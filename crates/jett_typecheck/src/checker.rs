use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use jett_common::{FileId, Span};
use jett_diagnostics::{Diagnostic, DiagnosticSink};
use jett_parser::ast::{
    self, BinOp, Block, Expr, FunctionDef, Item, Module, Stmt, StringPart, TypeExpr, UnaryOp,
    VerifyBlock,
};
use jett_resolve::resolver::ResolveResult;
use jett_resolve::scope::{DefId, DefKind};
use jett_types::{
    ActorDef as TypeActorDef, ActorMessageDef, BitfieldDef as TypeBitfieldDef,
    BitfieldFieldDef as TypeBitfieldFieldDef, BitfieldFieldKind as TypeBitfieldFieldKind,
    BitfieldId, EnumDef as TypeEnumDef, FunctionSig, InterfaceDef as TypeInterfaceDef,
    MachineDef as TypeMachineDef, MachineId, MachineStateDef as TypeMachineStateDef,
    MachineStateId, MachineTransitionDef as TypeMachineTransitionDef, ReflectionBitfieldFieldInfo,
    ReflectionBitfieldInfo, ReflectionFieldInfo, ReflectionMachineInfo, ReflectionMachineStateInfo,
    ReflectionMachineTransitionInfo, ReflectionMetadata, ReflectionTypeInfo, ReflectionVariantInfo,
    StructDef as TypeStructDef, StructId, Type, TypeId, TypeInterner, VariantDef,
};

use crate::capability;
use crate::errors;

/// The result of type checking.
#[derive(Debug)]
pub struct CheckResult {
    /// Diagnostics (errors and warnings) emitted during type checking.
    pub diagnostics: Vec<Diagnostic>,
    /// Map from expression spans to their inferred type.
    pub type_map: HashMap<Span, TypeId>,
    /// The type interner, containing all types encountered during checking.
    pub interner: TypeInterner,
    /// Checked reflection metadata snapshot for comptime reflection builtins.
    pub reflection_metadata: Arc<ReflectionMetadata>,
}

/// Type-check a resolved module.
pub fn check(module: &Module, resolve: &ResolveResult) -> CheckResult {
    let mut checker = TypeChecker::new(resolve);
    checker.check_module(module);

    let complexity_diagnostics = crate::complexity::check_complexity(module);

    // Run ownership analysis (linear type checking) after type checking.
    let ownership_diagnostics = crate::ownership::check_ownership(module, &checker.interner);

    let reflection_metadata = Arc::new(checker.build_reflection_metadata());

    let mut diagnostics = checker.sink.into_diagnostics();
    diagnostics.extend(complexity_diagnostics);
    diagnostics.extend(ownership_diagnostics);

    CheckResult {
        diagnostics,
        type_map: checker.type_map,
        interner: checker.interner,
        reflection_metadata,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
struct ReflectionParamFacts {
    type_info_kinds: Vec<(usize, String)>,
    type_info_primitives: Vec<(usize, Option<String>)>,
    type_kind_values: Vec<(usize, String)>,
    type_primitive_values: Vec<(usize, String)>,
}

impl ReflectionParamFacts {
    fn is_empty(&self) -> bool {
        self.type_info_kinds.is_empty()
            && self.type_info_primitives.is_empty()
            && self.type_kind_values.is_empty()
            && self.type_primitive_values.is_empty()
    }
}

#[derive(Debug, Clone)]
struct ReflectionTypeInfoStaticFacts {
    kind_tag: String,
    primitive_tag: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StaticReflectionEnumValue {
    TypeKind(String),
    TypePrimitive(String),
}

impl StaticReflectionEnumValue {
    fn variant_name(&self) -> &str {
        match self {
            Self::TypeKind(name) | Self::TypePrimitive(name) => name,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReflectionBranchContext {
    TopLevel,
    StaticReflectionBranch,
    RuntimeBranch,
}

impl ReflectionBranchContext {
    fn permits_shape_reflection(self) -> bool {
        matches!(self, Self::TopLevel | Self::StaticReflectionBranch)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MachineStateTruth {
    Is,
    IsNot,
}

// ---------------------------------------------------------------------------
// Internal type checker
// ---------------------------------------------------------------------------

struct TypeChecker<'a> {
    interner: TypeInterner,
    resolve: &'a ResolveResult,
    sink: DiagnosticSink,
    /// DefId → TypeId for variables, parameters, and functions.
    type_env: HashMap<DefId, TypeId>,
    /// Declaration span → DefId for locally declared names.
    decl_defs: HashMap<Span, DefId>,
    /// User-defined type name → TypeId.
    named_types: HashMap<String, TypeId>,
    /// Named types originating from compiler-shipped stdlib files.
    trusted_stdlib_named_types: HashMap<String, TypeId>,
    type_aliases: HashMap<String, ast::TypeAlias>,
    resolving_type_aliases: HashSet<String>,
    /// Expression span → TypeId (the output type map).
    type_map: HashMap<Span, TypeId>,
    /// The expected return type for the function currently being checked.
    current_return_type: Option<TypeId>,
    /// (interface, concrete type) -> implemented method signatures.
    interface_impls: HashMap<(TypeId, TypeId), HashMap<String, FunctionSig>>,
    /// concrete type -> all methods contributed by implement blocks.
    impl_methods_by_type: HashMap<TypeId, HashMap<String, FunctionSig>>,

    // -- Capability / purity tracking --
    /// Function name → is_pure.  Built during the first pass over the module.
    purity_map: HashMap<String, bool>,
    /// User-defined function name -> parameter and return types. Namespaced
    /// declarations are registered under their canonical `namespace.name`.
    function_signatures: HashMap<String, (Vec<TypeId>, TypeId)>,
    /// Function signatures originating from compiler-shipped stdlib files.
    trusted_stdlib_function_signatures: HashMap<String, (Vec<TypeId>, TypeId)>,
    /// Name of the function currently being type-checked (None outside functions).
    current_function_name: Option<String>,
    /// Whether the function currently being type-checked is pure.
    current_function_pure: bool,
    /// Whether we are inside a verify block.
    in_verify_block: bool,
    /// Whether we are inside a property block.
    in_property_block: bool,
    /// The name of the verify block currently being checked (for error messages).
    current_verify_name: Option<String>,
    /// Nesting depth inside a handle-block body. Used to validate `default`.
    handle_body_depth: usize,

    // -- Generic struct support --
    /// AST templates for user-defined generic structs (have type_params).
    generic_struct_templates: HashMap<String, ast::StructDef>,
    /// Cache of monomorphized generic struct instances: (name, concrete type args) → TypeId.
    monomorphized_structs: HashMap<(String, Vec<TypeId>), TypeId>,
    /// Checked reflection field snapshots keyed by the canonical owner TypeId.
    reflection_fields_by_id: HashMap<TypeId, (String, Vec<ReflectionFieldInfo>)>,
    /// Checked bitfield layout snapshots keyed by the canonical owner TypeId.
    reflection_bitfields_by_id: HashMap<TypeId, (String, ReflectionBitfieldInfo)>,
    /// Checked machine layout snapshots keyed by the canonical owner TypeId.
    reflection_machines_by_id: HashMap<TypeId, (String, ReflectionMachineInfo)>,
    /// Checked enum variant snapshots keyed by the canonical owner TypeId.
    reflection_variants_by_id: HashMap<TypeId, (String, Vec<ReflectionVariantInfo>)>,
    /// Active type variable substitution during monomorphization (type_param_name → TypeId).
    type_var_subst: HashMap<String, TypeId>,
    /// Source-level reflected kind tags for active type variables. This keeps
    /// simple aliases visible to `type.kind_tag[T]()` while their TypeId may
    /// resolve to the alias base type.
    type_var_kind_tags: HashMap<String, String>,
    /// Lexically scoped immutable `TypeInfo` locals known to come from
    /// `type.info[T]()` for the current concrete generic instantiation.
    reflection_type_info_kind_scopes: Vec<HashMap<DefId, String>>,
    /// Primitive tags for known `TypeInfo` locals. A present `None` means the
    /// TypeInfo is known, but it has no primitive tag.
    reflection_type_info_primitive_scopes: Vec<HashMap<DefId, Option<String>>>,
    /// Lexically scoped immutable `TypeKind` locals known from direct reflected
    /// kind facts.
    reflection_type_kind_value_scopes: Vec<HashMap<DefId, String>>,
    /// Lexically scoped immutable `TypePrimitive` locals known from direct
    /// reflected primitive facts.
    reflection_type_primitive_value_scopes: Vec<HashMap<DefId, String>>,
    /// Trusted field types currently available from direct `type.fields[T]()` loops.
    reflected_field_type_scopes: Vec<HashMap<String, Vec<TypeId>>>,
    /// Trusted TypeInfo types currently available from direct reflected `args` loops.
    reflected_type_info_scopes: Vec<HashMap<String, Vec<TypeId>>>,
    /// Trusted TypeVariant owners currently available from direct `type.variants[T]()` loops.
    reflected_variant_type_scopes: Vec<HashMap<String, TypeId>>,
    /// Trusted TypeMachineState owners currently available from direct `type.machine_states[T]()` loops.
    reflected_machine_state_type_scopes: Vec<HashMap<String, TypeId>>,

    // -- Generic function support --
    /// AST templates for user-defined generic functions (have type_params).
    generic_function_templates: HashMap<String, FunctionDef>,
    /// Generic function instantiations whose bodies have already been checked.
    checked_generic_function_instantiations:
        HashSet<(String, Vec<TypeId>, Vec<String>, ReflectionParamFacts)>,
    /// True while checking a generic instantiation whose type-param reflection
    /// is limited to directly evaluable branch conditions.
    specialize_reflection_branches: bool,

    // -- Actor support --
    /// The expected `responds T` type for the receive handler being checked.
    /// `None` when not inside a receive handler.
    current_respond_type: Option<TypeId>,
}

impl<'a> TypeChecker<'a> {
    fn new(resolve: &'a ResolveResult) -> Self {
        let decl_defs = resolve
            .scope_table
            .definitions
            .iter()
            .map(|def| (def.span, def.id))
            .collect();

        let mut checker = Self {
            interner: TypeInterner::new(),
            resolve,
            sink: DiagnosticSink::new(),
            type_env: HashMap::new(),
            decl_defs,
            named_types: HashMap::new(),
            trusted_stdlib_named_types: HashMap::new(),
            type_aliases: HashMap::new(),
            resolving_type_aliases: HashSet::new(),
            type_map: HashMap::new(),
            current_return_type: None,
            interface_impls: HashMap::new(),
            impl_methods_by_type: HashMap::new(),
            purity_map: HashMap::new(),
            function_signatures: HashMap::new(),
            trusted_stdlib_function_signatures: HashMap::new(),
            current_function_name: None,
            current_function_pure: false,
            in_verify_block: false,
            in_property_block: false,
            current_verify_name: None,
            handle_body_depth: 0,
            generic_struct_templates: HashMap::new(),
            monomorphized_structs: HashMap::new(),
            reflection_fields_by_id: HashMap::new(),
            reflection_bitfields_by_id: HashMap::new(),
            reflection_machines_by_id: HashMap::new(),
            reflection_variants_by_id: HashMap::new(),
            type_var_subst: HashMap::new(),
            type_var_kind_tags: HashMap::new(),
            reflection_type_info_kind_scopes: Vec::new(),
            reflection_type_info_primitive_scopes: Vec::new(),
            reflection_type_kind_value_scopes: Vec::new(),
            reflection_type_primitive_value_scopes: Vec::new(),
            reflected_field_type_scopes: Vec::new(),
            reflected_type_info_scopes: Vec::new(),
            reflected_variant_type_scopes: Vec::new(),
            reflected_machine_state_type_scopes: Vec::new(),
            generic_function_templates: HashMap::new(),
            checked_generic_function_instantiations: HashSet::new(),
            specialize_reflection_branches: false,
            current_respond_type: None,
        };
        checker.install_builtin_metadata_types();
        checker
    }

    fn install_builtin_metadata_types(&mut self) {
        let type_kind_eid = self.interner.add_enum(TypeEnumDef {
            name: "TypeKind".to_string(),
            variants: Self::metadata_unit_variants(&[
                "primitive_type",
                "alias_type",
                "refinement_type",
                "struct_type",
                "bitfield_type",
                "enum_type",
                "list_type",
                "set_type",
                "map_type",
                "optional_type",
                "result_type",
                "secret_type",
                "function_type",
                "machine_type",
                "machine_state_type",
                "unknown_type",
            ]),
        });
        let type_kind_ty = self.interner.intern(Type::Enum(type_kind_eid));
        self.named_types
            .insert("TypeKind".to_string(), type_kind_ty);

        let type_primitive_eid = self.interner.add_enum(TypeEnumDef {
            name: "TypePrimitive".to_string(),
            variants: Self::metadata_unit_variants(&[
                "int8_type",
                "int16_type",
                "int32_type",
                "int64_type",
                "uint8_type",
                "uint16_type",
                "uint32_type",
                "uint64_type",
                "float32_type",
                "float64_type",
                "string_type",
                "bool_type",
                "bytes_type",
                "nothing_type",
                "type_construction_type",
                "unknown_type",
            ]),
        });
        let type_primitive_ty = self.interner.intern(Type::Enum(type_primitive_eid));
        self.named_types
            .insert("TypePrimitive".to_string(), type_primitive_ty);

        let bitfield_shape_eid = self.interner.add_enum(TypeEnumDef {
            name: "TypeBitfieldFieldShape".to_string(),
            variants: Self::metadata_unit_variants(&["bits_field", "payload_field"]),
        });
        let bitfield_shape_ty = self.interner.intern(Type::Enum(bitfield_shape_eid));
        self.named_types
            .insert("TypeBitfieldFieldShape".to_string(), bitfield_shape_ty);

        let type_info_sid = self.interner.add_struct(TypeStructDef {
            name: "TypeInfo".to_string(),
            fields: Vec::new(),
            methods: Vec::new(),
        });
        let type_info_ty = self.interner.intern(Type::Struct(type_info_sid));
        let type_info_args_ty = self.interner.intern(Type::List(type_info_ty));
        let optional_type_primitive_ty = self.interner.intern(Type::Optional(type_primitive_ty));
        self.interner.update_struct(
            type_info_sid,
            TypeStructDef {
                name: "TypeInfo".to_string(),
                fields: vec![
                    ("type_name".to_string(), TypeInterner::STRING),
                    ("kind".to_string(), TypeInterner::STRING),
                    ("kind_tag".to_string(), type_kind_ty),
                    ("primitive_tag".to_string(), optional_type_primitive_ty),
                    ("has_secret".to_string(), TypeInterner::BOOL),
                    ("args".to_string(), type_info_args_ty),
                ],
                methods: Vec::new(),
            },
        );
        self.named_types
            .insert("TypeInfo".to_string(), type_info_ty);

        let optional_string_ty = self.interner.intern(Type::Optional(TypeInterner::STRING));
        let type_field_sid = self.interner.add_struct(TypeStructDef {
            name: "TypeField".to_string(),
            fields: vec![
                ("index".to_string(), TypeInterner::INT64),
                ("owner_type".to_string(), TypeInterner::STRING),
                ("owner_member".to_string(), optional_string_ty),
                ("name".to_string(), TypeInterner::STRING),
                ("type_name".to_string(), TypeInterner::STRING),
                ("kind".to_string(), TypeInterner::STRING),
                ("kind_tag".to_string(), type_kind_ty),
                ("serialize_name".to_string(), TypeInterner::STRING),
                ("has_secret".to_string(), TypeInterner::BOOL),
                ("type_info".to_string(), type_info_ty),
            ],
            methods: Vec::new(),
        });
        let type_field_ty = self.interner.intern(Type::Struct(type_field_sid));
        self.named_types
            .insert("TypeField".to_string(), type_field_ty);

        let optional_type_info_ty = self.interner.intern(Type::Optional(type_info_ty));
        let type_bitfield_field_sid = self.interner.add_struct(TypeStructDef {
            name: "TypeBitfieldField".to_string(),
            fields: vec![
                ("index".to_string(), TypeInterner::INT64),
                ("name".to_string(), TypeInterner::STRING),
                ("shape".to_string(), TypeInterner::STRING),
                ("shape_tag".to_string(), bitfield_shape_ty),
                ("width".to_string(), TypeInterner::INT64),
                ("type_info".to_string(), type_info_ty),
                ("enum_type".to_string(), optional_type_info_ty),
            ],
            methods: Vec::new(),
        });
        let type_bitfield_field_ty = self.interner.intern(Type::Struct(type_bitfield_field_sid));
        self.named_types
            .insert("TypeBitfieldField".to_string(), type_bitfield_field_ty);

        let type_bitfield_fields_ty = self.interner.intern(Type::List(type_bitfield_field_ty));
        let type_bitfield_sid = self.interner.add_struct(TypeStructDef {
            name: "TypeBitfield".to_string(),
            fields: vec![
                ("network_order".to_string(), TypeInterner::BOOL),
                ("fields".to_string(), type_bitfield_fields_ty),
            ],
            methods: Vec::new(),
        });
        let type_bitfield_ty = self.interner.intern(Type::Struct(type_bitfield_sid));
        self.named_types
            .insert("TypeBitfield".to_string(), type_bitfield_ty);

        let type_variant_fields_ty = self.interner.intern(Type::List(type_field_ty));
        let type_variant_sid = self.interner.add_struct(TypeStructDef {
            name: "TypeVariant".to_string(),
            fields: vec![
                ("index".to_string(), TypeInterner::INT64),
                ("owner_type".to_string(), TypeInterner::STRING),
                ("name".to_string(), TypeInterner::STRING),
                ("discriminant".to_string(), TypeInterner::INT64),
                ("has_secret".to_string(), TypeInterner::BOOL),
                ("fields".to_string(), type_variant_fields_ty),
            ],
            methods: Vec::new(),
        });
        let type_variant_ty = self.interner.intern(Type::Struct(type_variant_sid));
        self.named_types
            .insert("TypeVariant".to_string(), type_variant_ty);

        let type_machine_state_fields_ty = self.interner.intern(Type::List(type_field_ty));
        let type_machine_state_sid = self.interner.add_struct(TypeStructDef {
            name: "TypeMachineState".to_string(),
            fields: vec![
                ("index".to_string(), TypeInterner::INT64),
                ("owner_type".to_string(), TypeInterner::STRING),
                ("name".to_string(), TypeInterner::STRING),
                ("has_secret".to_string(), TypeInterner::BOOL),
                ("fields".to_string(), type_machine_state_fields_ty),
            ],
            methods: Vec::new(),
        });
        let type_machine_state_ty = self.interner.intern(Type::Struct(type_machine_state_sid));
        self.named_types
            .insert("TypeMachineState".to_string(), type_machine_state_ty);

        let type_machine_transition_sid = self.interner.add_struct(TypeStructDef {
            name: "TypeMachineTransition".to_string(),
            fields: vec![
                ("index".to_string(), TypeInterner::INT64),
                ("source_index".to_string(), TypeInterner::INT64),
                ("source".to_string(), TypeInterner::STRING),
                ("target_index".to_string(), TypeInterner::INT64),
                ("target".to_string(), TypeInterner::STRING),
            ],
            methods: Vec::new(),
        });
        let type_machine_transition_ty = self
            .interner
            .intern(Type::Struct(type_machine_transition_sid));
        self.named_types.insert(
            "TypeMachineTransition".to_string(),
            type_machine_transition_ty,
        );

        let type_machine_states_ty = self.interner.intern(Type::List(type_machine_state_ty));
        let type_machine_transitions_ty =
            self.interner.intern(Type::List(type_machine_transition_ty));
        let type_machine_sid = self.interner.add_struct(TypeStructDef {
            name: "TypeMachine".to_string(),
            fields: vec![
                ("states".to_string(), type_machine_states_ty),
                ("edges".to_string(), type_machine_transitions_ty),
            ],
            methods: Vec::new(),
        });
        let type_machine_ty = self.interner.intern(Type::Struct(type_machine_sid));
        self.named_types
            .insert("TypeMachine".to_string(), type_machine_ty);
    }

    fn metadata_unit_variants(names: &[&str]) -> Vec<VariantDef> {
        names
            .iter()
            .enumerate()
            .map(|(index, name)| VariantDef {
                name: (*name).to_string(),
                fields: Vec::new(),
                discriminant: index as i64,
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Utility: human-readable type name
    // ------------------------------------------------------------------

    fn type_name(&self, id: TypeId) -> String {
        match self.interner.resolve(id) {
            Type::Int8 => "int8".to_string(),
            Type::Int16 => "int16".to_string(),
            Type::Int32 => "int32".to_string(),
            Type::Int64 => "int64".to_string(),
            Type::Uint8 => "uint8".to_string(),
            Type::Uint16 => "uint16".to_string(),
            Type::Uint32 => "uint32".to_string(),
            Type::Uint64 => "uint64".to_string(),
            Type::Float32 => "float32".to_string(),
            Type::Float64 => "float64".to_string(),
            Type::String => "string".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Bytes => "bytes".to_string(),
            Type::Nothing => "nothing".to_string(),
            Type::TypeConstruction => "TypeConstruction".to_string(),
            Type::List(inner) => format!("list[{}]", self.type_name(*inner)),
            Type::Map(k, v) => format!("map[{}, {}]", self.type_name(*k), self.type_name(*v)),
            Type::Set(inner) => format!("set[{}]", self.type_name(*inner)),
            Type::Optional(inner) => format!("optional[{}]", self.type_name(*inner)),
            Type::Result(ok, err) => {
                format!("result[{}, {}]", self.type_name(*ok), self.type_name(*err))
            }
            Type::Secret(inner) => format!("secret[{}]", self.type_name(*inner)),
            Type::Struct(sid) => self.interner.resolve_struct(*sid).name.clone(),
            Type::Bitfield(bid) => self.interner.resolve_bitfield(*bid).name.clone(),
            Type::Enum(eid) => self.interner.resolve_enum(*eid).name.clone(),
            Type::Interface(iid) => self.interner.resolve_interface(*iid).name.clone(),
            Type::Actor(aid) => self.interner.resolve_actor(*aid).name.clone(),
            Type::Machine(mid) => self.interner.resolve_machine(*mid).name.clone(),
            Type::MachineState { machine, state } => {
                let machine_def = self.interner.resolve_machine(*machine);
                match machine_def.state(*state) {
                    Some(state_def) => format!("{} at {}", machine_def.name, state_def.name),
                    None => format!("{} at <unknown>", machine_def.name),
                }
            }
            Type::Function {
                params,
                return_type,
            } => {
                let params: Vec<String> = params.iter().map(|p| self.type_name(*p)).collect();
                format!(
                    "function({}) returns {}",
                    params.join(", "),
                    self.type_name(*return_type)
                )
            }
            Type::Refinement { name, .. } => name.clone(),
            Type::Error => "<error>".to_string(),
        }
    }

    fn item_file(item: &Item) -> FileId {
        match item {
            Item::Namespace(ns) => ns.span.file,
            Item::Function(func) => func.span.file,
            Item::Mutual(block) => block.span.file,
            Item::Interface(interface) => interface.span.file,
            Item::Implement(block) => block.span.file,
            Item::Struct(strukt) => strukt.span.file,
            Item::Bitfield(bitfield) => bitfield.span.file,
            Item::Enum(enm) => enm.span.file,
            Item::Machine(machine) => machine.span.file,
            Item::Actor(actor) => actor.span.file,
            Item::VarDecl(decl) => decl.span.file,
            Item::Verify(verify) => verify.span.file,
            Item::Property(prop) => prop.span.file,
            Item::TypeAlias(alias) => alias.span.file,
        }
    }

    fn update_current_namespace(
        item: &Item,
        current_file: &mut Option<FileId>,
        current_namespace: &mut Option<String>,
    ) {
        let item_file = Self::item_file(item);
        if current_file.is_some_and(|file| file != item_file) {
            *current_namespace = None;
        }
        *current_file = Some(item_file);

        if let Item::Namespace(ns) = item {
            *current_namespace = Some(ns.name.name.clone());
        }
    }

    fn namespace_qualified_name(namespace: Option<&str>, name: &str) -> Option<String> {
        namespace.map(|ns| format!("{ns}.{name}"))
    }

    fn canonical_name(namespace: Option<&str>, name: &str) -> String {
        Self::namespace_qualified_name(namespace, name).unwrap_or_else(|| name.to_string())
    }

    fn function_lookup_names(namespace: Option<&str>, name: &str) -> Vec<String> {
        vec![Self::canonical_name(namespace, name)]
    }

    fn type_lookup_names(namespace: Option<&str>, name: &str) -> Vec<String> {
        Self::function_lookup_names(namespace, name)
    }

    fn register_named_type(&mut self, namespace: Option<&str>, name: &str, ty: TypeId) {
        let canonical = Self::canonical_name(namespace, name);
        self.named_types.insert(canonical, ty);
    }

    fn register_trusted_stdlib_named_type(&mut self, canonical_name: &str, ty: TypeId, span: Span) {
        if span.file.is_stdlib() {
            self.trusted_stdlib_named_types
                .insert(canonical_name.to_string(), ty);
        }
    }

    fn register_generic_struct_template(
        &mut self,
        namespace: Option<&str>,
        name: &str,
        def: ast::StructDef,
    ) {
        let canonical = Self::canonical_name(namespace, name);
        self.generic_struct_templates.insert(canonical, def);
    }

    fn check_struct_json_serialize_names(&mut self, def: &ast::StructDef, namespace: Option<&str>) {
        let type_name = Self::canonical_name(namespace, &def.name.name);
        let mut seen = HashMap::new();
        for field in &def.fields {
            let serialize_name = field.serialize_name.as_deref().unwrap_or(&field.name.name);
            if let Some(previous_span) = seen.insert(serialize_name.to_string(), field.span) {
                self.sink.emit(errors::duplicate_json_serialize_name(
                    &type_name,
                    serialize_name,
                    field.span,
                    previous_span,
                ));
            }
        }
    }

    fn declaration_def_id(&self, span: Span) -> Option<DefId> {
        self.resolve
            .resolutions
            .get(&span)
            .copied()
            .or_else(|| self.decl_defs.get(&span).copied())
    }

    fn ident_def_id(&self, ident: &ast::Ident) -> Option<DefId> {
        self.resolve
            .resolutions
            .get(&ident.span)
            .copied()
            .or_else(|| self.decl_defs.get(&ident.span).copied())
    }

    fn ident_def_kind(&self, ident: &ast::Ident) -> Option<DefKind> {
        let def_id = self
            .resolve
            .resolutions
            .get(&ident.span)
            .copied()
            .or_else(|| self.decl_defs.get(&ident.span).copied())?;
        Some(self.resolve.scope_table.def(def_id).kind)
    }

    fn resolved_symbol_name(&self, name: &str, span: Span) -> String {
        self.resolve
            .resolutions
            .get(&span)
            .copied()
            .or_else(|| self.decl_defs.get(&span).copied())
            .map(|def_id| self.resolve.scope_table.def(def_id).name.clone())
            .unwrap_or_else(|| name.to_string())
    }

    fn resolved_or_expanded_name(&self, name: &str, span: Span) -> String {
        let Some((prefix, suffix)) = name.split_once('.') else {
            return self.resolved_symbol_name(name, span);
        };

        let Some(def_id) = self
            .resolve
            .resolutions
            .get(&span)
            .copied()
            .or_else(|| self.decl_defs.get(&span).copied())
        else {
            return name.to_string();
        };

        let def = self.resolve.scope_table.def(def_id);
        if def.kind == DefKind::Namespace {
            if def.name == prefix
                && let Some(target) = self.resolve.namespace_aliases.get(&def_id)
            {
                return format!("{target}.{suffix}");
            }
            return name.to_string();
        }

        def.name.clone()
    }

    fn expanded_dotted_expr_name(&self, expr: &Expr) -> Option<String> {
        let name = Self::extract_dotted_name(expr)?;
        Some(self.resolved_or_expanded_name(&name, expr.span()))
    }

    fn is_struct_type_name_expr(&self, expr: &Expr) -> bool {
        let Some(name) = self.expanded_dotted_expr_name(expr) else {
            return false;
        };
        self.named_types
            .get(&name)
            .is_some_and(|ty| matches!(self.interner.resolve(*ty), Type::Struct(_)))
    }

    fn is_bitfield_type_name_expr(&self, expr: &Expr) -> bool {
        let Some(name) = self.expanded_dotted_expr_name(expr) else {
            return false;
        };
        self.named_types
            .get(&name)
            .is_some_and(|ty| matches!(self.interner.resolve(*ty), Type::Bitfield(_)))
    }

    fn is_reflection_metadata_type_name(name: &str) -> bool {
        matches!(
            name,
            "TypeInfo"
                | "TypeField"
                | "TypeBitfield"
                | "TypeBitfieldField"
                | "TypeMachine"
                | "TypeMachineState"
                | "TypeMachineTransition"
                | "TypeVariant"
        )
    }

    /// Returns true if the type is numeric (any integer or float type).
    fn is_numeric(&self, id: TypeId) -> bool {
        matches!(
            self.interner.resolve(id),
            Type::Int8
                | Type::Int16
                | Type::Int32
                | Type::Int64
                | Type::Uint8
                | Type::Uint16
                | Type::Uint32
                | Type::Uint64
                | Type::Float32
                | Type::Float64
        )
    }

    fn int_literal_fits_type(&self, value: i128, id: TypeId) -> bool {
        match self.interner.resolve(id) {
            Type::Int8 => (i8::MIN as i128..=i8::MAX as i128).contains(&value),
            Type::Int16 => (i16::MIN as i128..=i16::MAX as i128).contains(&value),
            Type::Int32 => (i32::MIN as i128..=i32::MAX as i128).contains(&value),
            Type::Int64 => (i64::MIN as i128..=i64::MAX as i128).contains(&value),
            Type::Uint8 => (0..=u8::MAX as i128).contains(&value),
            Type::Uint16 => (0..=u16::MAX as i128).contains(&value),
            Type::Uint32 => (0..=u32::MAX as i128).contains(&value),
            Type::Uint64 => (0..=u64::MAX as i128).contains(&value),
            _ => false,
        }
    }

    fn int_literal_matches_expected_type(&self, value: i128, expected_ty: TypeId) -> bool {
        let expected_inner = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
        self.int_literal_fits_type(value, expected_inner)
    }

    fn float_literal_matches_expected_type(&self, expected_ty: TypeId) -> bool {
        let id = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
        matches!(self.interner.resolve(id), Type::Float32 | Type::Float64)
    }

    fn negated_literal_matches_expected_type(&self, operand: &Expr, expected_ty: TypeId) -> bool {
        match operand {
            Expr::IntLiteral(value, _) => value.checked_neg().is_some_and(|negated| {
                self.int_literal_matches_expected_type(negated, expected_ty)
            }),
            Expr::FloatLiteral(_, _) => self.float_literal_matches_expected_type(expected_ty),
            _ => false,
        }
    }

    fn expected_numeric_type(&self, expected_ty: TypeId) -> Option<TypeId> {
        let id = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
        self.is_numeric(id).then_some(id)
    }

    fn is_numeric_literal(expr: &Expr) -> bool {
        matches!(expr, Expr::IntLiteral(_, _) | Expr::FloatLiteral(_, _))
    }

    fn json_read_requires_view(&self, id: TypeId) -> bool {
        match self.interner.resolve(id) {
            Type::Int8
            | Type::Int16
            | Type::Int32
            | Type::Int64
            | Type::Uint8
            | Type::Uint16
            | Type::Uint32
            | Type::Uint64
            | Type::Float32
            | Type::Float64
            | Type::String
            | Type::Bool
            | Type::Nothing
            | Type::Error => false,
            Type::Refinement { base, .. } => self.json_read_requires_view(*base),
            _ => true,
        }
    }

    fn is_secret_type(&self, id: TypeId) -> bool {
        matches!(self.interner.resolve(id), Type::Secret(_))
    }

    fn is_refinement_type(&self, id: TypeId) -> bool {
        matches!(self.interner.resolve(id), Type::Refinement { .. })
    }

    fn refinement_base_type(&self, id: TypeId) -> Option<TypeId> {
        match self.interner.resolve(id) {
            Type::Refinement { base, .. } => Some(*base),
            _ => None,
        }
    }

    fn fully_coarsened_type(&self, mut id: TypeId) -> TypeId {
        while let Some(base) = self.refinement_base_type(id) {
            id = base;
        }
        id
    }

    fn can_coarsen_to(&self, mut source: TypeId, target: TypeId) -> bool {
        while let Some(base) = self.refinement_base_type(source) {
            if base == target {
                return true;
            }
            source = base;
        }
        false
    }

    fn refinement_boundary_input_type(&self, refinement_ty: TypeId) -> TypeId {
        let base_ty = self.fully_coarsened_type(refinement_ty);
        self.secret_inner_type(base_ty).unwrap_or(base_ty)
    }

    fn can_refine_from(&self, source: TypeId, mut refinement_ty: TypeId) -> bool {
        while let Some(base) = self.refinement_base_type(refinement_ty) {
            if base == source || self.secret_inner_type(base) == Some(source) {
                return true;
            }
            refinement_ty = base;
        }
        false
    }

    fn satisfies_expected_type(&self, expected: TypeId, got: TypeId) -> bool {
        self.types_compatible(expected, got)
            || (self.is_refinement_type(expected) && self.can_refine_from(got, expected))
    }

    fn type_id_is_named(&self, type_id: TypeId, name: &str) -> bool {
        self.named_types
            .get(name)
            .is_some_and(|named_type_id| *named_type_id == type_id)
    }

    fn type_requires_handle_error(&self, expected: TypeId, got: TypeId) -> bool {
        if matches!(self.interner.resolve(expected), Type::Result(_, _)) {
            return false;
        }
        match self.interner.resolve(got) {
            Type::Result(ok_ty, _) => self.satisfies_expected_type(expected, *ok_ty),
            _ => false,
        }
    }

    fn type_requires_bare_handle(&self, expected: TypeId, got: TypeId) -> bool {
        if matches!(
            self.interner.resolve(expected),
            Type::Result(_, _) | Type::Optional(_)
        ) {
            return false;
        }
        match self.interner.resolve(got) {
            Type::Optional(inner_ty) => self.satisfies_expected_type(expected, *inner_ty),
            _ => false,
        }
    }

    fn secret_inner_type(&self, id: TypeId) -> Option<TypeId> {
        match self.interner.resolve(id) {
            Type::Secret(inner) => Some(*inner),
            _ => None,
        }
    }

    fn strip_secret_type(&self, id: TypeId) -> (TypeId, bool) {
        match self.secret_inner_type(id) {
            Some(inner) => (inner, true),
            None => (id, false),
        }
    }

    fn maybe_wrap_secret(&mut self, ty: TypeId, tainted: bool) -> TypeId {
        if !tainted || ty == TypeInterner::ERROR || ty == TypeInterner::NOTHING {
            return ty;
        }
        if self.is_secret_type(ty) {
            return ty;
        }
        self.interner.intern(Type::Secret(ty))
    }

    fn is_secret_output_boundary(name: &str) -> bool {
        matches!(
            name,
            "Stdout.write"
                | "print"
                | "println"
                | "json.serialize"
                | "json.serialize_public"
                | "json.serialize_raw"
                | "Filesystem.write_file"
                | "log"
                | "http.respond"
        )
    }

    fn is_impure_builtin(name: &str) -> bool {
        matches!(
            name,
            "Stdout.write" | "Environment.args" | "Filesystem.read_file" | "Filesystem.write_file"
        )
    }

    fn is_secret_safe_builtin(name: &str) -> bool {
        matches!(name, "secret.redact" | "secret.compare")
    }

    fn is_secret_liftable_call(name: &str, callee_is_pure: bool) -> bool {
        callee_is_pure
            && !Self::is_secret_output_boundary(name)
            && !Self::is_secret_safe_builtin(name)
    }

    fn secret_argument_matches_param(&self, expected: TypeId, got: TypeId) -> (bool, bool) {
        if self.types_compatible(expected, got) {
            return (true, false);
        }

        let Some(inner) = self.secret_inner_type(got) else {
            return (false, false);
        };

        if self.types_compatible(expected, inner) {
            (true, true)
        } else {
            (false, false)
        }
    }

    fn type_contains_secret_data(&self, ty: TypeId) -> bool {
        let mut visited = HashSet::new();
        self.type_contains_secret_data_inner(ty, &mut visited)
    }

    fn type_contains_secret_data_inner(&self, ty: TypeId, visited: &mut HashSet<TypeId>) -> bool {
        if !visited.insert(ty) {
            return false;
        }

        match self.interner.resolve(ty) {
            Type::Secret(_) => true,
            Type::List(inner) | Type::Set(inner) | Type::Optional(inner) => {
                self.type_contains_secret_data_inner(*inner, visited)
            }
            Type::Map(key, value) | Type::Result(key, value) => {
                self.type_contains_secret_data_inner(*key, visited)
                    || self.type_contains_secret_data_inner(*value, visited)
            }
            Type::Struct(sid) => self
                .interner
                .resolve_struct(*sid)
                .fields
                .iter()
                .any(|(_, field_ty)| self.type_contains_secret_data_inner(*field_ty, visited)),
            Type::Bitfield(bid) => self
                .interner
                .resolve_bitfield(*bid)
                .fields
                .iter()
                .any(|field| self.type_contains_secret_data_inner(field.ty, visited)),
            Type::Enum(eid) => self
                .interner
                .resolve_enum(*eid)
                .variants
                .iter()
                .flat_map(|variant| variant.fields.iter())
                .any(|(_, field_ty)| self.type_contains_secret_data_inner(*field_ty, visited)),
            Type::Machine(mid) => self
                .interner
                .resolve_machine(*mid)
                .states
                .iter()
                .flat_map(|state| state.fields.iter())
                .any(|(_, field_ty)| self.type_contains_secret_data_inner(*field_ty, visited)),
            Type::MachineState { machine, state } => self
                .interner
                .resolve_machine(*machine)
                .state(*state)
                .is_some_and(|state_def| {
                    state_def.fields.iter().any(|(_, field_ty)| {
                        self.type_contains_secret_data_inner(*field_ty, visited)
                    })
                }),
            Type::Refinement { base, .. } => self.type_contains_secret_data_inner(*base, visited),
            _ => false,
        }
    }

    fn json_public_projection_allows_secret_data(&self, ty: TypeId) -> bool {
        let mut visited = HashSet::new();
        self.json_public_projection_allows_secret_data_inner(ty, &mut visited)
    }

    fn json_public_projection_allows_secret_data_inner(
        &self,
        ty: TypeId,
        visited: &mut HashSet<TypeId>,
    ) -> bool {
        if !visited.insert(ty) {
            return true;
        }

        match self.interner.resolve(ty) {
            Type::Secret(_) => false,
            Type::List(inner) | Type::Set(inner) | Type::Optional(inner) => {
                self.json_public_projection_allows_secret_data_inner(*inner, visited)
            }
            Type::Map(key, value) | Type::Result(key, value) => {
                self.json_public_projection_allows_secret_data_inner(*key, visited)
                    && self.json_public_projection_allows_secret_data_inner(*value, visited)
            }
            Type::Struct(_) | Type::Bitfield(_) | Type::Machine(_) | Type::MachineState { .. } => {
                true
            }
            Type::Enum(_) => !self.type_contains_secret_data(ty),
            Type::Refinement { base, .. } => {
                self.json_public_projection_allows_secret_data_inner(*base, visited)
            }
            _ => true,
        }
    }

    fn build_reflection_metadata(&mut self) -> ReflectionMetadata {
        let mut metadata = ReflectionMetadata::new();

        let type_ids = self.interner.type_ids().collect::<Vec<_>>();
        for type_id in type_ids {
            metadata.insert_type_info_for_id(type_id, self.reflection_type_info_for_type(type_id));
        }

        for (name, type_id) in self.named_types.clone() {
            if let Some(alias) = self.type_aliases.get(&name).cloned() {
                let info = self.reflection_type_info_for_alias(&name, &alias);
                if alias.constraint.is_some() {
                    metadata.insert_type_info_for_id(type_id, info);
                } else {
                    metadata.insert_type_info(info);
                }
            } else {
                let info = self.reflection_type_info_for_type_named(type_id, name);
                metadata.insert_type_info_for_id(type_id, info);
            }
        }

        for (type_id, (type_name, fields)) in self.reflection_fields_by_id.clone() {
            metadata.insert_type_fields_for_id(type_id, type_name, fields);
        }

        for (type_id, (type_name, bitfield)) in self.reflection_bitfields_by_id.clone() {
            metadata.insert_bitfield_for_id(type_id, type_name, bitfield);
        }

        for (type_id, (type_name, machine)) in self.reflection_machines_by_id.clone() {
            metadata.insert_machine_for_id(type_id, type_name, machine);
        }

        for (type_id, (type_name, variants)) in self.reflection_variants_by_id.clone() {
            metadata.insert_type_variants_for_id(type_id, type_name, variants);
        }

        metadata
    }

    fn reflection_type_info_for_alias(
        &mut self,
        name: &str,
        alias: &ast::TypeAlias,
    ) -> ReflectionTypeInfo {
        let base_ty = self.resolve_type_expr(&alias.base_type);
        let args = if base_ty == TypeInterner::ERROR {
            Vec::new()
        } else {
            vec![self.reflection_type_info_for_type(base_ty)]
        };
        let kind = if alias.constraint.is_some() {
            "refinement"
        } else {
            "alias"
        };
        let has_secret = base_ty != TypeInterner::ERROR && self.type_contains_secret_data(base_ty);
        ReflectionTypeInfo::new(name, kind, None, has_secret, args)
    }

    fn reflection_type_info_for_type(&self, type_id: TypeId) -> ReflectionTypeInfo {
        self.reflection_type_info_for_type_named(type_id, self.type_name(type_id))
    }

    fn reflection_type_info_for_type_named(
        &self,
        type_id: TypeId,
        type_name: String,
    ) -> ReflectionTypeInfo {
        let args = self
            .type_info_arg_types_for_type(type_id)
            .into_iter()
            .map(|arg| self.reflection_type_info_for_type(arg))
            .collect();
        self.reflection_type_info_for_type_named_with_args(type_id, type_name, args)
    }

    fn reflection_type_info_for_type_named_with_args(
        &self,
        type_id: TypeId,
        type_name: String,
        args: Vec<ReflectionTypeInfo>,
    ) -> ReflectionTypeInfo {
        ReflectionTypeInfo::new(
            type_name,
            self.reflection_kind_for_type(type_id),
            self.reflection_primitive_tag_for_type(type_id)
                .map(str::to_string),
            self.type_contains_secret_data(type_id),
            args,
        )
    }

    fn reflection_kind_for_type(&self, type_id: TypeId) -> &'static str {
        match self.interner.resolve(type_id) {
            Type::Int8
            | Type::Int16
            | Type::Int32
            | Type::Int64
            | Type::Uint8
            | Type::Uint16
            | Type::Uint32
            | Type::Uint64
            | Type::Float32
            | Type::Float64
            | Type::String
            | Type::Bool
            | Type::Bytes
            | Type::Nothing
            | Type::TypeConstruction => "primitive",
            Type::List(_) => "list",
            Type::Map(_, _) => "map",
            Type::Set(_) => "set",
            Type::Optional(_) => "optional",
            Type::Result(_, _) => "result",
            Type::Secret(_) => "secret",
            Type::Struct(_) => "struct",
            Type::Bitfield(_) => "bitfield",
            Type::Enum(_) => "enum",
            Type::Function { .. } => "function",
            Type::Refinement { .. } => "refinement",
            Type::Machine(_) => "machine",
            Type::MachineState { .. } => "machine_state",
            Type::Interface(_) | Type::Actor(_) | Type::Error => "unknown",
        }
    }

    fn reflection_kind_tag_for_type(&self, type_id: TypeId) -> &'static str {
        match self.interner.resolve(type_id) {
            Type::Int8
            | Type::Int16
            | Type::Int32
            | Type::Int64
            | Type::Uint8
            | Type::Uint16
            | Type::Uint32
            | Type::Uint64
            | Type::Float32
            | Type::Float64
            | Type::String
            | Type::Bool
            | Type::Bytes
            | Type::Nothing
            | Type::TypeConstruction => "primitive_type",
            Type::List(_) => "list_type",
            Type::Map(_, _) => "map_type",
            Type::Set(_) => "set_type",
            Type::Optional(_) => "optional_type",
            Type::Result(_, _) => "result_type",
            Type::Secret(_) => "secret_type",
            Type::Struct(_) => "struct_type",
            Type::Bitfield(_) => "bitfield_type",
            Type::Enum(_) => "enum_type",
            Type::Function { .. } => "function_type",
            Type::Refinement { .. } => "refinement_type",
            Type::Machine(_) => "machine_type",
            Type::MachineState { .. } => "machine_state_type",
            Type::Interface(_) | Type::Actor(_) | Type::Error => "unknown_type",
        }
    }

    fn reflection_kind_tag_for_type_expr(&self, ty: &TypeExpr, resolved_ty: TypeId) -> String {
        match ty {
            TypeExpr::View(inner, _) => self.reflection_kind_tag_for_type_expr(inner, resolved_ty),
            TypeExpr::StateQualified(_, _, _) => {
                self.reflection_kind_tag_for_type(resolved_ty).to_string()
            }
            TypeExpr::Named(ident) => {
                if let Some(kind_tag) = self.type_var_kind_tags.get(&ident.name) {
                    return kind_tag.clone();
                }

                let name = self.resolved_or_expanded_name(&ident.name, ident.span);
                if let Some(alias) = self
                    .type_aliases
                    .get(&name)
                    .or_else(|| self.type_aliases.get(&ident.name))
                {
                    if alias.constraint.is_some() {
                        "refinement_type".to_string()
                    } else {
                        "alias_type".to_string()
                    }
                } else if Self::reflection_primitive_tag_for_name(&name).is_some()
                    || Self::reflection_primitive_tag_for_name(&ident.name).is_some()
                {
                    "primitive_type".to_string()
                } else {
                    self.reflection_kind_tag_for_type(resolved_ty).to_string()
                }
            }
            TypeExpr::Generic(ident, _, _) => {
                let name = self.resolved_or_expanded_name(&ident.name, ident.span);
                match name.as_str() {
                    "list" => "list_type".to_string(),
                    "map" => "map_type".to_string(),
                    "set" => "set_type".to_string(),
                    "optional" => "optional_type".to_string(),
                    "result" => "result_type".to_string(),
                    "secret" => "secret_type".to_string(),
                    _ if self.generic_struct_templates.contains_key(&name) => {
                        "struct_type".to_string()
                    }
                    _ => self.reflection_kind_tag_for_type(resolved_ty).to_string(),
                }
            }
            TypeExpr::Function(_, _, _) => "function_type".to_string(),
        }
    }

    fn reflection_primitive_tag_for_type(&self, type_id: TypeId) -> Option<&'static str> {
        match self.interner.resolve(type_id) {
            Type::Int8 => Some("int8_type"),
            Type::Int16 => Some("int16_type"),
            Type::Int32 => Some("int32_type"),
            Type::Int64 => Some("int64_type"),
            Type::Uint8 => Some("uint8_type"),
            Type::Uint16 => Some("uint16_type"),
            Type::Uint32 => Some("uint32_type"),
            Type::Uint64 => Some("uint64_type"),
            Type::Float32 => Some("float32_type"),
            Type::Float64 => Some("float64_type"),
            Type::String => Some("string_type"),
            Type::Bool => Some("bool_type"),
            Type::Bytes => Some("bytes_type"),
            Type::Nothing => Some("nothing_type"),
            Type::TypeConstruction => Some("type_construction_type"),
            _ => None,
        }
    }

    fn reflection_primitive_tag_for_type_expr_static(
        &self,
        ty: &TypeExpr,
        resolved_ty: TypeId,
    ) -> Option<String> {
        if self.reflection_kind_tag_for_type_expr(ty, resolved_ty) != "primitive_type" {
            return None;
        }

        match ty {
            TypeExpr::Named(ident) => {
                let name = self.resolved_or_expanded_name(&ident.name, ident.span);
                Self::reflection_primitive_tag_for_name(&name)
                    .or_else(|| Self::reflection_primitive_tag_for_name(&ident.name))
                    .or_else(|| self.reflection_primitive_tag_for_type(resolved_ty))
                    .map(str::to_string)
            }
            TypeExpr::View(inner, _) => {
                self.reflection_primitive_tag_for_type_expr_static(inner, resolved_ty)
            }
            TypeExpr::StateQualified(_, _, _) => None,
            TypeExpr::Generic(_, _, _) | TypeExpr::Function(_, _, _) => self
                .reflection_primitive_tag_for_type(resolved_ty)
                .map(str::to_string),
        }
    }

    fn reflection_fields_for_struct_def(
        &mut self,
        def: &ast::StructDef,
        namespace: Option<&str>,
        resolved_fields: &[(String, TypeId)],
    ) -> Vec<ReflectionFieldInfo> {
        def.fields
            .iter()
            .zip(resolved_fields.iter())
            .enumerate()
            .map(|(index, (field, (_, field_ty)))| {
                self.reflection_field_info_for_type_expr(
                    index,
                    &field.name.name,
                    field.serialize_name.as_deref().unwrap_or(&field.name.name),
                    &field.ty,
                    namespace,
                    *field_ty,
                )
            })
            .collect()
    }

    fn reflection_fields_for_resolved_struct(
        &self,
        def: &ast::StructDef,
        resolved_fields: &[(String, TypeId)],
    ) -> Vec<ReflectionFieldInfo> {
        def.fields
            .iter()
            .zip(resolved_fields.iter())
            .enumerate()
            .map(|(index, (field, (_, field_ty)))| {
                self.reflection_field_info_for_type_id(
                    index,
                    &field.name.name,
                    field.serialize_name.as_deref().unwrap_or(&field.name.name),
                    *field_ty,
                )
            })
            .collect()
    }

    fn reflection_fields_for_bitfield_def(
        &mut self,
        def: &ast::BitfieldDef,
        namespace: Option<&str>,
        resolved_fields: &[TypeBitfieldFieldDef],
    ) -> Vec<ReflectionFieldInfo> {
        def.fields
            .iter()
            .zip(resolved_fields.iter())
            .enumerate()
            .map(|(index, (field, resolved_field))| {
                let ty = match &field.kind {
                    ast::BitfieldFieldKind::Bits {
                        as_type: Some(ty), ..
                    } => ty.clone(),
                    ast::BitfieldFieldKind::Bits {
                        width: 64,
                        as_type: None,
                    } => TypeExpr::Named(ast::Ident {
                        name: "uint64".to_string(),
                        span: field.span,
                    }),
                    ast::BitfieldFieldKind::Bits { as_type: None, .. } => {
                        TypeExpr::Named(ast::Ident {
                            name: "int64".to_string(),
                            span: field.span,
                        })
                    }
                    ast::BitfieldFieldKind::Payload(ty) => ty.clone(),
                };
                self.reflection_field_info_for_type_expr(
                    index,
                    &field.name.name,
                    &field.name.name,
                    &ty,
                    namespace,
                    resolved_field.ty,
                )
            })
            .collect()
    }

    fn reflection_bitfield_info_for_def(
        &mut self,
        def: &ast::BitfieldDef,
        namespace: Option<&str>,
        resolved_fields: &[TypeBitfieldFieldDef],
    ) -> ReflectionBitfieldInfo {
        let fields = def
            .fields
            .iter()
            .zip(resolved_fields.iter())
            .enumerate()
            .map(|(index, (field, resolved_field))| {
                let (shape, width, ty, enum_ty) = match &field.kind {
                    ast::BitfieldFieldKind::Bits { width, as_type } => {
                        let ty = as_type.clone().unwrap_or_else(|| {
                            let name = if *width == 64 { "uint64" } else { "int64" };
                            TypeExpr::Named(ast::Ident {
                                name: name.to_string(),
                                span: field.span,
                            })
                        });
                        ("bits", i64::from(*width), ty, as_type.as_ref())
                    }
                    ast::BitfieldFieldKind::Payload(ty) => ("payload", 0, ty.clone(), None),
                };
                let type_info =
                    self.reflection_type_info_for_type_expr(&ty, namespace, resolved_field.ty);
                let enum_type = enum_ty.map(|ty| {
                    let enum_ty = self.resolve_type_expr(ty);
                    self.reflection_type_info_for_type_expr(ty, namespace, enum_ty)
                });
                ReflectionBitfieldFieldInfo::new(
                    index,
                    &field.name.name,
                    shape,
                    width,
                    type_info,
                    enum_type,
                )
            })
            .collect();
        ReflectionBitfieldInfo::new(def.network_order, fields)
    }

    fn reflection_variants_for_enum_def(
        &mut self,
        def: &ast::EnumDef,
        namespace: Option<&str>,
        resolved_variants: &[VariantDef],
    ) -> Vec<ReflectionVariantInfo> {
        def.variants
            .iter()
            .zip(resolved_variants.iter())
            .enumerate()
            .map(|(variant_index, (variant, resolved_variant))| {
                let fields = variant
                    .fields
                    .iter()
                    .zip(resolved_variant.fields.iter())
                    .enumerate()
                    .map(|(field_index, (field, (_, field_ty)))| {
                        self.reflection_field_info_for_type_expr(
                            field_index,
                            &field.name.name,
                            field.serialize_name.as_deref().unwrap_or(&field.name.name),
                            &field.ty,
                            namespace,
                            *field_ty,
                        )
                    })
                    .collect::<Vec<_>>();
                let has_secret = fields.iter().any(|field| field.has_secret);
                ReflectionVariantInfo::new(
                    variant_index,
                    &variant.name.name,
                    resolved_variant.discriminant,
                    has_secret,
                    fields,
                )
            })
            .collect()
    }

    fn reflection_field_info_for_type_expr(
        &mut self,
        index: usize,
        name: &str,
        serialize_name: &str,
        ty: &TypeExpr,
        namespace: Option<&str>,
        resolved_ty: TypeId,
    ) -> ReflectionFieldInfo {
        let type_info = self.reflection_type_info_for_type_expr(ty, namespace, resolved_ty);
        Self::reflection_field_info_from_type_info(index, name, serialize_name, type_info)
    }

    fn reflection_field_info_for_type_id(
        &self,
        index: usize,
        name: &str,
        serialize_name: &str,
        ty: TypeId,
    ) -> ReflectionFieldInfo {
        let type_info = self.reflection_type_info_for_type(ty);
        Self::reflection_field_info_from_type_info(index, name, serialize_name, type_info)
    }

    fn reflection_field_info_from_type_info(
        index: usize,
        name: &str,
        serialize_name: &str,
        type_info: ReflectionTypeInfo,
    ) -> ReflectionFieldInfo {
        ReflectionFieldInfo::new(
            index,
            name,
            type_info.type_name.clone(),
            type_info.kind.clone(),
            serialize_name,
            type_info.has_secret,
            type_info,
        )
    }

    fn reflection_type_info_for_type_expr(
        &mut self,
        ty: &TypeExpr,
        namespace: Option<&str>,
        resolved_ty: TypeId,
    ) -> ReflectionTypeInfo {
        if let TypeExpr::View(inner, _) = ty {
            return self.reflection_type_info_for_type_expr(inner, namespace, resolved_ty);
        }

        let type_name = self.reflection_type_expr_display(ty, namespace);
        let args = self.reflection_type_info_args_for_type_expr(ty, namespace, resolved_ty);
        ReflectionTypeInfo::new(
            type_name,
            self.reflection_kind_for_type_expr(ty, namespace, resolved_ty),
            self.reflection_primitive_tag_for_type_expr_static(ty, resolved_ty),
            resolved_ty != TypeInterner::ERROR && self.type_contains_secret_data(resolved_ty),
            args,
        )
    }

    fn reflection_type_info_args_for_type_expr(
        &mut self,
        ty: &TypeExpr,
        namespace: Option<&str>,
        resolved_ty: TypeId,
    ) -> Vec<ReflectionTypeInfo> {
        match ty {
            TypeExpr::View(inner, _) => {
                self.reflection_type_info_args_for_type_expr(inner, namespace, resolved_ty)
            }
            TypeExpr::StateQualified(_, _, _) => Vec::new(),
            TypeExpr::Named(ident) => {
                let display_name = self.reflection_type_name_in_namespace(ident, namespace);
                if let Some(alias) = self.type_aliases.get(&display_name).cloned() {
                    let alias_namespace = display_name
                        .rsplit_once('.')
                        .map(|(namespace, _)| namespace)
                        .or(namespace);
                    let base_ty = self.resolve_type_expr(&alias.base_type);
                    if base_ty == TypeInterner::ERROR {
                        Vec::new()
                    } else {
                        vec![self.reflection_type_info_for_type_expr(
                            &alias.base_type,
                            alias_namespace,
                            base_ty,
                        )]
                    }
                } else {
                    self.type_info_arg_types_for_type(resolved_ty)
                        .into_iter()
                        .map(|arg| self.reflection_type_info_for_type(arg))
                        .collect()
                }
            }
            TypeExpr::Generic(_, args, _) => args
                .iter()
                .map(|arg| {
                    let arg_ty = self.resolve_type_expr(arg);
                    self.reflection_type_info_for_type_expr(arg, namespace, arg_ty)
                })
                .collect(),
            TypeExpr::Function(params, return_type, _) => params
                .iter()
                .chain(std::iter::once(return_type.as_ref()))
                .map(|arg| {
                    let arg_ty = self.resolve_type_expr(arg);
                    self.reflection_type_info_for_type_expr(arg, namespace, arg_ty)
                })
                .collect(),
        }
    }

    fn reflection_kind_for_type_expr(
        &self,
        ty: &TypeExpr,
        namespace: Option<&str>,
        resolved_ty: TypeId,
    ) -> &'static str {
        match ty {
            TypeExpr::View(inner, _) => {
                self.reflection_kind_for_type_expr(inner, namespace, resolved_ty)
            }
            TypeExpr::StateQualified(_, _, _) => self.reflection_kind_for_type(resolved_ty),
            TypeExpr::Named(ident) => {
                let name = self.reflection_type_name_in_namespace(ident, namespace);
                if let Some(alias) = self.type_aliases.get(&name) {
                    if alias.constraint.is_some() {
                        "refinement"
                    } else {
                        "alias"
                    }
                } else if Self::reflection_primitive_tag_for_name(&name).is_some() {
                    "primitive"
                } else {
                    self.reflection_kind_for_type(resolved_ty)
                }
            }
            TypeExpr::Generic(ident, _, _) => {
                let name = self.reflection_type_name_in_namespace(ident, namespace);
                match name.as_str() {
                    "list" => "list",
                    "map" => "map",
                    "set" => "set",
                    "optional" => "optional",
                    "result" => "result",
                    "secret" => "secret",
                    _ if self.generic_struct_templates.contains_key(&name) => "struct",
                    _ => self.reflection_kind_for_type(resolved_ty),
                }
            }
            TypeExpr::Function(_, _, _) => "function",
        }
    }

    fn reflection_primitive_tag_for_name(name: &str) -> Option<&'static str> {
        match name {
            "int8" => Some("int8_type"),
            "int16" => Some("int16_type"),
            "int32" => Some("int32_type"),
            "int64" => Some("int64_type"),
            "uint8" => Some("uint8_type"),
            "uint16" => Some("uint16_type"),
            "uint32" => Some("uint32_type"),
            "uint64" => Some("uint64_type"),
            "float32" => Some("float32_type"),
            "float64" => Some("float64_type"),
            "string" => Some("string_type"),
            "bool" => Some("bool_type"),
            "bytes" => Some("bytes_type"),
            "nothing" => Some("nothing_type"),
            "TypeConstruction" => Some("type_construction_type"),
            _ => None,
        }
    }

    fn reflection_type_expr_display(&self, ty: &TypeExpr, namespace: Option<&str>) -> String {
        match ty {
            TypeExpr::Named(ident) => self.reflection_type_name_in_namespace(ident, namespace),
            TypeExpr::Generic(ident, args, _) => {
                let args = args
                    .iter()
                    .map(|arg| self.reflection_type_expr_display(arg, namespace))
                    .collect::<Vec<_>>();
                format!(
                    "{}[{}]",
                    self.reflection_type_name_in_namespace(ident, namespace),
                    args.join(", ")
                )
            }
            TypeExpr::View(inner, _) => {
                format!(
                    "view {}",
                    self.reflection_type_expr_display(inner, namespace)
                )
            }
            TypeExpr::StateQualified(inner, state, _) => {
                format!(
                    "{} at {}",
                    self.reflection_type_expr_display(inner, namespace),
                    state.name
                )
            }
            TypeExpr::Function(params, return_type, _) => {
                let params = params
                    .iter()
                    .map(|param| self.reflection_type_expr_display(param, namespace))
                    .collect::<Vec<_>>();
                format!(
                    "function({}) returns {}",
                    params.join(", "),
                    self.reflection_type_expr_display(return_type, namespace)
                )
            }
        }
    }

    fn reflection_type_name_in_namespace(
        &self,
        ident: &ast::Ident,
        namespace: Option<&str>,
    ) -> String {
        if ident.name.contains('.') {
            return ident.name.clone();
        }
        if Self::reflection_primitive_tag_for_name(&ident.name).is_some() {
            return ident.name.clone();
        }
        if let Some(namespace) = namespace {
            let qualified = format!("{namespace}.{}", ident.name);
            if self.reflection_type_name_is_registered(&qualified) {
                return qualified;
            }
        }
        ident.name.clone()
    }

    fn reflection_type_name_is_registered(&self, name: &str) -> bool {
        self.named_types.contains_key(name)
            || self.type_aliases.contains_key(name)
            || self.generic_struct_templates.contains_key(name)
    }

    fn secret_field_names(&self, ty: TypeId) -> Vec<String> {
        match self.interner.resolve(ty) {
            Type::Struct(sid) => self
                .interner
                .resolve_struct(*sid)
                .fields
                .iter()
                .filter(|(_, field_ty)| self.type_contains_secret_data(*field_ty))
                .map(|(name, _)| name.clone())
                .collect(),
            Type::Bitfield(bid) => self
                .interner
                .resolve_bitfield(*bid)
                .fields
                .iter()
                .filter(|field| self.type_contains_secret_data(field.ty))
                .map(|field| field.name.clone())
                .collect(),
            _ => Vec::new(),
        }
    }

    fn json_non_string_map_key_types(&self, ty: TypeId) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut key_types = HashSet::new();
        let mut keys = Vec::new();
        self.collect_json_non_string_map_key_types(ty, &mut visited, &mut key_types, &mut keys);
        keys
    }

    fn json_unsupported_serialize_types(&self, ty: TypeId) -> Vec<String> {
        self.json_unsupported_data_types(ty, false, true)
    }

    fn json_unsupported_parse_types(&self, ty: TypeId) -> Vec<String> {
        self.json_unsupported_data_types(ty, true, true)
    }

    fn json_unsupported_data_types(
        &self,
        ty: TypeId,
        descend_into_secret: bool,
        allow_machine_values: bool,
    ) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut unsupported_types = HashSet::new();
        let mut unsupported = Vec::new();
        self.collect_json_unsupported_data_types(
            ty,
            descend_into_secret,
            allow_machine_values,
            &mut visited,
            &mut unsupported_types,
            &mut unsupported,
        );
        unsupported
    }

    fn push_json_unsupported_type(
        &self,
        ty: TypeId,
        unsupported_types: &mut HashSet<TypeId>,
        unsupported: &mut Vec<String>,
    ) {
        if unsupported_types.insert(ty) {
            unsupported.push(self.type_name(ty));
        }
    }

    fn collect_json_unsupported_data_types(
        &self,
        ty: TypeId,
        descend_into_secret: bool,
        allow_machine_values: bool,
        visited: &mut HashSet<TypeId>,
        unsupported_types: &mut HashSet<TypeId>,
        unsupported: &mut Vec<String>,
    ) {
        if !visited.insert(ty) {
            return;
        }

        match self.interner.resolve(ty) {
            Type::Int8
            | Type::Int16
            | Type::Int32
            | Type::Int64
            | Type::Uint8
            | Type::Uint16
            | Type::Uint32
            | Type::Uint64
            | Type::Float32
            | Type::Float64
            | Type::String
            | Type::Bool
            | Type::Bytes
            | Type::Nothing
            | Type::Error => {}
            Type::TypeConstruction
            | Type::Interface(_)
            | Type::Actor(_)
            | Type::Function { .. } => {
                self.push_json_unsupported_type(ty, unsupported_types, unsupported);
            }
            Type::Machine(mid) => {
                if !allow_machine_values {
                    self.push_json_unsupported_type(ty, unsupported_types, unsupported);
                    return;
                }
                for (_, field_ty) in self
                    .interner
                    .resolve_machine(*mid)
                    .states
                    .iter()
                    .flat_map(|state| state.fields.iter())
                {
                    self.collect_json_unsupported_data_types(
                        *field_ty,
                        descend_into_secret,
                        allow_machine_values,
                        visited,
                        unsupported_types,
                        unsupported,
                    );
                }
            }
            Type::MachineState { machine, state } => {
                if !allow_machine_values {
                    self.push_json_unsupported_type(ty, unsupported_types, unsupported);
                    return;
                }
                if let Some(state_def) = self.interner.resolve_machine(*machine).state(*state) {
                    for (_, field_ty) in &state_def.fields {
                        self.collect_json_unsupported_data_types(
                            *field_ty,
                            descend_into_secret,
                            allow_machine_values,
                            visited,
                            unsupported_types,
                            unsupported,
                        );
                    }
                }
            }
            Type::List(inner) | Type::Set(inner) | Type::Optional(inner) => self
                .collect_json_unsupported_data_types(
                    *inner,
                    descend_into_secret,
                    allow_machine_values,
                    visited,
                    unsupported_types,
                    unsupported,
                ),
            Type::Secret(inner) => {
                if descend_into_secret {
                    self.collect_json_unsupported_data_types(
                        *inner,
                        descend_into_secret,
                        allow_machine_values,
                        visited,
                        unsupported_types,
                        unsupported,
                    );
                }
            }
            Type::Map(_, value) => self.collect_json_unsupported_data_types(
                *value,
                descend_into_secret,
                allow_machine_values,
                visited,
                unsupported_types,
                unsupported,
            ),
            Type::Result(ok, err) => {
                self.collect_json_unsupported_data_types(
                    *ok,
                    descend_into_secret,
                    allow_machine_values,
                    visited,
                    unsupported_types,
                    unsupported,
                );
                self.collect_json_unsupported_data_types(
                    *err,
                    descend_into_secret,
                    allow_machine_values,
                    visited,
                    unsupported_types,
                    unsupported,
                );
            }
            Type::Struct(sid) => {
                for (_, field_ty) in &self.interner.resolve_struct(*sid).fields {
                    self.collect_json_unsupported_data_types(
                        *field_ty,
                        descend_into_secret,
                        allow_machine_values,
                        visited,
                        unsupported_types,
                        unsupported,
                    );
                }
            }
            Type::Bitfield(bid) => {
                for field in &self.interner.resolve_bitfield(*bid).fields {
                    self.collect_json_unsupported_data_types(
                        field.ty,
                        descend_into_secret,
                        allow_machine_values,
                        visited,
                        unsupported_types,
                        unsupported,
                    );
                }
            }
            Type::Enum(eid) => {
                for (_, field_ty) in self
                    .interner
                    .resolve_enum(*eid)
                    .variants
                    .iter()
                    .flat_map(|variant| variant.fields.iter())
                {
                    self.collect_json_unsupported_data_types(
                        *field_ty,
                        descend_into_secret,
                        allow_machine_values,
                        visited,
                        unsupported_types,
                        unsupported,
                    );
                }
            }
            Type::Refinement { base, .. } => self.collect_json_unsupported_data_types(
                *base,
                descend_into_secret,
                allow_machine_values,
                visited,
                unsupported_types,
                unsupported,
            ),
        }
    }

    fn collect_json_non_string_map_key_types(
        &self,
        ty: TypeId,
        visited: &mut HashSet<TypeId>,
        key_types: &mut HashSet<TypeId>,
        keys: &mut Vec<String>,
    ) {
        if !visited.insert(ty) {
            return;
        }

        match self.interner.resolve(ty) {
            Type::List(inner) | Type::Set(inner) | Type::Optional(inner) | Type::Secret(inner) => {
                self.collect_json_non_string_map_key_types(*inner, visited, key_types, keys);
            }
            Type::Map(key, value) => {
                if *key != TypeInterner::STRING && key_types.insert(*key) {
                    keys.push(self.type_name(*key));
                }
                self.collect_json_non_string_map_key_types(*key, visited, key_types, keys);
                self.collect_json_non_string_map_key_types(*value, visited, key_types, keys);
            }
            Type::Result(ok, err) => {
                self.collect_json_non_string_map_key_types(*ok, visited, key_types, keys);
                self.collect_json_non_string_map_key_types(*err, visited, key_types, keys);
            }
            Type::Struct(sid) => {
                for (_, field_ty) in &self.interner.resolve_struct(*sid).fields {
                    self.collect_json_non_string_map_key_types(*field_ty, visited, key_types, keys);
                }
            }
            Type::Bitfield(bid) => {
                for field in &self.interner.resolve_bitfield(*bid).fields {
                    self.collect_json_non_string_map_key_types(field.ty, visited, key_types, keys);
                }
            }
            Type::Enum(eid) => {
                for (_, field_ty) in self
                    .interner
                    .resolve_enum(*eid)
                    .variants
                    .iter()
                    .flat_map(|variant| variant.fields.iter())
                {
                    self.collect_json_non_string_map_key_types(*field_ty, visited, key_types, keys);
                }
            }
            Type::Machine(mid) => {
                for (_, field_ty) in self
                    .interner
                    .resolve_machine(*mid)
                    .states
                    .iter()
                    .flat_map(|state| state.fields.iter())
                {
                    self.collect_json_non_string_map_key_types(*field_ty, visited, key_types, keys);
                }
            }
            Type::MachineState { machine, state } => {
                if let Some(state_def) = self.interner.resolve_machine(*machine).state(*state) {
                    for (_, field_ty) in &state_def.fields {
                        self.collect_json_non_string_map_key_types(
                            *field_ty, visited, key_types, keys,
                        );
                    }
                }
            }
            Type::Refinement { base, .. } => {
                self.collect_json_non_string_map_key_types(*base, visited, key_types, keys);
            }
            _ => {}
        }
    }

    fn check_json_public_call_policy(
        &mut self,
        callee_name: Option<&str>,
        checked_arg_types: &[TypeId],
        args: &[ast::CallArg],
        return_type: TypeId,
    ) {
        let Some(callee_name) = callee_name else {
            return;
        };

        match callee_name {
            "json.serialize" => {
                let Some((&value_ty, arg)) = checked_arg_types.first().zip(args.first()) else {
                    return;
                };
                if self.type_contains_secret_data(value_ty) {
                    self.sink.emit(errors::type_contains_secret_data(
                        "json.serialize",
                        &self.type_name(value_ty),
                        &self.secret_field_names(value_ty),
                        arg.value.span(),
                    ));
                }
                self.check_json_public_serialize_policy(callee_name, value_ty, arg);
            }
            "json.serialize_public" => {
                let Some((&value_ty, arg)) = checked_arg_types.first().zip(args.first()) else {
                    return;
                };
                if !self.is_secret_type(value_ty)
                    && self.type_contains_secret_data(value_ty)
                    && !self.json_public_projection_allows_secret_data(value_ty)
                {
                    self.sink.emit(errors::type_contains_secret_data(
                        "json.serialize_public",
                        &self.type_name(value_ty),
                        &self.secret_field_names(value_ty),
                        arg.value.span(),
                    ));
                }
                self.check_json_public_serialize_policy(callee_name, value_ty, arg);
            }
            "json.parse" | "json.parse_exact" => {
                let Some(arg) = args.first() else {
                    return;
                };
                if let Type::Result(parsed_ty, _) = self.interner.resolve(return_type) {
                    for key_type in self.json_non_string_map_key_types(*parsed_ty) {
                        self.sink.emit(errors::json_map_key_must_be_string(
                            &key_type,
                            arg.value.span(),
                        ));
                    }
                    for unsupported_type in self.json_unsupported_parse_types(*parsed_ty) {
                        self.sink.emit(errors::json_unsupported_parse_type(
                            callee_name,
                            &unsupported_type,
                            arg.value.span(),
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    fn check_json_public_serialize_policy(
        &mut self,
        function_name: &str,
        value_ty: TypeId,
        arg: &ast::CallArg,
    ) {
        if !matches!(&arg.value, Expr::View(_, _)) && self.json_read_requires_view(value_ty) {
            self.sink.emit(errors::json_serialize_requires_view(
                function_name,
                &self.type_name(value_ty),
                arg.value.span(),
            ));
        }

        for key_type in self.json_non_string_map_key_types(value_ty) {
            self.sink.emit(errors::json_map_key_must_be_string(
                &key_type,
                arg.value.span(),
            ));
        }

        for unsupported_type in self.json_unsupported_serialize_types(value_ty) {
            self.sink.emit(errors::json_unsupported_serialize_type(
                function_name,
                &unsupported_type,
                arg.value.span(),
            ));
        }
    }

    fn types_compatible(&self, expected: TypeId, got: TypeId) -> bool {
        if expected == got || expected == TypeInterner::ERROR || got == TypeInterner::ERROR {
            return true;
        }

        match (self.interner.resolve(expected), self.interner.resolve(got)) {
            (Type::Interface(_), Type::Interface(_)) => expected == got,
            (Type::Interface(_), _) => self.interface_impls.contains_key(&(expected, got)),
            (Type::Secret(expected_inner), Type::Secret(got_inner)) => {
                self.types_compatible(*expected_inner, *got_inner)
            }
            (Type::Secret(expected_inner), _) => self.types_compatible(*expected_inner, got),
            (Type::List(expected_inner), Type::List(got_inner))
            | (Type::Optional(expected_inner), Type::Optional(got_inner)) => {
                self.types_compatible(*expected_inner, *got_inner)
            }
            (Type::Set(expected_inner), Type::Set(got_inner)) => {
                self.types_compatible(*expected_inner, *got_inner)
            }
            (Type::Map(expected_key, expected_val), Type::Map(got_key, got_val))
            | (Type::Result(expected_key, expected_val), Type::Result(got_key, got_val)) => {
                self.types_compatible(*expected_key, *got_key)
                    && self.types_compatible(*expected_val, *got_val)
            }
            (
                Type::Function {
                    params: expected_params,
                    return_type: expected_return,
                },
                Type::Function {
                    params: got_params,
                    return_type: got_return,
                },
            ) => {
                expected_params.len() == got_params.len()
                    && expected_params
                        .iter()
                        .zip(got_params.iter())
                        .all(|(expected, got)| self.types_compatible(*expected, *got))
                    && self.types_compatible(*expected_return, *got_return)
            }
            (Type::Machine(expected_machine), Type::MachineState { machine, .. }) => {
                expected_machine == machine
            }
            _ => false,
        }
    }

    /// Emit a builtin type-argument arity error without hiding the call shape.
    fn expect_no_type_args(&mut self, builtin_name: &str, type_args: &[TypeExpr], span: Span) {
        if !type_args.is_empty() {
            self.sink.emit(errors::unknown_type(
                &format!(
                    "{builtin_name} (expected 0 type arguments, got {})",
                    type_args.len()
                ),
                span,
            ));
        }
    }

    fn no_type_args_signature(
        &mut self,
        builtin_name: &str,
        type_args: &[TypeExpr],
        span: Span,
        params: Vec<TypeId>,
        return_type: TypeId,
    ) -> Option<(Vec<TypeId>, TypeId)> {
        self.expect_no_type_args(builtin_name, type_args, span);
        Some((params, return_type))
    }

    /// Extract T from an optional single-argument builtin type list.
    /// Uses ERROR as a wildcard when type args are absent.
    fn optional_type_arg(
        &mut self,
        builtin_name: &str,
        type_args: &[TypeExpr],
        span: Span,
    ) -> TypeId {
        match type_args.len() {
            0 => TypeInterner::ERROR,
            1 => self.resolve_type_expr(&type_args[0]),
            got => {
                self.sink.emit(errors::unknown_type(
                    &format!("{builtin_name} (expected 0 or 1 type arguments, got {got})"),
                    span,
                ));
                TypeInterner::ERROR
            }
        }
    }

    /// Extract (A, B) from an optional two-argument builtin type list.
    /// Uses ERROR as a wildcard when type args are absent.
    fn optional_two_type_args(
        &mut self,
        builtin_name: &str,
        type_args: &[TypeExpr],
        span: Span,
    ) -> (TypeId, TypeId) {
        match type_args.len() {
            0 => (TypeInterner::ERROR, TypeInterner::ERROR),
            2 => (
                self.resolve_type_expr(&type_args[0]),
                self.resolve_type_expr(&type_args[1]),
            ),
            got => {
                self.sink.emit(errors::unknown_type(
                    &format!("{builtin_name} (expected 0 or 2 type arguments, got {got})"),
                    span,
                ));
                (TypeInterner::ERROR, TypeInterner::ERROR)
            }
        }
    }

    /// Extract (A, B, C) from an optional three-argument builtin type list.
    /// Uses ERROR as a wildcard when type args are absent.
    fn optional_three_type_args(
        &mut self,
        builtin_name: &str,
        type_args: &[TypeExpr],
        span: Span,
    ) -> (TypeId, TypeId, TypeId) {
        match type_args.len() {
            0 => (
                TypeInterner::ERROR,
                TypeInterner::ERROR,
                TypeInterner::ERROR,
            ),
            3 => (
                self.resolve_type_expr(&type_args[0]),
                self.resolve_type_expr(&type_args[1]),
                self.resolve_type_expr(&type_args[2]),
            ),
            got => {
                self.sink.emit(errors::unknown_type(
                    &format!("{builtin_name} (expected 0 or 3 type arguments, got {got})"),
                    span,
                ));
                (
                    TypeInterner::ERROR,
                    TypeInterner::ERROR,
                    TypeInterner::ERROR,
                )
            }
        }
    }

    fn function_type(&mut self, params: Vec<TypeId>, return_type: TypeId) -> TypeId {
        self.interner.intern(Type::Function {
            params,
            return_type,
        })
    }

    fn map_values_type_args(
        &mut self,
        builtin_name: &str,
        type_args: &[TypeExpr],
        span: Span,
    ) -> (TypeId, TypeId, TypeId) {
        match type_args.len() {
            0 => (
                TypeInterner::ERROR,
                TypeInterner::ERROR,
                TypeInterner::ERROR,
            ),
            2 => {
                let (key, value) = self.optional_two_type_args(builtin_name, type_args, span);
                (key, value, value)
            }
            3 => self.optional_three_type_args(builtin_name, type_args, span),
            got => {
                self.sink.emit(errors::unknown_type(
                    &format!("{builtin_name} (expected 0, 2, or 3 type arguments, got {got})"),
                    span,
                ));
                (
                    TypeInterner::ERROR,
                    TypeInterner::ERROR,
                    TypeInterner::ERROR,
                )
            }
        }
    }

    /// Extract (key_type, value_type) from map builtin type args.
    /// Uses ERROR as a wildcard when type args are absent (matches any map).
    fn map_type_args(
        &mut self,
        builtin_name: &str,
        type_args: &[TypeExpr],
        span: Span,
    ) -> (TypeId, TypeId) {
        self.optional_two_type_args(builtin_name, type_args, span)
    }

    /// Extract T from set builtin type args.
    /// Uses ERROR as a wildcard when type args are absent.
    fn set_type_arg(&mut self, builtin_name: &str, type_args: &[TypeExpr], span: Span) -> TypeId {
        self.optional_type_arg(builtin_name, type_args, span)
    }

    fn extract_dotted_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Ident(ident) => Some(ident.name.clone()),
            Expr::FieldAccess(inner, field, _) => {
                let prefix = Self::extract_dotted_name(inner)?;
                Some(format!("{prefix}.{}", field.name))
            }
            _ => None,
        }
    }

    fn resolved_expr_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Ident(ident) => Some(self.resolved_symbol_name(&ident.name, ident.span)),
            Expr::FieldAccess(_, _, _) => self.expanded_dotted_expr_name(expr),
            _ => None,
        }
    }

    fn builtin_signature(
        &mut self,
        callee: &Expr,
        type_args: &[TypeExpr],
        span: Span,
    ) -> Option<(Vec<TypeId>, TypeId)> {
        let name = self.resolved_expr_name(callee)?;
        if let Some((type_name, method_name)) = name.rsplit_once('.')
            && let Some(&type_id) = self.named_types.get(type_name)
            && matches!(self.interner.resolve(type_id), Type::Bitfield(_))
        {
            match method_name {
                "to_bytes" => {
                    if !type_args.is_empty() {
                        self.sink.emit(errors::unknown_type(
                            &format!(
                                "{name} (expected 0 type arguments, got {})",
                                type_args.len()
                            ),
                            span,
                        ));
                    }
                    return Some((vec![type_id], TypeInterner::BYTES));
                }
                "from_bytes" => {
                    if !type_args.is_empty() {
                        self.sink.emit(errors::unknown_type(
                            &format!(
                                "{name} (expected 0 type arguments, got {})",
                                type_args.len()
                            ),
                            span,
                        ));
                    }
                    return Some((
                        vec![TypeInterner::BYTES],
                        self.interner
                            .intern(Type::Result(type_id, TypeInterner::STRING)),
                    ));
                }
                _ => {}
            }
        }

        match name.as_str() {
            "int64.from_string" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![TypeInterner::STRING],
                    self.interner
                        .intern(Type::Result(TypeInterner::INT64, TypeInterner::STRING)),
                ))
            }
            "uint64.from_string" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![TypeInterner::STRING],
                    self.interner
                        .intern(Type::Result(TypeInterner::UINT64, TypeInterner::STRING)),
                ))
            }
            "float64.from_string" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![TypeInterner::STRING],
                    self.interner
                        .intern(Type::Result(TypeInterner::FLOAT64, TypeInterner::STRING)),
                ))
            }
            "string.from_int64" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::INT64],
                TypeInterner::STRING,
            ),
            "string.from_uint64" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::UINT64],
                TypeInterner::STRING,
            ),
            "string.from_float64" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::FLOAT64],
                TypeInterner::STRING,
            ),
            "string.from_bool" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::BOOL],
                TypeInterner::STRING,
            ),
            "float64.from_int64" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::INT64],
                TypeInterner::FLOAT64,
            ),
            "string.length" | "string.char_count" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::INT64,
            ),
            "string.contains" | "string.starts_with" | "string.ends_with" => self
                .no_type_args_signature(
                    &name,
                    type_args,
                    span,
                    vec![TypeInterner::STRING, TypeInterner::STRING],
                    TypeInterner::BOOL,
                ),
            "string.trim" | "string.upper" | "string.lower" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::STRING,
            ),
            "string.replace" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![
                    TypeInterner::STRING,
                    TypeInterner::STRING,
                    TypeInterner::STRING,
                ],
                TypeInterner::STRING,
            ),
            "string.split" => {
                self.expect_no_type_args(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(TypeInterner::STRING));
                Some((vec![TypeInterner::STRING, TypeInterner::STRING], list_ty))
            }
            "string.join" => {
                self.expect_no_type_args(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(TypeInterner::STRING));
                Some((vec![list_ty, TypeInterner::STRING], TypeInterner::STRING))
            }
            "Environment.args" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![TypeInterner::ERROR],
                    self.interner.intern(Type::List(TypeInterner::STRING)),
                ))
            }
            "Filesystem.read_file" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![TypeInterner::ERROR, TypeInterner::STRING],
                    self.interner
                        .intern(Type::Result(TypeInterner::STRING, TypeInterner::STRING)),
                ))
            }
            "Filesystem.write_file" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![
                        TypeInterner::ERROR,
                        TypeInterner::STRING,
                        TypeInterner::STRING,
                    ],
                    self.interner
                        .intern(Type::Result(TypeInterner::NOTHING, TypeInterner::STRING)),
                ))
            }
            "Stdout.write" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::ERROR, TypeInterner::STRING],
                TypeInterner::NOTHING,
            ),
            "json.parse" | "json.parse_exact" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![TypeInterner::STRING], TypeInterner::ERROR));
                }

                let value_ty = self.resolve_type_expr(&type_args[0]);
                let result_ty = self
                    .interner
                    .intern(Type::Result(value_ty, TypeInterner::STRING));
                Some((vec![TypeInterner::STRING], result_ty))
            }
            "json.serialize" | "json.serialize_public" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![TypeInterner::ERROR], TypeInterner::ERROR));
                }

                let value_ty = self.resolve_type_expr(&type_args[0]);
                Some((vec![value_ty], TypeInterner::STRING))
            }
            "type.name" | "type.kind" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                Some((vec![], TypeInterner::STRING))
            }
            "type.kind_tag" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                Some((
                    vec![],
                    self.named_types
                        .get("TypeKind")
                        .copied()
                        .unwrap_or(TypeInterner::ERROR),
                ))
            }
            "type.primitive_tag" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let primitive_ty = self
                    .named_types
                    .get("TypePrimitive")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![], self.interner.intern(Type::Optional(primitive_ty))))
            }
            "type.has_secret" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                Some((vec![], TypeInterner::BOOL))
            }
            "type.info" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_info_ty = self
                    .named_types
                    .get("TypeInfo")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![], type_info_ty))
            }
            "type.arg" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![TypeInterner::INT64], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_info_ty = self
                    .named_types
                    .get("TypeInfo")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![TypeInterner::INT64], type_info_ty))
            }
            "type.construct_start" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let target_ty = self.resolve_type_expr(&type_args[0]);
                if !matches!(
                    self.interner.resolve(target_ty),
                    Type::Struct(_) | Type::Bitfield(_)
                ) {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} supports only structs and bitfields, got {}",
                            self.type_name(target_ty)
                        ),
                        span,
                    ));
                }
                Some((vec![], TypeInterner::TYPE_CONSTRUCTION))
            }
            "type.construct_variant_start" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![TypeInterner::ERROR], TypeInterner::ERROR));
                }
                let target_ty = self.resolve_type_expr(&type_args[0]);
                if !matches!(self.interner.resolve(target_ty), Type::Enum(_)) {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} supports only enums, got {}",
                            self.type_name(target_ty)
                        ),
                        span,
                    ));
                }
                let type_variant_ty = self
                    .named_types
                    .get("TypeVariant")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((
                    vec![type_variant_ty],
                    self.interner.intern(Type::Result(
                        TypeInterner::TYPE_CONSTRUCTION,
                        TypeInterner::STRING,
                    )),
                ))
            }
            "type.construct_machine_start" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![TypeInterner::ERROR], TypeInterner::ERROR));
                }
                let target_ty = self.resolve_type_expr(&type_args[0]);
                if !matches!(
                    self.interner.resolve(target_ty),
                    Type::Machine(_) | Type::MachineState { .. }
                ) {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} supports only machines and machine states, got {}",
                            self.type_name(target_ty)
                        ),
                        span,
                    ));
                }
                let type_machine_state_ty = self
                    .named_types
                    .get("TypeMachineState")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((
                    vec![type_machine_state_ty],
                    self.interner.intern(Type::Result(
                        TypeInterner::TYPE_CONSTRUCTION,
                        TypeInterner::STRING,
                    )),
                ))
            }
            "type.construct_put" => {
                if type_args.len() != 2 {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} (expected 2 type arguments, got {})",
                            type_args.len()
                        ),
                        span,
                    ));
                    return Some((
                        vec![
                            TypeInterner::TYPE_CONSTRUCTION,
                            self.named_types
                                .get("TypeField")
                                .copied()
                                .unwrap_or(TypeInterner::ERROR),
                            TypeInterner::ERROR,
                        ],
                        TypeInterner::ERROR,
                    ));
                }
                let target_ty = self.resolve_type_expr(&type_args[0]);
                if !matches!(
                    self.interner.resolve(target_ty),
                    Type::Struct(_)
                        | Type::Bitfield(_)
                        | Type::Enum(_)
                        | Type::Machine(_)
                        | Type::MachineState { .. }
                ) {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} supports only structs, bitfields, enums, and machines, got {}",
                            self.type_name(target_ty)
                        ),
                        span,
                    ));
                }
                let field_ty = self.resolve_type_expr(&type_args[1]);
                let type_field_ty = self
                    .named_types
                    .get("TypeField")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((
                    vec![TypeInterner::TYPE_CONSTRUCTION, type_field_ty, field_ty],
                    self.interner.intern(Type::Result(
                        TypeInterner::TYPE_CONSTRUCTION,
                        TypeInterner::STRING,
                    )),
                ))
            }
            "type.construct_finish" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![TypeInterner::TYPE_CONSTRUCTION], TypeInterner::ERROR));
                }
                let target_ty = self.resolve_type_expr(&type_args[0]);
                if !matches!(
                    self.interner.resolve(target_ty),
                    Type::Struct(_)
                        | Type::Bitfield(_)
                        | Type::Enum(_)
                        | Type::Machine(_)
                        | Type::MachineState { .. }
                ) {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} supports only structs, bitfields, enums, and machines, got {}",
                            self.type_name(target_ty)
                        ),
                        span,
                    ));
                }
                Some((
                    vec![TypeInterner::TYPE_CONSTRUCTION],
                    self.interner
                        .intern(Type::Result(target_ty, TypeInterner::STRING)),
                ))
            }
            "type.fields" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_field_ty = self
                    .named_types
                    .get("TypeField")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![], self.interner.intern(Type::List(type_field_ty))))
            }
            "type.bitfield_layout" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_bitfield_ty = self
                    .named_types
                    .get("TypeBitfield")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![], type_bitfield_ty))
            }
            "type.bitfield_fields" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_bitfield_field_ty = self
                    .named_types
                    .get("TypeBitfieldField")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((
                    vec![],
                    self.interner.intern(Type::List(type_bitfield_field_ty)),
                ))
            }
            "type.machine_layout" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_machine_ty = self
                    .named_types
                    .get("TypeMachine")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![], type_machine_ty))
            }
            "type.machine_states" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_machine_state_ty = self
                    .named_types
                    .get("TypeMachineState")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((
                    vec![],
                    self.interner.intern(Type::List(type_machine_state_ty)),
                ))
            }
            "type.machine_transitions" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_machine_transition_ty = self
                    .named_types
                    .get("TypeMachineTransition")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((
                    vec![],
                    self.interner.intern(Type::List(type_machine_transition_ty)),
                ))
            }
            "type.machine_state_value" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![TypeInterner::ERROR], TypeInterner::ERROR));
                }

                let value_ty = self.resolve_type_expr(&type_args[0]);
                let type_machine_state_ty = self
                    .named_types
                    .get("TypeMachineState")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![value_ty], type_machine_state_ty))
            }
            "type.variants" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let _ = self.resolve_type_expr(&type_args[0]);
                let type_variant_ty = self
                    .named_types
                    .get("TypeVariant")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![], self.interner.intern(Type::List(type_variant_ty))))
            }
            "type.variant_value" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!("{name} (expected 1 type argument, got {})", type_args.len()),
                        span,
                    ));
                    return Some((vec![TypeInterner::ERROR], TypeInterner::ERROR));
                }

                let value_ty = self.resolve_type_expr(&type_args[0]);
                let type_variant_ty = self
                    .named_types
                    .get("TypeVariant")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![value_ty], type_variant_ty))
            }
            "type.field_value" => {
                if type_args.len() != 2 {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} (expected 2 type arguments, got {})",
                            type_args.len()
                        ),
                        span,
                    ));
                    return Some((vec![TypeInterner::ERROR], TypeInterner::ERROR));
                }

                let value_ty = self.resolve_type_expr(&type_args[0]);
                let return_ty = self.resolve_type_expr(&type_args[1]);
                let type_field_ty = self
                    .named_types
                    .get("TypeField")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![value_ty, type_field_ty], return_ty))
            }
            "type.machine_field_value" => {
                if type_args.len() != 2 {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} (expected 2 type arguments, got {})",
                            type_args.len()
                        ),
                        span,
                    ));
                    return Some((vec![TypeInterner::ERROR], TypeInterner::ERROR));
                }

                let value_ty = self.resolve_type_expr(&type_args[0]);
                let return_ty = self.resolve_type_expr(&type_args[1]);
                let type_field_ty = self
                    .named_types
                    .get("TypeField")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![value_ty, type_field_ty], return_ty))
            }
            "type.variant_field_value" => {
                if type_args.len() != 2 {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{name} (expected 2 type arguments, got {})",
                            type_args.len()
                        ),
                        span,
                    ));
                    return Some((vec![TypeInterner::ERROR], TypeInterner::ERROR));
                }

                let value_ty = self.resolve_type_expr(&type_args[0]);
                let return_ty = self.resolve_type_expr(&type_args[1]);
                let type_field_ty = self
                    .named_types
                    .get("TypeField")
                    .copied()
                    .unwrap_or(TypeInterner::ERROR);
                Some((vec![value_ty, type_field_ty], return_ty))
            }
            "secret.redact" => {
                self.expect_no_type_args(&name, type_args, span);
                let secret_ty = self.interner.intern(Type::Secret(TypeInterner::ERROR));
                Some((vec![secret_ty], TypeInterner::STRING))
            }
            "secret.compare" => {
                self.expect_no_type_args(&name, type_args, span);
                let secret_ty = self.interner.intern(Type::Secret(TypeInterner::ERROR));
                Some((vec![secret_ty, secret_ty], TypeInterner::BOOL))
            }
            "bytes.new" => {
                self.no_type_args_signature(&name, type_args, span, vec![], TypeInterner::BYTES)
            }
            "bytes.length" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::BYTES],
                TypeInterner::INT64,
            ),
            "bytes.slice" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![
                    TypeInterner::BYTES,
                    TypeInterner::INT64,
                    TypeInterner::INT64,
                ],
                TypeInterner::BYTES,
            ),
            "bytes.concat" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::BYTES, TypeInterner::BYTES],
                TypeInterner::BYTES,
            ),
            "bytes.from_string" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::BYTES,
            ),
            "bytes.to_string" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![TypeInterner::BYTES],
                    self.interner
                        .intern(Type::Result(TypeInterner::STRING, TypeInterner::STRING)),
                ))
            }
            "bytes.get" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![TypeInterner::BYTES, TypeInterner::INT64],
                    self.interner.intern(Type::Optional(TypeInterner::INT64)),
                ))
            }
            "bytes.to_hex" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::BYTES],
                TypeInterner::STRING,
            ),
            "bytes.from_hex" => {
                self.expect_no_type_args(&name, type_args, span);
                Some((
                    vec![TypeInterner::STRING],
                    self.interner
                        .intern(Type::Result(TypeInterner::BYTES, TypeInterner::STRING)),
                ))
            }
            "list.new" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "list.new (expected 1 type argument, got {})",
                            type_args.len()
                        ),
                        span,
                    ));
                    return Some((vec![], TypeInterner::ERROR));
                }
                let inner = self.resolve_type_expr(&type_args[0]);
                Some((vec![], self.interner.intern(Type::List(inner))))
            }
            "list.append" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "list.append (expected 1 type argument, got {})",
                            type_args.len()
                        ),
                        span,
                    ));
                    return Some((
                        vec![TypeInterner::ERROR, TypeInterner::ERROR],
                        TypeInterner::ERROR,
                    ));
                }
                let inner = self.resolve_type_expr(&type_args[0]);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty, inner], list_ty))
            }
            "list.length" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                Some((
                    vec![self.interner.intern(Type::List(inner))],
                    TypeInterner::INT64,
                ))
            }
            "list.get" => {
                if type_args.len() != 1 {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "list.get (expected 1 type argument, got {})",
                            type_args.len()
                        ),
                        span,
                    ));
                    return Some((
                        vec![TypeInterner::ERROR, TypeInterner::INT64],
                        TypeInterner::ERROR,
                    ));
                }
                let inner = self.resolve_type_expr(&type_args[0]);
                Some((
                    vec![self.interner.intern(Type::List(inner)), TypeInterner::INT64],
                    self.interner.intern(Type::Optional(inner)),
                ))
            }
            // list builtins that transform a list → list
            "list.reverse" | "list.sort" | "list.unique" | "list.flatten" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty], list_ty))
            }
            "list.is_empty" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty], TypeInterner::BOOL))
            }
            "list.skip" | "list.take" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty, TypeInterner::INT64], list_ty))
            }
            "list.contains" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty, inner], TypeInterner::BOOL))
            }
            "list.index_of" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((
                    vec![list_ty, inner],
                    self.interner.intern(Type::Optional(TypeInterner::INT64)),
                ))
            }
            "list.remove" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty, TypeInterner::INT64], list_ty))
            }
            "list.concat" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty, list_ty], list_ty))
            }
            "list.zip" => {
                let (inner_a, inner_b) = self.optional_two_type_args(&name, type_args, span);
                let list_a = self.interner.intern(Type::List(inner_a));
                let list_b = self.interner.intern(Type::List(inner_b));
                let result_inner = self.interner.intern(Type::List(TypeInterner::ERROR));
                let result_ty = self.interner.intern(Type::List(result_inner));
                Some((vec![list_a, list_b], result_ty))
            }
            // math builtins
            "math.sqrt" | "math.log" | "math.log2" | "math.log10" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::FLOAT64],
                TypeInterner::FLOAT64,
            ),
            "math.pow" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::FLOAT64, TypeInterner::FLOAT64],
                TypeInterner::FLOAT64,
            ),
            "math.floor" | "math.ceil" | "math.round" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::FLOAT64],
                TypeInterner::FLOAT64,
            ),
            "math.clamp" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![
                    TypeInterner::FLOAT64,
                    TypeInterner::FLOAT64,
                    TypeInterner::FLOAT64,
                ],
                TypeInterner::FLOAT64,
            ),
            "math.average" | "math.median" => {
                let inner = match type_args.len() {
                    0 => TypeInterner::FLOAT64,
                    1 => self.resolve_type_expr(&type_args[0]),
                    got => {
                        self.sink.emit(errors::unknown_type(
                            &format!("{name} (expected 0 or 1 type arguments, got {got})"),
                            span,
                        ));
                        TypeInterner::ERROR
                    }
                };
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty], TypeInterner::FLOAT64))
            }
            "math.pi" | "math.e" => {
                self.no_type_args_signature(&name, type_args, span, vec![], TypeInterner::FLOAT64)
            }
            "math.sin" | "math.cos" | "math.tan" | "math.to_radians" | "math.to_degrees" => self
                .no_type_args_signature(
                    &name,
                    type_args,
                    span,
                    vec![TypeInterner::FLOAT64],
                    TypeInterner::FLOAT64,
                ),
            "math.mod" | "math.gcd" | "math.lcm" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::INT64, TypeInterner::INT64],
                TypeInterner::INT64,
            ),
            "math.is_even" | "math.is_odd" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::INT64],
                TypeInterner::BOOL,
            ),
            "math.factorial" | "math.sign" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::INT64],
                TypeInterner::INT64,
            ),
            "math.sum" => {
                self.expect_no_type_args(&name, type_args, span);
                let list_int = self.interner.intern(Type::List(TypeInterner::INT64));
                Some((vec![list_int], TypeInterner::INT64))
            }
            // string extras
            "string.reverse" | "string.trim_start" | "string.trim_end" => self
                .no_type_args_signature(
                    &name,
                    type_args,
                    span,
                    vec![TypeInterner::STRING],
                    TypeInterner::STRING,
                ),
            "string.after" | "string.before" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING, TypeInterner::STRING],
                TypeInterner::STRING,
            ),
            // string.chars / string.words / string.lines → list[string]
            "string.chars" | "string.words" | "string.lines" => {
                self.expect_no_type_args(&name, type_args, span);
                let list_str = self.interner.intern(Type::List(TypeInterner::STRING));
                Some((vec![TypeInterner::STRING], list_str))
            }
            // random builtins
            "random.int64" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::INT64, TypeInterner::INT64],
                TypeInterner::INT64,
            ),
            "random.float64" => {
                self.no_type_args_signature(&name, type_args, span, vec![], TypeInterner::FLOAT64)
            }
            "random.bool" => {
                self.no_type_args_signature(&name, type_args, span, vec![], TypeInterner::BOOL)
            }
            "random.choice" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty], self.interner.intern(Type::Optional(inner))))
            }
            "random.shuffle" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty], list_ty))
            }
            "string.is_empty" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::BOOL,
            ),
            "string.is_not_empty" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::BOOL,
            ),
            "string.repeat" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING, TypeInterner::INT64],
                TypeInterner::STRING,
            ),
            "string.slice" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![
                    TypeInterner::STRING,
                    TypeInterner::INT64,
                    TypeInterner::INT64,
                ],
                TypeInterner::STRING,
            ),
            // string.pad_left is the canonical name; pad_start/pad_end are aliases
            "string.pad_left" | "string.pad_start" | "string.pad_end" => self
                .no_type_args_signature(
                    &name,
                    type_args,
                    span,
                    vec![
                        TypeInterner::STRING,
                        TypeInterner::INT64,
                        TypeInterner::STRING,
                    ],
                    TypeInterner::STRING,
                ),
            "string.slugify" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::STRING,
            ),
            "string.truncate" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![
                    TypeInterner::STRING,
                    TypeInterner::INT64,
                    TypeInterner::STRING,
                ],
                TypeInterner::STRING,
            ),
            "string.between" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![
                    TypeInterner::STRING,
                    TypeInterner::STRING,
                    TypeInterner::STRING,
                ],
                TypeInterner::STRING,
            ),
            "map.new" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![], map_ty))
            }
            "map.length" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty], TypeInterner::INT64))
            }
            "map.is_empty" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty], TypeInterner::BOOL))
            }
            "map.has" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty, k], TypeInterner::BOOL))
            }
            "map.get" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty, k], self.interner.intern(Type::Optional(v))))
            }
            "map.insert" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty, k, v], map_ty))
            }
            "map.remove" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty, k], map_ty))
            }
            "map.keys" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty], self.interner.intern(Type::List(k))))
            }
            "map.values" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty], self.interner.intern(Type::List(v))))
            }
            "set.new" => {
                let inner = self.set_type_arg(&name, type_args, span);
                Some((vec![], self.interner.intern(Type::Set(inner))))
            }
            "set.add" | "set.remove" => {
                let inner = self.set_type_arg(&name, type_args, span);
                let set_ty = self.interner.intern(Type::Set(inner));
                Some((vec![set_ty, inner], set_ty))
            }
            "set.contains" => {
                let inner = self.set_type_arg(&name, type_args, span);
                let set_ty = self.interner.intern(Type::Set(inner));
                Some((vec![set_ty, inner], TypeInterner::BOOL))
            }
            "set.length" => {
                let inner = self.set_type_arg(&name, type_args, span);
                let set_ty = self.interner.intern(Type::Set(inner));
                Some((vec![set_ty], TypeInterner::INT64))
            }
            "set.is_empty" => {
                let inner = self.set_type_arg(&name, type_args, span);
                let set_ty = self.interner.intern(Type::Set(inner));
                Some((vec![set_ty], TypeInterner::BOOL))
            }
            "set.to_list" => {
                let inner = self.set_type_arg(&name, type_args, span);
                let set_ty = self.interner.intern(Type::Set(inner));
                Some((vec![set_ty], self.interner.intern(Type::List(inner))))
            }
            "set.union" | "set.intersection" | "set.difference" => {
                let inner = self.set_type_arg(&name, type_args, span);
                let set_ty = self.interner.intern(Type::Set(inner));
                Some((vec![set_ty, set_ty], set_ty))
            }

            // list.first / list.last — no fn arg
            "list.first" | "list.last" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty], self.interner.intern(Type::Optional(inner))))
            }
            // higher-order: list.filter[T](list, fn) -> list[T]
            "list.filter" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let predicate_ty = self.function_type(vec![inner], TypeInterner::BOOL);
                Some((vec![list_ty, predicate_ty], list_ty))
            }
            // higher-order: list.map[T, U](list, fn) -> list[U]
            "list.map" => {
                let (inner_t, inner_u) = self.optional_two_type_args(&name, type_args, span);
                let list_t = self.interner.intern(Type::List(inner_t));
                let list_u = self.interner.intern(Type::List(inner_u));
                let mapper_ty = self.function_type(vec![inner_t], inner_u);
                Some((vec![list_t, mapper_ty], list_u))
            }
            // higher-order: list.find[T](list, fn) -> optional[T]
            "list.find" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let predicate_ty = self.function_type(vec![inner], TypeInterner::BOOL);
                Some((
                    vec![list_ty, predicate_ty],
                    self.interner.intern(Type::Optional(inner)),
                ))
            }
            // higher-order: list.sort_by[T](list, fn) -> list[T]
            "list.sort_by" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let key_fn_ty = self.function_type(vec![inner], TypeInterner::INT64);
                Some((vec![list_ty, key_fn_ty], list_ty))
            }
            // higher-order: list.all / list.any [T](list, fn) -> bool
            "list.all" | "list.any" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let predicate_ty = self.function_type(vec![inner], TypeInterner::BOOL);
                Some((vec![list_ty, predicate_ty], TypeInterner::BOOL))
            }
            // higher-order: list.count[T](list, fn) -> int64
            "list.count" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let predicate_ty = self.function_type(vec![inner], TypeInterner::BOOL);
                Some((vec![list_ty, predicate_ty], TypeInterner::INT64))
            }
            // list.sum[T](list) -> T (numeric)
            "list.sum" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty], inner))
            }
            // list.group_by[T](list, fn) -> map[string, list[T]]
            "list.group_by" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let group_map_ty = self
                    .interner
                    .intern(Type::Map(TypeInterner::STRING, list_ty));
                let key_fn_ty = self.function_type(vec![inner], TypeInterner::STRING);
                Some((vec![list_ty, key_fn_ty], group_map_ty))
            }
            // list.reduce[T, U](list, initial, fn) -> U
            "list.reduce" => {
                let (inner, accumulator) = self.optional_two_type_args(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let reducer_ty = self.function_type(vec![accumulator, inner], accumulator);
                Some((vec![list_ty, accumulator, reducer_ty], accumulator))
            }
            // list.chunk[T](list, size) -> list[list[T]]
            "list.chunk" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let list_of_list = self.interner.intern(Type::List(list_ty));
                Some((vec![list_ty, TypeInterner::INT64], list_of_list))
            }
            // list.sort_by_index[T](list[T], index) -> list[T]
            // where T is the element type (e.g., list[string])
            "list.sort_by_index" => {
                let elem = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(elem));
                Some((vec![list_ty, TypeInterner::INT64], list_ty))
            }
            "list.is_sorted" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty], TypeInterner::BOOL))
            }
            "list.all_elements_in" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty, list_ty], TypeInterner::BOOL))
            }
            "list.enumerate" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                let pair_ty = self.interner.intern(Type::List(TypeInterner::ERROR));
                let pairs_ty = self.interner.intern(Type::List(pair_ty));
                Some((vec![list_ty], pairs_ty))
            }
            "list.from_set" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let set_ty = self.interner.intern(Type::Set(inner));
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![set_ty], list_ty))
            }
            "list.repeat" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![inner, TypeInterner::INT64], list_ty))
            }
            "list.last_index_of" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((
                    vec![list_ty, inner],
                    self.interner.intern(Type::Optional(TypeInterner::INT64)),
                ))
            }
            "list.insert_at" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty, TypeInterner::INT64, inner], list_ty))
            }
            "list.remove_at" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((vec![list_ty, TypeInterner::INT64], list_ty))
            }
            "list.swap" => {
                let inner = self.optional_type_arg(&name, type_args, span);
                let list_ty = self.interner.intern(Type::List(inner));
                Some((
                    vec![list_ty, TypeInterner::INT64, TypeInterner::INT64],
                    list_ty,
                ))
            }
            "list.flat_map" => {
                let (inner_t, inner_u) = self.optional_two_type_args(&name, type_args, span);
                let list_t = self.interner.intern(Type::List(inner_t));
                let list_u = self.interner.intern(Type::List(inner_u));
                let mapper_ty = self.function_type(vec![inner_t], list_u);
                Some((vec![list_t, mapper_ty], list_u))
            }
            // map extras
            "map.set" => {
                // alias for map.insert
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty, k, v], map_ty))
            }
            "map.get_or" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty, k, v], v))
            }
            "map.merge" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty, map_ty], map_ty))
            }
            "map.contains_key" => {
                // alias for map.has
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![map_ty, k], TypeInterner::BOOL))
            }
            "map.from_lists" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let keys_ty = self.interner.intern(Type::List(k));
                let values_ty = self.interner.intern(Type::List(v));
                let map_ty = self.interner.intern(Type::Map(k, v));
                Some((vec![keys_ty, values_ty], map_ty))
            }
            "map.entries" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                let pair_ty = self.interner.intern(Type::List(TypeInterner::ERROR));
                let entries_ty = self.interner.intern(Type::List(pair_ty));
                Some((vec![map_ty], entries_ty))
            }
            "map.filter" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                let predicate_ty = self.function_type(vec![k, v], TypeInterner::BOOL);
                Some((vec![map_ty, predicate_ty], map_ty))
            }
            "map.map_values" => {
                let (k, v, u) = self.map_values_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                let out_ty = self.interner.intern(Type::Map(k, u));
                let mapper_ty = self.function_type(vec![v], u);
                Some((vec![map_ty, mapper_ty], out_ty))
            }
            "map.for_each" => {
                let (k, v) = self.map_type_args(&name, type_args, span);
                let map_ty = self.interner.intern(Type::Map(k, v));
                let callback_ty = self.function_type(vec![k, v], TypeInterner::ERROR);
                Some((vec![map_ty, callback_ty], TypeInterner::NOTHING))
            }
            "uuid.new" => {
                self.no_type_args_signature(&name, type_args, span, vec![], TypeInterner::STRING)
            }
            // char-level string operations
            "string.take_chars" | "string.take_last_chars" | "string.drop_chars" => self
                .no_type_args_signature(
                    &name,
                    type_args,
                    span,
                    vec![TypeInterner::STRING, TypeInterner::INT64],
                    TypeInterner::STRING,
                ),
            "string.char_at" => {
                self.expect_no_type_args(&name, type_args, span);
                let opt_str = self.interner.intern(Type::Optional(TypeInterner::STRING));
                Some((vec![TypeInterner::STRING, TypeInterner::INT64], opt_str))
            }
            "string.index_of" => {
                self.expect_no_type_args(&name, type_args, span);
                let opt_int = self.interner.intern(Type::Optional(TypeInterner::INT64));
                Some((vec![TypeInterner::STRING, TypeInterner::STRING], opt_int))
            }
            "string.count" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING, TypeInterner::STRING],
                TypeInterner::INT64,
            ),
            "string.to_upper_first" | "string.to_lower_first" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::STRING,
            ),
            "string.center" | "string.ljust" | "string.rjust" | "string.zfill" => self
                .no_type_args_signature(
                    &name,
                    type_args,
                    span,
                    vec![TypeInterner::STRING, TypeInterner::INT64],
                    TypeInterner::STRING,
                ),
            "string.remove_prefix" | "string.remove_suffix" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING, TypeInterner::STRING],
                TypeInterner::STRING,
            ),
            "string.is_numeric" | "string.is_alpha" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::BOOL,
            ),
            // encoding module (all string → string)
            "encoding.base64_encode"
            | "encoding.base64_decode"
            | "encoding.hex_encode"
            | "encoding.hex_decode"
            | "encoding.url_encode"
            | "encoding.url_decode" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::STRING,
            ),
            // crypto module (string → string)
            "crypto.sha256" | "crypto.md5" => self.no_type_args_signature(
                &name,
                type_args,
                span,
                vec![TypeInterner::STRING],
                TypeInterner::STRING,
            ),
            "time.now_ms" | "time.now_s" => {
                self.no_type_args_signature(&name, type_args, span, vec![], TypeInterner::INT64)
            }
            "os.args" => {
                self.expect_no_type_args(&name, type_args, span);
                let list_string = self.interner.intern(Type::List(TypeInterner::STRING));
                Some((vec![], list_string))
            }
            "os.env" => {
                self.expect_no_type_args(&name, type_args, span);
                let opt_string = self.interner.intern(Type::Optional(TypeInterner::STRING));
                Some((vec![TypeInterner::STRING], opt_string))
            }
            "csv.parse" => {
                self.expect_no_type_args(&name, type_args, span);
                let list_string = self.interner.intern(Type::List(TypeInterner::STRING));
                let rows_ty = self.interner.intern(Type::List(list_string));
                Some((vec![TypeInterner::STRING], rows_ty))
            }
            "csv.stringify" => {
                self.expect_no_type_args(&name, type_args, span);
                let list_string = self.interner.intern(Type::List(TypeInterner::STRING));
                let rows_ty = self.interner.intern(Type::List(list_string));
                Some((vec![rows_ty], TypeInterner::STRING))
            }
            "csv.parse_with_header" => {
                self.expect_no_type_args(&name, type_args, span);
                let row_ty = self
                    .interner
                    .intern(Type::Map(TypeInterner::STRING, TypeInterner::STRING));
                let rows_ty = self.interner.intern(Type::List(row_ty));
                Some((vec![TypeInterner::STRING], rows_ty))
            }
            _ => None,
        }
    }

    // ------------------------------------------------------------------
    // Module
    // ------------------------------------------------------------------

    fn check_module(&mut self, module: &Module) {
        self.collect_type_aliases(module);

        // First pass: predeclare all user-defined types so function signatures,
        // fields, and methods can refer to them by name.
        let mut current_file = None;
        let mut current_namespace = None;
        for item in &module.items {
            Self::update_current_namespace(item, &mut current_file, &mut current_namespace);
            match item {
                Item::Interface(def) => {
                    self.predeclare_interface(def, current_namespace.as_deref())
                }
                Item::Struct(def) => self.predeclare_struct(def, current_namespace.as_deref()),
                Item::Bitfield(def) => self.predeclare_bitfield(def, current_namespace.as_deref()),
                Item::Enum(def) => self.predeclare_enum(def, current_namespace.as_deref()),
                Item::Machine(def) => self.predeclare_machine(def, current_namespace.as_deref()),
                Item::Actor(def) => self.predeclare_actor(def, current_namespace.as_deref()),
                _ => {}
            }
        }

        // Second pass: fill in the struct/enum contents now that all names exist.
        let mut current_file = None;
        let mut current_namespace = None;
        for item in &module.items {
            Self::update_current_namespace(item, &mut current_file, &mut current_namespace);
            match item {
                Item::Interface(def) => self.finish_interface(def, current_namespace.as_deref()),
                Item::Struct(def) => self.finish_struct(def, current_namespace.as_deref()),
                Item::Bitfield(def) => self.finish_bitfield(def, current_namespace.as_deref()),
                Item::Enum(def) => self.finish_enum(def, current_namespace.as_deref()),
                Item::Machine(def) => self.finish_machine(def, current_namespace.as_deref()),
                Item::Actor(def) => self.finish_actor(def, current_namespace.as_deref()),
                _ => {}
            }
        }

        // Third pass: register all top-level function signatures into the type env
        // and build the purity map.
        let mut current_file = None;
        let mut current_namespace = None;
        for item in &module.items {
            Self::update_current_namespace(item, &mut current_file, &mut current_namespace);
            match item {
                Item::Mutual(block) => {
                    for decl in &block.declarations {
                        let is_pure = Self::params_are_pure(&decl.params);
                        for name in Self::function_lookup_names(
                            current_namespace.as_deref(),
                            &decl.name.name,
                        ) {
                            self.purity_map.insert(name, is_pure);
                        }

                        if decl.type_params.is_empty() {
                            self.register_function_decl_sig(decl);
                            let signature = self.function_decl_signature(decl);
                            for name in Self::function_lookup_names(
                                current_namespace.as_deref(),
                                &decl.name.name,
                            ) {
                                if decl.span.file.is_stdlib() {
                                    self.trusted_stdlib_function_signatures
                                        .insert(name.clone(), signature.clone());
                                }
                                self.function_signatures.insert(name, signature.clone());
                            }
                        }
                    }
                }
                Item::Function(func) => {
                    if func.type_params.is_empty() {
                        self.register_function_sig(func);
                        let signature = self.function_signature(func);
                        for name in Self::function_lookup_names(
                            current_namespace.as_deref(),
                            &func.name.name,
                        ) {
                            if func.span.file.is_stdlib() {
                                self.trusted_stdlib_function_signatures
                                    .insert(name.clone(), signature.clone());
                            }
                            self.function_signatures.insert(name, signature.clone());
                        }
                    } else {
                        // Generic function — store the template; type checking happens at call sites.
                        for name in Self::function_lookup_names(
                            current_namespace.as_deref(),
                            &func.name.name,
                        ) {
                            self.generic_function_templates.insert(name, func.clone());
                        }
                    }
                    let is_pure = Self::function_is_pure(func);
                    for name in
                        Self::function_lookup_names(current_namespace.as_deref(), &func.name.name)
                    {
                        self.purity_map.insert(name, is_pure);
                    }
                }
                Item::Interface(def) => {
                    for method in &def.methods {
                        let is_pure = Self::params_are_pure(&method.params);
                        for owner_name in
                            Self::type_lookup_names(current_namespace.as_deref(), &def.name.name)
                        {
                            self.purity_map
                                .insert(format!("{owner_name}.{}", method.name.name), is_pure);
                        }
                    }
                }
                Item::Implement(block) => self.register_implement_block(block),
                Item::Struct(def) => {
                    for method in &def.methods {
                        let is_pure = Self::function_is_pure(method);
                        for owner_name in
                            Self::type_lookup_names(current_namespace.as_deref(), &def.name.name)
                        {
                            self.purity_map
                                .insert(format!("{owner_name}.{}", method.name.name), is_pure);
                        }
                    }
                }
                Item::Bitfield(_) => {}
                _ => {}
            }
        }

        self.validate_mutual_blocks(module);
        self.validate_implement_blocks(module);

        // Fourth pass: type-check function bodies, methods, and verify blocks.
        let mut current_file = None;
        let mut current_namespace = None;
        for item in &module.items {
            Self::update_current_namespace(item, &mut current_file, &mut current_namespace);
            match item {
                Item::TypeAlias(alias) => {
                    self.check_type_alias(alias, current_namespace.as_deref())
                }
                Item::Function(func) => {
                    if func.type_params.is_empty() {
                        self.check_function(func);
                    }
                    // Generic function bodies are checked at each call site.
                }
                Item::Implement(block) => self.check_implement_block(block),
                Item::Struct(def) => {
                    let owner_name = Self::namespace_qualified_name(
                        current_namespace.as_deref(),
                        &def.name.name,
                    )
                    .unwrap_or_else(|| def.name.name.clone());
                    for method in &def.methods {
                        self.check_method(&owner_name, method);
                    }
                }
                Item::Bitfield(_) => {}
                Item::Actor(def) => self.check_actor(def, current_namespace.as_deref()),
                Item::VarDecl(decl) => self.check_var_decl(decl),
                Item::Verify(verify) => self.check_verify_block(verify),
                Item::Property(prop) => self.check_property_block(prop),
                _ => {}
            }
        }
    }

    fn collect_type_aliases(&mut self, module: &Module) {
        let mut current_file = None;
        let mut current_namespace = None;
        for item in &module.items {
            Self::update_current_namespace(item, &mut current_file, &mut current_namespace);
            let Item::TypeAlias(alias) = item else {
                continue;
            };
            let canonical = Self::type_alias_canonical_name(alias, current_namespace.as_deref());
            self.type_aliases.insert(canonical, alias.clone());
        }
    }

    fn type_alias_canonical_name(alias: &ast::TypeAlias, namespace: Option<&str>) -> String {
        if alias.root_exported {
            alias.name.name.clone()
        } else {
            Self::canonical_name(namespace, &alias.name.name)
        }
    }

    /// Returns true if a function has no capability-type parameters (i.e. is pure).
    fn function_is_pure(func: &FunctionDef) -> bool {
        Self::params_are_pure(&func.params)
    }

    fn params_are_pure(params: &[ast::Param]) -> bool {
        !params
            .iter()
            .any(|p| capability::type_expr_is_capability(&p.ty))
    }

    fn check_type_alias(&mut self, alias: &ast::TypeAlias, namespace: Option<&str>) {
        let alias_name = Self::type_alias_canonical_name(alias, namespace);
        if alias.root_exported {
            let _ = self.resolve_type_alias(&alias_name, alias.name.span);
            return;
        }
        let Some(constraint) = &alias.constraint else {
            let _ = self.resolve_type_alias(&alias_name, alias.name.span);
            return;
        };

        let alias_ty = self.resolve_type_alias(&alias_name, alias.name.span);
        let Some(_base_ty) = self.refinement_base_type(alias_ty) else {
            return;
        };
        let constraint_value_ty = self.refinement_boundary_input_type(alias_ty);

        let saved_name = self.current_function_name.clone();
        let saved_pure = self.current_function_pure;
        self.current_function_name = Some(format!("type {alias_name}"));
        self.current_function_pure = true;

        if let Some(def_id) = self.declaration_def_id(constraint.span()) {
            self.type_env.insert(def_id, constraint_value_ty);
        }

        let constraint_ty = self.check_expr(constraint);
        if constraint_ty != TypeInterner::ERROR && constraint_ty != TypeInterner::BOOL {
            self.sink.emit(errors::refinement_constraint_not_bool(
                &alias_name,
                &self.type_name(constraint_ty),
                constraint.span(),
            ));
        }

        self.current_function_name = saved_name;
        self.current_function_pure = saved_pure;
    }

    // ------------------------------------------------------------------
    // Actors
    // ------------------------------------------------------------------

    fn predeclare_actor(&mut self, def: &ast::ActorDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let trusted_name = canonical_name.clone();
        let aid = self.interner.add_actor(TypeActorDef {
            name: canonical_name,
            capability_params: Vec::new(),
            state_fields: Vec::new(),
            messages: Vec::new(),
        });
        let ty = self.interner.intern(Type::Actor(aid));
        self.register_named_type(namespace, &def.name.name, ty);
        self.register_trusted_stdlib_named_type(&trusted_name, ty, def.name.span);
        if let Some(def_id) = self.declaration_def_id(def.name.span) {
            self.type_env.insert(def_id, ty);
        }
    }

    fn predeclare_machine(&mut self, def: &ast::MachineDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let trusted_name = canonical_name.clone();
        let mid = self.interner.add_machine(TypeMachineDef {
            name: canonical_name,
            states: Vec::new(),
            transitions: Vec::new(),
        });
        let ty = self.interner.intern(Type::Machine(mid));
        self.register_named_type(namespace, &def.name.name, ty);
        self.register_trusted_stdlib_named_type(&trusted_name, ty, def.name.span);

        if let Some(def_id) = self.declaration_def_id(def.name.span) {
            self.type_env.insert(def_id, ty);
        }
    }

    fn finish_machine(&mut self, def: &ast::MachineDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let Some(&ty) = self.named_types.get(&canonical_name) else {
            return;
        };
        let Type::Machine(mid) = *self.interner.resolve(ty) else {
            return;
        };

        let mut state_ids = HashMap::new();
        let mut states = Vec::new();
        let mut reflection_states = Vec::new();
        for state in &def.states {
            if let Some(previous_span) = state_ids.get(&state.name.name).map(|(_, span)| *span) {
                self.sink.emit(errors::duplicate_machine_state(
                    &canonical_name,
                    &state.name.name,
                    state.name.span,
                    previous_span,
                ));
                continue;
            }

            let state_id = jett_types::MachineStateId::new(states.len() as u32);
            state_ids.insert(state.name.name.clone(), (state_id, state.name.span));
            let resolved_fields = state
                .fields
                .iter()
                .map(|field| (field.name.name.clone(), self.resolve_type_expr(&field.ty)))
                .collect::<Vec<_>>();
            let reflection_fields = state
                .fields
                .iter()
                .zip(resolved_fields.iter())
                .enumerate()
                .map(|(index, (field, (_, field_ty)))| {
                    self.reflection_field_info_for_type_expr(
                        index,
                        &field.name.name,
                        field.serialize_name.as_deref().unwrap_or(&field.name.name),
                        &field.ty,
                        namespace,
                        *field_ty,
                    )
                })
                .collect::<Vec<_>>();
            let has_secret = reflection_fields.iter().any(|field| field.has_secret);
            reflection_states.push(ReflectionMachineStateInfo::new(
                state_id.index() as usize,
                &state.name.name,
                has_secret,
                reflection_fields,
            ));
            states.push(TypeMachineStateDef {
                name: state.name.name.clone(),
                fields: resolved_fields,
            });
        }

        let mut transitions = Vec::new();
        let mut reflection_transitions = Vec::new();
        for transition in &def.transitions {
            let transition_name = format!("{} to {}", transition.from.name, transition.to.name);
            let from = state_ids.get(&transition.from.name).map(|(id, _)| *id);
            let to = state_ids.get(&transition.to.name).map(|(id, _)| *id);

            match (from, to) {
                (Some(from), Some(to)) => {
                    let index = transitions.len();
                    transitions.push(TypeMachineTransitionDef { from, to });
                    reflection_transitions.push(ReflectionMachineTransitionInfo::new(
                        index,
                        from.index() as usize,
                        &transition.from.name,
                        to.index() as usize,
                        &transition.to.name,
                    ));
                }
                (None, _) => {
                    self.sink.emit(errors::invalid_machine_transition(
                        &canonical_name,
                        &transition_name,
                        &format!("unknown source state `{}`", transition.from.name),
                        transition.from.span,
                    ));
                }
                (_, None) => {
                    self.sink.emit(errors::invalid_machine_transition(
                        &canonical_name,
                        &transition_name,
                        &format!("unknown target state `{}`", transition.to.name),
                        transition.to.span,
                    ));
                }
            }
        }

        self.reflection_machines_by_id.insert(
            ty,
            (
                canonical_name.clone(),
                ReflectionMachineInfo::new(reflection_states, reflection_transitions),
            ),
        );

        self.interner.update_machine(
            mid,
            TypeMachineDef {
                name: canonical_name,
                states,
                transitions,
            },
        );
    }

    fn finish_actor(&mut self, def: &ast::ActorDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let Some(&ty) = self.named_types.get(&canonical_name) else {
            return;
        };
        let Type::Actor(aid) = *self.interner.resolve(ty) else {
            return;
        };

        let capability_params: Vec<(String, TypeId)> = def
            .capability_params
            .iter()
            .map(|p| (p.name.name.clone(), self.resolve_type_expr(&p.ty)))
            .collect();

        let state_fields: Vec<(String, TypeId)> = def
            .state_fields
            .iter()
            .map(|f| (f.name.name.clone(), self.resolve_type_expr(&f.ty)))
            .collect();

        let messages: Vec<ActorMessageDef> = def
            .handlers
            .iter()
            .map(|h| {
                let params = h
                    .params
                    .iter()
                    .map(|p| (p.name.name.clone(), self.resolve_type_expr(&p.ty)))
                    .collect();
                let responds = h
                    .responds
                    .as_ref()
                    .map(|t| self.resolve_type_expr(t))
                    .unwrap_or(TypeInterner::NOTHING);
                ActorMessageDef {
                    name: h.name.name.clone(),
                    params,
                    responds,
                }
            })
            .collect();

        self.interner.update_actor(
            aid,
            TypeActorDef {
                name: canonical_name,
                capability_params,
                state_fields,
                messages,
            },
        );
    }

    fn check_actor(&mut self, def: &ast::ActorDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let Some(&actor_ty) = self.named_types.get(&canonical_name) else {
            return;
        };
        let Type::Actor(aid) = *self.interner.resolve(actor_ty) else {
            return;
        };
        let actor_def = self.interner.resolve_actor(aid).clone();

        // Register state fields and capability params in a fresh type env scope.
        let mut local_env: HashMap<String, TypeId> = HashMap::new();
        for (name, ty) in &actor_def.capability_params {
            local_env.insert(name.clone(), *ty);
        }
        for (name, ty) in &actor_def.state_fields {
            local_env.insert(name.clone(), *ty);
        }

        for (param_ast, (_, param_ty)) in def
            .capability_params
            .iter()
            .zip(actor_def.capability_params.iter())
        {
            if let Some(def_id) = self.declaration_def_id(param_ast.name.span) {
                self.type_env.insert(def_id, *param_ty);
            }
        }

        // Type-check state field initializers.
        for field in &def.state_fields {
            let declared_ty = self.resolve_type_expr(&field.ty);
            let init_ty = self.check_expr_for_expected(&field.value, declared_ty, true);
            if init_ty != TypeInterner::ERROR
                && declared_ty != TypeInterner::ERROR
                && !self.types_compatible(declared_ty, init_ty)
            {
                self.sink.emit(errors::type_mismatch(
                    &self.type_name(declared_ty),
                    &self.type_name(init_ty),
                    field.value.span(),
                ));
            }
        }

        // Type-check each handler body.
        let prev_return = self.current_return_type;
        let prev_respond = self.current_respond_type;
        let prev_function_name = self.current_function_name.clone();

        for (handler_ast, handler_def) in def.handlers.iter().zip(actor_def.messages.iter()) {
            // Set up respond type.
            let responds_ty = if handler_def.responds == TypeInterner::NOTHING {
                None
            } else {
                Some(handler_def.responds)
            };
            self.current_respond_type = responds_ty;
            self.current_return_type = None;
            self.current_function_name =
                Some(format!("{}.{}", actor_def.name, handler_ast.name.name));

            // Register message params in the type env temporarily.
            for (param_ast, (_, param_ty)) in
                handler_ast.params.iter().zip(handler_def.params.iter())
            {
                if let Some(def_id) = self.declaration_def_id(param_ast.name.span) {
                    self.type_env.insert(def_id, *param_ty);
                }
            }

            // Register local_env vars into type_env using resolve declarations.
            for field in &def.state_fields {
                if let Some(def_id) = self.declaration_def_id(field.name.span) {
                    let ty = self.resolve_type_expr(&field.ty);
                    self.type_env.insert(def_id, ty);
                }
            }
            for cap in &def.capability_params {
                if let Some(def_id) = self.declaration_def_id(cap.name.span) {
                    let ty = self.resolve_type_expr(&cap.ty);
                    self.type_env.insert(def_id, ty);
                }
            }

            self.check_block(&handler_ast.body);
        }

        self.current_return_type = prev_return;
        self.current_respond_type = prev_respond;
        self.current_function_name = prev_function_name;
    }

    fn predeclare_struct(&mut self, def: &ast::StructDef, namespace: Option<&str>) {
        self.check_struct_json_serialize_names(def, namespace);
        if !def.type_params.is_empty() {
            // Generic struct — store the template for later monomorphization.
            self.register_generic_struct_template(namespace, &def.name.name, def.clone());
            return;
        }

        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let trusted_name = canonical_name.clone();
        let sid = self.interner.add_struct(TypeStructDef {
            name: canonical_name,
            fields: Vec::new(),
            methods: Vec::new(),
        });
        let ty = self.interner.intern(Type::Struct(sid));
        self.register_named_type(namespace, &def.name.name, ty);
        self.register_trusted_stdlib_named_type(&trusted_name, ty, def.name.span);

        if let Some(def_id) = self.declaration_def_id(def.name.span) {
            self.type_env.insert(def_id, ty);
        }
    }

    fn predeclare_interface(&mut self, def: &ast::InterfaceDecl, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let trusted_name = canonical_name.clone();
        let iid = self.interner.add_interface(TypeInterfaceDef {
            name: canonical_name,
            methods: Vec::new(),
        });
        let ty = self.interner.intern(Type::Interface(iid));
        self.register_named_type(namespace, &def.name.name, ty);
        self.register_trusted_stdlib_named_type(&trusted_name, ty, def.name.span);

        if let Some(def_id) = self.declaration_def_id(def.name.span) {
            self.type_env.insert(def_id, ty);
        }
    }

    fn predeclare_bitfield(&mut self, def: &ast::BitfieldDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let trusted_name = canonical_name.clone();
        let bid = self.interner.add_bitfield(TypeBitfieldDef {
            name: canonical_name,
            network_order: def.network_order,
            fields: Vec::new(),
        });
        let ty = self.interner.intern(Type::Bitfield(bid));
        self.register_named_type(namespace, &def.name.name, ty);
        self.register_trusted_stdlib_named_type(&trusted_name, ty, def.name.span);

        if let Some(def_id) = self.declaration_def_id(def.name.span) {
            self.type_env.insert(def_id, ty);
        }
    }

    fn finish_struct(&mut self, def: &ast::StructDef, namespace: Option<&str>) {
        if !def.type_params.is_empty() {
            // Generic structs are monomorphized on demand — nothing to finish here.
            return;
        }
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let Some(&ty) = self.named_types.get(&canonical_name) else {
            return;
        };
        let Type::Struct(sid) = *self.interner.resolve(ty) else {
            return;
        };

        let fields: Vec<(String, TypeId)> = def
            .fields
            .iter()
            .map(|field| (field.name.name.clone(), self.resolve_type_expr(&field.ty)))
            .collect();
        let reflection_fields =
            self.reflection_fields_for_struct_def(def, namespace, fields.as_slice());
        self.reflection_fields_by_id
            .insert(ty, (canonical_name.clone(), reflection_fields));
        let methods = def
            .methods
            .iter()
            .map(|method| self.method_signature(method))
            .collect();

        self.interner.update_struct(
            sid,
            TypeStructDef {
                name: canonical_name,
                fields,
                methods,
            },
        );
    }

    fn finish_interface(&mut self, def: &ast::InterfaceDecl, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let Some(&ty) = self.named_types.get(&canonical_name) else {
            return;
        };
        let Type::Interface(iid) = *self.interner.resolve(ty) else {
            return;
        };

        let methods = def
            .methods
            .iter()
            .map(|method| self.function_decl_method_signature(method))
            .collect();

        self.interner.update_interface(
            iid,
            TypeInterfaceDef {
                name: canonical_name,
                methods,
            },
        );
    }

    fn finish_bitfield(&mut self, def: &ast::BitfieldDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let Some(&ty) = self.named_types.get(&canonical_name) else {
            return;
        };
        let Type::Bitfield(bid) = *self.interner.resolve(ty) else {
            return;
        };

        let list_u8 = self.interner.intern(Type::List(TypeInterner::UINT8));
        let mut fields = Vec::with_capacity(def.fields.len());
        let mut bits_before_payload = 0usize;
        for (index, field) in def.fields.iter().enumerate() {
            let (ty, kind) = match &field.kind {
                ast::BitfieldFieldKind::Bits { width, as_type } => {
                    if *width == 0 {
                        self.sink.emit(errors::invalid_bitfield_field(
                            &def.name.name,
                            &field.name.name,
                            "bit width must be at least 1",
                            field.span,
                        ));
                    } else if *width > 64 {
                        self.sink.emit(errors::invalid_bitfield_field(
                            &def.name.name,
                            &field.name.name,
                            "bit width must be at most 64",
                            field.span,
                        ));
                    }

                    let ty = if let Some(ty_expr) = as_type {
                        let resolved = self.resolve_type_expr(ty_expr);
                        if resolved != TypeInterner::ERROR {
                            match self.interner.resolve(resolved) {
                                Type::Enum(eid) => {
                                    let enum_def = self.interner.resolve_enum(*eid);
                                    if enum_def
                                        .variants
                                        .iter()
                                        .any(|variant| !variant.fields.is_empty())
                                    {
                                        self.sink.emit(errors::invalid_bitfield_field(
                                            &def.name.name,
                                            &field.name.name,
                                            "`as` annotations require an enum with only unit variants",
                                            field.span,
                                        ));
                                    }

                                    let max_discriminant = enum_def
                                        .variants
                                        .iter()
                                        .map(|variant| variant.discriminant)
                                        .max()
                                        .unwrap_or(0);
                                    let fits = if max_discriminant < 0 {
                                        false
                                    } else if *width >= 63 {
                                        true
                                    } else {
                                        (max_discriminant as u64) < (1_u64 << *width)
                                    };
                                    if !fits {
                                        self.sink.emit(errors::invalid_bitfield_field(
                                            &def.name.name,
                                            &field.name.name,
                                            "enum annotation has a discriminant that does not fit in the declared bit width",
                                            field.span,
                                        ));
                                    }
                                }
                                _ => {
                                    self.sink.emit(errors::invalid_bitfield_field(
                                        &def.name.name,
                                        &field.name.name,
                                        "`as` annotations must name an enum type",
                                        field.span,
                                    ));
                                }
                            }
                        }
                        resolved
                    } else if *width == 64 {
                        TypeInterner::UINT64
                    } else {
                        TypeInterner::INT64
                    };
                    bits_before_payload += *width as usize;
                    (ty, TypeBitfieldFieldKind::Bits { width: *width })
                }
                ast::BitfieldFieldKind::Payload(ty_expr) => {
                    let resolved = self.resolve_type_expr(ty_expr);
                    if resolved != TypeInterner::ERROR && resolved != list_u8 {
                        self.sink.emit(errors::invalid_bitfield_field(
                            &def.name.name,
                            &field.name.name,
                            "payload fields must have type `list[uint8]`",
                            field.span,
                        ));
                    }
                    if index + 1 != def.fields.len() {
                        self.sink.emit(errors::invalid_bitfield_field(
                            &def.name.name,
                            &field.name.name,
                            "payload fields must be the final field",
                            field.span,
                        ));
                    }
                    if !bits_before_payload.is_multiple_of(8) {
                        self.sink.emit(errors::invalid_bitfield_field(
                            &def.name.name,
                            &field.name.name,
                            "payload fields must start on a byte boundary",
                            field.span,
                        ));
                    }
                    (resolved, TypeBitfieldFieldKind::Payload)
                }
            };

            fields.push(TypeBitfieldFieldDef {
                name: field.name.name.clone(),
                ty,
                kind,
            });
        }

        let reflection_fields =
            self.reflection_fields_for_bitfield_def(def, namespace, fields.as_slice());
        let reflection_bitfield =
            self.reflection_bitfield_info_for_def(def, namespace, fields.as_slice());
        self.reflection_fields_by_id
            .insert(ty, (canonical_name.clone(), reflection_fields));
        self.reflection_bitfields_by_id
            .insert(ty, (canonical_name.clone(), reflection_bitfield));

        self.interner.update_bitfield(
            bid,
            TypeBitfieldDef {
                name: canonical_name,
                network_order: def.network_order,
                fields,
            },
        );
    }

    fn predeclare_enum(&mut self, def: &ast::EnumDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let trusted_name = canonical_name.clone();
        let eid = self.interner.add_enum(TypeEnumDef {
            name: canonical_name,
            variants: Vec::new(),
        });
        let ty = self.interner.intern(Type::Enum(eid));
        self.register_named_type(namespace, &def.name.name, ty);
        self.register_trusted_stdlib_named_type(&trusted_name, ty, def.name.span);

        if let Some(def_id) = self.declaration_def_id(def.name.span) {
            self.type_env.insert(def_id, ty);
        }
    }

    fn finish_enum(&mut self, def: &ast::EnumDef, namespace: Option<&str>) {
        let canonical_name = Self::canonical_name(namespace, &def.name.name);
        let Some(&ty) = self.named_types.get(&canonical_name) else {
            return;
        };
        let Type::Enum(eid) = *self.interner.resolve(ty) else {
            return;
        };

        let mut next_discriminant = 0_i64;
        let mut seen_discriminants = HashMap::new();
        let variants: Vec<VariantDef> = def
            .variants
            .iter()
            .map(|variant| {
                if variant.discriminant.is_some() && !variant.fields.is_empty() {
                    self.sink
                        .emit(errors::enum_discriminant_requires_unit_variant(
                            &def.name.name,
                            &variant.name.name,
                            variant.span,
                        ));
                }

                let discriminant = variant.discriminant.unwrap_or(next_discriminant);
                next_discriminant = discriminant.saturating_add(1);

                if let Some(previous_span) =
                    seen_discriminants.insert(discriminant, variant.name.span)
                {
                    self.sink.emit(errors::duplicate_enum_discriminant(
                        &def.name.name,
                        &variant.name.name,
                        discriminant,
                        variant.name.span,
                        previous_span,
                    ));
                }

                VariantDef {
                    name: variant.name.name.clone(),
                    fields: variant
                        .fields
                        .iter()
                        .map(|field| (field.name.name.clone(), self.resolve_type_expr(&field.ty)))
                        .collect(),
                    discriminant,
                }
            })
            .collect();
        let reflection_variants =
            self.reflection_variants_for_enum_def(def, namespace, variants.as_slice());
        self.reflection_variants_by_id
            .insert(ty, (canonical_name.clone(), reflection_variants));

        self.interner.update_enum(
            eid,
            TypeEnumDef {
                name: canonical_name,
                variants,
            },
        );
    }

    fn method_signature(&mut self, func: &FunctionDef) -> FunctionSig {
        let params = func
            .params
            .iter()
            .map(|param| {
                (
                    param.name.name.clone(),
                    self.resolve_type_expr(&param.ty),
                    param.view,
                )
            })
            .collect();
        let return_type = func
            .return_type
            .as_ref()
            .map(|ty| self.resolve_type_expr(ty))
            .unwrap_or(TypeInterner::NOTHING);

        FunctionSig {
            name: func.name.name.clone(),
            params,
            return_type,
            is_pure: Self::function_is_pure(func),
        }
    }

    fn function_decl_method_signature(&mut self, decl: &ast::FunctionDecl) -> FunctionSig {
        let params = decl
            .params
            .iter()
            .map(|param| {
                (
                    param.name.name.clone(),
                    self.resolve_type_expr(&param.ty),
                    param.view,
                )
            })
            .collect();
        let return_type = decl
            .return_type
            .as_ref()
            .map(|ty| self.resolve_type_expr(ty))
            .unwrap_or(TypeInterner::NOTHING);

        FunctionSig {
            name: decl.name.name.clone(),
            params,
            return_type,
            is_pure: Self::params_are_pure(&decl.params),
        }
    }

    fn function_signature(&mut self, func: &FunctionDef) -> (Vec<TypeId>, TypeId) {
        let params = func
            .params
            .iter()
            .map(|p| self.resolve_type_expr(&p.ty))
            .collect();
        let return_type = func
            .return_type
            .as_ref()
            .map(|t| self.resolve_type_expr(t))
            .unwrap_or(TypeInterner::NOTHING);
        (params, return_type)
    }

    fn function_decl_signature(&mut self, decl: &ast::FunctionDecl) -> (Vec<TypeId>, TypeId) {
        let params = decl
            .params
            .iter()
            .map(|p| self.resolve_type_expr(&p.ty))
            .collect();
        let return_type = decl
            .return_type
            .as_ref()
            .map(|t| self.resolve_type_expr(t))
            .unwrap_or(TypeInterner::NOTHING);
        (params, return_type)
    }

    // ------------------------------------------------------------------
    // Function registration (builds FunctionType + binds to DefId)
    // ------------------------------------------------------------------

    fn register_function_decl_sig(&mut self, decl: &ast::FunctionDecl) {
        let (param_types, return_type) = self.function_decl_signature(decl);
        let fn_type = self.interner.intern(Type::Function {
            params: param_types,
            return_type,
        });

        if let Some(def_id) = self.declaration_def_id(decl.name.span) {
            self.type_env.insert(def_id, fn_type);
        }
    }

    fn register_function_sig(&mut self, func: &FunctionDef) {
        let param_types: Vec<TypeId> = func
            .params
            .iter()
            .map(|p| self.resolve_type_expr(&p.ty))
            .collect();

        let return_type = func
            .return_type
            .as_ref()
            .map(|t| self.resolve_type_expr(t))
            .unwrap_or(TypeInterner::NOTHING);

        let fn_type = self.interner.intern(Type::Function {
            params: param_types,
            return_type,
        });

        // Bind the function name's DefId to this function type.
        if let Some(def_id) = self.declaration_def_id(func.name.span) {
            self.type_env.insert(def_id, fn_type);
        }
    }

    fn register_implement_block(&mut self, block: &ast::ImplementBlock) {
        let interface_ty = self.resolve_type_expr(&TypeExpr::Named(block.interface_name.clone()));
        let owner_ty = self.resolve_type_expr(&block.for_type);
        if interface_ty == TypeInterner::ERROR || owner_ty == TypeInterner::ERROR {
            return;
        }

        let owner_name = self.type_name(owner_ty);
        let interface_name = self.type_name(interface_ty);
        let method_sigs: Vec<_> = block
            .methods
            .iter()
            .map(|method| {
                let sig = self.method_signature(method);
                (method.name.name.clone(), sig)
            })
            .collect();

        let impl_methods = self.impl_methods_by_type.entry(owner_ty).or_default();
        let interface_methods = self
            .interface_impls
            .entry((interface_ty, owner_ty))
            .or_default();

        for (method_name, sig) in method_sigs {
            self.purity_map
                .insert(format!("{owner_name}.{method_name}"), sig.is_pure);
            self.purity_map
                .insert(format!("{interface_name}.{method_name}"), sig.is_pure);
            impl_methods.insert(method_name.clone(), sig.clone());
            interface_methods.insert(method_name, sig);
        }
    }

    fn validate_implement_blocks(&mut self, module: &Module) {
        for item in &module.items {
            let Item::Implement(block) = item else {
                continue;
            };

            let interface_ty =
                self.resolve_type_expr(&TypeExpr::Named(block.interface_name.clone()));
            let owner_ty = self.resolve_type_expr(&block.for_type);
            if interface_ty == TypeInterner::ERROR || owner_ty == TypeInterner::ERROR {
                continue;
            }

            let Type::Interface(iid) = *self.interner.resolve(interface_ty) else {
                self.sink.emit(errors::expected_interface(
                    &block.interface_name.name,
                    block.interface_name.span,
                ));
                continue;
            };
            let interface_def = self.interner.resolve_interface(iid).clone();

            let Some(impl_methods) = self.interface_impls.get(&(interface_ty, owner_ty)).cloned()
            else {
                continue;
            };

            let mut seen = HashSet::new();
            for method in &block.methods {
                if !seen.insert(method.name.name.clone()) {
                    self.sink.emit(errors::duplicate_implemented_method(
                        &self.type_name(owner_ty),
                        &method.name.name,
                        method.name.span,
                    ));
                    continue;
                }

                let Some(interface_method) = interface_def
                    .methods
                    .iter()
                    .find(|candidate| candidate.name == method.name.name)
                    .cloned()
                else {
                    self.sink.emit(errors::interface_has_no_member(
                        &interface_def.name,
                        &method.name.name,
                        method.name.span,
                    ));
                    continue;
                };

                let impl_sig = impl_methods
                    .get(&method.name.name)
                    .expect("impl method must exist");
                if !self.implementation_matches_interface(owner_ty, impl_sig, &interface_method) {
                    self.sink
                        .emit(errors::implemented_method_signature_mismatch(
                            &interface_def.name,
                            &self.type_name(owner_ty),
                            &method.name.name,
                            method.name.span,
                        ));
                }
            }

            for interface_method in &interface_def.methods {
                if !seen.contains(&interface_method.name) {
                    self.sink.emit(errors::missing_implemented_method(
                        &interface_def.name,
                        &self.type_name(owner_ty),
                        &interface_method.name,
                        block.span,
                    ));
                }
            }
        }
    }

    fn implementation_matches_interface(
        &mut self,
        owner_ty: TypeId,
        impl_sig: &FunctionSig,
        interface_sig: &FunctionSig,
    ) -> bool {
        if impl_sig.params.len() != interface_sig.params.len() {
            return false;
        }

        for (index, (impl_param, interface_param)) in impl_sig
            .params
            .iter()
            .zip(interface_sig.params.iter())
            .enumerate()
        {
            if impl_param.0 != interface_param.0 || impl_param.2 != interface_param.2 {
                return false;
            }

            let expected_ty = if index == 0 {
                match self.interner.resolve(interface_param.1) {
                    Type::Interface(_) => owner_ty,
                    _ => interface_param.1,
                }
            } else {
                interface_param.1
            };

            if !self.types_compatible(expected_ty, impl_param.1)
                || !self.types_compatible(impl_param.1, expected_ty)
            {
                return false;
            }
        }

        self.types_compatible(interface_sig.return_type, impl_sig.return_type)
            && self.types_compatible(impl_sig.return_type, interface_sig.return_type)
    }

    fn validate_mutual_blocks(&mut self, module: &Module) {
        let mut function_defs: HashMap<String, &FunctionDef> = HashMap::new();
        let mut current_file = None;
        let mut current_namespace = None;
        for item in &module.items {
            Self::update_current_namespace(item, &mut current_file, &mut current_namespace);
            if let Item::Function(func) = item {
                function_defs.insert(
                    Self::canonical_name(current_namespace.as_deref(), &func.name.name),
                    func,
                );
            }
        }

        let mut current_file = None;
        let mut current_namespace = None;
        for item in &module.items {
            Self::update_current_namespace(item, &mut current_file, &mut current_namespace);
            let Item::Mutual(block) = item else {
                continue;
            };

            for decl in &block.declarations {
                let canonical_name =
                    Self::canonical_name(current_namespace.as_deref(), &decl.name.name);
                let Some(func) = function_defs.get(&canonical_name).copied() else {
                    self.sink.emit(errors::mutual_function_missing_definition(
                        &decl.name.name,
                        decl.name.span,
                    ));
                    continue;
                };

                if !self.function_matches_decl(func, decl) {
                    self.sink.emit(errors::mutual_signature_mismatch(
                        &decl.name.name,
                        func.name.span,
                    ));
                }
            }
        }
    }

    fn function_matches_decl(&mut self, func: &FunctionDef, decl: &ast::FunctionDecl) -> bool {
        if !func.type_params.is_empty() || !decl.type_params.is_empty() {
            return Self::generic_function_matches_decl(func, decl);
        }

        if func.params.len() != decl.params.len() {
            return false;
        }

        for (func_param, decl_param) in func.params.iter().zip(decl.params.iter()) {
            if func_param.name.name != decl_param.name.name
                || func_param.view != decl_param.view
                || func_param.mutable != decl_param.mutable
            {
                return false;
            }

            let func_ty = self.resolve_type_expr(&func_param.ty);
            let decl_ty = self.resolve_type_expr(&decl_param.ty);
            if !self.types_compatible(decl_ty, func_ty) || !self.types_compatible(func_ty, decl_ty)
            {
                return false;
            }
        }

        let func_return = func
            .return_type
            .as_ref()
            .map(|ty| self.resolve_type_expr(ty))
            .unwrap_or(TypeInterner::NOTHING);
        let decl_return = decl
            .return_type
            .as_ref()
            .map(|ty| self.resolve_type_expr(ty))
            .unwrap_or(TypeInterner::NOTHING);

        self.types_compatible(decl_return, func_return)
            && self.types_compatible(func_return, decl_return)
    }

    fn generic_function_matches_decl(func: &FunctionDef, decl: &ast::FunctionDecl) -> bool {
        if func.type_params.len() != decl.type_params.len()
            || func.params.len() != decl.params.len()
        {
            return false;
        }

        let type_params: HashMap<String, String> = decl
            .type_params
            .iter()
            .zip(func.type_params.iter())
            .map(|(decl_param, func_param)| (decl_param.name.clone(), func_param.name.clone()))
            .collect();

        for (func_param, decl_param) in func.params.iter().zip(decl.params.iter()) {
            if func_param.name.name != decl_param.name.name
                || func_param.view != decl_param.view
                || func_param.mutable != decl_param.mutable
                || !Self::type_expr_matches_decl(&func_param.ty, &decl_param.ty, &type_params)
            {
                return false;
            }
        }

        Self::return_type_matches_decl(
            func.return_type.as_ref(),
            decl.return_type.as_ref(),
            &type_params,
        )
    }

    fn return_type_matches_decl(
        func_ty: Option<&TypeExpr>,
        decl_ty: Option<&TypeExpr>,
        type_params: &HashMap<String, String>,
    ) -> bool {
        match (func_ty, decl_ty) {
            (None, None) => true,
            (Some(func_ty), Some(decl_ty)) => {
                Self::type_expr_matches_decl(func_ty, decl_ty, type_params)
            }
            (None, Some(decl_ty)) => Self::type_expr_is_nothing(decl_ty),
            (Some(func_ty), None) => Self::type_expr_is_nothing(func_ty),
        }
    }

    fn type_expr_is_nothing(ty: &TypeExpr) -> bool {
        matches!(ty, TypeExpr::Named(name) if name.name == "nothing")
    }

    fn type_expr_matches_decl(
        func_ty: &TypeExpr,
        decl_ty: &TypeExpr,
        type_params: &HashMap<String, String>,
    ) -> bool {
        match (func_ty, decl_ty) {
            (TypeExpr::Named(func_name), TypeExpr::Named(decl_name)) => {
                if let Some(expected_func_name) = type_params.get(&decl_name.name) {
                    func_name.name == *expected_func_name
                } else {
                    func_name.name == decl_name.name
                }
            }
            (
                TypeExpr::Generic(func_name, func_args, _),
                TypeExpr::Generic(decl_name, decl_args, _),
            ) => {
                func_name.name == decl_name.name
                    && func_args.len() == decl_args.len()
                    && func_args
                        .iter()
                        .zip(decl_args.iter())
                        .all(|(func_arg, decl_arg)| {
                            Self::type_expr_matches_decl(func_arg, decl_arg, type_params)
                        })
            }
            (TypeExpr::View(func_inner, _), TypeExpr::View(decl_inner, _)) => {
                Self::type_expr_matches_decl(func_inner, decl_inner, type_params)
            }
            (
                TypeExpr::StateQualified(func_inner, func_state, _),
                TypeExpr::StateQualified(decl_inner, decl_state, _),
            ) => {
                func_state.name == decl_state.name
                    && Self::type_expr_matches_decl(func_inner, decl_inner, type_params)
            }
            (
                TypeExpr::Function(func_params, func_return, _),
                TypeExpr::Function(decl_params, decl_return, _),
            ) => {
                func_params.len() == decl_params.len()
                    && func_params
                        .iter()
                        .zip(decl_params.iter())
                        .all(|(func_param, decl_param)| {
                            Self::type_expr_matches_decl(func_param, decl_param, type_params)
                        })
                    && Self::type_expr_matches_decl(func_return, decl_return, type_params)
            }
            _ => false,
        }
    }

    // ------------------------------------------------------------------
    // Function body
    // ------------------------------------------------------------------

    fn check_function(&mut self, func: &FunctionDef) {
        self.check_function_impl(func, func.name.name.clone());
    }

    fn check_method(&mut self, owner: &str, func: &FunctionDef) {
        self.check_function_impl(func, format!("{owner}.{}", func.name.name));
    }

    fn check_generic_function_instantiation(
        &mut self,
        function_name: &str,
        func: &FunctionDef,
        concrete_args: &[TypeId],
        subst: HashMap<String, TypeId>,
        kind_subst: HashMap<String, String>,
        param_facts: ReflectionParamFacts,
    ) {
        let uses_type_param_reflection = self.generic_function_uses_type_param_reflection(func);
        let branch_specializable = uses_type_param_reflection
            && self.generic_function_reflection_is_branch_specializable(func);
        if uses_type_param_reflection && !branch_specializable {
            return;
        }
        let specialize_reflection_branches = branch_specializable || !param_facts.is_empty();

        let kind_key = func
            .type_params
            .iter()
            .map(|param| {
                kind_subst
                    .get(&param.name)
                    .cloned()
                    .unwrap_or_else(|| "unknown_type".to_string())
            })
            .collect::<Vec<_>>();
        let cache_key = (
            function_name.to_string(),
            concrete_args.to_vec(),
            kind_key,
            param_facts.clone(),
        );
        if !self
            .checked_generic_function_instantiations
            .insert(cache_key)
        {
            return;
        }

        let type_arg_names = concrete_args
            .iter()
            .map(|&ty| self.type_name(ty))
            .collect::<Vec<_>>();
        let instantiated_name = format!("{function_name}[{}]", type_arg_names.join(", "));

        let old_subst = std::mem::replace(&mut self.type_var_subst, subst);
        let old_kind_subst = std::mem::replace(&mut self.type_var_kind_tags, kind_subst);
        let old_return_type = self.current_return_type;
        let old_function_name = self.current_function_name.clone();
        let old_function_pure = self.current_function_pure;
        let old_in_verify_block = self.in_verify_block;
        let old_in_property_block = self.in_property_block;
        let old_verify_name = self.current_verify_name.clone();
        let old_handle_body_depth = self.handle_body_depth;
        let old_respond_type = self.current_respond_type;
        let old_specialize_reflection_branches = self.specialize_reflection_branches;

        self.in_verify_block = false;
        self.in_property_block = false;
        self.current_verify_name = None;
        self.handle_body_depth = 0;
        self.current_respond_type = None;
        self.specialize_reflection_branches = specialize_reflection_branches;

        self.push_reflection_param_fact_scope(func, &param_facts);
        self.check_function_impl(func, instantiated_name);
        self.pop_reflection_local_fact_scope();

        self.type_var_subst = old_subst;
        self.type_var_kind_tags = old_kind_subst;
        self.current_return_type = old_return_type;
        self.current_function_name = old_function_name;
        self.current_function_pure = old_function_pure;
        self.in_verify_block = old_in_verify_block;
        self.in_property_block = old_in_property_block;
        self.current_verify_name = old_verify_name;
        self.handle_body_depth = old_handle_body_depth;
        self.current_respond_type = old_respond_type;
        self.specialize_reflection_branches = old_specialize_reflection_branches;
    }

    fn generic_function_uses_type_param_reflection(&self, func: &FunctionDef) -> bool {
        let type_params = func
            .type_params
            .iter()
            .map(|param| param.name.clone())
            .collect::<HashSet<_>>();
        self.block_uses_type_param_reflection(&func.body, &type_params)
    }

    fn generic_function_reflection_is_branch_specializable(&self, func: &FunctionDef) -> bool {
        let type_params = func
            .type_params
            .iter()
            .map(|param| param.name.clone())
            .collect::<HashSet<_>>();
        self.block_reflection_is_branch_specializable(&func.body, &type_params)
    }

    fn block_reflection_is_branch_specializable(
        &self,
        block: &Block,
        type_params: &HashSet<String>,
    ) -> bool {
        self.block_reflection_is_branch_specializable_in_context(
            block,
            type_params,
            ReflectionBranchContext::TopLevel,
        )
    }

    fn block_reflection_is_branch_specializable_in_context(
        &self,
        block: &Block,
        type_params: &HashSet<String>,
        context: ReflectionBranchContext,
    ) -> bool {
        block
            .stmts
            .iter()
            .all(|stmt| self.stmt_reflection_is_branch_specializable(stmt, type_params, context))
    }

    fn stmt_reflection_is_branch_specializable(
        &self,
        stmt: &Stmt,
        type_params: &HashSet<String>,
        context: ReflectionBranchContext,
    ) -> bool {
        if context == ReflectionBranchContext::StaticReflectionBranch {
            return true;
        }

        match stmt {
            Stmt::If(if_stmt) => {
                let condition_static =
                    self.expr_is_potential_static_reflection_condition(&if_stmt.condition);
                if self.expr_uses_type_param_reflection(&if_stmt.condition, type_params)
                    && !condition_static
                {
                    return false;
                }

                let then_context = if condition_static {
                    ReflectionBranchContext::StaticReflectionBranch
                } else {
                    ReflectionBranchContext::RuntimeBranch
                };
                let then_ok = self.block_reflection_is_branch_specializable_in_context(
                    &if_stmt.then_block,
                    type_params,
                    then_context,
                );

                let mut all_conditions_static = condition_static;
                let else_ifs_ok = if_stmt.else_ifs.iter().all(|(condition, block)| {
                    let else_if_static =
                        self.expr_is_potential_static_reflection_condition(condition);
                    all_conditions_static &= else_if_static;
                    if self.expr_uses_type_param_reflection(condition, type_params)
                        && !else_if_static
                    {
                        return false;
                    }
                    let else_if_context = if else_if_static {
                        ReflectionBranchContext::StaticReflectionBranch
                    } else {
                        ReflectionBranchContext::RuntimeBranch
                    };
                    self.block_reflection_is_branch_specializable_in_context(
                        block,
                        type_params,
                        else_if_context,
                    )
                });

                then_ok
                    && else_ifs_ok
                    && if_stmt.else_block.as_ref().is_none_or(|block| {
                        let else_context = if all_conditions_static {
                            ReflectionBranchContext::StaticReflectionBranch
                        } else {
                            ReflectionBranchContext::RuntimeBranch
                        };
                        self.block_reflection_is_branch_specializable_in_context(
                            block,
                            type_params,
                            else_context,
                        )
                    })
            }
            Stmt::VarDecl(decl) => {
                !self.expr_uses_type_param_reflection(&decl.value, type_params)
                    || (context.permits_shape_reflection()
                        && (self
                            .expr_is_direct_reflection_statement_source(&decl.value, type_params)
                            || (!decl.mutable
                                && self.expr_is_reflection_local_fact_source(
                                    &decl.value,
                                    type_params,
                                ))))
            }
            Stmt::Return(ret) => ret.value.as_ref().is_none_or(|expr| {
                !self.expr_uses_type_param_reflection(expr, type_params)
                    || (context.permits_shape_reflection()
                        && self.expr_is_direct_reflection_statement_source(expr, type_params))
            }),
            Stmt::ComptimeTypeBind(bind) => {
                self.comptime_type_bind_reflection_is_specializable(bind, type_params, context)
            }
            Stmt::For(for_stmt) => {
                self.for_reflection_is_branch_specializable(for_stmt, type_params, context)
            }
            Stmt::Match(match_stmt) => {
                self.match_reflection_is_branch_specializable(match_stmt, type_params, context)
            }
            _ => !self.stmt_uses_type_param_reflection(stmt, type_params),
        }
    }

    fn comptime_type_bind_reflection_is_specializable(
        &self,
        bind: &ast::ComptimeTypeBindStmt,
        type_params: &HashSet<String>,
        context: ReflectionBranchContext,
    ) -> bool {
        let source_mentions_type_param = comptime_type_info_binding(&bind.value)
            .or_else(|| comptime_type_arg_binding(&bind.value).map(|(ty, _)| ty))
            .is_some_and(|ty| Self::type_expr_mentions_type_param(ty, type_params));

        if source_mentions_type_param {
            return context.permits_shape_reflection()
                && self.block_reflection_is_branch_specializable_in_context(
                    &bind.body,
                    type_params,
                    context,
                );
        }

        !self.expr_uses_type_param_reflection(&bind.value, type_params)
            && self.block_reflection_is_branch_specializable_in_context(
                &bind.body,
                type_params,
                context,
            )
    }

    fn for_reflection_is_branch_specializable(
        &self,
        for_stmt: &ast::ForStmt,
        type_params: &HashSet<String>,
        context: ReflectionBranchContext,
    ) -> bool {
        if let Some(owner_ty) = direct_reflected_loop_owner_type(&for_stmt.iterable)
            && Self::type_expr_mentions_type_param(owner_ty, type_params)
        {
            return context.permits_shape_reflection()
                && self.block_reflection_is_branch_specializable_in_context(
                    &for_stmt.body,
                    type_params,
                    ReflectionBranchContext::StaticReflectionBranch,
                );
        }

        !self.expr_uses_type_param_reflection(&for_stmt.iterable, type_params)
            && self.block_reflection_is_branch_specializable_in_context(
                &for_stmt.body,
                type_params,
                context,
            )
    }

    fn match_reflection_is_branch_specializable(
        &self,
        match_stmt: &ast::MatchStmt,
        type_params: &HashSet<String>,
        _context: ReflectionBranchContext,
    ) -> bool {
        let scrutinee_static = self.expr_is_potential_static_reflection_value(&match_stmt.expr);
        if self.expr_uses_type_param_reflection(&match_stmt.expr, type_params) && !scrutinee_static
        {
            return false;
        }

        let arm_context = if scrutinee_static {
            ReflectionBranchContext::StaticReflectionBranch
        } else {
            ReflectionBranchContext::RuntimeBranch
        };

        match_stmt.arms.iter().all(|arm| {
            self.block_reflection_is_branch_specializable_in_context(
                &arm.body,
                type_params,
                arm_context,
            )
        })
    }

    fn expr_is_reflection_local_fact_source(
        &self,
        expr: &Expr,
        type_params: &HashSet<String>,
    ) -> bool {
        self.expr_is_type_info_reflection(expr, type_params)
            || self.expr_is_type_kind_reflection(expr, type_params)
            || self.expr_is_type_primitive_reflection_value(expr, type_params)
    }

    fn expr_is_direct_reflection_statement_source(
        &self,
        expr: &Expr,
        type_params: &HashSet<String>,
    ) -> bool {
        match expr {
            Expr::Paren(inner, _) => {
                self.expr_is_direct_reflection_statement_source(inner, type_params)
            }
            Expr::Handle(target, _, _, _) => {
                self.expr_is_direct_reflection_statement_source(target, type_params)
            }
            Expr::GenericCall(callee, type_args, _, _) => {
                self.resolved_expr_name(callee).is_some_and(|name| {
                    matches!(
                        name.as_str(),
                        "type.variant_value"
                            | "type.construct_start"
                            | "type.construct_variant_start"
                            | "type.construct_machine_start"
                            | "type.construct_finish"
                    )
                }) && type_args
                    .iter()
                    .any(|arg| Self::type_expr_mentions_type_param(arg, type_params))
            }
            _ => false,
        }
    }

    fn expr_is_potential_static_reflection_condition(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BoolLiteral(_, _) => true,
            Expr::Paren(inner, _) => self.expr_is_potential_static_reflection_condition(inner),
            Expr::Unary(UnaryOp::Not, inner, _) => {
                self.expr_is_potential_static_reflection_condition(inner)
            }
            Expr::Binary(lhs, BinOp::And | BinOp::Or, rhs, _) => {
                self.expr_is_potential_static_reflection_condition(lhs)
                    && self.expr_is_potential_static_reflection_condition(rhs)
            }
            Expr::Binary(lhs, BinOp::Eq | BinOp::NotEq, rhs, _) => {
                (self.expr_is_potential_static_reflection_value(lhs)
                    && self.expr_is_static_reflection_literal(rhs))
                    || (self.expr_is_static_reflection_literal(lhs)
                        && self.expr_is_potential_static_reflection_value(rhs))
            }
            _ => false,
        }
    }

    fn expr_is_potential_static_reflection_value(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Paren(inner, _) => self.expr_is_potential_static_reflection_value(inner),
            Expr::Ident(_) => true,
            Expr::GenericCall(callee, _, args, _) => {
                args.is_empty()
                    && self
                        .resolved_expr_name(callee)
                        .is_some_and(|name| name == "type.kind_tag")
            }
            Expr::FieldAccess(_, field, _) if field.name == "kind_tag" => true,
            Expr::Handle(target, _, body, _) => {
                self.expr_is_potential_optional_type_primitive_reflection(target)
                    && Self::handle_default_expr(body)
                        .is_some_and(|default| self.expr_is_type_primitive_literal(default))
            }
            _ => self.expr_is_static_reflection_value(expr, &HashSet::new()),
        }
    }

    fn expr_is_potential_optional_type_primitive_reflection(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Paren(inner, _) => {
                self.expr_is_potential_optional_type_primitive_reflection(inner)
            }
            Expr::FieldAccess(_, field, _) if field.name == "primitive_tag" => true,
            Expr::GenericCall(callee, _, args, _) => {
                args.is_empty()
                    && self
                        .resolved_expr_name(callee)
                        .is_some_and(|name| name == "type.primitive_tag")
            }
            _ => false,
        }
    }

    fn expr_is_static_reflection_value(&self, expr: &Expr, type_params: &HashSet<String>) -> bool {
        self.expr_is_type_kind_reflection(expr, type_params)
            || self.expr_is_type_primitive_reflection_value(expr, type_params)
    }

    fn expr_is_static_reflection_literal(&self, expr: &Expr) -> bool {
        self.expr_is_type_kind_literal(expr) || self.expr_is_type_primitive_literal(expr)
    }

    fn expr_is_type_kind_reflection(&self, expr: &Expr, type_params: &HashSet<String>) -> bool {
        match expr {
            Expr::GenericCall(callee, type_args, args, _) => {
                args.is_empty()
                    && self
                        .resolved_expr_name(callee)
                        .is_some_and(|name| name == "type.kind_tag")
                    && type_args.len() == 1
                    && Self::type_expr_mentions_type_param(&type_args[0], type_params)
            }
            Expr::FieldAccess(base, field, _) if field.name == "kind_tag" => {
                self.expr_is_type_info_reflection(base, type_params)
            }
            Expr::Paren(inner, _) => self.expr_is_type_kind_reflection(inner, type_params),
            _ => false,
        }
    }

    fn expr_is_type_primitive_reflection_value(
        &self,
        expr: &Expr,
        type_params: &HashSet<String>,
    ) -> bool {
        match expr {
            Expr::Handle(target, _, body, _) => {
                self.expr_is_optional_type_primitive_reflection(target, type_params)
                    && Self::handle_default_expr(body)
                        .is_some_and(|default| self.expr_is_type_primitive_literal(default))
            }
            Expr::Paren(inner, _) => {
                self.expr_is_type_primitive_reflection_value(inner, type_params)
            }
            _ => false,
        }
    }

    fn expr_is_optional_type_primitive_reflection(
        &self,
        expr: &Expr,
        type_params: &HashSet<String>,
    ) -> bool {
        match expr {
            Expr::GenericCall(callee, type_args, args, _) => {
                args.is_empty()
                    && self
                        .resolved_expr_name(callee)
                        .is_some_and(|name| name == "type.primitive_tag")
                    && type_args.len() == 1
                    && Self::type_expr_mentions_type_param(&type_args[0], type_params)
            }
            Expr::FieldAccess(base, field, _) if field.name == "primitive_tag" => {
                self.expr_is_type_info_reflection(base, type_params)
            }
            Expr::Paren(inner, _) => {
                self.expr_is_optional_type_primitive_reflection(inner, type_params)
            }
            _ => false,
        }
    }

    fn expr_is_type_info_reflection(&self, expr: &Expr, type_params: &HashSet<String>) -> bool {
        match expr {
            Expr::GenericCall(callee, type_args, args, _) => {
                args.is_empty()
                    && self
                        .resolved_expr_name(callee)
                        .is_some_and(|name| name == "type.info")
                    && type_args.len() == 1
                    && Self::type_expr_mentions_type_param(&type_args[0], type_params)
            }
            Expr::Paren(inner, _) => self.expr_is_type_info_reflection(inner, type_params),
            _ => false,
        }
    }

    fn expr_is_type_kind_literal(&self, expr: &Expr) -> bool {
        self.type_kind_literal_name(expr).is_some()
    }

    fn expr_is_type_primitive_literal(&self, expr: &Expr) -> bool {
        self.type_primitive_literal_name(expr).is_some()
    }

    fn type_kind_literal_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Paren(inner, _) => self.type_kind_literal_name(inner),
            Expr::EnumVariant(type_name, variant, _) if type_name.name == "TypeKind" => {
                Some(variant.name.clone())
            }
            Expr::FieldAccess(_, _, _) => {
                let name = self.expanded_dotted_expr_name(expr)?;
                let (type_name, variant_name) = name.rsplit_once('.')?;
                (type_name == "TypeKind").then(|| variant_name.to_string())
            }
            _ => None,
        }
    }

    fn type_primitive_literal_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Paren(inner, _) => self.type_primitive_literal_name(inner),
            Expr::EnumVariant(type_name, variant, _) if type_name.name == "TypePrimitive" => {
                Some(variant.name.clone())
            }
            Expr::FieldAccess(_, _, _) => {
                let name = self.expanded_dotted_expr_name(expr)?;
                let (type_name, variant_name) = name.rsplit_once('.')?;
                (type_name == "TypePrimitive").then(|| variant_name.to_string())
            }
            _ => None,
        }
    }

    fn handle_default_expr(body: &Block) -> Option<&Expr> {
        let Stmt::Expr(expr_stmt) = body.stmts.last()? else {
            return None;
        };
        let Expr::Default(default_value, _) = &expr_stmt.expr else {
            return None;
        };
        Some(default_value)
    }

    fn eval_static_type_condition(&mut self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::BoolLiteral(value, _) => Some(*value),
            Expr::Paren(inner, _) => self.eval_static_type_condition(inner),
            Expr::Unary(UnaryOp::Not, inner, _) => {
                self.eval_static_type_condition(inner).map(|value| !value)
            }
            Expr::Binary(lhs, BinOp::And, rhs, _) => match self.eval_static_type_condition(lhs)? {
                false => Some(false),
                true => self.eval_static_type_condition(rhs),
            },
            Expr::Binary(lhs, BinOp::Or, rhs, _) => match self.eval_static_type_condition(lhs)? {
                true => Some(true),
                false => self.eval_static_type_condition(rhs),
            },
            Expr::Binary(lhs, BinOp::Eq | BinOp::NotEq, rhs, _) => {
                let lhs_value = self.eval_static_reflection_enum_value(lhs)?;
                let rhs_value = self.eval_static_reflection_enum_value(rhs)?;
                let equal = lhs_value == rhs_value;
                Some(if matches!(expr, Expr::Binary(_, BinOp::Eq, _, _)) {
                    equal
                } else {
                    !equal
                })
            }
            _ => None,
        }
    }

    fn eval_static_reflection_enum_value(
        &mut self,
        expr: &Expr,
    ) -> Option<StaticReflectionEnumValue> {
        if let Some(kind_tag) = self.eval_type_kind_value(expr) {
            return Some(StaticReflectionEnumValue::TypeKind(kind_tag));
        }
        self.eval_type_primitive_value(expr)
            .map(StaticReflectionEnumValue::TypePrimitive)
    }

    fn eval_type_kind_value(&mut self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Paren(inner, _) => self.eval_type_kind_value(inner),
            Expr::Ident(ident) => self.reflection_type_kind_value_for_ident(ident),
            Expr::GenericCall(callee, type_args, args, _)
                if args.is_empty()
                    && self
                        .resolved_expr_name(callee)
                        .is_some_and(|name| name == "type.kind_tag") =>
            {
                self.reflection_type_arg_kind_tag(type_args)
            }
            Expr::FieldAccess(base, field, _) if field.name == "kind_tag" => {
                self.eval_type_info_type_arg_kind_tag(base)
            }
            _ => self.type_kind_literal_name(expr),
        }
    }

    fn eval_type_primitive_value(&mut self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Paren(inner, _) => self.eval_type_primitive_value(inner),
            Expr::Ident(ident) => self.reflection_type_primitive_value_for_ident(ident),
            Expr::Handle(target, _, body, _) => {
                match self.eval_optional_type_primitive_value(target)? {
                    Some(primitive_tag) => Some(primitive_tag),
                    None => Self::handle_default_expr(body)
                        .and_then(|default| self.eval_type_primitive_value(default)),
                }
            }
            _ => self.type_primitive_literal_name(expr),
        }
    }

    fn eval_optional_type_primitive_value(&mut self, expr: &Expr) -> Option<Option<String>> {
        match expr {
            Expr::Paren(inner, _) => self.eval_optional_type_primitive_value(inner),
            Expr::GenericCall(callee, type_args, args, _)
                if args.is_empty()
                    && self
                        .resolved_expr_name(callee)
                        .is_some_and(|name| name == "type.primitive_tag") =>
            {
                Some(self.reflection_type_arg_primitive_tag(type_args))
            }
            Expr::FieldAccess(base, field, _) if field.name == "primitive_tag" => {
                self.eval_type_info_primitive_tag(base)
            }
            _ => None,
        }
    }

    fn eval_type_info_type_arg_kind_tag(&mut self, expr: &Expr) -> Option<String> {
        self.eval_type_info_facts(expr).map(|facts| facts.kind_tag)
    }

    fn eval_type_info_primitive_tag(&mut self, expr: &Expr) -> Option<Option<String>> {
        self.eval_type_info_facts(expr)
            .map(|facts| facts.primitive_tag)
    }

    fn eval_type_info_facts(&mut self, expr: &Expr) -> Option<ReflectionTypeInfoStaticFacts> {
        match expr {
            Expr::Paren(inner, _) => self.eval_type_info_facts(inner),
            Expr::View(inner, _) => self.eval_type_info_facts(inner),
            Expr::Ident(ident) => {
                let kind_tag = self.reflection_type_info_kind_for_ident(ident)?;
                let primitive_tag = self
                    .reflection_type_info_primitive_for_ident(ident)
                    .unwrap_or(None);
                Some(ReflectionTypeInfoStaticFacts {
                    kind_tag,
                    primitive_tag,
                })
            }
            Expr::GenericCall(callee, type_args, args, _)
                if args.is_empty()
                    && self
                        .resolved_expr_name(callee)
                        .is_some_and(|name| name == "type.info") =>
            {
                let kind_tag = self.reflection_type_arg_kind_tag(type_args)?;
                let primitive_tag = self.reflection_type_arg_primitive_tag(type_args);
                Some(ReflectionTypeInfoStaticFacts {
                    kind_tag,
                    primitive_tag,
                })
            }
            _ => None,
        }
    }

    fn reflection_type_arg_kind_tag(&mut self, type_args: &[TypeExpr]) -> Option<String> {
        let [type_arg] = type_args else {
            return None;
        };
        let type_id = self.reflection_type_arg_id(type_args)?;
        Some(self.reflection_kind_tag_for_type_expr(type_arg, type_id))
    }

    fn reflection_type_arg_primitive_tag(&mut self, type_args: &[TypeExpr]) -> Option<String> {
        let [type_arg] = type_args else {
            return None;
        };
        let type_id = self.reflection_type_arg_id(type_args)?;
        self.reflection_primitive_tag_for_type_expr_static(type_arg, type_id)
    }

    fn reflection_type_arg_id(&mut self, type_args: &[TypeExpr]) -> Option<TypeId> {
        let [type_arg] = type_args else {
            return None;
        };
        if let TypeExpr::Named(ident) = type_arg
            && let Some(&type_id) = self.type_var_subst.get(&ident.name)
        {
            return Some(type_id);
        }
        let type_id = self.resolve_type_expr(type_arg);
        (type_id != TypeInterner::ERROR).then_some(type_id)
    }

    fn block_uses_type_param_reflection(
        &self,
        block: &Block,
        type_params: &HashSet<String>,
    ) -> bool {
        block
            .stmts
            .iter()
            .any(|stmt| self.stmt_uses_type_param_reflection(stmt, type_params))
    }

    fn block_type_param_reflection_span(
        &self,
        block: &Block,
        type_params: &HashSet<String>,
    ) -> Option<Span> {
        block
            .stmts
            .iter()
            .find_map(|stmt| self.stmt_type_param_reflection_span(stmt, type_params))
    }

    fn if_branch_type_param_reflection_span(
        &self,
        if_stmt: &ast::IfStmt,
        type_params: &HashSet<String>,
    ) -> Option<Span> {
        self.block_type_param_reflection_span(&if_stmt.then_block, type_params)
            .or_else(|| {
                if_stmt.else_ifs.iter().find_map(|(_, block)| {
                    self.block_type_param_reflection_span(block, type_params)
                })
            })
            .or_else(|| {
                if_stmt
                    .else_block
                    .as_ref()
                    .and_then(|block| self.block_type_param_reflection_span(block, type_params))
            })
    }

    fn stmt_type_param_reflection_span(
        &self,
        stmt: &Stmt,
        type_params: &HashSet<String>,
    ) -> Option<Span> {
        match stmt {
            Stmt::VarDecl(decl) => self
                .expr_uses_type_param_reflection(&decl.value, type_params)
                .then(|| decl.value.span()),
            Stmt::Assign(assign) => self
                .expr_uses_type_param_reflection(&assign.target, type_params)
                .then(|| assign.target.span())
                .or_else(|| {
                    self.expr_uses_type_param_reflection(&assign.value, type_params)
                        .then(|| assign.value.span())
                }),
            Stmt::Return(ret) => ret.value.as_ref().and_then(|expr| {
                self.expr_uses_type_param_reflection(expr, type_params)
                    .then(|| expr.span())
            }),
            Stmt::Respond(resp) => self
                .expr_uses_type_param_reflection(&resp.value, type_params)
                .then(|| resp.value.span()),
            Stmt::ComptimeTypeBind(bind) => self
                .expr_uses_type_param_reflection(&bind.value, type_params)
                .then(|| bind.value.span())
                .or_else(|| self.block_type_param_reflection_span(&bind.body, type_params)),
            Stmt::If(if_stmt) => self
                .expr_uses_type_param_reflection(&if_stmt.condition, type_params)
                .then(|| if_stmt.condition.span())
                .or_else(|| self.if_branch_type_param_reflection_span(if_stmt, type_params)),
            Stmt::For(for_stmt) => self
                .expr_uses_type_param_reflection(&for_stmt.iterable, type_params)
                .then(|| for_stmt.iterable.span())
                .or_else(|| self.block_type_param_reflection_span(&for_stmt.body, type_params)),
            Stmt::While(while_stmt) => self
                .expr_uses_type_param_reflection(&while_stmt.condition, type_params)
                .then(|| while_stmt.condition.span())
                .or_else(|| self.block_type_param_reflection_span(&while_stmt.body, type_params)),
            Stmt::Match(match_stmt) => self
                .expr_uses_type_param_reflection(&match_stmt.expr, type_params)
                .then(|| match_stmt.expr.span())
                .or_else(|| {
                    match_stmt.arms.iter().find_map(|arm| {
                        self.block_type_param_reflection_span(&arm.body, type_params)
                    })
                }),
            Stmt::Expr(expr_stmt) => self
                .expr_uses_type_param_reflection(&expr_stmt.expr, type_params)
                .then(|| expr_stmt.expr.span()),
            Stmt::Assert(assert_stmt) => self
                .expr_uses_type_param_reflection(&assert_stmt.condition, type_params)
                .then(|| assert_stmt.condition.span())
                .or_else(|| {
                    assert_stmt.message.as_ref().and_then(|message| {
                        self.expr_uses_type_param_reflection(message, type_params)
                            .then(|| message.span())
                    })
                }),
            Stmt::Breakpoint(breakpoint_stmt) => {
                breakpoint_stmt.condition.as_ref().and_then(|expr| {
                    self.expr_uses_type_param_reflection(expr, type_params)
                        .then(|| expr.span())
                })
            }
            Stmt::Trace(_) | Stmt::Use(_) | Stmt::Break(_) | Stmt::Continue(_) => None,
        }
    }

    fn stmt_uses_type_param_reflection(&self, stmt: &Stmt, type_params: &HashSet<String>) -> bool {
        match stmt {
            Stmt::VarDecl(decl) => self.expr_uses_type_param_reflection(&decl.value, type_params),
            Stmt::Assign(assign) => {
                self.expr_uses_type_param_reflection(&assign.target, type_params)
                    || self.expr_uses_type_param_reflection(&assign.value, type_params)
            }
            Stmt::Return(ret) => ret
                .value
                .as_ref()
                .is_some_and(|expr| self.expr_uses_type_param_reflection(expr, type_params)),
            Stmt::Respond(resp) => self.expr_uses_type_param_reflection(&resp.value, type_params),
            Stmt::ComptimeTypeBind(bind) => {
                self.expr_uses_type_param_reflection(&bind.value, type_params)
                    || self.block_uses_type_param_reflection(&bind.body, type_params)
            }
            Stmt::If(if_stmt) => {
                self.expr_uses_type_param_reflection(&if_stmt.condition, type_params)
                    || self.block_uses_type_param_reflection(&if_stmt.then_block, type_params)
                    || if_stmt.else_ifs.iter().any(|(condition, block)| {
                        self.expr_uses_type_param_reflection(condition, type_params)
                            || self.block_uses_type_param_reflection(block, type_params)
                    })
                    || if_stmt.else_block.as_ref().is_some_and(|block| {
                        self.block_uses_type_param_reflection(block, type_params)
                    })
            }
            Stmt::For(for_stmt) => {
                self.expr_uses_type_param_reflection(&for_stmt.iterable, type_params)
                    || self.block_uses_type_param_reflection(&for_stmt.body, type_params)
            }
            Stmt::While(while_stmt) => {
                self.expr_uses_type_param_reflection(&while_stmt.condition, type_params)
                    || self.block_uses_type_param_reflection(&while_stmt.body, type_params)
            }
            Stmt::Match(match_stmt) => {
                self.expr_uses_type_param_reflection(&match_stmt.expr, type_params)
                    || match_stmt
                        .arms
                        .iter()
                        .any(|arm| self.block_uses_type_param_reflection(&arm.body, type_params))
            }
            Stmt::Expr(expr_stmt) => {
                self.expr_uses_type_param_reflection(&expr_stmt.expr, type_params)
            }
            Stmt::Assert(assert_stmt) => {
                self.expr_uses_type_param_reflection(&assert_stmt.condition, type_params)
                    || assert_stmt.message.as_ref().is_some_and(|message| {
                        self.expr_uses_type_param_reflection(message, type_params)
                    })
            }
            Stmt::Breakpoint(breakpoint_stmt) => breakpoint_stmt
                .condition
                .as_ref()
                .is_some_and(|expr| self.expr_uses_type_param_reflection(expr, type_params)),
            Stmt::Trace(_) | Stmt::Use(_) | Stmt::Break(_) | Stmt::Continue(_) => false,
        }
    }

    fn expr_uses_type_param_reflection(&self, expr: &Expr, type_params: &HashSet<String>) -> bool {
        match expr {
            Expr::GenericCall(callee, type_args, args, _) => {
                let is_type_reflection = self
                    .resolved_expr_name(callee)
                    .is_some_and(|name| name.starts_with("type."));
                (is_type_reflection
                    && type_args
                        .iter()
                        .any(|arg| Self::type_expr_mentions_type_param(arg, type_params)))
                    || self.expr_uses_type_param_reflection(callee, type_params)
                    || args
                        .iter()
                        .any(|arg| self.expr_uses_type_param_reflection(&arg.value, type_params))
            }
            Expr::Call(callee, args, _) => {
                self.expr_uses_type_param_reflection(callee, type_params)
                    || args
                        .iter()
                        .any(|arg| self.expr_uses_type_param_reflection(&arg.value, type_params))
            }
            Expr::Binary(lhs, _, rhs, _) => {
                self.expr_uses_type_param_reflection(lhs, type_params)
                    || self.expr_uses_type_param_reflection(rhs, type_params)
            }
            Expr::Unary(_, inner, _)
            | Expr::FieldAccess(inner, _, _)
            | Expr::Paren(inner, _)
            | Expr::View(inner, _)
            | Expr::Ok(inner, _)
            | Expr::Fail(inner, _)
            | Expr::Some(inner, _)
            | Expr::Default(inner, _)
            | Expr::Declassify(inner, _)
            | Expr::Coarsen(inner, _)
            | Expr::At(inner, _, _)
            | Expr::Spawn(inner, _)
            | Expr::Send(inner, _)
            | Expr::Ask(inner, _)
            | Expr::Clone(inner, _)
            | Expr::Run(inner, _)
            | Expr::Join(inner, _)
            | Expr::Cancel(inner, _) => self.expr_uses_type_param_reflection(inner, type_params),
            Expr::ListConstruct(items, _) => items
                .iter()
                .any(|item| self.expr_uses_type_param_reflection(item, type_params)),
            Expr::MapConstruct(entries, _) => entries.iter().any(|(key, value)| {
                self.expr_uses_type_param_reflection(key, type_params)
                    || self.expr_uses_type_param_reflection(value, type_params)
            }),
            Expr::Handle(target, _, body, _) => {
                self.expr_uses_type_param_reflection(target, type_params)
                    || self.block_uses_type_param_reflection(body, type_params)
            }
            Expr::StringInterpolation(parts, _) => parts.iter().any(|part| match part {
                StringPart::Literal(_) => false,
                StringPart::Expr(expr) => self.expr_uses_type_param_reflection(expr, type_params),
            }),
            Expr::Pipeline(base, steps, _) => {
                self.expr_uses_type_param_reflection(base, type_params)
                    || steps.iter().any(|step| {
                        self.expr_uses_type_param_reflection(&step.function, type_params)
                            || step.extra_args.iter().any(|arg| {
                                self.expr_uses_type_param_reflection(&arg.value, type_params)
                            })
                    })
            }
            Expr::InlineFn(_, return_ty, body, _) => {
                return_ty
                    .as_ref()
                    .is_some_and(|ty| Self::type_expr_mentions_type_param(ty, type_params))
                    || self.block_uses_type_param_reflection(body, type_params)
            }
            Expr::IntLiteral(_, _)
            | Expr::FloatLiteral(_, _)
            | Expr::StringLiteral(_, _)
            | Expr::BoolLiteral(_, _)
            | Expr::Nothing(_)
            | Expr::Ident(_)
            | Expr::None(_)
            | Expr::EnumVariant(_, _, _)
            | Expr::Error(_) => false,
        }
    }

    fn type_expr_mentions_type_param(ty: &TypeExpr, type_params: &HashSet<String>) -> bool {
        match ty {
            TypeExpr::Named(ident) => type_params.contains(&ident.name),
            TypeExpr::Generic(_, args, _) => args
                .iter()
                .any(|arg| Self::type_expr_mentions_type_param(arg, type_params)),
            TypeExpr::View(inner, _) => Self::type_expr_mentions_type_param(inner, type_params),
            TypeExpr::StateQualified(inner, _, _) => {
                Self::type_expr_mentions_type_param(inner, type_params)
            }
            TypeExpr::Function(params, return_ty, _) => {
                params
                    .iter()
                    .any(|param| Self::type_expr_mentions_type_param(param, type_params))
                    || Self::type_expr_mentions_type_param(return_ty, type_params)
            }
        }
    }

    fn check_implement_block(&mut self, block: &ast::ImplementBlock) {
        let owner_ty = self.resolve_type_expr(&block.for_type);
        let owner_name = self.type_name(owner_ty);
        for method in &block.methods {
            self.check_method(&owner_name, method);
        }
    }

    fn check_function_impl(&mut self, func: &FunctionDef, function_name: String) {
        let return_type = func
            .return_type
            .as_ref()
            .map(|t| self.resolve_type_expr(t))
            .unwrap_or(TypeInterner::NOTHING);

        self.current_return_type = Some(return_type);

        // Set the purity context for this function.
        let is_pure = Self::function_is_pure(func);
        self.current_function_name = Some(function_name);
        self.current_function_pure = is_pure;

        // Bind parameter types into the type environment.
        for param in &func.params {
            let param_type = self.resolve_type_expr(&param.ty);
            if let Some(def_id) = self.declaration_def_id(param.name.span) {
                self.type_env.insert(def_id, param_type);
            }
        }

        self.check_block(&func.body);

        self.current_return_type = None;
        self.current_function_name = None;
        self.current_function_pure = false;
    }

    fn check_verify_block(&mut self, verify: &VerifyBlock) {
        self.in_verify_block = true;
        self.current_verify_name = Some(verify.name.name.clone());
        self.check_block(&verify.body);
        self.in_verify_block = false;
        self.current_verify_name = None;
    }

    fn check_property_block(&mut self, prop: &ast::PropertyBlock) {
        self.in_property_block = true;
        for given in &prop.givens {
            let given_type = self.resolve_type_expr(&given.ty);
            if let Some(def_id) = self.declaration_def_id(given.name.span) {
                self.type_env.insert(def_id, given_type);
            }
        }
        self.check_block(&prop.body);
        self.in_property_block = false;
    }

    // ------------------------------------------------------------------
    // Block
    // ------------------------------------------------------------------

    fn check_block(&mut self, block: &Block) {
        self.push_reflection_local_fact_scope();
        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }
        self.pop_reflection_local_fact_scope();
    }

    // ------------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------------

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::VarDecl(decl) => self.check_var_decl(decl),
            Stmt::Assign(assign) => self.check_assign(assign),
            Stmt::Return(ret) => self.check_return(ret),
            Stmt::ComptimeTypeBind(bind) => self.check_comptime_type_bind(bind),
            Stmt::If(if_stmt) => self.check_if(if_stmt),
            Stmt::For(for_stmt) => self.check_for(for_stmt),
            Stmt::While(while_stmt) => self.check_while(while_stmt),
            Stmt::Expr(expr_stmt) => {
                let ty = self.check_expr(&expr_stmt.expr);
                // Warn if a result or optional value is silently discarded.
                if ty != TypeInterner::ERROR {
                    let resolved = self.interner.resolve(ty);
                    if matches!(resolved, Type::Result(_, _)) {
                        self.sink.emit(errors::unhandled_result(expr_stmt.span));
                    } else if matches!(resolved, Type::Optional(_)) {
                        self.sink.emit(errors::unhandled_optional(expr_stmt.span));
                    }
                }
            }
            Stmt::Assert(assert_stmt) => self.check_assert(assert_stmt),
            Stmt::Trace(trace_stmt) => self.check_trace(trace_stmt),
            Stmt::Breakpoint(breakpoint_stmt) => self.check_breakpoint(breakpoint_stmt),
            Stmt::Match(match_stmt) => self.check_match(match_stmt),
            Stmt::Respond(resp) => self.check_respond(resp),
            Stmt::Use(_) | Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }

    fn check_comptime_type_bind(&mut self, bind: &ast::ComptimeTypeBindStmt) {
        self.check_expr(&bind.value);

        if let Some(bound_type_expr) = comptime_type_info_binding(&bind.value) {
            let bound_ty = self.resolve_type_expr(bound_type_expr);
            if bound_ty == TypeInterner::ERROR {
                self.check_block(&bind.body);
                return;
            }

            self.check_comptime_type_bind_body(&bind.name.name, bound_ty, &bind.body);
            return;
        }

        if let Some((source_type_expr, index)) = comptime_type_arg_binding(&bind.value) {
            let source_ty = self.resolve_type_expr(source_type_expr);
            if source_ty == TypeInterner::ERROR {
                self.check_block(&bind.body);
                return;
            }

            let arg_types = self.type_info_arg_types_for_type_expr(source_type_expr);
            if let Some(&bound_ty) = arg_types.get(index) {
                if bound_ty != TypeInterner::ERROR {
                    self.check_comptime_type_bind_body(&bind.name.name, bound_ty, &bind.body);
                }
                return;
            }

            self.sink
                .emit(errors::invalid_comptime_type_binding(bind.value.span()));
            self.check_block(&bind.body);
            return;
        }

        if let Some(field_name) = reflected_field_type_info_binding(&bind.value)
            && let Some(field_types) = self.reflected_field_types_for_name(field_name)
        {
            for field_ty in field_types {
                if field_ty != TypeInterner::ERROR {
                    self.check_comptime_type_bind_body(&bind.name.name, field_ty, &bind.body);
                }
            }
            return;
        }

        if let Some(info_name) = reflected_type_info_binding(&bind.value)
            && let Some(info_types) = self.reflected_type_info_types_for_name(info_name)
        {
            for info_ty in info_types {
                if info_ty != TypeInterner::ERROR {
                    self.check_comptime_type_bind_body(&bind.name.name, info_ty, &bind.body);
                }
            }
            return;
        }

        self.sink
            .emit(errors::invalid_comptime_type_binding(bind.value.span()));
        self.check_block(&bind.body);
    }

    fn check_comptime_type_bind_body(&mut self, name: &str, bound_ty: TypeId, body: &Block) {
        let previous = self.type_var_subst.insert(name.to_string(), bound_ty);
        self.check_block(body);
        if let Some(previous) = previous {
            self.type_var_subst.insert(name.to_string(), previous);
        } else {
            self.type_var_subst.remove(name);
        }
    }

    fn reflected_field_types_for_name(&self, name: &str) -> Option<Vec<TypeId>> {
        self.reflected_field_type_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .cloned()
    }

    fn reflected_type_info_types_for_name(&self, name: &str) -> Option<Vec<TypeId>> {
        self.reflected_type_info_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .cloned()
    }

    fn reflected_variant_owner_for_name(&self, name: &str) -> Option<TypeId> {
        self.reflected_variant_type_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .copied()
    }

    fn reflected_machine_state_owner_for_name(&self, name: &str) -> Option<TypeId> {
        self.reflected_machine_state_type_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .copied()
    }

    fn reflection_type_info_kind_for_ident(&self, ident: &ast::Ident) -> Option<String> {
        let def_id = self.ident_def_id(ident)?;
        self.reflection_type_info_kind_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(&def_id))
            .cloned()
    }

    fn reflection_type_info_primitive_for_ident(
        &self,
        ident: &ast::Ident,
    ) -> Option<Option<String>> {
        let def_id = self.ident_def_id(ident)?;
        self.reflection_type_info_primitive_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(&def_id))
            .cloned()
    }

    fn reflection_type_kind_value_for_ident(&self, ident: &ast::Ident) -> Option<String> {
        let def_id = self.ident_def_id(ident)?;
        self.reflection_type_kind_value_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(&def_id))
            .cloned()
    }

    fn reflection_type_primitive_value_for_ident(&self, ident: &ast::Ident) -> Option<String> {
        let def_id = self.ident_def_id(ident)?;
        self.reflection_type_primitive_value_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(&def_id))
            .cloned()
    }

    fn current_reflection_type_info_kind_scope_mut(
        &mut self,
    ) -> Option<&mut HashMap<DefId, String>> {
        self.reflection_type_info_kind_scopes.last_mut()
    }

    fn current_reflection_type_info_primitive_scope_mut(
        &mut self,
    ) -> Option<&mut HashMap<DefId, Option<String>>> {
        self.reflection_type_info_primitive_scopes.last_mut()
    }

    fn current_reflection_type_kind_value_scope_mut(
        &mut self,
    ) -> Option<&mut HashMap<DefId, String>> {
        self.reflection_type_kind_value_scopes.last_mut()
    }

    fn current_reflection_type_primitive_value_scope_mut(
        &mut self,
    ) -> Option<&mut HashMap<DefId, String>> {
        self.reflection_type_primitive_value_scopes.last_mut()
    }

    fn push_reflection_local_fact_scope(&mut self) {
        self.reflection_type_info_kind_scopes.push(HashMap::new());
        self.reflection_type_info_primitive_scopes
            .push(HashMap::new());
        self.reflection_type_kind_value_scopes.push(HashMap::new());
        self.reflection_type_primitive_value_scopes
            .push(HashMap::new());
    }

    fn push_reflection_param_fact_scope(
        &mut self,
        func: &FunctionDef,
        facts: &ReflectionParamFacts,
    ) {
        let mut type_info_scope = HashMap::new();
        for (index, kind_tag) in &facts.type_info_kinds {
            if let Some(param) = func.params.get(*index)
                && let Some(def_id) = self.declaration_def_id(param.name.span)
            {
                type_info_scope.insert(def_id, kind_tag.clone());
            }
        }

        let mut type_info_primitive_scope = HashMap::new();
        for (index, primitive_tag) in &facts.type_info_primitives {
            if let Some(param) = func.params.get(*index)
                && let Some(def_id) = self.declaration_def_id(param.name.span)
            {
                type_info_primitive_scope.insert(def_id, primitive_tag.clone());
            }
        }

        let mut type_kind_scope = HashMap::new();
        for (index, kind_tag) in &facts.type_kind_values {
            if let Some(param) = func.params.get(*index)
                && let Some(def_id) = self.declaration_def_id(param.name.span)
            {
                type_kind_scope.insert(def_id, kind_tag.clone());
            }
        }

        let mut type_primitive_scope = HashMap::new();
        for (index, primitive_tag) in &facts.type_primitive_values {
            if let Some(param) = func.params.get(*index)
                && let Some(def_id) = self.declaration_def_id(param.name.span)
            {
                type_primitive_scope.insert(def_id, primitive_tag.clone());
            }
        }

        self.reflection_type_info_kind_scopes.push(type_info_scope);
        self.reflection_type_info_primitive_scopes
            .push(type_info_primitive_scope);
        self.reflection_type_kind_value_scopes.push(type_kind_scope);
        self.reflection_type_primitive_value_scopes
            .push(type_primitive_scope);
    }

    fn pop_reflection_local_fact_scope(&mut self) {
        self.reflection_type_primitive_value_scopes.pop();
        self.reflection_type_kind_value_scopes.pop();
        self.reflection_type_info_primitive_scopes.pop();
        self.reflection_type_info_kind_scopes.pop();
    }

    fn clear_reflection_local_fact(&mut self, def_id: DefId) {
        for scope in &mut self.reflection_type_info_kind_scopes {
            scope.remove(&def_id);
        }
        for scope in &mut self.reflection_type_info_primitive_scopes {
            scope.remove(&def_id);
        }
        for scope in &mut self.reflection_type_kind_value_scopes {
            scope.remove(&def_id);
        }
        for scope in &mut self.reflection_type_primitive_value_scopes {
            scope.remove(&def_id);
        }
    }

    fn reflected_field_types_for_owner(&self, owner_ty: TypeId) -> Vec<TypeId> {
        match self.interner.resolve(owner_ty) {
            Type::Struct(sid) => self
                .interner
                .resolve_struct(*sid)
                .fields
                .iter()
                .map(|(_, ty)| *ty)
                .collect(),
            Type::Bitfield(bid) => self
                .interner
                .resolve_bitfield(*bid)
                .fields
                .iter()
                .map(|field| field.ty)
                .collect(),
            _ => Vec::new(),
        }
    }

    fn reflected_variant_field_types_for_owner(&self, owner_ty: TypeId) -> Vec<TypeId> {
        match self.interner.resolve(owner_ty) {
            Type::Enum(eid) => self
                .interner
                .resolve_enum(*eid)
                .variants
                .iter()
                .flat_map(|variant| variant.fields.iter().map(|(_, ty)| *ty))
                .collect(),
            _ => Vec::new(),
        }
    }

    fn reflected_machine_field_types_for_owner(&self, owner_ty: TypeId) -> Vec<TypeId> {
        match self.interner.resolve(owner_ty) {
            Type::Machine(mid) => self
                .interner
                .resolve_machine(*mid)
                .states
                .iter()
                .flat_map(|state| state.fields.iter().map(|(_, ty)| *ty))
                .collect(),
            Type::MachineState { machine, state } => self
                .interner
                .resolve_machine(*machine)
                .state(*state)
                .map(|state_def| state_def.fields.iter().map(|(_, ty)| *ty).collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn reflected_type_info_arg_types_for_iterable(
        &mut self,
        iterable: &Expr,
    ) -> Option<Vec<TypeId>> {
        let source = reflected_type_info_args_source(iterable)?;
        let source_types = match source {
            ReflectedTypeInfoSource::Direct(ty) => vec![self.resolve_type_expr(ty)],
            ReflectedTypeInfoSource::Field(field_name) => {
                self.reflected_field_types_for_name(field_name)?
            }
            ReflectedTypeInfoSource::TypeInfo(info_name) => {
                self.reflected_type_info_types_for_name(info_name)?
            }
        };

        Some(
            source_types
                .into_iter()
                .flat_map(|ty| self.type_info_arg_types_for_type(ty))
                .collect(),
        )
    }

    fn type_info_arg_types_for_type(&self, ty: TypeId) -> Vec<TypeId> {
        for ((_name, type_args), type_id) in &self.monomorphized_structs {
            if *type_id == ty {
                return type_args.clone();
            }
        }

        match self.interner.resolve(ty) {
            Type::List(inner) | Type::Set(inner) | Type::Optional(inner) | Type::Secret(inner) => {
                vec![*inner]
            }
            Type::Map(key, value) | Type::Result(key, value) => vec![*key, *value],
            Type::Refinement { base, .. } => vec![*base],
            Type::Function {
                params,
                return_type,
            } => params
                .iter()
                .copied()
                .chain(std::iter::once(*return_type))
                .collect(),
            _ => Vec::new(),
        }
    }

    fn type_info_arg_types_for_type_expr(&mut self, ty: &TypeExpr) -> Vec<TypeId> {
        match ty {
            TypeExpr::View(inner, _) => self.type_info_arg_types_for_type_expr(inner),
            TypeExpr::StateQualified(_, _, _) => Vec::new(),
            TypeExpr::Named(ident) if self.type_aliases.contains_key(&ident.name) => {
                let alias = self
                    .type_aliases
                    .get(&ident.name)
                    .cloned()
                    .expect("type alias existence checked before lookup");
                vec![self.resolve_type_expr(&alias.base_type)]
            }
            TypeExpr::Generic(_, args, _) => {
                args.iter().map(|arg| self.resolve_type_expr(arg)).collect()
            }
            TypeExpr::Function(params, return_type, _) => params
                .iter()
                .chain(std::iter::once(return_type.as_ref()))
                .map(|arg| self.resolve_type_expr(arg))
                .collect(),
            _ => {
                let resolved = self.resolve_type_expr(ty);
                self.type_info_arg_types_for_type(resolved)
            }
        }
    }

    fn push_reflected_field_type_scope(&mut self, field_name: &str, field_types: Vec<TypeId>) {
        let mut scope = HashMap::new();
        scope.insert(field_name.to_string(), field_types);
        self.reflected_field_type_scopes.push(scope);
    }

    fn pop_reflected_field_type_scope(&mut self) {
        self.reflected_field_type_scopes.pop();
    }

    fn push_reflected_type_info_scope(&mut self, info_name: &str, info_types: Vec<TypeId>) {
        let mut scope = HashMap::new();
        scope.insert(info_name.to_string(), info_types);
        self.reflected_type_info_scopes.push(scope);
    }

    fn pop_reflected_type_info_scope(&mut self) {
        self.reflected_type_info_scopes.pop();
    }

    fn push_reflected_variant_type_scope(&mut self, variant_name: &str, owner_ty: TypeId) {
        let mut scope = HashMap::new();
        scope.insert(variant_name.to_string(), owner_ty);
        self.reflected_variant_type_scopes.push(scope);
    }

    fn pop_reflected_variant_type_scope(&mut self) {
        self.reflected_variant_type_scopes.pop();
    }

    fn push_reflected_machine_state_type_scope(&mut self, state_name: &str, owner_ty: TypeId) {
        let mut scope = HashMap::new();
        scope.insert(state_name.to_string(), owner_ty);
        self.reflected_machine_state_type_scopes.push(scope);
    }

    fn pop_reflected_machine_state_type_scope(&mut self) {
        self.reflected_machine_state_type_scopes.pop();
    }

    fn check_trace(&mut self, trace_stmt: &ast::TraceStmt) {
        self.check_ident(&trace_stmt.name);
    }

    fn check_breakpoint(&mut self, breakpoint_stmt: &ast::BreakpointStmt) {
        if let Some(condition) = &breakpoint_stmt.condition {
            let cond_type = self.check_expr(condition);
            if cond_type != TypeInterner::ERROR && cond_type != TypeInterner::BOOL {
                self.sink.emit(errors::condition_not_bool(
                    &self.type_name(cond_type),
                    condition.span(),
                ));
            }
        }
    }

    fn check_respond(&mut self, resp: &ast::RespondStmt) {
        match self.current_respond_type {
            None => {
                self.check_expr(&resp.value);
                self.sink.emit(errors::respond_outside_handler(resp.span));
            }
            Some(expected) => {
                let val_ty = self.check_expr_for_expected(&resp.value, expected, false);
                if val_ty != TypeInterner::ERROR
                    && expected != TypeInterner::ERROR
                    && !self.types_compatible(expected, val_ty)
                {
                    self.sink.emit(errors::type_mismatch(
                        &self.type_name(expected),
                        &self.type_name(val_ty),
                        resp.value.span(),
                    ));
                }
            }
        }
    }

    fn check_expr_for_expected(
        &mut self,
        expr: &Expr,
        expected_ty: TypeId,
        allow_refinement_handle: bool,
    ) -> TypeId {
        let ty = match expr {
            Expr::IntLiteral(value, _)
                if self.int_literal_matches_expected_type(*value, expected_ty) =>
            {
                expected_ty
            }
            Expr::FloatLiteral(_, _) if self.float_literal_matches_expected_type(expected_ty) => {
                expected_ty
            }
            Expr::Unary(UnaryOp::Neg, inner, _)
                if self.negated_literal_matches_expected_type(inner, expected_ty) =>
            {
                expected_ty
            }
            Expr::Binary(lhs, op, rhs, span)
                if matches!(
                    op,
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Modulo
                ) && self.expected_numeric_type(expected_ty).is_some() =>
            {
                let operand_ty = self
                    .expected_numeric_type(expected_ty)
                    .expect("guard checked expected numeric type");
                self.check_binary_for_expected_numeric(lhs, *op, rhs, *span, operand_ty)
            }
            Expr::ListConstruct(elems, _span) => {
                let expected_inner = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
                match self.interner.resolve(expected_inner).clone() {
                    Type::List(expected_element) => self.check_list_construct_for_expected(
                        elems,
                        expected_ty,
                        expected_element,
                        allow_refinement_handle,
                    ),
                    _ => self.check_expr(expr),
                }
            }
            Expr::MapConstruct(entries, _span) => {
                let expected_inner = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
                match self.interner.resolve(expected_inner).clone() {
                    Type::Map(expected_key, expected_value) => self
                        .check_map_construct_for_expected(
                            entries,
                            expected_ty,
                            expected_key,
                            expected_value,
                            allow_refinement_handle,
                        ),
                    _ => self.check_expr(expr),
                }
            }
            Expr::Some(inner, _span) => {
                let expected_inner = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
                match self.interner.resolve(expected_inner).clone() {
                    Type::Optional(expected_payload) => self.check_wrapper_payload_for_expected(
                        inner,
                        expected_ty,
                        expected_payload,
                        allow_refinement_handle,
                    ),
                    _ => self.check_expr(expr),
                }
            }
            Expr::Ok(inner, _span) => {
                let expected_inner = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
                match self.interner.resolve(expected_inner).clone() {
                    Type::Result(expected_payload, _) => self.check_wrapper_payload_for_expected(
                        inner,
                        expected_ty,
                        expected_payload,
                        allow_refinement_handle,
                    ),
                    _ => self.check_expr(expr),
                }
            }
            Expr::Fail(inner, _span) => {
                let expected_inner = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
                match self.interner.resolve(expected_inner).clone() {
                    Type::Result(_, expected_payload) => self.check_wrapper_payload_for_expected(
                        inner,
                        expected_ty,
                        expected_payload,
                        allow_refinement_handle,
                    ),
                    _ => self.check_expr(expr),
                }
            }
            Expr::None(_) => {
                let expected_inner = self.secret_inner_type(expected_ty).unwrap_or(expected_ty);
                match self.interner.resolve(expected_inner) {
                    Type::Optional(_) => expected_ty,
                    _ => self.check_expr(expr),
                }
            }
            Expr::Coarsen(inner, span) => {
                let inner_ty = self.check_expr(inner);
                if inner_ty == TypeInterner::ERROR {
                    TypeInterner::ERROR
                } else if !self.is_refinement_type(inner_ty) {
                    self.sink.emit(errors::coarsen_requires_refinement(
                        &self.type_name(inner_ty),
                        *span,
                    ));
                    inner_ty
                } else if self.can_coarsen_to(inner_ty, expected_ty)
                    || self.fully_coarsened_type(inner_ty) == expected_ty
                {
                    expected_ty
                } else {
                    self.fully_coarsened_type(inner_ty)
                }
            }
            Expr::Handle(target, bind_name, body, span)
                if allow_refinement_handle && self.is_refinement_type(expected_ty) =>
            {
                let target_ty = self.check_expr(target);
                if matches!(
                    self.interner.resolve(target_ty),
                    Type::Result(_, _) | Type::Optional(_)
                ) {
                    self.check_handle_with_target_type(target_ty, bind_name.as_ref(), body, *span)
                } else {
                    self.check_refinement_handle_with_input_type(
                        expected_ty,
                        target_ty,
                        target.span(),
                        bind_name.as_ref(),
                        body,
                        *span,
                    )
                }
            }
            _ => {
                let actual_ty = self.check_expr(expr);
                if actual_ty != TypeInterner::ERROR
                    && self.type_requires_handle_error(expected_ty, actual_ty)
                {
                    self.sink
                        .emit(errors::result_requires_handle_error(expr.span()));
                    expected_ty
                } else if actual_ty != TypeInterner::ERROR
                    && self.type_requires_bare_handle(expected_ty, actual_ty)
                {
                    self.sink
                        .emit(errors::optional_requires_bare_handle(expr.span()));
                    expected_ty
                } else if allow_refinement_handle
                    && self.is_refinement_type(expected_ty)
                    && actual_ty != TypeInterner::ERROR
                    && actual_ty != expected_ty
                    && self.can_refine_from(actual_ty, expected_ty)
                {
                    self.sink.emit(errors::refinement_requires_handle_error(
                        &self.type_name(expected_ty),
                        &self.type_name(actual_ty),
                        expr.span(),
                    ));
                    expected_ty
                } else {
                    actual_ty
                }
            }
        };

        self.type_map.insert(expr.span(), ty);
        ty
    }

    fn check_refinement_handle_with_input_type(
        &mut self,
        refinement_ty: TypeId,
        input_ty: TypeId,
        target_span: Span,
        bind_name: Option<&ast::Ident>,
        body: &Block,
        span: Span,
    ) -> TypeId {
        let expected_input_ty = self.refinement_boundary_input_type(refinement_ty);
        let Some(_base_ty) = self.refinement_base_type(refinement_ty) else {
            return TypeInterner::ERROR;
        };

        if input_ty != TypeInterner::ERROR && !self.can_refine_from(input_ty, refinement_ty) {
            self.sink.emit(errors::type_mismatch(
                &self.type_name(expected_input_ty),
                &self.type_name(input_ty),
                target_span,
            ));
        }

        if bind_name.is_none() {
            self.sink.emit(errors::refinement_requires_handle_error(
                &self.type_name(refinement_ty),
                &self.type_name(expected_input_ty),
                span,
            ));
        }

        if let Some(name) = bind_name
            && let Some(def_id) = self.declaration_def_id(name.span)
        {
            self.type_env.insert(def_id, TypeInterner::STRING);
        }

        self.check_handle_body(body);
        self.validate_handle_terminator(body, refinement_ty);
        refinement_ty
    }

    fn record_reflection_local_fact(
        &mut self,
        decl: &ast::VarDecl,
        declared_type: TypeId,
        init_type: TypeId,
    ) {
        if !self.specialize_reflection_branches || decl.mutable {
            return;
        }

        let Some(def_id) = self.declaration_def_id(decl.name.span) else {
            return;
        };

        if !self.types_compatible(declared_type, init_type) {
            self.clear_reflection_local_fact(def_id);
            return;
        }

        if self.type_id_is_named(declared_type, "TypeInfo") {
            if let Some(facts) = self.eval_type_info_facts(&decl.value) {
                if let Some(scope) = self.current_reflection_type_info_kind_scope_mut() {
                    scope.insert(def_id, facts.kind_tag);
                }
                if let Some(scope) = self.current_reflection_type_info_primitive_scope_mut() {
                    scope.insert(def_id, facts.primitive_tag);
                }
            }
            return;
        }

        if self.type_id_is_named(declared_type, "TypeKind")
            && let Some(kind_tag) = self.eval_type_kind_value(&decl.value)
            && let Some(scope) = self.current_reflection_type_kind_value_scope_mut()
        {
            scope.insert(def_id, kind_tag);
        }

        if self.type_id_is_named(declared_type, "TypePrimitive")
            && let Some(primitive_tag) = self.eval_type_primitive_value(&decl.value)
            && let Some(scope) = self.current_reflection_type_primitive_value_scope_mut()
        {
            scope.insert(def_id, primitive_tag);
        }
    }

    fn check_var_decl(&mut self, decl: &ast::VarDecl) {
        let declared_type = self.resolve_type_expr(&decl.ty);
        let init_type = self.check_expr_for_expected(&decl.value, declared_type, true);

        // Bind the variable's DefId to its declared type.
        if let Some(def_id) = self.declaration_def_id(decl.name.span) {
            self.type_env.insert(def_id, declared_type);
        }

        // Check that the initializer type matches the declared type (skip if Error).
        if !self.types_compatible(declared_type, init_type) {
            self.sink.emit(errors::var_decl_type_mismatch(
                &decl.name.name,
                &self.type_name(declared_type),
                &self.type_name(init_type),
                decl.span,
            ));
        }

        self.record_reflection_local_fact(decl, declared_type, init_type);
    }

    fn check_assign(&mut self, assign: &ast::AssignStmt) {
        if let Expr::Ident(ident) = &assign.target
            && let Some(def_id) = self.ident_def_id(ident)
        {
            self.clear_reflection_local_fact(def_id);
        }

        let target_type = self.check_expr(&assign.target);
        let value_type = self.check_expr_for_expected(&assign.value, target_type, false);

        if !self.types_compatible(target_type, value_type) {
            self.sink.emit(errors::assign_type_mismatch(
                &self.type_name(target_type),
                &self.type_name(value_type),
                assign.span,
            ));
        }
    }

    fn check_return(&mut self, ret: &ast::ReturnStmt) {
        let ret_type = match &ret.value {
            Some(expr) => {
                if let Some(expected) = self.current_return_type {
                    self.check_expr_for_expected(expr, expected, false)
                } else {
                    self.check_expr(expr)
                }
            }
            None => TypeInterner::NOTHING,
        };

        if let Some(expected) = self.current_return_type
            && !self.satisfies_expected_type(expected, ret_type)
        {
            self.sink.emit(errors::return_type_mismatch(
                &self.type_name(expected),
                &self.type_name(ret_type),
                ret.span,
            ));
        }
    }

    fn check_condition_expr(&mut self, condition: &Expr) {
        let cond_type = self.check_expr(condition);
        if cond_type != TypeInterner::ERROR && cond_type != TypeInterner::BOOL {
            self.sink.emit(errors::condition_not_bool(
                &self.type_name(cond_type),
                condition.span(),
            ));
        }
    }

    fn check_if(&mut self, if_stmt: &ast::IfStmt) {
        if self.specialize_reflection_branches {
            if let Some(selected_branch) = self.static_reflection_if_branch(if_stmt) {
                self.check_condition_expr(&if_stmt.condition);
                for (else_if_cond, _) in &if_stmt.else_ifs {
                    self.check_condition_expr(else_if_cond);
                }

                if selected_branch == 0 {
                    let narrowing = self.machine_state_narrowing_for_branch(
                        Some(&if_stmt.condition),
                        None::<&Expr>,
                    );
                    self.check_block_with_type_override(&if_stmt.then_block, narrowing);
                    return;
                }

                for (index, (else_if_cond, else_if_block)) in if_stmt.else_ifs.iter().enumerate() {
                    if selected_branch == index + 1 {
                        let prior_conditions = std::iter::once(&if_stmt.condition).chain(
                            if_stmt
                                .else_ifs
                                .iter()
                                .take(index)
                                .map(|(condition, _)| condition),
                        );
                        let narrowing = self.machine_state_narrowing_for_branch(
                            Some(else_if_cond),
                            prior_conditions,
                        );
                        self.check_block_with_type_override(else_if_block, narrowing);
                        return;
                    }
                }

                if let Some(else_block) = &if_stmt.else_block {
                    let prior_conditions = std::iter::once(&if_stmt.condition)
                        .chain(if_stmt.else_ifs.iter().map(|(condition, _)| condition));
                    let narrowing = self.machine_state_narrowing_for_branch(None, prior_conditions);
                    self.check_block_with_type_override(else_block, narrowing);
                }
                return;
            }

            let active_type_params = self.type_var_subst.keys().cloned().collect::<HashSet<_>>();
            if let Some(span) =
                self.if_branch_type_param_reflection_span(if_stmt, &active_type_params)
            {
                self.check_condition_expr(&if_stmt.condition);
                for (else_if_cond, _) in &if_stmt.else_ifs {
                    self.check_condition_expr(else_if_cond);
                }
                self.sink.emit(errors::invalid_comptime_type_binding(span));
                return;
            }
        }

        self.check_condition_expr(&if_stmt.condition);
        let narrowing =
            self.machine_state_narrowing_for_branch(Some(&if_stmt.condition), None::<&Expr>);
        self.check_block_with_type_override(&if_stmt.then_block, narrowing);

        let mut prior_conditions = vec![&if_stmt.condition];
        for (else_if_cond, else_if_block) in &if_stmt.else_ifs {
            self.check_condition_expr(else_if_cond);
            let narrowing = self.machine_state_narrowing_for_branch(
                Some(else_if_cond),
                prior_conditions.iter().copied(),
            );
            self.check_block_with_type_override(else_if_block, narrowing);
            prior_conditions.push(else_if_cond);
        }

        if let Some(else_block) = &if_stmt.else_block {
            let narrowing = self.machine_state_narrowing_for_branch(None, prior_conditions);
            self.check_block_with_type_override(else_block, narrowing);
        }
    }

    fn check_block_with_type_override(
        &mut self,
        block: &Block,
        override_ty: Option<(DefId, TypeId)>,
    ) {
        let Some((def_id, narrowed_ty)) = override_ty else {
            self.check_block(block);
            return;
        };

        let previous = self.type_env.insert(def_id, narrowed_ty);
        self.check_block(block);
        if let Some(previous) = previous {
            self.type_env.insert(def_id, previous);
        } else {
            self.type_env.remove(&def_id);
        }
    }

    fn machine_state_narrowing_for_branch<'b>(
        &mut self,
        true_condition: Option<&'b Expr>,
        false_conditions: impl IntoIterator<Item = &'b Expr>,
    ) -> Option<(DefId, TypeId)> {
        let mut facts = Vec::new();
        let mut target_def = None;

        if let Some(condition) = true_condition
            && let Some((ident, state, truth)) = Self::machine_state_fact(condition, true)
        {
            let def_id = self.ident_def_id(ident)?;
            target_def = Some(def_id);
            facts.push((def_id, state, truth));
        }

        for condition in false_conditions {
            let Some((ident, state, truth)) = Self::machine_state_fact(condition, false) else {
                continue;
            };
            facts.push((self.ident_def_id(ident)?, state, truth));
        }

        let target_def = if let Some(target_def) = target_def {
            target_def
        } else {
            let mut candidate = None;
            for (def_id, _, _) in &facts {
                if candidate.is_some_and(|existing| existing != *def_id) {
                    return None;
                }
                candidate = Some(*def_id);
            }
            candidate?
        };

        let current_ty = *self.type_env.get(&target_def)?;
        let (machine, can_narrow_by_complement) = match self.interner.resolve(current_ty).clone() {
            Type::Machine(machine) => (machine, true),
            Type::MachineState { machine, .. } => (machine, false),
            _ => return None,
        };

        let mut required_state = None;
        let mut excluded_states = HashSet::new();
        let state_count = {
            let machine_def = self.interner.resolve_machine(machine);
            for (def_id, state, truth) in facts {
                if def_id != target_def {
                    continue;
                }
                let state_id = machine_def.state_id(&state.name)?;
                match truth {
                    MachineStateTruth::Is => {
                        if required_state.is_some_and(|existing| existing != state_id) {
                            return None;
                        }
                        required_state = Some(state_id);
                    }
                    MachineStateTruth::IsNot => {
                        excluded_states.insert(state_id);
                    }
                }
            }
            machine_def.states.len()
        };

        let state = if let Some(state) = required_state {
            if excluded_states.contains(&state) {
                return None;
            }
            state
        } else {
            if !can_narrow_by_complement || excluded_states.len() + 1 != state_count {
                return None;
            }
            (0..state_count).find_map(|index| {
                let state_id = MachineStateId::new(index as u32);
                (!excluded_states.contains(&state_id)).then_some(state_id)
            })?
        };

        let narrowed_ty = self.interner.intern(Type::MachineState { machine, state });
        Some((target_def, narrowed_ty))
    }

    fn machine_state_fact(
        condition: &Expr,
        condition_is_true: bool,
    ) -> Option<(&ast::Ident, &ast::Ident, MachineStateTruth)> {
        if let Some((ident, state)) = Self::positive_machine_state_check(condition) {
            return Some((
                ident,
                state,
                if condition_is_true {
                    MachineStateTruth::Is
                } else {
                    MachineStateTruth::IsNot
                },
            ));
        }
        let (ident, state) = Self::negative_machine_state_check(condition)?;
        Some((
            ident,
            state,
            if condition_is_true {
                MachineStateTruth::IsNot
            } else {
                MachineStateTruth::Is
            },
        ))
    }

    fn positive_machine_state_check(condition: &Expr) -> Option<(&ast::Ident, &ast::Ident)> {
        match condition {
            Expr::Paren(inner, _) => Self::positive_machine_state_check(inner),
            Expr::At(inner, state, _) => {
                let ident = Self::narrowable_ident_expr(inner)?;
                Some((ident, state))
            }
            _ => None,
        }
    }

    fn negative_machine_state_check(condition: &Expr) -> Option<(&ast::Ident, &ast::Ident)> {
        match condition {
            Expr::Paren(inner, _) => Self::negative_machine_state_check(inner),
            Expr::Unary(UnaryOp::Not, inner, _) => Self::positive_machine_state_check(inner),
            _ => None,
        }
    }

    fn narrowable_ident_expr(expr: &Expr) -> Option<&ast::Ident> {
        match expr {
            Expr::Ident(ident) => Some(ident),
            Expr::Paren(inner, _) => Self::narrowable_ident_expr(inner),
            _ => None,
        }
    }

    fn static_reflection_if_branch(&mut self, if_stmt: &ast::IfStmt) -> Option<usize> {
        if self.eval_static_type_condition(&if_stmt.condition)? {
            return Some(0);
        }

        for (index, (else_if_cond, _)) in if_stmt.else_ifs.iter().enumerate() {
            if self.eval_static_type_condition(else_if_cond)? {
                return Some(index + 1);
            }
        }

        Some(if_stmt.else_ifs.len() + 1)
    }

    fn static_reflection_match_arm(&mut self, match_stmt: &ast::MatchStmt) -> Option<usize> {
        let selected_value = self.eval_static_reflection_enum_value(&match_stmt.expr)?;

        for (index, arm) in match_stmt.arms.iter().enumerate() {
            match &arm.pattern {
                ast::Pattern::Ident(name) | ast::Pattern::Variant(name, _) => {
                    if selected_value.variant_name() == name.name {
                        return Some(index);
                    }
                }
                ast::Pattern::Other(_) => return Some(index),
            }
        }

        None
    }

    fn match_arm_type_param_reflection_span(
        &self,
        match_stmt: &ast::MatchStmt,
        type_params: &HashSet<String>,
    ) -> Option<Span> {
        match_stmt
            .arms
            .iter()
            .find_map(|arm| self.block_type_param_reflection_span(&arm.body, type_params))
    }

    fn check_for(&mut self, for_stmt: &ast::ForStmt) {
        let iterable_type = self.check_expr(&for_stmt.iterable);

        // The iterable must be list[T], string, or map[K,V].
        let resolved = self.interner.resolve(iterable_type);
        let elem_type = if iterable_type == TypeInterner::ERROR {
            TypeInterner::ERROR
        } else if let Type::List(inner) = resolved {
            *inner
        } else if iterable_type == TypeInterner::STRING {
            TypeInterner::STRING
        } else if let Type::Map(key_ty, _) = resolved {
            // Map iteration: first variable gets key type.
            *key_ty
        } else if let Type::Set(inner) = resolved {
            *inner
        } else {
            self.sink.emit(errors::not_iterable(
                &self.type_name(iterable_type),
                for_stmt.iterable.span(),
            ));
            TypeInterner::ERROR
        };

        // Bind the loop variable (key for maps, element for lists/strings).
        if let Some(def_id) = self.declaration_def_id(for_stmt.variable.span) {
            self.type_env.insert(def_id, elem_type);
        }

        // Bind the optional value variable (only for map iteration).
        if let Some(ref val_var) = for_stmt.value_variable {
            let val_type = if let Type::Map(_, val_ty) = self.interner.resolve(iterable_type) {
                *val_ty
            } else {
                TypeInterner::ERROR
            };
            if let Some(def_id) = self.declaration_def_id(val_var.span) {
                self.type_env.insert(def_id, val_type);
            }
        }

        let pushed_variant_scope =
            if let Some(owner_ty_expr) = comptime_type_variants_binding(&for_stmt.iterable) {
                let owner_ty = self.resolve_type_expr(owner_ty_expr);
                self.push_reflected_variant_type_scope(&for_stmt.variable.name, owner_ty);
                true
            } else {
                false
            };

        let pushed_machine_state_scope =
            if let Some(owner_ty_expr) = comptime_type_machine_states_binding(&for_stmt.iterable) {
                let owner_ty = self.resolve_type_expr(owner_ty_expr);
                self.push_reflected_machine_state_type_scope(&for_stmt.variable.name, owner_ty);
                true
            } else {
                false
            };

        let pushed_field_scope = if let Some(owner_ty_expr) =
            comptime_type_fields_binding(&for_stmt.iterable)
        {
            let owner_ty = self.resolve_type_expr(owner_ty_expr);
            let field_types = if owner_ty == TypeInterner::ERROR {
                Vec::new()
            } else {
                self.reflected_field_types_for_owner(owner_ty)
            };
            self.push_reflected_field_type_scope(&for_stmt.variable.name, field_types);
            true
        } else if let Some(owner_ty_expr) = comptime_type_variant_fields_binding(&for_stmt.iterable)
        {
            let owner_ty = self.resolve_type_expr(owner_ty_expr);
            let field_types = if owner_ty == TypeInterner::ERROR {
                Vec::new()
            } else {
                self.reflected_variant_field_types_for_owner(owner_ty)
            };
            self.push_reflected_field_type_scope(&for_stmt.variable.name, field_types);
            true
        } else if let Some(owner_ty_expr) = comptime_type_machine_fields_binding(&for_stmt.iterable)
        {
            let owner_ty = self.resolve_type_expr(owner_ty_expr);
            let field_types = if owner_ty == TypeInterner::ERROR {
                Vec::new()
            } else {
                self.reflected_machine_field_types_for_owner(owner_ty)
            };
            self.push_reflected_field_type_scope(&for_stmt.variable.name, field_types);
            true
        } else if let Some(fields_owner_name) =
            reflected_machine_state_fields_binding(&for_stmt.iterable)
                .or_else(|| reflected_variant_fields_binding(&for_stmt.iterable))
        {
            if let Some(owner_ty) = self.reflected_machine_state_owner_for_name(fields_owner_name) {
                let field_types = self.reflected_machine_field_types_for_owner(owner_ty);
                self.push_reflected_field_type_scope(&for_stmt.variable.name, field_types);
                true
            } else if let Some(owner_ty) = self.reflected_variant_owner_for_name(fields_owner_name)
            {
                let field_types = self.reflected_variant_field_types_for_owner(owner_ty);
                self.push_reflected_field_type_scope(&for_stmt.variable.name, field_types);
                true
            } else {
                false
            }
        } else {
            false
        };

        let pushed_type_info_scope = if let Some(info_types) =
            self.reflected_type_info_arg_types_for_iterable(&for_stmt.iterable)
        {
            self.push_reflected_type_info_scope(&for_stmt.variable.name, info_types);
            true
        } else {
            false
        };

        self.check_block(&for_stmt.body);

        if pushed_type_info_scope {
            self.pop_reflected_type_info_scope();
        }
        if pushed_field_scope {
            self.pop_reflected_field_type_scope();
        }
        if pushed_variant_scope {
            self.pop_reflected_variant_type_scope();
        }
        if pushed_machine_state_scope {
            self.pop_reflected_machine_state_type_scope();
        }
    }

    fn check_while(&mut self, while_stmt: &ast::WhileStmt) {
        let cond_type = self.check_expr(&while_stmt.condition);
        if cond_type != TypeInterner::ERROR && cond_type != TypeInterner::BOOL {
            self.sink.emit(errors::condition_not_bool(
                &self.type_name(cond_type),
                while_stmt.condition.span(),
            ));
        }
        self.check_block(&while_stmt.body);
    }

    fn check_assert(&mut self, assert_stmt: &ast::AssertStmt) {
        if !self.in_verify_block && !self.in_property_block {
            self.sink
                .emit(errors::assert_outside_test_block(assert_stmt.span));
        }
        let cond_type = self.check_expr(&assert_stmt.condition);
        if cond_type != TypeInterner::ERROR && cond_type != TypeInterner::BOOL {
            self.sink.emit(errors::assert_condition_not_bool(
                &self.type_name(cond_type),
                assert_stmt.condition.span(),
            ));
        }
        if let Some(msg) = &assert_stmt.message {
            self.check_expr(msg);
        }
    }

    fn check_match(&mut self, match_stmt: &ast::MatchStmt) {
        let expr_ty = self.check_expr(&match_stmt.expr);
        if expr_ty == TypeInterner::ERROR {
            for arm in &match_stmt.arms {
                self.check_block(&arm.body);
            }
            return;
        }

        let Type::Enum(enum_id) = *self.interner.resolve(expr_ty) else {
            self.sink.emit(errors::match_requires_enum(
                &self.type_name(expr_ty),
                match_stmt.expr.span(),
            ));
            for arm in &match_stmt.arms {
                self.check_block(&arm.body);
            }
            return;
        };

        let enum_def = self.interner.resolve_enum(enum_id).clone();
        let selected_static_arm = if self.specialize_reflection_branches {
            self.static_reflection_match_arm(match_stmt)
        } else {
            None
        };
        let unknown_reflection_span = if self.specialize_reflection_branches
            && selected_static_arm.is_none()
        {
            let active_type_params = self.type_var_subst.keys().cloned().collect::<HashSet<_>>();
            if self.expr_uses_type_param_reflection(&match_stmt.expr, &active_type_params) {
                Some(match_stmt.expr.span())
            } else {
                self.match_arm_type_param_reflection_span(match_stmt, &active_type_params)
            }
        } else {
            None
        };
        let mut covered = HashSet::new();
        let mut has_other = false;

        for (arm_index, arm) in match_stmt.arms.iter().enumerate() {
            let check_body = unknown_reflection_span.is_none()
                && selected_static_arm.is_none_or(|selected| selected == arm_index);

            match &arm.pattern {
                ast::Pattern::Ident(name) => {
                    if enum_def
                        .variants
                        .iter()
                        .any(|variant| variant.name == name.name)
                    {
                        covered.insert(name.name.clone());
                    } else {
                        self.sink.emit(errors::type_has_no_member(
                            &enum_def.name,
                            &name.name,
                            name.span,
                        ));
                    }
                }
                ast::Pattern::Variant(name, bindings) => {
                    if let Some(variant) = enum_def
                        .variants
                        .iter()
                        .find(|variant| variant.name == name.name)
                    {
                        covered.insert(name.name.clone());
                        if bindings.len() != variant.fields.len() {
                            self.sink.emit(errors::variant_binding_count_mismatch(
                                &name.name,
                                variant.fields.len(),
                                bindings.len(),
                                name.span,
                            ));
                        }

                        if check_body {
                            for (binding, (_, field_ty)) in
                                bindings.iter().zip(variant.fields.iter())
                            {
                                if let Some(def_id) = self.declaration_def_id(binding.span) {
                                    self.type_env.insert(def_id, *field_ty);
                                }
                            }
                        }
                    } else {
                        self.sink.emit(errors::type_has_no_member(
                            &enum_def.name,
                            &name.name,
                            name.span,
                        ));
                    }
                }
                ast::Pattern::Other(_) => {
                    has_other = true;
                }
            }

            if check_body {
                self.check_block(&arm.body);
            }
        }

        if let Some(span) = unknown_reflection_span {
            self.sink.emit(errors::invalid_comptime_type_binding(span));
        }

        if !has_other {
            for variant in &enum_def.variants {
                if !covered.contains(&variant.name) {
                    self.sink.emit(errors::non_exhaustive_match(
                        &enum_def.name,
                        &variant.name,
                        match_stmt.span,
                    ));
                    break;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Expressions
    // ------------------------------------------------------------------

    fn check_expr(&mut self, expr: &Expr) -> TypeId {
        let ty = match expr {
            Expr::IntLiteral(value, span) => {
                if self.int_literal_fits_type(*value, TypeInterner::INT64) {
                    TypeInterner::INT64
                } else if self.int_literal_fits_type(*value, TypeInterner::UINT64) {
                    TypeInterner::UINT64
                } else {
                    self.sink.emit(Diagnostic::error(
                        311,
                        "integer literal must fit in `int64` or `uint64`",
                        *span,
                    ));
                    TypeInterner::ERROR
                }
            }
            Expr::FloatLiteral(_, _) => TypeInterner::FLOAT64,
            Expr::StringLiteral(_, _) => TypeInterner::STRING,
            Expr::BoolLiteral(_, _) => TypeInterner::BOOL,
            Expr::Nothing(_) => TypeInterner::NOTHING,

            Expr::Ident(ident) => self.check_ident(ident),
            Expr::Binary(lhs, op, rhs, span) => self.check_binary(lhs, *op, rhs, *span),
            Expr::Unary(op, operand, span) => self.check_unary(*op, operand, *span),
            Expr::Call(callee, args, span) => self.check_call(callee, &[], args, *span),
            Expr::GenericCall(callee, type_args, args, span) => {
                self.check_call(callee, type_args, args, *span)
            }
            Expr::Paren(inner, _) => self.check_expr(inner),
            Expr::FieldAccess(base, field, span) => self.check_field_access(base, field, *span),
            Expr::View(inner, _) => self.check_expr(inner),

            Expr::ListConstruct(elems, _span) => self.check_list_construct(elems),
            Expr::MapConstruct(entries, _span) => self.check_map_construct(entries),

            Expr::Handle(target, bind_name, body, span) => {
                self.check_handle(target, bind_name.as_ref(), body, *span)
            }

            Expr::Ok(inner, _span) => {
                let inner_ty = self.check_expr(inner);
                // ok(T) → result[T, <error>] — the error type is unknown without context.
                // For now, produce result[T, nothing].
                self.interner
                    .intern(Type::Result(inner_ty, TypeInterner::ERROR))
            }
            Expr::Fail(inner, _span) => {
                let inner_ty = self.check_expr(inner);
                // fail(E) → result[<error>, E]
                self.interner
                    .intern(Type::Result(TypeInterner::ERROR, inner_ty))
            }
            Expr::Some(inner, _span) => {
                let inner_ty = self.check_expr(inner);
                self.interner.intern(Type::Optional(inner_ty))
            }
            Expr::None(_) => {
                // none → optional[<error>] (unknown inner type without context)
                self.interner.intern(Type::Optional(TypeInterner::ERROR))
            }
            Expr::Default(inner, span) => {
                if self.handle_body_depth == 0 {
                    self.sink.emit(errors::default_outside_handle(*span));
                }
                self.check_expr(inner)
            }

            Expr::StringInterpolation(parts, _) => {
                // Each interpolated expression must be displayable; the overall result is string.
                for part in parts {
                    if let StringPart::Expr(expr) = part {
                        let expr_ty = self.check_expr(expr);
                        if !self.is_displayable_type(expr_ty) {
                            self.sink.emit(errors::type_does_not_implement_interface(
                                &self.type_name(expr_ty),
                                "Displayable",
                                expr.span(),
                            ));
                        }
                    }
                }
                TypeInterner::STRING
            }
            Expr::Declassify(inner, span) => {
                let inner_ty = self.check_expr(inner);
                if let Some(unwrapped) = self.secret_inner_type(inner_ty) {
                    unwrapped
                } else {
                    self.sink.emit(errors::declassify_requires_secret(
                        &self.type_name(inner_ty),
                        *span,
                    ));
                    TypeInterner::ERROR
                }
            }
            Expr::Coarsen(inner, _) => {
                let inner_ty = self.check_expr(inner);
                if inner_ty != TypeInterner::ERROR && !self.is_refinement_type(inner_ty) {
                    self.sink.emit(errors::coarsen_requires_refinement(
                        &self.type_name(inner_ty),
                        expr.span(),
                    ));
                }
                self.fully_coarsened_type(inner_ty)
            }
            Expr::Pipeline(initial, steps, _) => self.check_pipeline(initial, steps),
            Expr::At(inner, state, span) => self.check_machine_state_check(inner, state, *span),
            Expr::Spawn(inner, _) => self.check_spawn(inner),
            Expr::Send(inner, _) => {
                self.check_send_ask_inner(inner);
                TypeInterner::NOTHING
            }
            Expr::Ask(inner, _) => self.check_send_ask_inner(inner),
            Expr::Clone(inner, _) => {
                // `clone expr` returns the same type as the expression.
                self.check_expr(inner)
            }
            Expr::Run(inner, _) => {
                // `run call` returns the same type as the call (pending tracked internally).
                self.check_expr(inner)
            }
            Expr::Join(inner, _) => {
                // `join task` returns result[T, string] so `handle error:` works.
                // If the task already has a result type, preserve it as-is.
                let inner_ty = self.check_expr(inner);
                match self.interner.resolve(inner_ty).clone() {
                    jett_types::Type::Result(_, _) => inner_ty,
                    _ => self
                        .interner
                        .intern(jett_types::Type::Result(inner_ty, TypeInterner::STRING)),
                }
            }
            Expr::Cancel(inner, _) => {
                // `cancel task` — checks the inner expression and returns nothing.
                self.check_expr(inner);
                TypeInterner::NOTHING
            }
            Expr::Error(_) => TypeInterner::ERROR,
            Expr::EnumVariant(type_name, variant, span) => {
                self.check_enum_variant(type_name, variant, &[], *span)
            }
            Expr::InlineFn(params, return_type, body, _) => {
                // Type-check the inline function body with parameters bound.
                let saved_return_type = self.current_return_type;
                let saved_fn_name = self.current_function_name.take();
                let saved_pure = self.current_function_pure;

                let ret = return_type
                    .as_ref()
                    .map(|t| self.resolve_type_expr(t))
                    .unwrap_or(TypeInterner::NOTHING);
                self.current_return_type = Some(ret);
                self.current_function_pure = false;

                for param in params {
                    let param_type = self.resolve_type_expr(&param.ty);
                    if let Some(def_id) = self.declaration_def_id(param.name.span) {
                        self.type_env.insert(def_id, param_type);
                    }
                }

                let param_types: Vec<_> = params
                    .iter()
                    .map(|p| self.resolve_type_expr(&p.ty))
                    .collect();

                self.check_block(body);

                self.current_return_type = saved_return_type;
                self.current_function_name = saved_fn_name;
                self.current_function_pure = saved_pure;

                self.interner.intern(Type::Function {
                    params: param_types,
                    return_type: ret,
                })
            }
        };

        // Record the type for this expression span.
        self.type_map.insert(expr.span(), ty);
        ty
    }

    fn check_pipeline(&mut self, initial: &Expr, steps: &[ast::PipelineStep]) -> TypeId {
        let mut current_ty = self.check_expr(initial);
        for step in steps {
            current_ty = self.check_pipeline_step(current_ty, step);
        }
        current_ty
    }

    fn pipeline_step_call_parts(
        step: &ast::PipelineStep,
    ) -> (&Expr, &[TypeExpr], &[ast::CallArg], bool) {
        let (function, piped_as_view) = match &step.function {
            Expr::View(inner, _) => (inner.as_ref(), true),
            _ => (&step.function, false),
        };
        match function {
            Expr::GenericCall(callee, type_args, args, _) => {
                (callee, type_args, args, piped_as_view)
            }
            _ => (function, &[], &step.extra_args, piped_as_view),
        }
    }

    fn check_pipeline_step(&mut self, current_ty: TypeId, step: &ast::PipelineStep) -> TypeId {
        let step_ty = self.check_pipeline_step_call(current_ty, step);
        if let Some(handle) = &step.handle {
            return self.check_handle_with_target_type(
                step_ty,
                handle.error_name.as_ref(),
                &handle.body,
                handle.span,
            );
        }
        step_ty
    }

    fn check_pipeline_step_call(&mut self, current_ty: TypeId, step: &ast::PipelineStep) -> TypeId {
        let (function, type_args, extra_args, piped_as_view) = Self::pipeline_step_call_parts(step);
        let callee_name = self.resolved_expr_name(function);
        let callee_is_pure = callee_name
            .as_deref()
            .map(|name| {
                if Self::is_impure_builtin(name) {
                    false
                } else {
                    self.purity_map.get(name).copied().unwrap_or(true)
                }
            })
            .unwrap_or(false);

        if let Some(callee_name) = callee_name.as_deref()
            && !callee_is_pure
        {
            if self.current_function_pure
                && let Some(caller_name) = &self.current_function_name
            {
                self.sink.emit(errors::pure_calls_impure(
                    caller_name,
                    callee_name,
                    step.span,
                ));
            }
            if self.in_verify_block
                && let Some(verify_name) = &self.current_verify_name
            {
                self.sink.emit(errors::verify_calls_impure(
                    verify_name,
                    callee_name,
                    step.span,
                ));
            }
        }

        if let Some(builtin_name) = callee_name.as_deref() {
            match builtin_name {
                "math.abs" => {
                    return self.check_math_abs_pipeline_step(
                        builtin_name,
                        current_ty,
                        type_args,
                        extra_args,
                        step.span,
                    );
                }
                "math.min" | "math.max" => {
                    return self.check_math_min_max_pipeline_step(
                        builtin_name,
                        current_ty,
                        type_args,
                        extra_args,
                        step.span,
                    );
                }
                _ => {}
            }
        }

        let builtin_signature = self.builtin_signature(function, type_args, step.span);
        if builtin_signature.is_none()
            && !type_args.is_empty()
            && let Some(return_type) = self.check_generic_function_pipeline_step(
                callee_name.as_deref(),
                type_args,
                current_ty,
                extra_args,
                step.span,
            )
        {
            return return_type;
        }

        let user_function_signature = if builtin_signature.is_none() {
            callee_name
                .as_deref()
                .and_then(|name| self.function_signatures.get(name).cloned())
        } else {
            None
        };

        let (param_types, return_type) = if let Some(signature) = builtin_signature {
            signature
        } else if let Some(signature) = user_function_signature {
            signature
        } else {
            self.check_expr(function);
            for arg in extra_args {
                self.check_expr(&arg.value);
            }
            self.sink.emit(errors::not_callable(
                callee_name.as_deref().unwrap_or("pipeline step"),
                step.span,
            ));
            return TypeInterner::ERROR;
        };

        let arg_count = extra_args.len() + 1;
        if arg_count != param_types.len() {
            let func_name = callee_name
                .clone()
                .unwrap_or_else(|| "<pipeline step>".to_string());
            self.sink.emit(errors::argument_count_mismatch(
                &func_name,
                param_types.len(),
                arg_count,
                step.span,
            ));
            for arg in extra_args {
                self.check_expr(&arg.value);
            }
            return return_type;
        }

        let mut tainted_return = false;
        let mut checked_arg_types = Vec::with_capacity(arg_count);
        checked_arg_types.push(current_ty);
        tainted_return |= self.check_argument_against_param_type(
            callee_name.as_deref(),
            callee_is_pure,
            "#1",
            param_types[0],
            current_ty,
            step.span,
        );

        for (index, arg) in extra_args.iter().enumerate() {
            let param_ty = param_types[index + 1];
            let arg_ty = self.check_expr_for_expected(&arg.value, param_ty, false);
            checked_arg_types.push(arg_ty);
            tainted_return |= self.check_argument_against_param_type(
                callee_name.as_deref(),
                callee_is_pure,
                &format!("#{}", index + 2),
                param_ty,
                arg_ty,
                arg.value.span(),
            );
        }

        if matches!(callee_name.as_deref(), Some("secret.compare"))
            && checked_arg_types.len() == 2
            && let (Some(lhs_inner), Some(rhs_inner)) = (
                self.secret_inner_type(checked_arg_types[0]),
                self.secret_inner_type(checked_arg_types[1]),
            )
            && (!self.types_compatible(lhs_inner, rhs_inner)
                || !self.types_compatible(rhs_inner, lhs_inner))
        {
            self.sink.emit(errors::argument_type_mismatch(
                "#2",
                &self.type_name(checked_arg_types[0]),
                &self.type_name(checked_arg_types[1]),
                step.span,
            ));
        }

        self.check_json_pipeline_public_call_policy(
            callee_name.as_deref(),
            current_ty,
            piped_as_view,
            step.span,
            return_type,
        );

        if let Some(callee_name) = callee_name.as_deref()
            && tainted_return
            && Self::is_secret_liftable_call(callee_name, callee_is_pure)
        {
            return self.maybe_wrap_secret(return_type, true);
        }

        return_type
    }

    fn check_generic_function_pipeline_step(
        &mut self,
        callee_name: Option<&str>,
        type_args: &[TypeExpr],
        current_ty: TypeId,
        extra_args: &[ast::CallArg],
        span: Span,
    ) -> Option<TypeId> {
        let function_name = callee_name?;
        let Some(template) = self.generic_function_templates.get(function_name).cloned() else {
            if self.function_signatures.contains_key(function_name) {
                self.sink.emit(errors::unknown_type(
                    &format!(
                        "{function_name} (expected 0 type argument(s), got {})",
                        type_args.len()
                    ),
                    span,
                ));
                for arg in extra_args {
                    self.check_expr(&arg.value);
                }
                return Some(TypeInterner::ERROR);
            }
            return None;
        };

        let concrete_args: Vec<TypeId> = type_args
            .iter()
            .map(|arg| self.resolve_type_expr(arg))
            .collect();

        if template.type_params.len() != concrete_args.len() {
            self.sink.emit(errors::unknown_type(
                &format!(
                    "{} (expected {} type argument(s), got {})",
                    function_name,
                    template.type_params.len(),
                    concrete_args.len()
                ),
                span,
            ));
            for arg in extra_args {
                self.check_expr(&arg.value);
            }
            return Some(TypeInterner::ERROR);
        }

        let subst: HashMap<String, TypeId> = template
            .type_params
            .iter()
            .zip(concrete_args.iter())
            .map(|(param, &ty)| (param.name.clone(), ty))
            .collect();
        let kind_subst: HashMap<String, String> = template
            .type_params
            .iter()
            .zip(type_args.iter().zip(concrete_args.iter()))
            .map(|(param, (type_arg, &resolved_ty))| {
                (
                    param.name.clone(),
                    self.reflection_kind_tag_for_type_expr(type_arg, resolved_ty),
                )
            })
            .collect();

        let old_subst = std::mem::replace(&mut self.type_var_subst, subst.clone());
        let param_types: Vec<TypeId> = template
            .params
            .iter()
            .map(|param| self.resolve_type_expr(&param.ty))
            .collect();
        let return_type = template
            .return_type
            .as_ref()
            .map(|ty| self.resolve_type_expr(ty))
            .unwrap_or(TypeInterner::NOTHING);
        self.type_var_subst = old_subst;

        let arg_count = extra_args.len() + 1;
        if arg_count != param_types.len() {
            self.sink.emit(errors::argument_count_mismatch(
                function_name,
                param_types.len(),
                arg_count,
                span,
            ));
            for arg in extra_args {
                self.check_expr(&arg.value);
            }
            return Some(TypeInterner::ERROR);
        }

        let mut arguments_match = true;
        if let Some(&expected) = param_types.first()
            && !self.types_compatible(expected, current_ty)
        {
            arguments_match = false;
            self.sink.emit(errors::type_mismatch(
                &self.type_name(expected),
                &self.type_name(current_ty),
                span,
            ));
        }

        for (arg, &expected) in extra_args.iter().zip(param_types.iter().skip(1)) {
            let got = self.check_expr_for_expected(&arg.value, expected, false);
            if !self.types_compatible(expected, got) {
                arguments_match = false;
                self.sink.emit(errors::type_mismatch(
                    &self.type_name(expected),
                    &self.type_name(got),
                    arg.value.span(),
                ));
            }
        }

        if arguments_match {
            self.check_generic_function_instantiation(
                function_name,
                &template,
                &concrete_args,
                subst,
                kind_subst,
                ReflectionParamFacts::default(),
            );
        }

        Some(return_type)
    }

    fn check_json_pipeline_public_call_policy(
        &mut self,
        callee_name: Option<&str>,
        value_ty: TypeId,
        piped_as_view: bool,
        span: Span,
        return_type: TypeId,
    ) {
        let Some(callee_name) = callee_name else {
            return;
        };

        match callee_name {
            "json.serialize" => {
                if self.type_contains_secret_data(value_ty) {
                    self.sink.emit(errors::type_contains_secret_data(
                        "json.serialize",
                        &self.type_name(value_ty),
                        &self.secret_field_names(value_ty),
                        span,
                    ));
                }
                self.check_json_pipeline_serialize_policy(
                    callee_name,
                    value_ty,
                    piped_as_view,
                    span,
                );
            }
            "json.serialize_public" => {
                if !self.is_secret_type(value_ty)
                    && self.type_contains_secret_data(value_ty)
                    && !self.json_public_projection_allows_secret_data(value_ty)
                {
                    self.sink.emit(errors::type_contains_secret_data(
                        "json.serialize_public",
                        &self.type_name(value_ty),
                        &self.secret_field_names(value_ty),
                        span,
                    ));
                }
                self.check_json_pipeline_serialize_policy(
                    callee_name,
                    value_ty,
                    piped_as_view,
                    span,
                );
            }
            "json.parse" | "json.parse_exact" => {
                if let Type::Result(parsed_ty, _) = self.interner.resolve(return_type) {
                    for key_type in self.json_non_string_map_key_types(*parsed_ty) {
                        self.sink
                            .emit(errors::json_map_key_must_be_string(&key_type, span));
                    }
                    for unsupported_type in self.json_unsupported_parse_types(*parsed_ty) {
                        self.sink.emit(errors::json_unsupported_parse_type(
                            callee_name,
                            &unsupported_type,
                            span,
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    fn check_json_pipeline_serialize_policy(
        &mut self,
        function_name: &str,
        value_ty: TypeId,
        piped_as_view: bool,
        span: Span,
    ) {
        if !piped_as_view && self.json_read_requires_view(value_ty) {
            self.sink.emit(errors::json_serialize_requires_view(
                function_name,
                &self.type_name(value_ty),
                span,
            ));
        }

        for key_type in self.json_non_string_map_key_types(value_ty) {
            self.sink
                .emit(errors::json_map_key_must_be_string(&key_type, span));
        }

        for unsupported_type in self.json_unsupported_serialize_types(value_ty) {
            self.sink.emit(errors::json_unsupported_serialize_type(
                function_name,
                &unsupported_type,
                span,
            ));
        }
    }

    fn check_argument_against_param_type(
        &mut self,
        callee_name: Option<&str>,
        callee_is_pure: bool,
        param_name: &str,
        param_ty: TypeId,
        arg_ty: TypeId,
        span: Span,
    ) -> bool {
        if arg_ty != TypeInterner::ERROR && self.type_requires_handle_error(param_ty, arg_ty) {
            self.sink.emit(errors::result_requires_handle_error(span));
            return false;
        }

        if arg_ty != TypeInterner::ERROR && self.type_requires_bare_handle(param_ty, arg_ty) {
            self.sink.emit(errors::optional_requires_bare_handle(span));
            return false;
        }

        if self.is_refinement_type(param_ty)
            && arg_ty != TypeInterner::ERROR
            && self.can_refine_from(arg_ty, param_ty)
        {
            self.sink.emit(errors::refinement_requires_handle_error(
                &self.type_name(param_ty),
                &self.type_name(arg_ty),
                span,
            ));
            return false;
        }

        if let Some(callee_name) = callee_name {
            if Self::is_secret_output_boundary(callee_name) && self.is_secret_type(arg_ty) {
                self.sink.emit(errors::secret_exposure(
                    callee_name,
                    &self.type_name(arg_ty),
                    span,
                ));
                return false;
            }

            if matches!(callee_name, "secret.redact" | "secret.compare")
                && !self.is_secret_type(arg_ty)
            {
                self.sink.emit(errors::secret_operation_requires_secret(
                    callee_name,
                    &self.type_name(arg_ty),
                    span,
                ));
                return false;
            }
        }

        let (matches, lifted_secret) = self.secret_argument_matches_param(param_ty, arg_ty);
        if matches {
            if lifted_secret {
                let allows_secret_lifting = callee_name
                    .map(|name| Self::is_secret_liftable_call(name, callee_is_pure))
                    .unwrap_or(callee_is_pure);

                if !allows_secret_lifting {
                    self.sink.emit(errors::secret_exposure(
                        callee_name.unwrap_or("<call>"),
                        &self.type_name(arg_ty),
                        span,
                    ));
                    return false;
                }
                return true;
            }
            return false;
        }

        if !self.types_compatible(param_ty, arg_ty) {
            self.sink.emit(errors::argument_type_mismatch(
                param_name,
                &self.type_name(param_ty),
                &self.type_name(arg_ty),
                span,
            ));
        }

        false
    }

    fn check_spawn(&mut self, inner: &Expr) -> TypeId {
        // `spawn ActorType(args)` — the inner expr should be a call to the actor type name.
        // We check the arguments but return the actor type.
        let (callee, args) = match inner {
            Expr::Call(callee, args, _span) => (callee.as_ref(), args.as_slice()),
            _ => {
                self.check_expr(inner);
                return TypeInterner::ERROR;
            }
        };

        let Some(actor_name) = self.expanded_dotted_expr_name(callee) else {
            for arg in args {
                self.check_expr(&arg.value);
            }
            self.check_expr(callee);
            return TypeInterner::ERROR;
        };

        if let Some(&ty) = self.named_types.get(&actor_name)
            && let Type::Actor(aid) = *self.interner.resolve(ty)
        {
            let actor_def = self.interner.resolve_actor(aid).clone();
            self.check_actor_argument_list(
                &actor_def.name,
                &actor_def.capability_params,
                args,
                inner.span(),
            );
            return ty;
        }

        for arg in args {
            self.check_expr(&arg.value);
        }
        self.check_expr(callee);
        TypeInterner::ERROR
    }

    fn check_send_ask_inner(&mut self, inner: &Expr) -> TypeId {
        // inner is `actor_expr.handler_name` or `actor_expr.handler_name(args)`
        // We check the actor expression and any args, and return the responds type.
        let (actor_expr, message_name, message_span, args) = match inner {
            Expr::Call(callee, args, _) => match callee.as_ref() {
                Expr::FieldAccess(base, field, _) => (
                    base.as_ref(),
                    &field.name,
                    field.span,
                    Some(args.as_slice()),
                ),
                _ => {
                    self.check_expr(inner);
                    return TypeInterner::ERROR;
                }
            },
            Expr::FieldAccess(base, field, _) => (base.as_ref(), &field.name, field.span, None),
            _ => {
                self.check_expr(inner);
                return TypeInterner::ERROR;
            }
        };

        let actor_ty = self.check_expr(actor_expr);

        // Look up the handler and return its responds type.
        if let Type::Actor(aid) = *self.interner.resolve(actor_ty) {
            let actor_def = self.interner.resolve_actor(aid).clone();
            if let Some(msg) = actor_def.messages.iter().find(|m| m.name == *message_name) {
                self.check_actor_message_args(&actor_def.name, msg, args, inner.span());
                return msg.responds;
            }
            self.sink.emit(errors::type_has_no_member(
                &actor_def.name,
                message_name,
                message_span,
            ));
        } else {
            if actor_ty != TypeInterner::ERROR {
                self.sink.emit(errors::type_has_no_member(
                    &self.type_name(actor_ty),
                    message_name,
                    message_span,
                ));
            }
            if let Some(arg_list) = args {
                for arg in arg_list {
                    self.check_expr(&arg.value);
                }
            }
        }
        TypeInterner::ERROR
    }

    fn check_actor_message_args(
        &mut self,
        actor_name: &str,
        msg: &ActorMessageDef,
        args: Option<&[ast::CallArg]>,
        span: Span,
    ) {
        self.check_actor_argument_list(
            &format!("{actor_name}.{}", msg.name),
            &msg.params,
            args.unwrap_or(&[]),
            span,
        );
    }

    fn check_actor_argument_list(
        &mut self,
        label: &str,
        params: &[(String, TypeId)],
        args: &[ast::CallArg],
        span: Span,
    ) {
        if args.len() != params.len() {
            self.sink.emit(errors::argument_count_mismatch(
                label,
                params.len(),
                args.len(),
                span,
            ));
            for arg in args {
                self.check_expr(&arg.value);
            }
            return;
        }

        for (arg, (param_name, param_ty)) in args.iter().zip(params.iter()) {
            let arg_ty = self.check_expr_for_expected(&arg.value, *param_ty, false);
            if arg_ty != TypeInterner::ERROR
                && *param_ty != TypeInterner::ERROR
                && !self.types_compatible(*param_ty, arg_ty)
            {
                self.sink.emit(errors::argument_type_mismatch(
                    param_name,
                    &self.type_name(*param_ty),
                    &self.type_name(arg_ty),
                    arg.value.span(),
                ));
            }
        }
    }

    fn check_ident(&mut self, ident: &ast::Ident) -> TypeId {
        if let Some(&def_id) = self
            .resolve
            .resolutions
            .get(&ident.span)
            .or_else(|| self.decl_defs.get(&ident.span))
            && let Some(&type_id) = self.type_env.get(&def_id)
        {
            return type_id;
        }
        // If name resolution didn't find this ident, the resolver already
        // emitted an error. We return Error to avoid cascading type errors.
        TypeInterner::ERROR
    }

    fn is_displayable_type(&self, ty: TypeId) -> bool {
        matches!(
            self.interner.resolve(ty),
            Type::Int8
                | Type::Int16
                | Type::Int32
                | Type::Int64
                | Type::Uint8
                | Type::Uint16
                | Type::Uint32
                | Type::Uint64
                | Type::Float32
                | Type::Float64
                | Type::String
                | Type::Bool
        ) || self.type_implements_named_interface(ty, "Displayable")
    }

    fn type_implements_named_interface(&self, ty: TypeId, interface_name: &str) -> bool {
        let Some(&interface_ty) = self.named_types.get(interface_name) else {
            return false;
        };
        matches!(self.interner.resolve(interface_ty), Type::Interface(_))
            && self.interface_impls.contains_key(&(interface_ty, ty))
    }

    fn check_binary(&mut self, lhs: &Expr, op: BinOp, rhs: &Expr, span: Span) -> TypeId {
        let (lhs_ty, rhs_ty) = if Self::is_numeric_literal(lhs) && !Self::is_numeric_literal(rhs) {
            let rhs_ty = self.check_expr(rhs);
            let (rhs_base, _) = self.strip_secret_type(rhs_ty);
            let lhs_ty = if self.is_numeric(rhs_base) {
                self.check_expr_for_expected(lhs, rhs_base, false)
            } else {
                self.check_expr(lhs)
            };
            (lhs_ty, rhs_ty)
        } else if !Self::is_numeric_literal(lhs) && Self::is_numeric_literal(rhs) {
            let lhs_ty = self.check_expr(lhs);
            let (lhs_base, _) = self.strip_secret_type(lhs_ty);
            let rhs_ty = if self.is_numeric(lhs_base) {
                self.check_expr_for_expected(rhs, lhs_base, false)
            } else {
                self.check_expr(rhs)
            };
            (lhs_ty, rhs_ty)
        } else {
            (self.check_expr(lhs), self.check_expr(rhs))
        };

        // If either side is an error, propagate.
        if lhs_ty == TypeInterner::ERROR || rhs_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }

        let (lhs_base, lhs_secret) = self.strip_secret_type(lhs_ty);
        let (rhs_base, rhs_secret) = self.strip_secret_type(rhs_ty);
        let tainted = lhs_secret || rhs_secret;

        match op {
            // Arithmetic operators: both sides must be the same numeric type.
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Modulo => {
                if !self.is_numeric(lhs_base) || !self.is_numeric(rhs_base) || lhs_base != rhs_base
                {
                    self.sink.emit(errors::binary_op_mismatch(
                        Self::binop_str(op),
                        &self.type_name(lhs_ty),
                        &self.type_name(rhs_ty),
                        span,
                    ));
                    return TypeInterner::ERROR;
                }
                self.maybe_wrap_secret(lhs_base, tainted)
            }

            // Comparison operators: both sides must be the same type, returns bool.
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                if lhs_base != rhs_base {
                    self.sink.emit(errors::binary_op_mismatch(
                        Self::binop_str(op),
                        &self.type_name(lhs_ty),
                        &self.type_name(rhs_ty),
                        span,
                    ));
                    return TypeInterner::ERROR;
                }
                self.maybe_wrap_secret(TypeInterner::BOOL, tainted)
            }

            // Logical operators: both sides must be bool.
            BinOp::And | BinOp::Or => {
                if lhs_base != TypeInterner::BOOL || rhs_base != TypeInterner::BOOL {
                    self.sink.emit(errors::binary_op_mismatch(
                        Self::binop_str(op),
                        &self.type_name(lhs_ty),
                        &self.type_name(rhs_ty),
                        span,
                    ));
                    return TypeInterner::ERROR;
                }
                self.maybe_wrap_secret(TypeInterner::BOOL, tainted)
            }
        }
    }

    fn check_binary_for_expected_numeric(
        &mut self,
        lhs: &Expr,
        op: BinOp,
        rhs: &Expr,
        span: Span,
        expected_operand_ty: TypeId,
    ) -> TypeId {
        let lhs_ty = self.check_expr_for_expected(lhs, expected_operand_ty, false);
        let rhs_ty = self.check_expr_for_expected(rhs, expected_operand_ty, false);

        if lhs_ty == TypeInterner::ERROR || rhs_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }

        let (lhs_base, lhs_secret) = self.strip_secret_type(lhs_ty);
        let (rhs_base, rhs_secret) = self.strip_secret_type(rhs_ty);
        if !self.is_numeric(lhs_base) || !self.is_numeric(rhs_base) || lhs_base != rhs_base {
            self.sink.emit(errors::binary_op_mismatch(
                Self::binop_str(op),
                &self.type_name(lhs_ty),
                &self.type_name(rhs_ty),
                span,
            ));
            return TypeInterner::ERROR;
        }

        self.maybe_wrap_secret(lhs_base, lhs_secret || rhs_secret)
    }

    fn check_unary(&mut self, op: UnaryOp, operand: &Expr, span: Span) -> TypeId {
        let operand_ty = self.check_expr(operand);

        if operand_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }

        let (operand_base, tainted) = self.strip_secret_type(operand_ty);

        match op {
            UnaryOp::Not => {
                if operand_base != TypeInterner::BOOL {
                    self.sink.emit(errors::unary_op_mismatch(
                        "not",
                        &self.type_name(operand_ty),
                        span,
                    ));
                    return TypeInterner::ERROR;
                }
                self.maybe_wrap_secret(TypeInterner::BOOL, tainted)
            }
            UnaryOp::Neg => {
                if !self.is_numeric(operand_base) {
                    self.sink.emit(errors::unary_op_mismatch(
                        "-",
                        &self.type_name(operand_ty),
                        span,
                    ));
                    return TypeInterner::ERROR;
                }
                self.maybe_wrap_secret(operand_base, tainted)
            }
        }
    }

    fn reflection_param_facts_for_call(
        &mut self,
        template: &FunctionDef,
        param_types: &[TypeId],
        args: &[ast::CallArg],
    ) -> ReflectionParamFacts {
        let mut facts = ReflectionParamFacts::default();

        for (index, ((param, &param_ty), arg)) in template
            .params
            .iter()
            .zip(param_types.iter())
            .zip(args.iter())
            .enumerate()
        {
            if param.mutable {
                continue;
            }

            if self.type_id_is_named(param_ty, "TypeInfo") {
                if let Some(info_facts) = self.eval_type_info_facts(&arg.value) {
                    facts.type_info_kinds.push((index, info_facts.kind_tag));
                    facts
                        .type_info_primitives
                        .push((index, info_facts.primitive_tag));
                }
                continue;
            }

            if self.type_id_is_named(param_ty, "TypeKind")
                && let Some(kind_tag) = self.eval_type_kind_value(&arg.value)
            {
                facts.type_kind_values.push((index, kind_tag));
            }

            if self.type_id_is_named(param_ty, "TypePrimitive")
                && let Some(primitive_tag) = self.eval_type_primitive_value(&arg.value)
            {
                facts.type_primitive_values.push((index, primitive_tag));
            }
        }

        facts
    }

    fn check_range_builtin_call(
        &mut self,
        name: &str,
        type_args: &[TypeExpr],
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        self.expect_no_type_args(name, type_args, span);
        let list_int = self.interner.intern(Type::List(TypeInterner::INT64));
        if !(1..=3).contains(&args.len()) {
            self.sink.emit(errors::argument_count_range_mismatch(
                name,
                1,
                3,
                args.len(),
                span,
            ));
            for arg in args {
                self.check_expr(&arg.value);
            }
            return list_int;
        }

        let param_name = format!("{name} argument");
        for arg in args {
            let got = self.check_expr_for_expected(&arg.value, TypeInterner::INT64, false);
            if !self.types_compatible(TypeInterner::INT64, got) {
                self.sink.emit(errors::argument_type_mismatch(
                    &param_name,
                    &self.type_name(TypeInterner::INT64),
                    &self.type_name(got),
                    arg.value.span(),
                ));
            }
        }
        list_int
    }

    fn check_print_builtin_call(
        &mut self,
        name: &str,
        type_args: &[TypeExpr],
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        self.expect_no_type_args(name, type_args, span);
        for arg in args {
            let arg_ty = self.check_expr(&arg.value);
            if self.is_secret_type(arg_ty) {
                self.sink.emit(errors::secret_exposure(
                    name,
                    &self.type_name(arg_ty),
                    arg.value.span(),
                ));
            }
        }
        TypeInterner::NOTHING
    }

    fn check_math_abs_pipeline_step(
        &mut self,
        name: &str,
        current_ty: TypeId,
        type_args: &[TypeExpr],
        extra_args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        self.expect_no_type_args(name, type_args, span);
        if !extra_args.is_empty() {
            self.sink.emit(errors::argument_count_mismatch(
                name,
                1,
                extra_args.len() + 1,
                span,
            ));
            for arg in extra_args {
                self.check_expr(&arg.value);
            }
            return TypeInterner::ERROR;
        }

        if current_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }

        match self.math_numeric_builtin_base(current_ty) {
            Some((base, tainted)) => self.maybe_wrap_secret(base, tainted),
            None => {
                self.sink.emit(errors::argument_type_mismatch(
                    "#1",
                    "int64 or float64",
                    &self.type_name(current_ty),
                    span,
                ));
                TypeInterner::ERROR
            }
        }
    }

    fn check_math_min_max_pipeline_step(
        &mut self,
        name: &str,
        current_ty: TypeId,
        type_args: &[TypeExpr],
        extra_args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        self.expect_no_type_args(name, type_args, span);
        if extra_args.len() != 1 {
            self.sink.emit(errors::argument_count_mismatch(
                name,
                2,
                extra_args.len() + 1,
                span,
            ));
            for arg in extra_args {
                self.check_expr(&arg.value);
            }
            return TypeInterner::ERROR;
        }

        if current_ty == TypeInterner::ERROR {
            self.check_expr(&extra_args[0].value);
            return TypeInterner::ERROR;
        }

        let Some((base, left_tainted)) = self.math_numeric_builtin_base(current_ty) else {
            self.sink.emit(errors::argument_type_mismatch(
                "#1",
                "int64 or float64",
                &self.type_name(current_ty),
                span,
            ));
            self.check_expr(&extra_args[0].value);
            return TypeInterner::ERROR;
        };

        let right_arg = &extra_args[0];
        let right_ty = self.check_expr_for_expected(&right_arg.value, base, false);
        if right_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }

        let Some((right_base, right_tainted)) = self.math_numeric_builtin_base(right_ty) else {
            self.sink.emit(errors::argument_type_mismatch(
                "#2",
                &self.type_name(base),
                &self.type_name(right_ty),
                right_arg.value.span(),
            ));
            return TypeInterner::ERROR;
        };

        if !self.types_compatible(base, right_base) || !self.types_compatible(right_base, base) {
            self.sink.emit(errors::argument_type_mismatch(
                "#2",
                &self.type_name(base),
                &self.type_name(right_ty),
                right_arg.value.span(),
            ));
            return TypeInterner::ERROR;
        }

        self.maybe_wrap_secret(base, left_tainted || right_tainted)
    }

    fn math_numeric_builtin_base(&self, ty: TypeId) -> Option<(TypeId, bool)> {
        let (base, tainted) = self.strip_secret_type(ty);
        if self.types_compatible(TypeInterner::INT64, base) {
            Some((TypeInterner::INT64, tainted))
        } else if self.types_compatible(TypeInterner::FLOAT64, base) {
            Some((TypeInterner::FLOAT64, tainted))
        } else {
            None
        }
    }

    fn check_math_abs_builtin_call(
        &mut self,
        name: &str,
        type_args: &[TypeExpr],
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        self.expect_no_type_args(name, type_args, span);
        if args.len() != 1 {
            self.sink
                .emit(errors::argument_count_mismatch(name, 1, args.len(), span));
            for arg in args {
                self.check_expr(&arg.value);
            }
            return TypeInterner::ERROR;
        }

        let arg_ty = self.check_expr(&args[0].value);
        if arg_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }

        match self.math_numeric_builtin_base(arg_ty) {
            Some((base, tainted)) => self.maybe_wrap_secret(base, tainted),
            None => {
                self.sink.emit(errors::argument_type_mismatch(
                    "value",
                    "int64 or float64",
                    &self.type_name(arg_ty),
                    args[0].value.span(),
                ));
                TypeInterner::ERROR
            }
        }
    }

    fn check_math_min_max_builtin_call(
        &mut self,
        name: &str,
        type_args: &[TypeExpr],
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        self.expect_no_type_args(name, type_args, span);
        if args.len() != 2 {
            self.sink
                .emit(errors::argument_count_mismatch(name, 2, args.len(), span));
            for arg in args {
                self.check_expr(&arg.value);
            }
            return TypeInterner::ERROR;
        }

        let left_ty = self.check_expr(&args[0].value);
        if left_ty == TypeInterner::ERROR {
            self.check_expr(&args[1].value);
            return TypeInterner::ERROR;
        }

        let Some((base, left_tainted)) = self.math_numeric_builtin_base(left_ty) else {
            self.sink.emit(errors::argument_type_mismatch(
                "left",
                "int64 or float64",
                &self.type_name(left_ty),
                args[0].value.span(),
            ));
            self.check_expr(&args[1].value);
            return TypeInterner::ERROR;
        };

        let right_ty = self.check_expr_for_expected(&args[1].value, base, false);
        if right_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }

        let Some((right_base, right_tainted)) = self.math_numeric_builtin_base(right_ty) else {
            self.sink.emit(errors::argument_type_mismatch(
                "right",
                &self.type_name(base),
                &self.type_name(right_ty),
                args[1].value.span(),
            ));
            return TypeInterner::ERROR;
        };

        if !self.types_compatible(base, right_base) || !self.types_compatible(right_base, base) {
            self.sink.emit(errors::argument_type_mismatch(
                "right",
                &self.type_name(base),
                &self.type_name(right_ty),
                args[1].value.span(),
            ));
            return TypeInterner::ERROR;
        }

        self.maybe_wrap_secret(base, left_tainted || right_tainted)
    }

    fn check_call(
        &mut self,
        callee: &Expr,
        type_args: &[TypeExpr],
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        let callee_name = self.resolved_expr_name(callee);
        let callee_is_pure = callee_name
            .as_deref()
            .map(|name| {
                if Self::is_impure_builtin(name) {
                    false
                } else {
                    self.purity_map.get(name).copied().unwrap_or(true)
                }
            })
            .unwrap_or(false);
        let builtin_signature = self.builtin_signature(callee, type_args, span);

        // -- Capability / purity check --
        // Extract the callee name so we can look it up in the purity map.
        if let Some(callee_name) = callee_name.as_deref()
            && !callee_is_pure
        {
            // E0500: pure function calls impure function
            if self.current_function_pure
                && let Some(caller_name) = &self.current_function_name
            {
                self.sink
                    .emit(errors::pure_calls_impure(caller_name, callee_name, span));
            }
            // E0501: verify block calls impure function
            if self.in_verify_block
                && let Some(verify_name) = &self.current_verify_name
            {
                self.sink
                    .emit(errors::verify_calls_impure(verify_name, callee_name, span));
            }
        }

        if let Some(builtin_name) = callee_name.as_deref() {
            match builtin_name {
                "range" | "list.range" => {
                    return self.check_range_builtin_call(builtin_name, type_args, args, span);
                }
                "print" | "println" => {
                    return self.check_print_builtin_call(builtin_name, type_args, args, span);
                }
                "math.abs" => {
                    return self.check_math_abs_builtin_call(builtin_name, type_args, args, span);
                }
                "math.min" | "math.max" => {
                    return self.check_math_min_max_builtin_call(
                        builtin_name,
                        type_args,
                        args,
                        span,
                    );
                }
                _ => {}
            }
        }

        // Check for generic function call: `name[T](args...)`.
        if builtin_signature.is_none()
            && !type_args.is_empty()
            && let Some(function_name) = callee_name.as_deref()
        {
            if let Some(template) = self.generic_function_templates.get(function_name).cloned() {
                let concrete_args: Vec<TypeId> = type_args
                    .iter()
                    .map(|a| self.resolve_type_expr(a))
                    .collect();

                if template.type_params.len() != concrete_args.len() {
                    self.sink.emit(errors::unknown_type(
                        &format!(
                            "{} (expected {} type argument(s), got {})",
                            function_name,
                            template.type_params.len(),
                            concrete_args.len()
                        ),
                        span,
                    ));
                    return TypeInterner::ERROR;
                }

                let subst: HashMap<String, TypeId> = template
                    .type_params
                    .iter()
                    .zip(concrete_args.iter())
                    .map(|(p, &ty)| (p.name.clone(), ty))
                    .collect();
                let kind_subst: HashMap<String, String> = template
                    .type_params
                    .iter()
                    .zip(type_args.iter().zip(concrete_args.iter()))
                    .map(|(param, (type_arg, &resolved_ty))| {
                        (
                            param.name.clone(),
                            self.reflection_kind_tag_for_type_expr(type_arg, resolved_ty),
                        )
                    })
                    .collect();

                let old_subst = std::mem::replace(&mut self.type_var_subst, subst.clone());

                let param_types: Vec<TypeId> = template
                    .params
                    .iter()
                    .map(|p| self.resolve_type_expr(&p.ty))
                    .collect();
                let return_type = template
                    .return_type
                    .as_ref()
                    .map(|t| self.resolve_type_expr(t))
                    .unwrap_or(TypeInterner::NOTHING);

                self.type_var_subst = old_subst;

                // Check argument count and types.
                if args.len() != param_types.len() {
                    self.sink.emit(errors::argument_count_mismatch(
                        function_name,
                        param_types.len(),
                        args.len(),
                        span,
                    ));
                    for arg in args {
                        self.check_expr(&arg.value);
                    }
                    return TypeInterner::ERROR;
                }
                let mut arguments_match = true;
                for (arg, &expected) in args.iter().zip(param_types.iter()) {
                    let got = self.check_expr_for_expected(&arg.value, expected, false);
                    if !self.types_compatible(expected, got) {
                        arguments_match = false;
                        self.sink.emit(errors::type_mismatch(
                            &self.type_name(expected),
                            &self.type_name(got),
                            arg.value.span(),
                        ));
                    }
                }
                if arguments_match {
                    let param_facts =
                        self.reflection_param_facts_for_call(&template, &param_types, args);
                    self.check_generic_function_instantiation(
                        function_name,
                        &template,
                        &concrete_args,
                        subst,
                        kind_subst,
                        param_facts,
                    );
                }
                return return_type;
            }

            if self.function_signatures.contains_key(function_name) {
                self.sink.emit(errors::unknown_type(
                    &format!(
                        "{function_name} (expected 0 type argument(s), got {})",
                        type_args.len()
                    ),
                    span,
                ));
                for arg in args {
                    self.check_expr(&arg.value);
                }
                return TypeInterner::ERROR;
            }
        }

        if type_args.is_empty()
            && let Some(mid) = self.machine_transition_owner(callee)
        {
            return self.check_machine_transition_call(mid, args, span);
        }

        // Check for generic struct construction: `Name[T, U](fields...)`.
        if !type_args.is_empty()
            && let Some(struct_name) = callee_name.as_deref()
            && self.generic_struct_templates.contains_key(struct_name)
        {
            let concrete_args: Vec<TypeId> = type_args
                .iter()
                .map(|a| self.resolve_type_expr(a))
                .collect();
            let mono_ty = self.monomorphize_struct(struct_name, &concrete_args, span);
            if mono_ty != TypeInterner::ERROR {
                let sid = match self.interner.resolve(mono_ty) {
                    Type::Struct(sid) => *sid,
                    _ => return TypeInterner::ERROR,
                };
                return self.check_struct_constructor(sid, args, span);
            }
            return TypeInterner::ERROR;
        }

        if type_args.is_empty()
            && let Some(type_name) = callee_name.as_deref()
            && let Some(type_id) = self.named_types.get(type_name).copied()
        {
            match self.interner.resolve(type_id).clone() {
                Type::Struct(sid) => {
                    if Self::is_reflection_metadata_type_name(type_name) {
                        self.sink
                            .emit(errors::reflection_metadata_constructor(type_name, span));
                        for arg in args {
                            self.check_expr(&arg.value);
                        }
                        return TypeInterner::ERROR;
                    }
                    return self.check_struct_constructor(sid, args, span);
                }
                Type::Bitfield(bid) => {
                    return self.check_bitfield_constructor(bid, args, span);
                }
                Type::Machine(mid) => {
                    return self.check_machine_constructor(mid, args, span);
                }
                _ => {}
            }
        }

        let user_function_signature = if type_args.is_empty() {
            if builtin_signature.is_none() {
                callee_name
                    .as_deref()
                    .and_then(|name| self.function_signatures.get(name).cloned())
            } else {
                None
            }
        } else {
            None
        };

        let (param_types, return_type) = if let Some(signature) = builtin_signature {
            signature
        } else if let Some(signature) = user_function_signature {
            signature
        } else {
            let callee_ty = self.check_expr(callee);

            if callee_ty == TypeInterner::ERROR {
                // Still check argument expressions so we populate the type map.
                for arg in args {
                    self.check_expr(&arg.value);
                }
                return TypeInterner::ERROR;
            }

            // The callee must be a function type.
            match self.interner.resolve(callee_ty).clone() {
                Type::Function {
                    params,
                    return_type,
                } => (params, return_type),
                Type::Struct(sid) if self.is_struct_type_name_expr(callee) => {
                    return self.check_struct_constructor(sid, args, span);
                }
                Type::Bitfield(bid) if self.is_bitfield_type_name_expr(callee) => {
                    return self.check_bitfield_constructor(bid, args, span);
                }
                Type::Machine(mid) => {
                    return self.check_machine_constructor(mid, args, span);
                }
                _ => {
                    self.sink
                        .emit(errors::not_callable(&self.type_name(callee_ty), span));
                    for arg in args {
                        self.check_expr(&arg.value);
                    }
                    return TypeInterner::ERROR;
                }
            }
        };

        // Check argument count.
        if args.len() != param_types.len() {
            let func_name = callee_name
                .clone()
                .unwrap_or_else(|| "<anonymous>".to_string());
            self.sink.emit(errors::argument_count_mismatch(
                &func_name,
                param_types.len(),
                args.len(),
                span,
            ));
            // Still type-check the provided arguments.
            for arg in args {
                self.check_expr(&arg.value);
            }
            return return_type;
        }

        // Check each argument type.
        let mut tainted_return = false;
        let mut checked_arg_types = Vec::with_capacity(args.len());
        for (i, arg) in args.iter().enumerate() {
            let param_ty = param_types[i];
            let arg_ty = self.check_expr_for_expected(&arg.value, param_ty, false);
            checked_arg_types.push(arg_ty);

            if self.is_refinement_type(param_ty)
                && arg_ty != TypeInterner::ERROR
                && self.can_refine_from(arg_ty, param_ty)
            {
                self.sink.emit(errors::refinement_requires_handle_error(
                    &self.type_name(param_ty),
                    &self.type_name(arg_ty),
                    arg.value.span(),
                ));
                continue;
            }

            if let Some(callee_name) = callee_name.as_deref() {
                if Self::is_secret_output_boundary(callee_name) && self.is_secret_type(arg_ty) {
                    self.sink.emit(errors::secret_exposure(
                        callee_name,
                        &self.type_name(arg_ty),
                        arg.value.span(),
                    ));
                    continue;
                }

                if matches!(callee_name, "secret.redact" | "secret.compare")
                    && !self.is_secret_type(arg_ty)
                {
                    self.sink.emit(errors::secret_operation_requires_secret(
                        callee_name,
                        &self.type_name(arg_ty),
                        arg.value.span(),
                    ));
                    continue;
                }
            }

            let (matches, lifted_secret) = self.secret_argument_matches_param(param_ty, arg_ty);
            if matches {
                if lifted_secret {
                    let allows_secret_lifting = callee_name
                        .as_deref()
                        .map(|name| Self::is_secret_liftable_call(name, callee_is_pure))
                        .unwrap_or(callee_is_pure);

                    if !allows_secret_lifting {
                        self.sink.emit(errors::secret_exposure(
                            callee_name.as_deref().unwrap_or("<call>"),
                            &self.type_name(arg_ty),
                            arg.value.span(),
                        ));
                        continue;
                    }
                    tainted_return = true;
                }
                continue;
            }

            if !self.types_compatible(param_ty, arg_ty) {
                let param_name = format!("#{}", i + 1);
                self.sink.emit(errors::argument_type_mismatch(
                    &param_name,
                    &self.type_name(param_ty),
                    &self.type_name(arg_ty),
                    arg.value.span(),
                ));
            }
        }

        if matches!(callee_name.as_deref(), Some("secret.compare"))
            && checked_arg_types.len() == 2
            && let (Some(lhs_inner), Some(rhs_inner)) = (
                self.secret_inner_type(checked_arg_types[0]),
                self.secret_inner_type(checked_arg_types[1]),
            )
            && (!self.types_compatible(lhs_inner, rhs_inner)
                || !self.types_compatible(rhs_inner, lhs_inner))
        {
            self.sink.emit(errors::argument_type_mismatch(
                "#2",
                &self.type_name(checked_arg_types[0]),
                &self.type_name(checked_arg_types[1]),
                args[1].value.span(),
            ));
        }

        self.check_json_public_call_policy(
            callee_name.as_deref(),
            &checked_arg_types,
            args,
            return_type,
        );

        if let Some(callee_name) = callee_name.as_deref()
            && tainted_return
            && Self::is_secret_liftable_call(callee_name, callee_is_pure)
        {
            return self.maybe_wrap_secret(return_type, true);
        }

        return_type
    }

    fn check_field_access(&mut self, base: &Expr, field: &ast::Ident, span: Span) -> TypeId {
        if let Expr::Ident(base_ident) = base {
            if self.ident_def_kind(base_ident) == Some(DefKind::Enum) {
                return self.check_enum_variant(base_ident, field, &[], span);
            }
            if self.ident_def_kind(base_ident) == Some(DefKind::Interface) {
                let base_ty = self.check_ident(base_ident);
                if let Some(method_ty) = self.check_interface_method(base_ty, field, span) {
                    return method_ty;
                }
            }
            if self.ident_def_kind(base_ident) == Some(DefKind::Struct) {
                let base_ty = self.check_ident(base_ident);
                if let Some(method_ty) = self.check_type_module_method(base_ty, field, span) {
                    return method_ty;
                }
            }
            if self.ident_def_kind(base_ident) == Some(DefKind::Bitfield) {
                let base_ty = self.check_ident(base_ident);
                if let Some(method_ty) = self.check_type_module_method(base_ty, field, span) {
                    return method_ty;
                }
            }
            if let Some(type_id) = self.named_types.get(&base_ident.name).copied() {
                if matches!(self.interner.resolve(type_id), Type::Enum(_)) {
                    return self.check_enum_variant(base_ident, field, &[], span);
                }
                if let Some(method_ty) = self.check_interface_method(type_id, field, span) {
                    return method_ty;
                }
                if let Some(method_ty) = self.check_type_module_method(type_id, field, span) {
                    return method_ty;
                }
            }
        }

        if !matches!(base, Expr::Ident(_))
            && let Some(type_name) = Self::extract_dotted_name(base)
        {
            let type_name = self.resolved_or_expanded_name(&type_name, span);
            if let Some(type_id) = self.named_types.get(&type_name).copied() {
                if matches!(self.interner.resolve(type_id), Type::Enum(_)) {
                    return self.check_enum_variant_by_type(type_id, field, &[], span);
                }
                if let Some(method_ty) = self.check_interface_method(type_id, field, span) {
                    return method_ty;
                }
                if let Some(method_ty) = self.check_type_module_method(type_id, field, span) {
                    return method_ty;
                }
            }
        }

        let base_ty = self.check_expr(base);
        if base_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }

        match self.interner.resolve(base_ty) {
            Type::Secret(inner) => match self.interner.resolve(*inner) {
                Type::Struct(sid) => {
                    let struct_def = self.interner.resolve_struct(*sid);
                    if let Some((_, field_ty)) = struct_def
                        .fields
                        .iter()
                        .find(|(name, _)| name == &field.name)
                    {
                        self.maybe_wrap_secret(*field_ty, true)
                    } else {
                        self.sink.emit(errors::type_has_no_member(
                            &format!("secret[{}]", struct_def.name),
                            &field.name,
                            span,
                        ));
                        TypeInterner::ERROR
                    }
                }
                Type::Bitfield(bid) => {
                    let bitfield_def = self.interner.resolve_bitfield(*bid);
                    if let Some(field_def) = bitfield_def
                        .fields
                        .iter()
                        .find(|candidate| candidate.name == field.name)
                    {
                        self.maybe_wrap_secret(field_def.ty, true)
                    } else {
                        self.sink.emit(errors::type_has_no_member(
                            &format!("secret[{}]", bitfield_def.name),
                            &field.name,
                            span,
                        ));
                        TypeInterner::ERROR
                    }
                }
                Type::MachineState { machine, state } => {
                    match self.machine_state_field_type(*machine, *state, &field.name) {
                        Ok(field_ty) => self.maybe_wrap_secret(field_ty, true),
                        Err(type_name) => {
                            self.sink.emit(errors::type_has_no_member(
                                &format!("secret[{type_name}]"),
                                &field.name,
                                span,
                            ));
                            TypeInterner::ERROR
                        }
                    }
                }
                _ => {
                    self.sink.emit(errors::type_has_no_member(
                        &self.type_name(base_ty),
                        &field.name,
                        span,
                    ));
                    TypeInterner::ERROR
                }
            },
            Type::Struct(sid) => {
                let struct_def = self.interner.resolve_struct(*sid);
                if let Some((_, field_ty)) = struct_def
                    .fields
                    .iter()
                    .find(|(name, _)| name == &field.name)
                {
                    *field_ty
                } else {
                    self.sink.emit(errors::type_has_no_member(
                        &struct_def.name,
                        &field.name,
                        span,
                    ));
                    TypeInterner::ERROR
                }
            }
            Type::Bitfield(bid) => {
                let bitfield_def = self.interner.resolve_bitfield(*bid);
                if let Some(field_def) = bitfield_def
                    .fields
                    .iter()
                    .find(|candidate| candidate.name == field.name)
                {
                    field_def.ty
                } else {
                    self.sink.emit(errors::type_has_no_member(
                        &bitfield_def.name,
                        &field.name,
                        span,
                    ));
                    TypeInterner::ERROR
                }
            }
            Type::MachineState { machine, state } => {
                match self.machine_state_field_type(*machine, *state, &field.name) {
                    Ok(field_ty) => field_ty,
                    Err(type_name) => {
                        self.sink
                            .emit(errors::type_has_no_member(&type_name, &field.name, span));
                        TypeInterner::ERROR
                    }
                }
            }
            _ => {
                self.sink.emit(errors::type_has_no_member(
                    &self.type_name(base_ty),
                    &field.name,
                    span,
                ));
                TypeInterner::ERROR
            }
        }
    }

    fn machine_state_field_type(
        &self,
        machine: MachineId,
        state: MachineStateId,
        field_name: &str,
    ) -> Result<TypeId, String> {
        let machine_def = self.interner.resolve_machine(machine);
        let Some(state_def) = machine_def.state(state) else {
            return Err(format!("{} at <unknown>", machine_def.name));
        };
        let type_name = format!("{} at {}", machine_def.name, state_def.name);
        state_def
            .fields
            .iter()
            .find(|(name, _)| name == field_name)
            .map(|(_, ty)| *ty)
            .ok_or(type_name)
    }

    fn check_bitfield_literal_range(
        &mut self,
        bitfield_name: &str,
        field_name: &str,
        width: u16,
        value: i128,
        span: Span,
    ) {
        let max_value = if width >= 63 {
            if width == 64 {
                u64::MAX as i128
            } else {
                i64::MAX as i128
            }
        } else {
            (1_i128 << width) - 1
        };

        if value < 0 || value > max_value {
            self.sink.emit(errors::bitfield_literal_out_of_range(
                bitfield_name,
                field_name,
                width,
                value,
                span,
            ));
        }
    }

    fn check_interface_method(
        &mut self,
        type_id: TypeId,
        field: &ast::Ident,
        span: Span,
    ) -> Option<TypeId> {
        let Type::Interface(iid) = *self.interner.resolve(type_id) else {
            return None;
        };

        let (interface_name, method) = {
            let interface_def = self.interner.resolve_interface(iid);
            (
                interface_def.name.clone(),
                interface_def
                    .methods
                    .iter()
                    .find(|m| m.name == field.name)
                    .cloned(),
            )
        };

        if let Some(method) = method {
            let params = method.params.iter().map(|(_, ty, _)| *ty).collect();
            return Some(self.interner.intern(Type::Function {
                params,
                return_type: method.return_type,
            }));
        }

        self.sink.emit(errors::interface_has_no_member(
            &interface_name,
            &field.name,
            span,
        ));
        Some(TypeInterner::ERROR)
    }

    fn check_type_module_method(
        &mut self,
        type_id: TypeId,
        field: &ast::Ident,
        span: Span,
    ) -> Option<TypeId> {
        if let Type::Struct(sid) = *self.interner.resolve(type_id) {
            let struct_def = self.interner.resolve_struct(sid);
            if let Some(method) = struct_def
                .methods
                .iter()
                .find(|m| m.name == field.name)
                .cloned()
            {
                let params = method.params.iter().map(|(_, ty, _)| *ty).collect();
                return Some(self.interner.intern(Type::Function {
                    params,
                    return_type: method.return_type,
                }));
            }
        }

        if matches!(self.interner.resolve(type_id), Type::Bitfield(_)) {
            match field.name.as_str() {
                "to_bytes" => {
                    return Some(self.interner.intern(Type::Function {
                        params: vec![type_id],
                        return_type: TypeInterner::BYTES,
                    }));
                }
                "from_bytes" => {
                    let result_ty = self
                        .interner
                        .intern(Type::Result(type_id, TypeInterner::STRING));
                    return Some(self.interner.intern(Type::Function {
                        params: vec![TypeInterner::BYTES],
                        return_type: result_ty,
                    }));
                }
                _ => {}
            }
        }

        if let Some(method) = self
            .impl_methods_by_type
            .get(&type_id)
            .and_then(|methods| methods.get(&field.name))
            .cloned()
        {
            let params = method.params.iter().map(|(_, ty, _)| *ty).collect();
            return Some(self.interner.intern(Type::Function {
                params,
                return_type: method.return_type,
            }));
        }

        self.sink.emit(errors::type_has_no_member(
            &self.type_name(type_id),
            &field.name,
            span,
        ));
        None
    }

    fn check_enum_variant(
        &mut self,
        type_name: &ast::Ident,
        variant: &ast::Ident,
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        let enum_ty = self.check_ident(type_name);
        if enum_ty == TypeInterner::ERROR {
            return TypeInterner::ERROR;
        }
        self.check_enum_variant_by_type(enum_ty, variant, args, span)
    }

    fn check_enum_variant_by_type(
        &mut self,
        enum_ty: TypeId,
        variant: &ast::Ident,
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        let Type::Enum(eid) = *self.interner.resolve(enum_ty) else {
            self.sink.emit(errors::type_has_no_member(
                &self.type_name(enum_ty),
                &variant.name,
                span,
            ));
            return TypeInterner::ERROR;
        };

        let enum_def = self.interner.resolve_enum(eid).clone();
        let Some(variant_def) = enum_def
            .variants
            .iter()
            .find(|candidate| candidate.name == variant.name)
            .cloned()
        else {
            self.sink.emit(errors::type_has_no_member(
                &enum_def.name,
                &variant.name,
                span,
            ));
            return TypeInterner::ERROR;
        };

        if args.is_empty() {
            if variant_def.fields.is_empty() {
                return enum_ty;
            }

            let params = variant_def.fields.iter().map(|(_, ty)| *ty).collect();
            return self.interner.intern(Type::Function {
                params,
                return_type: enum_ty,
            });
        }

        if args.len() != variant_def.fields.len() {
            self.sink.emit(errors::argument_count_mismatch(
                &format!("{}.{}", enum_def.name, variant_def.name),
                variant_def.fields.len(),
                args.len(),
                span,
            ));
        }

        for (index, arg) in args.iter().enumerate() {
            if let Some((field_name, expected_ty)) = variant_def.fields.get(index) {
                let arg_ty = self.check_expr_for_expected(&arg.value, *expected_ty, false);
                if !self.types_compatible(*expected_ty, arg_ty) {
                    self.sink.emit(errors::argument_type_mismatch(
                        field_name,
                        &self.type_name(*expected_ty),
                        &self.type_name(arg_ty),
                        arg.value.span(),
                    ));
                }
            }
        }

        enum_ty
    }

    fn check_struct_constructor(
        &mut self,
        sid: StructId,
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        let struct_def = self.interner.resolve_struct(sid).clone();
        let mut assigned = vec![false; struct_def.fields.len()];
        let validates_refinements = struct_def
            .fields
            .iter()
            .any(|(_, ty)| self.is_refinement_type(*ty));

        for arg in args {
            let Some(field_index) = (match &arg.name {
                Some(name) => struct_def
                    .fields
                    .iter()
                    .position(|(field_name, _)| field_name == &name.name),
                None => assigned.iter().position(|filled| !filled),
            }) else {
                if let Some(name) = &arg.name {
                    self.sink.emit(errors::type_has_no_member(
                        &struct_def.name,
                        &name.name,
                        arg.span,
                    ));
                } else {
                    self.sink.emit(errors::argument_count_mismatch(
                        &struct_def.name,
                        struct_def.fields.len(),
                        args.len(),
                        span,
                    ));
                }
                self.check_expr(&arg.value);
                continue;
            };

            if assigned[field_index] {
                self.sink.emit(errors::duplicate_constructor_field(
                    &struct_def.name,
                    &struct_def.fields[field_index].0,
                    arg.span,
                ));
                self.check_expr(&arg.value);
                continue;
            }

            assigned[field_index] = true;
            let expected_ty = struct_def.fields[field_index].1;
            let arg_ty = if self.is_refinement_type(expected_ty) {
                match &arg.value {
                    Expr::Handle(_, _, _, _) => {
                        self.check_expr_for_expected(&arg.value, expected_ty, true)
                    }
                    _ => self.check_expr(&arg.value),
                }
            } else {
                self.check_expr_for_expected(&arg.value, expected_ty, false)
            };
            if self.is_refinement_type(expected_ty) && self.can_refine_from(arg_ty, expected_ty) {
                continue;
            }
            if !self.types_compatible(expected_ty, arg_ty) {
                self.sink.emit(errors::argument_type_mismatch(
                    &struct_def.fields[field_index].0,
                    &self.type_name(expected_ty),
                    &self.type_name(arg_ty),
                    arg.value.span(),
                ));
            }
        }

        for (index, (field_name, _)) in struct_def.fields.iter().enumerate() {
            if !assigned[index] {
                self.sink.emit(errors::missing_constructor_field(
                    &struct_def.name,
                    field_name,
                    span,
                ));
            }
        }

        let struct_ty = self.interner.intern(Type::Struct(sid));
        if validates_refinements {
            self.interner
                .intern(Type::Result(struct_ty, TypeInterner::STRING))
        } else {
            struct_ty
        }
    }

    fn check_bitfield_constructor(
        &mut self,
        bid: BitfieldId,
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        let bitfield_def = self.interner.resolve_bitfield(bid).clone();
        let mut assigned = vec![false; bitfield_def.fields.len()];
        let mut requires_runtime_validation = false;

        for arg in args {
            let Some(field_index) = (match &arg.name {
                Some(name) => bitfield_def
                    .fields
                    .iter()
                    .position(|field| field.name == name.name),
                None => assigned.iter().position(|filled| !filled),
            }) else {
                if let Some(name) = &arg.name {
                    self.sink.emit(errors::type_has_no_member(
                        &bitfield_def.name,
                        &name.name,
                        arg.span,
                    ));
                } else {
                    self.sink.emit(errors::argument_count_mismatch(
                        &bitfield_def.name,
                        bitfield_def.fields.len(),
                        args.len(),
                        span,
                    ));
                }
                self.check_expr(&arg.value);
                continue;
            };

            if assigned[field_index] {
                self.sink.emit(errors::duplicate_constructor_field(
                    &bitfield_def.name,
                    &bitfield_def.fields[field_index].name,
                    arg.span,
                ));
                self.check_expr(&arg.value);
                continue;
            }

            assigned[field_index] = true;
            let field_def = &bitfield_def.fields[field_index];
            let arg_ty = self.check_expr_for_expected(&arg.value, field_def.ty, false);
            if !self.types_compatible(field_def.ty, arg_ty) {
                self.sink.emit(errors::argument_type_mismatch(
                    &field_def.name,
                    &self.type_name(field_def.ty),
                    &self.type_name(arg_ty),
                    arg.value.span(),
                ));
                continue;
            }

            if let TypeBitfieldFieldKind::Bits { width } = field_def.kind
                && (field_def.ty == TypeInterner::INT64 || field_def.ty == TypeInterner::UINT64)
            {
                match &arg.value {
                    Expr::IntLiteral(value, literal_span) => self.check_bitfield_literal_range(
                        &bitfield_def.name,
                        &field_def.name,
                        width,
                        *value,
                        *literal_span,
                    ),
                    _ => {
                        requires_runtime_validation = true;
                    }
                }
            }
        }

        for (index, field_def) in bitfield_def.fields.iter().enumerate() {
            if !assigned[index] {
                self.sink.emit(errors::missing_constructor_field(
                    &bitfield_def.name,
                    &field_def.name,
                    span,
                ));
            }
        }

        let bitfield_ty = self.interner.intern(Type::Bitfield(bid));
        if requires_runtime_validation {
            self.interner
                .intern(Type::Result(bitfield_ty, TypeInterner::STRING))
        } else {
            bitfield_ty
        }
    }

    fn check_machine_constructor(
        &mut self,
        mid: jett_types::MachineId,
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        let machine_def = self.interner.resolve_machine(mid).clone();
        if args.is_empty() {
            self.sink.emit(errors::invalid_machine_construction(
                &machine_def.name,
                "expected an initial state argument",
                span,
            ));
            return TypeInterner::ERROR;
        }

        let state_ident = match &args[0].value {
            Expr::Ident(ident) if args[0].name.is_none() => ident,
            _ => {
                self.sink.emit(errors::invalid_machine_construction(
                    &machine_def.name,
                    "first argument must be a bare state name",
                    args[0].span,
                ));
                for arg in &args[1..] {
                    self.check_expr(&arg.value);
                }
                return TypeInterner::ERROR;
            }
        };

        let Some(state_id) = machine_def.state_id(&state_ident.name) else {
            self.sink.emit(errors::invalid_machine_construction(
                &machine_def.name,
                &format!("unknown state `{}`", state_ident.name),
                state_ident.span,
            ));
            for arg in &args[1..] {
                self.check_expr(&arg.value);
            }
            return TypeInterner::ERROR;
        };

        let state_def = machine_def
            .state(state_id)
            .expect("state_id came from the same machine definition");
        let payload_args = &args[1..];
        if payload_args.len() != state_def.fields.len() {
            self.sink.emit(errors::invalid_machine_construction(
                &machine_def.name,
                &format!(
                    "state `{}` expects {} payload field(s), got {}",
                    state_def.name,
                    state_def.fields.len(),
                    payload_args.len()
                ),
                span,
            ));
        }

        for (arg, (field_name, expected_ty)) in payload_args.iter().zip(state_def.fields.iter()) {
            let arg_ty = self.check_expr_for_expected(&arg.value, *expected_ty, false);
            if !self.types_compatible(*expected_ty, arg_ty) {
                self.sink.emit(errors::argument_type_mismatch(
                    field_name,
                    &self.type_name(*expected_ty),
                    &self.type_name(arg_ty),
                    arg.value.span(),
                ));
            }
        }
        for arg in payload_args.iter().skip(state_def.fields.len()) {
            self.check_expr(&arg.value);
        }

        self.interner.intern(Type::MachineState {
            machine: mid,
            state: state_id,
        })
    }

    fn check_machine_transition_call(
        &mut self,
        mid: jett_types::MachineId,
        args: &[ast::CallArg],
        span: Span,
    ) -> TypeId {
        let machine_def = self.interner.resolve_machine(mid).clone();
        if args.len() < 2 {
            self.sink.emit(errors::invalid_machine_transition_call(
                &machine_def.name,
                "transition",
                "expected a source value and target state",
                span,
            ));
            for arg in args {
                self.check_expr(&arg.value);
            }
            return TypeInterner::ERROR;
        }

        let source_ty = self.check_expr(&args[0].value);
        let source_state = match self.interner.resolve(source_ty).clone() {
            Type::MachineState { machine, state } if machine == mid => Some(state),
            Type::Error => None,
            _ => {
                self.sink.emit(errors::invalid_machine_transition_call(
                    &machine_def.name,
                    "transition",
                    &format!(
                        "source value must be `{}` at a known state, got `{}`",
                        machine_def.name,
                        self.type_name(source_ty)
                    ),
                    args[0].value.span(),
                ));
                None
            }
        };

        let target_ident = match &args[1].value {
            Expr::Ident(ident) if args[1].name.is_none() => ident,
            _ => {
                self.sink.emit(errors::invalid_machine_transition_call(
                    &machine_def.name,
                    "transition",
                    "second argument must be a bare target state name",
                    args[1].span,
                ));
                for arg in &args[2..] {
                    self.check_expr(&arg.value);
                }
                return TypeInterner::ERROR;
            }
        };

        let Some(target_state) = machine_def.state_id(&target_ident.name) else {
            self.sink.emit(errors::invalid_machine_transition_call(
                &machine_def.name,
                &format!("to {}", target_ident.name),
                &format!("unknown target state `{}`", target_ident.name),
                target_ident.span,
            ));
            for arg in &args[2..] {
                self.check_expr(&arg.value);
            }
            return TypeInterner::ERROR;
        };

        if let Some(source_state) = source_state
            && !machine_def.has_transition(source_state, target_state)
        {
            let source_name = machine_def
                .state(source_state)
                .map(|state| state.name.as_str())
                .unwrap_or("<unknown>");
            self.sink.emit(errors::invalid_machine_transition_call(
                &machine_def.name,
                &format!("{source_name} to {}", target_ident.name),
                "edge is not declared",
                span,
            ));
        }

        let target_def = machine_def
            .state(target_state)
            .expect("state_id came from the same machine definition");
        let payload_args = &args[2..];
        if payload_args.len() != target_def.fields.len() {
            self.sink.emit(errors::invalid_machine_transition_call(
                &machine_def.name,
                &format!("to {}", target_def.name),
                &format!(
                    "target state `{}` expects {} payload field(s), got {}",
                    target_def.name,
                    target_def.fields.len(),
                    payload_args.len()
                ),
                span,
            ));
        }

        for (arg, (field_name, expected_ty)) in payload_args.iter().zip(target_def.fields.iter()) {
            let arg_ty = self.check_expr_for_expected(&arg.value, *expected_ty, false);
            if !self.types_compatible(*expected_ty, arg_ty) {
                self.sink.emit(errors::argument_type_mismatch(
                    field_name,
                    &self.type_name(*expected_ty),
                    &self.type_name(arg_ty),
                    arg.value.span(),
                ));
            }
        }
        for arg in payload_args.iter().skip(target_def.fields.len()) {
            self.check_expr(&arg.value);
        }

        self.interner.intern(Type::MachineState {
            machine: mid,
            state: target_state,
        })
    }

    fn machine_transition_owner(&self, callee: &Expr) -> Option<MachineId> {
        let Expr::FieldAccess(base, field, _) = callee else {
            return None;
        };
        if field.name != "transition" {
            return None;
        }

        if let Some(def_id) = self
            .resolve
            .resolutions
            .get(&callee.span())
            .copied()
            .or_else(|| self.decl_defs.get(&callee.span()).copied())
        {
            let def = self.resolve.scope_table.def(def_id);
            if def.kind == DefKind::Machine
                && let Some(owner_ty) = self.named_types.get(&def.name).copied()
                && let Type::Machine(mid) = *self.interner.resolve(owner_ty)
            {
                return Some(mid);
            }
        }

        let owner_name = self.expanded_dotted_expr_name(base)?;
        let owner_ty = self.named_types.get(&owner_name).copied()?;
        match self.interner.resolve(owner_ty) {
            Type::Machine(mid) => Some(*mid),
            _ => None,
        }
    }

    fn check_machine_state_check(
        &mut self,
        inner: &Expr,
        state: &ast::Ident,
        span: Span,
    ) -> TypeId {
        let inner_ty = self.check_expr(inner);
        let machine_id = match self.interner.resolve(inner_ty).clone() {
            Type::Machine(mid) => mid,
            Type::MachineState { machine, .. } => machine,
            Type::Error => return TypeInterner::BOOL,
            _ => {
                self.sink.emit(errors::invalid_machine_state_check(
                    &self.type_name(inner_ty),
                    &state.name,
                    "value is not a machine",
                    span,
                ));
                return TypeInterner::ERROR;
            }
        };

        let machine_def = self.interner.resolve_machine(machine_id);
        if machine_def.state_id(&state.name).is_none() {
            self.sink.emit(errors::invalid_machine_state_check(
                &machine_def.name,
                &state.name,
                "state is not declared on this machine",
                state.span,
            ));
            return TypeInterner::ERROR;
        }

        TypeInterner::BOOL
    }

    fn check_list_construct(&mut self, elems: &[Expr]) -> TypeId {
        if elems.is_empty() {
            // Empty list: list[<error>] since we can't infer the element type.
            return self.interner.intern(Type::List(TypeInterner::ERROR));
        }

        let first_ty = self.check_expr(&elems[0]);
        let (element_ty, mut tainted) = self.strip_secret_type(first_ty);
        for elem in &elems[1..] {
            let elem_ty = self.check_expr(elem);
            let (elem_base_ty, elem_secret) = self.strip_secret_type(elem_ty);
            if !self.types_compatible(element_ty, elem_base_ty)
                && !self.types_compatible(elem_base_ty, element_ty)
            {
                self.sink.emit(errors::type_mismatch(
                    &self.type_name(element_ty),
                    &self.type_name(elem_ty),
                    elem.span(),
                ));
            }
            tainted |= elem_secret;
        }

        let element_ty = self.maybe_wrap_secret(element_ty, tainted);
        self.interner.intern(Type::List(element_ty))
    }

    fn check_list_construct_for_expected(
        &mut self,
        elems: &[Expr],
        expected_list_ty: TypeId,
        expected_element_ty: TypeId,
        allow_refinement_handle: bool,
    ) -> TypeId {
        for elem in elems {
            let elem_ty = self.check_expr_with_optional_expected(
                elem,
                expected_element_ty,
                allow_refinement_handle,
            );
            if !self.types_compatible(expected_element_ty, elem_ty) {
                self.sink.emit(errors::type_mismatch(
                    &self.type_name(expected_element_ty),
                    &self.type_name(elem_ty),
                    elem.span(),
                ));
            }
        }

        expected_list_ty
    }

    fn check_map_construct(&mut self, entries: &[(Expr, Expr)]) -> TypeId {
        if entries.is_empty() {
            return self
                .interner
                .intern(Type::Map(TypeInterner::ERROR, TypeInterner::ERROR));
        }

        let first_key_ty = self.check_expr(&entries[0].0);
        let first_value_ty = self.check_expr(&entries[0].1);
        let (key_ty, mut key_tainted) = self.strip_secret_type(first_key_ty);
        let (value_ty, mut value_tainted) = self.strip_secret_type(first_value_ty);

        for (key_expr, value_expr) in &entries[1..] {
            let entry_key_ty = self.check_expr(key_expr);
            let (entry_key_base_ty, entry_key_secret) = self.strip_secret_type(entry_key_ty);
            if !self.types_compatible(key_ty, entry_key_base_ty)
                && !self.types_compatible(entry_key_base_ty, key_ty)
            {
                self.sink.emit(errors::type_mismatch(
                    &self.type_name(key_ty),
                    &self.type_name(entry_key_ty),
                    key_expr.span(),
                ));
            }
            key_tainted |= entry_key_secret;

            let entry_value_ty = self.check_expr(value_expr);
            let (entry_value_base_ty, entry_value_secret) = self.strip_secret_type(entry_value_ty);
            if !self.types_compatible(value_ty, entry_value_base_ty)
                && !self.types_compatible(entry_value_base_ty, value_ty)
            {
                self.sink.emit(errors::type_mismatch(
                    &self.type_name(value_ty),
                    &self.type_name(entry_value_ty),
                    value_expr.span(),
                ));
            }
            value_tainted |= entry_value_secret;
        }

        let key_ty = self.maybe_wrap_secret(key_ty, key_tainted);
        let value_ty = self.maybe_wrap_secret(value_ty, value_tainted);
        self.interner.intern(Type::Map(key_ty, value_ty))
    }

    fn check_map_construct_for_expected(
        &mut self,
        entries: &[(Expr, Expr)],
        expected_map_ty: TypeId,
        expected_key_ty: TypeId,
        expected_value_ty: TypeId,
        allow_refinement_handle: bool,
    ) -> TypeId {
        for (key_expr, value_expr) in entries {
            let key_ty = self.check_expr_with_optional_expected(
                key_expr,
                expected_key_ty,
                allow_refinement_handle,
            );
            if !self.types_compatible(expected_key_ty, key_ty) {
                self.sink.emit(errors::type_mismatch(
                    &self.type_name(expected_key_ty),
                    &self.type_name(key_ty),
                    key_expr.span(),
                ));
            }

            let value_ty = self.check_expr_with_optional_expected(
                value_expr,
                expected_value_ty,
                allow_refinement_handle,
            );
            if !self.types_compatible(expected_value_ty, value_ty) {
                self.sink.emit(errors::type_mismatch(
                    &self.type_name(expected_value_ty),
                    &self.type_name(value_ty),
                    value_expr.span(),
                ));
            }
        }

        expected_map_ty
    }

    fn check_expr_with_optional_expected(
        &mut self,
        expr: &Expr,
        expected_ty: TypeId,
        allow_refinement_handle: bool,
    ) -> TypeId {
        if expected_ty == TypeInterner::ERROR {
            self.check_expr(expr)
        } else {
            self.check_expr_for_expected(expr, expected_ty, allow_refinement_handle)
        }
    }

    fn check_wrapper_payload_for_expected(
        &mut self,
        payload: &Expr,
        expected_wrapper_ty: TypeId,
        expected_payload_ty: TypeId,
        allow_refinement_handle: bool,
    ) -> TypeId {
        let payload_ty = self.check_expr_with_optional_expected(
            payload,
            expected_payload_ty,
            allow_refinement_handle,
        );
        if !self.types_compatible(expected_payload_ty, payload_ty) {
            self.sink.emit(errors::type_mismatch(
                &self.type_name(expected_payload_ty),
                &self.type_name(payload_ty),
                payload.span(),
            ));
        }

        expected_wrapper_ty
    }

    fn check_handle(
        &mut self,
        target: &Expr,
        bind_name: Option<&ast::Ident>,
        body: &Block,
        span: Span,
    ) -> TypeId {
        let target_ty = self.check_expr(target);
        self.check_handle_with_target_type(target_ty, bind_name, body, span)
    }

    fn check_handle_with_target_type(
        &mut self,
        target_ty: TypeId,
        bind_name: Option<&ast::Ident>,
        body: &Block,
        span: Span,
    ) -> TypeId {
        if target_ty == TypeInterner::ERROR {
            self.check_handle_body(body);
            self.validate_handle_terminator(body, TypeInterner::ERROR);
            return TypeInterner::ERROR;
        }

        match self.interner.resolve(target_ty).clone() {
            Type::Result(ok_ty, err_ty) => {
                if bind_name.is_none() {
                    self.sink.emit(errors::result_requires_handle_error(span));
                }
                if let Some(name) = bind_name
                    && let Some(def_id) = self.declaration_def_id(name.span)
                {
                    self.type_env.insert(def_id, err_ty);
                }
                self.check_handle_body(body);
                self.validate_handle_terminator(body, ok_ty);
                ok_ty
            }
            Type::Optional(inner_ty) => {
                if bind_name.is_some() {
                    self.sink.emit(errors::optional_requires_bare_handle(span));
                }
                self.check_handle_body(body);
                self.validate_handle_terminator(body, inner_ty);
                inner_ty
            }
            _ => {
                self.sink.emit(errors::handle_requires_result_or_optional(
                    &self.type_name(target_ty),
                    span,
                ));
                self.check_handle_body(body);
                self.validate_handle_terminator(body, TypeInterner::ERROR);
                TypeInterner::ERROR
            }
        }
    }

    fn check_handle_body(&mut self, body: &Block) {
        self.handle_body_depth += 1;
        self.check_block(body);
        self.handle_body_depth -= 1;
    }

    fn validate_handle_terminator(&mut self, body: &Block, success_ty: TypeId) {
        let Some(last_stmt) = body.stmts.last() else {
            self.sink
                .emit(errors::handle_block_requires_return_or_default(body.span));
            return;
        };

        match last_stmt {
            Stmt::Return(_) => {}
            Stmt::Expr(expr_stmt) => {
                if let Expr::Default(default_value, _) = &expr_stmt.expr {
                    if success_ty != TypeInterner::ERROR {
                        let mut default_ty = self
                            .type_map
                            .get(&expr_stmt.expr.span())
                            .copied()
                            .unwrap_or(TypeInterner::ERROR);
                        if !self.types_compatible(success_ty, default_ty) {
                            default_ty =
                                self.check_expr_for_expected(default_value, success_ty, false);
                            self.type_map.insert(expr_stmt.expr.span(), default_ty);
                        }
                        if !self.types_compatible(success_ty, default_ty) {
                            self.sink.emit(errors::type_mismatch(
                                &self.type_name(success_ty),
                                &self.type_name(default_ty),
                                expr_stmt.expr.span(),
                            ));
                        }
                    }
                } else {
                    self.sink
                        .emit(errors::handle_block_requires_return_or_default(
                            expr_stmt.span,
                        ));
                }
            }
            _ => self
                .sink
                .emit(errors::handle_block_requires_return_or_default(stmt_span(
                    last_stmt,
                ))),
        }
    }

    // ------------------------------------------------------------------
    // Type expression resolution (AST TypeExpr → TypeId)
    // ------------------------------------------------------------------

    pub fn resolve_type_expr(&mut self, type_expr: &TypeExpr) -> TypeId {
        match type_expr {
            TypeExpr::Named(ident) => self.resolve_named_type(&ident.name, ident.span),
            TypeExpr::Generic(name, args, _span) => {
                self.resolve_generic_type(&name.name, args, name.span)
            }
            TypeExpr::View(inner, _span) => {
                // View types are transparent for type checking purposes.
                self.resolve_type_expr(inner)
            }
            TypeExpr::Function(param_types, return_type, _span) => {
                let params = param_types
                    .iter()
                    .map(|t| self.resolve_type_expr(t))
                    .collect();
                let ret = self.resolve_type_expr(return_type);
                self.interner.intern(Type::Function {
                    params,
                    return_type: ret,
                })
            }
            TypeExpr::StateQualified(base, state, _span) => {
                let base_ty = self.resolve_type_expr(base);
                self.resolve_machine_state_type(base_ty, state)
            }
        }
    }

    fn resolve_machine_state_type(&mut self, base_ty: TypeId, state: &ast::Ident) -> TypeId {
        let machine_id = match self.interner.resolve(base_ty).clone() {
            Type::Machine(mid) => mid,
            Type::MachineState { machine, .. } => machine,
            Type::Error => return TypeInterner::ERROR,
            _ => {
                self.sink.emit(errors::unknown_type(
                    &format!("{} at {}", self.type_name(base_ty), state.name),
                    state.span,
                ));
                return TypeInterner::ERROR;
            }
        };

        let machine_def = self.interner.resolve_machine(machine_id);
        let Some(state_id) = machine_def.state_id(&state.name) else {
            self.sink.emit(errors::unknown_type(
                &format!("{} at {}", machine_def.name, state.name),
                state.span,
            ));
            return TypeInterner::ERROR;
        };

        self.interner.intern(Type::MachineState {
            machine: machine_id,
            state: state_id,
        })
    }

    fn resolve_named_type(&mut self, name: &str, span: Span) -> TypeId {
        // Type variable substitution takes priority (active during monomorphization).
        if let Some(&ty) = self.type_var_subst.get(name) {
            return ty;
        }
        let lookup_name = self.resolved_or_expanded_name(name, span);
        match name {
            "int8" => TypeInterner::INT8,
            "int16" => TypeInterner::INT16,
            "int32" => TypeInterner::INT32,
            "int64" => TypeInterner::INT64,
            "uint8" => TypeInterner::UINT8,
            "uint16" => TypeInterner::UINT16,
            "uint32" => TypeInterner::UINT32,
            "uint64" => TypeInterner::UINT64,
            "float32" => TypeInterner::FLOAT32,
            "float64" => TypeInterner::FLOAT64,
            "string" => TypeInterner::STRING,
            "bool" => TypeInterner::BOOL,
            "bytes" => TypeInterner::BYTES,
            "nothing" => TypeInterner::NOTHING,
            "TypeConstruction" => TypeInterner::TYPE_CONSTRUCTION,
            _ if self.named_types.contains_key(&lookup_name) => self.named_types[&lookup_name],
            _ if self.type_aliases.contains_key(&lookup_name) => {
                self.resolve_type_alias(&lookup_name, span)
            }
            // Capability types are recognised but opaque — no further type
            // checking is performed on values of these types.
            _ if capability::is_capability_type(name) => TypeInterner::ERROR,
            _ => {
                self.sink.emit(errors::unknown_type(name, span));
                TypeInterner::ERROR
            }
        }
    }

    fn resolve_type_alias(&mut self, name: &str, span: Span) -> TypeId {
        if let Some(&ty) = self.named_types.get(name) {
            return ty;
        }

        if !self.resolving_type_aliases.insert(name.to_string()) {
            self.sink.emit(errors::unknown_type(name, span));
            return TypeInterner::ERROR;
        }

        let alias = self
            .type_aliases
            .get(name)
            .cloned()
            .expect("type alias existence checked before resolution");
        let base_ty = self.resolve_type_expr(&alias.base_type);
        let alias_ty = if alias.constraint.is_some() {
            self.interner.intern(Type::Refinement {
                name: name.to_string(),
                base: base_ty,
            })
        } else {
            base_ty
        };

        self.named_types.insert(name.to_string(), alias_ty);
        if alias.name.span.file.is_stdlib() {
            self.trusted_stdlib_named_types
                .insert(name.to_string(), alias_ty);
        }
        self.resolving_type_aliases.remove(name);
        alias_ty
    }

    fn resolve_generic_type(&mut self, name: &str, args: &[TypeExpr], span: Span) -> TypeId {
        let lookup_name = self.resolved_or_expanded_name(name, span);
        match name {
            "list" => {
                if args.len() == 1 {
                    let inner = self.resolve_type_expr(&args[0]);
                    self.interner.intern(Type::List(inner))
                } else {
                    self.sink.emit(errors::unknown_type(
                        &format!("list (expected 1 type argument, got {})", args.len()),
                        span,
                    ));
                    TypeInterner::ERROR
                }
            }
            "map" => {
                if args.len() == 2 {
                    let key = self.resolve_type_expr(&args[0]);
                    let val = self.resolve_type_expr(&args[1]);
                    self.interner.intern(Type::Map(key, val))
                } else {
                    self.sink.emit(errors::unknown_type(
                        &format!("map (expected 2 type arguments, got {})", args.len()),
                        span,
                    ));
                    TypeInterner::ERROR
                }
            }
            "set" => {
                if args.len() == 1 {
                    let inner = self.resolve_type_expr(&args[0]);
                    self.interner.intern(Type::Set(inner))
                } else {
                    self.sink.emit(errors::unknown_type(
                        &format!("set (expected 1 type argument, got {})", args.len()),
                        span,
                    ));
                    TypeInterner::ERROR
                }
            }
            "optional" => {
                if args.len() == 1 {
                    let inner = self.resolve_type_expr(&args[0]);
                    self.interner.intern(Type::Optional(inner))
                } else {
                    self.sink.emit(errors::unknown_type(
                        &format!("optional (expected 1 type argument, got {})", args.len()),
                        span,
                    ));
                    TypeInterner::ERROR
                }
            }
            "result" => {
                if args.len() == 2 {
                    let ok = self.resolve_type_expr(&args[0]);
                    let err = self.resolve_type_expr(&args[1]);
                    self.interner.intern(Type::Result(ok, err))
                } else {
                    self.sink.emit(errors::unknown_type(
                        &format!("result (expected 2 type arguments, got {})", args.len()),
                        span,
                    ));
                    TypeInterner::ERROR
                }
            }
            "secret" => {
                if args.len() == 1 {
                    let inner = self.resolve_type_expr(&args[0]);
                    self.interner.intern(Type::Secret(inner))
                } else {
                    self.sink.emit(errors::unknown_type(
                        &format!("secret (expected 1 type argument, got {})", args.len()),
                        span,
                    ));
                    TypeInterner::ERROR
                }
            }
            _ => {
                // Check if this is a user-defined generic struct.
                if self.generic_struct_templates.contains_key(&lookup_name) {
                    let concrete_args: Vec<TypeId> =
                        args.iter().map(|a| self.resolve_type_expr(a)).collect();
                    return self.monomorphize_struct(&lookup_name, &concrete_args, span);
                }
                self.sink.emit(errors::unknown_type(name, span));
                TypeInterner::ERROR
            }
        }
    }

    /// Monomorphize a generic struct with the given concrete type arguments.
    ///
    /// Returns the `TypeId` of the resulting `Type::Struct`, creating a new
    /// `StructId` on first use and caching it for subsequent calls with the
    /// same `(name, type_args)` key.
    fn monomorphize_struct(&mut self, name: &str, type_args: &[TypeId], span: Span) -> TypeId {
        // Check the cache first.
        let cache_key = (name.to_string(), type_args.to_vec());
        if let Some(&cached) = self.monomorphized_structs.get(&cache_key) {
            return cached;
        }

        let template = match self.generic_struct_templates.get(name).cloned() {
            Some(t) => t,
            None => return TypeInterner::ERROR,
        };

        if template.type_params.len() != type_args.len() {
            self.sink.emit(errors::unknown_type(
                &format!(
                    "{} (expected {} type argument(s), got {})",
                    name,
                    template.type_params.len(),
                    type_args.len()
                ),
                span,
            ));
            return TypeInterner::ERROR;
        }

        // Build substitution map: type param name → concrete TypeId.
        let substitution: HashMap<String, TypeId> = template
            .type_params
            .iter()
            .zip(type_args.iter())
            .map(|(param, &ty)| (param.name.clone(), ty))
            .collect();

        // Install the substitution, resolve fields, then restore.
        let old_subst = std::mem::replace(&mut self.type_var_subst, substitution);

        let fields: Vec<(String, TypeId)> = template
            .fields
            .iter()
            .map(|field| {
                let field_ty = self.resolve_type_expr(&field.ty);
                (field.name.name.clone(), field_ty)
            })
            .collect();

        let methods: Vec<FunctionSig> = template
            .methods
            .iter()
            .map(|method| self.method_signature(method))
            .collect();

        self.type_var_subst = old_subst;

        // Build a mangled name, e.g. "Pair[int64, string]".
        let type_arg_names: Vec<String> = type_args.iter().map(|&ty| self.type_name(ty)).collect();
        let mono_name = format!("{}[{}]", name, type_arg_names.join(", "));
        let reflection_fields = self.reflection_fields_for_resolved_struct(&template, &fields);

        let sid = self.interner.add_struct(TypeStructDef {
            name: mono_name.clone(),
            fields,
            methods,
        });
        let ty = self.interner.intern(Type::Struct(sid));

        self.reflection_fields_by_id
            .insert(ty, (mono_name, reflection_fields));
        self.monomorphized_structs.insert(cache_key, ty);
        ty
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn binop_str(op: BinOp) -> &'static str {
        match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Modulo => "modulo",
            BinOp::Eq => "==",
            BinOp::NotEq => "!=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::LtEq => "<=",
            BinOp::GtEq => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }
}

fn stmt_span(stmt: &Stmt) -> Span {
    match stmt {
        Stmt::VarDecl(v) => v.span,
        Stmt::Assign(a) => a.span,
        Stmt::Return(r) => r.span,
        Stmt::ComptimeTypeBind(b) => b.span,
        Stmt::If(i) => i.span,
        Stmt::For(f) => f.span,
        Stmt::While(w) => w.span,
        Stmt::Match(m) => m.span,
        Stmt::Expr(e) => e.span,
        Stmt::Use(u) => u.span,
        Stmt::Assert(a) => a.span,
        Stmt::Trace(t) => t.span,
        Stmt::Breakpoint(b) => b.span,
        Stmt::Respond(r) => r.span,
        Stmt::Break(span) | Stmt::Continue(span) => *span,
    }
}

fn comptime_type_info_binding(expr: &Expr) -> Option<&TypeExpr> {
    let Expr::GenericCall(callee, type_args, args, _) = expr else {
        return None;
    };
    if type_args.len() != 1 || !args.is_empty() || !is_type_info_callee(callee) {
        return None;
    }
    type_args.first()
}

fn comptime_type_arg_binding(expr: &Expr) -> Option<(&TypeExpr, usize)> {
    let Expr::GenericCall(callee, type_args, args, _) = expr else {
        return None;
    };
    if type_args.len() != 1 || args.len() != 1 || !is_type_arg_callee(callee) {
        return None;
    }
    let arg = args.first()?;
    if arg.name.is_some() {
        return None;
    }
    let Expr::IntLiteral(index, _) = &arg.value else {
        return None;
    };
    Some((type_args.first()?, usize::try_from(*index).ok()?))
}

fn comptime_type_fields_binding(expr: &Expr) -> Option<&TypeExpr> {
    let Expr::GenericCall(callee, type_args, args, _) = expr else {
        return None;
    };
    if type_args.len() != 1 || !args.is_empty() || !is_type_fields_callee(callee) {
        return None;
    }
    type_args.first()
}

fn comptime_type_variants_binding(expr: &Expr) -> Option<&TypeExpr> {
    let Expr::GenericCall(callee, type_args, args, _) = expr else {
        return None;
    };
    if type_args.len() != 1 || !args.is_empty() || !is_type_variants_callee(callee) {
        return None;
    }
    type_args.first()
}

fn direct_reflected_loop_owner_type(expr: &Expr) -> Option<&TypeExpr> {
    comptime_type_fields_binding(expr)
        .or_else(|| comptime_type_variants_binding(expr))
        .or_else(|| comptime_type_machine_states_binding(expr))
        .or_else(|| comptime_type_machine_fields_binding(expr))
}

fn comptime_type_variant_value_binding(expr: &Expr) -> Option<&TypeExpr> {
    let Expr::GenericCall(callee, type_args, args, _) = expr else {
        return None;
    };
    if type_args.len() != 1 || args.len() != 1 || !is_type_variant_value_callee(callee) {
        return None;
    }
    type_args.first()
}

fn comptime_type_variant_fields_binding(expr: &Expr) -> Option<&TypeExpr> {
    let Expr::FieldAccess(base, field, _) = expr else {
        return None;
    };
    if field.name != "fields" {
        return None;
    }
    comptime_type_variant_value_binding(base)
}

fn comptime_type_machine_state_value_binding(expr: &Expr) -> Option<&TypeExpr> {
    let Expr::GenericCall(callee, type_args, args, _) = expr else {
        return None;
    };
    if type_args.len() != 1 || args.len() != 1 || !is_type_machine_state_value_callee(callee) {
        return None;
    }
    type_args.first()
}

fn comptime_type_machine_states_binding(expr: &Expr) -> Option<&TypeExpr> {
    let Expr::GenericCall(callee, type_args, args, _) = expr else {
        return None;
    };
    if type_args.len() != 1 || !args.is_empty() || !is_type_machine_states_callee(callee) {
        return None;
    }
    type_args.first()
}

fn comptime_type_machine_fields_binding(expr: &Expr) -> Option<&TypeExpr> {
    let Expr::FieldAccess(base, field, _) = expr else {
        return None;
    };
    if field.name != "fields" {
        return None;
    }
    comptime_type_machine_state_value_binding(base)
}

fn reflected_machine_state_fields_binding(expr: &Expr) -> Option<&str> {
    let Expr::FieldAccess(base, field, _) = expr else {
        return None;
    };
    if field.name != "fields" {
        return None;
    }
    let Expr::Ident(ident) = base.as_ref() else {
        return None;
    };
    Some(&ident.name)
}

fn reflected_variant_fields_binding(expr: &Expr) -> Option<&str> {
    let Expr::FieldAccess(base, field, _) = expr else {
        return None;
    };
    if field.name != "fields" {
        return None;
    }
    let Expr::Ident(ident) = base.as_ref() else {
        return None;
    };
    Some(&ident.name)
}

fn reflected_field_type_info_binding(expr: &Expr) -> Option<&str> {
    let Expr::FieldAccess(base, field, _) = expr else {
        return None;
    };
    if field.name != "type_info" {
        return None;
    }
    let Expr::Ident(ident) = base.as_ref() else {
        return None;
    };
    Some(&ident.name)
}

enum ReflectedTypeInfoSource<'a> {
    Direct(&'a TypeExpr),
    Field(&'a str),
    TypeInfo(&'a str),
}

fn reflected_type_info_args_source(expr: &Expr) -> Option<ReflectedTypeInfoSource<'_>> {
    let Expr::FieldAccess(base, field, _) = expr else {
        return None;
    };
    if field.name != "args" {
        return None;
    }
    if let Some(ty) = comptime_type_info_binding(base) {
        return Some(ReflectedTypeInfoSource::Direct(ty));
    }
    if let Some(field_name) = reflected_field_type_info_binding(base) {
        return Some(ReflectedTypeInfoSource::Field(field_name));
    }
    reflected_type_info_binding(base).map(ReflectedTypeInfoSource::TypeInfo)
}

fn reflected_type_info_binding(expr: &Expr) -> Option<&str> {
    let Expr::Ident(ident) = expr else {
        return None;
    };
    Some(&ident.name)
}

fn is_type_info_callee(callee: &Expr) -> bool {
    let Expr::FieldAccess(base, field, _) = callee else {
        return false;
    };
    field.name == "info" && matches!(base.as_ref(), Expr::Ident(ident) if ident.name == "type")
}

fn is_type_arg_callee(callee: &Expr) -> bool {
    let Expr::FieldAccess(base, field, _) = callee else {
        return false;
    };
    field.name == "arg" && matches!(base.as_ref(), Expr::Ident(ident) if ident.name == "type")
}

fn is_type_fields_callee(callee: &Expr) -> bool {
    let Expr::FieldAccess(base, field, _) = callee else {
        return false;
    };
    field.name == "fields" && matches!(base.as_ref(), Expr::Ident(ident) if ident.name == "type")
}

fn is_type_variants_callee(callee: &Expr) -> bool {
    let Expr::FieldAccess(base, field, _) = callee else {
        return false;
    };
    field.name == "variants" && matches!(base.as_ref(), Expr::Ident(ident) if ident.name == "type")
}

fn is_type_variant_value_callee(callee: &Expr) -> bool {
    let Expr::FieldAccess(base, field, _) = callee else {
        return false;
    };
    field.name == "variant_value"
        && matches!(base.as_ref(), Expr::Ident(ident) if ident.name == "type")
}

fn is_type_machine_states_callee(callee: &Expr) -> bool {
    let Expr::FieldAccess(base, field, _) = callee else {
        return false;
    };
    field.name == "machine_states"
        && matches!(base.as_ref(), Expr::Ident(ident) if ident.name == "type")
}

fn is_type_machine_state_value_callee(callee: &Expr) -> bool {
    let Expr::FieldAccess(base, field, _) = callee else {
        return false;
    };
    field.name == "machine_state_value"
        && matches!(base.as_ref(), Expr::Ident(ident) if ident.name == "type")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use jett_common::{FileId, STDLIB_FILE_ID_START, Span};
    use jett_parser::{ast::*, parse};
    use jett_resolve::resolver::ResolveResult;
    use jett_resolve::scope::{DefKind, ScopeTable};

    /// Helper to create a span for tests.
    fn sp(start: u32, end: u32) -> Span {
        Span::new(FileId::new(0), start, end)
    }

    /// Helper: build a resolve result manually for testing.
    struct TestEnv {
        scope_table: ScopeTable,
        resolutions: HashMap<Span, DefId>,
    }

    impl TestEnv {
        fn new() -> Self {
            Self {
                scope_table: ScopeTable::new(),
                resolutions: HashMap::new(),
            }
        }

        fn def_var(&mut self, name: &str, span: Span) -> DefId {
            let def_id = self
                .scope_table
                .new_def(name.to_string(), DefKind::Variable, span);
            self.resolutions.insert(span, def_id);
            def_id
        }

        fn def_param(&mut self, name: &str, span: Span) -> DefId {
            let def_id = self
                .scope_table
                .new_def(name.to_string(), DefKind::Param, span);
            self.resolutions.insert(span, def_id);
            def_id
        }

        fn def_func(&mut self, name: &str, span: Span) -> DefId {
            let def_id = self
                .scope_table
                .new_def(name.to_string(), DefKind::Function, span);
            self.resolutions.insert(span, def_id);
            def_id
        }

        /// Also map an identifier reference span to a DefId.
        fn reference(&mut self, span: Span, def_id: DefId) {
            self.resolutions.insert(span, def_id);
        }

        fn into_resolve_result(self) -> ResolveResult {
            ResolveResult {
                scope_table: self.scope_table,
                resolutions: self.resolutions,
                namespace_aliases: HashMap::new(),
                diagnostics: Vec::new(),
            }
        }
    }

    fn ident(name: &str, span: Span) -> Ident {
        Ident {
            name: name.to_string(),
            span,
        }
    }

    fn check_source_result(source: &str) -> CheckResult {
        let file_id = FileId::new(0);
        check_source_result_with_file_id(source, file_id)
    }

    fn check_source_result_with_file_id(source: &str, file_id: FileId) -> CheckResult {
        let parse_result = parse(source, file_id);
        assert!(
            parse_result.errors.is_empty(),
            "unexpected parse errors: {:?}",
            parse_result.errors
        );

        let resolve_result = jett_resolve::resolve(&parse_result.module);
        let resolve_errors: Vec<_> = resolve_result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(
            resolve_errors.is_empty(),
            "unexpected resolve errors: {:?}",
            resolve_result.diagnostics
        );

        check(&parse_result.module, &resolve_result)
    }

    fn check_source_errors(source: &str) -> Vec<Diagnostic> {
        check_source_result(source)
            .diagnostics
            .into_iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect()
    }

    // ---------------------------------------------------------------
    // Test: simple function with parameters and return type
    // ---------------------------------------------------------------

    #[test]
    fn simple_function_params_and_return() {
        // function add(a: int64, b: int64) returns int64:
        //     return a + b
        let fn_name_span = sp(0, 3);
        let param_a_span = sp(4, 5);
        let param_b_span = sp(6, 7);
        let ref_a_span = sp(10, 11);
        let ref_b_span = sp(12, 13);
        let binop_span = sp(10, 13);
        let ret_span = sp(8, 13);
        let body_span = sp(8, 14);
        let func_span = sp(0, 14);

        let mut env = TestEnv::new();
        let fn_def_id = env.def_func("add", fn_name_span);
        let a_def_id = env.def_param("a", param_a_span);
        let b_def_id = env.def_param("b", param_b_span);
        env.reference(ref_a_span, a_def_id);
        env.reference(ref_b_span, b_def_id);
        // Also reference fn name for self-registration
        env.reference(fn_name_span, fn_def_id);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("add", fn_name_span),
                type_params: vec![],
                params: vec![
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("a", param_a_span),
                        ty: TypeExpr::Named(ident("int64", sp(100, 105))),
                        span: param_a_span,
                    },
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("b", param_b_span),
                        ty: TypeExpr::Named(ident("int64", sp(106, 111))),
                        span: param_b_span,
                    },
                ],
                return_type: Some(TypeExpr::Named(ident("int64", sp(112, 117)))),
                body: Block {
                    stmts: vec![Stmt::Return(ReturnStmt {
                        value: Some(Expr::Binary(
                            Box::new(Expr::Ident(ident("a", ref_a_span))),
                            BinOp::Add,
                            Box::new(Expr::Ident(ident("b", ref_b_span))),
                            binop_span,
                        )),
                        span: ret_span,
                    })],
                    span: body_span,
                },
                exported: false,
                span: func_span,
            })],
            span: sp(0, 14),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        // No errors expected.
        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);

        // The binary expression should be typed as int64.
        assert_eq!(result.type_map[&binop_span], TypeInterner::INT64);
    }

    // ---------------------------------------------------------------
    // Test: type mismatch error (int64 + string)
    // ---------------------------------------------------------------

    #[test]
    fn type_mismatch_int_plus_string() {
        // a: int64, b: string  →  a + b  should emit an error
        let fn_name_span = sp(0, 3);
        let param_a_span = sp(4, 5);
        let param_b_span = sp(6, 7);
        let ref_a_span = sp(10, 11);
        let ref_b_span = sp(12, 13);
        let binop_span = sp(10, 13);
        let body_span = sp(8, 14);
        let func_span = sp(0, 14);

        let mut env = TestEnv::new();
        let fn_def_id = env.def_func("bad", fn_name_span);
        let a_def_id = env.def_param("a", param_a_span);
        let b_def_id = env.def_param("b", param_b_span);
        env.reference(ref_a_span, a_def_id);
        env.reference(ref_b_span, b_def_id);
        env.reference(fn_name_span, fn_def_id);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("bad", fn_name_span),
                type_params: vec![],
                params: vec![
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("a", param_a_span),
                        ty: TypeExpr::Named(ident("int64", sp(100, 105))),
                        span: param_a_span,
                    },
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("b", param_b_span),
                        ty: TypeExpr::Named(ident("string", sp(106, 112))),
                        span: param_b_span,
                    },
                ],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(113, 120)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Binary(
                            Box::new(Expr::Ident(ident("a", ref_a_span))),
                            BinOp::Add,
                            Box::new(Expr::Ident(ident("b", ref_b_span))),
                            binop_span,
                        ),
                        span: binop_span,
                    })],
                    span: body_span,
                },
                exported: false,
                span: func_span,
            })],
            span: sp(0, 14),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "expected 1 error, got: {:?}", errors);
        assert_eq!(errors[0].code.code(), 301); // binary_op_mismatch
        assert!(errors[0].message.contains("int64"));
        assert!(errors[0].message.contains("string"));
    }

    // ---------------------------------------------------------------
    // Test: binary operator type checking (arithmetic, comparison, logic)
    // ---------------------------------------------------------------

    #[test]
    fn binary_operators_arithmetic_returns_same_type() {
        // 10 + 20 → int64
        let binop_span = sp(0, 5);

        let env = TestEnv::new();
        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Binary(
                            Box::new(Expr::IntLiteral(10, sp(0, 2))),
                            BinOp::Add,
                            Box::new(Expr::IntLiteral(20, sp(3, 5))),
                            binop_span,
                        ),
                        span: binop_span,
                    })],
                    span: sp(0, 5),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        assert!(
            result
                .diagnostics
                .iter()
                .all(|d| d.severity != jett_diagnostics::Severity::Error),
            "unexpected errors: {:?}",
            result.diagnostics
        );
        assert_eq!(result.type_map[&binop_span], TypeInterner::INT64);
    }

    #[test]
    fn binary_operators_comparison_returns_bool() {
        // 10 < 20 → bool
        let binop_span = sp(0, 5);

        let env = TestEnv::new();
        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Binary(
                            Box::new(Expr::IntLiteral(10, sp(0, 2))),
                            BinOp::Lt,
                            Box::new(Expr::IntLiteral(20, sp(3, 5))),
                            binop_span,
                        ),
                        span: binop_span,
                    })],
                    span: sp(0, 5),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        assert!(
            result
                .diagnostics
                .iter()
                .all(|d| d.severity != jett_diagnostics::Severity::Error),
            "unexpected errors: {:?}",
            result.diagnostics
        );
        assert_eq!(result.type_map[&binop_span], TypeInterner::BOOL);
    }

    #[test]
    fn binary_operators_logic_requires_bool() {
        // true && false → bool (ok)
        let binop_span = sp(0, 10);

        let env = TestEnv::new();
        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Binary(
                            Box::new(Expr::BoolLiteral(true, sp(0, 4))),
                            BinOp::And,
                            Box::new(Expr::BoolLiteral(false, sp(5, 10))),
                            binop_span,
                        ),
                        span: binop_span,
                    })],
                    span: sp(0, 10),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        assert!(
            result
                .diagnostics
                .iter()
                .all(|d| d.severity != jett_diagnostics::Severity::Error),
            "unexpected errors: {:?}",
            result.diagnostics
        );
        assert_eq!(result.type_map[&binop_span], TypeInterner::BOOL);
    }

    #[test]
    fn binary_operators_logic_error_on_non_bool() {
        // 42 && true → error
        let binop_span = sp(0, 10);

        let env = TestEnv::new();
        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Binary(
                            Box::new(Expr::IntLiteral(42, sp(0, 2))),
                            BinOp::And,
                            Box::new(Expr::BoolLiteral(true, sp(5, 10))),
                            binop_span,
                        ),
                        span: binop_span,
                    })],
                    span: sp(0, 10),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code.code(), 301);
    }

    // ---------------------------------------------------------------
    // Test: variable declaration type matching
    // ---------------------------------------------------------------

    #[test]
    fn var_decl_type_match_ok() {
        // int64 x = 42
        let var_name_span = sp(6, 7);
        let var_span = sp(0, 10);

        let mut env = TestEnv::new();
        env.def_var("x", var_name_span);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::VarDecl(VarDecl {
                        mutable: false,
                        ty: TypeExpr::Named(ident("int64", sp(0, 5))),
                        name: ident("x", var_name_span),
                        value: Expr::IntLiteral(42, sp(8, 10)),
                        span: var_span,
                    })],
                    span: sp(0, 10),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn var_decl_type_mismatch() {
        // string x = 42   →  error E0311
        let var_name_span = sp(7, 8);
        let var_span = sp(0, 12);

        let mut env = TestEnv::new();
        env.def_var("x", var_name_span);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::VarDecl(VarDecl {
                        mutable: false,
                        ty: TypeExpr::Named(ident("string", sp(0, 6))),
                        name: ident("x", var_name_span),
                        value: Expr::IntLiteral(42, sp(9, 11)),
                        span: var_span,
                    })],
                    span: sp(0, 12),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "expected 1 error, got: {:?}", errors);
        assert_eq!(errors[0].code.code(), 311);
        assert!(errors[0].message.contains("string"));
        assert!(errors[0].message.contains("int64"));
    }

    // ---------------------------------------------------------------
    // Test: function call argument type checking
    // ---------------------------------------------------------------

    #[test]
    fn function_call_correct_args() {
        // function add(a: int64, b: int64) returns int64
        // add(1, 2)  →  no error, result is int64
        let fn_name_span = sp(0, 3);
        let param_a_span = sp(4, 5);
        let param_b_span = sp(6, 7);
        let call_ref_span = sp(20, 23);
        let call_span = sp(20, 30);

        let mut env = TestEnv::new();
        let fn_def_id = env.def_func("add", fn_name_span);
        env.def_param("a", param_a_span);
        env.def_param("b", param_b_span);
        env.reference(call_ref_span, fn_def_id);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("add", fn_name_span),
                type_params: vec![],
                params: vec![
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("a", param_a_span),
                        ty: TypeExpr::Named(ident("int64", sp(100, 105))),
                        span: param_a_span,
                    },
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("b", param_b_span),
                        ty: TypeExpr::Named(ident("int64", sp(106, 111))),
                        span: param_b_span,
                    },
                ],
                return_type: Some(TypeExpr::Named(ident("int64", sp(112, 117)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("add", call_ref_span))),
                            vec![
                                CallArg {
                                    name: None,
                                    value: Expr::IntLiteral(1, sp(24, 25)),
                                    span: sp(24, 25),
                                },
                                CallArg {
                                    name: None,
                                    value: Expr::IntLiteral(2, sp(27, 28)),
                                    span: sp(27, 28),
                                },
                            ],
                            call_span,
                        ),
                        span: call_span,
                    })],
                    span: sp(20, 30),
                },
                exported: false,
                span: sp(0, 30),
            })],
            span: sp(0, 30),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
        assert_eq!(result.type_map[&call_span], TypeInterner::INT64);
    }

    #[test]
    fn function_call_wrong_arg_type() {
        // function add(a: int64, b: int64) returns int64
        // add(1, "hello")  →  error E0304
        let fn_name_span = sp(0, 3);
        let param_a_span = sp(4, 5);
        let param_b_span = sp(6, 7);
        let call_ref_span = sp(20, 23);
        let call_span = sp(20, 35);
        let bad_arg_span = sp(27, 34);

        let mut env = TestEnv::new();
        let fn_def_id = env.def_func("add", fn_name_span);
        env.def_param("a", param_a_span);
        env.def_param("b", param_b_span);
        env.reference(call_ref_span, fn_def_id);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("add", fn_name_span),
                type_params: vec![],
                params: vec![
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("a", param_a_span),
                        ty: TypeExpr::Named(ident("int64", sp(100, 105))),
                        span: param_a_span,
                    },
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("b", param_b_span),
                        ty: TypeExpr::Named(ident("int64", sp(106, 111))),
                        span: param_b_span,
                    },
                ],
                return_type: Some(TypeExpr::Named(ident("int64", sp(112, 117)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("add", call_ref_span))),
                            vec![
                                CallArg {
                                    name: None,
                                    value: Expr::IntLiteral(1, sp(24, 25)),
                                    span: sp(24, 25),
                                },
                                CallArg {
                                    name: None,
                                    value: Expr::StringLiteral("hello".to_string(), bad_arg_span),
                                    span: bad_arg_span,
                                },
                            ],
                            call_span,
                        ),
                        span: call_span,
                    })],
                    span: sp(20, 35),
                },
                exported: false,
                span: sp(0, 35),
            })],
            span: sp(0, 35),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "expected 1 error, got: {:?}", errors);
        assert_eq!(errors[0].code.code(), 304); // argument_type_mismatch
        assert!(errors[0].message.contains("int64"));
        assert!(errors[0].message.contains("string"));
    }

    #[test]
    fn function_call_wrong_arg_count() {
        // function add(a: int64, b: int64) returns int64
        // add(1)  →  error E0303
        let fn_name_span = sp(0, 3);
        let param_a_span = sp(4, 5);
        let param_b_span = sp(6, 7);
        let call_ref_span = sp(20, 23);
        let call_span = sp(20, 28);

        let mut env = TestEnv::new();
        let fn_def_id = env.def_func("add", fn_name_span);
        env.def_param("a", param_a_span);
        env.def_param("b", param_b_span);
        env.reference(call_ref_span, fn_def_id);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("add", fn_name_span),
                type_params: vec![],
                params: vec![
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("a", param_a_span),
                        ty: TypeExpr::Named(ident("int64", sp(100, 105))),
                        span: param_a_span,
                    },
                    Param {
                        view: false,
                        mutable: false,
                        name: ident("b", param_b_span),
                        ty: TypeExpr::Named(ident("int64", sp(106, 111))),
                        span: param_b_span,
                    },
                ],
                return_type: Some(TypeExpr::Named(ident("int64", sp(112, 117)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("add", call_ref_span))),
                            vec![CallArg {
                                name: None,
                                value: Expr::IntLiteral(1, sp(24, 25)),
                                span: sp(24, 25),
                            }],
                            call_span,
                        ),
                        span: call_span,
                    })],
                    span: sp(20, 28),
                },
                exported: false,
                span: sp(0, 28),
            })],
            span: sp(0, 28),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "expected 1 error, got: {:?}", errors);
        assert_eq!(errors[0].code.code(), 303); // argument_count_mismatch
    }

    // ---------------------------------------------------------------
    // Test: if condition must be bool
    // ---------------------------------------------------------------

    #[test]
    fn if_condition_must_be_bool() {
        // if 42:   →  error E0306
        let cond_span = sp(3, 5);
        let if_span = sp(0, 10);

        let env = TestEnv::new();
        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::If(IfStmt {
                        condition: Expr::IntLiteral(42, cond_span),
                        then_block: Block {
                            stmts: vec![],
                            span: sp(6, 10),
                        },
                        else_ifs: vec![],
                        else_block: None,
                        span: if_span,
                    })],
                    span: sp(0, 10),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "expected 1 error, got: {:?}", errors);
        assert_eq!(errors[0].code.code(), 306); // condition_not_bool
        assert!(errors[0].message.contains("int64"));
    }

    #[test]
    fn if_condition_bool_ok() {
        // if true:   →  no error
        let cond_span = sp(3, 7);

        let env = TestEnv::new();
        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::If(IfStmt {
                        condition: Expr::BoolLiteral(true, cond_span),
                        then_block: Block {
                            stmts: vec![],
                            span: sp(8, 10),
                        },
                        else_ifs: vec![],
                        else_block: None,
                        span: sp(0, 10),
                    })],
                    span: sp(0, 10),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    // ---------------------------------------------------------------
    // Test: return type checking
    // ---------------------------------------------------------------

    #[test]
    fn return_type_mismatch() {
        // function foo() returns int64:
        //     return "hello"
        // → error E0305
        let fn_name_span = sp(0, 3);
        let ret_span = sp(10, 25);

        let mut env = TestEnv::new();
        let fn_def_id = env.def_func("foo", fn_name_span);
        env.reference(fn_name_span, fn_def_id);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("foo", fn_name_span),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("int64", sp(100, 105)))),
                body: Block {
                    stmts: vec![Stmt::Return(ReturnStmt {
                        value: Some(Expr::StringLiteral("hello".to_string(), sp(17, 24))),
                        span: ret_span,
                    })],
                    span: sp(10, 25),
                },
                exported: false,
                span: sp(0, 25),
            })],
            span: sp(0, 25),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "expected 1 error, got: {:?}", errors);
        assert_eq!(errors[0].code.code(), 305); // return_type_mismatch
        assert!(errors[0].message.contains("int64"));
        assert!(errors[0].message.contains("string"));
    }

    #[test]
    fn return_type_correct() {
        // function foo() returns int64:
        //     return 42
        let fn_name_span = sp(0, 3);

        let mut env = TestEnv::new();
        let fn_def_id = env.def_func("foo", fn_name_span);
        env.reference(fn_name_span, fn_def_id);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("foo", fn_name_span),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("int64", sp(100, 105)))),
                body: Block {
                    stmts: vec![Stmt::Return(ReturnStmt {
                        value: Some(Expr::IntLiteral(42, sp(17, 19))),
                        span: sp(10, 19),
                    })],
                    span: sp(10, 19),
                },
                exported: false,
                span: sp(0, 19),
            })],
            span: sp(0, 19),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    // ---------------------------------------------------------------
    // Test: unary operators
    // ---------------------------------------------------------------

    #[test]
    fn unary_not_requires_bool() {
        // not 42  →  error
        let unary_span = sp(0, 6);

        let env = TestEnv::new();
        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::Expr(ExprStmt {
                        expr: Expr::Unary(
                            UnaryOp::Not,
                            Box::new(Expr::IntLiteral(42, sp(4, 6))),
                            unary_span,
                        ),
                        span: unary_span,
                    })],
                    span: sp(0, 6),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code.code(), 302); // unary_op_mismatch
    }

    // ---------------------------------------------------------------
    // Test: for loop iterable check
    // ---------------------------------------------------------------

    #[test]
    fn for_loop_requires_list() {
        // for x in 42:  →  error E0307
        let var_span = sp(4, 5);
        let iterable_span = sp(9, 11);
        let for_span = sp(0, 15);

        let mut env = TestEnv::new();
        env.def_var("x", var_span);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![Stmt::For(ForStmt {
                        variable: ident("x", var_span),
                        value_variable: None,
                        view: false,
                        iterable: Expr::IntLiteral(42, iterable_span),
                        body: Block {
                            stmts: vec![],
                            span: sp(12, 15),
                        },
                        span: for_span,
                    })],
                    span: sp(0, 15),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "expected 1 error, got: {:?}", errors);
        assert_eq!(errors[0].code.code(), 307);
    }

    // ---------------------------------------------------------------
    // Test: assignment type mismatch
    // ---------------------------------------------------------------

    #[test]
    fn assignment_type_mismatch() {
        // int64 x = 42
        // x = "hello"  →  error E0312
        let var_name_span = sp(6, 7);
        let var_span = sp(0, 10);
        let ref_x_span = sp(15, 16);
        let assign_span = sp(15, 26);

        let mut env = TestEnv::new();
        let x_def = env.def_var("x", var_name_span);
        env.reference(ref_x_span, x_def);

        let module = Module {
            items: vec![Item::Function(FunctionDef {
                name: ident("test", sp(50, 54)),
                type_params: vec![],
                params: vec![],
                return_type: Some(TypeExpr::Named(ident("nothing", sp(55, 62)))),
                body: Block {
                    stmts: vec![
                        Stmt::VarDecl(VarDecl {
                            mutable: true,
                            ty: TypeExpr::Named(ident("int64", sp(0, 5))),
                            name: ident("x", var_name_span),
                            value: Expr::IntLiteral(42, sp(8, 10)),
                            span: var_span,
                        }),
                        Stmt::Assign(AssignStmt {
                            target: Expr::Ident(ident("x", ref_x_span)),
                            value: Expr::StringLiteral("hello".to_string(), sp(19, 26)),
                            span: assign_span,
                        }),
                    ],
                    span: sp(0, 26),
                },
                exported: false,
                span: sp(50, 62),
            })],
            span: sp(0, 62),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "expected 1 error, got: {:?}", errors);
        assert_eq!(errors[0].code.code(), 312); // assign_type_mismatch
    }

    // ---------------------------------------------------------------
    // Test: resolve_type_expr for generic types
    // ---------------------------------------------------------------

    #[test]
    fn resolve_generic_types() {
        // Verify that type expressions like list[int64], result[string, int64] resolve correctly.
        let env = TestEnv::new();
        let resolve = env.into_resolve_result();
        let mut checker = TypeChecker::new(&resolve);

        let list_int = checker.resolve_type_expr(&TypeExpr::Generic(
            ident("list", sp(0, 4)),
            vec![TypeExpr::Named(ident("int64", sp(5, 10)))],
            sp(0, 11),
        ));
        assert_eq!(
            *checker.interner.resolve(list_int),
            Type::List(TypeInterner::INT64)
        );

        let result_type = checker.resolve_type_expr(&TypeExpr::Generic(
            ident("result", sp(0, 6)),
            vec![
                TypeExpr::Named(ident("string", sp(7, 13))),
                TypeExpr::Named(ident("int64", sp(15, 20))),
            ],
            sp(0, 21),
        ));
        assert_eq!(
            *checker.interner.resolve(result_type),
            Type::Result(TypeInterner::STRING, TypeInterner::INT64)
        );
    }

    // ---------------------------------------------------------------
    // Test: capability-based purity enforcement
    // ---------------------------------------------------------------

    /// Helper: create a minimal function definition.
    fn make_function(name: &str, name_span: Span, params: Vec<Param>, body: Block) -> FunctionDef {
        FunctionDef {
            name: ident(name, name_span),
            type_params: vec![],
            params,
            return_type: Some(TypeExpr::Named(ident("nothing", sp(200, 207)))),
            body,
            exported: false,
            span: Span::new(name_span.file, name_span.start, name_span.start + 50),
        }
    }

    #[test]
    fn pure_function_calling_pure_function_ok() {
        // function helper() returns nothing:
        //     return nothing
        // function caller() returns nothing:
        //     helper()
        let helper_name_span = sp(0, 6);
        let caller_name_span = sp(100, 106);
        let call_ref_span = sp(110, 116);
        let call_span = sp(110, 118);

        let mut env = TestEnv::new();
        let helper_def = env.def_func("helper", helper_name_span);
        let _caller_def = env.def_func("caller", caller_name_span);
        env.reference(helper_name_span, helper_def);
        env.reference(caller_name_span, _caller_def);
        env.reference(call_ref_span, helper_def);

        let module = Module {
            items: vec![
                Item::Function(make_function(
                    "helper",
                    helper_name_span,
                    vec![],
                    Block {
                        stmts: vec![],
                        span: sp(10, 20),
                    },
                )),
                Item::Function(make_function(
                    "caller",
                    caller_name_span,
                    vec![],
                    Block {
                        stmts: vec![Stmt::Expr(ExprStmt {
                            expr: Expr::Call(
                                Box::new(Expr::Ident(ident("helper", call_ref_span))),
                                vec![],
                                call_span,
                            ),
                            span: call_span,
                        })],
                        span: sp(107, 120),
                    },
                )),
            ],
            span: sp(0, 200),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn pure_function_calling_impure_function_error() {
        // function writer(view out: Stdout) returns nothing:
        //     return nothing
        // function caller() returns nothing:
        //     writer()          ← E0500
        let writer_name_span = sp(0, 6);
        let writer_param_span = sp(7, 10);
        let caller_name_span = sp(100, 106);
        let call_ref_span = sp(110, 116);
        let call_span = sp(110, 118);

        let mut env = TestEnv::new();
        let writer_def = env.def_func("writer", writer_name_span);
        let _caller_def = env.def_func("caller", caller_name_span);
        env.def_param("out", writer_param_span);
        env.reference(writer_name_span, writer_def);
        env.reference(caller_name_span, _caller_def);
        env.reference(call_ref_span, writer_def);

        let module = Module {
            items: vec![
                Item::Function(make_function(
                    "writer",
                    writer_name_span,
                    vec![Param {
                        view: true,
                        mutable: false,
                        name: ident("out", writer_param_span),
                        ty: TypeExpr::View(
                            Box::new(TypeExpr::Named(ident("Stdout", sp(11, 17)))),
                            sp(7, 17),
                        ),
                        span: writer_param_span,
                    }],
                    Block {
                        stmts: vec![],
                        span: sp(20, 30),
                    },
                )),
                Item::Function(make_function(
                    "caller",
                    caller_name_span,
                    vec![],
                    Block {
                        stmts: vec![Stmt::Expr(ExprStmt {
                            expr: Expr::Call(
                                Box::new(Expr::Ident(ident("writer", call_ref_span))),
                                vec![],
                                call_span,
                            ),
                            span: call_span,
                        })],
                        span: sp(107, 120),
                    },
                )),
            ],
            span: sp(0, 200),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        // Expect E0500 (pure calls impure) and E0303 (arg count mismatch — caller
        // passes 0 args but writer expects 1).  We only assert E0500 exists.
        let purity_errors: Vec<_> = errors.iter().filter(|d| d.code.code() == 500).collect();
        assert_eq!(
            purity_errors.len(),
            1,
            "expected 1 purity error (E0500), got: {:?}",
            purity_errors
        );
        assert!(purity_errors[0].message.contains("caller"));
        assert!(purity_errors[0].message.contains("writer"));
    }

    #[test]
    fn impure_function_calling_impure_function_ok() {
        // function writer(view out: Stdout) returns nothing:
        //     return nothing
        // function caller(view out: Stdout) returns nothing:
        //     writer()          ← ok, caller is also impure
        let writer_name_span = sp(0, 6);
        let writer_param_span = sp(7, 10);
        let caller_name_span = sp(100, 106);
        let caller_param_span = sp(107, 110);
        let call_ref_span = sp(150, 156);
        let call_span = sp(150, 158);

        let mut env = TestEnv::new();
        let writer_def = env.def_func("writer", writer_name_span);
        let _caller_def = env.def_func("caller", caller_name_span);
        env.def_param("out", writer_param_span);
        env.def_param("out2", caller_param_span);
        env.reference(writer_name_span, writer_def);
        env.reference(caller_name_span, _caller_def);
        env.reference(call_ref_span, writer_def);

        let module = Module {
            items: vec![
                Item::Function(make_function(
                    "writer",
                    writer_name_span,
                    vec![Param {
                        view: true,
                        mutable: false,
                        name: ident("out", writer_param_span),
                        ty: TypeExpr::View(
                            Box::new(TypeExpr::Named(ident("Stdout", sp(11, 17)))),
                            sp(7, 17),
                        ),
                        span: writer_param_span,
                    }],
                    Block {
                        stmts: vec![],
                        span: sp(20, 30),
                    },
                )),
                Item::Function(make_function(
                    "caller",
                    caller_name_span,
                    vec![Param {
                        view: true,
                        mutable: false,
                        name: ident("out2", caller_param_span),
                        ty: TypeExpr::View(
                            Box::new(TypeExpr::Named(ident("Stdout", sp(111, 117)))),
                            sp(107, 117),
                        ),
                        span: caller_param_span,
                    }],
                    Block {
                        stmts: vec![Stmt::Expr(ExprStmt {
                            expr: Expr::Call(
                                Box::new(Expr::Ident(ident("writer", call_ref_span))),
                                vec![],
                                call_span,
                            ),
                            span: call_span,
                        })],
                        span: sp(140, 160),
                    },
                )),
            ],
            span: sp(0, 300),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        // No purity errors expected (E0500).
        let purity_errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.code.code() == 500)
            .collect();
        assert!(
            purity_errors.is_empty(),
            "unexpected purity errors: {:?}",
            purity_errors
        );
    }

    #[test]
    fn function_with_stdout_param_is_impure() {
        // function printer(view out: Stdout) returns nothing
        // This is a unit-level check: the purity map marks this function as impure.
        let fn_name_span = sp(0, 7);
        let param_span = sp(8, 11);

        let mut env = TestEnv::new();
        let fn_def = env.def_func("printer", fn_name_span);
        env.def_param("out", param_span);
        env.reference(fn_name_span, fn_def);

        let func = make_function(
            "printer",
            fn_name_span,
            vec![Param {
                view: true,
                mutable: false,
                name: ident("out", param_span),
                ty: TypeExpr::View(
                    Box::new(TypeExpr::Named(ident("Stdout", sp(12, 18)))),
                    sp(8, 18),
                ),
                span: param_span,
            }],
            Block {
                stmts: vec![],
                span: sp(20, 30),
            },
        );

        let module = Module {
            items: vec![Item::Function(func)],
            span: sp(0, 100),
        };

        let resolve = env.into_resolve_result();
        let mut checker = TypeChecker::new(&resolve);
        checker.check_module(&module);

        // The purity map should mark "printer" as impure.
        assert_eq!(
            checker.purity_map.get("printer").copied(),
            Some(false),
            "function with Stdout param should be impure"
        );
    }

    #[test]
    fn function_without_capability_params_is_pure() {
        // function add(a: int64, b: int64) returns nothing
        let fn_name_span = sp(0, 3);
        let param_a_span = sp(4, 5);
        let param_b_span = sp(6, 7);

        let mut env = TestEnv::new();
        let fn_def = env.def_func("add", fn_name_span);
        env.def_param("a", param_a_span);
        env.def_param("b", param_b_span);
        env.reference(fn_name_span, fn_def);

        let func = make_function(
            "add",
            fn_name_span,
            vec![
                Param {
                    view: false,
                    mutable: false,
                    name: ident("a", param_a_span),
                    ty: TypeExpr::Named(ident("int64", sp(100, 105))),
                    span: param_a_span,
                },
                Param {
                    view: false,
                    mutable: false,
                    name: ident("b", param_b_span),
                    ty: TypeExpr::Named(ident("int64", sp(106, 111))),
                    span: param_b_span,
                },
            ],
            Block {
                stmts: vec![],
                span: sp(10, 20),
            },
        );

        let module = Module {
            items: vec![Item::Function(func)],
            span: sp(0, 100),
        };

        let resolve = env.into_resolve_result();
        let mut checker = TypeChecker::new(&resolve);
        checker.check_module(&module);

        assert_eq!(
            checker.purity_map.get("add").copied(),
            Some(true),
            "function without capability params should be pure"
        );
    }

    #[test]
    fn verify_block_calling_impure_function_error() {
        // function writer(view out: Stdout) returns nothing:
        //     return nothing
        // verify test_writer:
        //     writer()          ← E0501
        let writer_name_span = sp(0, 6);
        let writer_param_span = sp(7, 10);
        let call_ref_span = sp(110, 116);
        let call_span = sp(110, 118);
        let verify_span = sp(100, 130);

        let mut env = TestEnv::new();
        let writer_def = env.def_func("writer", writer_name_span);
        env.def_param("out", writer_param_span);
        env.reference(writer_name_span, writer_def);
        env.reference(call_ref_span, writer_def);

        let module = Module {
            items: vec![
                Item::Function(make_function(
                    "writer",
                    writer_name_span,
                    vec![Param {
                        view: true,
                        mutable: false,
                        name: ident("out", writer_param_span),
                        ty: TypeExpr::View(
                            Box::new(TypeExpr::Named(ident("Stdout", sp(11, 17)))),
                            sp(7, 17),
                        ),
                        span: writer_param_span,
                    }],
                    Block {
                        stmts: vec![],
                        span: sp(20, 30),
                    },
                )),
                Item::Verify(VerifyBlock {
                    name: ident("test_writer", sp(100, 111)),
                    body: Block {
                        stmts: vec![Stmt::Expr(ExprStmt {
                            expr: Expr::Call(
                                Box::new(Expr::Ident(ident("writer", call_ref_span))),
                                vec![],
                                call_span,
                            ),
                            span: call_span,
                        })],
                        span: sp(112, 130),
                    },
                    span: verify_span,
                }),
            ],
            span: sp(0, 200),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let verify_errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.code.code() == 501)
            .collect();
        assert_eq!(
            verify_errors.len(),
            1,
            "expected 1 verify purity error (E0501), got: {:?}",
            verify_errors
        );
        assert!(verify_errors[0].message.contains("test_writer"));
        assert!(verify_errors[0].message.contains("writer"));
    }

    #[test]
    fn verify_block_calling_pure_function_ok() {
        // function helper() returns nothing:
        //     return nothing
        // verify test_helper:
        //     helper()          ← ok
        let helper_name_span = sp(0, 6);
        let call_ref_span = sp(110, 116);
        let call_span = sp(110, 118);
        let verify_span = sp(100, 130);

        let mut env = TestEnv::new();
        let helper_def = env.def_func("helper", helper_name_span);
        env.reference(helper_name_span, helper_def);
        env.reference(call_ref_span, helper_def);

        let module = Module {
            items: vec![
                Item::Function(make_function(
                    "helper",
                    helper_name_span,
                    vec![],
                    Block {
                        stmts: vec![],
                        span: sp(10, 20),
                    },
                )),
                Item::Verify(VerifyBlock {
                    name: ident("test_helper", sp(100, 111)),
                    body: Block {
                        stmts: vec![Stmt::Expr(ExprStmt {
                            expr: Expr::Call(
                                Box::new(Expr::Ident(ident("helper", call_ref_span))),
                                vec![],
                                call_span,
                            ),
                            span: call_span,
                        })],
                        span: sp(112, 130),
                    },
                    span: verify_span,
                }),
            ],
            span: sp(0, 200),
        };

        let resolve = env.into_resolve_result();
        let result = check(&module, &resolve);

        let purity_errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.code.code() == 500 || d.code.code() == 501)
            .collect();
        assert!(
            purity_errors.is_empty(),
            "unexpected purity errors: {:?}",
            purity_errors
        );
    }

    #[test]
    fn mutual_block_allows_mutual_recursion() {
        let result = check_source_result(
            "\
mutual:
    function is_even(n: int64) returns bool
    function is_odd(n: int64) returns bool

function is_even(n: int64) returns bool:
    if n == 0:
        return true
    return is_odd(n - 1)

function is_odd(n: int64) returns bool:
    if n == 0:
        return false
    return is_even(n - 1)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn mutual_block_missing_definition_reports_error() {
        let errors = check_source_errors(
            "\
mutual:
    function is_even(n: int64) returns bool

function main() returns nothing:
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 325),
            "expected E0325, got: {:?}",
            errors
        );
    }

    #[test]
    fn mutual_block_signature_mismatch_reports_error() {
        let errors = check_source_errors(
            "\
mutual:
    function is_even(n: int64) returns bool

function is_even(value: string) returns bool:
    return true
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 326),
            "expected E0326, got: {:?}",
            errors
        );
    }

    #[test]
    fn user_defined_struct_constructor_and_field_access_typecheck_cleanly() {
        let result = check_source_result(
            "\
struct Point:
    x: int64
    y: int64

function sum(view point: Point) returns int64:
    return point.x + point.y

function main() returns int64:
    Point point = Point(x: 1, y: 2)
    return sum(view point)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn struct_method_call_typechecks_cleanly() {
        let result = check_source_result(
            "\
struct Point:
    x: int64
    y: int64

    function total(view self: Point) returns int64:
        return self.x + self.y

function main() returns int64:
    Point point = Point(x: 1, y: 2)
    return Point.total(view point)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn struct_constructor_missing_field_reports_error() {
        let errors = check_source_errors(
            "\
struct Point:
    x: int64
    y: int64

function main() returns Point:
    return Point(x: 1)
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 321),
            "expected E0321, got: {:?}",
            errors
        );
    }

    #[test]
    fn unknown_struct_field_reports_error() {
        let errors = check_source_errors(
            "\
struct Point:
    x: int64
    y: int64

function main(view point: Point) returns int64:
    return point.z
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 319),
            "expected E0319, got: {:?}",
            errors
        );
    }

    #[test]
    fn enum_variant_construction_and_match_typecheck_cleanly() {
        let result = check_source_result(
            "\
enum Shape:
    circle(radius: int64)
    rect(width: int64, height: int64)

function area(shape: Shape) returns int64:
    match shape:
        circle(radius):
            return radius * radius
        rect(width, height):
            return width * height

function main() returns int64:
    Shape shape = Shape.circle(3)
    return area(shape)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn non_exhaustive_enum_match_reports_error() {
        let errors = check_source_errors(
            "\
enum Color:
    red
    blue

function describe(color: Color) returns int64:
    match color:
        red:
            return 1
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 324),
            "expected E0324, got: {:?}",
            errors
        );
    }

    #[test]
    fn enum_pattern_binding_count_mismatch_reports_error() {
        let errors = check_source_errors(
            "\
enum Shape:
    rect(width: int64, height: int64)

function area(shape: Shape) returns int64:
    match shape:
        rect(width):
            return width
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 323),
            "expected E0323, got: {:?}",
            errors
        );
    }

    #[test]
    fn reflection_metadata_owner_tables_are_type_id_backed() {
        let result = check_source_result(
            "\
namespace models

export struct User:
    name: string
    score: int64

namespace packets

export enum Protocol:
    tcp = 6
    udp = 17

export bitfield Header:
    version: 4 bits
    protocol: 8 bits as Protocol
",
        );
        let metadata = result.reflection_metadata;

        let user_id = metadata
            .type_id_for_name("models.User")
            .expect("models.User should have a canonical TypeId");
        let user_fields = metadata
            .get_type_fields_for_id(user_id)
            .expect("struct fields should be keyed by owner TypeId");
        assert_eq!(user_fields[0].name, "name");
        assert_eq!(
            metadata
                .get_type_fields("models.User")
                .expect("legacy field lookup should bridge through TypeId")[1]
                .name,
            "score"
        );

        let header_id = metadata
            .type_id_for_name("packets.Header")
            .expect("packets.Header should have a canonical TypeId");
        let header = metadata
            .get_bitfield_for_id(header_id)
            .expect("bitfield metadata should be keyed by owner TypeId");
        assert_eq!(header.fields[1].name, "protocol");
        assert_eq!(
            metadata
                .get_bitfield("packets.Header")
                .expect("legacy bitfield lookup should bridge through TypeId")
                .fields[0]
                .width,
            4
        );

        let protocol_id = metadata
            .type_id_for_name("packets.Protocol")
            .expect("packets.Protocol should have a canonical TypeId");
        let variants = metadata
            .get_type_variants_for_id(protocol_id)
            .expect("enum variants should be keyed by owner TypeId");
        assert_eq!(variants[0].name, "tcp");
        assert_eq!(
            metadata
                .get_type_variants("packets.Protocol")
                .expect("legacy variant lookup should bridge through TypeId")[1]
                .discriminant,
            17
        );
    }

    #[test]
    fn reflection_metadata_generic_type_info_args_are_type_id_backed() {
        let result = check_source_result(
            "\
namespace accounts

export struct Box[T]:
    value: T

namespace audit

export struct Box[T]:
    value: T

namespace app

function use_boxes() returns string:
    use accounts as a
    use audit as au
    a.Box[int64] account = a.Box[int64](value: 1)
    au.Box[int64] audit_box = au.Box[int64](value: 2)
    TypeInfo account_info = type.info[a.Box[int64]]()
    TypeInfo audit_info = type.info[au.Box[int64]]()
    return \"{account_info.type_name}:{audit_info.type_name}:{account.value}:{audit_box.value}\"
",
        );
        let metadata = result.reflection_metadata;

        let account_info = metadata
            .get_type_info("accounts.Box[int64]")
            .expect("accounts.Box[int64] should have canonical type info");
        assert_eq!(account_info.args[0].type_name, "int64");

        let audit_info = metadata
            .get_type_info("audit.Box[int64]")
            .expect("audit.Box[int64] should have canonical type info");
        assert_eq!(audit_info.args[0].type_name, "int64");

        assert!(
            metadata.get_type_info("Box[int64]").is_none(),
            "ambiguous generic leaf metadata should not be published"
        );
    }

    #[test]
    fn resolved_namespaced_type_lookup_does_not_fall_back_to_root_leaf() {
        let type_span = sp(0, 4);
        let mut env = TestEnv::new();
        let namespaced_type =
            env.scope_table
                .new_def("models.User".to_string(), DefKind::Struct, type_span);
        env.reference(type_span, namespaced_type);
        let resolve = env.into_resolve_result();
        let mut checker = TypeChecker::new(&resolve);
        checker.register_named_type(None, "User", TypeInterner::INT64);

        assert_eq!(
            checker.resolve_named_type("User", type_span),
            TypeInterner::ERROR
        );

        let generic_span = sp(10, 14);
        let mut env = TestEnv::new();
        let namespaced_generic =
            env.scope_table
                .new_def("models.Box".to_string(), DefKind::Struct, generic_span);
        env.reference(generic_span, namespaced_generic);
        let resolve = env.into_resolve_result();
        let mut checker = TypeChecker::new(&resolve);
        checker.generic_struct_templates.insert(
            "Box".to_string(),
            ast::StructDef {
                name: ident("Box", generic_span),
                type_params: vec![ident("T", generic_span)],
                fields: Vec::new(),
                methods: Vec::new(),
                exported: false,
                span: generic_span,
            },
        );

        assert_eq!(
            checker.resolve_generic_type(
                "Box",
                &[TypeExpr::Named(ident("int64", sp(15, 20)))],
                generic_span,
            ),
            TypeInterner::ERROR
        );
    }

    #[test]
    fn raw_json_facade_signature_prefers_trusted_stdlib_declaration() {
        let result = check_source_result_with_file_id(
            "\
namespace json

export enum JsonTree:
    null

export function kind(view value: JsonTree) returns int64:
    return 1

namespace app

function main() returns int64:
    json.JsonTree tree = json.JsonTree.null
    return json.kind(view tree)
",
            FileId::new(STDLIB_FILE_ID_START),
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn raw_json_facade_uses_ordinary_source_signature_without_trusted_stdlib() {
        let errors = check_source_errors(
            "\
namespace json

export enum JsonTree:
    null

export function kind(view value: JsonTree) returns int64:
    return 1

namespace app

function main() returns string:
    json.JsonTree tree = json.JsonTree.null
    return json.kind(view tree)
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 305),
            "expected E0305, got: {:?}",
            errors
        );
    }

    #[test]
    fn result_handle_requires_error_keyword() {
        let errors = check_source_errors(
            "\
function main() returns int64:
    int64 parsed = int64.from_string(\"42\") handle:
        default 0
    return parsed
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 316),
            "expected E0316, got: {:?}",
            errors
        );
    }

    #[test]
    fn optional_handle_rejects_error_keyword() {
        let errors = check_source_errors(
            "\
function main() returns int64:
    int64 value = some(1) handle error:
        default 0
    return value
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 317),
            "expected E0317, got: {:?}",
            errors
        );
    }

    #[test]
    fn handle_block_requires_explicit_terminator() {
        let errors = check_source_errors(
            "\
function main() returns int64:
    int64 parsed = int64.from_string(\"oops\") handle error:
        int64 fallback = 0
    return parsed
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 318),
            "expected E0318, got: {:?}",
            errors
        );
    }

    #[test]
    fn handle_default_must_match_unwrapped_type() {
        let errors = check_source_errors(
            "\
function main() returns int64:
    int64 parsed = int64.from_string(\"oops\") handle error:
        default \"bad\"
    return parsed
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 300),
            "expected E0300, got: {:?}",
            errors
        );
    }

    #[test]
    fn handle_with_builtin_result_type_checks_cleanly() {
        let errors = check_source_errors(
            "\
function main() returns int64:
    int64 parsed = int64.from_string(\"42\") handle error:
        default 0
    return parsed
",
        );

        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn interface_implement_call_typechecks_cleanly() {
        let result = check_source_result(
            "\
interface Speaker:
    function speak(view self: Speaker) returns string

struct Dog:
    name: string

implement Speaker for Dog:
    function speak(view self: Dog) returns string:
        return self.name

function main() returns string:
    Dog dog = Dog(name: \"woof\")
    return Speaker.speak(view dog)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn implement_block_missing_method_reports_error() {
        let errors = check_source_errors(
            "\
interface Speaker:
    function speak(view self: Speaker) returns string
    function growl(view self: Speaker) returns string

struct Dog:
    name: string

implement Speaker for Dog:
    function speak(view self: Dog) returns string:
        return self.name
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 331),
            "expected E0331, got: {:?}",
            errors
        );
    }

    #[test]
    fn implement_block_signature_mismatch_reports_error() {
        let errors = check_source_errors(
            "\
interface Speaker:
    function speak(view self: Speaker) returns string

struct Dog:
    name: string

implement Speaker for Dog:
    function speak(self: Dog) returns int64:
        return 1
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 330),
            "expected E0330, got: {:?}",
            errors
        );
    }

    #[test]
    fn string_interpolation_accepts_user_defined_displayable_types() {
        let result = check_source_result(
            "\
interface Displayable:
    function display(view self: Displayable) returns string

struct User:
    name: string

implement Displayable for User:
    function display(view self: User) returns string:
        return self.name

function main() returns string:
    User user = User(name: \"Ada\")
    return \"user: {user}\"
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn secret_value_can_be_declassified() {
        let result = check_source_result(
            "\
function reveal(key: secret[string]) returns string:
    return declassify key

function main() returns string:
    secret[string] api_key = \"abc\"
    return reveal(api_key)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn declassify_requires_secret_type() {
        let errors = check_source_errors(
            "\
function main() returns string:
    return declassify \"abc\"
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 601),
            "expected E0601, got: {:?}",
            errors
        );
    }

    #[test]
    fn stdout_write_rejects_secret_values() {
        let errors = check_source_errors(
            "\
function main(view stdout: Stdout) returns nothing:
    secret[string] api_key = \"abc\"
    Stdout.write(view stdout, api_key)
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 600),
            "expected E0600, got: {:?}",
            errors
        );
    }

    #[test]
    fn secret_values_are_not_displayable() {
        let errors = check_source_errors(
            "\
function main() returns string:
    secret[string] api_key = \"abc\"
    return \"api key: {api_key}\"
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 332),
            "expected E0332, got: {:?}",
            errors
        );
    }

    #[test]
    fn pure_call_with_secret_argument_returns_secret() {
        let result = check_source_result(
            "\
function main() returns nothing:
    secret[string] api_key = \"abc\"
    secret[string] upper = string.upper(api_key)
    return nothing
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn pure_call_with_secret_argument_cannot_be_assigned_to_public_type() {
        let errors = check_source_errors(
            "\
function main() returns nothing:
    secret[string] api_key = \"abc\"
    string upper = string.upper(api_key)
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 311),
            "expected E0311, got: {:?}",
            errors
        );
    }

    #[test]
    fn secret_redact_returns_public_string() {
        let result = check_source_result(
            "\
function main() returns string:
    secret[string] api_key = \"abc\"
    string masked = secret.redact(api_key)
    return masked
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn secret_compare_returns_bool() {
        let result = check_source_result(
            "\
function main() returns bool:
    secret[string] stored = \"abc\"
    secret[string] computed = \"abc\"
    return secret.compare(stored, computed)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn secret_compare_requires_matching_secret_types() {
        let errors = check_source_errors(
            "\
function main() returns bool:
    secret[string] stored = \"abc\"
    secret[int64] computed = 1
    return secret.compare(stored, computed)
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 304),
            "expected E0304, got: {:?}",
            errors
        );
    }

    #[test]
    fn secret_redact_requires_secret_argument() {
        let errors = check_source_errors(
            "\
function main() returns string:
    return secret.redact(\"abc\")
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 602),
            "expected E0602, got: {:?}",
            errors
        );
    }

    #[test]
    fn field_access_on_secret_struct_stays_secret() {
        let result = check_source_result(
            "\
struct User:
    name: string

function main() returns nothing:
    secret[User] user = User(name: \"Ada\")
    secret[string] name = user.name
    return nothing
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn mixed_secret_and_public_list_becomes_secret_element_list() {
        let result = check_source_result(
            "\
function main() returns nothing:
    secret[string] api_key = \"abc\"
    list[secret[string]] items = list(\"prefix\", api_key, \"suffix\")
    return nothing
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn string_join_rejects_list_of_secret_strings() {
        let errors = check_source_errors(
            "\
function main() returns string:
    secret[string] api_key = \"abc\"
    list[secret[string]] items = list(\"prefix\", api_key, \"suffix\")
    return string.join(items, \"-\")
",
        );

        assert!(
            errors
                .iter()
                .any(|d| d.code.code() == 304 || d.code.code() == 305),
            "expected argument/return type mismatch, got: {:?}",
            errors
        );
    }

    #[test]
    fn filesystem_write_file_rejects_secret_string() {
        let errors = check_source_errors(
            "\
function main(view fs: Filesystem) returns nothing:
    secret[string] api_key = \"abc\"
    Filesystem.write_file(view fs, \"secret.txt\", api_key) handle error:
        return nothing
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 600),
            "expected E0600, got: {:?}",
            errors
        );
    }

    #[test]
    fn json_serialize_blocks_struct_with_secret_fields() {
        let errors = check_source_errors(
            "\
struct User:
    id: string
    api_key: secret[string]

function main(view user: User) returns string:
    return json.serialize[User](view user)
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 603),
            "expected E0603, got: {:?}",
            errors
        );
    }

    #[test]
    fn json_serialize_public_allows_struct_with_secret_fields() {
        let result = check_source_result(
            "\
struct User:
    id: string
    api_key: secret[string]

function main(view user: User) returns string:
    return json.serialize_public[User](view user)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn json_serialize_public_rejects_secret_wrapped_value() {
        let errors = check_source_errors(
            "\
struct User:
    id: string

function main() returns string:
    secret[User] user = User(id: \"1\")
    return json.serialize_public[User](user)
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 600),
            "expected E0600, got: {:?}",
            errors
        );
    }

    #[test]
    fn filesystem_read_file_returns_result_string() {
        let result = check_source_result(
            "\
function main(view fs: Filesystem) returns string:
    string raw = Filesystem.read_file(view fs, \"config.json\") handle error:
        default \"\"
    return raw
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn pure_function_lifts_secret_argument() {
        let result = check_source_result(
            "\
function upper(value: string) returns string:
    return string.upper(value)

function main() returns secret[string]:
    secret[string] api_key = \"abc\"
    return upper(api_key)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn impure_function_rejects_secret_argument_without_secret_param() {
        let errors = check_source_errors(
            "\
function emit(view stdout: Stdout, value: string) returns nothing:
    Stdout.write(view stdout, value)
    return nothing

function main(view stdout: Stdout) returns nothing:
    secret[string] api_key = \"abc\"
    emit(view stdout, api_key)
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 600),
            "expected E0600, got: {:?}",
            errors
        );
    }

    #[test]
    fn impure_function_accepts_explicit_secret_param() {
        let result = check_source_result(
            "\
function send_secret(view stdout: Stdout, value: secret[string]) returns nothing:
    string redacted = secret.redact(value)
    Stdout.write(view stdout, redacted)
    return nothing

function main(view stdout: Stdout) returns nothing:
    secret[string] api_key = \"abc\"
    send_secret(view stdout, api_key)
    return nothing
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn refinement_type_assignment_requires_handle_error() {
        let errors = check_source_errors(
            "\
type Port = int64 where value >= 1 && value <= 65535

function main() returns nothing:
    Port port = 8080
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 333),
            "expected E0333, got: {:?}",
            errors
        );
    }

    #[test]
    fn refinement_type_assignment_with_handle_and_coarsen_typechecks() {
        let result = check_source_result(
            "\
type Port = int64 where value >= 1 && value <= 65535

function main() returns int64:
    Port port = 8080 handle error:
        return 80
    return coarsen port
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn simple_type_alias_behaves_like_base_type() {
        let result = check_source_result(
            "\
type UserId = int64

function id(value: UserId) returns UserId:
    return value

function main() returns int64:
    UserId user_id = 42
    return id(user_id)
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn stdlib_root_json_value_alias_wins_over_legacy_primitive_type() {
        let file_id = FileId::new(STDLIB_FILE_ID_START);
        let source = "\
namespace json
export enum JsonTree:
    null = 0
export type JsonValue = JsonTree
export root type JsonValue = json.JsonTree
namespace app
function main() returns JsonValue:
    return json.JsonTree.null
";
        let parse_result = parse(source, file_id);
        assert!(
            parse_result.errors.is_empty(),
            "unexpected parse errors: {:?}",
            parse_result.errors
        );

        let resolve_result = jett_resolve::resolve(&parse_result.module);
        let resolve_errors: Vec<_> = resolve_result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(
            resolve_errors.is_empty(),
            "unexpected resolve errors: {:?}",
            resolve_result.diagnostics
        );

        let mut checker = TypeChecker::new(&resolve_result);
        checker.check_module(&parse_result.module);
        let typecheck_errors: Vec<_> = checker
            .sink
            .diagnostics()
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(
            typecheck_errors.is_empty(),
            "unexpected type errors: {:?}",
            checker.sink.diagnostics()
        );

        let span = Span::new(file_id, 0, 0);
        let bare_ty = checker.resolve_named_type("JsonValue", span);
        let tree_ty = checker.resolve_named_type("json.JsonTree", span);
        assert_eq!(bare_ty, tree_ty);

        let metadata = checker.build_reflection_metadata();
        let info = metadata
            .get_type_info("JsonValue")
            .expect("root JsonValue alias should have reflection metadata");
        assert_eq!(info.kind, "alias");
        assert_eq!(info.primitive_tag, None);
        assert_eq!(info.args.len(), 1);
        assert_eq!(info.args[0].type_name, "json.JsonTree");
        assert_eq!(info.args[0].kind, "enum");
    }

    #[test]
    fn bare_json_value_requires_stdlib_root_alias() {
        let errors = check_source_errors(
            "\
function main() returns JsonValue:
    return nothing
",
        );

        assert!(
            errors
                .iter()
                .any(|diag| diag.code.code() == 309 && diag.message.contains("JsonValue")),
            "expected JsonValue to be unknown without the stdlib root alias, got {errors:?}"
        );
    }

    #[test]
    fn coarsen_can_target_refinement_ancestors() {
        let result = check_source_result(
            "\
type NonEmpty = string where string.char_count(value) > 0
type Password = NonEmpty where string.char_count(value) > 8

function main() returns string:
    Password password = \"hunter42!\" handle error:
        return \"\"
    NonEmpty non_empty = coarsen password
    return coarsen non_empty
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn refinement_constraint_must_return_bool() {
        let errors = check_source_errors(
            "\
type Broken = int64 where 42

function main() returns nothing:
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 335),
            "expected E0335, got: {:?}",
            errors
        );
    }

    #[test]
    fn struct_constructor_with_refinement_field_requires_handle() {
        let errors = check_source_errors(
            "\
type Age = int64 where value >= 0 && value < 150

struct User:
    age: Age

function main() returns nothing:
    User user = User(age: 42)
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 316),
            "expected E0316, got: {:?}",
            errors
        );
    }

    #[test]
    fn struct_constructor_with_refinement_field_handle_typechecks() {
        let result = check_source_result(
            "\
type Age = int64 where value >= 0 && value < 150

struct User:
    age: Age

function main() returns nothing:
    User user = User(age: 42) handle error:
        return nothing
    return nothing
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn refined_return_type_accepts_base_expression() {
        let result = check_source_result(
            "\
type Percentage = float64 where value >= 0.0 && value <= 100.0

function calculate_grade(score: int64, total: int64) returns Percentage:
    float64 score_f = float64.from_int64(score)
    float64 total_f = float64.from_int64(total)
    return score_f / total_f * 100.0
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn refined_return_type_still_requires_handle_for_result_values() {
        let errors = check_source_errors(
            "\
type Port = int64 where value >= 1 && value <= 65535

function parse_port(raw: string) returns Port:
    return int64.from_string(raw)
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 316),
            "expected E0316, got: {:?}",
            errors
        );
    }

    #[test]
    fn function_call_into_refinement_parameter_reports_boundary_error() {
        let errors = check_source_errors(
            "\
type Password = string where string.char_count(value) > 8

function create_user(password: Password) returns nothing:
    return nothing

function main() returns nothing:
    create_user(\"hunter42!\")
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 333),
            "expected E0333, got: {:?}",
            errors
        );
    }

    #[test]
    fn bitfield_constructor_with_literals_typechecks() {
        let result = check_source_result(
            "\
bitfield TcpFlags:
    syn: 1 bit
    ack: 1 bit

function main() returns int64:
    TcpFlags flags = TcpFlags(syn: 0, ack: 1)
    return flags.ack
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn bitfield_constructor_with_dynamic_int_requires_handle() {
        let errors = check_source_errors(
            "\
bitfield TcpFlags:
    syn: 1 bit
    ack: 1 bit

function main(bit: int64) returns nothing:
    TcpFlags flags = TcpFlags(syn: bit, ack: 0)
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 316),
            "expected E0316, got: {:?}",
            errors
        );
    }

    #[test]
    fn bitfield_literal_out_of_range_reports_error() {
        let errors = check_source_errors(
            "\
bitfield ColorChannel:
    red: 8 bits
    green: 8 bits

function main() returns nothing:
    ColorChannel color = ColorChannel(red: 300, green: 1)
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 337),
            "expected E0337, got: {:?}",
            errors
        );
    }

    #[test]
    fn bitfield_payload_must_be_list_of_uint8() {
        let errors = check_source_errors(
            "\
bitfield Packet:
    header: 8 bits
    payload: bytes

function main() returns nothing:
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 336),
            "expected E0336, got: {:?}",
            errors
        );
    }

    #[test]
    fn bitfield_binary_roundtrip_typechecks() {
        let result = check_source_result(
            "\
bitfield network IpHeader:
    version: 4 bits
    header_length: 4 bits
    total_length: 16 bits

function main() returns int64:
    IpHeader header = IpHeader(version: 4, header_length: 5, total_length: 500)
    bytes raw = IpHeader.to_bytes(header)
    IpHeader decoded = IpHeader.from_bytes(raw) handle error:
        return 0
    return decoded.total_length
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn bitfield_from_bytes_requires_handle() {
        let errors = check_source_errors(
            "\
bitfield network IpHeader:
    version: 4 bits
    header_length: 4 bits
    total_length: 16 bits

function main() returns nothing:
    bytes raw = IpHeader.to_bytes(IpHeader(version: 4, header_length: 5, total_length: 500))
    IpHeader decoded = IpHeader.from_bytes(raw)
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 316),
            "expected E0316, got: {:?}",
            errors
        );
    }

    #[test]
    fn trace_statement_reads_variable_without_error() {
        let result = check_source_result(
            "\
function main() returns nothing:
    int64 total = 42
    trace total
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn enum_with_explicit_discriminants_typechecks() {
        let result = check_source_result(
            "\
enum IpProtocol:
    icmp = 1
    tcp = 6
    udp = 17

bitfield network IpHeader:
    protocol: 8 bits as IpProtocol

function main() returns nothing:
    IpHeader header = IpHeader(protocol: IpProtocol.tcp)
    bytes raw = IpHeader.to_bytes(header)
    IpHeader decoded = IpHeader.from_bytes(raw) handle error:
        return nothing
    IpProtocol protocol = decoded.protocol
    return nothing
",
        );

        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn duplicate_enum_discriminant_reports_error() {
        let errors = check_source_errors(
            "\
enum Bad:
    first = 1
    second = 1

function main() returns nothing:
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 339),
            "expected E0339, got: {:?}",
            errors
        );
    }

    #[test]
    fn enum_discriminant_requires_unit_variant() {
        let errors = check_source_errors(
            "\
enum Bad:
    named(value: int64) = 1

function main() returns nothing:
    return nothing
",
        );

        assert!(
            errors.iter().any(|d| d.code.code() == 338),
            "expected E0338, got: {:?}",
            errors
        );
    }
}
