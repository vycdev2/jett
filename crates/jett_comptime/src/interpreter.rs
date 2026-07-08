use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use rand::Rng;

use jett_common::{FileId, Span, is_json_raw_facade, json_public_bridge_spec};
use jett_parser::ast::{
    ActorDef, BinOp, BitfieldDef, BitfieldFieldKind, Block, CallArg, EnumDef, Expr, FunctionDef,
    Ident, ImplementBlock, InterfaceDecl, Item, MachineDef, Module, Pattern, PipelineStep,
    PipelineStepHandle, Stmt, StringPart, StructDef, TypeAlias, TypeExpr, UnaryOp,
};
use jett_types::{
    ReflectionBitfieldFieldInfo, ReflectionBitfieldInfo, ReflectionFieldInfo,
    ReflectionMachineInfo, ReflectionMachineStateInfo, ReflectionMachineTransitionInfo,
    ReflectionMetadata, ReflectionTypeInfo, ReflectionVariantInfo,
};

use crate::value::Value;

// ---------------------------------------------------------------------------
// Built-in argument checking (must be defined before call_builtin uses it)
// ---------------------------------------------------------------------------

/// Check that a built-in function received the expected number of arguments.
/// Returns `Some(Err(...))` on mismatch (suitable for early-return from
/// `call_builtin` via `if let`), or `None` if the count is correct.
fn check_args(name: &str, expected: usize, args: &[Value]) -> Option<Result<Value, String>> {
    if args.len() != expected {
        Some(Err(format!(
            "{name} expects {expected} argument(s), got {}",
            args.len()
        )))
    } else {
        None
    }
}

/// Convenience macro: invoke `check_args` and, if the count is wrong,
/// immediately return the error wrapped in `Some`.
macro_rules! require_args {
    ($name:expr, $expected:expr, $args:expr) => {
        if let Some(err) = check_args($name, $expected, $args) {
            return Some(err);
        }
    };
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// A single lexical scope mapping variable names to values.
pub type Environment = HashMap<String, Value>;

// ---------------------------------------------------------------------------
// Control-flow signals
// ---------------------------------------------------------------------------

/// Internal signal used to propagate `return` and loop control flow through
/// the recursive interpreter.
#[derive(Debug)]
enum Signal {
    Return(Value),
    Default(Value),
    Respond(Value),
    Break,
    Continue,
}

#[derive(Debug)]
enum ExprFlow {
    Value(Value),
    Signal(Signal),
}

macro_rules! value_or_signal {
    ($self:expr, $expr:expr) => {
        match $self.eval_expr_flow($expr)? {
            ExprFlow::Value(value) => value,
            ExprFlow::Signal(signal) => return Ok(ExprFlow::Signal(signal)),
        }
    };
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/// A tree-walking interpreter that evaluates Jett AST nodes at compile time.
///
/// The interpreter maintains a stack of environments (scopes) and a registry
/// of user-defined functions.  It is intentionally simple: no heap, no GC,
/// no closures — just enough to execute `verify` blocks and `comptime`
/// expressions.
/// A registered refinement type alias with its base type name and constraint.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RefinementDef {
    base_type_name: String,
    constraint: Expr,
}

#[derive(Debug, Clone)]
struct ReflectionField {
    name: String,
    ty: TypeExpr,
    serialize_name: String,
}

#[derive(Debug, Clone)]
struct ReflectedFieldBinding {
    index: usize,
    owner_type: String,
    owner_member: Option<String>,
    name: String,
    ty: TypeExpr,
}

#[derive(Debug, Clone)]
struct TypeFieldMetadata {
    index: usize,
    owner_type: String,
    owner_member: Option<String>,
    name: String,
    type_name: String,
}

#[derive(Debug, Clone)]
struct ReflectedTypeInfoBinding {
    ty: TypeExpr,
}

#[derive(Debug, Clone)]
struct ReflectedVariantBinding {
    ty: TypeExpr,
    index: usize,
    owner_type: String,
    name: String,
    discriminant: i64,
}

#[derive(Debug, Clone)]
struct ReflectedMachineStateBinding {
    ty: TypeExpr,
    index: usize,
    owner_type: String,
    name: String,
}

#[derive(Debug, Clone)]
struct ReflectedVariantFieldOwner {
    ty: TypeExpr,
    variant: Option<String>,
}

#[derive(Debug, Clone)]
struct ReflectedMachineFieldOwner {
    ty: TypeExpr,
}

#[derive(Debug, Clone)]
struct ReflectionVariant {
    name: String,
    discriminant: i64,
    fields: Vec<ReflectionField>,
}

#[derive(Debug, Clone)]
struct ReflectionBitfieldField {
    name: String,
    shape: String,
    width: i64,
    ty: TypeExpr,
    enum_ty: Option<TypeExpr>,
}

#[derive(Debug, Clone)]
struct ReflectionBitfield {
    network_order: bool,
    fields: Vec<ReflectionBitfieldField>,
}

#[derive(Debug, Clone)]
struct ReflectionMachine {
    states: Vec<ReflectionMachineState>,
    edges: Vec<ReflectionMachineTransition>,
}

#[derive(Debug, Clone)]
struct ReflectionMachineState {
    name: String,
    fields: Vec<ReflectionField>,
}

#[derive(Debug, Clone)]
struct ReflectionMachineTransition {
    source_index: usize,
    source: String,
    target_index: usize,
    target: String,
}

pub struct Interpreter {
    /// Stack of lexical scopes. The last element is the innermost scope.
    scopes: Vec<Environment>,
    /// Runtime type annotations for variables declared in the matching scope.
    variable_type_scopes: Vec<HashMap<String, TypeExpr>>,
    /// Stack of block-scoped namespace aliases introduced by `use`.
    namespace_alias_scopes: Vec<HashMap<String, String>>,
    /// User-defined functions available for calling.
    functions: HashMap<String, FunctionDef>,
    /// Function registry entries that came from compiler-shipped stdlib files.
    trusted_stdlib_functions: HashSet<String>,
    /// Registered user-defined structs available for construction and field access.
    structs: HashMap<String, StructDef>,
    /// Registered user-defined bitfields available for construction and field access.
    bitfields: HashMap<String, BitfieldDef>,
    /// Registered enums for bitfield enum annotations and runtime mapping.
    enums: HashMap<String, EnumDef>,
    /// Interface dotted name -> concrete runtime type -> concrete dotted function name.
    interface_methods: HashMap<String, HashMap<String, String>>,
    /// Registered type alias base expressions.
    type_alias_bases: HashMap<String, TypeExpr>,
    /// Registered type aliases: name -> (base_type_name, optional constraint).
    type_aliases: HashMap<String, Option<RefinementDef>>,
    /// Registered state machine definitions: name -> MachineDef.
    machines: HashMap<String, MachineDef>,
    /// Registered actor definitions: type_name -> ActorDef (AST).
    actor_defs: HashMap<String, ActorDef>,
    /// Active generic type argument substitutions for interpreted generic functions.
    type_arg_scopes: Vec<HashMap<String, TypeExpr>>,
    /// Namespace of the qualified function body currently executing.
    current_namespace: Option<String>,
    /// Trusted field metadata currently produced by direct `type.fields[T]()` loops.
    reflected_field_scopes: Vec<HashMap<String, ReflectedFieldBinding>>,
    /// Trusted TypeInfo metadata currently produced by direct reflected `args` loops.
    reflected_type_info_scopes: Vec<HashMap<String, ReflectedTypeInfoBinding>>,
    /// Trusted TypeVariant metadata currently produced by direct `type.variants[T]()` loops.
    reflected_variant_scopes: Vec<HashMap<String, ReflectedVariantBinding>>,
    /// Trusted TypeMachineState metadata currently produced by direct `type.machine_states[T]()` loops.
    reflected_machine_state_scopes: Vec<HashMap<String, ReflectedMachineStateBinding>>,
    /// Checked reflection metadata snapshot, when supplied by the driver.
    reflection_metadata: Option<Arc<ReflectionMetadata>>,
    /// Checked expression type names keyed by source span, when supplied by
    /// the driver after type checking.
    checked_expression_types: Option<Arc<HashMap<Span, String>>>,
    /// Live actor instances keyed by unique ID.
    actor_instances: HashMap<u64, ActorInstance>,
    /// Next actor instance ID.
    next_actor_id: u64,
    /// Recorded debug output lines (`trace`, `breakpoint`).
    debug_output: Vec<String>,
    /// Whether debug output should print as the program runs.
    emit_runtime_debug: bool,
    /// Optional captured stdout for driver tests.
    stdout_capture: Option<String>,
}

/// Runtime state of a spawned actor instance.
struct ActorInstance {
    /// Name of the actor type (e.g. `"Counter"`).
    type_name: String,
    /// Current values of the actor's mutable state fields.
    state: HashMap<String, Value>,
    /// Capability values passed at spawn time.
    capabilities: HashMap<String, Value>,
}

impl Interpreter {
    /// Create a new interpreter with an empty global scope.
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            variable_type_scopes: vec![HashMap::new()],
            namespace_alias_scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            trusted_stdlib_functions: HashSet::new(),
            structs: HashMap::new(),
            bitfields: HashMap::new(),
            enums: HashMap::new(),
            interface_methods: HashMap::new(),
            type_alias_bases: HashMap::new(),
            type_aliases: HashMap::new(),
            machines: HashMap::new(),
            actor_defs: HashMap::new(),
            type_arg_scopes: Vec::new(),
            current_namespace: None,
            reflected_field_scopes: Vec::new(),
            reflected_type_info_scopes: Vec::new(),
            reflected_variant_scopes: Vec::new(),
            reflected_machine_state_scopes: Vec::new(),
            reflection_metadata: None,
            checked_expression_types: None,
            actor_instances: HashMap::new(),
            next_actor_id: 0,
            debug_output: Vec::new(),
            emit_runtime_debug: false,
            stdout_capture: None,
        }
    }

    /// Create an interpreter that emits debug output during execution.
    pub fn new_runtime() -> Self {
        let mut interp = Self::new();
        interp.emit_runtime_debug = true;
        interp
    }

    /// Attach checked reflection metadata produced by the typechecker.
    pub fn set_reflection_metadata(&mut self, metadata: Arc<ReflectionMetadata>) {
        self.reflection_metadata = Some(metadata);
    }

    /// Attach checked expression type names produced by the typechecker.
    pub fn set_checked_expression_types(&mut self, types: Arc<HashMap<Span, String>>) {
        self.checked_expression_types = Some(types);
    }

    /// Drain any debug lines recorded so far.
    pub fn take_debug_output(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug_output)
    }

    /// Capture runtime stdout writes instead of printing them directly.
    pub fn enable_stdout_capture(&mut self) {
        self.stdout_capture = Some(String::new());
    }

    /// Drain captured stdout. Returns an empty string when capture is disabled.
    pub fn take_stdout_output(&mut self) -> String {
        self.stdout_capture
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    // -- Scope management ---------------------------------------------------

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.variable_type_scopes.push(HashMap::new());
        self.namespace_alias_scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.variable_type_scopes.pop();
        self.namespace_alias_scopes.pop();
    }

    fn set_variable(&mut self, name: &str, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    fn set_variable_with_type(&mut self, name: &str, value: Value, ty: TypeExpr) {
        self.set_variable(name, value);
        if let Some(scope) = self.variable_type_scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn get_variable(&self, name: &str) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        None
    }

    fn get_variable_type(&self, name: &str) -> Option<&TypeExpr> {
        for scope in self.variable_type_scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// Reassign an existing variable in the nearest enclosing scope that
    /// contains it.  Returns `Err` if the variable was never declared.
    fn assign_variable(&mut self, name: &str, value: Value) -> Result<(), String> {
        for index in (0..self.scopes.len()).rev() {
            if self.scopes[index].contains_key(name) {
                let ty = self.variable_type_scopes[index].get(name).cloned();
                let value = if let Some(ty) = ty {
                    self.normalize_value_for_type(&ty, value)?
                } else {
                    value
                };
                self.scopes[index].insert(name.to_string(), value);
                return Ok(());
            }
        }
        Err(format!("undefined variable '{name}'"))
    }

    fn emit_debug_line(&mut self, line: String) {
        self.debug_output.push(line.clone());
        if self.emit_runtime_debug {
            println!("{line}");
        }
    }

    fn write_stdout(&mut self, text: &str) {
        if let Some(capture) = self.stdout_capture.as_mut() {
            capture.push_str(text);
        } else {
            print!("{text}");
        }
    }

    fn write_stdout_line(&mut self, text: &str) {
        if let Some(capture) = self.stdout_capture.as_mut() {
            capture.push_str(text);
            capture.push('\n');
        } else {
            println!("{text}");
        }
    }

    fn trace_variable(&mut self, name: &str) -> Result<(), String> {
        let value = self
            .get_variable(name)
            .cloned()
            .ok_or_else(|| format!("undefined variable '{name}'"))?;
        let label = self.debug_binding_label(name, &value);
        self.emit_debug_line(format!("trace {label}"));
        Ok(())
    }

    fn debug_binding_label(&self, name: &str, value: &Value) -> String {
        self.get_variable_type(name)
            .map(|ty| format!("{name}: {} = {value}", type_expr_display(ty)))
            .unwrap_or_else(|| format!("{name} = {value}"))
    }

    fn function_namespace(name: &str) -> Option<String> {
        name.rsplit_once('.')
            .map(|(namespace, _)| namespace.to_string())
    }

    fn current_qualified_name(&self, name: &str) -> Option<String> {
        if name.contains('.') {
            return None;
        }
        self.current_namespace
            .as_ref()
            .map(|namespace| format!("{namespace}.{name}"))
    }

    fn registry_name<T>(&self, registry: &HashMap<String, T>, name: &str) -> Option<String> {
        self.expand_namespace_alias_name(name)
            .filter(|expanded| registry.contains_key(expanded))
            .or_else(|| {
                if name.contains('.') {
                    registry.contains_key(name).then(|| name.to_string())
                } else {
                    None
                }
            })
            .or_else(|| {
                self.current_qualified_name(name)
                    .filter(|qualified| registry.contains_key(qualified))
            })
            .or_else(|| {
                if self.current_namespace.is_none() && !name.contains('.') {
                    registry.contains_key(name).then(|| name.to_string())
                } else {
                    None
                }
            })
    }

    fn use_bound_name(path: &str, alias: Option<&Ident>) -> String {
        alias
            .map(|ident| ident.name.clone())
            .unwrap_or_else(|| path.rsplit('.').next().unwrap_or(path).to_string())
    }

    fn set_namespace_alias(&mut self, bound_name: String, target: String) {
        if let Some(scope) = self.namespace_alias_scopes.last_mut() {
            scope.insert(bound_name, target);
        }
    }

    fn expand_namespace_alias_name(&self, name: &str) -> Option<String> {
        let (prefix, suffix) = name.split_once('.')?;
        self.namespace_alias_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(prefix))
            .map(|target| format!("{target}.{suffix}"))
    }

    fn runtime_name(&self, name: &str) -> String {
        self.registry_name(&self.functions, name)
            .or_else(|| self.expand_namespace_alias_name(name))
            .unwrap_or_else(|| name.to_string())
    }

    fn hit_breakpoint(&mut self) {
        let mut bindings = BTreeMap::new();
        for (scope, type_scope) in self.scopes.iter().zip(self.variable_type_scopes.iter()) {
            for (name, value) in scope {
                bindings.insert(name.clone(), (value.clone(), type_scope.get(name).cloned()));
            }
        }

        if bindings.is_empty() {
            self.emit_debug_line("breakpoint hit".to_string());
            return;
        }

        let fields: Vec<String> = bindings
            .into_iter()
            .map(|(name, (value, ty))| {
                ty.map(|ty| format!("{name}: {} = {value}", type_expr_display(&ty)))
                    .unwrap_or_else(|| format!("{name} = {value}"))
            })
            .collect();
        self.emit_debug_line(format!("breakpoint hit: {}", fields.join(", ")));
    }

    // -- Public scope management (for property-based testing) ---------------

    /// Push a new scope (public wrapper for use by verify/property runners).
    pub fn push_scope_public(&mut self) {
        self.push_scope();
    }

    /// Pop the current scope (public wrapper for use by verify/property runners).
    pub fn pop_scope_public(&mut self) {
        self.pop_scope();
    }

    /// Set a variable in the current scope (public wrapper for use by
    /// verify/property runners).
    pub fn set_variable_public(&mut self, name: &str, value: Value) {
        self.set_variable(name, value);
    }

    // -- Public helpers -----------------------------------------------------

    /// Register a function definition so it can be called later.
    pub fn register_function(&mut self, func: &FunctionDef) {
        self.register_function_named(&func.name.name, func, func.span.file.is_stdlib());
    }

    fn register_function_named(&mut self, name: &str, func: &FunctionDef, trusted_stdlib: bool) {
        self.functions.insert(name.to_string(), func.clone());
        if trusted_stdlib {
            self.trusted_stdlib_functions.insert(name.to_string());
        } else {
            self.trusted_stdlib_functions.remove(name);
        }
    }

    /// Register a function under its canonical runtime name.
    pub fn register_function_in_namespace(&mut self, namespace: Option<&str>, func: &FunctionDef) {
        let trusted_stdlib = func.span.file.is_stdlib();
        match namespace {
            Some(namespace) => self.register_function_named(
                &format!("{namespace}.{}", func.name.name),
                func,
                trusted_stdlib,
            ),
            None => self.register_function_named(&func.name.name, func, trusted_stdlib),
        }
    }

    /// Register all runtime-visible declarations from a parsed module.
    pub fn register_module(&mut self, module: &Module) {
        let mut current_file = None;
        let mut current_namespace = None;
        for item in &module.items {
            Self::update_current_namespace(item, &mut current_file, &mut current_namespace);
            match item {
                Item::Function(func) => {
                    self.register_function_in_namespace(current_namespace.as_deref(), func)
                }
                Item::TypeAlias(alias) => {
                    if alias.root_exported {
                        self.register_type_alias(alias);
                    } else {
                        self.register_type_alias_in_namespace(current_namespace.as_deref(), alias);
                    }
                }
                Item::Interface(interface) => self.register_interface(interface),
                Item::Implement(block) => self.register_implement_block(block),
                Item::Struct(strukt) => {
                    self.register_struct_in_namespace(current_namespace.as_deref(), strukt)
                }
                Item::Enum(enm) => {
                    self.register_enum_in_namespace(current_namespace.as_deref(), enm)
                }
                Item::Bitfield(bitfield) => {
                    self.register_bitfield_in_namespace(current_namespace.as_deref(), bitfield)
                }
                Item::Machine(machine) => {
                    self.register_machine_in_namespace(current_namespace.as_deref(), machine)
                }
                Item::Actor(actor) => {
                    self.register_actor_in_namespace(current_namespace.as_deref(), actor)
                }
                _ => {}
            }
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

    /// Register a user-defined struct so it can be constructed and its methods
    /// called with dotted syntax like `Point.total(view p)`.
    pub fn register_struct(&mut self, strukt: &StructDef) {
        self.structs
            .insert(strukt.name.name.clone(), strukt.clone());

        for method in &strukt.methods {
            self.functions.insert(
                format!("{}.{}", strukt.name.name, method.name.name),
                method.clone(),
            );
        }
    }

    pub fn register_struct_in_namespace(&mut self, namespace: Option<&str>, strukt: &StructDef) {
        if let Some(namespace) = namespace {
            let mut qualified = strukt.clone();
            qualified.name.name = format!("{namespace}.{}", strukt.name.name);
            self.register_struct(&qualified);
        } else {
            self.register_struct(strukt);
        }
    }

    /// Register a user-defined actor so it can be spawned and messaged.
    pub fn register_actor(&mut self, actor: &ActorDef) {
        self.actor_defs
            .insert(actor.name.name.clone(), actor.clone());
    }

    pub fn register_actor_in_namespace(&mut self, namespace: Option<&str>, actor: &ActorDef) {
        if let Some(namespace) = namespace {
            let mut qualified = actor.clone();
            qualified.name.name = format!("{namespace}.{}", actor.name.name);
            self.register_actor(&qualified);
        } else {
            self.register_actor(actor);
        }
    }

    /// Register a user-defined bitfield so it can be constructed and its
    /// fields can be read like a struct value.
    pub fn register_bitfield(&mut self, bitfield: &BitfieldDef) {
        self.bitfields
            .insert(bitfield.name.name.clone(), bitfield.clone());
    }

    pub fn register_bitfield_in_namespace(
        &mut self,
        namespace: Option<&str>,
        bitfield: &BitfieldDef,
    ) {
        if let Some(namespace) = namespace {
            let mut qualified = bitfield.clone();
            qualified.name.name = format!("{namespace}.{}", bitfield.name.name);
            self.register_bitfield(&qualified);
        } else {
            self.register_bitfield(bitfield);
        }
    }

    /// Register an enum definition so runtime bitfield conversions can map
    /// between stored integers and named variants.
    pub fn register_enum(&mut self, enm: &EnumDef) {
        self.enums.insert(enm.name.name.clone(), enm.clone());
    }

    pub fn register_enum_in_namespace(&mut self, namespace: Option<&str>, enm: &EnumDef) {
        if let Some(namespace) = namespace {
            let mut qualified = enm.clone();
            qualified.name.name = format!("{namespace}.{}", enm.name.name);
            self.register_enum(&qualified);
        } else {
            self.register_enum(enm);
        }
    }

    /// Register an interface declaration. Interfaces carry no runtime state,
    /// but keeping the entry point makes module registration symmetric.
    pub fn register_interface(&mut self, _interface: &InterfaceDecl) {}

    /// Register an `implement Interface for Type` block so interface-qualified
    /// calls can dispatch to the concrete method body at runtime.
    pub fn register_implement_block(&mut self, block: &ImplementBlock) {
        let interface_name = block.interface_name.name.clone();
        let owner_name = type_expr_name(&block.for_type);

        for method in &block.methods {
            let concrete_name = format!("{}.{}", owner_name, method.name.name);
            let interface_method_name = format!("{}.{}", interface_name, method.name.name);

            self.functions.insert(concrete_name.clone(), method.clone());
            self.interface_methods
                .entry(interface_method_name)
                .or_default()
                .insert(owner_name.clone(), concrete_name);
        }
    }

    /// Register a state machine definition so it can be used for construction
    /// and transitions.
    pub fn register_machine(&mut self, machine: &MachineDef) {
        self.machines
            .insert(machine.name.name.clone(), machine.clone());
    }

    pub fn register_machine_in_namespace(&mut self, namespace: Option<&str>, machine: &MachineDef) {
        if let Some(namespace) = namespace {
            let mut qualified = machine.clone();
            qualified.name.name = format!("{namespace}.{}", machine.name.name);
            self.register_machine(&qualified);
        } else {
            self.register_machine(machine);
        }
    }

    /// Register a type alias so the interpreter can validate refinement
    /// constraints when values are assigned to the type.
    pub fn register_type_alias(&mut self, alias: &TypeAlias) {
        let base_ty = alias.base_type.clone();
        let base_name = type_expr_name(&base_ty);
        self.type_alias_bases
            .insert(alias.name.name.clone(), base_ty);
        let def = alias.constraint.as_ref().map(|c| RefinementDef {
            base_type_name: base_name,
            constraint: c.clone(),
        });
        self.type_aliases.insert(alias.name.name.clone(), def);
    }

    pub fn register_type_alias_in_namespace(&mut self, namespace: Option<&str>, alias: &TypeAlias) {
        if let Some(namespace) = namespace {
            let mut qualified = alias.clone();
            qualified.name.name = format!("{namespace}.{}", alias.name.name);
            self.register_type_alias(&qualified);
        } else {
            self.register_type_alias(alias);
        }
    }

    /// Check a value against a refinement type's constraint.
    /// Returns `Ok(())` if valid, or `Err(message)` if the constraint fails.
    fn check_refinement(&mut self, type_name: &str, value: &Value) -> Result<(), String> {
        let type_name = self
            .registry_name(&self.type_aliases, type_name)
            .unwrap_or_else(|| type_name.to_string());

        if let Some(base_ty) = self.type_alias_bases.get(&type_name).cloned() {
            let namespace =
                Self::type_name_namespace(&type_name).or(self.current_namespace.as_deref());
            let base_type_name =
                type_expr_name(&self.substitute_type_expr_in_namespace(&base_ty, namespace));
            self.check_refinement(&base_type_name, value)?;
        }

        let def = match self.type_aliases.get(&type_name) {
            Some(Some(def)) => def.clone(),
            Some(None) => return Ok(()), // simple alias, no constraint
            None => return Ok(()),       // not a known type alias
        };

        self.push_scope();
        self.set_variable("value", value.clone());
        let result = self.eval_expr(&def.constraint);
        self.pop_scope();

        match result {
            Ok(Value::Bool(true)) => Ok(()),
            Ok(Value::Bool(false)) => Err(format!(
                "refinement type constraint failed for '{type_name}'"
            )),
            Ok(other) => Err(format!(
                "refinement constraint for '{type_name}' must return bool, got {other}"
            )),
            Err(e) => Err(format!(
                "error evaluating refinement constraint for '{type_name}': {e}"
            )),
        }
    }

    /// Public wrapper for subsystems such as property generation that need to
    /// reuse the interpreter's registered refinement constraints.
    pub fn check_refinement_type(&mut self, type_name: &str, value: &Value) -> Result<(), String> {
        self.check_refinement(type_name, value)
    }

    fn finish_refinement_boundary(
        &mut self,
        type_name: &str,
        value: Value,
        bind_name: Option<&Ident>,
        body: &Block,
    ) -> Result<ExprFlow, String> {
        match self.check_refinement(type_name, &value) {
            Ok(()) => Ok(ExprFlow::Value(value)),
            Err(message) => self.exec_handle_block(bind_name, Some(Value::String(message)), body),
        }
    }

    fn normalize_value_for_type_name(
        &self,
        type_name: &str,
        value: Value,
    ) -> Result<Value, String> {
        match type_name {
            "uint64" => match value {
                Value::Int64(n) if n >= 0 => Ok(Value::Uint64(n as u64)),
                Value::Int64(n) => Err(format!("uint64 value cannot be negative: {n}")),
                other => Ok(other),
            },
            _ => Ok(value),
        }
    }

    fn normalize_value_for_type(&self, ty: &TypeExpr, value: Value) -> Result<Value, String> {
        self.normalize_value_for_type_name(&type_expr_name(ty), value)
    }

    fn type_name_has_refinement(&self, type_name: &str) -> bool {
        let type_name = self
            .registry_name(&self.type_aliases, type_name)
            .unwrap_or_else(|| type_name.to_string());
        match self.type_aliases.get(&type_name) {
            Some(Some(_)) => true,
            Some(None) => self.type_alias_bases.get(&type_name).is_some_and(|base| {
                let namespace =
                    Self::type_name_namespace(&type_name).or(self.current_namespace.as_deref());
                let base_name =
                    type_expr_name(&self.substitute_type_expr_in_namespace(base, namespace));
                self.type_name_has_refinement(&base_name)
            }),
            None => false,
        }
    }

    /// Return an immutable reference to the current (flat) environment.
    /// Useful for `eval_assert` which needs to inspect the environment.
    pub fn current_env(&self) -> &Environment {
        self.scopes.last().unwrap()
    }

    // -- Expression evaluation ----------------------------------------------

    /// Evaluate an expression, returning its [`Value`].
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<Value, String> {
        match self.eval_expr_flow(expr)? {
            ExprFlow::Value(value) => Ok(value),
            ExprFlow::Signal(Signal::Default(_)) => {
                Err("`default` can only be used inside a `handle` block".to_string())
            }
            ExprFlow::Signal(Signal::Return(_)) => {
                Err("`return` cannot escape expression evaluation".to_string())
            }
            ExprFlow::Signal(Signal::Break) => {
                Err("`break` cannot escape expression evaluation".to_string())
            }
            ExprFlow::Signal(Signal::Continue) => {
                Err("`continue` cannot escape expression evaluation".to_string())
            }
            ExprFlow::Signal(Signal::Respond(_)) => {
                Err("`respond` cannot escape expression evaluation".to_string())
            }
        }
    }

    fn eval_expr_flow(&mut self, expr: &Expr) -> Result<ExprFlow, String> {
        let flow = self.eval_expr_flow_inner(expr)?;
        match flow {
            ExprFlow::Value(value) => Ok(ExprFlow::Value(
                self.normalize_value_for_checked_expr(expr, value)?,
            )),
            ExprFlow::Signal(signal) => Ok(ExprFlow::Signal(signal)),
        }
    }

    fn normalize_value_for_checked_expr(&self, expr: &Expr, value: Value) -> Result<Value, String> {
        let Some(type_name) = self
            .checked_expression_types
            .as_ref()
            .and_then(|types| types.get(&expr.span()))
        else {
            return Ok(value);
        };
        self.normalize_value_for_type_name(type_name, value)
    }

    fn eval_expr_flow_inner(&mut self, expr: &Expr) -> Result<ExprFlow, String> {
        match expr {
            // Literals
            Expr::IntLiteral(n, _) => {
                if (i64::MIN as i128..=i64::MAX as i128).contains(n) {
                    Ok(ExprFlow::Value(Value::Int64(*n as i64)))
                } else if (0..=u64::MAX as i128).contains(n) {
                    Ok(ExprFlow::Value(Value::Uint64(*n as u64)))
                } else {
                    Err(format!("integer literal '{n}' is out of runtime range"))
                }
            }
            Expr::FloatLiteral(n, _) => Ok(ExprFlow::Value(Value::Float64(*n))),
            Expr::StringLiteral(s, _) => Ok(ExprFlow::Value(Value::String(s.clone()))),
            Expr::StringInterpolation(parts, _) => {
                let mut result = String::new();
                for part in parts {
                    match part {
                        StringPart::Literal(s) => result.push_str(s),
                        StringPart::Expr(expr) => {
                            let val = value_or_signal!(self, expr);
                            result.push_str(&val.to_string());
                        }
                    }
                }
                Ok(ExprFlow::Value(Value::String(result)))
            }
            Expr::BoolLiteral(b, _) => Ok(ExprFlow::Value(Value::Bool(*b))),
            Expr::Nothing(_) => Ok(ExprFlow::Value(Value::Nothing)),
            Expr::Ok(inner, _) => {
                let value = value_or_signal!(self, inner);
                Ok(ExprFlow::Value(Value::ResultOk(Box::new(value))))
            }
            Expr::Fail(inner, _) => {
                let value = value_or_signal!(self, inner);
                Ok(ExprFlow::Value(Value::ResultFail(Box::new(value))))
            }
            Expr::Some(inner, _) => {
                let value = value_or_signal!(self, inner);
                Ok(ExprFlow::Value(Value::OptionalSome(Box::new(value))))
            }
            Expr::None(_) => Ok(ExprFlow::Value(Value::OptionalNone)),
            Expr::Default(inner, _) => {
                let value = value_or_signal!(self, inner);
                Ok(ExprFlow::Signal(Signal::Default(value)))
            }

            // Variables
            Expr::Ident(ident) => {
                if let Some(val) = self.get_variable(&ident.name).cloned() {
                    Ok(ExprFlow::Value(val))
                } else if let Some(func_name) = self.registry_name(&self.functions, &ident.name) {
                    let func = self
                        .functions
                        .get(&func_name)
                        .expect("registry lookup returned an existing function")
                        .clone();
                    // Named function reference — wrap as a function value.
                    Ok(ExprFlow::Value(Value::Function {
                        params: func.params.clone(),
                        body: func.body.clone(),
                        captures: HashMap::new(),
                    }))
                } else {
                    Err(format!("undefined variable '{}'", ident.name))
                }
            }

            // Parenthesized
            Expr::Paren(inner, _) => self.eval_expr_flow(inner),
            Expr::View(inner, _) => self.eval_expr_flow(inner),
            Expr::Declassify(inner, _) => self.eval_expr_flow(inner),

            // Binary operations
            Expr::Binary(lhs, op, rhs, _) => {
                let left = value_or_signal!(self, lhs);
                // Short-circuit for logical operators
                match op {
                    BinOp::And => {
                        if let Value::Bool(false) = left {
                            return Ok(ExprFlow::Value(Value::Bool(false)));
                        }
                        let right = value_or_signal!(self, rhs);
                        return Ok(ExprFlow::Value(eval_binary_op(&left, *op, &right)?));
                    }
                    BinOp::Or => {
                        if let Value::Bool(true) = left {
                            return Ok(ExprFlow::Value(Value::Bool(true)));
                        }
                        let right = value_or_signal!(self, rhs);
                        return Ok(ExprFlow::Value(eval_binary_op(&left, *op, &right)?));
                    }
                    _ => {}
                }
                let right = value_or_signal!(self, rhs);
                Ok(ExprFlow::Value(eval_binary_op(&left, *op, &right)?))
            }

            // Unary operations
            Expr::Unary(op, operand, _) => match op {
                UnaryOp::Not => {
                    let val = value_or_signal!(self, operand);
                    match val {
                        Value::Bool(b) => Ok(ExprFlow::Value(Value::Bool(!b))),
                        _ => Err("'not' requires a boolean operand".to_string()),
                    }
                }
                UnaryOp::Neg => self.eval_negation_flow(operand),
            },

            // Function / method calls
            Expr::Call(callee, args, _) => self.eval_call_flow(callee, &[], args),
            Expr::GenericCall(callee, type_args, args, _) => {
                self.eval_call_flow(callee, type_args, args)
            }

            // List construction
            Expr::ListConstruct(elems, _) => {
                let mut vals = Vec::with_capacity(elems.len());
                for elem in elems {
                    vals.push(value_or_signal!(self, elem));
                }
                Ok(ExprFlow::Value(Value::List(vals)))
            }

            Expr::MapConstruct(entries, _) => {
                let mut pairs = Vec::with_capacity(entries.len());
                for (key_expr, val_expr) in entries {
                    let k = value_or_signal!(self, key_expr);
                    let v = value_or_signal!(self, val_expr);
                    pairs.push((k, v));
                }
                Ok(ExprFlow::Value(Value::Map(pairs)))
            }

            Expr::Handle(target, bind_name, body, _) => {
                let target_value = value_or_signal!(self, target);
                match target_value {
                    Value::ResultOk(value) => Ok(ExprFlow::Value(*value)),
                    Value::ResultFail(error) => {
                        self.exec_handle_block(bind_name.as_ref(), Some(*error), body)
                    }
                    Value::OptionalSome(value) => Ok(ExprFlow::Value(*value)),
                    Value::OptionalNone => self.exec_handle_block(None, None, body),
                    other => Err(format!(
                        "handle block requires a result or optional value, got {other}"
                    )),
                }
            }

            // Enum variant reference: `Color.red`
            Expr::EnumVariant(type_name, variant, _) => Ok(ExprFlow::Value(Value::Enum {
                type_name: self
                    .registry_name(&self.enums, &type_name.name)
                    .unwrap_or_else(|| type_name.name.clone()),
                variant: variant.name.clone(),
                fields: vec![],
            })),

            // Field access: struct field access, or enum variant like `Color.red`
            Expr::FieldAccess(obj, field, _) => {
                if let Some(owner_name) = Self::dotted_expr_name(obj)
                    && let Some(enum_name) = self.registry_name(&self.enums, &owner_name)
                {
                    return Ok(ExprFlow::Value(Value::Enum {
                        type_name: enum_name,
                        variant: field.name.clone(),
                        fields: vec![],
                    }));
                }
                match obj.as_ref() {
                    Expr::Ident(ident) => {
                        if let Some(value) = self.get_variable(&ident.name).cloned() {
                            self.eval_value_field_access(value, &field.name)
                        } else {
                            // Treat as enum variant: Type.variant
                            if let Some(enum_name) = self.registry_name(&self.enums, &ident.name) {
                                Ok(ExprFlow::Value(Value::Enum {
                                    type_name: enum_name,
                                    variant: field.name.clone(),
                                    fields: vec![],
                                }))
                            } else {
                                Ok(ExprFlow::Value(Value::Enum {
                                    type_name: ident.name.clone(),
                                    variant: field.name.clone(),
                                    fields: vec![],
                                }))
                            }
                        }
                    }
                    _ => {
                        let base = value_or_signal!(self, obj);
                        self.eval_value_field_access(base, &field.name)
                    }
                }
            }

            // Coarsen: strip refinement type, returning the underlying value.
            // In the interpreter, the value is already the base type at
            // runtime, so coarsen is a no-op.
            Expr::Coarsen(inner, _) => self.eval_expr_flow(inner),

            // Pipeline: `expr into f into g(extra)`
            // Evaluate the initial expression, then for each step, call the
            // function with the accumulated value as the first argument plus
            // any extra args.
            Expr::Pipeline(initial, steps, _) => {
                let mut value = value_or_signal!(self, initial);
                for step in steps {
                    value = match self.eval_pipeline_step(step, value)? {
                        ExprFlow::Value(next) => next,
                        ExprFlow::Signal(signal) => return Ok(ExprFlow::Signal(signal)),
                    };
                }
                Ok(ExprFlow::Value(value))
            }

            // State check: `expr at state_name`
            Expr::At(expr, state_name, _) => {
                let val = value_or_signal!(self, expr);
                match val {
                    Value::Machine { state, .. } => {
                        Ok(ExprFlow::Value(Value::Bool(state == state_name.name)))
                    }
                    _ => Err(format!("'at' requires a machine value, got {val}")),
                }
            }

            Expr::Clone(inner, _) => {
                // `clone expr` — deep clone (all Values are Clone so this is a no-op copy).
                self.eval_expr_flow(inner)
            }

            Expr::Spawn(inner, _) => {
                // `spawn ActorType(cap1: val1, ...)` — create a new actor instance.
                let (source_actor_name, args) = match inner.as_ref() {
                    Expr::Call(callee, args, _) => match callee.as_ref() {
                        Expr::Ident(ident) => (ident.name.clone(), args),
                        Expr::FieldAccess(_, _, _) => {
                            let name = Self::dotted_expr_name(callee)
                                .ok_or_else(|| "spawn: expected actor type name".to_string())?;
                            (name, args)
                        }
                        _ => return Err("spawn: expected actor type name".to_string()),
                    },
                    _ => return Err("spawn: expected call expression".to_string()),
                };
                let actor_name = self
                    .registry_name(&self.actor_defs, &source_actor_name)
                    .unwrap_or(source_actor_name);

                let actor_def = self
                    .actor_defs
                    .get(&actor_name)
                    .ok_or_else(|| format!("unknown actor type '{actor_name}'"))?
                    .clone();

                // Evaluate capability args.
                let mut capabilities = HashMap::new();
                for (arg, param) in args.iter().zip(actor_def.capability_params.iter()) {
                    let val = value_or_signal!(self, &arg.value);
                    let param_ty = self.substitute_type_expr(&param.ty);
                    let val = self.normalize_value_for_type(&param_ty, val)?;
                    let name = arg
                        .name
                        .as_ref()
                        .map(|n| n.name.clone())
                        .unwrap_or_else(|| param.name.name.clone());
                    capabilities.insert(name, val);
                }

                // Evaluate state field initializers in a temp scope with capabilities in scope.
                self.push_scope();
                for (name, val) in &capabilities {
                    if let Some(param) = actor_def
                        .capability_params
                        .iter()
                        .find(|param| param.name.name == *name)
                    {
                        let param_ty = self.substitute_type_expr(&param.ty);
                        self.set_variable_with_type(name, val.clone(), param_ty);
                    } else {
                        self.set_variable(name, val.clone());
                    }
                }
                let mut state = HashMap::new();
                for field in &actor_def.state_fields {
                    let val = value_or_signal!(self, &field.value);
                    let field_ty = self.substitute_type_expr(&field.ty);
                    let val = self.normalize_value_for_type(&field_ty, val)?;
                    state.insert(field.name.name.clone(), val);
                }
                self.pop_scope();

                let id = self.next_actor_id;
                self.next_actor_id += 1;
                self.actor_instances.insert(
                    id,
                    ActorInstance {
                        type_name: actor_name.clone(),
                        state,
                        capabilities,
                    },
                );

                Ok(ExprFlow::Value(Value::Actor(id)))
            }

            Expr::Send(inner, _) => {
                self.eval_actor_message(inner, false)?;
                Ok(ExprFlow::Value(Value::Nothing))
            }

            Expr::Ask(inner, _) => {
                let val = self.eval_actor_message(inner, true)?;
                Ok(ExprFlow::Value(val))
            }

            // Structured concurrency — sequential simulation:
            // `run call` evaluates immediately and wraps in Pending.
            Expr::Run(inner, _) => {
                let val = value_or_signal!(self, inner);
                Ok(ExprFlow::Value(Value::Pending(Box::new(val))))
            }

            // `join pending` unwraps the Pending, returning result[T, error]
            // so that a `handle error:` block can handle failures.
            Expr::Join(inner, _) => {
                let val = value_or_signal!(self, inner);
                let result = match val {
                    Value::Pending(inner_val) => match *inner_val {
                        Value::ResultOk(_) | Value::ResultFail(_) => *inner_val,
                        other => Value::ResultOk(Box::new(other)),
                    },
                    Value::Nothing => {
                        Value::ResultFail(Box::new(Value::String("task was cancelled".to_string())))
                    }
                    other => Value::ResultOk(Box::new(other)),
                };
                Ok(ExprFlow::Value(result))
            }

            // `cancel task` — in the sequential simulation this is a no-op;
            // the task has already completed.
            Expr::Cancel(inner, _) => {
                value_or_signal!(self, inner);
                Ok(ExprFlow::Value(Value::Nothing))
            }

            Expr::InlineFn(params, _return_type, body, _) => {
                // Capture the current environment (all visible variables) for closure semantics.
                let mut captures = HashMap::new();
                for scope in &self.scopes {
                    for (name, value) in scope {
                        captures.insert(name.clone(), value.clone());
                    }
                }
                Ok(ExprFlow::Value(Value::Function {
                    params: params.clone(),
                    body: body.clone(),
                    captures,
                }))
            }

            // Unsupported expressions produce a clear error.
            _ => Err(format!(
                "unsupported expression in comptime: {:?}",
                std::mem::discriminant(expr)
            )),
        }
    }

    fn eval_negation_flow(&mut self, operand: &Expr) -> Result<ExprFlow, String> {
        if let Expr::IntLiteral(value, _) = operand {
            let negated = value
                .checked_neg()
                .ok_or_else(|| format!("integer literal '-{value}' is out of runtime range"))?;
            if (i64::MIN as i128..=i64::MAX as i128).contains(&negated) {
                return Ok(ExprFlow::Value(Value::Int64(negated as i64)));
            }
            return Err(format!(
                "integer literal '-{value}' is out of runtime range"
            ));
        }

        let val = value_or_signal!(self, operand);
        match val {
            Value::Int64(n) => n
                .checked_neg()
                .map(|value| ExprFlow::Value(Value::Int64(value)))
                .ok_or_else(|| "integer negation overflow".to_string()),
            Value::Float64(n) => Ok(ExprFlow::Value(Value::Float64(-n))),
            _ => Err("unary '-' requires a numeric operand".to_string()),
        }
    }

    /// Evaluate a single pipeline step: call the step's function with the
    /// accumulated `piped_value` as the first argument, followed by any
    /// extra arguments.
    fn eval_call_flow(
        &mut self,
        callee: &Expr,
        type_args: &[TypeExpr],
        args: &[CallArg],
    ) -> Result<ExprFlow, String> {
        // Check for machine construction/transition BEFORE evaluating args,
        // since state-name arguments are bare identifiers (not variables) and
        // would fail evaluation.
        match callee {
            Expr::Ident(ident) if self.registry_name(&self.structs, &ident.name).is_some() => {}
            Expr::Ident(ident) => {
                if let Some(machine_name) = self.registry_name(&self.machines, &ident.name) {
                    return Ok(ExprFlow::Value(
                        self.construct_machine(&machine_name, args)?,
                    ));
                }
            }
            Expr::FieldAccess(obj, field, _) => {
                if let Some(name) = Self::extract_dotted_name(obj, &field.name)
                    && let Some(machine_name) = self.registry_name(&self.machines, &name)
                {
                    return Ok(ExprFlow::Value(
                        self.construct_machine(&machine_name, args)?,
                    ));
                }
                if let Some(owner_name) = Self::dotted_expr_name(obj)
                    && field.name == "transition"
                    && let Some(machine_name) = self.registry_name(&self.machines, &owner_name)
                {
                    return Ok(ExprFlow::Value(
                        self.machine_transition(&machine_name, args)?,
                    ));
                }
            }
            _ => {}
        }

        let mut arg_values = Vec::with_capacity(args.len());
        for arg in args {
            arg_values.push(value_or_signal!(self, &arg.value));
        }

        match callee {
            Expr::Ident(ident) => {
                if let Some(name) = self.registry_name(&self.structs, &ident.name) {
                    return Ok(ExprFlow::Value(
                        self.construct_struct(&name, args, arg_values)?,
                    ));
                }
                if let Some(name) = self.registry_name(&self.bitfields, &ident.name) {
                    return Ok(ExprFlow::Value(
                        self.construct_bitfield(&name, args, arg_values)?,
                    ));
                }
                let name = self
                    .registry_name(&self.functions, &ident.name)
                    .unwrap_or_else(|| ident.name.clone());
                Ok(ExprFlow::Value(self.call_function_with_type_args(
                    &name, type_args, arg_values,
                )?))
            }
            // Handle enum variant construction: Type.variant(args)
            Expr::EnumVariant(type_name, variant, _) => {
                let enum_name = self
                    .registry_name(&self.enums, &type_name.name)
                    .unwrap_or_else(|| type_name.name.clone());
                Ok(ExprFlow::Value(self.construct_enum_variant(
                    &enum_name,
                    &variant.name,
                    arg_values,
                )?))
            }
            // Handle dotted names: string.trim(...), Stdout.write(...), etc.
            // Also handles enum variant construction: Shape.circle(5.0)
            Expr::FieldAccess(obj, field, _) => {
                let dotted = Self::extract_dotted_name(obj, &field.name);
                if let Some(ref name) = dotted {
                    if let Some(struct_name) = self.registry_name(&self.structs, name) {
                        return Ok(ExprFlow::Value(self.construct_struct(
                            &struct_name,
                            args,
                            arg_values,
                        )?));
                    }
                    if let Some(bitfield_name) = self.registry_name(&self.bitfields, name) {
                        return Ok(ExprFlow::Value(self.construct_bitfield(
                            &bitfield_name,
                            args,
                            arg_values,
                        )?));
                    }
                    // Try higher-order built-ins first (require &mut self).
                    if let Some(result) = self.call_higher_order_builtin(name, arg_values.clone()) {
                        return Ok(ExprFlow::Value(result?));
                    }
                    let runtime_name = self.runtime_name(name);
                    if self.is_trusted_stdlib_first_function(&runtime_name) {
                        return Ok(ExprFlow::Value(self.call_user_function_with_type_args(
                            &runtime_name,
                            type_args,
                            arg_values,
                        )?));
                    }
                    // Try type-reflection built-ins before ordinary built-ins.
                    if let Some(result) =
                        self.call_builtin_with_type_args(name, type_args, &arg_values)
                    {
                        return Ok(ExprFlow::Value(result?));
                    }
                    if let Some(result) = self.call_builtin(name, &arg_values) {
                        return Ok(ExprFlow::Value(result?));
                    }
                    // Try user-defined dotted functions.
                    if self.functions.contains_key(runtime_name.as_str())
                        || self
                            .resolve_interface_dispatch(&runtime_name, &arg_values)
                            .is_some()
                    {
                        return Ok(ExprFlow::Value(self.call_function_with_type_args(
                            &runtime_name,
                            type_args,
                            arg_values,
                        )?));
                    }
                }
                // Fall through to enum variant construction if no built-in or
                // user function matched.
                if let Some(owner_name) = Self::dotted_expr_name(obj)
                    && let Some(enum_name) = self.registry_name(&self.enums, &owner_name)
                {
                    return Ok(ExprFlow::Value(self.construct_enum_variant(
                        &enum_name,
                        &field.name,
                        arg_values,
                    )?));
                }
                if let Expr::Ident(ident) = obj.as_ref()
                    && let Some(enum_name) = self.registry_name(&self.enums, &ident.name)
                {
                    return Ok(ExprFlow::Value(self.construct_enum_variant(
                        &enum_name,
                        &field.name,
                        arg_values,
                    )?));
                }
                match dotted {
                    Some(name) => Ok(ExprFlow::Value(
                        self.call_function_with_type_args(&name, type_args, arg_values)?,
                    )),
                    None => Err("only named function calls are supported in comptime".to_string()),
                }
            }
            _ => Err("only named function calls are supported in comptime".to_string()),
        }
    }

    fn construct_enum_variant(
        &self,
        enum_name: &str,
        variant_name: &str,
        arg_values: Vec<Value>,
    ) -> Result<Value, String> {
        let Some(enm) = self.enums.get(enum_name) else {
            return Ok(Value::Enum {
                type_name: enum_name.to_string(),
                variant: variant_name.to_string(),
                fields: arg_values,
            });
        };
        let variant = enm
            .variants
            .iter()
            .find(|candidate| candidate.name.name == variant_name)
            .ok_or_else(|| format!("enum '{enum_name}' has no variant '{variant_name}'"))?;

        if arg_values.len() != variant.fields.len() {
            return Err(format!(
                "enum variant '{}.{}' expects {} field argument(s), got {}",
                enum_name,
                variant_name,
                variant.fields.len(),
                arg_values.len()
            ));
        }

        let mut fields = Vec::with_capacity(variant.fields.len());
        for (field, value) in variant.fields.iter().zip(arg_values) {
            let field_ty = self.substitute_type_expr(&field.ty);
            fields.push(self.normalize_value_for_type(&field_ty, value)?);
        }

        Ok(Value::Enum {
            type_name: enum_name.to_string(),
            variant: variant_name.to_string(),
            fields,
        })
    }

    fn eval_pipeline_step(
        &mut self,
        step: &PipelineStep,
        piped_value: Value,
    ) -> Result<ExprFlow, String> {
        let flow = self.eval_pipeline_step_call(step, piped_value)?;
        let Some(handle) = &step.handle else {
            return Ok(flow);
        };
        let value = match flow {
            ExprFlow::Value(value) => value,
            ExprFlow::Signal(signal) => return Ok(ExprFlow::Signal(signal)),
        };
        self.eval_pipeline_step_handle(value, handle)
    }

    fn eval_pipeline_step_handle(
        &mut self,
        step_value: Value,
        handle: &PipelineStepHandle,
    ) -> Result<ExprFlow, String> {
        match step_value {
            Value::ResultOk(value) => Ok(ExprFlow::Value(*value)),
            Value::ResultFail(error) => {
                self.exec_handle_block(handle.error_name.as_ref(), Some(*error), &handle.body)
            }
            Value::OptionalSome(value) => Ok(ExprFlow::Value(*value)),
            Value::OptionalNone => self.exec_handle_block(None, None, &handle.body),
            other => Err(format!(
                "handle block requires a result or optional value, got {other}"
            )),
        }
    }

    fn eval_pipeline_step_call(
        &mut self,
        step: &PipelineStep,
        piped_value: Value,
    ) -> Result<ExprFlow, String> {
        let function = match &step.function {
            Expr::View(inner, _) => inner.as_ref(),
            _ => &step.function,
        };
        let (function, type_args, extra_args): (&Expr, &[TypeExpr], &[CallArg]) = match function {
            Expr::GenericCall(callee, type_args, args, _) => (callee, type_args, args),
            _ => (function, &[], &step.extra_args),
        };

        // Build argument list: piped value first, then extra args.
        let mut arg_values = vec![piped_value];
        for arg in extra_args {
            arg_values.push(value_or_signal!(self, &arg.value));
        }

        // Resolve the function name from the expression.
        match function {
            Expr::Ident(ident) => {
                let name = self
                    .registry_name(&self.functions, &ident.name)
                    .unwrap_or_else(|| ident.name.clone());
                Ok(ExprFlow::Value(self.call_function_with_type_args(
                    &name, type_args, arg_values,
                )?))
            }
            Expr::FieldAccess(obj, field, _) => {
                let dotted = Self::extract_dotted_name(obj, &field.name);
                if let Some(ref name) = dotted {
                    // Check higher-order builtins first (need &mut self).
                    if let Some(result) = self.call_higher_order_builtin(name, arg_values.clone()) {
                        return Ok(ExprFlow::Value(result?));
                    }
                    let runtime_name = self.runtime_name(name);
                    if self.is_trusted_stdlib_first_function(&runtime_name) {
                        return Ok(ExprFlow::Value(self.call_user_function_with_type_args(
                            &runtime_name,
                            type_args,
                            arg_values,
                        )?));
                    }
                    if let Some(result) =
                        self.call_builtin_with_type_args(name, type_args, &arg_values)
                    {
                        return Ok(ExprFlow::Value(result?));
                    }
                    if let Some(result) = self.call_builtin(name, &arg_values) {
                        return Ok(ExprFlow::Value(result?));
                    }
                    if self.functions.contains_key(runtime_name.as_str()) {
                        return Ok(ExprFlow::Value(self.call_function_with_type_args(
                            &runtime_name,
                            type_args,
                            arg_values,
                        )?));
                    }
                }
                match dotted {
                    Some(name) => {
                        let runtime_name = self.runtime_name(&name);
                        Ok(ExprFlow::Value(self.call_function_with_type_args(
                            &runtime_name,
                            type_args,
                            arg_values,
                        )?))
                    }
                    None => {
                        Err("only named function calls are supported in pipeline steps".to_string())
                    }
                }
            }
            _ => Err("only named function calls are supported in pipeline steps".to_string()),
        }
    }

    fn exec_handle_block(
        &mut self,
        bind_name: Option<&Ident>,
        bind_value: Option<Value>,
        body: &Block,
    ) -> Result<ExprFlow, String> {
        self.push_scope();
        if let (Some(name), Some(value)) = (bind_name, bind_value) {
            self.set_variable(&name.name, value);
        }

        let mut signal = None;
        for stmt in &body.stmts {
            if let Some(next) = self.exec_stmt_inner(stmt)? {
                signal = Some(next);
                break;
            }
        }
        self.pop_scope();

        match signal {
            Some(Signal::Default(value)) => Ok(ExprFlow::Value(value)),
            Some(other) => Ok(ExprFlow::Signal(other)),
            None => Err("handle block must end with return or default".to_string()),
        }
    }

    // -- Statement execution ------------------------------------------------

    /// Execute a single statement.  Returns `Ok(None)` for normal flow, or
    /// a [`Signal`] if control flow must be altered.
    fn exec_stmt_inner(&mut self, stmt: &Stmt) -> Result<Option<Signal>, String> {
        match stmt {
            Stmt::VarDecl(decl) => {
                let declared_ty = self.substitute_type_expr(&decl.ty);
                let type_name = type_expr_name(&declared_ty);
                let val = if self.type_aliases.contains_key(&type_name) {
                    match &decl.value {
                        Expr::Handle(target, bind_name, body, _) => {
                            let target_value = match self.eval_expr_flow(target)? {
                                ExprFlow::Value(value) => value,
                                ExprFlow::Signal(signal) => return Ok(Some(signal)),
                            };
                            let flow = match target_value {
                                Value::ResultOk(value) | Value::OptionalSome(value) => self
                                    .finish_refinement_boundary(
                                        &type_name,
                                        *value,
                                        bind_name.as_ref(),
                                        body,
                                    )?,
                                Value::ResultFail(error) => {
                                    self.exec_handle_block(bind_name.as_ref(), Some(*error), body)?
                                }
                                Value::OptionalNone => self.exec_handle_block(None, None, body)?,
                                value => self.finish_refinement_boundary(
                                    &type_name,
                                    value,
                                    bind_name.as_ref(),
                                    body,
                                )?,
                            };
                            match flow {
                                ExprFlow::Value(value) => value,
                                ExprFlow::Signal(signal) => return Ok(Some(signal)),
                            }
                        }
                        _ => {
                            let val = match self.eval_expr_flow(&decl.value)? {
                                ExprFlow::Value(value) => value,
                                ExprFlow::Signal(signal) => return Ok(Some(signal)),
                            };
                            self.check_refinement(&type_name, &val)?;
                            val
                        }
                    }
                } else {
                    match self.eval_expr_flow(&decl.value)? {
                        ExprFlow::Value(value) => value,
                        ExprFlow::Signal(signal) => return Ok(Some(signal)),
                    }
                };
                let val = self.normalize_value_for_type(&declared_ty, val)?;
                self.set_variable_with_type(&decl.name.name, val, declared_ty);
                Ok(None)
            }

            Stmt::ComptimeTypeBind(bind) => {
                let bound_type_expr = if let Some(bound_type_expr) =
                    comptime_type_info_binding(&bind.value)
                {
                    self.substitute_type_expr(bound_type_expr)
                } else if let Some((source_ty, index)) = comptime_type_arg_binding(&bind.value) {
                    let source_ty = self.substitute_type_expr(source_ty);
                    self.checked_type_info_arg_types(&source_ty)
                        .unwrap_or_else(|| self.type_info_arg_types(&source_ty))
                        .get(index)
                        .cloned()
                        .ok_or_else(|| {
                            format!(
                                "`comptime type` type.arg index {index} is out of range for type '{}'",
                                type_expr_display(&source_ty)
                            )
                        })?
                } else if let Some(field_name) = reflected_field_type_info_binding(&bind.value) {
                    self.bound_reflected_field_type(field_name)?
                } else if let Some(info_name) = reflected_type_info_binding(&bind.value) {
                    self.bound_reflected_type_info_type(info_name)?
                } else {
                    return Err("`comptime type` currently requires a direct `type.info[T]()` initializer or trusted reflected metadata".to_string());
                };
                let mut scope = HashMap::new();
                scope.insert(bind.name.name.clone(), bound_type_expr);
                self.type_arg_scopes.push(scope);
                let result = self.exec_block_inner(&bind.body);
                self.type_arg_scopes.pop();
                result
            }

            Stmt::Assign(assign) => {
                let val = match self.eval_expr_flow(&assign.value)? {
                    ExprFlow::Value(value) => value,
                    ExprFlow::Signal(signal) => return Ok(Some(signal)),
                };
                match &assign.target {
                    Expr::Ident(ident) => {
                        // If variable doesn't exist yet, create it (handles parser producing
                        // AssignStmt instead of VarDecl for `Type name = expr` patterns)
                        if self.get_variable(&ident.name).is_none() {
                            self.set_variable(&ident.name, val);
                        } else {
                            self.assign_variable(&ident.name, val)?;
                        }
                    }
                    _ => {
                        return Err(
                            "only simple variable assignment is supported in comptime".to_string()
                        );
                    }
                }
                Ok(None)
            }

            Stmt::Return(ret) => {
                let val = match &ret.value {
                    Some(expr) => match self.eval_expr_flow(expr)? {
                        ExprFlow::Value(value) => value,
                        ExprFlow::Signal(signal) => return Ok(Some(signal)),
                    },
                    None => Value::Nothing,
                };
                Ok(Some(Signal::Return(val)))
            }

            Stmt::If(if_stmt) => {
                let cond = match self.eval_expr_flow(&if_stmt.condition)? {
                    ExprFlow::Value(value) => value,
                    ExprFlow::Signal(signal) => return Ok(Some(signal)),
                };
                if is_truthy(&cond)? {
                    return self.exec_block_inner(&if_stmt.then_block);
                }
                for (else_if_cond, else_if_block) in &if_stmt.else_ifs {
                    let val = match self.eval_expr_flow(else_if_cond)? {
                        ExprFlow::Value(value) => value,
                        ExprFlow::Signal(signal) => return Ok(Some(signal)),
                    };
                    if is_truthy(&val)? {
                        return self.exec_block_inner(else_if_block);
                    }
                }
                if let Some(else_block) = &if_stmt.else_block {
                    return self.exec_block_inner(else_block);
                }
                Ok(None)
            }

            Stmt::For(for_stmt) => {
                let reflected_field_bindings =
                    self.reflected_field_loop_bindings(&for_stmt.iterable)?;
                let reflected_variant_bindings =
                    self.reflected_variant_loop_bindings(&for_stmt.iterable)?;
                let reflected_machine_state_bindings =
                    self.reflected_machine_state_loop_bindings(&for_stmt.iterable)?;
                let reflected_variant_field_owner =
                    self.reflected_variant_field_loop_owner(&for_stmt.iterable)?;
                let reflected_machine_field_owner =
                    self.reflected_machine_field_loop_owner(&for_stmt.iterable)?;
                let reflected_type_info_bindings =
                    self.reflected_type_info_arg_loop_bindings(&for_stmt.iterable)?;
                let iterable = match self.eval_expr_flow(&for_stmt.iterable)? {
                    ExprFlow::Value(value) => value,
                    ExprFlow::Signal(signal) => return Ok(Some(signal)),
                };
                match iterable {
                    Value::List(items) => {
                        for (index, item) in items.into_iter().enumerate() {
                            self.push_scope();
                            let loop_item = item.clone();
                            self.set_variable(&for_stmt.variable.name, item);

                            let pushed_field_scope = reflected_field_bindings
                                .as_ref()
                                .and_then(|bindings| bindings.get(index))
                                .map(|binding| {
                                    let mut scope = HashMap::new();
                                    scope.insert(for_stmt.variable.name.clone(), binding.clone());
                                    self.reflected_field_scopes.push(scope);
                                })
                                .is_some();
                            let pushed_variant_scope = reflected_variant_bindings
                                .as_ref()
                                .and_then(|bindings| bindings.get(index))
                                .map(|binding| {
                                    let mut scope = HashMap::new();
                                    scope.insert(for_stmt.variable.name.clone(), binding.clone());
                                    self.reflected_variant_scopes.push(scope);
                                })
                                .is_some();
                            let pushed_machine_state_scope = reflected_machine_state_bindings
                                .as_ref()
                                .and_then(|bindings| bindings.get(index))
                                .map(|binding| {
                                    let mut scope = HashMap::new();
                                    scope.insert(for_stmt.variable.name.clone(), binding.clone());
                                    self.reflected_machine_state_scopes.push(scope);
                                })
                                .is_some();
                            let pushed_variant_field_scope = reflected_variant_field_owner
                                .as_ref()
                                .map(|owner_ty| {
                                    self.reflected_variant_field_binding_for_value(
                                        &owner_ty.ty,
                                        owner_ty.variant.as_deref(),
                                        &loop_item,
                                    )
                                })
                                .transpose()?
                                .map(|binding| {
                                    let mut scope = HashMap::new();
                                    scope.insert(for_stmt.variable.name.clone(), binding);
                                    self.reflected_field_scopes.push(scope);
                                })
                                .is_some();
                            let pushed_machine_field_scope = reflected_machine_field_owner
                                .as_ref()
                                .map(|owner_ty| {
                                    self.reflected_machine_field_binding_for_value(
                                        &owner_ty.ty,
                                        &loop_item,
                                    )
                                })
                                .transpose()?
                                .map(|binding| {
                                    let mut scope = HashMap::new();
                                    scope.insert(for_stmt.variable.name.clone(), binding);
                                    self.reflected_field_scopes.push(scope);
                                })
                                .is_some();
                            let pushed_type_info_scope = reflected_type_info_bindings
                                .as_ref()
                                .and_then(|bindings| bindings.get(index))
                                .map(|binding| {
                                    let mut scope = HashMap::new();
                                    scope.insert(for_stmt.variable.name.clone(), binding.clone());
                                    self.reflected_type_info_scopes.push(scope);
                                })
                                .is_some();

                            let signal = self.exec_block_inner(&for_stmt.body);
                            if pushed_type_info_scope {
                                self.reflected_type_info_scopes.pop();
                            }
                            if pushed_machine_field_scope {
                                self.reflected_field_scopes.pop();
                            }
                            if pushed_variant_field_scope {
                                self.reflected_field_scopes.pop();
                            }
                            if pushed_field_scope {
                                self.reflected_field_scopes.pop();
                            }
                            if pushed_variant_scope {
                                self.reflected_variant_scopes.pop();
                            }
                            if pushed_machine_state_scope {
                                self.reflected_machine_state_scopes.pop();
                            }
                            self.pop_scope();
                            let signal = signal?;
                            match signal {
                                Some(Signal::Break) => break,
                                Some(Signal::Continue) => continue,
                                Some(other) => return Ok(Some(other)),
                                None => {}
                            }
                        }
                    }
                    Value::String(s) => {
                        for ch in s.chars() {
                            self.push_scope();
                            self.set_variable(
                                &for_stmt.variable.name,
                                Value::String(ch.to_string()),
                            );
                            let signal = self.exec_block_inner(&for_stmt.body)?;
                            self.pop_scope();
                            match signal {
                                Some(Signal::Break) => break,
                                Some(Signal::Continue) => continue,
                                Some(other) => return Ok(Some(other)),
                                None => {}
                            }
                        }
                    }
                    Value::Map(entries) => {
                        for (key, val) in entries {
                            self.push_scope();
                            self.set_variable(&for_stmt.variable.name, key);
                            if let Some(ref val_var) = for_stmt.value_variable {
                                self.set_variable(&val_var.name, val);
                            }
                            let signal = self.exec_block_inner(&for_stmt.body)?;
                            self.pop_scope();
                            match signal {
                                Some(Signal::Break) => break,
                                Some(Signal::Continue) => continue,
                                Some(other) => return Ok(Some(other)),
                                None => {}
                            }
                        }
                    }
                    Value::Set(items) => {
                        for item in items {
                            self.push_scope();
                            self.set_variable(&for_stmt.variable.name, item);
                            let signal = self.exec_block_inner(&for_stmt.body)?;
                            self.pop_scope();
                            match signal {
                                Some(Signal::Break) => break,
                                Some(Signal::Continue) => continue,
                                Some(other) => return Ok(Some(other)),
                                None => {}
                            }
                        }
                    }
                    _ => {
                        return Err(
                            "for loop requires a list, string, map, or set value".to_string()
                        );
                    }
                }
                Ok(None)
            }

            Stmt::While(while_stmt) => {
                loop {
                    let cond = match self.eval_expr_flow(&while_stmt.condition)? {
                        ExprFlow::Value(value) => value,
                        ExprFlow::Signal(signal) => return Ok(Some(signal)),
                    };
                    if !is_truthy(&cond)? {
                        break;
                    }
                    self.push_scope();
                    let signal = self.exec_block_inner(&while_stmt.body)?;
                    self.pop_scope();
                    match signal {
                        Some(Signal::Break) => break,
                        Some(Signal::Continue) => continue,
                        Some(other) => return Ok(Some(other)),
                        None => {}
                    }
                }
                Ok(None)
            }

            Stmt::Match(match_stmt) => {
                let val = match self.eval_expr_flow(&match_stmt.expr)? {
                    ExprFlow::Value(value) => value,
                    ExprFlow::Signal(signal) => return Ok(Some(signal)),
                };
                let (variant_name, fields) = match &val {
                    Value::Enum {
                        variant, fields, ..
                    } => (variant.clone(), fields.clone()),
                    _ => return Err(format!("match requires an enum value, got {val}")),
                };

                for arm in &match_stmt.arms {
                    match &arm.pattern {
                        Pattern::Ident(ident) => {
                            if ident.name == variant_name {
                                return self.exec_block_inner(&arm.body);
                            }
                        }
                        Pattern::Variant(name, bindings) => {
                            if name.name == variant_name {
                                self.push_scope();
                                for (binding, field_val) in bindings.iter().zip(fields.iter()) {
                                    self.set_variable(&binding.name, field_val.clone());
                                }
                                let result = self.exec_block_inner(&arm.body);
                                self.pop_scope();
                                return result;
                            }
                        }
                        Pattern::Other(_) => {
                            return self.exec_block_inner(&arm.body);
                        }
                    }
                }
                Ok(None)
            }

            Stmt::Expr(expr_stmt) => {
                // Type names appearing as bare ExprStmt (from parser producing ExprStmt
                // instead of VarDecl for `Type name = expr`) are harmless — ignore errors.
                match self.eval_expr_flow(&expr_stmt.expr) {
                    Ok(ExprFlow::Value(_)) => Ok(None),
                    Ok(ExprFlow::Signal(signal)) => Ok(Some(signal)),
                    Err(_) => Ok(None),
                }
            }

            Stmt::Assert(assert_stmt) => {
                let cond = match self.eval_expr_flow(&assert_stmt.condition)? {
                    ExprFlow::Value(value) => value,
                    ExprFlow::Signal(signal) => return Ok(Some(signal)),
                };
                match cond {
                    Value::Bool(true) => Ok(None),
                    Value::Bool(false) => {
                        let msg = if let Some(msg_expr) = &assert_stmt.message {
                            match self.eval_expr_flow(msg_expr)? {
                                ExprFlow::Value(Value::String(s)) => s,
                                ExprFlow::Value(other) => other.to_string(),
                                ExprFlow::Signal(signal) => return Ok(Some(signal)),
                            }
                        } else {
                            "assertion failed".to_string()
                        };
                        Err(msg)
                    }
                    _ => Err("assert condition must be a boolean".to_string()),
                }
            }

            Stmt::Trace(trace_stmt) => {
                self.trace_variable(&trace_stmt.name.name)?;
                Ok(None)
            }

            Stmt::Breakpoint(breakpoint_stmt) => {
                let should_break = if let Some(condition) = &breakpoint_stmt.condition {
                    match self.eval_expr_flow(condition)? {
                        ExprFlow::Value(Value::Bool(value)) => value,
                        ExprFlow::Value(other) => {
                            return Err(format!("breakpoint condition must be bool, got {other}"));
                        }
                        ExprFlow::Signal(signal) => return Ok(Some(signal)),
                    }
                } else {
                    true
                };

                if should_break {
                    self.hit_breakpoint();
                }
                Ok(None)
            }

            Stmt::Respond(resp) => {
                let val = self.eval_expr(&resp.value)?;
                Ok(Some(Signal::Respond(val)))
            }

            Stmt::Break(_) => Ok(Some(Signal::Break)),
            Stmt::Continue(_) => Ok(Some(Signal::Continue)),

            Stmt::Use(use_decl) => {
                let bound_name = Self::use_bound_name(&use_decl.path.name, use_decl.alias.as_ref());
                self.set_namespace_alias(bound_name, use_decl.path.name.clone());
                Ok(None)
            }
        }
    }

    /// Execute a single statement.  Converts the internal signal into a
    /// public-facing result.
    pub fn exec_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match self.exec_stmt_inner(stmt)? {
            None | Some(Signal::Break) | Some(Signal::Continue) | Some(Signal::Return(_)) => Ok(()),
            Some(Signal::Default(_)) => {
                Err("`default` can only be used inside a `handle` block".to_string())
            }
            Some(Signal::Respond(_)) => {
                Err("`respond` can only be used inside a `receive` handler".to_string())
            }
        }
    }

    // -- Block execution ----------------------------------------------------

    /// Execute a block (list of statements), propagating control-flow signals.
    fn exec_block_inner(&mut self, block: &Block) -> Result<Option<Signal>, String> {
        self.push_scope();
        let mut result = None;
        for stmt in &block.stmts {
            if let Some(signal) = self.exec_stmt_inner(stmt)? {
                result = Some(signal);
                break;
            }
        }
        self.pop_scope();
        Ok(result)
    }

    /// Execute a block, returning the value produced by a `return` statement
    /// (if any).
    pub fn exec_block(&mut self, block: &Block) -> Result<Option<Value>, String> {
        match self.exec_block_inner(block)? {
            Some(Signal::Return(v)) => Ok(Some(v)),
            Some(Signal::Default(_)) => {
                Err("`default` can only be used inside a `handle` block".to_string())
            }
            _ => Ok(None),
        }
    }

    // -- Dotted name extraction -----------------------------------------------

    /// Recursively extract a dotted name from nested FieldAccess nodes.
    /// e.g. `Expr::FieldAccess(Expr::Ident("Stdout"), "write")` → `"Stdout.write"`
    fn extract_dotted_name(expr: &Expr, suffix: &str) -> Option<String> {
        match expr {
            Expr::Ident(ident) => Some(format!("{}.{}", ident.name, suffix)),
            Expr::FieldAccess(inner, field, _) => {
                let inner_name = Self::extract_dotted_name(inner, &field.name)?;
                Some(format!("{inner_name}.{suffix}"))
            }
            _ => None,
        }
    }

    fn dotted_expr_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Ident(ident) => Some(ident.name.clone()),
            Expr::FieldAccess(inner, field, _) => Self::extract_dotted_name(inner, &field.name),
            _ => None,
        }
    }

    /// Execute an actor message (send or ask).
    ///
    /// `inner` is the expression after the `send`/`ask` keyword:
    ///   - `actor_expr.handler_name`   (no args)
    ///   - `actor_expr.handler_name(args...)` (with args)
    ///
    /// When `is_ask` is true, a `respond value` inside the handler returns
    /// `value` as the result; for `send` the result is always `nothing`.
    fn eval_actor_message(&mut self, inner: &Expr, is_ask: bool) -> Result<Value, String> {
        // Decompose `actor_expr.handler_name` or `actor_expr.handler_name(args)`.
        let (actor_expr, handler_name, call_args) = match inner {
            Expr::Call(callee, args, _) => match callee.as_ref() {
                Expr::FieldAccess(base, field, _) => {
                    (base.as_ref(), field.name.clone(), Some(args.as_slice()))
                }
                _ => {
                    return Err(
                        "send/ask: expected actor.handler or actor.handler(args)".to_string()
                    );
                }
            },
            Expr::FieldAccess(base, field, _) => (base.as_ref(), field.name.clone(), None),
            _ => return Err("send/ask: expected actor.handler expression".to_string()),
        };

        // Evaluate actor handle.
        let actor_val = match self.eval_expr_flow(actor_expr)? {
            ExprFlow::Value(v) => v,
            ExprFlow::Signal(s) => {
                return Err(format!("send/ask: actor expression returned signal: {s:?}"));
            }
        };
        let actor_id = match actor_val {
            Value::Actor(id) => id,
            _ => return Err(format!("send/ask: expected actor value, got {actor_val}")),
        };

        // Evaluate message arguments.
        let mut arg_values = Vec::new();
        if let Some(args) = call_args {
            for arg in args {
                let val = match self.eval_expr_flow(&arg.value)? {
                    ExprFlow::Value(v) => v,
                    ExprFlow::Signal(s) => {
                        return Err(format!("send/ask: arg expression returned signal: {s:?}"));
                    }
                };
                arg_values.push(val);
            }
        }

        // Clone the actor def and instance state to avoid borrow conflicts.
        let instance = self
            .actor_instances
            .get(&actor_id)
            .ok_or_else(|| format!("unknown actor instance #{actor_id}"))?;
        let type_name = instance.type_name.clone();
        let state_snapshot = instance.state.clone();
        let caps_snapshot = instance.capabilities.clone();

        let actor_def = self
            .actor_defs
            .get(&type_name)
            .ok_or_else(|| format!("unknown actor type '{type_name}'"))?
            .clone();

        let handler = actor_def
            .handlers
            .iter()
            .find(|h| h.name.name == handler_name)
            .ok_or_else(|| format!("actor '{type_name}' has no handler '{handler_name}'"))?
            .clone();
        let mut normalized_args = Vec::with_capacity(arg_values.len());
        for (param, value) in handler.params.iter().zip(arg_values) {
            let param_ty = self.substitute_type_expr(&param.ty);
            normalized_args.push(self.normalize_value_for_type(&param_ty, value)?);
        }

        // Execute handler body in a new scope with state + caps + params.
        self.push_scope();
        for (name, val) in &state_snapshot {
            if let Some(field) = actor_def
                .state_fields
                .iter()
                .find(|field| field.name.name == *name)
            {
                let field_ty = self.substitute_type_expr(&field.ty);
                self.set_variable_with_type(name, val.clone(), field_ty);
            } else {
                self.set_variable(name, val.clone());
            }
        }
        for (name, val) in &caps_snapshot {
            if let Some(param) = actor_def
                .capability_params
                .iter()
                .find(|param| param.name.name == *name)
            {
                let param_ty = self.substitute_type_expr(&param.ty);
                self.set_variable_with_type(name, val.clone(), param_ty);
            } else {
                self.set_variable(name, val.clone());
            }
        }
        for (param, val) in handler.params.iter().zip(normalized_args) {
            let param_ty = self.substitute_type_expr(&param.ty);
            self.set_variable_with_type(&param.name.name, val, param_ty);
        }

        // Execute the handler body, collecting signals.
        let mut respond_value = Value::Nothing;
        for stmt in &handler.body.stmts {
            match self.exec_stmt_inner(stmt)? {
                Some(Signal::Respond(val)) => {
                    respond_value = val;
                    break;
                }
                Some(Signal::Return(_)) => break,
                Some(Signal::Break) | Some(Signal::Continue) => break,
                Some(Signal::Default(_)) => break,
                None => {}
            }
        }

        // Collect updated state field values before popping scope.
        let mut updated_state = state_snapshot;
        for field in &actor_def.state_fields {
            let name = &field.name.name;
            // Check innermost scope(s) for the updated value.
            if let Some(val) = self.scopes.last().and_then(|s| s.get(name)).cloned() {
                let field_ty = self.substitute_type_expr(&field.ty);
                let val = self.normalize_value_for_type(&field_ty, val)?;
                updated_state.insert(name.clone(), val);
            }
        }

        self.pop_scope();

        // Write updated state back to the actor instance.
        if let Some(instance) = self.actor_instances.get_mut(&actor_id) {
            instance.state = updated_state;
        }

        if is_ask {
            if let Some(responds) = &handler.responds {
                let responds = self.substitute_type_expr(responds);
                respond_value = self.normalize_value_for_type(&responds, respond_value)?;
            }
            Ok(respond_value)
        } else {
            Ok(Value::Nothing)
        }
    }

    fn call_bitfield_builtin(&self, name: &str, args: &[Value]) -> Option<Result<Value, String>> {
        let (bitfield_name, method_name) = name.rsplit_once('.')?;
        if !self.bitfields.contains_key(bitfield_name) {
            return None;
        }

        match method_name {
            "to_bytes" => Some(self.bitfield_to_bytes(bitfield_name, args)),
            "from_bytes" => Some(self.bitfield_from_bytes(bitfield_name, args)),
            _ => None,
        }
    }

    fn bitfield_to_bytes(&self, bitfield_name: &str, args: &[Value]) -> Result<Value, String> {
        if args.len() != 1 {
            return Err(format!(
                "{}.to_bytes expects 1 argument(s), got {}",
                bitfield_name,
                args.len()
            ));
        }
        let bitfield = self
            .bitfields
            .get(bitfield_name)
            .ok_or_else(|| format!("undefined bitfield '{bitfield_name}'"))?;
        let bytes = self.encode_bitfield_value(bitfield, &args[0])?;
        Ok(Value::Bytes(bytes))
    }

    fn bitfield_from_bytes(&self, bitfield_name: &str, args: &[Value]) -> Result<Value, String> {
        if args.len() != 1 {
            return Err(format!(
                "{}.from_bytes expects 1 argument(s), got {}",
                bitfield_name,
                args.len()
            ));
        }
        let bitfield = self
            .bitfields
            .get(bitfield_name)
            .ok_or_else(|| format!("undefined bitfield '{bitfield_name}'"))?;

        let Value::Bytes(bytes) = &args[0] else {
            return Err(format!(
                "{bitfield_name}.from_bytes expects a bytes argument"
            ));
        };

        match self.decode_bitfield_value(bitfield, bytes) {
            Ok(value) => Ok(Value::ResultOk(Box::new(value))),
            Err(message) => Ok(Value::ResultFail(Box::new(Value::String(message)))),
        }
    }

    fn encode_bitfield_value(
        &self,
        bitfield: &BitfieldDef,
        value: &Value,
    ) -> Result<Vec<u8>, String> {
        let Value::Struct { type_name, fields } = value else {
            return Err(format!(
                "{}.to_bytes expects a {} value",
                bitfield.name.name, bitfield.name.name
            ));
        };
        if type_name != &bitfield.name.name {
            return Err(format!(
                "{}.to_bytes expects a {} value, got {}",
                bitfield.name.name, bitfield.name.name, type_name
            ));
        }

        let mut bits = Vec::new();
        for field in &bitfield.fields {
            let (_, field_value) = fields
                .iter()
                .find(|(name, _)| name == &field.name.name)
                .ok_or_else(|| {
                    format!(
                        "bitfield '{}' is missing field '{}'",
                        bitfield.name.name, field.name.name
                    )
                })?;

            match &field.kind {
                BitfieldFieldKind::Bits { width, as_type } => {
                    let numeric = self.bitfield_field_numeric_value(
                        bitfield,
                        &field.name.name,
                        *width,
                        as_type.as_ref(),
                        field_value,
                    )?;
                    let byte_aligned = bits.len() % 8 == 0;
                    self.push_encoded_bits(
                        &mut bits,
                        numeric,
                        *width,
                        bitfield.network_order,
                        byte_aligned,
                    );
                }
                BitfieldFieldKind::Payload(_) => {
                    if bits.len() % 8 != 0 {
                        return Err(format!(
                            "bitfield '{}' payload field '{}' must begin on a byte boundary",
                            bitfield.name.name, field.name.name
                        ));
                    }
                    let payload = self.value_to_byte_list(field_value).map_err(|message| {
                        format!(
                            "bitfield '{}' field '{}': {}",
                            bitfield.name.name, field.name.name, message
                        )
                    })?;
                    let mut bytes = Self::bits_to_bytes(&bits);
                    bytes.extend(payload);
                    return Ok(bytes);
                }
            }
        }

        Ok(Self::bits_to_bytes(&bits))
    }

    fn decode_bitfield_value(&self, bitfield: &BitfieldDef, bytes: &[u8]) -> Result<Value, String> {
        let mut bit_index = 0usize;
        let mut fields = Vec::with_capacity(bitfield.fields.len());

        for field in &bitfield.fields {
            match &field.kind {
                BitfieldFieldKind::Bits { width, as_type } => {
                    if *width > 64 {
                        return Err(format!(
                            "bitfield '{}' field '{}' is {} bit(s) wide; the current runtime supports at most 64 bit(s)",
                            bitfield.name.name, field.name.name, width,
                        ));
                    }
                    let numeric = if *width > 8
                        && width % 8 == 0
                        && !bitfield.network_order
                        && bit_index.is_multiple_of(8)
                    {
                        let byte_count = (*width as usize) / 8;
                        let start = bit_index / 8;
                        let end = start + byte_count;
                        if end > bytes.len() {
                            return Err(format!(
                                "bitfield '{}.from_bytes' expected at least {} byte(s), got {}",
                                bitfield.name.name,
                                end,
                                bytes.len()
                            ));
                        }
                        bit_index += *width as usize;
                        let mut value = 0u64;
                        for (shift, byte) in bytes[start..end].iter().enumerate() {
                            value |= (*byte as u64) << (shift * 8);
                        }
                        value
                    } else {
                        Self::read_bits(bytes, &mut bit_index, *width).ok_or_else(|| {
                            format!(
                                "bitfield '{}.from_bytes' expected {} bit(s), got {} byte(s)",
                                bitfield.name.name,
                                bit_index + (*width as usize),
                                bytes.len()
                            )
                        })?
                    };

                    let value = if let Some(enum_ty) = as_type {
                        let enum_name = type_expr_name(enum_ty);
                        self.enum_value_from_numeric(&enum_name, numeric)
                            .map_err(|message| {
                                format!(
                                    "bitfield '{}' field '{}': {}",
                                    bitfield.name.name, field.name.name, message
                                )
                            })?
                    } else {
                        if *width == 64 {
                            Value::Uint64(numeric)
                        } else {
                            Value::Int64(numeric as i64)
                        }
                    };
                    fields.push((field.name.name.clone(), value));
                }
                BitfieldFieldKind::Payload(_) => {
                    if !bit_index.is_multiple_of(8) {
                        return Err(format!(
                            "bitfield '{}' payload field '{}' must begin on a byte boundary",
                            bitfield.name.name, field.name.name
                        ));
                    }
                    let start = bit_index / 8;
                    let payload = bytes[start..]
                        .iter()
                        .map(|byte| Value::Int64(*byte as i64))
                        .collect();
                    bit_index = bytes.len() * 8;
                    fields.push((field.name.name.clone(), Value::List(payload)));
                }
            }
        }

        let consumed_bytes = bit_index.div_ceil(8);
        if consumed_bytes != bytes.len() {
            return Err(format!(
                "bitfield '{}.from_bytes' expected {} byte(s), got {}",
                bitfield.name.name,
                consumed_bytes,
                bytes.len()
            ));
        }

        Ok(Value::Struct {
            type_name: bitfield.name.name.clone(),
            fields,
        })
    }

    fn bitfield_field_numeric_value(
        &self,
        bitfield: &BitfieldDef,
        field_name: &str,
        width: u16,
        as_type: Option<&TypeExpr>,
        value: &Value,
    ) -> Result<u64, String> {
        if let Some(enum_ty) = as_type {
            let enum_name = type_expr_name(enum_ty);
            let Value::Enum {
                type_name,
                variant,
                fields,
            } = value
            else {
                return Err(format!(
                    "field '{}' expects enum '{}'",
                    field_name, enum_name
                ));
            };
            if !fields.is_empty() {
                return Err(format!(
                    "field '{}' enum '{}' must use unit variants",
                    field_name, enum_name
                ));
            }
            if type_name != &enum_name {
                return Err(format!(
                    "field '{}' expects enum '{}', got '{}'",
                    field_name, enum_name, type_name
                ));
            }
            let numeric = self.enum_numeric_value(type_name, variant)?;
            if !Self::fits_in_bits(numeric, width) {
                return Err(format!(
                    "bitfield '{}' field '{}' is {} bit(s) wide and cannot hold enum variant '{}.{}'",
                    bitfield.name.name, field_name, width, type_name, variant
                ));
            }
            return Ok(numeric);
        }

        Self::plain_bitfield_field_numeric_value(&bitfield.name.name, field_name, width, value)
    }

    fn enum_numeric_value(&self, enum_name: &str, variant_name: &str) -> Result<u64, String> {
        let enm = self
            .enums
            .get(enum_name)
            .ok_or_else(|| format!("unknown enum '{}'", enum_name))?;
        let mut next_discriminant = 0_i64;
        for variant in &enm.variants {
            let discriminant = variant.discriminant.unwrap_or(next_discriminant);
            next_discriminant = discriminant.saturating_add(1);
            if variant.name.name == variant_name {
                if discriminant < 0 {
                    return Err(format!(
                        "enum '{}.{}' has negative discriminant {}",
                        enum_name, variant_name, discriminant
                    ));
                }
                return Ok(discriminant as u64);
            }
        }
        Err(format!(
            "enum '{}' has no variant '{}'",
            enum_name, variant_name
        ))
    }

    fn checked_enum_numeric_value(
        &self,
        enum_name: &str,
        variant_name: &str,
    ) -> Result<u64, String> {
        if let Some(variants) = self
            .reflection_metadata
            .as_ref()
            .and_then(|metadata| metadata.get_type_variants(enum_name))
        {
            for variant in variants {
                if variant.name == variant_name {
                    if variant.discriminant < 0 {
                        return Err(format!(
                            "enum '{}.{}' has negative discriminant {}",
                            enum_name, variant_name, variant.discriminant
                        ));
                    }
                    return Ok(variant.discriminant as u64);
                }
            }
            return Err(format!(
                "enum '{}' has no variant '{}'",
                enum_name, variant_name
            ));
        }

        if self
            .reflection_metadata
            .as_ref()
            .and_then(|metadata| metadata.get_type_info(enum_name))
            .is_some_and(|info| info.kind == "enum")
        {
            return Err(format!(
                "checked reflection metadata for type '{enum_name}' is missing variant metadata"
            ));
        }

        self.enum_numeric_value(enum_name, variant_name)
    }

    fn checked_bitfield_field_numeric_value(
        &self,
        bitfield_name: &str,
        field_name: &str,
        width: i64,
        enum_type: Option<&ReflectionTypeInfo>,
        value: &Value,
    ) -> Result<u64, String> {
        let width = width as u16;
        if let Some(enum_type) = enum_type {
            let enum_name = Self::reflection_type_base_name(enum_type);
            let Value::Enum {
                type_name,
                variant,
                fields,
            } = value
            else {
                return Err(format!(
                    "field '{}' expects enum '{}'",
                    field_name, enum_name
                ));
            };
            if !fields.is_empty() {
                return Err(format!(
                    "field '{}' enum '{}' must use unit variants",
                    field_name, enum_name
                ));
            }
            if type_name != &enum_name {
                return Err(format!(
                    "field '{}' expects enum '{}', got '{}'",
                    field_name, enum_name, type_name
                ));
            }
            let numeric = self.checked_enum_numeric_value(type_name, variant)?;
            if !Self::fits_in_bits(numeric, width) {
                return Err(format!(
                    "bitfield '{}' field '{}' is {} bit(s) wide and cannot hold enum variant '{}.{}'",
                    bitfield_name, field_name, width, type_name, variant
                ));
            }
            return Ok(numeric);
        }

        Self::plain_bitfield_field_numeric_value(bitfield_name, field_name, width, value)
    }

    fn plain_bitfield_field_numeric_value(
        bitfield_name: &str,
        field_name: &str,
        width: u16,
        value: &Value,
    ) -> Result<u64, String> {
        if width > 64 {
            return Err(format!(
                "bitfield '{}' field '{}' is {} bit(s) wide; the current runtime supports at most 64 bit(s)",
                bitfield_name, field_name, width,
            ));
        }

        let numeric = match value {
            Value::Int64(int_value) if *int_value >= 0 => *int_value as u64,
            Value::Int64(int_value) => {
                return Err(format!(
                    "bitfield '{}' field '{}' is {} bit(s) wide and cannot hold '{}'",
                    bitfield_name, field_name, width, int_value
                ));
            }
            Value::Uint64(uint_value) => *uint_value,
            _ => {
                return Err(format!("field '{}' expects int64 or uint64", field_name));
            }
        };

        if !Self::fits_in_bits(numeric, width) {
            return Err(format!(
                "bitfield '{}' field '{}' is {} bit(s) wide and cannot hold '{}'",
                bitfield_name, field_name, width, value
            ));
        }
        Ok(numeric)
    }

    fn normalized_plain_bitfield_field_value(
        bitfield_name: &str,
        field_name: &str,
        width: u16,
        value: &Value,
    ) -> Result<Value, String> {
        let numeric =
            Self::plain_bitfield_field_numeric_value(bitfield_name, field_name, width, value)?;
        if width == 64 {
            Ok(Value::Uint64(numeric))
        } else {
            Ok(Value::Int64(numeric as i64))
        }
    }

    fn enum_value_from_numeric(&self, enum_name: &str, numeric: u64) -> Result<Value, String> {
        let enm = self
            .enums
            .get(enum_name)
            .ok_or_else(|| format!("unknown enum '{}'", enum_name))?;
        let mut next_discriminant = 0_i64;
        for variant in &enm.variants {
            let discriminant = variant.discriminant.unwrap_or(next_discriminant);
            next_discriminant = discriminant.saturating_add(1);
            if discriminant >= 0 && discriminant as u64 == numeric {
                return Ok(Value::Enum {
                    type_name: enum_name.to_string(),
                    variant: variant.name.name.clone(),
                    fields: vec![],
                });
            }
        }
        Err(format!(
            "enum '{}' has no variant for value {}",
            enum_name, numeric
        ))
    }

    fn value_to_byte_list(&self, value: &Value) -> Result<Vec<u8>, String> {
        match value {
            Value::Bytes(bytes) => Ok(bytes.clone()),
            Value::List(items) => items
                .iter()
                .map(|item| match item {
                    Value::Int64(value) if (0..=255).contains(value) => Ok(*value as u8),
                    Value::Int64(value) => Err(format!("byte value out of range: {}", value)),
                    other => Err(format!("payload expects list[uint8], found {}", other)),
                })
                .collect(),
            other => Err(format!(
                "payload expects list[uint8] or bytes, found {}",
                other
            )),
        }
    }

    fn fits_in_bits(value: u64, width: u16) -> bool {
        if width == 64 {
            true
        } else if width > 64 {
            false
        } else {
            value < (1_u64 << width)
        }
    }

    fn push_encoded_bits(
        &self,
        bits: &mut Vec<bool>,
        value: u64,
        width: u16,
        network_order: bool,
        byte_aligned: bool,
    ) {
        if width > 8 && width.is_multiple_of(8) && !network_order && byte_aligned {
            let byte_count = (width / 8) as usize;
            for byte_index in 0..byte_count {
                let byte = ((value >> (byte_index * 8)) & 0xFF) as u8;
                for bit_shift in (0..8).rev() {
                    bits.push(((byte >> bit_shift) & 1) == 1);
                }
            }
            return;
        }

        for bit_shift in (0..width).rev() {
            bits.push(((value >> bit_shift) & 1) == 1);
        }
    }

    fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(bits.len().div_ceil(8));
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (index, bit) in chunk.iter().enumerate() {
                if *bit {
                    byte |= 1 << (7 - index);
                }
            }
            bytes.push(byte);
        }
        bytes
    }

    fn read_bits(bytes: &[u8], bit_index: &mut usize, width: u16) -> Option<u64> {
        let mut value = 0u64;
        for _ in 0..width {
            let byte_index = *bit_index / 8;
            if byte_index >= bytes.len() {
                return None;
            }
            let bit_in_byte = 7 - (*bit_index % 8);
            let bit = (bytes[byte_index] >> bit_in_byte) & 1;
            value = (value << 1) | (bit as u64);
            *bit_index += 1;
        }
        Some(value)
    }

    fn call_builtin_with_type_args(
        &mut self,
        name: &str,
        type_args: &[TypeExpr],
        args: &[Value],
    ) -> Option<Result<Value, String>> {
        let is_typed_builtin = matches!(
            name,
            "type.name"
                | "type.kind"
                | "type.kind_tag"
                | "type.primitive_tag"
                | "type.has_secret"
                | "type.info"
                | "type.arg"
                | "type.fields"
                | "type.bitfield_layout"
                | "type.bitfield_fields"
                | "type.machine_layout"
                | "type.machine_states"
                | "type.machine_transitions"
                | "type.machine_state_value"
                | "type.variants"
                | "type.variant_value"
                | "type.field_value"
                | "type.machine_field_value"
                | "type.variant_field_value"
                | "type.construct_start"
                | "type.construct_variant_start"
                | "type.construct_machine_start"
                | "type.construct_put"
                | "type.construct_finish"
                | "json.parse"
                | "json.parse_exact"
                | "json.serialize"
                | "json.serialize_public"
        );
        if !is_typed_builtin {
            return None;
        }
        let expected_type_arg_count = if matches!(
            name,
            "type.field_value"
                | "type.machine_field_value"
                | "type.variant_field_value"
                | "type.construct_put"
        ) {
            2
        } else {
            1
        };
        if type_args.len() != expected_type_arg_count {
            return Some(Err(format!(
                "{name} expects {expected_type_arg_count} type argument(s), got {}",
                type_args.len()
            )));
        }

        let ty = self.substitute_type_expr(&type_args[0]);
        if let Some(error) = self.missing_checked_type_info_error(&ty) {
            return Some(Err(error));
        }
        Some(match name {
            "type.name" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                Ok(Value::String(
                    self.checked_type_info(&ty)
                        .map(|info| info.type_name.clone())
                        .unwrap_or_else(|| type_expr_display(&ty)),
                ))
            }
            "type.kind" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                Ok(Value::String(
                    self.checked_type_kind(&ty)
                        .unwrap_or_else(|| self.type_expr_kind(&ty))
                        .to_string(),
                ))
            }
            "type.kind_tag" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                let kind = self
                    .checked_type_kind(&ty)
                    .unwrap_or_else(|| self.type_expr_kind(&ty));
                Ok(Self::type_kind_tag_value(kind))
            }
            "type.primitive_tag" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                Ok(self
                    .checked_type_info(&ty)
                    .map(|info| Self::primitive_tag_value(info.primitive_tag.as_deref()))
                    .unwrap_or_else(|| self.type_primitive_tag_value(&ty)))
            }
            "type.has_secret" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                Ok(Value::Bool(
                    self.checked_type_has_secret(&ty)
                        .unwrap_or_else(|| self.type_expr_has_secret(&ty)),
                ))
            }
            "type.info" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                Ok(self
                    .checked_type_info_value(&ty)
                    .unwrap_or_else(|| self.type_info_value(&ty)))
            }
            "type.arg" => {
                if let Some(err) = check_args(name, 1, args) {
                    return Some(err);
                }
                let index = match args.first() {
                    Some(Value::Int64(index)) if *index >= 0 => *index as usize,
                    Some(other) => {
                        return Some(Err(format!(
                            "type.arg expects a non-negative int64 index, got {other}"
                        )));
                    }
                    None => unreachable!("type.arg argument count was already checked"),
                };
                if let Some(result) = self.checked_type_arg_value(&ty, index) {
                    return Some(result);
                }
                let arg_types = self.type_info_arg_types(&ty);
                let Some(arg_ty) = arg_types.get(index) else {
                    return Some(Err(format!(
                        "type.arg index {index} is out of range for type '{}'",
                        type_expr_display(&ty)
                    )));
                };
                Ok(self.type_info_value(arg_ty))
            }
            "type.construct_start" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                Ok(Value::TypeConstruction {
                    type_name: type_expr_display(&ty),
                    variant: None,
                    state: None,
                    fields: Vec::new(),
                })
            }
            "type.construct_variant_start" => {
                if let Some(err) = check_args(name, 1, args) {
                    return Some(err);
                }
                self.reflected_construct_variant_start(&ty, &args[0])
            }
            "type.construct_machine_start" => {
                if let Some(err) = check_args(name, 1, args) {
                    return Some(err);
                }
                self.reflected_construct_machine_start(&ty, &args[0])
            }
            "type.construct_put" => {
                if let Some(err) = check_args(name, 3, args) {
                    return Some(err);
                }
                let expected_field_ty = self.substitute_type_expr(&type_args[1]);
                self.reflected_construct_put(&ty, &expected_field_ty, &args[0], &args[1], &args[2])
            }
            "type.construct_finish" => {
                if let Some(err) = check_args(name, 1, args) {
                    return Some(err);
                }
                self.reflected_construct_finish(&ty, &args[0])
            }
            "type.fields" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                if let Some(value) = self.checked_type_fields_value(&ty) {
                    return Some(Ok(value));
                }
                if self.checked_metadata_kind_is(&ty, &["struct", "bitfield"]) {
                    return Some(Err(format!(
                        "checked reflection metadata for type '{}' is missing field metadata",
                        type_expr_display(&ty)
                    )));
                }
                Ok(Value::List(
                    self.type_expr_fields(&ty)
                        .into_iter()
                        .enumerate()
                        .map(|(index, field)| {
                            self.type_field_value(index, &type_expr_display(&ty), None, field)
                        })
                        .collect(),
                ))
            }
            "type.bitfield_layout" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                if let Some(value) = self.checked_bitfield_value(&ty) {
                    return Some(Ok(value));
                }
                if self.checked_metadata_kind_is(&ty, &["bitfield"]) {
                    return Some(Err(format!(
                        "checked reflection metadata for type '{}' is missing bitfield metadata",
                        type_expr_display(&ty)
                    )));
                }
                Ok(self.type_bitfield_value(self.type_expr_bitfield(&ty)))
            }
            "type.bitfield_fields" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                if let Some(value) = self.checked_bitfield_fields_value(&ty) {
                    return Some(Ok(value));
                }
                if self.checked_metadata_kind_is(&ty, &["bitfield"]) {
                    return Some(Err(format!(
                        "checked reflection metadata for type '{}' is missing bitfield metadata",
                        type_expr_display(&ty)
                    )));
                }
                Ok(Value::List(
                    self.type_expr_bitfield_fields(&ty)
                        .into_iter()
                        .enumerate()
                        .map(|(index, field)| self.type_bitfield_field_value(index, field))
                        .collect(),
                ))
            }
            "type.machine_layout" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                if let Some(value) = self.checked_machine_value(&ty) {
                    return Some(Ok(value));
                }
                if self.checked_metadata_kind_is(&ty, &["machine", "machine_state"]) {
                    return Some(Err(format!(
                        "checked reflection metadata for type '{}' is missing machine metadata",
                        type_expr_display(&ty)
                    )));
                }
                Ok(self.type_machine_value(&type_expr_name(&ty), self.type_expr_machine(&ty)))
            }
            "type.machine_states" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                if let Some(value) = self.checked_machine_states_value(&ty) {
                    return Some(Ok(value));
                }
                if self.checked_metadata_kind_is(&ty, &["machine", "machine_state"]) {
                    return Some(Err(format!(
                        "checked reflection metadata for type '{}' is missing machine metadata",
                        type_expr_display(&ty)
                    )));
                }
                Ok(Value::List(
                    self.type_expr_machine(&ty)
                        .states
                        .into_iter()
                        .enumerate()
                        .map(|(index, state)| {
                            self.type_machine_state_value(index, &type_expr_name(&ty), state)
                        })
                        .collect(),
                ))
            }
            "type.machine_transitions" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                if let Some(value) = self.checked_machine_transitions_value(&ty) {
                    return Some(Ok(value));
                }
                if self.checked_metadata_kind_is(&ty, &["machine", "machine_state"]) {
                    return Some(Err(format!(
                        "checked reflection metadata for type '{}' is missing machine metadata",
                        type_expr_display(&ty)
                    )));
                }
                Ok(Value::List(
                    self.type_expr_machine(&ty)
                        .edges
                        .into_iter()
                        .enumerate()
                        .map(|(index, transition)| {
                            Self::type_machine_transition_value(index, transition)
                        })
                        .collect(),
                ))
            }
            "type.machine_state_value" => {
                if let Some(err) = check_args(name, 1, args) {
                    return Some(err);
                }
                self.reflected_machine_state_value(&args[0], &ty)
            }
            "type.variants" => {
                if let Some(err) = check_args(name, 0, args) {
                    return Some(err);
                }
                if let Some(value) = self.checked_type_variants_value(&ty) {
                    return Some(Ok(value));
                }
                if self.checked_metadata_kind_is(&ty, &["enum"]) {
                    return Some(Err(format!(
                        "checked reflection metadata for type '{}' is missing variant metadata",
                        type_expr_display(&ty)
                    )));
                }
                Ok(Value::List(
                    self.type_expr_variants(&ty)
                        .into_iter()
                        .enumerate()
                        .map(|(index, variant)| {
                            self.type_variant_value(index, &type_expr_display(&ty), variant)
                        })
                        .collect(),
                ))
            }
            "type.variant_value" => {
                if let Some(err) = check_args(name, 1, args) {
                    return Some(err);
                }
                self.reflected_variant_value(&args[0], &ty)
            }
            "type.field_value" => {
                if let Some(err) = check_args(name, 2, args) {
                    return Some(err);
                }
                let expected_field_ty = self.substitute_type_expr(&type_args[1]);
                self.reflected_field_value(&args[0], &ty, &args[1], &expected_field_ty)
            }
            "type.machine_field_value" => {
                if let Some(err) = check_args(name, 2, args) {
                    return Some(err);
                }
                let expected_field_ty = self.substitute_type_expr(&type_args[1]);
                self.reflected_machine_field_value(&args[0], &ty, &args[1], &expected_field_ty)
            }
            "type.variant_field_value" => {
                if let Some(err) = check_args(name, 2, args) {
                    return Some(err);
                }
                let expected_field_ty = self.substitute_type_expr(&type_args[1]);
                self.reflected_variant_field_value(&args[0], &ty, &args[1], &expected_field_ty)
            }
            "json.parse" | "json.parse_exact" => {
                if let Some(err) = check_args(name, 1, args) {
                    return Some(err);
                }
                self.call_trusted_json_public_bridge(name, type_args, args)
            }
            "json.serialize" | "json.serialize_public" => {
                if let Some(err) = check_args(name, 1, args) {
                    return Some(err);
                }
                if name == "json.serialize"
                    && self
                        .checked_type_has_secret(&ty)
                        .unwrap_or_else(|| self.type_expr_has_secret(&ty))
                {
                    return Some(Err(format!(
                        "json.serialize cannot serialize secret-containing type '{}'",
                        type_expr_display(&ty)
                    )));
                }
                self.call_trusted_json_public_bridge(name, type_args, args)
            }
            _ => return None,
        })
    }

    fn call_trusted_json_public_bridge(
        &mut self,
        public_name: &str,
        type_args: &[TypeExpr],
        args: &[Value],
    ) -> Result<Value, String> {
        let hook = json_public_bridge_spec(public_name)
            .expect("JSON public bridge should have a public bridge spec")
            .hook;
        if !self.has_trusted_stdlib_function(hook) {
            return Err(format!(
                "{public_name} requires trusted stdlib hook '{hook}'"
            ));
        }
        if !self.has_trusted_stdlib_function(public_name) {
            return Err(format!(
                "{public_name} requires trusted stdlib wrapper '{public_name}'"
            ));
        }
        self.call_user_function_with_type_args(public_name, type_args, args.to_vec())
    }

    fn current_type_binding(&self, name: &str) -> Option<TypeExpr> {
        self.type_arg_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).cloned())
    }

    fn substitute_type_expr(&self, ty: &TypeExpr) -> TypeExpr {
        self.substitute_type_expr_in_namespace(ty, self.current_namespace.as_deref())
    }

    fn substitute_type_expr_in_namespace(
        &self,
        ty: &TypeExpr,
        namespace: Option<&str>,
    ) -> TypeExpr {
        match ty {
            TypeExpr::Named(ident) => {
                if let Some(bound) = self.current_type_binding(&ident.name) {
                    self.substitute_type_expr_in_namespace(&bound, namespace)
                } else {
                    TypeExpr::Named(self.expand_type_ident(ident, namespace))
                }
            }
            TypeExpr::Generic(ident, args, span) => TypeExpr::Generic(
                self.expand_type_ident(ident, namespace),
                args.iter()
                    .map(|arg| self.substitute_type_expr_in_namespace(arg, namespace))
                    .collect(),
                *span,
            ),
            TypeExpr::View(inner, span) => TypeExpr::View(
                Box::new(self.substitute_type_expr_in_namespace(inner, namespace)),
                *span,
            ),
            TypeExpr::StateQualified(inner, state, span) => TypeExpr::StateQualified(
                Box::new(self.substitute_type_expr_in_namespace(inner, namespace)),
                state.clone(),
                *span,
            ),
            TypeExpr::Function(params, return_type, span) => TypeExpr::Function(
                params
                    .iter()
                    .map(|param| self.substitute_type_expr_in_namespace(param, namespace))
                    .collect(),
                Box::new(self.substitute_type_expr_in_namespace(return_type, namespace)),
                *span,
            ),
        }
    }

    fn expand_type_ident(&self, ident: &Ident, namespace: Option<&str>) -> Ident {
        let mut expanded = ident.clone();
        if let Some(name) = self.expand_namespace_alias_name(&ident.name) {
            expanded.name = name;
        } else if !ident.name.contains('.')
            && !Self::is_builtin_type_name(&ident.name)
            && let Some(namespace) = namespace
        {
            let qualified = format!("{namespace}.{}", ident.name);
            if self.type_name_is_registered(&qualified) {
                expanded.name = qualified;
            }
        }
        expanded
    }

    fn is_builtin_type_name(name: &str) -> bool {
        matches!(
            name,
            "int8"
                | "int16"
                | "int32"
                | "int64"
                | "uint8"
                | "uint16"
                | "uint32"
                | "uint64"
                | "float32"
                | "float64"
                | "string"
                | "bool"
                | "bytes"
                | "nothing"
                | "TypeConstruction"
        )
    }

    fn type_name_is_registered(&self, name: &str) -> bool {
        self.structs.contains_key(name)
            || self.bitfields.contains_key(name)
            || self.enums.contains_key(name)
            || self.type_alias_bases.contains_key(name)
            || self.type_aliases.contains_key(name)
            || self.machines.contains_key(name)
            || self.actor_defs.contains_key(name)
    }

    fn substitute_type_expr_with_map(
        &self,
        ty: &TypeExpr,
        substitutions: &HashMap<String, TypeExpr>,
    ) -> TypeExpr {
        self.substitute_type_expr_with_map_in_namespace(
            ty,
            substitutions,
            self.current_namespace.as_deref(),
        )
    }

    fn substitute_type_expr_with_map_in_namespace(
        &self,
        ty: &TypeExpr,
        substitutions: &HashMap<String, TypeExpr>,
        namespace: Option<&str>,
    ) -> TypeExpr {
        match ty {
            TypeExpr::Named(ident) => {
                if let Some(bound) = substitutions
                    .get(&ident.name)
                    .cloned()
                    .or_else(|| self.current_type_binding(&ident.name))
                {
                    self.substitute_type_expr_with_map_in_namespace(
                        &bound,
                        substitutions,
                        namespace,
                    )
                } else {
                    TypeExpr::Named(self.expand_type_ident(ident, namespace))
                }
            }
            TypeExpr::Generic(ident, args, span) => TypeExpr::Generic(
                self.expand_type_ident(ident, namespace),
                args.iter()
                    .map(|arg| {
                        self.substitute_type_expr_with_map_in_namespace(
                            arg,
                            substitutions,
                            namespace,
                        )
                    })
                    .collect(),
                *span,
            ),
            TypeExpr::View(inner, span) => TypeExpr::View(
                Box::new(self.substitute_type_expr_with_map_in_namespace(
                    inner,
                    substitutions,
                    namespace,
                )),
                *span,
            ),
            TypeExpr::StateQualified(inner, state, span) => TypeExpr::StateQualified(
                Box::new(self.substitute_type_expr_with_map_in_namespace(
                    inner,
                    substitutions,
                    namespace,
                )),
                state.clone(),
                *span,
            ),
            TypeExpr::Function(params, return_type, span) => TypeExpr::Function(
                params
                    .iter()
                    .map(|param| {
                        self.substitute_type_expr_with_map_in_namespace(
                            param,
                            substitutions,
                            namespace,
                        )
                    })
                    .collect(),
                Box::new(self.substitute_type_expr_with_map_in_namespace(
                    return_type,
                    substitutions,
                    namespace,
                )),
                *span,
            ),
        }
    }

    fn type_expr_kind(&self, ty: &TypeExpr) -> &'static str {
        let ty = self.substitute_type_expr(ty);
        self.type_expr_kind_inner(&ty)
    }

    fn type_expr_kind_inner(&self, ty: &TypeExpr) -> &'static str {
        match ty {
            TypeExpr::Named(ident) => {
                if let Some(bound) = self.current_type_binding(&ident.name) {
                    return self.type_expr_kind_inner(&bound);
                }
                if self.type_aliases.contains_key(&ident.name) {
                    if self
                        .type_aliases
                        .get(&ident.name)
                        .is_some_and(|def| def.is_some())
                    {
                        "refinement"
                    } else {
                        "alias"
                    }
                } else if Self::is_builtin_type_name(&ident.name) {
                    "primitive"
                } else if self.structs.contains_key(&ident.name) {
                    "struct"
                } else if self.enums.contains_key(&ident.name) {
                    "enum"
                } else if self.bitfields.contains_key(&ident.name) {
                    "bitfield"
                } else if self.machines.contains_key(&ident.name) {
                    "machine"
                } else {
                    "named"
                }
            }
            TypeExpr::Generic(ident, _, _) => {
                if self.structs.contains_key(&ident.name) {
                    "struct"
                } else {
                    match ident.name.as_str() {
                        "list" => "list",
                        "map" => "map",
                        "set" => "set",
                        "optional" => "optional",
                        "result" => "result",
                        "secret" => "secret",
                        other if self.structs.contains_key(other) => "struct",
                        _ => "generic",
                    }
                }
            }
            TypeExpr::View(inner, _) => self.type_expr_kind_inner(inner),
            TypeExpr::StateQualified(_, _, _) => "machine_state",
            TypeExpr::Function(_, _, _) => "function",
        }
    }

    fn type_kind_tag_value(kind: &str) -> Value {
        let variant = match kind {
            "primitive" => "primitive_type",
            "alias" => "alias_type",
            "refinement" => "refinement_type",
            "struct" => "struct_type",
            "bitfield" => "bitfield_type",
            "enum" => "enum_type",
            "list" => "list_type",
            "set" => "set_type",
            "map" => "map_type",
            "optional" => "optional_type",
            "result" => "result_type",
            "secret" => "secret_type",
            "function" => "function_type",
            "machine" => "machine_type",
            "machine_state" => "machine_state_type",
            _ => "unknown_type",
        };
        Value::Enum {
            type_name: "TypeKind".to_string(),
            variant: variant.to_string(),
            fields: Vec::new(),
        }
    }

    fn type_primitive_tag_value(&self, ty: &TypeExpr) -> Value {
        let ty = self.substitute_type_expr(ty);
        if let TypeExpr::View(inner, _) = &ty {
            return self.type_primitive_tag_value(inner);
        }
        if let TypeExpr::StateQualified(_, _, _) = &ty {
            return Self::primitive_tag_value(None);
        }

        let variant = match &ty {
            TypeExpr::Named(ident) => match ident.name.as_str() {
                _ if self.type_aliases.contains_key(&ident.name) => None,
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
            },
            _ => None,
        };

        Self::primitive_tag_value(variant)
    }

    fn primitive_tag_value(variant: Option<&str>) -> Value {
        variant
            .map(|variant| {
                Value::OptionalSome(Box::new(Value::Enum {
                    type_name: "TypePrimitive".to_string(),
                    variant: variant.to_string(),
                    fields: Vec::new(),
                }))
            })
            .unwrap_or(Value::OptionalNone)
    }

    fn optional_string_value(value: Option<&str>) -> Value {
        value
            .map(|value| Value::OptionalSome(Box::new(Value::String(value.to_string()))))
            .unwrap_or(Value::OptionalNone)
    }

    fn bitfield_shape_tag_value(shape: &str) -> Value {
        let variant = match shape {
            "payload" => "payload_field",
            _ => "bits_field",
        };
        Value::Enum {
            type_name: "TypeBitfieldFieldShape".to_string(),
            variant: variant.to_string(),
            fields: Vec::new(),
        }
    }

    fn type_expr_has_secret(&self, ty: &TypeExpr) -> bool {
        let ty = self.substitute_type_expr(ty);
        self.type_expr_has_secret_inner(&ty, &mut HashSet::new())
    }

    fn type_expr_has_secret_inner(&self, ty: &TypeExpr, visited: &mut HashSet<String>) -> bool {
        match ty {
            TypeExpr::Named(ident) => {
                if let Some(bound) = self.current_type_binding(&ident.name) {
                    return self.type_expr_has_secret_inner(&bound, visited);
                }
                if let Some(base_ty) = self.type_alias_bases.get(&ident.name).cloned() {
                    return self
                        .type_expr_has_secret_inner(&self.substitute_type_expr(&base_ty), visited);
                }
                if !visited.insert(type_expr_display(ty)) {
                    return false;
                }
                if let Some(strukt) = self.structs.get(&ident.name) {
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    return strukt.fields.iter().any(|field| {
                        self.type_expr_has_secret_inner(
                            &self.substitute_type_expr_in_namespace(&field.ty, namespace),
                            visited,
                        )
                    });
                }
                if let Some(enum_def) = self.enums.get(&ident.name) {
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    return enum_def
                        .variants
                        .iter()
                        .flat_map(|variant| variant.fields.iter())
                        .any(|field| {
                            self.type_expr_has_secret_inner(
                                &self.substitute_type_expr_in_namespace(&field.ty, namespace),
                                visited,
                            )
                        });
                }
                if let Some(bitfield) = self.bitfields.get(&ident.name) {
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    return bitfield.fields.iter().any(|field| match &field.kind {
                        BitfieldFieldKind::Bits { as_type, .. } => {
                            as_type.as_ref().is_some_and(|ty| {
                                self.type_expr_has_secret_inner(
                                    &self.substitute_type_expr_in_namespace(ty, namespace),
                                    visited,
                                )
                            })
                        }
                        BitfieldFieldKind::Payload(ty) => self.type_expr_has_secret_inner(
                            &self.substitute_type_expr_in_namespace(ty, namespace),
                            visited,
                        ),
                    });
                }
                if let Some(machine) = self.machines.get(&ident.name) {
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    return machine
                        .states
                        .iter()
                        .flat_map(|state| state.fields.iter())
                        .any(|field| {
                            self.type_expr_has_secret_inner(
                                &self.substitute_type_expr_in_namespace(&field.ty, namespace),
                                visited,
                            )
                        });
                }
                false
            }
            TypeExpr::Generic(ident, args, _) => {
                if ident.name == "secret" {
                    return true;
                }
                if let Some(strukt) = self.structs.get(&ident.name) {
                    if !visited.insert(type_expr_display(ty)) {
                        return false;
                    }
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    let substitutions = self.generic_type_substitutions(strukt, args, namespace);
                    return strukt.fields.iter().any(|field| {
                        let field_ty = self.substitute_type_expr_with_map_in_namespace(
                            &field.ty,
                            &substitutions,
                            namespace,
                        );
                        self.type_expr_has_secret_inner(&field_ty, visited)
                    });
                }
                args.iter()
                    .any(|arg| self.type_expr_has_secret_inner(arg, visited))
            }
            TypeExpr::View(inner, _) => self.type_expr_has_secret_inner(inner, visited),
            TypeExpr::StateQualified(inner, state, _) => {
                let inner = self.substitute_type_expr(inner);
                if let TypeExpr::Named(ident) = &inner {
                    let qualified_name = self
                        .current_namespace
                        .as_deref()
                        .map(|namespace| format!("{namespace}.{}", ident.name));
                    let machine = self.machines.get(&ident.name).or_else(|| {
                        qualified_name
                            .as_ref()
                            .and_then(|name| self.machines.get(name))
                    });
                    if let Some(machine) = machine {
                        if !visited.insert(type_expr_display(ty)) {
                            return false;
                        }
                        let namespace = Self::type_name_namespace(&ident.name)
                            .or(self.current_namespace.as_deref());
                        return machine
                            .states
                            .iter()
                            .find(|candidate| candidate.name.name == state.name)
                            .is_some_and(|state_def| {
                                state_def.fields.iter().any(|field| {
                                    self.type_expr_has_secret_inner(
                                        &self.substitute_type_expr_in_namespace(
                                            &field.ty, namespace,
                                        ),
                                        visited,
                                    )
                                })
                            });
                    }
                }
                self.type_expr_has_secret_inner(&inner, visited)
            }
            TypeExpr::Function(params, return_type, _) => {
                params
                    .iter()
                    .any(|param| self.type_expr_has_secret_inner(param, visited))
                    || self.type_expr_has_secret_inner(return_type, visited)
            }
        }
    }

    fn type_expr_fields(&self, ty: &TypeExpr) -> Vec<ReflectionField> {
        let ty = self.substitute_type_expr(ty);
        self.type_expr_fields_inner(&ty)
    }

    fn type_expr_fields_inner(&self, ty: &TypeExpr) -> Vec<ReflectionField> {
        match ty {
            TypeExpr::Named(ident) => {
                if let Some(strukt) = self.structs.get(&ident.name) {
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    return strukt
                        .fields
                        .iter()
                        .map(|field| ReflectionField {
                            name: field.name.name.clone(),
                            ty: self.substitute_type_expr_in_namespace(&field.ty, namespace),
                            serialize_name: field
                                .serialize_name
                                .clone()
                                .unwrap_or_else(|| field.name.name.clone()),
                        })
                        .collect();
                }
                if let Some(bitfield) = self.bitfields.get(&ident.name) {
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    return bitfield
                        .fields
                        .iter()
                        .map(|field| ReflectionField {
                            name: field.name.name.clone(),
                            ty: match &field.kind {
                                BitfieldFieldKind::Bits { as_type, .. } => {
                                    as_type.clone().unwrap_or_else(|| {
                                        TypeExpr::Named(Ident {
                                            name: "int64".to_string(),
                                            span: field.span,
                                        })
                                    })
                                }
                                BitfieldFieldKind::Payload(ty) => {
                                    self.substitute_type_expr_in_namespace(ty, namespace)
                                }
                            },
                            serialize_name: field.name.name.clone(),
                        })
                        .collect();
                }
                Vec::new()
            }
            TypeExpr::Generic(ident, args, _) => self
                .structs
                .get(&ident.name)
                .map(|strukt| {
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    let substitutions = self.generic_type_substitutions(strukt, args, namespace);
                    strukt
                        .fields
                        .iter()
                        .map(|field| ReflectionField {
                            name: field.name.name.clone(),
                            ty: self.substitute_type_expr_with_map_in_namespace(
                                &field.ty,
                                &substitutions,
                                namespace,
                            ),
                            serialize_name: field
                                .serialize_name
                                .clone()
                                .unwrap_or_else(|| field.name.name.clone()),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            TypeExpr::View(inner, _) | TypeExpr::StateQualified(inner, _, _) => {
                self.type_expr_fields_inner(inner)
            }
            TypeExpr::Function(_, _, _) => Vec::new(),
        }
    }

    fn type_expr_bitfield(&self, ty: &TypeExpr) -> ReflectionBitfield {
        let ty = self.substitute_type_expr(ty);
        self.type_expr_bitfield_inner(&ty)
    }

    fn type_expr_bitfield_inner(&self, ty: &TypeExpr) -> ReflectionBitfield {
        match ty {
            TypeExpr::Named(ident) => self
                .bitfields
                .get(&ident.name)
                .map(|bitfield| ReflectionBitfield {
                    network_order: bitfield.network_order,
                    fields: self.reflection_bitfield_fields(
                        bitfield,
                        Self::type_name_namespace(&ident.name)
                            .or(self.current_namespace.as_deref()),
                    ),
                })
                .unwrap_or_else(|| ReflectionBitfield {
                    network_order: false,
                    fields: Vec::new(),
                }),
            TypeExpr::View(inner, _) | TypeExpr::StateQualified(inner, _, _) => {
                self.type_expr_bitfield_inner(inner)
            }
            TypeExpr::Generic(_, _, _) | TypeExpr::Function(_, _, _) => ReflectionBitfield {
                network_order: false,
                fields: Vec::new(),
            },
        }
    }

    fn type_expr_bitfield_fields(&self, ty: &TypeExpr) -> Vec<ReflectionBitfieldField> {
        self.type_expr_bitfield(ty).fields
    }

    fn reflection_bitfield_fields(
        &self,
        bitfield: &BitfieldDef,
        namespace: Option<&str>,
    ) -> Vec<ReflectionBitfieldField> {
        bitfield
            .fields
            .iter()
            .map(|field| match &field.kind {
                BitfieldFieldKind::Bits { width, as_type } => {
                    let ty = as_type.clone().unwrap_or_else(|| {
                        TypeExpr::Named(Ident {
                            name: "int64".to_string(),
                            span: field.span,
                        })
                    });
                    ReflectionBitfieldField {
                        name: field.name.name.clone(),
                        shape: "bits".to_string(),
                        width: *width as i64,
                        ty: self.substitute_type_expr_in_namespace(&ty, namespace),
                        enum_ty: as_type
                            .as_ref()
                            .map(|ty| self.substitute_type_expr_in_namespace(ty, namespace)),
                    }
                }
                BitfieldFieldKind::Payload(ty) => ReflectionBitfieldField {
                    name: field.name.name.clone(),
                    shape: "payload".to_string(),
                    width: 0,
                    ty: self.substitute_type_expr_in_namespace(ty, namespace),
                    enum_ty: None,
                },
            })
            .collect()
    }

    fn type_expr_machine(&self, ty: &TypeExpr) -> ReflectionMachine {
        let ty = self.substitute_type_expr(ty);
        self.type_expr_machine_inner(&ty)
    }

    fn type_expr_machine_inner(&self, ty: &TypeExpr) -> ReflectionMachine {
        match ty {
            TypeExpr::Named(ident) => {
                let machine_name = encoded_machine_base_name(&ident.name);
                self.machines
                    .get(machine_name)
                    .map(|machine| {
                        let namespace = Self::type_name_namespace(machine_name)
                            .or(self.current_namespace.as_deref());
                        let states = machine
                            .states
                            .iter()
                            .map(|state| ReflectionMachineState {
                                name: state.name.name.clone(),
                                fields: state
                                    .fields
                                    .iter()
                                    .map(|field| ReflectionField {
                                        name: field.name.name.clone(),
                                        ty: self.substitute_type_expr_in_namespace(
                                            &field.ty, namespace,
                                        ),
                                        serialize_name: field
                                            .serialize_name
                                            .clone()
                                            .unwrap_or_else(|| field.name.name.clone()),
                                    })
                                    .collect(),
                            })
                            .collect::<Vec<_>>();
                        let edges = machine
                            .transitions
                            .iter()
                            .filter_map(|transition| {
                                let source_index = states
                                    .iter()
                                    .position(|state| state.name == transition.from.name)?;
                                let target_index = states
                                    .iter()
                                    .position(|state| state.name == transition.to.name)?;
                                Some(ReflectionMachineTransition {
                                    source_index,
                                    source: transition.from.name.clone(),
                                    target_index,
                                    target: transition.to.name.clone(),
                                })
                            })
                            .collect();
                        ReflectionMachine { states, edges }
                    })
                    .unwrap_or_else(|| ReflectionMachine {
                        states: Vec::new(),
                        edges: Vec::new(),
                    })
            }
            TypeExpr::View(inner, _) | TypeExpr::StateQualified(inner, _, _) => {
                self.type_expr_machine_inner(inner)
            }
            TypeExpr::Generic(_, _, _) | TypeExpr::Function(_, _, _) => ReflectionMachine {
                states: Vec::new(),
                edges: Vec::new(),
            },
        }
    }

    fn type_machine_value(&self, owner_type: &str, machine: ReflectionMachine) -> Value {
        let states = machine
            .states
            .into_iter()
            .enumerate()
            .map(|(index, state)| self.type_machine_state_value(index, owner_type, state))
            .collect();
        let edges = machine
            .edges
            .into_iter()
            .enumerate()
            .map(|(index, transition)| Self::type_machine_transition_value(index, transition))
            .collect();
        Value::Struct {
            type_name: "TypeMachine".to_string(),
            fields: vec![
                ("states".to_string(), Value::List(states)),
                ("edges".to_string(), Value::List(edges)),
            ],
        }
    }

    fn type_machine_state_value(
        &self,
        index: usize,
        owner_type: &str,
        state: ReflectionMachineState,
    ) -> Value {
        let has_secret = state
            .fields
            .iter()
            .any(|field| self.type_expr_has_secret(&field.ty));
        let owner_member = state.name.clone();
        let fields = state
            .fields
            .into_iter()
            .enumerate()
            .map(|(index, field)| {
                self.type_field_value(index, owner_type, Some(&owner_member), field)
            })
            .collect();
        Value::Struct {
            type_name: "TypeMachineState".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(index as i64)),
                (
                    "owner_type".to_string(),
                    Value::String(owner_type.to_string()),
                ),
                ("name".to_string(), Value::String(state.name)),
                ("has_secret".to_string(), Value::Bool(has_secret)),
                ("fields".to_string(), Value::List(fields)),
            ],
        }
    }

    fn type_machine_transition_value(
        index: usize,
        transition: ReflectionMachineTransition,
    ) -> Value {
        Value::Struct {
            type_name: "TypeMachineTransition".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(index as i64)),
                (
                    "source_index".to_string(),
                    Value::Int64(transition.source_index as i64),
                ),
                ("source".to_string(), Value::String(transition.source)),
                (
                    "target_index".to_string(),
                    Value::Int64(transition.target_index as i64),
                ),
                ("target".to_string(), Value::String(transition.target)),
            ],
        }
    }

    fn type_expr_variants(&self, ty: &TypeExpr) -> Vec<ReflectionVariant> {
        let ty = self.substitute_type_expr(ty);
        self.type_expr_variants_inner(&ty)
    }

    fn type_expr_variants_inner(&self, ty: &TypeExpr) -> Vec<ReflectionVariant> {
        match ty {
            TypeExpr::Named(ident) => self
                .enums
                .get(&ident.name)
                .map(|enum_def| {
                    let namespace = Self::type_name_namespace(&ident.name)
                        .or(self.current_namespace.as_deref());
                    let mut next_discriminant = 0_i64;
                    enum_def
                        .variants
                        .iter()
                        .map(|variant| {
                            let discriminant = variant.discriminant.unwrap_or(next_discriminant);
                            next_discriminant = discriminant.saturating_add(1);
                            ReflectionVariant {
                                name: variant.name.name.clone(),
                                discriminant,
                                fields: variant
                                    .fields
                                    .iter()
                                    .map(|field| ReflectionField {
                                        name: field.name.name.clone(),
                                        ty: self.substitute_type_expr_in_namespace(
                                            &field.ty, namespace,
                                        ),
                                        serialize_name: field
                                            .serialize_name
                                            .clone()
                                            .unwrap_or_else(|| field.name.name.clone()),
                                    })
                                    .collect(),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default(),
            TypeExpr::View(inner, _) | TypeExpr::StateQualified(inner, _, _) => {
                self.type_expr_variants_inner(inner)
            }
            TypeExpr::Generic(_, _, _) | TypeExpr::Function(_, _, _) => Vec::new(),
        }
    }

    fn generic_type_substitutions(
        &self,
        strukt: &StructDef,
        args: &[TypeExpr],
        namespace: Option<&str>,
    ) -> HashMap<String, TypeExpr> {
        strukt
            .type_params
            .iter()
            .zip(args.iter())
            .map(|(param, arg)| {
                (
                    param.name.clone(),
                    self.substitute_type_expr_in_namespace(arg, namespace),
                )
            })
            .collect()
    }

    fn type_name_namespace(name: &str) -> Option<&str> {
        name.rsplit_once('.').map(|(namespace, _)| namespace)
    }

    fn type_field_value(
        &self,
        index: usize,
        owner_type: &str,
        owner_member: Option<&str>,
        field: ReflectionField,
    ) -> Value {
        let kind = self.type_expr_kind(&field.ty).to_string();
        let kind_tag = Self::type_kind_tag_value(&kind);
        let has_secret = self.type_expr_has_secret(&field.ty);
        let type_info = self.type_info_value(&field.ty);
        Value::Struct {
            type_name: "TypeField".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(index as i64)),
                (
                    "owner_type".to_string(),
                    Value::String(owner_type.to_string()),
                ),
                (
                    "owner_member".to_string(),
                    Self::optional_string_value(owner_member),
                ),
                ("name".to_string(), Value::String(field.name)),
                (
                    "type_name".to_string(),
                    Value::String(type_expr_display(&field.ty)),
                ),
                ("kind".to_string(), Value::String(kind)),
                ("kind_tag".to_string(), kind_tag),
                (
                    "serialize_name".to_string(),
                    Value::String(field.serialize_name),
                ),
                ("has_secret".to_string(), Value::Bool(has_secret)),
                ("type_info".to_string(), type_info),
            ],
        }
    }

    fn reflected_field_loop_bindings(
        &self,
        iterable: &Expr,
    ) -> Result<Option<Vec<ReflectedFieldBinding>>, String> {
        let Some(owner_ty) = comptime_type_fields_binding(iterable) else {
            return Ok(None);
        };
        let owner_ty = self.substitute_type_expr(owner_ty);
        let owner_type = type_expr_display(&owner_ty);
        if let Some(fields) = self.checked_type_fields(&owner_ty) {
            return Ok(Some(
                fields
                    .iter()
                    .map(|field| ReflectedFieldBinding {
                        index: field.index,
                        owner_type: owner_type.clone(),
                        owner_member: None,
                        name: field.name.clone(),
                        ty: Self::reflection_type_info_type_expr(&field.type_info),
                    })
                    .collect(),
            ));
        }
        if self.checked_metadata_kind_is(&owner_ty, &["struct", "bitfield"]) {
            return Err(self.missing_checked_metadata_error(&owner_ty, "field"));
        }

        Ok(Some(
            self.type_expr_fields(&owner_ty)
                .into_iter()
                .enumerate()
                .map(|(index, field)| ReflectedFieldBinding {
                    index,
                    owner_type: owner_type.clone(),
                    owner_member: None,
                    name: field.name,
                    ty: field.ty,
                })
                .collect(),
        ))
    }

    fn reflected_variant_loop_bindings(
        &self,
        iterable: &Expr,
    ) -> Result<Option<Vec<ReflectedVariantBinding>>, String> {
        let Some(owner_ty) = comptime_type_variants_binding(iterable) else {
            return Ok(None);
        };
        let owner_ty = self.substitute_type_expr(owner_ty);
        let owner_type = type_expr_display(&owner_ty);
        if let Some(variants) = self.checked_type_variants(&owner_ty) {
            return Ok(Some(
                variants
                    .iter()
                    .map(|variant| ReflectedVariantBinding {
                        ty: owner_ty.clone(),
                        index: variant.index,
                        owner_type: owner_type.clone(),
                        name: variant.name.clone(),
                        discriminant: variant.discriminant,
                    })
                    .collect(),
            ));
        }
        if self.checked_metadata_kind_is(&owner_ty, &["enum"]) {
            return Err(self.missing_checked_metadata_error(&owner_ty, "variant"));
        }

        Ok(Some(
            self.type_expr_variants(&owner_ty)
                .into_iter()
                .enumerate()
                .map(|(index, variant)| ReflectedVariantBinding {
                    ty: owner_ty.clone(),
                    index,
                    owner_type: owner_type.clone(),
                    name: variant.name,
                    discriminant: variant.discriminant,
                })
                .collect(),
        ))
    }

    fn reflected_machine_state_loop_bindings(
        &self,
        iterable: &Expr,
    ) -> Result<Option<Vec<ReflectedMachineStateBinding>>, String> {
        let Some(owner_ty) = comptime_type_machine_states_binding(iterable) else {
            return Ok(None);
        };
        let owner_ty = self.substitute_type_expr(owner_ty);
        let owner_type = type_expr_name(&owner_ty);
        if let Some(machine) = self.checked_machine(&owner_ty) {
            return Ok(Some(
                machine
                    .states
                    .iter()
                    .map(|state| ReflectedMachineStateBinding {
                        ty: owner_ty.clone(),
                        index: state.index,
                        owner_type: owner_type.clone(),
                        name: state.name.clone(),
                    })
                    .collect(),
            ));
        }
        if self.checked_metadata_kind_is(&owner_ty, &["machine", "machine_state"]) {
            return Err(self.missing_checked_metadata_error(&owner_ty, "machine"));
        }

        Ok(Some(
            self.type_expr_machine(&owner_ty)
                .states
                .into_iter()
                .enumerate()
                .map(|(index, state)| ReflectedMachineStateBinding {
                    ty: owner_ty.clone(),
                    index,
                    owner_type: owner_type.clone(),
                    name: state.name,
                })
                .collect(),
        ))
    }

    fn reflected_variant_field_loop_owner(
        &self,
        iterable: &Expr,
    ) -> Result<Option<ReflectedVariantFieldOwner>, String> {
        if let Some(ty) = comptime_type_variant_fields_binding(iterable) {
            return Ok(Some(ReflectedVariantFieldOwner {
                ty: self.substitute_type_expr(ty),
                variant: None,
            }));
        }

        let Some(variant_name) = reflected_variant_fields_binding(iterable) else {
            return Ok(None);
        };
        let Some(binding) = self.maybe_bound_reflected_variant(variant_name)? else {
            return Ok(None);
        };
        Ok(Some(ReflectedVariantFieldOwner {
            ty: binding.ty,
            variant: Some(binding.name),
        }))
    }

    fn reflected_machine_field_loop_owner(
        &self,
        iterable: &Expr,
    ) -> Result<Option<ReflectedMachineFieldOwner>, String> {
        if let Some(ty) = comptime_type_machine_fields_binding(iterable) {
            return Ok(Some(ReflectedMachineFieldOwner {
                ty: self.substitute_type_expr(ty),
            }));
        }

        let Some(state_name) = reflected_machine_state_fields_binding(iterable) else {
            return Ok(None);
        };
        let Some(binding) = self.maybe_bound_reflected_machine_state(state_name)? else {
            return Ok(None);
        };
        Ok(Some(ReflectedMachineFieldOwner { ty: binding.ty }))
    }

    fn reflected_variant_field_binding_for_value(
        &self,
        owner_ty: &TypeExpr,
        selected_variant: Option<&str>,
        field_metadata: &Value,
    ) -> Result<ReflectedFieldBinding, String> {
        let metadata = Self::type_field_metadata_for(field_metadata, "type.field_value")?;
        let expected_owner = type_expr_display(owner_ty);
        if metadata.owner_type != expected_owner
            || selected_variant
                .is_some_and(|selected| metadata.owner_member.as_deref() != Some(selected))
        {
            return Err(format!(
                "`comptime type`: field metadata belongs to '{}', expected '{}'",
                Self::field_metadata_owner_label(&metadata),
                Self::field_owner_label(&expected_owner, selected_variant)
            ));
        }
        let Some(metadata_owner_member) = metadata.owner_member.as_deref() else {
            return Err(format!(
                "`comptime type`: field metadata belongs to '{}', expected payload field metadata for '{}'",
                Self::field_metadata_owner_label(&metadata),
                expected_owner
            ));
        };
        if let Some(variants) = self.checked_type_variants(owner_ty) {
            for variant in variants {
                if matches!(selected_variant, Some(selected) if selected != variant.name) {
                    continue;
                }
                if metadata_owner_member != variant.name {
                    continue;
                }
                let Some(field) = variant.fields.get(metadata.index) else {
                    continue;
                };
                if field.name == metadata.name && field.type_name == metadata.type_name {
                    return Ok(ReflectedFieldBinding {
                        index: metadata.index,
                        owner_type: expected_owner.clone(),
                        owner_member: metadata.owner_member.clone(),
                        name: field.name.clone(),
                        ty: Self::reflection_type_info_type_expr(&field.type_info),
                    });
                }
            }
            return Err(format!(
                "`comptime type` reflected enum field metadata '{}' does not match any payload field on type '{}'",
                metadata.name,
                type_expr_display(owner_ty)
            ));
        }
        if self.checked_metadata_kind_is(owner_ty, &["enum"]) {
            return Err(self.missing_checked_metadata_error(owner_ty, "variant"));
        }

        for variant in self.type_expr_variants(owner_ty) {
            if matches!(selected_variant, Some(selected) if selected != variant.name) {
                continue;
            }
            if metadata_owner_member != variant.name {
                continue;
            }
            let Some(field) = variant.fields.get(metadata.index) else {
                continue;
            };
            if field.name == metadata.name && type_expr_display(&field.ty) == metadata.type_name {
                return Ok(ReflectedFieldBinding {
                    index: metadata.index,
                    owner_type: expected_owner.clone(),
                    owner_member: metadata.owner_member.clone(),
                    name: field.name.clone(),
                    ty: field.ty.clone(),
                });
            }
        }
        Err(format!(
            "`comptime type` reflected enum field metadata '{}' does not match any payload field on type '{}'",
            metadata.name,
            type_expr_display(owner_ty)
        ))
    }

    fn reflected_machine_field_binding_for_value(
        &self,
        owner_ty: &TypeExpr,
        field_metadata: &Value,
    ) -> Result<ReflectedFieldBinding, String> {
        let metadata = Self::type_field_metadata_for(field_metadata, "type.machine_field_value")?;
        let expected_owner = type_expr_name(owner_ty);
        if metadata.owner_type != expected_owner {
            return Err(format!(
                "`comptime type`: field metadata belongs to '{}', expected '{}'",
                Self::field_metadata_owner_label(&metadata),
                expected_owner
            ));
        }
        let Some(metadata_owner_member) = metadata.owner_member.as_deref() else {
            return Err(format!(
                "`comptime type`: field metadata belongs to '{}', expected state payload field metadata for '{}'",
                Self::field_metadata_owner_label(&metadata),
                expected_owner
            ));
        };
        if let Some(machine) = self.checked_machine(owner_ty) {
            for state in &machine.states {
                if metadata_owner_member != state.name {
                    continue;
                }
                let Some(field) = state.fields.get(metadata.index) else {
                    continue;
                };
                if field.name == metadata.name && field.type_name == metadata.type_name {
                    return Ok(ReflectedFieldBinding {
                        index: metadata.index,
                        owner_type: expected_owner.clone(),
                        owner_member: metadata.owner_member.clone(),
                        name: field.name.clone(),
                        ty: Self::reflection_type_info_type_expr(&field.type_info),
                    });
                }
            }
            return Err(format!(
                "`comptime type` reflected machine field metadata '{}' does not match any payload field on type '{}'",
                metadata.name,
                type_expr_display(owner_ty)
            ));
        }
        if self.checked_metadata_kind_is(owner_ty, &["machine", "machine_state"]) {
            return Err(self.missing_checked_metadata_error(owner_ty, "machine"));
        }

        for state in self.type_expr_machine(owner_ty).states {
            if metadata_owner_member != state.name {
                continue;
            }
            let Some(field) = state.fields.get(metadata.index) else {
                continue;
            };
            if field.name == metadata.name && type_expr_display(&field.ty) == metadata.type_name {
                return Ok(ReflectedFieldBinding {
                    index: metadata.index,
                    owner_type: expected_owner.clone(),
                    owner_member: metadata.owner_member.clone(),
                    name: field.name.clone(),
                    ty: field.ty.clone(),
                });
            }
        }
        Err(format!(
            "`comptime type` reflected machine field metadata '{}' does not match any payload field on type '{}'",
            metadata.name,
            type_expr_display(owner_ty)
        ))
    }

    fn maybe_bound_reflected_variant(
        &self,
        variant_name: &str,
    ) -> Result<Option<ReflectedVariantBinding>, String> {
        let binding = self
            .reflected_variant_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(variant_name));
        let Some(binding) = binding else {
            return Ok(None);
        };
        let current_value = self.get_variable(variant_name).ok_or_else(|| {
            format!("`comptime type` reflected variant '{variant_name}' is not in scope")
        })?;
        let (index, owner_type, name, discriminant) =
            Self::type_variant_metadata(current_value, "`comptime type`")?;
        if index != binding.index
            || owner_type != binding.owner_type
            || name != binding.name
            || discriminant != binding.discriminant
        {
            return Err("`comptime type` reflected variant metadata no longer matches the trusted `type.variants[T]()` loop item".to_string());
        }
        Ok(Some(binding.clone()))
    }

    fn maybe_bound_reflected_machine_state(
        &self,
        state_name: &str,
    ) -> Result<Option<ReflectedMachineStateBinding>, String> {
        let binding = self
            .reflected_machine_state_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(state_name));
        let Some(binding) = binding else {
            return Ok(None);
        };
        let current_value = self.get_variable(state_name).ok_or_else(|| {
            format!("`comptime type` reflected machine state '{state_name}' is not in scope")
        })?;
        let (index, owner_type, name) =
            Self::type_machine_state_metadata(current_value, "`comptime type`")?;
        if index != binding.index || owner_type != binding.owner_type || name != binding.name {
            return Err("`comptime type` reflected machine state metadata no longer matches the trusted `type.machine_states[T]()` loop item".to_string());
        }
        Ok(Some(binding.clone()))
    }

    fn reflected_type_info_arg_loop_bindings(
        &self,
        iterable: &Expr,
    ) -> Result<Option<Vec<ReflectedTypeInfoBinding>>, String> {
        let Some(source_ty) = reflected_type_info_args_source(iterable) else {
            return Ok(None);
        };
        let ty = match source_ty {
            ReflectedTypeInfoSource::Direct(ty) => self.substitute_type_expr(ty),
            ReflectedTypeInfoSource::Field(field_name) => {
                let Some(ty) = self.maybe_bound_reflected_field_type(field_name)? else {
                    return Ok(None);
                };
                ty
            }
            ReflectedTypeInfoSource::TypeInfo(info_name) => {
                let Some(ty) = self.maybe_bound_reflected_type_info_type(info_name)? else {
                    return Ok(None);
                };
                ty
            }
        };
        Ok(Some(
            self.checked_type_info_arg_types(&ty)
                .unwrap_or_else(|| self.type_info_arg_types(&ty))
                .into_iter()
                .map(|ty| ReflectedTypeInfoBinding { ty })
                .collect(),
        ))
    }

    fn bound_reflected_field_type(&self, field_name: &str) -> Result<TypeExpr, String> {
        self.maybe_bound_reflected_field_type(field_name)?
            .ok_or_else(|| {
                "`comptime type` can only bind field.type_info inside direct trusted reflection field loops"
                    .to_string()
            })
    }

    fn maybe_bound_reflected_field_type(
        &self,
        field_name: &str,
    ) -> Result<Option<TypeExpr>, String> {
        let binding = self
            .reflected_field_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(field_name));
        let Some(binding) = binding else {
            return Ok(None);
        };
        let current_value = self.get_variable(field_name).ok_or_else(|| {
            format!("`comptime type` reflected field '{field_name}' is not in scope")
        })?;
        let metadata = Self::type_field_metadata(current_value).map_err(|_| {
            "`comptime type` reflected field binding requires the current TypeField value"
                .to_string()
        })?;
        let expected_type_name = type_expr_display(&binding.ty);
        if metadata.index != binding.index
            || metadata.owner_type != binding.owner_type
            || metadata.owner_member != binding.owner_member
            || metadata.name != binding.name
            || metadata.type_name != expected_type_name
        {
            return Err(format!(
                "`comptime type` reflected field metadata no longer matches the trusted `type.fields[T]()` loop item (expected #{} {}: {}, got #{} {}: {})",
                binding.index,
                binding.name,
                expected_type_name,
                metadata.index,
                metadata.name,
                metadata.type_name
            ));
        }
        Ok(Some(self.substitute_type_expr(&binding.ty)))
    }

    fn bound_reflected_type_info_type(&self, info_name: &str) -> Result<TypeExpr, String> {
        self.maybe_bound_reflected_type_info_type(info_name)?
            .ok_or_else(|| {
                "`comptime type` can only bind TypeInfo values from trusted reflected `args` loops"
                    .to_string()
            })
    }

    fn maybe_bound_reflected_type_info_type(
        &self,
        info_name: &str,
    ) -> Result<Option<TypeExpr>, String> {
        let binding = self
            .reflected_type_info_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(info_name));
        let Some(binding) = binding else {
            return Ok(None);
        };
        let current_value = self.get_variable(info_name).ok_or_else(|| {
            format!("`comptime type` reflected TypeInfo '{info_name}' is not in scope")
        })?;
        let type_name = Self::type_info_metadata(current_value).map_err(|_| {
            "`comptime type` reflected TypeInfo binding requires the current TypeInfo value"
                .to_string()
        })?;
        let expected_type_name = type_expr_display(&binding.ty);
        if type_name != expected_type_name {
            return Err("`comptime type` reflected TypeInfo metadata no longer matches the trusted `args` loop item".to_string());
        }
        Ok(Some(self.substitute_type_expr(&binding.ty)))
    }

    fn type_info_arg_types(&self, ty: &TypeExpr) -> Vec<TypeExpr> {
        let ty = self.substitute_type_expr(ty);
        if let TypeExpr::View(inner, _) = &ty {
            return self.type_info_arg_types(inner);
        }

        match &ty {
            TypeExpr::Named(ident) if self.type_aliases.contains_key(&ident.name) => self
                .type_alias_bases
                .get(&ident.name)
                .map(|base_ty| vec![self.substitute_type_expr(base_ty)])
                .unwrap_or_default(),
            TypeExpr::Generic(_, args, _) => args
                .iter()
                .map(|arg| self.substitute_type_expr(arg))
                .collect(),
            TypeExpr::Function(params, return_type, _) => params
                .iter()
                .chain(std::iter::once(return_type.as_ref()))
                .map(|arg| self.substitute_type_expr(arg))
                .collect(),
            _ => Vec::new(),
        }
    }

    fn reflection_compare_type_name(&self, type_name: &str) -> String {
        self.reflection_compare_type_name_inner(type_name, &mut HashSet::new())
    }

    fn reflection_compare_type_name_inner(
        &self,
        type_name: &str,
        visited: &mut HashSet<String>,
    ) -> String {
        if !visited.insert(type_name.to_string()) {
            return type_name.to_string();
        }

        if let Some(base_ty) = self.type_alias_bases.get(type_name) {
            let namespace =
                Self::type_name_namespace(type_name).or(self.current_namespace.as_deref());
            let base_ty = self.substitute_type_expr_in_namespace(base_ty, namespace);
            return self.reflection_compare_type_expr_inner(&base_ty, visited);
        }

        if let Some((owner, args)) = Self::split_generic_type_display(type_name) {
            let owner = self.reflection_compare_type_name_inner(owner, visited);
            let args = Self::split_type_display_args(args)
                .into_iter()
                .map(|arg| self.reflection_compare_type_name_inner(arg.trim(), visited))
                .collect::<Vec<_>>();
            return format!("{owner}[{}]", args.join(", "));
        }

        type_name.to_string()
    }

    fn reflection_compare_type_expr(&self, ty: &TypeExpr) -> String {
        let ty = self.substitute_type_expr(ty);
        self.reflection_compare_type_expr_inner(&ty, &mut HashSet::new())
    }

    fn reflection_compare_type_expr_inner(
        &self,
        ty: &TypeExpr,
        visited: &mut HashSet<String>,
    ) -> String {
        match ty {
            TypeExpr::Named(ident) => self.reflection_compare_type_name_inner(&ident.name, visited),
            TypeExpr::Generic(ident, args, _) => {
                let owner = self.reflection_compare_type_name_inner(&ident.name, visited);
                let args = args
                    .iter()
                    .map(|arg| self.reflection_compare_type_expr_inner(arg, visited))
                    .collect::<Vec<_>>();
                format!("{owner}[{}]", args.join(", "))
            }
            TypeExpr::View(inner, _) => {
                format!(
                    "view {}",
                    self.reflection_compare_type_expr_inner(inner, visited)
                )
            }
            TypeExpr::StateQualified(inner, state, _) => {
                format!(
                    "{} at {}",
                    self.reflection_compare_type_expr_inner(inner, visited),
                    state.name
                )
            }
            TypeExpr::Function(params, return_type, _) => {
                let params = params
                    .iter()
                    .map(|param| self.reflection_compare_type_expr_inner(param, visited))
                    .collect::<Vec<_>>();
                format!(
                    "function({}) returns {}",
                    params.join(", "),
                    self.reflection_compare_type_expr_inner(return_type, visited)
                )
            }
        }
    }

    fn reflection_type_names_match(&self, actual: &str, expected: &str) -> bool {
        self.reflection_compare_type_name(actual) == self.reflection_compare_type_name(expected)
    }

    fn reflection_type_name_matches_expr(&self, actual: &str, expected: &TypeExpr) -> bool {
        self.reflection_compare_type_name(actual) == self.reflection_compare_type_expr(expected)
    }

    fn split_generic_type_display(type_name: &str) -> Option<(&str, &str)> {
        let open = type_name.find('[')?;
        if !type_name.ends_with(']') {
            return None;
        }
        let owner = &type_name[..open];
        if owner.is_empty() {
            return None;
        }

        let mut depth = 0_i32;
        for (index, ch) in type_name.char_indices().skip(open) {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 && index != type_name.len() - 1 {
                        return None;
                    }
                }
                _ => {}
            }
            if depth < 0 {
                return None;
            }
        }
        (depth == 0).then_some((owner, &type_name[open + 1..type_name.len() - 1]))
    }

    fn split_type_display_args(args: &str) -> Vec<&str> {
        let mut parts = Vec::new();
        let mut start = 0;
        let mut depth = 0_i32;
        for (index, ch) in args.char_indices() {
            match ch {
                '[' | '(' => depth += 1,
                ']' | ')' => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(&args[start..index]);
                    start = index + 1;
                }
                _ => {}
            }
        }
        parts.push(&args[start..]);
        parts
    }

    fn type_bitfield_value(&self, bitfield: ReflectionBitfield) -> Value {
        let fields = bitfield
            .fields
            .into_iter()
            .enumerate()
            .map(|(index, field)| self.type_bitfield_field_value(index, field))
            .collect();
        Value::Struct {
            type_name: "TypeBitfield".to_string(),
            fields: vec![
                (
                    "network_order".to_string(),
                    Value::Bool(bitfield.network_order),
                ),
                ("fields".to_string(), Value::List(fields)),
            ],
        }
    }

    fn type_bitfield_field_value(&self, index: usize, field: ReflectionBitfieldField) -> Value {
        let enum_type = field
            .enum_ty
            .as_ref()
            .map(|ty| Value::OptionalSome(Box::new(self.type_info_value(ty))))
            .unwrap_or(Value::OptionalNone);
        Value::Struct {
            type_name: "TypeBitfieldField".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(index as i64)),
                ("name".to_string(), Value::String(field.name)),
                ("shape".to_string(), Value::String(field.shape.clone())),
                (
                    "shape_tag".to_string(),
                    Self::bitfield_shape_tag_value(&field.shape),
                ),
                ("width".to_string(), Value::Int64(field.width)),
                ("type_info".to_string(), self.type_info_value(&field.ty)),
                ("enum_type".to_string(), enum_type),
            ],
        }
    }

    fn type_variant_value(
        &self,
        index: usize,
        owner_type: &str,
        variant: ReflectionVariant,
    ) -> Value {
        let has_secret = variant
            .fields
            .iter()
            .any(|field| self.type_expr_has_secret(&field.ty));
        let owner_member = variant.name.clone();
        let fields = variant
            .fields
            .into_iter()
            .enumerate()
            .map(|(index, field)| {
                self.type_field_value(index, owner_type, Some(&owner_member), field)
            })
            .collect::<Vec<_>>();

        Value::Struct {
            type_name: "TypeVariant".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(index as i64)),
                (
                    "owner_type".to_string(),
                    Value::String(owner_type.to_string()),
                ),
                ("name".to_string(), Value::String(variant.name)),
                (
                    "discriminant".to_string(),
                    Value::Int64(variant.discriminant),
                ),
                ("has_secret".to_string(), Value::Bool(has_secret)),
                ("fields".to_string(), Value::List(fields)),
            ],
        }
    }

    fn checked_type_info(&self, ty: &TypeExpr) -> Option<&ReflectionTypeInfo> {
        let metadata = self.reflection_metadata.as_ref()?;
        let type_name = type_expr_display(ty);
        metadata.get_type_info(&type_name)
    }

    fn checked_type_has_secret(&self, ty: &TypeExpr) -> Option<bool> {
        Some(self.checked_type_info(ty)?.has_secret)
    }

    fn checked_type_kind(&self, ty: &TypeExpr) -> Option<&str> {
        Some(self.checked_type_info(ty)?.kind.as_str())
    }

    fn checked_metadata_kind_is(&self, ty: &TypeExpr, kinds: &[&str]) -> bool {
        self.checked_type_kind(ty)
            .is_some_and(|kind| kinds.contains(&kind))
    }

    fn missing_checked_metadata_error(&self, ty: &TypeExpr, metadata: &str) -> String {
        format!(
            "checked reflection metadata for type '{}' is missing {metadata} metadata",
            type_expr_display(ty)
        )
    }

    fn missing_checked_type_info_error(&self, ty: &TypeExpr) -> Option<String> {
        let metadata = self.reflection_metadata.as_ref()?;
        let type_name = type_expr_display(ty);
        if metadata.type_id_for_name(&type_name).is_some()
            && metadata.get_type_info(&type_name).is_none()
        {
            return Some(self.missing_checked_metadata_error(ty, "type info"));
        }
        None
    }

    fn checked_construction_kind(&self, ty: &TypeExpr) -> Option<&'static str> {
        if let Some(kind) = self.checked_type_kind(ty) {
            return match kind {
                "struct" => Some("struct"),
                "bitfield" => Some("bitfield"),
                "enum" => Some("enum"),
                "machine" => Some("machine"),
                "machine_state" => Some("machine_state"),
                _ => None,
            };
        }

        if self.checked_type_variants(ty).is_some() {
            Some("enum")
        } else if self.checked_bitfield(ty).is_some() {
            Some("bitfield")
        } else if self.checked_machine(ty).is_some() {
            match ty {
                TypeExpr::StateQualified(_, _, _) => Some("machine_state"),
                TypeExpr::Named(ident) if encoded_machine_state_name(&ident.name).is_some() => {
                    Some("machine_state")
                }
                _ => Some("machine"),
            }
        } else if self.checked_type_fields(ty).is_some() {
            Some("struct")
        } else {
            None
        }
    }

    fn checked_type_info_value(&self, ty: &TypeExpr) -> Option<Value> {
        self.checked_type_info(ty)
            .map(Self::reflection_type_info_value)
    }

    fn checked_type_arg_value(&self, ty: &TypeExpr, index: usize) -> Option<Result<Value, String>> {
        let info = self.checked_type_info(ty)?;
        Some(
            info.args
                .get(index)
                .map(Self::reflection_type_info_value)
                .ok_or_else(|| {
                    format!(
                        "type.arg index {index} is out of range for type '{}'",
                        type_expr_display(ty)
                    )
                }),
        )
    }

    fn checked_type_info_arg_types(&self, ty: &TypeExpr) -> Option<Vec<TypeExpr>> {
        let info = self.checked_type_info(ty)?;
        Some(
            info.args
                .iter()
                .map(Self::reflection_type_info_type_expr)
                .collect(),
        )
    }

    fn reflection_type_info_type_expr(info: &ReflectionTypeInfo) -> TypeExpr {
        let span = Self::reflection_type_span();
        if info.kind == "function" && !info.args.is_empty() {
            let mut args = info
                .args
                .iter()
                .map(Self::reflection_type_info_type_expr)
                .collect::<Vec<_>>();
            let return_type = args.pop().unwrap_or_else(|| {
                TypeExpr::Named(Ident {
                    name: "nothing".to_string(),
                    span,
                })
            });
            return TypeExpr::Function(args, Box::new(return_type), span);
        }

        if !matches!(info.kind.as_str(), "alias" | "refinement")
            && !info.args.is_empty()
            && let Some((generic_name, _)) = info.type_name.split_once('[')
        {
            let args = info
                .args
                .iter()
                .map(Self::reflection_type_info_type_expr)
                .collect();
            return TypeExpr::Generic(
                Ident {
                    name: generic_name.to_string(),
                    span,
                },
                args,
                span,
            );
        }

        TypeExpr::Named(Ident {
            name: info.type_name.clone(),
            span,
        })
    }

    fn reflection_type_span() -> jett_common::Span {
        jett_common::Span::new(FileId::new(0), 0, 0)
    }

    fn checked_type_fields(&self, ty: &TypeExpr) -> Option<&[ReflectionFieldInfo]> {
        let metadata = self.reflection_metadata.as_ref()?;
        let type_name = type_expr_display(ty);
        metadata.get_type_fields(&type_name)
    }

    fn checked_type_fields_value(&self, ty: &TypeExpr) -> Option<Value> {
        let fields = self.checked_type_fields(ty)?;
        let owner_type = type_expr_display(ty);
        Some(Value::List(
            fields
                .iter()
                .map(|field| Self::reflection_field_info_value(&owner_type, None, field))
                .collect(),
        ))
    }

    fn reflection_field_info_value(
        owner_type: &str,
        owner_member: Option<&str>,
        field: &ReflectionFieldInfo,
    ) -> Value {
        Value::Struct {
            type_name: "TypeField".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(field.index as i64)),
                (
                    "owner_type".to_string(),
                    Value::String(owner_type.to_string()),
                ),
                (
                    "owner_member".to_string(),
                    Self::optional_string_value(owner_member),
                ),
                ("name".to_string(), Value::String(field.name.clone())),
                (
                    "type_name".to_string(),
                    Value::String(field.type_name.clone()),
                ),
                ("kind".to_string(), Value::String(field.kind.clone())),
                (
                    "kind_tag".to_string(),
                    Self::type_kind_tag_value(&field.kind),
                ),
                (
                    "serialize_name".to_string(),
                    Value::String(field.serialize_name.clone()),
                ),
                ("has_secret".to_string(), Value::Bool(field.has_secret)),
                (
                    "type_info".to_string(),
                    Self::reflection_type_info_value(&field.type_info),
                ),
            ],
        }
    }

    fn reflection_field_refinement_name(field: &ReflectionFieldInfo) -> String {
        Self::reflection_type_base_name(&field.type_info)
    }

    fn reflection_type_base_name(info: &ReflectionTypeInfo) -> String {
        if info.kind == "function" {
            return "function".to_string();
        }
        info.type_name
            .split_once('[')
            .map(|(base, _)| base)
            .unwrap_or(&info.type_name)
            .to_string()
    }

    fn checked_bitfield(&self, ty: &TypeExpr) -> Option<&ReflectionBitfieldInfo> {
        let metadata = self.reflection_metadata.as_ref()?;
        let type_name = type_expr_display(ty);
        metadata.get_bitfield(&type_name)
    }

    fn checked_bitfield_value(&self, ty: &TypeExpr) -> Option<Value> {
        self.checked_bitfield(ty)
            .map(Self::reflection_bitfield_info_value)
    }

    fn checked_bitfield_fields_value(&self, ty: &TypeExpr) -> Option<Value> {
        let bitfield = self.checked_bitfield(ty)?;
        Some(Value::List(
            bitfield
                .fields
                .iter()
                .map(Self::reflection_bitfield_field_info_value)
                .collect(),
        ))
    }

    fn reflection_bitfield_info_value(bitfield: &ReflectionBitfieldInfo) -> Value {
        let fields = bitfield
            .fields
            .iter()
            .map(Self::reflection_bitfield_field_info_value)
            .collect();
        Value::Struct {
            type_name: "TypeBitfield".to_string(),
            fields: vec![
                (
                    "network_order".to_string(),
                    Value::Bool(bitfield.network_order),
                ),
                ("fields".to_string(), Value::List(fields)),
            ],
        }
    }

    fn reflection_bitfield_field_info_value(field: &ReflectionBitfieldFieldInfo) -> Value {
        let enum_type = field
            .enum_type
            .as_ref()
            .map(|info| Value::OptionalSome(Box::new(Self::reflection_type_info_value(info))))
            .unwrap_or(Value::OptionalNone);
        Value::Struct {
            type_name: "TypeBitfieldField".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(field.index as i64)),
                ("name".to_string(), Value::String(field.name.clone())),
                ("shape".to_string(), Value::String(field.shape.clone())),
                (
                    "shape_tag".to_string(),
                    Self::bitfield_shape_tag_value(&field.shape),
                ),
                ("width".to_string(), Value::Int64(field.width)),
                (
                    "type_info".to_string(),
                    Self::reflection_type_info_value(&field.type_info),
                ),
                ("enum_type".to_string(), enum_type),
            ],
        }
    }

    fn checked_machine(&self, ty: &TypeExpr) -> Option<&ReflectionMachineInfo> {
        let metadata = self.reflection_metadata.as_ref()?;
        let type_name = type_expr_display(ty);
        let base_name = type_expr_name(ty);
        metadata
            .get_machine(&type_name)
            .or_else(|| metadata.get_machine(&base_name))
            .or_else(|| {
                if let TypeExpr::StateQualified(inner, _, _) = ty {
                    metadata.get_machine(&type_expr_display(inner))
                } else {
                    None
                }
            })
    }

    fn checked_machine_value(&self, ty: &TypeExpr) -> Option<Value> {
        self.checked_machine(ty)
            .map(|machine| Self::reflection_machine_info_value(&type_expr_name(ty), machine))
    }

    fn checked_machine_states_value(&self, ty: &TypeExpr) -> Option<Value> {
        let machine = self.checked_machine(ty)?;
        let owner_type = type_expr_name(ty);
        Some(Value::List(
            machine
                .states
                .iter()
                .map(|state| Self::reflection_machine_state_info_value(&owner_type, state))
                .collect(),
        ))
    }

    fn checked_machine_transitions_value(&self, ty: &TypeExpr) -> Option<Value> {
        let machine = self.checked_machine(ty)?;
        Some(Value::List(
            machine
                .edges
                .iter()
                .map(Self::reflection_machine_transition_info_value)
                .collect(),
        ))
    }

    fn reflection_machine_info_value(owner_type: &str, machine: &ReflectionMachineInfo) -> Value {
        let states = machine
            .states
            .iter()
            .map(|state| Self::reflection_machine_state_info_value(owner_type, state))
            .collect();
        let edges = machine
            .edges
            .iter()
            .map(Self::reflection_machine_transition_info_value)
            .collect();
        Value::Struct {
            type_name: "TypeMachine".to_string(),
            fields: vec![
                ("states".to_string(), Value::List(states)),
                ("edges".to_string(), Value::List(edges)),
            ],
        }
    }

    fn reflection_machine_state_info_value(
        owner_type: &str,
        state: &ReflectionMachineStateInfo,
    ) -> Value {
        let owner_member = state.name.as_str();
        let fields = state
            .fields
            .iter()
            .map(|field| Self::reflection_field_info_value(owner_type, Some(owner_member), field))
            .collect();
        Value::Struct {
            type_name: "TypeMachineState".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(state.index as i64)),
                (
                    "owner_type".to_string(),
                    Value::String(owner_type.to_string()),
                ),
                ("name".to_string(), Value::String(state.name.clone())),
                ("has_secret".to_string(), Value::Bool(state.has_secret)),
                ("fields".to_string(), Value::List(fields)),
            ],
        }
    }

    fn reflection_machine_transition_info_value(
        transition: &ReflectionMachineTransitionInfo,
    ) -> Value {
        Value::Struct {
            type_name: "TypeMachineTransition".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(transition.index as i64)),
                (
                    "source_index".to_string(),
                    Value::Int64(transition.source_index as i64),
                ),
                (
                    "source".to_string(),
                    Value::String(transition.source.clone()),
                ),
                (
                    "target_index".to_string(),
                    Value::Int64(transition.target_index as i64),
                ),
                (
                    "target".to_string(),
                    Value::String(transition.target.clone()),
                ),
            ],
        }
    }

    fn checked_type_variants(&self, ty: &TypeExpr) -> Option<&[ReflectionVariantInfo]> {
        let metadata = self.reflection_metadata.as_ref()?;
        let type_name = type_expr_display(ty);
        metadata.get_type_variants(&type_name)
    }

    fn checked_type_variants_value(&self, ty: &TypeExpr) -> Option<Value> {
        let variants = self.checked_type_variants(ty)?;
        let owner_type = type_expr_display(ty);
        Some(Value::List(
            variants
                .iter()
                .map(|variant| Self::reflection_variant_info_value(&owner_type, variant))
                .collect(),
        ))
    }

    fn reflection_variant_info_value(owner_type: &str, variant: &ReflectionVariantInfo) -> Value {
        let owner_member = variant.name.as_str();
        let fields = variant
            .fields
            .iter()
            .map(|field| Self::reflection_field_info_value(owner_type, Some(owner_member), field))
            .collect();
        Value::Struct {
            type_name: "TypeVariant".to_string(),
            fields: vec![
                ("index".to_string(), Value::Int64(variant.index as i64)),
                (
                    "owner_type".to_string(),
                    Value::String(owner_type.to_string()),
                ),
                ("name".to_string(), Value::String(variant.name.clone())),
                (
                    "discriminant".to_string(),
                    Value::Int64(variant.discriminant),
                ),
                ("has_secret".to_string(), Value::Bool(variant.has_secret)),
                ("fields".to_string(), Value::List(fields)),
            ],
        }
    }

    fn reflection_type_info_value(info: &ReflectionTypeInfo) -> Value {
        let arg_values = info
            .args
            .iter()
            .map(Self::reflection_type_info_value)
            .collect();
        Value::Struct {
            type_name: "TypeInfo".to_string(),
            fields: vec![
                (
                    "type_name".to_string(),
                    Value::String(info.type_name.clone()),
                ),
                ("kind".to_string(), Value::String(info.kind.clone())),
                (
                    "kind_tag".to_string(),
                    Self::type_kind_tag_value(&info.kind),
                ),
                (
                    "primitive_tag".to_string(),
                    Self::primitive_tag_value(info.primitive_tag.as_deref()),
                ),
                ("has_secret".to_string(), Value::Bool(info.has_secret)),
                ("args".to_string(), Value::List(arg_values)),
            ],
        }
    }

    fn type_info_value(&self, ty: &TypeExpr) -> Value {
        let ty = self.substitute_type_expr(ty);
        if let TypeExpr::View(inner, _) = &ty {
            return self.type_info_value(inner);
        }

        let arg_values = match &ty {
            TypeExpr::Named(ident) if self.type_aliases.contains_key(&ident.name) => self
                .type_alias_bases
                .get(&ident.name)
                .map(|base_ty| vec![self.type_info_value(base_ty)])
                .unwrap_or_default(),
            TypeExpr::Generic(_, args, _) => args
                .iter()
                .map(|arg| self.type_info_value(arg))
                .collect::<Vec<_>>(),
            TypeExpr::Function(params, return_type, _) => params
                .iter()
                .chain(std::iter::once(return_type.as_ref()))
                .map(|arg| self.type_info_value(arg))
                .collect(),
            _ => Vec::new(),
        };

        Value::Struct {
            type_name: "TypeInfo".to_string(),
            fields: vec![
                (
                    "type_name".to_string(),
                    Value::String(type_expr_display(&ty)),
                ),
                (
                    "kind".to_string(),
                    Value::String(self.type_expr_kind(&ty).to_string()),
                ),
                (
                    "kind_tag".to_string(),
                    Self::type_kind_tag_value(self.type_expr_kind(&ty)),
                ),
                (
                    "primitive_tag".to_string(),
                    self.type_primitive_tag_value(&ty),
                ),
                (
                    "has_secret".to_string(),
                    Value::Bool(self.type_expr_has_secret(&ty)),
                ),
                ("args".to_string(), Value::List(arg_values)),
            ],
        }
    }

    fn reflected_field_value(
        &self,
        value: &Value,
        owner_ty: &TypeExpr,
        field_metadata: &Value,
        expected_field_ty: &TypeExpr,
    ) -> Result<Value, String> {
        let metadata = Self::type_field_metadata_for(field_metadata, "type.field_value")?;
        let expected_owner = type_expr_display(owner_ty);
        Self::validate_field_metadata_owner(&metadata, &expected_owner, None, "type.field_value")?;
        let (field_name, actual_type_name) = if let Some(fields) =
            self.checked_type_fields(owner_ty)
        {
            let field = fields.get(metadata.index).ok_or_else(|| {
                format!(
                    "type.field_value: type '{}' has no field at index {}",
                    type_expr_display(owner_ty),
                    metadata.index
                )
            })?;

            if field.name != metadata.name {
                return Err(format!(
                    "type.field_value: field metadata '{}' does not match field '{}' on type '{}'",
                    metadata.name,
                    field.name,
                    type_expr_display(owner_ty)
                ));
            }

            if !self.reflection_type_names_match(&field.type_name, &metadata.type_name) {
                return Err(format!(
                    "type.field_value: field metadata for '{}' has type '{}', but type '{}' reports '{}'",
                    metadata.name,
                    metadata.type_name,
                    type_expr_display(owner_ty),
                    field.type_name
                ));
            }
            (field.name.clone(), field.type_name.clone())
        } else if self.checked_metadata_kind_is(owner_ty, &["struct", "bitfield"]) {
            return Err(self.missing_checked_metadata_error(owner_ty, "field"));
        } else {
            let owner_fields = self.type_expr_fields(owner_ty);
            let field = owner_fields.get(metadata.index).ok_or_else(|| {
                format!(
                    "type.field_value: type '{}' has no field at index {}",
                    type_expr_display(owner_ty),
                    metadata.index
                )
            })?;

            if field.name != metadata.name {
                return Err(format!(
                    "type.field_value: field metadata '{}' does not match field '{}' on type '{}'",
                    metadata.name,
                    field.name,
                    type_expr_display(owner_ty)
                ));
            }

            let actual_type_name = type_expr_display(&field.ty);
            if !self.reflection_type_names_match(&actual_type_name, &metadata.type_name) {
                return Err(format!(
                    "type.field_value: field metadata for '{}' has type '{}', but type '{}' reports '{}'",
                    metadata.name,
                    metadata.type_name,
                    type_expr_display(owner_ty),
                    actual_type_name
                ));
            }
            (field.name.clone(), actual_type_name)
        };

        let expected_type_name = type_expr_display(expected_field_ty);
        if !self.reflection_type_name_matches_expr(&actual_type_name, expected_field_ty) {
            return Err(format!(
                "type.field_value: field '{}' has type '{}', requested '{}'",
                metadata.name, actual_type_name, expected_type_name
            ));
        }

        match value {
            Value::Struct { fields, .. } => fields
                .iter()
                .find(|(name, _)| name == &field_name)
                .map(|(_, field_value)| field_value.clone())
                .ok_or_else(|| format!("type.field_value: value is missing field '{field_name}'")),
            other => Err(format!(
                "type.field_value: expected struct value for '{}', got {other}",
                type_expr_display(owner_ty)
            )),
        }
    }

    fn reflected_machine_state_value(
        &self,
        value: &Value,
        owner_ty: &TypeExpr,
    ) -> Result<Value, String> {
        let Value::Machine {
            type_name, state, ..
        } = value
        else {
            return Err(format!(
                "type.machine_state_value: expected machine value for '{}', got {value}",
                type_expr_display(owner_ty)
            ));
        };

        let expected_type_name = type_expr_name(owner_ty);
        if type_name != &expected_type_name {
            return Err(format!(
                "type.machine_state_value: expected machine '{}', got '{}'",
                expected_type_name, type_name
            ));
        }

        if let Some(expected_state) = type_expr_state_name(owner_ty)
            && state != expected_state
        {
            return Err(format!(
                "type.machine_state_value: expected machine state '{} at {}', got '{} at {}'",
                expected_type_name, expected_state, type_name, state
            ));
        }

        if let Some(machine) = self.checked_machine(owner_ty) {
            return machine
                .states
                .iter()
                .find(|candidate| candidate.name == *state)
                .map(|candidate| {
                    Self::reflection_machine_state_info_value(&expected_type_name, candidate)
                })
                .ok_or_else(|| {
                    format!(
                        "type.machine_state_value: machine '{}' has no state '{}'",
                        expected_type_name, state
                    )
                });
        }

        if self.checked_metadata_kind_is(owner_ty, &["machine", "machine_state"]) {
            return Err(self.missing_checked_metadata_error(owner_ty, "machine"));
        }

        self.type_expr_machine(owner_ty)
            .states
            .into_iter()
            .enumerate()
            .find(|(_, candidate)| candidate.name == *state)
            .map(|(index, state)| self.type_machine_state_value(index, &expected_type_name, state))
            .ok_or_else(|| {
                format!(
                    "type.machine_state_value: machine '{}' has no state '{}'",
                    expected_type_name, state
                )
            })
    }

    fn reflected_machine_field_value(
        &self,
        value: &Value,
        owner_ty: &TypeExpr,
        field_metadata: &Value,
        expected_field_ty: &TypeExpr,
    ) -> Result<Value, String> {
        let metadata = Self::type_field_metadata_for(field_metadata, "type.machine_field_value")?;
        let Value::Machine {
            type_name,
            state,
            fields,
        } = value
        else {
            return Err(format!(
                "type.machine_field_value: expected machine value for '{}', got {value}",
                type_expr_display(owner_ty)
            ));
        };

        let expected_type_name = type_expr_name(owner_ty);
        Self::validate_field_metadata_owner(
            &metadata,
            &expected_type_name,
            Some(state),
            "type.machine_field_value",
        )?;
        if type_name != &expected_type_name {
            return Err(format!(
                "type.machine_field_value: expected machine '{}', got '{}'",
                expected_type_name, type_name
            ));
        }

        if let Some(expected_state) = type_expr_state_name(owner_ty)
            && state != expected_state
        {
            return Err(format!(
                "type.machine_field_value: expected machine state '{} at {}', got '{} at {}'",
                expected_type_name, expected_state, type_name, state
            ));
        }

        let (field_name, actual_type_name) = if let Some(machine) = self.checked_machine(owner_ty) {
            let state_metadata = machine
                .states
                .iter()
                .find(|candidate| candidate.name == *state)
                .ok_or_else(|| {
                    format!(
                        "type.machine_field_value: machine '{}' has no state '{}'",
                        expected_type_name, state
                    )
                })?;
            let field = state_metadata.fields.get(metadata.index).ok_or_else(|| {
                format!(
                    "type.machine_field_value: state '{}.{}' has no payload field at index {}",
                    expected_type_name, state, metadata.index
                )
            })?;

            if field.name != metadata.name {
                return Err(format!(
                    "type.machine_field_value: field metadata '{}' does not match payload field '{}' on state '{}.{}'",
                    metadata.name, field.name, expected_type_name, state
                ));
            }

            if !self.reflection_type_names_match(&field.type_name, &metadata.type_name) {
                return Err(format!(
                    "type.machine_field_value: field metadata for '{}' has type '{}', but state '{}.{}' reports '{}'",
                    metadata.name, metadata.type_name, expected_type_name, state, field.type_name
                ));
            }
            (field.name.clone(), field.type_name.clone())
        } else if self.checked_metadata_kind_is(owner_ty, &["machine", "machine_state"]) {
            return Err(self.missing_checked_metadata_error(owner_ty, "machine"));
        } else {
            let machine = self.type_expr_machine(owner_ty);
            let state_metadata = machine
                .states
                .iter()
                .find(|candidate| candidate.name == *state)
                .ok_or_else(|| {
                    format!(
                        "type.machine_field_value: machine '{}' has no state '{}'",
                        expected_type_name, state
                    )
                })?;
            let field = state_metadata.fields.get(metadata.index).ok_or_else(|| {
                format!(
                    "type.machine_field_value: state '{}.{}' has no payload field at index {}",
                    expected_type_name, state, metadata.index
                )
            })?;

            if field.name != metadata.name {
                return Err(format!(
                    "type.machine_field_value: field metadata '{}' does not match payload field '{}' on state '{}.{}'",
                    metadata.name, field.name, expected_type_name, state
                ));
            }

            let actual_type_name = type_expr_display(&field.ty);
            if !self.reflection_type_names_match(&actual_type_name, &metadata.type_name) {
                return Err(format!(
                    "type.machine_field_value: field metadata for '{}' has type '{}', but state '{}.{}' reports '{}'",
                    metadata.name, metadata.type_name, expected_type_name, state, actual_type_name
                ));
            }
            (field.name.clone(), actual_type_name)
        };

        let expected_type_name = type_expr_display(expected_field_ty);
        if !self.reflection_type_name_matches_expr(&actual_type_name, expected_field_ty) {
            return Err(format!(
                "type.machine_field_value: field '{}' has type '{}', requested '{}'",
                metadata.name, actual_type_name, expected_type_name
            ));
        }

        fields.get(metadata.index).cloned().ok_or_else(|| {
            format!("type.machine_field_value: value is missing payload field '{field_name}'")
        })
    }

    fn reflected_variant_value(&self, value: &Value, owner_ty: &TypeExpr) -> Result<Value, String> {
        let Value::Enum {
            type_name, variant, ..
        } = value
        else {
            return Err(format!(
                "type.variant_value: expected enum value for '{}', got {value}",
                type_expr_display(owner_ty)
            ));
        };

        let expected_type_name = type_expr_name(owner_ty);
        if type_name != &expected_type_name {
            return Err(format!(
                "type.variant_value: expected enum '{}', got '{}'",
                expected_type_name, type_name
            ));
        }

        if let Some(variants) = self.checked_type_variants(owner_ty) {
            return variants
                .iter()
                .find(|candidate| candidate.name == *variant)
                .map(|candidate| {
                    Self::reflection_variant_info_value(&type_expr_display(owner_ty), candidate)
                })
                .ok_or_else(|| {
                    format!(
                        "type.variant_value: enum '{}' has no variant '{}'",
                        expected_type_name, variant
                    )
                });
        }

        if self.checked_metadata_kind_is(owner_ty, &["enum"]) {
            return Err(self.missing_checked_metadata_error(owner_ty, "variant"));
        }

        self.type_expr_variants(owner_ty)
            .into_iter()
            .enumerate()
            .find(|(_, candidate)| candidate.name == *variant)
            .map(|(index, variant)| self.type_variant_value(index, &expected_type_name, variant))
            .ok_or_else(|| {
                format!(
                    "type.variant_value: enum '{}' has no variant '{}'",
                    expected_type_name, variant
                )
            })
    }

    fn reflected_variant_field_value(
        &self,
        value: &Value,
        owner_ty: &TypeExpr,
        field_metadata: &Value,
        expected_field_ty: &TypeExpr,
    ) -> Result<Value, String> {
        let metadata = Self::type_field_metadata_for(field_metadata, "type.variant_field_value")?;
        let Value::Enum {
            type_name,
            variant,
            fields,
        } = value
        else {
            return Err(format!(
                "type.variant_field_value: expected enum value for '{}', got {value}",
                type_expr_display(owner_ty)
            ));
        };

        let expected_type_name = type_expr_name(owner_ty);
        let expected_owner = type_expr_display(owner_ty);
        Self::validate_field_metadata_owner(
            &metadata,
            &expected_owner,
            Some(variant),
            "type.variant_field_value",
        )?;
        if type_name != &expected_type_name {
            return Err(format!(
                "type.variant_field_value: expected enum '{}', got '{}'",
                expected_type_name, type_name
            ));
        }

        if let Some(variants) = self.checked_type_variants(owner_ty) {
            let variant_metadata = variants
                .iter()
                .find(|candidate| candidate.name == *variant)
                .ok_or_else(|| {
                    format!(
                        "type.variant_field_value: enum '{}' has no variant '{}'",
                        expected_type_name, variant
                    )
                })?;
            let field = variant_metadata.fields.get(metadata.index).ok_or_else(|| {
                format!(
                    "type.variant_field_value: variant '{}.{}' has no payload field at index {}",
                    expected_type_name, variant, metadata.index
                )
            })?;

            if field.name != metadata.name {
                return Err(format!(
                    "type.variant_field_value: field metadata '{}' does not match payload field '{}' on variant '{}.{}'",
                    metadata.name, field.name, expected_type_name, variant
                ));
            }

            if !self.reflection_type_names_match(&field.type_name, &metadata.type_name) {
                return Err(format!(
                    "type.variant_field_value: field metadata for '{}' has type '{}', but variant '{}.{}' reports '{}'",
                    metadata.name, metadata.type_name, expected_type_name, variant, field.type_name
                ));
            }

            let expected_field_type_name = type_expr_display(expected_field_ty);
            if !self.reflection_type_name_matches_expr(&field.type_name, expected_field_ty) {
                return Err(format!(
                    "type.variant_field_value: field '{}' has type '{}', requested '{}'",
                    metadata.name, field.type_name, expected_field_type_name
                ));
            }

            return fields.get(metadata.index).cloned().ok_or_else(|| {
                format!(
                    "type.variant_field_value: value is missing payload field '{}'",
                    field.name
                )
            });
        }

        if self.checked_metadata_kind_is(owner_ty, &["enum"]) {
            return Err(self.missing_checked_metadata_error(owner_ty, "variant"));
        }

        let variants = self.type_expr_variants(owner_ty);
        let variant_metadata = variants
            .iter()
            .find(|candidate| candidate.name == *variant)
            .ok_or_else(|| {
                format!(
                    "type.variant_field_value: enum '{}' has no variant '{}'",
                    expected_type_name, variant
                )
            })?;
        let field = variant_metadata.fields.get(metadata.index).ok_or_else(|| {
            format!(
                "type.variant_field_value: variant '{}.{}' has no payload field at index {}",
                expected_type_name, variant, metadata.index
            )
        })?;

        if field.name != metadata.name {
            return Err(format!(
                "type.variant_field_value: field metadata '{}' does not match payload field '{}' on variant '{}.{}'",
                metadata.name, field.name, expected_type_name, variant
            ));
        }

        let actual_type_name = type_expr_display(&field.ty);
        if !self.reflection_type_names_match(&actual_type_name, &metadata.type_name) {
            return Err(format!(
                "type.variant_field_value: field metadata for '{}' has type '{}', but variant '{}.{}' reports '{}'",
                metadata.name, metadata.type_name, expected_type_name, variant, actual_type_name
            ));
        }

        let expected_field_type_name = type_expr_display(expected_field_ty);
        if !self.reflection_type_name_matches_expr(&actual_type_name, expected_field_ty) {
            return Err(format!(
                "type.variant_field_value: field '{}' has type '{}', requested '{}'",
                metadata.name, actual_type_name, expected_field_type_name
            ));
        }

        fields.get(metadata.index).cloned().ok_or_else(|| {
            format!(
                "type.variant_field_value: value is missing payload field '{}'",
                field.name
            )
        })
    }

    fn reflected_construct_put(
        &self,
        owner_ty: &TypeExpr,
        expected_field_ty: &TypeExpr,
        builder: &Value,
        field_metadata: &Value,
        value: &Value,
    ) -> Result<Value, String> {
        let Value::TypeConstruction {
            type_name,
            variant,
            state,
            fields: existing_fields,
        } = builder
        else {
            return Err(format!(
                "type.construct_put: first argument must be TypeConstruction, got {builder}"
            ));
        };

        let expected_owner = type_expr_display(owner_ty);
        if type_name != &expected_owner {
            return Ok(result_fail(format!(
                "type.construct_put: builder for '{}' cannot construct '{}'",
                type_name, expected_owner
            )));
        }
        let owner_kind = self
            .checked_construction_kind(owner_ty)
            .unwrap_or_else(|| self.type_expr_kind(owner_ty));
        if !matches!(
            owner_kind,
            "struct" | "bitfield" | "enum" | "machine" | "machine_state"
        ) {
            return Ok(result_fail(format!(
                "type.construct_put supports only structs, bitfields, enums, and machines, got '{}'",
                type_expr_display(owner_ty)
            )));
        }

        let metadata = match Self::type_field_metadata_for(field_metadata, "type.construct_put") {
            Ok(metadata) => metadata,
            Err(message) => return Ok(result_fail(message)),
        };

        let (field_name, actual_type_name) = if owner_kind == "enum" {
            if state.is_some() {
                return Ok(result_fail(format!(
                    "type.construct_put: machine state builder cannot construct '{}'",
                    expected_owner
                )));
            }
            let Some(variant_name) = variant.as_ref() else {
                return Ok(result_fail(
                    "type.construct_put: enum construction requires type.construct_variant_start"
                        .to_string(),
                ));
            };
            if let Err(message) = Self::validate_field_metadata_owner(
                &metadata,
                &expected_owner,
                Some(variant_name),
                "type.construct_put",
            ) {
                return Ok(result_fail(message));
            }

            if let Some(variants) = self.checked_type_variants(owner_ty) {
                let Some(variant) = variants
                    .iter()
                    .find(|candidate| candidate.name == *variant_name)
                else {
                    return Ok(result_fail(format!(
                        "type.construct_put: enum '{}' has no variant '{}'",
                        expected_owner, variant_name
                    )));
                };
                let Some(field) = variant.fields.get(metadata.index) else {
                    return Ok(result_fail(format!(
                        "type.construct_put: variant '{}.{}' has no payload field at index {}",
                        expected_owner, variant_name, metadata.index
                    )));
                };
                (field.name.clone(), field.type_name.clone())
            } else {
                if self.checked_metadata_kind_is(owner_ty, &["enum"]) {
                    return Ok(result_fail(
                        self.missing_checked_metadata_error(owner_ty, "variant"),
                    ));
                }
                let Some(variant) = self
                    .type_expr_variants(owner_ty)
                    .into_iter()
                    .find(|candidate| candidate.name == *variant_name)
                else {
                    return Ok(result_fail(format!(
                        "type.construct_put: enum '{}' has no variant '{}'",
                        expected_owner, variant_name
                    )));
                };
                let Some(field) = variant.fields.get(metadata.index) else {
                    return Ok(result_fail(format!(
                        "type.construct_put: variant '{}.{}' has no payload field at index {}",
                        expected_owner, variant_name, metadata.index
                    )));
                };
                (field.name.clone(), type_expr_display(&field.ty))
            }
        } else if matches!(owner_kind, "machine" | "machine_state") {
            if variant.is_some() {
                return Ok(result_fail(format!(
                    "type.construct_put: enum variant builder cannot construct '{}'",
                    expected_owner
                )));
            }
            let Some(state_name) = state.as_ref() else {
                return Ok(result_fail(
                    "type.construct_put: machine construction requires type.construct_machine_start"
                        .to_string(),
                ));
            };
            let expected_machine = type_expr_name(owner_ty);
            if let Err(message) = Self::validate_field_metadata_owner(
                &metadata,
                &expected_machine,
                Some(state_name),
                "type.construct_put",
            ) {
                return Ok(result_fail(message));
            }

            if let Some(machine) = self.checked_machine(owner_ty) {
                let Some(state_metadata) = machine
                    .states
                    .iter()
                    .find(|candidate| candidate.name == *state_name)
                else {
                    return Ok(result_fail(format!(
                        "type.construct_put: machine '{}' has no state '{}'",
                        expected_machine, state_name
                    )));
                };
                let Some(field) = state_metadata.fields.get(metadata.index) else {
                    return Ok(result_fail(format!(
                        "type.construct_put: state '{}.{}' has no payload field at index {}",
                        expected_machine, state_name, metadata.index
                    )));
                };
                (field.name.clone(), field.type_name.clone())
            } else {
                if self.checked_metadata_kind_is(owner_ty, &["machine", "machine_state"]) {
                    return Ok(result_fail(
                        self.missing_checked_metadata_error(owner_ty, "machine"),
                    ));
                }
                let machine = self.type_expr_machine(owner_ty);
                let Some(state_metadata) = machine
                    .states
                    .iter()
                    .find(|candidate| candidate.name == *state_name)
                else {
                    return Ok(result_fail(format!(
                        "type.construct_put: machine '{}' has no state '{}'",
                        expected_machine, state_name
                    )));
                };
                let Some(field) = state_metadata.fields.get(metadata.index) else {
                    return Ok(result_fail(format!(
                        "type.construct_put: state '{}.{}' has no payload field at index {}",
                        expected_machine, state_name, metadata.index
                    )));
                };
                (field.name.clone(), type_expr_display(&field.ty))
            }
        } else {
            if variant.is_some() {
                return Ok(result_fail(format!(
                    "type.construct_put: enum variant builder cannot construct '{}'",
                    expected_owner
                )));
            }
            if state.is_some() {
                return Ok(result_fail(format!(
                    "type.construct_put: machine state builder cannot construct '{}'",
                    expected_owner
                )));
            }
            if let Err(message) = Self::validate_field_metadata_owner(
                &metadata,
                &expected_owner,
                None,
                "type.construct_put",
            ) {
                return Ok(result_fail(message));
            }

            if let Some(fields) = self.checked_type_fields(owner_ty) {
                let Some(field) = fields.get(metadata.index) else {
                    return Ok(result_fail(format!(
                        "type.construct_put: type '{}' has no field at index {}",
                        expected_owner, metadata.index
                    )));
                };
                (field.name.clone(), field.type_name.clone())
            } else {
                if self.checked_metadata_kind_is(owner_ty, &["struct", "bitfield"]) {
                    return Ok(result_fail(
                        self.missing_checked_metadata_error(owner_ty, "field"),
                    ));
                }
                let fields = self.type_expr_fields(owner_ty);
                let Some(field) = fields.get(metadata.index) else {
                    return Ok(result_fail(format!(
                        "type.construct_put: type '{}' has no field at index {}",
                        expected_owner, metadata.index
                    )));
                };
                (field.name.clone(), type_expr_display(&field.ty))
            }
        };

        if field_name != metadata.name {
            if let Some(variant_name) = variant.as_ref() {
                return Ok(result_fail(format!(
                    "type.construct_put: field metadata '{}' does not match payload field '{}' on variant '{}.{}'",
                    metadata.name, field_name, expected_owner, variant_name
                )));
            } else if let Some(state_name) = state.as_ref() {
                return Ok(result_fail(format!(
                    "type.construct_put: field metadata '{}' does not match payload field '{}' on state '{}.{}'",
                    metadata.name,
                    field_name,
                    type_expr_name(owner_ty),
                    state_name
                )));
            } else {
                return Ok(result_fail(format!(
                    "type.construct_put: field metadata '{}' does not match field '{}' on '{}'",
                    metadata.name, field_name, expected_owner
                )));
            }
        }

        if !self.reflection_type_names_match(&actual_type_name, &metadata.type_name) {
            return Ok(result_fail(format!(
                "type.construct_put: field metadata for '{}' has type '{}', but '{}' reports '{}'",
                metadata.name, metadata.type_name, expected_owner, actual_type_name
            )));
        }

        let expected_field_type_name = type_expr_display(expected_field_ty);
        if !self.reflection_type_name_matches_expr(&actual_type_name, expected_field_ty) {
            return Ok(result_fail(format!(
                "type.construct_put: field '{}' has type '{}', provided as '{}'",
                metadata.name, actual_type_name, expected_field_type_name
            )));
        }

        if existing_fields
            .iter()
            .any(|(index, _, _, _)| *index == metadata.index)
        {
            return Ok(result_fail(format!(
                "type.construct_put: field '{}' was provided more than once",
                metadata.name
            )));
        }

        let value = match self.normalize_value_for_type_name(&actual_type_name, value.clone()) {
            Ok(value) => value,
            Err(message) => return Ok(result_fail(message)),
        };

        let mut fields = existing_fields.clone();
        fields.push((metadata.index, metadata.name, actual_type_name, value));
        Ok(result_ok(Value::TypeConstruction {
            type_name: type_name.clone(),
            variant: variant.clone(),
            state: state.clone(),
            fields,
        }))
    }

    fn reflected_construct_variant_start(
        &self,
        owner_ty: &TypeExpr,
        variant_metadata: &Value,
    ) -> Result<Value, String> {
        let owner_kind = self
            .checked_construction_kind(owner_ty)
            .unwrap_or_else(|| self.type_expr_kind(owner_ty));
        if owner_kind != "enum" {
            return Ok(result_fail(format!(
                "type.construct_variant_start supports only enums, got '{}'",
                type_expr_display(owner_ty)
            )));
        }

        let expected_owner = type_expr_display(owner_ty);
        let (variant_index, metadata_owner, metadata_name, metadata_discriminant) =
            match Self::type_variant_metadata(variant_metadata, "type.construct_variant_start") {
                Ok(metadata) => metadata,
                Err(message) => return Ok(result_fail(message)),
            };
        if metadata_owner != expected_owner {
            return Ok(result_fail(format!(
                "type.construct_variant_start: variant metadata belongs to '{}', expected '{}'",
                metadata_owner, expected_owner
            )));
        }
        let metadata_fields = match Self::type_variant_payload_field_metadata(
            variant_metadata,
            "type.construct_variant_start",
        ) {
            Ok(fields) => fields,
            Err(message) => return Ok(result_fail(message)),
        };

        let variant_name = if let Some(variants) = self.checked_type_variants(owner_ty) {
            let Some(variant) = variants.get(variant_index) else {
                return Ok(result_fail(format!(
                    "type.construct_variant_start: enum '{}' has no variant at index {}",
                    expected_owner, variant_index
                )));
            };
            if variant.name != metadata_name {
                return Ok(result_fail(format!(
                    "type.construct_variant_start: variant metadata '{}' does not match variant '{}' on '{}'",
                    metadata_name, variant.name, expected_owner
                )));
            }
            if variant.discriminant != metadata_discriminant {
                return Ok(result_fail(format!(
                    "type.construct_variant_start: variant '{}.{}' has discriminant {}, metadata reports {}",
                    expected_owner, variant.name, variant.discriminant, metadata_discriminant
                )));
            }
            if metadata_fields.len() != variant.fields.len() {
                return Ok(result_fail(format!(
                    "type.construct_variant_start: variant '{}.{}' expects {} payload field(s), metadata reports {}",
                    expected_owner,
                    variant.name,
                    variant.fields.len(),
                    metadata_fields.len()
                )));
            }
            for (index, field) in variant.fields.iter().enumerate() {
                let metadata_field = &metadata_fields[index];
                if let Err(message) = Self::validate_field_metadata_owner(
                    metadata_field,
                    &expected_owner,
                    Some(&variant.name),
                    "type.construct_variant_start",
                ) {
                    return Ok(result_fail(message));
                }
                if metadata_field.index != index
                    || metadata_field.name != field.name
                    || metadata_field.type_name != field.type_name
                {
                    return Ok(result_fail(format!(
                        "type.construct_variant_start: payload field metadata at index {} does not match variant '{}.{}'",
                        index, expected_owner, variant.name
                    )));
                }
            }
            variant.name.clone()
        } else {
            if self.checked_metadata_kind_is(owner_ty, &["enum"]) {
                return Ok(result_fail(
                    self.missing_checked_metadata_error(owner_ty, "variant"),
                ));
            }
            let variants = self.type_expr_variants(owner_ty);
            let Some(variant) = variants.get(variant_index) else {
                return Ok(result_fail(format!(
                    "type.construct_variant_start: enum '{}' has no variant at index {}",
                    expected_owner, variant_index
                )));
            };
            if variant.name != metadata_name {
                return Ok(result_fail(format!(
                    "type.construct_variant_start: variant metadata '{}' does not match variant '{}' on '{}'",
                    metadata_name, variant.name, expected_owner
                )));
            }
            if variant.discriminant != metadata_discriminant {
                return Ok(result_fail(format!(
                    "type.construct_variant_start: variant '{}.{}' has discriminant {}, metadata reports {}",
                    expected_owner, variant.name, variant.discriminant, metadata_discriminant
                )));
            }
            if metadata_fields.len() != variant.fields.len() {
                return Ok(result_fail(format!(
                    "type.construct_variant_start: variant '{}.{}' expects {} payload field(s), metadata reports {}",
                    expected_owner,
                    variant.name,
                    variant.fields.len(),
                    metadata_fields.len()
                )));
            }
            for (index, field) in variant.fields.iter().enumerate() {
                let metadata_field = &metadata_fields[index];
                if let Err(message) = Self::validate_field_metadata_owner(
                    metadata_field,
                    &expected_owner,
                    Some(&variant.name),
                    "type.construct_variant_start",
                ) {
                    return Ok(result_fail(message));
                }
                let actual_type_name = type_expr_display(&field.ty);
                if metadata_field.index != index
                    || metadata_field.name != field.name
                    || metadata_field.type_name != actual_type_name
                {
                    return Ok(result_fail(format!(
                        "type.construct_variant_start: payload field metadata at index {} does not match variant '{}.{}'",
                        index, expected_owner, variant.name
                    )));
                }
            }
            variant.name.clone()
        };

        Ok(result_ok(Value::TypeConstruction {
            type_name: expected_owner,
            variant: Some(variant_name),
            state: None,
            fields: Vec::new(),
        }))
    }

    fn reflected_construct_machine_start(
        &self,
        owner_ty: &TypeExpr,
        state_metadata: &Value,
    ) -> Result<Value, String> {
        let owner_kind = self
            .checked_construction_kind(owner_ty)
            .unwrap_or_else(|| self.type_expr_kind(owner_ty));
        if !matches!(owner_kind, "machine" | "machine_state") {
            return Ok(result_fail(format!(
                "type.construct_machine_start supports only machines and machine states, got '{}'",
                type_expr_display(owner_ty)
            )));
        }

        let expected_owner = type_expr_display(owner_ty);
        let expected_machine = type_expr_name(owner_ty);
        let (state_index, metadata_owner, metadata_name) =
            match Self::type_machine_state_metadata(state_metadata, "type.construct_machine_start")
            {
                Ok(metadata) => metadata,
                Err(message) => return Ok(result_fail(message)),
            };
        if metadata_owner != expected_machine {
            return Ok(result_fail(format!(
                "type.construct_machine_start: state metadata belongs to machine '{}', expected '{}'",
                metadata_owner, expected_machine
            )));
        }
        let metadata_fields = match Self::type_machine_state_payload_field_metadata(
            state_metadata,
            "type.construct_machine_start",
        ) {
            Ok(fields) => fields,
            Err(message) => return Ok(result_fail(message)),
        };

        let state_name = if let Some(machine) = self.checked_machine(owner_ty) {
            let Some(state) = machine.states.get(state_index) else {
                return Ok(result_fail(format!(
                    "type.construct_machine_start: machine '{}' has no state at index {}",
                    expected_machine, state_index
                )));
            };
            if state.name != metadata_name {
                return Ok(result_fail(format!(
                    "type.construct_machine_start: state metadata '{}' does not match state '{}' on '{}'",
                    metadata_name, state.name, expected_machine
                )));
            }
            if metadata_fields.len() != state.fields.len() {
                return Ok(result_fail(format!(
                    "type.construct_machine_start: state '{}.{}' expects {} payload field(s), metadata reports {}",
                    expected_machine,
                    state.name,
                    state.fields.len(),
                    metadata_fields.len()
                )));
            }
            for (index, field) in state.fields.iter().enumerate() {
                let metadata_field = &metadata_fields[index];
                if let Err(message) = Self::validate_field_metadata_owner(
                    metadata_field,
                    &expected_machine,
                    Some(&state.name),
                    "type.construct_machine_start",
                ) {
                    return Ok(result_fail(message));
                }
                if metadata_field.index != index
                    || metadata_field.name != field.name
                    || metadata_field.type_name != field.type_name
                {
                    return Ok(result_fail(format!(
                        "type.construct_machine_start: payload field metadata at index {} does not match state '{}.{}'",
                        index, expected_machine, state.name
                    )));
                }
            }
            state.name.clone()
        } else {
            if self.checked_metadata_kind_is(owner_ty, &["machine", "machine_state"]) {
                return Ok(result_fail(
                    self.missing_checked_metadata_error(owner_ty, "machine"),
                ));
            }
            let machine = self.type_expr_machine(owner_ty);
            let Some(state) = machine.states.get(state_index) else {
                return Ok(result_fail(format!(
                    "type.construct_machine_start: machine '{}' has no state at index {}",
                    expected_machine, state_index
                )));
            };
            if state.name != metadata_name {
                return Ok(result_fail(format!(
                    "type.construct_machine_start: state metadata '{}' does not match state '{}' on '{}'",
                    metadata_name, state.name, expected_machine
                )));
            }
            if metadata_fields.len() != state.fields.len() {
                return Ok(result_fail(format!(
                    "type.construct_machine_start: state '{}.{}' expects {} payload field(s), metadata reports {}",
                    expected_machine,
                    state.name,
                    state.fields.len(),
                    metadata_fields.len()
                )));
            }
            for (index, field) in state.fields.iter().enumerate() {
                let metadata_field = &metadata_fields[index];
                if let Err(message) = Self::validate_field_metadata_owner(
                    metadata_field,
                    &expected_machine,
                    Some(&state.name),
                    "type.construct_machine_start",
                ) {
                    return Ok(result_fail(message));
                }
                let actual_type_name = type_expr_display(&field.ty);
                if metadata_field.index != index
                    || metadata_field.name != field.name
                    || metadata_field.type_name != actual_type_name
                {
                    return Ok(result_fail(format!(
                        "type.construct_machine_start: payload field metadata at index {} does not match state '{}.{}'",
                        index, expected_machine, state.name
                    )));
                }
            }
            state.name.clone()
        };

        Ok(result_ok(Value::TypeConstruction {
            type_name: expected_owner,
            variant: None,
            state: Some(state_name),
            fields: Vec::new(),
        }))
    }

    fn reflected_construct_finish(
        &mut self,
        owner_ty: &TypeExpr,
        builder: &Value,
    ) -> Result<Value, String> {
        let Value::TypeConstruction {
            type_name,
            variant,
            state,
            fields,
        } = builder
        else {
            return Err(format!(
                "type.construct_finish: first argument must be TypeConstruction, got {builder}"
            ));
        };

        let expected_owner = type_expr_display(owner_ty);
        if type_name != &expected_owner {
            return Ok(result_fail(format!(
                "type.construct_finish: builder for '{}' cannot construct '{}'",
                type_name, expected_owner
            )));
        }
        let ast_owner_kind = self.type_expr_kind(owner_ty);
        let owner_kind = match self.checked_construction_kind(owner_ty) {
            Some("struct") => "struct",
            Some("enum") => "enum",
            Some("bitfield") => "bitfield",
            Some("machine") => "machine",
            Some("machine_state") => "machine_state",
            _ => ast_owner_kind,
        };
        if !matches!(
            owner_kind,
            "struct" | "bitfield" | "enum" | "machine" | "machine_state"
        ) {
            return Ok(result_fail(format!(
                "type.construct_finish supports only structs, bitfields, enums, and machines, got '{}'",
                type_expr_display(owner_ty)
            )));
        }

        if owner_kind == "enum" {
            if state.is_some() {
                return Ok(result_fail(format!(
                    "type.construct_finish: machine state builder cannot construct '{}'",
                    expected_owner
                )));
            }
            return self.reflected_construct_finish_enum(
                owner_ty,
                variant.as_deref(),
                fields,
                &expected_owner,
            );
        }
        if matches!(owner_kind, "machine" | "machine_state") {
            if variant.is_some() {
                return Ok(result_fail(format!(
                    "type.construct_finish: enum variant builder cannot construct '{}'",
                    expected_owner
                )));
            }
            return self.reflected_construct_finish_machine(
                owner_ty,
                state.as_deref(),
                fields,
                &expected_owner,
            );
        }
        if owner_kind == "bitfield" {
            if variant.is_some() {
                return Ok(result_fail(format!(
                    "type.construct_finish: enum variant builder cannot construct '{}'",
                    expected_owner
                )));
            }
            if state.is_some() {
                return Ok(result_fail(format!(
                    "type.construct_finish: machine state builder cannot construct '{}'",
                    expected_owner
                )));
            }
            return self.reflected_construct_finish_bitfield(owner_ty, fields, &expected_owner);
        }
        if variant.is_some() {
            return Ok(result_fail(format!(
                "type.construct_finish: enum variant builder cannot construct '{}'",
                expected_owner
            )));
        }
        if state.is_some() {
            return Ok(result_fail(format!(
                "type.construct_finish: machine state builder cannot construct '{}'",
                expected_owner
            )));
        }

        if let Some(reflected_fields) = self
            .checked_type_fields(owner_ty)
            .map(|fields| fields.to_vec())
        {
            return self.reflected_construct_finish_struct_checked(
                owner_ty,
                fields,
                &expected_owner,
                &reflected_fields,
            );
        }

        if self.checked_metadata_kind_is(owner_ty, &["struct"]) {
            return Ok(result_fail(
                self.missing_checked_metadata_error(owner_ty, "field"),
            ));
        }

        let reflected_fields = self.type_expr_fields(owner_ty);
        let mut struct_fields = Vec::with_capacity(reflected_fields.len());
        for (index, field) in reflected_fields.iter().enumerate() {
            let Some((_, _, _, value)) = fields
                .iter()
                .find(|(field_index, _, _, _)| *field_index == index)
            else {
                return Ok(result_fail(format!(
                    "type.construct_finish: '{}' is missing required field '{}'",
                    expected_owner, field.name
                )));
            };

            let field_ty = self.substitute_type_expr(&field.ty);
            let type_name = type_expr_name(&field_ty);
            let value = match self.normalize_value_for_type(&field_ty, value.clone()) {
                Ok(value) => value,
                Err(message) => return Ok(result_fail(message)),
            };
            if let Err(message) = self.check_refinement(&type_name, &value) {
                return Ok(result_fail(message));
            }
            struct_fields.push((field.name.clone(), value));
        }

        Ok(result_ok(Value::Struct {
            type_name: type_expr_name(owner_ty),
            fields: struct_fields,
        }))
    }

    fn reflected_construct_finish_struct_checked(
        &mut self,
        owner_ty: &TypeExpr,
        fields: &[(usize, String, String, Value)],
        expected_owner: &str,
        reflected_fields: &[ReflectionFieldInfo],
    ) -> Result<Value, String> {
        let mut struct_fields = Vec::with_capacity(reflected_fields.len());
        for (index, field) in reflected_fields.iter().enumerate() {
            let Some((_, _, _, value)) = fields
                .iter()
                .find(|(field_index, _, _, _)| *field_index == index)
            else {
                return Ok(result_fail(format!(
                    "type.construct_finish: '{}' is missing required field '{}'",
                    expected_owner, field.name
                )));
            };

            let value = match self.normalize_value_for_type_name(&field.type_name, value.clone()) {
                Ok(value) => value,
                Err(message) => return Ok(result_fail(message)),
            };
            let type_name = Self::reflection_field_refinement_name(field);
            if let Err(message) = self.check_refinement(&type_name, &value) {
                return Ok(result_fail(message));
            }
            struct_fields.push((field.name.clone(), value));
        }

        Ok(result_ok(Value::Struct {
            type_name: type_expr_name(owner_ty),
            fields: struct_fields,
        }))
    }

    fn reflected_construct_finish_enum(
        &mut self,
        owner_ty: &TypeExpr,
        selected_variant: Option<&str>,
        fields: &[(usize, String, String, Value)],
        expected_owner: &str,
    ) -> Result<Value, String> {
        let Some(variant_name) = selected_variant else {
            return Ok(result_fail(
                "type.construct_finish: enum construction requires type.construct_variant_start"
                    .to_string(),
            ));
        };
        if let Some(variants) = self
            .checked_type_variants(owner_ty)
            .map(|variants| variants.to_vec())
        {
            let Some(variant) = variants
                .into_iter()
                .find(|candidate| candidate.name == variant_name)
            else {
                return Ok(result_fail(format!(
                    "type.construct_finish: enum '{}' has no variant '{}'",
                    expected_owner, variant_name
                )));
            };

            let mut enum_fields = Vec::with_capacity(variant.fields.len());
            for (index, field) in variant.fields.iter().enumerate() {
                let Some((_, _, _, value)) = fields
                    .iter()
                    .find(|(field_index, _, _, _)| *field_index == index)
                else {
                    return Ok(result_fail(format!(
                        "type.construct_finish: variant '{}.{}' is missing required payload field '{}'",
                        expected_owner, variant.name, field.name
                    )));
                };

                let value =
                    match self.normalize_value_for_type_name(&field.type_name, value.clone()) {
                        Ok(value) => value,
                        Err(message) => return Ok(result_fail(message)),
                    };
                let type_name = Self::reflection_field_refinement_name(field);
                if let Err(message) = self.check_refinement(&type_name, &value) {
                    return Ok(result_fail(message));
                }
                enum_fields.push(value);
            }

            return Ok(result_ok(Value::Enum {
                type_name: type_expr_name(owner_ty),
                variant: variant.name,
                fields: enum_fields,
            }));
        }

        if self.checked_metadata_kind_is(owner_ty, &["enum"]) {
            return Ok(result_fail(
                self.missing_checked_metadata_error(owner_ty, "variant"),
            ));
        }

        let Some(variant) = self
            .type_expr_variants(owner_ty)
            .into_iter()
            .find(|candidate| candidate.name == variant_name)
        else {
            return Ok(result_fail(format!(
                "type.construct_finish: enum '{}' has no variant '{}'",
                expected_owner, variant_name
            )));
        };

        let mut enum_fields = Vec::with_capacity(variant.fields.len());
        for (index, field) in variant.fields.iter().enumerate() {
            let Some((_, _, _, value)) = fields
                .iter()
                .find(|(field_index, _, _, _)| *field_index == index)
            else {
                return Ok(result_fail(format!(
                    "type.construct_finish: variant '{}.{}' is missing required payload field '{}'",
                    expected_owner, variant.name, field.name
                )));
            };

            let field_ty = self.substitute_type_expr(&field.ty);
            let type_name = type_expr_name(&field_ty);
            let value = match self.normalize_value_for_type(&field_ty, value.clone()) {
                Ok(value) => value,
                Err(message) => return Ok(result_fail(message)),
            };
            if let Err(message) = self.check_refinement(&type_name, &value) {
                return Ok(result_fail(message));
            }
            enum_fields.push(value);
        }

        Ok(result_ok(Value::Enum {
            type_name: type_expr_name(owner_ty),
            variant: variant.name,
            fields: enum_fields,
        }))
    }

    fn reflected_construct_finish_machine(
        &mut self,
        owner_ty: &TypeExpr,
        selected_state: Option<&str>,
        fields: &[(usize, String, String, Value)],
        _expected_owner: &str,
    ) -> Result<Value, String> {
        let Some(state_name) = selected_state else {
            return Ok(result_fail(
                "type.construct_finish: machine construction requires type.construct_machine_start"
                    .to_string(),
            ));
        };

        let machine_name = type_expr_name(owner_ty);
        if let Some(expected_state) = type_expr_state_name(owner_ty)
            && state_name != expected_state
        {
            return Ok(result_fail(format!(
                "type.construct_finish: machine target '{} at {}' cannot finish state '{}'",
                machine_name, expected_state, state_name
            )));
        }

        if let Some(machine) = self.checked_machine(owner_ty).cloned() {
            let Some(state) = machine
                .states
                .into_iter()
                .find(|candidate| candidate.name == state_name)
            else {
                return Ok(result_fail(format!(
                    "type.construct_finish: machine '{}' has no state '{}'",
                    machine_name, state_name
                )));
            };

            let mut machine_fields = Vec::with_capacity(state.fields.len());
            for (index, field) in state.fields.iter().enumerate() {
                let Some((_, _, _, value)) = fields
                    .iter()
                    .find(|(field_index, _, _, _)| *field_index == index)
                else {
                    return Ok(result_fail(format!(
                        "type.construct_finish: state '{}.{}' is missing required payload field '{}'",
                        machine_name, state.name, field.name
                    )));
                };

                let value =
                    match self.normalize_value_for_type_name(&field.type_name, value.clone()) {
                        Ok(value) => value,
                        Err(message) => return Ok(result_fail(message)),
                    };
                let type_name = Self::reflection_field_refinement_name(field);
                if let Err(message) = self.check_refinement(&type_name, &value) {
                    return Ok(result_fail(message));
                }
                machine_fields.push(value);
            }

            return Ok(result_ok(Value::Machine {
                type_name: machine_name,
                state: state.name,
                fields: machine_fields,
            }));
        }

        if self.checked_metadata_kind_is(owner_ty, &["machine", "machine_state"]) {
            return Ok(result_fail(
                self.missing_checked_metadata_error(owner_ty, "machine"),
            ));
        }

        let Some(state) = self
            .type_expr_machine(owner_ty)
            .states
            .into_iter()
            .find(|candidate| candidate.name == state_name)
        else {
            return Ok(result_fail(format!(
                "type.construct_finish: machine '{}' has no state '{}'",
                machine_name, state_name
            )));
        };

        let mut machine_fields = Vec::with_capacity(state.fields.len());
        for (index, field) in state.fields.iter().enumerate() {
            let Some((_, _, _, value)) = fields
                .iter()
                .find(|(field_index, _, _, _)| *field_index == index)
            else {
                return Ok(result_fail(format!(
                    "type.construct_finish: state '{}.{}' is missing required payload field '{}'",
                    machine_name, state.name, field.name
                )));
            };

            let field_ty = self.substitute_type_expr(&field.ty);
            let type_name = type_expr_name(&field_ty);
            let value = match self.normalize_value_for_type(&field_ty, value.clone()) {
                Ok(value) => value,
                Err(message) => return Ok(result_fail(message)),
            };
            if let Err(message) = self.check_refinement(&type_name, &value) {
                return Ok(result_fail(message));
            }
            machine_fields.push(value);
        }

        Ok(result_ok(Value::Machine {
            type_name: machine_name,
            state: state.name,
            fields: machine_fields,
        }))
    }

    fn reflected_construct_finish_bitfield(
        &self,
        owner_ty: &TypeExpr,
        fields: &[(usize, String, String, Value)],
        expected_owner: &str,
    ) -> Result<Value, String> {
        let bitfield_name = type_expr_name(owner_ty);
        if let Some(bitfield) = self.checked_bitfield(owner_ty).cloned() {
            let mut bitfield_fields = Vec::with_capacity(bitfield.fields.len());
            for (index, field) in bitfield.fields.iter().enumerate() {
                let Some((_, _, _, value)) = fields
                    .iter()
                    .find(|(field_index, _, _, _)| *field_index == index)
                else {
                    return Ok(result_fail(format!(
                        "type.construct_finish: '{}' is missing required field '{}'",
                        expected_owner, field.name
                    )));
                };

                let mut value = value.clone();
                if field.shape == "bits" {
                    if field.enum_type.is_some() {
                        if let Err(message) = self.checked_bitfield_field_numeric_value(
                            &bitfield_name,
                            &field.name,
                            field.width,
                            field.enum_type.as_ref(),
                            &value,
                        ) {
                            return Ok(result_fail(message));
                        }
                    } else {
                        value = match Self::normalized_plain_bitfield_field_value(
                            &bitfield_name,
                            &field.name,
                            field.width as u16,
                            &value,
                        ) {
                            Ok(value) => value,
                            Err(message) => return Ok(result_fail(message)),
                        };
                    }
                }

                bitfield_fields.push((field.name.clone(), value));
            }

            return Ok(result_ok(Value::Struct {
                type_name: bitfield_name,
                fields: bitfield_fields,
            }));
        }

        if self.checked_metadata_kind_is(owner_ty, &["bitfield"]) {
            return Ok(result_fail(
                self.missing_checked_metadata_error(owner_ty, "bitfield"),
            ));
        }

        let bitfield =
            self.bitfields.get(&bitfield_name).cloned().ok_or_else(|| {
                format!("type.construct_finish: unknown bitfield '{bitfield_name}'")
            })?;
        let reflected_fields = self.type_expr_fields(owner_ty);
        let mut bitfield_fields = Vec::with_capacity(reflected_fields.len());

        for (index, field) in reflected_fields.iter().enumerate() {
            let Some((_, _, _, value)) = fields
                .iter()
                .find(|(field_index, _, _, _)| *field_index == index)
            else {
                return Ok(result_fail(format!(
                    "type.construct_finish: '{}' is missing required field '{}'",
                    expected_owner, field.name
                )));
            };

            if let Some(field_def) = bitfield.fields.get(index)
                && let BitfieldFieldKind::Bits { width, as_type } = &field_def.kind
            {
                let value = if as_type.is_none() {
                    match Self::normalized_plain_bitfield_field_value(
                        &bitfield.name.name,
                        &field_def.name.name,
                        *width,
                        value,
                    ) {
                        Ok(value) => value,
                        Err(message) => return Ok(result_fail(message)),
                    }
                } else {
                    if let Err(message) = self.bitfield_field_numeric_value(
                        &bitfield,
                        &field_def.name.name,
                        *width,
                        as_type.as_ref(),
                        value,
                    ) {
                        return Ok(result_fail(message));
                    }
                    value.clone()
                };
                bitfield_fields.push((field.name.clone(), value));
                continue;
            }

            bitfield_fields.push((field.name.clone(), value.clone()));
        }

        Ok(result_ok(Value::Struct {
            type_name: bitfield_name,
            fields: bitfield_fields,
        }))
    }

    fn type_field_metadata(value: &Value) -> Result<TypeFieldMetadata, String> {
        Self::type_field_metadata_for(value, "type.field_value")
    }

    fn type_variant_metadata(
        value: &Value,
        caller: &str,
    ) -> Result<(usize, String, String, i64), String> {
        let Value::Struct { type_name, fields } = value else {
            return Err(format!(
                "{caller}: argument must be TypeVariant, got {value}"
            ));
        };
        if type_name != "TypeVariant" {
            return Err(format!(
                "{caller}: argument must be TypeVariant, got {type_name}"
            ));
        }

        let field_value = |name: &str| {
            fields
                .iter()
                .find(|(field_name, _)| field_name == name)
                .map(|(_, field_value)| field_value)
                .ok_or_else(|| format!("{caller}: TypeVariant is missing '{name}'"))
        };

        let index = match field_value("index")? {
            Value::Int64(index) if *index >= 0 => *index as usize,
            other => {
                return Err(format!(
                    "{caller}: TypeVariant.index must be a non-negative int64, got {other}"
                ));
            }
        };
        let owner_type = match field_value("owner_type")? {
            Value::String(owner_type) => owner_type.clone(),
            other => {
                return Err(format!(
                    "{caller}: TypeVariant.owner_type must be string, got {other}"
                ));
            }
        };
        let name = match field_value("name")? {
            Value::String(name) => name.clone(),
            other => {
                return Err(format!(
                    "{caller}: TypeVariant.name must be string, got {other}"
                ));
            }
        };
        let discriminant = match field_value("discriminant")? {
            Value::Int64(discriminant) => *discriminant,
            other => {
                return Err(format!(
                    "{caller}: TypeVariant.discriminant must be int64, got {other}"
                ));
            }
        };

        Ok((index, owner_type, name, discriminant))
    }

    fn type_variant_payload_field_metadata(
        value: &Value,
        caller: &str,
    ) -> Result<Vec<TypeFieldMetadata>, String> {
        let Value::Struct { type_name, fields } = value else {
            return Err(format!(
                "{caller}: argument must be TypeVariant, got {value}"
            ));
        };
        if type_name != "TypeVariant" {
            return Err(format!(
                "{caller}: argument must be TypeVariant, got {type_name}"
            ));
        }
        let field_values = fields
            .iter()
            .find(|(field_name, _)| field_name == "fields")
            .map(|(_, field_value)| field_value)
            .ok_or_else(|| format!("{caller}: TypeVariant is missing 'fields'"))?;
        let Value::List(field_values) = field_values else {
            return Err(format!(
                "{caller}: TypeVariant.fields must be list[TypeField], got {field_values}"
            ));
        };

        field_values
            .iter()
            .map(|field| Self::type_field_metadata_for(field, caller))
            .collect()
    }

    fn type_machine_state_metadata(
        value: &Value,
        caller: &str,
    ) -> Result<(usize, String, String), String> {
        let Value::Struct { type_name, fields } = value else {
            return Err(format!(
                "{caller}: argument must be TypeMachineState, got {value}"
            ));
        };
        if type_name != "TypeMachineState" {
            return Err(format!(
                "{caller}: argument must be TypeMachineState, got {type_name}"
            ));
        }

        let field_value = |name: &str| {
            fields
                .iter()
                .find(|(field_name, _)| field_name == name)
                .map(|(_, field_value)| field_value)
                .ok_or_else(|| format!("{caller}: TypeMachineState is missing '{name}'"))
        };

        let index = match field_value("index")? {
            Value::Int64(index) if *index >= 0 => *index as usize,
            other => {
                return Err(format!(
                    "{caller}: TypeMachineState.index must be a non-negative int64, got {other}"
                ));
            }
        };
        let owner_type = match field_value("owner_type")? {
            Value::String(owner_type) => owner_type.clone(),
            other => {
                return Err(format!(
                    "{caller}: TypeMachineState.owner_type must be string, got {other}"
                ));
            }
        };
        let name = match field_value("name")? {
            Value::String(name) => name.clone(),
            other => {
                return Err(format!(
                    "{caller}: TypeMachineState.name must be string, got {other}"
                ));
            }
        };

        Ok((index, owner_type, name))
    }

    fn type_machine_state_payload_field_metadata(
        value: &Value,
        caller: &str,
    ) -> Result<Vec<TypeFieldMetadata>, String> {
        let Value::Struct { type_name, fields } = value else {
            return Err(format!(
                "{caller}: argument must be TypeMachineState, got {value}"
            ));
        };
        if type_name != "TypeMachineState" {
            return Err(format!(
                "{caller}: argument must be TypeMachineState, got {type_name}"
            ));
        }
        let field_values = fields
            .iter()
            .find(|(field_name, _)| field_name == "fields")
            .map(|(_, field_value)| field_value)
            .ok_or_else(|| format!("{caller}: TypeMachineState is missing 'fields'"))?;
        let Value::List(field_values) = field_values else {
            return Err(format!(
                "{caller}: TypeMachineState.fields must be list[TypeField], got {field_values}"
            ));
        };

        field_values
            .iter()
            .map(|field| Self::type_field_metadata_for(field, caller))
            .collect()
    }

    fn type_field_metadata_for(value: &Value, caller: &str) -> Result<TypeFieldMetadata, String> {
        let Value::Struct { type_name, fields } = value else {
            return Err(format!(
                "{caller}: second argument must be TypeField, got {value}"
            ));
        };
        if type_name != "TypeField" {
            return Err(format!(
                "{caller}: second argument must be TypeField, got {type_name}"
            ));
        }

        let field_value = |name: &str| {
            fields
                .iter()
                .find(|(field_name, _)| field_name == name)
                .map(|(_, field_value)| field_value)
                .ok_or_else(|| format!("{caller}: TypeField is missing '{name}'"))
        };

        let index = match field_value("index")? {
            Value::Int64(index) if *index >= 0 => *index as usize,
            other => {
                return Err(format!(
                    "{caller}: TypeField.index must be a non-negative int64, got {other}"
                ));
            }
        };
        let owner_type = match field_value("owner_type")? {
            Value::String(owner_type) => owner_type.clone(),
            other => {
                return Err(format!(
                    "{caller}: TypeField.owner_type must be string, got {other}"
                ));
            }
        };
        let owner_member = match field_value("owner_member")? {
            Value::OptionalNone => None,
            Value::OptionalSome(value) => match value.as_ref() {
                Value::String(owner_member) => Some(owner_member.clone()),
                other => {
                    return Err(format!(
                        "{caller}: TypeField.owner_member must be optional[string], got some({other})"
                    ));
                }
            },
            other => {
                return Err(format!(
                    "{caller}: TypeField.owner_member must be optional[string], got {other}"
                ));
            }
        };
        let name = match field_value("name")? {
            Value::String(name) => name.clone(),
            other => {
                return Err(format!(
                    "{caller}: TypeField.name must be string, got {other}"
                ));
            }
        };
        let type_name = match field_value("type_name")? {
            Value::String(type_name) => type_name.clone(),
            other => {
                return Err(format!(
                    "{caller}: TypeField.type_name must be string, got {other}"
                ));
            }
        };

        Ok(TypeFieldMetadata {
            index,
            owner_type,
            owner_member,
            name,
            type_name,
        })
    }

    fn field_owner_label(owner_type: &str, owner_member: Option<&str>) -> String {
        owner_member
            .map(|member| format!("{owner_type}.{member}"))
            .unwrap_or_else(|| owner_type.to_string())
    }

    fn field_metadata_owner_label(metadata: &TypeFieldMetadata) -> String {
        Self::field_owner_label(&metadata.owner_type, metadata.owner_member.as_deref())
    }

    fn validate_field_metadata_owner(
        metadata: &TypeFieldMetadata,
        expected_owner_type: &str,
        expected_owner_member: Option<&str>,
        caller: &str,
    ) -> Result<(), String> {
        if metadata.owner_type == expected_owner_type
            && metadata.owner_member.as_deref() == expected_owner_member
        {
            return Ok(());
        }

        Err(format!(
            "{caller}: field metadata belongs to '{}', expected '{}'",
            Self::field_metadata_owner_label(metadata),
            Self::field_owner_label(expected_owner_type, expected_owner_member)
        ))
    }

    fn type_info_metadata(value: &Value) -> Result<String, String> {
        let Value::Struct { type_name, fields } = value else {
            return Err(format!("expected TypeInfo, got {value}"));
        };
        if type_name != "TypeInfo" {
            return Err(format!("expected TypeInfo, got {type_name}"));
        }

        match fields
            .iter()
            .find(|(field_name, _)| field_name == "type_name")
            .map(|(_, field_value)| field_value)
        {
            Some(Value::String(type_name)) => Ok(type_name.clone()),
            Some(other) => Err(format!("TypeInfo.type_name must be string, got {other}")),
            None => Err("TypeInfo is missing 'type_name'".to_string()),
        }
    }

    fn parse_hex_bytes(
        raw: &str,
        even_length_error: &str,
        invalid_hex_error: &str,
    ) -> Result<Vec<u8>, String> {
        let hex = raw.strip_prefix("0x").unwrap_or(raw);
        if !hex.len().is_multiple_of(2) {
            return Err(even_length_error.to_string());
        }
        if !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(invalid_hex_error.to_string());
        }
        let mut bytes = Vec::with_capacity(hex.len() / 2);
        for pair in hex.as_bytes().chunks_exact(2) {
            let pair = std::str::from_utf8(pair).map_err(|_| invalid_hex_error.to_string())?;
            let byte = u8::from_str_radix(pair, 16).map_err(|_| invalid_hex_error.to_string())?;
            bytes.push(byte);
        }
        Ok(bytes)
    }

    // =========================================================================
    // Built-in implementations
    //
    // This function handles two distinct categories:
    //
    //   COMPILER PRIMITIVES — operations that are semantically special and
    //   will remain as interpreter/compiler built-ins permanently (I/O
    //   capabilities, JSON serialization, secret-taint ops, random).
    //
    //   STANDARD LIBRARY STUBS — functions that belong to the Jett standard
    //   library (stdlib/*.jett) but are implemented here as interpreter
    //   builtins until Phase D code generation is complete.  Once codegen
    //   exists these should migrate to actual Jett source files.
    // =========================================================================

    fn call_builtin(&mut self, name: &str, args: &[Value]) -> Option<Result<Value, String>> {
        if let Some(result) = self.call_bitfield_builtin(name, args) {
            return Some(result);
        }

        match name {
            // =================================================================
            // COMPILER PRIMITIVES
            // =================================================================

            // -- I/O (capability-simulated) -----------------------------------
            "Stdout.write" => {
                // Stdout.write(stdout, message) — ignore capability, print message
                if args.len() < 2 {
                    return Some(Err(format!(
                        "Stdout.write expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                // The first arg is the capability (ignored), second is the message.
                self.write_stdout(&format!("{}", args[1]));
                Some(Ok(Value::Nothing))
            }

            // -- Secret-safe operations --------------------------------------
            "secret.redact" => {
                require_args!(name, 1, args);
                Some(Ok(Value::String("***".to_string())))
            }

            "secret.compare" => {
                require_args!(name, 2, args);
                Some(Ok(Value::Bool(args[0] == args[1])))
            }

            // -- Random operations (stdlib/random.jett) -----------------------
            "random.int64" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Int64(lo), Value::Int64(hi)) => {
                        if lo >= hi {
                            Some(Err(format!(
                                "random.int64: lo ({lo}) must be less than hi ({hi})"
                            )))
                        } else {
                            let n = rand::thread_rng().gen_range(*lo..*hi);
                            Some(Ok(Value::Int64(n)))
                        }
                    }
                    _ => Some(Err(format!("{name} expects two int64 arguments"))),
                }
            }

            "random.float64" => {
                require_args!(name, 0, args);
                let f: f64 = rand::thread_rng().gen_range(0.0f64..1.0f64);
                Some(Ok(Value::Float64(f)))
            }

            "random.bool" => {
                require_args!(name, 0, args);
                Some(Ok(Value::Bool(rand::thread_rng().gen_bool(0.5))))
            }

            "random.choice" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        if items.is_empty() {
                            Some(Ok(Value::OptionalNone))
                        } else {
                            let idx = rand::thread_rng().gen_range(0..items.len());
                            Some(Ok(Value::OptionalSome(Box::new(items[idx].clone()))))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "random.shuffle" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        use rand::seq::SliceRandom;
                        let mut shuffled = items.clone();
                        shuffled.shuffle(&mut rand::thread_rng());
                        Some(Ok(Value::List(shuffled)))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            // =================================================================
            // STANDARD LIBRARY STUBS
            // (will migrate to stdlib/*.jett once codegen is available)
            // =================================================================

            // -- String operations (stdlib/string.jett) -----------------------
            "string.length" | "string.char_count" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::Int64(string_grapheme_count(s) as i64))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.contains" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::String(substr)) => {
                        Some(Ok(Value::Bool(string_contains_grapheme(s, substr))))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }

            "string.trim" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::String(s.trim().to_string()))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.upper" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::String(s.to_uppercase()))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.lower" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::String(s.to_lowercase()))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.replace" => {
                require_args!(name, 3, args);
                match (&args[0], &args[1], &args[2]) {
                    (Value::String(s), Value::String(from), Value::String(to)) => {
                        Some(Ok(Value::String(s.replace(from.as_str(), to.as_str()))))
                    }
                    _ => Some(Err(format!("{name} expects three string arguments"))),
                }
            }

            "string.split" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::String(delim)) => {
                        let parts: Vec<Value> = s
                            .split(delim.as_str())
                            .map(|p| Value::String(p.to_string()))
                            .collect();
                        Some(Ok(Value::List(parts)))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }

            "string.join" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::String(sep)) => {
                        let strs: Result<Vec<String>, String> = items
                            .iter()
                            .map(|v| match v {
                                Value::String(s) => Ok(s.clone()),
                                _ => Err(format!("{name} requires a list of strings, found {v}")),
                            })
                            .collect();
                        match strs {
                            Ok(parts) => Some(Ok(Value::String(parts.join(sep)))),
                            Err(e) => Some(Err(e)),
                        }
                    }
                    _ => Some(Err(format!("{name} expects a list and a string separator"))),
                }
            }

            "string.starts_with" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::String(prefix)) => {
                        Some(Ok(Value::Bool(string_starts_with_grapheme(s, prefix))))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }

            "string.ends_with" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::String(suffix)) => {
                        Some(Ok(Value::Bool(string_ends_with_grapheme(s, suffix))))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }

            "string.is_empty" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::Bool(s.is_empty()))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.slice" => {
                require_args!(name, 3, args);
                match (&args[0], &args[1], &args[2]) {
                    (Value::String(s), Value::Int64(start), Value::Int64(end)) => {
                        let graphemes = string_graphemes(s);
                        let len = graphemes.len() as i64;
                        let start = (*start).clamp(0, len) as usize;
                        let end = (*end).clamp(0, len) as usize;
                        let result = graphemes[start.min(end)..end].concat();
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!(
                        "{name} expects a string and two int64 indices"
                    ))),
                }
            }

            "string.repeat" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = (*n).max(0) as usize;
                        Some(Ok(Value::String(s.repeat(n))))
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64"))),
                }
            }

            // string.pad_start is an alias for string.pad_left
            "string.pad_start" => self.call_builtin("string.pad_left", args),

            "string.pad_end" => {
                require_args!(name, 3, args);
                match (&args[0], &args[1], &args[2]) {
                    (Value::String(s), Value::Int64(width), Value::String(pad)) => {
                        let pad_unit = first_grapheme_or_space(pad);
                        let current_len = string_grapheme_count(s);
                        let width = (*width).max(0) as usize;
                        if current_len >= width {
                            Some(Ok(Value::String(s.clone())))
                        } else {
                            let padding = pad_unit.repeat(width - current_len);
                            Some(Ok(Value::String(format!("{s}{padding}"))))
                        }
                    }
                    _ => Some(Err(format!(
                        "{name} expects a string, int64 width, and string pad char"
                    ))),
                }
            }

            // -- Type conversions (stdlib/string.jett, stdlib/int64.jett) -----
            "string.from_int64" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Int64(n) => Some(Ok(Value::String(n.to_string()))),
                    _ => Some(Err(format!("{name} expects an int64 argument"))),
                }
            }
            "string.from_uint64" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Uint64(n) => Some(Ok(Value::String(n.to_string()))),
                    Value::Int64(n) if *n >= 0 => Some(Ok(Value::String(n.to_string()))),
                    _ => Some(Err(format!("{name} expects a uint64 argument"))),
                }
            }

            "string.is_not_empty" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::Bool(!s.is_empty()))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.slugify" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let slug: String = s
                            .to_lowercase()
                            .chars()
                            .map(|c| if c.is_alphanumeric() { c } else { '-' })
                            .collect::<String>()
                            .split('-')
                            .filter(|part| !part.is_empty())
                            .collect::<Vec<_>>()
                            .join("-");
                        Some(Ok(Value::String(slug)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.truncate" => {
                if args.len() != 3 {
                    return Some(Err(format!("{name} expects 3 arguments")));
                }
                match (&args[0], &args[1], &args[2]) {
                    (Value::String(s), Value::Int64(max_len), Value::String(suffix)) => {
                        let max = (*max_len).max(0) as usize;
                        let graphemes = string_graphemes(s);
                        let result = if graphemes.len() <= max {
                            s.clone()
                        } else {
                            // Keep first `max` graphemes, then append suffix.
                            let kept = graphemes[..max].concat();
                            format!("{kept}{suffix}")
                        };
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects (string, int64, string)"))),
                }
            }

            "string.between" => {
                if args.len() != 3 {
                    return Some(Err(format!("{name} expects 3 arguments")));
                }
                match (&args[0], &args[1], &args[2]) {
                    (Value::String(s), Value::String(start), Value::String(end)) => {
                        // Returns "" when the markers are not found (design doc shows plain string)
                        let result =
                            if let Some((_, start_end, _)) = string_find_grapheme_match(s, start) {
                                let after_start = &s[start_end..];
                                if let Some((end_pos, _, _)) =
                                    string_find_grapheme_match(after_start, end)
                                {
                                    after_start[..end_pos].to_string()
                                } else {
                                    String::new()
                                }
                            } else {
                                String::new()
                            };
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects (string, string, string)"))),
                }
            }

            // string.pad_left is the canonical name (design doc); pad_start is an alias
            "string.pad_left" => {
                require_args!(name, 3, args);
                match (&args[0], &args[1], &args[2]) {
                    (Value::String(s), Value::Int64(width), Value::String(pad)) => {
                        let pad_unit = first_grapheme_or_space(pad);
                        let current_len = string_grapheme_count(s);
                        let width = (*width).max(0) as usize;
                        if current_len >= width {
                            Some(Ok(Value::String(s.clone())))
                        } else {
                            let padding = pad_unit.repeat(width - current_len);
                            Some(Ok(Value::String(format!("{padding}{s}"))))
                        }
                    }
                    _ => Some(Err(format!("{name} expects (string, int64, string)"))),
                }
            }

            // -- int64 / float64 conversions ----------------------------------
            "int64.from_string" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => match s.parse::<i64>() {
                        Ok(n) => Some(Ok(Value::ResultOk(Box::new(Value::Int64(n))))),
                        Err(_) => Some(Ok(Value::ResultFail(Box::new(Value::String(format!(
                            "int64.from_string: cannot parse '{s}' as int64"
                        )))))),
                    },
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }
            "uint64.from_string" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => match s.parse::<u64>() {
                        Ok(n) => Some(Ok(Value::ResultOk(Box::new(Value::Uint64(n))))),
                        Err(_) => Some(Ok(Value::ResultFail(Box::new(Value::String(format!(
                            "uint64.from_string: cannot parse '{s}' as uint64"
                        )))))),
                    },
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // -- float64 conversions ------------------------------------------
            "float64.from_int64" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Int64(n) => Some(Ok(Value::Float64(*n as f64))),
                    _ => Some(Err(format!("{name} expects an int64 argument"))),
                }
            }
            "float64.from_string" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => match s.parse::<f64>() {
                        Ok(n) => Some(Ok(Value::ResultOk(Box::new(Value::Float64(n))))),
                        Err(_) => Some(Ok(Value::ResultFail(Box::new(Value::String(format!(
                            "float64.from_string: cannot parse '{s}' as float64"
                        )))))),
                    },
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // -- Additional string conversions --------------------------------
            "string.from_float64" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::String(format!("{n}")))),
                    _ => Some(Err(format!("{name} expects a float64 argument"))),
                }
            }
            "string.from_bool" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Bool(b) => Some(Ok(Value::String(format!("{b}")))),
                    _ => Some(Err(format!("{name} expects a bool argument"))),
                }
            }

            // -- print (debugging helper) -------------------------------------
            "print" => {
                let output: Vec<String> = args.iter().map(|v| format!("{v}")).collect();
                self.write_stdout(&output.join(" "));
                Some(Ok(Value::Nothing))
            }
            "println" => {
                let output: Vec<String> = args.iter().map(|v| format!("{v}")).collect();
                self.write_stdout_line(&output.join(" "));
                Some(Ok(Value::Nothing))
            }

            // -- List operations (stdlib/list.jett) ---------------------------
            "list.new" => {
                require_args!(name, 0, args);
                Some(Ok(Value::List(vec![])))
            }

            "list.length" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => Some(Ok(Value::Int64(items.len() as i64))),
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.append" => {
                require_args!(name, 2, args);
                match &args[0] {
                    Value::List(items) => {
                        let mut new_list = items.clone();
                        new_list.push(args[1].clone());
                        Some(Ok(Value::List(new_list)))
                    }
                    _ => Some(Err(format!("{name} expects a list as first argument"))),
                }
            }

            "list.get" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::Int64(index)) => {
                        let idx = *index as usize;
                        if idx < items.len() {
                            Some(Ok(Value::OptionalSome(Box::new(items[idx].clone()))))
                        } else {
                            Some(Ok(Value::OptionalNone))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a list and an int64 index"))),
                }
            }

            "list.first" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        if items.is_empty() {
                            Some(Ok(Value::OptionalNone))
                        } else {
                            Some(Ok(Value::OptionalSome(Box::new(items[0].clone()))))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.last" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        if items.is_empty() {
                            Some(Ok(Value::OptionalNone))
                        } else {
                            Some(Ok(Value::OptionalSome(Box::new(
                                items[items.len() - 1].clone(),
                            ))))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.is_empty" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => Some(Ok(Value::Bool(items.is_empty()))),
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.skip" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::Int64(n)) => {
                        let n = (*n).max(0) as usize;
                        Some(Ok(Value::List(items[n.min(items.len())..].to_vec())))
                    }
                    _ => Some(Err(format!("{name} expects a list and an int64"))),
                }
            }

            "list.take" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::Int64(n)) => {
                        let n = (*n).max(0) as usize;
                        Some(Ok(Value::List(items[..n.min(items.len())].to_vec())))
                    }
                    _ => Some(Err(format!("{name} expects a list and an int64"))),
                }
            }

            "list.reverse" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        let mut reversed = items.clone();
                        reversed.reverse();
                        Some(Ok(Value::List(reversed)))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.sort" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        let mut sorted = items.clone();
                        sorted.sort_by(|a, b| match (a, b) {
                            (Value::Int64(x), Value::Int64(y)) => x.cmp(y),
                            (Value::Float64(x), Value::Float64(y)) => {
                                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                            }
                            (Value::String(x), Value::String(y)) => x.cmp(y),
                            (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
                            _ => std::cmp::Ordering::Equal,
                        });
                        Some(Ok(Value::List(sorted)))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.contains" => {
                require_args!(name, 2, args);
                match &args[0] {
                    Value::List(items) => Some(Ok(Value::Bool(items.contains(&args[1])))),
                    _ => Some(Err(format!("{name} expects a list as first argument"))),
                }
            }

            "list.index_of" => {
                require_args!(name, 2, args);
                match &args[0] {
                    Value::List(items) => {
                        let idx = items.iter().position(|v| v == &args[1]);
                        Some(Ok(match idx {
                            Some(i) => Value::OptionalSome(Box::new(Value::Int64(i as i64))),
                            None => Value::OptionalNone,
                        }))
                    }
                    _ => Some(Err(format!("{name} expects a list as first argument"))),
                }
            }

            "list.remove" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::Int64(index)) => {
                        let idx = *index as usize;
                        if idx < items.len() {
                            let mut new_list = items.clone();
                            new_list.remove(idx);
                            Some(Ok(Value::List(new_list)))
                        } else {
                            Some(Err(format!("{name}: index {index} out of bounds")))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a list and an int64 index"))),
                }
            }

            "list.concat" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(a), Value::List(b)) => {
                        let mut result = a.clone();
                        result.extend(b.iter().cloned());
                        Some(Ok(Value::List(result)))
                    }
                    _ => Some(Err(format!("{name} expects two list arguments"))),
                }
            }

            "list.flatten" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        let mut flat = Vec::new();
                        for item in items {
                            match item {
                                Value::List(inner) => flat.extend(inner.iter().cloned()),
                                other => flat.push(other.clone()),
                            }
                        }
                        Some(Ok(Value::List(flat)))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.unique" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        let mut seen = Vec::new();
                        for item in items {
                            if !seen.contains(item) {
                                seen.push(item.clone());
                            }
                        }
                        Some(Ok(Value::List(seen)))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.zip" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(a), Value::List(b)) => {
                        let pairs: Vec<Value> = a
                            .iter()
                            .zip(b.iter())
                            .map(|(x, y)| Value::List(vec![x.clone(), y.clone()]))
                            .collect();
                        Some(Ok(Value::List(pairs)))
                    }
                    _ => Some(Err(format!("{name} expects two list arguments"))),
                }
            }

            // -- Map operations (stdlib/map.jett) -----------------------------
            "map.new" => Some(Ok(Value::Map(Vec::new()))),
            "map.length" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Map(entries) => Some(Ok(Value::Int64(entries.len() as i64))),
                    _ => Some(Err(format!("{name} expects a map argument"))),
                }
            }
            "map.has" => {
                require_args!(name, 2, args);
                match &args[0] {
                    Value::Map(entries) => {
                        let found = entries.iter().any(|(k, _)| k == &args[1]);
                        Some(Ok(Value::Bool(found)))
                    }
                    _ => Some(Err(format!("{name} expects a map as first argument"))),
                }
            }
            "map.get" => {
                require_args!(name, 2, args);
                match &args[0] {
                    Value::Map(entries) => {
                        let val = entries
                            .iter()
                            .find(|(k, _)| k == &args[1])
                            .map(|(_, v)| v.clone());
                        Some(Ok(match val {
                            Some(v) => Value::OptionalSome(Box::new(v)),
                            None => Value::OptionalNone,
                        }))
                    }
                    _ => Some(Err(format!("{name} expects a map as first argument"))),
                }
            }
            "map.insert" => {
                require_args!(name, 3, args);
                match args[0].clone() {
                    Value::Map(mut entries) => {
                        let key = args[1].clone();
                        let val = args[2].clone();
                        if let Some(entry) = entries.iter_mut().find(|(k, _)| k == &key) {
                            entry.1 = val;
                        } else {
                            entries.push((key, val));
                        }
                        Some(Ok(Value::Map(entries)))
                    }
                    _ => Some(Err(format!("{name} expects a map as first argument"))),
                }
            }
            "map.remove" => {
                require_args!(name, 2, args);
                match args[0].clone() {
                    Value::Map(mut entries) => {
                        entries.retain(|(k, _)| k != &args[1]);
                        Some(Ok(Value::Map(entries)))
                    }
                    _ => Some(Err(format!("{name} expects a map as first argument"))),
                }
            }
            "map.keys" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Map(entries) => {
                        let keys = entries.iter().map(|(k, _)| k.clone()).collect();
                        Some(Ok(Value::List(keys)))
                    }
                    _ => Some(Err(format!("{name} expects a map argument"))),
                }
            }
            "map.values" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Map(entries) => {
                        let vals = entries.iter().map(|(_, v)| v.clone()).collect();
                        Some(Ok(Value::List(vals)))
                    }
                    _ => Some(Err(format!("{name} expects a map argument"))),
                }
            }
            "map.is_empty" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Map(entries) => Some(Ok(Value::Bool(entries.is_empty()))),
                    _ => Some(Err(format!("{name} expects a map argument"))),
                }
            }

            // map.set is the canonical name (design doc); insert is an alias
            "map.set" => self.call_builtin("map.insert", args),

            "map.get_or" => {
                if args.len() != 3 {
                    return Some(Err(format!("{name} expects 3 arguments")));
                }
                match &args[0] {
                    Value::Map(entries) => {
                        let key = &args[1];
                        let default = args[2].clone();
                        let found = entries
                            .iter()
                            .find(|(k, _)| k == key)
                            .map(|(_, v)| v.clone())
                            .unwrap_or(default);
                        Some(Ok(found))
                    }
                    _ => Some(Err(format!("{name} expects a map argument"))),
                }
            }

            "map.merge" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Map(a), Value::Map(b)) => {
                        let mut merged = a.clone();
                        for (k, v) in b {
                            if let Some(pos) = merged.iter().position(|(mk, _)| mk == k) {
                                merged[pos].1 = v.clone();
                            } else {
                                merged.push((k.clone(), v.clone()));
                            }
                        }
                        Some(Ok(Value::Map(merged)))
                    }
                    _ => Some(Err(format!("{name} expects two map arguments"))),
                }
            }

            "map.contains_key" => self.call_builtin("map.has", args),

            "map.from_lists" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(keys), Value::List(values)) => {
                        let entries: Vec<(Value, Value)> = keys
                            .iter()
                            .zip(values.iter())
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        Some(Ok(Value::Map(entries)))
                    }
                    _ => Some(Err(format!("{name} expects two list arguments"))),
                }
            }

            "map.entries" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Map(entries) => {
                        let pairs = entries
                            .iter()
                            .map(|(k, v)| Value::List(vec![k.clone(), v.clone()]))
                            .collect();
                        Some(Ok(Value::List(pairs)))
                    }
                    _ => Some(Err(format!("{name} expects a map argument"))),
                }
            }

            // -- Set operations (stdlib/set.jett) -----------------------------
            "set.new" => Some(Ok(Value::Set(Vec::new()))),
            "set.add" => {
                require_args!(name, 2, args);
                match args[0].clone() {
                    Value::Set(mut items) => {
                        let val = args[1].clone();
                        if !items.contains(&val) {
                            items.push(val);
                        }
                        Some(Ok(Value::Set(items)))
                    }
                    _ => Some(Err(format!("{name} expects a set as first argument"))),
                }
            }
            "set.remove" => {
                require_args!(name, 2, args);
                match args[0].clone() {
                    Value::Set(mut items) => {
                        items.retain(|v| v != &args[1]);
                        Some(Ok(Value::Set(items)))
                    }
                    _ => Some(Err(format!("{name} expects a set as first argument"))),
                }
            }
            "set.contains" => {
                require_args!(name, 2, args);
                match &args[0] {
                    Value::Set(items) => Some(Ok(Value::Bool(items.contains(&args[1])))),
                    _ => Some(Err(format!("{name} expects a set as first argument"))),
                }
            }
            "set.length" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Set(items) => Some(Ok(Value::Int64(items.len() as i64))),
                    _ => Some(Err(format!("{name} expects a set argument"))),
                }
            }
            "set.is_empty" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Set(items) => Some(Ok(Value::Bool(items.is_empty()))),
                    _ => Some(Err(format!("{name} expects a set argument"))),
                }
            }
            "set.to_list" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Set(items) => Some(Ok(Value::List(items.clone()))),
                    _ => Some(Err(format!("{name} expects a set argument"))),
                }
            }
            "set.union" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Set(a), Value::Set(b)) => {
                        let mut result = a.clone();
                        for item in b {
                            if !result.contains(item) {
                                result.push(item.clone());
                            }
                        }
                        Some(Ok(Value::Set(result)))
                    }
                    _ => Some(Err(format!("{name} expects two set arguments"))),
                }
            }
            "set.intersection" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Set(a), Value::Set(b)) => {
                        let result: Vec<Value> =
                            a.iter().filter(|v| b.contains(v)).cloned().collect();
                        Some(Ok(Value::Set(result)))
                    }
                    _ => Some(Err(format!("{name} expects two set arguments"))),
                }
            }
            "set.difference" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Set(a), Value::Set(b)) => {
                        let result: Vec<Value> =
                            a.iter().filter(|v| !b.contains(v)).cloned().collect();
                        Some(Ok(Value::Set(result)))
                    }
                    _ => Some(Err(format!("{name} expects two set arguments"))),
                }
            }

            // -- Additional list operations ------------------------------------
            "list.chunk" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::Int64(size)) => {
                        let size = (*size).max(1) as usize;
                        let chunks: Vec<Value> = items
                            .chunks(size)
                            .map(|c| Value::List(c.to_vec()))
                            .collect();
                        Some(Ok(Value::List(chunks)))
                    }
                    _ => Some(Err(format!(
                        "{name} expects a list and an int64 chunk size"
                    ))),
                }
            }

            "list.sort_by_index" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::Int64(idx)) => {
                        let idx = *idx as usize;
                        let mut sorted = items.clone();
                        sorted.sort_by(|a, b| {
                            let va = match a {
                                Value::List(l) => l.get(idx).cloned(),
                                _ => None,
                            };
                            let vb = match b {
                                Value::List(l) => l.get(idx).cloned(),
                                _ => None,
                            };
                            match (va, vb) {
                                (Some(Value::String(sa)), Some(Value::String(sb))) => sa.cmp(&sb),
                                (Some(Value::Int64(ia)), Some(Value::Int64(ib))) => ia.cmp(&ib),
                                _ => std::cmp::Ordering::Equal,
                            }
                        });
                        Some(Ok(Value::List(sorted)))
                    }
                    _ => Some(Err(format!(
                        "{name} expects a list of lists and an int64 index"
                    ))),
                }
            }

            "list.is_sorted" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        let sorted = items.windows(2).all(|w| match (&w[0], &w[1]) {
                            (Value::Int64(a), Value::Int64(b)) => a <= b,
                            (Value::Float64(a), Value::Float64(b)) => a <= b,
                            (Value::String(a), Value::String(b)) => a <= b,
                            _ => true,
                        });
                        Some(Ok(Value::Bool(sorted)))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "list.all_elements_in" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::List(pool)) => {
                        let all_in = items.iter().all(|item| pool.contains(item));
                        Some(Ok(Value::Bool(all_in)))
                    }
                    _ => Some(Err(format!("{name} expects two list arguments"))),
                }
            }

            // -- Math operations (stdlib/math.jett) ---------------------------
            "math.abs" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Int64(n) => Some(Ok(Value::Int64(n.abs()))),
                    Value::Float64(n) => Some(Ok(Value::Float64(n.abs()))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }

            "math.min" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Int64(a), Value::Int64(b)) => Some(Ok(Value::Int64(*a.min(b)))),
                    (Value::Float64(a), Value::Float64(b)) => Some(Ok(Value::Float64(a.min(*b)))),
                    _ => Some(Err(format!(
                        "{name} expects two arguments of the same numeric type"
                    ))),
                }
            }

            "math.max" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Int64(a), Value::Int64(b)) => Some(Ok(Value::Int64(*a.max(b)))),
                    (Value::Float64(a), Value::Float64(b)) => Some(Ok(Value::Float64(a.max(*b)))),
                    _ => Some(Err(format!(
                        "{name} expects two arguments of the same numeric type"
                    ))),
                }
            }

            "math.sqrt" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.sqrt()))),
                    Value::Int64(n) => Some(Ok(Value::Float64((*n as f64).sqrt()))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }

            "math.pow" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Float64(base), Value::Float64(exp)) => {
                        Some(Ok(Value::Float64(base.powf(*exp))))
                    }
                    (Value::Int64(base), Value::Int64(exp)) => {
                        let exp_u = (*exp).max(0) as u32;
                        Some(Ok(Value::Int64(base.pow(exp_u))))
                    }
                    (Value::Float64(base), Value::Int64(exp)) => {
                        Some(Ok(Value::Float64(base.powi(*exp as i32))))
                    }
                    _ => Some(Err(format!("{name} expects numeric arguments"))),
                }
            }

            "math.floor" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.floor()))),
                    Value::Int64(n) => Some(Ok(Value::Int64(*n))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }

            "math.ceil" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.ceil()))),
                    Value::Int64(n) => Some(Ok(Value::Int64(*n))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }

            "math.round" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.round()))),
                    Value::Int64(n) => Some(Ok(Value::Int64(*n))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }

            "math.clamp" => {
                require_args!(name, 3, args);
                match (&args[0], &args[1], &args[2]) {
                    (Value::Int64(v), Value::Int64(lo), Value::Int64(hi)) => {
                        Some(Ok(Value::Int64((*v).clamp(*lo, *hi))))
                    }
                    (Value::Float64(v), Value::Float64(lo), Value::Float64(hi)) => {
                        Some(Ok(Value::Float64(v.clamp(*lo, *hi))))
                    }
                    _ => Some(Err(format!(
                        "{name} expects three arguments of the same numeric type"
                    ))),
                }
            }

            "math.log" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.ln()))),
                    Value::Int64(n) => Some(Ok(Value::Float64((*n as f64).ln()))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }

            "math.log2" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.log2()))),
                    Value::Int64(n) => Some(Ok(Value::Float64((*n as f64).log2()))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }

            "math.log10" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.log10()))),
                    Value::Int64(n) => Some(Ok(Value::Float64((*n as f64).log10()))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }

            "math.average" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) if !items.is_empty() => {
                        let sum: f64 = items
                            .iter()
                            .map(|v| match v {
                                Value::Int64(n) => *n as f64,
                                Value::Float64(n) => *n,
                                _ => 0.0,
                            })
                            .sum();
                        Some(Ok(Value::Float64(sum / items.len() as f64)))
                    }
                    Value::List(_) => Some(Err("math.average: list is empty".to_string())),
                    _ => Some(Err(format!("{name} expects a list of numbers"))),
                }
            }

            "math.median" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) if !items.is_empty() => {
                        let mut nums: Vec<f64> = items
                            .iter()
                            .map(|v| match v {
                                Value::Int64(n) => *n as f64,
                                Value::Float64(n) => *n,
                                _ => 0.0,
                            })
                            .collect();
                        nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let mid = nums.len() / 2;
                        let median = if nums.len().is_multiple_of(2) {
                            (nums[mid - 1] + nums[mid]) / 2.0
                        } else {
                            nums[mid]
                        };
                        Some(Ok(Value::Float64(median)))
                    }
                    Value::List(_) => Some(Err("math.median: list is empty".to_string())),
                    _ => Some(Err(format!("{name} expects a list of numbers"))),
                }
            }

            // -- Math constants and extras -----------------------------------------
            "math.pi" => {
                require_args!(name, 0, args);
                Some(Ok(Value::Float64(std::f64::consts::PI)))
            }
            "math.e" => {
                require_args!(name, 0, args);
                Some(Ok(Value::Float64(std::f64::consts::E)))
            }
            "math.sin" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.sin()))),
                    Value::Int64(n) => Some(Ok(Value::Float64((*n as f64).sin()))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }
            "math.cos" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.cos()))),
                    Value::Int64(n) => Some(Ok(Value::Float64((*n as f64).cos()))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }
            "math.tan" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(n) => Some(Ok(Value::Float64(n.tan()))),
                    Value::Int64(n) => Some(Ok(Value::Float64((*n as f64).tan()))),
                    _ => Some(Err(format!("{name} expects a numeric argument"))),
                }
            }
            "math.mod" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Int64(a), Value::Int64(b)) => {
                        if *b == 0 {
                            Some(Err("math.mod: division by zero".to_string()))
                        } else {
                            Some(Ok(Value::Int64(a % b)))
                        }
                    }
                    _ => Some(Err(format!("{name} expects two int64 arguments"))),
                }
            }
            "math.is_even" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Int64(n) => Some(Ok(Value::Bool(n % 2 == 0))),
                    _ => Some(Err(format!("{name} expects an int64 argument"))),
                }
            }
            "math.is_odd" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Int64(n) => Some(Ok(Value::Bool(n % 2 != 0))),
                    _ => Some(Err(format!("{name} expects an int64 argument"))),
                }
            }
            "math.sum" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        let mut total: i64 = 0;
                        for item in items {
                            match item {
                                Value::Int64(n) => total += n,
                                _ => {
                                    return Some(Err(
                                        "math.sum: list must contain int64 values".to_string()
                                    ));
                                }
                            }
                        }
                        Some(Ok(Value::Int64(total)))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "math.gcd" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Int64(a), Value::Int64(b)) => {
                        let (mut x, mut y) = (a.abs(), b.abs());
                        while y != 0 {
                            let t = y;
                            y = x % y;
                            x = t;
                        }
                        Some(Ok(Value::Int64(x)))
                    }
                    _ => Some(Err(format!("{name} expects two int64 arguments"))),
                }
            }
            "math.lcm" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Int64(a), Value::Int64(b)) => {
                        if *a == 0 && *b == 0 {
                            Some(Ok(Value::Int64(0)))
                        } else {
                            let (mut x, mut y) = (a.abs(), b.abs());
                            let product = x * y;
                            while y != 0 {
                                let t = y;
                                y = x % y;
                                x = t;
                            }
                            Some(Ok(Value::Int64(product / x)))
                        }
                    }
                    _ => Some(Err(format!("{name} expects two int64 arguments"))),
                }
            }
            "math.factorial" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Int64(n) => {
                        if *n < 0 {
                            Some(Err(
                                "math.factorial: argument must be non-negative".to_string()
                            ))
                        } else {
                            let mut result: i64 = 1;
                            for i in 2..=*n {
                                result = result.saturating_mul(i);
                            }
                            Some(Ok(Value::Int64(result)))
                        }
                    }
                    _ => Some(Err(format!("{name} expects an int64 argument"))),
                }
            }
            "math.sign" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Int64(n) => {
                        let s = if *n < 0 {
                            -1
                        } else if *n > 0 {
                            1
                        } else {
                            0
                        };
                        Some(Ok(Value::Int64(s)))
                    }
                    _ => Some(Err(format!("{name} expects an int64 argument"))),
                }
            }
            "math.to_radians" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(deg) => Some(Ok(Value::Float64(deg.to_radians()))),
                    _ => Some(Err(format!("{name} expects a float64 argument"))),
                }
            }
            "math.to_degrees" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Float64(rad) => Some(Ok(Value::Float64(rad.to_degrees()))),
                    _ => Some(Err(format!("{name} expects a float64 argument"))),
                }
            }

            // -- list.enumerate (returns list of [index, value] pairs) ----------
            "list.enumerate" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(items) => {
                        let enumerated: Vec<Value> = items
                            .iter()
                            .enumerate()
                            .map(|(i, v)| Value::List(vec![Value::Int64(i as i64), v.clone()]))
                            .collect();
                        Some(Ok(Value::List(enumerated)))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            // -- list.from_set (convert set to list) ----------------------------
            "list.from_set" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Set(items) => Some(Ok(Value::List(items.clone()))),
                    _ => Some(Err(format!("{name} expects a set argument"))),
                }
            }

            // -- list.repeat, list.range, list.last_index_of, list.insert_at, list.remove_at, list.swap
            "list.repeat" => {
                require_args!(name, 2, args);
                match &args[1] {
                    Value::Int64(count) => {
                        let n = (*count).max(0) as usize;
                        let items: Vec<Value> = std::iter::repeat_n(args[0].clone(), n).collect();
                        Some(Ok(Value::List(items)))
                    }
                    _ => Some(Err(format!("{name} expects a value and an int64 count"))),
                }
            }
            "list.range" => self.call_builtin("range", args),
            "list.last_index_of" => {
                require_args!(name, 2, args);
                match &args[0] {
                    Value::List(items) => {
                        let idx = items.iter().rposition(|v| v == &args[1]);
                        Some(Ok(match idx {
                            Some(i) => Value::OptionalSome(Box::new(Value::Int64(i as i64))),
                            None => Value::OptionalNone,
                        }))
                    }
                    _ => Some(Err(format!("{name} expects a list as first argument"))),
                }
            }
            "list.insert_at" => {
                require_args!(name, 3, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::Int64(index)) => {
                        let idx = *index as usize;
                        if idx <= items.len() {
                            let mut new_list = items.clone();
                            new_list.insert(idx, args[2].clone());
                            Some(Ok(Value::List(new_list)))
                        } else {
                            Some(Err(format!("{name}: index {index} out of bounds")))
                        }
                    }
                    _ => Some(Err(format!(
                        "{name} expects a list, an int64 index, and a value"
                    ))),
                }
            }
            "list.remove_at" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::List(items), Value::Int64(index)) => {
                        let idx = *index as usize;
                        if idx < items.len() {
                            let mut new_list = items.clone();
                            new_list.remove(idx);
                            Some(Ok(Value::List(new_list)))
                        } else {
                            Some(Err(format!("{name}: index {index} out of bounds")))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a list and an int64 index"))),
                }
            }
            "list.swap" => {
                require_args!(name, 3, args);
                match (&args[0], &args[1], &args[2]) {
                    (Value::List(items), Value::Int64(i), Value::Int64(j)) => {
                        let a = *i as usize;
                        let b = *j as usize;
                        if a >= items.len() || b >= items.len() {
                            Some(Err(format!("{name}: index out of bounds")))
                        } else {
                            let mut new_list = items.clone();
                            new_list.swap(a, b);
                            Some(Ok(Value::List(new_list)))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a list and two int64 indices"))),
                }
            }

            // -- Additional string operations (stdlib/string.jett) ------------
            "string.reverse" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let reversed = string_graphemes(s).into_iter().rev().collect();
                        Some(Ok(Value::String(reversed)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.after" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::String(marker)) => {
                        let result =
                            if let Some((_, end, _)) = string_find_grapheme_match(s, marker) {
                                s[end..].to_string()
                            } else {
                                String::new()
                            };
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }

            "string.before" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::String(marker)) => {
                        let result =
                            if let Some((start, _, _)) = string_find_grapheme_match(s, marker) {
                                s[..start].to_string()
                            } else {
                                s.clone()
                            };
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }

            "string.trim_start" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::String(s.trim_start().to_string()))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.trim_end" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::String(s.trim_end().to_string()))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // string.chars / string.words / string.lines — yield list[string]
            "string.chars" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let chars: Vec<Value> = string_graphemes(s)
                            .into_iter()
                            .map(|cluster| Value::String(cluster.to_string()))
                            .collect();
                        Some(Ok(Value::List(chars)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.words" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let words: Vec<Value> = s
                            .split_whitespace()
                            .map(|w| Value::String(w.to_string()))
                            .collect();
                        Some(Ok(Value::List(words)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.lines" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let lines: Vec<Value> =
                            s.lines().map(|l| Value::String(l.to_string())).collect();
                        Some(Ok(Value::List(lines)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // -- UUID operations (stdlib/uuid.jett) ---------------------------
            "uuid.new" => {
                require_args!(name, 0, args);
                // Generate a UUID v4 using rand
                let mut rng = rand::thread_rng();
                let mut b = [0u8; 16];
                for byte in b.iter_mut() {
                    *byte = rand::Rng::r#gen(&mut rng);
                }
                // Set version 4 bits
                b[6] = (b[6] & 0x0F) | 0x40;
                // Set variant bits
                b[8] = (b[8] & 0x3F) | 0x80;
                let uuid = format!(
                    "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    b[4],
                    b[5],
                    b[6],
                    b[7],
                    b[8],
                    b[9],
                    b[10],
                    b[11],
                    b[12],
                    b[13],
                    b[14],
                    b[15]
                );
                Some(Ok(Value::String(uuid)))
            }

            // -- Additional char-level string operations -----------------------
            "string.take_chars" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = (*n).max(0) as usize;
                        let result = string_graphemes(s).into_iter().take(n).collect();
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64"))),
                }
            }

            "string.take_last_chars" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = (*n).max(0) as usize;
                        let graphemes = string_graphemes(s);
                        let start = graphemes.len().saturating_sub(n);
                        let result = graphemes[start..].concat();
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64"))),
                }
            }

            "string.drop_chars" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = (*n).max(0) as usize;
                        let result = string_graphemes(s).into_iter().skip(n).collect();
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64"))),
                }
            }

            "string.char_at" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(i)) => {
                        let result = if *i < 0 {
                            Value::OptionalNone
                        } else {
                            match string_graphemes(s).get(*i as usize) {
                                Some(cluster) => Value::OptionalSome(Box::new(Value::String(
                                    (*cluster).to_string(),
                                ))),
                                None => Value::OptionalNone,
                            }
                        };
                        Some(Ok(result))
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64 index"))),
                }
            }

            "string.index_of" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(haystack), Value::String(needle)) => {
                        let result = match string_find_grapheme_match(haystack, needle) {
                            Some((_, _, index)) => {
                                Value::OptionalSome(Box::new(Value::Int64(index as i64)))
                            }
                            None => Value::OptionalNone,
                        };
                        Some(Ok(result))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }
            "string.count" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(haystack), Value::String(needle)) => {
                        let count = string_count_grapheme_matches(haystack, needle) as i64;
                        Some(Ok(Value::Int64(count)))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }
            "string.to_upper_first" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let graphemes = string_graphemes(s);
                        let result = match graphemes.split_first() {
                            Some((first, rest)) => {
                                let mut chars = first.chars();
                                let upper = chars
                                    .next()
                                    .map(|c| c.to_uppercase().collect::<String>())
                                    .unwrap_or_default();
                                format!("{upper}{}{}", chars.as_str(), rest.concat())
                            }
                            None => String::new(),
                        };
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }
            "string.to_lower_first" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let graphemes = string_graphemes(s);
                        let result = match graphemes.split_first() {
                            Some((first, rest)) => {
                                let mut chars = first.chars();
                                let lower = chars
                                    .next()
                                    .map(|c| c.to_lowercase().collect::<String>())
                                    .unwrap_or_default();
                                format!("{lower}{}{}", chars.as_str(), rest.concat())
                            }
                            None => String::new(),
                        };
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // -- String formatting operations ----------------------------------
            "string.center" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(width)) => {
                        let width = (*width).max(0) as usize;
                        let grapheme_len = string_grapheme_count(s);
                        if grapheme_len >= width {
                            Some(Ok(Value::String(s.clone())))
                        } else {
                            let total_pad = width - grapheme_len;
                            let left_pad = total_pad / 2;
                            let right_pad = total_pad - left_pad;
                            let left: String = std::iter::repeat_n(' ', left_pad).collect();
                            let right: String = std::iter::repeat_n(' ', right_pad).collect();
                            Some(Ok(Value::String(format!("{left}{s}{right}"))))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64 width"))),
                }
            }

            "string.ljust" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(width)) => {
                        let width = (*width).max(0) as usize;
                        let grapheme_len = string_grapheme_count(s);
                        if grapheme_len >= width {
                            Some(Ok(Value::String(s.clone())))
                        } else {
                            let padding: String =
                                std::iter::repeat_n(' ', width - grapheme_len).collect();
                            Some(Ok(Value::String(format!("{s}{padding}"))))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64 width"))),
                }
            }

            "string.rjust" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(width)) => {
                        let width = (*width).max(0) as usize;
                        let grapheme_len = string_grapheme_count(s);
                        if grapheme_len >= width {
                            Some(Ok(Value::String(s.clone())))
                        } else {
                            let padding: String =
                                std::iter::repeat_n(' ', width - grapheme_len).collect();
                            Some(Ok(Value::String(format!("{padding}{s}"))))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64 width"))),
                }
            }

            "string.zfill" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::Int64(width)) => {
                        let width = (*width).max(0) as usize;
                        let grapheme_len = string_grapheme_count(s);
                        if grapheme_len >= width {
                            Some(Ok(Value::String(s.clone())))
                        } else {
                            // Handle optional leading sign
                            let (sign, digits) = if s.starts_with('-') || s.starts_with('+') {
                                (&s[..1], &s[1..])
                            } else {
                                ("", s.as_str())
                            };
                            let zeros: String =
                                std::iter::repeat_n('0', width - grapheme_len).collect();
                            Some(Ok(Value::String(format!("{sign}{zeros}{digits}"))))
                        }
                    }
                    _ => Some(Err(format!("{name} expects a string and an int64 width"))),
                }
            }

            "string.remove_prefix" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::String(prefix)) => {
                        let result = if string_starts_with_grapheme(s, prefix) {
                            s[prefix.len()..].to_string()
                        } else {
                            s.clone()
                        };
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }

            "string.remove_suffix" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::String(s), Value::String(suffix)) => {
                        let result = if string_ends_with_grapheme(s, suffix) {
                            s[..s.len() - suffix.len()].to_string()
                        } else {
                            s.clone()
                        };
                        Some(Ok(Value::String(result)))
                    }
                    _ => Some(Err(format!("{name} expects two string arguments"))),
                }
            }

            "string.is_numeric" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::Bool(
                        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()),
                    ))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "string.is_alpha" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::Bool(
                        !s.is_empty() && s.chars().all(|c| c.is_alphabetic()),
                    ))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // -- Encoding operations (stdlib/encoding.jett) -------------------
            "encoding.base64_encode" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let encoded = base64_encode(s.as_bytes());
                        Some(Ok(Value::String(encoded)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "encoding.base64_decode" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => match base64_decode(s) {
                        Ok(bytes) => match String::from_utf8(bytes) {
                            Ok(decoded) => Some(Ok(Value::String(decoded))),
                            Err(_) => Some(Err(
                                "encoding.base64_decode: decoded bytes are not valid UTF-8"
                                    .to_string(),
                            )),
                        },
                        Err(e) => Some(Err(format!("encoding.base64_decode: {e}"))),
                    },
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "encoding.hex_encode" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let hex: String = s.bytes().map(|b| format!("{b:02x}")).collect();
                        Some(Ok(Value::String(hex)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "encoding.hex_decode" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        if s.len() % 2 != 0 {
                            return Some(Err(
                                "encoding.hex_decode: odd-length hex string".to_string()
                            ));
                        }
                        let bytes: Result<Vec<u8>, _> = (0..s.len() / 2)
                            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16))
                            .collect();
                        match bytes {
                            Ok(b) => match String::from_utf8(b) {
                                Ok(decoded) => Some(Ok(Value::String(decoded))),
                                Err(_) => {
                                    Some(Err("encoding.hex_decode: bytes are not valid UTF-8"
                                        .to_string()))
                                }
                            },
                            Err(_) => Some(Err(
                                "encoding.hex_decode: invalid hex characters".to_string()
                            )),
                        }
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "encoding.url_encode" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let encoded: String = s
                            .bytes()
                            .flat_map(|b| {
                                if b.is_ascii_alphanumeric()
                                    || b == b'-'
                                    || b == b'_'
                                    || b == b'.'
                                    || b == b'~'
                                {
                                    vec![b as char]
                                } else {
                                    format!("%{b:02X}").chars().collect()
                                }
                            })
                            .collect();
                        Some(Ok(Value::String(encoded)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "encoding.url_decode" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let bytes = s.as_bytes();
                        let mut result = Vec::new();
                        let mut i = 0;
                        while i < bytes.len() {
                            if bytes[i] == b'%' && i + 2 < bytes.len() {
                                if let Ok(b) = u8::from_str_radix(
                                    std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""),
                                    16,
                                ) {
                                    result.push(b);
                                    i += 3;
                                    continue;
                                }
                            } else if bytes[i] == b'+' {
                                result.push(b' ');
                                i += 1;
                                continue;
                            }
                            result.push(bytes[i]);
                            i += 1;
                        }
                        match String::from_utf8(result) {
                            Ok(decoded) => Some(Ok(Value::String(decoded))),
                            Err(_) => Some(Err(
                                "encoding.url_decode: result is not valid UTF-8".to_string()
                            )),
                        }
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // -- Crypto operations (stdlib/crypto.jett) -----------------------
            "crypto.sha256" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::String(sha256_hash(s.as_bytes())))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "crypto.md5" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::String(md5_hash(s.as_bytes())))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // -- Time operations (stdlib/time.jett) -------------------------------
            "time.now_ms" => {
                require_args!(name, 0, args);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64;
                Some(Ok(Value::Int64(now)))
            }
            "time.now_s" => {
                require_args!(name, 0, args);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                Some(Ok(Value::Int64(now)))
            }

            // -- OS operations (stdlib/os.jett) ---------------------------------
            "os.env" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(key) => {
                        let result = match std::env::var(key) {
                            Ok(val) => Value::OptionalSome(Box::new(Value::String(val))),
                            Err(_) => Value::OptionalNone,
                        };
                        Some(Ok(result))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }
            "os.args" => {
                require_args!(name, 0, args);
                let args_list: Vec<Value> = std::env::args().map(Value::String).collect();
                Some(Ok(Value::List(args_list)))
            }

            // -- CSV operations (stdlib/csv.jett) ---------------------------------
            "csv.parse" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let rows: Vec<Value> = parse_csv_records(s)
                            .into_iter()
                            .map(|row| Value::List(row.into_iter().map(Value::String).collect()))
                            .collect();
                        Some(Ok(Value::List(rows)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "csv.stringify" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::List(rows) => {
                        let mut lines = Vec::new();
                        for row in rows {
                            match row {
                                Value::List(cols) => {
                                    let fields: Vec<String> = cols
                                        .iter()
                                        .map(|v| match v {
                                            Value::String(s) => csv_quote_field(s),
                                            other => csv_quote_field(&format!("{other}")),
                                        })
                                        .collect();
                                    lines.push(fields.join(","));
                                }
                                _ => {
                                    return Some(Err(format!("{name} expects list[list[string]]")));
                                }
                            }
                        }
                        Some(Ok(Value::String(lines.join("\n"))))
                    }
                    _ => Some(Err(format!("{name} expects a list argument"))),
                }
            }

            "csv.parse_with_header" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => {
                        let mut records = parse_csv_records(s).into_iter();
                        let headers: Vec<String> = match records.next() {
                            Some(header_row) => header_row,
                            None => return Some(Ok(Value::List(Vec::new()))),
                        };
                        let rows: Vec<Value> = records
                            .map(|cols| {
                                let entries: Vec<(Value, Value)> = headers
                                    .iter()
                                    .zip(cols)
                                    .map(|(h, c)| (Value::String(h.clone()), Value::String(c)))
                                    .collect();
                                Value::Map(entries)
                            })
                            .collect();
                        Some(Ok(Value::List(rows)))
                    }
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            // -- Range generation ---------------------------------------------------
            "range" => {
                match args.len() {
                    // range(end) — 0 to end exclusive
                    1 => match &args[0] {
                        Value::Int64(end) => {
                            let items: Vec<Value> = (0..*end).map(Value::Int64).collect();
                            Some(Ok(Value::List(items)))
                        }
                        _ => Some(Err(format!("{name} expects int64 arguments"))),
                    },
                    // range(start, end) — start to end exclusive
                    2 => match (&args[0], &args[1]) {
                        (Value::Int64(start), Value::Int64(end)) => {
                            let items: Vec<Value> = (*start..*end).map(Value::Int64).collect();
                            Some(Ok(Value::List(items)))
                        }
                        _ => Some(Err(format!("{name} expects int64 arguments"))),
                    },
                    // range(start, end, step)
                    3 => match (&args[0], &args[1], &args[2]) {
                        (Value::Int64(start), Value::Int64(end), Value::Int64(step)) => {
                            if *step == 0 {
                                return Some(Err("range step cannot be zero".to_string()));
                            }
                            let mut items = Vec::new();
                            let mut i = *start;
                            if *step > 0 {
                                while i < *end {
                                    items.push(Value::Int64(i));
                                    i += step;
                                }
                            } else {
                                while i > *end {
                                    items.push(Value::Int64(i));
                                    i += step;
                                }
                            }
                            Some(Ok(Value::List(items)))
                        }
                        _ => Some(Err(format!("{name} expects int64 arguments"))),
                    },
                    _ => Some(Err(format!("{name} expects 1, 2, or 3 arguments"))),
                }
            }

            // -- Bytes operations (stdlib/bytes.jett) ---------------------------
            "bytes.new" => {
                require_args!(name, 0, args);
                Some(Ok(Value::Bytes(Vec::new())))
            }

            "bytes.length" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Bytes(b) => Some(Ok(Value::Int64(b.len() as i64))),
                    _ => Some(Err(format!("{name} expects a bytes argument"))),
                }
            }

            "bytes.slice" => {
                require_args!(name, 3, args);
                match (&args[0], &args[1], &args[2]) {
                    (Value::Bytes(b), Value::Int64(start), Value::Int64(end)) => {
                        let len = b.len() as i64;
                        let start = (*start).clamp(0, len) as usize;
                        let end = (*end).clamp(0, len) as usize;
                        let result = b[start.min(end)..end].to_vec();
                        Some(Ok(Value::Bytes(result)))
                    }
                    _ => Some(Err(format!(
                        "{name} expects a bytes value and two int64 indices"
                    ))),
                }
            }

            "bytes.concat" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Bytes(a), Value::Bytes(b)) => {
                        let mut result = a.clone();
                        result.extend(b.iter());
                        Some(Ok(Value::Bytes(result)))
                    }
                    _ => Some(Err(format!("{name} expects two bytes arguments"))),
                }
            }

            "bytes.from_string" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(s) => Some(Ok(Value::Bytes(s.as_bytes().to_vec()))),
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "bytes.to_string" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Bytes(b) => match String::from_utf8(b.clone()) {
                        Ok(s) => Some(Ok(Value::ResultOk(Box::new(Value::String(s))))),
                        Err(e) => Some(Ok(Value::ResultFail(Box::new(Value::String(format!(
                            "invalid UTF-8: {e}"
                        )))))),
                    },
                    _ => Some(Err(format!("{name} expects a bytes argument"))),
                }
            }

            "bytes.to_hex" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::Bytes(b) => {
                        let hex: String = b.iter().map(|byte| format!("{byte:02x}")).collect();
                        Some(Ok(Value::String(hex)))
                    }
                    _ => Some(Err(format!("{name} expects a bytes argument"))),
                }
            }

            "bytes.from_hex" => {
                require_args!(name, 1, args);
                match &args[0] {
                    Value::String(raw) => match Self::parse_hex_bytes(
                        raw,
                        "bytes.from_hex: expected even-length hex string",
                        "bytes.from_hex: expected hex string",
                    ) {
                        Ok(bytes) => Some(Ok(Value::ResultOk(Box::new(Value::Bytes(bytes))))),
                        Err(error) => Some(Ok(Value::ResultFail(Box::new(Value::String(error))))),
                    },
                    _ => Some(Err(format!("{name} expects a string argument"))),
                }
            }

            "bytes.get" => {
                require_args!(name, 2, args);
                match (&args[0], &args[1]) {
                    (Value::Bytes(b), Value::Int64(index)) => {
                        let idx = *index as usize;
                        if idx < b.len() {
                            Some(Ok(Value::OptionalSome(Box::new(Value::Int64(
                                b[idx] as i64,
                            )))))
                        } else {
                            Some(Ok(Value::OptionalNone))
                        }
                    }
                    _ => Some(Err(format!(
                        "{name} expects a bytes value and an int64 index"
                    ))),
                }
            }

            // Not a built-in
            _ => None,
        }
    }

    // -- Function calls -----------------------------------------------------

    /// Call a registered function by name with the given arguments.
    /// Built-in standard library functions are checked first; if the name
    /// does not match a built-in, user-defined functions are consulted.
    pub fn call_function(&mut self, name: &str, args: Vec<Value>) -> Result<Value, String> {
        self.call_function_with_type_args(name, &[], args)
    }

    /// Call a function while resolving unqualified names as if execution is
    /// currently inside `namespace`.
    pub fn call_function_in_namespace(
        &mut self,
        namespace: Option<&str>,
        name: &str,
        args: Vec<Value>,
    ) -> Result<Value, String> {
        let saved_namespace = self.current_namespace.clone();
        self.current_namespace = namespace.map(str::to_string);
        let runtime_name = self
            .registry_name(&self.functions, name)
            .unwrap_or_else(|| name.to_string());
        let result = self.call_function(&runtime_name, args);
        self.current_namespace = saved_namespace;
        result
    }

    /// Execute a block while resolving unqualified names as if execution is
    /// currently inside `namespace`.
    pub fn exec_block_in_namespace(
        &mut self,
        namespace: Option<&str>,
        block: &Block,
    ) -> Result<Option<Value>, String> {
        let saved_namespace = self.current_namespace.clone();
        self.current_namespace = namespace.map(str::to_string);
        let result = self.exec_block(block);
        self.current_namespace = saved_namespace;
        result
    }

    fn call_function_with_type_args(
        &mut self,
        name: &str,
        type_args: &[TypeExpr],
        args: Vec<Value>,
    ) -> Result<Value, String> {
        if self.is_trusted_stdlib_first_function(name) {
            return self.call_user_function_with_type_args(name, type_args, args);
        }

        // Check higher-order built-ins first (require &mut self).
        if let Some(result) = self.call_higher_order_builtin(name, args.clone()) {
            return result;
        }
        if let Some(result) = self.call_builtin_with_type_args(name, type_args, &args) {
            return result;
        }
        // Check built-in functions first.
        if let Some(result) = self.call_builtin(name, &args) {
            return result;
        }

        self.call_user_function_with_type_args(name, type_args, args)
    }

    fn is_trusted_stdlib_first_function(&self, name: &str) -> bool {
        self.has_trusted_stdlib_function(name)
            && (name.starts_with("json.json_") || is_json_raw_facade(name))
    }

    fn has_trusted_stdlib_function(&self, name: &str) -> bool {
        self.trusted_stdlib_functions.contains(name) && self.functions.contains_key(name)
    }

    fn call_user_function_with_type_args(
        &mut self,
        name: &str,
        type_args: &[TypeExpr],
        args: Vec<Value>,
    ) -> Result<Value, String> {
        // Check if the name refers to a variable holding a function value (closure).
        if let Some(fn_val) = self.get_variable(name).cloned()
            && matches!(fn_val, Value::Function { .. })
        {
            return self.call_fn_value(fn_val, args);
        }

        let resolved_name = self
            .resolve_interface_dispatch(name, &args)
            .unwrap_or_else(|| name.to_string());

        // Look up the function definition.
        let func = self
            .functions
            .get(&resolved_name)
            .ok_or_else(|| format!("undefined function '{name}'"))?
            .clone();

        if args.len() != func.params.len() {
            return Err(format!(
                "function '{}' expects {} argument(s), got {}",
                resolved_name,
                func.params.len(),
                args.len()
            ));
        }

        let type_scope = self.type_scope_for_function(&func, type_args)?;
        self.type_arg_scopes.push(type_scope);

        let saved_namespace = self.current_namespace.clone();
        self.current_namespace = Self::function_namespace(&resolved_name);

        self.push_scope();
        let call_result = (|| {
            for (param, arg) in func.params.iter().zip(args) {
                let param_ty = self.substitute_type_expr(&param.ty);
                let type_name = type_expr_name(&param_ty);
                let arg = self.normalize_value_for_type(&param_ty, arg)?;
                self.check_refinement(&type_name, &arg)?;
                self.set_variable_with_type(&param.name.name, arg, param_ty);
            }

            let result = self.exec_block_inner(&func.body)?;
            let mut value = match result {
                Some(Signal::Return(v)) => v,
                Some(Signal::Default(_)) => {
                    return Err("`default` can only be used inside a `handle` block".to_string());
                }
                _ => Value::Nothing,
            };

            if let Some(return_type) = &func.return_type {
                let return_type = self.substitute_type_expr(return_type);
                let type_name = type_expr_name(&return_type);
                value = self.normalize_value_for_type(&return_type, value)?;
                self.check_refinement(&type_name, &value)?;
            }

            Ok(value)
        })();
        self.pop_scope();
        self.type_arg_scopes.pop();
        self.current_namespace = saved_namespace;
        call_result
    }

    fn type_scope_for_function(
        &self,
        func: &FunctionDef,
        type_args: &[TypeExpr],
    ) -> Result<HashMap<String, TypeExpr>, String> {
        if func.type_params.is_empty() && type_args.is_empty() {
            return Ok(HashMap::new());
        }
        if func.type_params.len() != type_args.len() {
            return Err(format!(
                "function '{}' expects {} type argument(s), got {}",
                func.name.name,
                func.type_params.len(),
                type_args.len()
            ));
        }

        let raw: HashMap<String, TypeExpr> = func
            .type_params
            .iter()
            .zip(type_args.iter())
            .map(|(param, arg)| (param.name.clone(), self.substitute_type_expr(arg)))
            .collect();

        Ok(raw
            .iter()
            .map(|(name, ty)| (name.clone(), self.substitute_type_expr_with_map(ty, &raw)))
            .collect())
    }

    /// Try to call a higher-order built-in that requires `&mut self` (because
    /// it needs to invoke a user-supplied function value).  Returns `None` if
    /// the name is not a higher-order built-in.
    fn call_higher_order_builtin(
        &mut self,
        name: &str,
        args: Vec<Value>,
    ) -> Option<Result<Value, String>> {
        match name {
            "list.filter" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.filter expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.filter: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                let mut result = Vec::new();
                for item in items {
                    match self.call_fn_value(fn_val.clone(), vec![item.clone()]) {
                        Ok(Value::Bool(true)) => result.push(item),
                        Ok(Value::Bool(false)) => {}
                        Ok(other) => {
                            return Some(Err(format!(
                                "list.filter: predicate returned {other}, expected bool"
                            )));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::List(result)))
            }
            "list.map" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.map expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.map: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                let mut result = Vec::new();
                for item in items {
                    match self.call_fn_value(fn_val.clone(), vec![item]) {
                        Ok(v) => result.push(v),
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::List(result)))
            }
            "list.find" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.find expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.find: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                for item in items {
                    match self.call_fn_value(fn_val.clone(), vec![item.clone()]) {
                        Ok(Value::Bool(true)) => {
                            return Some(Ok(Value::OptionalSome(Box::new(item))));
                        }
                        Ok(Value::Bool(false)) => {}
                        Ok(other) => {
                            return Some(Err(format!(
                                "list.find: predicate returned {other}, expected bool"
                            )));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::OptionalNone))
            }
            "list.sort_by" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.sort_by expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.sort_by: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                // Compute keys for each item.
                let mut keyed: Vec<(i64, Value)> = Vec::new();
                for item in items {
                    match self.call_fn_value(fn_val.clone(), vec![item.clone()]) {
                        Ok(Value::Int64(k)) => keyed.push((k, item)),
                        Ok(other) => {
                            return Some(Err(format!(
                                "list.sort_by: key function returned {other}, expected int64"
                            )));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                keyed.sort_by_key(|(k, _)| *k);
                Some(Ok(Value::List(keyed.into_iter().map(|(_, v)| v).collect())))
            }
            "list.all" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.all expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.all: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                for item in items {
                    match self.call_fn_value(fn_val.clone(), vec![item]) {
                        Ok(Value::Bool(false)) => return Some(Ok(Value::Bool(false))),
                        Ok(Value::Bool(true)) => {}
                        Ok(other) => {
                            return Some(Err(format!(
                                "list.all: predicate returned {other}, expected bool"
                            )));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::Bool(true)))
            }
            "list.any" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.any expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.any: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                for item in items {
                    match self.call_fn_value(fn_val.clone(), vec![item]) {
                        Ok(Value::Bool(true)) => return Some(Ok(Value::Bool(true))),
                        Ok(Value::Bool(false)) => {}
                        Ok(other) => {
                            return Some(Err(format!(
                                "list.any: predicate returned {other}, expected bool"
                            )));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::Bool(false)))
            }
            "list.count" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.count expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.count: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                let mut count = 0i64;
                for item in items {
                    match self.call_fn_value(fn_val.clone(), vec![item]) {
                        Ok(Value::Bool(true)) => count += 1,
                        Ok(Value::Bool(false)) => {}
                        Ok(other) => {
                            return Some(Err(format!(
                                "list.count: predicate returned {other}, expected bool"
                            )));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::Int64(count)))
            }
            "list.sum" => {
                if args.len() != 1 {
                    return Some(Err(format!(
                        "list.sum expects 1 argument, got {}",
                        args.len()
                    )));
                }
                match &args[0] {
                    Value::List(items) => {
                        if items.is_empty() {
                            return Some(Ok(Value::Int64(0)));
                        }
                        // Detect int64 vs float64 from first element.
                        match &items[0] {
                            Value::Int64(_) => {
                                let mut total = 0i64;
                                for item in items {
                                    match item {
                                        Value::Int64(n) => total += n,
                                        _ => return Some(Err("list.sum: mixed types".into())),
                                    }
                                }
                                Some(Ok(Value::Int64(total)))
                            }
                            Value::Float64(_) => {
                                let mut total = 0.0f64;
                                for item in items {
                                    match item {
                                        Value::Float64(n) => total += n,
                                        _ => return Some(Err("list.sum: mixed types".into())),
                                    }
                                }
                                Some(Ok(Value::Float64(total)))
                            }
                            _ => Some(Err(
                                "list.sum: list elements must be int64 or float64".into()
                            )),
                        }
                    }
                    _ => Some(Err("list.sum: argument must be a list".into())),
                }
            }
            "list.group_by" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.group_by expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.group_by: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                let mut groups: Vec<(Value, Value)> = Vec::new();
                for item in items {
                    let key = match self.call_fn_value(fn_val.clone(), vec![item.clone()]) {
                        Ok(k) => k,
                        Err(e) => return Some(Err(e)),
                    };
                    if let Some((_, group)) = groups.iter_mut().find(|(k, _)| k == &key) {
                        if let Value::List(v) = group {
                            v.push(item);
                        }
                    } else {
                        groups.push((key, Value::List(vec![item])));
                    }
                }
                Some(Ok(Value::Map(groups)))
            }

            // list.reduce[T, U](list, initial, fn(acc, item) -> acc)
            "list.reduce" => {
                if args.len() != 3 {
                    return Some(Err(format!(
                        "list.reduce expects 3 arguments (list, initial, fn), got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.reduce: first argument must be a list".into())),
                };
                let mut acc = args[1].clone();
                let fn_val = args[2].clone();
                for item in items {
                    acc = match self.call_fn_value(fn_val.clone(), vec![acc, item]) {
                        Ok(v) => v,
                        Err(e) => return Some(Err(e)),
                    };
                }
                Some(Ok(acc))
            }

            "list.flat_map" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "list.flat_map expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let items = match &args[0] {
                    Value::List(v) => v.clone(),
                    _ => return Some(Err("list.flat_map: first argument must be a list".into())),
                };
                let fn_val = args[1].clone();
                let mut result = Vec::new();
                for item in items {
                    match self.call_fn_value(fn_val.clone(), vec![item]) {
                        Ok(Value::List(inner)) => result.extend(inner),
                        Ok(other) => {
                            return Some(Err(format!(
                                "list.flat_map: function returned {other}, expected a list"
                            )));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::List(result)))
            }

            // -- Map higher-order builtins ------------------------------------
            "map.filter" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "map.filter expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let entries = match &args[0] {
                    Value::Map(v) => v.clone(),
                    _ => return Some(Err("map.filter: first argument must be a map".into())),
                };
                let fn_val = args[1].clone();
                let mut result = Vec::new();
                for (k, v) in entries {
                    match self.call_fn_value(fn_val.clone(), vec![k.clone(), v.clone()]) {
                        Ok(Value::Bool(true)) => result.push((k, v)),
                        Ok(Value::Bool(false)) => {}
                        Ok(other) => {
                            return Some(Err(format!(
                                "map.filter: predicate returned {other}, expected bool"
                            )));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::Map(result)))
            }
            "map.map_values" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "map.map_values expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let entries = match &args[0] {
                    Value::Map(v) => v.clone(),
                    _ => return Some(Err("map.map_values: first argument must be a map".into())),
                };
                let fn_val = args[1].clone();
                let mut result = Vec::new();
                for (k, v) in entries {
                    match self.call_fn_value(fn_val.clone(), vec![v]) {
                        Ok(new_v) => result.push((k, new_v)),
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::Map(result)))
            }
            "map.for_each" => {
                if args.len() != 2 {
                    return Some(Err(format!(
                        "map.for_each expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                let entries = match &args[0] {
                    Value::Map(v) => v.clone(),
                    _ => return Some(Err("map.for_each: first argument must be a map".into())),
                };
                let fn_val = args[1].clone();
                for (k, v) in entries {
                    match self.call_fn_value(fn_val.clone(), vec![k, v]) {
                        Ok(_) => {}
                        Err(e) => return Some(Err(e)),
                    }
                }
                Some(Ok(Value::Nothing))
            }

            _ => None,
        }
    }

    /// Call a `Value::Function` (inline function) with the given arguments.
    fn call_fn_value(&mut self, fn_val: Value, args: Vec<Value>) -> Result<Value, String> {
        match fn_val {
            Value::Function {
                params,
                body,
                captures,
            } => {
                if args.len() != params.len() {
                    return Err(format!(
                        "inline function expects {} argument(s), got {}",
                        params.len(),
                        args.len()
                    ));
                }
                let mut normalized_args = Vec::with_capacity(args.len());
                for (param, arg) in params.iter().zip(args) {
                    let param_ty = self.substitute_type_expr(&param.ty);
                    normalized_args.push(self.normalize_value_for_type(&param_ty, arg)?);
                }

                // Push the captured environment as a scope, then the parameter scope on top.
                self.push_scope();
                for (name, value) in &captures {
                    self.set_variable(name, value.clone());
                }
                self.push_scope();
                for (param, arg) in params.iter().zip(normalized_args) {
                    let param_ty = self.substitute_type_expr(&param.ty);
                    self.set_variable_with_type(&param.name.name, arg, param_ty);
                }
                let result = self.exec_block_inner(&body)?;
                self.pop_scope(); // params
                self.pop_scope(); // captures
                Ok(match result {
                    Some(Signal::Return(v)) => v,
                    _ => Value::Nothing,
                })
            }
            other => Err(format!("expected function value, got {other}")),
        }
    }

    fn resolve_interface_dispatch(&self, name: &str, args: &[Value]) -> Option<String> {
        if self.functions.contains_key(name) {
            return Some(name.to_string());
        }

        let receiver_type = runtime_type_name(args.first()?)?;
        self.interface_methods
            .get(name)
            .and_then(|methods| methods.get(&receiver_type))
            .cloned()
    }

    fn construct_struct(
        &mut self,
        struct_name: &str,
        args: &[CallArg],
        arg_values: Vec<Value>,
    ) -> Result<Value, String> {
        let strukt = self
            .structs
            .get(struct_name)
            .ok_or_else(|| format!("undefined struct '{struct_name}'"))?
            .clone();
        let validates_refinements = strukt
            .fields
            .iter()
            .any(|field| self.type_name_has_refinement(&type_expr_name(&field.ty)));

        if args.len() > strukt.fields.len() {
            return Err(format!(
                "struct '{}' expects {} field argument(s), got {}",
                struct_name,
                strukt.fields.len(),
                args.len()
            ));
        }

        let mut fields: Vec<Option<Value>> = vec![None; strukt.fields.len()];
        for (arg, value) in args.iter().zip(arg_values) {
            let field_index = if let Some(name) = &arg.name {
                strukt
                    .fields
                    .iter()
                    .position(|field| field.name.name == name.name)
                    .ok_or_else(|| format!("struct '{struct_name}' has no field '{}'", name.name))?
            } else {
                let Some(index) = fields.iter().position(|value| value.is_none()) else {
                    return Err(format!(
                        "struct '{}' expects {} field argument(s), got {}",
                        struct_name,
                        strukt.fields.len(),
                        args.len()
                    ));
                };
                index
            };

            if fields[field_index].is_some() {
                return Err(format!(
                    "struct '{}' received field '{}' more than once",
                    struct_name, strukt.fields[field_index].name.name
                ));
            }

            let field_ty = self.substitute_type_expr(&strukt.fields[field_index].ty);
            let type_name = type_expr_name(&field_ty);
            let value = self.normalize_value_for_type(&field_ty, value)?;
            if let Err(message) = self.check_refinement(&type_name, &value) {
                if validates_refinements {
                    return Ok(Value::ResultFail(Box::new(Value::String(message))));
                }
                return Err(message);
            }
            fields[field_index] = Some(value);
        }

        for (index, field) in strukt.fields.iter().enumerate() {
            if fields[index].is_none() {
                return Err(format!(
                    "struct '{}' is missing required field '{}'",
                    struct_name, field.name.name
                ));
            }
        }

        let fields = strukt
            .fields
            .iter()
            .zip(fields)
            .map(|(field, value)| (field.name.name.clone(), value.unwrap()))
            .collect();

        let value = Value::Struct {
            type_name: struct_name.to_string(),
            fields,
        };

        if validates_refinements {
            Ok(Value::ResultOk(Box::new(value)))
        } else {
            Ok(value)
        }
    }

    fn construct_bitfield(
        &mut self,
        bitfield_name: &str,
        args: &[CallArg],
        arg_values: Vec<Value>,
    ) -> Result<Value, String> {
        let bitfield = self
            .bitfields
            .get(bitfield_name)
            .ok_or_else(|| format!("undefined bitfield '{bitfield_name}'"))?
            .clone();

        if args.len() > bitfield.fields.len() {
            return Err(format!(
                "bitfield '{}' expects {} field argument(s), got {}",
                bitfield_name,
                bitfield.fields.len(),
                args.len()
            ));
        }

        let mut fields: Vec<Option<Value>> = vec![None; bitfield.fields.len()];
        let mut requires_runtime_validation = false;
        for (arg, value) in args.iter().zip(arg_values) {
            let field_index = if let Some(name) = &arg.name {
                bitfield
                    .fields
                    .iter()
                    .position(|field| field.name.name == name.name)
                    .ok_or_else(|| {
                        format!("bitfield '{bitfield_name}' has no field '{}'", name.name)
                    })?
            } else {
                let Some(index) = fields.iter().position(|value| value.is_none()) else {
                    return Err(format!(
                        "bitfield '{}' expects {} field argument(s), got {}",
                        bitfield_name,
                        bitfield.fields.len(),
                        args.len()
                    ));
                };
                index
            };

            if fields[field_index].is_some() {
                return Err(format!(
                    "bitfield '{}' received field '{}' more than once",
                    bitfield_name, bitfield.fields[field_index].name.name
                ));
            }

            if let BitfieldFieldKind::Bits { width, as_type } = &bitfield.fields[field_index].kind
                && as_type.is_none()
            {
                let literal_input = matches!(arg.value, Expr::IntLiteral(_, _));
                let normalized = match Self::normalized_plain_bitfield_field_value(
                    bitfield_name,
                    &bitfield.fields[field_index].name.name,
                    *width,
                    &value,
                ) {
                    Ok(value) => value,
                    Err(message) if literal_input => return Err(message),
                    Err(message) => {
                        return Ok(Value::ResultFail(Box::new(Value::String(message))));
                    }
                };

                if !literal_input {
                    requires_runtime_validation = true;
                }

                fields[field_index] = Some(normalized);
                continue;
            }

            fields[field_index] = Some(value);
        }

        for (index, field) in bitfield.fields.iter().enumerate() {
            if fields[index].is_none() {
                return Err(format!(
                    "bitfield '{}' is missing required field '{}'",
                    bitfield_name, field.name.name
                ));
            }
        }

        let value = Value::Struct {
            type_name: bitfield_name.to_string(),
            fields: bitfield
                .fields
                .iter()
                .zip(fields)
                .map(|(field, value)| (field.name.name.clone(), value.unwrap()))
                .collect(),
        };

        if requires_runtime_validation {
            Ok(Value::ResultOk(Box::new(value)))
        } else {
            Ok(value)
        }
    }

    fn eval_value_field_access(&self, value: Value, field_name: &str) -> Result<ExprFlow, String> {
        match value {
            Value::Struct { type_name, fields } => fields
                .into_iter()
                .find(|(name, _)| name == field_name)
                .map(|(_, value)| ExprFlow::Value(value))
                .ok_or_else(|| format!("struct '{type_name}' has no field '{field_name}'")),
            Value::Machine {
                type_name,
                state,
                fields,
            } => {
                let machine = self
                    .machines
                    .get(&type_name)
                    .ok_or_else(|| format!("unknown machine '{type_name}'"))?;
                let state_def = machine
                    .states
                    .iter()
                    .find(|candidate| candidate.name.name == state)
                    .ok_or_else(|| format!("machine '{type_name}' has no state '{state}'"))?;
                let field_index = state_def
                    .fields
                    .iter()
                    .position(|candidate| candidate.name.name == field_name)
                    .ok_or_else(|| {
                        format!("machine '{type_name}' state '{state}' has no field '{field_name}'")
                    })?;
                fields.get(field_index).cloned().map(ExprFlow::Value).ok_or_else(|| {
                    format!(
                        "machine '{type_name}' state '{state}' field '{field_name}' has no runtime value"
                    )
                })
            }
            other => Err(format!("field access is not supported on {other}")),
        }
    }

    // -- Machine operations -------------------------------------------------

    /// Construct a machine value: `MachineName(state_name, fields...)`
    /// The first argument is a bare identifier naming the initial state (NOT
    /// evaluated).  Remaining arguments are evaluated as field values.
    fn construct_machine(&mut self, machine_name: &str, args: &[CallArg]) -> Result<Value, String> {
        if args.is_empty() {
            return Err(format!(
                "machine '{machine_name}' construction requires at least a state name"
            ));
        }

        // The first argument should be an identifier naming the state.
        let state_name = match &args[0].value {
            Expr::Ident(ident) => ident.name.clone(),
            _ => {
                return Err(format!(
                    "machine '{machine_name}' construction: first argument must be a state name"
                ));
            }
        };

        let machine_def = self.machines.get(machine_name).unwrap().clone();

        // Validate that the state exists.
        if !machine_def.states.iter().any(|s| s.name.name == state_name) {
            return Err(format!(
                "machine '{machine_name}' has no state '{state_name}'"
            ));
        }

        // Evaluate field arguments (args[1..]).
        let mut fields = Vec::new();
        for arg in &args[1..] {
            fields.push(self.eval_expr(&arg.value)?);
        }

        Ok(Value::Machine {
            type_name: machine_name.to_string(),
            state: state_name,
            fields,
        })
    }

    /// Perform a machine transition: `MachineName.transition(value, target_state, fields...)`
    /// - arg 0: the current machine value (evaluated)
    /// - arg 1: a bare identifier naming the target state (NOT evaluated)
    /// - args 2..: fields for the target state (evaluated)
    fn machine_transition(
        &mut self,
        machine_name: &str,
        args: &[CallArg],
    ) -> Result<Value, String> {
        if args.len() < 2 {
            return Err(format!(
                "{machine_name}.transition requires at least 2 arguments (value, target_state)"
            ));
        }

        // arg 0 is evaluated — it should be the current machine value
        let current_value = self.eval_expr(&args[0].value)?;
        let current_state = match &current_value {
            Value::Machine {
                type_name, state, ..
            } => {
                if type_name != machine_name {
                    return Err(format!(
                        "{machine_name}.transition: expected a {machine_name} value, got {type_name}"
                    ));
                }
                state.clone()
            }
            _ => {
                return Err(format!(
                    "{machine_name}.transition: first argument must be a machine value"
                ));
            }
        };

        // arg 1 should be a bare identifier naming the target state
        let target_state = match &args[1].value {
            Expr::Ident(ident) => ident.name.clone(),
            _ => {
                return Err(format!(
                    "{machine_name}.transition: second argument must be a state name"
                ));
            }
        };

        let machine_def = self.machines.get(machine_name).unwrap().clone();

        // Validate that the target state exists.
        if !machine_def
            .states
            .iter()
            .any(|s| s.name.name == target_state)
        {
            return Err(format!(
                "machine '{machine_name}' has no state '{target_state}'"
            ));
        }

        // Validate that the transition is allowed.
        let transition_allowed = machine_def
            .transitions
            .iter()
            .any(|t| t.from.name == current_state && t.to.name == target_state);
        if !transition_allowed {
            return Err(format!(
                "machine '{machine_name}': transition from '{current_state}' to '{target_state}' is not allowed"
            ));
        }

        // Evaluate field arguments (args[2..]).
        let mut fields = Vec::new();
        for arg in &args[2..] {
            fields.push(self.eval_expr(&arg.value)?);
        }

        Ok(Value::Machine {
            type_name: machine_name.to_string(),
            state: target_state,
            fields,
        })
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_truthy(val: &Value) -> Result<bool, String> {
    match val {
        Value::Bool(b) => Ok(*b),
        _ => Err(format!("expected boolean, got {val}")),
    }
}

fn string_grapheme_count(s: &str) -> usize {
    string_graphemes(s).len()
}

fn first_grapheme_or_space(s: &str) -> String {
    string_graphemes(s)
        .first()
        .map(|cluster| (*cluster).to_string())
        .unwrap_or_else(|| " ".to_string())
}

fn string_contains_grapheme(haystack: &str, needle: &str) -> bool {
    string_find_grapheme_match(haystack, needle).is_some()
}

fn string_starts_with_grapheme(haystack: &str, prefix: &str) -> bool {
    prefix.is_empty()
        || (haystack.starts_with(prefix) && string_is_grapheme_boundary(haystack, prefix.len()))
}

fn string_ends_with_grapheme(haystack: &str, suffix: &str) -> bool {
    if suffix.is_empty() {
        return true;
    }
    haystack.ends_with(suffix)
        && string_is_grapheme_boundary(haystack, haystack.len() - suffix.len())
}

fn string_find_grapheme_match(haystack: &str, needle: &str) -> Option<(usize, usize, usize)> {
    if needle.is_empty() {
        return Some((0, 0, 0));
    }

    let boundaries = string_grapheme_boundaries(haystack);
    for (grapheme_index, start) in boundaries
        .iter()
        .take(boundaries.len().saturating_sub(1))
        .enumerate()
    {
        let start = *start;
        if haystack[start..].starts_with(needle) {
            let end = start + needle.len();
            if boundaries.binary_search(&end).is_ok() {
                return Some((start, end, grapheme_index));
            }
        }
    }
    None
}

fn string_count_grapheme_matches(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }

    let boundaries = string_grapheme_boundaries(haystack);
    let mut count = 0;
    let mut boundary_index = 0;
    while boundary_index + 1 < boundaries.len() {
        let start = boundaries[boundary_index];
        if haystack[start..].starts_with(needle) {
            let end = start + needle.len();
            if let Ok(end_boundary_index) = boundaries.binary_search(&end) {
                count += 1;
                boundary_index = end_boundary_index;
                continue;
            }
        }
        boundary_index += 1;
    }
    count
}

fn string_is_grapheme_boundary(s: &str, byte_index: usize) -> bool {
    string_grapheme_boundaries(s)
        .binary_search(&byte_index)
        .is_ok()
}

fn string_grapheme_boundaries(s: &str) -> Vec<usize> {
    let mut boundaries = vec![0];
    let mut offset = 0;
    for cluster in string_graphemes(s) {
        offset += cluster.len();
        boundaries.push(offset);
    }
    boundaries
}

fn string_graphemes(s: &str) -> Vec<&str> {
    if s.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut cluster_start = 0;
    let mut previous = None;
    let mut regional_indicator_count = 0usize;

    for (index, ch) in s.char_indices() {
        if index == 0 {
            regional_indicator_count = if is_regional_indicator(ch) { 1 } else { 0 };
            previous = Some(ch);
            continue;
        }

        let joins_previous = is_grapheme_extend(ch)
            || ch == '\u{200D}'
            || previous == Some('\u{200D}')
            || previous == Some('\r') && ch == '\n'
            || is_regional_indicator(ch) && regional_indicator_count % 2 == 1;

        if !joins_previous {
            clusters.push(&s[cluster_start..index]);
            cluster_start = index;
            regional_indicator_count = 0;
        }

        if is_regional_indicator(ch) {
            regional_indicator_count += 1;
        } else if !is_grapheme_extend(ch) && ch != '\u{200D}' {
            regional_indicator_count = 0;
        }
        previous = Some(ch);
    }

    clusters.push(&s[cluster_start..]);
    clusters
}

fn is_grapheme_extend(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0300..=0x036F
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
            | 0xE0100..=0xE01EF
            | 0x1F3FB..=0x1F3FF
    )
}

fn is_regional_indicator(ch: char) -> bool {
    matches!(ch as u32, 0x1F1E6..=0x1F1FF)
}

fn uint64_arithmetic_operand(value: i64) -> Result<u64, String> {
    if value >= 0 {
        Ok(value as u64)
    } else {
        Err(format!(
            "uint64 arithmetic cannot use negative int64 operand {value}"
        ))
    }
}

fn eval_uint64_arithmetic(left: u64, op: BinOp, right: u64) -> Result<Value, String> {
    match op {
        BinOp::Add => left
            .checked_add(right)
            .map(Value::Uint64)
            .ok_or_else(|| format!("uint64 overflow: {left} + {right}")),
        BinOp::Sub => left
            .checked_sub(right)
            .map(Value::Uint64)
            .ok_or_else(|| format!("uint64 overflow: {left} - {right}")),
        BinOp::Mul => left
            .checked_mul(right)
            .map(Value::Uint64)
            .ok_or_else(|| format!("uint64 overflow: {left} * {right}")),
        BinOp::Div => left
            .checked_div(right)
            .map(Value::Uint64)
            .ok_or_else(|| "division by zero".to_string()),
        BinOp::Modulo => {
            if right == 0 {
                Err("modulo by zero".to_string())
            } else {
                Ok(Value::Uint64(left % right))
            }
        }
        _ => Err("unsupported uint64 arithmetic operator".to_string()),
    }
}

fn compare_uint64_values(left: u64, op: BinOp, right: u64) -> Result<Value, String> {
    let result = match op {
        BinOp::Eq => left == right,
        BinOp::NotEq => left != right,
        BinOp::Lt => left < right,
        BinOp::Gt => left > right,
        BinOp::LtEq => left <= right,
        BinOp::GtEq => left >= right,
        _ => return Err("unsupported uint64 comparison operator".to_string()),
    };
    Ok(Value::Bool(result))
}

fn compare_uint64_i64(left: u64, op: BinOp, right: i64) -> Result<Value, String> {
    if right < 0 {
        let result = match op {
            BinOp::Eq => false,
            BinOp::NotEq => true,
            BinOp::Lt => false,
            BinOp::Gt => true,
            BinOp::LtEq => false,
            BinOp::GtEq => true,
            _ => return Err("unsupported uint64 comparison operator".to_string()),
        };
        return Ok(Value::Bool(result));
    }
    compare_uint64_values(left, op, right as u64)
}

fn compare_i64_uint64(left: i64, op: BinOp, right: u64) -> Result<Value, String> {
    if left < 0 {
        let result = match op {
            BinOp::Eq => false,
            BinOp::NotEq => true,
            BinOp::Lt => true,
            BinOp::Gt => false,
            BinOp::LtEq => true,
            BinOp::GtEq => false,
            _ => return Err("unsupported uint64 comparison operator".to_string()),
        };
        return Ok(Value::Bool(result));
    }
    compare_uint64_values(left as u64, op, right)
}

fn eval_binary_op(left: &Value, op: BinOp, right: &Value) -> Result<Value, String> {
    match (left, op, right) {
        // -- Integer arithmetic -----------------------------------------------
        (Value::Int64(a), BinOp::Add, Value::Int64(b)) => a
            .checked_add(*b)
            .map(Value::Int64)
            .ok_or_else(|| format!("integer overflow: {a} + {b}")),
        (Value::Int64(a), BinOp::Sub, Value::Int64(b)) => a
            .checked_sub(*b)
            .map(Value::Int64)
            .ok_or_else(|| format!("integer overflow: {a} - {b}")),
        (Value::Int64(a), BinOp::Mul, Value::Int64(b)) => a
            .checked_mul(*b)
            .map(Value::Int64)
            .ok_or_else(|| format!("integer overflow: {a} * {b}")),
        (Value::Int64(a), BinOp::Div, Value::Int64(b)) => {
            if *b == 0 {
                Err("division by zero".to_string())
            } else {
                a.checked_div(*b)
                    .map(Value::Int64)
                    .ok_or_else(|| format!("integer overflow: {a} / {b}"))
            }
        }
        (Value::Int64(a), BinOp::Modulo, Value::Int64(b)) => {
            if *b == 0 {
                Err("modulo by zero".to_string())
            } else {
                a.checked_rem(*b)
                    .map(Value::Int64)
                    .ok_or_else(|| format!("integer overflow: {a} % {b}"))
            }
        }
        (
            Value::Uint64(a),
            op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Modulo),
            Value::Uint64(b),
        ) => eval_uint64_arithmetic(*a, op, *b),
        (
            Value::Uint64(a),
            op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Modulo),
            Value::Int64(b),
        ) => eval_uint64_arithmetic(*a, op, uint64_arithmetic_operand(*b)?),
        (
            Value::Int64(a),
            op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Modulo),
            Value::Uint64(b),
        ) => eval_uint64_arithmetic(uint64_arithmetic_operand(*a)?, op, *b),

        // -- Float arithmetic ------------------------------------------------
        (Value::Float64(a), BinOp::Add, Value::Float64(b)) => Ok(Value::Float64(a + b)),
        (Value::Float64(a), BinOp::Sub, Value::Float64(b)) => Ok(Value::Float64(a - b)),
        (Value::Float64(a), BinOp::Mul, Value::Float64(b)) => Ok(Value::Float64(a * b)),
        (Value::Float64(a), BinOp::Div, Value::Float64(b)) => Ok(Value::Float64(a / b)),
        (Value::Float64(a), BinOp::Modulo, Value::Float64(b)) => Ok(Value::Float64(a % b)),

        // -- String concatenation --------------------------------------------
        (Value::String(a), BinOp::Add, Value::String(b)) => Ok(Value::String(format!("{a}{b}"))),

        // -- Integer comparisons ---------------------------------------------
        (Value::Int64(a), BinOp::Eq, Value::Int64(b)) => Ok(Value::Bool(a == b)),
        (Value::Int64(a), BinOp::NotEq, Value::Int64(b)) => Ok(Value::Bool(a != b)),
        (Value::Int64(a), BinOp::Lt, Value::Int64(b)) => Ok(Value::Bool(a < b)),
        (Value::Int64(a), BinOp::Gt, Value::Int64(b)) => Ok(Value::Bool(a > b)),
        (Value::Int64(a), BinOp::LtEq, Value::Int64(b)) => Ok(Value::Bool(a <= b)),
        (Value::Int64(a), BinOp::GtEq, Value::Int64(b)) => Ok(Value::Bool(a >= b)),
        (
            Value::Uint64(a),
            op @ (BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq),
            Value::Uint64(b),
        ) => compare_uint64_values(*a, op, *b),
        (
            Value::Uint64(a),
            op @ (BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq),
            Value::Int64(b),
        ) => compare_uint64_i64(*a, op, *b),
        (
            Value::Int64(a),
            op @ (BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq),
            Value::Uint64(b),
        ) => compare_i64_uint64(*a, op, *b),

        // -- Float comparisons -----------------------------------------------
        (Value::Float64(a), BinOp::Eq, Value::Float64(b)) => Ok(Value::Bool(a == b)),
        (Value::Float64(a), BinOp::NotEq, Value::Float64(b)) => Ok(Value::Bool(a != b)),
        (Value::Float64(a), BinOp::Lt, Value::Float64(b)) => Ok(Value::Bool(a < b)),
        (Value::Float64(a), BinOp::Gt, Value::Float64(b)) => Ok(Value::Bool(a > b)),
        (Value::Float64(a), BinOp::LtEq, Value::Float64(b)) => Ok(Value::Bool(a <= b)),
        (Value::Float64(a), BinOp::GtEq, Value::Float64(b)) => Ok(Value::Bool(a >= b)),

        // -- String comparisons ----------------------------------------------
        (Value::String(a), BinOp::Eq, Value::String(b)) => Ok(Value::Bool(a == b)),
        (Value::String(a), BinOp::NotEq, Value::String(b)) => Ok(Value::Bool(a != b)),

        // -- Boolean comparisons ---------------------------------------------
        (Value::Bool(a), BinOp::Eq, Value::Bool(b)) => Ok(Value::Bool(a == b)),
        (Value::Bool(a), BinOp::NotEq, Value::Bool(b)) => Ok(Value::Bool(a != b)),

        // -- Boolean logic ---------------------------------------------------
        (Value::Bool(a), BinOp::And, Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
        (Value::Bool(a), BinOp::Or, Value::Bool(b)) => Ok(Value::Bool(*a || *b)),

        // -- Enum equality ---------------------------------------------------
        (Value::Enum { .. }, BinOp::Eq, Value::Enum { .. }) => Ok(Value::Bool(left == right)),
        (Value::Enum { .. }, BinOp::NotEq, Value::Enum { .. }) => Ok(Value::Bool(left != right)),

        // -- Nothing equality ------------------------------------------------
        (Value::Nothing, BinOp::Eq, Value::Nothing) => Ok(Value::Bool(true)),
        (Value::Nothing, BinOp::NotEq, Value::Nothing) => Ok(Value::Bool(false)),

        _ => Err(format!(
            "unsupported binary operation: {left} {op:?} {right}"
        )),
    }
}

/// Extract the simple name from a `TypeExpr` (e.g. `"int64"`, `"Port"`, `"list"`).
fn type_expr_name(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named(ident) => encoded_machine_base_name(&ident.name).to_string(),
        TypeExpr::Generic(ident, _, _) => ident.name.clone(),
        TypeExpr::View(inner, _) => type_expr_name(inner),
        TypeExpr::StateQualified(inner, _, _) => type_expr_name(inner),
        TypeExpr::Function(_, _, _) => "function".to_string(),
    }
}

fn type_expr_state_name(ty: &TypeExpr) -> Option<&str> {
    match ty {
        TypeExpr::Named(ident) => encoded_machine_state_name(&ident.name),
        TypeExpr::View(inner, _) => type_expr_state_name(inner),
        TypeExpr::StateQualified(_, state, _) => Some(&state.name),
        _ => None,
    }
}

fn encoded_machine_base_name(name: &str) -> &str {
    name.split_once(" at ")
        .map(|(base, _)| base)
        .unwrap_or(name)
}

fn encoded_machine_state_name(name: &str) -> Option<&str> {
    name.split_once(" at ").map(|(_, state)| state)
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

fn result_ok(value: Value) -> Value {
    Value::ResultOk(Box::new(value))
}

fn result_fail(message: String) -> Value {
    Value::ResultFail(Box::new(Value::String(message)))
}

fn type_expr_display(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named(ident) => ident.name.clone(),
        TypeExpr::Generic(ident, args, _) => {
            let args = args.iter().map(type_expr_display).collect::<Vec<_>>();
            format!("{}[{}]", ident.name, args.join(", "))
        }
        TypeExpr::View(inner, _) => format!("view {}", type_expr_display(inner)),
        TypeExpr::StateQualified(inner, state, _) => {
            format!("{} at {}", type_expr_display(inner), state.name)
        }
        TypeExpr::Function(params, return_type, _) => {
            let params = params.iter().map(type_expr_display).collect::<Vec<_>>();
            format!(
                "function({}) returns {}",
                params.join(", "),
                type_expr_display(return_type)
            )
        }
    }
}

fn runtime_type_name(value: &Value) -> Option<String> {
    match value {
        Value::Int64(_) => Some("int64".to_string()),
        Value::Uint64(_) => Some("uint64".to_string()),
        Value::Float64(_) => Some("float64".to_string()),
        Value::String(_) => Some("string".to_string()),
        Value::Bool(_) => Some("bool".to_string()),
        Value::List(_) => Some("list".to_string()),
        Value::Bytes(_) => Some("bytes".to_string()),
        Value::ResultOk(_) | Value::ResultFail(_) => Some("result".to_string()),
        Value::OptionalSome(_) | Value::OptionalNone => Some("optional".to_string()),
        Value::Nothing => Some("nothing".to_string()),
        Value::TypeConstruction { .. } => Some("TypeConstruction".to_string()),
        Value::Struct { type_name, .. }
        | Value::Enum { type_name, .. }
        | Value::Machine { type_name, .. } => Some(type_name.clone()),
        Value::Error(_) => None,
        Value::Actor(_) => Some("actor".to_string()),
        Value::Pending(_) => Some("pending".to_string()),
        Value::Map(_) => Some("map".to_string()),
        Value::Set(_) => Some("set".to_string()),
        Value::Function { .. } => Some("function".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use jett_common::{FileId, STDLIB_FILE_ID_START, Span};
    use jett_parser::ast::*;

    use super::*;

    const JSON_RAW_FACADE_NAMES: &[&str] = &[
        "json.parse_raw",
        "json.serialize_raw",
        "json.kind",
        "json.is_null",
        "json.is_bool",
        "json.is_number",
        "json.is_string",
        "json.is_array",
        "json.is_object",
        "json.field",
        "json.index",
        "json.array_length",
        "json.object_keys",
        "json.as_string",
        "json.as_int64",
        "json.as_uint64",
        "json.as_float64",
        "json.as_bool",
        "json.object_field",
        "json.array_index",
        "json.require_field",
        "json.require_index",
    ];

    /// Helper: create a dummy span for test AST nodes.
    fn sp() -> Span {
        Span::new(FileId::new(0), 0, 0)
    }

    fn stdlib_sp() -> Span {
        Span::new(FileId::new(STDLIB_FILE_ID_START), 0, 0)
    }

    /// Helper: create an Ident.
    fn ident(name: &str) -> Ident {
        Ident {
            name: name.to_string(),
            span: sp(),
        }
    }

    /// Helper: create an int literal expression.
    fn int(n: i64) -> Expr {
        Expr::IntLiteral(n.into(), sp())
    }

    /// Helper: create a float literal expression.
    fn float(n: f64) -> Expr {
        Expr::FloatLiteral(n, sp())
    }

    /// Helper: create a string literal expression.
    fn string(s: &str) -> Expr {
        Expr::StringLiteral(s.to_string(), sp())
    }

    /// Helper: create a bool literal expression.
    fn bool_expr(b: bool) -> Expr {
        Expr::BoolLiteral(b, sp())
    }

    /// Helper: create an identifier expression.
    fn var(name: &str) -> Expr {
        Expr::Ident(ident(name))
    }

    /// Helper: create a binary expression.
    fn binary(lhs: Expr, op: BinOp, rhs: Expr) -> Expr {
        Expr::Binary(Box::new(lhs), op, Box::new(rhs), sp())
    }

    /// Helper: create a simple type expression.
    fn type_named(name: &str) -> TypeExpr {
        TypeExpr::Named(ident(name))
    }

    fn type_alias(name: &str, base: &str, constraint: Option<Expr>) -> TypeAlias {
        TypeAlias {
            name: ident(name),
            base_type: type_named(base),
            constraint,
            exported: false,
            root_exported: false,
            span: sp(),
        }
    }

    /// Helper: create a variable declaration statement.
    fn var_decl(name: &str, value: Expr) -> Stmt {
        Stmt::VarDecl(VarDecl {
            mutable: true,
            ty: type_named("int64"),
            name: ident(name),
            value,
            span: sp(),
        })
    }

    fn typed_var_decl(type_name: &str, name: &str, value: Expr) -> Stmt {
        Stmt::VarDecl(VarDecl {
            mutable: true,
            ty: type_named(type_name),
            name: ident(name),
            value,
            span: sp(),
        })
    }

    /// Helper: create an assignment statement.
    fn assign(name: &str, value: Expr) -> Stmt {
        Stmt::Assign(AssignStmt {
            target: var(name),
            value,
            span: sp(),
        })
    }

    /// Helper: create an assert statement.
    fn assert_stmt(condition: Expr, message: Option<Expr>) -> Stmt {
        Stmt::Assert(AssertStmt {
            condition,
            message,
            span: sp(),
        })
    }

    /// Helper: create a block from statements.
    fn block(stmts: Vec<Stmt>) -> Block {
        Block { stmts, span: sp() }
    }

    /// Helper: create a return statement.
    fn return_stmt(value: Expr) -> Stmt {
        Stmt::Return(ReturnStmt {
            value: Some(value),
            span: sp(),
        })
    }

    fn respond_stmt(value: Expr) -> Stmt {
        Stmt::Respond(RespondStmt { value, span: sp() })
    }

    /// Helper: create a function call expression.
    fn call(name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call(
            Box::new(var(name)),
            args.into_iter()
                .map(|value| CallArg {
                    name: None,
                    value,
                    span: sp(),
                })
                .collect(),
            sp(),
        )
    }

    /// Helper: create a FunctionDef.
    fn func_def(name: &str, params: Vec<(&str, &str)>, body: Block) -> FunctionDef {
        FunctionDef {
            name: ident(name),
            type_params: vec![],
            params: params
                .into_iter()
                .map(|(pname, ptype)| Param {
                    view: false,
                    mutable: false,
                    name: ident(pname),
                    ty: type_named(ptype),
                    span: sp(),
                })
                .collect(),
            return_type: None,
            body,
            exported: false,
            span: sp(),
        }
    }

    fn generic_json_hook(name: &str, body: Block) -> FunctionDef {
        FunctionDef {
            name: ident(name),
            type_params: vec![ident("T")],
            params: vec![Param {
                view: false,
                mutable: false,
                name: ident("raw"),
                ty: type_named("string"),
                span: sp(),
            }],
            return_type: None,
            body,
            exported: false,
            span: sp(),
        }
    }

    fn json_tree_null() -> Value {
        Value::Enum {
            type_name: "json.JsonTree".to_string(),
            variant: "null".to_string(),
            fields: Vec::new(),
        }
    }

    fn string_type_info() -> ReflectionTypeInfo {
        ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        )
    }

    fn uint64_type_info() -> ReflectionTypeInfo {
        ReflectionTypeInfo::new(
            "uint64",
            "primitive",
            Some("uint64_type".to_string()),
            false,
            Vec::new(),
        )
    }

    fn string_field_info(index: usize, name: &str) -> ReflectionFieldInfo {
        ReflectionFieldInfo::new(
            index,
            name,
            "string",
            "primitive",
            name,
            false,
            string_type_info(),
        )
    }

    fn uint64_field_info(index: usize, name: &str) -> ReflectionFieldInfo {
        ReflectionFieldInfo::new(
            index,
            name,
            "uint64",
            "primitive",
            name,
            false,
            uint64_type_info(),
        )
    }

    #[test]
    fn type_info_uses_checked_reflection_metadata_when_available() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "int64",
            "primitive",
            Some("int64_type".to_string()),
            true,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let value = interp
            .call_builtin_with_type_args("type.info", &[type_named("int64")], &[])
            .expect("type.info should be a typed builtin")
            .expect("type.info should evaluate");

        let Value::Struct { fields, .. } = value else {
            panic!("expected TypeInfo struct");
        };
        let has_secret = fields
            .iter()
            .find_map(|(name, value)| {
                if name == "has_secret" {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("TypeInfo.has_secret field should exist");
        assert_eq!(has_secret, &Value::Bool(true));
    }

    #[test]
    fn bound_type_reflection_reports_missing_checked_type_info() {
        let mut metadata = ReflectionMetadata::new();
        metadata.bind_type_name("Box", jett_types::TypeInterner::STRING);

        let mut interp = Interpreter::new();
        interp.register_struct(&struct_def("Box", vec![("value", "string")], vec![]));
        interp.set_reflection_metadata(Arc::new(metadata));

        let ty = type_named("Box");
        let info_error = interp
            .call_builtin_with_type_args("type.info", std::slice::from_ref(&ty), &[])
            .expect("type.info should be a typed builtin")
            .expect_err("bound missing checked type info should be an error");
        let kind_error = interp
            .call_builtin_with_type_args("type.kind", std::slice::from_ref(&ty), &[])
            .expect("type.kind should be a typed builtin")
            .expect_err("bound missing checked type info should be an error");

        assert_eq!(
            info_error,
            "checked reflection metadata for type 'Box' is missing type info metadata"
        );
        assert_eq!(kind_error, info_error);
    }

    #[test]
    fn direct_type_reflection_uses_checked_metadata_when_available() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Token",
            "primitive",
            Some("string_type".to_string()),
            true,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let ty = type_named("Token");
        let name = interp
            .call_builtin_with_type_args("type.name", std::slice::from_ref(&ty), &[])
            .expect("type.name should be a typed builtin")
            .expect("type.name should evaluate");
        let kind = interp
            .call_builtin_with_type_args("type.kind", std::slice::from_ref(&ty), &[])
            .expect("type.kind should be a typed builtin")
            .expect("type.kind should evaluate");
        let kind_tag = interp
            .call_builtin_with_type_args("type.kind_tag", std::slice::from_ref(&ty), &[])
            .expect("type.kind_tag should be a typed builtin")
            .expect("type.kind_tag should evaluate");
        let primitive_tag = interp
            .call_builtin_with_type_args("type.primitive_tag", std::slice::from_ref(&ty), &[])
            .expect("type.primitive_tag should be a typed builtin")
            .expect("type.primitive_tag should evaluate");
        let has_secret = interp
            .call_builtin_with_type_args("type.has_secret", &[ty], &[])
            .expect("type.has_secret should be a typed builtin")
            .expect("type.has_secret should evaluate");

        assert_eq!(name, Value::String("Token".to_string()));
        assert_eq!(kind, Value::String("primitive".to_string()));
        assert_eq!(
            kind_tag,
            Value::Enum {
                type_name: "TypeKind".to_string(),
                variant: "primitive_type".to_string(),
                fields: Vec::new(),
            }
        );
        assert_eq!(
            primitive_tag,
            Value::OptionalSome(Box::new(Value::Enum {
                type_name: "TypePrimitive".to_string(),
                variant: "string_type".to_string(),
                fields: Vec::new(),
            }))
        );
        assert_eq!(has_secret, Value::Bool(true));
    }

    #[test]
    fn direct_has_secret_fallback_uses_state_qualified_machine_payload() {
        let mut interp = Interpreter::new();
        let token_ty = TypeExpr::Generic(ident("secret"), vec![type_named("string")], sp());
        interp.register_machine(&machine_def(
            "TokenLifecycle",
            vec![
                ("issued", vec![("token", token_ty)]),
                ("revoked", Vec::new()),
            ],
            vec![("issued", "revoked")],
        ));

        let machine_has_secret = interp
            .call_builtin_with_type_args("type.has_secret", &[type_named("TokenLifecycle")], &[])
            .expect("type.has_secret should be a typed builtin")
            .expect("machine secret reflection should evaluate");
        let issued_ty = TypeExpr::StateQualified(
            Box::new(type_named("TokenLifecycle")),
            ident("issued"),
            sp(),
        );
        let issued_has_secret = interp
            .call_builtin_with_type_args("type.has_secret", &[issued_ty], &[])
            .expect("type.has_secret should be a typed builtin")
            .expect("issued state secret reflection should evaluate");
        let revoked_ty = TypeExpr::StateQualified(
            Box::new(type_named("TokenLifecycle")),
            ident("revoked"),
            sp(),
        );
        let revoked_has_secret = interp
            .call_builtin_with_type_args("type.has_secret", &[revoked_ty], &[])
            .expect("type.has_secret should be a typed builtin")
            .expect("revoked state secret reflection should evaluate");

        assert_eq!(machine_has_secret, Value::Bool(true));
        assert_eq!(issued_has_secret, Value::Bool(true));
        assert_eq!(revoked_has_secret, Value::Bool(false));
    }

    #[test]
    fn direct_reflection_fallback_normalizes_current_namespace_types() {
        let mut interp = Interpreter::new();
        interp.register_struct_in_namespace(
            Some("models"),
            &struct_def("User", vec![("name", "string")], vec![]),
        );
        interp.register_enum_in_namespace(
            Some("models"),
            &enum_def_with_values("Status", vec![("active", 0)]),
        );
        interp.register_bitfield_in_namespace(
            Some("models"),
            &bitfield_def(
                "Header",
                vec![(
                    "version",
                    BitfieldFieldKind::Bits {
                        width: 4,
                        as_type: None,
                    },
                )],
                false,
            ),
        );
        interp.current_namespace = Some("models".to_string());

        let user_ty = type_named("User");
        assert_eq!(interp.type_expr_kind(&user_ty), "struct");
        let user_fields = interp.type_expr_fields(&user_ty);
        assert_eq!(user_fields.len(), 1);
        assert_eq!(user_fields[0].name, "name");

        let status_ty = type_named("Status");
        assert_eq!(interp.type_expr_kind(&status_ty), "enum");
        let variants = interp.type_expr_variants(&status_ty);
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].name, "active");

        let header_ty = type_named("Header");
        assert_eq!(interp.type_expr_kind(&header_ty), "bitfield");
        let bitfield = interp.type_expr_bitfield(&header_ty);
        assert_eq!(bitfield.fields.len(), 1);
        assert_eq!(bitfield.fields[0].name, "version");
    }

    #[test]
    fn registry_lookup_preserves_root_only_outside_namespace() {
        let mut interp = Interpreter::new();
        interp.register_function(&func_def(
            "helper",
            vec![],
            block(vec![return_stmt(string("root"))]),
        ));

        assert_eq!(
            interp.registry_name(&interp.functions, "helper").as_deref(),
            Some("helper")
        );

        interp.current_namespace = Some("alpha".to_string());
        assert_eq!(interp.registry_name(&interp.functions, "helper"), None);

        interp.register_function_in_namespace(
            Some("alpha"),
            &func_def(
                "helper",
                vec![],
                block(vec![return_stmt(string("namespace"))]),
            ),
        );
        assert_eq!(
            interp.registry_name(&interp.functions, "helper").as_deref(),
            Some("alpha.helper")
        );
    }

    #[test]
    fn direct_json_value_reflection_prefers_registered_alias_fallback() {
        let mut interp = Interpreter::new();
        interp.register_enum(&enum_def_with_values("json.JsonTree", vec![("null", 0)]));
        interp.register_type_alias(&type_alias("JsonValue", "json.JsonTree", None));

        let ty = type_named("JsonValue");
        let kind = interp
            .call_builtin_with_type_args("type.kind", std::slice::from_ref(&ty), &[])
            .expect("type.kind should be a typed builtin")
            .expect("type.kind should evaluate");
        let kind_tag = interp
            .call_builtin_with_type_args("type.kind_tag", std::slice::from_ref(&ty), &[])
            .expect("type.kind_tag should be a typed builtin")
            .expect("type.kind_tag should evaluate");
        let primitive_tag = interp
            .call_builtin_with_type_args("type.primitive_tag", std::slice::from_ref(&ty), &[])
            .expect("type.primitive_tag should be a typed builtin")
            .expect("type.primitive_tag should evaluate");
        let info = interp
            .call_builtin_with_type_args("type.info", &[ty], &[])
            .expect("type.info should be a typed builtin")
            .expect("type.info should evaluate");

        assert_eq!(kind, Value::String("alias".to_string()));
        assert_eq!(
            kind_tag,
            Value::Enum {
                type_name: "TypeKind".to_string(),
                variant: "alias_type".to_string(),
                fields: Vec::new(),
            }
        );
        assert_eq!(primitive_tag, Value::OptionalNone);

        let Value::Struct { fields, .. } = info else {
            panic!("expected TypeInfo struct");
        };
        let field = |field_name: &str| {
            fields
                .iter()
                .find_map(|(name, value)| (name == field_name).then_some(value))
                .expect("TypeInfo field should exist")
        };
        assert_eq!(field("kind"), &Value::String("alias".to_string()));
        assert_eq!(field("primitive_tag"), &Value::OptionalNone);

        let Value::List(args) = field("args") else {
            panic!("expected TypeInfo.args list");
        };
        assert_eq!(args.len(), 1);
        let Value::Struct {
            fields: arg_fields, ..
        } = &args[0]
        else {
            panic!("expected nested TypeInfo");
        };
        let arg_field = |field_name: &str| {
            arg_fields
                .iter()
                .find_map(|(name, value)| (name == field_name).then_some(value))
                .expect("nested TypeInfo field should exist")
        };
        assert_eq!(
            arg_field("type_name"),
            &Value::String("json.JsonTree".to_string())
        );
        assert_eq!(arg_field("kind"), &Value::String("enum".to_string()));
    }

    #[test]
    fn direct_json_value_reflection_requires_registered_alias() {
        let mut interp = Interpreter::new();
        let ty = type_named("JsonValue");

        let kind = interp
            .call_builtin_with_type_args("type.kind", std::slice::from_ref(&ty), &[])
            .expect("type.kind should be a typed builtin")
            .expect("type.kind should evaluate");
        let primitive_tag = interp
            .call_builtin_with_type_args("type.primitive_tag", &[ty], &[])
            .expect("type.primitive_tag should be a typed builtin")
            .expect("type.primitive_tag should evaluate");

        assert_eq!(kind, Value::String("named".to_string()));
        assert_eq!(primitive_tag, Value::OptionalNone);
    }

    #[test]
    fn json_serialize_secret_gate_uses_checked_reflection_metadata_when_available() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "SecretBox",
            "struct",
            None,
            true,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let value = Value::Struct {
            type_name: "SecretBox".to_string(),
            fields: Vec::new(),
        };
        let result = interp
            .call_builtin_with_type_args("json.serialize", &[type_named("SecretBox")], &[value])
            .expect("json.serialize should be a typed builtin");

        assert_eq!(
            result,
            Err("json.serialize cannot serialize secret-containing type 'SecretBox'".to_string())
        );
    }

    #[test]
    fn type_arg_uses_checked_reflection_metadata_when_available() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "list[int64]",
            "list",
            None,
            false,
            vec![ReflectionTypeInfo::new(
                "int64",
                "primitive",
                Some("int64_type".to_string()),
                true,
                Vec::new(),
            )],
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let list_int = TypeExpr::Generic(ident("list"), vec![type_named("int64")], sp());
        let value = interp
            .call_builtin_with_type_args("type.arg", &[list_int], &[Value::Int64(0)])
            .expect("type.arg should be a typed builtin")
            .expect("type.arg should evaluate");

        let Value::Struct { fields, .. } = value else {
            panic!("expected TypeInfo struct");
        };
        let has_secret = fields
            .iter()
            .find_map(|(name, value)| {
                if name == "has_secret" {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("TypeInfo.has_secret field should exist");
        assert_eq!(has_secret, &Value::Bool(true));
    }

    #[test]
    fn type_info_arg_loop_uses_checked_reflection_metadata_when_available() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "list[int64]",
            "list",
            None,
            false,
            vec![ReflectionTypeInfo::new(
                "string",
                "primitive",
                Some("string_type".to_string()),
                false,
                Vec::new(),
            )],
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let list_int = TypeExpr::Generic(ident("list"), vec![type_named("int64")], sp());
        let type_info_call = Expr::GenericCall(
            Box::new(field_access(var("type"), "info")),
            vec![list_int],
            Vec::new(),
            sp(),
        );
        let args_iterable = field_access(type_info_call, "args");
        let bindings = interp
            .reflected_type_info_arg_loop_bindings(&args_iterable)
            .expect("type.info args loop should be recognized")
            .expect("type.info args loop should produce bindings");

        assert_eq!(bindings.len(), 1);
        assert_eq!(type_expr_display(&bindings[0].ty), "string");
    }

    #[test]
    fn comptime_type_arg_binding_uses_checked_reflection_metadata_when_available() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "list[int64]",
            "list",
            None,
            false,
            vec![ReflectionTypeInfo::new(
                "string",
                "primitive",
                Some("string_type".to_string()),
                false,
                Vec::new(),
            )],
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));
        interp.set_variable("bound_name", Value::String("unset".to_string()));

        let stmt = Stmt::ComptimeTypeBind(ComptimeTypeBindStmt {
            name: ident("Item"),
            value: Expr::GenericCall(
                Box::new(field_access(var("type"), "arg")),
                vec![TypeExpr::Generic(
                    ident("list"),
                    vec![type_named("int64")],
                    sp(),
                )],
                vec![CallArg {
                    name: None,
                    value: Expr::IntLiteral(0, sp()),
                    span: sp(),
                }],
                sp(),
            ),
            body: block(vec![assign(
                "bound_name",
                Expr::GenericCall(
                    Box::new(field_access(var("type"), "name")),
                    vec![type_named("Item")],
                    Vec::new(),
                    sp(),
                ),
            )]),
            span: sp(),
        });

        interp.exec_stmt(&stmt).expect("binding should execute");
        assert_eq!(
            interp
                .get_variable("bound_name")
                .expect("bound_name should be set by the comptime type body"),
            &Value::String("string".to_string())
        );
    }

    #[test]
    fn type_fields_uses_checked_reflection_metadata_when_available() {
        let value_info = ReflectionTypeInfo::new(
            "secret[string]",
            "secret",
            None,
            true,
            vec![ReflectionTypeInfo::new(
                "string",
                "primitive",
                Some("string_type".to_string()),
                false,
                Vec::new(),
            )],
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_fields(
            "Box",
            vec![ReflectionFieldInfo::new(
                0,
                "value",
                "secret[string]",
                "secret",
                "jsonValue",
                true,
                value_info,
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let value = interp
            .call_builtin_with_type_args("type.fields", &[type_named("Box")], &[])
            .expect("type.fields should be a typed builtin")
            .expect("type.fields should evaluate");

        let Value::List(fields) = value else {
            panic!("expected list of TypeField values");
        };
        assert_eq!(fields.len(), 1);
        let Value::Struct { fields, .. } = &fields[0] else {
            panic!("expected TypeField struct");
        };
        let serialize_name = fields
            .iter()
            .find_map(|(name, value)| {
                if name == "serialize_name" {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("TypeField.serialize_name should exist");
        let has_secret = fields
            .iter()
            .find_map(|(name, value)| {
                if name == "has_secret" {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("TypeField.has_secret should exist");
        assert_eq!(serialize_name, &Value::String("jsonValue".to_string()));
        assert_eq!(has_secret, &Value::Bool(true));
    }

    #[test]
    fn type_fields_reports_missing_checked_struct_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Box",
            "struct",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let err = interp
            .call_builtin_with_type_args("type.fields", &[type_named("Box")], &[])
            .expect("type.fields should be a typed builtin")
            .expect_err("missing checked fields should be an error");

        assert!(err.contains("missing field metadata"));
    }

    #[test]
    fn type_field_value_reports_missing_checked_field_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Box",
            "struct",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let field =
            Interpreter::reflection_field_info_value("Box", None, &string_field_info(0, "value"));
        let value = Value::Struct {
            type_name: "Box".to_string(),
            fields: vec![("value".to_string(), Value::String("ok".to_string()))],
        };
        let err = interp
            .call_builtin_with_type_args(
                "type.field_value",
                &[type_named("Box"), type_named("string")],
                &[value, field],
            )
            .expect("type.field_value should be a typed builtin")
            .expect_err("missing checked field metadata should be an error");

        assert!(err.contains("missing field metadata"));
    }

    #[test]
    fn reflected_field_loop_uses_checked_reflection_metadata_when_available() {
        let value_info = ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_fields(
            "Box",
            vec![ReflectionFieldInfo::new(
                0,
                "value",
                "string",
                "primitive",
                "value",
                false,
                value_info,
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let iterable = Expr::GenericCall(
            Box::new(field_access(var("type"), "fields")),
            vec![type_named("Box")],
            Vec::new(),
            sp(),
        );
        let bindings = interp
            .reflected_field_loop_bindings(&iterable)
            .expect("type.fields loop should not error")
            .expect("type.fields loop should be recognized");

        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].index, 0);
        assert_eq!(bindings[0].name, "value");
        assert_eq!(type_expr_display(&bindings[0].ty), "string");
    }

    #[test]
    fn reflected_field_loop_reports_missing_checked_field_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Box",
            "struct",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let iterable = Expr::GenericCall(
            Box::new(field_access(var("type"), "fields")),
            vec![type_named("Box")],
            Vec::new(),
            sp(),
        );
        let err = interp
            .reflected_field_loop_bindings(&iterable)
            .expect_err("missing checked fields should be an error");

        assert!(err.contains("missing field metadata"));
    }

    #[test]
    fn type_field_value_uses_checked_reflection_metadata_when_available() {
        let value_info = ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_fields(
            "Box",
            vec![ReflectionFieldInfo::new(
                0,
                "value",
                "string",
                "primitive",
                "value",
                false,
                value_info,
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let field = interp
            .call_builtin_with_type_args("type.fields", &[type_named("Box")], &[])
            .expect("type.fields should be a typed builtin")
            .expect("type.fields should evaluate");
        let Value::List(fields) = field else {
            panic!("expected list of TypeField values");
        };
        let value = Value::Struct {
            type_name: "Box".to_string(),
            fields: vec![("value".to_string(), Value::String("ok".to_string()))],
        };
        let reflected = interp
            .call_builtin_with_type_args(
                "type.field_value",
                &[type_named("Box"), type_named("string")],
                &[value, fields[0].clone()],
            )
            .expect("type.field_value should be a typed builtin")
            .expect("type.field_value should evaluate");

        assert_eq!(reflected, Value::String("ok".to_string()));
    }

    #[test]
    fn type_bitfield_uses_checked_reflection_metadata_when_available() {
        let int_info = ReflectionTypeInfo::new(
            "int64",
            "primitive",
            Some("int64_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_bitfield(
            "Header",
            ReflectionBitfieldInfo::new(
                true,
                vec![ReflectionBitfieldFieldInfo::new(
                    0, "version", "bits", 3, int_info, None,
                )],
            ),
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let layout = interp
            .call_builtin_with_type_args("type.bitfield_layout", &[type_named("Header")], &[])
            .expect("type.bitfield_layout should be a typed builtin")
            .expect("type.bitfield_layout should evaluate");
        let Value::Struct { fields, .. } = layout else {
            panic!("expected TypeBitfield struct");
        };
        let network_order = fields
            .iter()
            .find_map(|(name, value)| {
                if name == "network_order" {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("TypeBitfield.network_order should exist");
        assert_eq!(network_order, &Value::Bool(true));

        let fields = interp
            .call_builtin_with_type_args("type.bitfield_fields", &[type_named("Header")], &[])
            .expect("type.bitfield_fields should be a typed builtin")
            .expect("type.bitfield_fields should evaluate");
        let Value::List(fields) = fields else {
            panic!("expected list of TypeBitfieldField values");
        };
        let Value::Struct { fields, .. } = &fields[0] else {
            panic!("expected TypeBitfieldField struct");
        };
        let width = fields
            .iter()
            .find_map(|(name, value)| if name == "width" { Some(value) } else { None })
            .expect("TypeBitfieldField.width should exist");
        assert_eq!(width, &Value::Int64(3));
    }

    #[test]
    fn type_bitfield_reports_missing_checked_bitfield_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Header",
            "bitfield",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let err = interp
            .call_builtin_with_type_args("type.bitfield_layout", &[type_named("Header")], &[])
            .expect("type.bitfield_layout should be a typed builtin")
            .expect_err("missing checked bitfield layout should be an error");

        assert!(err.contains("missing bitfield metadata"));
    }

    #[test]
    fn type_machine_uses_checked_reflection_metadata_when_available() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_machine(
            "Session",
            ReflectionMachineInfo::new(
                vec![
                    ReflectionMachineStateInfo::new(0, "guest", false, Vec::new()),
                    ReflectionMachineStateInfo::new(
                        1,
                        "logged_in",
                        false,
                        vec![string_field_info(0, "user_id")],
                    ),
                ],
                vec![ReflectionMachineTransitionInfo::new(
                    0,
                    0,
                    "guest",
                    1,
                    "logged_in",
                )],
            ),
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let layout = interp
            .call_builtin_with_type_args("type.machine_layout", &[type_named("Session")], &[])
            .expect("type.machine_layout should be a typed builtin")
            .expect("type.machine_layout should evaluate");
        let Value::Struct { fields, .. } = layout else {
            panic!("expected TypeMachine struct");
        };
        let edges = fields
            .iter()
            .find_map(|(name, value)| if name == "edges" { Some(value) } else { None })
            .expect("TypeMachine.edges should exist");
        let Value::List(edges) = edges else {
            panic!("expected TypeMachine.edges to be a list");
        };
        assert_eq!(edges.len(), 1);

        let states = interp
            .call_builtin_with_type_args("type.machine_states", &[type_named("Session")], &[])
            .expect("type.machine_states should be a typed builtin")
            .expect("type.machine_states should evaluate");
        let Value::List(states) = states else {
            panic!("expected list of TypeMachineState values");
        };
        let Value::Struct { fields, .. } = &states[1] else {
            panic!("expected TypeMachineState struct");
        };
        let state_fields = fields
            .iter()
            .find_map(|(name, value)| if name == "fields" { Some(value) } else { None })
            .expect("TypeMachineState.fields should exist");
        let Value::List(state_fields) = state_fields else {
            panic!("expected TypeMachineState.fields to be a list");
        };
        assert_eq!(state_fields.len(), 1);

        let edges = interp
            .call_builtin_with_type_args("type.machine_transitions", &[type_named("Session")], &[])
            .expect("type.machine_transitions should be a typed builtin")
            .expect("type.machine_transitions should evaluate");
        let Value::List(edges) = edges else {
            panic!("expected list of TypeMachineTransition values");
        };
        let Value::Struct { fields, .. } = &edges[0] else {
            panic!("expected TypeMachineTransition struct");
        };
        let target = fields
            .iter()
            .find_map(|(name, value)| if name == "target" { Some(value) } else { None })
            .expect("TypeMachineTransition.target should exist");
        assert_eq!(target, &Value::String("logged_in".to_string()));

        let state_ty =
            TypeExpr::StateQualified(Box::new(type_named("Session")), ident("logged_in"), sp());
        let layout = interp
            .call_builtin_with_type_args("type.machine_layout", &[state_ty], &[])
            .expect("type.machine_layout should be a typed builtin")
            .expect("state-qualified machine layout should evaluate");
        let Value::Struct { fields, .. } = layout else {
            panic!("expected state-qualified TypeMachine struct");
        };
        let states = fields
            .iter()
            .find_map(|(name, value)| if name == "states" { Some(value) } else { None })
            .expect("TypeMachine.states should exist");
        let Value::List(states) = states else {
            panic!("expected TypeMachine.states to be a list");
        };
        assert_eq!(states.len(), 2);
    }

    #[test]
    fn reflected_machine_field_binding_requires_state_owner_member() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_machine(
            "Session",
            ReflectionMachineInfo::new(
                vec![ReflectionMachineStateInfo::new(
                    0,
                    "logged_in",
                    false,
                    vec![string_field_info(0, "user_id")],
                )],
                Vec::new(),
            ),
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let field = Interpreter::reflection_field_info_value(
            "Session",
            None,
            &string_field_info(0, "user_id"),
        );
        let err = interp
            .reflected_machine_field_binding_for_value(&type_named("Session"), &field)
            .expect_err("machine payload field metadata should include a state owner member");

        assert!(err.contains("expected state payload field metadata"));
    }

    #[test]
    fn type_machine_reports_missing_checked_machine_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Session",
            "machine",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let err = interp
            .call_builtin_with_type_args("type.machine_layout", &[type_named("Session")], &[])
            .expect("type.machine_layout should be a typed builtin")
            .expect_err("missing checked machine layout should be an error");

        assert!(err.contains("missing machine metadata"));
    }

    #[test]
    fn type_variants_uses_checked_reflection_metadata_when_available() {
        let field_info = ReflectionTypeInfo::new(
            "secret[string]",
            "secret",
            None,
            true,
            vec![ReflectionTypeInfo::new(
                "string",
                "primitive",
                Some("string_type".to_string()),
                false,
                Vec::new(),
            )],
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_variants(
            "Choice",
            vec![ReflectionVariantInfo::new(
                1,
                "token",
                7,
                true,
                vec![ReflectionFieldInfo::new(
                    0,
                    "value",
                    "secret[string]",
                    "secret",
                    "jsonValue",
                    true,
                    field_info,
                )],
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let value = interp
            .call_builtin_with_type_args("type.variants", &[type_named("Choice")], &[])
            .expect("type.variants should be a typed builtin")
            .expect("type.variants should evaluate");

        let Value::List(variants) = value else {
            panic!("expected list of TypeVariant values");
        };
        let Value::Struct { fields, .. } = &variants[0] else {
            panic!("expected TypeVariant struct");
        };
        let discriminant = fields
            .iter()
            .find_map(|(name, value)| {
                if name == "discriminant" {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("TypeVariant.discriminant should exist");
        let has_secret = fields
            .iter()
            .find_map(|(name, value)| {
                if name == "has_secret" {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("TypeVariant.has_secret should exist");
        assert_eq!(discriminant, &Value::Int64(7));
        assert_eq!(has_secret, &Value::Bool(true));
    }

    #[test]
    fn type_variants_reports_missing_checked_enum_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Shape",
            "enum",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let err = interp
            .call_builtin_with_type_args("type.variants", &[type_named("Shape")], &[])
            .expect("type.variants should be a typed builtin")
            .expect_err("missing checked enum variants should be an error");

        assert!(err.contains("missing variant metadata"));
    }

    #[test]
    fn type_variant_value_reports_missing_checked_variant_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Choice",
            "enum",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let value = Value::Enum {
            type_name: "Choice".to_string(),
            variant: "token".to_string(),
            fields: Vec::new(),
        };
        let err = interp
            .call_builtin_with_type_args("type.variant_value", &[type_named("Choice")], &[value])
            .expect("type.variant_value should be a typed builtin")
            .expect_err("missing checked variant metadata should be an error");

        assert!(err.contains("missing variant metadata"));
    }

    #[test]
    fn reflected_variant_loops_use_checked_reflection_metadata_when_available() {
        let field_info = ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_variants(
            "Choice",
            vec![ReflectionVariantInfo::new(
                0,
                "token",
                7,
                false,
                vec![ReflectionFieldInfo::new(
                    0,
                    "value",
                    "string",
                    "primitive",
                    "value",
                    false,
                    field_info,
                )],
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let variant_iterable = Expr::GenericCall(
            Box::new(field_access(var("type"), "variants")),
            vec![type_named("Choice")],
            Vec::new(),
            sp(),
        );
        let variant_bindings = interp
            .reflected_variant_loop_bindings(&variant_iterable)
            .expect("type.variants loop should not error")
            .expect("type.variants loop should be recognized");
        assert_eq!(variant_bindings.len(), 1);
        assert_eq!(variant_bindings[0].name, "token");
        assert_eq!(variant_bindings[0].discriminant, 7);

        let variants = interp
            .call_builtin_with_type_args("type.variants", &[type_named("Choice")], &[])
            .expect("type.variants should be a typed builtin")
            .expect("type.variants should evaluate");
        let Value::List(variants) = variants else {
            panic!("expected list of TypeVariant values");
        };
        let Value::Struct {
            fields: variant_fields,
            ..
        } = &variants[0]
        else {
            panic!("expected TypeVariant");
        };
        let payload_fields = variant_fields
            .iter()
            .find_map(
                |(name, value)| {
                    if name == "fields" { Some(value) } else { None }
                },
            )
            .expect("TypeVariant.fields should exist");
        let Value::List(payload_fields) = payload_fields else {
            panic!("expected TypeVariant.fields list");
        };

        let field_binding = interp
            .reflected_variant_field_binding_for_value(
                &type_named("Choice"),
                Some("token"),
                &payload_fields[0],
            )
            .expect("payload field binding should be checked");
        assert_eq!(field_binding.index, 0);
        assert_eq!(field_binding.name, "value");
        assert_eq!(type_expr_display(&field_binding.ty), "string");
    }

    #[test]
    fn reflected_variant_field_binding_requires_variant_owner_member() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_variants(
            "Choice",
            vec![ReflectionVariantInfo::new(
                0,
                "token",
                7,
                false,
                vec![string_field_info(0, "value")],
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let field = Interpreter::reflection_field_info_value(
            "Choice",
            None,
            &string_field_info(0, "value"),
        );
        let err = interp
            .reflected_variant_field_binding_for_value(&type_named("Choice"), None, &field)
            .expect_err("enum payload field metadata should include a variant owner member");

        assert!(err.contains("expected payload field metadata"));
    }

    #[test]
    fn reflected_variant_loop_reports_missing_checked_variant_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Choice",
            "enum",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let variant_iterable = Expr::GenericCall(
            Box::new(field_access(var("type"), "variants")),
            vec![type_named("Choice")],
            Vec::new(),
            sp(),
        );
        let err = interp
            .reflected_variant_loop_bindings(&variant_iterable)
            .expect_err("missing checked variants should be an error");

        assert!(err.contains("missing variant metadata"));
    }

    #[test]
    fn reflected_variant_field_binding_reports_missing_checked_variant_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Choice",
            "enum",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let field = Interpreter::reflection_field_info_value(
            "Choice",
            Some("token"),
            &string_field_info(0, "value"),
        );
        let err = interp
            .reflected_variant_field_binding_for_value(&type_named("Choice"), Some("token"), &field)
            .expect_err("missing checked variants should be an error");

        assert!(err.contains("missing variant metadata"));
    }

    #[test]
    fn type_variant_value_uses_checked_reflection_metadata_when_available() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_variants(
            "Choice",
            vec![ReflectionVariantInfo::new(0, "token", 7, false, Vec::new())],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let value = Value::Enum {
            type_name: "Choice".to_string(),
            variant: "token".to_string(),
            fields: Vec::new(),
        };
        let variant = interp
            .call_builtin_with_type_args("type.variant_value", &[type_named("Choice")], &[value])
            .expect("type.variant_value should be a typed builtin")
            .expect("type.variant_value should evaluate");

        let Value::Struct { fields, .. } = variant else {
            panic!("expected TypeVariant struct");
        };
        let discriminant = fields
            .iter()
            .find_map(|(name, value)| {
                if name == "discriminant" {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("TypeVariant.discriminant should exist");
        assert_eq!(discriminant, &Value::Int64(7));
    }

    #[test]
    fn type_variant_field_value_uses_checked_reflection_metadata_when_available() {
        let field_type = ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_variants(
            "Choice",
            vec![ReflectionVariantInfo::new(
                0,
                "token",
                7,
                false,
                vec![ReflectionFieldInfo::new(
                    0,
                    "value",
                    "string",
                    "primitive",
                    "value",
                    false,
                    field_type,
                )],
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let value = Value::Enum {
            type_name: "Choice".to_string(),
            variant: "token".to_string(),
            fields: vec![Value::String("ok".to_string())],
        };
        let variant = interp
            .call_builtin_with_type_args(
                "type.variant_value",
                &[type_named("Choice")],
                &[value.clone()],
            )
            .expect("type.variant_value should be a typed builtin")
            .expect("type.variant_value should evaluate");
        let Value::Struct { fields, .. } = variant else {
            panic!("expected TypeVariant struct");
        };
        let field = fields
            .iter()
            .find_map(
                |(name, value)| {
                    if name == "fields" { Some(value) } else { None }
                },
            )
            .expect("TypeVariant.fields should exist");
        let Value::List(fields) = field else {
            panic!("expected TypeVariant.fields list");
        };

        let reflected = interp
            .call_builtin_with_type_args(
                "type.variant_field_value",
                &[type_named("Choice"), type_named("string")],
                &[value, fields[0].clone()],
            )
            .expect("type.variant_field_value should be a typed builtin")
            .expect("type.variant_field_value should evaluate");

        assert_eq!(reflected, Value::String("ok".to_string()));
    }

    #[test]
    fn type_construct_put_and_finish_use_checked_reflection_metadata_when_available() {
        let field_type = ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Box",
            "struct",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_type_fields(
            "Box",
            vec![ReflectionFieldInfo::new(
                0,
                "value",
                "string",
                "primitive",
                "value",
                false,
                field_type,
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Box")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let type_fields = interp
            .call_builtin_with_type_args("type.fields", &[type_named("Box")], &[])
            .expect("type.fields should be a typed builtin")
            .expect("type.fields should evaluate");
        let Value::List(type_fields) = type_fields else {
            panic!("expected list of TypeField values");
        };

        let updated = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Box"), type_named("string")],
                &[
                    builder,
                    type_fields[0].clone(),
                    Value::String("ok".to_string()),
                ],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(updated) = updated else {
            panic!("expected successful construction update");
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Box")],
                &[(*updated).clone()],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");
        let Value::ResultOk(finished) = finished else {
            panic!("expected successful construction finish");
        };
        let Value::Struct {
            type_name: finished_type_name,
            fields: finished_fields,
        } = *finished
        else {
            panic!("expected constructed struct");
        };
        assert_eq!(finished_type_name, "Box");
        assert_eq!(
            finished_fields,
            vec![("value".to_string(), Value::String("ok".to_string()))]
        );

        let Value::TypeConstruction {
            type_name,
            variant,
            state,
            fields,
        } = *updated
        else {
            panic!("expected TypeConstruction");
        };

        assert_eq!(type_name, "Box");
        assert_eq!(variant, None);
        assert_eq!(state, None);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, 0);
        assert_eq!(fields[0].1, "value");
        assert_eq!(fields[0].2, "string");
        assert_eq!(fields[0].3, Value::String("ok".to_string()));
    }

    #[test]
    fn type_construct_put_and_finish_normalize_checked_uint64_struct_field() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Counter",
            "struct",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_type_fields("Counter", vec![uint64_field_info(0, "value")]);

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Counter")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let type_fields = interp
            .call_builtin_with_type_args("type.fields", &[type_named("Counter")], &[])
            .expect("type.fields should be a typed builtin")
            .expect("type.fields should evaluate");
        let Value::List(type_fields) = type_fields else {
            panic!("expected list of TypeField values");
        };

        let updated = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Counter"), type_named("uint64")],
                &[builder, type_fields[0].clone(), Value::Int64(42)],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(updated) = updated else {
            panic!("expected successful construction update");
        };
        let Value::TypeConstruction { fields, .. } = updated.as_ref() else {
            panic!("expected TypeConstruction");
        };
        assert_eq!(fields[0].3, Value::Uint64(42));

        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Counter")],
                &[(*updated).clone()],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");
        let Value::ResultOk(finished) = finished else {
            panic!("expected successful construction finish");
        };

        assert_eq!(
            *finished,
            Value::Struct {
                type_name: "Counter".to_string(),
                fields: vec![("value".to_string(), Value::Uint64(42))],
            }
        );
    }

    #[test]
    fn type_construct_enum_payload_and_finish_use_checked_reflection_metadata_when_available() {
        let field_type = ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Choice",
            "enum",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_type_variants(
            "Choice",
            vec![ReflectionVariantInfo::new(
                0,
                "token",
                7,
                false,
                vec![ReflectionFieldInfo::new(
                    0,
                    "value",
                    "string",
                    "primitive",
                    "value",
                    false,
                    field_type,
                )],
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let variants = interp
            .call_builtin_with_type_args("type.variants", &[type_named("Choice")], &[])
            .expect("type.variants should be a typed builtin")
            .expect("type.variants should evaluate");
        let Value::List(variants) = variants else {
            panic!("expected list of TypeVariant values");
        };
        let Value::Struct {
            fields: variant_fields,
            ..
        } = &variants[0]
        else {
            panic!("expected TypeVariant");
        };
        let payload_fields = variant_fields
            .iter()
            .find_map(
                |(name, value)| {
                    if name == "fields" { Some(value) } else { None }
                },
            )
            .expect("TypeVariant.fields should exist");
        let Value::List(payload_fields) = payload_fields else {
            panic!("expected TypeVariant.fields list");
        };

        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_variant_start",
                &[type_named("Choice")],
                &[variants[0].clone()],
            )
            .expect("type.construct_variant_start should be a typed builtin")
            .expect("type.construct_variant_start should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful enum construction start");
        };

        let updated = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Choice"), type_named("string")],
                &[
                    *builder,
                    payload_fields[0].clone(),
                    Value::String("ok".to_string()),
                ],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(updated) = updated else {
            panic!("expected successful enum payload update");
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Choice")],
                &[(*updated).clone()],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");
        let Value::ResultOk(finished) = finished else {
            panic!("expected successful enum construction finish");
        };
        let Value::Enum {
            type_name: finished_type_name,
            variant: finished_variant,
            fields: finished_fields,
        } = *finished
        else {
            panic!("expected constructed enum");
        };
        assert_eq!(finished_type_name, "Choice");
        assert_eq!(finished_variant, "token");
        assert_eq!(finished_fields, vec![Value::String("ok".to_string())]);

        let Value::TypeConstruction {
            type_name,
            variant,
            state,
            fields,
        } = *updated
        else {
            panic!("expected TypeConstruction");
        };

        assert_eq!(type_name, "Choice");
        assert_eq!(variant, Some("token".to_string()));
        assert_eq!(state, None);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, 0);
        assert_eq!(fields[0].1, "value");
        assert_eq!(fields[0].2, "string");
        assert_eq!(fields[0].3, Value::String("ok".to_string()));
    }

    #[test]
    fn type_construct_put_and_finish_normalize_checked_uint64_enum_payload() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Choice",
            "enum",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_type_variants(
            "Choice",
            vec![ReflectionVariantInfo::new(
                0,
                "token",
                7,
                false,
                vec![uint64_field_info(0, "value")],
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let variants = interp
            .call_builtin_with_type_args("type.variants", &[type_named("Choice")], &[])
            .expect("type.variants should be a typed builtin")
            .expect("type.variants should evaluate");
        let Value::List(variants) = variants else {
            panic!("expected list of TypeVariant values");
        };
        let Value::Struct {
            fields: variant_fields,
            ..
        } = &variants[0]
        else {
            panic!("expected TypeVariant");
        };
        let payload_fields = variant_fields
            .iter()
            .find_map(
                |(name, value)| {
                    if name == "fields" { Some(value) } else { None }
                },
            )
            .expect("TypeVariant.fields should exist");
        let Value::List(payload_fields) = payload_fields else {
            panic!("expected TypeVariant.fields list");
        };

        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_variant_start",
                &[type_named("Choice")],
                &[variants[0].clone()],
            )
            .expect("type.construct_variant_start should be a typed builtin")
            .expect("type.construct_variant_start should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful enum construction start");
        };
        let updated = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Choice"), type_named("uint64")],
                &[*builder, payload_fields[0].clone(), Value::Int64(42)],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(updated) = updated else {
            panic!("expected successful enum payload update");
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Choice")],
                &[(*updated).clone()],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");
        let Value::ResultOk(finished) = finished else {
            panic!("expected successful enum construction finish");
        };

        assert_eq!(
            *finished,
            Value::Enum {
                type_name: "Choice".to_string(),
                variant: "token".to_string(),
                fields: vec![Value::Uint64(42)],
            }
        );
    }

    #[test]
    fn type_construct_finish_checked_struct_reports_missing_field() {
        let field_type = ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Box",
            "struct",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_type_fields(
            "Box",
            vec![ReflectionFieldInfo::new(
                0,
                "value",
                "string",
                "primitive",
                "value",
                false,
                field_type,
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Box")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let finished = interp
            .call_builtin_with_type_args("type.construct_finish", &[type_named("Box")], &[builder])
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_fail(
                "type.construct_finish: 'Box' is missing required field 'value'".to_string()
            )
        );
    }

    #[test]
    fn type_construct_put_reports_missing_checked_field_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Box",
            "struct",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Box")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let field =
            Interpreter::reflection_field_info_value("Box", None, &string_field_info(0, "value"));
        let updated = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Box"), type_named("string")],
                &[builder, field, Value::String("ok".to_string())],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");

        assert_eq!(
            updated,
            result_fail(
                "checked reflection metadata for type 'Box' is missing field metadata".to_string()
            )
        );
    }

    #[test]
    fn type_construct_finish_reports_missing_checked_field_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Box",
            "struct",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Box")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let finished = interp
            .call_builtin_with_type_args("type.construct_finish", &[type_named("Box")], &[builder])
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_fail(
                "checked reflection metadata for type 'Box' is missing field metadata".to_string()
            )
        );
    }

    #[test]
    fn type_construct_finish_checked_enum_reports_missing_payload_field() {
        let field_type = ReflectionTypeInfo::new(
            "string",
            "primitive",
            Some("string_type".to_string()),
            false,
            Vec::new(),
        );
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Choice",
            "enum",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_type_variants(
            "Choice",
            vec![ReflectionVariantInfo::new(
                0,
                "token",
                7,
                false,
                vec![ReflectionFieldInfo::new(
                    0,
                    "value",
                    "string",
                    "primitive",
                    "value",
                    false,
                    field_type,
                )],
            )],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let variants = interp
            .call_builtin_with_type_args("type.variants", &[type_named("Choice")], &[])
            .expect("type.variants should be a typed builtin")
            .expect("type.variants should evaluate");
        let Value::List(variants) = variants else {
            panic!("expected list of TypeVariant values");
        };
        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_variant_start",
                &[type_named("Choice")],
                &[variants[0].clone()],
            )
            .expect("type.construct_variant_start should be a typed builtin")
            .expect("type.construct_variant_start should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful enum construction start");
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Choice")],
                &[*builder],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_fail(
                "type.construct_finish: variant 'Choice.token' is missing required payload field 'value'"
                    .to_string()
            )
        );
    }

    #[test]
    fn type_construct_variant_start_reports_missing_checked_variant_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Choice",
            "enum",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let variant = Interpreter::reflection_variant_info_value(
            "Choice",
            &ReflectionVariantInfo::new(0, "token", 7, false, Vec::new()),
        );
        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_variant_start",
                &[type_named("Choice")],
                &[variant],
            )
            .expect("type.construct_variant_start should be a typed builtin")
            .expect("type.construct_variant_start should evaluate");

        assert_eq!(
            builder,
            result_fail(
                "checked reflection metadata for type 'Choice' is missing variant metadata"
                    .to_string()
            )
        );
    }

    #[test]
    fn type_construct_finish_reports_missing_checked_variant_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Choice",
            "enum",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let builder = Value::TypeConstruction {
            type_name: "Choice".to_string(),
            variant: Some("token".to_string()),
            state: None,
            fields: Vec::new(),
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Choice")],
                &[builder],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_fail(
                "checked reflection metadata for type 'Choice' is missing variant metadata"
                    .to_string()
            )
        );
    }

    #[test]
    fn type_construct_finish_reports_missing_checked_bitfield_metadata() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Header",
            "bitfield",
            None,
            false,
            Vec::new(),
        ));

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Header")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Header")],
                &[builder],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_fail(
                "checked reflection metadata for type 'Header' is missing bitfield metadata"
                    .to_string()
            )
        );
    }

    #[test]
    fn type_construct_bitfield_finish_uses_checked_reflection_metadata_when_available() {
        let int_info = ReflectionTypeInfo::new(
            "int64",
            "primitive",
            Some("int64_type".to_string()),
            false,
            Vec::new(),
        );
        let protocol_info = ReflectionTypeInfo::new("IpProtocol", "enum", None, false, Vec::new());
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Header",
            "bitfield",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_type_fields(
            "Header",
            vec![
                ReflectionFieldInfo::new(
                    0,
                    "version",
                    "int64",
                    "primitive",
                    "version",
                    false,
                    int_info.clone(),
                ),
                ReflectionFieldInfo::new(
                    1,
                    "protocol",
                    "IpProtocol",
                    "enum",
                    "protocol",
                    false,
                    protocol_info.clone(),
                ),
            ],
        );
        metadata.insert_bitfield(
            "Header",
            ReflectionBitfieldInfo::new(
                true,
                vec![
                    ReflectionBitfieldFieldInfo::new(0, "version", "bits", 4, int_info, None),
                    ReflectionBitfieldFieldInfo::new(
                        1,
                        "protocol",
                        "bits",
                        8,
                        protocol_info.clone(),
                        Some(protocol_info),
                    ),
                ],
            ),
        );
        metadata.insert_type_variants(
            "IpProtocol",
            vec![ReflectionVariantInfo::new(0, "tcp", 6, false, Vec::new())],
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let type_fields = interp
            .call_builtin_with_type_args("type.fields", &[type_named("Header")], &[])
            .expect("type.fields should be a typed builtin")
            .expect("type.fields should evaluate");
        let Value::List(type_fields) = type_fields else {
            panic!("expected list of TypeField values");
        };
        let protocol = Value::Enum {
            type_name: "IpProtocol".to_string(),
            variant: "tcp".to_string(),
            fields: Vec::new(),
        };

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Header")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Header"), type_named("int64")],
                &[builder, type_fields[0].clone(), Value::Int64(4)],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful version field update");
        };
        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Header"), type_named("IpProtocol")],
                &[(*builder).clone(), type_fields[1].clone(), protocol.clone()],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful protocol field update");
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Header")],
                &[(*builder).clone()],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_ok(Value::Struct {
                type_name: "Header".to_string(),
                fields: vec![
                    ("version".to_string(), Value::Int64(4)),
                    ("protocol".to_string(), protocol.clone()),
                ],
            })
        );

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Header")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Header"), type_named("int64")],
                &[builder, type_fields[0].clone(), Value::Int64(16)],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful wide version field update");
        };
        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Header"), type_named("IpProtocol")],
                &[(*builder).clone(), type_fields[1].clone(), protocol],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful wide protocol field update");
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Header")],
                &[*builder],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_fail(
                "bitfield 'Header' field 'version' is 4 bit(s) wide and cannot hold '16'"
                    .to_string()
            )
        );
    }

    #[test]
    fn type_construct_bitfield_finish_normalizes_checked_uint64_bits() {
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Header",
            "bitfield",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_bitfield(
            "Header",
            ReflectionBitfieldInfo::new(
                true,
                vec![ReflectionBitfieldFieldInfo::new(
                    0,
                    "wide",
                    "bits",
                    64,
                    uint64_type_info(),
                    None,
                )],
            ),
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let builder = Value::TypeConstruction {
            type_name: "Header".to_string(),
            variant: None,
            state: None,
            fields: vec![(
                0,
                "wide".to_string(),
                "uint64".to_string(),
                Value::Int64(42),
            )],
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Header")],
                &[builder],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_ok(Value::Struct {
                type_name: "Header".to_string(),
                fields: vec![("wide".to_string(), Value::Uint64(42))],
            })
        );
    }

    #[test]
    fn type_construct_bitfield_finish_requires_checked_enum_variant_metadata() {
        let int_info = ReflectionTypeInfo::new(
            "int64",
            "primitive",
            Some("int64_type".to_string()),
            false,
            Vec::new(),
        );
        let protocol_info = ReflectionTypeInfo::new("IpProtocol", "enum", None, false, Vec::new());
        let mut metadata = ReflectionMetadata::new();
        metadata.insert_type_info(ReflectionTypeInfo::new(
            "Header",
            "bitfield",
            None,
            false,
            Vec::new(),
        ));
        metadata.insert_type_info(protocol_info.clone());
        metadata.insert_type_fields(
            "Header",
            vec![
                ReflectionFieldInfo::new(
                    0,
                    "version",
                    "int64",
                    "primitive",
                    "version",
                    false,
                    int_info.clone(),
                ),
                ReflectionFieldInfo::new(
                    1,
                    "protocol",
                    "IpProtocol",
                    "enum",
                    "protocol",
                    false,
                    protocol_info.clone(),
                ),
            ],
        );
        metadata.insert_bitfield(
            "Header",
            ReflectionBitfieldInfo::new(
                true,
                vec![
                    ReflectionBitfieldFieldInfo::new(0, "version", "bits", 4, int_info, None),
                    ReflectionBitfieldFieldInfo::new(
                        1,
                        "protocol",
                        "bits",
                        8,
                        protocol_info.clone(),
                        Some(protocol_info),
                    ),
                ],
            ),
        );

        let mut interp = Interpreter::new();
        interp.set_reflection_metadata(Arc::new(metadata));

        let type_fields = interp
            .call_builtin_with_type_args("type.fields", &[type_named("Header")], &[])
            .expect("type.fields should be a typed builtin")
            .expect("type.fields should evaluate");
        let Value::List(type_fields) = type_fields else {
            panic!("expected list of TypeField values");
        };
        let protocol = Value::Enum {
            type_name: "IpProtocol".to_string(),
            variant: "tcp".to_string(),
            fields: Vec::new(),
        };

        let builder = interp
            .call_builtin_with_type_args("type.construct_start", &[type_named("Header")], &[])
            .expect("type.construct_start should be a typed builtin")
            .expect("type.construct_start should evaluate");
        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Header"), type_named("int64")],
                &[builder, type_fields[0].clone(), Value::Int64(4)],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful version field update");
        };
        let builder = interp
            .call_builtin_with_type_args(
                "type.construct_put",
                &[type_named("Header"), type_named("IpProtocol")],
                &[(*builder).clone(), type_fields[1].clone(), protocol],
            )
            .expect("type.construct_put should be a typed builtin")
            .expect("type.construct_put should evaluate");
        let Value::ResultOk(builder) = builder else {
            panic!("expected successful protocol field update");
        };
        let finished = interp
            .call_builtin_with_type_args(
                "type.construct_finish",
                &[type_named("Header")],
                &[*builder],
            )
            .expect("type.construct_finish should be a typed builtin")
            .expect("type.construct_finish should evaluate");

        assert_eq!(
            finished,
            result_fail(
                "checked reflection metadata for type 'IpProtocol' is missing variant metadata"
                    .to_string()
            )
        );
    }

    /// Helper: create a field access expression.
    fn field_access(base: Expr, field: &str) -> Expr {
        Expr::FieldAccess(Box::new(base), ident(field), sp())
    }

    /// Helper: create a named call argument.
    fn named_arg(name: &str, value: Expr) -> CallArg {
        CallArg {
            name: Some(ident(name)),
            value,
            span: sp(),
        }
    }

    /// Helper: create a dotted call expression like `Point.total(view point)`.
    fn dotted_call(module: &str, func_name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call(
            Box::new(field_access(var(module), func_name)),
            args.into_iter()
                .map(|value| CallArg {
                    name: None,
                    value,
                    span: sp(),
                })
                .collect(),
            sp(),
        )
    }

    /// Helper: create a simple struct definition.
    fn struct_def(name: &str, fields: Vec<(&str, &str)>, methods: Vec<FunctionDef>) -> StructDef {
        StructDef {
            name: ident(name),
            type_params: vec![],
            fields: fields
                .into_iter()
                .map(|(field_name, field_ty)| FieldDef {
                    name: ident(field_name),
                    ty: type_named(field_ty),
                    serialize_name: None,
                    span: sp(),
                })
                .collect(),
            methods,
            exported: false,
            span: sp(),
        }
    }

    fn machine_def(
        name: &str,
        states: Vec<(&str, Vec<(&str, TypeExpr)>)>,
        transitions: Vec<(&str, &str)>,
    ) -> MachineDef {
        MachineDef {
            name: ident(name),
            exported: false,
            states: states
                .into_iter()
                .map(|(state_name, fields)| jett_parser::ast::MachineState {
                    name: ident(state_name),
                    fields: fields
                        .into_iter()
                        .map(|(field_name, ty)| FieldDef {
                            name: ident(field_name),
                            ty,
                            serialize_name: None,
                            span: sp(),
                        })
                        .collect(),
                    span: sp(),
                })
                .collect(),
            transitions: transitions
                .into_iter()
                .map(|(from, to)| jett_parser::ast::MachineTransition {
                    from: ident(from),
                    to: ident(to),
                    span: sp(),
                })
                .collect(),
            span: sp(),
        }
    }

    fn enum_def_with_values(name: &str, variants: Vec<(&str, i64)>) -> EnumDef {
        EnumDef {
            name: ident(name),
            variants: variants
                .into_iter()
                .map(|(variant_name, discriminant)| jett_parser::ast::Variant {
                    name: ident(variant_name),
                    fields: vec![],
                    discriminant: Some(discriminant),
                    span: sp(),
                })
                .collect(),
            exported: false,
            span: sp(),
        }
    }

    fn enum_def_with_field(
        name: &str,
        variant_name: &str,
        field_name: &str,
        field_ty: &str,
    ) -> EnumDef {
        EnumDef {
            name: ident(name),
            variants: vec![jett_parser::ast::Variant {
                name: ident(variant_name),
                fields: vec![FieldDef {
                    name: ident(field_name),
                    ty: type_named(field_ty),
                    serialize_name: None,
                    span: sp(),
                }],
                discriminant: None,
                span: sp(),
            }],
            exported: false,
            span: sp(),
        }
    }

    /// Helper: create a simple bitfield definition.
    fn bitfield_def(
        name: &str,
        fields: Vec<(&str, BitfieldFieldKind)>,
        network_order: bool,
    ) -> BitfieldDef {
        BitfieldDef {
            name: ident(name),
            network_order,
            fields: fields
                .into_iter()
                .map(|(field_name, kind)| jett_parser::ast::BitfieldFieldDef {
                    name: ident(field_name),
                    kind,
                    span: sp(),
                })
                .collect(),
            exported: false,
            span: sp(),
        }
    }

    fn interface_decl(
        name: &str,
        methods: Vec<(&str, Vec<(&str, &str, bool)>, &str)>,
    ) -> InterfaceDecl {
        InterfaceDecl {
            name: ident(name),
            methods: methods
                .into_iter()
                .map(|(method_name, params, return_type)| FunctionDecl {
                    name: ident(method_name),
                    type_params: vec![],
                    params: params
                        .into_iter()
                        .map(|(param_name, param_ty, view)| Param {
                            view,
                            mutable: false,
                            name: ident(param_name),
                            ty: type_named(param_ty),
                            span: sp(),
                        })
                        .collect(),
                    return_type: Some(type_named(return_type)),
                    exported: false,
                    span: sp(),
                })
                .collect(),
            exported: false,
            span: sp(),
        }
    }

    fn implement_block(
        interface_name: &str,
        for_type: &str,
        methods: Vec<FunctionDef>,
    ) -> ImplementBlock {
        ImplementBlock {
            interface_name: ident(interface_name),
            for_type: type_named(for_type),
            methods,
            span: sp(),
        }
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn eval_integer_addition() {
        let mut interp = Interpreter::new();
        let expr = binary(int(2), BinOp::Add, int(3));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(5));
    }

    #[test]
    fn eval_integer_subtraction() {
        let mut interp = Interpreter::new();
        let expr = binary(int(10), BinOp::Sub, int(4));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(6));
    }

    #[test]
    fn eval_integer_multiplication() {
        let mut interp = Interpreter::new();
        let expr = binary(int(3), BinOp::Mul, int(7));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(21));
    }

    #[test]
    fn eval_integer_division() {
        let mut interp = Interpreter::new();
        let expr = binary(int(15), BinOp::Div, int(3));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(5));
    }

    #[test]
    fn eval_integer_modulo() {
        let mut interp = Interpreter::new();
        let expr = binary(int(17), BinOp::Modulo, int(5));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(2));
    }

    #[test]
    fn eval_float_addition() {
        let mut interp = Interpreter::new();
        let expr = binary(float(1.5), BinOp::Add, float(2.5));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Float64(4.0));
    }

    #[test]
    fn eval_complex_arithmetic() {
        // (2 + 3) * 4 == 20
        let mut interp = Interpreter::new();
        let sum = binary(int(2), BinOp::Add, int(3));
        let product = binary(sum, BinOp::Mul, int(4));
        assert_eq!(interp.eval_expr(&product).unwrap(), Value::Int64(20));
    }

    #[test]
    fn eval_division_by_zero() {
        let mut interp = Interpreter::new();
        let expr = binary(int(10), BinOp::Div, int(0));
        assert!(interp.eval_expr(&expr).is_err());
    }

    #[test]
    fn eval_integer_division_overflow_reports_error() {
        let mut interp = Interpreter::new();
        let expr = binary(int(i64::MIN), BinOp::Div, int(-1));
        let err = interp.eval_expr(&expr).unwrap_err();
        assert_eq!(
            err,
            "integer overflow: -9223372036854775808 / -1".to_string()
        );
    }

    #[test]
    fn eval_integer_modulo_overflow_reports_error() {
        let mut interp = Interpreter::new();
        let expr = binary(int(i64::MIN), BinOp::Modulo, int(-1));
        let err = interp.eval_expr(&expr).unwrap_err();
        assert_eq!(
            err,
            "integer overflow: -9223372036854775808 % -1".to_string()
        );
    }

    // -----------------------------------------------------------------------
    // Boolean logic
    // -----------------------------------------------------------------------

    #[test]
    fn eval_and_true_true() {
        let mut interp = Interpreter::new();
        let expr = binary(bool_expr(true), BinOp::And, bool_expr(true));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn eval_and_true_false() {
        let mut interp = Interpreter::new();
        let expr = binary(bool_expr(true), BinOp::And, bool_expr(false));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(false));
    }

    #[test]
    fn eval_or_false_true() {
        let mut interp = Interpreter::new();
        let expr = binary(bool_expr(false), BinOp::Or, bool_expr(true));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn eval_or_false_false() {
        let mut interp = Interpreter::new();
        let expr = binary(bool_expr(false), BinOp::Or, bool_expr(false));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(false));
    }

    #[test]
    fn eval_not_true() {
        let mut interp = Interpreter::new();
        let expr = Expr::Unary(UnaryOp::Not, Box::new(bool_expr(true)), sp());
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(false));
    }

    #[test]
    fn eval_not_false() {
        let mut interp = Interpreter::new();
        let expr = Expr::Unary(UnaryOp::Not, Box::new(bool_expr(false)), sp());
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn eval_comparison_eq() {
        let mut interp = Interpreter::new();
        let expr = binary(int(5), BinOp::Eq, int(5));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn eval_comparison_neq() {
        let mut interp = Interpreter::new();
        let expr = binary(int(5), BinOp::NotEq, int(3));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn eval_comparison_lt() {
        let mut interp = Interpreter::new();
        let expr = binary(int(3), BinOp::Lt, int(5));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn eval_comparison_gte() {
        let mut interp = Interpreter::new();
        let expr = binary(int(5), BinOp::GtEq, int(5));
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    // -----------------------------------------------------------------------
    // String literals
    // -----------------------------------------------------------------------

    #[test]
    fn eval_string_literal() {
        let mut interp = Interpreter::new();
        let expr = string("hello");
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn eval_string_concatenation() {
        let mut interp = Interpreter::new();
        let expr = binary(string("hello "), BinOp::Add, string("world"));
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("hello world".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // String interpolation
    // -----------------------------------------------------------------------

    #[test]
    fn string_interpolation_simple() {
        // "hello {name}" with name = "world"
        let mut interp = Interpreter::new();
        interp
            .exec_stmt(&var_decl("name", string("world")))
            .unwrap();
        let expr = Expr::StringInterpolation(
            vec![
                StringPart::Literal("hello ".to_string()),
                StringPart::Expr(Box::new(var("name"))),
            ],
            sp(),
        );
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("hello world".to_string())
        );
    }

    #[test]
    fn string_interpolation_multiple() {
        // "{a} + {b} = {c}" with a=2, b=3, c=5
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("a", int(2))).unwrap();
        interp.exec_stmt(&var_decl("b", int(3))).unwrap();
        interp.exec_stmt(&var_decl("c", int(5))).unwrap();
        let expr = Expr::StringInterpolation(
            vec![
                StringPart::Expr(Box::new(var("a"))),
                StringPart::Literal(" + ".to_string()),
                StringPart::Expr(Box::new(var("b"))),
                StringPart::Literal(" = ".to_string()),
                StringPart::Expr(Box::new(var("c"))),
            ],
            sp(),
        );
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("2 + 3 = 5".to_string())
        );
    }

    #[test]
    fn string_interpolation_with_function_call() {
        // "result: {add(2, 3)}" with a function add(a, b) returns a + b
        let mut interp = Interpreter::new();
        let add_fn = func_def(
            "add",
            vec![("a", "int64"), ("b", "int64")],
            Block {
                stmts: vec![Stmt::Return(ReturnStmt {
                    value: Some(binary(var("a"), BinOp::Add, var("b"))),
                    span: sp(),
                })],
                span: sp(),
            },
        );
        interp.register_function(&add_fn);
        let expr = Expr::StringInterpolation(
            vec![
                StringPart::Literal("result: ".to_string()),
                StringPart::Expr(Box::new(call("add", vec![int(2), int(3)]))),
            ],
            sp(),
        );
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("result: 5".to_string())
        );
    }

    #[test]
    fn string_interpolation_plain_string_still_works() {
        // "plain string" with no interpolation should still work as StringLiteral
        let mut interp = Interpreter::new();
        let expr = string("plain string");
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("plain string".to_string())
        );
    }

    #[test]
    fn declassify_is_a_runtime_no_op() {
        let mut interp = Interpreter::new();
        interp.set_variable_public("api_key", Value::String("abc".to_string()));

        let expr = Expr::Declassify(Box::new(var("api_key")), sp());
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("abc".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // Trusted stdlib hooks
    // -----------------------------------------------------------------------

    #[test]
    fn json_parse_bridge_requires_trusted_stdlib_hook() {
        let mut interp = Interpreter::new();
        let fake_hook = generic_json_hook(
            "json_parse_reflected",
            block(vec![return_stmt(string("fake"))]),
        );
        interp.register_function_named("json.json_parse_reflected", &fake_hook, false);

        let err = interp
            .call_function_with_type_args(
                "json.parse",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.parse requires trusted stdlib hook 'json.json_parse_reflected'"
        );
    }

    #[test]
    fn json_parse_bridge_requires_trusted_public_wrapper() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_parse_reflected",
            block(vec![return_stmt(string("private hook"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);

        let err = interp
            .call_function_with_type_args(
                "json.parse",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.parse requires trusted stdlib wrapper 'json.parse'"
        );
    }

    #[test]
    fn json_public_bridges_all_require_trusted_public_wrapper() {
        let cases = [
            ("json.parse", "json_parse_reflected", "\"value\""),
            (
                "json.parse_exact",
                "json_parse_exact_reflected",
                "\"value\"",
            ),
            ("json.serialize", "json_serialize_reflected", "value"),
            (
                "json.serialize_public",
                "json_serialize_public_reflected",
                "value",
            ),
        ];

        for (public_name, hook_name, arg) in cases {
            let mut interp = Interpreter::new();
            let mut trusted_hook =
                generic_json_hook(hook_name, block(vec![return_stmt(string("private hook"))]));
            trusted_hook.span = stdlib_sp();
            interp.register_function_in_namespace(Some("json"), &trusted_hook);

            let err = interp
                .call_function_with_type_args(
                    public_name,
                    &[type_named("string")],
                    vec![Value::String(arg.to_string())],
                )
                .unwrap_err();

            assert_eq!(
                err,
                format!("{public_name} requires trusted stdlib wrapper '{public_name}'")
            );
        }
    }

    #[test]
    fn json_public_bridges_all_reject_untrusted_public_wrapper() {
        let cases = [
            ("json.parse", "json_parse_reflected", "parse", "\"value\""),
            (
                "json.parse_exact",
                "json_parse_exact_reflected",
                "parse_exact",
                "\"value\"",
            ),
            (
                "json.serialize",
                "json_serialize_reflected",
                "serialize",
                "value",
            ),
            (
                "json.serialize_public",
                "json_serialize_public_reflected",
                "serialize_public",
                "value",
            ),
        ];

        for (public_name, hook_name, wrapper_name, arg) in cases {
            let mut interp = Interpreter::new();
            let mut trusted_hook =
                generic_json_hook(hook_name, block(vec![return_stmt(string("private hook"))]));
            trusted_hook.span = stdlib_sp();
            interp.register_function_in_namespace(Some("json"), &trusted_hook);
            let fake_wrapper =
                generic_json_hook(wrapper_name, block(vec![return_stmt(string("fake"))]));
            interp.register_function_named(public_name, &fake_wrapper, false);

            let err = interp
                .call_function_with_type_args(
                    public_name,
                    &[type_named("string")],
                    vec![Value::String(arg.to_string())],
                )
                .unwrap_err();

            assert_eq!(
                err,
                format!("{public_name} requires trusted stdlib wrapper '{public_name}'")
            );
        }
    }

    #[test]
    fn json_parse_bridge_uses_trusted_public_wrapper() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_parse_reflected",
            block(vec![return_stmt(string("private hook"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);
        let mut public_wrapper =
            generic_json_hook("parse", block(vec![return_stmt(string("public wrapper"))]));
        public_wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &public_wrapper);

        let value = interp
            .call_function_with_type_args(
                "json.parse",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap();

        assert_eq!(value, Value::String("public wrapper".to_string()));
    }

    #[test]
    fn untrusted_json_public_wrapper_cannot_satisfy_bridge() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_parse_reflected",
            block(vec![return_stmt(string("private hook"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);

        let fake_wrapper = generic_json_hook("parse", block(vec![return_stmt(string("fake"))]));
        interp.register_function_named("json.parse", &fake_wrapper, false);

        let err = interp
            .call_function_with_type_args(
                "json.parse",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.parse requires trusted stdlib wrapper 'json.parse'"
        );
    }

    #[test]
    fn trusted_json_public_wrapper_still_requires_trusted_private_hook() {
        let mut interp = Interpreter::new();
        let fake_hook = generic_json_hook(
            "json_parse_reflected",
            block(vec![return_stmt(string("fake"))]),
        );
        interp.register_function_named("json.json_parse_reflected", &fake_hook, false);
        let mut public_wrapper =
            generic_json_hook("parse", block(vec![return_stmt(string("public wrapper"))]));
        public_wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &public_wrapper);

        let err = interp
            .call_function_with_type_args(
                "json.parse",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.parse requires trusted stdlib hook 'json.json_parse_reflected'"
        );
    }

    #[test]
    fn json_parse_exact_bridge_requires_trusted_stdlib_hook() {
        let mut interp = Interpreter::new();
        let fake_hook = generic_json_hook(
            "json_parse_exact_reflected",
            block(vec![return_stmt(string("fake exact"))]),
        );
        interp.register_function_named("json.json_parse_exact_reflected", &fake_hook, false);

        let err = interp
            .call_function_with_type_args(
                "json.parse_exact",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.parse_exact requires trusted stdlib hook 'json.json_parse_exact_reflected'"
        );
    }

    #[test]
    fn json_parse_exact_bridge_uses_trusted_public_wrapper() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_parse_exact_reflected",
            block(vec![return_stmt(string("private exact"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);
        let mut public_wrapper = generic_json_hook(
            "parse_exact",
            block(vec![return_stmt(string("public exact"))]),
        );
        public_wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &public_wrapper);

        let value = interp
            .call_function_with_type_args(
                "json.parse_exact",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap();

        assert_eq!(value, Value::String("public exact".to_string()));
    }

    #[test]
    fn json_serialize_bridge_requires_trusted_stdlib_hook() {
        let mut interp = Interpreter::new();
        let fake_hook = generic_json_hook(
            "json_serialize_reflected",
            block(vec![return_stmt(string("fake"))]),
        );
        interp.register_function_named("json.json_serialize_reflected", &fake_hook, false);

        let err = interp
            .call_function_with_type_args(
                "json.serialize",
                &[type_named("string")],
                vec![Value::String("value".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.serialize requires trusted stdlib hook 'json.json_serialize_reflected'"
        );
    }

    #[test]
    fn json_serialize_bridge_uses_trusted_public_wrapper() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_serialize_reflected",
            block(vec![return_stmt(string("private serialized"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);
        let mut public_wrapper = generic_json_hook(
            "serialize",
            block(vec![return_stmt(string("public serialized"))]),
        );
        public_wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &public_wrapper);

        let value = interp
            .call_function_with_type_args(
                "json.serialize",
                &[type_named("string")],
                vec![Value::String("value".to_string())],
            )
            .unwrap();

        assert_eq!(value, Value::String("public serialized".to_string()));
    }

    #[test]
    fn json_serialize_public_bridge_requires_trusted_stdlib_hook() {
        let mut interp = Interpreter::new();
        let fake_hook = generic_json_hook(
            "json_serialize_public_reflected",
            block(vec![return_stmt(string("fake"))]),
        );
        interp.register_function_named("json.json_serialize_public_reflected", &fake_hook, false);

        let err = interp
            .call_function_with_type_args(
                "json.serialize_public",
                &[type_named("string")],
                vec![Value::String("value".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.serialize_public requires trusted stdlib hook 'json.json_serialize_public_reflected'"
        );
    }

    #[test]
    fn json_serialize_public_bridge_uses_trusted_public_wrapper() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_serialize_public_reflected",
            block(vec![return_stmt(string("private public serialized"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);
        let mut public_wrapper = generic_json_hook(
            "serialize_public",
            block(vec![return_stmt(string("public serialized"))]),
        );
        public_wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &public_wrapper);

        let value = interp
            .call_function_with_type_args(
                "json.serialize_public",
                &[type_named("string")],
                vec![Value::String("value".to_string())],
            )
            .unwrap();

        assert_eq!(value, Value::String("public serialized".to_string()));
    }

    #[test]
    fn json_parse_raw_without_stdlib_wrapper_is_undefined() {
        let mut interp = Interpreter::new();

        let err = interp
            .call_function("json.parse_raw", vec![Value::String("null".to_string())])
            .unwrap_err();

        assert_eq!(err, "undefined function 'json.parse_raw'");
    }

    #[test]
    fn json_raw_facades_without_stdlib_wrappers_are_undefined() {
        let mut interp = Interpreter::new();

        for name in JSON_RAW_FACADE_NAMES {
            let err = interp.call_function(name, Vec::new()).unwrap_err();
            assert_eq!(err, format!("undefined function '{name}'"));
        }
    }

    #[test]
    fn json_parse_raw_uses_public_stdlib_wrapper() {
        let mut interp = Interpreter::new();
        let mut wrapper = func_def(
            "parse_raw",
            vec![("raw", "string")],
            block(vec![return_stmt(string("public raw"))]),
        );
        wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &wrapper);

        let value = interp
            .call_function(
                "json.parse_raw",
                vec![Value::String("not json".to_string())],
            )
            .unwrap();

        assert_eq!(value, Value::String("public raw".to_string()));
    }

    #[test]
    fn json_raw_helpers_use_public_stdlib_wrappers() {
        let mut interp = Interpreter::new();

        for name in JSON_RAW_FACADE_NAMES {
            let short_name = name
                .strip_prefix("json.")
                .expect("raw facade names should be qualified");
            let mut wrapper = func_def(
                short_name,
                vec![("value", "JsonTree")],
                block(vec![return_stmt(string("public tree"))]),
            );
            wrapper.span = stdlib_sp();
            interp.register_function_in_namespace(Some("json"), &wrapper);

            assert!(
                interp.is_trusted_stdlib_first_function(name),
                "{name} should run through the trusted stdlib wrapper path"
            );
        }
    }

    #[test]
    fn public_raw_facade_wrapper_runs_before_private_helper() {
        let mut interp = Interpreter::new();
        let mut public_wrapper = func_def(
            "kind",
            vec![("value", "JsonTree")],
            block(vec![return_stmt(string("public wrapper"))]),
        );
        public_wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &public_wrapper);

        let mut trusted_hook = func_def(
            "json_tree_kind",
            vec![("value", "JsonTree")],
            block(vec![return_stmt(string("hook fallback"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);

        let value = interp
            .call_function("json.kind", vec![json_tree_null()])
            .unwrap();

        assert_eq!(value, Value::String("public wrapper".to_string()));
    }

    #[test]
    fn dotted_raw_facade_expr_runs_public_wrapper() {
        let mut interp = Interpreter::new();
        interp.set_variable_public("tree", json_tree_null());

        let mut public_wrapper = func_def(
            "kind",
            vec![("value", "JsonTree")],
            block(vec![return_stmt(string("public wrapper"))]),
        );
        public_wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &public_wrapper);

        let value = interp
            .eval_expr(&dotted_call("json", "kind", vec![var("tree")]))
            .unwrap();

        assert_eq!(value, Value::String("public wrapper".to_string()));
    }

    #[test]
    fn pipeline_raw_facade_step_runs_public_wrapper() {
        let mut interp = Interpreter::new();
        interp.set_variable_public("tree", json_tree_null());

        let mut public_wrapper = func_def(
            "kind",
            vec![("value", "JsonTree")],
            block(vec![return_stmt(string("public wrapper"))]),
        );
        public_wrapper.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &public_wrapper);

        let value = interp
            .eval_expr(&Expr::Pipeline(
                Box::new(var("tree")),
                vec![PipelineStep {
                    function: field_access(var("json"), "kind"),
                    extra_args: Vec::new(),
                    handle: None,
                    span: sp(),
                }],
                sp(),
            ))
            .unwrap();

        assert_eq!(value, Value::String("public wrapper".to_string()));
    }

    #[test]
    fn raw_facade_wrapper_is_ordinary_function_without_builtin_fallback() {
        let mut interp = Interpreter::new();
        let wrapper = func_def(
            "kind",
            vec![("value", "JsonTree")],
            block(vec![return_stmt(string("ordinary wrapper"))]),
        );
        interp.register_function_named("json.kind", &wrapper, false);

        let value = interp
            .call_function("json.kind", vec![json_tree_null()])
            .unwrap();

        assert_eq!(value, Value::String("ordinary wrapper".to_string()));
    }

    #[test]
    fn untrusted_registration_removes_previous_json_hook_trust() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_parse_reflected",
            block(vec![return_stmt(string("trusted"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);

        let fake_hook = generic_json_hook(
            "json_parse_reflected",
            block(vec![return_stmt(string("fake"))]),
        );
        interp.register_function_named("json.json_parse_reflected", &fake_hook, false);

        let err = interp
            .call_function_with_type_args(
                "json.parse",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.parse requires trusted stdlib hook 'json.json_parse_reflected'"
        );
    }

    #[test]
    fn untrusted_registration_removes_previous_json_parse_exact_hook_trust() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_parse_exact_reflected",
            block(vec![return_stmt(string("trusted exact"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);

        let fake_hook = generic_json_hook(
            "json_parse_exact_reflected",
            block(vec![return_stmt(string("fake exact"))]),
        );
        interp.register_function_named("json.json_parse_exact_reflected", &fake_hook, false);

        let err = interp
            .call_function_with_type_args(
                "json.parse_exact",
                &[type_named("string")],
                vec![Value::String("\"value\"".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.parse_exact requires trusted stdlib hook 'json.json_parse_exact_reflected'"
        );
    }

    #[test]
    fn untrusted_registration_removes_previous_json_serialize_hook_trust() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_serialize_reflected",
            block(vec![return_stmt(string("trusted"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);

        let fake_hook = generic_json_hook(
            "json_serialize_reflected",
            block(vec![return_stmt(string("fake"))]),
        );
        interp.register_function_named("json.json_serialize_reflected", &fake_hook, false);

        let err = interp
            .call_function_with_type_args(
                "json.serialize",
                &[type_named("string")],
                vec![Value::String("value".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.serialize requires trusted stdlib hook 'json.json_serialize_reflected'"
        );
    }

    #[test]
    fn untrusted_registration_removes_previous_json_serialize_public_hook_trust() {
        let mut interp = Interpreter::new();
        let mut trusted_hook = generic_json_hook(
            "json_serialize_public_reflected",
            block(vec![return_stmt(string("trusted"))]),
        );
        trusted_hook.span = stdlib_sp();
        interp.register_function_in_namespace(Some("json"), &trusted_hook);

        let fake_hook = generic_json_hook(
            "json_serialize_public_reflected",
            block(vec![return_stmt(string("fake"))]),
        );
        interp.register_function_named("json.json_serialize_public_reflected", &fake_hook, false);

        let err = interp
            .call_function_with_type_args(
                "json.serialize_public",
                &[type_named("string")],
                vec![Value::String("value".to_string())],
            )
            .unwrap_err();

        assert_eq!(
            err,
            "json.serialize_public requires trusted stdlib hook 'json.json_serialize_public_reflected'"
        );
    }

    // -----------------------------------------------------------------------
    // Variable declaration and lookup
    // -----------------------------------------------------------------------

    #[test]
    fn variable_declaration_and_lookup() {
        let mut interp = Interpreter::new();
        let decl = var_decl("x", int(42));
        interp.exec_stmt(&decl).unwrap();
        let result = interp.eval_expr(&var("x")).unwrap();
        assert_eq!(result, Value::Int64(42));
    }

    #[test]
    fn uint64_variable_declaration_normalizes_small_literal_carrier() {
        let mut interp = Interpreter::new();
        interp
            .exec_stmt(&typed_var_decl("uint64", "x", int(42)))
            .unwrap();

        assert_eq!(interp.eval_expr(&var("x")).unwrap(), Value::Uint64(42));
    }

    #[test]
    fn uint64_function_boundaries_normalize_small_literal_carrier() {
        let mut interp = Interpreter::new();
        let mut echo = func_def(
            "echo_u64",
            vec![("value", "uint64")],
            block(vec![return_stmt(var("value"))]),
        );
        echo.return_type = Some(type_named("uint64"));
        interp.register_function(&echo);

        let mut make = func_def("make_u64", vec![], block(vec![return_stmt(int(7))]));
        make.return_type = Some(type_named("uint64"));
        interp.register_function(&make);

        assert_eq!(
            interp
                .call_function("echo_u64", vec![Value::Int64(42)])
                .unwrap(),
            Value::Uint64(42)
        );
        assert_eq!(
            interp.call_function("make_u64", Vec::new()).unwrap(),
            Value::Uint64(7)
        );
    }

    #[test]
    fn uint64_inline_function_parameter_normalizes_small_literal_carrier() {
        let mut interp = Interpreter::new();
        let fn_value = Value::Function {
            params: vec![Param {
                view: false,
                mutable: false,
                name: ident("value"),
                ty: type_named("uint64"),
                span: sp(),
            }],
            body: block(vec![return_stmt(var("value"))]),
            captures: HashMap::new(),
        };

        assert_eq!(
            interp
                .call_fn_value(fn_value, vec![Value::Int64(42)])
                .unwrap(),
            Value::Uint64(42)
        );
    }

    #[test]
    fn uint64_actor_boundaries_normalize_small_literal_carriers() {
        let mut interp = Interpreter::new();
        let uint64_param = |name: &str| Param {
            view: false,
            mutable: false,
            name: ident(name),
            ty: type_named("uint64"),
            span: sp(),
        };
        let actor = ActorDef {
            name: ident("Probe"),
            capability_params: vec![uint64_param("seed")],
            state_fields: vec![VarDecl {
                mutable: true,
                ty: type_named("uint64"),
                name: ident("stored"),
                value: var("seed"),
                span: sp(),
            }],
            handlers: vec![
                ReceiveHandler {
                    name: ident("echo"),
                    params: vec![uint64_param("value")],
                    responds: Some(type_named("uint64")),
                    body: block(vec![respond_stmt(var("value"))]),
                    span: sp(),
                },
                ReceiveHandler {
                    name: ident("stored"),
                    params: vec![],
                    responds: Some(type_named("uint64")),
                    body: block(vec![respond_stmt(var("stored"))]),
                    span: sp(),
                },
                ReceiveHandler {
                    name: ident("reset"),
                    params: vec![],
                    responds: Some(type_named("uint64")),
                    body: block(vec![assign("stored", int(7)), respond_stmt(var("stored"))]),
                    span: sp(),
                },
            ],
            exported: false,
            span: sp(),
        };
        interp.register_actor(&actor);

        let actor_value = interp
            .eval_expr(&Expr::Spawn(
                Box::new(Expr::Call(
                    Box::new(var("Probe")),
                    vec![named_arg("seed", int(42))],
                    sp(),
                )),
                sp(),
            ))
            .unwrap();
        interp.set_variable("probe", actor_value);

        let ask_echo = Expr::Ask(
            Box::new(Expr::Call(
                Box::new(field_access(var("probe"), "echo")),
                vec![CallArg {
                    name: None,
                    value: int(5),
                    span: sp(),
                }],
                sp(),
            )),
            sp(),
        );
        let ask_stored = Expr::Ask(Box::new(field_access(var("probe"), "stored")), sp());
        let ask_reset = Expr::Ask(Box::new(field_access(var("probe"), "reset")), sp());

        assert_eq!(interp.eval_expr(&ask_echo).unwrap(), Value::Uint64(5));
        assert_eq!(interp.eval_expr(&ask_stored).unwrap(), Value::Uint64(42));
        assert_eq!(interp.eval_expr(&ask_reset).unwrap(), Value::Uint64(7));
        assert_eq!(interp.eval_expr(&ask_stored).unwrap(), Value::Uint64(7));
    }

    #[test]
    fn variable_assignment() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("x", int(1))).unwrap();
        interp.exec_stmt(&assign("x", int(99))).unwrap();
        assert_eq!(interp.eval_expr(&var("x")).unwrap(), Value::Int64(99));
    }

    #[test]
    fn uint64_variable_assignment_normalizes_small_literal_carrier() {
        let mut interp = Interpreter::new();
        interp
            .exec_stmt(&typed_var_decl("uint64", "x", int(1)))
            .unwrap();
        interp.exec_stmt(&assign("x", int(42))).unwrap();

        assert_eq!(interp.eval_expr(&var("x")).unwrap(), Value::Uint64(42));
    }

    #[test]
    fn uint64_parameter_assignment_normalizes_small_literal_carrier() {
        let mut interp = Interpreter::new();
        let reassign = func_def(
            "reassign_u64",
            vec![("value", "uint64")],
            block(vec![assign("value", int(7)), return_stmt(var("value"))]),
        );
        interp.register_function(&reassign);

        assert_eq!(
            interp
                .call_function("reassign_u64", vec![Value::Int64(1)])
                .unwrap(),
            Value::Uint64(7)
        );
    }

    #[test]
    fn trace_stmt_records_current_value() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("total", int(42))).unwrap();

        interp
            .exec_stmt(&Stmt::Trace(TraceStmt {
                name: ident("total"),
                span: sp(),
            }))
            .unwrap();

        assert_eq!(interp.take_debug_output(), vec!["trace total: int64 = 42"]);
    }

    #[test]
    fn breakpoint_stmt_records_visible_bindings() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("total", int(42))).unwrap();

        interp
            .exec_stmt(&Stmt::Breakpoint(BreakpointStmt {
                condition: Some(bool_expr(true)),
                span: sp(),
            }))
            .unwrap();

        assert_eq!(
            interp.take_debug_output(),
            vec!["breakpoint hit: total: int64 = 42"]
        );
    }

    #[test]
    fn breakpoint_stmt_skips_when_condition_is_false() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("total", int(42))).unwrap();

        interp
            .exec_stmt(&Stmt::Breakpoint(BreakpointStmt {
                condition: Some(bool_expr(false)),
                span: sp(),
            }))
            .unwrap();

        assert!(interp.take_debug_output().is_empty());
    }

    #[test]
    fn undefined_variable_error() {
        let mut interp = Interpreter::new();
        assert!(interp.eval_expr(&var("nonexistent")).is_err());
    }

    // -----------------------------------------------------------------------
    // If/else branching
    // -----------------------------------------------------------------------

    #[test]
    fn if_then_branch_taken() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("result", int(0))).unwrap();
        let if_stmt = Stmt::If(IfStmt {
            condition: bool_expr(true),
            then_block: block(vec![assign("result", int(1))]),
            else_ifs: vec![],
            else_block: None,
            span: sp(),
        });
        interp.exec_stmt(&if_stmt).unwrap();
        assert_eq!(interp.eval_expr(&var("result")).unwrap(), Value::Int64(1));
    }

    #[test]
    fn if_else_branch_taken() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("result", int(0))).unwrap();
        let if_stmt = Stmt::If(IfStmt {
            condition: bool_expr(false),
            then_block: block(vec![assign("result", int(1))]),
            else_ifs: vec![],
            else_block: Some(block(vec![assign("result", int(2))])),
            span: sp(),
        });
        interp.exec_stmt(&if_stmt).unwrap();
        assert_eq!(interp.eval_expr(&var("result")).unwrap(), Value::Int64(2));
    }

    #[test]
    fn if_else_if_branch_taken() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("result", int(0))).unwrap();
        let if_stmt = Stmt::If(IfStmt {
            condition: bool_expr(false),
            then_block: block(vec![assign("result", int(1))]),
            else_ifs: vec![(bool_expr(true), block(vec![assign("result", int(3))]))],
            else_block: Some(block(vec![assign("result", int(2))])),
            span: sp(),
        });
        interp.exec_stmt(&if_stmt).unwrap();
        assert_eq!(interp.eval_expr(&var("result")).unwrap(), Value::Int64(3));
    }

    // -----------------------------------------------------------------------
    // For loop
    // -----------------------------------------------------------------------

    #[test]
    fn for_loop_over_list() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("sum", int(0))).unwrap();

        let list = Expr::ListConstruct(vec![int(1), int(2), int(3)], sp());
        let for_stmt = Stmt::For(ForStmt {
            variable: ident("item"),
            value_variable: None,
            view: false,
            iterable: list,
            body: block(vec![assign(
                "sum",
                binary(var("sum"), BinOp::Add, var("item")),
            )]),
            span: sp(),
        });
        interp.exec_stmt(&for_stmt).unwrap();
        assert_eq!(interp.eval_expr(&var("sum")).unwrap(), Value::Int64(6));
    }

    #[test]
    fn for_loop_with_break() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("count", int(0))).unwrap();

        let list = Expr::ListConstruct(vec![int(1), int(2), int(3), int(4), int(5)], sp());
        let for_stmt = Stmt::For(ForStmt {
            variable: ident("item"),
            value_variable: None,
            view: false,
            iterable: list,
            body: block(vec![
                // if item == 4: break
                Stmt::If(IfStmt {
                    condition: binary(var("item"), BinOp::Eq, int(4)),
                    then_block: block(vec![Stmt::Break(sp())]),
                    else_ifs: vec![],
                    else_block: None,
                    span: sp(),
                }),
                assign("count", binary(var("count"), BinOp::Add, int(1))),
            ]),
            span: sp(),
        });
        interp.exec_stmt(&for_stmt).unwrap();
        // Should have counted 1, 2, 3 (stopped before 4)
        assert_eq!(interp.eval_expr(&var("count")).unwrap(), Value::Int64(3));
    }

    // -----------------------------------------------------------------------
    // While loop
    // -----------------------------------------------------------------------

    #[test]
    fn while_loop_with_counter() {
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("i", int(0))).unwrap();
        interp.exec_stmt(&var_decl("sum", int(0))).unwrap();

        // while i < 5: sum = sum + i; i = i + 1
        let while_stmt = Stmt::While(WhileStmt {
            condition: binary(var("i"), BinOp::Lt, int(5)),
            body: block(vec![
                assign("sum", binary(var("sum"), BinOp::Add, var("i"))),
                assign("i", binary(var("i"), BinOp::Add, int(1))),
            ]),
            span: sp(),
        });
        interp.exec_stmt(&while_stmt).unwrap();
        // sum = 0 + 1 + 2 + 3 + 4 = 10
        assert_eq!(interp.eval_expr(&var("sum")).unwrap(), Value::Int64(10));
    }

    // -----------------------------------------------------------------------
    // Function calls
    // -----------------------------------------------------------------------

    #[test]
    fn function_call_add() {
        let mut interp = Interpreter::new();

        // function add(a: int64, b: int64) returns int64:
        //     return a + b
        let add_fn = func_def(
            "add",
            vec![("a", "int64"), ("b", "int64")],
            block(vec![return_stmt(binary(var("a"), BinOp::Add, var("b")))]),
        );
        interp.register_function(&add_fn);

        let result = interp
            .eval_expr(&call("add", vec![int(3), int(4)]))
            .unwrap();
        assert_eq!(result, Value::Int64(7));
    }

    #[test]
    fn function_call_no_return() {
        let mut interp = Interpreter::new();

        // function noop():
        //     pass (empty body)
        let noop_fn = func_def("noop", vec![], block(vec![]));
        interp.register_function(&noop_fn);

        let result = interp.eval_expr(&call("noop", vec![])).unwrap();
        assert_eq!(result, Value::Nothing);
    }

    #[test]
    fn function_wrong_arg_count() {
        let mut interp = Interpreter::new();
        let add_fn = func_def(
            "add",
            vec![("a", "int64"), ("b", "int64")],
            block(vec![return_stmt(binary(var("a"), BinOp::Add, var("b")))]),
        );
        interp.register_function(&add_fn);

        let result = interp.eval_expr(&call("add", vec![int(1)]));
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // User-defined structs
    // -----------------------------------------------------------------------

    #[test]
    fn struct_constructor_returns_struct_value() {
        let mut interp = Interpreter::new();
        interp.register_struct(&struct_def(
            "Point",
            vec![("x", "int64"), ("y", "int64")],
            vec![],
        ));

        let expr = Expr::Call(
            Box::new(var("Point")),
            vec![named_arg("x", int(3)), named_arg("y", int(4))],
            sp(),
        );

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::Struct {
                type_name: "Point".to_string(),
                fields: vec![
                    ("x".to_string(), Value::Int64(3)),
                    ("y".to_string(), Value::Int64(4)),
                ],
            }
        );
    }

    #[test]
    fn struct_constructor_normalizes_uint64_field_carrier() {
        let mut interp = Interpreter::new();
        interp.register_struct(&struct_def("Packet", vec![("serial", "uint64")], vec![]));

        let expr = Expr::Call(
            Box::new(var("Packet")),
            vec![named_arg("serial", int(42))],
            sp(),
        );

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::Struct {
                type_name: "Packet".to_string(),
                fields: vec![("serial".to_string(), Value::Uint64(42))],
            }
        );
    }

    #[test]
    fn struct_constructor_with_refinement_field_returns_result_ok() {
        let mut interp = Interpreter::new();
        interp.register_type_alias(&type_alias(
            "Age",
            "int64",
            Some(binary(
                binary(var("value"), BinOp::GtEq, int(0)),
                BinOp::And,
                binary(var("value"), BinOp::Lt, int(150)),
            )),
        ));
        interp.register_struct(&struct_def("User", vec![("age", "Age")], vec![]));

        let expr = Expr::Call(Box::new(var("User")), vec![named_arg("age", int(42))], sp());

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultOk(Box::new(Value::Struct {
                type_name: "User".to_string(),
                fields: vec![("age".to_string(), Value::Int64(42))],
            }))
        );
    }

    #[test]
    fn struct_constructor_with_refinement_field_returns_result_fail() {
        let mut interp = Interpreter::new();
        interp.register_type_alias(&type_alias(
            "Age",
            "int64",
            Some(binary(
                binary(var("value"), BinOp::GtEq, int(0)),
                BinOp::And,
                binary(var("value"), BinOp::Lt, int(150)),
            )),
        ));
        interp.register_struct(&struct_def("User", vec![("age", "Age")], vec![]));

        let expr = Expr::Call(
            Box::new(var("User")),
            vec![named_arg("age", int(200))],
            sp(),
        );

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultFail(Box::new(Value::String(
                "refinement type constraint failed for 'Age'".to_string(),
            )))
        );
    }

    #[test]
    fn bitfield_constructor_with_literals_returns_value() {
        let mut interp = Interpreter::new();
        interp.register_bitfield(&bitfield_def(
            "TcpFlags",
            vec![
                (
                    "syn",
                    BitfieldFieldKind::Bits {
                        width: 1,
                        as_type: None,
                    },
                ),
                (
                    "ack",
                    BitfieldFieldKind::Bits {
                        width: 1,
                        as_type: None,
                    },
                ),
            ],
            false,
        ));

        let expr = Expr::Call(
            Box::new(var("TcpFlags")),
            vec![named_arg("syn", int(0)), named_arg("ack", int(1))],
            sp(),
        );

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::Struct {
                type_name: "TcpFlags".to_string(),
                fields: vec![
                    ("syn".to_string(), Value::Int64(0)),
                    ("ack".to_string(), Value::Int64(1)),
                ],
            }
        );
    }

    #[test]
    fn bitfield_constructor_with_dynamic_field_returns_result_ok() {
        let mut interp = Interpreter::new();
        interp.register_bitfield(&bitfield_def(
            "TcpFlags",
            vec![
                (
                    "syn",
                    BitfieldFieldKind::Bits {
                        width: 1,
                        as_type: None,
                    },
                ),
                (
                    "ack",
                    BitfieldFieldKind::Bits {
                        width: 1,
                        as_type: None,
                    },
                ),
            ],
            false,
        ));
        interp.set_variable("bit", Value::Int64(1));

        let expr = Expr::Call(
            Box::new(var("TcpFlags")),
            vec![named_arg("syn", var("bit")), named_arg("ack", int(0))],
            sp(),
        );

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultOk(Box::new(Value::Struct {
                type_name: "TcpFlags".to_string(),
                fields: vec![
                    ("syn".to_string(), Value::Int64(1)),
                    ("ack".to_string(), Value::Int64(0)),
                ],
            }))
        );
    }

    #[test]
    fn bitfield_constructor_with_dynamic_field_returns_result_fail() {
        let mut interp = Interpreter::new();
        interp.register_bitfield(&bitfield_def(
            "TcpFlags",
            vec![
                (
                    "syn",
                    BitfieldFieldKind::Bits {
                        width: 1,
                        as_type: None,
                    },
                ),
                (
                    "ack",
                    BitfieldFieldKind::Bits {
                        width: 1,
                        as_type: None,
                    },
                ),
            ],
            false,
        ));
        interp.set_variable("bit", Value::Int64(2));

        let expr = Expr::Call(
            Box::new(var("TcpFlags")),
            vec![named_arg("syn", var("bit")), named_arg("ack", int(0))],
            sp(),
        );

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultFail(Box::new(Value::String(
                "bitfield 'TcpFlags' field 'syn' is 1 bit(s) wide and cannot hold '2'".to_string(),
            )))
        );
    }

    #[test]
    fn bitfield_field_access_reads_registered_field() {
        let mut interp = Interpreter::new();
        interp.register_bitfield(&bitfield_def(
            "TcpFlags",
            vec![
                (
                    "syn",
                    BitfieldFieldKind::Bits {
                        width: 1,
                        as_type: None,
                    },
                ),
                (
                    "ack",
                    BitfieldFieldKind::Bits {
                        width: 1,
                        as_type: None,
                    },
                ),
                (
                    "payload",
                    BitfieldFieldKind::Payload(TypeExpr::Generic(
                        ident("list"),
                        vec![type_named("uint8")],
                        sp(),
                    )),
                ),
            ],
            false,
        ));

        let expr = field_access(
            Expr::Call(
                Box::new(var("TcpFlags")),
                vec![
                    named_arg("syn", int(1)),
                    named_arg("ack", int(0)),
                    named_arg("payload", Expr::ListConstruct(vec![int(1), int(2)], sp())),
                ],
                sp(),
            ),
            "ack",
        );

        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(0));
    }

    #[test]
    fn bitfield_to_bytes_packs_network_order_fields() {
        let mut interp = Interpreter::new();
        interp.register_bitfield(&bitfield_def(
            "IpHeader",
            vec![
                (
                    "version",
                    BitfieldFieldKind::Bits {
                        width: 4,
                        as_type: None,
                    },
                ),
                (
                    "header_length",
                    BitfieldFieldKind::Bits {
                        width: 4,
                        as_type: None,
                    },
                ),
                (
                    "total_length",
                    BitfieldFieldKind::Bits {
                        width: 16,
                        as_type: None,
                    },
                ),
            ],
            true,
        ));

        let expr = dotted_call(
            "IpHeader",
            "to_bytes",
            vec![Expr::Call(
                Box::new(var("IpHeader")),
                vec![
                    named_arg("version", int(4)),
                    named_arg("header_length", int(5)),
                    named_arg("total_length", int(500)),
                ],
                sp(),
            )],
        );

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::Bytes(vec![0x45, 0x01, 0xF4])
        );
    }

    #[test]
    fn bitfield_from_bytes_unpacks_network_order_fields() {
        let mut interp = Interpreter::new();
        interp.register_bitfield(&bitfield_def(
            "IpHeader",
            vec![
                (
                    "version",
                    BitfieldFieldKind::Bits {
                        width: 4,
                        as_type: None,
                    },
                ),
                (
                    "header_length",
                    BitfieldFieldKind::Bits {
                        width: 4,
                        as_type: None,
                    },
                ),
                (
                    "total_length",
                    BitfieldFieldKind::Bits {
                        width: 16,
                        as_type: None,
                    },
                ),
            ],
            true,
        ));

        let expr = dotted_call("IpHeader", "from_bytes", vec![var("raw")]);
        interp.set_variable("raw", Value::Bytes(vec![0x45, 0x01, 0xF4]));

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultOk(Box::new(Value::Struct {
                type_name: "IpHeader".to_string(),
                fields: vec![
                    ("version".to_string(), Value::Int64(4)),
                    ("header_length".to_string(), Value::Int64(5)),
                    ("total_length".to_string(), Value::Int64(500)),
                ],
            }))
        );
    }

    #[test]
    fn bitfield_to_bytes_uses_enum_annotations() {
        let mut interp = Interpreter::new();
        interp.register_enum(&enum_def_with_values(
            "IpProtocol",
            vec![("icmp", 1), ("tcp", 6), ("udp", 17)],
        ));
        interp.register_bitfield(&bitfield_def(
            "Header",
            vec![(
                "protocol",
                BitfieldFieldKind::Bits {
                    width: 8,
                    as_type: Some(type_named("IpProtocol")),
                },
            )],
            true,
        ));

        let expr = dotted_call(
            "Header",
            "to_bytes",
            vec![Expr::Call(
                Box::new(var("Header")),
                vec![named_arg(
                    "protocol",
                    Expr::EnumVariant(ident("IpProtocol"), ident("tcp"), sp()),
                )],
                sp(),
            )],
        );

        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bytes(vec![6]));
    }

    #[test]
    fn bitfield_from_bytes_decodes_enum_annotations() {
        let mut interp = Interpreter::new();
        interp.register_enum(&enum_def_with_values(
            "IpProtocol",
            vec![("icmp", 1), ("tcp", 6), ("udp", 17)],
        ));
        interp.register_bitfield(&bitfield_def(
            "Header",
            vec![(
                "protocol",
                BitfieldFieldKind::Bits {
                    width: 8,
                    as_type: Some(type_named("IpProtocol")),
                },
            )],
            true,
        ));

        let expr = dotted_call("Header", "from_bytes", vec![var("raw")]);
        interp.set_variable("raw", Value::Bytes(vec![17]));

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultOk(Box::new(Value::Struct {
                type_name: "Header".to_string(),
                fields: vec![(
                    "protocol".to_string(),
                    Value::Enum {
                        type_name: "IpProtocol".to_string(),
                        variant: "udp".to_string(),
                        fields: vec![],
                    },
                )],
            }))
        );
    }

    #[test]
    fn enum_variant_constructor_normalizes_uint64_payload_carrier() {
        let mut interp = Interpreter::new();
        interp.register_enum(&enum_def_with_field("Event", "serial", "value", "uint64"));

        let expr = Expr::Call(
            Box::new(Expr::EnumVariant(ident("Event"), ident("serial"), sp())),
            vec![CallArg {
                name: None,
                value: int(42),
                span: sp(),
            }],
            sp(),
        );

        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::Enum {
                type_name: "Event".to_string(),
                variant: "serial".to_string(),
                fields: vec![Value::Uint64(42)],
            }
        );
    }

    #[test]
    fn function_parameter_refinement_rejects_invalid_argument() {
        let mut interp = Interpreter::new();
        interp.register_type_alias(&type_alias(
            "Age",
            "int64",
            Some(binary(
                binary(var("value"), BinOp::GtEq, int(0)),
                BinOp::And,
                binary(var("value"), BinOp::Lt, int(150)),
            )),
        ));

        let mut accept_age = func_def(
            "accept_age",
            vec![("age", "Age")],
            block(vec![return_stmt(var("age"))]),
        );
        accept_age.return_type = Some(type_named("int64"));
        interp.register_function(&accept_age);

        assert_eq!(
            interp
                .call_function("accept_age", vec![Value::Int64(200)])
                .unwrap_err(),
            "refinement type constraint failed for 'Age'".to_string()
        );
    }

    #[test]
    fn function_with_refinement_return_accepts_valid_value() {
        let mut interp = Interpreter::new();
        interp.register_type_alias(&type_alias(
            "Port",
            "int64",
            Some(binary(
                binary(var("value"), BinOp::GtEq, int(1)),
                BinOp::And,
                binary(var("value"), BinOp::LtEq, int(65535)),
            )),
        ));

        let mut default_port =
            func_def("default_port", vec![], block(vec![return_stmt(int(8080))]));
        default_port.return_type = Some(type_named("Port"));
        interp.register_function(&default_port);

        assert_eq!(
            interp.call_function("default_port", vec![]).unwrap(),
            Value::Int64(8080)
        );
    }

    #[test]
    fn function_with_refinement_return_rejects_invalid_value() {
        let mut interp = Interpreter::new();
        interp.register_type_alias(&type_alias(
            "Port",
            "int64",
            Some(binary(
                binary(var("value"), BinOp::GtEq, int(1)),
                BinOp::And,
                binary(var("value"), BinOp::LtEq, int(65535)),
            )),
        ));

        let mut invalid_port =
            func_def("invalid_port", vec![], block(vec![return_stmt(int(70000))]));
        invalid_port.return_type = Some(type_named("Port"));
        interp.register_function(&invalid_port);

        assert_eq!(
            interp.call_function("invalid_port", vec![]).unwrap_err(),
            "refinement type constraint failed for 'Port'".to_string()
        );
    }

    #[test]
    fn struct_field_access_reads_registered_field() {
        let mut interp = Interpreter::new();
        interp.register_struct(&struct_def(
            "Point",
            vec![("x", "int64"), ("y", "int64")],
            vec![],
        ));

        let expr = field_access(
            Expr::Call(
                Box::new(var("Point")),
                vec![named_arg("x", int(8)), named_arg("y", int(13))],
                sp(),
            ),
            "x",
        );

        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(8));
    }

    #[test]
    fn struct_method_call_uses_struct_fields() {
        let mut interp = Interpreter::new();
        let mut total_method = func_def(
            "total",
            vec![("self", "Point")],
            block(vec![return_stmt(binary(
                field_access(var("self"), "x"),
                BinOp::Add,
                field_access(var("self"), "y"),
            ))]),
        );
        total_method.params[0].view = true;

        interp.register_struct(&struct_def(
            "Point",
            vec![("x", "int64"), ("y", "int64")],
            vec![total_method],
        ));
        interp
            .exec_stmt(&Stmt::VarDecl(VarDecl {
                mutable: false,
                ty: type_named("Point"),
                name: ident("point"),
                value: Expr::Call(
                    Box::new(var("Point")),
                    vec![named_arg("x", int(10)), named_arg("y", int(20))],
                    sp(),
                ),
                span: sp(),
            }))
            .unwrap();

        let expr = dotted_call(
            "Point",
            "total",
            vec![Expr::View(Box::new(var("point")), sp())],
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(30));
    }

    #[test]
    fn interface_call_dispatches_to_struct_implementation() {
        let mut interp = Interpreter::new();
        interp.register_interface(&interface_decl(
            "Speaker",
            vec![("speak", vec![("self", "Speaker", true)], "string")],
        ));
        interp.register_struct(&struct_def("Dog", vec![("name", "string")], vec![]));

        let mut speak_method = func_def(
            "speak",
            vec![("self", "Dog")],
            block(vec![return_stmt(field_access(var("self"), "name"))]),
        );
        speak_method.params[0].view = true;
        interp.register_implement_block(&implement_block("Speaker", "Dog", vec![speak_method]));

        interp
            .exec_stmt(&Stmt::VarDecl(VarDecl {
                mutable: false,
                ty: type_named("Dog"),
                name: ident("dog"),
                value: Expr::Call(
                    Box::new(var("Dog")),
                    vec![named_arg("name", string("woof"))],
                    sp(),
                ),
                span: sp(),
            }))
            .unwrap();

        let expr = dotted_call(
            "Speaker",
            "speak",
            vec![Expr::View(Box::new(var("dog")), sp())],
        );
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("woof".to_string())
        );
    }

    #[test]
    fn interface_call_dispatches_to_primitive_implementation() {
        let mut interp = Interpreter::new();
        interp.register_interface(&interface_decl(
            "Displayable",
            vec![("display", vec![("self", "Displayable", true)], "string")],
        ));

        let mut display_method = func_def(
            "display",
            vec![("self", "int64")],
            block(vec![return_stmt(string("forty-two"))]),
        );
        display_method.params[0].view = true;
        interp.register_implement_block(&implement_block(
            "Displayable",
            "int64",
            vec![display_method],
        ));

        let expr = dotted_call(
            "Displayable",
            "display",
            vec![Expr::View(Box::new(int(42)), sp())],
        );
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("forty-two".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // Nested function calls
    // -----------------------------------------------------------------------

    #[test]
    fn nested_function_calls() {
        let mut interp = Interpreter::new();

        // function double(x: int64) returns int64:
        //     return x * 2
        let double_fn = func_def(
            "double",
            vec![("x", "int64")],
            block(vec![return_stmt(binary(var("x"), BinOp::Mul, int(2)))]),
        );
        interp.register_function(&double_fn);

        // function add(a: int64, b: int64) returns int64:
        //     return a + b
        let add_fn = func_def(
            "add",
            vec![("a", "int64"), ("b", "int64")],
            block(vec![return_stmt(binary(var("a"), BinOp::Add, var("b")))]),
        );
        interp.register_function(&add_fn);

        // add(double(3), double(5)) == 16
        let expr = call(
            "add",
            vec![call("double", vec![int(3)]), call("double", vec![int(5)])],
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(16));
    }

    #[test]
    fn recursive_function_call() {
        let mut interp = Interpreter::new();

        // function factorial(n: int64) returns int64:
        //     if n <= 1:
        //         return 1
        //     return n * factorial(n - 1)
        let factorial_fn = func_def(
            "factorial",
            vec![("n", "int64")],
            block(vec![
                Stmt::If(IfStmt {
                    condition: binary(var("n"), BinOp::LtEq, int(1)),
                    then_block: block(vec![return_stmt(int(1))]),
                    else_ifs: vec![],
                    else_block: None,
                    span: sp(),
                }),
                return_stmt(binary(
                    var("n"),
                    BinOp::Mul,
                    call("factorial", vec![binary(var("n"), BinOp::Sub, int(1))]),
                )),
            ]),
        );
        interp.register_function(&factorial_fn);

        let result = interp.eval_expr(&call("factorial", vec![int(5)])).unwrap();
        assert_eq!(result, Value::Int64(120));
    }

    // -----------------------------------------------------------------------
    // Assert
    // -----------------------------------------------------------------------

    #[test]
    fn assert_passing() {
        let mut interp = Interpreter::new();
        // assert (2 + 3) == 5
        let stmt = assert_stmt(
            binary(binary(int(2), BinOp::Add, int(3)), BinOp::Eq, int(5)),
            None,
        );
        interp.exec_stmt(&stmt).unwrap(); // should not error
    }

    #[test]
    fn assert_failing() {
        let mut interp = Interpreter::new();
        // 2 + 2 = 4, which is not a boolean — type error
        let stmt = assert_stmt(binary(int(2), BinOp::Add, int(2)), None);
        assert!(interp.exec_stmt(&stmt).is_err());
    }

    #[test]
    fn assert_failing_with_bool() {
        let mut interp = Interpreter::new();
        let stmt = assert_stmt(
            binary(int(1), BinOp::Eq, int(2)),
            Some(string("one should not equal two")),
        );
        let err = interp.exec_stmt(&stmt).unwrap_err();
        assert_eq!(err, "one should not equal two");
    }

    #[test]
    fn assert_passing_bool() {
        let mut interp = Interpreter::new();
        let stmt = assert_stmt(bool_expr(true), None);
        interp.exec_stmt(&stmt).unwrap();
    }

    // -----------------------------------------------------------------------
    // Block execution returns value from return stmt
    // -----------------------------------------------------------------------

    #[test]
    fn exec_block_with_return() {
        let mut interp = Interpreter::new();
        let b = block(vec![
            var_decl("x", int(10)),
            return_stmt(binary(var("x"), BinOp::Mul, int(2))),
        ]);
        let result = interp.exec_block(&b).unwrap();
        assert_eq!(result, Some(Value::Int64(20)));
    }

    #[test]
    fn exec_block_without_return() {
        let mut interp = Interpreter::new();
        let b = block(vec![var_decl("x", int(10))]);
        let result = interp.exec_block(&b).unwrap();
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // Match statement
    // -----------------------------------------------------------------------

    /// Helper: create an enum value expression (via EnumVariant AST node).
    fn enum_variant(type_name: &str, variant_name: &str) -> Expr {
        Expr::EnumVariant(ident(type_name), ident(variant_name), sp())
    }

    /// Helper: create a match statement.
    fn match_stmt(expr: Expr, arms: Vec<(Pattern, Vec<Stmt>)>) -> Stmt {
        Stmt::Match(MatchStmt {
            expr,
            arms: arms
                .into_iter()
                .map(|(pattern, stmts)| MatchArm {
                    pattern,
                    body: block(stmts),
                    span: sp(),
                })
                .collect(),
            span: sp(),
        })
    }

    #[test]
    fn match_simple_enum_variant() {
        // match Color.green:
        //     red:  result = "red"
        //     green: result = "green"
        //     blue:  result = "blue"
        let mut interp = Interpreter::new();
        interp
            .exec_stmt(&var_decl("result", string("none")))
            .unwrap();

        let m = match_stmt(
            enum_variant("Color", "green"),
            vec![
                (
                    Pattern::Ident(ident("red")),
                    vec![assign("result", string("red"))],
                ),
                (
                    Pattern::Ident(ident("green")),
                    vec![assign("result", string("green"))],
                ),
                (
                    Pattern::Ident(ident("blue")),
                    vec![assign("result", string("blue"))],
                ),
            ],
        );
        interp.exec_stmt(&m).unwrap();
        assert_eq!(
            interp.eval_expr(&var("result")).unwrap(),
            Value::String("green".to_string())
        );
    }

    #[test]
    fn match_destructuring_binds_variables() {
        // shape = Shape.circle(5)
        // match shape:
        //     circle(r):
        //         result = r
        //     rect(w, h):
        //         result = w + h
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("result", int(0))).unwrap();
        interp.set_variable(
            "shape",
            Value::Enum {
                type_name: "Shape".to_string(),
                variant: "circle".to_string(),
                fields: vec![Value::Int64(5)],
            },
        );

        let m = match_stmt(
            var("shape"),
            vec![
                (
                    Pattern::Variant(ident("circle"), vec![ident("r")]),
                    vec![assign("result", var("r"))],
                ),
                (
                    Pattern::Variant(ident("rect"), vec![ident("w"), ident("h")]),
                    vec![assign("result", binary(var("w"), BinOp::Add, var("h")))],
                ),
            ],
        );
        interp.exec_stmt(&m).unwrap();
        assert_eq!(interp.eval_expr(&var("result")).unwrap(), Value::Int64(5));
    }

    #[test]
    fn match_destructuring_rect_variant() {
        // shape = Shape.rect(3, 4)
        // match shape:
        //     circle(r):
        //         result = r
        //     rect(w, h):
        //         result = w + h
        let mut interp = Interpreter::new();
        interp.exec_stmt(&var_decl("result", int(0))).unwrap();
        interp.set_variable(
            "shape",
            Value::Enum {
                type_name: "Shape".to_string(),
                variant: "rect".to_string(),
                fields: vec![Value::Int64(3), Value::Int64(4)],
            },
        );

        let m = match_stmt(
            var("shape"),
            vec![
                (
                    Pattern::Variant(ident("circle"), vec![ident("r")]),
                    vec![assign("result", var("r"))],
                ),
                (
                    Pattern::Variant(ident("rect"), vec![ident("w"), ident("h")]),
                    vec![assign("result", binary(var("w"), BinOp::Add, var("h")))],
                ),
            ],
        );
        interp.exec_stmt(&m).unwrap();
        assert_eq!(interp.eval_expr(&var("result")).unwrap(), Value::Int64(7));
    }

    #[test]
    fn match_other_catch_all() {
        // match Color.blue:
        //     red: result = "red"
        //     other: result = "other"
        let mut interp = Interpreter::new();
        interp
            .exec_stmt(&var_decl("result", string("none")))
            .unwrap();

        let m = match_stmt(
            enum_variant("Color", "blue"),
            vec![
                (
                    Pattern::Ident(ident("red")),
                    vec![assign("result", string("red"))],
                ),
                (
                    Pattern::Other(sp()),
                    vec![assign("result", string("other"))],
                ),
            ],
        );
        interp.exec_stmt(&m).unwrap();
        assert_eq!(
            interp.eval_expr(&var("result")).unwrap(),
            Value::String("other".to_string())
        );
    }

    #[test]
    fn match_with_return_signal() {
        // Test that return inside a match arm propagates correctly.
        let mut interp = Interpreter::new();

        // function pick(color: ?) returns string:
        //     match color:
        //         red: return "red"
        //         other: return "unknown"
        let b = block(vec![match_stmt(
            var("color"),
            vec![
                (
                    Pattern::Ident(ident("red")),
                    vec![return_stmt(string("red"))],
                ),
                (Pattern::Other(sp()), vec![return_stmt(string("unknown"))]),
            ],
        )]);

        // Execute as a block with a variable set up.
        interp.set_variable(
            "color",
            Value::Enum {
                type_name: "Color".to_string(),
                variant: "red".to_string(),
                fields: vec![],
            },
        );
        let result = interp.exec_block(&b).unwrap();
        assert_eq!(result, Some(Value::String("red".to_string())));
    }
}

// ---------------------------------------------------------------------------
// Base64 helpers (no external crate dependency)
// ---------------------------------------------------------------------------

fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i] as u32;
        let b1 = if i + 1 < data.len() {
            data[i + 1] as u32
        } else {
            0
        };
        let b2 = if i + 2 < data.len() {
            data[i + 2] as u32
        } else {
            0
        };
        let combined = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((combined >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((combined >> 12) & 63) as usize] as char);
        if i + 1 < data.len() {
            out.push(ALPHABET[((combined >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if i + 2 < data.len() {
            out.push(ALPHABET[(combined & 63) as usize] as char);
        } else {
            out.push('=');
        }
        i += 3;
    }
    out
}

// ---------------------------------------------------------------------------
// SHA-256 helper (no external crate dependency)
// ---------------------------------------------------------------------------

fn sha256_hash(data: &[u8]) -> String {
    #[rustfmt::skip]
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

// ---------------------------------------------------------------------------
// MD5 helper (no external crate dependency)
// ---------------------------------------------------------------------------

fn md5_hash(data: &[u8]) -> String {
    #[rustfmt::skip]
    const S: [u32; 64] = [
        7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
        5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
        4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
        6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,
    ];
    #[rustfmt::skip]
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    ];
    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());
    for chunk in msg.chunks(64) {
        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = u32::from_le_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        let mut a = a0;
        let mut b = b0;
        let mut c = c0;
        let mut d = d0;
        for i in 0..64usize {
            let (f, g) = if i < 16 {
                ((b & c) | ((!b) & d), i)
            } else if i < 32 {
                ((d & b) | ((!d) & c), (5 * i + 1) % 16)
            } else if i < 48 {
                (b ^ c ^ d, (3 * i + 5) % 16)
            } else {
                (c ^ (b | (!d)), (7 * i) % 16)
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = temp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a0.to_le_bytes());
    out[4..8].copy_from_slice(&b0.to_le_bytes());
    out[8..12].copy_from_slice(&c0.to_le_bytes());
    out[12..16].copy_from_slice(&d0.to_le_bytes());
    out.iter().map(|b| format!("{b:02x}")).collect()
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    fn char_val(c: u8) -> Result<u32, String> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(format!("invalid base64 character: {c:?}")),
        }
    }
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err("base64 string length must be a multiple of 4".to_string());
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let v0 = char_val(bytes[i])?;
        let v1 = char_val(bytes[i + 1])?;
        let v2 = char_val(bytes[i + 2])?;
        let v3 = char_val(bytes[i + 3])?;
        let combined = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
        out.push(((combined >> 16) & 0xFF) as u8);
        if bytes[i + 2] != b'=' {
            out.push(((combined >> 8) & 0xFF) as u8);
        }
        if bytes[i + 3] != b'=' {
            out.push((combined & 0xFF) as u8);
        }
        i += 4;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CSV helpers
// ---------------------------------------------------------------------------

/// Parse CSV records, handling quoted fields with embedded commas, quotes
/// (escaped as `""`), and newlines.
fn parse_csv_records(input: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut has_record_data = false;
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                // Check for escaped quote ""
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            has_record_data = true;
            in_quotes = true;
        } else if ch == ',' {
            has_record_data = true;
            row.push(std::mem::take(&mut current));
        } else if ch == '\n' {
            push_csv_record(&mut records, &mut row, &mut current, &mut has_record_data);
        } else if ch == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            push_csv_record(&mut records, &mut row, &mut current, &mut has_record_data);
        } else {
            has_record_data = true;
            current.push(ch);
        }
    }

    push_csv_record(&mut records, &mut row, &mut current, &mut has_record_data);
    records
}

fn push_csv_record(
    records: &mut Vec<Vec<String>>,
    row: &mut Vec<String>,
    current: &mut String,
    has_record_data: &mut bool,
) {
    if *has_record_data || !current.is_empty() || !row.is_empty() {
        row.push(std::mem::take(current));
        records.push(std::mem::take(row));
    } else {
        row.clear();
        current.clear();
    }
    *has_record_data = false;
}

/// Quote a CSV field if it contains commas, quotes, or newlines.
fn csv_quote_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod builtin_tests {
    use jett_common::{FileId, Span};
    use jett_parser::ast::*;

    use super::*;

    fn sp() -> Span {
        Span::new(FileId::new(0), 0, 0)
    }

    fn ident(name: &str) -> Ident {
        Ident {
            name: name.to_string(),
            span: sp(),
        }
    }

    fn int(n: i64) -> Expr {
        Expr::IntLiteral(n.into(), sp())
    }

    fn float(n: f64) -> Expr {
        Expr::FloatLiteral(n, sp())
    }

    fn string(s: &str) -> Expr {
        Expr::StringLiteral(s.to_string(), sp())
    }

    fn var(name: &str) -> Expr {
        Expr::Ident(ident(name))
    }

    fn dotted_call(module: &str, func_name: &str, args: Vec<Expr>) -> Expr {
        let callee = Expr::FieldAccess(Box::new(var(module)), ident(func_name), sp());
        Expr::Call(
            Box::new(callee),
            args.into_iter()
                .map(|value| CallArg {
                    name: None,
                    value,
                    span: sp(),
                })
                .collect(),
            sp(),
        )
    }

    fn default_stmt(value: Expr) -> Stmt {
        Stmt::Expr(ExprStmt {
            expr: Expr::Default(Box::new(value), sp()),
            span: sp(),
        })
    }

    fn return_stmt(value: Expr) -> Stmt {
        Stmt::Return(ReturnStmt {
            value: Some(value),
            span: sp(),
        })
    }

    fn block(stmts: Vec<Stmt>) -> Block {
        Block { stmts, span: sp() }
    }

    fn func_def(name: &str, body: Block) -> FunctionDef {
        FunctionDef {
            name: ident(name),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body,
            exported: false,
            span: sp(),
        }
    }

    #[test]
    fn builtin_stdout_write() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("Stdout", "write", vec![string("fake_cap"), string("hello")]);
        let result = interp.eval_expr(&expr).unwrap();
        assert_eq!(result, Value::Nothing);
    }

    #[test]
    fn stdout_output_can_be_captured() {
        let mut interp = Interpreter::new();
        interp.enable_stdout_capture();

        assert_eq!(
            interp
                .call_builtin(
                    "Stdout.write",
                    &[Value::Nothing, Value::String("hello ".to_string())],
                )
                .expect("Stdout.write should be a builtin")
                .expect("Stdout.write should succeed"),
            Value::Nothing
        );
        assert_eq!(
            interp
                .call_builtin(
                    "print",
                    &[Value::String("score".to_string()), Value::Int64(7)],
                )
                .expect("print should be a builtin")
                .expect("print should succeed"),
            Value::Nothing
        );
        assert_eq!(
            interp
                .call_builtin("println", &[Value::Bool(true)])
                .expect("println should be a builtin")
                .expect("println should succeed"),
            Value::Nothing
        );

        assert_eq!(interp.take_stdout_output(), "hello score 7true\n");
        assert_eq!(interp.take_stdout_output(), "");
    }

    #[test]
    fn builtin_string_char_count() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("string", "char_count", vec![string("hello")]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(5));
    }

    #[test]
    fn builtin_string_char_count_unicode() {
        let mut interp = Interpreter::new();
        let expr = dotted_call(
            "string",
            "char_count",
            vec![string("\u{00e9}\u{00e9}\u{00e9}")],
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(3));
    }

    #[test]
    fn builtin_string_length() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("string", "length", vec![string("test")]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(4));
    }

    #[test]
    fn builtin_string_contains_true() {
        let mut interp = Interpreter::new();
        let expr = dotted_call(
            "string",
            "contains",
            vec![string("hello world"), string("world")],
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn builtin_string_contains_false() {
        let mut interp = Interpreter::new();
        let expr = dotted_call(
            "string",
            "contains",
            vec![string("hello world"), string("xyz")],
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(false));
    }

    #[test]
    fn builtin_string_trim() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("string", "trim", vec![string("  hello  ")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn builtin_string_upper() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("string", "upper", vec![string("hello")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("HELLO".to_string())
        );
    }

    #[test]
    fn builtin_string_lower() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("string", "lower", vec![string("HELLO")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn builtin_string_split() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("string", "split", vec![string("a,b,c"), string(",")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::List(vec![
                Value::String("a".to_string()),
                Value::String("b".to_string()),
                Value::String("c".to_string()),
            ])
        );
    }

    #[test]
    fn builtin_string_join() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![string("a"), string("b"), string("c")], sp());
        let expr = dotted_call("string", "join", vec![list_expr, string(", ")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("a, b, c".to_string())
        );
    }

    #[test]
    fn builtin_string_starts_with() {
        let mut interp = Interpreter::new();
        let expr = dotted_call(
            "string",
            "starts_with",
            vec![string("hello world"), string("hello")],
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn builtin_string_ends_with() {
        let mut interp = Interpreter::new();
        let expr = dotted_call(
            "string",
            "ends_with",
            vec![string("hello world"), string("world")],
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn builtin_string_from_int64() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("string", "from_int64", vec![int(42)]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("42".to_string())
        );
    }

    #[test]
    fn builtin_string_from_uint64() {
        let mut interp = Interpreter::new();
        assert_eq!(
            interp
                .call_builtin("string.from_uint64", &[Value::Uint64(u64::MAX)])
                .expect("string.from_uint64 should be a builtin")
                .expect("string.from_uint64 should succeed"),
            Value::String("18446744073709551615".to_string())
        );
    }

    #[test]
    fn builtin_secret_redact() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("secret", "redact", vec![string("top-secret")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::String("***".to_string())
        );
    }

    #[test]
    fn builtin_secret_compare() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("secret", "compare", vec![string("a"), string("a")]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn builtin_list_length() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![int(1), int(2), int(3)], sp());
        let expr = dotted_call("list", "length", vec![list_expr]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(3));
    }

    #[test]
    fn builtin_list_append() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![int(1), int(2)], sp());
        let expr = dotted_call("list", "append", vec![list_expr, int(3)]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)])
        );
    }

    #[test]
    fn builtin_list_get() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![int(10), int(20), int(30)], sp());
        let expr = dotted_call("list", "get", vec![list_expr, int(1)]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::OptionalSome(Box::new(Value::Int64(20)))
        );
    }

    #[test]
    fn builtin_list_get_out_of_bounds() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![int(10)], sp());
        let expr = dotted_call("list", "get", vec![list_expr, int(5)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::OptionalNone);
    }

    #[test]
    fn builtin_list_first() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![int(10), int(20)], sp());
        let expr = dotted_call("list", "first", vec![list_expr]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::OptionalSome(Box::new(Value::Int64(10)))
        );
    }

    #[test]
    fn builtin_list_last() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![int(10), int(20)], sp());
        let expr = dotted_call("list", "last", vec![list_expr]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::OptionalSome(Box::new(Value::Int64(20)))
        );
    }

    #[test]
    fn builtin_list_new() {
        let mut interp = Interpreter::new();
        let expr = Expr::GenericCall(
            Box::new(Expr::FieldAccess(Box::new(var("list")), ident("new"), sp())),
            vec![TypeExpr::Named(ident("int64"))],
            vec![],
            sp(),
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::List(vec![]));
    }

    #[test]
    fn builtin_list_is_empty_true() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![], sp());
        let expr = dotted_call("list", "is_empty", vec![list_expr]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn builtin_list_is_empty_false() {
        let mut interp = Interpreter::new();
        let list_expr = Expr::ListConstruct(vec![int(1)], sp());
        let expr = dotted_call("list", "is_empty", vec![list_expr]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Bool(false));
    }

    #[test]
    fn builtin_math_abs_positive() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("math", "abs", vec![int(5)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(5));
    }

    #[test]
    fn builtin_math_abs_negative() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("math", "abs", vec![int(-7)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(7));
    }

    #[test]
    fn builtin_math_abs_float() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("math", "abs", vec![float(-3.5)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Float64(3.5));
    }

    #[test]
    fn builtin_math_min_int() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("math", "min", vec![int(3), int(7)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(3));
    }

    #[test]
    fn builtin_math_max_int() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("math", "max", vec![int(3), int(7)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(7));
    }

    #[test]
    fn builtin_math_min_float() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("math", "min", vec![float(1.5), float(2.5)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Float64(1.5));
    }

    #[test]
    fn builtin_math_max_float() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("math", "max", vec![float(1.5), float(2.5)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Float64(2.5));
    }

    #[test]
    fn builtin_int64_from_string() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("int64", "from_string", vec![string("123")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultOk(Box::new(Value::Int64(123)))
        );
    }

    #[test]
    fn builtin_int64_from_string_error() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("int64", "from_string", vec![string("abc")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultFail(Box::new(Value::String(
                "int64.from_string: cannot parse 'abc' as int64".to_string(),
            )))
        );
    }

    #[test]
    fn builtin_uint64_from_string_max() {
        let mut interp = Interpreter::new();
        let expr = dotted_call(
            "uint64",
            "from_string",
            vec![string("18446744073709551615")],
        );
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultOk(Box::new(Value::Uint64(u64::MAX)))
        );
    }

    #[test]
    fn builtin_uint64_from_string_error() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("uint64", "from_string", vec![string("-1")]);
        assert_eq!(
            interp.eval_expr(&expr).unwrap(),
            Value::ResultFail(Box::new(Value::String(
                "uint64.from_string: cannot parse '-1' as uint64".to_string(),
            )))
        );
    }

    #[test]
    fn handle_result_uses_default_value() {
        let mut interp = Interpreter::new();
        let expr = Expr::Handle(
            Box::new(dotted_call("int64", "from_string", vec![string("abc")])),
            Some(ident("error")),
            block(vec![default_stmt(int(7))]),
            sp(),
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(7));
    }

    #[test]
    fn handle_optional_uses_default_value() {
        let mut interp = Interpreter::new();
        let expr = Expr::Handle(
            Box::new(Expr::None(sp())),
            None,
            block(vec![default_stmt(int(5))]),
            sp(),
        );
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Int64(5));
    }

    #[test]
    fn handle_return_exits_enclosing_function() {
        let mut interp = Interpreter::new();
        let parse_fn = func_def(
            "parse_or_default",
            block(vec![
                Stmt::VarDecl(VarDecl {
                    mutable: false,
                    ty: TypeExpr::Named(ident("int64")),
                    name: ident("parsed"),
                    value: Expr::Handle(
                        Box::new(dotted_call("int64", "from_string", vec![string("abc")])),
                        Some(ident("error")),
                        block(vec![return_stmt(int(9))]),
                        sp(),
                    ),
                    span: sp(),
                }),
                return_stmt(var("parsed")),
            ]),
        );
        interp.register_function(&parse_fn);

        let result = interp.call_function("parse_or_default", vec![]).unwrap();
        assert_eq!(result, Value::Int64(9));
    }

    #[test]
    fn builtin_float64_from_int64() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("float64", "from_int64", vec![int(42)]);
        assert_eq!(interp.eval_expr(&expr).unwrap(), Value::Float64(42.0));
    }

    #[test]
    fn builtin_wrong_arg_count() {
        let mut interp = Interpreter::new();
        let expr = dotted_call("string", "trim", vec![string("a"), string("b")]);
        assert!(interp.eval_expr(&expr).is_err());
    }

    #[test]
    fn pipeline_string_trim_and_upper() {
        // "  hello  " into string.trim into string.upper => "HELLO"
        let mut interp = Interpreter::new();
        let initial = string("  hello  ");
        let steps = vec![
            PipelineStep {
                function: Expr::FieldAccess(Box::new(var("string")), ident("trim"), sp()),
                extra_args: vec![],
                handle: None,
                span: sp(),
            },
            PipelineStep {
                function: Expr::FieldAccess(Box::new(var("string")), ident("upper"), sp()),
                extra_args: vec![],
                handle: None,
                span: sp(),
            },
        ];
        let expr = Expr::Pipeline(Box::new(initial), steps, sp());
        let result = interp.eval_expr(&expr).unwrap();
        assert_eq!(result, Value::String("HELLO".to_string()));
    }

    #[test]
    fn pipeline_string_replace_with_extra_args() {
        // "hello world" into string.replace("world", "jett") => "hello jett"
        let mut interp = Interpreter::new();
        let initial = string("hello world");
        let steps = vec![PipelineStep {
            function: Expr::FieldAccess(Box::new(var("string")), ident("replace"), sp()),
            extra_args: vec![
                CallArg {
                    name: None,
                    value: string("world"),
                    span: sp(),
                },
                CallArg {
                    name: None,
                    value: string("jett"),
                    span: sp(),
                },
            ],
            handle: None,
            span: sp(),
        }];
        let expr = Expr::Pipeline(Box::new(initial), steps, sp());
        let result = interp.eval_expr(&expr).unwrap();
        assert_eq!(result, Value::String("hello jett".to_string()));
    }

    #[test]
    fn test_sha256_known_vectors() {
        // NIST test vectors
        assert_eq!(
            sha256_hash(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "sha256 of empty string"
        );
        assert_eq!(
            sha256_hash(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "sha256 of 'abc'"
        );
    }

    #[test]
    fn test_md5_known_vectors() {
        assert_eq!(md5_hash(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hash(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    }
}
