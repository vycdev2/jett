use jett_common::{FileId, Span};
use jett_diagnostics::Diagnostic;
use jett_parser::ast::{
    BitfieldDef, BitfieldFieldKind, EnumDef, FieldDef, FunctionDef, GivenDecl, Item, Module,
    PropertyBlock, StructDef, TypeAlias, TypeExpr, VerifyBlock,
};
use jett_types::ReflectionMetadata;
use std::collections::HashMap;
use std::sync::Arc;

use crate::interpreter::Interpreter;
use crate::value::Value;

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
    let item_file = item_file(item);
    if current_file.is_some_and(|file| file != item_file) {
        *current_namespace = None;
    }
    *current_file = Some(item_file);

    if let Item::Namespace(ns) = item {
        *current_namespace = Some(ns.name.name.clone());
    }
}

// ---------------------------------------------------------------------------
// Default iteration count for property-based testing
// ---------------------------------------------------------------------------

const PROPERTY_DEFAULT_ITERATIONS: usize = 100;
const SHRINK_MAX_STEPS: usize = 50;
const VERIFY_STACK_SIZE: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
struct PropertyEnumDef {
    type_name: String,
    namespace: Option<String>,
    def: EnumDef,
}

#[derive(Debug, Clone)]
struct PropertyStructDef {
    type_name: String,
    namespace: Option<String>,
    def: StructDef,
}

#[derive(Debug, Clone)]
struct PropertyBitfieldDef {
    type_name: String,
    namespace: Option<String>,
    def: BitfieldDef,
}

#[derive(Debug, Clone)]
struct PropertyTypeAliasDef {
    type_name: String,
    namespace: Option<String>,
    def: TypeAlias,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// An error produced during compile-time evaluation.
#[derive(Debug, Clone)]
pub struct ComptimeError {
    pub message: String,
    pub span: Span,
}

impl ComptimeError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

// ---------------------------------------------------------------------------
// Verify result
// ---------------------------------------------------------------------------

/// Outcome of running a single verify or property block.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub name: String,
    pub span: Span,
    pub passed: bool,
    pub error: Option<String>,
    /// If this was a property block, how many iterations were run.
    pub iterations: Option<usize>,
    /// If this was a property block, whether it is a property (true) or verify (false).
    pub is_property: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run all verify blocks in the module and return any diagnostics produced by
/// assertion failures.
///
/// This looks for:
/// 1. `Item::Verify` blocks (the primary mechanism).
/// 2. Legacy fallback: top-level zero-argument functions whose bodies contain
///    `assert` statements (kept for backward compatibility).
pub fn run_verify_blocks(module: &Module) -> Vec<Diagnostic> {
    let results = run_verify_blocks_detailed(module);
    verify_results_to_diagnostics(results)
}

/// Run all verify blocks with checked reflection metadata from type checking.
pub fn run_verify_blocks_with_metadata(
    module: &Module,
    metadata: Arc<ReflectionMetadata>,
) -> Vec<Diagnostic> {
    let results = run_verify_blocks_detailed_with_metadata(module, Some(metadata));
    verify_results_to_diagnostics(results)
}

/// Run all verify blocks with checked metadata and expression type facts from
/// type checking.
pub fn run_verify_blocks_with_metadata_and_expression_types(
    module: &Module,
    metadata: Arc<ReflectionMetadata>,
    expression_types: Arc<HashMap<Span, String>>,
) -> Vec<Diagnostic> {
    let results = run_verify_blocks_detailed_with_metadata_and_expression_types(
        module,
        Some(metadata),
        Some(expression_types),
    );
    verify_results_to_diagnostics(results)
}

fn verify_results_to_diagnostics(results: Vec<VerifyResult>) -> Vec<Diagnostic> {
    results
        .into_iter()
        .filter_map(|r| {
            if r.passed {
                None
            } else {
                Some(Diagnostic::error(
                    9000,
                    format!(
                        "comptime verify failed in '{}': {}",
                        r.name,
                        r.error.unwrap_or_default()
                    ),
                    r.span,
                ))
            }
        })
        .collect()
}

/// Run all verify blocks and return structured results.  Used by the
/// `jett test` command for per-block reporting.
pub fn run_verify_blocks_detailed(module: &Module) -> Vec<VerifyResult> {
    run_verify_blocks_detailed_with_metadata(module, None)
}

pub fn run_verify_blocks_detailed_with_metadata(
    module: &Module,
    metadata: Option<Arc<ReflectionMetadata>>,
) -> Vec<VerifyResult> {
    run_verify_blocks_detailed_with_metadata_and_expression_types(module, metadata, None)
}

pub fn run_verify_blocks_detailed_with_metadata_and_expression_types(
    module: &Module,
    metadata: Option<Arc<ReflectionMetadata>>,
    expression_types: Option<Arc<HashMap<Span, String>>>,
) -> Vec<VerifyResult> {
    let module_for_thread = module.clone();
    let thread_metadata = metadata.clone();
    let thread_expression_types = expression_types.clone();
    match std::thread::Builder::new()
        .name("jett-verify".to_string())
        .stack_size(VERIFY_STACK_SIZE)
        .spawn(move || {
            run_verify_blocks_detailed_inner(
                &module_for_thread,
                thread_metadata,
                thread_expression_types,
            )
        }) {
        Ok(handle) => match handle.join() {
            Ok(results) => results,
            Err(payload) => std::panic::resume_unwind(payload),
        },
        Err(_) => run_verify_blocks_detailed_inner(module, metadata, expression_types),
    }
}

fn run_verify_blocks_detailed_inner(
    module: &Module,
    metadata: Option<Arc<ReflectionMetadata>>,
    expression_types: Option<Arc<HashMap<Span, String>>>,
) -> Vec<VerifyResult> {
    let mut interp = Interpreter::new();
    if let Some(metadata) = metadata {
        interp.set_reflection_metadata(metadata);
    }
    if let Some(expression_types) = expression_types {
        interp.set_checked_expression_types(expression_types);
    }
    let mut results = Vec::new();

    // First pass: register all functions and type aliases so verify blocks
    // can call them and use refinement types.
    interp.register_module(module);
    let mut legacy_verify_functions: Vec<(Option<String>, FunctionDef)> = Vec::new();
    let mut verify_blocks: Vec<(Option<String>, VerifyBlock)> = Vec::new();
    let mut property_blocks: Vec<(Option<String>, PropertyBlock)> = Vec::new();
    let mut property_enums: Vec<PropertyEnumDef> = Vec::new();
    let mut property_structs: Vec<PropertyStructDef> = Vec::new();
    let mut property_bitfields: Vec<PropertyBitfieldDef> = Vec::new();
    let mut property_type_aliases: Vec<PropertyTypeAliasDef> = Vec::new();
    let mut current_file = None;
    let mut current_namespace = None;
    for item in &module.items {
        update_current_namespace(item, &mut current_file, &mut current_namespace);
        match item {
            Item::Struct(strukt) => {
                let type_name = current_namespace
                    .as_ref()
                    .map(|namespace| format!("{namespace}.{}", strukt.name.name))
                    .unwrap_or_else(|| strukt.name.name.clone());
                property_structs.push(PropertyStructDef {
                    type_name,
                    namespace: current_namespace.clone(),
                    def: strukt.clone(),
                });
            }
            Item::Bitfield(bitfield) => {
                let type_name = current_namespace
                    .as_ref()
                    .map(|namespace| format!("{namespace}.{}", bitfield.name.name))
                    .unwrap_or_else(|| bitfield.name.name.clone());
                property_bitfields.push(PropertyBitfieldDef {
                    type_name,
                    namespace: current_namespace.clone(),
                    def: bitfield.clone(),
                });
            }
            Item::Enum(enm) => {
                let type_name = current_namespace
                    .as_ref()
                    .map(|namespace| format!("{namespace}.{}", enm.name.name))
                    .unwrap_or_else(|| enm.name.name.clone());
                property_enums.push(PropertyEnumDef {
                    type_name,
                    namespace: current_namespace.clone(),
                    def: enm.clone(),
                });
            }
            Item::TypeAlias(alias) => {
                let type_name = current_namespace
                    .as_ref()
                    .map(|namespace| format!("{namespace}.{}", alias.name.name))
                    .unwrap_or_else(|| alias.name.name.clone());
                property_type_aliases.push(PropertyTypeAliasDef {
                    type_name,
                    namespace: current_namespace.clone(),
                    def: alias.clone(),
                });
            }
            Item::Function(func) => {
                if has_assert_stmts(func) && func.params.is_empty() && func.name.name != "main" {
                    legacy_verify_functions.push((current_namespace.clone(), func.clone()));
                }
            }
            Item::Verify(vb) => {
                verify_blocks.push((current_namespace.clone(), vb.clone()));
            }
            Item::Property(pb) => {
                property_blocks.push((current_namespace.clone(), pb.clone()));
            }
            _ => {}
        }
    }

    // Execute proper verify blocks.
    for (namespace, vb) in &verify_blocks {
        match interp.exec_block_in_namespace(namespace.as_deref(), &vb.body) {
            Ok(_) => {
                results.push(VerifyResult {
                    name: vb.name.name.clone(),
                    span: vb.name.span,
                    passed: true,
                    error: None,
                    iterations: None,
                    is_property: false,
                });
            }
            Err(msg) => {
                results.push(VerifyResult {
                    name: vb.name.name.clone(),
                    span: vb.name.span,
                    passed: false,
                    error: Some(msg),
                    iterations: None,
                    is_property: false,
                });
            }
        }
    }

    // Execute legacy verify functions (zero-arg functions with asserts).
    for (namespace, func) in &legacy_verify_functions {
        match interp.call_function_in_namespace(namespace.as_deref(), &func.name.name, vec![]) {
            Ok(_) => {
                results.push(VerifyResult {
                    name: func.name.name.clone(),
                    span: func.name.span,
                    passed: true,
                    error: None,
                    iterations: None,
                    is_property: false,
                });
            }
            Err(msg) => {
                results.push(VerifyResult {
                    name: func.name.name.clone(),
                    span: func.name.span,
                    passed: false,
                    error: Some(msg),
                    iterations: None,
                    is_property: false,
                });
            }
        }
    }

    // Execute property blocks.
    for (namespace, pb) in &property_blocks {
        let result = run_property_block(
            &mut interp,
            namespace.as_deref(),
            pb,
            &property_enums,
            &property_structs,
            &property_bitfields,
            &property_type_aliases,
        );
        results.push(result);
    }

    results
}

/// Evaluate a single pure function at compile time with the given arguments.
pub fn eval_function(func: &FunctionDef, args: Vec<Value>) -> Result<Value, ComptimeError> {
    let mut interp = Interpreter::new();
    interp.register_function(func);
    interp
        .call_function(&func.name.name, args)
        .map_err(|msg| ComptimeError::new(msg, func.span))
}

/// Evaluate an assert condition in the given environment.
///
/// Returns `Ok(())` if the assertion passes, or a `ComptimeError` if it
/// fails or evaluates to a non-boolean value.
pub fn eval_assert(
    condition: &jett_parser::ast::Expr,
    interp: &mut Interpreter,
) -> Result<(), ComptimeError> {
    let span = condition.span();
    match interp.eval_expr(condition) {
        Ok(Value::Bool(true)) => Ok(()),
        Ok(Value::Bool(false)) => Err(ComptimeError::new("assertion failed", span)),
        Ok(other) => Err(ComptimeError::new(
            format!("assert condition must be boolean, got {other}"),
            span,
        )),
        Err(msg) => Err(ComptimeError::new(msg, span)),
    }
}

// ---------------------------------------------------------------------------
// Property-based testing
// ---------------------------------------------------------------------------

/// Run a single property block for `PROPERTY_DEFAULT_ITERATIONS` iterations.
/// Each iteration generates random values for each `given` parameter, binds
/// them into the interpreter, and executes the body.
/// Generate candidate shrunk versions of a value (ordered from simplest to most complex).
fn shrink_value(value: &Value) -> Vec<Value> {
    match value {
        Value::Int64(n) => {
            let mut candidates = Vec::new();
            if *n != 0 {
                candidates.push(Value::Int64(0));
            }
            if *n > 1 {
                candidates.push(Value::Int64(n / 2));
                candidates.push(Value::Int64(n - 1));
            }
            if *n < -1 {
                candidates.push(Value::Int64(n / 2));
                candidates.push(Value::Int64(n + 1));
                candidates.push(Value::Int64(-n)); // try positive version
            }
            if *n == -1 {
                candidates.push(Value::Int64(1));
            }
            candidates
        }
        Value::Uint64(n) => {
            let mut candidates = Vec::new();
            if *n != 0 {
                candidates.push(Value::Uint64(0));
            }
            if *n > 1 {
                candidates.push(Value::Uint64(n / 2));
                candidates.push(Value::Uint64(n - 1));
            }
            candidates
        }
        Value::Float64(f) => {
            let mut candidates = Vec::new();
            if *f != 0.0 {
                candidates.push(Value::Float64(0.0));
            }
            if f.abs() > 1.0 {
                candidates.push(Value::Float64(f / 2.0));
                candidates.push(Value::Float64(f.floor()));
            }
            if *f < 0.0 {
                candidates.push(Value::Float64(-f)); // try positive version
            }
            candidates
        }
        Value::String(s) if !s.is_empty() => {
            let mut candidates = Vec::new();
            candidates.push(Value::String(String::new()));
            let chars: Vec<char> = s.chars().collect();
            if chars.len() > 1 {
                // Remove first half
                candidates.push(Value::String(chars[chars.len() / 2..].iter().collect()));
                // Remove second half
                candidates.push(Value::String(chars[..chars.len() / 2].iter().collect()));
                // Remove last character
                candidates.push(Value::String(chars[..chars.len() - 1].iter().collect()));
            }
            candidates
        }
        Value::Bytes(bytes) if !bytes.is_empty() => {
            let mut candidates = Vec::new();
            candidates.push(Value::Bytes(Vec::new()));
            if bytes.len() > 1 {
                candidates.push(Value::Bytes(bytes[bytes.len() / 2..].to_vec()));
                candidates.push(Value::Bytes(bytes[..bytes.len() / 2].to_vec()));
                candidates.push(Value::Bytes(bytes[..bytes.len() - 1].to_vec()));
            }
            candidates
        }
        Value::OptionalSome(value) => {
            let mut candidates = vec![Value::OptionalNone];
            for shrunk_value in shrink_value(value) {
                candidates.push(Value::OptionalSome(Box::new(shrunk_value)));
            }
            candidates
        }
        Value::ResultOk(value) => shrink_value(value)
            .into_iter()
            .map(|value| Value::ResultOk(Box::new(value)))
            .collect(),
        Value::ResultFail(error) => shrink_value(error)
            .into_iter()
            .map(|error| Value::ResultFail(Box::new(error)))
            .collect(),
        Value::List(items) if !items.is_empty() => {
            let mut candidates = Vec::new();
            candidates.push(Value::List(vec![]));
            if items.len() > 1 {
                // First half
                candidates.push(Value::List(items[..items.len() / 2].to_vec()));
                // Second half
                candidates.push(Value::List(items[items.len() / 2..].to_vec()));
                // Remove last element
                candidates.push(Value::List(items[..items.len() - 1].to_vec()));
            }
            // Try shrinking individual elements
            for (i, item) in items.iter().enumerate() {
                for shrunk_item in shrink_value(item) {
                    let mut new_list = items.clone();
                    new_list[i] = shrunk_item;
                    candidates.push(Value::List(new_list));
                }
            }
            candidates
        }
        Value::Set(items) if !items.is_empty() => {
            let mut candidates = Vec::new();
            candidates.push(Value::Set(vec![]));
            if items.len() > 1 {
                candidates.push(Value::Set(items[..items.len() / 2].to_vec()));
                candidates.push(Value::Set(items[items.len() / 2..].to_vec()));
                candidates.push(Value::Set(items[..items.len() - 1].to_vec()));
            }
            for (i, item) in items.iter().enumerate() {
                for shrunk_item in shrink_value(item) {
                    let mut new_set = items.clone();
                    new_set[i] = shrunk_item;
                    candidates.push(Value::Set(unique_values(new_set)));
                }
            }
            candidates
        }
        Value::Map(entries) if !entries.is_empty() => {
            let mut candidates = Vec::new();
            candidates.push(Value::Map(vec![]));
            if entries.len() > 1 {
                candidates.push(Value::Map(entries[..entries.len() / 2].to_vec()));
                candidates.push(Value::Map(entries[entries.len() / 2..].to_vec()));
                candidates.push(Value::Map(entries[..entries.len() - 1].to_vec()));
            }
            for (i, (key, value)) in entries.iter().enumerate() {
                for shrunk_key in shrink_value(key) {
                    if entries
                        .iter()
                        .enumerate()
                        .any(|(other_index, (other_key, _))| {
                            other_index != i && *other_key == shrunk_key
                        })
                    {
                        continue;
                    }
                    let mut new_map = entries.clone();
                    new_map[i].0 = shrunk_key;
                    candidates.push(Value::Map(new_map));
                }
                for shrunk_value in shrink_value(value) {
                    let mut new_map = entries.clone();
                    new_map[i].1 = shrunk_value;
                    candidates.push(Value::Map(new_map));
                }
            }
            candidates
        }
        Value::Enum {
            type_name,
            variant,
            fields,
        } if !fields.is_empty() => {
            let mut candidates = Vec::new();
            for (i, field) in fields.iter().enumerate() {
                for shrunk_field in shrink_value(field) {
                    let mut new_fields = fields.clone();
                    new_fields[i] = shrunk_field;
                    candidates.push(Value::Enum {
                        type_name: type_name.clone(),
                        variant: variant.clone(),
                        fields: new_fields,
                    });
                }
            }
            candidates
        }
        Value::Struct { type_name, fields } if !fields.is_empty() => {
            let mut candidates = Vec::new();
            for (i, (_, field)) in fields.iter().enumerate() {
                for shrunk_field in shrink_value(field) {
                    let mut new_fields = fields.clone();
                    new_fields[i].1 = shrunk_field;
                    candidates.push(Value::Struct {
                        type_name: type_name.clone(),
                        fields: new_fields,
                    });
                }
            }
            candidates
        }
        _ => vec![], // Bool, Nothing, etc. cannot be shrunk further
    }
}

fn unique_values(values: Vec<Value>) -> Vec<Value> {
    let mut unique = Vec::new();
    for value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique
}

/// Try to find simpler inputs that still cause the property to fail.
/// Returns the shrunk inputs as a Vec<Value> in the same order as `failing`.
fn shrink_inputs(
    interp: &mut Interpreter,
    namespace: Option<&str>,
    pb: &PropertyBlock,
    failing: Vec<Value>,
) -> Vec<Value> {
    let mut current = failing;

    'outer: for _ in 0..SHRINK_MAX_STEPS {
        // Try to shrink each input one at a time.
        for i in 0..current.len() {
            let candidates = shrink_value(&current[i]);
            for candidate in candidates {
                let mut attempt = current.clone();
                attempt[i] = candidate;

                // Run the property with the candidate inputs.
                interp.push_scope_public();
                for (given, value) in pb.givens.iter().zip(attempt.iter()) {
                    interp.set_variable_public(&given.name.name, value.clone());
                }
                let result = interp.exec_block_in_namespace(namespace, &pb.body);
                interp.pop_scope_public();

                if result.is_err() {
                    // Still fails — use the simpler version.
                    current = attempt;
                    continue 'outer;
                }
            }
        }
        // No further shrinking possible.
        break;
    }

    current
}

fn run_property_block(
    interp: &mut Interpreter,
    namespace: Option<&str>,
    pb: &PropertyBlock,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> VerifyResult {
    let iterations = PROPERTY_DEFAULT_ITERATIONS;

    // Pre-compute the value pools for each given declaration.
    let pools: Vec<Vec<Value>> = pb
        .givens
        .iter()
        .map(|g| {
            generate_values_for_type_in_namespace(
                interp,
                &g.ty,
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            )
        })
        .collect();

    // If any pool is empty, we cannot test.
    for (i, pool) in pools.iter().enumerate() {
        if pool.is_empty() {
            return VerifyResult {
                name: pb.name.name.clone(),
                span: pb.name.span,
                passed: false,
                error: Some(format!(
                    "unsupported type for property given '{}': cannot generate values",
                    pb.givens[i].name.name,
                )),
                iterations: Some(0),
                is_property: true,
            };
        }
    }

    for iteration in 0..iterations {
        // Pick values for this iteration: cycle through the pool.
        let chosen: Vec<(&GivenDecl, Value)> = pb
            .givens
            .iter()
            .zip(pools.iter())
            .map(|(given, pool)| {
                let idx = iteration % pool.len();
                (given, pool[idx].clone())
            })
            .collect();

        // Push a scope, bind the given values, execute the body.
        interp.push_scope_public();
        for (given, value) in &chosen {
            interp.set_variable_public(&given.name.name, value.clone());
        }

        let exec_result = interp.exec_block_in_namespace(namespace, &pb.body);
        interp.pop_scope_public();

        if let Err(msg) = exec_result {
            // Shrink the failing inputs to find a simpler counterexample.
            let failing_values: Vec<Value> = chosen.iter().map(|(_, v)| v.clone()).collect();
            let shrunk = shrink_inputs(interp, namespace, pb, failing_values);

            let input_desc: Vec<String> = pb
                .givens
                .iter()
                .zip(shrunk.iter())
                .map(|(given, value)| format!("{} = {}", given.name.name, value))
                .collect();
            return VerifyResult {
                name: pb.name.name.clone(),
                span: pb.name.span,
                passed: false,
                error: Some(format!(
                    "{} (counterexample: {})",
                    msg,
                    input_desc.join(", ")
                )),
                iterations: Some(iteration + 1),
                is_property: true,
            };
        }
    }

    VerifyResult {
        name: pb.name.name.clone(),
        span: pb.name.span,
        passed: true,
        error: None,
        iterations: Some(iterations),
        is_property: true,
    }
}

/// Generate a pool of test values for a given type expression.
#[cfg(test)]
fn generate_values_for_type(ty: &TypeExpr) -> Vec<Value> {
    let mut interp = Interpreter::new();
    generate_values_for_type_in_namespace(&mut interp, ty, None, &[], &[], &[], &[])
}

fn generate_signed_integer_values(min: i64, max: i64) -> Vec<Value> {
    unique_values(
        [0, 1, -1, 42, -42, 100, max, min]
            .into_iter()
            .filter(|value| *value >= min && *value <= max)
            .map(Value::Int64)
            .collect(),
    )
}

fn generate_unsigned_integer_values(max: i64) -> Vec<Value> {
    unique_values(
        [0, 1, 42, 100, max]
            .into_iter()
            .filter(|value| *value >= 0 && *value <= max)
            .map(Value::Int64)
            .collect(),
    )
}

fn generate_uint64_values() -> Vec<Value> {
    unique_values(vec![
        Value::Uint64(0),
        Value::Uint64(1),
        Value::Uint64(42),
        Value::Uint64(100),
        Value::Uint64(i64::MAX as u64),
        Value::Uint64(i64::MAX as u64 + 1),
        Value::Uint64(u64::MAX),
    ])
}

fn generate_float_values() -> Vec<Value> {
    vec![
        Value::Float64(0.0),
        Value::Float64(1.0),
        Value::Float64(-1.0),
        Value::Float64(std::f64::consts::PI),
        Value::Float64(-0.0),
    ]
}

fn generate_values_for_type_in_namespace(
    interp: &mut Interpreter,
    ty: &TypeExpr,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    match ty {
        TypeExpr::Named(ident) => match ident.name.as_str() {
            "int8" => generate_signed_integer_values(i8::MIN as i64, i8::MAX as i64),
            "int16" => generate_signed_integer_values(i16::MIN as i64, i16::MAX as i64),
            "int32" => generate_signed_integer_values(i32::MIN as i64, i32::MAX as i64),
            "int64" => generate_signed_integer_values(i64::MIN, i64::MAX),
            "uint8" => generate_unsigned_integer_values(u8::MAX as i64),
            "uint16" => generate_unsigned_integer_values(u16::MAX as i64),
            "uint32" => generate_unsigned_integer_values(u32::MAX as i64),
            "uint64" => generate_uint64_values(),
            "string" => vec![
                Value::String(String::new()),
                Value::String("a".to_string()),
                Value::String("hello".to_string()),
                Value::String("hello world".to_string()),
                Value::String("123".to_string()),
            ],
            "bool" => vec![Value::Bool(true), Value::Bool(false)],
            "float32" | "float64" => generate_float_values(),
            "bytes" => vec![
                Value::Bytes(Vec::new()),
                Value::Bytes(vec![0]),
                Value::Bytes(b"hello".to_vec()),
                Value::Bytes(vec![0, 1, 2, 127, 255]),
            ],
            "nothing" => vec![Value::Nothing],
            _ => {
                if let Some(alias_values) = generate_type_alias_values_for_type_name(
                    interp,
                    &ident.name,
                    namespace,
                    enum_defs,
                    struct_defs,
                    bitfield_defs,
                    type_alias_defs,
                ) {
                    return alias_values;
                }

                let enum_values = generate_enum_values_for_type_name(
                    interp,
                    &ident.name,
                    namespace,
                    enum_defs,
                    struct_defs,
                    bitfield_defs,
                    type_alias_defs,
                );
                if enum_values.is_empty() {
                    let struct_values = generate_struct_values_for_type_name(
                        interp,
                        &ident.name,
                        namespace,
                        enum_defs,
                        struct_defs,
                        bitfield_defs,
                        type_alias_defs,
                    );
                    if struct_values.is_empty() {
                        generate_bitfield_values_for_type_name(
                            interp,
                            &ident.name,
                            namespace,
                            enum_defs,
                            struct_defs,
                            bitfield_defs,
                            type_alias_defs,
                        )
                    } else {
                        struct_values
                    }
                } else {
                    enum_values
                }
            }
        },
        TypeExpr::Generic(ident, args, _) => match ident.name.as_str() {
            "list" if args.len() == 1 => generate_list_values_for_type(
                interp,
                &args[0],
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            ),
            "set" if args.len() == 1 => generate_set_values_for_type(
                interp,
                &args[0],
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            ),
            "map" if args.len() == 2 => generate_map_values_for_type(
                interp,
                &args[0],
                &args[1],
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            ),
            "optional" if args.len() == 1 => generate_optional_values_for_type(
                interp,
                &args[0],
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            ),
            "result" if args.len() == 2 => generate_result_values_for_type(
                interp,
                &args[0],
                &args[1],
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            ),
            _ => generate_generic_struct_values_for_type_name(
                interp,
                &ident.name,
                args,
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            ),
        },
        TypeExpr::View(inner, _) => generate_values_for_type_in_namespace(
            interp,
            inner,
            namespace,
            enum_defs,
            struct_defs,
            bitfield_defs,
            type_alias_defs,
        ),
        TypeExpr::StateQualified(inner, _, _) => generate_values_for_type_in_namespace(
            interp,
            inner,
            namespace,
            enum_defs,
            struct_defs,
            bitfield_defs,
            type_alias_defs,
        ),
        TypeExpr::Function(_, _, _) => vec![], // cannot generate function values
    }
}

fn generate_list_values_for_type(
    interp: &mut Interpreter,
    inner_ty: &TypeExpr,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let inner_values = generate_values_for_type_in_namespace(
        interp,
        inner_ty,
        namespace,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if inner_values.is_empty() {
        return Vec::new();
    }

    let mut values = vec![Value::List(vec![])];
    values.push(Value::List(vec![inner_values[0].clone()]));

    let sample: Vec<Value> = inner_values.iter().take(3).cloned().collect();
    if sample.len() > 1 {
        values.push(Value::List(sample));
    }

    if inner_values.len() > 1 {
        values.push(Value::List(vec![
            inner_values[1].clone(),
            inner_values[1].clone(),
            inner_values[1].clone(),
        ]));
    }

    values
}

fn generate_set_values_for_type(
    interp: &mut Interpreter,
    inner_ty: &TypeExpr,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let inner_values = generate_values_for_type_in_namespace(
        interp,
        inner_ty,
        namespace,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if inner_values.is_empty() {
        return Vec::new();
    }

    let mut unique_inner_values = Vec::new();
    for value in inner_values {
        if !unique_inner_values.contains(&value) {
            unique_inner_values.push(value);
        }
    }

    let mut values = vec![Value::Set(vec![])];
    values.push(Value::Set(vec![unique_inner_values[0].clone()]));

    let sample: Vec<Value> = unique_inner_values.iter().take(3).cloned().collect();
    if sample.len() > 1 {
        values.push(Value::Set(sample));
    }

    values
}

#[allow(clippy::too_many_arguments)]
fn generate_map_values_for_type(
    interp: &mut Interpreter,
    key_ty: &TypeExpr,
    value_ty: &TypeExpr,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let key_values = generate_values_for_type_in_namespace(
        interp,
        key_ty,
        namespace,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    let value_values = generate_values_for_type_in_namespace(
        interp,
        value_ty,
        namespace,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if key_values.is_empty() || value_values.is_empty() {
        return Vec::new();
    }

    let mut unique_keys = Vec::new();
    for key in key_values {
        if !unique_keys.contains(&key) {
            unique_keys.push(key);
        }
    }

    let mut maps = vec![Value::Map(vec![])];
    maps.push(Value::Map(vec![(
        unique_keys[0].clone(),
        value_values[0].clone(),
    )]));

    let mut sample = Vec::new();
    for (index, key) in unique_keys.iter().take(3).enumerate() {
        sample.push((
            key.clone(),
            value_values[index % value_values.len()].clone(),
        ));
    }
    if sample.len() > 1 {
        maps.push(Value::Map(sample));
    }

    maps
}

fn generate_optional_values_for_type(
    interp: &mut Interpreter,
    inner_ty: &TypeExpr,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let inner_values = generate_values_for_type_in_namespace(
        interp,
        inner_ty,
        namespace,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if inner_values.is_empty() {
        return Vec::new();
    }

    let mut values = vec![Value::OptionalNone];
    values.extend(
        inner_values
            .into_iter()
            .take(3)
            .map(|value| Value::OptionalSome(Box::new(value))),
    );
    values
}

#[allow(clippy::too_many_arguments)]
fn generate_result_values_for_type(
    interp: &mut Interpreter,
    ok_ty: &TypeExpr,
    error_ty: &TypeExpr,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let ok_values = generate_values_for_type_in_namespace(
        interp,
        ok_ty,
        namespace,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    let error_values = generate_values_for_type_in_namespace(
        interp,
        error_ty,
        namespace,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if ok_values.is_empty() || error_values.is_empty() {
        return Vec::new();
    }

    let mut values: Vec<Value> = ok_values
        .into_iter()
        .take(3)
        .map(|value| Value::ResultOk(Box::new(value)))
        .collect();
    values.extend(
        error_values
            .into_iter()
            .take(3)
            .map(|error| Value::ResultFail(Box::new(error))),
    );
    values
}

fn generate_type_alias_values_for_type_name(
    interp: &mut Interpreter,
    name: &str,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Option<Vec<Value>> {
    let alias_def = find_property_type_alias(name, namespace, type_alias_defs)?;
    Some(generate_type_alias_values(
        interp,
        alias_def,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    ))
}

fn find_property_type_alias<'a>(
    name: &str,
    namespace: Option<&str>,
    type_alias_defs: &'a [PropertyTypeAliasDef],
) -> Option<&'a PropertyTypeAliasDef> {
    if name.contains('.') {
        return type_alias_defs
            .iter()
            .find(|alias_def| alias_def.type_name == name);
    }

    if let Some(namespace) = namespace {
        let scoped_name = format!("{namespace}.{name}");
        if let Some(alias_def) = type_alias_defs
            .iter()
            .find(|alias_def| alias_def.type_name == scoped_name)
        {
            return Some(alias_def);
        }
    }

    type_alias_defs
        .iter()
        .find(|alias_def| alias_def.type_name == name)
}

fn generate_type_alias_values(
    interp: &mut Interpreter,
    alias_def: &PropertyTypeAliasDef,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let base_values = generate_values_for_type_in_namespace(
        interp,
        &alias_def.def.base_type,
        alias_def.namespace.as_deref(),
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if alias_def.def.constraint.is_none() {
        return base_values;
    }

    base_values
        .into_iter()
        .filter(|value| {
            interp
                .check_refinement_type(&alias_def.type_name, value)
                .is_ok()
        })
        .collect()
}

fn generate_enum_values_for_type_name(
    interp: &mut Interpreter,
    name: &str,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let Some(enum_def) = find_property_enum(name, namespace, enum_defs) else {
        return Vec::new();
    };

    generate_enum_values(
        interp,
        enum_def,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    )
}

fn find_property_enum<'a>(
    name: &str,
    namespace: Option<&str>,
    enum_defs: &'a [PropertyEnumDef],
) -> Option<&'a PropertyEnumDef> {
    if name.contains('.') {
        return enum_defs.iter().find(|enum_def| enum_def.type_name == name);
    }

    if let Some(namespace) = namespace {
        let scoped_name = format!("{namespace}.{name}");
        if let Some(enum_def) = enum_defs
            .iter()
            .find(|enum_def| enum_def.type_name == scoped_name)
        {
            return Some(enum_def);
        }
    }

    enum_defs.iter().find(|enum_def| enum_def.type_name == name)
}

fn generate_enum_values(
    interp: &mut Interpreter,
    enum_def: &PropertyEnumDef,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let field_namespace = enum_def.namespace.as_deref();
    let mut values = Vec::new();
    for variant in &enum_def.def.variants {
        if variant.fields.is_empty() {
            values.push(Value::Enum {
                type_name: enum_def.type_name.clone(),
                variant: variant.name.name.clone(),
                fields: vec![],
            });
            continue;
        }

        let field_pools = generate_enum_field_pools(
            interp,
            &variant.fields,
            field_namespace,
            enum_defs,
            struct_defs,
            bitfield_defs,
            type_alias_defs,
        );
        if field_pools.is_empty() {
            return Vec::new();
        }

        let sample_count = field_pools.iter().map(Vec::len).max().unwrap_or(0).min(3);
        for sample_index in 0..sample_count {
            let fields = field_pools
                .iter()
                .map(|pool| pool[sample_index % pool.len()].clone())
                .collect();
            values.push(Value::Enum {
                type_name: enum_def.type_name.clone(),
                variant: variant.name.name.clone(),
                fields,
            });
        }
    }
    values
}

fn generate_enum_field_pools(
    interp: &mut Interpreter,
    fields: &[FieldDef],
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Vec<Value>> {
    let mut pools = Vec::new();
    for field in fields {
        let values = generate_values_for_type_in_namespace(
            interp,
            &field.ty,
            namespace,
            enum_defs,
            struct_defs,
            bitfield_defs,
            type_alias_defs,
        );
        if values.is_empty() {
            return Vec::new();
        }
        pools.push(values);
    }
    pools
}

fn generate_struct_values_for_type_name(
    interp: &mut Interpreter,
    name: &str,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let Some(struct_def) = find_property_struct(name, namespace, struct_defs) else {
        return Vec::new();
    };

    generate_struct_values(
        interp,
        struct_def,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    )
}

fn find_property_struct<'a>(
    name: &str,
    namespace: Option<&str>,
    struct_defs: &'a [PropertyStructDef],
) -> Option<&'a PropertyStructDef> {
    if name.contains('.') {
        return struct_defs
            .iter()
            .find(|struct_def| struct_def.type_name == name);
    }

    if let Some(namespace) = namespace {
        let scoped_name = format!("{namespace}.{name}");
        if let Some(struct_def) = struct_defs
            .iter()
            .find(|struct_def| struct_def.type_name == scoped_name)
        {
            return Some(struct_def);
        }
    }

    struct_defs
        .iter()
        .find(|struct_def| struct_def.type_name == name)
}

fn generate_struct_values(
    interp: &mut Interpreter,
    struct_def: &PropertyStructDef,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    if !struct_def.def.type_params.is_empty() {
        return Vec::new();
    }

    if struct_def.def.fields.is_empty() {
        return vec![Value::Struct {
            type_name: struct_def.type_name.clone(),
            fields: vec![],
        }];
    }

    let field_pools = generate_struct_field_pools(
        interp,
        &struct_def.def.fields,
        struct_def.namespace.as_deref(),
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if field_pools.is_empty() {
        return Vec::new();
    }

    let sample_count = field_pools.iter().map(Vec::len).max().unwrap_or(0).min(3);
    let mut values = Vec::new();
    for sample_index in 0..sample_count {
        let fields = struct_def
            .def
            .fields
            .iter()
            .zip(field_pools.iter())
            .map(|(field, pool)| {
                (
                    field.name.name.clone(),
                    pool[sample_index % pool.len()].clone(),
                )
            })
            .collect();
        values.push(Value::Struct {
            type_name: struct_def.type_name.clone(),
            fields,
        });
    }
    values
}

#[allow(clippy::too_many_arguments)]
fn generate_generic_struct_values_for_type_name(
    interp: &mut Interpreter,
    name: &str,
    args: &[TypeExpr],
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let Some(struct_def) = find_property_struct(name, namespace, struct_defs) else {
        return Vec::new();
    };
    if struct_def.def.type_params.len() != args.len() {
        return Vec::new();
    }

    let substitutions: HashMap<String, TypeExpr> = struct_def
        .def
        .type_params
        .iter()
        .zip(args.iter())
        .map(|(param, arg)| {
            (
                param.name.clone(),
                qualify_property_type_expr_in_namespace(
                    arg,
                    namespace,
                    enum_defs,
                    struct_defs,
                    bitfield_defs,
                    type_alias_defs,
                ),
            )
        })
        .collect();

    generate_generic_struct_values(
        interp,
        struct_def,
        &substitutions,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    )
}

fn generate_generic_struct_values(
    interp: &mut Interpreter,
    struct_def: &PropertyStructDef,
    substitutions: &HashMap<String, TypeExpr>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let field_namespace = struct_def.namespace.as_deref();
    let substituted_fields: Vec<FieldDef> = struct_def
        .def
        .fields
        .iter()
        .map(|field| FieldDef {
            name: field.name.clone(),
            ty: substitute_property_type_params(&field.ty, substitutions),
            serialize_name: field.serialize_name.clone(),
            span: field.span,
        })
        .collect();

    if substituted_fields.is_empty() {
        return vec![Value::Struct {
            type_name: struct_def.type_name.clone(),
            fields: vec![],
        }];
    }

    let field_pools = generate_struct_field_pools(
        interp,
        &substituted_fields,
        field_namespace,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if field_pools.is_empty() {
        return Vec::new();
    }

    let sample_count = field_pools.iter().map(Vec::len).max().unwrap_or(0).min(3);
    let mut values = Vec::new();
    for sample_index in 0..sample_count {
        let fields = substituted_fields
            .iter()
            .zip(field_pools.iter())
            .map(|(field, pool)| {
                (
                    field.name.name.clone(),
                    pool[sample_index % pool.len()].clone(),
                )
            })
            .collect();
        values.push(Value::Struct {
            type_name: struct_def.type_name.clone(),
            fields,
        });
    }
    values
}

fn generate_struct_field_pools(
    interp: &mut Interpreter,
    fields: &[FieldDef],
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Vec<Value>> {
    let mut pools = Vec::new();
    for field in fields {
        let values = generate_values_for_type_in_namespace(
            interp,
            &field.ty,
            namespace,
            enum_defs,
            struct_defs,
            bitfield_defs,
            type_alias_defs,
        );
        if values.is_empty() {
            return Vec::new();
        }
        pools.push(values);
    }
    pools
}

fn generate_bitfield_values_for_type_name(
    interp: &mut Interpreter,
    name: &str,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    let Some(bitfield_def) = find_property_bitfield(name, namespace, bitfield_defs) else {
        return Vec::new();
    };

    generate_bitfield_values(
        interp,
        bitfield_def,
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    )
}

fn find_property_bitfield<'a>(
    name: &str,
    namespace: Option<&str>,
    bitfield_defs: &'a [PropertyBitfieldDef],
) -> Option<&'a PropertyBitfieldDef> {
    if name.contains('.') {
        return bitfield_defs
            .iter()
            .find(|bitfield_def| bitfield_def.type_name == name);
    }

    if let Some(namespace) = namespace {
        let scoped_name = format!("{namespace}.{name}");
        if let Some(bitfield_def) = bitfield_defs
            .iter()
            .find(|bitfield_def| bitfield_def.type_name == scoped_name)
        {
            return Some(bitfield_def);
        }
    }

    bitfield_defs
        .iter()
        .find(|bitfield_def| bitfield_def.type_name == name)
}

fn generate_bitfield_values(
    interp: &mut Interpreter,
    bitfield_def: &PropertyBitfieldDef,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Value> {
    if bitfield_def.def.fields.is_empty() {
        return vec![Value::Struct {
            type_name: bitfield_def.type_name.clone(),
            fields: vec![],
        }];
    }

    let field_pools = generate_bitfield_field_pools(
        interp,
        &bitfield_def.def.fields,
        bitfield_def.namespace.as_deref(),
        enum_defs,
        struct_defs,
        bitfield_defs,
        type_alias_defs,
    );
    if field_pools.is_empty() {
        return Vec::new();
    }

    let sample_count = field_pools.iter().map(Vec::len).max().unwrap_or(0).min(3);
    let mut values = Vec::new();
    for sample_index in 0..sample_count {
        let fields = bitfield_def
            .def
            .fields
            .iter()
            .zip(field_pools.iter())
            .map(|(field, pool)| {
                (
                    field.name.name.clone(),
                    pool[sample_index % pool.len()].clone(),
                )
            })
            .collect();
        values.push(Value::Struct {
            type_name: bitfield_def.type_name.clone(),
            fields,
        });
    }
    values
}

fn generate_bitfield_field_pools(
    interp: &mut Interpreter,
    fields: &[jett_parser::ast::BitfieldFieldDef],
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> Vec<Vec<Value>> {
    let mut pools = Vec::new();
    for field in fields {
        let values = match &field.kind {
            BitfieldFieldKind::Bits { width, as_type } => {
                if let Some(enum_ty) = as_type {
                    generate_bitfield_enum_values(enum_ty, *width, namespace, enum_defs)
                } else {
                    generate_bit_width_values(*width)
                }
            }
            BitfieldFieldKind::Payload(ty) => generate_values_for_type_in_namespace(
                interp,
                ty,
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            ),
        };
        if values.is_empty() {
            return Vec::new();
        }
        pools.push(values);
    }
    pools
}

fn generate_bit_width_values(width: u16) -> Vec<Value> {
    if width == 0 {
        return Vec::new();
    }

    let max_value = if width >= 63 {
        i64::MAX
    } else {
        (1_i64 << width) - 1
    };
    let candidates = [0, 1.min(max_value), 42.min(max_value), max_value];
    unique_values(
        candidates
            .into_iter()
            .map(Value::Int64)
            .collect::<Vec<Value>>(),
    )
}

fn generate_bitfield_enum_values(
    enum_ty: &TypeExpr,
    width: u16,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
) -> Vec<Value> {
    let Some(enum_name) = property_type_expr_named_type(enum_ty) else {
        return Vec::new();
    };
    let Some(enum_def) = find_property_enum(&enum_name, namespace, enum_defs) else {
        return Vec::new();
    };

    let mut values = Vec::new();
    for (variant, discriminant) in enum_def
        .def
        .variants
        .iter()
        .zip(property_enum_discriminants(&enum_def.def))
    {
        if variant.fields.is_empty()
            && discriminant >= 0
            && property_value_fits_in_bits(discriminant as u64, width)
        {
            values.push(Value::Enum {
                type_name: enum_def.type_name.clone(),
                variant: variant.name.name.clone(),
                fields: vec![],
            });
        }
    }
    values
}

fn property_type_expr_named_type(ty: &TypeExpr) -> Option<String> {
    match ty {
        TypeExpr::Named(ident) => Some(ident.name.clone()),
        TypeExpr::View(inner, _) => property_type_expr_named_type(inner),
        TypeExpr::StateQualified(inner, _, _) => property_type_expr_named_type(inner),
        TypeExpr::Generic(_, _, _) | TypeExpr::Function(_, _, _) => None,
    }
}

fn property_enum_discriminants(enum_def: &EnumDef) -> Vec<i64> {
    let mut next_discriminant = 0_i64;
    let mut discriminants = Vec::with_capacity(enum_def.variants.len());
    for variant in &enum_def.variants {
        let discriminant = variant.discriminant.unwrap_or(next_discriminant);
        next_discriminant = discriminant.saturating_add(1);
        discriminants.push(discriminant);
    }
    discriminants
}

fn property_value_fits_in_bits(value: u64, width: u16) -> bool {
    width >= 64 || value < (1_u64 << width)
}

fn substitute_property_type_params(
    ty: &TypeExpr,
    substitutions: &HashMap<String, TypeExpr>,
) -> TypeExpr {
    match ty {
        TypeExpr::Named(ident) => substitutions
            .get(&ident.name)
            .cloned()
            .unwrap_or_else(|| TypeExpr::Named(ident.clone())),
        TypeExpr::Generic(ident, args, span) => TypeExpr::Generic(
            ident.clone(),
            args.iter()
                .map(|arg| substitute_property_type_params(arg, substitutions))
                .collect(),
            *span,
        ),
        TypeExpr::View(inner, span) => TypeExpr::View(
            Box::new(substitute_property_type_params(inner, substitutions)),
            *span,
        ),
        TypeExpr::StateQualified(inner, state, span) => TypeExpr::StateQualified(
            Box::new(substitute_property_type_params(inner, substitutions)),
            state.clone(),
            *span,
        ),
        TypeExpr::Function(params, return_ty, span) => TypeExpr::Function(
            params
                .iter()
                .map(|param| substitute_property_type_params(param, substitutions))
                .collect(),
            Box::new(substitute_property_type_params(return_ty, substitutions)),
            *span,
        ),
    }
}

fn qualify_property_type_expr_in_namespace(
    ty: &TypeExpr,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> TypeExpr {
    match ty {
        TypeExpr::Named(ident) => TypeExpr::Named(qualify_property_type_ident_in_namespace(
            ident,
            namespace,
            enum_defs,
            struct_defs,
            bitfield_defs,
            type_alias_defs,
        )),
        TypeExpr::Generic(ident, args, span) => TypeExpr::Generic(
            qualify_property_type_ident_in_namespace(
                ident,
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            ),
            args.iter()
                .map(|arg| {
                    qualify_property_type_expr_in_namespace(
                        arg,
                        namespace,
                        enum_defs,
                        struct_defs,
                        bitfield_defs,
                        type_alias_defs,
                    )
                })
                .collect(),
            *span,
        ),
        TypeExpr::View(inner, span) => TypeExpr::View(
            Box::new(qualify_property_type_expr_in_namespace(
                inner,
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            )),
            *span,
        ),
        TypeExpr::StateQualified(inner, state, span) => TypeExpr::StateQualified(
            Box::new(qualify_property_type_expr_in_namespace(
                inner,
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            )),
            state.clone(),
            *span,
        ),
        TypeExpr::Function(params, return_ty, span) => TypeExpr::Function(
            params
                .iter()
                .map(|param| {
                    qualify_property_type_expr_in_namespace(
                        param,
                        namespace,
                        enum_defs,
                        struct_defs,
                        bitfield_defs,
                        type_alias_defs,
                    )
                })
                .collect(),
            Box::new(qualify_property_type_expr_in_namespace(
                return_ty,
                namespace,
                enum_defs,
                struct_defs,
                bitfield_defs,
                type_alias_defs,
            )),
            *span,
        ),
    }
}

fn qualify_property_type_ident_in_namespace(
    ident: &jett_parser::ast::Ident,
    namespace: Option<&str>,
    enum_defs: &[PropertyEnumDef],
    struct_defs: &[PropertyStructDef],
    bitfield_defs: &[PropertyBitfieldDef],
    type_alias_defs: &[PropertyTypeAliasDef],
) -> jett_parser::ast::Ident {
    if ident.name.contains('.') {
        return ident.clone();
    }

    let Some(namespace) = namespace else {
        return ident.clone();
    };
    let qualified = format!("{namespace}.{}", ident.name);
    if enum_defs
        .iter()
        .any(|enum_def| enum_def.type_name == qualified)
        || struct_defs
            .iter()
            .any(|struct_def| struct_def.type_name == qualified)
        || bitfield_defs
            .iter()
            .any(|bitfield_def| bitfield_def.type_name == qualified)
        || type_alias_defs
            .iter()
            .any(|alias_def| alias_def.type_name == qualified)
    {
        let mut qualified_ident = ident.clone();
        qualified_ident.name = qualified;
        qualified_ident
    } else {
        ident.clone()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` if the function body contains at least one `assert` statement
/// (at the top level of the body).
fn has_assert_stmts(func: &FunctionDef) -> bool {
    func.body
        .stmts
        .iter()
        .any(|s| matches!(s, jett_parser::ast::Stmt::Assert(_)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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

    fn var(name: &str) -> Expr {
        Expr::Ident(ident(name))
    }

    fn binary(lhs: Expr, op: BinOp, rhs: Expr) -> Expr {
        Expr::Binary(Box::new(lhs), op, Box::new(rhs), sp())
    }

    fn string(s: &str) -> Expr {
        Expr::StringLiteral(s.to_string(), sp())
    }

    fn type_named(name: &str) -> TypeExpr {
        TypeExpr::Named(ident(name))
    }

    fn block(stmts: Vec<Stmt>) -> Block {
        Block { stmts, span: sp() }
    }

    fn return_stmt(value: Expr) -> Stmt {
        Stmt::Return(ReturnStmt {
            value: Some(value),
            span: sp(),
        })
    }

    fn assert_stmt_ast(condition: Expr) -> Stmt {
        Stmt::Assert(AssertStmt {
            condition,
            message: None,
            span: sp(),
        })
    }

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

    fn field_access(base: Expr, field: &str) -> Expr {
        Expr::FieldAccess(Box::new(base), ident(field), sp())
    }

    fn named_arg(name: &str, value: Expr) -> CallArg {
        CallArg {
            name: Some(ident(name)),
            value,
            span: sp(),
        }
    }

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

    fn generic_struct_def(
        name: &str,
        type_params: Vec<&str>,
        fields: Vec<(&str, TypeExpr)>,
    ) -> StructDef {
        StructDef {
            name: ident(name),
            type_params: type_params.into_iter().map(ident).collect(),
            fields: fields
                .into_iter()
                .map(|(field_name, ty)| FieldDef {
                    name: ident(field_name),
                    ty,
                    serialize_name: None,
                    span: sp(),
                })
                .collect(),
            methods: vec![],
            exported: false,
            span: sp(),
        }
    }

    fn enum_def(name: &str, variants: Vec<Variant>) -> EnumDef {
        EnumDef {
            name: ident(name),
            variants,
            exported: false,
            span: sp(),
        }
    }

    fn enum_variant(name: &str, fields: Vec<FieldDef>) -> Variant {
        Variant {
            name: ident(name),
            fields,
            discriminant: None,
            span: sp(),
        }
    }

    fn enum_variant_with_discriminant(name: &str, discriminant: i64) -> Variant {
        Variant {
            name: ident(name),
            fields: vec![],
            discriminant: Some(discriminant),
            span: sp(),
        }
    }

    fn enum_field(name: &str, ty: TypeExpr) -> FieldDef {
        FieldDef {
            name: ident(name),
            ty,
            serialize_name: None,
            span: sp(),
        }
    }

    fn bitfield_def(name: &str, fields: Vec<(&str, BitfieldFieldKind)>) -> BitfieldDef {
        BitfieldDef {
            name: ident(name),
            network_order: false,
            fields: fields
                .into_iter()
                .map(|(field_name, kind)| BitfieldFieldDef {
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

    #[test]
    fn eval_function_simple() {
        let add = func_def(
            "add",
            vec![("a", "int64"), ("b", "int64")],
            block(vec![return_stmt(binary(var("a"), BinOp::Add, var("b")))]),
        );
        let result = eval_function(&add, vec![Value::Int64(10), Value::Int64(20)]).unwrap();
        assert_eq!(result, Value::Int64(30));
    }

    #[test]
    fn eval_function_error() {
        let bad = func_def(
            "bad",
            vec![("a", "int64")],
            block(vec![return_stmt(binary(var("a"), BinOp::Div, int(0)))]),
        );
        let result = eval_function(&bad, vec![Value::Int64(1)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("division by zero"));
    }

    #[test]
    fn eval_assert_passes() {
        let mut interp = crate::interpreter::Interpreter::new();
        let cond = binary(int(1), BinOp::Eq, int(1));
        eval_assert(&cond, &mut interp).unwrap();
    }

    #[test]
    fn eval_assert_fails() {
        let mut interp = crate::interpreter::Interpreter::new();
        let cond = binary(int(1), BinOp::Eq, int(2));
        assert!(eval_assert(&cond, &mut interp).is_err());
    }

    #[test]
    fn run_verify_blocks_passing() {
        let verify_fn = func_def(
            "test_verify",
            vec![],
            block(vec![assert_stmt_ast(
                binary(int(2), BinOp::Add, int(3)).pipe_eq(int(5)),
            )]),
        );
        let module = Module {
            items: vec![Item::Function(verify_fn)],
            span: sp(),
        };
        let diags = run_verify_blocks(&module);
        assert!(diags.is_empty());
    }

    #[test]
    fn run_verify_blocks_failing() {
        // assert 1 == 2  -> should produce a diagnostic
        let verify_fn = func_def(
            "test_verify",
            vec![],
            block(vec![assert_stmt_ast(binary(int(1), BinOp::Eq, int(2)))]),
        );
        let module = Module {
            items: vec![Item::Function(verify_fn)],
            span: sp(),
        };
        let diags = run_verify_blocks(&module);
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("comptime verify failed"));
    }

    #[test]
    fn run_verify_blocks_calls_helper_function() {
        // function double(x: int64) returns int64:
        //     return x * 2
        let double_fn = func_def(
            "double",
            vec![("x", "int64")],
            block(vec![return_stmt(binary(var("x"), BinOp::Mul, int(2)))]),
        );

        // function test_verify():
        //     assert double(5) == 10
        let call_double = Expr::Call(
            Box::new(var("double")),
            vec![CallArg {
                name: None,
                value: int(5),
                span: sp(),
            }],
            sp(),
        );
        let verify_fn = func_def(
            "test_verify",
            vec![],
            block(vec![assert_stmt_ast(binary(
                call_double,
                BinOp::Eq,
                int(10),
            ))]),
        );

        let module = Module {
            items: vec![Item::Function(double_fn), Item::Function(verify_fn)],
            span: sp(),
        };
        let diags = run_verify_blocks(&module);
        assert!(diags.is_empty(), "expected no diagnostics, got: {diags:?}");
    }

    /// Helper trait to build `a == b` more concisely in tests.
    trait PipeEq {
        fn pipe_eq(self, other: Expr) -> Expr;
    }

    impl PipeEq for Expr {
        fn pipe_eq(self, other: Expr) -> Expr {
            Expr::Binary(Box::new(self), BinOp::Eq, Box::new(other), sp())
        }
    }

    // -----------------------------------------------------------------------
    // Item::Verify block tests
    // -----------------------------------------------------------------------

    fn verify_block_item(name: &str, body: Block) -> VerifyBlock {
        VerifyBlock {
            name: ident(name),
            body,
            span: sp(),
        }
    }

    #[test]
    fn verify_block_passing() {
        // function add(a: int64, b: int64) returns int64:
        //     return a + b
        let add_fn = func_def(
            "add",
            vec![("a", "int64"), ("b", "int64")],
            block(vec![return_stmt(binary(var("a"), BinOp::Add, var("b")))]),
        );

        // verify add:
        //     assert add(2, 3) == 5
        let call_add = Expr::Call(
            Box::new(var("add")),
            vec![
                CallArg {
                    name: None,
                    value: int(2),
                    span: sp(),
                },
                CallArg {
                    name: None,
                    value: int(3),
                    span: sp(),
                },
            ],
            sp(),
        );
        let vb = verify_block_item(
            "add",
            block(vec![assert_stmt_ast(binary(call_add, BinOp::Eq, int(5)))]),
        );

        let module = Module {
            items: vec![Item::Function(add_fn), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(results[0].passed, "expected verify block to pass");
        assert_eq!(results[0].name, "add");
    }

    #[test]
    fn verify_block_failing() {
        // function add(a: int64, b: int64) returns int64:
        //     return a + b
        let add_fn = func_def(
            "add",
            vec![("a", "int64"), ("b", "int64")],
            block(vec![return_stmt(binary(var("a"), BinOp::Add, var("b")))]),
        );

        // verify add:
        //     assert add(2, 3) == 99   <-- wrong!
        let call_add = Expr::Call(
            Box::new(var("add")),
            vec![
                CallArg {
                    name: None,
                    value: int(2),
                    span: sp(),
                },
                CallArg {
                    name: None,
                    value: int(3),
                    span: sp(),
                },
            ],
            sp(),
        );
        let vb = verify_block_item(
            "add",
            block(vec![assert_stmt_ast(binary(call_add, BinOp::Eq, int(99)))]),
        );

        let module = Module {
            items: vec![Item::Function(add_fn), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(!results[0].passed, "expected verify block to fail");
        assert_eq!(results[0].name, "add");
        assert!(results[0].error.is_some());
    }

    #[test]
    fn verify_block_can_construct_and_call_struct_methods() {
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

        let point_struct = struct_def(
            "Point",
            vec![("x", "int64"), ("y", "int64")],
            vec![total_method],
        );

        let point_ctor = Expr::Call(
            Box::new(var("Point")),
            vec![named_arg("x", int(2)), named_arg("y", int(3))],
            sp(),
        );
        let point_total = dotted_call(
            "Point",
            "total",
            vec![Expr::View(Box::new(var("point")), sp())],
        );
        let vb = verify_block_item(
            "point_total",
            block(vec![
                var_decl_stmt("Point", "point", point_ctor),
                assert_stmt_ast(binary(field_access(var("point"), "x"), BinOp::Eq, int(2))),
                assert_stmt_ast(binary(point_total, BinOp::Eq, int(5))),
            ]),
        );

        let module = Module {
            items: vec![Item::Struct(point_struct), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected struct verify block to pass: {:?}",
            results[0].error
        );
        assert_eq!(results[0].name, "point_total");
    }

    #[test]
    fn verify_block_can_call_interface_methods() {
        let speaker = interface_decl(
            "Speaker",
            vec![("speak", vec![("self", "Speaker", true)], "string")],
        );
        let dog = struct_def("Dog", vec![("name", "string")], vec![]);

        let mut speak = func_def(
            "speak",
            vec![("self", "Dog")],
            block(vec![return_stmt(field_access(var("self"), "name"))]),
        );
        speak.params[0].view = true;
        let dog_speaker = implement_block("Speaker", "Dog", vec![speak]);

        let dog_ctor = Expr::Call(
            Box::new(var("Dog")),
            vec![named_arg("name", string("bark"))],
            sp(),
        );
        let speaker_call = dotted_call(
            "Speaker",
            "speak",
            vec![Expr::View(Box::new(var("dog")), sp())],
        );
        let vb = verify_block_item(
            "speaker_dispatch",
            block(vec![
                var_decl_stmt("Dog", "dog", dog_ctor),
                assert_stmt_ast(binary(speaker_call, BinOp::Eq, string("bark"))),
            ]),
        );

        let module = Module {
            items: vec![
                Item::Interface(speaker),
                Item::Struct(dog),
                Item::Implement(dog_speaker),
                Item::Verify(vb),
            ],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected interface verify block to pass: {:?}",
            results[0].error
        );
        assert_eq!(results[0].name, "speaker_dispatch");
    }

    // -----------------------------------------------------------------------
    // Refinement type tests
    // -----------------------------------------------------------------------

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

    fn var_decl_stmt(ty_name: &str, var_name: &str, value: Expr) -> Stmt {
        Stmt::VarDecl(VarDecl {
            mutable: false,
            ty: type_named(ty_name),
            name: ident(var_name),
            value,
            span: sp(),
        })
    }

    #[test]
    fn refinement_type_valid_port() {
        // type Port = int64 where value >= 1 && value <= 65535
        // verify port_test:
        //     Port p = 8080
        //     assert p == 8080
        let constraint = binary(
            binary(var("value"), BinOp::GtEq, int(1)),
            BinOp::And,
            binary(var("value"), BinOp::LtEq, int(65535)),
        );
        let port_alias = type_alias("Port", "int64", Some(constraint));

        let vb = verify_block_item(
            "port_test",
            block(vec![
                var_decl_stmt("Port", "p", int(8080)),
                assert_stmt_ast(binary(var("p"), BinOp::Eq, int(8080))),
            ]),
        );

        let module = Module {
            items: vec![Item::TypeAlias(port_alias), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected verify block to pass: {:?}",
            results[0].error
        );
    }

    #[test]
    fn refinement_type_invalid_port() {
        // type Port = int64 where value >= 1 && value <= 65535
        // verify port_test:
        //     Port p = 0     # should fail — 0 is not >= 1
        let constraint = binary(
            binary(var("value"), BinOp::GtEq, int(1)),
            BinOp::And,
            binary(var("value"), BinOp::LtEq, int(65535)),
        );
        let port_alias = type_alias("Port", "int64", Some(constraint));

        let vb = verify_block_item("port_test", block(vec![var_decl_stmt("Port", "p", int(0))]));

        let module = Module {
            items: vec![Item::TypeAlias(port_alias), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            !results[0].passed,
            "expected verify block to fail for invalid port"
        );
        assert!(
            results[0].error.as_ref().unwrap().contains("refinement"),
            "error should mention refinement: {:?}",
            results[0].error,
        );
    }

    #[test]
    fn refinement_type_boundary_valid() {
        // Port p = 1 should pass (boundary value)
        let constraint = binary(
            binary(var("value"), BinOp::GtEq, int(1)),
            BinOp::And,
            binary(var("value"), BinOp::LtEq, int(65535)),
        );
        let port_alias = type_alias("Port", "int64", Some(constraint));

        let vb = verify_block_item(
            "port_boundary",
            block(vec![
                var_decl_stmt("Port", "p", int(1)),
                assert_stmt_ast(binary(var("p"), BinOp::Eq, int(1))),
            ]),
        );

        let module = Module {
            items: vec![Item::TypeAlias(port_alias), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(results[0].passed, "expected boundary value 1 to pass");
    }

    #[test]
    fn coarsen_strips_refinement() {
        // type Port = int64 where value >= 1 && value <= 65535
        // verify coarsen_test:
        //     Port p = 8080
        //     int64 raw = coarsen p
        //     assert raw == 8080
        let constraint = binary(
            binary(var("value"), BinOp::GtEq, int(1)),
            BinOp::And,
            binary(var("value"), BinOp::LtEq, int(65535)),
        );
        let port_alias = type_alias("Port", "int64", Some(constraint));

        let coarsen_expr = Expr::Coarsen(Box::new(var("p")), sp());

        let vb = verify_block_item(
            "coarsen_test",
            block(vec![
                var_decl_stmt("Port", "p", int(8080)),
                var_decl_stmt("int64", "raw", coarsen_expr),
                assert_stmt_ast(binary(var("raw"), BinOp::Eq, int(8080))),
            ]),
        );

        let module = Module {
            items: vec![Item::TypeAlias(port_alias), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected coarsen test to pass: {:?}",
            results[0].error
        );
    }

    #[test]
    fn simple_type_alias_no_constraint() {
        // type UserId = int64
        // verify alias_test:
        //     UserId id = 42
        //     assert id == 42
        let alias = type_alias("UserId", "int64", None);

        let vb = verify_block_item(
            "alias_test",
            block(vec![
                var_decl_stmt("UserId", "id", int(42)),
                assert_stmt_ast(binary(var("id"), BinOp::Eq, int(42))),
            ]),
        );

        let module = Module {
            items: vec![Item::TypeAlias(alias), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected simple alias to pass: {:?}",
            results[0].error
        );
    }

    // -----------------------------------------------------------------------
    // State machine tests
    // -----------------------------------------------------------------------

    fn machine_def() -> MachineDef {
        MachineDef {
            name: ident("UserAuth"),
            exported: false,
            states: vec![
                MachineState {
                    name: ident("guest"),
                    fields: vec![],
                    span: sp(),
                },
                MachineState {
                    name: ident("logged_in"),
                    fields: vec![FieldDef {
                        name: ident("user_id"),
                        ty: type_named("string"),
                        serialize_name: None,
                        span: sp(),
                    }],
                    span: sp(),
                },
                MachineState {
                    name: ident("banned"),
                    fields: vec![FieldDef {
                        name: ident("user_id"),
                        ty: type_named("string"),
                        serialize_name: None,
                        span: sp(),
                    }],
                    span: sp(),
                },
            ],
            transitions: vec![
                MachineTransition {
                    from: ident("guest"),
                    to: ident("logged_in"),
                    span: sp(),
                },
                MachineTransition {
                    from: ident("logged_in"),
                    to: ident("guest"),
                    span: sp(),
                },
                MachineTransition {
                    from: ident("logged_in"),
                    to: ident("banned"),
                    span: sp(),
                },
            ],
            span: sp(),
        }
    }

    #[test]
    fn machine_construct_and_check_state() {
        // machine UserAuth: ...
        // verify machine_test:
        //     UserAuth session = UserAuth(guest)
        //     assert session at guest

        let construct = Expr::Call(
            Box::new(var("UserAuth")),
            vec![CallArg {
                name: None,
                value: var("guest"),
                span: sp(),
            }],
            sp(),
        );

        let at_check = Expr::At(Box::new(var("session")), ident("guest"), sp());

        let vb = verify_block_item(
            "machine_test",
            block(vec![
                var_decl_stmt("UserAuth", "session", construct),
                assert_stmt_ast(at_check),
            ]),
        );

        let module = Module {
            items: vec![Item::Machine(machine_def()), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected machine construct + at check to pass: {:?}",
            results[0].error
        );
    }

    #[test]
    fn machine_transition_valid() {
        // machine UserAuth: ...
        // verify transition_test:
        //     UserAuth session = UserAuth(guest)
        //     UserAuth session2 = UserAuth.transition(session, logged_in, "user_123")
        //     assert session2 at logged_in

        let construct = Expr::Call(
            Box::new(var("UserAuth")),
            vec![CallArg {
                name: None,
                value: var("guest"),
                span: sp(),
            }],
            sp(),
        );

        let transition_call = Expr::Call(
            Box::new(Expr::FieldAccess(
                Box::new(var("UserAuth")),
                ident("transition"),
                sp(),
            )),
            vec![
                CallArg {
                    name: None,
                    value: var("session"),
                    span: sp(),
                },
                CallArg {
                    name: None,
                    value: var("logged_in"),
                    span: sp(),
                },
                CallArg {
                    name: None,
                    value: Expr::StringLiteral("user_123".to_string(), sp()),
                    span: sp(),
                },
            ],
            sp(),
        );

        let at_check = Expr::At(Box::new(var("session2")), ident("logged_in"), sp());

        let vb = verify_block_item(
            "transition_test",
            block(vec![
                var_decl_stmt("UserAuth", "session", construct),
                var_decl_stmt("UserAuth", "session2", transition_call),
                assert_stmt_ast(at_check),
            ]),
        );

        let module = Module {
            items: vec![Item::Machine(machine_def()), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected valid transition to pass: {:?}",
            results[0].error
        );
    }

    #[test]
    fn machine_transition_invalid_rejected() {
        // machine UserAuth: ...
        // verify invalid_transition_test:
        //     UserAuth session = UserAuth(guest)
        //     UserAuth session2 = UserAuth.transition(session, banned, "user_123")
        //     ^^ This should fail because guest -> banned is not an allowed transition.

        let construct = Expr::Call(
            Box::new(var("UserAuth")),
            vec![CallArg {
                name: None,
                value: var("guest"),
                span: sp(),
            }],
            sp(),
        );

        let transition_call = Expr::Call(
            Box::new(Expr::FieldAccess(
                Box::new(var("UserAuth")),
                ident("transition"),
                sp(),
            )),
            vec![
                CallArg {
                    name: None,
                    value: var("session"),
                    span: sp(),
                },
                CallArg {
                    name: None,
                    value: var("banned"),
                    span: sp(),
                },
                CallArg {
                    name: None,
                    value: Expr::StringLiteral("user_123".to_string(), sp()),
                    span: sp(),
                },
            ],
            sp(),
        );

        let vb = verify_block_item(
            "invalid_transition_test",
            block(vec![
                var_decl_stmt("UserAuth", "session", construct),
                var_decl_stmt("UserAuth", "session2", transition_call),
            ]),
        );

        let module = Module {
            items: vec![Item::Machine(machine_def()), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(!results[0].passed, "expected invalid transition to fail");
        let err = results[0].error.as_ref().unwrap();
        assert!(
            err.contains("not allowed"),
            "expected 'not allowed' in error message, got: {err}"
        );
    }

    #[test]
    fn machine_at_check_wrong_state() {
        // machine UserAuth: ...
        // verify at_wrong_state:
        //     UserAuth session = UserAuth(guest)
        //     assert session at logged_in  <-- should be false

        let construct = Expr::Call(
            Box::new(var("UserAuth")),
            vec![CallArg {
                name: None,
                value: var("guest"),
                span: sp(),
            }],
            sp(),
        );

        // `session at logged_in` should evaluate to false
        let at_check = Expr::At(Box::new(var("session")), ident("logged_in"), sp());

        let vb = verify_block_item(
            "at_wrong_state",
            block(vec![
                var_decl_stmt("UserAuth", "session", construct),
                // assert NOT(session at logged_in)
                assert_stmt_ast(Expr::Unary(UnaryOp::Not, Box::new(at_check), sp())),
            ]),
        );

        let module = Module {
            items: vec![Item::Machine(machine_def()), Item::Verify(vb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected at-check for wrong state to pass (not at logged_in): {:?}",
            results[0].error
        );
    }

    // -----------------------------------------------------------------------
    // Property block tests
    // -----------------------------------------------------------------------

    fn type_generic(name: &str, args: Vec<TypeExpr>) -> TypeExpr {
        TypeExpr::Generic(ident(name), args, sp())
    }

    fn property_block_item(name: &str, givens: Vec<GivenDecl>, body: Block) -> PropertyBlock {
        PropertyBlock {
            name: ident(name),
            givens,
            body,
            span: sp(),
        }
    }

    fn given_decl(name: &str, ty: TypeExpr) -> GivenDecl {
        GivenDecl {
            name: ident(name),
            ty,
            span: sp(),
        }
    }

    fn generic_call(module: &str, func_name: &str, args: Vec<TypeExpr>, values: Vec<Expr>) -> Expr {
        Expr::GenericCall(
            Box::new(field_access(var(module), func_name)),
            args,
            values
                .into_iter()
                .map(|value| CallArg {
                    name: None,
                    value,
                    span: sp(),
                })
                .collect(),
            sp(),
        )
    }

    #[test]
    fn property_block_passing_int64() {
        // property int_identity:
        //     given x: int64
        //     assert x == x
        let pb = property_block_item(
            "int_identity",
            vec![given_decl("x", type_named("int64"))],
            block(vec![assert_stmt_ast(binary(var("x"), BinOp::Eq, var("x")))]),
        );

        let module = Module {
            items: vec![Item::Property(pb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected property to pass: {:?}",
            results[0].error
        );
        assert!(results[0].is_property);
        assert_eq!(results[0].iterations, Some(100));
    }

    #[test]
    fn property_generator_supports_generic_collections_of_generated_types() {
        let string_lists =
            generate_values_for_type(&type_generic("list", vec![type_named("string")]));
        assert!(
            string_lists.iter().any(|value| {
                matches!(value, Value::List(items) if items.iter().any(|item| matches!(item, Value::String(_))))
            }),
            "expected list[string] generation to include string payloads"
        );

        let bool_lists = generate_values_for_type(&type_generic("list", vec![type_named("bool")]));
        assert!(
            bool_lists.iter().any(|value| {
                matches!(value, Value::List(items) if items.iter().any(|item| matches!(item, Value::Bool(_))))
            }),
            "expected list[bool] generation to include bool payloads"
        );

        let string_sets =
            generate_values_for_type(&type_generic("set", vec![type_named("string")]));
        assert!(
            string_sets.iter().any(|value| {
                matches!(value, Value::Set(items) if items.iter().any(|item| matches!(item, Value::String(_))))
            }),
            "expected set[string] generation to include string payloads"
        );

        let string_int_maps = generate_values_for_type(&type_generic(
            "map",
            vec![type_named("string"), type_named("int64")],
        ));
        assert!(
            string_int_maps.iter().any(|value| {
                matches!(value, Value::Map(entries) if entries.iter().any(|(key, value)| {
                    matches!(key, Value::String(_)) && matches!(value, Value::Int64(_))
                }))
            }),
            "expected map[string, int64] generation to include string/int64 entries"
        );

        let optional_strings =
            generate_values_for_type(&type_generic("optional", vec![type_named("string")]));
        assert!(optional_strings.contains(&Value::OptionalNone));
        assert!(
            optional_strings.iter().any(|value| {
                matches!(value, Value::OptionalSome(inner) if matches!(inner.as_ref(), Value::String(_)))
            }),
            "expected optional[string] generation to include string payloads"
        );

        let int_string_results = generate_values_for_type(&type_generic(
            "result",
            vec![type_named("int64"), type_named("string")],
        ));
        assert!(
            int_string_results
                .iter()
                .any(|value| matches!(value, Value::ResultOk(inner) if matches!(inner.as_ref(), Value::Int64(_)))),
            "expected result[int64, string] generation to include ok payloads"
        );
        assert!(
            int_string_results
                .iter()
                .any(|value| matches!(value, Value::ResultFail(inner) if matches!(inner.as_ref(), Value::String(_)))),
            "expected result[int64, string] generation to include fail payloads"
        );
    }

    #[test]
    fn property_generator_supports_bytes() {
        let byte_values = generate_values_for_type(&type_named("bytes"));
        assert!(byte_values.contains(&Value::Bytes(Vec::new())));
        assert!(
            byte_values
                .iter()
                .any(|value| matches!(value, Value::Bytes(bytes) if !bytes.is_empty())),
            "expected bytes generation to include non-empty payloads"
        );
    }

    #[test]
    fn property_generator_supports_sized_numeric_primitives() {
        let int8_values = generate_values_for_type(&type_named("int8"));
        assert!(!int8_values.is_empty());
        assert!(int8_values.iter().all(|value| {
            matches!(value, Value::Int64(n) if (i8::MIN as i64..=i8::MAX as i64).contains(n))
        }));

        let int16_values = generate_values_for_type(&type_named("int16"));
        assert!(!int16_values.is_empty());
        assert!(int16_values.iter().all(|value| {
            matches!(value, Value::Int64(n) if (i16::MIN as i64..=i16::MAX as i64).contains(n))
        }));

        let int32_values = generate_values_for_type(&type_named("int32"));
        assert!(!int32_values.is_empty());
        assert!(int32_values.iter().all(|value| {
            matches!(value, Value::Int64(n) if (i32::MIN as i64..=i32::MAX as i64).contains(n))
        }));

        let uint32_values = generate_values_for_type(&type_named("uint32"));
        assert!(!uint32_values.is_empty());
        assert!(uint32_values.iter().all(|value| {
            matches!(value, Value::Int64(n) if (0..=u32::MAX as i64).contains(n))
        }));

        let uint64_values = generate_values_for_type(&type_named("uint64"));
        assert!(!uint64_values.is_empty());
        assert!(
            uint64_values
                .iter()
                .all(|value| matches!(value, Value::Uint64(_)))
        );
        assert!(
            uint64_values
                .iter()
                .any(|value| matches!(value, Value::Uint64(n) if *n > i64::MAX as u64))
        );

        let float32_values = generate_values_for_type(&type_named("float32"));
        assert!(!float32_values.is_empty());
        assert!(
            float32_values
                .iter()
                .all(|value| matches!(value, Value::Float64(_)))
        );
    }

    #[test]
    fn property_generator_supports_enums_in_namespace() {
        let mut interp = Interpreter::new();
        let enum_defs = vec![PropertyEnumDef {
            type_name: "app.PropertyChoice".to_string(),
            namespace: Some("app".to_string()),
            def: enum_def(
                "PropertyChoice",
                vec![
                    enum_variant("empty", vec![]),
                    enum_variant("score", vec![enum_field("value", type_named("int64"))]),
                ],
            ),
        }];

        let values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_named("PropertyChoice"),
            Some("app"),
            &enum_defs,
            &[],
            &[],
            &[],
        );
        assert!(values.contains(&Value::Enum {
            type_name: "app.PropertyChoice".to_string(),
            variant: "empty".to_string(),
            fields: vec![],
        }));
        assert!(
            values.iter().any(|value| {
                matches!(
                    value,
                    Value::Enum {
                        type_name,
                        variant,
                        fields,
                    } if type_name == "app.PropertyChoice"
                        && variant == "score"
                        && matches!(fields.as_slice(), [Value::Int64(_)])
                )
            }),
            "expected enum generation to include payload variants"
        );
    }

    #[test]
    fn property_generator_supports_structs_in_namespace() {
        let mut interp = Interpreter::new();
        let struct_defs = vec![PropertyStructDef {
            type_name: "app.PropertyUser".to_string(),
            namespace: Some("app".to_string()),
            def: struct_def(
                "PropertyUser",
                vec![("name", "string"), ("score", "int64")],
                vec![],
            ),
        }];

        let values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_named("PropertyUser"),
            Some("app"),
            &[],
            &struct_defs,
            &[],
            &[],
        );
        assert!(
            values.iter().any(|value| {
                matches!(
                    value,
                    Value::Struct { type_name, fields }
                        if type_name == "app.PropertyUser"
                            && fields.iter().any(|(name, value)| {
                                name == "name" && matches!(value, Value::String(_))
                            })
                            && fields.iter().any(|(name, value)| {
                                name == "score" && matches!(value, Value::Int64(_))
                            })
                )
            }),
            "expected struct generation to include generated fields"
        );
    }

    #[test]
    fn property_generator_resolves_struct_fields_in_owner_namespace() {
        let mut interp = Interpreter::new();
        let enum_defs = vec![PropertyEnumDef {
            type_name: "models.Status".to_string(),
            namespace: Some("models".to_string()),
            def: enum_def(
                "Status",
                vec![
                    enum_variant("active", vec![]),
                    enum_variant("disabled", vec![]),
                ],
            ),
        }];
        let struct_defs = vec![PropertyStructDef {
            type_name: "models.User".to_string(),
            namespace: Some("models".to_string()),
            def: struct_def("User", vec![("status", "Status")], vec![]),
        }];

        let values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_named("models.User"),
            Some("tests"),
            &enum_defs,
            &struct_defs,
            &[],
            &[],
        );
        assert!(
            values.iter().any(|value| {
                matches!(
                    value,
                    Value::Struct { fields, .. }
                        if matches!(
                            fields.as_slice(),
                            [(_, Value::Enum { type_name, .. })] if type_name == "models.Status"
                        )
                )
            }),
            "expected struct field generation to resolve unqualified field types in the owner namespace"
        );
    }

    #[test]
    fn property_generator_supports_bitfields_in_namespace() {
        let mut interp = Interpreter::new();
        let enum_defs = vec![PropertyEnumDef {
            type_name: "app.Protocol".to_string(),
            namespace: Some("app".to_string()),
            def: enum_def(
                "Protocol",
                vec![
                    enum_variant_with_discriminant("icmp", 1),
                    enum_variant_with_discriminant("tcp", 6),
                    enum_variant_with_discriminant("udp", 17),
                ],
            ),
        }];
        let bitfield_defs = vec![PropertyBitfieldDef {
            type_name: "app.Header".to_string(),
            namespace: Some("app".to_string()),
            def: bitfield_def(
                "Header",
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
                        "protocol",
                        BitfieldFieldKind::Bits {
                            width: 8,
                            as_type: Some(type_named("Protocol")),
                        },
                    ),
                    (
                        "payload",
                        BitfieldFieldKind::Payload(type_generic("list", vec![type_named("uint8")])),
                    ),
                ],
            ),
        }];

        let values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_named("Header"),
            Some("app"),
            &enum_defs,
            &[],
            &bitfield_defs,
            &[],
        );
        assert!(
            values.iter().any(|value| {
                matches!(
                    value,
                    Value::Struct { type_name, fields }
                        if type_name == "app.Header"
                            && fields.iter().any(|(name, value)| {
                                name == "version"
                                    && matches!(value, Value::Int64(n) if (0..=15).contains(n))
                            })
                            && fields.iter().any(|(name, value)| {
                                name == "protocol"
                                    && matches!(
                                        value,
                                        Value::Enum { type_name, fields, .. }
                                            if type_name == "app.Protocol" && fields.is_empty()
                                    )
                            })
                            && fields.iter().any(|(name, value)| {
                                name == "payload"
                                    && matches!(
                                        value,
                                        Value::List(items)
                                            if !items.is_empty()
                                                && items.iter().all(|item| {
                                                    matches!(item, Value::Int64(n) if (0..=255).contains(n))
                                                })
                                    )
                            })
                )
            }),
            "expected bitfield generation to include valid bit, enum, and payload fields"
        );
    }

    #[test]
    fn property_generator_supports_type_aliases_and_refinements() {
        let mut interp = Interpreter::new();
        let alias = type_alias("NameAlias", "string", None);
        let constraint = binary(var("value"), BinOp::Gt, int(0));
        let refinement = type_alias("Positive", "int64", Some(constraint));
        interp.register_type_alias_in_namespace(Some("app"), &alias);
        interp.register_type_alias_in_namespace(Some("app"), &refinement);
        let type_alias_defs = vec![
            PropertyTypeAliasDef {
                type_name: "app.NameAlias".to_string(),
                namespace: Some("app".to_string()),
                def: alias,
            },
            PropertyTypeAliasDef {
                type_name: "app.Positive".to_string(),
                namespace: Some("app".to_string()),
                def: refinement,
            },
        ];

        let alias_values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_named("NameAlias"),
            Some("app"),
            &[],
            &[],
            &[],
            &type_alias_defs,
        );
        assert!(
            alias_values
                .iter()
                .any(|value| matches!(value, Value::String(_))),
            "expected simple alias generation to use base values"
        );

        let refined_values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_named("Positive"),
            Some("app"),
            &[],
            &[],
            &[],
            &type_alias_defs,
        );
        assert!(!refined_values.is_empty());
        assert!(
            refined_values
                .iter()
                .all(|value| matches!(value, Value::Int64(n) if *n > 0)),
            "expected refinement generation to keep only values accepted by the constraint"
        );
    }

    #[test]
    fn property_generator_filters_refined_struct_fields() {
        let mut interp = Interpreter::new();
        let constraint = binary(var("value"), BinOp::Gt, int(0));
        let refinement = type_alias("Positive", "int64", Some(constraint));
        interp.register_type_alias_in_namespace(Some("app"), &refinement);
        let type_alias_defs = vec![PropertyTypeAliasDef {
            type_name: "app.Positive".to_string(),
            namespace: Some("app".to_string()),
            def: refinement,
        }];
        let struct_defs = vec![PropertyStructDef {
            type_name: "app.Score".to_string(),
            namespace: Some("app".to_string()),
            def: struct_def("Score", vec![("value", "Positive")], vec![]),
        }];

        let values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_named("Score"),
            Some("app"),
            &[],
            &struct_defs,
            &[],
            &type_alias_defs,
        );
        assert!(
            values.iter().all(|value| {
                matches!(
                    value,
                    Value::Struct { fields, .. }
                        if matches!(fields.as_slice(), [(_, Value::Int64(n))] if *n > 0)
                )
            }),
            "expected refined struct fields to be generated through refinement filtering"
        );
    }

    #[test]
    fn property_generator_supports_generic_structs() {
        let mut interp = Interpreter::new();
        let struct_defs = vec![PropertyStructDef {
            type_name: "app.Box".to_string(),
            namespace: Some("app".to_string()),
            def: generic_struct_def("Box", vec!["T"], vec![("value", type_named("T"))]),
        }];

        let values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_generic("Box", vec![type_named("int64")]),
            Some("app"),
            &[],
            &struct_defs,
            &[],
            &[],
        );
        assert!(
            values.iter().any(|value| {
                matches!(
                    value,
                    Value::Struct { type_name, fields }
                        if type_name == "app.Box"
                            && matches!(fields.as_slice(), [(_, Value::Int64(_))])
                )
            }),
            "expected generic struct generation to substitute field type parameters"
        );
    }

    #[test]
    fn property_generator_supports_generic_structs_with_refined_args() {
        let mut interp = Interpreter::new();
        let constraint = binary(var("value"), BinOp::Gt, int(0));
        let refinement = type_alias("Positive", "int64", Some(constraint));
        interp.register_type_alias_in_namespace(Some("app"), &refinement);
        let type_alias_defs = vec![PropertyTypeAliasDef {
            type_name: "app.Positive".to_string(),
            namespace: Some("app".to_string()),
            def: refinement,
        }];
        let struct_defs = vec![PropertyStructDef {
            type_name: "app.Box".to_string(),
            namespace: Some("app".to_string()),
            def: generic_struct_def("Box", vec!["T"], vec![("value", type_named("T"))]),
        }];

        let values = generate_values_for_type_in_namespace(
            &mut interp,
            &type_generic("Box", vec![type_named("Positive")]),
            Some("app"),
            &[],
            &struct_defs,
            &[],
            &type_alias_defs,
        );
        assert!(
            values.iter().all(|value| {
                matches!(
                    value,
                    Value::Struct { fields, .. }
                        if matches!(fields.as_slice(), [(_, Value::Int64(n))] if *n > 0)
                )
            }),
            "expected generic struct type arguments to resolve in the use-site namespace"
        );
    }

    #[test]
    fn property_shrinker_simplifies_sets_and_maps() {
        let set_candidates = shrink_value(&Value::Set(vec![Value::Int64(5)]));
        assert!(set_candidates.contains(&Value::Set(vec![])));
        assert!(set_candidates.contains(&Value::Set(vec![Value::Int64(0)])));

        let map_candidates = shrink_value(&Value::Map(vec![(
            Value::String("score".to_string()),
            Value::Int64(5),
        )]));
        assert!(map_candidates.contains(&Value::Map(vec![])));
        assert!(map_candidates.contains(&Value::Map(vec![(
            Value::String("score".to_string()),
            Value::Int64(0),
        )])));
    }

    #[test]
    fn property_shrinker_simplifies_bytes() {
        let candidates = shrink_value(&Value::Bytes(b"hello".to_vec()));
        assert!(candidates.contains(&Value::Bytes(Vec::new())));
        assert!(candidates.contains(&Value::Bytes(b"he".to_vec())));
    }

    #[test]
    fn property_shrinker_simplifies_optional_and_result_values() {
        let optional_candidates = shrink_value(&Value::OptionalSome(Box::new(Value::Int64(5))));
        assert!(optional_candidates.contains(&Value::OptionalNone));
        assert!(optional_candidates.contains(&Value::OptionalSome(Box::new(Value::Int64(0)))));

        let ok_candidates = shrink_value(&Value::ResultOk(Box::new(Value::Int64(5))));
        assert!(ok_candidates.contains(&Value::ResultOk(Box::new(Value::Int64(0)))));

        let fail_candidates = shrink_value(&Value::ResultFail(Box::new(Value::String(
            "error".to_string(),
        ))));
        assert!(
            fail_candidates.contains(&Value::ResultFail(Box::new(Value::String(String::new(),))))
        );
    }

    #[test]
    fn property_shrinker_simplifies_enum_payloads() {
        let candidates = shrink_value(&Value::Enum {
            type_name: "PropertyChoice".to_string(),
            variant: "score".to_string(),
            fields: vec![Value::Int64(5)],
        });
        assert!(candidates.contains(&Value::Enum {
            type_name: "PropertyChoice".to_string(),
            variant: "score".to_string(),
            fields: vec![Value::Int64(0)],
        }));
    }

    #[test]
    fn property_shrinker_simplifies_struct_fields() {
        let candidates = shrink_value(&Value::Struct {
            type_name: "PropertyUser".to_string(),
            fields: vec![
                ("name".to_string(), Value::String("Ada".to_string())),
                ("score".to_string(), Value::Int64(5)),
            ],
        });
        assert!(candidates.contains(&Value::Struct {
            type_name: "PropertyUser".to_string(),
            fields: vec![
                ("name".to_string(), Value::String(String::new())),
                ("score".to_string(), Value::Int64(5)),
            ],
        }));
        assert!(candidates.contains(&Value::Struct {
            type_name: "PropertyUser".to_string(),
            fields: vec![
                ("name".to_string(), Value::String("Ada".to_string())),
                ("score".to_string(), Value::Int64(0)),
            ],
        }));
    }

    #[test]
    fn property_block_string_list_length_non_negative() {
        let pb = property_block_item(
            "string_list_length_non_negative",
            vec![given_decl(
                "items",
                type_generic("list", vec![type_named("string")]),
            )],
            block(vec![
                var_decl_stmt(
                    "int64",
                    "len",
                    generic_call(
                        "list",
                        "length",
                        vec![type_named("string")],
                        vec![var("items")],
                    ),
                ),
                assert_stmt_ast(binary(var("len"), BinOp::GtEq, int(0))),
            ]),
        );

        let module = Module {
            items: vec![Item::Property(pb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected property to pass: {:?}",
            results[0].error
        );
        assert!(results[0].is_property);
        assert_eq!(results[0].iterations, Some(100));
    }

    #[test]
    fn property_block_list_length_non_negative() {
        // function list_length(view items: list[int64]) returns int64:
        //     return list.length(items)
        //
        // property list_length_non_negative:
        //     given items: list[int64]
        //     int64 len = list_length(items)
        //     assert len >= 0

        let length_fn = func_def(
            "list_length",
            vec![("items", "list")],
            block(vec![return_stmt(Expr::Call(
                Box::new(Expr::FieldAccess(
                    Box::new(var("list")),
                    ident("length"),
                    sp(),
                )),
                vec![CallArg {
                    name: None,
                    value: var("items"),
                    span: sp(),
                }],
                sp(),
            ))]),
        );

        let pb = property_block_item(
            "list_length_non_negative",
            vec![given_decl(
                "items",
                type_generic("list", vec![type_named("int64")]),
            )],
            block(vec![
                var_decl_stmt(
                    "int64",
                    "len",
                    Expr::Call(
                        Box::new(var("list_length")),
                        vec![CallArg {
                            name: None,
                            value: var("items"),
                            span: sp(),
                        }],
                        sp(),
                    ),
                ),
                assert_stmt_ast(binary(var("len"), BinOp::GtEq, int(0))),
            ]),
        );

        let module = Module {
            items: vec![Item::Function(length_fn), Item::Property(pb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected property to pass: {:?}",
            results[0].error
        );
        assert!(results[0].is_property);
    }

    #[test]
    fn property_block_failing_detects_bug() {
        // property all_ints_positive:
        //     given x: int64
        //     assert x > 0
        //
        // This should fail because the int64 pool includes 0, -1, -42, i64::MIN.
        let pb = property_block_item(
            "all_ints_positive",
            vec![given_decl("x", type_named("int64"))],
            block(vec![assert_stmt_ast(binary(var("x"), BinOp::Gt, int(0)))]),
        );

        let module = Module {
            items: vec![Item::Property(pb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(!results[0].passed, "expected property to fail");
        assert!(results[0].is_property);
        let err = results[0].error.as_ref().unwrap();
        assert!(
            err.contains("counterexample:") || err.contains("input:"),
            "error should contain input values: {err}"
        );
    }

    #[test]
    fn property_block_with_function_call() {
        // function negate(x: int64) returns int64:
        //     return 0 - x
        //
        // property negate_inverts_sign:
        //     given x: int64
        //     int64 neg = negate(x)
        //     int64 neg_neg = negate(neg)
        //     assert neg_neg == x
        //
        // Note: negate(i64::MIN) overflows, which the property correctly catches.
        // We use a function that returns the same value to avoid overflow.
        let identity_fn = func_def(
            "identity",
            vec![("x", "int64")],
            block(vec![return_stmt(var("x"))]),
        );

        let pb = property_block_item(
            "identity_round_trip",
            vec![given_decl("x", type_named("int64"))],
            block(vec![
                var_decl_stmt(
                    "int64",
                    "y",
                    Expr::Call(
                        Box::new(var("identity")),
                        vec![CallArg {
                            name: None,
                            value: var("x"),
                            span: sp(),
                        }],
                        sp(),
                    ),
                ),
                assert_stmt_ast(binary(var("y"), BinOp::Eq, var("x"))),
            ]),
        );

        let module = Module {
            items: vec![Item::Function(identity_fn), Item::Property(pb)],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "expected property to pass: {:?}",
            results[0].error
        );
        assert!(results[0].is_property);
        assert_eq!(results[0].iterations, Some(100));
    }

    #[test]
    fn property_and_verify_blocks_coexist() {
        // function is_non_negative(x: int64) returns bool:
        //     return x >= 0
        //
        // verify is_non_negative:
        //     assert is_non_negative(5) == true
        //
        // property bool_result:
        //     given x: int64
        //     bool result = is_non_negative(x)
        //     # result is always either true or false — trivially true
        //     assert result == true || result == false
        let is_nn_fn = FunctionDef {
            name: ident("is_non_negative"),
            type_params: vec![],
            params: vec![Param {
                view: false,
                mutable: false,
                name: ident("x"),
                ty: type_named("int64"),
                span: sp(),
            }],
            return_type: Some(type_named("bool")),
            body: block(vec![return_stmt(binary(var("x"), BinOp::GtEq, int(0)))]),
            exported: false,
            span: sp(),
        };

        let call_nn = |arg: Expr| -> Expr {
            Expr::Call(
                Box::new(var("is_non_negative")),
                vec![CallArg {
                    name: None,
                    value: arg,
                    span: sp(),
                }],
                sp(),
            )
        };

        let vb = verify_block_item(
            "is_non_negative",
            block(vec![assert_stmt_ast(binary(
                call_nn(int(5)),
                BinOp::Eq,
                Expr::BoolLiteral(true, sp()),
            ))]),
        );

        let pb = property_block_item(
            "bool_result",
            vec![given_decl("x", type_named("int64"))],
            block(vec![
                Stmt::VarDecl(VarDecl {
                    mutable: false,
                    ty: type_named("bool"),
                    name: ident("result"),
                    value: call_nn(var("x")),
                    span: sp(),
                }),
                assert_stmt_ast(binary(
                    binary(var("result"), BinOp::Eq, Expr::BoolLiteral(true, sp())),
                    BinOp::Or,
                    binary(var("result"), BinOp::Eq, Expr::BoolLiteral(false, sp())),
                )),
            ]),
        );

        let module = Module {
            items: vec![
                Item::Function(is_nn_fn),
                Item::Verify(vb),
                Item::Property(pb),
            ],
            span: sp(),
        };
        let results = run_verify_blocks_detailed(&module);
        assert_eq!(results.len(), 2);

        // First result is the verify block
        assert!(!results[0].is_property);
        assert!(results[0].passed, "verify block should pass");

        // Second result is the property block
        assert!(results[1].is_property);
        assert!(
            results[1].passed,
            "property block should pass: {:?}",
            results[1].error
        );
        assert_eq!(results[1].iterations, Some(100));
    }
}
