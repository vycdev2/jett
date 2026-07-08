use jett_common::{FileId, STDLIB_FILE_ID_START, Span};
use jett_comptime::value::Value;
use jett_comptime::verify::{
    run_verify_blocks_detailed_with_metadata_and_expression_types,
    run_verify_blocks_with_metadata_and_expression_types,
};
use jett_diagnostics::Diagnostic;
use jett_fmt::{FormatResult, format_source};
use jett_parser::ast::{FunctionDecl, FunctionDef, Item, Module, Param, TypeExpr};
use jett_parser::parse;
use jett_resolve::resolve;
use jett_typecheck::{CheckResult, check};
use jett_types::ReflectionMetadata;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

const RUNTIME_STACK_SIZE: usize = 8 * 1024 * 1024;

struct DiscoveredModules {
    modules: Vec<Module>,
    diagnostics: Vec<Diagnostic>,
    files: HashMap<FileId, PathBuf>,
}

impl DiscoveredModules {
    fn extend(&mut self, other: DiscoveredModules) {
        self.modules.extend(other.modules);
        self.diagnostics.extend(other.diagnostics);
        self.files.extend(other.files);
    }
}

/// Result of compiling a single file.
pub struct BuildResult {
    pub diagnostics: Vec<Diagnostic>,
    pub has_errors: bool,
    /// The source text that was compiled (for diagnostic rendering).
    pub source: String,
    /// The file path that was compiled (for diagnostic rendering).
    pub file_path: String,
    /// Checked reflection metadata for runtime reflection/JSON hooks.
    pub reflection_metadata: Option<Arc<ReflectionMetadata>>,
    /// Checked expression type names for runtime normalization at expression-only sites.
    pub checked_expression_types: Option<Arc<HashMap<Span, String>>>,
}

/// Captured output from running a Jett program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutput {
    pub stdout: String,
    pub debug_output: Vec<String>,
}

/// A single definition visible through the namespace query surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryDefinition {
    pub name: String,
    pub kind: jett_resolve::scope::DefKind,
    pub namespace: Option<String>,
    pub visibility: jett_resolve::scope::DefVisibility,
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

/// Result of `jett query --agent --namespaces`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceQueryResult {
    pub definitions: Vec<QueryDefinition>,
}

/// A top-level symbol declared in a single source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSymbolQueryEntry {
    pub name: String,
    pub kind: String,
    pub namespace: Option<String>,
    pub visibility: jett_resolve::scope::DefVisibility,
    pub signature: Option<String>,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

/// Result of `jett query --agent --symbols file.jett`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSymbolsQueryResult {
    pub file_path: String,
    pub symbols: Vec<FileSymbolQueryEntry>,
}

/// Result of `jett query --agent --type-at file:line:column`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAtQueryResult {
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub type_name: Option<String>,
    pub span_line: Option<u32>,
    pub span_column: Option<u32>,
    pub span_end_line: Option<u32>,
    pub span_end_column: Option<u32>,
}

/// The resolved declaration target for a definition-at query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefinitionQueryTarget {
    pub name: String,
    pub kind: jett_resolve::scope::DefKind,
    pub namespace: Option<String>,
    pub visibility: jett_resolve::scope::DefVisibility,
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

/// Result of `jett query --agent --definition-at file:line:column`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefinitionAtQueryResult {
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub target: Option<DefinitionQueryTarget>,
}

/// A single use site returned by a references-at query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceQueryEntry {
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

/// Result of `jett query --agent --references-at file:line:column`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferencesAtQueryResult {
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub target: Option<DefinitionQueryTarget>,
    pub references: Vec<ReferenceQueryEntry>,
}

/// A single completion candidate visible at a source position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionQueryEntry {
    pub name: String,
    pub kind: jett_resolve::scope::DefKind,
    pub namespace: Option<String>,
    pub visibility: jett_resolve::scope::DefVisibility,
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
    pub match_kind: CompletionMatchKind,
    pub rank: u32,
    pub signature: Option<String>,
}

/// How a completion candidate matched the cursor prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompletionMatchKind {
    EmptyPrefix,
    Exact,
    QualifiedPrefix,
    LeafPrefix,
}

/// Result of `jett query --agent --complete-at file:line:column`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionsQueryResult {
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub prefix: String,
    pub candidates: Vec<CompletionQueryEntry>,
}

/// A single parameter in a queried function signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureParam {
    pub name: String,
    pub type_name: String,
    pub view: bool,
    pub mutable: bool,
}

/// Result of `jett query --agent --signature function.name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureQueryResult {
    pub name: String,
    pub type_params: Vec<String>,
    pub params: Vec<SignatureParam>,
    pub return_type: String,
    pub file_path: String,
}

#[derive(Clone, Copy)]
struct RunOptions {
    capture_stdout: bool,
    emit_runtime_debug: bool,
}

/// Run the full compilation pipeline on in-memory source text.
/// Used by the LSP server to validate documents without touching the filesystem.
pub fn build_source(source: &str, file_path: &str) -> BuildResult {
    let file_id = FileId::new(0);
    let mut all_diagnostics = Vec::new();

    // Phase 1+2: Lex + Parse
    let mut parse_result = parse(source, file_id);
    all_diagnostics.extend(parse_result.errors.clone());

    let has_parse_errors = has_error_diagnostics(&all_diagnostics);
    if has_parse_errors {
        return BuildResult {
            has_errors: true,
            diagnostics: all_diagnostics,
            source: source.to_string(),
            file_path: file_path.to_string(),
            reflection_metadata: None,
            checked_expression_types: None,
        };
    }

    // Phase 3: Resolve names
    let support_modules = discover_stdlib_modules_with_diagnostics();
    all_diagnostics.extend(support_modules.diagnostics);
    if has_error_diagnostics(&all_diagnostics) {
        return BuildResult {
            has_errors: true,
            diagnostics: all_diagnostics,
            source: source.to_string(),
            file_path: file_path.to_string(),
            reflection_metadata: None,
            checked_expression_types: None,
        };
    }
    prepend_support_modules(&mut parse_result.module, support_modules.modules);

    let resolve_result = resolve(&parse_result.module);
    all_diagnostics.extend(resolve_result.diagnostics.clone());

    let has_resolve_errors = has_error_diagnostics(&all_diagnostics);
    if has_resolve_errors {
        return BuildResult {
            has_errors: true,
            diagnostics: all_diagnostics,
            source: source.to_string(),
            file_path: file_path.to_string(),
            reflection_metadata: None,
            checked_expression_types: None,
        };
    }

    // Phase 4: Type check
    let check_result = check(&parse_result.module, &resolve_result);
    all_diagnostics.extend(check_result.diagnostics.clone());

    let has_typecheck_errors = has_error_diagnostics(&all_diagnostics);
    if has_typecheck_errors {
        return BuildResult {
            has_errors: true,
            diagnostics: all_diagnostics,
            source: source.to_string(),
            file_path: file_path.to_string(),
            reflection_metadata: None,
            checked_expression_types: None,
        };
    }

    // Phase 5: Execute verify blocks at compile time
    let reflection_metadata = check_result.reflection_metadata.clone();
    let checked_expression_types = Arc::new(expression_type_names(&check_result));
    let verify_diagnostics = run_verify_blocks_with_metadata_and_expression_types(
        &parse_result.module,
        check_result.reflection_metadata,
        checked_expression_types.clone(),
    );
    all_diagnostics.extend(verify_diagnostics);

    let has_errors = has_error_diagnostics(&all_diagnostics);

    BuildResult {
        has_errors,
        diagnostics: all_diagnostics,
        source: source.to_string(),
        file_path: file_path.to_string(),
        reflection_metadata: Some(reflection_metadata),
        checked_expression_types: Some(checked_expression_types),
    }
}

/// Return the inferred type name at the given (1-based) line and column in `source`.
/// Returns `None` if the position is outside any typed expression or if the file
/// does not compile cleanly past the parse phase.
pub fn hover_type(source: &str, line: u32, col: u32) -> Option<String> {
    let file_id = FileId::new(0);

    // Convert 1-based (line, col) to a byte offset.
    let offset = line_col_to_offset(source, line, col)?;

    let mut parse_result = parse(source, file_id);
    if parse_result
        .errors
        .iter()
        .any(|d| d.severity == jett_diagnostics::Severity::Error)
    {
        return None;
    }

    prepend_support_modules(&mut parse_result.module, discover_stdlib_modules());

    let resolve_result = resolve(&parse_result.module);
    if resolve_result
        .diagnostics
        .iter()
        .any(|d| d.severity == jett_diagnostics::Severity::Error)
    {
        return None;
    }

    let check_result = check(&parse_result.module, &resolve_result);

    // Find the smallest span in type_map that contains `offset`.
    let mut best: Option<(u32, jett_types::TypeId)> = None;
    for (span, ty_id) in &check_result.type_map {
        if span.file == file_id && span.start <= offset && offset <= span.end {
            let len = span.end - span.start;
            if best.is_none() || len < best.unwrap().0 {
                best = Some((len, *ty_id));
            }
        }
    }

    best.map(|(_, ty_id)| check_result.interner.type_name(ty_id))
}

/// Return the inferred type name at a source position in a file.
///
/// This query parses, resolves, and typechecks with stdlib plus sibling project
/// modules, but it does not execute verify/property blocks.
pub fn query_type_at(path: &Path, line: u32, column: u32) -> Result<TypeAtQueryResult, String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let file_path = path.display().to_string();
    let file_id = FileId::new(0);
    let Some(offset) = line_col_to_offset(&source, line, column) else {
        return Err(format!(
            "position {line}:{column} is outside {}",
            path.display()
        ));
    };

    let mut parse_result = parse(&source, file_id);
    let parse_errors = error_messages_from_diagnostics(&parse_result.errors);
    if !parse_errors.is_empty() {
        return Err(format!("parse errors:\n{}", parse_errors.join("\n")));
    }

    let mut support_modules = discover_stdlib_modules_with_diagnostics();
    support_modules.extend(discover_project_modules_with_diagnostics(path));
    let support_errors = error_messages_from_diagnostics(&support_modules.diagnostics);
    if !support_errors.is_empty() {
        return Err(format!(
            "support parse errors:\n{}",
            support_errors.join("\n")
        ));
    }
    prepend_support_modules(&mut parse_result.module, support_modules.modules);

    let resolve_result = resolve(&parse_result.module);
    let resolve_errors = error_messages_from_diagnostics(&resolve_result.diagnostics);
    if !resolve_errors.is_empty() {
        return Err(format!("resolution errors:\n{}", resolve_errors.join("\n")));
    }

    let check_result = check(&parse_result.module, &resolve_result);
    let type_errors = error_messages_from_diagnostics(&check_result.diagnostics);
    if !type_errors.is_empty() {
        return Err(format!("type errors:\n{}", type_errors.join("\n")));
    }

    let mut best: Option<(u32, Span, jett_types::TypeId)> = None;
    for (span, ty_id) in &check_result.type_map {
        if span.file == file_id && span.start <= offset && offset <= span.end {
            let len = span.end - span.start;
            if best.is_none() || len < best.unwrap().0 {
                best = Some((len, *span, *ty_id));
            }
        }
    }
    let (type_name, span_line, span_column, span_end_line, span_end_column) =
        if let Some((_, span, ty_id)) = best {
            let (span_line, span_column) = jett_diagnostics::render::line_col(&source, span.start);
            let (span_end_line, span_end_column) =
                jett_diagnostics::render::line_col(&source, span.end);
            (
                Some(check_result.interner.type_name(ty_id)),
                Some(span_line as u32),
                Some(span_column as u32),
                Some(span_end_line as u32),
                Some(span_end_column as u32),
            )
        } else {
            (None, None, None, None, None)
        };

    Ok(TypeAtQueryResult {
        file_path,
        line,
        column,
        type_name,
        span_line,
        span_column,
        span_end_line,
        span_end_column,
    })
}

/// Return the resolved definition target at a source position in a file.
///
/// This query parses and resolves with stdlib plus sibling project modules, but
/// it does not typecheck or execute verify/property blocks.
pub fn query_definition_at(
    path: &Path,
    line: u32,
    column: u32,
) -> Result<DefinitionAtQueryResult, String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let file_path = path.display().to_string();
    let file_id = FileId::new(0);
    let Some(offset) = line_col_to_offset(&source, line, column) else {
        return Err(format!(
            "position {line}:{column} is outside {}",
            path.display()
        ));
    };

    let mut parse_result = parse(&source, file_id);
    let parse_errors = error_messages_from_diagnostics(&parse_result.errors);
    if !parse_errors.is_empty() {
        return Err(format!("parse errors:\n{}", parse_errors.join("\n")));
    }

    let mut support_modules = discover_stdlib_modules_with_diagnostics();
    support_modules.extend(discover_project_modules_with_diagnostics(path));
    let support_errors = error_messages_from_diagnostics(&support_modules.diagnostics);
    if !support_errors.is_empty() {
        return Err(format!(
            "support parse errors:\n{}",
            support_errors.join("\n")
        ));
    }

    let mut file_paths = support_modules.files.clone();
    let display_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    file_paths.insert(file_id, display_path);

    prepend_support_modules(&mut parse_result.module, support_modules.modules);

    let resolve_result = resolve(&parse_result.module);
    let resolve_errors = error_messages_from_diagnostics(&resolve_result.diagnostics);
    if !resolve_errors.is_empty() {
        return Err(format!("resolution errors:\n{}", resolve_errors.join("\n")));
    }

    let mut best_def: Option<(u32, jett_resolve::scope::DefId)> = None;
    for (span, def_id) in &resolve_result.resolutions {
        if span.file == file_id && span.start <= offset && offset <= span.end {
            let len = span.end - span.start;
            if best_def.is_none() || len < best_def.unwrap().0 {
                best_def = Some((len, *def_id));
            }
        }
    }

    let target = best_def.and_then(|(_, def_id)| {
        let def = resolve_result.scope_table.def(def_id);
        let (file_path, line, column, end_line, end_column) =
            span_location(Some(&source), def.span, &file_paths)?;
        Some(DefinitionQueryTarget {
            name: def.name.clone(),
            kind: def.kind,
            namespace: def.namespace.clone(),
            visibility: def.visibility,
            file_path,
            line,
            column,
            end_line,
            end_column,
        })
    });

    Ok(DefinitionAtQueryResult {
        file_path,
        line,
        column,
        target,
    })
}

/// Return all resolver-visible references to the symbol at a source position.
///
/// This uses the same resolver map as definition-at and returns use sites only;
/// the declaration itself is reported separately as `target`.
pub fn query_references_at(
    path: &Path,
    line: u32,
    column: u32,
) -> Result<ReferencesAtQueryResult, String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let file_path = path.display().to_string();
    let file_id = FileId::new(0);
    let Some(offset) = line_col_to_offset(&source, line, column) else {
        return Err(format!(
            "position {line}:{column} is outside {}",
            path.display()
        ));
    };

    let mut parse_result = parse(&source, file_id);
    let parse_errors = error_messages_from_diagnostics(&parse_result.errors);
    if !parse_errors.is_empty() {
        return Err(format!("parse errors:\n{}", parse_errors.join("\n")));
    }

    let mut support_modules = discover_stdlib_modules_with_diagnostics();
    support_modules.extend(discover_project_modules_with_diagnostics(path));
    let support_errors = error_messages_from_diagnostics(&support_modules.diagnostics);
    if !support_errors.is_empty() {
        return Err(format!(
            "support parse errors:\n{}",
            support_errors.join("\n")
        ));
    }

    let mut file_paths = support_modules.files.clone();
    let display_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    file_paths.insert(file_id, display_path);

    prepend_support_modules(&mut parse_result.module, support_modules.modules);

    let resolve_result = resolve(&parse_result.module);
    let resolve_errors = error_messages_from_diagnostics(&resolve_result.diagnostics);
    if !resolve_errors.is_empty() {
        return Err(format!("resolution errors:\n{}", resolve_errors.join("\n")));
    }

    let Some((_, target_def_id)) = best_resolved_definition_at(&resolve_result, file_id, offset)
    else {
        return Ok(ReferencesAtQueryResult {
            file_path,
            line,
            column,
            target: None,
            references: Vec::new(),
        });
    };

    let def = resolve_result.scope_table.def(target_def_id);
    let target = span_location(Some(&source), def.span, &file_paths).map(
        |(file_path, line, column, end_line, end_column)| DefinitionQueryTarget {
            name: def.name.clone(),
            kind: def.kind,
            namespace: def.namespace.clone(),
            visibility: def.visibility,
            file_path,
            line,
            column,
            end_line,
            end_column,
        },
    );

    let mut references = Vec::new();
    for (span, def_id) in &resolve_result.resolutions {
        if *def_id == target_def_id
            && let Some((file_path, line, column, end_line, end_column)) =
                span_location(Some(&source), *span, &file_paths)
        {
            references.push(ReferenceQueryEntry {
                file_path,
                line,
                column,
                end_line,
                end_column,
            });
        }
    }
    references.sort_by(|left, right| {
        left.file_path
            .cmp(&right.file_path)
            .then_with(|| left.line.cmp(&right.line))
            .then_with(|| left.column.cmp(&right.column))
    });

    Ok(ReferencesAtQueryResult {
        file_path,
        line,
        column,
        target,
        references,
    })
}

/// Return a list of (name, kind) completion candidates visible in `source`.
/// Runs parse + resolve and collects all definitions from the scope table.
pub fn completions(source: &str) -> Vec<(String, jett_resolve::scope::DefKind)> {
    completions_for_namespace(source, None)
}

/// Return completion candidates visible at the given (1-based) line and column.
pub fn completions_at(
    source: &str,
    line: u32,
    col: u32,
) -> Vec<(String, jett_resolve::scope::DefKind)> {
    let file_id = FileId::new(0);
    let Some(offset) = line_col_to_offset(source, line, col) else {
        return Vec::new();
    };

    let parsed = parse(source, file_id);
    let current_namespace = namespace_at_offset(&parsed.module, file_id, offset);
    let support_modules = discover_stdlib_modules();
    let current_namespace = current_namespace
        .filter(|namespace| !support_modules_declare_namespace(&support_modules, namespace));
    completions_for_namespace_with_support(source, current_namespace.as_deref(), support_modules)
}

/// Return completion candidates visible at a source position in a file.
pub fn query_completions_at(
    path: &Path,
    line: u32,
    column: u32,
) -> Result<CompletionsQueryResult, String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let file_path = path.display().to_string();
    let file_id = FileId::new(0);
    let Some(offset) = line_col_to_offset(&source, line, column) else {
        return Err(format!(
            "position {line}:{column} is outside {}",
            path.display()
        ));
    };
    let prefix = completion_prefix_at(&source, offset);

    let parsed = parse(&source, file_id);
    let parse_errors = error_messages_from_diagnostics(&parsed.errors);
    if !parse_errors.is_empty() {
        return Err(format!("parse errors:\n{}", parse_errors.join("\n")));
    }

    let mut support_modules = discover_stdlib_modules_with_diagnostics();
    support_modules.extend(discover_project_modules_with_diagnostics(path));
    let support_errors = error_messages_from_diagnostics(&support_modules.diagnostics);
    if !support_errors.is_empty() {
        return Err(format!(
            "support parse errors:\n{}",
            support_errors.join("\n")
        ));
    }

    let mut file_paths = support_modules.files.clone();
    let display_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    file_paths.insert(file_id, display_path);

    let mut definitions = query_builtin_definitions();
    for module in &support_modules.modules {
        append_module_query_definitions(&mut definitions, module, None, &file_paths);
    }
    append_module_query_definitions(&mut definitions, &parsed.module, Some(&source), &file_paths);

    let mut signatures = HashMap::new();
    for module in &support_modules.modules {
        append_module_signature_displays(&mut signatures, module, &file_paths);
    }
    append_module_signature_displays(&mut signatures, &parsed.module, &file_paths);

    let mut candidates: Vec<CompletionQueryEntry> = definitions
        .into_iter()
        .filter_map(|definition| {
            let match_kind = completion_match_kind(&definition.name, &prefix)?;
            Some(CompletionQueryEntry {
                signature: signatures.get(&definition.name).cloned(),
                name: definition.name,
                kind: definition.kind,
                namespace: definition.namespace,
                visibility: definition.visibility,
                file_path: definition.file_path,
                line: definition.line,
                column: definition.column,
                end_line: definition.end_line,
                end_column: definition.end_column,
                match_kind,
                rank: completion_rank(match_kind),
            })
        })
        .collect();
    candidates.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| query_kind_name(left.kind).cmp(query_kind_name(right.kind)))
    });
    candidates.dedup_by(|left, right| left.name == right.name && left.kind == right.kind);

    Ok(CompletionsQueryResult {
        file_path,
        line,
        column,
        prefix,
        candidates,
    })
}

/// Return the source-level signature for a public function.
pub fn query_signature(
    start_dir: &Path,
    function_name: &str,
) -> Result<Option<SignatureQueryResult>, String> {
    let mut support_modules = discover_stdlib_modules_with_diagnostics();
    support_modules.extend(discover_query_project_modules_with_diagnostics(start_dir));

    let support_errors = error_messages_from_diagnostics(&support_modules.diagnostics);
    if !support_errors.is_empty() {
        return Err(format!(
            "query support parse errors:\n{}",
            support_errors.join("\n")
        ));
    }

    for module in &support_modules.modules {
        if let Some(signature) =
            module_signature_query_result(module, &support_modules.files, function_name)
        {
            return Ok(Some(signature));
        }
    }

    Ok(None)
}

fn module_signature_query_result(
    module: &Module,
    file_paths: &HashMap<FileId, PathBuf>,
    function_name: &str,
) -> Option<SignatureQueryResult> {
    let mut current_namespace: Option<String> = None;
    for item in &module.items {
        match item {
            Item::Namespace(ns) => current_namespace = Some(ns.name.name.clone()),
            Item::Function(func) => {
                if let Some(signature) =
                    function_signature_query_result(func, current_namespace.as_deref(), file_paths)
                    && signature.name == function_name
                {
                    return Some(signature);
                }
            }
            Item::Mutual(block) => {
                for decl in &block.declarations {
                    if let Some(signature) = function_decl_signature_query_result(
                        decl,
                        current_namespace.as_deref(),
                        file_paths,
                    ) && signature.name == function_name
                    {
                        return Some(signature);
                    }
                }
            }
            Item::Interface(_)
            | Item::Implement(_)
            | Item::Struct(_)
            | Item::Bitfield(_)
            | Item::Enum(_)
            | Item::Machine(_)
            | Item::Actor(_)
            | Item::VarDecl(_)
            | Item::Verify(_)
            | Item::Property(_)
            | Item::TypeAlias(_) => {}
        }
    }
    None
}

fn append_module_signature_displays(
    signatures: &mut HashMap<String, String>,
    module: &Module,
    file_paths: &HashMap<FileId, PathBuf>,
) {
    let mut current_namespace: Option<String> = None;
    for item in &module.items {
        match item {
            Item::Namespace(ns) => current_namespace = Some(ns.name.name.clone()),
            Item::Function(func) => {
                if let Some(signature) =
                    function_signature_query_result(func, current_namespace.as_deref(), file_paths)
                {
                    signatures
                        .entry(signature.name.clone())
                        .or_insert_with(|| signature_display(&signature));
                }
            }
            Item::Mutual(block) => {
                for decl in &block.declarations {
                    if let Some(signature) = function_decl_signature_query_result(
                        decl,
                        current_namespace.as_deref(),
                        file_paths,
                    ) {
                        signatures
                            .entry(signature.name.clone())
                            .or_insert_with(|| signature_display(&signature));
                    }
                }
            }
            Item::Interface(_)
            | Item::Implement(_)
            | Item::Struct(_)
            | Item::Bitfield(_)
            | Item::Enum(_)
            | Item::Machine(_)
            | Item::Actor(_)
            | Item::VarDecl(_)
            | Item::Verify(_)
            | Item::Property(_)
            | Item::TypeAlias(_) => {}
        }
    }
}

fn function_signature_query_result(
    func: &FunctionDef,
    namespace: Option<&str>,
    file_paths: &HashMap<FileId, PathBuf>,
) -> Option<SignatureQueryResult> {
    if namespace.is_some() && !func.exported {
        return None;
    }

    Some(signature_query_result(
        &func.name.name,
        &func.type_params,
        &func.params,
        func.return_type.as_ref(),
        namespace,
        func.span.file,
        file_paths,
    ))
}

fn function_decl_signature_query_result(
    decl: &FunctionDecl,
    namespace: Option<&str>,
    file_paths: &HashMap<FileId, PathBuf>,
) -> Option<SignatureQueryResult> {
    if namespace.is_some() && !decl.exported {
        return None;
    }

    Some(signature_query_result(
        &decl.name.name,
        &decl.type_params,
        &decl.params,
        decl.return_type.as_ref(),
        namespace,
        decl.span.file,
        file_paths,
    ))
}

fn signature_display(signature: &SignatureQueryResult) -> String {
    let type_params = if signature.type_params.is_empty() {
        String::new()
    } else {
        format!("[{}]", signature.type_params.join(", "))
    };
    let params: Vec<String> = signature
        .params
        .iter()
        .map(|param| {
            let mut prefix = String::new();
            if param.view {
                prefix.push_str("view ");
            }
            if param.mutable {
                prefix.push_str("mutable ");
            }
            format!("{prefix}{}: {}", param.name, param.type_name)
        })
        .collect();
    format!(
        "{}{}({}) returns {}",
        signature.name,
        type_params,
        params.join(", "),
        signature.return_type
    )
}

fn signature_query_result(
    leaf_name: &str,
    type_params: &[jett_parser::ast::Ident],
    params: &[Param],
    return_type: Option<&TypeExpr>,
    namespace: Option<&str>,
    file: FileId,
    file_paths: &HashMap<FileId, PathBuf>,
) -> SignatureQueryResult {
    let name = namespace
        .map(|namespace| format!("{namespace}.{leaf_name}"))
        .unwrap_or_else(|| leaf_name.to_string());
    let type_params: Vec<String> = type_params.iter().map(|param| param.name.clone()).collect();
    let params = params
        .iter()
        .map(|param| SignatureParam {
            name: param.name.name.clone(),
            type_name: signature_type_expr_name(&param.ty, namespace, &type_params),
            view: param.view,
            mutable: param.mutable,
        })
        .collect();
    let return_type = return_type
        .map(|ty| signature_type_expr_name(ty, namespace, &type_params))
        .unwrap_or_else(|| "nothing".to_string());

    SignatureQueryResult {
        name,
        type_params,
        params,
        return_type,
        file_path: query_file_path(file, file_paths),
    }
}

fn signature_type_expr_name(
    ty: &TypeExpr,
    namespace: Option<&str>,
    type_params: &[String],
) -> String {
    match ty {
        TypeExpr::Named(ident) => signature_named_type_name(&ident.name, namespace, type_params),
        TypeExpr::Generic(name, args, _) => {
            let name = signature_generic_type_name(&name.name, namespace);
            let args: Vec<String> = args
                .iter()
                .map(|arg| signature_type_expr_name(arg, namespace, type_params))
                .collect();
            format!("{}[{}]", name, args.join(", "))
        }
        TypeExpr::View(inner, _) => {
            format!(
                "view {}",
                signature_type_expr_name(inner, namespace, type_params)
            )
        }
        TypeExpr::StateQualified(inner, state, _) => {
            format!(
                "{} at {}",
                signature_type_expr_name(inner, namespace, type_params),
                state.name
            )
        }
        TypeExpr::Function(params, ret, _) => {
            let params: Vec<String> = params
                .iter()
                .map(|param| signature_type_expr_name(param, namespace, type_params))
                .collect();
            format!(
                "function({}) returns {}",
                params.join(", "),
                signature_type_expr_name(ret, namespace, type_params)
            )
        }
    }
}

fn signature_named_type_name(
    name: &str,
    namespace: Option<&str>,
    type_params: &[String],
) -> String {
    if name.contains('.')
        || type_params.iter().any(|type_param| type_param == name)
        || signature_builtin_type_name(name)
    {
        return name.to_string();
    }

    namespace
        .map(|namespace| format!("{namespace}.{name}"))
        .unwrap_or_else(|| name.to_string())
}

fn signature_generic_type_name(name: &str, namespace: Option<&str>) -> String {
    if name.contains('.') || signature_builtin_generic_type_name(name) {
        return name.to_string();
    }

    namespace
        .map(|namespace| format!("{namespace}.{name}"))
        .unwrap_or_else(|| name.to_string())
}

fn signature_builtin_type_name(name: &str) -> bool {
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
            | "TypeInfo"
            | "TypeKind"
            | "TypePrimitive"
            | "TypeField"
            | "TypeBitfield"
            | "TypeBitfieldField"
            | "TypeBitfieldFieldShape"
            | "TypeMachine"
            | "TypeMachineState"
            | "TypeMachineTransition"
            | "TypeVariant"
            | "Stdout"
            | "Stderr"
            | "Stdin"
            | "Filesystem"
            | "Network"
            | "Clock"
            | "Random"
            | "Process"
            | "Environment"
    )
}

fn signature_builtin_generic_type_name(name: &str) -> bool {
    matches!(
        name,
        "list" | "map" | "set" | "optional" | "result" | "secret"
    )
}

fn completions_for_namespace(
    source: &str,
    current_namespace: Option<&str>,
) -> Vec<(String, jett_resolve::scope::DefKind)> {
    completions_for_namespace_with_support(source, current_namespace, discover_stdlib_modules())
}

fn completions_for_namespace_with_support(
    source: &str,
    current_namespace: Option<&str>,
    support_modules: Vec<Module>,
) -> Vec<(String, jett_resolve::scope::DefKind)> {
    use jett_resolve::scope::DefVisibility;

    let file_id = FileId::new(0);
    let mut parse_result = parse(source, file_id);
    if parse_result
        .errors
        .iter()
        .any(|d| d.severity == jett_diagnostics::Severity::Error)
    {
        return Vec::new();
    }
    prepend_support_modules(&mut parse_result.module, support_modules);

    let resolve_result = resolve(&parse_result.module);
    resolve_result
        .scope_table
        .definitions
        .iter()
        .filter(|def| {
            def.namespace.is_none()
                || def.visibility == DefVisibility::Public
                || def.namespace.as_deref() == current_namespace
        })
        .map(|def| (def.name.clone(), def.kind))
        .collect()
}

fn completion_prefix_at(source: &str, offset: u32) -> String {
    let mut end = offset as usize;
    if end > source.len() {
        end = source.len();
    }
    while end > 0 && !source.is_char_boundary(end) {
        end -= 1;
    }

    let mut start = end;
    while start > 0 {
        let Some(ch) = source[..start].chars().next_back() else {
            break;
        };
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            start -= ch.len_utf8();
        } else {
            break;
        }
    }

    source[start..end].to_string()
}

fn completion_match_kind(name: &str, prefix: &str) -> Option<CompletionMatchKind> {
    if prefix.is_empty() {
        return Some(CompletionMatchKind::EmptyPrefix);
    }

    let leaf = name.rsplit_once('.').map_or(name, |(_, leaf)| leaf);
    if name == prefix || leaf == prefix {
        return Some(CompletionMatchKind::Exact);
    }
    if name.starts_with(prefix) {
        return Some(CompletionMatchKind::QualifiedPrefix);
    }
    if prefix.contains('.') {
        return None;
    }

    leaf.starts_with(prefix)
        .then_some(CompletionMatchKind::LeafPrefix)
}

fn completion_rank(match_kind: CompletionMatchKind) -> u32 {
    match match_kind {
        CompletionMatchKind::Exact => 0,
        CompletionMatchKind::QualifiedPrefix => 10,
        CompletionMatchKind::LeafPrefix => 20,
        CompletionMatchKind::EmptyPrefix => 100,
    }
}

/// Return the public namespace and definition registry available from `start_dir`.
///
/// If `start_dir` is inside a project, project `.jett` files are included with
/// compiler-shipped stdlib modules. Without a `jett.proj`, the query still
/// returns stdlib and language built-ins so agents can discover the base surface.
pub fn query_namespaces(start_dir: &Path) -> Result<NamespaceQueryResult, String> {
    let mut support_modules = discover_stdlib_modules_with_diagnostics();
    support_modules.extend(discover_query_project_modules_with_diagnostics(start_dir));

    let support_errors = error_messages_from_diagnostics(&support_modules.diagnostics);
    if !support_errors.is_empty() {
        return Err(format!(
            "query support parse errors:\n{}",
            support_errors.join("\n")
        ));
    }

    let mut definitions = query_builtin_definitions();
    for module in &support_modules.modules {
        append_module_query_definitions(&mut definitions, module, None, &support_modules.files);
    }

    definitions.sort_by(|left, right| {
        left.namespace
            .cmp(&right.namespace)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| query_kind_name(left.kind).cmp(query_kind_name(right.kind)))
    });
    definitions.dedup_by(|left, right| {
        left.name == right.name
            && left.kind == right.kind
            && left.namespace == right.namespace
            && left.visibility == right.visibility
    });

    Ok(NamespaceQueryResult { definitions })
}

/// Return a file-local outline of top-level declarations, including private
/// symbols that are intentionally omitted from the global namespace query.
pub fn query_file_symbols(path: &Path) -> Result<FileSymbolsQueryResult, String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let parsed = parse(&source, FileId::new(0));
    let parse_errors = error_messages_from_diagnostics(&parsed.errors);
    if !parse_errors.is_empty() {
        return Err(format!("parse errors:\n{}", parse_errors.join("\n")));
    }

    let mut symbols = Vec::new();
    let mut file_paths = HashMap::new();
    let display_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    file_paths.insert(FileId::new(0), display_path);
    append_file_symbol_query_entries(&mut symbols, &parsed.module, &source, &file_paths);
    Ok(FileSymbolsQueryResult {
        file_path: path.display().to_string(),
        symbols,
    })
}

fn append_file_symbol_query_entries(
    symbols: &mut Vec<FileSymbolQueryEntry>,
    module: &Module,
    source: &str,
    file_paths: &HashMap<FileId, PathBuf>,
) {
    let mut current_namespace: Option<String> = None;
    for item in &module.items {
        match item {
            Item::Namespace(ns) => {
                current_namespace = Some(ns.name.name.clone());
                push_file_symbol_query_entry(
                    symbols,
                    ns.name.name.clone(),
                    "namespace",
                    None,
                    jett_resolve::scope::DefVisibility::Public,
                    None,
                    ns.name.span,
                    source,
                );
            }
            Item::Function(func) => {
                let signature = file_symbol_function_signature(
                    &func.name.name,
                    &func.type_params,
                    &func.params,
                    func.return_type.as_ref(),
                    current_namespace.as_deref(),
                    func.span.file,
                    file_paths,
                );
                push_file_symbol_query_entry(
                    symbols,
                    file_symbol_name(&func.name.name, current_namespace.as_deref()),
                    "function",
                    current_namespace.clone(),
                    file_symbol_visibility(current_namespace.as_deref(), func.exported),
                    Some(signature),
                    func.name.span,
                    source,
                );
            }
            Item::Mutual(block) => {
                for decl in &block.declarations {
                    let signature = file_symbol_function_signature(
                        &decl.name.name,
                        &decl.type_params,
                        &decl.params,
                        decl.return_type.as_ref(),
                        current_namespace.as_deref(),
                        decl.span.file,
                        file_paths,
                    );
                    push_file_symbol_query_entry(
                        symbols,
                        file_symbol_name(&decl.name.name, current_namespace.as_deref()),
                        "function",
                        current_namespace.clone(),
                        file_symbol_visibility(current_namespace.as_deref(), decl.exported),
                        Some(signature),
                        decl.name.span,
                        source,
                    );
                }
            }
            Item::Interface(interface) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&interface.name.name, current_namespace.as_deref()),
                "interface",
                current_namespace.clone(),
                file_symbol_visibility(current_namespace.as_deref(), interface.exported),
                None,
                interface.name.span,
                source,
            ),
            Item::Implement(block) => push_file_symbol_query_entry(
                symbols,
                format!(
                    "implement {} for {}",
                    block.interface_name.name,
                    type_expr_name(&block.for_type)
                ),
                "implement",
                current_namespace.clone(),
                jett_resolve::scope::DefVisibility::Private,
                None,
                block.interface_name.span,
                source,
            ),
            Item::Struct(strukt) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&strukt.name.name, current_namespace.as_deref()),
                "struct",
                current_namespace.clone(),
                file_symbol_visibility(current_namespace.as_deref(), strukt.exported),
                None,
                strukt.name.span,
                source,
            ),
            Item::Bitfield(bitfield) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&bitfield.name.name, current_namespace.as_deref()),
                "bitfield",
                current_namespace.clone(),
                file_symbol_visibility(current_namespace.as_deref(), bitfield.exported),
                None,
                bitfield.name.span,
                source,
            ),
            Item::Enum(enm) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&enm.name.name, current_namespace.as_deref()),
                "enum",
                current_namespace.clone(),
                file_symbol_visibility(current_namespace.as_deref(), enm.exported),
                None,
                enm.name.span,
                source,
            ),
            Item::Machine(machine) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&machine.name.name, current_namespace.as_deref()),
                "machine",
                current_namespace.clone(),
                file_symbol_visibility(current_namespace.as_deref(), machine.exported),
                None,
                machine.name.span,
                source,
            ),
            Item::Actor(actor) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&actor.name.name, current_namespace.as_deref()),
                "actor",
                current_namespace.clone(),
                file_symbol_visibility(current_namespace.as_deref(), actor.exported),
                None,
                actor.name.span,
                source,
            ),
            Item::VarDecl(decl) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&decl.name.name, current_namespace.as_deref()),
                "variable",
                current_namespace.clone(),
                jett_resolve::scope::DefVisibility::Private,
                None,
                decl.name.span,
                source,
            ),
            Item::Verify(verify) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&verify.name.name, current_namespace.as_deref()),
                "verify",
                current_namespace.clone(),
                jett_resolve::scope::DefVisibility::Private,
                None,
                verify.name.span,
                source,
            ),
            Item::Property(prop) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&prop.name.name, current_namespace.as_deref()),
                "property",
                current_namespace.clone(),
                jett_resolve::scope::DefVisibility::Private,
                None,
                prop.name.span,
                source,
            ),
            Item::TypeAlias(alias) => push_file_symbol_query_entry(
                symbols,
                file_symbol_name(&alias.name.name, current_namespace.as_deref()),
                "type",
                current_namespace.clone(),
                file_symbol_visibility(
                    current_namespace.as_deref(),
                    alias.exported || alias.root_exported,
                ),
                None,
                alias.name.span,
                source,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_file_symbol_query_entry(
    symbols: &mut Vec<FileSymbolQueryEntry>,
    name: String,
    kind: &str,
    namespace: Option<String>,
    visibility: jett_resolve::scope::DefVisibility,
    signature: Option<String>,
    span: Span,
    source: &str,
) {
    let (line, column) = jett_diagnostics::render::line_col(source, span.start);
    let (end_line, end_column) = jett_diagnostics::render::line_col(source, span.end);
    symbols.push(FileSymbolQueryEntry {
        name,
        kind: kind.to_string(),
        namespace,
        visibility,
        signature,
        line: line as u32,
        column: column as u32,
        end_line: end_line as u32,
        end_column: end_column as u32,
    });
}

fn file_symbol_name(leaf_name: &str, namespace: Option<&str>) -> String {
    namespace
        .map(|namespace| format!("{namespace}.{leaf_name}"))
        .unwrap_or_else(|| leaf_name.to_string())
}

fn file_symbol_visibility(
    namespace: Option<&str>,
    exported: bool,
) -> jett_resolve::scope::DefVisibility {
    if namespace.is_none() || exported {
        jett_resolve::scope::DefVisibility::Public
    } else {
        jett_resolve::scope::DefVisibility::Private
    }
}

fn file_symbol_function_signature(
    leaf_name: &str,
    type_params: &[jett_parser::ast::Ident],
    params: &[Param],
    return_type: Option<&TypeExpr>,
    namespace: Option<&str>,
    file: FileId,
    file_paths: &HashMap<FileId, PathBuf>,
) -> String {
    let signature = signature_query_result(
        leaf_name,
        type_params,
        params,
        return_type,
        namespace,
        file,
        file_paths,
    );
    signature_display(&signature)
}

fn query_builtin_definitions() -> Vec<QueryDefinition> {
    use jett_resolve::scope::DefVisibility;

    let module = Module {
        items: Vec::new(),
        span: Span::new(FileId::new(0), 0, 0),
    };
    let resolve_result = resolve(&module);
    resolve_result
        .scope_table
        .definitions
        .iter()
        .filter(|def| def.visibility == DefVisibility::Public && query_surface_kind(def.kind))
        .map(|def| QueryDefinition {
            name: def.name.clone(),
            kind: def.kind,
            namespace: def.namespace.clone(),
            visibility: def.visibility,
            file_path: "builtin".to_string(),
            line: 0,
            column: 0,
            end_line: 0,
            end_column: 0,
        })
        .collect()
}

fn append_module_query_definitions(
    definitions: &mut Vec<QueryDefinition>,
    module: &Module,
    current_source: Option<&str>,
    file_paths: &HashMap<FileId, PathBuf>,
) {
    use jett_resolve::scope::DefKind;

    let mut current_namespace: Option<String> = None;
    for item in &module.items {
        match item {
            Item::Namespace(ns) => {
                current_namespace = Some(ns.name.name.clone());
                push_query_definition(
                    definitions,
                    ns.name.name.clone(),
                    DefKind::Namespace,
                    None,
                    ns.name.span,
                    current_source,
                    file_paths,
                );
            }
            Item::Function(func) => push_exported_query_definition(
                definitions,
                &func.name.name,
                DefKind::Function,
                current_namespace.as_deref(),
                func.exported,
                func.name.span,
                current_source,
                file_paths,
            ),
            Item::Mutual(block) => {
                for decl in &block.declarations {
                    push_exported_query_definition(
                        definitions,
                        &decl.name.name,
                        DefKind::Function,
                        current_namespace.as_deref(),
                        decl.exported,
                        decl.name.span,
                        current_source,
                        file_paths,
                    );
                }
            }
            Item::Interface(interface) => push_exported_query_definition(
                definitions,
                &interface.name.name,
                DefKind::Interface,
                current_namespace.as_deref(),
                interface.exported,
                interface.name.span,
                current_source,
                file_paths,
            ),
            Item::Struct(strukt) => push_exported_query_definition(
                definitions,
                &strukt.name.name,
                DefKind::Struct,
                current_namespace.as_deref(),
                strukt.exported,
                strukt.name.span,
                current_source,
                file_paths,
            ),
            Item::Bitfield(bitfield) => push_exported_query_definition(
                definitions,
                &bitfield.name.name,
                DefKind::Bitfield,
                current_namespace.as_deref(),
                bitfield.exported,
                bitfield.name.span,
                current_source,
                file_paths,
            ),
            Item::Enum(enm) => push_exported_query_definition(
                definitions,
                &enm.name.name,
                DefKind::Enum,
                current_namespace.as_deref(),
                enm.exported,
                enm.name.span,
                current_source,
                file_paths,
            ),
            Item::Machine(machine) => push_exported_query_definition(
                definitions,
                &machine.name.name,
                DefKind::Machine,
                current_namespace.as_deref(),
                machine.exported,
                machine.name.span,
                current_source,
                file_paths,
            ),
            Item::Actor(actor) => push_exported_query_definition(
                definitions,
                &actor.name.name,
                DefKind::Actor,
                current_namespace.as_deref(),
                actor.exported,
                actor.name.span,
                current_source,
                file_paths,
            ),
            Item::TypeAlias(alias) => {
                if alias.root_exported {
                    push_query_definition(
                        definitions,
                        alias.name.name.clone(),
                        DefKind::Type,
                        None,
                        alias.name.span,
                        current_source,
                        file_paths,
                    );
                }
                push_exported_query_definition(
                    definitions,
                    &alias.name.name,
                    DefKind::Type,
                    current_namespace.as_deref(),
                    alias.exported,
                    alias.name.span,
                    current_source,
                    file_paths,
                );
            }
            Item::Implement(_) | Item::VarDecl(_) | Item::Verify(_) | Item::Property(_) => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_exported_query_definition(
    definitions: &mut Vec<QueryDefinition>,
    leaf_name: &str,
    kind: jett_resolve::scope::DefKind,
    namespace: Option<&str>,
    exported: bool,
    span: Span,
    current_source: Option<&str>,
    file_paths: &HashMap<FileId, PathBuf>,
) {
    if namespace.is_some() && !exported {
        return;
    }

    let name = namespace
        .map(|namespace| format!("{namespace}.{leaf_name}"))
        .unwrap_or_else(|| leaf_name.to_string());
    push_query_definition(
        definitions,
        name,
        kind,
        namespace.map(str::to_string),
        span,
        current_source,
        file_paths,
    );
}

fn push_query_definition(
    definitions: &mut Vec<QueryDefinition>,
    name: String,
    kind: jett_resolve::scope::DefKind,
    namespace: Option<String>,
    span: Span,
    current_source: Option<&str>,
    file_paths: &HashMap<FileId, PathBuf>,
) {
    let (file_path, line, column, end_line, end_column) =
        span_location(current_source, span, file_paths)
            .unwrap_or_else(|| (query_file_path(span.file, file_paths), 0, 0, 0, 0));
    definitions.push(QueryDefinition {
        name,
        kind,
        namespace,
        visibility: jett_resolve::scope::DefVisibility::Public,
        file_path,
        line,
        column,
        end_line,
        end_column,
    });
}

fn query_surface_kind(kind: jett_resolve::scope::DefKind) -> bool {
    use jett_resolve::scope::DefKind;

    matches!(
        kind,
        DefKind::Function
            | DefKind::Interface
            | DefKind::Struct
            | DefKind::Bitfield
            | DefKind::Enum
            | DefKind::Machine
            | DefKind::Actor
            | DefKind::Type
            | DefKind::Constant
            | DefKind::Namespace
    )
}

fn query_file_path(file: FileId, file_paths: &HashMap<FileId, PathBuf>) -> String {
    file_paths
        .get(&file)
        .map(|path| display_query_path(path))
        .unwrap_or_else(|| "builtin".to_string())
}

fn display_query_path(path: &Path) -> String {
    let displayed = path.display().to_string();
    displayed
        .strip_prefix(r"\\?\")
        .unwrap_or(&displayed)
        .to_string()
}

fn best_resolved_definition_at(
    resolve_result: &jett_resolve::ResolveResult,
    file_id: FileId,
    offset: u32,
) -> Option<(u32, jett_resolve::scope::DefId)> {
    let mut best_def: Option<(u32, jett_resolve::scope::DefId)> = None;
    for (span, def_id) in &resolve_result.resolutions {
        if span.file == file_id && span.start <= offset && offset <= span.end {
            let len = span.end - span.start;
            if best_def.is_none() || len < best_def.unwrap().0 {
                best_def = Some((len, *def_id));
            }
        }
    }
    best_def
}

fn span_location(
    current_source: Option<&str>,
    span: Span,
    file_paths: &HashMap<FileId, PathBuf>,
) -> Option<(String, u32, u32, u32, u32)> {
    if span.start == 0 && span.end == 0 {
        return Some(("builtin".to_string(), 0, 0, 0, 0));
    }

    let source = if span.file == FileId::new(0) {
        Cow::Borrowed(current_source?)
    } else {
        Cow::Owned(fs::read_to_string(file_paths.get(&span.file)?).ok()?)
    };
    let (line, column) = jett_diagnostics::render::line_col(&source, span.start);
    let (end_line, end_column) = jett_diagnostics::render::line_col(&source, span.end);
    Some((
        query_file_path(span.file, file_paths),
        line as u32,
        column as u32,
        end_line as u32,
        end_column as u32,
    ))
}

/// Stable text label for a resolved definition kind.
pub fn query_kind_name(kind: jett_resolve::scope::DefKind) -> &'static str {
    use jett_resolve::scope::DefKind;

    match kind {
        DefKind::Function => "function",
        DefKind::Interface => "interface",
        DefKind::Struct => "struct",
        DefKind::Bitfield => "bitfield",
        DefKind::Enum => "enum",
        DefKind::Machine => "machine",
        DefKind::Actor => "actor",
        DefKind::Variable => "variable",
        DefKind::Param => "param",
        DefKind::Type => "type",
        DefKind::Constant => "constant",
        DefKind::Namespace => "namespace",
    }
}

/// Stable text label for a resolved definition visibility.
pub fn query_visibility_name(visibility: jett_resolve::scope::DefVisibility) -> &'static str {
    use jett_resolve::scope::DefVisibility;

    match visibility {
        DefVisibility::Public => "public",
        DefVisibility::Private => "private",
    }
}

/// Stable text label for a completion prefix match kind.
pub fn completion_match_kind_name(match_kind: CompletionMatchKind) -> &'static str {
    match match_kind {
        CompletionMatchKind::EmptyPrefix => "empty_prefix",
        CompletionMatchKind::Exact => "exact",
        CompletionMatchKind::QualifiedPrefix => "qualified_prefix",
        CompletionMatchKind::LeafPrefix => "leaf_prefix",
    }
}

fn support_modules_declare_namespace(modules: &[Module], namespace: &str) -> bool {
    modules.iter().any(|module| {
        module.items.iter().any(|item| match item {
            Item::Namespace(ns) => ns.span.file.is_stdlib() && ns.name.name == namespace,
            _ => false,
        })
    })
}

fn namespace_at_offset(module: &Module, file_id: FileId, offset: u32) -> Option<String> {
    let mut current_namespace = None;
    for item in &module.items {
        if item_file(item) != file_id {
            continue;
        }
        if item_span(item).start > offset {
            break;
        }
        if let Item::Namespace(ns) = item {
            current_namespace = Some(ns.name.name.clone());
        }
    }
    current_namespace
}

fn item_span(item: &Item) -> jett_common::Span {
    match item {
        Item::Namespace(ns) => ns.span,
        Item::Function(func) => func.span,
        Item::Mutual(block) => block.span,
        Item::Interface(interface) => interface.span,
        Item::Implement(block) => block.span,
        Item::Struct(strukt) => strukt.span,
        Item::Bitfield(bitfield) => bitfield.span,
        Item::Enum(enm) => enm.span,
        Item::Machine(machine) => machine.span,
        Item::Actor(actor) => actor.span,
        Item::VarDecl(decl) => decl.span,
        Item::Verify(verify) => verify.span,
        Item::Property(prop) => prop.span,
        Item::TypeAlias(alias) => alias.span,
    }
}

/// Return the byte span of the definition of the symbol at the given (1-based)
/// line and column in `source`.  Returns `None` if no definition is found.
pub fn goto_definition(source: &str, line: u32, col: u32) -> Option<(u32, u32)> {
    let file_id = FileId::new(0);

    let offset = line_col_to_offset(source, line, col)?;

    let mut parse_result = parse(source, file_id);
    if parse_result
        .errors
        .iter()
        .any(|d| d.severity == jett_diagnostics::Severity::Error)
    {
        return None;
    }

    prepend_support_modules(&mut parse_result.module, discover_stdlib_modules());

    let resolve_result = resolve(&parse_result.module);

    // Find the reference span that covers `offset`.
    let mut best_def: Option<(u32, jett_resolve::scope::DefId)> = None;
    for (span, def_id) in &resolve_result.resolutions {
        if span.file == file_id && span.start <= offset && offset <= span.end {
            let len = span.end - span.start;
            if best_def.is_none() || len < best_def.unwrap().0 {
                best_def = Some((len, *def_id));
            }
        }
    }

    best_def.and_then(|(_, def_id)| {
        let def_info = resolve_result.scope_table.def(def_id);
        if def_info.span.file == file_id {
            Some((def_info.span.start, def_info.span.end))
        } else {
            None
        }
    })
}

/// Convert a 1-based line+column to a byte offset in `source`.
fn line_col_to_offset(source: &str, line: u32, col: u32) -> Option<u32> {
    if line == 0 || col == 0 {
        return None;
    }
    let mut current_line = 1u32;
    let mut line_start = 0usize;
    for (i, ch) in source.char_indices() {
        if current_line == line {
            // col is 1-based within the line; advance col-1 chars.
            let col_offset = source[line_start..]
                .char_indices()
                .nth((col - 1) as usize)
                .map(|(o, _)| o)
                .unwrap_or(source.len() - line_start);
            return Some((line_start + col_offset) as u32);
        }
        if ch == '\n' {
            current_line += 1;
            line_start = i + 1;
        }
    }
    if current_line == line {
        return Some(line_start as u32);
    }
    None
}

/// Run the full compilation pipeline on a single file: lex → parse → resolve → typecheck.
/// Does not produce executable output yet — just validates the source.
pub fn build_file(path: &Path) -> BuildResult {
    build_file_inner(path, true)
}

fn build_file_inner(path: &Path, include_project: bool) -> BuildResult {
    let file_path_str = path.display().to_string();

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return BuildResult {
                diagnostics: vec![Diagnostic::error(
                    0,
                    format!("failed to read {}: {}", path.display(), e),
                    jett_common::Span::new(FileId::new(0), 0, 0),
                )],
                has_errors: true,
                source: String::new(),
                file_path: file_path_str,
                reflection_metadata: None,
                checked_expression_types: None,
            };
        }
    };

    let file_id = FileId::new(0);
    let mut all_diagnostics = Vec::new();

    // Phase 1+2: Lex + Parse (parse internally calls tokenize)
    let mut parse_result = parse(&source, file_id);
    all_diagnostics.extend(parse_result.errors.clone());

    // If there are parse errors, stop here — resolve/typecheck won't produce useful results
    let has_parse_errors = has_error_diagnostics(&all_diagnostics);
    if has_parse_errors {
        return BuildResult {
            has_errors: true,
            diagnostics: all_diagnostics,
            source,
            file_path: file_path_str,
            reflection_metadata: None,
            checked_expression_types: None,
        };
    }

    // Multi-file: prepend stdlib and sibling project modules so
    // resolver/typechecker can see cross-file definitions (functions, types,
    // etc.).
    let mut support_modules = discover_stdlib_modules_with_diagnostics();
    if include_project {
        support_modules.extend(discover_project_modules_with_diagnostics(path));
    }
    all_diagnostics.extend(support_modules.diagnostics);
    if has_error_diagnostics(&all_diagnostics) {
        return BuildResult {
            has_errors: true,
            diagnostics: all_diagnostics,
            source,
            file_path: file_path_str,
            reflection_metadata: None,
            checked_expression_types: None,
        };
    }
    prepend_support_modules(&mut parse_result.module, support_modules.modules);

    // Phase 3: Resolve names
    let resolve_result = resolve(&parse_result.module);
    all_diagnostics.extend(resolve_result.diagnostics.clone());

    let has_resolve_errors = has_error_diagnostics(&all_diagnostics);
    if has_resolve_errors {
        return BuildResult {
            has_errors: true,
            diagnostics: all_diagnostics,
            source,
            file_path: file_path_str,
            reflection_metadata: None,
            checked_expression_types: None,
        };
    }

    // Phase 4: Type check
    let check_result = check(&parse_result.module, &resolve_result);
    all_diagnostics.extend(check_result.diagnostics.clone());

    let has_typecheck_errors = has_error_diagnostics(&all_diagnostics);
    if has_typecheck_errors {
        return BuildResult {
            has_errors: true,
            diagnostics: all_diagnostics,
            source,
            file_path: file_path_str,
            reflection_metadata: None,
            checked_expression_types: None,
        };
    }

    // Phase 5: Execute verify blocks at compile time
    let reflection_metadata = check_result.reflection_metadata.clone();
    let checked_expression_types = Arc::new(expression_type_names(&check_result));
    let verify_diagnostics = run_verify_blocks_with_metadata_and_expression_types(
        &parse_result.module,
        check_result.reflection_metadata,
        checked_expression_types.clone(),
    );
    all_diagnostics.extend(verify_diagnostics);

    let has_errors = has_error_diagnostics(&all_diagnostics);

    BuildResult {
        has_errors,
        diagnostics: all_diagnostics,
        source,
        file_path: file_path_str,
        reflection_metadata: Some(reflection_metadata),
        checked_expression_types: Some(checked_expression_types),
    }
}

/// Register all items from a parsed module into an interpreter.
fn register_module_items(
    interp: &mut jett_comptime::interpreter::Interpreter,
    module: &jett_parser::ast::Module,
) {
    interp.register_module(module);
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

fn has_error_diagnostics(diagnostics: &[Diagnostic]) -> bool {
    diagnostics
        .iter()
        .any(|d| d.severity == jett_diagnostics::Severity::Error)
}

fn expression_type_names(check_result: &CheckResult) -> HashMap<Span, String> {
    check_result
        .type_map
        .iter()
        .map(|(span, ty_id)| (*span, check_result.interner.type_name(*ty_id)))
        .collect()
}

fn update_current_namespace(
    item: &Item,
    current_file: &mut Option<FileId>,
    current_namespace: &mut Option<String>,
) {
    let file = item_file(item);
    if current_file.is_some_and(|current| current != file) {
        *current_namespace = None;
    }
    *current_file = Some(file);

    if let Item::Namespace(ns) = item {
        *current_namespace = Some(ns.name.name.clone());
    }
}

fn find_main_function(module: &Module) -> Option<(Option<String>, &FunctionDef)> {
    let mut current_file = None;
    let mut current_namespace = None;

    for item in &module.items {
        update_current_namespace(item, &mut current_file, &mut current_namespace);
        if let Item::Function(func) = item
            && func.name.name == "main"
        {
            return Some((current_namespace.clone(), func));
        }
    }

    None
}

fn prepend_support_modules(module: &mut Module, support_modules: Vec<Module>) {
    if support_modules.is_empty() {
        return;
    }

    let mut merged_items = Vec::new();
    for support in support_modules {
        merged_items.extend(support.items);
    }
    merged_items.append(&mut module.items);
    module.items = merged_items;
}

/// Discover and parse compiler-shipped stdlib modules.
fn discover_stdlib_modules() -> Vec<Module> {
    discover_stdlib_modules_with_diagnostics().modules
}

fn discover_stdlib_modules_with_diagnostics() -> DiscoveredModules {
    discover_modules_in_dir(&stdlib_root(), None, STDLIB_FILE_ID_START, "stdlib")
}

fn stdlib_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("stdlib")
}

/// Discover and parse all sibling .jett files in the project (if a jett.proj exists).
/// Returns parsed modules for files other than the entry file.
fn discover_project_modules(entry_path: &Path) -> Vec<Module> {
    discover_project_modules_with_diagnostics(entry_path).modules
}

fn discover_project_modules_with_diagnostics(entry_path: &Path) -> DiscoveredModules {
    let canon = entry_path.canonicalize().ok();
    let project_root = find_project_root(entry_path).ok();
    let Some(root) = project_root else {
        return DiscoveredModules {
            modules: Vec::new(),
            diagnostics: Vec::new(),
            files: HashMap::new(),
        };
    };
    discover_modules_in_dir(&root, canon.as_deref(), 1, "project")
}

fn discover_query_project_modules_with_diagnostics(start_dir: &Path) -> DiscoveredModules {
    let Ok(root) = find_project_root(start_dir) else {
        return DiscoveredModules {
            modules: Vec::new(),
            diagnostics: Vec::new(),
            files: HashMap::new(),
        };
    };
    discover_modules_in_dir(&root, None, 1, "project")
}

fn discover_modules_in_dir(
    root: &Path,
    skip_canon: Option<&Path>,
    start_file_id: u32,
    module_kind: &str,
) -> DiscoveredModules {
    let mut files = Vec::new();
    if let Err(err) = collect_jett_files(root, &mut files) {
        return DiscoveredModules {
            modules: Vec::new(),
            diagnostics: vec![Diagnostic::error(
                0,
                format!(
                    "failed to scan {module_kind} modules in {}: {err}",
                    root.display()
                ),
                jett_common::Span::new(FileId::new(start_file_id), 0, 0),
            )],
            files: HashMap::new(),
        };
    }
    files.sort();

    let mut modules = Vec::new();
    let mut diagnostics = Vec::new();
    let mut module_files = HashMap::new();
    for (idx, file_path) in files.iter().enumerate() {
        // Skip the entry file when parsing project siblings.
        let should_skip = skip_canon
            .map(|skip| file_path.canonicalize().ok().as_deref() == Some(skip))
            .unwrap_or(false);
        if should_skip {
            continue;
        }
        let file_id = FileId::new(start_file_id + idx as u32);
        let source = match fs::read_to_string(file_path) {
            Ok(source) => source,
            Err(err) => {
                diagnostics.push(Diagnostic::error(
                    0,
                    format!(
                        "failed to read {module_kind} module {}: {err}",
                        file_path.display()
                    ),
                    jett_common::Span::new(file_id, 0, 0),
                ));
                continue;
            }
        };
        let parsed = parse(&source, file_id);
        if has_error_diagnostics(&parsed.errors) {
            for mut diagnostic in parsed.errors {
                if diagnostic.severity == jett_diagnostics::Severity::Error {
                    diagnostic.message = format!(
                        "failed to parse {module_kind} module {}: {}",
                        file_path.display(),
                        diagnostic.message
                    );
                    diagnostics.push(diagnostic);
                }
            }
        } else {
            let display_path = file_path
                .canonicalize()
                .unwrap_or_else(|_| file_path.clone());
            module_files.insert(file_id, display_path);
            modules.push(parsed.module);
        }
    }
    DiscoveredModules {
        modules,
        diagnostics,
        files: module_files,
    }
}

/// Run a .jett file using the tree-walking interpreter.
/// First validates (lex → parse → resolve → typecheck → verify), then executes main().
/// If a jett.proj exists, also loads sibling .jett files so cross-file calls work.
pub fn run_file(path: &Path) -> Result<(), String> {
    run_file_with_options(
        path,
        RunOptions {
            capture_stdout: false,
            emit_runtime_debug: true,
        },
    )
    .map(|_| ())
}

/// Run a .jett file and capture runtime stdout produced by `Stdout.write`,
/// `print`, and `println`.
pub fn run_file_capture_stdout(path: &Path) -> Result<String, String> {
    run_file_with_options(
        path,
        RunOptions {
            capture_stdout: true,
            emit_runtime_debug: false,
        },
    )
    .map(|output| output.stdout)
}

/// Run a .jett file and capture stdout plus trace/breakpoint debug lines.
pub fn run_file_capture_output(path: &Path) -> Result<RunOutput, String> {
    run_file_with_options(
        path,
        RunOptions {
            capture_stdout: true,
            emit_runtime_debug: false,
        },
    )
}

fn run_file_with_options(path: &Path, options: RunOptions) -> Result<RunOutput, String> {
    let thread_path = path.to_path_buf();
    let fallback_path = thread_path.clone();
    match thread::Builder::new()
        .name("jett-runtime".to_string())
        .stack_size(RUNTIME_STACK_SIZE)
        .spawn(move || run_file_inner(&thread_path, options))
    {
        Ok(handle) => match handle.join() {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        },
        Err(_) => run_file_inner(&fallback_path, options),
    }
}

fn run_file_inner(path: &Path, options: RunOptions) -> Result<RunOutput, String> {
    let build = build_file(path);

    if build.has_errors {
        let errors: Vec<String> = build
            .diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect();
        return Err(format!(
            "cannot run — compilation errors:\n{}",
            errors.join("\n")
        ));
    }

    // Parse again to get the module for interpretation
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let file_id = FileId::new(0);
    let parse_result = parse(&source, file_id);
    let module = parse_result.module;

    let Some((main_namespace, main_func)) = find_main_function(&module) else {
        return Err("runtime error: no `main` function found".to_string());
    };

    let main_args = default_runtime_args_for_main(main_func)?;

    use jett_comptime::interpreter::Interpreter;
    let mut interp = if options.emit_runtime_debug {
        Interpreter::new_runtime()
    } else {
        Interpreter::new()
    };
    if let Some(metadata) = build.reflection_metadata.clone() {
        interp.set_reflection_metadata(metadata);
    }
    if let Some(expression_types) = build.checked_expression_types.clone() {
        interp.set_checked_expression_types(expression_types);
    }
    if options.capture_stdout {
        interp.enable_stdout_capture();
    }

    // Register compiler-shipped stdlib modules before project and entry files.
    for module in discover_stdlib_modules() {
        register_module_items(&mut interp, &module);
    }

    // Register items from sibling project files first (so they're available to main file).
    let sibling_modules = discover_project_modules(path);
    for module in &sibling_modules {
        register_module_items(&mut interp, module);
    }

    // Register items from the entry file (may override sibling definitions).
    register_module_items(&mut interp, &module);

    // Call main()
    match interp.call_function_in_namespace(main_namespace.as_deref(), "main", main_args) {
        Ok(_) => Ok(RunOutput {
            stdout: interp.take_stdout_output(),
            debug_output: interp.take_debug_output(),
        }),
        Err(e) => Err(format!("runtime error: {}", e)),
    }
}

fn default_runtime_args_for_main(main: &FunctionDef) -> Result<Vec<Value>, String> {
    main.params
        .iter()
        .map(default_runtime_arg_for_param)
        .collect()
}

fn default_runtime_arg_for_param(param: &Param) -> Result<Value, String> {
    if type_expr_is_capability(&param.ty) {
        return Ok(Value::Nothing);
    }

    Err(format!(
        "runtime error: `main` parameter `{}` has unsupported type `{}`; only zero-argument or capability-only `main` functions can be run right now",
        param.name.name,
        type_expr_name(&param.ty)
    ))
}

fn type_expr_is_capability(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Named(ident) => matches!(
            ident.name.as_str(),
            "Stdout"
                | "Stderr"
                | "Stdin"
                | "Filesystem"
                | "Network"
                | "Clock"
                | "Random"
                | "Process"
                | "Environment"
        ),
        TypeExpr::View(inner, _) => type_expr_is_capability(inner),
        TypeExpr::StateQualified(inner, _, _) => type_expr_is_capability(inner),
        TypeExpr::Generic(_, _, _) => false,
        TypeExpr::Function(_, _, _) => false,
    }
}

fn type_expr_name(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named(ident) => ident.name.clone(),
        TypeExpr::Generic(name, args, _) => {
            let args: Vec<String> = args.iter().map(type_expr_name).collect();
            format!("{}[{}]", name.name, args.join(", "))
        }
        TypeExpr::View(inner, _) => format!("view {}", type_expr_name(inner)),
        TypeExpr::StateQualified(inner, state, _) => {
            format!("{} at {}", type_expr_name(inner), state.name)
        }
        TypeExpr::Function(params, ret, _) => {
            let params: Vec<String> = params.iter().map(type_expr_name).collect();
            format!(
                "function({}) returns {}",
                params.join(", "),
                type_expr_name(ret)
            )
        }
    }
}

/// Format a single .jett file and return the formatted source.
pub fn format_file(path: &Path) -> Result<FormatResult, String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

    let file_id = FileId::new(0);
    Ok(format_source(&source, file_id))
}

/// Format a .jett file in place (overwrite with formatted version).
pub fn format_file_in_place(path: &Path) -> Result<(), String> {
    let result = format_file(path)?;

    if !result.errors.is_empty() {
        return Err(format!(
            "cannot format {} — lexer errors:\n{}",
            path.display(),
            result.errors.join("\n")
        ));
    }

    fs::write(path, &result.output)
        .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// A single block result in a test run.
pub struct TestBlockResult {
    pub name: String,
    pub passed: bool,
    pub error: Option<String>,
    pub is_property: bool,
    pub iterations: Option<usize>,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

/// Result of running `jett test` on a single file.
pub struct TestResult {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    /// The file that was tested.
    pub file_path: String,
    /// Per-block results.
    pub blocks: Vec<TestBlockResult>,
}

/// Result of running `jett test` across an entire project.
pub struct ProjectTestResult {
    pub total_files: usize,
    pub total_blocks: usize,
    pub total_passed: usize,
    pub total_failed: usize,
    /// Per-file results.
    pub file_results: Vec<TestResult>,
}

/// One source file included in a generated bundle.
pub struct BundleFileResult {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// Result of bundling a project into one distributable file.
pub struct BundleResult {
    pub project_root: String,
    pub output_path: String,
    pub files: Vec<BundleFileResult>,
}

/// Parse a .jett file and run all verify blocks, reporting per-block results.
pub fn test_file(path: &Path) -> Result<TestResult, String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

    let file_id = FileId::new(0);
    let mut parse_result = parse(&source, file_id);

    // If there are parse errors, report and bail.
    let has_parse_errors = parse_result
        .errors
        .iter()
        .any(|d| d.severity == jett_diagnostics::Severity::Error);
    if has_parse_errors {
        let msgs: Vec<String> = parse_result
            .errors
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect();
        return Err(format!("parse errors:\n{}", msgs.join("\n")));
    }

    let mut support_modules = discover_stdlib_modules_with_diagnostics();
    support_modules.extend(discover_project_modules_with_diagnostics(path));
    let support_errors = error_messages_from_diagnostics(&support_modules.diagnostics);
    if !support_errors.is_empty() {
        return Err(format!(
            "support parse errors:\n{}",
            support_errors.join("\n")
        ));
    }
    strip_test_items_from_support_modules(&mut support_modules.modules);
    prepend_support_modules(&mut parse_result.module, support_modules.modules);

    let resolve_result = resolve(&parse_result.module);
    let resolve_errors = error_messages_from_diagnostics(&resolve_result.diagnostics);
    if !resolve_errors.is_empty() {
        return Err(format!("resolution errors:\n{}", resolve_errors.join("\n")));
    }

    let check_result = check(&parse_result.module, &resolve_result);
    let type_errors = error_messages_from_diagnostics(&check_result.diagnostics);
    if !type_errors.is_empty() {
        return Err(format!("type errors:\n{}", type_errors.join("\n")));
    }

    let checked_expression_types = Arc::new(expression_type_names(&check_result));
    let results = run_verify_blocks_detailed_with_metadata_and_expression_types(
        &parse_result.module,
        Some(check_result.reflection_metadata),
        Some(checked_expression_types),
    );

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    let blocks = results
        .into_iter()
        .map(|r| {
            let (line, column) = jett_diagnostics::render::line_col(&source, r.span.start);
            let (end_line, end_column) = jett_diagnostics::render::line_col(&source, r.span.end);
            TestBlockResult {
                name: r.name,
                passed: r.passed,
                error: r.error,
                is_property: r.is_property,
                iterations: r.iterations,
                line: line as u32,
                column: column as u32,
                end_line: end_line as u32,
                end_column: end_column as u32,
            }
        })
        .collect();

    Ok(TestResult {
        total,
        passed,
        failed,
        file_path: path.display().to_string(),
        blocks,
    })
}

fn strip_test_items_from_support_modules(modules: &mut [Module]) {
    for module in modules {
        module
            .items
            .retain(|item| !matches!(item, Item::Verify(_) | Item::Property(_)));
    }
}

fn error_messages_from_diagnostics(diagnostics: &[Diagnostic]) -> Vec<String> {
    diagnostics
        .iter()
        .filter(|d| d.severity == jett_diagnostics::Severity::Error)
        .map(|d| format!("{}: {}", d.code, d.message))
        .collect()
}

/// Discover all `.jett` files under a project root (walks up from `start_dir`
/// to find `jett.proj`, then collects all `.jett` files in the project) and
/// run verify blocks in each one.
pub fn test_project(start_dir: &Path) -> Result<ProjectTestResult, String> {
    let project_dir = find_project_root(start_dir)?;
    let mut files = Vec::new();
    collect_jett_files(&project_dir, &mut files)
        .map_err(|e| format!("error scanning project: {e}"))?;

    if files.is_empty() {
        return Err(format!(
            "no .jett files found in project at {}",
            project_dir.display()
        ));
    }

    files.sort();

    let mut file_results = Vec::new();
    for file_path in &files {
        file_results.push(test_file(file_path)?);
    }

    let total_files = file_results.len();
    let total_blocks: usize = file_results.iter().map(|r| r.total).sum();
    let total_passed: usize = file_results.iter().map(|r| r.passed).sum();
    let total_failed: usize = file_results.iter().map(|r| r.failed).sum();

    Ok(ProjectTestResult {
        total_files,
        total_blocks,
        total_passed,
        total_failed,
        file_results,
    })
}

/// Bundle all project `.jett` files into a single file, then validate it before
/// writing the output path.
pub fn bundle_project(start_dir: &Path, output: &Path) -> Result<BundleResult, String> {
    let project_dir = find_project_root(start_dir)?;
    let output_abs = if output.is_absolute() {
        output.to_path_buf()
    } else {
        project_dir.join(output)
    };

    let mut files = Vec::new();
    collect_jett_files(&project_dir, &mut files)
        .map_err(|e| format!("error scanning project: {e}"))?;
    files.sort();
    files.retain(|path| path != &output_abs);

    if files.is_empty() {
        return Err(format!(
            "no .jett files found in project at {}",
            project_dir.display()
        ));
    }

    let mut bundled = String::new();
    let mut current_line = 1_u32;
    bundled.push_str("# Generated by jett bundle.\n");
    bundled.push_str(&format!("# Project root: {}\n\n", project_dir.display()));
    current_line += 3;

    let mut bundled_files = Vec::new();
    for file in &files {
        let source = fs::read_to_string(file)
            .map_err(|e| format!("failed to read {}: {}", file.display(), e))?;
        let relative = file.strip_prefix(&project_dir).unwrap_or(file);
        bundled.push_str(&format!("# --- file: {} ---\n", relative.display()));
        current_line += 1;

        let start_line = current_line;
        let source_line_count = source.lines().count().max(1) as u32;
        bundled.push_str(&source);
        if !source.ends_with('\n') {
            bundled.push('\n');
        }
        let end_line = start_line + source_line_count - 1;
        current_line = end_line + 1;
        bundled.push('\n');
        current_line += 1;

        bundled_files.push(BundleFileResult {
            path: display_query_path(relative),
            start_line,
            end_line,
        });
    }

    let validation = build_source(&bundled, &output_abs.display().to_string());
    if validation.has_errors {
        let errors = error_messages_from_diagnostics(&validation.diagnostics);
        return Err(format!(
            "candidate bundle failed validation:\n{}",
            errors.join("\n")
        ));
    }

    if let Some(parent) = output_abs.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {}", parent.display(), e))?;
    }
    fs::write(&output_abs, bundled)
        .map_err(|e| format!("failed to write {}: {}", output_abs.display(), e))?;

    Ok(BundleResult {
        project_root: project_dir.display().to_string(),
        output_path: output_abs.display().to_string(),
        files: bundled_files,
    })
}

// ---------------------------------------------------------------------------
// Helpers — project file discovery for `jett test`
// ---------------------------------------------------------------------------

/// Walk up from `start_dir` to find a directory containing `jett.proj`.
fn find_project_root(start_dir: &Path) -> Result<std::path::PathBuf, String> {
    let start = if start_dir.is_file() {
        start_dir.parent().unwrap_or(start_dir).to_path_buf()
    } else {
        start_dir.to_path_buf()
    };

    let mut current = start.as_path();
    loop {
        if current.join("jett.proj").exists() {
            return Ok(current.to_path_buf());
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => {
                return Err("no jett.proj found in current directory or any parent".to_string());
            }
        }
    }
}

/// Recursively collect all `.jett` files in a directory, skipping hidden dirs
/// and `target/`.
fn collect_jett_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !dir_name.starts_with('.') && dir_name != "target" {
                collect_jett_files(&path, out)?;
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("jett") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{nanos}"))
    }

    #[test]
    fn support_module_parse_errors_are_reported() {
        let root = temp_test_dir("jett_driver_support_parse_errors");
        fs::create_dir_all(&root).expect("temp support dir should be created");
        let broken = root.join("broken.jett");
        fs::write(&broken, "namespace broken\nfunction nope(\n")
            .expect("broken support fixture should be written");

        let discovered = discover_modules_in_dir(&root, None, STDLIB_FILE_ID_START, "stdlib");
        let errors = error_messages_from_diagnostics(&discovered.diagnostics);

        fs::remove_dir_all(&root).expect("temp support dir should be removed");

        assert!(
            discovered.modules.is_empty(),
            "parse-broken support file should not be loaded"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("failed to parse stdlib module")
                    && error.contains("broken.jett")),
            "expected support parse diagnostic to mention the broken stdlib module, got {errors:?}"
        );
        assert!(
            errors.iter().all(|error| !error.contains("undefined name")),
            "support parse errors should surface before resolver fallout, got {errors:?}"
        );
    }

    #[test]
    fn query_namespaces_lists_public_project_definitions() {
        let root = temp_test_dir("jett_driver_query_namespaces");
        fs::create_dir_all(&root).expect("temp project dir should be created");
        fs::write(root.join("jett.proj"), "name: query_fixture\n")
            .expect("project marker should be written");
        fs::write(
            root.join("api.jett"),
            "namespace api\n\nexport function login() returns int64:\n    return 1\n\nfunction hidden() returns int64:\n    return 2\n\nexport struct User:\n    id: int64\n",
        )
        .expect("query fixture should be written");

        let result = query_namespaces(&root).expect("query should succeed");

        fs::remove_dir_all(&root).expect("temp project dir should be removed");

        assert!(
            result
                .definitions
                .iter()
                .any(|def| def.name == "api" && query_kind_name(def.kind) == "namespace"),
            "expected namespace row in query result"
        );
        let login = result
            .definitions
            .iter()
            .find(|def| def.name == "api.login")
            .expect("expected exported function row in query result");
        assert_eq!(query_kind_name(login.kind), "function");
        assert_eq!(login.namespace.as_deref(), Some("api"));
        assert_eq!(
            (login.line, login.column, login.end_line, login.end_column),
            (3, 17, 3, 22)
        );

        let user = result
            .definitions
            .iter()
            .find(|def| def.name == "api.User")
            .expect("expected exported struct row in query result");
        assert_eq!(query_kind_name(user.kind), "struct");
        assert_eq!(user.namespace.as_deref(), Some("api"));
        assert_eq!(
            (user.line, user.column, user.end_line, user.end_column),
            (9, 15, 9, 19)
        );
        assert!(
            result
                .definitions
                .iter()
                .all(|def| def.name != "api.hidden"),
            "private namespaced definitions should not appear in global query results"
        );
    }

    #[test]
    fn query_file_symbols_lists_private_and_public_declarations() {
        let root = temp_test_dir("jett_driver_query_file_symbols");
        fs::create_dir_all(&root).expect("temp query dir should be created");
        let file = root.join("api.jett");
        fs::write(
            &file,
            "namespace api\n\nexport function login() returns int64:\n    return 1\n\nfunction hidden() returns int64:\n    return 2\n\nverify api_checks:\n    assert login() == 1\n",
        )
        .expect("symbols fixture should be written");

        let result = query_file_symbols(&file).expect("symbols query should succeed");

        fs::remove_dir_all(&root).expect("temp query dir should be removed");

        let login = result
            .symbols
            .iter()
            .find(|symbol| symbol.name == "api.login")
            .expect("expected exported function in file symbols");
        assert_eq!(login.kind, "function");
        assert_eq!(login.visibility, jett_resolve::scope::DefVisibility::Public);
        assert_eq!(
            (login.line, login.column, login.end_line, login.end_column),
            (3, 17, 3, 22)
        );

        let hidden = result
            .symbols
            .iter()
            .find(|symbol| symbol.name == "api.hidden")
            .expect("expected private function in file symbols");
        assert_eq!(hidden.kind, "function");
        assert_eq!(
            hidden.visibility,
            jett_resolve::scope::DefVisibility::Private
        );
        assert_eq!(
            hidden.signature.as_deref(),
            Some("api.hidden() returns int64")
        );
        assert_eq!(
            (
                hidden.line,
                hidden.column,
                hidden.end_line,
                hidden.end_column
            ),
            (6, 10, 6, 16)
        );
        assert!(
            result
                .symbols
                .iter()
                .any(|symbol| symbol.name == "api.api_checks" && symbol.kind == "verify"),
            "expected verify block in file symbols, got {:?}",
            result.symbols
        );
    }

    #[test]
    fn query_type_at_returns_type_for_file_position() {
        let root = temp_test_dir("jett_driver_query_type_at");
        fs::create_dir_all(&root).expect("temp query dir should be created");
        let file = root.join("main.jett");
        fs::write(
            &file,
            "namespace app\n\nfunction main() returns nothing:\n    int64 total = 1 + 2\n    return nothing\n",
        )
        .expect("query type fixture should be written");

        let result = query_type_at(&file, 4, 19).expect("type query should succeed");

        fs::remove_dir_all(&root).expect("temp query dir should be removed");

        assert_eq!(result.type_name, Some("int64".to_string()));
        assert_eq!(
            (
                result.span_line,
                result.span_column,
                result.span_end_line,
                result.span_end_column
            ),
            (Some(4), Some(19), Some(4), Some(20))
        );
    }

    #[test]
    fn query_definition_at_returns_cross_file_definition() {
        let root = temp_test_dir("jett_driver_query_definition_at");
        fs::create_dir_all(&root).expect("temp query dir should be created");
        fs::write(root.join("jett.proj"), "name: query_fixture\n")
            .expect("project marker should be written");
        let models = root.join("models.jett");
        fs::write(
            &models,
            "namespace models\n\nexport struct User:\n    id: int64\n",
        )
        .expect("models fixture should be written");
        let file = root.join("main.jett");
        fs::write(
            &file,
            "namespace app\n\nfunction make() returns models.User:\n    models.User user = models.User(id: 1)\n    return user\n",
        )
        .expect("main fixture should be written");

        let result = query_definition_at(&file, 4, 12).expect("definition query should succeed");

        fs::remove_dir_all(&root).expect("temp query dir should be removed");

        let target = result.target.expect("definition target should be found");
        assert_eq!(target.name, "models.User");
        assert_eq!(query_kind_name(target.kind), "struct");
        assert_eq!(target.namespace.as_deref(), Some("models"));
        assert!(
            target.file_path.ends_with("models.jett"),
            "expected target file to be models.jett, got {}",
            target.file_path
        );
        assert_eq!(
            (
                target.line,
                target.column,
                target.end_line,
                target.end_column
            ),
            (3, 15, 3, 19)
        );
    }

    #[test]
    fn query_references_at_returns_cross_file_use_sites() {
        let root = temp_test_dir("jett_driver_query_references_at");
        fs::create_dir_all(&root).expect("temp query dir should be created");
        fs::write(root.join("jett.proj"), "name: query_fixture\n")
            .expect("project marker should be written");
        fs::write(
            root.join("util.jett"),
            "namespace util\n\nexport function helper(n: int64) returns int64:\n    return n\n",
        )
        .expect("support fixture should be written");
        let file = root.join("main.jett");
        fs::write(
            &file,
            "namespace app\n\nfunction main() returns int64:\n    int64 a = util.helper(1)\n    int64 b = util.helper(2)\n    return a + b\n",
        )
        .expect("main fixture should be written");

        let result = query_references_at(&file, 4, 15).expect("references query should succeed");

        fs::remove_dir_all(&root).expect("temp query dir should be removed");

        let target = result.target.expect("reference target should be found");
        assert_eq!(target.name, "util.helper");
        assert_eq!(query_kind_name(target.kind), "function");
        assert_eq!(result.references.len(), 2);
        assert!(
            result.references.iter().any(|reference| (
                reference.line,
                reference.column,
                reference.end_line,
                reference.end_column
            ) == (4, 15, 4, 26)),
            "expected first call site in references, got {:?}",
            result.references
        );
        assert!(
            result.references.iter().any(|reference| (
                reference.line,
                reference.column,
                reference.end_line,
                reference.end_column
            ) == (5, 15, 5, 26)),
            "expected second call site in references, got {:?}",
            result.references
        );
    }

    #[test]
    fn query_completions_at_includes_project_definitions() {
        let root = temp_test_dir("jett_driver_query_completions_at");
        fs::create_dir_all(&root).expect("temp query dir should be created");
        fs::write(root.join("jett.proj"), "name: query_fixture\n")
            .expect("project marker should be written");
        fs::write(
            root.join("util.jett"),
            "namespace util\n\nexport function helper() returns int64:\n    return 1\n",
        )
        .expect("support fixture should be written");
        let file = root.join("main.jett");
        fs::write(
            &file,
            "namespace app\n\nfunction main() returns nothing:\n    return nothing\n",
        )
        .expect("main fixture should be written");

        let result = query_completions_at(&file, 4, 5).expect("completion query should succeed");

        fs::remove_dir_all(&root).expect("temp query dir should be removed");

        assert_eq!(result.prefix, "");
        let helper = result
            .candidates
            .iter()
            .find(|candidate| {
                candidate.name == "util.helper" && query_kind_name(candidate.kind) == "function"
            })
            .expect("expected completion query to include exported project helper");
        assert_eq!(helper.namespace.as_deref(), Some("util"));
        assert_eq!(
            helper.visibility,
            jett_resolve::scope::DefVisibility::Public
        );
        assert!(
            helper.file_path.ends_with("util.jett"),
            "expected helper source file, got {}",
            helper.file_path
        );
        assert_eq!(
            (
                helper.line,
                helper.column,
                helper.end_line,
                helper.end_column
            ),
            (3, 17, 3, 23)
        );
        assert_eq!(helper.match_kind, CompletionMatchKind::EmptyPrefix);
        assert_eq!(helper.rank, 100);
    }

    #[test]
    fn query_completions_at_filters_by_cursor_prefix() {
        let root = temp_test_dir("jett_driver_query_completions_prefix");
        fs::create_dir_all(&root).expect("temp query dir should be created");
        fs::write(root.join("jett.proj"), "name: query_fixture\n")
            .expect("project marker should be written");
        fs::write(
            root.join("util.jett"),
            "namespace util\n\nexport function helper(n: int64) returns int64:\n    return n\n",
        )
        .expect("support fixture should be written");
        fs::write(
            root.join("other.jett"),
            "namespace other\n\nexport function helper(n: int64) returns int64:\n    return n\n",
        )
        .expect("other fixture should be written");
        let file = root.join("main.jett");
        fs::write(
            &file,
            "namespace app\n\nfunction main() returns int64:\n    int64 value = util.helper(1)\n    return value\n",
        )
        .expect("main fixture should be written");

        let result = query_completions_at(&file, 4, 21).expect("completion query should succeed");

        fs::remove_dir_all(&root).expect("temp query dir should be removed");

        assert_eq!(result.prefix, "ut");
        assert!(
            result
                .candidates
                .iter()
                .any(|candidate| candidate.name == "util.helper"),
            "expected util.helper for prefix `ut`, got {:?}",
            result.candidates
        );
        let util_helper = result
            .candidates
            .iter()
            .find(|candidate| candidate.name == "util.helper")
            .expect("expected util.helper candidate");
        assert_eq!(
            (
                util_helper.line,
                util_helper.column,
                util_helper.end_line,
                util_helper.end_column
            ),
            (3, 17, 3, 23)
        );
        assert_eq!(util_helper.match_kind, CompletionMatchKind::QualifiedPrefix);
        assert_eq!(util_helper.rank, 10);
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| candidate.name.starts_with("ut")
                    || candidate
                        .name
                        .rsplit_once('.')
                        .is_some_and(|(_, leaf)| leaf.starts_with("ut"))),
            "expected all candidates to match prefix `ut`, got {:?}",
            result.candidates
        );
    }

    #[test]
    fn query_completions_at_ranks_leaf_prefix_matches() {
        let root = temp_test_dir("jett_driver_query_completions_leaf_prefix");
        fs::create_dir_all(&root).expect("temp query dir should be created");
        fs::write(root.join("jett.proj"), "name: query_fixture\n")
            .expect("project marker should be written");
        fs::write(
            root.join("util.jett"),
            "namespace util\n\nexport function helper(n: int64) returns int64:\n    return n\n",
        )
        .expect("support fixture should be written");
        let file = root.join("main.jett");
        fs::write(
            &file,
            "namespace app\n\nfunction main() returns int64:\n    int64 value = hel\n    return value\n",
        )
        .expect("main fixture should be written");

        let result = query_completions_at(&file, 4, 22).expect("completion query should succeed");

        fs::remove_dir_all(&root).expect("temp query dir should be removed");

        assert_eq!(result.prefix, "hel");
        let helper = result
            .candidates
            .iter()
            .find(|candidate| candidate.name == "util.helper")
            .expect("expected util.helper candidate");
        assert_eq!(
            (
                helper.line,
                helper.column,
                helper.end_line,
                helper.end_column
            ),
            (3, 17, 3, 23)
        );
        assert_eq!(helper.match_kind, CompletionMatchKind::LeafPrefix);
        assert_eq!(helper.rank, 20);
    }

    #[test]
    fn query_signature_reports_stdlib_function_signature() {
        let result = query_signature(Path::new("."), "json.parse")
            .expect("signature query should succeed")
            .expect("json.parse signature should be found");

        assert_eq!(result.name, "json.parse");
        assert_eq!(result.type_params, vec!["T".to_string()]);
        assert_eq!(result.params.len(), 1);
        assert_eq!(result.params[0].name, "raw");
        assert_eq!(result.params[0].type_name, "string");
        assert_eq!(result.return_type, "result[T, string]");

        let raw_result = query_signature(Path::new("."), "json.parse_raw")
            .expect("signature query should succeed")
            .expect("json.parse_raw signature should be found");
        assert_eq!(raw_result.return_type, "result[json.JsonTree, string]");
    }

    #[test]
    fn bundle_project_writes_validated_single_file() {
        let root = temp_test_dir("jett_driver_bundle_project");
        fs::create_dir_all(root.join("src")).expect("temp bundle dir should be created");
        fs::write(root.join("jett.proj"), "name: bundle_fixture\n")
            .expect("project marker should be written");
        fs::write(
            root.join("src").join("core.jett"),
            "namespace core\n\nexport function answer() returns int64:\n    return 42\n",
        )
        .expect("bundle source should be written");
        let output = root.join("dist").join("lib.jett");

        let result = bundle_project(&root, &output).expect("bundle should succeed");

        let bundled = fs::read_to_string(&output).expect("bundle output should be readable");
        fs::remove_dir_all(&root).expect("temp bundle dir should be removed");

        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].path.replace('\\', "/"), "src/core.jett");
        assert_eq!(result.files[0].start_line, 5);
        assert_eq!(result.files[0].end_line, 8);
        assert!(bundled.contains("# Generated by jett bundle."));
        assert!(bundled.contains("namespace core"));
        assert!(
            result
                .output_path
                .replace('\\', "/")
                .ends_with("dist/lib.jett")
        );
    }
}
