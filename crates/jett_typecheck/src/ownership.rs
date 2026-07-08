use std::collections::HashMap;

use jett_common::{Span, is_json_implicit_view_facade};
use jett_diagnostics::Diagnostic;
use jett_parser::ast::{self, Block, CallArg, Expr, FunctionDef, Item, Module, Stmt, StringPart};
use jett_types::{Type, TypeId, TypeInterner};

// ---------------------------------------------------------------------------
// Ownership state
// ---------------------------------------------------------------------------

/// Tracks the ownership state of a variable through the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipState {
    /// The variable holds an owned value.
    Owned,
    /// The variable is a `view` (read-only borrow).
    Viewed,
    /// The variable has been moved/consumed and is no longer valid.
    Consumed,
    /// The variable was produced by `run` and cannot be used until `join`ed.
    Pending,
    /// The variable has not been assigned yet.
    Uninitialized,
}

// ---------------------------------------------------------------------------
// Variable info tracked by the ownership checker
// ---------------------------------------------------------------------------

/// Information about a tracked variable.
#[derive(Debug, Clone)]
struct VarInfo {
    state: OwnershipState,
    /// Whether the variable was declared `mutable`.
    mutable: bool,
    /// The type of the variable (used to determine if it's a copyable primitive).
    type_id: TypeId,
    /// The span where the variable was consumed (for "previously consumed here" labels).
    consumed_span: Option<Span>,
}

// ---------------------------------------------------------------------------
// Ownership error constructors (E0400–E0499)
// ---------------------------------------------------------------------------

/// E0400: Use after move — a variable was consumed and then used again.
fn use_after_move(name: &str, use_span: Span, consumed_span: Span) -> Diagnostic {
    Diagnostic::error(
        400,
        format!("`{name}` was consumed and cannot be used again"),
        use_span,
    )
    .with_label(consumed_span, format!("`{name}` was consumed here"))
}

/// E0401: Cannot consume a view parameter — view parameters are read-only.
fn cannot_consume_view(name: &str, span: Span) -> Diagnostic {
    Diagnostic::error(
        401,
        format!("cannot consume `{name}` because it is a view parameter"),
        span,
    )
}

// ---------------------------------------------------------------------------
// Ownership checker
// ---------------------------------------------------------------------------

/// Performs ownership analysis (linear type checking) on a type-checked module.
///
/// This is Phase C from the architecture: tracking move/view/consume states for
/// variables. It runs after type checking completes and adds any ownership errors
/// to the diagnostic list.
pub struct OwnershipChecker<'a> {
    /// Maps variable names to their ownership info within the current scope.
    states: HashMap<String, VarInfo>,
    /// Collected diagnostics.
    diagnostics: Vec<Diagnostic>,
    /// The type interner, needed to check whether a type is a copyable primitive.
    interner: &'a TypeInterner,
}

impl<'a> OwnershipChecker<'a> {
    pub fn new(interner: &'a TypeInterner) -> Self {
        Self {
            states: HashMap::new(),
            diagnostics: Vec::new(),
            interner,
        }
    }

    /// Run ownership analysis on a module and return collected diagnostics.
    pub fn check_module(mut self, module: &Module) -> Vec<Diagnostic> {
        for item in &module.items {
            match item {
                Item::Function(func) => self.check_function(func),
                Item::VarDecl(decl) => self.check_var_decl(decl),
                _ => {}
            }
        }
        self.diagnostics
    }

    // ------------------------------------------------------------------
    // Primitives: implicitly copyable types
    // ------------------------------------------------------------------

    /// Returns `true` if the type is a primitive that is implicitly copyable
    /// and therefore not subject to linear consumption rules.
    fn is_copyable(&self, type_id: TypeId) -> bool {
        matches!(
            self.interner.resolve(type_id),
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
                | Type::Error
        )
    }

    // ------------------------------------------------------------------
    // Function
    // ------------------------------------------------------------------

    fn check_function(&mut self, func: &FunctionDef) {
        // Save the outer scope and start a fresh scope for this function.
        let saved = std::mem::take(&mut self.states);

        // Register parameters.
        for param in &func.params {
            let type_id = self.resolve_type_for_ownership(&param.ty);
            let state = if param.view {
                OwnershipState::Viewed
            } else {
                OwnershipState::Owned
            };
            self.states.insert(
                param.name.name.clone(),
                VarInfo {
                    state,
                    mutable: param.mutable,
                    type_id,
                    consumed_span: None,
                },
            );
        }

        self.check_block(&func.body);

        // Restore the outer scope.
        self.states = saved;
    }

    // ------------------------------------------------------------------
    // Block
    // ------------------------------------------------------------------

    fn check_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.check_stmt(stmt);
            if Self::stmt_definitely_exits(stmt) {
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------------

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::VarDecl(decl) => self.check_var_decl(decl),
            Stmt::Assign(assign) => self.check_assign(assign),
            Stmt::Return(ret) => self.check_return(ret),
            Stmt::ComptimeTypeBind(bind) => self.check_block(&bind.body),
            Stmt::If(if_stmt) => self.check_if(if_stmt),
            Stmt::For(for_stmt) => self.check_for(for_stmt),
            Stmt::While(while_stmt) => self.check_while(while_stmt),
            Stmt::Expr(expr_stmt) => {
                self.check_expr_ownership(&expr_stmt.expr);
            }
            Stmt::Assert(assert_stmt) => self.check_assert(assert_stmt),
            Stmt::Trace(trace_stmt) => self.check_trace(trace_stmt),
            Stmt::Breakpoint(breakpoint_stmt) => self.check_breakpoint(breakpoint_stmt),
            Stmt::Match(match_stmt) => self.check_match(match_stmt),
            Stmt::Respond(resp) => {
                self.check_expr_ownership(&resp.value);
            }
            Stmt::Use(_) | Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }

    fn check_trace(&mut self, trace_stmt: &ast::TraceStmt) {
        let expr = Expr::Ident(trace_stmt.name.clone());
        self.check_expr_ownership(&expr);
    }

    fn check_breakpoint(&mut self, breakpoint_stmt: &ast::BreakpointStmt) {
        if let Some(condition) = &breakpoint_stmt.condition {
            self.check_expr_ownership(condition);
        }
    }

    fn check_var_decl(&mut self, decl: &ast::VarDecl) {
        // Check the initializer expression for ownership violations.
        self.check_expr_ownership(&decl.value);

        let type_id = self.resolve_type_for_ownership(&decl.ty);
        self.states.insert(
            decl.name.name.clone(),
            VarInfo {
                state: OwnershipState::Owned,
                mutable: decl.mutable,
                type_id,
                consumed_span: None,
            },
        );
    }

    fn check_assign(&mut self, assign: &ast::AssignStmt) {
        // Check the value expression.
        self.check_expr_ownership(&assign.value);

        // If the target is an identifier, check if it's mutable and handle rebinding.
        if let Expr::Ident(ident) = &assign.target {
            if let Some(info) = self.states.get_mut(&ident.name)
                && info.mutable
            {
                // Mutable variable: rebinding resets state to Owned.
                info.state = OwnershipState::Owned;
                info.consumed_span = None;
            }
        } else {
            self.check_expr_ownership(&assign.target);
        }
    }

    fn check_return(&mut self, ret: &ast::ReturnStmt) {
        if let Some(expr) = &ret.value {
            self.check_expr_ownership(expr);
        }
    }

    fn check_if(&mut self, if_stmt: &ast::IfStmt) {
        self.check_expr_ownership(&if_stmt.condition);
        let mut condition_state = self.states.clone();
        let mut fallthrough_states = Vec::new();

        let then_state = self.check_block_from_state(&condition_state, &if_stmt.then_block);
        if Self::block_can_fall_through(&if_stmt.then_block) {
            fallthrough_states.push(then_state);
        }

        for (cond, block) in &if_stmt.else_ifs {
            self.states = condition_state.clone();
            self.check_expr_ownership(cond);
            condition_state = self.states.clone();
            let branch_state = self.check_block_from_state(&condition_state, block);
            if Self::block_can_fall_through(block) {
                fallthrough_states.push(branch_state);
            }
        }

        if let Some(else_block) = &if_stmt.else_block {
            let else_state = self.check_block_from_state(&condition_state, else_block);
            if Self::block_can_fall_through(else_block) {
                fallthrough_states.push(else_state);
            }
        } else {
            fallthrough_states.push(condition_state.clone());
        }

        self.states = self.merge_fallthrough_states(&condition_state, &fallthrough_states);
    }

    fn check_block_from_state(
        &mut self,
        state: &HashMap<String, VarInfo>,
        block: &Block,
    ) -> HashMap<String, VarInfo> {
        self.states = state.clone();
        self.check_block(block);
        self.states.clone()
    }

    fn merge_fallthrough_states(
        &self,
        baseline: &HashMap<String, VarInfo>,
        branches: &[HashMap<String, VarInfo>],
    ) -> HashMap<String, VarInfo> {
        if branches.is_empty() {
            return baseline.clone();
        }

        let mut merged = baseline.clone();
        for (name, baseline_info) in baseline {
            let mut branch_infos = branches
                .iter()
                .filter_map(|branch| branch.get(name))
                .peekable();
            let Some(first_info) = branch_infos.peek().copied() else {
                continue;
            };
            let mut result = first_info.clone();
            for branch_info in branch_infos {
                if branch_info.state == OwnershipState::Consumed {
                    result.state = OwnershipState::Consumed;
                    result.consumed_span = branch_info.consumed_span;
                    break;
                }
                if result.state != OwnershipState::Consumed {
                    result.state = branch_info.state;
                    result.consumed_span = branch_info.consumed_span;
                }
            }
            if branches.iter().any(|branch| !branch.contains_key(name)) {
                result = baseline_info.clone();
            }
            merged.insert(name.clone(), result);
        }
        merged
    }

    fn block_can_fall_through(block: &Block) -> bool {
        !block.stmts.iter().any(Self::stmt_definitely_exits)
    }

    fn stmt_definitely_exits(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Return(_) | Stmt::Respond(_) => true,
            Stmt::If(if_stmt) => {
                if if_stmt.else_block.is_none() {
                    return false;
                }
                !Self::block_can_fall_through(&if_stmt.then_block)
                    && if_stmt
                        .else_ifs
                        .iter()
                        .all(|(_, block)| !Self::block_can_fall_through(block))
                    && if_stmt
                        .else_block
                        .as_ref()
                        .is_some_and(|block| !Self::block_can_fall_through(block))
            }
            _ => false,
        }
    }

    fn check_for(&mut self, for_stmt: &ast::ForStmt) {
        // Check the iterable expression.
        // If `view` is used, the iterable stays owned; otherwise it is consumed.
        if for_stmt.view {
            // `for item in view items:` — just check that the iterable is usable,
            // don't consume it.
            self.check_expr_ownership(&for_stmt.iterable);
        } else {
            // `for item in items:` — this consumes the iterable.
            self.consume_expr(&for_stmt.iterable, for_stmt.iterable.span());
        }

        // The loop variable is a fresh owned (or viewed) binding.
        // We don't know the exact type here without the type map, so we use ERROR
        // as a placeholder — it will be treated as copyable for ownership purposes.
        self.states.insert(
            for_stmt.variable.name.clone(),
            VarInfo {
                state: if for_stmt.view {
                    OwnershipState::Viewed
                } else {
                    OwnershipState::Owned
                },
                mutable: false,
                type_id: TypeInterner::ERROR, // Element type not tracked here
                consumed_span: None,
            },
        );

        self.check_block(&for_stmt.body);
    }

    fn check_while(&mut self, while_stmt: &ast::WhileStmt) {
        self.check_expr_ownership(&while_stmt.condition);
        self.check_block(&while_stmt.body);
    }

    fn check_assert(&mut self, assert_stmt: &ast::AssertStmt) {
        self.check_expr_ownership(&assert_stmt.condition);
        if let Some(msg) = &assert_stmt.message {
            self.check_expr_ownership(msg);
        }
    }

    fn check_match(&mut self, match_stmt: &ast::MatchStmt) {
        self.check_expr_ownership(&match_stmt.expr);
        for arm in &match_stmt.arms {
            self.check_block(&arm.body);
        }
    }

    // ------------------------------------------------------------------
    // Expression ownership checking
    // ------------------------------------------------------------------

    /// Check an expression for ownership violations. This is a read-only
    /// traversal that verifies variable states are valid but does NOT consume
    /// anything (consumption happens explicitly through `consume_expr`).
    fn check_expr_ownership(&mut self, expr: &Expr) {
        match expr {
            Expr::Ident(ident) => {
                // Reading a variable: check if it has been consumed.
                if let Some(info) = self.states.get(&ident.name)
                    && info.state == OwnershipState::Consumed
                    && let Some(consumed_span) = info.consumed_span
                {
                    self.diagnostics
                        .push(use_after_move(&ident.name, ident.span, consumed_span));
                }
            }
            Expr::Binary(lhs, _, rhs, _) => {
                self.check_expr_ownership(lhs);
                self.check_expr_ownership(rhs);
            }
            Expr::Unary(_, operand, _) => {
                self.check_expr_ownership(operand);
            }
            Expr::Call(callee, args, span) => {
                self.check_call_ownership(callee, args, *span);
            }
            Expr::GenericCall(callee, _, args, span) => {
                self.check_call_ownership(callee, args, *span);
            }
            Expr::Paren(inner, _) => {
                self.check_expr_ownership(inner);
            }
            Expr::View(inner, _) => {
                // `view expr` — just check the inner expression, no consumption.
                self.check_expr_ownership(inner);
            }
            Expr::FieldAccess(base, _, _) => {
                // Field access is an implicit view — does not consume the base.
                self.check_expr_ownership(base);
            }
            Expr::ListConstruct(elems, _) => {
                for elem in elems {
                    self.check_expr_ownership(elem);
                }
            }
            Expr::MapConstruct(entries, _) => {
                for (k, v) in entries {
                    self.check_expr_ownership(k);
                    self.check_expr_ownership(v);
                }
            }
            Expr::Handle(target, _, body, _) => {
                self.check_expr_ownership(target);
                self.check_block(body);
            }
            Expr::Ok(inner, _) | Expr::Fail(inner, _) | Expr::Some(inner, _) => {
                self.check_expr_ownership(inner);
            }
            Expr::Default(inner, _) => {
                self.check_expr_ownership(inner);
            }
            Expr::StringInterpolation(parts, _) => {
                for part in parts {
                    if let StringPart::Expr(expr) = part {
                        self.check_expr_ownership(expr);
                    }
                }
            }
            Expr::Declassify(inner, _) => {
                self.check_expr_ownership(inner);
            }
            Expr::Coarsen(inner, _) => {
                self.check_expr_ownership(inner);
            }
            Expr::Pipeline(initial, steps, _) => {
                self.check_expr_ownership(initial);
                for step in steps {
                    self.check_expr_ownership(&step.function);
                    for arg in &step.extra_args {
                        self.check_expr_ownership(&arg.value);
                    }
                }
            }
            Expr::At(inner, _, _) => {
                self.check_expr_ownership(inner);
            }
            Expr::Spawn(inner, _)
            | Expr::Send(inner, _)
            | Expr::Ask(inner, _)
            | Expr::Clone(inner, _)
            | Expr::Run(inner, _)
            | Expr::Join(inner, _)
            | Expr::Cancel(inner, _) => {
                self.check_expr_ownership(inner);
            }
            Expr::InlineFn(_, _, body, _) => {
                // The inline function body is checked in its own scope; it
                // captures variables by reference (view), so we just check the
                // body without consuming any outer variables.
                self.check_block(body);
            }
            // Literals and other leaves have no ownership effects.
            Expr::IntLiteral(_, _)
            | Expr::FloatLiteral(_, _)
            | Expr::StringLiteral(_, _)
            | Expr::BoolLiteral(_, _)
            | Expr::Nothing(_)
            | Expr::None(_)
            | Expr::EnumVariant(_, _, _)
            | Expr::Error(_) => {}
        }
    }

    /// Check a function call for ownership effects.
    ///
    /// Arguments passed with `view` (i.e., `Expr::View(inner)`) do not consume
    /// the inner value. Arguments passed without `view` consume the value if it
    /// is a non-copyable variable.
    fn check_call_ownership(&mut self, callee: &Expr, args: &[CallArg], _span: Span) {
        // Check the callee itself (e.g., reading a function variable).
        self.check_expr_ownership(callee);

        // Map and list builtins that operate on a collection without consuming it.
        // The first argument (the collection) is implicitly viewed.
        let callee_name = Self::dotted_name_str(callee);
        let collection_view_builtins: &[&str] = &[
            "map.length",
            "map.has",
            "map.get",
            "map.insert",
            "map.remove",
            "map.keys",
            "map.values",
            "map.is_empty",
            "list.length",
            "list.get",
            "list.first",
            "list.last",
            "list.append",
            "list.is_empty",
            "list.skip",
            "list.take",
            "list.reverse",
            "list.sort",
            "list.contains",
            "list.index_of",
            "list.remove",
            "list.concat",
            "list.flatten",
            "list.unique",
            "list.zip",
            "list.filter",
            "list.map",
            "list.find",
            "list.sort_by",
            "list.all",
            "list.any",
            "list.count",
            "list.sum",
            "list.group_by",
            "random.choice",
            "random.shuffle",
            "math.average",
            "math.median",
            "list.chunk",
            "list.sort_by_index",
            "list.is_sorted",
            "list.all_elements_in",
            "map.get_or",
            "map.merge",
            "map.contains_key",
            "map.set",
        ];
        let first_arg_is_view = callee_name
            .as_deref()
            .map(|n| collection_view_builtins.contains(&n) || is_json_implicit_view_facade(n))
            .unwrap_or(false);

        for (i, arg) in args.iter().enumerate() {
            match &arg.value {
                Expr::View(inner, _) => {
                    // `view x` — the argument is passed as a view; no consumption.
                    self.check_expr_ownership(inner);
                }
                _ if first_arg_is_view && i == 0 => {
                    // First argument to a collection builtin is implicitly viewed.
                    self.check_expr_ownership(&arg.value);
                }
                _ => {
                    // Non-view argument — consumes the value.
                    self.consume_expr(&arg.value, arg.span);
                }
            }
        }
    }

    fn dotted_name_str(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Ident(ident) => Some(ident.name.clone()),
            Expr::FieldAccess(inner, field, _) => {
                let prefix = Self::dotted_name_str(inner)?;
                Some(format!("{prefix}.{}", field.name))
            }
            _ => None,
        }
    }

    /// Consume an expression. If the expression is a simple identifier for a
    /// non-copyable variable, mark it as consumed. If it was already consumed,
    /// emit a use-after-move error.
    fn consume_expr(&mut self, expr: &Expr, span: Span) {
        match expr {
            Expr::Ident(ident) => {
                self.consume_variable(&ident.name, ident.span, span);
            }
            Expr::View(inner, _) => {
                // `view x` at the expression level — does not consume.
                self.check_expr_ownership(inner);
            }
            _ => {
                // For compound expressions, just traverse them for ownership checks.
                self.check_expr_ownership(expr);
            }
        }
    }

    /// Consume a named variable. Handles all the ownership state transitions
    /// and error reporting.
    fn consume_variable(&mut self, name: &str, name_span: Span, consume_span: Span) {
        if let Some(info) = self.states.get(name) {
            // Copyable primitives are never consumed.
            if self.is_copyable(info.type_id) {
                return;
            }

            match info.state {
                OwnershipState::Consumed => {
                    // Already consumed — use-after-move error.
                    if let Some(prev_span) = info.consumed_span {
                        self.diagnostics
                            .push(use_after_move(name, name_span, prev_span));
                    }
                }
                OwnershipState::Viewed => {
                    // View parameter — cannot be consumed.
                    self.diagnostics.push(cannot_consume_view(name, name_span));
                }
                OwnershipState::Owned => {
                    // Consume it.
                    let info = self.states.get_mut(name).unwrap();
                    info.state = OwnershipState::Consumed;
                    info.consumed_span = Some(consume_span);
                }
                OwnershipState::Pending | OwnershipState::Uninitialized => {
                    // Pending/Uninitialized — these would be caught by other passes.
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Type resolution helper (simplified — just for ownership tracking)
    // ------------------------------------------------------------------

    /// Resolve a type expression to a TypeId for ownership tracking purposes.
    /// This is a simplified version that only needs to distinguish primitives
    /// from non-primitives.
    fn resolve_type_for_ownership(&self, type_expr: &ast::TypeExpr) -> TypeId {
        match type_expr {
            ast::TypeExpr::Named(ident) => match ident.name.as_str() {
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
                "nothing" => TypeInterner::NOTHING,
                "TypeConstruction" => TypeInterner::TYPE_CONSTRUCTION,
                "TypeKind" | "TypePrimitive" => TypeInterner::INT64,
                // Any other named type is a struct/enum — not copyable.
                _ => TypeInterner::BYTES, // Use BYTES as a non-copyable stand-in
            },
            ast::TypeExpr::Generic(_, _, _) => {
                // Generic types (list[T], map[K,V], etc.) are handled by
                // is_type_expr_copyable which inspects the AST directly.
                TypeInterner::BYTES
            }
            ast::TypeExpr::View(inner, _) => {
                // View of a type — resolve the inner type.
                self.resolve_type_for_ownership(inner)
            }
            ast::TypeExpr::StateQualified(_, _, _) => {
                // State-qualified machine values are non-copyable.
                TypeInterner::BYTES
            }
            ast::TypeExpr::Function(_, _, _) => {
                // Function types are non-copyable (closures carry captured env).
                TypeInterner::BYTES
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Integration entry point
// ---------------------------------------------------------------------------

/// Run ownership analysis on a module and return diagnostics.
///
/// Called from the main `check` function in checker.rs after type checking.
pub fn check_ownership(module: &Module, interner: &TypeInterner) -> Vec<Diagnostic> {
    let checker = OwnershipChecker::new(interner);
    checker.check_module(module)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use jett_common::{FileId, Span};
    use jett_parser::ast::*;

    /// Helper to create a span for tests.
    fn sp(start: u32, end: u32) -> Span {
        Span::new(FileId::new(0), start, end)
    }

    fn ident(name: &str, span: Span) -> Ident {
        Ident {
            name: name.to_string(),
            span,
        }
    }

    /// Helper to create a module with a single function.
    fn module_with_func(func: FunctionDef) -> Module {
        Module {
            items: vec![Item::Function(func)],
            span: sp(0, 1000),
        }
    }

    /// Extract error diagnostics from a result.
    fn errors(diagnostics: &[Diagnostic]) -> Vec<&Diagnostic> {
        diagnostics
            .iter()
            .filter(|d| d.severity == jett_diagnostics::Severity::Error)
            .collect()
    }

    // ---------------------------------------------------------------
    // Test: Variable consumed after passing to function
    // ---------------------------------------------------------------

    #[test]
    fn variable_consumed_after_passing_to_function() {
        // function process(data: list[int64]) returns nothing:
        //     return nothing
        //
        // function example() returns nothing:
        //     list[int64] items = list(1, 2, 3)
        //     process(items)
        //     # items is consumed here

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(8, 15)))),
            body: Block {
                stmts: vec![
                    // list[int64] items = list(1, 2, 3)
                    Stmt::VarDecl(VarDecl {
                        mutable: false,
                        ty: TypeExpr::Generic(
                            ident("list", sp(20, 24)),
                            vec![TypeExpr::Named(ident("int64", sp(25, 30)))],
                            sp(20, 31),
                        ),
                        name: ident("items", sp(32, 37)),
                        value: Expr::ListConstruct(
                            vec![
                                Expr::IntLiteral(1, sp(40, 41)),
                                Expr::IntLiteral(2, sp(42, 43)),
                                Expr::IntLiteral(3, sp(44, 45)),
                            ],
                            sp(39, 46),
                        ),
                        span: sp(20, 46),
                    }),
                    // process(items)
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(50, 57)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("items", sp(58, 63))),
                                span: sp(58, 63),
                            }],
                            sp(50, 64),
                        ),
                        span: sp(50, 64),
                    }),
                ],
                span: sp(20, 64),
            },
            exported: false,
            span: sp(0, 64),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        // items is consumed by process() — no error (it's a valid consumption).
        let errs = errors(&diagnostics);
        assert!(errs.is_empty(), "unexpected errors: {:?}", errs);
    }

    // ---------------------------------------------------------------
    // Test: Use-after-move error
    // ---------------------------------------------------------------

    #[test]
    fn use_after_move_error() {
        // function example() returns nothing:
        //     list[int64] items = list(1, 2, 3)
        //     process(items)        # items consumed
        //     process(items)        # ERROR: use after move

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(8, 15)))),
            body: Block {
                stmts: vec![
                    Stmt::VarDecl(VarDecl {
                        mutable: false,
                        ty: TypeExpr::Generic(
                            ident("list", sp(20, 24)),
                            vec![TypeExpr::Named(ident("int64", sp(25, 30)))],
                            sp(20, 31),
                        ),
                        name: ident("items", sp(32, 37)),
                        value: Expr::ListConstruct(
                            vec![Expr::IntLiteral(1, sp(40, 41))],
                            sp(39, 42),
                        ),
                        span: sp(20, 42),
                    }),
                    // process(items)  — first call consumes items
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(50, 57)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("items", sp(58, 63))),
                                span: sp(58, 63),
                            }],
                            sp(50, 64),
                        ),
                        span: sp(50, 64),
                    }),
                    // process(items)  — second call: items already consumed
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(70, 77)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("items", sp(78, 83))),
                                span: sp(78, 83),
                            }],
                            sp(70, 84),
                        ),
                        span: sp(70, 84),
                    }),
                ],
                span: sp(20, 84),
            },
            exported: false,
            span: sp(0, 84),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        let errs = errors(&diagnostics);
        assert_eq!(errs.len(), 1, "expected 1 error, got: {:?}", errs);
        assert_eq!(errs[0].code.code(), 400);
        assert!(errs[0].message.contains("items"));
        assert!(errs[0].message.contains("consumed"));
    }

    // ---------------------------------------------------------------
    // Test: View parameter allows continued use
    // ---------------------------------------------------------------

    #[test]
    fn view_parameter_allows_continued_use() {
        // function example() returns nothing:
        //     list[int64] items = list(1, 2, 3)
        //     process(view items)     # items stays owned (view argument)
        //     process(view items)     # still valid

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(8, 15)))),
            body: Block {
                stmts: vec![
                    Stmt::VarDecl(VarDecl {
                        mutable: false,
                        ty: TypeExpr::Generic(
                            ident("list", sp(20, 24)),
                            vec![TypeExpr::Named(ident("int64", sp(25, 30)))],
                            sp(20, 31),
                        ),
                        name: ident("items", sp(32, 37)),
                        value: Expr::ListConstruct(
                            vec![Expr::IntLiteral(1, sp(40, 41))],
                            sp(39, 42),
                        ),
                        span: sp(20, 42),
                    }),
                    // process(view items) — first call, view argument
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(50, 57)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::View(
                                    Box::new(Expr::Ident(ident("items", sp(63, 68)))),
                                    sp(58, 68),
                                ),
                                span: sp(58, 68),
                            }],
                            sp(50, 69),
                        ),
                        span: sp(50, 69),
                    }),
                    // process(view items) — second call, still valid
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(75, 82)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::View(
                                    Box::new(Expr::Ident(ident("items", sp(88, 93)))),
                                    sp(83, 93),
                                ),
                                span: sp(83, 93),
                            }],
                            sp(75, 94),
                        ),
                        span: sp(75, 94),
                    }),
                ],
                span: sp(20, 94),
            },
            exported: false,
            span: sp(0, 94),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        let errs = errors(&diagnostics);
        assert!(errs.is_empty(), "unexpected errors: {:?}", errs);
        assert!(
            diagnostics.is_empty(),
            "viewed values may go out of scope without warnings: {:?}",
            diagnostics
        );
    }

    // ---------------------------------------------------------------
    // Test: Primitives are copyable (no move tracking)
    // ---------------------------------------------------------------

    #[test]
    fn primitives_are_copyable_no_move_tracking() {
        // function example() returns nothing:
        //     int64 x = 42
        //     process(x)
        //     process(x)    # Valid — int64 is implicitly copyable

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(8, 15)))),
            body: Block {
                stmts: vec![
                    Stmt::VarDecl(VarDecl {
                        mutable: false,
                        ty: TypeExpr::Named(ident("int64", sp(20, 25))),
                        name: ident("x", sp(26, 27)),
                        value: Expr::IntLiteral(42, sp(30, 32)),
                        span: sp(20, 32),
                    }),
                    // process(x) — first call
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(40, 47)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("x", sp(48, 49))),
                                span: sp(48, 49),
                            }],
                            sp(40, 50),
                        ),
                        span: sp(40, 50),
                    }),
                    // process(x) — second call, still valid because int64 is copyable
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(55, 62)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("x", sp(63, 64))),
                                span: sp(63, 64),
                            }],
                            sp(55, 65),
                        ),
                        span: sp(55, 65),
                    }),
                ],
                span: sp(20, 65),
            },
            exported: false,
            span: sp(0, 65),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        let errs = errors(&diagnostics);
        assert!(
            errs.is_empty(),
            "primitives should be copyable, got errors: {:?}",
            errs
        );
    }

    // ---------------------------------------------------------------
    // Test: Mutable variable can be rebound after move
    // ---------------------------------------------------------------

    #[test]
    fn mutable_variable_can_be_rebound_after_move() {
        // function example() returns nothing:
        //     mutable list[int64] items = list(1, 2)
        //     process(items)                 # items consumed
        //     items = list(3, 4)             # rebind: items is owned again
        //     process(items)                 # valid

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(8, 15)))),
            body: Block {
                stmts: vec![
                    // mutable list[int64] items = list(1, 2)
                    Stmt::VarDecl(VarDecl {
                        mutable: true,
                        ty: TypeExpr::Generic(
                            ident("list", sp(20, 24)),
                            vec![TypeExpr::Named(ident("int64", sp(25, 30)))],
                            sp(20, 31),
                        ),
                        name: ident("items", sp(32, 37)),
                        value: Expr::ListConstruct(
                            vec![
                                Expr::IntLiteral(1, sp(40, 41)),
                                Expr::IntLiteral(2, sp(42, 43)),
                            ],
                            sp(39, 44),
                        ),
                        span: sp(20, 44),
                    }),
                    // process(items) — consumes items
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(50, 57)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("items", sp(58, 63))),
                                span: sp(58, 63),
                            }],
                            sp(50, 64),
                        ),
                        span: sp(50, 64),
                    }),
                    // items = list(3, 4) — rebind
                    Stmt::Assign(AssignStmt {
                        target: Expr::Ident(ident("items", sp(70, 75))),
                        value: Expr::ListConstruct(
                            vec![
                                Expr::IntLiteral(3, sp(80, 81)),
                                Expr::IntLiteral(4, sp(82, 83)),
                            ],
                            sp(79, 84),
                        ),
                        span: sp(70, 84),
                    }),
                    // process(items) — valid again
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(90, 97)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("items", sp(98, 103))),
                                span: sp(98, 103),
                            }],
                            sp(90, 104),
                        ),
                        span: sp(90, 104),
                    }),
                ],
                span: sp(20, 104),
            },
            exported: false,
            span: sp(0, 104),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        let errs = errors(&diagnostics);
        assert!(
            errs.is_empty(),
            "mutable rebinding should reset ownership, got errors: {:?}",
            errs
        );
    }

    // ---------------------------------------------------------------
    // Test: For loop consumes iterable
    // ---------------------------------------------------------------

    #[test]
    fn for_loop_consumes_iterable() {
        // function example() returns nothing:
        //     list[int64] items = list(1, 2, 3)
        //     for item in items:
        //         pass
        //     process(items)   # ERROR: items was consumed by for loop

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(8, 15)))),
            body: Block {
                stmts: vec![
                    // list[int64] items = list(1, 2, 3)
                    Stmt::VarDecl(VarDecl {
                        mutable: false,
                        ty: TypeExpr::Generic(
                            ident("list", sp(20, 24)),
                            vec![TypeExpr::Named(ident("int64", sp(25, 30)))],
                            sp(20, 31),
                        ),
                        name: ident("items", sp(32, 37)),
                        value: Expr::ListConstruct(
                            vec![
                                Expr::IntLiteral(1, sp(40, 41)),
                                Expr::IntLiteral(2, sp(42, 43)),
                                Expr::IntLiteral(3, sp(44, 45)),
                            ],
                            sp(39, 46),
                        ),
                        span: sp(20, 46),
                    }),
                    // for item in items:
                    Stmt::For(ForStmt {
                        variable: ident("item", sp(50, 54)),
                        value_variable: None,
                        view: false, // Not a view iteration — consumes items
                        iterable: Expr::Ident(ident("items", sp(58, 63))),
                        body: Block {
                            stmts: vec![],
                            span: sp(65, 70),
                        },
                        span: sp(47, 70),
                    }),
                    // process(items)  — ERROR: items consumed by for loop
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(75, 82)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("items", sp(83, 88))),
                                span: sp(83, 88),
                            }],
                            sp(75, 89),
                        ),
                        span: sp(75, 89),
                    }),
                ],
                span: sp(20, 89),
            },
            exported: false,
            span: sp(0, 89),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        let errs = errors(&diagnostics);
        assert_eq!(
            errs.len(),
            1,
            "expected 1 use-after-move error, got: {:?}",
            errs
        );
        assert_eq!(errs[0].code.code(), 400);
        assert!(errs[0].message.contains("items"));
    }

    // ---------------------------------------------------------------
    // Test: For loop with view does NOT consume iterable
    // ---------------------------------------------------------------

    #[test]
    fn for_loop_view_does_not_consume_iterable() {
        // function example() returns nothing:
        //     list[int64] items = list(1, 2, 3)
        //     for item in view items:
        //         pass
        //     process(items)   # Valid — items was viewed, not consumed

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(8, 15)))),
            body: Block {
                stmts: vec![
                    Stmt::VarDecl(VarDecl {
                        mutable: false,
                        ty: TypeExpr::Generic(
                            ident("list", sp(20, 24)),
                            vec![TypeExpr::Named(ident("int64", sp(25, 30)))],
                            sp(20, 31),
                        ),
                        name: ident("items", sp(32, 37)),
                        value: Expr::ListConstruct(
                            vec![Expr::IntLiteral(1, sp(40, 41))],
                            sp(39, 42),
                        ),
                        span: sp(20, 42),
                    }),
                    // for item in view items:
                    Stmt::For(ForStmt {
                        variable: ident("item", sp(50, 54)),
                        value_variable: None,
                        view: true, // View iteration — does NOT consume items
                        iterable: Expr::Ident(ident("items", sp(63, 68))),
                        body: Block {
                            stmts: vec![],
                            span: sp(70, 75),
                        },
                        span: sp(47, 75),
                    }),
                    // process(items) — valid, items was not consumed
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(80, 87)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("items", sp(88, 93))),
                                span: sp(88, 93),
                            }],
                            sp(80, 94),
                        ),
                        span: sp(80, 94),
                    }),
                ],
                span: sp(20, 94),
            },
            exported: false,
            span: sp(0, 94),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        let errs = errors(&diagnostics);
        assert!(errs.is_empty(), "unexpected errors: {:?}", errs);
    }

    // ---------------------------------------------------------------
    // Test: Cannot consume a view parameter
    // ---------------------------------------------------------------

    #[test]
    fn cannot_consume_view_parameter() {
        // function example(view data: list[int64]) returns nothing:
        //     process(data)    # ERROR: cannot consume a view parameter

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![Param {
                view: true,
                mutable: false,
                name: ident("data", sp(8, 12)),
                ty: TypeExpr::Generic(
                    ident("list", sp(14, 18)),
                    vec![TypeExpr::Named(ident("int64", sp(19, 24)))],
                    sp(14, 25),
                ),
                span: sp(8, 25),
            }],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(30, 37)))),
            body: Block {
                stmts: vec![
                    // process(data)  — ERROR: data is a view parameter
                    Stmt::Expr(ExprStmt {
                        expr: Expr::Call(
                            Box::new(Expr::Ident(ident("process", sp(40, 47)))),
                            vec![CallArg {
                                name: None,
                                value: Expr::Ident(ident("data", sp(48, 52))),
                                span: sp(48, 52),
                            }],
                            sp(40, 53),
                        ),
                        span: sp(40, 53),
                    }),
                ],
                span: sp(40, 53),
            },
            exported: false,
            span: sp(0, 53),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        let errs = errors(&diagnostics);
        assert_eq!(errs.len(), 1, "expected 1 error, got: {:?}", errs);
        assert_eq!(errs[0].code.code(), 401);
        assert!(errs[0].message.contains("data"));
        assert!(errs[0].message.contains("view"));
    }

    // ---------------------------------------------------------------
    // Test: Owned linear values may go out of scope without warnings
    // ---------------------------------------------------------------

    #[test]
    fn owned_linear_value_can_exit_scope_without_warning() {
        // function example(data: list[int64]) returns nothing:
        //     return nothing
        // No warning: scope exit is a valid ownership endpoint.

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![Param {
                view: false,
                mutable: false,
                name: ident("data", sp(8, 12)),
                ty: TypeExpr::Generic(
                    ident("list", sp(14, 18)),
                    vec![TypeExpr::Named(ident("int64", sp(19, 24)))],
                    sp(14, 25),
                ),
                span: sp(8, 25),
            }],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(30, 37)))),
            body: Block {
                stmts: vec![Stmt::Return(ReturnStmt {
                    value: Some(Expr::Nothing(sp(40, 47))),
                    span: sp(40, 47),
                })],
                span: sp(40, 47),
            },
            exported: false,
            span: sp(0, 47),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        assert!(
            diagnostics.is_empty(),
            "owned values may go out of scope without warnings: {:?}",
            diagnostics
        );
    }

    // ---------------------------------------------------------------
    // Test: View parameter for primitives — no consumption concern
    // ---------------------------------------------------------------

    #[test]
    fn view_param_with_primitive_no_errors() {
        // function example(view x: int64) returns nothing:
        //     return nothing
        // No warnings — primitives are copyable, even when viewed.

        let func = FunctionDef {
            name: ident("example", sp(0, 7)),
            type_params: vec![],
            params: vec![Param {
                view: true,
                mutable: false,
                name: ident("x", sp(8, 9)),
                ty: TypeExpr::Named(ident("int64", sp(11, 16))),
                span: sp(8, 16),
            }],
            return_type: Some(TypeExpr::Named(ident("nothing", sp(20, 27)))),
            body: Block {
                stmts: vec![Stmt::Return(ReturnStmt {
                    value: Some(Expr::Nothing(sp(30, 37))),
                    span: sp(30, 37),
                })],
                span: sp(30, 37),
            },
            exported: false,
            span: sp(0, 37),
        };

        let interner = TypeInterner::new();
        let module = module_with_func(func);
        let diagnostics = check_ownership(&module, &interner);

        let errs = errors(&diagnostics);
        assert!(errs.is_empty(), "unexpected errors: {:?}", errs);
        assert!(
            diagnostics.is_empty(),
            "unexpected diagnostics: {:?}",
            diagnostics
        );
    }
}
