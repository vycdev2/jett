use jett_common::Span;
use jett_diagnostics::Diagnostic;
use jett_lexer::{Token, TokenKind};

use crate::ast::*;

// ---------------------------------------------------------------------------
// Parse result
// ---------------------------------------------------------------------------

/// Result of parsing a source file.
pub struct ParseResult {
    pub module: Module,
    pub errors: Vec<Diagnostic>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub struct Parser<'src> {
    source: &'src str,
    tokens: Vec<Token>,
    pos: usize,
    errors: Vec<Diagnostic>,
}

impl<'src> Parser<'src> {
    pub fn new(source: &'src str, tokens: Vec<Token>) -> Self {
        Self {
            source,
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    // =======================================================================
    // Token helpers
    // =======================================================================

    fn peek(&self) -> TokenKind {
        self.tokens.get(self.pos).map_or(TokenKind::Eof, |t| t.kind)
    }

    fn peek_token(&self) -> &Token {
        static EOF_TOKEN: std::sync::LazyLock<Token> = std::sync::LazyLock::new(|| Token {
            kind: TokenKind::Eof,
            span: Span::new(jett_common::FileId::new(0), 0, 0),
        });
        self.tokens.get(self.pos).unwrap_or(&EOF_TOKEN)
    }

    fn peek_nth(&self, n: usize) -> TokenKind {
        self.tokens
            .get(self.pos + n)
            .map_or(TokenKind::Eof, |t| t.kind)
    }

    fn advance(&mut self) -> Token {
        let tok = self.peek_token().clone();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn eat(&mut self, kind: TokenKind) -> Option<Token> {
        if self.peek() == kind {
            Some(self.advance())
        } else {
            None
        }
    }

    fn eat_contextual_ident(&mut self, text: &str) -> Option<Token> {
        let tok = self.peek_token().clone();
        if tok.kind == TokenKind::Ident && self.token_text(&tok) == text {
            Some(self.advance())
        } else {
            None
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Token {
        if self.peek() == kind {
            self.advance()
        } else {
            let tok = self.peek_token().clone();
            self.error(
                format!("expected {:?}, found {:?}", kind, tok.kind),
                tok.span,
            );
            tok
        }
    }

    fn token_text(&self, token: &Token) -> &'src str {
        let start = token.span.start as usize;
        let end = token.span.end as usize;
        if end <= self.source.len() {
            &self.source[start..end]
        } else {
            ""
        }
    }

    fn parse_i64_token(&mut self, token: &Token, context: &str) -> i64 {
        let text = self.token_text(token).to_string();
        text.parse::<i64>().unwrap_or_else(|_| {
            self.error(
                format!("{context} must fit in the supported signed 64-bit literal range"),
                token.span,
            );
            0
        })
    }

    fn parse_i128_token(&mut self, token: &Token, context: &str) -> i128 {
        let text = self.token_text(token).to_string();
        text.parse::<i128>().unwrap_or_else(|_| {
            self.error(format!("{context} is too large"), token.span);
            0
        })
    }

    fn skip_newlines(&mut self) {
        while self.peek() == TokenKind::Newline {
            self.advance();
        }
    }

    fn error(&mut self, message: impl Into<String>, span: Span) {
        self.errors.push(Diagnostic::error(1000, message, span));
    }

    /// Skip tokens until we reach one of the recovery points (Newline, Dedent, Eof).
    fn synchronize(&mut self) {
        loop {
            match self.peek() {
                TokenKind::Newline => {
                    self.advance();
                    return;
                }
                TokenKind::Dedent | TokenKind::Eof => return,
                _ => {
                    self.advance();
                }
            }
        }
    }

    // =======================================================================
    // Entry point
    // =======================================================================

    pub fn parse(mut self) -> ParseResult {
        let start_span = self.peek_token().span;
        self.skip_newlines();
        let mut items = Vec::new();

        while self.peek() != TokenKind::Eof {
            let before = self.pos;
            match self.parse_item() {
                Some(item) => items.push(item),
                None => {
                    // Error recovery: skip to next line
                    self.synchronize();
                    if self.pos == before && self.peek() != TokenKind::Eof {
                        self.advance();
                    }
                }
            }
            self.skip_newlines();
        }

        let end_span = if let Some(last) = self.tokens.last() {
            last.span
        } else {
            start_span
        };

        ParseResult {
            module: Module {
                span: start_span.merge(end_span),
                items,
            },
            errors: self.errors,
        }
    }

    // =======================================================================
    // Items
    // =======================================================================

    fn parse_item(&mut self) -> Option<Item> {
        self.skip_newlines();
        let export_span = self.eat(TokenKind::Export).map(|tok| tok.span);
        let exported = export_span.is_some();
        let root_exported = exported && self.eat_contextual_ident("root").is_some();
        if root_exported && self.peek() != TokenKind::Type {
            let tok = self.peek_token().clone();
            self.error(
                format!("expected `type` after `export root`, found {:?}", tok.kind),
                tok.span,
            );
            return None;
        }
        if exported
            && !matches!(
                self.peek(),
                TokenKind::Function
                    | TokenKind::Interface
                    | TokenKind::Struct
                    | TokenKind::Bitfield
                    | TokenKind::Enum
                    | TokenKind::Machine
                    | TokenKind::Actor
                    | TokenKind::Type
            )
            && !(self.peek() == TokenKind::Network && self.peek_nth(1) == TokenKind::Bitfield)
        {
            let tok = self.peek_token().clone();
            self.error(
                format!(
                    "expected exportable item (function, interface, struct, bitfield, enum, machine, actor, or type), found {:?}",
                    tok.kind
                ),
                tok.span,
            );
            return None;
        }
        match self.peek() {
            TokenKind::Namespace => Some(Item::Namespace(self.parse_namespace())),
            TokenKind::Function => Some(Item::Function(self.parse_function(exported, export_span))),
            TokenKind::Mutual => Some(Item::Mutual(self.parse_mutual_block())),
            TokenKind::Interface => {
                Some(Item::Interface(self.parse_interface(exported, export_span)))
            }
            TokenKind::Implement => Some(Item::Implement(self.parse_implement())),
            TokenKind::Struct => Some(Item::Struct(self.parse_struct(exported, export_span))),
            TokenKind::Bitfield => Some(Item::Bitfield(self.parse_bitfield(
                false,
                exported,
                export_span,
            ))),
            TokenKind::Network if self.peek_nth(1) == TokenKind::Bitfield => Some(Item::Bitfield(
                self.parse_bitfield(true, exported, export_span),
            )),
            TokenKind::Enum => Some(Item::Enum(self.parse_enum(exported, export_span))),
            TokenKind::Machine => Some(Item::Machine(self.parse_machine(exported))),
            TokenKind::Actor => Some(Item::Actor(self.parse_actor(exported, export_span))),
            TokenKind::Verify => Some(Item::Verify(self.parse_verify_block())),
            TokenKind::Property => Some(Item::Property(self.parse_property_block())),
            TokenKind::Type => Some(Item::TypeAlias(self.parse_type_alias(
                exported,
                root_exported,
                export_span,
            ))),
            TokenKind::Mutable => Some(Item::VarDecl(self.parse_var_decl())),
            // Could be a variable declaration: `Type name = expr`
            _ if !exported && self.looks_like_var_decl() => {
                Some(Item::VarDecl(self.parse_var_decl()))
            }
            _ => {
                let tok = self.peek_token().clone();
                if exported {
                    self.error(
                        format!(
                            "expected exportable item (function, interface, struct, bitfield, enum, machine, actor, or type), found {:?}",
                            tok.kind
                        ),
                        tok.span,
                    );
                } else {
                    self.error(
                        format!("expected item (namespace, function, mutual, interface, implement, struct, bitfield, enum, machine, type, property, or variable), found {:?}", tok.kind),
                        tok.span,
                    );
                }
                None
            }
        }
    }

    fn parse_namespace(&mut self) -> NamespaceDecl {
        let kw = self.expect(TokenKind::Namespace);
        let name = self.parse_qualified_ident();
        NamespaceDecl {
            span: kw.span.merge(name.span),
            name,
        }
    }

    fn parse_verify_block(&mut self) -> VerifyBlock {
        let kw = self.expect(TokenKind::Verify);
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);
        let body = self.parse_block();
        let end_span = body.span;
        VerifyBlock {
            span: kw.span.merge(end_span),
            name,
            body,
        }
    }

    fn parse_property_block(&mut self) -> PropertyBlock {
        let kw = self.expect(TokenKind::Property);
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);

        // Expect an indented block containing `given` declarations followed by body statements.
        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut givens = Vec::new();
        let mut stmts = Vec::new();
        let mut last_span = indent_tok.span;

        self.skip_newlines();

        // Parse `given` declarations first.
        while self.peek() == TokenKind::Given {
            let given_kw = self.advance();
            let given_name = self.parse_ident();
            self.expect(TokenKind::Colon);
            let given_ty = self.parse_type();
            let end = given_ty.span();
            givens.push(GivenDecl {
                span: given_kw.span.merge(end),
                name: given_name,
                ty: given_ty,
            });
            last_span = givens.last().unwrap().span;
            self.skip_newlines();
        }

        // Parse remaining statements as the body.
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            match self.parse_stmt() {
                Some(stmt) => {
                    last_span = stmt_span(&stmt);
                    stmts.push(stmt);
                }
                None => {
                    self.synchronize();
                }
            }
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        let body = Block {
            stmts,
            span: indent_tok.span.merge(last_span),
        };

        PropertyBlock {
            span: kw.span.merge(last_span),
            name,
            givens,
            body,
        }
    }

    fn parse_type_alias(
        &mut self,
        exported: bool,
        root_exported: bool,
        export_span: Option<Span>,
    ) -> TypeAlias {
        let kw = self.expect(TokenKind::Type);
        let start_span = export_span.unwrap_or(kw.span);
        let name = self.parse_ident();
        self.expect(TokenKind::Eq);
        let base_type = self.parse_type();

        let constraint = if self.eat(TokenKind::Where).is_some() {
            Some(self.parse_expr())
        } else {
            None
        };

        let end_span = constraint.as_ref().map_or(base_type.span(), |c| c.span());

        TypeAlias {
            span: start_span.merge(end_span),
            name,
            base_type,
            constraint,
            exported,
            root_exported,
        }
    }

    fn parse_function(&mut self, exported: bool, export_span: Option<Span>) -> FunctionDef {
        let kw = self.expect(TokenKind::Function);
        let start_span = export_span.unwrap_or(kw.span);
        let decl = self.parse_function_decl_rest(kw.span, exported);
        self.expect(TokenKind::Colon);
        let body = self.parse_block();
        let end_span = body.span;

        FunctionDef {
            span: start_span.merge(end_span),
            name: decl.name,
            type_params: decl.type_params,
            params: decl.params,
            return_type: decl.return_type,
            body,
            exported,
        }
    }

    fn parse_function_decl_rest(&mut self, start_span: Span, exported: bool) -> FunctionDecl {
        let name = self.parse_ident();

        // Optional generic type parameters: `[T, U, ...]`
        let type_params = if self.peek() == TokenKind::LBracket {
            self.advance(); // consume `[`
            let mut params = Vec::new();
            if self.peek() != TokenKind::RBracket {
                params.push(self.parse_ident());
                while self.eat(TokenKind::Comma).is_some() {
                    params.push(self.parse_ident());
                }
            }
            self.expect(TokenKind::RBracket);
            params
        } else {
            vec![]
        };

        self.expect(TokenKind::LParen);
        let params = self.parse_params();
        self.expect(TokenKind::RParen);

        let return_type = if self.eat(TokenKind::Returns).is_some() {
            Some(self.parse_type())
        } else {
            None
        };

        let end_span = return_type.as_ref().map_or(name.span, TypeExpr::span);

        FunctionDecl {
            span: start_span.merge(end_span),
            name,
            type_params,
            params,
            return_type,
            exported,
        }
    }

    fn parse_mutual_block(&mut self) -> MutualBlock {
        let kw = self.expect(TokenKind::Mutual);
        self.expect(TokenKind::Colon);

        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut declarations = Vec::new();
        let mut last_span = indent_tok.span;

        self.skip_newlines();
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            let export_span = self.eat(TokenKind::Export).map(|tok| tok.span);
            let exported = export_span.is_some();
            let func_kw = self.expect(TokenKind::Function);
            let start_span = export_span.unwrap_or(func_kw.span);
            let decl = self.parse_function_decl_rest(start_span, exported);
            last_span = decl.span;
            declarations.push(decl);
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        MutualBlock {
            declarations,
            span: kw.span.merge(last_span),
        }
    }

    fn parse_interface(&mut self, exported: bool, export_span: Option<Span>) -> InterfaceDecl {
        let kw = self.expect(TokenKind::Interface);
        let start_span = export_span.unwrap_or(kw.span);
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);

        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut methods = Vec::new();
        let mut last_span = indent_tok.span;

        self.skip_newlines();
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            let func_kw = self.expect(TokenKind::Function);
            let decl = self.parse_function_decl_rest(func_kw.span, false);
            last_span = decl.span;
            methods.push(decl);
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        InterfaceDecl {
            name,
            methods,
            exported,
            span: start_span.merge(last_span),
        }
    }

    fn parse_implement(&mut self) -> ImplementBlock {
        let kw = self.expect(TokenKind::Implement);
        let interface_name = self.parse_qualified_ident();
        self.expect(TokenKind::For);
        let for_type = self.parse_type();
        self.expect(TokenKind::Colon);

        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut methods = Vec::new();
        let mut last_span = indent_tok.span;

        self.skip_newlines();
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            let method = self.parse_function(false, None);
            last_span = method.span;
            methods.push(method);
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        ImplementBlock {
            interface_name,
            for_type,
            methods,
            span: kw.span.merge(last_span),
        }
    }

    fn parse_params(&mut self) -> Vec<Param> {
        let mut params = Vec::new();
        if self.peek() == TokenKind::RParen {
            return params;
        }
        params.push(self.parse_param());
        while self.eat(TokenKind::Comma).is_some() {
            if self.peek() == TokenKind::RParen {
                break;
            }
            params.push(self.parse_param());
        }
        params
    }

    fn parse_param(&mut self) -> Param {
        let start = self.peek_token().span;
        let view = self.eat(TokenKind::View).is_some();
        let mutable = self.eat(TokenKind::Mutable).is_some();
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);
        let ty = self.parse_type();
        let end = ty.span();
        Param {
            view,
            mutable,
            name,
            ty,
            span: start.merge(end),
        }
    }

    fn parse_struct(&mut self, exported: bool, export_span: Option<Span>) -> StructDef {
        let kw = self.expect(TokenKind::Struct);
        let start_span = export_span.unwrap_or(kw.span);
        let name = self.parse_ident();

        // Optional generic type parameters: `[T, U, ...]`
        let type_params = if self.peek() == TokenKind::LBracket {
            self.advance(); // consume `[`
            let mut params = Vec::new();
            if self.peek() != TokenKind::RBracket {
                params.push(self.parse_ident());
                while self.eat(TokenKind::Comma).is_some() {
                    params.push(self.parse_ident());
                }
            }
            self.expect(TokenKind::RBracket);
            params
        } else {
            vec![]
        };

        self.expect(TokenKind::Colon);

        // Expect an indented block of fields and methods
        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        let mut last_span = indent_tok.span;

        self.skip_newlines();
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            if self.peek() == TokenKind::Function {
                let func = self.parse_function(false, None);
                last_span = func.span;
                methods.push(func);
            } else {
                // Field: `name: Type`
                let field = self.parse_field_def();
                last_span = field.span;
                fields.push(field);
            }
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        StructDef {
            span: start_span.merge(last_span),
            name,
            type_params,
            fields,
            methods,
            exported,
        }
    }

    fn parse_field_def(&mut self) -> FieldDef {
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);
        let ty = self.parse_type();
        let mut end = ty.span();
        let serialize_name = if self.eat(TokenKind::Serialize).is_some() {
            let tok = self.expect(TokenKind::StringLiteral);
            end = tok.span;
            let text = self.token_text(&tok);
            let raw = if text.len() >= 2 {
                &text[1..text.len() - 1]
            } else {
                text
            };
            Some(unescape_string(raw))
        } else {
            None
        };
        FieldDef {
            span: name.span.merge(end),
            name,
            ty,
            serialize_name,
        }
    }

    fn parse_bitfield(
        &mut self,
        leading_network: bool,
        exported: bool,
        export_span: Option<Span>,
    ) -> BitfieldDef {
        let mut start_span = export_span.unwrap_or_else(|| self.peek_token().span);
        let mut network_order = false;

        if leading_network {
            let network_kw = self.expect(TokenKind::Network);
            if export_span.is_none() {
                start_span = network_kw.span;
            }
            network_order = true;
        }

        let bitfield_kw = self.expect(TokenKind::Bitfield);
        if !leading_network && export_span.is_none() {
            start_span = bitfield_kw.span;
        }
        if self.eat(TokenKind::Network).is_some() {
            network_order = true;
        }

        let name = self.parse_ident();
        self.expect(TokenKind::Colon);

        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut fields = Vec::new();
        let mut last_span = indent_tok.span;

        self.skip_newlines();
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            let field = self.parse_bitfield_field_def();
            last_span = field.span;
            fields.push(field);
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        BitfieldDef {
            span: start_span.merge(last_span),
            name,
            network_order,
            fields,
            exported,
        }
    }

    fn parse_bitfield_field_def(&mut self) -> BitfieldFieldDef {
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);

        let (kind, end_span) = if self.peek() == TokenKind::IntLiteral {
            let width_tok = self.advance();
            let width = self
                .token_text(&width_tok)
                .parse::<u16>()
                .unwrap_or_else(|_| {
                    self.error("bitfield width must be a base-10 integer", width_tok.span);
                    0
                });
            let unit_tok = if self.peek() == TokenKind::Bit {
                self.advance()
            } else {
                self.expect(TokenKind::Bits)
            };
            let as_type = if self.eat(TokenKind::As).is_some() {
                Some(self.parse_type())
            } else {
                None
            };
            let end_span = as_type.as_ref().map_or(unit_tok.span, TypeExpr::span);
            (BitfieldFieldKind::Bits { width, as_type }, end_span)
        } else {
            let ty = self.parse_type();
            let end_span = ty.span();
            (BitfieldFieldKind::Payload(ty), end_span)
        };

        BitfieldFieldDef {
            span: name.span.merge(end_span),
            name,
            kind,
        }
    }

    fn parse_enum(&mut self, exported: bool, export_span: Option<Span>) -> EnumDef {
        let kw = self.expect(TokenKind::Enum);
        let start_span = export_span.unwrap_or(kw.span);
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);

        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut variants = Vec::new();
        let mut last_span = indent_tok.span;

        self.skip_newlines();
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            let variant = self.parse_variant();
            last_span = variant.span;
            variants.push(variant);
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        EnumDef {
            span: start_span.merge(last_span),
            name,
            variants,
            exported,
        }
    }

    fn parse_variant(&mut self) -> Variant {
        let name = self.parse_ident();
        let mut fields = Vec::new();
        let mut discriminant = None;
        let mut end_span = name.span;

        if self.eat(TokenKind::LParen).is_some() {
            if self.peek() != TokenKind::RParen {
                fields.push(self.parse_field_def());
                while self.eat(TokenKind::Comma).is_some() {
                    if self.peek() == TokenKind::RParen {
                        break;
                    }
                    fields.push(self.parse_field_def());
                }
            }
            end_span = self.expect(TokenKind::RParen).span;
        }

        if self.eat(TokenKind::Eq).is_some() {
            let value_tok = self.expect(TokenKind::IntLiteral);
            discriminant = Some(self.parse_i64_token(&value_tok, "enum discriminant"));
            end_span = value_tok.span;
        }

        Variant {
            span: name.span.merge(end_span),
            name,
            fields,
            discriminant,
        }
    }

    // =======================================================================
    // State machines
    // =======================================================================

    fn parse_machine(&mut self, exported: bool) -> MachineDef {
        let kw = self.expect(TokenKind::Machine);
        let name = self.parse_ident();
        self.expect(TokenKind::Colon);

        // Outer indent for the machine body
        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut states = Vec::new();
        let mut transitions = Vec::new();
        let mut last_span = indent_tok.span;

        // Parse `states:` block
        self.skip_newlines();
        if self.peek() == TokenKind::States {
            self.advance(); // consume `states`
            self.expect(TokenKind::Colon);

            self.skip_newlines();
            self.expect(TokenKind::Indent);
            self.skip_newlines();
            while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
                let state = self.parse_machine_state();
                last_span = state.span;
                states.push(state);
                self.skip_newlines();
            }
            if self.peek() == TokenKind::Dedent {
                last_span = self.advance().span;
            }
        }

        // Parse `transitions:` block
        self.skip_newlines();
        if self.peek() == TokenKind::Transitions {
            self.advance(); // consume `transitions`
            self.expect(TokenKind::Colon);

            self.skip_newlines();
            self.expect(TokenKind::Indent);
            self.skip_newlines();
            while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
                let transition = self.parse_machine_transition();
                last_span = transition.span;
                transitions.push(transition);
                self.skip_newlines();
            }
            if self.peek() == TokenKind::Dedent {
                last_span = self.advance().span;
            }
        }

        // Outer dedent
        self.skip_newlines();
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        MachineDef {
            span: kw.span.merge(last_span),
            name,
            exported,
            states,
            transitions,
        }
    }

    /// Parse a single state: `guest` or `logged_in(user_id: string)`.
    fn parse_machine_state(&mut self) -> MachineState {
        let name = self.parse_ident();
        let mut fields = Vec::new();
        let mut end_span = name.span;

        if self.eat(TokenKind::LParen).is_some() {
            if self.peek() != TokenKind::RParen {
                fields.push(self.parse_field_def());
                while self.eat(TokenKind::Comma).is_some() {
                    if self.peek() == TokenKind::RParen {
                        break;
                    }
                    fields.push(self.parse_field_def());
                }
            }
            end_span = self.expect(TokenKind::RParen).span;
        }

        MachineState {
            span: name.span.merge(end_span),
            name,
            fields,
        }
    }

    /// Parse a single transition: `guest to logged_in`.
    fn parse_machine_transition(&mut self) -> MachineTransition {
        let from = self.parse_ident();
        self.expect(TokenKind::To);
        let to = self.parse_ident();
        MachineTransition {
            span: from.span.merge(to.span),
            from,
            to,
        }
    }

    // =======================================================================
    // Actors
    // =======================================================================

    fn parse_actor(&mut self, exported: bool, export_span: Option<Span>) -> ActorDef {
        let kw = self.expect(TokenKind::Actor);
        let start_span = export_span.unwrap_or(kw.span);
        let name = self.parse_ident();

        // Optional capability parameters: `actor Counter(stdout: Stdout):`
        let capability_params = if self.peek() == TokenKind::LParen {
            self.advance(); // consume `(`
            let mut params = Vec::new();
            if self.peek() != TokenKind::RParen {
                params.push(self.parse_param());
                while self.eat(TokenKind::Comma).is_some() {
                    if self.peek() == TokenKind::RParen {
                        break;
                    }
                    params.push(self.parse_param());
                }
            }
            self.expect(TokenKind::RParen);
            params
        } else {
            vec![]
        };

        self.expect(TokenKind::Colon);

        // Outer indent for the actor body
        self.skip_newlines();
        self.expect(TokenKind::Indent);

        let mut state_fields = Vec::new();
        let mut handlers = Vec::new();
        let mut last_span = name.span;

        loop {
            self.skip_newlines();
            match self.peek() {
                TokenKind::Dedent | TokenKind::Eof => break,
                TokenKind::Receive => {
                    let handler = self.parse_receive_handler();
                    last_span = handler.span;
                    handlers.push(handler);
                }
                TokenKind::Mutable => {
                    let field = self.parse_var_decl();
                    last_span = field.span;
                    state_fields.push(field);
                }
                _ if self.looks_like_var_decl() => {
                    let field = self.parse_var_decl();
                    last_span = field.span;
                    state_fields.push(field);
                }
                _ => break,
            }
        }

        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        ActorDef {
            span: start_span.merge(last_span),
            name,
            capability_params,
            state_fields,
            handlers,
            exported,
        }
    }

    fn parse_receive_handler(&mut self) -> ReceiveHandler {
        let kw = self.expect(TokenKind::Receive);
        let name = self.parse_ident();

        // Optional message parameters: `receive process(data: Payload):`
        let params = if self.peek() == TokenKind::LParen {
            self.advance(); // consume `(`
            let mut params = Vec::new();
            if self.peek() != TokenKind::RParen {
                params.push(self.parse_param());
                while self.eat(TokenKind::Comma).is_some() {
                    if self.peek() == TokenKind::RParen {
                        break;
                    }
                    params.push(self.parse_param());
                }
            }
            self.expect(TokenKind::RParen);
            params
        } else {
            vec![]
        };

        // Optional `responds T` annotation
        let responds = if self.eat(TokenKind::Responds).is_some() {
            Some(self.parse_type())
        } else {
            None
        };

        self.expect(TokenKind::Colon);
        let body = self.parse_block();
        let end_span = body.span;

        ReceiveHandler {
            span: kw.span.merge(end_span),
            name,
            params,
            responds,
            body,
        }
    }

    // =======================================================================
    // Types
    // =======================================================================

    fn parse_type(&mut self) -> TypeExpr {
        let mut ty = if self.eat(TokenKind::View).is_some() {
            let start = self.tokens[self.pos - 1].span;
            let inner = self.parse_type();
            let span = start.merge(inner.span());
            TypeExpr::View(Box::new(inner), span)
        } else if self.peek() == TokenKind::Function {
            // Function type: `function(T, U) returns V`
            let start = self.advance().span;
            self.expect(TokenKind::LParen);
            let mut param_types = Vec::new();
            if self.peek() != TokenKind::RParen {
                param_types.push(self.parse_type());
                while self.eat(TokenKind::Comma).is_some() {
                    if self.peek() == TokenKind::RParen {
                        break;
                    }
                    param_types.push(self.parse_type());
                }
            }
            self.expect(TokenKind::RParen);
            self.expect(TokenKind::Returns);
            let return_type = self.parse_type();
            let span = start.merge(return_type.span());
            TypeExpr::Function(param_types, Box::new(return_type), span)
        } else {
            let ident = self.parse_type_ident();

            // Check for generic parameters: `list[string]`, `map[string, int64]`
            if self.peek() == TokenKind::LBracket {
                let start = ident.span;
                self.advance(); // consume `[`
                let mut args = Vec::new();
                if self.peek() != TokenKind::RBracket {
                    args.push(self.parse_type());
                    while self.eat(TokenKind::Comma).is_some() {
                        args.push(self.parse_type());
                    }
                }
                let end = self.expect(TokenKind::RBracket).span;
                TypeExpr::Generic(ident, args, start.merge(end))
            } else {
                TypeExpr::Named(ident)
            }
        };

        while self.eat(TokenKind::At).is_some() {
            let state = self.parse_ident();
            let span = ty.span().merge(state.span);
            ty = TypeExpr::StateQualified(Box::new(ty), state, span);
        }

        ty
    }

    /// Parse a type name — could be a keyword type (`int64`, `string`, etc.) or an identifier.
    fn parse_type_ident(&mut self) -> Ident {
        let tok = self.peek_token().clone();
        match tok.kind {
            // Built-in type keywords
            TokenKind::Int8
            | TokenKind::Int16
            | TokenKind::Int32
            | TokenKind::Int64
            | TokenKind::Uint8
            | TokenKind::Uint16
            | TokenKind::Uint32
            | TokenKind::Uint64
            | TokenKind::Float32
            | TokenKind::Float64
            | TokenKind::String_
            | TokenKind::Bool_
            | TokenKind::Bytes_
            | TokenKind::List_
            | TokenKind::Map_
            | TokenKind::Set_
            | TokenKind::Nothing
            | TokenKind::Result
            | TokenKind::Optional
            | TokenKind::Secret => {
                self.advance();
                Ident {
                    name: self.token_text(&tok).to_string(),
                    span: tok.span,
                }
            }
            TokenKind::Ident => self.parse_qualified_ident(),
            _ => {
                self.error(format!("expected type, found {:?}", tok.kind), tok.span);
                self.advance();
                Ident {
                    name: "<error>".to_string(),
                    span: tok.span,
                }
            }
        }
    }

    fn parse_qualified_ident(&mut self) -> Ident {
        let first = self.parse_ident();
        if first.name == "<error>" {
            return first;
        }

        let mut name = first.name;
        let mut span = first.span;
        while self.peek() == TokenKind::Dot {
            self.advance();
            let part = self.parse_ident();
            if part.name == "<error>" {
                break;
            }
            span = span.merge(part.span);
            name.push('.');
            name.push_str(&part.name);
        }

        Ident { name, span }
    }

    // =======================================================================
    // Blocks and Statements
    // =======================================================================

    /// Parse a block that may be either indented (multi-line) or a single
    /// statement on the same line (used for inline function bodies).
    ///
    /// - If the next non-newline token is `Indent`, delegates to `parse_block`.
    /// - Otherwise, parses exactly one statement as the body.
    fn parse_block_or_single_stmt(&mut self) -> Block {
        self.skip_newlines();
        if self.peek() == TokenKind::Indent {
            return self.parse_block();
        }
        // Single-statement body (no indentation change).
        let start = self.peek_token().span;
        let stmts = if let Some(stmt) = self.parse_stmt() {
            let end = stmt_span(&stmt);
            return Block {
                stmts: vec![stmt],
                span: start.merge(end),
            };
        } else {
            vec![]
        };
        Block { stmts, span: start }
    }

    fn parse_block(&mut self) -> Block {
        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let start = indent_tok.span;
        let mut stmts = Vec::new();
        let mut last_span = start;

        self.skip_newlines();
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            match self.parse_stmt() {
                Some(stmt) => {
                    last_span = stmt_span(&stmt);
                    stmts.push(stmt);
                }
                None => {
                    self.synchronize();
                }
            }
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        Block {
            stmts,
            span: start.merge(last_span),
        }
    }

    fn parse_stmt(&mut self) -> Option<Stmt> {
        self.skip_newlines();
        match self.peek() {
            TokenKind::Return => Some(self.parse_return_stmt()),
            TokenKind::If => Some(self.parse_if_stmt()),
            TokenKind::For => Some(self.parse_for_stmt()),
            TokenKind::While => Some(self.parse_while_stmt()),
            TokenKind::Match => Some(self.parse_match_stmt()),
            TokenKind::Use => Some(self.parse_use_stmt()),
            TokenKind::Assert => Some(self.parse_assert_stmt()),
            TokenKind::Trace => Some(self.parse_trace_stmt()),
            TokenKind::Breakpoint => Some(self.parse_breakpoint_stmt()),
            TokenKind::Respond => Some(self.parse_respond_stmt()),
            TokenKind::Comptime => Some(self.parse_comptime_type_bind_stmt()),
            TokenKind::Break => {
                let tok = self.advance();
                Some(Stmt::Break(tok.span))
            }
            TokenKind::Continue => {
                let tok = self.advance();
                Some(Stmt::Continue(tok.span))
            }
            TokenKind::Mutable => Some(Stmt::VarDecl(self.parse_var_decl())),
            _ if self.looks_like_var_decl() => Some(Stmt::VarDecl(self.parse_var_decl())),
            TokenKind::Eof => None,
            TokenKind::Dedent => None,
            _ => {
                // Try to parse as expression statement (could also be an assignment).
                let expr = self.parse_expr();
                if self.eat(TokenKind::Eq).is_some() {
                    // Assignment: `target = value`
                    let value = self.parse_expr();
                    let span = expr.span().merge(value.span());
                    Some(Stmt::Assign(AssignStmt {
                        target: expr,
                        value,
                        span,
                    }))
                } else {
                    // Check for handle block attached to expression
                    let expr = self.maybe_parse_handle(expr);
                    let span = expr.span();
                    Some(Stmt::Expr(ExprStmt { expr, span }))
                }
            }
        }
    }

    fn parse_return_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::Return);
        // Check if there's an expression on the same line
        let value = if self.peek() == TokenKind::Newline
            || self.peek() == TokenKind::Dedent
            || self.peek() == TokenKind::Eof
        {
            None
        } else {
            Some(self.parse_expr())
        };
        let end = value.as_ref().map_or(kw.span, |e| e.span());
        Stmt::Return(ReturnStmt {
            value,
            span: kw.span.merge(end),
        })
    }

    fn parse_respond_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::Respond);
        let value = self.parse_expr();
        let span = kw.span.merge(value.span());
        Stmt::Respond(RespondStmt { value, span })
    }

    fn parse_comptime_type_bind_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::Comptime);
        self.expect(TokenKind::Type);
        let name = self.parse_ident();
        self.expect(TokenKind::Eq);
        let value = self.parse_expr();
        self.expect(TokenKind::Colon);
        let body = self.parse_block();
        let span = kw.span.merge(body.span);
        Stmt::ComptimeTypeBind(ComptimeTypeBindStmt {
            name,
            value,
            body,
            span,
        })
    }

    fn parse_if_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::If);
        let condition = self.parse_expr();
        self.expect(TokenKind::Colon);
        let then_block = self.parse_block();

        let mut else_ifs = Vec::new();
        let mut else_block = None;
        let mut end_span = then_block.span;

        // Parse `else if` and `else` chains
        self.skip_newlines();
        while self.peek() == TokenKind::Else {
            self.advance(); // consume `else`
            if self.eat(TokenKind::If).is_some() {
                let cond = self.parse_expr();
                self.expect(TokenKind::Colon);
                let block = self.parse_block();
                end_span = block.span;
                else_ifs.push((cond, block));
                self.skip_newlines();
            } else {
                self.expect(TokenKind::Colon);
                let block = self.parse_block();
                end_span = block.span;
                else_block = Some(block);
                break;
            }
        }

        Stmt::If(IfStmt {
            condition,
            then_block,
            else_ifs,
            else_block,
            span: kw.span.merge(end_span),
        })
    }

    fn parse_for_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::For);
        let variable = self.parse_ident();

        // Check for `key, value` destructuring: `for key, value in map:`
        let value_variable = if self.eat(TokenKind::Comma).is_some() {
            Some(self.parse_ident())
        } else {
            None
        };

        self.expect(TokenKind::In);

        // Check for `view` keyword before iterable
        let view = self.eat(TokenKind::View).is_some();
        let iterable = self.parse_expr();
        self.expect(TokenKind::Colon);
        let body = self.parse_block();

        Stmt::For(ForStmt {
            span: kw.span.merge(body.span),
            variable,
            value_variable,
            view,
            iterable,
            body,
        })
    }

    fn parse_while_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::While);
        let condition = self.parse_expr();
        self.expect(TokenKind::Colon);
        let body = self.parse_block();

        Stmt::While(WhileStmt {
            condition,
            body: body.clone(),
            span: kw.span.merge(body.span),
        })
    }

    fn parse_match_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::Match);
        let expr = self.parse_expr();
        self.expect(TokenKind::Colon);

        self.skip_newlines();
        let indent_tok = self.expect(TokenKind::Indent);
        let mut arms = Vec::new();
        let mut last_span = indent_tok.span;

        self.skip_newlines();
        while self.peek() != TokenKind::Dedent && self.peek() != TokenKind::Eof {
            let arm = self.parse_match_arm();
            last_span = arm.span;
            arms.push(arm);
            self.skip_newlines();
        }
        if self.peek() == TokenKind::Dedent {
            last_span = self.advance().span;
        }

        Stmt::Match(MatchStmt {
            expr,
            arms,
            span: kw.span.merge(last_span),
        })
    }

    fn parse_match_arm(&mut self) -> MatchArm {
        let start = self.peek_token().span;

        let pattern = if self.peek() == TokenKind::Other {
            let tok = self.advance();
            // Check if this is `other` as catch-all (followed by `:`) or
            // `other` as an identifier pattern
            if self.peek() == TokenKind::Colon {
                Pattern::Other(tok.span)
            } else if self.peek() == TokenKind::LParen {
                // other(bindings) — destructuring with name "other"
                let name = Ident {
                    name: self.token_text(&tok).to_string(),
                    span: tok.span,
                };
                self.advance(); // consume `(`
                let mut bindings = Vec::new();
                if self.peek() != TokenKind::RParen {
                    bindings.push(self.parse_ident());
                    while self.eat(TokenKind::Comma).is_some() {
                        if self.peek() == TokenKind::RParen {
                            break;
                        }
                        bindings.push(self.parse_ident());
                    }
                }
                self.expect(TokenKind::RParen);
                Pattern::Variant(name, bindings)
            } else {
                Pattern::Other(tok.span)
            }
        } else {
            let name = self.parse_ident();
            if self.peek() == TokenKind::LParen {
                // Destructuring: `variant(a, b)`
                self.advance(); // consume `(`
                let mut bindings = Vec::new();
                if self.peek() != TokenKind::RParen {
                    bindings.push(self.parse_ident());
                    while self.eat(TokenKind::Comma).is_some() {
                        if self.peek() == TokenKind::RParen {
                            break;
                        }
                        bindings.push(self.parse_ident());
                    }
                }
                self.expect(TokenKind::RParen);
                Pattern::Variant(name, bindings)
            } else {
                Pattern::Ident(name)
            }
        };

        self.expect(TokenKind::Colon);
        let body = self.parse_block();

        MatchArm {
            span: start.merge(body.span),
            pattern,
            body,
        }
    }

    fn parse_use_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::Use);
        let mut path = self.parse_ident();
        // Support dotted paths: `use net.http`
        while self.eat(TokenKind::Dot).is_some() {
            let segment = self.parse_ident();
            path = Ident {
                name: format!("{}.{}", path.name, segment.name),
                span: path.span.merge(segment.span),
            };
        }
        let alias = if self.eat(TokenKind::As).is_some() {
            Some(self.parse_ident())
        } else {
            None
        };
        let end = alias.as_ref().map_or(path.span, |a| a.span);
        Stmt::Use(UseDecl {
            path,
            alias,
            span: kw.span.merge(end),
        })
    }

    fn parse_assert_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::Assert);
        let condition = self.parse_expr();
        // Optional message (string literal)
        let message =
            if self.peek() == TokenKind::StringLiteral || self.peek() == TokenKind::StringStart {
                Some(self.parse_expr())
            } else {
                None
            };
        let end = message.as_ref().map_or(condition.span(), |m| m.span());
        Stmt::Assert(AssertStmt {
            condition,
            message,
            span: kw.span.merge(end),
        })
    }

    fn parse_trace_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::Trace);
        let name = self.parse_ident();
        let span = kw.span.merge(name.span);
        Stmt::Trace(TraceStmt { name, span })
    }

    fn parse_breakpoint_stmt(&mut self) -> Stmt {
        let kw = self.expect(TokenKind::Breakpoint);
        let condition = if matches!(
            self.peek(),
            TokenKind::Newline | TokenKind::Dedent | TokenKind::Eof
        ) {
            None
        } else {
            Some(self.parse_expr())
        };
        let end = condition.as_ref().map_or(kw.span, Expr::span);
        Stmt::Breakpoint(BreakpointStmt {
            condition,
            span: kw.span.merge(end),
        })
    }

    fn parse_var_decl(&mut self) -> VarDecl {
        let start = self.peek_token().span;
        let mutable = self.eat(TokenKind::Mutable).is_some();
        let ty = self.parse_type();
        let name = self.parse_ident();
        self.expect(TokenKind::Eq);
        let value = self.parse_expr();
        // Check for handle block after the expression
        let value = self.maybe_parse_handle(value);
        let end = value.span();
        VarDecl {
            mutable,
            ty,
            name,
            value,
            span: start.merge(end),
        }
    }

    /// Try to determine if the current position looks like a variable declaration.
    /// A var decl starts with a type followed by an identifier followed by `=`.
    /// We look ahead to check this pattern.
    fn looks_like_var_decl(&self) -> bool {
        // Must start with something that could be a type
        if !self.is_type_start(self.peek()) {
            return false;
        }

        // Walk forward to find the pattern: Type [GenericArgs] Ident =
        let mut lookahead = 0;

        // Skip the type name
        let first = self.peek_nth(lookahead);
        if !self.is_type_start(first) {
            return false;
        }

        // Handle function types: `function(T, U) returns V name =`
        if first == TokenKind::Function {
            lookahead += 1;
            if self.peek_nth(lookahead) != TokenKind::LParen {
                return false;
            }
            lookahead += 1;
            let mut depth = 1;
            while depth > 0 && lookahead < 40 {
                match self.peek_nth(lookahead) {
                    TokenKind::LParen => depth += 1,
                    TokenKind::RParen => depth -= 1,
                    TokenKind::Eof => return false,
                    _ => {}
                }
                lookahead += 1;
            }
            // Skip `returns Type` — the return type can itself be complex
            // (e.g. `function(int64) returns function(int64) returns int64`)
            if self.peek_nth(lookahead) != TokenKind::Returns {
                return false;
            }
            lookahead += 1;
            // Skip the return type recursively (simplified: skip one or more tokens
            // until we find an identifier followed by `=`)
            return self.scan_past_type_for_var_decl(lookahead);
        }

        lookahead += 1;
        lookahead = self.skip_dotted_type_path(lookahead);

        // Handle generic args: Type[...]
        if self.peek_nth(lookahead) == TokenKind::LBracket {
            lookahead += 1;
            let mut depth = 1;
            while depth > 0 && lookahead < 20 {
                match self.peek_nth(lookahead) {
                    TokenKind::LBracket => depth += 1,
                    TokenKind::RBracket => depth -= 1,
                    TokenKind::Eof => return false,
                    _ => {}
                }
                lookahead += 1;
            }
        }
        lookahead = self.skip_state_type_qualifiers(lookahead);

        // Now we should see an identifier (or contextual keyword) followed by `=`
        let name_kind = self.peek_nth(lookahead);
        if (name_kind == TokenKind::Ident || self.is_contextual_ident(name_kind))
            && self.peek_nth(lookahead + 1) == TokenKind::Eq
        {
            return true;
        }

        false
    }

    fn is_type_start(&self, kind: TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Int8
                | TokenKind::Int16
                | TokenKind::Int32
                | TokenKind::Int64
                | TokenKind::Uint8
                | TokenKind::Uint16
                | TokenKind::Uint32
                | TokenKind::Uint64
                | TokenKind::Float32
                | TokenKind::Float64
                | TokenKind::String_
                | TokenKind::Bool_
                | TokenKind::Bytes_
                | TokenKind::List_
                | TokenKind::Map_
                | TokenKind::Set_
                | TokenKind::Nothing
                | TokenKind::Result
                | TokenKind::Optional
                | TokenKind::Secret
                | TokenKind::Function
                | TokenKind::Ident
        )
    }

    /// Starting at `lookahead`, skip past one type expression, then check for `ident =`.
    /// Used by `looks_like_var_decl` to handle function type return types.
    fn scan_past_type_for_var_decl(&self, mut lookahead: usize) -> bool {
        if lookahead >= 60 {
            return false;
        }
        let kind = self.peek_nth(lookahead);
        if kind == TokenKind::Function {
            // Nested function type: `function(…) returns T`
            lookahead += 1;
            if self.peek_nth(lookahead) != TokenKind::LParen {
                return false;
            }
            lookahead += 1;
            let mut depth = 1;
            while depth > 0 && lookahead < 60 {
                match self.peek_nth(lookahead) {
                    TokenKind::LParen => depth += 1,
                    TokenKind::RParen => depth -= 1,
                    TokenKind::Eof => return false,
                    _ => {}
                }
                lookahead += 1;
            }
            if self.peek_nth(lookahead) != TokenKind::Returns {
                return false;
            }
            lookahead += 1;
            return self.scan_past_type_for_var_decl(lookahead);
        }
        if !self.is_type_start(kind) {
            return false;
        }
        lookahead += 1;
        lookahead = self.skip_dotted_type_path(lookahead);
        // Skip generic args: Type[…]
        if self.peek_nth(lookahead) == TokenKind::LBracket {
            lookahead += 1;
            let mut depth = 1;
            while depth > 0 && lookahead < 60 {
                match self.peek_nth(lookahead) {
                    TokenKind::LBracket => depth += 1,
                    TokenKind::RBracket => depth -= 1,
                    TokenKind::Eof => return false,
                    _ => {}
                }
                lookahead += 1;
            }
        }
        lookahead = self.skip_state_type_qualifiers(lookahead);
        let name_kind = self.peek_nth(lookahead);
        (name_kind == TokenKind::Ident || self.is_contextual_ident(name_kind))
            && self.peek_nth(lookahead + 1) == TokenKind::Eq
    }

    fn skip_dotted_type_path(&self, mut lookahead: usize) -> usize {
        while self.peek_nth(lookahead) == TokenKind::Dot
            && (self.peek_nth(lookahead + 1) == TokenKind::Ident
                || self.is_contextual_ident(self.peek_nth(lookahead + 1)))
        {
            lookahead += 2;
        }
        lookahead
    }

    fn skip_state_type_qualifiers(&self, mut lookahead: usize) -> usize {
        while self.peek_nth(lookahead) == TokenKind::At
            && (self.peek_nth(lookahead + 1) == TokenKind::Ident
                || self.is_contextual_ident(self.peek_nth(lookahead + 1)))
        {
            lookahead += 2;
        }
        lookahead
    }

    // =======================================================================
    // Handle blocks
    // =======================================================================

    fn parse_handle_suffix(&mut self, target_span: Span) -> (Option<Ident>, Block, Span) {
        self.advance(); // consume `handle`

        // `handle error:` or `handle:`
        let error_name = if self.peek() == TokenKind::Error {
            let tok = self.advance();
            Some(Ident {
                name: self.token_text(&tok).to_string(),
                span: tok.span,
            })
        } else {
            None
        };

        self.expect(TokenKind::Colon);
        let block = self.parse_block_or_single_stmt();
        let span = target_span.merge(block.span);
        (error_name, block, span)
    }

    fn maybe_parse_handle(&mut self, expr: Expr) -> Expr {
        if self.peek() != TokenKind::Handle {
            return expr;
        }
        let (error_name, block, span) = self.parse_handle_suffix(expr.span());
        Expr::Handle(Box::new(expr), error_name, block, span)
    }

    // =======================================================================
    // Expressions — Pratt parser
    // =======================================================================

    fn parse_expr(&mut self) -> Expr {
        let expr = self.parse_expr_bp(0);
        // Check for pipeline: `expr into f into g(...)` — may be on the same line or
        // on subsequent indented lines.
        if self.peek() == TokenKind::Into {
            return self.parse_pipeline(expr, false);
        }
        // Multi-line form: newline + Indent + into ...
        let saved = self.pos;
        self.skip_newlines();
        if self.peek() == TokenKind::Indent {
            self.advance(); // consume Indent
            if self.peek() == TokenKind::Into {
                return self.parse_pipeline(expr, true);
            }
            // Not a pipeline — backtrack
            self.pos = saved;
        } else {
            self.pos = saved;
        }
        expr
    }

    /// Parse a pipeline starting after the initial expression.
    /// `indented` is true when we consumed an `Indent` before the first `into`
    /// (multi-line form); in that case we consume the matching `Dedent` at the end.
    fn parse_pipeline(&mut self, initial: Expr, indented: bool) -> Expr {
        let start = initial.span();
        let mut steps = Vec::new();

        while self.eat(TokenKind::Into).is_some() {
            let step_start = self.peek_token().span;
            // Parse the function reference (identifier, possibly dotted like `string.trim`)
            let func = self.parse_prefix();
            // Allow postfix field access to form dotted names like `string.trim`
            let func = self.parse_postfix_chain(func);

            // Check for extra arguments: `into f(extra1, extra2)`
            let (extra_args, step_end) = if self.peek() == TokenKind::LParen {
                self.advance(); // consume `(`
                let mut args = Vec::new();
                if self.peek() != TokenKind::RParen {
                    args.push(self.parse_call_arg());
                    while self.eat(TokenKind::Comma).is_some() {
                        if self.peek() == TokenKind::RParen {
                            break;
                        }
                        args.push(self.parse_call_arg());
                    }
                }
                let close = self.expect(TokenKind::RParen);
                (args, close.span)
            } else {
                (Vec::new(), func.span())
            };

            let mut span = step_start.merge(step_end);
            let handle = if self.peek() == TokenKind::Handle {
                let (error_name, body, handle_span) = self.parse_handle_suffix(span);
                span = handle_span;
                Some(PipelineStepHandle {
                    error_name,
                    body,
                    span: handle_span,
                })
            } else {
                None
            };

            steps.push(PipelineStep {
                function: func,
                extra_args,
                handle,
                span,
            });

            // Allow `into` steps on subsequent lines at the same indentation level.
            if indented {
                self.skip_newlines();
            }
        }

        if indented {
            // Consume the Dedent that closes the indented pipeline block.
            self.eat(TokenKind::Dedent);
        }

        let end = steps.last().map_or(start, |s| s.span);
        Expr::Pipeline(Box::new(initial), steps, start.merge(end))
    }

    /// Parse postfix field-access chain (`.field`) without call args.
    /// Used by pipeline parsing to build dotted names like `string.trim`.
    fn parse_postfix_chain(&mut self, mut expr: Expr) -> Expr {
        if self.peek() == TokenKind::LBracket && self.looks_like_generic_args() {
            expr = self.parse_generic_call(expr);
            return expr;
        }

        while self.peek() == TokenKind::Dot {
            self.advance(); // consume `.`
            let field = self.parse_ident();
            let span = expr.span().merge(field.span);
            expr = Expr::FieldAccess(Box::new(expr), field, span);
            if self.peek() == TokenKind::LBracket && self.looks_like_generic_args() {
                expr = self.parse_generic_call(expr);
            }
        }
        expr
    }

    /// Pratt parser: parse expression with minimum binding power.
    fn parse_expr_bp(&mut self, min_bp: u8) -> Expr {
        let mut lhs = self.parse_prefix();

        loop {
            // Check for postfix/infix operators
            let kind = self.peek();

            // Postfix: field access `.`, function call `(`
            match kind {
                TokenKind::Dot => {
                    let (l_bp, _) = (14, 15); // high precedence
                    if l_bp < min_bp {
                        break;
                    }
                    self.advance(); // consume `.`
                    let field = self.parse_ident();
                    let span = lhs.span().merge(field.span);
                    lhs = Expr::FieldAccess(Box::new(lhs), field, span);
                    // Check for generic/method call: `expr.method[T](args)` / `expr.method(args)`
                    if self.peek() == TokenKind::LBracket && self.looks_like_generic_args() {
                        lhs = self.parse_generic_call(lhs);
                    } else if self.peek() == TokenKind::LParen {
                        lhs = self.parse_call_args(lhs);
                    }
                    continue;
                }
                TokenKind::LParen => {
                    let (l_bp, _) = (14, 15);
                    if l_bp < min_bp {
                        break;
                    }
                    lhs = self.parse_call_args(lhs);
                    continue;
                }
                TokenKind::LBracket => {
                    // Generic call: `name[Type](args)`
                    // But only if followed by something that looks like a type
                    // and eventually `](` — this is tricky. For now, handle the
                    // common case.
                    let (l_bp, _) = (14, 15);
                    if l_bp < min_bp {
                        break;
                    }
                    if self.looks_like_generic_args() {
                        lhs = self.parse_generic_call(lhs);
                        continue;
                    }
                    break;
                }
                _ => {}
            }

            // Handle block as postfix
            if kind == TokenKind::Handle {
                let (l_bp, _) = (1, 2);
                if l_bp < min_bp {
                    break;
                }
                lhs = self.maybe_parse_handle(lhs);
                continue;
            }

            // `expr at state_name` — machine state check (postfix keyword)
            if kind == TokenKind::At {
                let (l_bp, _) = (7, 8); // same precedence as comparisons
                if l_bp < min_bp {
                    break;
                }
                self.advance(); // consume `at`
                let state_name = self.parse_ident();
                let span = lhs.span().merge(state_name.span);
                lhs = Expr::At(Box::new(lhs), state_name, span);
                continue;
            }

            // Infix binary operators
            if let Some((l_bp, r_bp)) = infix_binding_power(kind) {
                if l_bp < min_bp {
                    break;
                }
                let op_tok = self.advance();
                let op = token_to_binop(op_tok.kind);
                let rhs = self.parse_expr_bp(r_bp);
                let span = lhs.span().merge(rhs.span());
                lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs), span);
                continue;
            }

            break;
        }

        lhs
    }

    fn parse_prefix(&mut self) -> Expr {
        let tok = self.peek_token().clone();
        match tok.kind {
            TokenKind::IntLiteral => {
                self.advance();
                let value = self.parse_i128_token(&tok, "integer literal");
                Expr::IntLiteral(value, tok.span)
            }
            TokenKind::FloatLiteral => {
                self.advance();
                let text = self.token_text(&tok);
                let value = text.parse::<f64>().unwrap_or(0.0);
                Expr::FloatLiteral(value, tok.span)
            }
            TokenKind::StringLiteral => {
                self.advance();
                let text = self.token_text(&tok);
                // Strip the surrounding quotes
                let raw = if text.len() >= 2 {
                    &text[1..text.len() - 1]
                } else {
                    text
                };
                let inner = unescape_string(raw);
                Expr::StringLiteral(inner, tok.span)
            }
            TokenKind::StringStart => {
                // String interpolation: StringStart expr (StringMid expr)* StringEnd
                self.advance();
                let start_span = tok.span;
                let start_text = self.token_text(&tok);
                let mut parts = Vec::new();

                // StringStart includes the opening quote — strip it
                let literal = if !start_text.is_empty() {
                    &start_text[1..]
                } else {
                    start_text
                };
                if !literal.is_empty() {
                    parts.push(StringPart::Literal(unescape_string(literal)));
                }

                // Parse interpolated expression
                let expr = self.parse_expr();
                parts.push(StringPart::Expr(Box::new(expr)));

                // Consume StringMid/StringEnd segments
                loop {
                    match self.peek() {
                        TokenKind::StringEnd => {
                            let end_tok = self.peek_token().clone();
                            self.advance();
                            let end_text = self.token_text(&end_tok);
                            // StringEnd includes the closing quote — strip it
                            let literal = if !end_text.is_empty() {
                                &end_text[..end_text.len() - 1]
                            } else {
                                end_text
                            };
                            if !literal.is_empty() {
                                parts.push(StringPart::Literal(unescape_string(literal)));
                            }
                            let span = start_span.merge(end_tok.span);
                            return Expr::StringInterpolation(parts, span);
                        }
                        TokenKind::StringMid => {
                            let mid_tok = self.peek_token().clone();
                            self.advance();
                            let mid_text = self.token_text(&mid_tok);
                            if !mid_text.is_empty() {
                                parts.push(StringPart::Literal(unescape_string(mid_text)));
                            }
                            // Parse the next interpolated expression
                            let expr = self.parse_expr();
                            parts.push(StringPart::Expr(Box::new(expr)));
                        }
                        TokenKind::Eof => {
                            return Expr::StringInterpolation(parts, start_span);
                        }
                        _ => {
                            // Unexpected token — error recovery
                            self.advance();
                        }
                    }
                }
            }
            TokenKind::True => {
                self.advance();
                Expr::BoolLiteral(true, tok.span)
            }
            TokenKind::False => {
                self.advance();
                Expr::BoolLiteral(false, tok.span)
            }
            TokenKind::Nothing => {
                self.advance();
                Expr::Nothing(tok.span)
            }
            TokenKind::None => {
                self.advance();
                Expr::None(tok.span)
            }
            TokenKind::Not | TokenKind::Bang => {
                self.advance();
                let operand = self.parse_expr_bp(13); // unary has high precedence
                let span = tok.span.merge(operand.span());
                Expr::Unary(UnaryOp::Not, Box::new(operand), span)
            }
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_expr_bp(13);
                let span = tok.span.merge(operand.span());
                Expr::Unary(UnaryOp::Neg, Box::new(operand), span)
            }
            TokenKind::LParen => {
                self.advance();
                let inner = self.parse_expr();
                let close = self.expect(TokenKind::RParen);
                Expr::Paren(Box::new(inner), tok.span.merge(close.span))
            }
            TokenKind::View => {
                self.advance();
                let inner = self.parse_expr_bp(13);
                let span = tok.span.merge(inner.span());
                Expr::View(Box::new(inner), span)
            }
            TokenKind::Declassify => {
                self.advance();
                let inner = self.parse_expr_bp(13);
                let span = tok.span.merge(inner.span());
                Expr::Declassify(Box::new(inner), span)
            }
            TokenKind::Coarsen => {
                self.advance();
                let inner = self.parse_expr_bp(13);
                let span = tok.span.merge(inner.span());
                Expr::Coarsen(Box::new(inner), span)
            }
            TokenKind::Spawn => {
                self.advance();
                let inner = self.parse_expr_bp(13);
                let span = tok.span.merge(inner.span());
                Expr::Spawn(Box::new(inner), span)
            }
            TokenKind::Clone => {
                self.advance();
                let inner = self.parse_expr_bp(13);
                let span = tok.span.merge(inner.span());
                Expr::Clone(Box::new(inner), span)
            }
            TokenKind::Send => {
                self.advance();
                let inner = self.parse_expr_bp(0);
                let span = tok.span.merge(inner.span());
                Expr::Send(Box::new(inner), span)
            }
            TokenKind::Ask => {
                self.advance();
                let inner = self.parse_expr_bp(0);
                let span = tok.span.merge(inner.span());
                Expr::Ask(Box::new(inner), span)
            }
            TokenKind::Run => {
                self.advance();
                let inner = self.parse_expr_bp(0);
                let span = tok.span.merge(inner.span());
                Expr::Run(Box::new(inner), span)
            }
            TokenKind::Join => {
                self.advance();
                // Use min_bp=2 so the `handle` postfix (l_bp=1) is NOT consumed
                // inside `join`; instead it wraps the whole `join expr` expression.
                let inner = self.parse_expr_bp(2);
                let span = tok.span.merge(inner.span());
                Expr::Join(Box::new(inner), span)
            }
            TokenKind::Cancel => {
                self.advance();
                let inner = self.parse_expr_bp(2);
                let span = tok.span.merge(inner.span());
                Expr::Cancel(Box::new(inner), span)
            }
            TokenKind::Ok => {
                self.advance();
                self.expect(TokenKind::LParen);
                let inner = self.parse_expr();
                let close = self.expect(TokenKind::RParen);
                Expr::Ok(Box::new(inner), tok.span.merge(close.span))
            }
            TokenKind::Fail => {
                self.advance();
                self.expect(TokenKind::LParen);
                let inner = self.parse_expr();
                let close = self.expect(TokenKind::RParen);
                Expr::Fail(Box::new(inner), tok.span.merge(close.span))
            }
            TokenKind::Some => {
                self.advance();
                self.expect(TokenKind::LParen);
                let inner = self.parse_expr();
                let close = self.expect(TokenKind::RParen);
                Expr::Some(Box::new(inner), tok.span.merge(close.span))
            }
            TokenKind::Default => {
                self.advance();
                let inner = self.parse_expr();
                let span = tok.span.merge(inner.span());
                Expr::Default(Box::new(inner), span)
            }
            TokenKind::List_ => {
                self.advance();
                if self.peek() == TokenKind::LParen {
                    self.expect(TokenKind::LParen);
                    let mut items = Vec::new();
                    if self.peek() != TokenKind::RParen {
                        items.push(self.parse_expr());
                        while self.eat(TokenKind::Comma).is_some() {
                            if self.peek() == TokenKind::RParen {
                                break;
                            }
                            items.push(self.parse_expr());
                        }
                    }
                    let close = self.expect(TokenKind::RParen);
                    Expr::ListConstruct(items, tok.span.merge(close.span))
                } else {
                    Expr::Ident(Ident {
                        name: self.token_text(&tok).to_string(),
                        span: tok.span,
                    })
                }
            }
            TokenKind::Map_ => {
                self.advance();
                if self.peek() == TokenKind::LParen {
                    self.expect(TokenKind::LParen);
                    let mut pairs = Vec::new();
                    if self.peek() != TokenKind::RParen {
                        let key = self.parse_expr();
                        self.expect(TokenKind::Colon);
                        let val = self.parse_expr();
                        pairs.push((key, val));
                        while self.eat(TokenKind::Comma).is_some() {
                            if self.peek() == TokenKind::RParen {
                                break;
                            }
                            let key = self.parse_expr();
                            self.expect(TokenKind::Colon);
                            let val = self.parse_expr();
                            pairs.push((key, val));
                        }
                    }
                    let close = self.expect(TokenKind::RParen);
                    Expr::MapConstruct(pairs, tok.span.merge(close.span))
                } else {
                    Expr::Ident(Ident {
                        name: self.token_text(&tok).to_string(),
                        span: tok.span,
                    })
                }
            }
            TokenKind::Function => {
                // Inline function expression (anonymous, no name).
                // Syntax: `function(params) returns Type: body`
                //   - body may be indented (multi-line) or on the same line (single statement)
                self.advance(); // consume `function`
                self.expect(TokenKind::LParen);
                let params = self.parse_params();
                self.expect(TokenKind::RParen);
                let return_type = if self.eat(TokenKind::Returns).is_some() {
                    Some(self.parse_type())
                } else {
                    None
                };
                self.expect(TokenKind::Colon);
                let body = self.parse_block_or_single_stmt();
                let span = tok.span.merge(body.span);
                Expr::InlineFn(params, return_type, body, span)
            }
            TokenKind::Self_ => {
                self.advance();
                Expr::Ident(Ident {
                    name: "self".to_string(),
                    span: tok.span,
                })
            }
            TokenKind::Value => {
                self.advance();
                Expr::Ident(Ident {
                    name: "value".to_string(),
                    span: tok.span,
                })
            }
            TokenKind::Ident => {
                let ident = self.parse_ident();
                Expr::Ident(ident)
            }
            kind if self.is_contextual_ident(kind) => {
                let ident = self.parse_ident();
                Expr::Ident(ident)
            }
            // Type keywords that can also appear as expressions (for Type.method calls)
            kind if self.is_type_start(kind) => {
                let ident = self.parse_type_ident();
                Expr::Ident(ident)
            }
            _ => {
                self.error(
                    format!("expected expression, found {:?}", tok.kind),
                    tok.span,
                );
                self.advance();
                Expr::Error(tok.span)
            }
        }
    }

    fn parse_call_args(&mut self, callee: Expr) -> Expr {
        let start = callee.span();
        self.expect(TokenKind::LParen);
        let mut args = Vec::new();

        if self.peek() != TokenKind::RParen {
            args.push(self.parse_call_arg());
            while self.eat(TokenKind::Comma).is_some() {
                if self.peek() == TokenKind::RParen {
                    break;
                }
                args.push(self.parse_call_arg());
            }
        }

        let close = self.expect(TokenKind::RParen);
        Expr::Call(Box::new(callee), args, start.merge(close.span))
    }

    fn parse_call_arg(&mut self) -> CallArg {
        // Check for named argument: `name: expr`
        // But be careful — `name: expr` could also be `view name` or just `expr`.
        // Named args: Ident Colon Expr
        if (self.peek() == TokenKind::Ident || self.is_contextual_ident(self.peek()))
            && self.peek_nth(1) == TokenKind::Colon
        {
            let name_tok = self.peek_token().clone();
            let name_text = self.token_text(&name_tok).to_string();
            self.advance(); // ident
            self.advance(); // colon
            let value = self.parse_expr();
            let span = name_tok.span.merge(value.span());
            return CallArg {
                name: Some(Ident {
                    name: name_text,
                    span: name_tok.span,
                }),
                value,
                span,
            };
        }

        let value = self.parse_expr();
        let span = value.span();
        CallArg {
            name: None,
            value,
            span,
        }
    }

    fn looks_like_generic_args(&self) -> bool {
        // Quick check: `[` followed by a type name and eventually `](`
        if self.peek() != TokenKind::LBracket {
            return false;
        }
        let mut i = 1;
        let mut depth = 1;
        while depth > 0 && i < 20 {
            match self.peek_nth(i) {
                TokenKind::LBracket => depth += 1,
                TokenKind::RBracket => depth -= 1,
                TokenKind::Eof | TokenKind::Newline => return false,
                _ => {}
            }
            i += 1;
        }
        // After `]`, we should see `(`
        self.peek_nth(i) == TokenKind::LParen
            || depth == 0 && self.peek_nth(i - 1 + 1) == TokenKind::LParen
    }

    fn parse_generic_call(&mut self, callee: Expr) -> Expr {
        let start = callee.span();
        self.expect(TokenKind::LBracket);
        let mut type_args = Vec::new();
        if self.peek() != TokenKind::RBracket {
            type_args.push(self.parse_type());
            while self.eat(TokenKind::Comma).is_some() {
                type_args.push(self.parse_type());
            }
        }
        self.expect(TokenKind::RBracket);

        self.expect(TokenKind::LParen);
        let mut args = Vec::new();
        if self.peek() != TokenKind::RParen {
            args.push(self.parse_call_arg());
            while self.eat(TokenKind::Comma).is_some() {
                if self.peek() == TokenKind::RParen {
                    break;
                }
                args.push(self.parse_call_arg());
            }
        }
        let close = self.expect(TokenKind::RParen);
        Expr::GenericCall(Box::new(callee), type_args, args, start.merge(close.span))
    }

    // =======================================================================
    // Identifier
    // =======================================================================

    fn parse_ident(&mut self) -> Ident {
        let tok = self.peek_token().clone();
        if tok.kind == TokenKind::Ident || self.is_contextual_ident(tok.kind) {
            self.advance();
            Ident {
                name: self.token_text(&tok).to_string(),
                span: tok.span,
            }
        } else {
            self.error(
                format!("expected identifier, found {:?}", tok.kind),
                tok.span,
            );
            // Don't advance — let the caller decide what to do
            Ident {
                name: "<error>".to_string(),
                span: tok.span,
            }
        }
    }

    /// Keywords that can appear as identifiers in certain contexts
    /// (e.g., parameter names, variable names, use aliases, field names).
    fn is_contextual_ident(&self, kind: TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Self_
                | TokenKind::Other
                | TokenKind::Error
                | TokenKind::Value
                | TokenKind::Serialize
                | TokenKind::Network
                | TokenKind::Default
                | TokenKind::Ok
                | TokenKind::Fail
                | TokenKind::Clone
                | TokenKind::Send
                | TokenKind::Run
                | TokenKind::Join
                | TokenKind::Cancel
                | TokenKind::Trace
                | TokenKind::Transition
                | TokenKind::Type
                | TokenKind::Bit
                | TokenKind::Bits
                | TokenKind::States
                | TokenKind::Map_
                | TokenKind::List_
        )
    }
}

// ===========================================================================
// Operator precedence tables
// ===========================================================================

/// Returns (left_binding_power, right_binding_power) for infix operators.
/// Higher numbers = tighter binding.
/// Process escape sequences in a raw string (after the surrounding quotes are stripped).
fn unescape_string(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('0') => result.push('\0'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else if c == '{' {
            if chars.as_str().starts_with('{') {
                chars.next();
                result.push('{');
            } else {
                result.push(c);
            }
        } else if c == '}' {
            if chars.as_str().starts_with('}') {
                chars.next();
                result.push('}');
            } else {
                result.push(c);
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn infix_binding_power(kind: TokenKind) -> Option<(u8, u8)> {
    match kind {
        TokenKind::PipePipe | TokenKind::Or => Some((1, 2)),
        TokenKind::AmpAmp | TokenKind::And => Some((3, 4)),
        TokenKind::EqEq | TokenKind::NotEq => Some((5, 6)),
        TokenKind::Lt | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq => Some((7, 8)),
        TokenKind::Plus | TokenKind::Minus => Some((9, 10)),
        TokenKind::Star | TokenKind::Slash | TokenKind::Modulo => Some((11, 12)),
        _ => None,
    }
}

fn token_to_binop(kind: TokenKind) -> BinOp {
    match kind {
        TokenKind::Plus => BinOp::Add,
        TokenKind::Minus => BinOp::Sub,
        TokenKind::Star => BinOp::Mul,
        TokenKind::Slash => BinOp::Div,
        TokenKind::Modulo => BinOp::Modulo,
        TokenKind::EqEq => BinOp::Eq,
        TokenKind::NotEq => BinOp::NotEq,
        TokenKind::Lt => BinOp::Lt,
        TokenKind::Gt => BinOp::Gt,
        TokenKind::LtEq => BinOp::LtEq,
        TokenKind::GtEq => BinOp::GtEq,
        TokenKind::AmpAmp | TokenKind::And => BinOp::And,
        TokenKind::PipePipe | TokenKind::Or => BinOp::Or,
        _ => unreachable!("not a binary operator: {:?}", kind),
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
        Stmt::Break(s) | Stmt::Continue(s) => *s,
    }
}

// ===========================================================================
// Public convenience function
// ===========================================================================

/// Parse source text into an AST. Lexes first, then parses.
pub fn parse(source: &str, file: jett_common::FileId) -> ParseResult {
    let lex_result = jett_lexer::tokenize(source, file);

    // Convert lex errors to diagnostics
    let mut diagnostics: Vec<Diagnostic> = lex_result
        .errors
        .iter()
        .map(|e| Diagnostic::error(999, &e.message, e.span))
        .collect();

    let parser = Parser::new(source, lex_result.tokens);
    let mut result = parser.parse();
    diagnostics.append(&mut result.errors);
    result.errors = diagnostics;
    result
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use jett_common::FileId;

    fn parse_str(source: &str) -> ParseResult {
        parse(source, FileId::new(0))
    }

    // -----------------------------------------------------------------------
    // Parsing a simple function
    // -----------------------------------------------------------------------

    #[test]
    fn parse_simple_function() {
        let src = "\
function add(a: int64, b: int64) returns int64:
    return a + b
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 1);
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.name.name, "add");
                assert_eq!(f.params.len(), 2);
                assert_eq!(f.params[0].name.name, "a");
                assert_eq!(f.params[1].name.name, "b");
                assert!(f.return_type.is_some());
                assert_eq!(f.body.stmts.len(), 1);
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_function_returns_nothing() {
        let src = "\
function greet(view stdout: Stdout, name: string) returns nothing:
    return nothing
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.name.name, "greet");
                assert_eq!(f.params.len(), 2);
                assert!(f.params[0].view);
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_exported_declarations() {
        let src = "\
export function exposed() returns nothing:
    return nothing

export struct User:
    id: int64

export enum Color:
    red

export bitfield Flags:
    active: 1 bit

export network bitfield WireFlags:
    active: 1 bit

export type Port = int64

export interface Speaker:
    function speak(view self: Speaker) returns string

export machine Session:
    states:
        guest

export actor Worker:
    receive ping:
        return nothing
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 9);
        assert!(matches!(&result.module.items[0], Item::Function(f) if f.exported));
        assert!(matches!(&result.module.items[1], Item::Struct(s) if s.exported));
        assert!(matches!(&result.module.items[2], Item::Enum(e) if e.exported));
        assert!(
            matches!(&result.module.items[3], Item::Bitfield(b) if b.exported && !b.network_order)
        );
        assert!(
            matches!(&result.module.items[4], Item::Bitfield(b) if b.exported && b.network_order)
        );
        assert!(matches!(&result.module.items[5], Item::TypeAlias(t) if t.exported));
        assert!(matches!(&result.module.items[6], Item::Interface(i) if i.exported));
        assert!(matches!(&result.module.items[7], Item::Machine(m) if m.exported));
        assert!(matches!(&result.module.items[8], Item::Actor(a) if a.exported));
    }

    #[test]
    fn parse_export_root_type_alias() {
        let src = "\
export root type JsonValue = json.JsonTree
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 1);
        assert!(
            matches!(&result.module.items[0], Item::TypeAlias(t) if t.exported && t.root_exported && t.name.name == "JsonValue")
        );
    }

    #[test]
    fn parse_export_root_rejects_non_type_item() {
        let src = "\
export root function parse() returns nothing:
    return nothing
";
        let result = parse_str(src);
        assert!(!result.errors.is_empty());
        assert!(
            result.errors[0]
                .message
                .contains("expected `type` after `export root`"),
            "unexpected errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn parse_export_rejects_non_exportable_item() {
        let src = "\
export namespace app
";
        let result = parse_str(src);
        assert!(!result.errors.is_empty());
        assert!(
            result.errors[0]
                .message
                .contains("expected exportable item"),
            "unexpected errors: {:?}",
            result.errors
        );
    }

    // -----------------------------------------------------------------------
    // Parsing variable declarations
    // -----------------------------------------------------------------------

    #[test]
    fn parse_var_decl_immutable() {
        let src = "\
function main() returns nothing:
    int64 x = 42
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.body.stmts.len(), 1);
                match &f.body.stmts[0] {
                    Stmt::VarDecl(v) => {
                        assert!(!v.mutable);
                        assert_eq!(v.name.name, "x");
                        matches!(&v.value, Expr::IntLiteral(42, _));
                    }
                    other => panic!("expected VarDecl, got {:?}", other),
                }
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_var_decl_mutable() {
        let src = "\
function main() returns nothing:
    mutable int64 counter = 0
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::VarDecl(v) => {
                    assert!(v.mutable);
                    assert_eq!(v.name.name, "counter");
                }
                other => panic!("expected VarDecl, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Parsing if/else
    // -----------------------------------------------------------------------

    #[test]
    fn parse_if_else() {
        let src = "\
function classify(x: int64) returns string:
    if x > 0:
        return \"positive\"
    else if x == 0:
        return \"zero\"
    else:
        return \"negative\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.body.stmts.len(), 1);
                match &f.body.stmts[0] {
                    Stmt::If(if_stmt) => {
                        assert_eq!(if_stmt.else_ifs.len(), 1);
                        assert!(if_stmt.else_block.is_some());
                    }
                    other => panic!("expected If, got {:?}", other),
                }
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_if_no_else() {
        let src = "\
function check(x: int64) returns nothing:
    if x > 0:
        return nothing
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::If(if_stmt) => {
                    assert!(if_stmt.else_ifs.is_empty());
                    assert!(if_stmt.else_block.is_none());
                }
                other => panic!("expected If, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Parsing for/while loops
    // -----------------------------------------------------------------------

    #[test]
    fn parse_for_loop() {
        let src = "\
function process(items: list[string]) returns nothing:
    for item in items:
        break
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::For(for_stmt) => {
                    assert_eq!(for_stmt.variable.name, "item");
                    assert!(!for_stmt.view);
                }
                other => panic!("expected For, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_for_loop_with_view() {
        let src = "\
function process(items: list[string]) returns nothing:
    for item in view items:
        continue
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::For(for_stmt) => {
                    assert!(for_stmt.view);
                }
                other => panic!("expected For, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_while_loop() {
        let src = "\
function countdown(mutable count: int64) returns nothing:
    while count > 0:
        count = count - 1
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.body.stmts.len(), 1);
                match &f.body.stmts[0] {
                    Stmt::While(w) => {
                        assert_eq!(w.body.stmts.len(), 1);
                    }
                    other => panic!("expected While, got {:?}", other),
                }
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_mutual_block() {
        let src = "\
mutual:
    function is_even(n: int64) returns bool
    function is_odd(n: int64) returns bool
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Mutual(block) => {
                assert_eq!(block.declarations.len(), 2);
                assert_eq!(block.declarations[0].name.name, "is_even");
                assert_eq!(block.declarations[1].name.name, "is_odd");
                assert!(!block.declarations[0].exported);
                assert!(matches!(
                    block.declarations[0].return_type,
                    Some(TypeExpr::Named(ref ident)) if ident.name == "bool"
                ));
            }
            other => panic!("expected Mutual, got {:?}", other),
        }
    }

    #[test]
    fn parse_exported_mutual_declaration() {
        let src = "\
mutual:
    export function parse(raw: string) returns int64
    function parse_value(raw: string) returns int64
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Mutual(block) => {
                assert_eq!(block.declarations.len(), 2);
                assert!(block.declarations[0].exported);
                assert_eq!(block.declarations[0].name.name, "parse");
                assert!(!block.declarations[1].exported);
                assert_eq!(block.declarations[1].name.name, "parse_value");
            }
            other => panic!("expected Mutual, got {:?}", other),
        }
    }

    #[test]
    fn parse_interface_declaration() {
        let src = "\
interface Speaker:
    function speak(view self: Speaker) returns string
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Interface(interface) => {
                assert_eq!(interface.name.name, "Speaker");
                assert_eq!(interface.methods.len(), 1);
                assert_eq!(interface.methods[0].name.name, "speak");
            }
            other => panic!("expected Interface, got {:?}", other),
        }
    }

    #[test]
    fn parse_implement_block() {
        let src = "\
implement Speaker for Dog:
    function speak(view self: Dog) returns string:
        return \"woof\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Implement(block) => {
                assert_eq!(block.interface_name.name, "Speaker");
                assert!(matches!(
                    block.for_type,
                    TypeExpr::Named(ref ident) if ident.name == "Dog"
                ));
                assert_eq!(block.methods.len(), 1);
                assert_eq!(block.methods[0].name.name, "speak");
            }
            other => panic!("expected Implement, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Parsing struct definitions
    // -----------------------------------------------------------------------

    #[test]
    fn parse_struct_with_fields() {
        let src = "\
struct Point:
    x: float64
    y: float64
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Struct(s) => {
                assert_eq!(s.name.name, "Point");
                assert_eq!(s.fields.len(), 2);
                assert_eq!(s.fields[0].name.name, "x");
                assert_eq!(s.fields[1].name.name, "y");
                assert!(s.methods.is_empty());
            }
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn parse_struct_with_methods() {
        let src = "\
struct Point:
    x: float64
    y: float64

    function distance(view self: Point, view other: Point) returns float64:
        return 0.0
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Struct(s) => {
                assert_eq!(s.fields.len(), 2);
                assert_eq!(s.methods.len(), 1);
                assert_eq!(s.methods[0].name.name, "distance");
            }
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn parse_struct_field_serialize_name() {
        let src = "\
struct ApiResponse:
    user_name: string serialize \"userName\"
    total_count: int64 serialize \"totalCount\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Struct(s) => {
                assert_eq!(s.fields.len(), 2);
                assert_eq!(s.fields[0].serialize_name.as_deref(), Some("userName"));
                assert_eq!(s.fields[1].serialize_name.as_deref(), Some("totalCount"));
            }
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn parse_bitfield_with_width_fields() {
        let src = "\
bitfield TcpFlags:
    syn: 1 bit
    ack: 1 bit
    window_size: 16 bits
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Bitfield(bitfield) => {
                assert_eq!(bitfield.name.name, "TcpFlags");
                assert!(!bitfield.network_order);
                assert_eq!(bitfield.fields.len(), 3);
                match &bitfield.fields[2].kind {
                    BitfieldFieldKind::Bits { width, as_type } => {
                        assert_eq!(*width, 16);
                        assert!(as_type.is_none());
                    }
                    other => panic!("expected width field, got {:?}", other),
                }
            }
            other => panic!("expected Bitfield, got {:?}", other),
        }
    }

    #[test]
    fn parse_bitfield_with_network_modifier_and_payload() {
        let src = "\
bitfield network DnsHeader:
    id: 16 bits
    payload: list[uint8]
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Bitfield(bitfield) => {
                assert_eq!(bitfield.name.name, "DnsHeader");
                assert!(bitfield.network_order);
                assert_eq!(bitfield.fields.len(), 2);
                match &bitfield.fields[1].kind {
                    BitfieldFieldKind::Payload(TypeExpr::Generic(name, args, _)) => {
                        assert_eq!(name.name, "list");
                        assert_eq!(args.len(), 1);
                    }
                    other => panic!("expected payload field, got {:?}", other),
                }
            }
            other => panic!("expected Bitfield, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Parsing enum definitions
    // -----------------------------------------------------------------------

    #[test]
    fn parse_simple_enum() {
        let src = "\
enum Color:
    red
    green
    blue
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Enum(e) => {
                assert_eq!(e.name.name, "Color");
                assert_eq!(e.variants.len(), 3);
                assert_eq!(e.variants[0].name.name, "red");
                assert_eq!(e.variants[1].name.name, "green");
                assert_eq!(e.variants[2].name.name, "blue");
            }
            other => panic!("expected Enum, got {:?}", other),
        }
    }

    #[test]
    fn parse_enum_with_data() {
        let src = "\
enum Shape:
    circle(radius: float64)
    rect(width: float64, height: float64)
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Enum(e) => {
                assert_eq!(e.variants.len(), 2);
                assert_eq!(e.variants[0].fields.len(), 1);
                assert_eq!(e.variants[1].fields.len(), 2);
            }
            other => panic!("expected Enum, got {:?}", other),
        }
    }

    #[test]
    fn parse_enum_with_explicit_discriminants() {
        let src = "\
enum IpProtocol:
    icmp = 1
    tcp = 6
    udp = 17
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Enum(e) => {
                assert_eq!(e.variants.len(), 3);
                assert_eq!(e.variants[0].discriminant, Some(1));
                assert_eq!(e.variants[1].discriminant, Some(6));
                assert_eq!(e.variants[2].discriminant, Some(17));
            }
            other => panic!("expected Enum, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Parsing expressions with operator precedence
    // -----------------------------------------------------------------------

    #[test]
    fn parse_arithmetic_precedence() {
        // `a + b * c` should parse as `a + (b * c)`
        let src = "\
function f() returns int64:
    return a + b * c
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Return(r) => {
                    let val = r.value.as_ref().unwrap();
                    // Should be Binary(a, Add, Binary(b, Mul, c))
                    match val {
                        Expr::Binary(lhs, BinOp::Add, rhs, _) => {
                            assert!(matches!(lhs.as_ref(), Expr::Ident(_)));
                            assert!(matches!(rhs.as_ref(), Expr::Binary(_, BinOp::Mul, _, _)));
                        }
                        other => panic!("expected Binary Add, got {:?}", other),
                    }
                }
                other => panic!("expected Return, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_comparison_and_logic() {
        // `x > 0 && y < 10` should parse as `(x > 0) && (y < 10)`
        let src = "\
function f() returns bool:
    return x > 0 && y < 10
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Return(r) => {
                    let val = r.value.as_ref().unwrap();
                    match val {
                        Expr::Binary(_, BinOp::And, _, _) => { /* correct */ }
                        other => panic!("expected Binary And, got {:?}", other),
                    }
                }
                other => panic!("expected Return, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_unary_not() {
        let src = "\
function f() returns bool:
    return not x
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Return(r) => {
                    let val = r.value.as_ref().unwrap();
                    assert!(matches!(val, Expr::Unary(UnaryOp::Not, _, _)));
                }
                other => panic!("expected Return, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Parsing function calls and field access
    // -----------------------------------------------------------------------

    #[test]
    fn parse_function_call() {
        let src = "\
function f() returns nothing:
    foo(1, 2, 3)
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Expr(e) => match &e.expr {
                    Expr::Call(callee, args, _) => {
                        assert!(matches!(callee.as_ref(), Expr::Ident(_)));
                        assert_eq!(args.len(), 3);
                    }
                    other => panic!("expected Call, got {:?}", other),
                },
                other => panic!("expected Expr, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_method_call() {
        let src = "\
function f() returns nothing:
    Point.distance(view p1, view p2)
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Expr(e) => match &e.expr {
                    Expr::Call(callee, args, _) => {
                        // callee should be FieldAccess(Ident("Point"), "distance")
                        assert!(matches!(callee.as_ref(), Expr::FieldAccess(_, _, _)));
                        assert_eq!(args.len(), 2);
                    }
                    other => panic!("expected Call, got {:?}", other),
                },
                other => panic!("expected Expr, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Namespace
    // -----------------------------------------------------------------------

    #[test]
    fn parse_namespace() {
        let src = "namespace myapp\n";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 1);
        match &result.module.items[0] {
            Item::Namespace(ns) => {
                assert_eq!(ns.name.name, "myapp");
            }
            other => panic!("expected Namespace, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Use declarations
    // -----------------------------------------------------------------------

    #[test]
    fn parse_use_with_alias() {
        let src = "\
function f() returns nothing:
    use math
    use net as network
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.body.stmts.len(), 2);
                match &f.body.stmts[0] {
                    Stmt::Use(u) => {
                        assert_eq!(u.path.name, "math");
                        assert!(u.alias.is_none());
                    }
                    other => panic!("expected Use, got {:?}", other),
                }
                match &f.body.stmts[1] {
                    Stmt::Use(u) => {
                        assert_eq!(u.path.name, "net");
                        assert_eq!(u.alias.as_ref().unwrap().name, "network");
                    }
                    other => panic!("expected Use, got {:?}", other),
                }
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Assert
    // -----------------------------------------------------------------------

    #[test]
    fn parse_assert() {
        let src = "\
function f() returns nothing:
    assert x > 0
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Assert(a) => {
                    assert!(a.message.is_none());
                }
                other => panic!("expected Assert, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_trace_statement() {
        let src = "\
function main() returns nothing:
    int64 total = 42
    trace total
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[1] {
                Stmt::Trace(trace_stmt) => assert_eq!(trace_stmt.name.name, "total"),
                other => panic!("expected Trace, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_breakpoint_statement_with_condition() {
        let src = "\
function main() returns nothing:
    breakpoint 1 < 2
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Breakpoint(breakpoint_stmt) => {
                    assert!(matches!(
                        breakpoint_stmt.condition.as_ref(),
                        Some(Expr::Binary(_, BinOp::Lt, _, _))
                    ));
                }
                other => panic!("expected Breakpoint, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Error recovery
    // -----------------------------------------------------------------------

    #[test]
    fn error_recovery_continues_after_bad_statement() {
        let src = "\
function f() returns nothing:
    ??? bad
    int64 x = 42
";
        let result = parse_str(src);
        // Should have at least one error from the lexer or parser
        assert!(!result.errors.is_empty());
        // But should still have parsed the function with at least some content
        assert_eq!(result.module.items.len(), 1);
        match &result.module.items[0] {
            Item::Function(f) => {
                // We recovered and parsed the valid var decl
                let has_var_decl = f.body.stmts.iter().any(|s| matches!(s, Stmt::VarDecl(_)));
                assert!(
                    has_var_decl,
                    "should have recovered and parsed var decl, stmts: {:?}",
                    f.body.stmts
                );
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn error_recovery_multiple_items() {
        // Even if first item fails, second should parse
        let src = "\
namespace myapp

function f() returns nothing:
    return nothing
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 2);
    }

    // -----------------------------------------------------------------------
    // List and map construction
    // -----------------------------------------------------------------------

    #[test]
    fn parse_list_construction() {
        let src = "\
function f() returns nothing:
    list[string] names = list(\"alice\", \"bob\")
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::VarDecl(v) => match &v.value {
                    Expr::ListConstruct(items, _) => {
                        assert_eq!(items.len(), 2);
                    }
                    other => panic!("expected ListConstruct, got {:?}", other),
                },
                other => panic!("expected VarDecl, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Handle blocks
    // -----------------------------------------------------------------------

    #[test]
    fn parse_handle_error_block() {
        let src = "\
function f() returns nothing:
    string content = read_file(path) handle error:
        return nothing
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::VarDecl(v) => match &v.value {
                    Expr::Handle(_, error_name, _, _) => {
                        assert!(error_name.is_some());
                        assert_eq!(error_name.as_ref().unwrap().name, "error");
                    }
                    other => panic!("expected Handle, got {:?}", other),
                },
                other => panic!("expected VarDecl, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Global variable declarations
    // -----------------------------------------------------------------------

    #[test]
    fn parse_global_constants() {
        let src = "\
namespace config

int64 MAX_RETRIES = 5
string DEFAULT_HOST = \"localhost\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 3);
        match &result.module.items[1] {
            Item::VarDecl(v) => {
                assert_eq!(v.name.name, "MAX_RETRIES");
            }
            other => panic!("expected VarDecl, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Assignment
    // -----------------------------------------------------------------------

    #[test]
    fn parse_assignment() {
        let src = "\
function f(mutable x: int64) returns nothing:
    x = x + 1
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Assign(a) => {
                    assert!(matches!(&a.target, Expr::Ident(_)));
                }
                other => panic!("expected Assign, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Match statements
    // -----------------------------------------------------------------------

    #[test]
    fn parse_match_simple_patterns() {
        let src = "\
function f(color: Color) returns string:
    match color:
        red:
            return \"red\"
        green:
            return \"green\"
        blue:
            return \"blue\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.body.stmts.len(), 1);
                match &f.body.stmts[0] {
                    Stmt::Match(m) => {
                        assert_eq!(m.arms.len(), 3);
                        assert!(
                            matches!(&m.arms[0].pattern, Pattern::Ident(id) if id.name == "red")
                        );
                        assert!(
                            matches!(&m.arms[1].pattern, Pattern::Ident(id) if id.name == "green")
                        );
                        assert!(
                            matches!(&m.arms[2].pattern, Pattern::Ident(id) if id.name == "blue")
                        );
                        // Each arm body has one return statement
                        assert_eq!(m.arms[0].body.stmts.len(), 1);
                    }
                    other => panic!("expected Match, got {:?}", other),
                }
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_match_with_destructuring() {
        let src = "\
function f(shape: Shape) returns nothing:
    match shape:
        circle(r):
            return r
        rect(w, h):
            return w
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Match(m) => {
                    assert_eq!(m.arms.len(), 2);
                    match &m.arms[0].pattern {
                        Pattern::Variant(name, bindings) => {
                            assert_eq!(name.name, "circle");
                            assert_eq!(bindings.len(), 1);
                            assert_eq!(bindings[0].name, "r");
                        }
                        other => panic!("expected Variant pattern, got {:?}", other),
                    }
                    match &m.arms[1].pattern {
                        Pattern::Variant(name, bindings) => {
                            assert_eq!(name.name, "rect");
                            assert_eq!(bindings.len(), 2);
                            assert_eq!(bindings[0].name, "w");
                            assert_eq!(bindings[1].name, "h");
                        }
                        other => panic!("expected Variant pattern, got {:?}", other),
                    }
                }
                other => panic!("expected Match, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Verify blocks
    // -----------------------------------------------------------------------

    #[test]
    fn parse_function_with_verify_block() {
        let src = "\
function add(a: int64, b: int64) returns int64:
    return a + b

verify add:
    assert add(2, 3) == 5
    assert add(0, 0) == 0
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 2);

        // First item is the function
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.name.name, "add");
                assert_eq!(f.params.len(), 2);
            }
            other => panic!("expected Function, got {:?}", other),
        }

        // Second item is the verify block
        match &result.module.items[1] {
            Item::Verify(vb) => {
                assert_eq!(vb.name.name, "add");
                assert_eq!(vb.body.stmts.len(), 2);
                assert!(matches!(&vb.body.stmts[0], Stmt::Assert(_)));
                assert!(matches!(&vb.body.stmts[1], Stmt::Assert(_)));
            }
            other => panic!("expected Verify, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // String interpolation parsing
    // -----------------------------------------------------------------------

    /// Helper: extract the expression from a single-statement function body.
    fn extract_expr_from_return(result: &ParseResult) -> &Expr {
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Return(r) => r.value.as_ref().unwrap(),
                other => panic!("expected Return, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_string_interpolation_simple() {
        let src = "\
function greet(name: string) returns string:
    return \"hello {name}\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::StringInterpolation(parts, _) => {
                assert_eq!(parts.len(), 2);
                match &parts[0] {
                    StringPart::Literal(s) => assert_eq!(s, "hello "),
                    other => panic!("expected Literal, got {:?}", other),
                }
                match &parts[1] {
                    StringPart::Expr(e) => match e.as_ref() {
                        Expr::Ident(ident) => assert_eq!(ident.name, "name"),
                        other => panic!("expected Ident, got {:?}", other),
                    },
                    other => panic!("expected Expr, got {:?}", other),
                }
            }
            other => panic!("expected StringInterpolation, got {:?}", other),
        }
    }

    #[test]
    fn parse_string_interpolation_multiple() {
        let src = "\
function fmt(a: int64, b: int64, c: int64) returns string:
    return \"{a} + {b} = {c}\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::StringInterpolation(parts, _) => {
                // {a} then " + " then {b} then " = " then {c}
                assert_eq!(parts.len(), 5);
                assert!(matches!(&parts[0], StringPart::Expr(_)));
                assert!(matches!(&parts[1], StringPart::Literal(s) if s == " + "));
                assert!(matches!(&parts[2], StringPart::Expr(_)));
                assert!(matches!(&parts[3], StringPart::Literal(s) if s == " = "));
                assert!(matches!(&parts[4], StringPart::Expr(_)));
            }
            other => panic!("expected StringInterpolation, got {:?}", other),
        }
    }

    #[test]
    fn parse_string_interpolation_with_expression() {
        let src = "\
function fmt(a: int64, b: int64) returns string:
    return \"result: {a + b}\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::StringInterpolation(parts, _) => {
                assert_eq!(parts.len(), 2);
                match &parts[0] {
                    StringPart::Literal(s) => assert_eq!(s, "result: "),
                    other => panic!("expected Literal, got {:?}", other),
                }
                match &parts[1] {
                    StringPart::Expr(e) => {
                        assert!(matches!(e.as_ref(), Expr::Binary(_, BinOp::Add, _, _)));
                    }
                    other => panic!("expected Expr, got {:?}", other),
                }
            }
            other => panic!("expected StringInterpolation, got {:?}", other),
        }
    }

    #[test]
    fn parse_string_interpolation_with_contextual_identifier() {
        let src = "\
function f() returns string:
    return \"error: {error}\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::StringInterpolation(parts, _) => {
                assert_eq!(parts.len(), 2);
                match &parts[1] {
                    StringPart::Expr(e) => {
                        assert!(matches!(e.as_ref(), Expr::Ident(id) if id.name == "error"));
                    }
                    other => panic!("expected Expr, got {:?}", other),
                }
            }
            other => panic!("expected StringInterpolation, got {:?}", other),
        }
    }

    #[test]
    fn parse_plain_string_no_interpolation() {
        let src = "\
function greet() returns string:
    return \"plain string\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::StringLiteral(s, _) => assert_eq!(s, "plain string"),
            other => panic!("expected StringLiteral, got {:?}", other),
        }
    }

    #[test]
    fn parse_string_escaped_braces() {
        let src = "\
function braces() returns string:
    return \"{{key}}\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::StringLiteral(s, _) => assert_eq!(s, "{key}"),
            other => panic!("expected StringLiteral, got {:?}", other),
        }
    }

    #[test]
    fn parse_string_interpolation_with_escaped_braces() {
        let src = "\
function wrapped(name: string) returns string:
    return \"{{{name}}}\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::StringInterpolation(parts, _) => {
                assert_eq!(parts.len(), 3);
                assert!(matches!(&parts[0], StringPart::Literal(s) if s == "{"));
                assert!(matches!(&parts[1], StringPart::Expr(_)));
                assert!(matches!(&parts[2], StringPart::Literal(s) if s == "}"));
            }
            other => panic!("expected StringInterpolation, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Pipeline parsing
    // -----------------------------------------------------------------------

    #[test]
    fn parse_simple_pipeline() {
        // `x into f into g` should parse as Pipeline(x, [f, g])
        let src = "\
function transform(x: string) returns string:
    return x into f into g
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::Pipeline(initial, steps, _) => {
                assert!(matches!(initial.as_ref(), Expr::Ident(i) if i.name == "x"));
                assert_eq!(steps.len(), 2);
                assert!(matches!(&steps[0].function, Expr::Ident(i) if i.name == "f"));
                assert!(steps[0].extra_args.is_empty());
                assert!(matches!(&steps[1].function, Expr::Ident(i) if i.name == "g"));
                assert!(steps[1].extra_args.is_empty());
            }
            other => panic!("expected Pipeline, got {:?}", other),
        }
    }

    #[test]
    fn parse_pipeline_with_extra_args() {
        // `x into f(y)` should parse as Pipeline(x, [f with extra_args=[y]])
        let src = "\
function transform(x: string, y: string) returns string:
    return x into f(y)
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::Pipeline(initial, steps, _) => {
                assert!(matches!(initial.as_ref(), Expr::Ident(i) if i.name == "x"));
                assert_eq!(steps.len(), 1);
                assert!(matches!(&steps[0].function, Expr::Ident(i) if i.name == "f"));
                assert_eq!(steps[0].extra_args.len(), 1);
            }
            other => panic!("expected Pipeline, got {:?}", other),
        }
    }

    #[test]
    fn parse_pipeline_with_dotted_function() {
        // `x into string.trim into string.upper`
        let src = "\
function transform(x: string) returns string:
    return x into string.trim into string.upper
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::Pipeline(initial, steps, _) => {
                assert!(matches!(initial.as_ref(), Expr::Ident(i) if i.name == "x"));
                assert_eq!(steps.len(), 2);
                // First step should be string.trim (FieldAccess)
                assert!(
                    matches!(&steps[0].function, Expr::FieldAccess(_, field, _) if field.name == "trim")
                );
                assert!(steps[0].extra_args.is_empty());
                // Second step should be string.upper (FieldAccess)
                assert!(
                    matches!(&steps[1].function, Expr::FieldAccess(_, field, _) if field.name == "upper")
                );
                assert!(steps[1].extra_args.is_empty());
            }
            other => panic!("expected Pipeline, got {:?}", other),
        }
    }

    #[test]
    fn parse_pipeline_dotted_with_extra_args() {
        // `x into string.replace("old", "new")`
        let src = "\
function transform(x: string) returns string:
    return x into string.replace(\"old\", \"new\")
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::Pipeline(initial, steps, _) => {
                assert!(matches!(initial.as_ref(), Expr::Ident(i) if i.name == "x"));
                assert_eq!(steps.len(), 1);
                assert!(
                    matches!(&steps[0].function, Expr::FieldAccess(_, field, _) if field.name == "replace")
                );
                assert_eq!(steps[0].extra_args.len(), 2);
            }
            other => panic!("expected Pipeline, got {:?}", other),
        }
    }

    #[test]
    fn parse_pipeline_step_handle() {
        let src = "\
function transform(x: string, fallback: int64) returns int64:
    return x
        into int64.from_string() handle error:
            default fallback
        into plus_one
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::Pipeline(initial, steps, _) => {
                assert!(matches!(initial.as_ref(), Expr::Ident(i) if i.name == "x"));
                assert_eq!(steps.len(), 2);
                let handle = steps[0].handle.as_ref().expect("first step handle");
                assert!(matches!(&handle.error_name, Some(i) if i.name == "error"));
                assert_eq!(handle.body.stmts.len(), 1);
                assert!(steps[1].handle.is_none());
            }
            other => panic!("expected Pipeline, got {:?}", other),
        }
    }

    #[test]
    fn parse_dotted_generic_call() {
        let src = "\
function first(view items: list[int64]) returns optional[int64]:
    return list.get[int64](view items, 0)
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::GenericCall(callee, type_args, args, _) => {
                assert!(matches!(
                    callee.as_ref(),
                    Expr::FieldAccess(inner, field, _)
                        if matches!(inner.as_ref(), Expr::Ident(id) if id.name == "list")
                        && field.name == "get"
                ));
                assert_eq!(type_args.len(), 1);
                assert_eq!(args.len(), 2);
            }
            other => panic!("expected GenericCall, got {:?}", other),
        }
    }

    #[test]
    fn parse_json_serialize_generic_call() {
        let src = "\
function dump(view user: User) returns string:
    return json.serialize[User](view user)
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = extract_expr_from_return(&result);
        match expr {
            Expr::GenericCall(callee, type_args, args, _) => {
                assert!(matches!(
                    callee.as_ref(),
                    Expr::FieldAccess(inner, field, _)
                        if matches!(inner.as_ref(), Expr::Ident(id) if id.name == "json")
                        && field.name == "serialize"
                ));
                assert_eq!(type_args.len(), 1);
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected GenericCall, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Refinement type declarations
    // -----------------------------------------------------------------------

    #[test]
    fn parse_refinement_type_simple() {
        let src = "type Port = int64 where value >= 1 && value <= 65535\n";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 1);
        match &result.module.items[0] {
            Item::TypeAlias(ta) => {
                assert_eq!(ta.name.name, "Port");
                match &ta.base_type {
                    TypeExpr::Named(ident) => assert_eq!(ident.name, "int64"),
                    other => panic!("expected Named type, got {:?}", other),
                }
                assert!(ta.constraint.is_some(), "expected a where constraint");
            }
            other => panic!("expected TypeAlias, got {:?}", other),
        }
    }

    #[test]
    fn parse_type_alias_no_constraint() {
        let src = "type UserId = int64\n";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::TypeAlias(ta) => {
                assert_eq!(ta.name.name, "UserId");
                assert!(ta.constraint.is_none(), "expected no where constraint");
            }
            other => panic!("expected TypeAlias, got {:?}", other),
        }
    }

    #[test]
    fn parse_refinement_type_with_function_call() {
        // type Email = string where string.contains(value, "@")
        let src = "type Email = string where string.contains(value, \"@\")\n";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::TypeAlias(ta) => {
                assert_eq!(ta.name.name, "Email");
                assert!(ta.constraint.is_some());
            }
            other => panic!("expected TypeAlias, got {:?}", other),
        }
    }

    #[test]
    fn parse_coarsen_expression() {
        let src = "\
function process(p: Port) returns nothing:
    int64 raw = coarsen p
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::VarDecl(v) => match &v.value {
                    Expr::Coarsen(inner, _) => {
                        assert!(matches!(inner.as_ref(), Expr::Ident(i) if i.name == "p"));
                    }
                    other => panic!("expected Coarsen expression, got {:?}", other),
                },
                other => panic!("expected VarDecl, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_secret_type_in_var_decl() {
        let src = "\
function main() returns nothing:
    secret[string] api_key = \"abc\"
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::VarDecl(v) => match &v.ty {
                    TypeExpr::Generic(name, args, _) => {
                        assert_eq!(name.name, "secret");
                        assert_eq!(args.len(), 1);
                        assert!(
                            matches!(&args[0], TypeExpr::Named(inner) if inner.name == "string")
                        );
                    }
                    other => panic!("expected Generic type, got {:?}", other),
                },
                other => panic!("expected VarDecl, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_declassify_expression() {
        let src = "\
function reveal(key: secret[string]) returns string:
    return declassify key
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Return(ret) => match ret.value.as_ref() {
                    Some(Expr::Declassify(inner, _)) => {
                        assert!(matches!(inner.as_ref(), Expr::Ident(i) if i.name == "key"));
                    }
                    other => panic!("expected Declassify expression, got {:?}", other),
                },
                other => panic!("expected Return, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // State machine declarations
    // -----------------------------------------------------------------------

    #[test]
    fn parse_machine_declaration() {
        let src = "\
machine UserAuth:
    states:
        guest
        logged_in(user_id: string)
        banned(user_id: string)

    transitions:
        guest to logged_in
        logged_in to guest
        logged_in to banned
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 1);
        match &result.module.items[0] {
            Item::Machine(m) => {
                assert_eq!(m.name.name, "UserAuth");
                assert!(!m.exported);

                // States
                assert_eq!(m.states.len(), 3);
                assert_eq!(m.states[0].name.name, "guest");
                assert!(m.states[0].fields.is_empty());
                assert_eq!(m.states[1].name.name, "logged_in");
                assert_eq!(m.states[1].fields.len(), 1);
                assert_eq!(m.states[1].fields[0].name.name, "user_id");
                assert_eq!(m.states[2].name.name, "banned");
                assert_eq!(m.states[2].fields.len(), 1);

                // Transitions
                assert_eq!(m.transitions.len(), 3);
                assert_eq!(m.transitions[0].from.name, "guest");
                assert_eq!(m.transitions[0].to.name, "logged_in");
                assert_eq!(m.transitions[1].from.name, "logged_in");
                assert_eq!(m.transitions[1].to.name, "guest");
                assert_eq!(m.transitions[2].from.name, "logged_in");
                assert_eq!(m.transitions[2].to.name, "banned");
            }
            other => panic!("expected Machine, got {:?}", other),
        }
    }

    #[test]
    fn parse_at_expression() {
        let src = "\
function f(session: UserAuth) returns bool:
    return session at guest
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::Return(r) => match r.value.as_ref().unwrap() {
                    Expr::At(expr, state_name, _) => {
                        assert!(matches!(expr.as_ref(), Expr::Ident(i) if i.name == "session"));
                        assert_eq!(state_name.name, "guest");
                    }
                    other => panic!("expected At expression, got {:?}", other),
                },
                other => panic!("expected Return, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_state_qualified_type() {
        let src = "\
function capture(payment: Payment at authorized) returns Payment at captured:
    return payment
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => {
                match &f.params[0].ty {
                    TypeExpr::StateQualified(base, state, _) => {
                        assert!(matches!(base.as_ref(), TypeExpr::Named(i) if i.name == "Payment"));
                        assert_eq!(state.name, "authorized");
                    }
                    other => panic!("expected state-qualified parameter type, got {:?}", other),
                }

                match f.return_type.as_ref().expect("return type") {
                    TypeExpr::StateQualified(base, state, _) => {
                        assert!(matches!(base.as_ref(), TypeExpr::Named(i) if i.name == "Payment"));
                        assert_eq!(state.name, "captured");
                    }
                    other => panic!("expected state-qualified return type, got {:?}", other),
                }
            }
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_state_qualified_local_var_decl() {
        let src = "\
function capture(payment: Payment at captured) returns string:
    Payment at captured current = payment
    return current.receipt_id
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[0] {
                Stmt::VarDecl(v) => {
                    assert_eq!(v.name.name, "current");
                    match &v.ty {
                        TypeExpr::StateQualified(base, state, _) => {
                            assert!(
                                matches!(base.as_ref(), TypeExpr::Named(i) if i.name == "Payment")
                            );
                            assert_eq!(state.name, "captured");
                        }
                        other => panic!("expected state-qualified local type, got {:?}", other),
                    }
                }
                other => panic!("expected VarDecl, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    #[test]
    fn parse_comptime_type_bind_statement() {
        let src = "\
function f() returns string:
    string name = \"missing\"
    comptime type Bound = type.info[int64]():
        name = type.name[Bound]()
    return name
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module.items[0] {
            Item::Function(f) => match &f.body.stmts[1] {
                Stmt::ComptimeTypeBind(stmt) => {
                    assert_eq!(stmt.name.name, "Bound");
                    assert_eq!(stmt.body.stmts.len(), 1);
                    assert!(matches!(&stmt.body.stmts[0], Stmt::Assign(_)));
                }
                other => panic!("expected ComptimeTypeBind, got {:?}", other),
            },
            other => panic!("expected Function, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Property blocks
    // -----------------------------------------------------------------------

    #[test]
    fn parse_property_block_with_givens() {
        let src = "\
function identity(x: int64) returns int64:
    return x

property identity_preserves_value:
    given x: int64
    assert identity(x) == x
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 2);

        // First item is the function
        match &result.module.items[0] {
            Item::Function(f) => {
                assert_eq!(f.name.name, "identity");
            }
            other => panic!("expected Function, got {:?}", other),
        }

        // Second item is the property block
        match &result.module.items[1] {
            Item::Property(pb) => {
                assert_eq!(pb.name.name, "identity_preserves_value");
                assert_eq!(pb.givens.len(), 1);
                assert_eq!(pb.givens[0].name.name, "x");
                match &pb.givens[0].ty {
                    TypeExpr::Named(ident) => assert_eq!(ident.name, "int64"),
                    other => panic!("expected Named type, got {:?}", other),
                }
                assert_eq!(pb.body.stmts.len(), 1);
                assert!(matches!(&pb.body.stmts[0], Stmt::Assert(_)));
            }
            other => panic!("expected Property, got {:?}", other),
        }
    }

    #[test]
    fn parse_property_block_multiple_givens() {
        let src = "\
property multi_given:
    given a: int64
    given b: string
    given c: bool
    assert a == a
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 1);

        match &result.module.items[0] {
            Item::Property(pb) => {
                assert_eq!(pb.name.name, "multi_given");
                assert_eq!(pb.givens.len(), 3);
                assert_eq!(pb.givens[0].name.name, "a");
                assert_eq!(pb.givens[1].name.name, "b");
                assert_eq!(pb.givens[2].name.name, "c");
                assert_eq!(pb.body.stmts.len(), 1);
            }
            other => panic!("expected Property, got {:?}", other),
        }
    }

    #[test]
    fn parse_property_block_with_generic_given() {
        let src = "\
property list_property:
    given items: list[int64]
    assert items == items
";
        let result = parse_str(src);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.module.items.len(), 1);

        match &result.module.items[0] {
            Item::Property(pb) => {
                assert_eq!(pb.name.name, "list_property");
                assert_eq!(pb.givens.len(), 1);
                assert_eq!(pb.givens[0].name.name, "items");
                match &pb.givens[0].ty {
                    TypeExpr::Generic(ident, args, _) => {
                        assert_eq!(ident.name, "list");
                        assert_eq!(args.len(), 1);
                    }
                    other => panic!("expected Generic type, got {:?}", other),
                }
            }
            other => panic!("expected Property, got {:?}", other),
        }
    }
}
