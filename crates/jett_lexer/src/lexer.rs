use jett_common::{FileId, Span};

use crate::token::{Token, TokenKind};

/// The Jett lexer. Takes source text and produces a sequence of tokens.
///
/// Handles strict indentation (4 spaces only), string interpolation,
/// keywords, identifiers, literals, and all symbols.
pub struct Lexer<'src> {
    source: &'src str,
    file: FileId,
    /// Byte position in source
    pos: usize,
    /// All produced tokens
    tokens: Vec<Token>,
    /// Comment trivia skipped by the parser-facing token stream.
    comments: Vec<CommentTrivia>,
    /// Stack of indentation levels (in number of spaces). Always starts with [0].
    indent_stack: Vec<u32>,
    /// Whether we are at the beginning of a line (for indentation processing).
    at_line_start: bool,
    /// Tracks whether we have emitted at least one non-structural token on the current
    /// logical line. Used to avoid emitting Newline before any real content.
    emitted_content: bool,
    /// Errors collected during lexing.
    errors: Vec<LexError>,
}

/// A lexer error with location information.
#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

/// A skipped source comment kept for tools that need source-preserving trivia.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentTrivia {
    pub span: Span,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str, file: FileId) -> Self {
        Self {
            source,
            file,
            pos: 0,
            tokens: Vec::new(),
            comments: Vec::new(),
            indent_stack: vec![0],
            at_line_start: true,
            emitted_content: false,
            errors: Vec::new(),
        }
    }

    /// Tokenize the entire source and return all tokens plus any errors.
    pub fn tokenize(mut self) -> LexResult<'src> {
        while self.pos < self.source.len() {
            if self.at_line_start {
                self.process_line_start();
            } else {
                self.process_token();
            }
        }

        // At EOF, emit dedents back to level 0
        let eof_pos = self.pos as u32;
        while self.indent_stack.len() > 1 {
            self.indent_stack.pop();
            self.tokens.push(Token {
                kind: TokenKind::Dedent,
                span: Span::new(self.file, eof_pos, eof_pos),
            });
        }

        // Emit Eof
        self.tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span::new(self.file, eof_pos, eof_pos),
        });

        LexResult {
            source: self.source,
            tokens: self.tokens,
            comments: self.comments,
            errors: self.errors,
        }
    }

    // ---------------------------------------------------------------
    // Line-start processing (indentation, blank lines, comments)
    // ---------------------------------------------------------------

    fn process_line_start(&mut self) {
        let line_start = self.pos;

        // Count leading spaces and check for tabs
        let mut spaces: u32 = 0;
        let mut has_tab = false;
        while self.pos < self.source.len() {
            match self.current_byte() {
                b' ' => {
                    spaces += 1;
                    self.pos += 1;
                }
                b'\t' => {
                    has_tab = true;
                    self.pos += 1;
                    spaces += 1; // count it so we can continue, but we'll report an error
                }
                _ => break,
            }
        }

        if has_tab {
            self.errors.push(LexError {
                message: "tabs are not allowed, use 4 spaces for indentation".into(),
                span: Span::new(self.file, line_start as u32, self.pos as u32),
            });
            self.emit(TokenKind::InvalidToken, line_start, self.pos);
        }

        // Check if rest of line is blank or comment-only
        if self.at_end() || self.current_byte() == b'\n' {
            // Blank line — skip the newline
            if !self.at_end() {
                self.pos += 1; // skip '\n'
            }
            // Stay at_line_start for next line
            return;
        }

        if self.current_byte() == b'\r' {
            if self.peek_byte(1) == Some(b'\n') {
                self.pos += 2;
            } else {
                self.pos += 1;
            }
            return;
        }

        if self.current_byte() == b'#' {
            // Comment line — skip to end of line
            self.skip_comment();
            // After comment, we'll hit newline or EOF, handle in next iteration
            self.skip_newline();
            // stay at_line_start
            return;
        }

        // Check for trailing whitespace on the previous token line? No — trailing whitespace
        // is checked at end-of-line during token processing.

        // Validate indentation: must be multiple of 4
        if !has_tab && !spaces.is_multiple_of(4) {
            self.errors.push(LexError {
                message: format!(
                    "indentation must be a multiple of 4 spaces, found {} spaces",
                    spaces
                ),
                span: Span::new(self.file, line_start as u32, self.pos as u32),
            });
            self.emit(TokenKind::InvalidToken, line_start, self.pos);
        }

        let current_indent = *self.indent_stack.last().unwrap();
        let indent_pos = self.pos as u32;

        if spaces > current_indent {
            // Indentation increased — must be exactly one level (4 spaces more)
            if spaces != current_indent + 4 {
                self.errors.push(LexError {
                    message: format!(
                        "indentation increased by {} spaces, expected increase of exactly 4",
                        spaces - current_indent
                    ),
                    span: Span::new(self.file, line_start as u32, self.pos as u32),
                });
                // Still push the level so we can continue lexing
            }
            self.indent_stack.push(spaces);
            self.tokens.push(Token {
                kind: TokenKind::Indent,
                span: Span::new(self.file, indent_pos, indent_pos),
            });
        } else if spaces < current_indent {
            // Dedent — may need multiple dedent tokens
            while self.indent_stack.len() > 1 && *self.indent_stack.last().unwrap() > spaces {
                self.indent_stack.pop();
                self.tokens.push(Token {
                    kind: TokenKind::Dedent,
                    span: Span::new(self.file, indent_pos, indent_pos),
                });
            }
            if *self.indent_stack.last().unwrap() != spaces {
                self.errors.push(LexError {
                    message: format!(
                        "dedent to {} spaces does not match any outer indentation level",
                        spaces
                    ),
                    span: Span::new(self.file, line_start as u32, self.pos as u32),
                });
            }
        }
        // else: same indentation level, no Indent/Dedent needed

        self.at_line_start = false;
    }

    // ---------------------------------------------------------------
    // Token processing (non-line-start)
    // ---------------------------------------------------------------

    fn process_token(&mut self) {
        // Skip spaces (horizontal whitespace within a line)
        if self.current_byte() == b' ' {
            self.pos += 1;
            while !self.at_end() && self.current_byte() == b' ' {
                self.pos += 1;
            }
            return;
        }

        // Newline
        if self.current_byte() == b'\n' || self.current_byte() == b'\r' {
            self.handle_newline();
            return;
        }

        // Tab within a line
        if self.current_byte() == b'\t' {
            let start = self.pos;
            self.pos += 1;
            self.errors.push(LexError {
                message: "tabs are not allowed".into(),
                span: Span::new(self.file, start as u32, self.pos as u32),
            });
            self.emit(TokenKind::InvalidToken, start, self.pos);
            return;
        }

        // Comment
        if self.current_byte() == b'#' {
            self.skip_comment();
            return;
        }

        // String literal
        if self.current_byte() == b'"' {
            self.lex_string();
            return;
        }

        // Number literal
        if self.current_byte().is_ascii_digit() {
            self.lex_number();
            return;
        }

        // Identifier or keyword
        if is_ident_start(self.current_byte()) {
            self.lex_ident_or_keyword();
            return;
        }

        // Symbols
        self.lex_symbol();
    }

    // ---------------------------------------------------------------
    // Newline handling
    // ---------------------------------------------------------------

    fn handle_newline(&mut self) {
        // Check for trailing whitespace before this newline:
        // We look backwards from pos to see if there were spaces before the newline.
        self.check_trailing_whitespace();

        let start = self.pos;
        self.skip_newline();

        // Emit Newline only if we've emitted real content
        if self.emitted_content {
            self.tokens.push(Token {
                kind: TokenKind::Newline,
                span: Span::new(self.file, start as u32, self.pos as u32),
            });
        }

        self.at_line_start = true;
        self.emitted_content = false;
    }

    fn skip_newline(&mut self) {
        if !self.at_end() && self.current_byte() == b'\r' {
            self.pos += 1;
        }
        if !self.at_end() && self.current_byte() == b'\n' {
            self.pos += 1;
        }
    }

    fn skip_comment(&mut self) {
        let start = self.pos;
        while !self.at_end() && self.current_byte() != b'\n' && self.current_byte() != b'\r' {
            self.pos += 1;
        }
        self.comments.push(CommentTrivia {
            span: Span::new(self.file, start as u32, self.pos as u32),
        });
    }

    fn check_trailing_whitespace(&mut self) {
        // Walk backwards to find if spaces precede the newline
        if self.pos > 0 {
            let mut check = self.pos - 1;
            let mut found_trailing = false;
            while check > 0 && self.source.as_bytes()[check] == b' ' {
                found_trailing = true;
                check -= 1;
            }
            // Only report trailing whitespace if those spaces are not the only thing on the line
            // (blank-line whitespace is handled separately at line start)
            if found_trailing {
                let trail_start = check + 1;
                // Make sure there was actual content before the trailing spaces on this line
                // (i.e., this is not a whitespace-only line)
                let mut has_content = false;
                let mut scan = check;
                loop {
                    let b = self.source.as_bytes()[scan];
                    if b == b'\n' || b == b'\r' {
                        break;
                    }
                    if b != b' ' && b != b'\t' {
                        has_content = true;
                        break;
                    }
                    if scan == 0 {
                        break;
                    }
                    scan -= 1;
                }
                if has_content {
                    self.errors.push(LexError {
                        message: "trailing whitespace is not allowed".into(),
                        span: Span::new(self.file, trail_start as u32, self.pos as u32),
                    });
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // String lexing (with interpolation support)
    // ---------------------------------------------------------------

    fn lex_string(&mut self) {
        let quote_start = self.pos;
        self.pos += 1; // skip opening '"'

        // Scan the string to see if it contains interpolation
        let mut has_interpolation = false;
        {
            let mut scan = self.pos;
            while scan < self.source.len() {
                let b = self.source.as_bytes()[scan];
                if b == b'"' {
                    break;
                }
                if b == b'\n' || b == b'\r' {
                    break; // unterminated
                }
                // Skip escape sequences so \" doesn't prematurely end the string.
                if b == b'\\' && scan + 1 < self.source.len() {
                    scan += 2;
                    continue;
                }
                if b == b'{' {
                    // Check for escaped brace {{
                    if scan + 1 < self.source.len() && self.source.as_bytes()[scan + 1] == b'{' {
                        scan += 2;
                        continue;
                    }
                    has_interpolation = true;
                    break;
                }
                if b == b'}' {
                    // Check for escaped brace }}
                    if scan + 1 < self.source.len() && self.source.as_bytes()[scan + 1] == b'}' {
                        scan += 2;
                        continue;
                    }
                }
                scan += 1;
            }
        }

        if !has_interpolation {
            // Simple string literal — no interpolation
            self.lex_simple_string(quote_start);
        } else {
            // String with interpolation
            self.lex_interpolated_string(quote_start);
        }
    }

    fn lex_simple_string(&mut self, quote_start: usize) {
        // pos is right after the opening quote
        while !self.at_end() {
            let b = self.current_byte();
            if b == b'"' {
                self.pos += 1; // consume closing quote
                self.emit(TokenKind::StringLiteral, quote_start, self.pos);
                self.emitted_content = true;
                return;
            }
            if b == b'\n' || b == b'\r' {
                // Unterminated string
                self.errors.push(LexError {
                    message: "unterminated string literal".into(),
                    span: Span::new(self.file, quote_start as u32, self.pos as u32),
                });
                self.emit(TokenKind::InvalidToken, quote_start, self.pos);
                self.emitted_content = true;
                return;
            }
            // Skip escape sequences (e.g., \", \\, \n, \t)
            if b == b'\\' && self.peek_byte(1).is_some() {
                self.pos += 2;
                continue;
            }
            // Handle {{ and }} as literal braces (just skip past them)
            if b == b'{' && self.peek_byte(1) == Some(b'{') {
                self.pos += 2;
                continue;
            }
            if b == b'}' && self.peek_byte(1) == Some(b'}') {
                self.pos += 2;
                continue;
            }
            self.pos += 1;
        }

        // Reached end of file without closing quote
        self.errors.push(LexError {
            message: "unterminated string literal".into(),
            span: Span::new(self.file, quote_start as u32, self.pos as u32),
        });
        self.emit(TokenKind::InvalidToken, quote_start, self.pos);
        self.emitted_content = true;
    }

    fn lex_interpolated_string(&mut self, quote_start: usize) {
        // pos is right after the opening quote
        // Emit StringStart from the opening quote to the first `{`
        let start_begin = quote_start;

        // Scan to first unescaped `{`
        let seg_start = self.pos;
        while !self.at_end() {
            let b = self.current_byte();
            if b == b'\n' || b == b'\r' {
                // No interpolation found after all — shouldn't happen since we pre-scanned,
                // but handle it gracefully.
                self.pos += 1;
                self.emit(TokenKind::StringLiteral, quote_start, self.pos);
                self.emitted_content = true;
                return;
            }
            // Skip escape sequences so \" doesn't end the string
            if b == b'\\' && self.peek_byte(1).is_some() {
                let next = self.peek_byte(1).unwrap();
                if next == b'"' {
                    // \\" is an escaped quote — skip both
                    self.pos += 2;
                    continue;
                }
                self.pos += 2;
                continue;
            }
            if b == b'"' {
                // Unescaped closing quote
                self.pos += 1;
                self.emit(TokenKind::StringLiteral, quote_start, self.pos);
                self.emitted_content = true;
                return;
            }
            if b == b'{' {
                if self.peek_byte(1) == Some(b'{') {
                    self.pos += 2;
                    continue;
                }
                break;
            }
            if b == b'}' && self.peek_byte(1) == Some(b'}') {
                self.pos += 2;
                continue;
            }
            self.pos += 1;
        }

        // Emit StringStart: from quote_start (including the ") to the position just before `{`
        let _ = seg_start; // seg_start is start of string content after "
        self.emit(TokenKind::StringStart, start_begin, self.pos);
        self.emitted_content = true;

        // Now lex interpolation segments
        loop {
            if self.at_end() {
                self.errors.push(LexError {
                    message: "unterminated string literal".into(),
                    span: Span::new(self.file, quote_start as u32, self.pos as u32),
                });
                return;
            }

            // We should be at `{`
            if self.current_byte() != b'{' {
                // Unexpected — this is a bug in our lexer logic
                self.errors.push(LexError {
                    message: "expected '{' in string interpolation".into(),
                    span: Span::new(self.file, self.pos as u32, (self.pos + 1) as u32),
                });
                return;
            }

            self.pos += 1; // skip `{`

            // Lex tokens inside the interpolation until we find the matching `}`
            let mut brace_depth = 1u32;
            while !self.at_end() && brace_depth > 0 {
                // Skip whitespace
                while !self.at_end() && self.current_byte() == b' ' {
                    self.pos += 1;
                }
                if self.at_end() {
                    break;
                }

                let b = self.current_byte();
                if b == b'}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        self.pos += 1; // skip closing `}`
                        break;
                    }
                    self.pos += 1;
                    continue;
                }
                if b == b'{' {
                    brace_depth += 1;
                    self.pos += 1;
                    continue;
                }

                // Lex a token inside the interpolation expression
                if b == b'\n' || b == b'\r' {
                    // Newline inside interpolation — error
                    self.errors.push(LexError {
                        message: "newline inside string interpolation".into(),
                        span: Span::new(self.file, self.pos as u32, (self.pos + 1) as u32),
                    });
                    self.pos += 1;
                    break;
                }
                if b == b'"' {
                    // Nested string inside interpolation
                    self.lex_string();
                } else if b.is_ascii_digit() {
                    self.lex_number();
                } else if is_ident_start(b) {
                    self.lex_ident_or_keyword();
                } else {
                    self.lex_symbol();
                }
            }

            if brace_depth > 0 {
                self.errors.push(LexError {
                    message: "unterminated interpolation in string".into(),
                    span: Span::new(self.file, quote_start as u32, self.pos as u32),
                });
                return;
            }

            // After closing `}`, scan the next string segment
            let seg_start = self.pos;
            let mut found_interp = false;
            while !self.at_end() {
                let b = self.current_byte();
                // Skip escape sequences
                if b == b'\\' && self.peek_byte(1).is_some() {
                    self.pos += 2;
                    continue;
                }
                if b == b'"' {
                    // End of string — emit StringEnd
                    self.pos += 1; // consume closing quote
                    self.emit(TokenKind::StringEnd, seg_start, self.pos);
                    return;
                }
                if b == b'\n' || b == b'\r' {
                    self.errors.push(LexError {
                        message: "unterminated string literal".into(),
                        span: Span::new(self.file, quote_start as u32, self.pos as u32),
                    });
                    self.emit(TokenKind::InvalidToken, seg_start, self.pos);
                    return;
                }
                if b == b'{' {
                    if self.peek_byte(1) == Some(b'{') {
                        self.pos += 2;
                        continue;
                    }
                    // Another interpolation — emit StringMid
                    self.emit(TokenKind::StringMid, seg_start, self.pos);
                    found_interp = true;
                    break;
                }
                if b == b'}' && self.peek_byte(1) == Some(b'}') {
                    self.pos += 2;
                    continue;
                }
                self.pos += 1;
            }

            if !found_interp {
                // Reached EOF — unterminated
                self.errors.push(LexError {
                    message: "unterminated string literal".into(),
                    span: Span::new(self.file, quote_start as u32, self.pos as u32),
                });
                self.emit(TokenKind::InvalidToken, seg_start, self.pos);
                return;
            }
            // loop back to lex the next `{...}` interpolation
        }
    }

    // ---------------------------------------------------------------
    // Number lexing
    // ---------------------------------------------------------------

    fn lex_number(&mut self) {
        let start = self.pos;
        while !self.at_end() && self.current_byte().is_ascii_digit() {
            self.pos += 1;
        }

        // Check for float: digits followed by '.' followed by digit
        if !self.at_end()
            && self.current_byte() == b'.'
            && self.peek_byte(1).is_some_and(|b| b.is_ascii_digit())
        {
            self.pos += 1; // skip '.'
            while !self.at_end() && self.current_byte().is_ascii_digit() {
                self.pos += 1;
            }
            self.emit(TokenKind::FloatLiteral, start, self.pos);
        } else {
            self.emit(TokenKind::IntLiteral, start, self.pos);
        }
        self.emitted_content = true;
    }

    // ---------------------------------------------------------------
    // Identifier / keyword lexing
    // ---------------------------------------------------------------

    fn lex_ident_or_keyword(&mut self) {
        let start = self.pos;
        while !self.at_end() && is_ident_continue(self.current_byte()) {
            self.pos += 1;
        }

        let text = &self.source[start..self.pos];
        let kind = keyword_lookup(text).unwrap_or(TokenKind::Ident);
        self.emit(kind, start, self.pos);
        self.emitted_content = true;
    }

    // ---------------------------------------------------------------
    // Symbol lexing
    // ---------------------------------------------------------------

    fn lex_symbol(&mut self) {
        let start = self.pos;
        let b = self.current_byte();
        let next = self.peek_byte(1);

        let kind = match b {
            b'=' => {
                if next == Some(b'=') {
                    self.pos += 2;
                    TokenKind::EqEq
                } else {
                    self.pos += 1;
                    TokenKind::Eq
                }
            }
            b'!' => {
                if next == Some(b'=') {
                    self.pos += 2;
                    TokenKind::NotEq
                } else {
                    self.pos += 1;
                    TokenKind::Bang
                }
            }
            b'<' => {
                if next == Some(b'=') {
                    self.pos += 2;
                    TokenKind::LtEq
                } else {
                    self.pos += 1;
                    TokenKind::Lt
                }
            }
            b'>' => {
                if next == Some(b'=') {
                    self.pos += 2;
                    TokenKind::GtEq
                } else {
                    self.pos += 1;
                    TokenKind::Gt
                }
            }
            b'+' => {
                self.pos += 1;
                TokenKind::Plus
            }
            b'-' => {
                self.pos += 1;
                TokenKind::Minus
            }
            b'*' => {
                self.pos += 1;
                TokenKind::Star
            }
            b'/' => {
                self.pos += 1;
                TokenKind::Slash
            }
            b'&' => {
                if next == Some(b'&') {
                    self.pos += 2;
                    TokenKind::AmpAmp
                } else {
                    self.pos += 1;
                    self.errors.push(LexError {
                        message: "unexpected character '&', did you mean '&&'?".into(),
                        span: Span::new(self.file, start as u32, self.pos as u32),
                    });
                    TokenKind::InvalidToken
                }
            }
            b'|' => {
                if next == Some(b'|') {
                    self.pos += 2;
                    TokenKind::PipePipe
                } else {
                    self.pos += 1;
                    self.errors.push(LexError {
                        message: "unexpected character '|', did you mean '||'?".into(),
                        span: Span::new(self.file, start as u32, self.pos as u32),
                    });
                    TokenKind::InvalidToken
                }
            }
            b'.' => {
                self.pos += 1;
                TokenKind::Dot
            }
            b',' => {
                self.pos += 1;
                TokenKind::Comma
            }
            b':' => {
                self.pos += 1;
                TokenKind::Colon
            }
            b'(' => {
                self.pos += 1;
                TokenKind::LParen
            }
            b')' => {
                self.pos += 1;
                TokenKind::RParen
            }
            b'[' => {
                self.pos += 1;
                TokenKind::LBracket
            }
            b']' => {
                self.pos += 1;
                TokenKind::RBracket
            }
            b'#' => {
                // This shouldn't normally be reached because we handle # earlier,
                // but just in case:
                self.skip_comment();
                return;
            }
            _ => {
                self.pos += 1;
                self.errors.push(LexError {
                    message: format!(
                        "unexpected character '{}'",
                        self.source[start..self.pos].chars().next().unwrap_or('?')
                    ),
                    span: Span::new(self.file, start as u32, self.pos as u32),
                });
                TokenKind::InvalidToken
            }
        };

        self.emit(kind, start, self.pos);
        self.emitted_content = true;
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    fn at_end(&self) -> bool {
        self.pos >= self.source.len()
    }

    fn current_byte(&self) -> u8 {
        self.source.as_bytes()[self.pos]
    }

    fn peek_byte(&self, offset: usize) -> Option<u8> {
        self.source.as_bytes().get(self.pos + offset).copied()
    }

    fn emit(&mut self, kind: TokenKind, start: usize, end: usize) {
        self.tokens.push(Token {
            kind,
            span: Span::new(self.file, start as u32, end as u32),
        });
    }
}

/// Result of lexing.
pub struct LexResult<'src> {
    pub source: &'src str,
    pub tokens: Vec<Token>,
    pub comments: Vec<CommentTrivia>,
    pub errors: Vec<LexError>,
}

impl<'src> LexResult<'src> {
    /// Returns the text corresponding to a token's span.
    pub fn token_text(&self, token: &Token) -> &'src str {
        &self.source[token.span.start as usize..token.span.end as usize]
    }

    /// Returns the text corresponding to a comment trivia span.
    pub fn comment_text(&self, comment: &CommentTrivia) -> &'src str {
        &self.source[comment.span.start as usize..comment.span.end as usize]
    }
}

// ---------------------------------------------------------------
// Keyword lookup
// ---------------------------------------------------------------

fn keyword_lookup(text: &str) -> Option<TokenKind> {
    match text {
        // Keywords
        "function" => Some(TokenKind::Function),
        "return" => Some(TokenKind::Return),
        "returns" => Some(TokenKind::Returns),
        "if" => Some(TokenKind::If),
        "else" => Some(TokenKind::Else),
        "for" => Some(TokenKind::For),
        "in" => Some(TokenKind::In),
        "into" => Some(TokenKind::Into),
        "while" => Some(TokenKind::While),
        "struct" => Some(TokenKind::Struct),
        "enum" => Some(TokenKind::Enum),
        "match" => Some(TokenKind::Match),
        "use" => Some(TokenKind::Use),
        "mutable" => Some(TokenKind::Mutable),
        "handle" => Some(TokenKind::Handle),
        "error" => Some(TokenKind::Error),
        "default" => Some(TokenKind::Default),
        "result" => Some(TokenKind::Result),
        "ok" => Some(TokenKind::Ok),
        "fail" => Some(TokenKind::Fail),
        "clone" => Some(TokenKind::Clone),
        "view" => Some(TokenKind::View),
        "type" => Some(TokenKind::Type),
        "where" => Some(TokenKind::Where),
        "machine" => Some(TokenKind::Machine),
        "states" => Some(TokenKind::States),
        "transitions" => Some(TokenKind::Transitions),
        "to" => Some(TokenKind::To),
        "at" => Some(TokenKind::At),
        "is" => Some(TokenKind::Is),
        "actor" => Some(TokenKind::Actor),
        "receive" => Some(TokenKind::Receive),
        "responds" => Some(TokenKind::Responds),
        "send" => Some(TokenKind::Send),
        "ask" => Some(TokenKind::Ask),
        "respond" => Some(TokenKind::Respond),
        "spawn" => Some(TokenKind::Spawn),
        "run" => Some(TokenKind::Run),
        "join" => Some(TokenKind::Join),
        "cancel" => Some(TokenKind::Cancel),
        "comptime" => Some(TokenKind::Comptime),
        "verify" => Some(TokenKind::Verify),
        "property" => Some(TokenKind::Property),
        "given" => Some(TokenKind::Given),
        "trace" => Some(TokenKind::Trace),
        "breakpoint" => Some(TokenKind::Breakpoint),
        "secret" => Some(TokenKind::Secret),
        "declassify" => Some(TokenKind::Declassify),
        "coarsen" => Some(TokenKind::Coarsen),
        "serialize" => Some(TokenKind::Serialize),
        "namespace" => Some(TokenKind::Namespace),
        "export" => Some(TokenKind::Export),
        "bitfield" => Some(TokenKind::Bitfield),
        "bit" => Some(TokenKind::Bit),
        "bits" => Some(TokenKind::Bits),
        "network" => Some(TokenKind::Network),
        "implement" => Some(TokenKind::Implement),
        "interface" => Some(TokenKind::Interface),
        "mutual" => Some(TokenKind::Mutual),
        "assert" => Some(TokenKind::Assert),
        "some" => Some(TokenKind::Some),
        "none" => Some(TokenKind::None),
        "nothing" => Some(TokenKind::Nothing),
        "true" => Some(TokenKind::True),
        "false" => Some(TokenKind::False),
        "modulo" => Some(TokenKind::Modulo),
        "as" => Some(TokenKind::As),
        "break" => Some(TokenKind::Break),
        "continue" => Some(TokenKind::Continue),
        "and" => Some(TokenKind::And),
        "or" => Some(TokenKind::Or),
        "within" => Some(TokenKind::Within),
        "self" => Some(TokenKind::Self_),
        "value" => Some(TokenKind::Value),
        "transition" => Some(TokenKind::Transition),
        "optional" => Some(TokenKind::Optional),
        "other" => Some(TokenKind::Other),
        "not" => Some(TokenKind::Not),

        // Type keywords
        "int8" => Some(TokenKind::Int8),
        "int16" => Some(TokenKind::Int16),
        "int32" => Some(TokenKind::Int32),
        "int64" => Some(TokenKind::Int64),
        "uint8" => Some(TokenKind::Uint8),
        "uint16" => Some(TokenKind::Uint16),
        "uint32" => Some(TokenKind::Uint32),
        "uint64" => Some(TokenKind::Uint64),
        "float32" => Some(TokenKind::Float32),
        "float64" => Some(TokenKind::Float64),
        "string" => Some(TokenKind::String_),
        "bool" => Some(TokenKind::Bool_),
        "bytes" => Some(TokenKind::Bytes_),
        "list" => Some(TokenKind::List_),
        "map" => Some(TokenKind::Map_),
        "set" => Some(TokenKind::Set_),

        _ => None,
    }
}

// ---------------------------------------------------------------
// Character classification
// ---------------------------------------------------------------

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Convenience function: tokenize source and return (tokens, errors).
pub fn tokenize(source: &str, file: FileId) -> LexResult<'_> {
    Lexer::new(source, file).tokenize()
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> FileId {
        FileId::new(0)
    }

    fn kinds(source: &str) -> Vec<TokenKind> {
        let result = tokenize(source, file());
        result.tokens.iter().map(|t| t.kind).collect()
    }

    fn kinds_no_eof(source: &str) -> Vec<TokenKind> {
        let mut k = kinds(source);
        if k.last() == Some(&TokenKind::Eof) {
            k.pop();
        }
        k
    }

    fn text_of<'a>(result: &'a LexResult<'a>, idx: usize) -> &'a str {
        result.token_text(&result.tokens[idx])
    }

    fn comment_texts(source: &str) -> Vec<String> {
        let result = tokenize(source, file());
        result
            .comments
            .iter()
            .map(|comment| result.comment_text(comment).to_string())
            .collect()
    }

    // ===========================================
    // Basic keyword recognition
    // ===========================================

    #[test]
    fn test_all_keywords() {
        let pairs = vec![
            ("function", TokenKind::Function),
            ("return", TokenKind::Return),
            ("returns", TokenKind::Returns),
            ("if", TokenKind::If),
            ("else", TokenKind::Else),
            ("for", TokenKind::For),
            ("in", TokenKind::In),
            ("into", TokenKind::Into),
            ("while", TokenKind::While),
            ("struct", TokenKind::Struct),
            ("enum", TokenKind::Enum),
            ("match", TokenKind::Match),
            ("use", TokenKind::Use),
            ("mutable", TokenKind::Mutable),
            ("handle", TokenKind::Handle),
            ("error", TokenKind::Error),
            ("default", TokenKind::Default),
            ("result", TokenKind::Result),
            ("ok", TokenKind::Ok),
            ("fail", TokenKind::Fail),
            ("clone", TokenKind::Clone),
            ("view", TokenKind::View),
            ("type", TokenKind::Type),
            ("where", TokenKind::Where),
            ("machine", TokenKind::Machine),
            ("states", TokenKind::States),
            ("transitions", TokenKind::Transitions),
            ("to", TokenKind::To),
            ("at", TokenKind::At),
            ("is", TokenKind::Is),
            ("actor", TokenKind::Actor),
            ("receive", TokenKind::Receive),
            ("responds", TokenKind::Responds),
            ("send", TokenKind::Send),
            ("ask", TokenKind::Ask),
            ("respond", TokenKind::Respond),
            ("spawn", TokenKind::Spawn),
            ("run", TokenKind::Run),
            ("join", TokenKind::Join),
            ("cancel", TokenKind::Cancel),
            ("comptime", TokenKind::Comptime),
            ("verify", TokenKind::Verify),
            ("property", TokenKind::Property),
            ("given", TokenKind::Given),
            ("trace", TokenKind::Trace),
            ("breakpoint", TokenKind::Breakpoint),
            ("secret", TokenKind::Secret),
            ("declassify", TokenKind::Declassify),
            ("coarsen", TokenKind::Coarsen),
            ("serialize", TokenKind::Serialize),
            ("namespace", TokenKind::Namespace),
            ("export", TokenKind::Export),
            ("bitfield", TokenKind::Bitfield),
            ("bit", TokenKind::Bit),
            ("bits", TokenKind::Bits),
            ("network", TokenKind::Network),
            ("implement", TokenKind::Implement),
            ("interface", TokenKind::Interface),
            ("mutual", TokenKind::Mutual),
            ("assert", TokenKind::Assert),
            ("some", TokenKind::Some),
            ("none", TokenKind::None),
            ("nothing", TokenKind::Nothing),
            ("true", TokenKind::True),
            ("false", TokenKind::False),
            ("modulo", TokenKind::Modulo),
            ("as", TokenKind::As),
            ("break", TokenKind::Break),
            ("continue", TokenKind::Continue),
            ("and", TokenKind::And),
            ("within", TokenKind::Within),
            ("self", TokenKind::Self_),
            ("value", TokenKind::Value),
            ("transition", TokenKind::Transition),
            ("optional", TokenKind::Optional),
            ("other", TokenKind::Other),
            ("not", TokenKind::Not),
        ];

        for (text, expected_kind) in pairs {
            let k = kinds_no_eof(text);
            assert_eq!(
                k,
                vec![expected_kind],
                "keyword '{}' should produce {:?}",
                text,
                expected_kind
            );
        }
    }

    #[test]
    fn test_all_type_keywords() {
        let pairs = vec![
            ("int8", TokenKind::Int8),
            ("int16", TokenKind::Int16),
            ("int32", TokenKind::Int32),
            ("int64", TokenKind::Int64),
            ("uint8", TokenKind::Uint8),
            ("uint16", TokenKind::Uint16),
            ("uint32", TokenKind::Uint32),
            ("uint64", TokenKind::Uint64),
            ("float32", TokenKind::Float32),
            ("float64", TokenKind::Float64),
            ("string", TokenKind::String_),
            ("bool", TokenKind::Bool_),
            ("bytes", TokenKind::Bytes_),
            ("list", TokenKind::List_),
            ("map", TokenKind::Map_),
            ("set", TokenKind::Set_),
        ];

        for (text, expected_kind) in pairs {
            let k = kinds_no_eof(text);
            assert_eq!(
                k,
                vec![expected_kind],
                "type keyword '{}' should produce {:?}",
                text,
                expected_kind
            );
        }
    }

    #[test]
    fn test_identifier_not_keyword() {
        let k = kinds_no_eof("my_var");
        assert_eq!(k, vec![TokenKind::Ident]);

        let result = tokenize("hello_world", file());
        assert_eq!(text_of(&result, 0), "hello_world");
    }

    #[test]
    fn test_keyword_prefix_is_identifier() {
        // "functions" is not a keyword — "function" is, but "functions" has extra chars
        let k = kinds_no_eof("functions");
        assert_eq!(k, vec![TokenKind::Ident]);

        let k = kinds_no_eof("returnable");
        assert_eq!(k, vec![TokenKind::Ident]);
    }

    // ===========================================
    // Integer and float literals
    // ===========================================

    #[test]
    fn test_integer_literal() {
        let k = kinds_no_eof("42");
        assert_eq!(k, vec![TokenKind::IntLiteral]);

        let result = tokenize("12345", file());
        assert_eq!(text_of(&result, 0), "12345");
    }

    #[test]
    fn test_float_literal() {
        let k = kinds_no_eof("3.14");
        assert_eq!(k, vec![TokenKind::FloatLiteral]);

        let result = tokenize("2.718", file());
        assert_eq!(text_of(&result, 0), "2.718");
    }

    #[test]
    fn test_integer_followed_by_dot_no_digit() {
        // "42." without a digit after is int + dot, not float
        let k = kinds_no_eof("42.foo");
        assert_eq!(
            k,
            vec![TokenKind::IntLiteral, TokenKind::Dot, TokenKind::Ident]
        );
    }

    #[test]
    fn test_zero() {
        let k = kinds_no_eof("0");
        assert_eq!(k, vec![TokenKind::IntLiteral]);
    }

    #[test]
    fn test_float_zero() {
        let k = kinds_no_eof("0.0");
        assert_eq!(k, vec![TokenKind::FloatLiteral]);
    }

    // ===========================================
    // All symbol tokens
    // ===========================================

    #[test]
    fn test_single_char_symbols() {
        let pairs = vec![
            ("=", TokenKind::Eq),
            ("+", TokenKind::Plus),
            ("-", TokenKind::Minus),
            ("*", TokenKind::Star),
            ("/", TokenKind::Slash),
            ("!", TokenKind::Bang),
            (".", TokenKind::Dot),
            (",", TokenKind::Comma),
            (":", TokenKind::Colon),
            ("(", TokenKind::LParen),
            (")", TokenKind::RParen),
            ("[", TokenKind::LBracket),
            ("]", TokenKind::RBracket),
            ("<", TokenKind::Lt),
            (">", TokenKind::Gt),
        ];
        for (text, expected) in pairs {
            let k = kinds_no_eof(text);
            assert_eq!(k, vec![expected], "symbol '{}' failed", text);
        }
    }

    #[test]
    fn test_double_char_symbols() {
        let pairs = vec![
            ("==", TokenKind::EqEq),
            ("!=", TokenKind::NotEq),
            ("<=", TokenKind::LtEq),
            (">=", TokenKind::GtEq),
            ("&&", TokenKind::AmpAmp),
            ("||", TokenKind::PipePipe),
        ];
        for (text, expected) in pairs {
            let k = kinds_no_eof(text);
            assert_eq!(k, vec![expected], "symbol '{}' failed", text);
        }
    }

    #[test]
    fn test_symbol_adjacency() {
        // "a+b" should tokenize as ident, plus, ident
        let k = kinds_no_eof("a+b");
        assert_eq!(k, vec![TokenKind::Ident, TokenKind::Plus, TokenKind::Ident]);
    }

    #[test]
    fn test_eq_vs_eqeq() {
        let k = kinds_no_eof("x == y");
        assert_eq!(k, vec![TokenKind::Ident, TokenKind::EqEq, TokenKind::Ident]);

        let k = kinds_no_eof("x = y");
        assert_eq!(k, vec![TokenKind::Ident, TokenKind::Eq, TokenKind::Ident]);
    }

    // ===========================================
    // String literals
    // ===========================================

    #[test]
    fn test_simple_string() {
        let k = kinds_no_eof("\"hello\"");
        assert_eq!(k, vec![TokenKind::StringLiteral]);

        let result = tokenize("\"hello world\"", file());
        assert_eq!(text_of(&result, 0), "\"hello world\"");
    }

    #[test]
    fn test_empty_string() {
        let k = kinds_no_eof("\"\"");
        assert_eq!(k, vec![TokenKind::StringLiteral]);
    }

    #[test]
    fn test_string_interpolation_simple() {
        // "hello {name}" -> StringStart, Ident, StringEnd
        let k = kinds_no_eof("\"hello {name}\"");
        assert_eq!(
            k,
            vec![
                TokenKind::StringStart,
                TokenKind::Ident,
                TokenKind::StringEnd
            ]
        );
    }

    #[test]
    fn test_string_interpolation_expression() {
        // "{a + b}" -> StringStart, Ident, Plus, Ident, StringEnd
        let k = kinds_no_eof("\"{a + b}\"");
        assert_eq!(
            k,
            vec![
                TokenKind::StringStart,
                TokenKind::Ident,
                TokenKind::Plus,
                TokenKind::Ident,
                TokenKind::StringEnd,
            ]
        );
    }

    #[test]
    fn test_string_interpolation_multiple() {
        // "{a} and {b}" -> StringStart, Ident, StringMid, Ident, StringEnd
        let k = kinds_no_eof("\"{a} and {b}\"");
        assert_eq!(
            k,
            vec![
                TokenKind::StringStart,
                TokenKind::Ident,
                TokenKind::StringMid,
                TokenKind::Ident,
                TokenKind::StringEnd,
            ]
        );
    }

    #[test]
    fn test_string_escaped_braces() {
        // "{{key}}" has escaped braces, no interpolation
        let k = kinds_no_eof("\"{{key}}\"");
        assert_eq!(k, vec![TokenKind::StringLiteral]);
    }

    #[test]
    fn test_string_interpolation_with_dot() {
        // "total: {order.total}" -> StringStart, Ident, Dot, Ident, StringEnd
        let k = kinds_no_eof("\"total: {order.total}\"");
        assert_eq!(
            k,
            vec![
                TokenKind::StringStart,
                TokenKind::Ident,
                TokenKind::Dot,
                TokenKind::Ident,
                TokenKind::StringEnd,
            ]
        );
    }

    // ===========================================
    // Comments
    // ===========================================

    #[test]
    fn test_comment_only_line() {
        let k = kinds_no_eof("# this is a comment");
        assert_eq!(k, Vec::<TokenKind>::new());
        assert_eq!(
            comment_texts("# this is a comment"),
            vec!["# this is a comment"]
        );
    }

    #[test]
    fn test_comment_after_code() {
        // "x # comment" -> just Ident (comment is skipped)
        let k = kinds_no_eof("x # comment");
        assert_eq!(k, vec![TokenKind::Ident]);
        assert_eq!(comment_texts("x # comment"), vec!["# comment"]);
    }

    #[test]
    fn test_only_comments() {
        let source = "# line 1\n# line 2\n# line 3\n";
        let k = kinds_no_eof(source);
        assert_eq!(k, Vec::<TokenKind>::new());
        assert_eq!(
            comment_texts(source),
            vec!["# line 1", "# line 2", "# line 3"]
        );
    }

    // ===========================================
    // Newlines
    // ===========================================

    #[test]
    fn test_newline_between_statements() {
        let source = "x\ny";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![TokenKind::Ident, TokenKind::Newline, TokenKind::Ident]
        );
    }

    #[test]
    fn test_blank_lines_collapsed() {
        let source = "x\n\n\ny";
        let k = kinds_no_eof(source);
        // Blank lines are skipped; only one Newline between x and y
        assert_eq!(
            k,
            vec![TokenKind::Ident, TokenKind::Newline, TokenKind::Ident]
        );
    }

    #[test]
    fn test_no_leading_newline() {
        let source = "\n\nx";
        let k = kinds_no_eof(source);
        assert_eq!(k, vec![TokenKind::Ident]);
    }

    // ===========================================
    // Indentation (Indent/Dedent generation)
    // ===========================================

    #[test]
    fn test_simple_indent() {
        let source = "if x:\n    return y";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::If,
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Return,
                TokenKind::Ident,
                TokenKind::Dedent, // emitted at EOF
            ]
        );
    }

    #[test]
    fn test_indent_dedent() {
        let source = "if x:\n    y\nz";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::If,
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident,
                TokenKind::Newline,
                TokenKind::Dedent,
                TokenKind::Ident,
            ]
        );
    }

    #[test]
    fn test_multiple_dedents() {
        let source = "a:\n    b:\n        c\nd";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident,
                TokenKind::Newline,
                TokenKind::Dedent,
                TokenKind::Dedent,
                TokenKind::Ident,
            ]
        );
    }

    #[test]
    fn test_deeply_nested_indentation() {
        let source = "a:\n    b:\n        c:\n            d\n";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident,
                TokenKind::Newline,
                // 3 dedents at EOF
                TokenKind::Dedent,
                TokenKind::Dedent,
                TokenKind::Dedent,
            ]
        );
    }

    #[test]
    fn test_indent_with_blank_lines() {
        let source = "a:\n    b\n\n    c";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident,
                TokenKind::Newline,
                // blank line is skipped
                TokenKind::Ident,
                TokenKind::Dedent, // at EOF
            ]
        );
    }

    // ===========================================
    // Indentation errors
    // ===========================================

    #[test]
    fn test_tab_indentation_error() {
        let source = "if x:\n\ty";
        let result = tokenize(source, file());
        assert!(
            !result.errors.is_empty(),
            "should report error for tab indentation"
        );
        assert!(result.errors[0].message.contains("tabs"));
    }

    #[test]
    fn test_non_multiple_of_4_error() {
        let source = "if x:\n  y";
        let result = tokenize(source, file());
        assert!(
            !result.errors.is_empty(),
            "should report error for 2-space indent"
        );
        assert!(result.errors[0].message.contains("multiple of 4"));
    }

    // ===========================================
    // EOF
    // ===========================================

    #[test]
    fn test_eof_token() {
        let source = "x";
        let k = kinds(source);
        assert_eq!(k.last(), Some(&TokenKind::Eof));
    }

    // ===========================================
    // Edge cases
    // ===========================================

    #[test]
    fn test_empty_input() {
        let k = kinds("");
        assert_eq!(k, vec![TokenKind::Eof]);
    }

    #[test]
    fn test_whitespace_only() {
        let k = kinds("   ");
        // Whitespace only with no newline — pos starts at_line_start, the
        // spaces are consumed, then it's blank (at end), so no tokens emitted.
        assert_eq!(k, vec![TokenKind::Eof]);
    }

    #[test]
    fn test_newlines_only() {
        let k = kinds("\n\n\n");
        assert_eq!(k, vec![TokenKind::Eof]);
    }

    #[test]
    fn test_error_recovery_unknown_char() {
        let source = "x @ y";
        let result = tokenize(source, file());
        assert!(!result.errors.is_empty());
        // The lexer should continue and produce tokens for x, @(invalid), y
        let k: Vec<_> = result.tokens.iter().map(|t| t.kind).collect();
        assert!(k.contains(&TokenKind::Ident));
        assert!(k.contains(&TokenKind::InvalidToken));
    }

    #[test]
    fn test_unterminated_string() {
        let source = "\"hello\nworld";
        let result = tokenize(source, file());
        assert!(!result.errors.is_empty());
        assert!(result.errors[0].message.contains("unterminated"));
    }

    // ===========================================
    // Full program snippets
    // ===========================================

    #[test]
    fn test_function_definition() {
        let source = "function add(a: int64, b: int64) returns int64:\n    return a + b";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Function,
                TokenKind::Ident, // add
                TokenKind::LParen,
                TokenKind::Ident, // a
                TokenKind::Colon,
                TokenKind::Int64,
                TokenKind::Comma,
                TokenKind::Ident, // b
                TokenKind::Colon,
                TokenKind::Int64,
                TokenKind::RParen,
                TokenKind::Returns,
                TokenKind::Int64,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Return,
                TokenKind::Ident, // a
                TokenKind::Plus,
                TokenKind::Ident,  // b
                TokenKind::Dedent, // at EOF
            ]
        );
    }

    #[test]
    fn test_variable_declaration() {
        let source = "int64 x = 42";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Int64,
                TokenKind::Ident,
                TokenKind::Eq,
                TokenKind::IntLiteral,
            ]
        );
    }

    #[test]
    fn test_if_else() {
        let source = "if x > 0:\n    return true\nelse:\n    return false";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::If,
                TokenKind::Ident,
                TokenKind::Gt,
                TokenKind::IntLiteral,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Return,
                TokenKind::True,
                TokenKind::Newline,
                TokenKind::Dedent,
                TokenKind::Else,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Return,
                TokenKind::False,
                TokenKind::Dedent, // at EOF
            ]
        );
    }

    #[test]
    fn test_struct_definition() {
        let source = "struct Point:\n    x: float64\n    y: float64";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Struct,
                TokenKind::Ident, // Point
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident, // x
                TokenKind::Colon,
                TokenKind::Float64,
                TokenKind::Newline,
                TokenKind::Ident, // y
                TokenKind::Colon,
                TokenKind::Float64,
                TokenKind::Dedent, // at EOF
            ]
        );
    }

    #[test]
    fn test_string_interpolation_in_context() {
        let source = "\"hello {name}\"";
        let result = tokenize(source, file());
        let k: Vec<_> = result.tokens.iter().map(|t| t.kind).collect();
        assert_eq!(
            k,
            vec![
                TokenKind::StringStart,
                TokenKind::Ident,
                TokenKind::StringEnd,
                TokenKind::Eof,
            ]
        );
        // StringStart span covers "hello
        assert_eq!(text_of(&result, 0), "\"hello ");
        // Ident span covers name
        assert_eq!(text_of(&result, 1), "name");
        // StringEnd span covers "
        assert_eq!(text_of(&result, 2), "\"");
    }

    #[test]
    fn test_comparison_operators() {
        let source = "a <= b && c >= d || e != f";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Ident,
                TokenKind::LtEq,
                TokenKind::Ident,
                TokenKind::AmpAmp,
                TokenKind::Ident,
                TokenKind::GtEq,
                TokenKind::Ident,
                TokenKind::PipePipe,
                TokenKind::Ident,
                TokenKind::NotEq,
                TokenKind::Ident,
            ]
        );
    }

    #[test]
    fn test_method_call_syntax() {
        let source = "Point.distance(view p1, view p2)";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Ident, // Point
                TokenKind::Dot,
                TokenKind::Ident, // distance
                TokenKind::LParen,
                TokenKind::View,
                TokenKind::Ident, // p1
                TokenKind::Comma,
                TokenKind::View,
                TokenKind::Ident, // p2
                TokenKind::RParen,
            ]
        );
    }

    #[test]
    fn test_list_type() {
        let source = "list[int64]";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::List_,
                TokenKind::LBracket,
                TokenKind::Int64,
                TokenKind::RBracket
            ]
        );
    }

    #[test]
    fn test_modulo_keyword() {
        let source = "n modulo 2";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![TokenKind::Ident, TokenKind::Modulo, TokenKind::IntLiteral]
        );
    }

    #[test]
    fn test_namespace_declaration() {
        let source = "namespace myapp";
        let k = kinds_no_eof(source);
        assert_eq!(k, vec![TokenKind::Namespace, TokenKind::Ident]);
    }

    #[test]
    fn test_multiple_tokens_on_line() {
        let source = "int64 x = a + b * c";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Int64,
                TokenKind::Ident,
                TokenKind::Eq,
                TokenKind::Ident,
                TokenKind::Plus,
                TokenKind::Ident,
                TokenKind::Star,
                TokenKind::Ident,
            ]
        );
    }

    #[test]
    fn test_crlf_line_endings() {
        let source = "x\r\ny";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![TokenKind::Ident, TokenKind::Newline, TokenKind::Ident]
        );
    }

    #[test]
    fn test_single_ampersand_error() {
        let source = "a & b";
        let result = tokenize(source, file());
        assert!(!result.errors.is_empty());
        assert!(result.errors[0].message.contains("&"));
    }

    #[test]
    fn test_single_pipe_error() {
        let source = "a | b";
        let result = tokenize(source, file());
        assert!(!result.errors.is_empty());
        assert!(result.errors[0].message.contains("|"));
    }

    #[test]
    fn test_span_positions() {
        let source = "int64 x";
        let result = tokenize(source, file());
        // "int64" at 0..5
        assert_eq!(result.tokens[0].span.start, 0);
        assert_eq!(result.tokens[0].span.end, 5);
        // "x" at 6..7
        assert_eq!(result.tokens[1].span.start, 6);
        assert_eq!(result.tokens[1].span.end, 7);
    }

    #[test]
    fn test_comment_at_line_start_with_indent() {
        let source = "a:\n    b\n    # comment\n    c";
        let k = kinds_no_eof(source);
        // The comment line is skipped entirely
        assert_eq!(
            k,
            vec![
                TokenKind::Ident,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident, // b
                TokenKind::Newline,
                TokenKind::Ident, // c
                TokenKind::Dedent,
            ]
        );
    }

    #[test]
    fn test_for_loop() {
        let source = "for item in items:\n    process(item)";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::For,
                TokenKind::Ident, // item
                TokenKind::In,
                TokenKind::Ident, // items
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Ident, // process
                TokenKind::LParen,
                TokenKind::Ident, // item
                TokenKind::RParen,
                TokenKind::Dedent,
            ]
        );
    }

    #[test]
    fn test_trailing_whitespace_error() {
        // Line with trailing spaces before newline
        let source = "x   \ny";
        let result = tokenize(source, file());
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.message.contains("trailing whitespace")),
            "should report trailing whitespace error"
        );
    }

    #[test]
    fn test_mutable_variable() {
        let source = "mutable int64 counter = 0";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Mutable,
                TokenKind::Int64,
                TokenKind::Ident,
                TokenKind::Eq,
                TokenKind::IntLiteral,
            ]
        );
    }

    #[test]
    fn test_result_type() {
        let source = "result[string, string]";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Result,
                TokenKind::LBracket,
                TokenKind::String_,
                TokenKind::Comma,
                TokenKind::String_,
                TokenKind::RBracket,
            ]
        );
    }

    #[test]
    fn test_handle_error_block() {
        let source = "x handle error:\n    return fail(error)";
        let k = kinds_no_eof(source);
        assert_eq!(
            k,
            vec![
                TokenKind::Ident,
                TokenKind::Handle,
                TokenKind::Error,
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Return,
                TokenKind::Fail,
                TokenKind::LParen,
                TokenKind::Error,
                TokenKind::RParen,
                TokenKind::Dedent,
            ]
        );
    }

    #[test]
    fn test_hash_is_comment_not_token() {
        // # at the start of a line is a comment, not a Hash token
        let source = "# comment\nx";
        let k = kinds_no_eof(source);
        assert_eq!(k, vec![TokenKind::Ident]);
    }

    #[test]
    fn test_hash_after_space_is_comment() {
        // # after code on the same line is a comment
        let source = "x + y # this is a comment";
        let k = kinds_no_eof(source);
        assert_eq!(k, vec![TokenKind::Ident, TokenKind::Plus, TokenKind::Ident]);
    }
}
