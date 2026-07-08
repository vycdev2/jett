use std::collections::HashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

/// The Jett LSP backend.
pub struct JettBackend {
    client: Client,
    /// In-memory document store: URI → source text.
    documents: tokio::sync::RwLock<HashMap<Url, String>>,
}

impl JettBackend {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Run the Jett compiler pipeline on the given source text and publish
    /// diagnostics back to the client.
    async fn validate(&self, uri: Url, text: &str) {
        let file_path = uri
            .to_file_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| uri.to_string());

        let result = jett_driver::build_source(text, &file_path);

        let diagnostics: Vec<Diagnostic> = result
            .diagnostics
            .iter()
            .map(|d| {
                let (start_line, start_col) =
                    jett_diagnostics::render::line_col(&result.source, d.span.start);
                let (end_line, end_col) =
                    jett_diagnostics::render::line_col(&result.source, d.span.end);

                let severity = match d.severity {
                    jett_diagnostics::Severity::Error => Some(DiagnosticSeverity::ERROR),
                    jett_diagnostics::Severity::Warning => Some(DiagnosticSeverity::WARNING),
                    jett_diagnostics::Severity::Info => Some(DiagnosticSeverity::INFORMATION),
                };

                // LSP positions are 0-based; line_col returns 1-based.
                let range = Range::new(
                    Position::new(start_line as u32 - 1, start_col as u32 - 1),
                    Position::new(end_line as u32 - 1, end_col as u32 - 1),
                );

                Diagnostic {
                    range,
                    severity,
                    code: Some(NumberOrString::String(d.code.to_string())),
                    source: Some("jett".to_string()),
                    message: d.message.clone(),
                    ..Diagnostic::default()
                }
            })
            .collect();

        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for JettBackend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                completion_provider: Some(CompletionOptions::default()),
                ..ServerCapabilities::default()
            },
            ..InitializeResult::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Jett language server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let text = params.text_document.text.clone();
        self.documents
            .write()
            .await
            .insert(uri.clone(), text.clone());
        self.validate(uri, &text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        // We requested FULL sync, so the last content change is the full text.
        if let Some(change) = params.content_changes.into_iter().last() {
            let uri = params.text_document.uri.clone();
            let text = change.text.clone();
            self.documents
                .write()
                .await
                .insert(uri.clone(), text.clone());
            self.validate(uri, &text).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.documents
            .write()
            .await
            .remove(&params.text_document.uri);
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let docs = self.documents.read().await;
        let Some(source) = docs.get(uri) else {
            return Ok(None);
        };

        // LSP positions are 0-based; hover_type expects 1-based.
        let line = position.line + 1;
        let col = position.character + 1;

        let type_info = jett_driver::hover_type(source, line, col);

        let Some(type_str) = type_info else {
            return Ok(None);
        };

        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::PlainText,
                value: type_str,
            }),
            range: None,
        }))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let docs = self.documents.read().await;
        let Some(source) = docs.get(uri) else {
            return Ok(None);
        };

        let line = position.line + 1;
        let col = position.character + 1;

        let Some((start, end)) = jett_driver::goto_definition(source, line, col) else {
            return Ok(None);
        };

        let (start_line, start_col) = jett_diagnostics::render::line_col(source, start);
        let (end_line, end_col) = jett_diagnostics::render::line_col(source, end);

        let range = Range::new(
            Position::new(start_line as u32 - 1, start_col as u32 - 1),
            Position::new(end_line as u32 - 1, end_col as u32 - 1),
        );

        Ok(Some(GotoDefinitionResponse::Scalar(Location {
            uri: uri.clone(),
            range,
        })))
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;

        let docs = self.documents.read().await;
        let Some(source) = docs.get(uri) else {
            return Ok(None);
        };

        let position = params.text_document_position.position;
        let candidates =
            jett_driver::completions_at(source, position.line + 1, position.character + 1);
        if candidates.is_empty() {
            return Ok(None);
        }

        use jett_resolve::scope::DefKind;
        let items: Vec<CompletionItem> = candidates
            .into_iter()
            .map(|(name, kind)| {
                let kind = match kind {
                    DefKind::Function => CompletionItemKind::FUNCTION,
                    DefKind::Struct => CompletionItemKind::STRUCT,
                    DefKind::Enum => CompletionItemKind::ENUM,
                    DefKind::Interface => CompletionItemKind::INTERFACE,
                    DefKind::Machine => CompletionItemKind::CLASS,
                    DefKind::Actor => CompletionItemKind::CLASS,
                    DefKind::Variable | DefKind::Param => CompletionItemKind::VARIABLE,
                    DefKind::Type => CompletionItemKind::TYPE_PARAMETER,
                    DefKind::Constant => CompletionItemKind::CONSTANT,
                    DefKind::Namespace => CompletionItemKind::MODULE,
                    DefKind::Bitfield => CompletionItemKind::STRUCT,
                };
                CompletionItem {
                    label: name,
                    kind: Some(kind),
                    ..CompletionItem::default()
                }
            })
            .collect();

        Ok(Some(CompletionResponse::Array(items)))
    }
}

/// Start the LSP server on stdin/stdout. This is the main entry point called
/// by `jett lsp`.
pub async fn run_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = tower_lsp::LspService::new(JettBackend::new);
    tower_lsp::Server::new(stdin, stdout, socket)
        .serve(service)
        .await;
}

#[cfg(test)]
mod tests {
    /// Verify that `build_source` produces diagnostics for invalid Jett code.
    /// This exercises the same path the LSP uses to validate documents.
    #[test]
    fn build_source_returns_diagnostics_for_bad_code() {
        let source = "this is not valid jett code !!!";
        let result = jett_driver::build_source(source, "test.jett");
        assert!(
            result.has_errors,
            "expected errors for invalid source, got none"
        );
        assert!(
            !result.diagnostics.is_empty(),
            "expected at least one diagnostic"
        );
    }

    /// Verify that valid (empty) source produces no errors.
    #[test]
    fn build_source_empty_is_ok() {
        let result = jett_driver::build_source("", "empty.jett");
        assert!(
            !result.has_errors,
            "expected no errors for empty source, got: {:?}",
            result
                .diagnostics
                .iter()
                .map(|d| &d.message)
                .collect::<Vec<_>>()
        );
    }

    /// Verify that hover_type returns a type for a known expression.
    #[test]
    fn hover_type_returns_type_for_identifier() {
        let source = "namespace test\n\nfunction main() returns nothing:\n    int64 x = 42\n    return nothing\n";
        // Line 4, col 5 = start of "int64 x" — the literal 42 is on the same line
        // col 15 = the '4' in '42'
        let ty = jett_driver::hover_type(source, 4, 15);
        assert_eq!(ty, Some("int64".to_string()), "expected int64 hover type");
    }
}
