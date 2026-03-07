//! Native Gemini REST provider implementation.
//!
//! This provider bypasses rig-core to support Gemini 3's required
//! `thought_signature` in tool calls and results.

use async_trait::async_trait;
use reqwest::Client;
use rust_decimal::Decimal;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};

use crate::config::GeminiConfig;
use crate::error::LlmError;
use crate::llm::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, Role, ToolCall,
    ToolCompletionRequest, ToolCompletionResponse,
};

/// Gemini REST API request.
#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
enum GeminiPart {
    Text { text: String },
    FunctionCall {
        name: String,
        args: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        name: String,
        response: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

/// Gemini REST API response.
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContentResponse,
    #[serde(rename = "finishReason")]
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct GeminiContentResponse {
    role: String,
    parts: Vec<GeminiPartResponse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPartResponse {
    text: Option<String>,
    function_call: Option<GeminiFunctionCallResponse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFunctionCallResponse {
    name: String,
    args: serde_json::Value,
    thought_signature: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
}

/// Native Gemini REST provider.
pub struct GeminiRestProvider {
    client: Client,
    config: GeminiConfig,
}

impl GeminiRestProvider {
    /// Create a new Gemini REST provider.
    pub fn new(config: GeminiConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        Self { client, config }
    }

    fn map_role(role: Role) -> String {
        match role {
            Role::User => "user".to_string(),
            Role::Assistant => "model".to_string(),
            Role::System => "user".to_string(), // Gemini uses system instruction or user prefix
            Role::Tool => "function".to_string(),
        }
    }

    fn convert_messages(&self, messages: &[ChatMessage]) -> Vec<GeminiContent> {
        messages
            .iter()
            .map(|m| {
                let role = Self::map_role(m.role);
                let mut parts = Vec::new();

                if !m.content.is_empty() {
                    parts.push(GeminiPart::Text {
                        text: m.content.clone(),
                    });
                }

                if let Some(ref tool_calls) = m.tool_calls {
                    for tc in tool_calls {
                        parts.push(GeminiPart::FunctionCall {
                            name: tc.name.clone(),
                            args: tc.arguments.clone(),
                            thought_signature: tc.thought_signature.clone(),
                        });
                    }
                }

                if m.role == Role::Tool {
                    if let Some(ref name) = m.name {
                        // content is already a JSON string in Thread::messages()
                        let response = serde_json::from_str(&m.content)
                            .unwrap_or_else(|_| serde_json::json!({ "result": m.content }));
                        parts.push(GeminiPart::FunctionResponse {
                            name: name.clone(),
                            response,
                        });
                    }
                }

                GeminiContent { role, parts }
            })
            .collect()
    }
}

#[async_trait]
impl LlmProvider for GeminiRestProvider {
    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
        // Dummy pricing or fetch from config
        (Decimal::ZERO, Decimal::ZERO)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.config.model,
            self.config.api_key.expose_secret()
        );

        let g_request = GeminiRequest {
            contents: self.convert_messages(&request.messages),
            tools: None,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: request.max_tokens,
                temperature: request.temperature,
                stop_sequences: request.stop_sequences,
            }),
        };

        let response = self
            .client
            .post(&url)
            .json(&g_request)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "gemini".to_string(),
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::RequestFailed {
                provider: "gemini".to_string(),
                reason: format!("HTTP {}: {}", status, body),
            });
        }

        let g_response: GeminiResponse = response.json().await.map_err(|e| LlmError::RequestFailed {
            provider: "gemini".to_string(),
            reason: format!("Failed to parse JSON: {}", e),
        })?;

        let candidate = g_response
            .candidates
            .get(0)
            .ok_or_else(|| LlmError::RequestFailed {
                provider: "gemini".to_string(),
                reason: "No candidates in response".to_string(),
            })?;

        let content = candidate
            .content
            .parts
            .iter()
            .filter_map(|p| p.text.clone())
            .collect::<Vec<_>>()
            .join("\n");

        let (input_tokens, output_tokens) = g_response
            .usage_metadata
            .map(|u| (u.prompt_token_count, u.candidates_token_count))
            .unwrap_or((0, 0));

        Ok(CompletionResponse {
            content,
            input_tokens,
            output_tokens,
            finish_reason: match candidate.finish_reason.as_str() {
                "STOP" => FinishReason::Stop,
                "MAX_TOKENS" => FinishReason::MaxTokens,
                _ => FinishReason::Other,
            },
        })
    }

    async fn complete_with_tools(
        &self,
        request: ToolCompletionRequest,
    ) -> Result<ToolCompletionResponse, LlmError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.config.model,
            self.config.api_key.expose_secret()
        );

        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(vec![GeminiTool {
                function_declarations: request
                    .tools
                    .iter()
                    .map(|t| GeminiFunctionDeclaration {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    })
                    .collect(),
            }])
        };

        let g_request = GeminiRequest {
            contents: self.convert_messages(&request.messages),
            tools,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: request.max_tokens,
                temperature: request.temperature,
                stop_sequences: None,
            }),
        };

        let response = self
            .client
            .post(&url)
            .json(&g_request)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "gemini".to_string(),
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::RequestFailed {
                provider: "gemini".to_string(),
                reason: format!("HTTP {}: {}", status, body),
            });
        }

        let g_response: GeminiResponse = response.json().await.map_err(|e| LlmError::RequestFailed {
            provider: "gemini".to_string(),
            reason: format!("Failed to parse JSON: {}", e),
        })?;

        let candidate = g_response
            .candidates
            .get(0)
            .ok_or_else(|| LlmError::RequestFailed {
                provider: "gemini".to_string(),
                reason: "No candidates in response".to_string(),
            })?;

        let mut content = None;
        let mut tool_calls = Vec::new();

        for part in &candidate.content.parts {
            if let Some(ref text) = part.text {
                if !text.is_empty() {
                    content = Some(text.clone());
                }
            }
            if let Some(ref fc) = part.function_call {
                tool_calls.push(ToolCall {
                    id: uuid::Uuid::new_v4().to_string(), // Gemini doesn't always provide call IDs, but OpenAI protocol needs them
                    name: fc.name.clone(),
                    arguments: fc.args.clone(),
                    thought_signature: fc.thought_signature.clone(),
                });
            }
        }

        let (input_tokens, output_tokens) = g_response
            .usage_metadata
            .map(|u| (u.prompt_token_count, u.candidates_token_count))
            .unwrap_or((0, 0));

        Ok(ToolCompletionResponse {
            content,
            tool_calls,
            input_tokens,
            output_tokens,
            finish_reason: match candidate.finish_reason.as_str() {
                "STOP" => FinishReason::Stop,
                "MAX_TOKENS" => FinishReason::MaxTokens,
                "SAFETY" => FinishReason::ContentFilter,
                _ => FinishReason::Other,
            },
        })
    }
}
