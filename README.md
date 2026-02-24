# LLM Verifier

Local FastAPI webapp for managing chat threads, system prompts, and multi-agent verification (master + responder/verifier/polisher) with streaming.

## Run

```powershell
cd C:\Users\kunle\Projects\llm-verifier-app
.\start-server.bat
```

Or:

```powershell
python -m uvicorn app.main:app --reload
```

To auto-restart the server if it crashes:

```powershell
.\watch-server.bat
```

Open http://127.0.0.1:8000

Logs are written to `logs\server.log` and also printed to the console when using the batch scripts.

## Notes

- Model config stored at `config/models.json` (plain text, local only).
- Verifier is disabled by default; toggle **Verifier** to switch per message (Save Config persists the default).
- Chat history stored in `data/chat.db`.
- Streaming chat uses `/api/threads/{thread_id}/messages/stream` (SSE).
- Supported models: Azure OpenAI and Azure Grok (Entra ID), plus Gemini (API key).
- Threads auto-title using the final-response model after each exchange.
- Master agent uses dynamic routing: if a task is simple and responder confidence is above the configured threshold, verifier can be skipped and output goes directly to polisher.

## Multi-agent architecture

- **Master routing agent** decides execution path for each request.
- **Responder agent** creates the first draft.
- **Verifier agent** validates/corrects when routing requires verification.
- **Polisher agent** rewrites the final user-facing response.

Default path is:

`responder -> verifier -> polisher`

Shortcut path (when enabled and confidence rule is met):

`responder -> polisher`

## Streaming behavior

SSE now emits agent-based stages and routing metadata:

- `status` / `token` events for `responder`, `verifier`, `polisher`
- `routing` event with selected path and skip reason
- `saved` / `title` / `done` events as before

## Agent tools status

- Microsoft Agent Framework dependency is included (`agent-framework`, preview).
- Tool abstractions are added (`ToolRegistry`, `WebSearchTool`, tool traces) for per-agent tool usage.
- `tools.web_search_enabled` is persisted in config and ready for runtime wiring to a concrete search backend.

## Azure OpenAI (Entra ID)

Sign in with Azure CLI:

```powershell
az login
```

Example `config/models.json`:

```json
{
  "primary_model_id": "azure-openai",
  "verifier_model_id": "gemini",
  "verifier_enabled": false,
  "agents": {
    "responder_model_id": "azure-openai",
    "verifier_model_id": "gemini",
    "polisher_model_id": "azure-openai"
  },
  "routing": {
    "confidence_threshold": 0.95,
    "enable_verifier_shortcut": true
  },
  "tools": {
    "web_search_enabled": true
  },
  "models": [
    {
      "id": "azure-openai",
      "type": "azure-openai",
      "label": "Azure OpenAI",
      "endpoint": "https://<resource-name>.openai.azure.com",
      "deployment": "<deployment-name>",
      "api_version": "2025-04-01-preview",
      "api_type": "chat-completions",
      "instructions": ""
    },
    {
      "id": "azure-grok",
      "type": "azure-openai",
      "label": "Azure Grok",
      "endpoint": "https://<resource-name>.openai.azure.com",
      "deployment": "<deployment-name>",
      "api_version": "2025-04-01-preview",
      "api_type": "chat-completions",
      "instructions": ""
    },
    {
      "id": "gemini",
      "type": "gemini",
      "label": "Gemini",
      "api_key": "<gemini-api-key>",
      "model": "gemini-3-pro-preview",
      "instructions": ""
    }
  ]
}
```
