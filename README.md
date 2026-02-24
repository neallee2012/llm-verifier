# LLM Verifier

Local FastAPI webapp for managing chat threads, system prompts, and multi-model verification (primary + verifier) with streaming.

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
- Threads auto-title using the primary model after each exchange.

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
