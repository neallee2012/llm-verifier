# Plan

Problem: Build a local FastAPI webapp with system prompt editing, chat thread/history management, SQLite persistence, multi‑model verification (primary + verifier), and configurable model settings.

Approach: Create a small FastAPI backend with SQLite tables for threads/messages/settings, store model config in a local JSON file, implement Azure OpenAI and Gemini clients with a primary+verifier flow, and serve a simple HTML/JS UI for threads, chat, system prompt, and config.

## Workplan
- [x] Initialize project structure and dependencies.
- [x] Implement SQLite schema and data access (threads, messages, settings).
- [x] Implement config load/save + LLM client calls (OpenAI, Azure OpenAI, Gemini, verification flow).
- [x] Build FastAPI API routes and static file serving.
- [x] Build frontend UI (threads list, chat history, system prompt editor, config form).
- [x] Add streaming chat endpoint and UI status updates.
- [x] Auto-generate thread titles from conversation topics.
- [x] Run a basic smoke check and document run steps.

Notes: Model configs are stored in a local JSON file as requested (plain text, local only). Verifier model is run after primary and both outputs are saved to chat history. Streaming uses SSE and updates the UI with model status. Threads are auto-titled by the primary model.
