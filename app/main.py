from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.db.database import (
    add_message,
    create_thread,
    delete_thread,
    get_setting,
    get_thread,
    init_db,
    list_messages,
    list_threads,
    set_setting,
    update_thread_title,
)
from app.services.llm import call_model
from app.services.model_config import (
    find_model,
    load_config,
    resolve_agent_model_id,
    save_config,
)
from app.services.orchestration import MultiAgentService

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_verifier")


def _merge_instructions(*parts: str | None) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _truncate(text: str, limit: int = 200) -> str:
    cleaned = " ".join(text.strip().splitlines())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _clean_title(title: str) -> str:
    cleaned = " ".join(title.strip().strip('"').splitlines()).strip()
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rsplit(" ", 1)[0] or cleaned[:80]
    return cleaned


async def _generate_thread_title(
    messages: list[dict[str, str]], primary_model: dict[str, Any]
) -> str:
    if not messages:
        return ""
    prompt = "Create a concise thread title (max 6 words). Return only the title."
    excerpt = "\n".join(
        f"{msg['role']}: {_truncate(msg['content'])}" for msg in messages[-6:]
    )
    title_messages = [{"role": "user", "content": f"{prompt}\n\nConversation:\n{excerpt}"}]
    title = await call_model(primary_model, title_messages, "")
    return _clean_title(title)


async def _maybe_update_thread_title(
    thread_id: str, primary_model: dict[str, Any]
) -> dict[str, Any] | None:
    title_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in list_messages(thread_id)
        if msg["role"] in {"user", "assistant"}
    ]
    if not title_messages:
        return None
    try:
        title = await _generate_thread_title(title_messages, primary_model)
    except (ValueError, httpx.HTTPError) as exc:
        logger.warning("Auto-title failed: %s", exc)
        return None
    if not title:
        return None
    current = get_thread(thread_id)
    if current and title == (current.get("title") or ""):
        return None
    return update_thread_title(thread_id, title)


class ThreadCreate(BaseModel):
    title: str | None = None


class ThreadUpdate(BaseModel):
    title: str


class MessageCreate(BaseModel):
    content: str
    images: list[str] | None = None
    verifier_enabled: bool | None = None


class SystemPromptUpdate(BaseModel):
    content: str


def _build_history(thread_id: str, include_images: bool = True) -> list[dict]:
    """Build LLM message history, converting images to multimodal content format."""
    history = []
    for msg in list_messages(thread_id):
        if msg["role"] not in {"user", "assistant"}:
            continue
        if include_images and msg.get("images"):
            content_parts: list[dict] = [{"type": "text", "text": msg["content"]}]
            for img in msg["images"]:
                content_parts.append({"type": "image_url", "image_url": {"url": img}})
            history.append({"role": msg["role"], "content": content_parts})
        else:
            history.append({"role": msg["role"], "content": msg["content"]})
    return history


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    load_config()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/threads")
def api_list_threads() -> list[dict[str, Any]]:
    return list_threads()


@app.post("/api/threads")
def api_create_thread(payload: ThreadCreate) -> dict[str, Any]:
    return create_thread(payload.title)


@app.put("/api/threads/{thread_id}")
def api_update_thread(thread_id: str, payload: ThreadUpdate) -> dict[str, Any]:
    thread = update_thread_title(thread_id, payload.title)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.delete("/api/threads/{thread_id}")
def api_delete_thread(thread_id: str) -> dict[str, Any]:
    if not delete_thread(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"ok": True}


@app.get("/api/threads/{thread_id}/messages")
def api_list_messages(thread_id: str) -> list[dict[str, Any]]:
    if not get_thread(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    return list_messages(thread_id)


@app.post("/api/threads/{thread_id}/messages")
async def api_add_message(thread_id: str, payload: MessageCreate) -> dict[str, Any]:
    if not get_thread(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    user_message = add_message(thread_id, "user", payload.content, images=payload.images or None)
    responder_history = _build_history(thread_id)
    text_history = _build_history(thread_id, include_images=False)
    system_prompt = get_setting("system_prompt", "")

    config = load_config()
    verifier_enabled = (
        payload.verifier_enabled
        if payload.verifier_enabled is not None
        else bool(config.get("verifier_enabled"))
    )
    service = MultiAgentService(config, system_prompt)

    try:
        orchestration = await service.run(
            thread_id=thread_id,
            user_message=payload.content,
            responder_history=responder_history,
            text_history=text_history,
            verifier_enabled=verifier_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="LLM request failed") from exc

    responder_model_id = resolve_agent_model_id(config, "responder")
    verifier_model_id = resolve_agent_model_id(config, "verifier")
    polisher_model_id = resolve_agent_model_id(config, "polisher")
    responder_message = add_message(
        thread_id,
        "assistant_draft",
        orchestration.responder.content,
        model=responder_model_id,
    )
    verifier_message = None
    if orchestration.verifier is not None:
        verifier_message = add_message(
            thread_id,
            "verifier",
            orchestration.verifier.content,
            model=verifier_model_id,
        )
    assistant_message = add_message(
        thread_id,
        "assistant",
        orchestration.polisher.content,
        model=polisher_model_id,
    )
    title_model = find_model(config, polisher_model_id)
    updated_thread = await _maybe_update_thread_title(thread_id, title_model)
    response: dict[str, Any] = {
        "messages": [user_message, responder_message, assistant_message],
        "routing": {
            "path": orchestration.decision.path,
            "reason": orchestration.decision.reason,
            "skipped_verifier": orchestration.decision.skipped_verifier,
            "confidence_threshold": orchestration.decision.confidence_threshold,
            "responder_confidence": orchestration.decision.responder_confidence,
            "is_simple_task": orchestration.decision.is_simple_task,
        },
    }
    if verifier_message:
        response["messages"].insert(2, verifier_message)
    if updated_thread:
        response["thread"] = updated_thread
    return response


@app.post("/api/threads/{thread_id}/messages/stream")
async def api_stream_message(
    thread_id: str, request: Request, payload: MessageCreate
) -> StreamingResponse:
    if not get_thread(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    user_message = add_message(thread_id, "user", payload.content, images=payload.images or None)
    responder_history = _build_history(thread_id)
    text_history = _build_history(thread_id, include_images=False)
    system_prompt = get_setting("system_prompt", "")

    config = load_config()
    verifier_enabled = (
        payload.verifier_enabled
        if payload.verifier_enabled is not None
        else bool(config.get("verifier_enabled"))
    )
    service = MultiAgentService(config, system_prompt)
    responder_model_id = resolve_agent_model_id(config, "responder")
    verifier_model_id = resolve_agent_model_id(config, "verifier")
    polisher_model_id = resolve_agent_model_id(config, "polisher")

    async def event_stream() -> Any:
        agent_outputs: dict[str, dict[str, Any]] = {}
        try:
            async for event in service.stream(
                thread_id=thread_id,
                user_message=payload.content,
                responder_history=responder_history,
                text_history=text_history,
                verifier_enabled=verifier_enabled,
            ):
                if await request.is_disconnected():
                    return
                event_type = event.get("event", "message")
                payload_data = {k: v for k, v in event.items() if k != "event"}
                if event_type == "agent_output":
                    stage = payload_data.get("stage")
                    if isinstance(stage, str):
                        agent_outputs[stage] = payload_data
                    continue
                yield f"event: {event_type}\ndata: {json.dumps(payload_data)}\n\n"
        except ValueError as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
            return
        except httpx.HTTPError:
            yield f"event: error\ndata: {json.dumps({'message': 'LLM request failed'})}\n\n"
            return

        responder_output = agent_outputs.get("responder")
        polisher_output = agent_outputs.get("polisher")
        verifier_output = agent_outputs.get("verifier")
        if responder_output:
            responder_message = add_message(
                thread_id,
                "assistant_draft",
                responder_output.get("content", ""),
                model=responder_output.get("model") or responder_model_id,
            )
            yield f"event: saved\ndata: {json.dumps({'message': responder_message})}\n\n"
        if verifier_output:
            verifier_message = add_message(
                thread_id,
                "verifier",
                verifier_output.get("content", ""),
                model=verifier_output.get("model") or verifier_model_id,
            )
            yield f"event: saved\ndata: {json.dumps({'message': verifier_message})}\n\n"
        if not polisher_output:
            yield f"event: error\ndata: {json.dumps({'message': 'Polisher output missing'})}\n\n"
            return
        assistant_message = add_message(
            thread_id,
            "assistant",
            polisher_output.get("content", ""),
            model=polisher_output.get("model") or polisher_model_id,
        )
        yield f"event: saved\ndata: {json.dumps({'message': assistant_message})}\n\n"
        title_model = find_model(config, polisher_model_id)
        updated_thread = await _maybe_update_thread_title(thread_id, title_model)
        if updated_thread:
            yield f"event: title\ndata: {json.dumps({'thread': updated_thread})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/system-prompt")
def api_get_system_prompt() -> dict[str, Any]:
    return {"content": get_setting("system_prompt", "")}


@app.put("/api/system-prompt")
def api_set_system_prompt(payload: SystemPromptUpdate) -> dict[str, Any]:
    set_setting("system_prompt", payload.content)
    return {"content": payload.content}


@app.get("/api/config")
def api_get_config() -> dict[str, Any]:
    return load_config()


@app.put("/api/config")
def api_set_config(config: dict[str, Any]) -> dict[str, Any]:
    save_config(config)
    return config

