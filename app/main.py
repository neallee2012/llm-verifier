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
from app.services.llm import call_model, stream_model
from app.services.model_config import find_model, load_config, save_config

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
    history = _build_history(thread_id)
    system_prompt = get_setting("system_prompt", "")

    config = load_config()
    verifier_enabled = (
        payload.verifier_enabled
        if payload.verifier_enabled is not None
        else bool(config.get("verifier_enabled"))
    )
    primary_model = find_model(config, config.get("primary_model_id"))
    verifier_model = None
    if verifier_enabled:
        verifier_model = find_model(config, config.get("verifier_model_id"))
    primary_instructions = _merge_instructions(
        system_prompt, primary_model.get("instructions")
    )

    try:
        primary_response = await call_model(
            primary_model, history, primary_instructions
        )
        verifier_response = None
        if verifier_enabled:
            verifier_prompt = (
                "You are an industry expert and consultant. Review the primary assistant "
                "response for accuracy, completeness, and relevance. Use the primary response "
                "as input, provide an executive summary, then start to correct any issues, and provide the final answer to the user. "
            )
            verifier_prompt = _merge_instructions(
                verifier_prompt, system_prompt, verifier_model.get("instructions")
            )
            verifier_messages = [
                *_build_history(thread_id, include_images=False),
                {"role": "assistant", "content": primary_response},
            ]
            verifier_response = await call_model(
                verifier_model, verifier_messages, verifier_prompt
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="LLM request failed") from exc

    assistant_message = add_message(
        thread_id, "assistant", primary_response, model=primary_model["id"]
    )
    verifier_message = None
    if verifier_enabled:
        verifier_message = add_message(
            thread_id, "verifier", verifier_response, model=verifier_model["id"]
        )
    updated_thread = await _maybe_update_thread_title(thread_id, primary_model)
    response: dict[str, Any] = {"messages": [user_message, assistant_message]}
    if verifier_message:
        response["messages"].append(verifier_message)
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
    history = _build_history(thread_id)
    system_prompt = get_setting("system_prompt", "")

    config = load_config()
    verifier_enabled = (
        payload.verifier_enabled
        if payload.verifier_enabled is not None
        else bool(config.get("verifier_enabled"))
    )
    primary_model = find_model(config, config.get("primary_model_id"))
    verifier_model = None
    if verifier_enabled:
        verifier_model = find_model(config, config.get("verifier_model_id"))
    primary_instructions = _merge_instructions(
        system_prompt, primary_model.get("instructions")
    )

    async def event_stream() -> Any:
        yield f"event: status\ndata: {json.dumps({'stage': 'primary', 'status': 'started'})}\n\n"
        primary_text = ""
        try:
            streaming_text = ""

            def _capture_primary_final(text: str) -> None:
                nonlocal primary_text
                primary_text = text

            async for chunk in stream_model(
                primary_model,
                history,
                primary_instructions,
                on_final_text=_capture_primary_final,
            ):
                streaming_text += chunk
                yield f"event: token\ndata: {json.dumps({'stage': 'primary', 'delta': chunk})}\n\n"
            if streaming_text and len(streaming_text) > len(primary_text):
                primary_text = streaming_text
            yield f"event: status\ndata: {json.dumps({'stage': 'primary', 'status': 'done'})}\n\n"
        except ValueError as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
            return
        except httpx.HTTPError:
            yield f"event: error\ndata: {json.dumps({'message': 'LLM request failed'})}\n\n"
            return

        assistant_message = add_message(
            thread_id, "assistant", primary_text, model=primary_model["id"]
        )
        yield f"event: saved\ndata: {json.dumps({'message': assistant_message})}\n\n"

        verifier_message = None
        if verifier_enabled:
            verifier_prompt = (
                "You are an industry expert and consultant. Review the primary assistant "
                "response for accuracy, completeness, and relevance. Use the primary response "
                "as input, illustrate your judgment first and list issues you find, then correct any issues, and provide the final answer to the user. "
                "Respond with the final answer only."
            )
            verifier_prompt = _merge_instructions(
                verifier_prompt, system_prompt, verifier_model.get("instructions")
            )
            verifier_messages = [
                *_build_history(thread_id, include_images=False),
                {"role": "assistant", "content": primary_text},
            ]

            yield f"event: status\ndata: {json.dumps({'stage': 'verifier', 'status': 'started'})}\n\n"
            verifier_text = ""
            try:
                async for chunk in stream_model(
                    verifier_model, verifier_messages, verifier_prompt
                ):
                    verifier_text += chunk
                    yield f"event: token\ndata: {json.dumps({'stage': 'verifier', 'delta': chunk})}\n\n"
                yield f"event: status\ndata: {json.dumps({'stage': 'verifier', 'status': 'done'})}\n\n"
            except ValueError as exc:
                yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
                return
            except httpx.HTTPError as exc:
                logger.error("Verifier LLM request failed: %s", exc)
                yield f"event: error\ndata: {json.dumps({'message': 'Verifier LLM request failed'})}\n\n"
                return

            verifier_message = add_message(
                thread_id, "verifier", verifier_text, model=verifier_model["id"]
            )
            yield f"event: saved\ndata: {json.dumps({'message': verifier_message})}\n\n"
        updated_thread = await _maybe_update_thread_title(thread_id, primary_model)
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

