from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

from azure.identity import AzureCliCredential

import httpx

logger = logging.getLogger("llm_verifier")


def _format_messages(messages: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
    if system_prompt:
        return [{"role": "system", "content": system_prompt}, *messages]
    return messages


def _parse_azure_endpoint(endpoint: str) -> tuple[str, str | None, bool]:
    parts = urlsplit(endpoint)
    if parts.scheme and parts.netloc:
        base = f"{parts.scheme}://{parts.netloc}"
        query_version = parse_qs(parts.query).get("api-version", [None])[0]
        uses_responses = "responses" in parts.path.lower()
        return base, query_version, uses_responses
    return endpoint.rstrip("/"), None, False


def _extract_response_text(data: dict[str, Any]) -> str:
    if "choices" in data:
        choices = data["choices"]
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                text = choice.get("text")
                if isinstance(text, str):
                    return text
    if "output_text" in data:
        return data["output_text"]
    if "output" in data:
        for item in data["output"]:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if "text" in part:
                            return part["text"]
                        if "output_text" in part:
                            return part["output_text"]
    raise ValueError("Unsupported response format from Azure OpenAI")


def _extract_text_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts: list[str] = []
        text = value.get("text")
        if isinstance(text, str):
            parts.append(text)
        output_text = value.get("output_text")
        if isinstance(output_text, str):
            parts.append(output_text)
        content = value.get("content")
        if content is not None:
            parts.extend(_extract_text_parts(content))
        return parts
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_text_parts(item))
        return parts
    return []


def _extract_stream_deltas(data: dict[str, Any]) -> list[str]:
    event_type = data.get("type")
    if isinstance(event_type, str) and "delta" not in event_type:
        return []

    deltas: list[str] = []
    if "choices" in data and isinstance(data["choices"], list):
        for choice in data["choices"]:
            if not isinstance(choice, dict):
                continue
            deltas.extend(_extract_text_parts(choice.get("delta")))
            deltas.extend(_extract_text_parts(choice.get("text")))
        return deltas
    deltas.extend(_extract_text_parts(data.get("delta")))
    deltas.extend(_extract_text_parts(data.get("output_text")))
    deltas.extend(_extract_text_parts(data.get("text")))
    deltas.extend(_extract_text_parts(data.get("output")))
    return deltas


def _extract_stream_fallback_text(data: dict[str, Any]) -> str | None:
    for candidate in (data.get("response"), data):
        if isinstance(candidate, dict):
            try:
                return _extract_response_text(candidate)
            except (ValueError, IndexError):
                continue
    return None


def _format_gemini_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content")
        if isinstance(content, str):
            contents.append({"role": role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append({"text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        mime, _, b64 = url.partition(";base64,")
                        mime = mime.replace("data:", "")
                        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            if parts:
                contents.append({"role": role, "parts": parts})
    return contents


def _extract_gemini_text(data: dict[str, Any], allow_empty: bool = False) -> str:
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        if isinstance(candidate, dict):
            content = candidate.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    text = "".join(
                        part.get("text", "")
                        for part in parts
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    )
                    if text:
                        return text
            text = candidate.get("text")
            if isinstance(text, str):
                return text
    text = data.get("text")
    if isinstance(text, str):
        return text
    if allow_empty:
        return ""
    raise ValueError("Unsupported response format from Gemini")


def _log_http_error(service: str, exc: httpx.HTTPError) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        request = exc.request
        response = exc.response
        try:
            body = response.text
        except httpx.ResponseNotRead:
            body = "(streaming response not read)"
        logger.error(
            "%s request failed: %s %s -> %s\n%s",
            service,
            request.method,
            request.url,
            response.status_code,
            body,
        )
        return
    request = getattr(exc, "request", None)
    if request is not None:
        logger.error(
            "%s request error: %s %s -> %s",
            service,
            request.method,
            request.url,
            exc,
        )
    else:
        logger.error("%s request error: %s", service, exc)


async def _call_azure_openai(
    model: dict[str, Any], messages: list[dict[str, str]], system_prompt: str
) -> str:
    endpoint = model.get("endpoint")
    auth_method = model.get("auth_method") or "entra_id"
    deployment = model.get("deployment")
    api_version = model.get("api_version")
    if not endpoint or not deployment:
        raise ValueError("Azure OpenAI endpoint and deployment are required")
    if auth_method != "entra_id":
        raise ValueError("Azure models require Entra ID authentication")

    base, api_version_from_endpoint, uses_responses = _parse_azure_endpoint(endpoint)
    api_version = api_version or api_version_from_endpoint
    if not api_version:
        raise ValueError("Azure OpenAI api_version is required")

    api_type = model.get("api_type") or ("responses" if uses_responses else "chat-completions")
    if api_type == "responses":
        url = f"{base}/openai/responses"
        payload = {"model": deployment, "input": messages, "temperature": 1}
        if system_prompt:
            payload["instructions"] = system_prompt
    else:
        url = f"{base}/openai/deployments/{deployment}/chat/completions"
        payload = {
            "messages": _format_messages(messages, system_prompt),
            "temperature": 1,
        }
    token = AzureCliCredential().get_token(
        "https://cognitiveservices.azure.com/.default"
    )
    headers = {"Authorization": f"Bearer {token.token}"}
    params = {"api-version": api_version}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                url, headers=headers, params=params, json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _log_http_error("Azure OpenAI", exc)
            raise
        data = response.json()
    return _extract_response_text(data)


async def _call_gemini(
    model: dict[str, Any], messages: list[dict[str, str]], system_prompt: str
) -> str:
    api_key = model.get("api_key")
    model_name = model.get("model")
    if not api_key or not model_name:
        raise ValueError("Gemini api_key and model are required")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload: dict[str, Any] = {"contents": _format_gemini_contents(messages)}
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                url, headers=headers, params=params, json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _log_http_error("Gemini", exc)
            raise
        data = response.json()
    return _extract_gemini_text(data)


async def _stream_azure_openai(
    model: dict[str, Any],
    messages: list[dict[str, str]],
    system_prompt: str,
    on_final_text: Callable[[str], None] | None = None,
) -> AsyncIterator[str]:
    endpoint = model.get("endpoint")
    auth_method = model.get("auth_method") or "entra_id"
    deployment = model.get("deployment")
    api_version = model.get("api_version")
    if not endpoint or not deployment:
        raise ValueError("Azure OpenAI endpoint and deployment are required")
    if auth_method != "entra_id":
        raise ValueError("Azure models require Entra ID authentication")

    base, api_version_from_endpoint, uses_responses = _parse_azure_endpoint(endpoint)
    api_version = api_version or api_version_from_endpoint
    if not api_version:
        raise ValueError("Azure OpenAI api_version is required")

    api_type = model.get("api_type") or ("responses" if uses_responses else "chat-completions")
    if api_type == "responses":
        url = f"{base}/openai/responses"
        payload = {
            "model": deployment,
            "input": messages,
            "temperature": 1,
            "stream": True,
        }
        if system_prompt:
            payload["instructions"] = system_prompt
    else:
        url = f"{base}/openai/deployments/{deployment}/chat/completions"
        payload = {
            "messages": _format_messages(messages, system_prompt),
            "temperature": 1,
            "stream": True,
        }
    token = AzureCliCredential().get_token(
        "https://cognitiveservices.azure.com/.default"
    )
    headers = {"Authorization": f"Bearer {token.token}"}
    params = {"api-version": api_version}
    saw_delta = False
    final_text: str | None = None

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            async with client.stream(
                "POST", url, headers=headers, params=params, json=payload
            ) as response:
                if response.status_code >= 400:
                    err_body = await response.aread()
                    logger.error(
                        "Azure OpenAI stream error: POST %s -> %s\n%s",
                        url, response.status_code, err_body.decode(errors="replace"),
                    )
                response.raise_for_status()
                buffer: list[str] = []
                async for line in response.aiter_lines():
                    if not line:
                        if not buffer:
                            continue
                        raw = "\n".join(buffer).strip()
                        buffer.clear()
                        if not raw:
                            continue
                        if raw == "[DONE]":
                            break
                        data = json.loads(raw)
                        deltas = _extract_stream_deltas(data)
                        if deltas:
                            saw_delta = True
                            for delta in deltas:
                                if delta:
                                    yield delta
                        candidate = _extract_stream_fallback_text(data)
                        if candidate and (final_text is None or len(candidate) > len(final_text)):
                            final_text = candidate
                        continue
                    if line.startswith("data:"):
                        buffer.append(line[5:].strip())
                if buffer:
                    raw = "\n".join(buffer).strip()
                    if raw and raw != "[DONE]":
                        data = json.loads(raw)
                        deltas = _extract_stream_deltas(data)
                        if deltas:
                            saw_delta = True
                            for delta in deltas:
                                if delta:
                                    yield delta
                        candidate = _extract_stream_fallback_text(data)
                        if candidate and (final_text is None or len(candidate) > len(final_text)):
                            final_text = candidate
                if not saw_delta and final_text:
                    yield final_text
                if final_text and on_final_text:
                    on_final_text(final_text)
        except httpx.HTTPError as exc:
            _log_http_error("Azure OpenAI", exc)
            raise


async def _stream_gemini(
    model: dict[str, Any], messages: list[dict[str, str]], system_prompt: str
) -> AsyncIterator[str]:
    api_key = model.get("api_key")
    model_name = model.get("model")
    if not api_key or not model_name:
        raise ValueError("Gemini api_key and model are required")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent"
    )
    payload: dict[str, Any] = {"contents": _format_gemini_contents(messages)}
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key, "alt": "sse"}
    last_text = ""

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            async with client.stream(
                "POST", url, headers=headers, params=params, json=payload
            ) as response:
                response.raise_for_status()
                buffer: list[str] = []
                async for line in response.aiter_lines():
                    if not line:
                        if not buffer:
                            continue
                        raw = "\n".join(buffer).strip()
                        buffer.clear()
                        if not raw:
                            continue
                        if raw == "[DONE]":
                            return
                        data = json.loads(raw)
                        text = _extract_gemini_text(data, allow_empty=True)
                        if text:
                            delta = text[len(last_text) :] if text.startswith(last_text) else text
                            last_text = text
                            if delta:
                                yield delta
                        continue
                    if line.startswith("data:"):
                        buffer.append(line[5:].strip())
                if buffer:
                    raw = "\n".join(buffer).strip()
                    if raw and raw != "[DONE]":
                        data = json.loads(raw)
                        text = _extract_gemini_text(data, allow_empty=True)
                        if text:
                            delta = text[len(last_text) :] if text.startswith(last_text) else text
                            if delta:
                                yield delta
        except httpx.HTTPError as exc:
            _log_http_error("Gemini", exc)
            raise


async def call_model(model: dict[str, Any], messages: list[dict[str, str]], system_prompt: str) -> str:
    model_type = model.get("type")
    if model_type == "azure-openai":
        return await _call_azure_openai(model, messages, system_prompt)
    if model_type == "gemini":
        return await _call_gemini(model, messages, system_prompt)
    raise ValueError(f"Unsupported model type '{model_type}'")


async def stream_model(
    model: dict[str, Any],
    messages: list[dict[str, str]],
    system_prompt: str,
    on_final_text: Callable[[str], None] | None = None,
) -> AsyncIterator[str]:
    model_type = model.get("type")
    if model_type == "azure-openai":
        async for chunk in _stream_azure_openai(
            model, messages, system_prompt, on_final_text=on_final_text
        ):
            yield chunk
        return
    if model_type == "gemini":
        text = ""
        async for chunk in _stream_gemini(model, messages, system_prompt):
            text += chunk
            yield chunk
        if on_final_text:
            on_final_text(text)
        return
    text = await call_model(model, messages, system_prompt)
    if on_final_text:
        on_final_text(text)
    yield text
