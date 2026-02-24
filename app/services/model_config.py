from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "models.json"

_DEFAULT_CONFIG: dict[str, Any] = {
    "primary_model_id": "azure-openai",
    "verifier_model_id": "gemini",
    "verifier_enabled": False,
    "models": [
        {
            "id": "azure-openai",
            "type": "azure-openai",
            "label": "Azure OpenAI",
            "endpoint": "",
            "deployment": "",
            "api_version": "2025-04-01-preview",
            "api_type": "chat-completions",
            "instructions": "",
        },
        {
            "id": "azure-grok",
            "type": "azure-openai",
            "label": "Azure Grok",
            "endpoint": "",
            "deployment": "",
            "api_version": "2025-04-01-preview",
            "api_type": "chat-completions",
            "instructions": "",
        },
        {
            "id": "gemini",
            "type": "gemini",
            "label": "Gemini",
            "api_key": "",
            "model": "gemini-3-pro-preview",
            "instructions": "",
        },
    ],
}


_config_cache: dict[str, Any] | None = None


def _apply_defaults(config: dict[str, Any]) -> bool:
    updated = False
    if "primary_model_id" not in config:
        config["primary_model_id"] = _DEFAULT_CONFIG["primary_model_id"]
        updated = True
    if "verifier_model_id" not in config:
        config["verifier_model_id"] = _DEFAULT_CONFIG["verifier_model_id"]
        updated = True
    if "verifier_enabled" not in config:
        config["verifier_enabled"] = _DEFAULT_CONFIG["verifier_enabled"]
        updated = True

    models = config.get("models")
    defaults = _DEFAULT_CONFIG["models"]
    normalized_models: list[dict[str, Any]] = []
    existing: dict[str, dict[str, Any]] = {}
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and model.get("id"):
                existing[model["id"]] = model
    for template in defaults:
        model = existing.get(template["id"], {})
        normalized_models.append({key: model.get(key, value) for key, value in template.items()})
    if config.get("models") != normalized_models:
        config["models"] = normalized_models
        updated = True

    return updated


def load_config() -> dict[str, Any]:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if CONFIG_PATH.exists():
        _config_cache = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        _config_cache = json.loads(json.dumps(_DEFAULT_CONFIG))
        save_config(_config_cache)
        return _config_cache
    if _apply_defaults(_config_cache):
        save_config(_config_cache)
    return _config_cache


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    global _config_cache
    _config_cache = config


def find_model(config: dict[str, Any], model_id: str | None) -> dict[str, Any]:
    if not model_id:
        raise ValueError("Model id is required")
    for model in config.get("models", []):
        if model.get("id") == model_id:
            return model
    raise ValueError(f"Model id '{model_id}' not found")
