from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class ToolCallTrace:
    tool_name: str
    input_text: str
    output_text: str | None
    started_at: str
    finished_at: str
    error: str | None = None


@runtime_checkable
class AgentTool(Protocol):
    name: str
    description: str

    async def run(self, input_text: str, context: dict[str, Any] | None = None) -> str:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def list(self) -> list[AgentTool]:
        return list(self._tools.values())


class WebSearchTool:
    name = "web_search"
    description = "Searches the web for current information and returns a concise summary."

    def __init__(
        self, search_client: Callable[[str], Awaitable[str]] | None = None
    ) -> None:
        self._search_client = search_client

    async def run(self, input_text: str, context: dict[str, Any] | None = None) -> str:
        if self._search_client is None:
            raise ValueError("Web search client is not configured")
        return await self._search_client(input_text)


def create_default_tool_registry(
    web_search_client: Callable[[str], Awaitable[str]] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool(search_client=web_search_client))
    return registry


def build_framework_tools(
    registry: ToolRegistry, traces: list[ToolCallTrace]
) -> list[Callable[[str], Awaitable[str]]]:
    wrappers: list[Callable[[str], Awaitable[str]]] = []
    for tool in registry.list():
        wrappers.append(_wrap_tool(tool, traces))
    return wrappers


def _wrap_tool(
    tool: AgentTool, traces: list[ToolCallTrace]
) -> Callable[[str], Awaitable[str]]:
    async def _invoke(input_text: str) -> str:
        started_at = _utc_now()
        output_text: str | None = None
        error_text: str | None = None
        try:
            output_text = await tool.run(input_text)
            return output_text
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            traces.append(
                ToolCallTrace(
                    tool_name=tool.name,
                    input_text=input_text,
                    output_text=output_text,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    error=error_text,
                )
            )

    _invoke.__name__ = tool.name.replace("-", "_")
    _invoke.__doc__ = tool.description
    return _invoke


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
