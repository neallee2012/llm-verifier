from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from .agent_tools import ToolCallTrace, ToolRegistry, build_framework_tools

RESPONDER_AGENT_ID = "responder"
VERIFIER_AGENT_ID = "verifier"
POLISHER_AGENT_ID = "polisher"


@dataclass(slots=True)
class AgentTask:
    thread_id: str
    user_message: str
    history: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_user_message(self, user_message: str) -> AgentTask:
        return AgentTask(
            thread_id=self.thread_id,
            user_message=user_message,
            history=self.history,
            system_prompt=self.system_prompt,
            metadata=self.metadata,
        )


@dataclass(slots=True)
class SubAgentResult:
    agent_id: str
    content: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoutingDecision:
    path: list[str]
    reason: str
    skipped_verifier: bool
    confidence_threshold: float
    responder_confidence: float | None
    is_simple_task: bool


@dataclass(slots=True)
class OrchestrationResult:
    decision: RoutingDecision
    results: dict[str, SubAgentResult]
    final_output: str


@runtime_checkable
class SubAgent(Protocol):
    agent_id: str

    async def run(self, task: AgentTask) -> SubAgentResult:
        ...


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, SubAgent] = {}

    def register(self, agent: SubAgent) -> None:
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> SubAgent:
        if agent_id not in self._agents:
            raise ValueError(f"Agent '{agent_id}' is not registered")
        return self._agents[agent_id]

    def has(self, agent_id: str) -> bool:
        return agent_id in self._agents


class MasterRoutingAgent:
    def __init__(self, confidence_threshold: float = 0.95) -> None:
        self.confidence_threshold = confidence_threshold

    def decide_route(
        self, task: AgentTask, responder_result: SubAgentResult
    ) -> RoutingDecision:
        if bool(task.metadata.get("disable_verifier", False)):
            return RoutingDecision(
                path=[RESPONDER_AGENT_ID, POLISHER_AGENT_ID],
                reason="Verifier disabled by configuration",
                skipped_verifier=True,
                confidence_threshold=self.confidence_threshold,
                responder_confidence=responder_result.confidence,
                is_simple_task=True,
            )

        is_simple = self._is_simple_task(task)
        if not bool(task.metadata.get("enable_verifier_shortcut", True)):
            is_simple = False
        confidence = self._resolve_responder_confidence(task, responder_result)
        skip_verifier = (
            is_simple
            and confidence is not None
            and confidence >= self.confidence_threshold
        )
        path = [RESPONDER_AGENT_ID]
        if not skip_verifier:
            path.append(VERIFIER_AGENT_ID)
        path.append(POLISHER_AGENT_ID)
        reason = (
            "Skipped verifier: simple task with high responder confidence"
            if skip_verifier
            else "Included verifier: complex task or confidence below threshold"
        )
        return RoutingDecision(
            path=path,
            reason=reason,
            skipped_verifier=skip_verifier,
            confidence_threshold=self.confidence_threshold,
            responder_confidence=confidence,
            is_simple_task=is_simple,
        )

    def _is_simple_task(self, task: AgentTask) -> bool:
        configured = task.metadata.get("is_simple")
        if isinstance(configured, bool):
            return configured
        text = task.user_message.strip().lower()
        if not text:
            return True
        if len(text) > 220 or "\n" in text:
            return False
        complex_markers = (
            "architecture",
            "analyze",
            "比較",
            "分析",
            "設計",
            "step by step",
            "tradeoff",
            "workflow",
        )
        return not any(marker in text for marker in complex_markers)

    def _resolve_responder_confidence(
        self, task: AgentTask, responder_result: SubAgentResult
    ) -> float | None:
        if responder_result.confidence is not None:
            return responder_result.confidence
        meta_confidence = responder_result.metadata.get("confidence")
        if meta_confidence is None:
            meta_confidence = task.metadata.get("responder_confidence")
        parsed = _coerce_confidence(meta_confidence)
        if parsed is not None:
            return parsed
        match = re.search(
            r"confidence\s*[:=]\s*(\d+(?:\.\d+)?)\s*(%)?",
            responder_result.content,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        value = float(match.group(1))
        if match.group(2):
            value = value / 100
        return _coerce_confidence(value)


class MultiAgentOrchestrator:
    def __init__(self, master: MasterRoutingAgent | None = None) -> None:
        self.registry = AgentRegistry()
        self.master = master or MasterRoutingAgent()

    def register(self, agent: SubAgent) -> None:
        self.registry.register(agent)

    async def run(self, task: AgentTask) -> OrchestrationResult:
        responder = self.registry.get(RESPONDER_AGENT_ID)
        responder_result = await responder.run(task)
        decision = self.master.decide_route(task, responder_result)

        results: dict[str, SubAgentResult] = {RESPONDER_AGENT_ID: responder_result}
        running_output = responder_result.content
        for agent_id in decision.path[1:]:
            next_agent = self.registry.get(agent_id)
            step_result = await next_agent.run(task.with_user_message(running_output))
            results[agent_id] = step_result
            running_output = step_result.content
        return OrchestrationResult(
            decision=decision,
            results=results,
            final_output=running_output,
        )


class AgentFrameworkSubAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        name: str,
        client: Any,
        instructions: str,
        tool_registry: ToolRegistry | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        try:
            from agent_framework import Agent  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "agent-framework is required. Install with 'pip install agent-framework --pre'."
            ) from exc

        self.agent_id = agent_id
        self.tool_traces: list[ToolCallTrace] = []
        framework_tools = list(tools or [])
        if tool_registry is not None:
            framework_tools.extend(build_framework_tools(tool_registry, self.tool_traces))
        self._agent = Agent(
            client=client,
            name=name,
            instructions=instructions,
            tools=framework_tools,
        )

    async def run(self, task: AgentTask) -> SubAgentResult:
        start_trace_index = len(self.tool_traces)
        response = await self._agent.run(task.user_message)
        current_traces = [asdict(trace) for trace in self.tool_traces[start_trace_index:]]
        return SubAgentResult(
            agent_id=self.agent_id,
            content=str(response),
            confidence=_coerce_confidence(task.metadata.get(f"{self.agent_id}_confidence")),
            metadata={"tool_traces": current_traces},
        )


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 1 else None
