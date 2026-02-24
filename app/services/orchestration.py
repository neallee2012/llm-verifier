from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .agent_runtime import (
    AgentTask,
    MasterRoutingAgent,
    POLISHER_AGENT_ID,
    RESPONDER_AGENT_ID,
    RoutingDecision,
    SubAgentResult,
    VERIFIER_AGENT_ID,
)
from .llm import call_model, stream_model
from .model_config import find_model, resolve_agent_model_id


def _merge_instructions(*parts: str | None) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


@dataclass(slots=True)
class OrchestrationOutput:
    responder: SubAgentResult
    verifier: SubAgentResult | None
    polisher: SubAgentResult
    decision: RoutingDecision


class MultiAgentService:
    def __init__(self, config: dict[str, Any], system_prompt: str) -> None:
        self.config = config
        self.system_prompt = system_prompt
        routing = config.get("routing", {})
        threshold = routing.get("confidence_threshold", 0.95)
        self.master = MasterRoutingAgent(confidence_threshold=float(threshold))
        self.enable_verifier_shortcut = bool(
            routing.get("enable_verifier_shortcut", True)
        )

    async def run(
        self,
        *,
        thread_id: str,
        user_message: str,
        responder_history: list[dict[str, Any]],
        text_history: list[dict[str, Any]],
        verifier_enabled: bool,
    ) -> OrchestrationOutput:
        responder_model = self._find_agent_model(RESPONDER_AGENT_ID)
        verifier_model = self._find_agent_model(VERIFIER_AGENT_ID) if verifier_enabled else None
        polisher_model = self._find_agent_model(POLISHER_AGENT_ID)

        task = AgentTask(
            thread_id=thread_id,
            user_message=user_message,
            history=text_history,
            system_prompt=self.system_prompt,
            metadata={
                "original_user_message": user_message,
                "disable_verifier": not verifier_enabled,
                "enable_verifier_shortcut": self.enable_verifier_shortcut,
            },
        )

        responder_text = await call_model(
            responder_model,
            [*responder_history, {"role": "user", "content": user_message}],
            _merge_instructions(self.system_prompt, responder_model.get("instructions")),
        )
        responder_result = SubAgentResult(
            agent_id=RESPONDER_AGENT_ID,
            content=responder_text,
            confidence=_estimate_responder_confidence(user_message, responder_text),
        )
        decision = self.master.decide_route(task, responder_result)

        verifier_result: SubAgentResult | None = None
        latest_content = responder_text
        if VERIFIER_AGENT_ID in decision.path:
            if verifier_model is None:
                raise ValueError("Verifier model is not configured")
            verifier_messages = [
                *text_history,
                {"role": "user", "content": self._verifier_user_prompt(user_message, latest_content)},
            ]
            verifier_text = await call_model(
                verifier_model,
                verifier_messages,
                _merge_instructions(
                    "You are a verification specialist. Validate factual correctness, identify issues, then provide a corrected draft.",
                    self.system_prompt,
                    verifier_model.get("instructions"),
                ),
            )
            verifier_result = SubAgentResult(
                agent_id=VERIFIER_AGENT_ID,
                content=verifier_text,
            )
            latest_content = verifier_text

        polisher_messages = [
            *text_history,
            {"role": "user", "content": self._polisher_user_prompt(user_message, latest_content)},
        ]
        polisher_text = await call_model(
            polisher_model,
            polisher_messages,
            _merge_instructions(
                "You are a final-response polisher. Produce the final user-facing answer only.",
                self.system_prompt,
                polisher_model.get("instructions"),
            ),
        )
        polisher_result = SubAgentResult(
            agent_id=POLISHER_AGENT_ID,
            content=polisher_text,
        )
        return OrchestrationOutput(
            responder=responder_result,
            verifier=verifier_result,
            polisher=polisher_result,
            decision=decision,
        )

    async def stream(
        self,
        *,
        thread_id: str,
        user_message: str,
        responder_history: list[dict[str, Any]],
        text_history: list[dict[str, Any]],
        verifier_enabled: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        responder_model = self._find_agent_model(RESPONDER_AGENT_ID)
        verifier_model = self._find_agent_model(VERIFIER_AGENT_ID) if verifier_enabled else None
        polisher_model = self._find_agent_model(POLISHER_AGENT_ID)

        task = AgentTask(
            thread_id=thread_id,
            user_message=user_message,
            history=text_history,
            system_prompt=self.system_prompt,
            metadata={
                "original_user_message": user_message,
                "disable_verifier": not verifier_enabled,
                "enable_verifier_shortcut": self.enable_verifier_shortcut,
            },
        )

        yield {"event": "status", "stage": RESPONDER_AGENT_ID, "status": "started"}
        responder_text = ""
        async for chunk in stream_model(
            responder_model,
            [*responder_history, {"role": "user", "content": user_message}],
            _merge_instructions(self.system_prompt, responder_model.get("instructions")),
        ):
            responder_text += chunk
            yield {"event": "token", "stage": RESPONDER_AGENT_ID, "delta": chunk}
        yield {"event": "status", "stage": RESPONDER_AGENT_ID, "status": "done"}

        responder_result = SubAgentResult(
            agent_id=RESPONDER_AGENT_ID,
            content=responder_text,
            confidence=_estimate_responder_confidence(user_message, responder_text),
        )
        decision = self.master.decide_route(task, responder_result)
        yield {
            "event": "routing",
            "decision": {
                "path": decision.path,
                "reason": decision.reason,
                "skipped_verifier": decision.skipped_verifier,
                "confidence_threshold": decision.confidence_threshold,
                "responder_confidence": decision.responder_confidence,
                "is_simple_task": decision.is_simple_task,
            },
        }

        latest_content = responder_text
        if VERIFIER_AGENT_ID in decision.path:
            if verifier_model is None:
                raise ValueError("Verifier model is not configured")
            yield {"event": "status", "stage": VERIFIER_AGENT_ID, "status": "started"}
            verifier_text = ""
            verifier_messages = [
                *text_history,
                {"role": "user", "content": self._verifier_user_prompt(user_message, latest_content)},
            ]
            async for chunk in stream_model(
                verifier_model,
                verifier_messages,
                _merge_instructions(
                    "You are a verification specialist. Validate factual correctness, identify issues, then provide a corrected draft.",
                    self.system_prompt,
                    verifier_model.get("instructions"),
                ),
            ):
                verifier_text += chunk
                yield {"event": "token", "stage": VERIFIER_AGENT_ID, "delta": chunk}
            yield {"event": "status", "stage": VERIFIER_AGENT_ID, "status": "done"}
            latest_content = verifier_text
            yield {
                "event": "agent_output",
                "stage": VERIFIER_AGENT_ID,
                "content": verifier_text,
                "model": verifier_model.get("id"),
            }

        yield {"event": "status", "stage": POLISHER_AGENT_ID, "status": "started"}
        polisher_text = ""
        polisher_messages = [
            *text_history,
            {"role": "user", "content": self._polisher_user_prompt(user_message, latest_content)},
        ]
        async for chunk in stream_model(
            polisher_model,
            polisher_messages,
            _merge_instructions(
                "You are a final-response polisher. Produce the final user-facing answer only.",
                self.system_prompt,
                polisher_model.get("instructions"),
            ),
        ):
            polisher_text += chunk
            yield {"event": "token", "stage": POLISHER_AGENT_ID, "delta": chunk}
        yield {"event": "status", "stage": POLISHER_AGENT_ID, "status": "done"}
        yield {
            "event": "agent_output",
            "stage": RESPONDER_AGENT_ID,
            "content": responder_text,
            "model": responder_model.get("id"),
            "confidence": responder_result.confidence,
        }
        yield {
            "event": "agent_output",
            "stage": POLISHER_AGENT_ID,
            "content": polisher_text,
            "model": polisher_model.get("id"),
        }

    def _find_agent_model(self, agent_id: str) -> dict[str, Any]:
        model_id = resolve_agent_model_id(self.config, agent_id)
        return find_model(self.config, model_id)

    def _verifier_user_prompt(self, original_user_message: str, responder_output: str) -> str:
        return (
            "Original user question:\n"
            f"{original_user_message}\n\n"
            "Responder draft:\n"
            f"{responder_output}\n\n"
            "Check correctness/completeness and return an improved draft."
        )

    def _polisher_user_prompt(self, original_user_message: str, draft_output: str) -> str:
        return (
            "Original user question:\n"
            f"{original_user_message}\n\n"
            "Current draft:\n"
            f"{draft_output}\n\n"
            "Rewrite into a polished final answer for the user."
        )


def _estimate_responder_confidence(user_message: str, responder_text: str) -> float:
    if not responder_text.strip():
        return 0.0
    text = user_message.strip()
    if len(text) <= 80 and "\n" not in text:
        return 0.97
    if len(text) <= 220:
        return 0.93
    return 0.88
