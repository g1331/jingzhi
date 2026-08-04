from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RoleName(StrEnum):
    UTILITY = "utility"
    TRANSCRIPT_CORRECTION = "transcript_correction"
    INSTANT_ANSWER = "instant_answer"
    DEEP_ANALYSIS = "deep_analysis"


class ReasoningLevel(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


REASONING_EFFORT = {
    ReasoningLevel.FAST: "low",
    ReasoningLevel.BALANCED: "medium",
    ReasoningLevel.DEEP: "high",
}


@dataclass(frozen=True, slots=True)
class ModelConnection:
    id: str
    name: str
    base_url: str = ""
    api_key: str = ""
    api_mode: str = "responses"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Connection ID is required")
        if not self.name.strip():
            raise ValueError("Connection name is required")
        if self.api_mode not in {"responses", "chat_completions"}:
            raise ValueError("API mode must be responses or chat_completions")


@dataclass(frozen=True, slots=True)
class ModelFallback:
    connection_id: str
    model: str
    cross_connection_authorized: bool = False


@dataclass(frozen=True, slots=True)
class ModelRole:
    name: RoleName
    connection_id: str
    model: str
    reasoning: ReasoningLevel
    fallbacks: tuple[ModelFallback, ...] = ()

    def __post_init__(self) -> None:
        if len(self.fallbacks) > 2:
            raise ValueError("A model role supports at most two ordered fallbacks")


def default_roles(
    model: str = "gpt-5.5", correction_model: str | None = None
) -> tuple[ModelRole, ...]:
    correction_model = correction_model or model
    return (
        ModelRole(RoleName.UTILITY, "default", model, ReasoningLevel.FAST),
        ModelRole(
            RoleName.TRANSCRIPT_CORRECTION,
            "default",
            correction_model,
            ReasoningLevel.FAST,
        ),
        ModelRole(RoleName.INSTANT_ANSWER, "default", model, ReasoningLevel.BALANCED),
        ModelRole(RoleName.DEEP_ANALYSIS, "default", model, ReasoningLevel.DEEP),
    )
