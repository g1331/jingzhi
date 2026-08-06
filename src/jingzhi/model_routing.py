from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from jingzhi.database import (
    Database,
    ModelInvocationEvidenceRecord,
    ModelInvocationRecord,
)
from jingzhi.llm import OpenAIContextModel
from jingzhi.model_roles import (
    REASONING_EFFORT,
    ModelConnection,
    ModelRole,
    RoleName,
)
from jingzhi.provider_settings import SavedProviderSettings

InvocationEvidence = ModelInvocationEvidenceRecord
T = TypeVar("T")
AdapterFactory = Callable[[ModelConnection, str, str | None], Any]


@dataclass(frozen=True, slots=True)
class RoutedResult(Generic[T]):
    value: T
    invocation: ModelInvocationRecord


def invocation_connection_json(invocation: ModelInvocationRecord) -> str:
    return json.dumps(
        {
            "connection_id": invocation.connection_id,
            "connection_name": invocation.connection_name,
            "base_url": invocation.base_url,
            "api_mode": invocation.api_mode,
            "role": invocation.role,
            "reasoning_level": invocation.reasoning_level,
            "fallback_reason": invocation.fallback_reason,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class ModelRoutingError(RuntimeError):
    def __init__(self, cause: Exception, last_invocation: ModelInvocationRecord) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.last_invocation = last_invocation
        self.request_id = getattr(cause, "request_id", None)


class RoutedTranscriptCorrectionModel:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self.model = router._role(RoleName.TRANSCRIPT_CORRECTION).model

    def correct(self, request: Any) -> dict[int, str]:
        evidence = tuple(
            InvocationEvidence(
                stable_id=(
                    f"transcript-version:{segment.version_id}"
                    if segment.version_id is not None
                    else f"transcript:{segment.id}"
                ),
                kind="transcript",
                source=segment.source,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                transcript_version_id=segment.version_id,
            )
            for segment in request.context_segments
        ) + tuple(
            InvocationEvidence(
                stable_id=f"frame:{frame.id}",
                kind="frame",
                source=frame.source_id,
                start_ms=frame.ts_ms,
                end_ms=frame.ts_ms,
                frame_id=frame.id,
            )
            for frame in request.frames
        )
        try:
            result = self.router.invoke(
                RoleName.TRANSCRIPT_CORRECTION,
                lambda adapter: adapter.correct(request),
                session_id=request.session_id,
                evidence=evidence,
                task_type="transcript_correction",
                task_payload_json=json.dumps(
                    {
                        "session_id": request.session_id,
                        "window_start_ms": request.window_start_ms,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        except Exception as exc:
            invocation = getattr(exc, "last_invocation", None)
            if invocation is not None:
                self.model = invocation.model
            raise
        self.model = result.invocation.model
        return result.value


class ModelRouter:
    """Routes a model role through its authorized ordered fallback chain."""

    def __init__(
        self,
        database: Database,
        settings: SavedProviderSettings,
        *,
        adapter_factory: AdapterFactory | None = None,
    ) -> None:
        self.database = database
        self.connections = {connection.id: connection for connection in settings.connections}
        self.roles = {role.name: role for role in settings.roles}
        self.adapter_factory = adapter_factory or self._openai_adapter

    @staticmethod
    def _openai_adapter(
        connection: ModelConnection, model: str, reasoning_effort: str | None
    ) -> OpenAIContextModel:
        return OpenAIContextModel(
            model,
            api_key=connection.api_key,
            base_url=connection.base_url,
            api_mode=connection.api_mode,
            reasoning_effort=reasoning_effort,
        )

    def invoke(
        self,
        role_name: RoleName,
        operation: Callable[[Any], T],
        *,
        session_id: str | None = None,
        evidence: tuple[InvocationEvidence, ...] = (),
        task_type: str | None = None,
        task_payload_json: str | None = None,
    ) -> RoutedResult[T]:
        role = self._role(role_name)
        targets = [(role.connection_id, role.model)]
        targets.extend(
            (fallback.connection_id, fallback.model)
            for fallback in role.fallbacks
            if fallback.connection_id == role.connection_id or fallback.cross_connection_authorized
        )
        fallback_reason: str | None = None
        last_error: Exception | None = None
        last_invocation: ModelInvocationRecord | None = None
        attempt_ids: list[int] = []
        for connection_id, model in targets:
            connection = self._connection(connection_id)
            invocation_id = self.database.start_model_invocation(
                session_id=session_id,
                role=role.name.value,
                connection_id=connection.id,
                connection_name=connection.name,
                base_url=connection.base_url,
                api_mode=connection.api_mode,
                model=model,
                reasoning_level=role.reasoning.value,
                fallback_reason=fallback_reason,
                evidence=evidence,
                task_type=task_type,
                task_payload_json=task_payload_json,
            )
            attempt_ids.append(invocation_id)
            try:
                adapter = self.adapter_factory(
                    connection,
                    model,
                    REASONING_EFFORT[role.reasoning],
                )
                value = operation(adapter)
            except Exception as exc:  # noqa: BLE001 - provider boundary records every failure
                last_invocation = self.database.finish_model_invocation(
                    invocation_id,
                    "failed",
                    request_id=getattr(exc, "request_id", None),
                    error=str(exc),
                )
                fallback_reason = str(exc)
                last_error = exc
                continue
            invocation = self.database.finish_model_invocation(
                invocation_id,
                "succeeded",
                request_id=getattr(value, "request_id", None),
            )
            self.database.resolve_model_fallbacks(attempt_ids[:-1])
            return RoutedResult(value, invocation)
        if len(attempt_ids) > 1:
            self.database.resolve_model_fallbacks(attempt_ids[:-1])
        if last_error is None or last_invocation is None:
            raise RuntimeError(f"No model target is configured for role {role_name.value}")
        try:
            last_error.__dict__["last_invocation"] = last_invocation
        except (AttributeError, TypeError):
            raise ModelRoutingError(last_error, last_invocation) from last_error
        raise last_error

    def _role(self, name: RoleName) -> ModelRole:
        try:
            return self.roles[name]
        except KeyError as exc:
            raise RuntimeError(f"Model role is not configured: {name.value}") from exc

    def _connection(self, connection_id: str) -> ModelConnection:
        try:
            return self.connections[connection_id]
        except KeyError as exc:
            raise RuntimeError(f"Model connection is not configured: {connection_id}") from exc
