from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock

from jingzhi.context import ContextAssembler
from jingzhi.database import (
    CrossSessionEvidenceRecord,
    CrossSessionSearchResult,
    CrossSessionSynthesisRecord,
    Database,
    ModelInvocationEvidenceRecord,
)
from jingzhi.llm import SynthesisModelResult
from jingzhi.model_roles import RoleName
from jingzhi.model_routing import ModelRouter, invocation_connection_json

MAX_SYNTHESIS_EVIDENCE = 24
MAX_SYNTHESIS_CHARACTERS = 24_000
_SYNTHESIS_PROMPT_OVERHEAD = 800
_SYNTHESIS_EVIDENCE_OVERHEAD = 180


class CrossSessionSynthesisError(RuntimeError):
    def __init__(self, message: str, synthesis_id: int) -> None:
        super().__init__(message)
        self.synthesis_id = synthesis_id


@dataclass(frozen=True, slots=True)
class CrossSessionSynthesisPreview:
    question: str
    stable_ids: tuple[str, ...]
    evidence_count: int
    character_count: int
    frame_count: int
    connection_name: str | None
    model: str | None
    reasoning_level: str | None
    can_synthesize: bool
    reason: str | None = None


class CrossSessionSynthesisService:
    _retry_lock = Lock()

    def __init__(self, database: Database, router: ModelRouter) -> None:
        self.database = database
        self.router = router

    def search(self, query: str, *, limit: int = 50) -> list[CrossSessionSearchResult]:
        return self.database.cross_session_search(query, limit=limit)

    def evidence_candidates(self, stable_ids: tuple[str, ...]) -> list[CrossSessionEvidenceRecord]:
        return self.database.cross_session_evidence_candidates(stable_ids)

    def selected_evidence(
        self, stable_ids: tuple[str, ...]
    ) -> tuple[CrossSessionEvidenceRecord, ...]:
        return tuple(self.database.cross_session_selected_evidence(stable_ids))

    def preview(self, question: str, stable_ids: tuple[str, ...]) -> CrossSessionSynthesisPreview:
        question = question.strip()
        selected_ids = tuple(dict.fromkeys(item.strip() for item in stable_ids if item.strip()))
        role = self.router.roles.get(RoleName.DEEP_ANALYSIS)
        connection = self.router.connections.get(role.connection_id) if role else None
        base = {
            "question": question,
            "stable_ids": selected_ids,
            "connection_name": connection.name if connection else None,
            "model": role.model if role else None,
            "reasoning_level": role.reasoning.value if role else None,
        }
        if not question:
            return CrossSessionSynthesisPreview(
                **base,
                evidence_count=0,
                character_count=0,
                frame_count=0,
                can_synthesize=False,
                reason="请先填写综合问题。",
            )
        if not selected_ids:
            return CrossSessionSynthesisPreview(
                **base,
                evidence_count=0,
                character_count=0,
                frame_count=0,
                can_synthesize=False,
                reason="请至少选择一条证据。",
            )
        if len(selected_ids) > MAX_SYNTHESIS_EVIDENCE:
            return CrossSessionSynthesisPreview(
                **base,
                evidence_count=len(selected_ids),
                character_count=0,
                frame_count=0,
                can_synthesize=False,
                reason=f"一次最多选择 {MAX_SYNTHESIS_EVIDENCE} 条证据。",
            )
        selected = self.selected_evidence(selected_ids)
        if len(selected) != len(selected_ids):
            return CrossSessionSynthesisPreview(
                **base,
                evidence_count=len(selected),
                character_count=sum(len(item.content_text or "") for item in selected),
                frame_count=sum(item.kind == "frame" for item in selected),
                can_synthesize=False,
                reason="部分证据已被删除、回收或不再可用，请重新搜索并选择。",
            )
        character_count = sum(len(item.content_text or "") for item in selected)
        frame_count = sum(item.kind == "frame" for item in selected)
        try:
            estimated_request_size = self._estimate_request_size(question, selected)
        except OSError as exc:
            return CrossSessionSynthesisPreview(
                **base,
                evidence_count=len(selected),
                character_count=character_count,
                frame_count=frame_count,
                can_synthesize=False,
                reason=f"所选关键帧文件不可用：{exc}",
            )
        if estimated_request_size > MAX_SYNTHESIS_CHARACTERS:
            return CrossSessionSynthesisPreview(
                **base,
                evidence_count=len(selected),
                character_count=character_count,
                frame_count=frame_count,
                can_synthesize=False,
                reason=(
                    f"所选证据过长或提示过大，预计约 {estimated_request_size} 字，超过单次综合上限 "
                    f"{MAX_SYNTHESIS_CHARACTERS} 字，请减少证据。"
                ),
            )
        if role is None or connection is None:
            return CrossSessionSynthesisPreview(
                **base,
                evidence_count=len(selected),
                character_count=character_count,
                frame_count=frame_count,
                can_synthesize=False,
                reason="尚未配置深度分析模型。",
            )
        return CrossSessionSynthesisPreview(
            **base,
            evidence_count=len(selected),
            character_count=character_count,
            frame_count=frame_count,
            can_synthesize=True,
        )

    @staticmethod
    def _estimate_request_size(
        question: str, evidence: tuple[CrossSessionEvidenceRecord, ...]
    ) -> int:
        estimate = len(question) + _SYNTHESIS_PROMPT_OVERHEAD
        for item in evidence:
            estimate += (
                len(item.stable_id)
                + len(item.session_id)
                + len(item.session_title)
                + len(item.source)
                + len(item.content_text or "")
                + _SYNTHESIS_EVIDENCE_OVERHEAD
            )
            if item.kind == "frame":
                if item.resource_path is None or not item.resource_path.is_file():
                    raise OSError(str(item.resource_path or item.stable_id))
                estimate += (item.resource_path.stat().st_size * 4 + 2) // 3
        return estimate

    def retry(self, synthesis_id: int) -> CrossSessionSynthesisRecord:
        with self._retry_lock:
            failed = self.database.cross_session_synthesis(synthesis_id)
            if failed is None or failed.request_status != "failed":
                raise LookupError("找不到可重试的跨会话综合")
            if self.database.cross_session_successor_exists(synthesis_id):
                raise LookupError("该跨会话综合已经重试")
            if not self.database.claim_cross_session_retry(synthesis_id):
                raise LookupError("该跨会话综合已经重试")
            if failed.model_invocation_id is not None:
                self.database.resolve_model_fallbacks([failed.model_invocation_id])
            stable_ids = tuple(
                item.stable_id
                for item in self.database.cross_session_synthesis_evidence(synthesis_id)
            )
            try:
                return self.synthesize(failed.question, stable_ids, retry_of_id=synthesis_id)
            except CrossSessionSynthesisError:
                raise
            except Exception:
                self.database.release_cross_session_retry_claim(synthesis_id)
                raise

    def synthesize(
        self,
        question: str,
        stable_ids: tuple[str, ...],
        *,
        retry_of_id: int | None = None,
    ) -> CrossSessionSynthesisRecord:
        preview = self.preview(question, stable_ids)
        if not preview.can_synthesize:
            raise RuntimeError(preview.reason or "无法开始跨会话综合")
        selected = self.selected_evidence(preview.stable_ids)
        context = ContextAssembler(self.database).for_cross_session(selected)
        evidence = tuple(
            ModelInvocationEvidenceRecord(
                stable_id=item.stable_id,
                kind=item.kind,
                source=item.source,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                transcript_version_id=item.transcript_version_id,
                frame_id=item.frame_id,
                answer_version_id=item.answer_version_id,
                material_version_id=item.material_version_id,
                content_text=item.content_text,
                resource_path=item.resource_path,
            )
            for item in selected
        )
        try:
            routed = self.router.invoke(
                RoleName.DEEP_ANALYSIS,
                lambda model: model.synthesize(question.strip(), context),
                session_id=None,
                evidence=evidence,
                task_type="cross_session",
                task_payload_json=json.dumps(
                    {
                        "question": question.strip(),
                        "stable_ids": list(preview.stable_ids),
                        "evidence": [
                            {
                                "stable_id": item.stable_id,
                                "session_id": item.session_id,
                                "session_title": item.session_title,
                                "kind": item.kind,
                                "source": item.source,
                                "start_ms": item.start_ms,
                                "end_ms": item.end_ms,
                                "content_text": item.content_text,
                                "resource_path": (
                                    str(item.resource_path) if item.resource_path else None
                                ),
                                "transcript_version_id": item.transcript_version_id,
                                "frame_id": item.frame_id,
                                "answer_version_id": item.answer_version_id,
                                "material_version_id": item.material_version_id,
                            }
                            for item in selected
                        ],
                        "retry_of_id": retry_of_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        except Exception as exc:
            invocation = getattr(exc, "last_invocation", None)
            if invocation is not None:
                failed = self.database.record_cross_session_synthesis(
                    question=question.strip(),
                    answer=None,
                    model=invocation.model,
                    connection_json=invocation_connection_json(invocation),
                    model_invocation_id=invocation.id,
                    request_status="failed",
                    request_id=getattr(exc, "request_id", None),
                    error=str(exc),
                    evidence_state="exact",
                    evidence=selected,
                    retry_of_id=retry_of_id,
                )
                raise CrossSessionSynthesisError(str(exc), failed.id) from exc
            raise
        result = routed.value
        if isinstance(result, SynthesisModelResult):
            answer = result.text
            request_id = result.request_id
            model = result.model or routed.invocation.model
        elif isinstance(result, str):
            answer = result.strip()
            request_id = None
            model = routed.invocation.model
        else:
            failed = self.database.record_cross_session_synthesis(
                question=question.strip(),
                answer=None,
                model=routed.invocation.model,
                connection_json=invocation_connection_json(routed.invocation),
                model_invocation_id=routed.invocation.id,
                request_status="failed",
                request_id=None,
                error="综合模型返回了不支持的结果",
                evidence_state="exact",
                evidence=selected,
                retry_of_id=retry_of_id,
            )
            raise CrossSessionSynthesisError("综合模型返回了不支持的结果", failed.id)
        if not answer:
            failed = self.database.record_cross_session_synthesis(
                question=question.strip(),
                answer=None,
                model=model,
                connection_json=invocation_connection_json(routed.invocation),
                model_invocation_id=routed.invocation.id,
                request_status="failed",
                request_id=request_id,
                error="综合模型没有返回内容",
                evidence_state="exact",
                evidence=selected,
                retry_of_id=retry_of_id,
            )
            raise CrossSessionSynthesisError("综合模型没有返回内容", failed.id)
        if retry_of_id is not None:
            failed = self.database.cross_session_synthesis(retry_of_id)
            if failed is not None and failed.model_invocation_id is not None:
                self.database.resolve_model_fallbacks([failed.model_invocation_id])
        return self.database.record_cross_session_synthesis(
            question=question.strip(),
            answer=answer,
            model=model,
            connection_json=invocation_connection_json(routed.invocation),
            model_invocation_id=routed.invocation.id,
            request_status="succeeded",
            request_id=request_id,
            error=None,
            evidence_state="exact",
            evidence=selected,
            retry_of_id=retry_of_id,
        )
