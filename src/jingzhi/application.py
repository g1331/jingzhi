from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from jingzhi.archive import ArchiveManager, RestorePreview, RestoreResult
from jingzhi.capture.devices import RecordingSelection
from jingzhi.context import ContextAssembler, QuestionContext
from jingzhi.cross_session import CrossSessionSynthesisPreview
from jingzhi.database import (
    AnswerEvidenceRecord,
    AnswerVersionRecord,
    CrossSessionEvidenceRecord,
    CrossSessionSearchResult,
    CrossSessionSynthesisEvidenceRecord,
    CrossSessionSynthesisRecord,
    Database,
    MaterialEvidenceRecord,
    QuestionNoteRecord,
    SessionAnswerRecord,
    SessionMaterialVersionRecord,
    SessionNotificationKind,
    SessionRecord,
    SourceEventRecord,
    TimelineEventRecord,
    TimelineFrameRecord,
    TimelineQuestionRecord,
    TimelineTranscriptRecord,
    TranscriptCorrectionRunRecord,
    TranscriptCorrectionSettingsRecord,
    TranscriptVersionRecord,
)
from jingzhi.diagnostics import AudioRecoveryReport, RuntimeMetrics
from jingzhi.materials import MaterialGenerationPreview
from jingzhi.model_roles import RoleName
from jingzhi.model_routing import InvocationEvidence, ModelRouter, invocation_connection_json
from jingzhi.storage import storage_reader, storage_writer
from jingzhi.transcript_correction import (
    TranscriptCorrectionModel,
    TranscriptCorrectionProcessor,
)

logger = logging.getLogger(__name__)


class QuestionAnsweringService:
    """Owns question anchors, exact model evidence, and immutable answer versions."""

    def __init__(self, database: Database, router: ModelRouter) -> None:
        self.database = database
        self.router = router

    @storage_writer("创建问题锚点")
    def create_anchor(
        self, session_id: str, asked_at_ms: int, *, lookback_ms: int = 2 * 60_000
    ) -> int:
        if lookback_ms <= 0:
            raise ValueError("Question range must be greater than zero")
        return self.database.create_question(
            session_id,
            asked_at_ms,
            "",
            max(0, asked_at_ms - lookback_ms),
            asked_at_ms,
            state="draft",
        )

    @storage_writer("修改问题范围")
    def set_anchor_range(self, question_id: int, lookback_ms: int) -> None:
        if lookback_ms <= 0:
            raise ValueError("Question range must be greater than zero")
        question = self.database.question(question_id)
        if question is None or question.state != "draft":
            raise RuntimeError("The pending question anchor is unavailable")
        self.database.update_question_range(
            question_id, max(0, question.asked_at_ms - lookback_ms), question.asked_at_ms
        )

    @storage_writer("取消问题")
    def cancel_anchor(self, question_id: int) -> bool:
        return self.database.delete_pending_question(question_id)

    @storage_writer("提交问题")
    def submit(self, question_id: int, question: str) -> AnswerVersionRecord:
        question = question.strip()
        if not question:
            raise ValueError("Question is required")
        anchor = self.database.question(question_id)
        if (
            anchor is None
            or anchor.state != "draft"
            or anchor.context_start_ms is None
            or anchor.context_end_ms is None
        ):
            raise RuntimeError("The pending question anchor is unavailable")
        self.database.submit_question(question_id, question)
        context = ContextAssembler(self.database).for_anchor(
            anchor.session_id, anchor.context_start_ms, anchor.context_end_ms
        )
        return self._answer(question_id, question, context)

    def ask(
        self,
        session_id: str,
        asked_at_ms: int,
        question: str,
        *,
        lookback_ms: int = 2 * 60_000,
    ) -> AnswerVersionRecord:
        question_id = self.create_anchor(session_id, asked_at_ms, lookback_ms=lookback_ms)
        return self.submit(question_id, question)

    @storage_writer("重新回答问题")
    def reanswer(self, question_id: int) -> AnswerVersionRecord:
        question = self.database.question(question_id)
        if question is None:
            raise KeyError(f"Unknown question: {question_id}")
        if question.context_start_ms is None or question.context_end_ms is None:
            raise RuntimeError("The original question anchor is unavailable")
        context = ContextAssembler(self.database).for_anchor(
            question.session_id, question.context_start_ms, question.context_end_ms
        )
        return self._answer(question.id, question.question, context)

    def _answer(
        self, question_id: int, question: str, context: QuestionContext
    ) -> AnswerVersionRecord:
        evidence = context.persistence_items()
        invocation_evidence = tuple(
            InvocationEvidence(
                stable_id=item["stable_id"],
                kind=item["kind"],
                source=item["source"],
                start_ms=item["start_ms"],
                end_ms=item["end_ms"],
                transcript_version_id=item.get("transcript_version_id"),
                frame_id=item.get("frame_id"),
            )
            for item in evidence
        )
        anchor = self.database.question(question_id)
        if anchor is None:
            raise RuntimeError("Question anchor is unavailable")
        try:
            routed = self.router.invoke(
                RoleName.INSTANT_ANSWER,
                lambda model: model.answer(question, context),
                session_id=anchor.session_id,
                evidence=invocation_evidence,
                task_type="answer",
                task_payload_json=json.dumps(
                    {"question_id": question_id, "evidence": evidence},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        except Exception as exc:
            invocation = getattr(exc, "last_invocation", None)
            if invocation is None:
                raise
            self.database.record_answer_version(
                question_id,
                model=invocation.model,
                connection_json=invocation_connection_json(invocation),
                request_status="failed",
                request_id=invocation.request_id,
                answer=None,
                error=str(exc),
                evidence_state="exact",
                evidence=evidence,
                model_invocation_id=invocation.id,
            )
            raise
        result = routed.value
        self.database.resolve_retryable_model_tasks(
            "answer",
            anchor.session_id,
            payload_key="question_id",
            payload_value=question_id,
        )
        return self.database.record_answer_version(
            question_id,
            model=result.model or routed.invocation.model,
            connection_json=invocation_connection_json(routed.invocation),
            request_status="succeeded",
            request_id=result.request_id,
            answer=result.text,
            error=None,
            evidence_state="exact",
            evidence=evidence,
            model_invocation_id=routed.invocation.id,
        )


class RecordingAdapter(Protocol):
    @property
    def is_recording(self) -> bool: ...

    def start(self, title: str, *, selection: RecordingSelection | None = None) -> str: ...

    def stop(self) -> str | None: ...

    def pause(self) -> bool: ...

    def resume(self) -> bool: ...

    @property
    def is_paused(self) -> bool: ...

    def recording_status(self) -> RecordingStatus: ...


@dataclass(frozen=True, slots=True)
class RecordingStatus:
    state: str
    duration_ms: int
    display_count: int
    system_audio: bool
    microphone: bool
    failed_sources: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class AnswerEvidenceSummary:
    state: str
    frame_count: int
    transcript_count: int
    start_ms: int | None
    end_ms: int | None
    stable_ids: tuple[str, ...]


ANSWER_BOUNDARY_HEADINGS = ("会话证据确认", "补充解释", "无法确认")


def present_answer(answer: str, summary: AnswerEvidenceSummary | None) -> str:
    content = answer.strip()
    if not content:
        return "## 无法确认\n\n模型没有返回可展示的回答。"

    has_exact_evidence = bool(
        summary and summary.state == "exact" and (summary.frame_count or summary.transcript_count)
    )
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    allowed_headings = {f"## {heading}" for heading in ANSWER_BOUNDARY_HEADINGS}
    has_boundary = bool(
        lines
        and lines[0] in allowed_headings
        and all(line in allowed_headings for line in lines if line.startswith("## "))
    )
    if has_exact_evidence and has_boundary:
        return content
    if summary is not None and summary.state == "unavailable":
        reason = "此历史回答的精确会话证据不可恢复，不能重新判断哪些内容得到过证据确认。"
    elif not has_exact_evidence:
        reason = "当前回答没有可核验的会话证据。"
    else:
        reason = "模型回答未标明依据边界，以下内容不能视为会话证据确认。"
    quoted = "\n".join(f"> {line}" if line else ">" for line in content.splitlines())
    return f"## 无法确认\n\n{reason}\n\n### 模型原始回复\n\n{quoted}"


@dataclass(frozen=True, slots=True)
class SessionTimeline:
    session: SessionRecord
    frames: tuple[TimelineFrameRecord, ...]
    transcripts: tuple[TimelineTranscriptRecord, ...]
    questions: tuple[TimelineQuestionRecord, ...]
    duration_ms: int

    window_start_ms: int
    window_end_ms: int
    events: tuple[TimelineEventRecord, ...] = ()
    answer_frame_ids: frozenset[int] = frozenset()
    answer_transcript_ids: frozenset[int] = frozenset()
    selected_answer_id: int | None = None
    answer_evidence_summary: AnswerEvidenceSummary | None = None


@dataclass(frozen=True, slots=True)
class SessionNotification:
    session_id: str
    title: str
    kind: SessionNotificationKind


class JingzhiApplicationService:
    """Application boundary used by the Qt UI and hardware-free use-case tests."""

    def runtime_metrics(self) -> RuntimeMetrics:
        method = getattr(self.recorder, "runtime_metrics", None)
        if method is None:
            raise RuntimeError("Runtime metrics are unavailable for this recorder")
        return method()

    def recover_pending_audio(self, *, include_failed: bool = False) -> AudioRecoveryReport:
        method = getattr(self.recorder, "recover_pending_audio", None)
        if method is None:
            raise RuntimeError("Pending audio recovery is unavailable for this recorder")
        return method(include_failed=include_failed)

    def retry_failed_audio(self) -> AudioRecoveryReport:
        method = getattr(self.recorder, "retry_failed_audio", None)
        if method is None:
            raise RuntimeError("Failed audio retry is unavailable for this recorder")
        return method()

    def failed_audio_chunk_count(self) -> int:
        method = getattr(self.recorder, "database", None)
        if method is None:
            return 0
        _pending, failed = method.recovery_audio_counts()
        return int(failed)

    def retryable_model_task_count(self) -> int:
        method = getattr(self.recorder, "database", None)
        if method is None:
            return 0
        return int(method.retryable_model_task_count())

    def failed_correction_run_count(self) -> int:
        method = getattr(self.recorder, "database", None)
        if method is None:
            return 0
        return int(method.failed_correction_run_count())

    def retry_failed_correction_runs(self) -> int:
        method = getattr(self.recorder, "retry_failed_correction_runs", None)
        if method is None:
            raise RuntimeError("Failed correction retry is unavailable for this recorder")
        return int(method())

    def __init__(
        self,
        database: Database,
        *,
        recorder: RecordingAdapter,
        now: Callable[[], datetime] | None = None,
        correction_model: TranscriptCorrectionModel | None = None,
    ) -> None:
        self.database = database
        self.recorder = recorder
        self.database.reclaim_orphan_cross_session_retry_claims()
        self.archive = ArchiveManager(database, source_busy_reason=self.archive_storage_busy_reason)
        self._now = now or (lambda: datetime.now(UTC))
        self.correction_model = correction_model
        active_session_id = getattr(recorder, "session_id", None) if recorder.is_recording else None
        restart_at = self._now_utc().isoformat()
        self.database.interrupt_recording_sessions(restart_at, exclude_session_id=active_session_id)
        interrupted_models = self.database.running_model_invocations()
        self.database.recover_running_model_invocations(restart_at)
        start_recovered_correction_tasks = getattr(
            recorder, "start_recovered_correction_tasks", None
        )
        if callable(start_recovered_correction_tasks):
            start_recovered_correction_tasks()
        self._materialize_interrupted_model_tasks(interrupted_models)

    def _materialize_interrupted_model_tasks(self, invocations) -> None:  # type: ignore[no-untyped-def]
        for invocation in invocations:
            if not invocation.task_type or not invocation.task_payload_json:
                continue
            try:
                payload = json.loads(invocation.task_payload_json)
                if invocation.task_type == "answer":
                    self._materialize_interrupted_answer(invocation, payload)
                elif invocation.task_type == "material":
                    self._materialize_interrupted_material(invocation, payload)
                elif invocation.task_type == "cross_session":
                    self._materialize_interrupted_cross_session(invocation, payload)
            except Exception:
                logger.exception("Could not materialize interrupted model task %s", invocation.id)

    def _materialize_interrupted_answer(self, invocation, payload) -> None:  # type: ignore[no-untyped-def]
        question_id = int(payload["question_id"])
        question = self.database.question(question_id)
        if question is None:
            return
        if any(
            version.model_invocation_id == invocation.id
            for version in self.database.answer_versions(question_id)
        ):
            return
        raw_evidence = payload.get("evidence")
        evidence = (
            [dict(item) for item in raw_evidence if isinstance(item, dict)]
            if isinstance(raw_evidence, list)
            else []
        )
        if (
            not evidence
            and question.context_start_ms is not None
            and question.context_end_ms is not None
        ):
            try:
                evidence = (
                    ContextAssembler(self.database)
                    .for_anchor(
                        question.session_id, question.context_start_ms, question.context_end_ms
                    )
                    .persistence_items()
                )
            except Exception:
                logger.debug("Could not restore answer evidence for %s", question_id, exc_info=True)
        self.database.record_answer_version(
            question_id,
            model=invocation.model,
            connection_json=invocation_connection_json(invocation),
            request_status="failed",
            request_id=invocation.request_id,
            answer=None,
            error=invocation.error or "应用异常退出，回答任务可重试",
            evidence_state="exact" if evidence else "unavailable",
            evidence=evidence,
            model_invocation_id=invocation.id,
        )

    def _materialize_interrupted_material(self, invocation, payload) -> None:  # type: ignore[no-untyped-def]
        session_id = str(payload["session_id"])
        if self.database.session(session_id) is None:
            return
        if any(
            version.model_invocation_id == invocation.id
            for version in self.database.session_material_versions(session_id)
        ):
            return
        template_id = payload.get("template_id")
        raw_evidence = payload.get("evidence")
        evidence = (
            [dict(item) for item in raw_evidence if isinstance(item, dict)]
            if isinstance(raw_evidence, list)
            else []
        )
        if not evidence:
            try:
                evidence = (
                    ContextAssembler(self.database).for_material(session_id).persistence_items()
                )
            except Exception:
                logger.debug(
                    "Could not restore material evidence for %s", session_id, exc_info=True
                )
        self.database.record_material_version(
            session_id,
            kind="generated",
            content="模型任务在应用异常退出前未完成，可从材料入口重试。",
            template_id=str(template_id) if template_id is not None else None,
            model=invocation.model,
            connection_json=invocation_connection_json(invocation),
            model_invocation_id=invocation.id,
            request_status="failed",
            request_id=invocation.request_id,
            error=invocation.error or "应用异常退出，材料任务可重试",
            evidence_state="exact" if evidence else "unavailable",
            evidence=evidence,
        )

    def _materialize_interrupted_cross_session(self, invocation, payload) -> None:  # type: ignore[no-untyped-def]
        question = str(payload["question"]).strip()
        stable_ids = tuple(str(item) for item in payload["stable_ids"])
        if self.database.cross_session_synthesis_for_invocation(invocation.id) is not None:
            return
        raw_evidence = payload.get("evidence")
        if isinstance(raw_evidence, list):
            evidence = tuple(
                CrossSessionEvidenceRecord(
                    stable_id=str(item["stable_id"]),
                    session_id=str(item["session_id"]),
                    session_title=str(item["session_title"]),
                    kind=str(item["kind"]),
                    source=str(item["source"]),
                    start_ms=int(item["start_ms"]),
                    end_ms=int(item["end_ms"]),
                    content_text=(
                        str(item["content_text"]) if item.get("content_text") is not None else None
                    ),
                    resource_path=(
                        Path(str(item["resource_path"])) if item.get("resource_path") else None
                    ),
                    transcript_version_id=(
                        int(item["transcript_version_id"])
                        if item.get("transcript_version_id") is not None
                        else None
                    ),
                    frame_id=(int(item["frame_id"]) if item.get("frame_id") is not None else None),
                    answer_version_id=(
                        int(item["answer_version_id"])
                        if item.get("answer_version_id") is not None
                        else None
                    ),
                    material_version_id=(
                        int(item["material_version_id"])
                        if item.get("material_version_id") is not None
                        else None
                    ),
                )
                for item in raw_evidence
                if isinstance(item, dict)
            )
        else:
            evidence = tuple(self.database.cross_session_evidence_candidates(stable_ids))
        self.database.record_cross_session_synthesis(
            question=question,
            answer=None,
            model=invocation.model,
            connection_json=invocation_connection_json(invocation),
            model_invocation_id=invocation.id,
            request_status="failed",
            request_id=invocation.request_id,
            error=invocation.error or "应用异常退出，跨会话综合任务可重试",
            evidence_state="exact" if evidence else "unavailable",
            evidence=tuple(evidence),
            retry_of_id=(int(payload["retry_of_id"]) if payload.get("retry_of_id") else None),
        )

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording

    @storage_reader("导出会话")
    def export_session(self, session_id: str, destination: Path) -> Path:
        self._ensure_archive_source_idle()
        return self.archive.export_session(session_id, destination)

    @storage_reader("创建完整备份")
    def create_backup(self, destination: Path) -> Path:
        self._ensure_archive_source_idle()
        return self.archive.create_backup(destination)

    @storage_reader("检查完整备份")
    def preview_restore(self, archive: Path, target_dir: Path) -> RestorePreview:
        self._ensure_archive_source_idle()
        return self.archive.preview_restore(archive, target_dir)

    @storage_writer("恢复完整备份")
    def restore_backup(self, archive: Path, target_dir: Path) -> RestoreResult:
        self._ensure_archive_source_idle()
        return self.archive.restore_backup(archive, target_dir)

    def archive_storage_busy_reason(self) -> str | None:
        busy_reason = getattr(self.recorder, "storage_busy_reason", None)
        if callable(busy_reason):
            reason = busy_reason()
            if reason:
                return str(reason)
        if self.recorder.is_recording:
            return "正在记录会话"
        return None

    def _ensure_archive_source_idle(self) -> None:
        reason = self.archive_storage_busy_reason()
        if reason:
            raise RuntimeError(f"归档需要等待当前写入完成：{reason}")

    def session_storage_busy_reason(self, session_id: str) -> str | None:
        if getattr(self.recorder, "session_id", None) != session_id:
            return None
        return self.archive_storage_busy_reason()

    def cross_session_search(
        self, query: str, *, limit: int = 50
    ) -> list[CrossSessionSearchResult]:
        return self.database.cross_session_search(query, limit=limit)

    def cross_session_evidence_candidates(
        self, stable_ids: tuple[str, ...]
    ) -> list[CrossSessionEvidenceRecord]:
        return self.database.cross_session_evidence_candidates(stable_ids)

    def failed_cross_session_syntheses(
        self, *, limit: int = 10
    ) -> tuple[CrossSessionSynthesisRecord, ...]:
        method = getattr(self.recorder, "failed_cross_session_syntheses", None)
        if not callable(method):
            return ()
        return method(limit=limit)

    def cross_session_synthesis_preview(
        self, question: str, stable_ids: tuple[str, ...]
    ) -> CrossSessionSynthesisPreview:
        method = getattr(self.recorder, "cross_session_synthesis_preview", None)
        if not callable(method):
            raise TypeError("当前应用未配置跨会话综合模型")
        return method(question, stable_ids)

    def synthesize_cross_session(
        self, question: str, stable_ids: tuple[str, ...]
    ) -> CrossSessionSynthesisRecord:
        method = getattr(self.recorder, "synthesize_cross_session", None)
        if not callable(method):
            raise TypeError("当前应用未配置跨会话综合模型")
        return method(question, stable_ids)

    def retry_cross_session_synthesis(self, synthesis_id: int) -> CrossSessionSynthesisRecord:
        method = getattr(self.recorder, "retry_cross_session_synthesis", None)
        if not callable(method):
            raise TypeError("当前应用未配置跨会话综合模型")
        return method(synthesis_id)

    def cross_session_synthesis_evidence(
        self, synthesis_id: int
    ) -> tuple[CrossSessionSynthesisEvidenceRecord, ...]:
        return self.database.cross_session_synthesis_evidence(synthesis_id)

    def start_session(
        self,
        title: str,
        *,
        selection: RecordingSelection | None = None,
    ) -> str:
        return self.recorder.start(title, selection=selection)

    def stop_session(self) -> str | None:
        return self.recorder.stop()

    @property
    def supports_pause(self) -> bool:
        return callable(getattr(self.recorder, "pause", None)) and callable(
            getattr(self.recorder, "resume", None)
        )

    @property
    def is_paused(self) -> bool:
        return bool(getattr(self.recorder, "is_paused", False))

    def pause_session(self) -> bool:
        method = getattr(self.recorder, "pause", None)
        if not callable(method):
            raise TypeError("当前录制适配器不支持暂停")
        return bool(method())

    def resume_session(self) -> bool:
        method = getattr(self.recorder, "resume", None)
        if not callable(method):
            raise TypeError("当前录制适配器不支持恢复")
        return bool(method())

    def recording_status(self) -> RecordingStatus:
        method = getattr(self.recorder, "recording_status", None)
        if callable(method):
            return method()
        if not self.recorder.is_recording:
            return RecordingStatus("idle", 0, 0, False, False)
        return RecordingStatus(
            "paused" if self.is_paused else "recording",
            0,
            0,
            False,
            False,
        )

    def begin_question(self, lookback_ms: int = 2 * 60_000) -> int:
        return self.recorder.capture_question_anchor(lookback_ms)

    def set_question_range(self, lookback_ms: int) -> None:
        self.recorder.set_question_range(lookback_ms)

    def cancel_question(self) -> bool:
        return self.recorder.cancel_question()

    def submit_question(self, question: str) -> str:
        return self.recorder.answer(question)

    def start_question_voice(self) -> None:
        self.recorder.start_question_voice()

    def finish_question_voice(self) -> str:
        return self.recorder.finish_question_voice()

    def list_sessions(
        self,
        *,
        query: str = "",
        status: str = "all",
        newest_first: bool = True,
    ) -> list[SessionRecord]:
        current_session_id = (
            getattr(self.recorder, "session_id", None) if self.recorder.is_recording else None
        )
        records = self.database.list_sessions(
            query=query,
            status="all" if status == "unfinished" else status,
            newest_first=newest_first,
            current_session_id=current_session_id,
        )
        if status == "unfinished":
            return [item for item in records if item.status in {"recording", "interrupted"}]
        return records

    @storage_writer("固定会话")
    def pin_session(self, session_id: str, pinned: bool) -> None:
        now = self._now_utc().isoformat()
        if not self.database.set_session_pinned(
            session_id,
            pinned_at_utc=now if pinned else None,
            retention_started_at_utc=now,
        ):
            raise KeyError(f"Unknown active session: {session_id}")

    @storage_writer("移入回收区")
    def delete_session(self, session_id: str) -> None:
        busy_reason = self.session_storage_busy_reason(session_id)
        if busy_reason:
            raise RuntimeError(f"会话仍在写入：{busy_reason}")
        now = self._now_utc()
        if not self.database.move_session_to_trash(
            session_id, now.isoformat(), (now + timedelta(days=7)).isoformat()
        ):
            raise KeyError(f"Unknown deletable session: {session_id}")

    @storage_writer("恢复会话")
    def restore_session(self, session_id: str) -> None:
        if not self.database.restore_session(session_id, self._now_utc().isoformat()):
            raise KeyError(f"Unknown trashed session: {session_id}")

    @storage_writer("完成中断会话")
    def complete_interrupted_session(self, session_id: str) -> None:
        busy_reason = self.session_storage_busy_reason(session_id)
        if busy_reason:
            raise RuntimeError(f"会话仍在写入：{busy_reason}")
        if not self.database.complete_interrupted_session(session_id, self._now_utc().isoformat()):
            raise KeyError(f"Unknown interrupted session: {session_id}")

    def source_events(
        self, session_id: str, start_ms: int = 0, end_ms: int | None = None
    ) -> list[SourceEventRecord]:
        return self.database.source_events(session_id, start_ms, end_ms)

    @storage_writer("确认数据缺口")
    def confirm_data_gap(self, session_id: str, source_event_id: int) -> int:
        source_event = self.database.source_event(source_event_id)
        if source_event is None or source_event.session_id != session_id:
            raise KeyError(f"Unknown source event for session: {source_event_id}")
        return self.database.confirm_data_gap(source_event_id)

    @storage_writer("执行会话清理")
    def run_session_maintenance(self) -> tuple[SessionNotification, ...]:
        now = self._now_utc()
        notices: list[SessionNotification] = []

        def append_final_delete_failure(session_id: str, title: str) -> None:
            if self.database.record_final_delete_failure(session_id, now.isoformat()):
                notices.append(
                    SessionNotification(
                        session_id, title, SessionNotificationKind.FINAL_DELETE_FAILED
                    )
                )

        for pending in self.database.pending_media_deletions():
            if self.session_storage_busy_reason(pending.session_id):
                continue
            try:
                finalized = self.database.finalize_pending_media_deletion(pending.session_id)
            except OSError:
                append_final_delete_failure(pending.session_id, pending.title)
                continue
            if finalized:
                notices.append(
                    SessionNotification(
                        pending.session_id,
                        pending.title,
                        SessionNotificationKind.PERMANENTLY_DELETED,
                    )
                )

        for session in self.database.list_sessions():
            if (
                self.session_storage_busy_reason(session.id)
                or session.pinned
                or session.status == "recording"
            ):
                continue
            anchor_text = (
                session.retention_started_at_utc or session.ended_at_utc or session.started_at_utc
            )
            due_at = self._as_utc(datetime.fromisoformat(anchor_text)) + timedelta(days=30)
            remaining = due_at - now
            if remaining <= timedelta(0):
                expires_at = now + timedelta(days=7)
                if self.database.move_session_to_trash(
                    session.id,
                    now.isoformat(),
                    expires_at.isoformat(),
                    allow_pinned=False,
                    expected_session=session,
                ) and self.database.record_session_notification(
                    session.id,
                    SessionNotificationKind.MOVED_TO_TRASH,
                    now.isoformat(),
                    expected_trashed=True,
                ):
                    notices.append(
                        SessionNotification(
                            session.id, session.title, SessionNotificationKind.MOVED_TO_TRASH
                        )
                    )
                continue
            kind = None
            if remaining <= timedelta(days=1):
                kind = SessionNotificationKind.RETENTION_1D
            elif remaining <= timedelta(days=7):
                kind = SessionNotificationKind.RETENTION_7D
            if kind and self.database.record_session_notification(
                session.id,
                kind,
                now.isoformat(),
                expected_session=session,
            ):
                notices.append(SessionNotification(session.id, session.title, kind))

        for session in self.database.list_sessions(status="trash"):
            if self.session_storage_busy_reason(session.id):
                continue
            if session.trash_expires_at_utc is None:
                continue
            expires_at = self._as_utc(datetime.fromisoformat(session.trash_expires_at_utc))
            if expires_at > now:
                continue
            try:
                deleted = self.database.permanently_delete_session(session.id)
            except OSError:
                append_final_delete_failure(session.id, session.title)
                continue
            if deleted:
                notices.append(
                    SessionNotification(
                        session.id, session.title, SessionNotificationKind.PERMANENTLY_DELETED
                    )
                )
        return tuple(notices)

    def _now_utc(self) -> datetime:
        return self._as_utc(self._now())

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def open_session(
        self,
        session_id: str,
        *,
        window_start_ms: int = 0,
        window_duration_ms: int | None = None,
        answer_version_id: int | None = None,
    ) -> SessionTimeline:
        session = self.database.get_session(session_id)
        if session is None:
            raise KeyError(f"Unknown session: {session_id}")
        duration_ms = self._current_duration(session)
        if window_duration_ms is None:
            start_ms = 0
            end_ms = duration_ms
        else:
            maximum_start_ms = max(0, duration_ms - window_duration_ms)
            start_ms = min(max(0, window_start_ms), maximum_start_ms)
            end_ms = min(duration_ms, start_ms + window_duration_ms)

        selected_answer = None
        if answer_version_id is not None:
            selected_answer = next(
                (
                    answer
                    for answer in self.database.session_answers(session_id)
                    if answer.id == answer_version_id
                ),
                None,
            )
            if selected_answer is None:
                raise KeyError(f"Unknown answer version for session: {answer_version_id}")

        transcripts = self._timeline_transcripts(session_id, start_ms, end_ms)
        answer_frame_ids: frozenset[int] = frozenset()
        answer_transcript_ids: frozenset[int] = frozenset()
        answer_evidence_summary: AnswerEvidenceSummary | None = None
        if selected_answer is not None:
            evidence = (
                self.database.answer_evidence(selected_answer.id)
                if selected_answer.evidence_state == "exact"
                else []
            )
            if selected_answer.evidence_state == "exact":
                answer_frame_ids = frozenset(
                    item.frame_id
                    for item in evidence
                    if item.kind == "frame" and item.frame_id is not None
                )
                version_ids = tuple(
                    item.transcript_version_id
                    for item in evidence
                    if item.kind == "transcript" and item.transcript_version_id is not None
                )
                cited_transcripts = self.database.timeline_transcript_versions(
                    session_id, version_ids, start_ms, end_ms
                )
                cited_by_segment = {item.id: item for item in cited_transcripts}
                transcripts = [cited_by_segment.get(item.id, item) for item in transcripts]
                answer_transcript_ids = frozenset(cited_by_segment)
            evidence_start_ms = min((item.start_ms for item in evidence), default=None)
            evidence_end_ms = max((item.end_ms for item in evidence), default=None)
            answer_evidence_summary = AnswerEvidenceSummary(
                state=selected_answer.evidence_state,
                frame_count=sum(item.kind == "frame" for item in evidence),
                transcript_count=sum(item.kind == "transcript" for item in evidence),
                start_ms=evidence_start_ms,
                end_ms=evidence_end_ms,
                stable_ids=tuple(item.stable_id for item in evidence),
            )

        return SessionTimeline(
            session=session,
            frames=tuple(self.database.timeline_frames(session_id, start_ms, end_ms)),
            transcripts=tuple(transcripts),
            events=tuple(self.database.timeline_events(session_id, start_ms, end_ms)),
            questions=tuple(self.database.timeline_questions(session_id, start_ms, end_ms)),
            duration_ms=duration_ms,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            answer_frame_ids=answer_frame_ids,
            answer_transcript_ids=answer_transcript_ids,
            selected_answer_id=selected_answer.id if selected_answer is not None else None,
            answer_evidence_summary=answer_evidence_summary,
        )

    def _timeline_transcripts(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[TimelineTranscriptRecord]:
        transcripts = self.database.timeline_transcripts(session_id, start_ms, end_ms)
        settings = self.database.transcript_correction_settings(session_id)
        if settings.enabled:
            transcripts.extend(self.database.recognizing_transcripts(session_id, start_ms, end_ms))
            transcripts.sort(key=lambda item: (item.start_ms, item.id))
        return transcripts

    def _current_duration(self, session: SessionRecord) -> int:
        if session.status != "recording":
            return session.duration_ms
        started = datetime.fromisoformat(session.started_at_utc)
        now = self._now()
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        return max(session.duration_ms, int((now - started).total_seconds() * 1000))

    @storage_writer("保存字幕校订配置")
    def configure_transcript_correction(
        self, session_id: str, *, enabled: bool, window_seconds: int
    ) -> None:
        self.database.configure_transcript_correction(
            session_id, enabled=enabled, window_ms=window_seconds * 1000
        )

    def transcript_versions(self, segment_id: int) -> list[TranscriptVersionRecord]:
        return self.database.transcript_versions(segment_id)

    def timeline_transcript_version(
        self, session_id: str, version_id: int, start_ms: int, end_ms: int
    ) -> TimelineTranscriptRecord | None:
        records = self.database.timeline_transcript_versions(
            session_id, (version_id,), start_ms, end_ms
        )
        return records[0] if records else None

    def answer_evidence_entries(
        self, session_id: str, answer_version_id: int
    ) -> list[AnswerEvidenceRecord]:
        if not any(
            answer.id == answer_version_id for answer in self.database.session_answers(session_id)
        ):
            raise KeyError(f"Unknown answer version for session: {answer_version_id}")
        return self.database.answer_evidence(answer_version_id)

    def resolve_answer_evidence(
        self, session_id: str, answer_version_id: int, stable_id: str
    ) -> TimelineFrameRecord | TimelineTranscriptRecord:
        evidence = next(
            (
                item
                for item in self.answer_evidence_entries(session_id, answer_version_id)
                if item.stable_id == stable_id
            ),
            None,
        )
        if evidence is None:
            raise LookupError("证据入口不属于当前回答")

        if evidence.kind == "frame":
            frame_id = evidence.frame_id
            if frame_id is None or stable_id != f"frame:{frame_id}":
                raise ValueError("证据链接协议不受支持")
            frame = next(
                (
                    item
                    for item in self.database.timeline_frames(
                        session_id, evidence.start_ms, evidence.end_ms
                    )
                    if item.id == frame_id
                ),
                None,
            )
            if frame is None:
                raise LookupError("证据目标不属于当前会话")
            session_media_dir = (self.database.path.parent / "sessions" / session_id).resolve()
            if not frame.path.resolve().is_relative_to(session_media_dir):
                raise PermissionError("证据链接不能访问当前会话目录以外的文件")
            return frame

        if evidence.kind == "transcript":
            version_id = evidence.transcript_version_id
            if version_id is None or stable_id != f"transcript-version:{version_id}":
                raise ValueError("证据链接协议不受支持")
            transcripts = self.database.timeline_transcript_versions(
                session_id,
                (version_id,),
                evidence.start_ms,
                evidence.end_ms,
            )
            if not transcripts:
                raise LookupError("证据目标不属于当前会话")
            return transcripts[0]

        raise ValueError("证据链接协议不受支持")

    def latest_question_id(self, session_id: str) -> int | None:
        return self.database.latest_question_id(session_id)

    def transcript_correction_settings(self, session_id: str) -> TranscriptCorrectionSettingsRecord:
        return self.database.transcript_correction_settings(session_id)

    def session_answers(self, session_id: str) -> list[SessionAnswerRecord]:
        return self.database.session_answers(session_id)

    def session_materials(self, session_id: str) -> list[SessionMaterialVersionRecord]:
        return self.database.session_material_versions(session_id)

    def material_generation_mode(self):
        mode = getattr(self.recorder, "material_generation_mode", None)
        return mode() if callable(mode) else None

    def material_generation_preview(self, session_id: str) -> MaterialGenerationPreview:
        preview = getattr(self.recorder, "material_generation_preview", None)
        if not callable(preview):
            raise TypeError("Session material generation is not configured")
        return preview(session_id)

    @storage_writer("生成会话材料")
    def generate_material(
        self, session_id: str, *, template_id: str | None = None
    ) -> SessionMaterialVersionRecord:
        generator = getattr(self.recorder, "generate_material", None)
        if not callable(generator):
            raise TypeError("Session material generation is not configured")
        return generator(session_id, template_id=template_id)

    @storage_writer("编辑会话材料")
    def edit_material(
        self, session_id: str, material_version_id: int, content: str
    ) -> SessionMaterialVersionRecord:
        material = self.database.material_version(material_version_id)
        if material is None or material.session_id != session_id:
            raise KeyError(f"Unknown material version for session: {material_version_id}")
        editor = getattr(self.recorder, "edit_material", None)
        if not callable(editor):
            raise TypeError("Session material editing is not configured")
        return editor(material_version_id, content)

    def material_evidence_entries(
        self, session_id: str, material_version_id: int
    ) -> list[MaterialEvidenceRecord]:
        material = self.database.material_version(material_version_id)
        if material is None or material.session_id != session_id:
            raise KeyError(f"Unknown material version for session: {material_version_id}")
        return self.database.material_evidence(material_version_id)

    @storage_writer("添加问题附注")
    def add_question_note(
        self, session_id: str, question_id: int, content: str
    ) -> QuestionNoteRecord:
        question = self.database.question(question_id)
        if question is None or question.session_id != session_id or question.state != "submitted":
            raise KeyError(f"Unknown submitted question for session: {question_id}")
        return self.database.add_question_note(question_id, content)

    def question_notes(self, session_id: str, question_id: int) -> list[QuestionNoteRecord]:
        question = self.database.question(question_id)
        if question is None or question.session_id != session_id:
            raise KeyError(f"Unknown question for session: {question_id}")
        return self.database.question_notes(question_id)

    @storage_writer("编辑字幕")
    def edit_transcript(self, segment_id: int, text: str) -> int:
        version_id = self.database.add_transcript_version(segment_id, "user_edit", text)
        assert version_id is not None
        return version_id

    @storage_writer("撤销字幕校订")
    def undo_transcript_correction(self, segment_id: int) -> None:
        self.database.undo_transcript_correction(segment_id)

    @storage_writer("运行字幕校订")
    def run_transcript_correction(
        self, session_id: str, *, window_start_ms: int
    ) -> TranscriptCorrectionRunRecord:
        settings = self.database.transcript_correction_settings(session_id)
        if not settings.enabled:
            raise RuntimeError("Transcript correction is disabled")
        model = self.correction_model
        if model is None:
            factory = getattr(self.recorder, "transcript_correction_model", None)
            if callable(factory):
                model = factory()
        if model is None:
            raise RuntimeError("Transcript correction model is not configured")
        return TranscriptCorrectionProcessor(self.database, model).run(
            session_id, window_start_ms=window_start_ms
        )
