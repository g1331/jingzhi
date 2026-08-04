from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from jingzhi.capture.devices import RecordingSelection
from jingzhi.context import ContextAssembler, QuestionContext
from jingzhi.database import (
    AnswerEvidenceRecord,
    AnswerVersionRecord,
    Database,
    SessionAnswerRecord,
    SessionNotificationKind,
    SessionRecord,
    TimelineFrameRecord,
    TimelineQuestionRecord,
    TimelineTranscriptRecord,
    TranscriptCorrectionRunRecord,
    TranscriptCorrectionSettingsRecord,
    TranscriptVersionRecord,
)
from jingzhi.model_roles import RoleName
from jingzhi.model_routing import InvocationEvidence, ModelRouter
from jingzhi.storage import storage_writer
from jingzhi.transcript_correction import (
    TranscriptCorrectionModel,
    TranscriptCorrectionProcessor,
)


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
            )
        except Exception as exc:
            invocation = getattr(exc, "last_invocation", None)
            if invocation is None:
                raise
            self.database.record_answer_version(
                question_id,
                model=invocation.model,
                connection_json=self._connection_json(invocation),
                request_status="failed",
                request_id=invocation.request_id,
                answer=None,
                error=str(exc),
                evidence_state="exact",
                evidence=evidence,
            )
            raise
        result = routed.value
        return self.database.record_answer_version(
            question_id,
            model=result.model or routed.invocation.model,
            connection_json=self._connection_json(routed.invocation),
            request_status="succeeded",
            request_id=result.request_id,
            answer=result.text,
            error=None,
            evidence_state="exact",
            evidence=evidence,
        )

    @staticmethod
    def _connection_json(invocation) -> str:
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


class RecordingAdapter(Protocol):
    @property
    def is_recording(self) -> bool: ...

    def start(self, title: str, *, selection: RecordingSelection | None = None) -> str: ...

    def stop(self) -> str | None: ...


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
        self._now = now or (lambda: datetime.now(UTC))
        self.correction_model = correction_model
        active_session_id = getattr(recorder, "session_id", None) if recorder.is_recording else None
        self.database.interrupt_recording_sessions(
            self._now_utc().isoformat(), exclude_session_id=active_session_id
        )

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording

    def start_session(
        self,
        title: str,
        *,
        selection: RecordingSelection | None = None,
    ) -> str:
        return self.recorder.start(title, selection=selection)

    def stop_session(self) -> str | None:
        return self.recorder.stop()

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
        if self.recorder.is_recording and session_id == getattr(self.recorder, "session_id", None):
            raise RuntimeError("正在记录的会话不能删除")
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
        if not self.database.complete_interrupted_session(session_id, self._now_utc().isoformat()):
            raise KeyError(f"Unknown interrupted session: {session_id}")

    @storage_writer("执行会话清理")
    def run_session_maintenance(self) -> tuple[SessionNotification, ...]:
        now = self._now_utc()
        notices: list[SessionNotification] = []
        for pending in self.database.pending_media_deletions():
            try:
                finalized = self.database.finalize_pending_media_deletion(pending.session_id)
            except Exception:  # noqa: BLE001 - cleanup failure remains retryable
                notices.append(
                    SessionNotification(
                        pending.session_id,
                        pending.title,
                        SessionNotificationKind.FINAL_DELETE_FAILED,
                    )
                )
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
            if session.pinned or session.status == "recording":
                continue
            anchor_text = (
                session.retention_started_at_utc or session.ended_at_utc or session.started_at_utc
            )
            due_at = self._as_utc(datetime.fromisoformat(anchor_text)) + timedelta(days=30)
            remaining = due_at - now
            if remaining <= timedelta(0):
                expires_at = now + timedelta(days=7)
                if self.database.move_session_to_trash(
                    session.id, now.isoformat(), expires_at.isoformat()
                ):
                    self.database.record_session_notification(
                        session.id, SessionNotificationKind.MOVED_TO_TRASH, now.isoformat()
                    )
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
                session.id, kind, now.isoformat()
            ):
                notices.append(SessionNotification(session.id, session.title, kind))

        for session in self.database.list_sessions(status="trash"):
            if session.trash_expires_at_utc is None:
                continue
            expires_at = self._as_utc(datetime.fromisoformat(session.trash_expires_at_utc))
            if expires_at > now:
                continue
            try:
                deleted = self.database.permanently_delete_session(session.id)
            except Exception:  # noqa: BLE001 - cleanup failure remains retryable
                notices.append(
                    SessionNotification(
                        session.id,
                        session.title,
                        SessionNotificationKind.FINAL_DELETE_FAILED,
                    )
                )
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
