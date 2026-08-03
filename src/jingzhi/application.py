from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Protocol

from jingzhi.context import ContextAssembler, QuestionContext
from jingzhi.database import (
    AnswerVersionRecord,
    Database,
    SessionAnswerRecord,
    SessionRecord,
    TimelineFrameRecord,
    TimelineQuestionRecord,
    TimelineTranscriptRecord,
    TranscriptCorrectionRunRecord,
    TranscriptCorrectionSettingsRecord,
    TranscriptVersionRecord,
)
from jingzhi.llm import AnswerModelResult
from jingzhi.transcript_correction import (
    TranscriptCorrectionModel,
    TranscriptCorrectionProcessor,
)


@dataclass(frozen=True, slots=True)
class ModelConnectionSnapshot:
    model: str
    base_url: str
    api_mode: str


class AnswerModel(Protocol):
    def answer(self, question: str, context: QuestionContext) -> AnswerModelResult: ...


class QuestionAnsweringService:
    """Owns question anchors, exact model evidence, and immutable answer versions."""

    def __init__(
        self,
        database: Database,
        model: AnswerModel,
        connection: ModelConnectionSnapshot,
    ) -> None:
        self.database = database
        self.model = model
        self.connection = connection

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

    def set_anchor_range(self, question_id: int, lookback_ms: int) -> None:
        if lookback_ms <= 0:
            raise ValueError("Question range must be greater than zero")
        question = self.database.question(question_id)
        if question is None or question.state != "draft":
            raise RuntimeError("The pending question anchor is unavailable")
        self.database.update_question_range(
            question_id, max(0, question.asked_at_ms - lookback_ms), question.asked_at_ms
        )

    def cancel_anchor(self, question_id: int) -> bool:
        return self.database.delete_pending_question(question_id)

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
        connection_json = json.dumps(asdict(self.connection), ensure_ascii=False, sort_keys=True)
        evidence = context.persistence_items()
        try:
            result = self.model.answer(question, context)
        except Exception as exc:
            self.database.record_answer_version(
                question_id,
                model=self.connection.model,
                connection_json=connection_json,
                request_status="failed",
                request_id=getattr(exc, "request_id", None),
                answer=None,
                error=str(exc),
                evidence_state="exact",
                evidence=evidence,
            )
            raise
        return self.database.record_answer_version(
            question_id,
            model=result.model or self.connection.model,
            connection_json=connection_json,
            request_status="succeeded",
            request_id=result.request_id,
            answer=result.text,
            error=None,
            evidence_state="exact",
            evidence=evidence,
        )


class RecordingAdapter(Protocol):
    @property
    def is_recording(self) -> bool: ...

    def start(
        self,
        title: str,
        *,
        capture_system_audio: bool | None = None,
        capture_microphone: bool | None = None,
    ) -> str: ...

    def stop(self) -> str | None: ...


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
    answer_evidence_state: str | None = None


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

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording

    def start_session(
        self,
        title: str,
        *,
        capture_system_audio: bool | None = None,
        capture_microphone: bool | None = None,
    ) -> str:
        return self.recorder.start(
            title,
            capture_system_audio=capture_system_audio,
            capture_microphone=capture_microphone,
        )

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

    def list_sessions(self) -> list[SessionRecord]:
        return self.database.list_sessions()

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
        if selected_answer is not None and selected_answer.evidence_state == "exact":
            evidence = self.database.answer_evidence(selected_answer.id)
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
            answer_evidence_state=(
                selected_answer.evidence_state if selected_answer is not None else None
            ),
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

    def configure_transcript_correction(
        self, session_id: str, *, enabled: bool, window_seconds: int
    ) -> None:
        self.database.configure_transcript_correction(
            session_id, enabled=enabled, window_ms=window_seconds * 1000
        )

    def transcript_versions(self, segment_id: int) -> list[TranscriptVersionRecord]:
        return self.database.transcript_versions(segment_id)

    def latest_question_id(self, session_id: str) -> int | None:
        return self.database.latest_question_id(session_id)

    def transcript_correction_settings(self, session_id: str) -> TranscriptCorrectionSettingsRecord:
        return self.database.transcript_correction_settings(session_id)

    def session_answers(self, session_id: str) -> list[SessionAnswerRecord]:
        return self.database.session_answers(session_id)

    def edit_transcript(self, segment_id: int, text: str) -> int:
        version_id = self.database.add_transcript_version(segment_id, "user_edit", text)
        assert version_id is not None
        return version_id

    def undo_transcript_correction(self, segment_id: int) -> None:
        self.database.undo_transcript_correction(segment_id)

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
