from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from jingzhi.database import (
    Database,
    SessionRecord,
    TimelineFrameRecord,
    TimelineQuestionRecord,
    TimelineTranscriptRecord,
    TranscriptCorrectionRunRecord,
    TranscriptCorrectionSettingsRecord,
    TranscriptVersionRecord,
)
from jingzhi.transcript_correction import (
    TranscriptCorrectionModel,
    TranscriptCorrectionProcessor,
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

    def list_sessions(self) -> list[SessionRecord]:
        return self.database.list_sessions()

    def open_session(
        self,
        session_id: str,
        *,
        window_start_ms: int = 0,
        window_duration_ms: int | None = None,
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
        return SessionTimeline(
            session=session,
            frames=tuple(self.database.timeline_frames(session_id, start_ms, end_ms)),
            transcripts=tuple(self._timeline_transcripts(session_id, start_ms, end_ms)),
            questions=tuple(self.database.timeline_questions(session_id, start_ms, end_ms)),
            duration_ms=duration_ms,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
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

    def transcript_correction_settings(self, session_id: str) -> TranscriptCorrectionSettingsRecord:
        return self.database.transcript_correction_settings(session_id)

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
