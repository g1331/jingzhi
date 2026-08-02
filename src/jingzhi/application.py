from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from jingzhi.database import (
    Database,
    SessionRecord,
    TimelineFrameRecord,
    TimelineTranscriptRecord,
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
    duration_ms: int
    window_start_ms: int
    window_end_ms: int


class JingzhiApplicationService:
    """Application boundary used by the Qt UI and hardware-free use-case tests."""

    def __init__(
        self,
        database: Database,
        *,
        recorder: RecordingAdapter,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self.recorder = recorder
        self._now = now or (lambda: datetime.now(UTC))

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
        start_ms = max(0, window_start_ms)
        end_ms = duration_ms if window_duration_ms is None else start_ms + window_duration_ms
        return SessionTimeline(
            session=session,
            frames=tuple(self.database.timeline_frames(session_id, start_ms, end_ms)),
            transcripts=tuple(self.database.timeline_transcripts(session_id, start_ms, end_ms)),
            duration_ms=duration_ms,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
        )

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
