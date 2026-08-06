from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from jingzhi.database import Database, TranscriptCorrectionRunRecord


CORRECTION_WINDOW_SECONDS = (15, 30, 60)
CORRECTION_WINDOW_MS = tuple(seconds * 1000 for seconds in CORRECTION_WINDOW_SECONDS)


@dataclass(frozen=True, slots=True)
class CorrectionSegment:
    id: int
    start_ms: int
    end_ms: int
    source: str
    text: str
    version_id: int | None = None


@dataclass(frozen=True, slots=True)
class CorrectionFrame:
    id: int
    source_id: str
    ts_ms: int
    path: Path


@dataclass(frozen=True, slots=True)
class CorrectionRequest:
    session_id: str
    window_start_ms: int
    window_end_ms: int
    target_segments: tuple[CorrectionSegment, ...]
    context_segments: tuple[CorrectionSegment, ...]
    frames: tuple[CorrectionFrame, ...]


class TranscriptCorrectionModel(Protocol):
    model: str

    def correct(self, request: CorrectionRequest) -> dict[int, str]: ...


class CorrectionWindowBatcher:
    """Serializes correction windows and reschedules them when new segments arrive."""

    def __init__(self, window_seconds: int) -> None:
        if window_seconds not in CORRECTION_WINDOW_SECONDS:
            raise ValueError("Correction window must be 15, 30, or 60 seconds")
        self.window_ms = window_seconds * 1000
        self._pending: set[tuple[str, int]] = set()
        self._dirty: set[tuple[str, int]] = set()
        self._lock = Lock()

    def add_segment(self, session_id: str, start_ms: int) -> tuple[tuple[str, int], ...]:
        window = (session_id, start_ms // self.window_ms * self.window_ms)
        with self._lock:
            if window in self._pending:
                self._dirty.add(window)
                return ()
            self._pending.add(window)
        return (window,)

    def register(self, window: tuple[str, int]) -> None:
        with self._lock:
            self._pending.add(window)
            self._dirty.discard(window)

    def start(self, window: tuple[str, int]) -> None:
        with self._lock:
            if window not in self._pending:
                raise RuntimeError("Correction window is not pending")
            self._dirty.discard(window)

    def complete(self, window: tuple[str, int]) -> tuple[tuple[str, int], ...]:
        with self._lock:
            if window not in self._pending:
                raise RuntimeError("Correction window is not pending")
            if window in self._dirty:
                self._dirty.remove(window)
                return (window,)
            self._pending.remove(window)
        return ()


class TranscriptCorrectionProcessor:
    """Runs one correction window while preserving raw and user-authored versions."""

    def __init__(self, database: Database, model: TranscriptCorrectionModel) -> None:
        self.database = database
        self.model = model

    def run(self, session_id: str, *, window_start_ms: int) -> TranscriptCorrectionRunRecord:
        settings = self.database.transcript_correction_settings(session_id)
        if not settings.enabled:
            raise RuntimeError("Transcript correction is disabled")

        start_ms = max(0, window_start_ms)
        end_ms = start_ms + settings.window_ms
        target_records = [
            item
            for item in self.database.correction_segments(session_id, start_ms, end_ms)
            if not any(
                version.kind in {"correction", "user_edit"}
                for version in self.database.transcript_versions(item.id)
            )
        ]
        context_records = self.database.correction_segments(
            session_id, max(0, start_ms - settings.window_ms), end_ms + settings.window_ms
        )
        frame_records = self.database.representative_frames(session_id, start_ms, end_ms)
        request = CorrectionRequest(
            session_id=session_id,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            target_segments=tuple(
                CorrectionSegment(
                    item.id, item.start_ms, item.end_ms, item.source, item.text, item.version_id
                )
                for item in target_records
            ),
            context_segments=tuple(
                CorrectionSegment(
                    item.id, item.start_ms, item.end_ms, item.source, item.text, item.version_id
                )
                for item in context_records
            ),
            frames=tuple(
                CorrectionFrame(item.id, item.source_id, item.ts_ms, item.path)
                for item in frame_records
            ),
        )
        run_id = self.database.start_correction_run(session_id, start_ms, end_ms, self.model.model)
        if not request.target_segments:
            return self.database.finish_correction_run(run_id, "corrected")
        try:
            corrected = self.model.correct(request)
            target_ids = {segment.id for segment in request.target_segments}
            returned_ids = set(corrected)
            if returned_ids != target_ids:
                missing_ids = sorted(target_ids - returned_ids)
                unknown_ids = sorted(returned_ids - target_ids)
                raise ValueError(
                    "Correction returned an incomplete segment map: "
                    f"missing={missing_ids}, unknown={unknown_ids}"
                )
            for segment in request.target_segments:
                text = corrected.get(segment.id)
                if text is not None:
                    self.database.add_transcript_version(
                        segment.id,
                        "correction",
                        text,
                        model=self.model.model,
                    )
        except Exception as exc:  # noqa: BLE001 - model boundary records any provider failure
            return self.database.finish_correction_run(
                run_id,
                "failed",
                error_source=self.model.model,
                error=str(exc),
            )
        return self.database.finish_correction_run(run_id, "corrected")
