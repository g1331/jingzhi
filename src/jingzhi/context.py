from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jingzhi.database import Database


@dataclass(frozen=True, slots=True)
class QuestionContext:
    start_ms: int
    end_ms: int
    transcript: str
    frame_paths: tuple[Path, ...]


class ContextAssembler:
    def __init__(self, database: Database) -> None:
        self.database = database

    def around_question(
        self, session_id: str, asked_at_ms: int, lookback_ms: int = 8 * 60_000
    ) -> QuestionContext:
        start_ms = max(0, asked_at_ms - lookback_ms)
        end_ms = asked_at_ms
        segments = self.database.transcripts_between(session_id, start_ms, end_ms)
        transcript = "\n".join(
            f"[{item.start_ms / 1000:8.1f}s][{item.source}] {item.text}" for item in segments
        )
        frames = self.database.nearest_frames(session_id, asked_at_ms, start_ms, end_ms, limit=4)
        return QuestionContext(
            start_ms=start_ms,
            end_ms=end_ms,
            transcript=transcript,
            frame_paths=tuple(item.path for item in sorted(frames, key=lambda item: item.ts_ms)),
        )
