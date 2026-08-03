from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from jingzhi.database import Database


@dataclass(frozen=True, slots=True)
class TranscriptEvidence:
    stable_id: str
    segment_id: int
    version_id: int
    source: str
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    stable_id: str
    frame_id: int
    source: str
    ts_ms: int
    path: Path
    image_url: str


@dataclass(frozen=True, slots=True)
class QuestionContext:
    start_ms: int
    end_ms: int
    transcripts: tuple[TranscriptEvidence, ...]
    frames: tuple[FrameEvidence, ...]

    @property
    def transcript(self) -> str:
        return "\n".join(
            f"[{item.stable_id}][{item.start_ms / 1000:8.1f}s][{item.source}] {item.text}"
            for item in self.transcripts
        )

    @property
    def frame_paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.frames)

    def persistence_items(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = [
            {
                "stable_id": item.stable_id,
                "kind": "transcript",
                "source": item.source,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "transcript_version_id": item.version_id,
                "content_text": item.text,
            }
            for item in self.transcripts
        ]
        items.extend(
            {
                "stable_id": item.stable_id,
                "kind": "frame",
                "source": item.source,
                "start_ms": item.ts_ms,
                "end_ms": item.ts_ms,
                "frame_id": item.frame_id,
                "resource_path": str(item.path),
            }
            for item in self.frames
        )
        return items


class ContextAssembler:
    def __init__(self, database: Database) -> None:
        self.database = database

    def around_question(
        self, session_id: str, asked_at_ms: int, lookback_ms: int = 8 * 60_000
    ) -> QuestionContext:
        start_ms = max(0, asked_at_ms - lookback_ms)
        return self.for_anchor(session_id, start_ms, asked_at_ms)

    def for_anchor(self, session_id: str, start_ms: int, end_ms: int) -> QuestionContext:
        transcripts = self.database.answer_transcripts_between(session_id, start_ms, end_ms)
        frames = self.database.answer_frames_near(session_id, end_ms, start_ms, end_ms, limit=4)
        frame_evidence: list[FrameEvidence] = []
        for frame in sorted(frames, key=lambda item: (item.ts_ms, item.frame_id)):
            try:
                image_bytes = frame.path.read_bytes()
            except OSError:
                continue
            media_type = "image/webp" if frame.path.suffix.lower() == ".webp" else "image/png"
            image_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
            frame_evidence.append(
                FrameEvidence(
                    stable_id=f"frame:{frame.frame_id}",
                    frame_id=frame.frame_id,
                    source=frame.source,
                    ts_ms=frame.ts_ms,
                    path=frame.path,
                    image_url=image_url,
                )
            )
        return QuestionContext(
            start_ms=start_ms,
            end_ms=end_ms,
            transcripts=tuple(
                TranscriptEvidence(
                    stable_id=f"transcript-version:{item.version_id}",
                    segment_id=item.segment_id,
                    version_id=item.version_id,
                    source=item.source,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    text=item.text,
                )
                for item in transcripts
            ),
            frames=tuple(frame_evidence),
        )
