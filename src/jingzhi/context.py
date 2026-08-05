from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from jingzhi.database import CrossSessionEvidenceRecord, Database


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


@dataclass(frozen=True, slots=True)
class SynthesisEvidence:
    stable_id: str
    session_id: str
    session_title: str
    kind: str
    source: str
    start_ms: int
    end_ms: int
    text: str | None
    image_url: str | None


@dataclass(frozen=True, slots=True)
class SynthesisContext:
    evidence: tuple[SynthesisEvidence, ...]

    @property
    def prompt_text(self) -> str:
        sections = [
            "以下内容仅包含用户明确选择并授权用于本次综合的证据。",
            "只能根据这些证据回答问题；不要补充未提供的会话内容，也不要把推测写成事实。",
        ]
        for item in self.evidence:
            time_range = (
                f"{item.start_ms / 1000:.1f}s-{item.end_ms / 1000:.1f}s"
                if item.start_ms != item.end_ms
                else f"{item.start_ms / 1000:.1f}s"
            )
            content = item.text or "（关键帧图像见附件）"
            sections.append(
                f"[{item.stable_id}] 会话={item.session_title}({item.session_id}) "
                f"时间={time_range} 类型={item.kind} 来源={item.source}\n{content}"
            )
        return "\n\n".join(sections)

    @property
    def image_urls(self) -> tuple[str, ...]:
        return tuple(item.image_url for item in self.evidence if item.image_url is not None)


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

    def for_cross_session(
        self, records: tuple[CrossSessionEvidenceRecord, ...]
    ) -> SynthesisContext:
        evidence: list[SynthesisEvidence] = []
        for record in records:
            image_url = None
            if record.kind == "frame":
                if record.resource_path is None:
                    raise RuntimeError(f"关键帧 {record.stable_id} 缺少文件路径")
                try:
                    image_bytes = record.resource_path.read_bytes()
                except OSError as exc:
                    raise RuntimeError(f"关键帧 {record.stable_id} 无法读取") from exc
                suffix = record.resource_path.suffix.lower()
                media_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                }.get(suffix, "image/png")
                image_url = (
                    f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
                )
            evidence.append(
                SynthesisEvidence(
                    stable_id=record.stable_id,
                    session_id=record.session_id,
                    session_title=record.session_title,
                    kind=record.kind,
                    source=record.source,
                    start_ms=record.start_ms,
                    end_ms=record.end_ms,
                    text=record.content_text,
                    image_url=image_url,
                )
            )
        return SynthesisContext(tuple(evidence))

    def for_material(self, session_id: str) -> QuestionContext:
        """Build the exact effective transcript evidence for a whole session."""
        transcripts = self.database.all_effective_transcripts(session_id)
        if not transcripts:
            raise RuntimeError("No transcript is available yet")
        end_ms = max(item.end_ms for item in transcripts)
        return QuestionContext(
            start_ms=0,
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
            frames=(),
        )
