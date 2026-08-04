from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from jingzhi.capture.audio import AudioChunk
from jingzhi.database import Database
from jingzhi.whisper_settings import WhisperSettings, whisper_transcribe_options

logger = logging.getLogger(__name__)


def _transcribe_audio(model, path: Path, settings: WhisperSettings):  # type: ignore[no-untyped-def]
    return model.transcribe(str(path), **whisper_transcribe_options(settings))


class WhisperQuestionTranscriber:
    """Transcribes a completed question clip without persisting it as session evidence."""

    def __init__(self, settings: WhisperSettings, model_dir: Path | None = None) -> None:
        self.settings = settings
        self.model_dir = model_dir
        self._model = None

    def transcribe(self, path: Path) -> str:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.settings.model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
                download_root=str(self.model_dir) if self.model_dir is not None else None,
            )
        segments, _info = _transcribe_audio(self._model, path, self.settings)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        if not text:
            raise RuntimeError("No speech was recognized in the question recording")
        return text


class TranscriptionWorker(threading.Thread):
    def __init__(
        self,
        *,
        database: Database,
        chunk_queue: queue.Queue[AudioChunk | None],
        settings: WhisperSettings,
        model_dir: Path | None = None,
        on_segment: Callable[[int, int, str, str], None] | None = None,
        on_persisted_segment: Callable[[str, int, int], None] | None = None,
        on_recognition_started: Callable[[int, int, str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name="transcription", daemon=True)
        self.database = database
        self.chunk_queue = chunk_queue
        self.settings = settings
        self.model_dir = model_dir
        self.on_segment = on_segment
        self.on_persisted_segment = on_persisted_segment
        self.on_recognition_started = on_recognition_started
        self.on_error = on_error

    def run(self) -> None:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                self.settings.model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
                download_root=str(self.model_dir) if self.model_dir is not None else None,
            )
        except Exception as exc:
            logger.exception("Could not initialize transcription model")
            if self.on_error:
                self.on_error(f"Could not initialize Whisper model: {exc}")
            return

        while True:
            chunk = self.chunk_queue.get()
            if chunk is None:
                self.chunk_queue.task_done()
                return
            try:
                if self.on_recognition_started:
                    self.on_recognition_started(chunk.start_ms, chunk.end_ms, chunk.source)
                segments, info = _transcribe_audio(model, chunk.path, self.settings)
                persisted_segments: list[tuple[int, int, str, str]] = []
                correction_candidates: list[tuple[str, int, int]] = []
                for segment in segments:
                    text = segment.text.strip()
                    if not text:
                        continue
                    start_ms = chunk.start_ms + round(segment.start * 1000)
                    end_ms = chunk.start_ms + round(segment.end * 1000)
                    confidence = None
                    if getattr(segment, "avg_logprob", None) is not None:
                        confidence = float(segment.avg_logprob)
                    segment_id = self.database.add_transcript(
                        chunk.session_id,
                        chunk.id,
                        chunk.source,
                        start_ms,
                        end_ms,
                        text,
                        getattr(info, "language", None),
                        confidence,
                    )
                    correction_candidates.append((chunk.session_id, segment_id, start_ms))
                    persisted_segments.append((start_ms, end_ms, chunk.source, text))
                self.database.set_chunk_state(chunk.id, "transcribed")
                if self.on_persisted_segment:
                    for candidate in correction_candidates:
                        self.on_persisted_segment(*candidate)
                if self.on_segment:
                    if persisted_segments:
                        for persisted in persisted_segments:
                            self.on_segment(*persisted)
                    else:
                        self.on_segment(chunk.start_ms, chunk.end_ms, chunk.source, "")
            except Exception as exc:
                logger.exception("Transcription failed for %s", chunk.path)
                self.database.set_chunk_state(chunk.id, "failed", str(exc))
                if self.on_error:
                    self.on_error(f"Transcription failed for {chunk.path.name}: {exc}")
            finally:
                self.chunk_queue.task_done()
