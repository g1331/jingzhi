from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from jingzhi.capture.audio import AudioChunk
from jingzhi.database import Database

logger = logging.getLogger(__name__)


class TranscriptionWorker(threading.Thread):
    def __init__(
        self,
        *,
        database: Database,
        chunk_queue: queue.Queue[AudioChunk | None],
        model_name: str,
        device: str,
        compute_type: str,
        on_segment: Callable[[int, int, str, str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name="transcription", daemon=True)
        self.database = database
        self.chunk_queue = chunk_queue
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.on_segment = on_segment
        self.on_error = on_error

    def run(self) -> None:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
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
                segments, info = model.transcribe(
                    str(chunk.path),
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 400},
                )
                for segment in segments:
                    text = segment.text.strip()
                    if not text:
                        continue
                    start_ms = chunk.start_ms + round(segment.start * 1000)
                    end_ms = chunk.start_ms + round(segment.end * 1000)
                    confidence = None
                    if getattr(segment, "avg_logprob", None) is not None:
                        confidence = float(segment.avg_logprob)
                    self.database.add_transcript(
                        chunk.session_id,
                        chunk.id,
                        chunk.source,
                        start_ms,
                        end_ms,
                        text,
                        getattr(info, "language", None),
                        confidence,
                    )
                    if self.on_segment:
                        self.on_segment(start_ms, end_ms, chunk.source, text)
                self.database.set_chunk_state(chunk.id, "transcribed")
            except Exception as exc:
                logger.exception("Transcription failed for %s", chunk.path)
                self.database.set_chunk_state(chunk.id, "failed", str(exc))
                if self.on_error:
                    self.on_error(f"Transcription failed for {chunk.path.name}: {exc}")
            finally:
                self.chunk_queue.task_done()
