from __future__ import annotations

import queue

from jingzhi.capture.audio import AudioChunk
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.transcribe import TranscriptionWorker


def test_whisper_initialization_failure_marks_queued_audio_failed(tmp_path, monkeypatch) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("恢复", "2026-01-01T00:00:00+00:00")
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")
    chunk_id = database.add_audio_chunk(session_id, "microphone", 0, 2_000, audio_path)
    chunks: queue.Queue[AudioChunk | None] = queue.Queue()
    chunks.put(AudioChunk(chunk_id, session_id, "microphone", 0, 2_000, audio_path))
    chunks.put(None)

    class BrokenWhisperModel:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("model unavailable")

    monkeypatch.setattr("faster_whisper.WhisperModel", BrokenWhisperModel)
    errors: list[str] = []
    worker = TranscriptionWorker(
        database=database,
        chunk_queue=chunks,
        settings=Settings(data_dir=tmp_path).whisper,
        on_error=errors.append,
    )

    worker.run()

    assert database.audio_chunk_state(chunk_id) == "failed"
    assert errors and "model unavailable" in errors[0]
