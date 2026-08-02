from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from jingzhi.capture.audio import AudioCaptureWorker, AudioChunk
from jingzhi.capture.screen import ScreenCaptureWorker
from jingzhi.clock import SessionClock
from jingzhi.config import Settings
from jingzhi.context import ContextAssembler
from jingzhi.database import Database
from jingzhi.llm import OpenAIContextModel
from jingzhi.provider_settings import ProviderSettingsStore, SavedProviderSettings
from jingzhi.transcribe import TranscriptionWorker


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        *,
        on_segment: Callable[[int, int, str, str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.database = Database(self.settings.data_dir / "jingzhi.sqlite3")
        self.on_segment = on_segment
        self.on_error = on_error
        self.session_id: str | None = None
        self.clock: SessionClock | None = None
        self.stop_event: threading.Event | None = None
        self.workers: list[threading.Thread] = []
        self.chunk_queue: queue.Queue[AudioChunk | None] | None = None
        self.transcriber: TranscriptionWorker | None = None
        self.llm_model = settings.llm_model
        self.llm_base_url = settings.llm_base_url
        self.llm_api_key = settings.llm_api_key
        self.llm_api_mode = settings.llm_api_mode
        self.provider_settings_store = ProviderSettingsStore(settings.data_dir)

    def configure_provider(self, *, model: str, base_url: str, api_key: str, api_mode: str) -> None:
        model = model.strip()
        if not model:
            raise ValueError("Model is required")
        if api_mode not in {"responses", "chat_completions"}:
            raise ValueError("API mode must be responses or chat_completions")
        self.llm_model = model
        self.llm_base_url = base_url.strip()
        self.llm_api_key = api_key.strip()
        self.llm_api_mode = api_mode

    def _context_model(self) -> OpenAIContextModel:
        return OpenAIContextModel(
            self.llm_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            api_mode=self.llm_api_mode,
        )

    def save_provider(self) -> None:
        self.provider_settings_store.save(
            SavedProviderSettings(
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                model=self.llm_model,
                api_mode=self.llm_api_mode,
            )
        )

    @property
    def is_recording(self) -> bool:
        return self.session_id is not None and self.stop_event is not None

    def start(
        self,
        title: str,
        *,
        capture_system_audio: bool | None = None,
        capture_microphone: bool | None = None,
    ) -> str:
        if self.is_recording:
            raise RuntimeError("A session is already recording")
        clock = SessionClock.start()
        session_id = self.database.create_session(
            title.strip() or "Untitled session", clock.started_at_utc
        )
        session_dir = self.settings.data_dir / "sessions" / session_id
        stop_event = threading.Event()
        chunk_queue: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=16)

        self.session_id = session_id
        self.clock = clock
        self.stop_event = stop_event
        self.chunk_queue = chunk_queue
        self.workers = [
            ScreenCaptureWorker(
                database=self.database,
                session_id=session_id,
                clock=clock,
                output_dir=session_dir / "frames",
                stop_event=stop_event,
                interval_s=self.settings.screen_interval_s,
                hash_distance=self.settings.screen_hash_distance,
                on_error=self.on_error,
            )
        ]
        system_audio_enabled = (
            self.settings.capture_system_audio
            if capture_system_audio is None
            else capture_system_audio
        )
        microphone_enabled = (
            self.settings.capture_microphone if capture_microphone is None else capture_microphone
        )
        for source, enabled in (
            ("system", system_audio_enabled),
            ("microphone", microphone_enabled),
        ):
            if enabled:
                self.workers.append(
                    AudioCaptureWorker(
                        database=self.database,
                        session_id=session_id,
                        clock=clock,
                        source=source,
                        output_dir=session_dir / "audio",
                        stop_event=stop_event,
                        chunk_queue=chunk_queue,
                        sample_rate=self.settings.audio_capture_rate,
                        storage_sample_rate=self.settings.audio_storage_rate,
                        chunk_s=self.settings.audio_chunk_s,
                        on_error=self.on_error,
                    )
                )
        self.transcriber = TranscriptionWorker(
            database=self.database,
            chunk_queue=chunk_queue,
            model_name=self.settings.whisper_model,
            device=self.settings.whisper_device,
            compute_type=self.settings.whisper_compute_type,
            on_segment=self.on_segment,
            on_error=self.on_error,
        )
        self.transcriber.start()
        for worker in self.workers:
            worker.start()
        return session_id

    def test_provider(self) -> str:
        return self._context_model().test_connection()

    def stop(self) -> str | None:
        if not self.is_recording:
            return None
        assert self.stop_event is not None
        assert self.chunk_queue is not None
        session_id = self.session_id
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=self.settings.audio_chunk_s + 3)
        if self.transcriber and self.transcriber.is_alive():
            self.chunk_queue.put(None)
            self.transcriber.join(timeout=120)
        ended_at = datetime.now(UTC).isoformat()
        assert session_id is not None
        all_workers_stopped = all(not worker.is_alive() for worker in self.workers)
        transcriber_stopped = self.transcriber is None or not self.transcriber.is_alive()
        status = "complete" if all_workers_stopped and transcriber_stopped else "interrupted"
        self.database.finish_session(session_id, ended_at, status)
        self.stop_event = None
        self.workers = []
        self.chunk_queue = None
        self.transcriber = None
        return session_id

    def answer(self, question: str) -> str:
        if self.session_id is None or self.clock is None:
            raise RuntimeError("Start a study session before asking a question")
        asked_at_ms = self.clock.now_ms()
        context = ContextAssembler(self.database).around_question(self.session_id, asked_at_ms)
        model = self._context_model()
        try:
            answer = model.answer(question, context)
            self.database.add_question(
                self.session_id,
                asked_at_ms,
                question,
                answer,
                context.start_ms,
                context.end_ms,
            )
            return answer
        except Exception as exc:
            self.database.add_question(
                self.session_id,
                asked_at_ms,
                question,
                None,
                context.start_ms,
                context.end_ms,
                str(exc),
            )
            raise

    def summarize(self) -> dict:
        if self.session_id is None:
            raise RuntimeError("There is no current session")
        transcript = "\n".join(
            f"[{item.start_ms / 1000:.1f}s][{item.source}] {item.text}"
            for item in self.database.all_transcripts(self.session_id)
        )
        if not transcript:
            raise RuntimeError("No transcript is available yet")
        result = self._context_model().summarize(transcript)
        created_at = datetime.now(UTC).isoformat()
        for kind in ("summary", "knowledge_points", "mistakes"):
            self.database.add_artifact(
                self.session_id,
                kind,
                created_at,
                json.dumps(result.get(kind), ensure_ascii=False),
                self.llm_model,
            )
        return result
