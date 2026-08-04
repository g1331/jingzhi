from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from jingzhi.application import ModelConnectionSnapshot, QuestionAnsweringService
from jingzhi.capture.audio import AudioCaptureWorker, AudioChunk, QuestionVoiceRecorder
from jingzhi.capture.screen import ScreenCaptureWorker
from jingzhi.clock import SessionClock
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.llm import OpenAIContextModel
from jingzhi.provider_settings import ProviderSettingsStore, SavedProviderSettings
from jingzhi.transcribe import TranscriptionWorker, WhisperQuestionTranscriber
from jingzhi.transcript_correction import (
    CORRECTION_WINDOW_SECONDS,
    CorrectionWindowBatcher,
    TranscriptCorrectionProcessor,
)


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
        self.correction_enabled = settings.transcript_correction_enabled
        self.correction_window_seconds = settings.transcript_correction_window_seconds
        self.correction_model = settings.transcript_correction_model
        self.correction_queue: queue.Queue[tuple[str, int] | None] | None = None
        self.correction_worker: threading.Thread | None = None
        self.correction_batcher = CorrectionWindowBatcher(self.correction_window_seconds)
        self.correction_flush_event = threading.Event()
        self.llm_model = settings.llm_model
        self.llm_base_url = settings.llm_base_url
        self.llm_api_key = settings.llm_api_key
        self.llm_api_mode = settings.llm_api_mode
        self.provider_settings_store = ProviderSettingsStore(settings.data_dir)
        self.last_question_id: int | None = None
        self.pending_question_id: int | None = None
        self.question_voice_recorder: QuestionVoiceRecorder | None = None
        self.question_transcriber: WhisperQuestionTranscriber | None = None

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

    def transcript_correction_model(self) -> OpenAIContextModel:
        return OpenAIContextModel(
            self.correction_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            api_mode=self.llm_api_mode,
        )

    def configure_transcript_correction(
        self, *, enabled: bool, window_seconds: int, model: str
    ) -> None:
        if window_seconds not in CORRECTION_WINDOW_SECONDS:
            raise ValueError("Correction window must be 15, 30, or 60 seconds")
        model = model.strip()
        if not model:
            raise ValueError("Correction model is required")
        if window_seconds != self.correction_window_seconds:
            if self.is_recording:
                raise RuntimeError("Cannot change the correction window during a recording")
            self.correction_batcher = CorrectionWindowBatcher(window_seconds)
        self.correction_enabled = enabled
        self.correction_window_seconds = window_seconds
        self.correction_model = model
        if self.session_id is not None:
            self.database.configure_transcript_correction(
                self.session_id,
                enabled=enabled,
                window_ms=window_seconds * 1000,
            )
            if enabled:
                self._start_correction_worker()

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
        self.last_question_id = None
        self.pending_question_id = None
        self.chunk_queue = chunk_queue
        self.database.configure_transcript_correction(
            session_id,
            enabled=self.correction_enabled,
            window_ms=self.correction_window_seconds * 1000,
        )
        self.correction_flush_event.clear()
        if self.correction_enabled:
            self._start_correction_worker()
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
            on_persisted_segment=self._enqueue_correction,
            on_recognition_started=(
                (lambda start, end, source: self.on_segment(start, end, source, ""))
                if self.on_segment
                else None
            ),
            on_error=self.on_error,
        )
        self.transcriber.start()
        for worker in self.workers:
            worker.start()
        return session_id

    def _start_correction_worker(self) -> None:
        if self.correction_worker is not None and self.correction_worker.is_alive():
            return
        self.correction_queue = queue.Queue()

        def work() -> None:
            assert self.correction_queue is not None
            while True:
                item = self.correction_queue.get()
                try:
                    if item is None:
                        if self.correction_queue.empty():
                            return
                        self.correction_queue.put(None)
                        continue
                    self.correction_batcher.start(item)
                    session_id, window_start_ms = item
                    window_ms = self.correction_window_seconds * 1000
                    ready_at_ms = (
                        window_start_ms + window_ms + round(self.settings.audio_chunk_s * 1000)
                    )
                    now_ms = self.clock.now_ms() if self.clock is not None else ready_at_ms
                    wait_seconds = max(0, ready_at_ms - now_ms) / 1000
                    self.correction_flush_event.wait(wait_seconds)
                    settings = self.database.transcript_correction_settings(session_id)
                    if not settings.enabled:
                        continue
                    result = TranscriptCorrectionProcessor(
                        self.database, self.transcript_correction_model()
                    ).run(session_id, window_start_ms=window_start_ms)
                    if result.state == "failed" and self.on_error:
                        self.on_error(
                            f"Transcript correction failed via {result.error_source}: {result.error}"
                        )
                    if self.on_segment:
                        self.on_segment(window_start_ms, window_start_ms, "", "")
                finally:
                    if item is not None:
                        for retry in self.correction_batcher.complete(item):
                            self.correction_queue.put(retry)
                    self.correction_queue.task_done()

        self.correction_worker = threading.Thread(
            target=work, name="transcript-correction", daemon=True
        )
        self.correction_worker.start()

    def _enqueue_correction(self, session_id: str, _segment_id: int, start_ms: int) -> None:
        if not self.correction_enabled or self.correction_queue is None:
            return
        for item in self.correction_batcher.add_segment(session_id, start_ms):
            self.correction_queue.put(item)

    def test_provider(self) -> str:
        return self._context_model().test_connection()

    def stop(self) -> str | None:
        self.cancel_question()
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
        self.correction_flush_event.set()
        if self.correction_worker and self.correction_worker.is_alive():
            assert self.correction_queue is not None
            self.correction_queue.put(None)
            self.correction_worker.join(timeout=120)
        ended_at = datetime.now(UTC).isoformat()
        assert session_id is not None
        all_workers_stopped = all(not worker.is_alive() for worker in self.workers)
        transcriber_stopped = self.transcriber is None or not self.transcriber.is_alive()
        correction_stopped = self.correction_worker is None or not self.correction_worker.is_alive()
        status = (
            "complete"
            if all_workers_stopped and transcriber_stopped and correction_stopped
            else "interrupted"
        )
        self.database.finish_session(session_id, ended_at, status)
        self.stop_event = None
        self.workers = []
        self.chunk_queue = None
        self.transcriber = None
        self.correction_queue = None
        self.correction_worker = None
        self.correction_batcher = CorrectionWindowBatcher(self.correction_window_seconds)
        self.correction_flush_event.clear()
        return session_id

    def _question_service(self) -> QuestionAnsweringService:
        return QuestionAnsweringService(
            self.database,
            self._context_model(),
            ModelConnectionSnapshot(self.llm_model, self.llm_base_url, self.llm_api_mode),
        )

    def capture_question_anchor(self, lookback_ms: int = 2 * 60_000) -> int:
        if self.session_id is None or self.clock is None:
            raise RuntimeError("Start a study session before asking a question")
        if self.pending_question_id is None:
            self.pending_question_id = self._question_service().create_anchor(
                self.session_id, self.clock.now_ms(), lookback_ms=lookback_ms
            )
        return self.pending_question_id

    def set_question_range(self, lookback_ms: int) -> None:
        if self.pending_question_id is None:
            raise RuntimeError("There is no pending question anchor")
        self._question_service().set_anchor_range(self.pending_question_id, lookback_ms)

    def cancel_question(self) -> bool:
        voice_recorder = self.question_voice_recorder
        self.question_voice_recorder = None
        if voice_recorder is not None:
            voice_recorder.cancel()
        if self.pending_question_id is None:
            return False
        question_id = self.pending_question_id
        self.pending_question_id = None
        return self._question_service().cancel_anchor(question_id)

    def start_question_voice(self) -> None:
        if self.session_id is None or self.clock is None:
            raise RuntimeError("Start a study session before asking a question")
        self.capture_question_anchor()
        if self.question_voice_recorder is not None:
            raise RuntimeError("A question voice recording is already active")
        recorder = QuestionVoiceRecorder(
            self.settings.audio_capture_rate, self.settings.audio_storage_rate
        )
        path = (
            self.settings.data_dir
            / "sessions"
            / self.session_id
            / "questions"
            / f"{self.clock.now_ms():012d}.flac"
        )
        recorder.start(path)
        self.question_voice_recorder = recorder

    def finish_question_voice(self) -> str:
        recorder = self.question_voice_recorder
        if recorder is None:
            raise RuntimeError("No question voice recording is active")
        self.question_voice_recorder = None
        path = recorder.stop()
        if self.question_transcriber is None:
            self.question_transcriber = WhisperQuestionTranscriber(
                self.settings.whisper_model,
                self.settings.whisper_device,
                self.settings.whisper_compute_type,
            )
        try:
            return self.question_transcriber.transcribe(path)
        finally:
            path.unlink(missing_ok=True)

    def answer(self, question: str) -> str:
        if self.session_id is None or self.clock is None:
            raise RuntimeError("Start a study session before asking a question")
        question_id = self.capture_question_anchor()
        self.pending_question_id = None
        self.last_question_id = question_id
        result = self._question_service().submit(question_id, question)
        assert result.answer is not None
        return result.answer

    def reanswer_question(self, question_id: int) -> str:
        result = self._question_service().reanswer(question_id)
        self.last_question_id = question_id
        assert result.answer is not None
        return result.answer

    def reanswer_last_question(self) -> str:
        if self.session_id is None:
            raise RuntimeError("There is no current session")
        if self.last_question_id is None:
            self.last_question_id = self.database.latest_question_id(self.session_id)
        if self.last_question_id is None:
            raise RuntimeError("There is no question to answer again")
        question = self.database.question(self.last_question_id)
        if question is None or question.session_id != self.session_id:
            raise RuntimeError("The selected question does not belong to the current session")
        return self.reanswer_question(self.last_question_id)

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
