from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime

from jingzhi.application import QuestionAnsweringService, RecordingStatus
from jingzhi.capture.audio import AudioCaptureWorker, AudioChunk, QuestionVoiceRecorder
from jingzhi.capture.devices import (
    DeviceCatalog,
    RecordingSelection,
    ResolvedRecordingSelection,
    WindowsDeviceCatalog,
)
from jingzhi.capture.screen import ScreenCaptureWorker
from jingzhi.clock import SessionClock
from jingzhi.config import Settings
from jingzhi.context import ContextAssembler
from jingzhi.cross_session import (
    CrossSessionSynthesisPreview,
    CrossSessionSynthesisService,
)
from jingzhi.database import (
    CrossSessionEvidenceRecord,
    CrossSessionSearchResult,
    CrossSessionSynthesisRecord,
    Database,
    SessionMaterialVersionRecord,
    SourceEventRecord,
    TimelineEventKind,
)
from jingzhi.diagnostics import (
    AudioRecoveryReport,
    RuntimeMetrics,
    SystemMetricsSampler,
    directory_size,
    disk_free_bytes,
    now_utc_iso,
)
from jingzhi.llm import MaterialModelResult
from jingzhi.material_settings import MaterialGenerationMode, MaterialGenerationSettingsStore
from jingzhi.materials import MaterialGenerationPreview, legacy_summary_to_markdown
from jingzhi.model_roles import ModelRole, RoleName
from jingzhi.model_routing import (
    InvocationEvidence,
    ModelRouter,
    RoutedTranscriptCorrectionModel,
    invocation_connection_json,
)
from jingzhi.provider_settings import ProviderSettingsStore, SavedProviderSettings
from jingzhi.recording_settings import RecordingPreferences, resolve_recording_selection
from jingzhi.storage import canonical_whisper_repository_id, storage_writer
from jingzhi.transcribe import TranscriptionWorker, WhisperQuestionTranscriber
from jingzhi.transcript_correction import (
    CORRECTION_WINDOW_SECONDS,
    CorrectionWindowBatcher,
    TranscriptCorrectionProcessor,
)
from jingzhi.whisper_settings import (
    BenchmarkResult,
    DownloadState,
    WhisperBenchmark,
    WhisperCapabilities,
    WhisperModelDownloader,
    WhisperSettings,
    WhisperSettingsStore,
    detect_whisper_capabilities,
    resolve_whisper_settings,
)

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        *,
        on_segment: Callable[[int, int, str, str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_source_event: Callable[[SourceEventRecord], None] | None = None,
        device_catalog: DeviceCatalog | None = None,
        whisper_capabilities: WhisperCapabilities | None = None,
    ) -> None:
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.database = Database(self.settings.data_dir / "jingzhi.sqlite3")
        self.on_segment = on_segment
        self.on_source_event = on_source_event
        self.on_error = on_error
        self.device_catalog = device_catalog or WindowsDeviceCatalog()
        self.session_id: str | None = None
        self.clock: SessionClock | None = None
        self.stop_event: threading.Event | None = None
        self.workers: list[threading.Thread] = []
        self.pause_event = threading.Event()
        self.recording_selection: ResolvedRecordingSelection | None = None
        self._pause_event_id: int | None = None
        self._failed_sources: set[str] = set()
        self._source_event_keys: set[tuple[str, str, int]] = set()
        self._lifecycle_lock = threading.RLock()
        self._source_event_lock = threading.RLock()
        self.chunk_queue: queue.Queue[AudioChunk | None] | None = None
        self.transcriber: TranscriptionWorker | None = None
        self.correction_enabled = settings.transcript_correction_enabled
        self.correction_window_seconds = settings.transcript_correction_window_seconds
        self.provider_settings = settings.provider_settings
        self.correction_queue: queue.Queue[tuple[str, int] | None] | None = None
        self.correction_worker: threading.Thread | None = None
        self.correction_batcher = CorrectionWindowBatcher(self.correction_window_seconds)
        self.correction_flush_event = threading.Event()
        self.provider_settings_store = ProviderSettingsStore(settings.data_dir)
        self.material_generation_settings_store = MaterialGenerationSettingsStore(settings.data_dir)
        self._material_generation_mode = self.material_generation_settings_store.load()
        self.whisper_settings_store = WhisperSettingsStore(settings.data_dir)
        self.whisper_settings = settings.whisper
        self.whisper_capabilities = whisper_capabilities or detect_whisper_capabilities()
        self.actual_whisper_settings = resolve_whisper_settings(
            self.whisper_settings, self.whisper_capabilities
        ).settings
        self.last_question_id: int | None = None
        self.pending_question_id: int | None = None
        self.question_voice_recorder: QuestionVoiceRecorder | None = None
        self.question_transcriber: WhisperQuestionTranscriber | None = None
        self._transcription_metrics_lock = threading.RLock()
        self._transcription_audio_ms = 0
        self._transcription_processing_seconds = 0.0
        self._resource_sampler = SystemMetricsSampler()
        recovered_correction_items = self.database.recover_running_correction_runs(
            datetime.now(UTC).isoformat()
        )
        self.recovered_correction_items = tuple(recovered_correction_items)

    def start_recovered_correction_tasks(self) -> None:
        retryable_items = [
            item
            for item in self.recovered_correction_items
            if self.database.transcript_correction_settings(item[0]).enabled
        ]
        if not retryable_items:
            return
        self._start_correction_worker()
        assert self.correction_queue is not None
        for item in retryable_items:
            self.correction_batcher.register(item)
            self.database.resolve_retryable_model_tasks(
                "transcript_correction",
                item[0],
                payload_key="window_start_ms",
                payload_value=item[1],
            )
            self.correction_queue.put(item)
        self.recovered_correction_items = ()

    def retry_failed_correction_runs(self) -> int:
        items = self.database.failed_correction_windows()
        if not items:
            return 0
        self._start_correction_worker()
        assert self.correction_queue is not None
        for item in items:
            self.correction_batcher.register(item)
            self.database.resolve_retryable_model_tasks(
                "transcript_correction",
                item[0],
                payload_key="window_start_ms",
                payload_value=item[1],
            )
            self.correction_queue.put(item)
        return len(items)

    def configure_provider(self, settings: SavedProviderSettings) -> None:
        connection_ids = {connection.id for connection in settings.connections}
        if not connection_ids:
            raise ValueError("At least one model connection is required")
        roles = {role.name: role for role in settings.roles}
        missing = set(RoleName) - roles.keys()
        if missing:
            names = ", ".join(sorted(role.value for role in missing))
            raise ValueError(f"Model roles are missing: {names}")
        for role in roles.values():
            if role.connection_id not in connection_ids:
                raise ValueError(f"Unknown connection for role {role.name.value}")
            if not role.model.strip():
                raise ValueError(f"Model is required for role {role.name.value}")
        self.provider_settings = settings

    def model_role(self, name: RoleName) -> ModelRole:
        for role in self.provider_settings.roles:
            if role.name == name:
                return role
        raise RuntimeError(f"Model role is not configured: {name.value}")

    def _model_router(self) -> ModelRouter:
        return ModelRouter(self.database, self.provider_settings)

    def transcript_correction_model(self) -> RoutedTranscriptCorrectionModel:
        return RoutedTranscriptCorrectionModel(self._model_router())

    @storage_writer("保存字幕校订配置")
    def configure_transcript_correction(self, *, enabled: bool, window_seconds: int) -> None:
        if window_seconds not in CORRECTION_WINDOW_SECONDS:
            raise ValueError("Correction window must be 15, 30, or 60 seconds")
        if window_seconds != self.correction_window_seconds:
            if self.is_recording:
                raise RuntimeError("Cannot change the correction window during a recording")
            self.correction_batcher = CorrectionWindowBatcher(window_seconds)
        self.correction_enabled = enabled
        self.correction_window_seconds = window_seconds
        if self.session_id is not None:
            self.database.configure_transcript_correction(
                self.session_id,
                enabled=enabled,
                window_ms=window_seconds * 1000,
            )
            if enabled:
                self._start_correction_worker()

    @storage_writer("保存模型连接配置")
    def save_provider(self) -> None:
        self.provider_settings_store.save(self.provider_settings)

    def configure_whisper(self, settings: WhisperSettings) -> str:
        if self.is_recording:
            raise RuntimeError("Cannot change Whisper settings during a recording")
        resolved = resolve_whisper_settings(settings, self.whisper_capabilities)
        self.whisper_settings = settings
        self.actual_whisper_settings = resolved.settings
        self.question_transcriber = None
        return resolved.fallback_advice

    @storage_writer("保存 Whisper 配置")
    def save_whisper(self) -> None:
        self.whisper_settings_store.save(self.whisper_settings)

    @storage_writer("运行 Whisper 样本测试")
    def benchmark_whisper(self, sample_path) -> BenchmarkResult:  # type: ignore[no-untyped-def]
        resolved = resolve_whisper_settings(self.whisper_settings, self.whisper_capabilities)
        result = WhisperBenchmark(model_dir=self.settings.model_dir).run(
            resolved.settings, sample_path
        )
        self.whisper_settings = replace(self.whisper_settings, first_run_completed=True)
        self.actual_whisper_settings = resolved.settings
        self.save_whisper()
        return result

    @storage_writer("下载 Whisper 模型")
    def prepare_whisper_model(
        self,
        *,
        on_progress: Callable[[DownloadState], None],
        cancel_event: threading.Event,
    ) -> DownloadState:
        return WhisperModelDownloader(model_dir=self.settings.model_dir).prepare(
            self.whisper_settings.model,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

    @property
    def is_recording(self) -> bool:
        return self.session_id is not None and self.stop_event is not None

    @property
    def is_paused(self) -> bool:
        return self.is_recording and self.pause_event.is_set()

    def recording_status(self) -> RecordingStatus:
        selection = self.recording_selection
        if not self.is_recording or self.clock is None:
            return RecordingStatus("idle", 0, 0, False, False)
        with self._source_event_lock:
            failed_sources = frozenset(self._failed_sources)
        return RecordingStatus(
            "paused" if self.is_paused else "recording",
            self.clock.now_ms(),
            len(selection.displays) if selection is not None else 0,
            selection is not None and selection.system_audio is not None,
            selection is not None and selection.microphone is not None,
            failed_sources,
        )

    def storage_busy_reason(self) -> str | None:
        if self.is_recording:
            return "正在录制会话"
        if self.question_voice_recorder is not None:
            return "正在录制语音提问"
        if any(worker.is_alive() for worker in self.workers):
            return "会话采集线程仍在写入数据"
        if self.transcriber is not None and self.transcriber.is_alive():
            return "字幕转写仍在写入数据"
        if self.correction_worker is not None and self.correction_worker.is_alive():
            return "字幕校订仍在写入数据"
        write_thread_names = {
            "answer-question",
            "reanswer-question",
            "generate-material",
            "edit-material",
            "summarize-session",
        }
        if any(thread.name in write_thread_names for thread in threading.enumerate()):
            return "问答任务仍在写入会话数据"
        return None

    def _source_failure(
        self, source: str, kind: str, start_ms: int, end_ms: int, message: str
    ) -> None:
        session_id = self.session_id
        if session_id is None:
            return
        end_ms = max(start_ms, end_ms)
        key = (source, kind, start_ms)
        try:
            with self._source_event_lock:
                if key in self._source_event_keys:
                    return
                event_id = self.database.record_source_event(
                    session_id, source, kind, start_ms, end_ms, message
                )
                self._source_event_keys.add(key)
                self._failed_sources.add(source)
        except Exception as exc:  # persistence boundary must not kill capture workers
            logger.exception("Could not persist source failure for %s", source)
            if self.on_error:
                try:
                    self.on_error(f"来源故障无法写入本地存储：{source}；{exc}")
                except Exception:
                    logger.exception("Source failure persistence error callback failed")
            return
        event = self.database.source_event(event_id)
        if event is not None and self.on_source_event:
            self.on_source_event(event)
        if self.on_error:
            self.on_error(
                f"来源故障：{source}，会话时间 {start_ms / 1000:.1f}–{end_ms / 1000:.1f} 秒；"
                f"{message}。确认数据缺失后才会写入时间线。"
            )

    def _record_transcription_metrics(self, audio_ms: int, processing_seconds: float) -> None:
        with self._transcription_metrics_lock:
            self._transcription_audio_ms += max(0, audio_ms)
            self._transcription_processing_seconds += max(0.0, processing_seconds)

    def runtime_metrics(self) -> RuntimeMetrics:
        session_id = self.session_id
        counts = (
            self.database.session_runtime_counts(session_id) if session_id is not None else None
        )
        global_pending_audio, global_failed_audio = self.database.recovery_audio_counts()
        duration_ms = self.recording_status().duration_ms if self.is_recording else 0
        storage_path = self.settings.data_dir
        if session_id is not None:
            storage_path = self.settings.data_dir / "sessions" / session_id
        with self._transcription_metrics_lock:
            audio_ms = self._transcription_audio_ms
            processing_seconds = self._transcription_processing_seconds
        if counts is not None:
            audio_ms = max(audio_ms, counts.transcribed_audio_ms)
        realtime_factor = (
            processing_seconds / (audio_ms / 1000)
            if audio_ms > 0 and processing_seconds > 0
            else None
        )
        correction_backlog = counts.correction_backlog if counts is not None else 0
        if self.correction_queue is not None:
            correction_backlog += self.correction_queue.qsize()
        resources = self._resource_sampler.sample()
        return RuntimeMetrics(
            session_id=session_id,
            duration_ms=duration_ms,
            frame_count=counts.frame_count if counts is not None else 0,
            storage_bytes=directory_size(storage_path),
            free_bytes=disk_free_bytes(self.settings.data_dir),
            transcribed_audio_ms=audio_ms,
            transcription_realtime_factor=realtime_factor,
            correction_backlog=correction_backlog,
            pending_audio_chunks=(
                counts.pending_audio_chunks if counts is not None else global_pending_audio
            ),
            failed_audio_chunks=(
                counts.failed_audio_chunks if counts is not None else global_failed_audio
            ),
            retryable_model_tasks=self.database.retryable_model_task_count(),
            cpu_percent=resources.cpu_percent,
            memory_used_bytes=resources.memory_used_bytes,
            memory_total_bytes=resources.memory_total_bytes,
            gpu_utilization_percent=resources.gpu_utilization_percent,
            gpu_memory_used_bytes=resources.gpu_memory_used_bytes,
            gpu_memory_total_bytes=resources.gpu_memory_total_bytes,
            sampled_at_utc=now_utc_iso(),
        )

    def recover_pending_audio(self, *, include_failed: bool = False) -> AudioRecoveryReport:
        with self._lifecycle_lock:
            if self.is_recording:
                raise RuntimeError("Cannot recover audio while a session is recording")
            if self.transcriber is not None and self.transcriber.is_alive():
                raise RuntimeError("Audio recovery is already running")
            self.database.reconcile_transcribed_audio_chunks()
            records = self.database.pending_audio_chunks(include_failed=include_failed)
            if not records:
                return AudioRecoveryReport(queued_chunks=0, missing_chunks=0)
            recovery_queue: queue.Queue[AudioChunk | None] = queue.Queue(
                maxsize=max(16, len(records) + 1)
            )
            missing = 0
            queued_chunks = 0
            for record in records:
                if not record.path.is_file():
                    self.database.set_chunk_state(
                        record.id, "failed", "音频文件不存在，无法恢复待转写任务"
                    )
                    missing += 1
                    continue
                recovery_queue.put(
                    AudioChunk(
                        id=record.id,
                        session_id=record.session_id,
                        source=record.source,
                        start_ms=record.start_ms,
                        end_ms=record.end_ms,
                        path=record.path,
                    )
                )
                queued_chunks += 1
            if recovery_queue.empty():
                return AudioRecoveryReport(queued_chunks=0, missing_chunks=missing)
            if any(
                self.database.transcript_correction_settings(record.session_id).enabled
                for record in records
                if record.path.is_file()
            ):
                self._start_correction_worker()
            recovery_queue.put(None)
            self.chunk_queue = recovery_queue
            self.transcriber = TranscriptionWorker(
                database=self.database,
                chunk_queue=recovery_queue,
                settings=self.actual_whisper_settings,
                model_dir=self.settings.model_dir,
                on_segment=self.on_segment,
                on_persisted_segment=self._enqueue_correction,
                on_recognition_started=(
                    (lambda start, end, source: self.on_segment(start, end, source, ""))
                    if self.on_segment
                    else None
                ),
                on_metrics=self._record_transcription_metrics,
                on_error=self.on_error,
            )
            self.transcriber.start()
            return AudioRecoveryReport(queued_chunks=queued_chunks, missing_chunks=missing)

    def retry_failed_audio(self) -> AudioRecoveryReport:
        with self._lifecycle_lock:
            if self.is_recording:
                raise RuntimeError("Cannot retry audio while a session is recording")
            if self.transcriber is not None and self.transcriber.is_alive():
                raise RuntimeError("Audio recovery is already running")
            self.database.retry_failed_audio_chunks()
            return self.recover_pending_audio()

    def whisper_model_in_use(self, repository_id: str) -> bool:
        active_repository = canonical_whisper_repository_id(self.actual_whisper_settings.model)
        if repository_id != active_repository:
            return False
        model_threads = {"whisper-model-download", "whisper-benchmark"}
        return (
            self.storage_busy_reason() is not None
            or self.question_transcriber is not None
            or any(thread.name in model_threads for thread in threading.enumerate())
        )

    @storage_writer("开始会话")
    def start(
        self,
        title: str,
        *,
        selection: RecordingSelection | None = None,
    ) -> str:
        with self._lifecycle_lock:
            return self._start_unlocked(title, selection=selection)

    def _start_unlocked(
        self,
        title: str,
        *,
        selection: RecordingSelection | None = None,
    ) -> str:
        if self.is_recording:
            raise RuntimeError("A session is already recording")
        busy_reason = self.storage_busy_reason()
        if busy_reason:
            raise RuntimeError(f"Cannot start a session while {busy_reason}")
        snapshot = self.device_catalog.snapshot()
        if selection is None:
            selection = RecordingSelection(
                display_ids=(),
                system_audio_id=(
                    next(
                        (item.id for item in snapshot.system_audio if item.is_default),
                        snapshot.system_audio[0].id if snapshot.system_audio else None,
                    )
                    if self.settings.capture_system_audio
                    else None
                ),
                microphone_id=(
                    next(
                        (item.id for item in snapshot.microphones if item.is_default),
                        snapshot.microphones[0].id if snapshot.microphones else None,
                    )
                    if self.settings.capture_microphone
                    else None
                ),
                estimated_duration_minutes=60,
            )
        resolved_selection = resolve_recording_selection(
            RecordingPreferences(
                display_ids=selection.display_ids,
                system_audio_id=selection.system_audio_id,
                microphone_id=selection.microphone_id,
                system_audio_enabled=selection.system_audio_id is not None,
                microphone_enabled=selection.microphone_id is not None,
                estimated_duration_minutes=selection.estimated_duration_minutes,
            ),
            snapshot,
        )
        self.recording_selection = resolved_selection
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
        self.pause_event.clear()
        self._pause_event_id = None
        self._failed_sources.clear()
        self._source_event_keys.clear()
        with self._transcription_metrics_lock:
            self._transcription_audio_ms = 0
            self._transcription_processing_seconds = 0.0
        resolved_whisper = resolve_whisper_settings(
            self.whisper_settings, self.whisper_capabilities
        )
        self.actual_whisper_settings = resolved_whisper.settings
        self.database.record_whisper_run(
            session_id=session_id,
            requested=self.whisper_settings,
            actual=self.actual_whisper_settings,
            fallback_advice=resolved_whisper.fallback_advice,
        )
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
                display=display,
                output_dir=session_dir / "frames" / f"display-{index:02d}",
                stop_event=stop_event,
                pause_event=self.pause_event,
                interval_s=self.settings.screen_interval_s,
                hash_distance=self.settings.screen_hash_distance,
                on_error=self.on_error,
                on_failure=self._source_failure,
            )
            for index, display in enumerate(resolved_selection.displays, start=1)
        ]
        for source, device in (
            ("system", resolved_selection.system_audio),
            ("microphone", resolved_selection.microphone),
        ):
            if device is not None:
                self.workers.append(
                    AudioCaptureWorker(
                        database=self.database,
                        session_id=session_id,
                        clock=clock,
                        source=source,
                        device=device,
                        device_catalog=self.device_catalog,
                        output_dir=session_dir / "audio",
                        stop_event=stop_event,
                        pause_event=self.pause_event,
                        chunk_queue=chunk_queue,
                        sample_rate=self.settings.audio_capture_rate,
                        storage_sample_rate=self.settings.audio_storage_rate,
                        chunk_s=self.settings.audio_chunk_s,
                        on_error=self.on_error,
                        on_failure=self._source_failure,
                    )
                )
        self.transcriber = TranscriptionWorker(
            database=self.database,
            chunk_queue=chunk_queue,
            settings=self.actual_whisper_settings,
            model_dir=self.settings.model_dir,
            on_segment=self.on_segment,
            on_persisted_segment=self._enqueue_correction,
            on_recognition_started=(
                (lambda start, end, source: self.on_segment(start, end, source, ""))
                if self.on_segment
                else None
            ),
            on_metrics=self._record_transcription_metrics,
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
                    settings = self.database.transcript_correction_settings(session_id)
                    window_ms = settings.window_ms
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
        if self.correction_queue is None:
            return
        for item in self.correction_batcher.add_segment(session_id, start_ms):
            self.correction_queue.put(item)

    @storage_writer("测试模型连接")
    def test_provider(self) -> str:
        result = self._model_router().invoke(
            RoleName.UTILITY,
            lambda model: model.test_connection(),
            session_id=self.session_id,
            task_type="provider_test",
        )
        return result.value

    @storage_writer("结束会话")
    def stop(self) -> str | None:
        with self._lifecycle_lock:
            return self._stop_unlocked()

    def _stop_unlocked(self) -> str | None:
        if self.is_paused:
            assert self.clock is not None
            self._close_pause(self.clock.now_ms())
        self.cancel_question()
        if not self.is_recording:
            return None
        assert self.stop_event is not None
        assert self.chunk_queue is not None
        session_id = self.session_id
        ended_at = datetime.now(UTC).isoformat()
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
        self.workers = [worker for worker in self.workers if worker.is_alive()]
        if transcriber_stopped:
            self.chunk_queue = None
            self.transcriber = None
        if correction_stopped:
            self.correction_queue = None
            self.correction_worker = None
            self.correction_batcher = CorrectionWindowBatcher(self.correction_window_seconds)
            self.correction_flush_event.clear()
        return session_id

    @storage_writer("暂停会话")
    def pause(self) -> bool:
        with self._lifecycle_lock:
            if not self.is_recording or self.is_paused:
                return False
            assert self.session_id is not None
            assert self.clock is not None
            start_ms = self.clock.now_ms()
            event_id = self.database.add_timeline_event(
                self.session_id,
                TimelineEventKind.PAUSE,
                None,
                start_ms,
                start_ms,
                "用户主动暂停，所有来源停止采集",
            )
            self._pause_event_id = event_id
            self.pause_event.set()
            return True

    @storage_writer("恢复会话")
    def resume(self) -> bool:
        with self._lifecycle_lock:
            if not self.is_recording or not self.is_paused:
                return False
            assert self.clock is not None
            self._close_pause(self.clock.now_ms())
            return True

    def _close_pause(self, end_ms: int) -> None:
        event_id = self._pause_event_id
        if event_id is not None and not self.database.finish_timeline_event(event_id, end_ms):
            raise RuntimeError(f"Unable to close pause timeline event {event_id}")
        self.pause_event.clear()
        self._pause_event_id = None

    def _question_service(self) -> QuestionAnsweringService:
        return QuestionAnsweringService(self.database, self._model_router())

    @storage_writer("创建问题锚点")
    def capture_question_anchor(self, lookback_ms: int = 2 * 60_000) -> int:
        if self.session_id is None or self.clock is None:
            raise RuntimeError("Start a study session before asking a question")
        if self.pending_question_id is None:
            self.pending_question_id = self._question_service().create_anchor(
                self.session_id, self.clock.now_ms(), lookback_ms=lookback_ms
            )
        return self.pending_question_id

    @storage_writer("修改问题范围")
    def set_question_range(self, lookback_ms: int) -> None:
        if self.pending_question_id is None:
            raise RuntimeError("There is no pending question anchor")
        self._question_service().set_anchor_range(self.pending_question_id, lookback_ms)

    @storage_writer("取消问题")
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

    @storage_writer("录制语音提问")
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

    @storage_writer("转写语音提问")
    def finish_question_voice(self) -> str:
        recorder = self.question_voice_recorder
        if recorder is None:
            raise RuntimeError("No question voice recording is active")
        self.question_voice_recorder = None
        path = recorder.stop()
        if self.question_transcriber is None:
            resolved = resolve_whisper_settings(self.whisper_settings, self.whisper_capabilities)
            self.actual_whisper_settings = resolved.settings
            self.question_transcriber = WhisperQuestionTranscriber(
                self.actual_whisper_settings, self.settings.model_dir
            )
        try:
            return self.question_transcriber.transcribe(path)
        finally:
            path.unlink(missing_ok=True)

    @storage_writer("提交问题")
    def answer(self, question: str) -> str:
        if self.session_id is None or self.clock is None:
            raise RuntimeError("Start a study session before asking a question")
        question_id = self.capture_question_anchor()
        self.pending_question_id = None
        self.last_question_id = question_id
        result = self._question_service().submit(question_id, question)
        assert result.answer is not None
        return result.answer

    @storage_writer("重新回答问题")
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

    def material_generation_mode(self) -> MaterialGenerationMode | None:
        return self._material_generation_mode

    @storage_writer("保存会话材料生成策略")
    def set_material_generation_mode(self, mode: MaterialGenerationMode) -> None:
        self.material_generation_settings_store.save(mode)
        self._material_generation_mode = mode

    def material_generation_preview(self, session_id: str) -> MaterialGenerationPreview:
        transcripts = self.database.all_effective_transcripts(session_id)
        role = self.model_role(RoleName.DEEP_ANALYSIS)
        connection = next(
            (item for item in self.provider_settings.connections if item.id == role.connection_id),
            None,
        )
        if connection is None:
            raise RuntimeError(f"Model connection is not configured: {role.connection_id}")
        return MaterialGenerationPreview(
            session_id=session_id,
            transcript_count=len(transcripts),
            character_count=sum(len(item.text) for item in transcripts),
            connection_name=connection.name,
            model=role.model,
            base_url=connection.base_url,
            reasoning_level=role.reasoning.value,
        )

    @storage_writer("生成会话材料")
    def generate_material(
        self, session_id: str | None = None, *, template_id: str | None = None
    ) -> SessionMaterialVersionRecord:
        target_session_id = session_id or self.session_id
        if target_session_id is None:
            raise RuntimeError("There is no session to generate material for")
        context = ContextAssembler(self.database).for_material(target_session_id)
        persistence_items = context.persistence_items()
        evidence = tuple(
            InvocationEvidence(
                stable_id=str(item["stable_id"]),
                kind=str(item["kind"]),
                source=str(item["source"]),
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                transcript_version_id=(
                    int(item["transcript_version_id"])
                    if item.get("transcript_version_id") is not None
                    else None
                ),
                frame_id=(int(item["frame_id"]) if item.get("frame_id") is not None else None),
            )
            for item in persistence_items
        )

        def generate(model):  # type: ignore[no-untyped-def]
            method = getattr(model, "generate_material", None)
            if callable(method):
                return method(context, template_id=template_id)
            legacy = model.summarize(context.transcript)
            if isinstance(legacy, MaterialModelResult):
                return legacy
            if isinstance(legacy, str):
                return MaterialModelResult(legacy)
            if isinstance(legacy, Mapping):
                return MaterialModelResult(legacy_summary_to_markdown(legacy))
            raise TypeError("Material model returned an unsupported result")

        routed = self._model_router().invoke(
            RoleName.DEEP_ANALYSIS,
            generate,
            session_id=target_session_id,
            evidence=evidence,
            task_type="material",
            task_payload_json=json.dumps(
                {
                    "session_id": target_session_id,
                    "template_id": template_id,
                    "evidence": persistence_items,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        result = routed.value
        if isinstance(result, MaterialModelResult):
            content = result.content
            request_id = result.request_id
            model_name = result.model or routed.invocation.model
        elif isinstance(result, str):
            content = result
            request_id = None
            model_name = routed.invocation.model
        else:
            raise TypeError("Material model returned an unsupported result")
        self.database.resolve_retryable_model_tasks("material", target_session_id)
        return self.database.record_material_version(
            target_session_id,
            kind="generated",
            content=content,
            template_id=template_id,
            model=model_name,
            connection_json=invocation_connection_json(routed.invocation),
            model_invocation_id=routed.invocation.id,
            request_status="succeeded",
            request_id=request_id,
            error=None,
            evidence_state="exact",
            evidence=persistence_items,
        )

    @storage_writer("编辑会话材料")
    def edit_material(self, material_version_id: int, content: str) -> SessionMaterialVersionRecord:
        return self.database.record_material_edit(material_version_id, content)

    def material_versions(self, session_id: str) -> list[SessionMaterialVersionRecord]:
        return self.database.session_material_versions(session_id)

    def cross_session_search(
        self, query: str, *, limit: int = 50
    ) -> list[CrossSessionSearchResult]:
        return self.database.cross_session_search(query, limit=limit)

    def cross_session_evidence_candidates(
        self, stable_ids: tuple[str, ...]
    ) -> list[CrossSessionEvidenceRecord]:
        return self.database.cross_session_evidence_candidates(stable_ids)

    def failed_cross_session_syntheses(
        self, *, limit: int = 10
    ) -> tuple[CrossSessionSynthesisRecord, ...]:
        return self.database.cross_session_syntheses(limit=limit, request_status="failed")

    def cross_session_synthesis_preview(
        self, question: str, stable_ids: tuple[str, ...]
    ) -> CrossSessionSynthesisPreview:
        return CrossSessionSynthesisService(self.database, self._model_router()).preview(
            question, stable_ids
        )

    @storage_writer("执行跨会话综合")
    def synthesize_cross_session(
        self, question: str, stable_ids: tuple[str, ...]
    ) -> CrossSessionSynthesisRecord:
        return CrossSessionSynthesisService(self.database, self._model_router()).synthesize(
            question, stable_ids
        )

    @storage_writer("重试跨会话综合")
    def retry_cross_session_synthesis(self, synthesis_id: int) -> CrossSessionSynthesisRecord:
        return CrossSessionSynthesisService(self.database, self._model_router()).retry(synthesis_id)

    @storage_writer("生成会话材料")
    def summarize(self) -> str:
        """Compatibility entry point retained for callers of the old summary action."""
        return self.generate_material().content
