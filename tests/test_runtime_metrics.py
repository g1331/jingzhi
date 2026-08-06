from __future__ import annotations

import queue
import sqlite3
from pathlib import Path
from typing import ClassVar

from jingzhi.application import JingzhiApplicationService
from jingzhi.capture.devices import DeviceSnapshot
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.diagnostics import ResourceMetrics, format_bytes, format_runtime_metrics
from jingzhi.session import SessionManager
from jingzhi.whisper_settings import WhisperCapabilities


class EmptyDeviceCatalog:
    def snapshot(self) -> DeviceSnapshot:
        return DeviceSnapshot((), (), ())

    def microphone_level(self, _device) -> float:
        return 0.0

    def audio_locator(self, _identifier: str):
        raise LookupError("No audio devices")


class CapturingTranscriptionWorker:
    instances: ClassVar[list[CapturingTranscriptionWorker]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return False


def _manager(tmp_path: Path) -> SessionManager:
    return SessionManager(
        Settings(data_dir=tmp_path),
        device_catalog=EmptyDeviceCatalog(),
        whisper_capabilities=WhisperCapabilities(
            devices=("cpu",), compute_types={"cpu": ("int8", "float32")}
        ),
    )


def test_session_runtime_counts_include_frames_audio_transcripts_and_correction_backlog(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = database.create_session("指标", "2026-01-01T00:00:00+00:00")
    frame = tmp_path / "frame.webp"
    frame.write_bytes(b"frame")
    database.add_frame(session_id, 1_000, frame, "hash", (100, 100))
    transcribed = tmp_path / "transcribed.wav"
    transcribed.write_bytes(b"audio")
    transcribed_id = database.add_audio_chunk(session_id, "system", 0, 2_000, transcribed)
    database.set_chunk_state(transcribed_id, "transcribed")
    database.add_transcript(session_id, transcribed_id, "system", 0, 2_000, "字幕", "zh", 0.9)
    pending = tmp_path / "pending.wav"
    pending.write_bytes(b"pending")
    database.add_audio_chunk(session_id, "microphone", 2_000, 4_000, pending)
    failed = tmp_path / "failed.wav"
    failed.write_bytes(b"failed")
    failed_id = database.add_audio_chunk(session_id, "system", 4_000, 6_000, failed)
    database.set_chunk_state(failed_id, "failed", "transcriber failed")
    database.start_correction_run(session_id, 0, 2_000, "correction-model")

    counts = database.session_runtime_counts(session_id)

    assert counts.frame_count == 1
    assert counts.transcribed_audio_ms == 2_000
    assert counts.transcript_count == 1
    assert counts.correction_backlog == 1
    assert counts.pending_audio_chunks == 1
    assert counts.failed_audio_chunks == 1


def test_pending_audio_recovery_queues_existing_chunks_and_marks_missing_files_failed(
    tmp_path: Path, monkeypatch
) -> None:
    CapturingTranscriptionWorker.instances.clear()
    monkeypatch.setattr("jingzhi.session.TranscriptionWorker", CapturingTranscriptionWorker)
    manager = _manager(tmp_path)
    session_id = manager.database.create_session("恢复", "2026-01-01T00:00:00+00:00")
    audio = tmp_path / "recover.wav"
    audio.write_bytes(b"audio")
    manager.database.add_audio_chunk(session_id, "system", 0, 2_000, audio)
    missing_id = manager.database.add_audio_chunk(
        session_id, "microphone", 2_000, 4_000, tmp_path / "missing.wav"
    )

    report = manager.recover_pending_audio()

    assert report.queued_chunks == 1
    assert report.missing_chunks == 1
    worker = CapturingTranscriptionWorker.instances[0]
    assert worker.started
    recovered = worker.kwargs["chunk_queue"]
    assert isinstance(recovered, queue.Queue)
    first = recovered.get_nowait()
    assert first is not None and first.id != missing_id
    assert recovered.get_nowait() is None
    assert manager.database.audio_chunk_state(missing_id) == "failed"


def test_v15_database_migrates_model_task_columns_before_normalization(tmp_path: Path) -> None:
    path = tmp_path / "v15.sqlite3"
    Database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE model_invocations DROP COLUMN task_type")
        connection.execute("ALTER TABLE model_invocations DROP COLUMN task_payload_json")
        connection.execute("ALTER TABLE model_invocations DROP COLUMN retryable")
        connection.execute("DELETE FROM schema_migrations WHERE version > 15")
        connection.execute("PRAGMA user_version = 15")

    migrated = Database(path)
    with migrated.connect() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(model_invocations)")
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert {"retryable", "task_type", "task_payload_json"} <= columns
    assert version == 17


def test_runtime_metrics_format_unavailable_gpu_explicitly(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager._resource_sampler.sample = lambda: ResourceMetrics(  # type: ignore[method-assign]
        cpu_percent=23.0,
        memory_used_bytes=1024 * 1024,
        memory_total_bytes=4 * 1024 * 1024,
        gpu_utilization_percent=None,
        gpu_memory_used_bytes=None,
        gpu_memory_total_bytes=None,
    )

    metrics = manager.runtime_metrics()
    text = format_runtime_metrics(metrics)

    assert format_bytes(1024 * 1024) == "1.0MB"
    assert "GPU 未检测" in text
    assert "CPU 23%" in text
    assert metrics.free_bytes >= 0


def test_application_start_marks_running_model_tasks_retryable(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    invocation_id = manager.database.start_model_invocation(
        session_id=None,
        role="utility",
        connection_id="connection",
        connection_name="连接",
        base_url="https://example.test/v1",
        api_mode="responses",
        model="utility-model",
        reasoning_level="fast",
        fallback_reason=None,
    )

    JingzhiApplicationService(manager.database, recorder=manager)

    assert manager.database.model_invocation(invocation_id).status == "failed"
    assert manager.database.retryable_model_task_count() == 0


def test_restart_materializes_interrupted_answer_task_for_retry(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session_id = manager.database.create_session("回答恢复", "2026-01-01T00:00:00+00:00")
    question_id = manager.database.create_question(session_id, 1_000, "刚才说了什么？", 0, 2_000)
    invocation_id = manager.database.start_model_invocation(
        session_id=session_id,
        role="instant_answer",
        connection_id="connection",
        connection_name="连接",
        base_url="https://example.test/v1",
        api_mode="responses",
        model="answer-model",
        reasoning_level="balanced",
        fallback_reason=None,
        task_type="answer",
        task_payload_json=(
            f'{{"question_id": {question_id}, "evidence": '
            '[{"stable_id": "transcript:original", "kind": "transcript", '
            '"source": "microphone", "start_ms": 0, "end_ms": 1000, '
            '"transcript_version_id": null, "content_text": "原始证据"}]}'
        ),
    )

    JingzhiApplicationService(manager.database, recorder=manager)

    versions = manager.database.answer_versions(question_id)
    assert len(versions) == 1
    assert versions[0].request_status == "failed"
    assert versions[0].model_invocation_id == invocation_id
    assert manager.database.answer_evidence(versions[0].id)[0].stable_id == "transcript:original"
    assert manager.database.retryable_model_task_count() == 1


def test_failed_audio_chunks_can_be_requeued_for_manual_retry(tmp_path: Path, monkeypatch) -> None:
    CapturingTranscriptionWorker.instances.clear()
    monkeypatch.setattr("jingzhi.session.TranscriptionWorker", CapturingTranscriptionWorker)
    manager = _manager(tmp_path)
    session_id = manager.database.create_session("重试", "2026-01-01T00:00:00+00:00")
    audio = tmp_path / "retry.wav"
    audio.write_bytes(b"audio")
    chunk_id = manager.database.add_audio_chunk(session_id, "system", 0, 2_000, audio)
    manager.database.set_chunk_state(chunk_id, "failed", "temporary")

    report = manager.retry_failed_audio()

    assert report.queued_chunks == 1
    assert manager.database.audio_chunk_state(chunk_id) == "pending"
