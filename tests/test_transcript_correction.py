from __future__ import annotations

import sqlite3
from pathlib import Path

from PIL import Image

from jingzhi.application import JingzhiApplicationService
from jingzhi.database import Database
from jingzhi.transcript_correction import CorrectionRequest, CorrectionWindowBatcher


class NoHardwareRecorder:
    is_recording = False

    def start(self, title: str, **_kwargs) -> str:
        raise AssertionError(f"Unexpected hardware start for {title}")

    def stop(self) -> None:
        return None


class FakeCorrectionModel:
    model = "correction-small"

    def __init__(self, replacements: dict[int, str] | None = None) -> None:
        self.replacements = replacements or {}
        self.requests: list[CorrectionRequest] = []

    def correct(self, request: CorrectionRequest) -> dict[int, str]:
        self.requests.append(request)
        return {
            segment.id: self.replacements.get(segment.id, segment.text)
            for segment in request.target_segments
        }


def _add_segment(
    database: Database,
    session_id: str,
    tmp_path: Path,
    *,
    start_ms: int,
    end_ms: int,
    text: str,
) -> int:
    chunk_id = database.add_audio_chunk(
        session_id,
        "system",
        start_ms,
        end_ms,
        tmp_path / f"audio-{start_ms}.wav",
    )
    segment_id = database.add_transcript(
        session_id,
        chunk_id,
        "system",
        start_ms,
        end_ms,
        text,
        "zh",
        -0.1,
    )
    database.set_chunk_state(chunk_id, "transcribed")
    return segment_id


def test_original_transcript_is_saved_as_immutable_first_version_and_hidden_state_by_default(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("字幕会话", "2026-08-03T00:00:00+00:00")
    segment_id = _add_segment(
        database,
        session_id,
        tmp_path,
        start_ms=1_000,
        end_ms=2_000,
        text="原始字幕",
    )
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())

    transcript = service.open_session(session_id).transcripts[0]
    versions = service.transcript_versions(segment_id)

    assert transcript.text == "原始字幕"
    assert transcript.version_kind == "original"
    assert transcript.correction_state is None
    assert [(version.kind, version.text) for version in versions] == [("original", "原始字幕")]


def test_recognition_state_only_appears_when_correction_is_enabled(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("识别状态", "2026-08-03T00:00:00+00:00")
    database.add_audio_chunk(session_id, "system", 1_000, 9_000, tmp_path / "pending.wav")
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())

    assert service.open_session(session_id).transcripts == ()

    service.configure_transcript_correction(session_id, enabled=True, window_seconds=30)

    recognizing = service.open_session(session_id).transcripts
    assert [(item.correction_state, item.text) for item in recognizing] == [
        ("recognizing", "正在识别…")
    ]


def test_correction_uses_window_neighbors_and_source_labeled_representative_frames(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("校订会话", "2026-08-03T00:00:00+00:00")
    neighbor_id = _add_segment(
        database,
        session_id,
        tmp_path,
        start_ms=1_000,
        end_ms=2_000,
        text="上一句提到单纯形法",
    )
    target_id = _add_segment(
        database,
        session_id,
        tmp_path,
        start_ms=16_000,
        end_ms=18_000,
        text="换入便量",
    )
    frame_path = tmp_path / "display-2.webp"
    Image.new("RGB", (320, 180), "navy").save(frame_path)
    database.add_frame(
        session_id,
        17_000,
        frame_path,
        "hash",
        (320, 180),
        source_id="display:2",
    )
    model = FakeCorrectionModel({target_id: "换入变量"})
    service = JingzhiApplicationService(
        database,
        recorder=NoHardwareRecorder(),
        correction_model=model,
    )
    service.configure_transcript_correction(session_id, enabled=True, window_seconds=15)

    before = service.open_session(session_id).transcripts
    result = service.run_transcript_correction(session_id, window_start_ms=15_000)
    after = service.open_session(session_id).transcripts

    assert [item.correction_state for item in before] == ["pending", "pending"]
    assert result.state == "corrected"
    assert [item.text for item in after] == ["上一句提到单纯形法", "换入变量"]
    assert after[1].version_kind == "correction"
    request = model.requests[0]
    assert [item.id for item in request.target_segments] == [target_id]
    assert neighbor_id in {item.id for item in request.context_segments}
    assert [(frame.source_id, frame.ts_ms, frame.path) for frame in request.frames] == [
        ("display:2", 17_000, frame_path)
    ]


def test_user_edit_wins_and_undo_restores_original_without_deleting_history(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("版本会话", "2026-08-03T00:00:00+00:00")
    segment_id = _add_segment(
        database,
        session_id,
        tmp_path,
        start_ms=2_000,
        end_ms=4_000,
        text="元始字幕",
    )
    model = FakeCorrectionModel({segment_id: "原始字幕"})
    service = JingzhiApplicationService(
        database,
        recorder=NoHardwareRecorder(),
        correction_model=model,
    )
    service.configure_transcript_correction(session_id, enabled=True, window_seconds=30)
    service.run_transcript_correction(session_id, window_start_ms=0)

    service.edit_transcript(segment_id, "用户确认的字幕")
    model.replacements[segment_id] = "后续自动校订"
    service.run_transcript_correction(session_id, window_start_ms=0)

    edited = service.open_session(session_id).transcripts[0]
    assert edited.text == "用户确认的字幕"
    assert edited.version_kind == "user_edit"
    assert [version.kind for version in service.transcript_versions(segment_id)] == [
        "original",
        "correction",
        "user_edit",
    ]
    assert len(model.requests) == 1

    service.undo_transcript_correction(segment_id)
    still_edited = service.open_session(session_id).transcripts[0]
    assert still_edited.text == "用户确认的字幕"


def test_correction_failure_falls_back_to_original_and_records_source(tmp_path: Path) -> None:
    class FailingModel(FakeCorrectionModel):
        def correct(self, request: CorrectionRequest) -> dict[int, str]:
            self.requests.append(request)
            raise RuntimeError("provider unavailable")

    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("失败回退", "2026-08-03T00:00:00+00:00")
    _add_segment(
        database,
        session_id,
        tmp_path,
        start_ms=1_000,
        end_ms=2_000,
        text="始终可见的原文",
    )
    service = JingzhiApplicationService(
        database,
        recorder=NoHardwareRecorder(),
        correction_model=FailingModel(),
    )
    service.configure_transcript_correction(session_id, enabled=True, window_seconds=60)

    result = service.run_transcript_correction(session_id, window_start_ms=0)
    transcript = service.open_session(session_id).transcripts[0]

    assert result.state == "failed"
    assert result.error_source == "correction-small"
    assert "provider unavailable" in (result.error or "")
    assert transcript.text == "始终可见的原文"
    assert transcript.version_kind == "original"


def test_incomplete_correction_result_is_recorded_as_failure(tmp_path: Path) -> None:
    class IncompleteModel(FakeCorrectionModel):
        def correct(self, request: CorrectionRequest) -> dict[int, str]:
            self.requests.append(request)
            return {request.target_segments[0].id: "只返回一句"}

    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("不完整校订", "2026-08-03T00:00:00+00:00")
    _add_segment(
        database,
        session_id,
        tmp_path,
        start_ms=1_000,
        end_ms=2_000,
        text="第一句",
    )
    _add_segment(
        database,
        session_id,
        tmp_path,
        start_ms=3_000,
        end_ms=4_000,
        text="第二句",
    )
    model = IncompleteModel()
    service = JingzhiApplicationService(
        database, recorder=NoHardwareRecorder(), correction_model=model
    )
    service.configure_transcript_correction(session_id, enabled=True, window_seconds=15)

    result = service.run_transcript_correction(session_id, window_start_ms=0)

    assert result.state == "failed"
    assert "missing=" in (result.error or "")
    assert [item.version_kind for item in service.open_session(session_id).transcripts] == [
        "original",
        "original",
    ]


def test_window_batcher_emits_each_window_once() -> None:
    batcher = CorrectionWindowBatcher(15)

    assert batcher.add_segment("session", 1_000) == (("session", 0),)
    assert batcher.add_segment("session", 12_000) == ()
    assert batcher.add_segment("session", 16_000) == (("session", 15_000),)
    assert batcher.add_segment("session", 18_000) == ()


def test_representative_frames_include_latest_per_source_and_at_most_two_changes(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("关键帧选择", "2026-08-03T00:00:00+00:00")
    frame_ids: list[int] = []
    for index, image_hash in enumerate(("0" * 64, "f" * 64, "0" * 64, "1" * 64)):
        path = tmp_path / f"frame-{index}.webp"
        Image.new("RGB", (16, 16), "white").save(path)
        frame_ids.append(
            database.add_frame(
                session_id,
                1_000 + index * 1_000,
                path,
                image_hash,
                (16, 16),
                source_id="display:1",
            )
        )
    second_source = tmp_path / "display-2.webp"
    Image.new("RGB", (16, 16), "navy").save(second_source)
    display_2_id = database.add_frame(
        session_id,
        2_500,
        second_source,
        "a" * 64,
        (16, 16),
        source_id="display:2",
    )

    selected = database.representative_frames(session_id, 0, 10_000)

    selected_ids = {item.id for item in selected}
    assert frame_ids[-1] in selected_ids
    assert display_2_id in selected_ids
    assert len(selected) == 4  # two latest-source frames plus at most two high-change frames


def test_legacy_transcript_rows_are_migrated_to_original_versions(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    database = Database(path)
    session_id = database.create_session("旧字幕", "2026-08-03T00:00:00+00:00")
    segment_id = _add_segment(
        database,
        session_id,
        tmp_path,
        start_ms=1_000,
        end_ms=2_000,
        text="旧数据",
    )
    with database.connect() as connection:
        connection.execute("DROP TABLE transcript_versions")
        connection.execute("DELETE FROM schema_migrations WHERE version > 2")
        connection.execute("PRAGMA user_version = 2")

    migrated = Database(path)

    assert [(item.kind, item.text) for item in migrated.transcript_versions(segment_id)] == [
        ("original", "旧数据")
    ]
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] >= 3
