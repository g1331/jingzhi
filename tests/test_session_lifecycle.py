from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from jingzhi.application import JingzhiApplicationService
from jingzhi.database import Database


class IdleRecorder:
    session_id: str | None = None
    is_recording = False


def _service(database: Database, now: list[datetime]) -> JingzhiApplicationService:
    return JingzhiApplicationService(database, recorder=IdleRecorder(), now=lambda: now[0])


def _completed_session(database: Database, title: str, ended_at: datetime) -> str:
    session_id = database.create_session(title, (ended_at - timedelta(hours=1)).isoformat())
    database.finish_session(session_id, ended_at.isoformat(), "complete")
    return session_id


def test_session_retention_notifications_pin_restore_and_restart_are_idempotent(
    tmp_path: Path,
) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    now = [ended_at + timedelta(days=23)]
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = _completed_session(database, "保留策略", ended_at)
    service = _service(database, now)

    assert [notice.kind for notice in service.run_session_maintenance()] == ["retention_7d"]
    assert service.run_session_maintenance() == ()
    assert _service(Database(database.path), now).run_session_maintenance() == ()

    service.pin_session(session_id, True)
    now[0] = ended_at + timedelta(days=60)
    assert service.run_session_maintenance() == ()
    assert service.list_sessions()[0].pinned is True

    service.pin_session(session_id, False)
    unpinned_at = now[0]
    now[0] = unpinned_at + timedelta(days=29)
    assert [notice.kind for notice in service.run_session_maintenance()] == ["retention_1d"]
    now[0] = unpinned_at + timedelta(days=30)
    assert [notice.kind for notice in service.run_session_maintenance()] == ["moved_to_trash"]
    assert service.list_sessions(status="trash")[0].id == session_id

    service.restore_session(session_id)
    restored = service.list_sessions()
    assert [item.id for item in restored] == [session_id]
    assert restored[0].trashed_at_utc is None
    now[0] += timedelta(days=29)
    assert service.run_session_maintenance()
    now[0] += timedelta(days=1)
    service.run_session_maintenance()
    assert service.list_sessions(status="trash")[0].id == session_id


def test_final_delete_removes_database_fts_and_session_media(tmp_path: Path) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    now = [ended_at + timedelta(days=30)]
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = _completed_session(database, "最终删除", ended_at)
    media_dir = tmp_path / "sessions" / session_id
    frame_path = media_dir / "frames" / "frame.webp"
    audio_path = media_dir / "audio" / "chunk.flac"
    frame_path.parent.mkdir(parents=True)
    audio_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"frame")
    audio_path.write_bytes(b"audio")
    database.add_frame(session_id, 1_000, frame_path, "hash", (10, 10))
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 2_000, audio_path)
    database.add_transcript(
        session_id,
        chunk_id,
        "system",
        0,
        2_000,
        "必须从全文索引删除",
        "zh",
        1.0,
    )
    service = _service(database, now)

    service.run_session_maintenance()
    now[0] += timedelta(days=7)
    notices = service.run_session_maintenance()

    assert [notice.kind for notice in notices] == ["permanently_deleted"]
    assert database.get_session(session_id) is None
    assert not media_dir.exists()
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM transcript_fts WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM answer_evidence WHERE resource_path LIKE ?",
                (f"%{session_id}%",),
            ).fetchone()[0]
            == 0
        )


def test_failed_media_deletion_keeps_database_consistent_for_retry(
    tmp_path: Path, monkeypatch
) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    now = [ended_at + timedelta(days=30)]
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = _completed_session(database, "删除失败", ended_at)
    media_dir = tmp_path / "sessions" / session_id
    media_dir.mkdir(parents=True)
    (media_dir / "frame.webp").write_bytes(b"frame")
    service = _service(database, now)
    service.run_session_maintenance()
    now[0] += timedelta(days=7)
    monkeypatch.setattr(
        "jingzhi.database.shutil.rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("媒体正在使用")),
    )

    assert [notice.kind for notice in service.run_session_maintenance()] == ["final_delete_failed"]
    assert database.get_session(session_id) is None
    assert database.pending_media_deletions()
    assert not media_dir.exists()
    monkeypatch.undo()

    assert [notice.kind for notice in service.run_session_maintenance()] == ["permanently_deleted"]
    assert not database.pending_media_deletions()


def test_session_library_search_filter_sort_and_interrupted_recovery(tmp_path: Path) -> None:
    now = [datetime(2026, 8, 4, 12, tzinfo=UTC)]
    database = Database(tmp_path / "jingzhi.sqlite3")
    older_id = _completed_session(database, "旧会议", now[0] - timedelta(days=2))
    newer_id = _completed_session(database, "项目复盘", now[0] - timedelta(days=1))
    chunk_id = database.add_audio_chunk(newer_id, "microphone", 0, 1_000, tmp_path / "search.flac")
    segment_id = database.add_transcript(
        newer_id, chunk_id, "microphone", 0, 1_000, "讨论季度路线图", "zh", 1.0
    )
    database.add_transcript_version(segment_id, "correction", "讨论校订术语")
    interrupted_id = database.create_session("崩溃前会话", now[0].isoformat())
    service = _service(database, now)

    assert [item.id for item in service.list_sessions()] == [
        interrupted_id,
        newer_id,
        older_id,
    ]
    assert service.list_sessions()[0].status == "interrupted"
    assert [item.id for item in service.list_sessions(query="校订术语")] == [newer_id]
    assert service.list_sessions(query="季度路线图") == []
    assert [item.id for item in service.list_sessions(status="complete", newest_first=False)] == [
        older_id,
        newer_id,
    ]

    service.complete_interrupted_session(interrupted_id)
    assert database.get_session(interrupted_id).status == "complete"  # type: ignore[union-attr]


def test_stopped_session_is_not_treated_as_current_and_can_be_deleted(tmp_path: Path) -> None:
    now = [datetime(2026, 8, 4, 12, tzinfo=UTC)]
    database = Database(tmp_path / "stale-current.sqlite3")
    session_id = _completed_session(database, "刚结束", now[0] - timedelta(hours=1))
    recorder = IdleRecorder()
    recorder.session_id = session_id
    service = JingzhiApplicationService(database, recorder=recorder, now=lambda: now[0])

    assert service.list_sessions()[0].id == session_id
    service.delete_session(session_id)
    assert service.list_sessions(status="trash")[0].id == session_id
