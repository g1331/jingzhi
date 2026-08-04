from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jingzhi.application import JingzhiApplicationService
from jingzhi.database import Database, ModelInvocationEvidenceRecord
from jingzhi.whisper_settings import WhisperSettings


class IdleRecorder:
    session_id: str | None = None
    is_recording = False


class ActiveRecorder(IdleRecorder):
    is_recording = True


class BusyRecorder(IdleRecorder):
    def storage_busy_reason(self) -> str:
        return "会话采集线程仍在写入数据"


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
    restarted = _service(Database(database.path), now)
    assert restarted.list_sessions(status="trash")[0].id == session_id
    service = restarted

    service.restore_session(session_id)
    restored = service.list_sessions()
    assert [item.id for item in restored] == [session_id]
    assert restored[0].trashed_at_utc is None
    now[0] += timedelta(days=29)
    assert service.run_session_maintenance()
    now[0] += timedelta(days=1)
    service.run_session_maintenance()
    assert service.list_sessions(status="trash")[0].id == session_id


def test_retention_notice_skips_session_pinned_after_snapshot(tmp_path: Path, monkeypatch) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    now = [ended_at + timedelta(days=23)]
    database = Database(tmp_path / "retention-notice-race.sqlite3")
    session_id = _completed_session(database, "通知竞态", ended_at)
    service = _service(database, now)
    original_record = database.record_session_notification

    def pin_then_record(candidate_id: str, kind, notified_at: str, **kwargs) -> bool:
        assert database.set_session_pinned(candidate_id, now[0].isoformat(), now[0].isoformat())
        return original_record(candidate_id, kind, notified_at, **kwargs)

    monkeypatch.setattr(database, "record_session_notification", pin_then_record)
    assert service.run_session_maintenance() == ()
    pinned = database.get_session(session_id)
    assert pinned is not None
    assert pinned.pinned is True


def test_moved_notice_skips_session_restored_after_move(tmp_path: Path, monkeypatch) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    now = [ended_at + timedelta(days=30)]
    database = Database(tmp_path / "moved-notice-race.sqlite3")
    session_id = _completed_session(database, "回收通知竞态", ended_at)
    service = _service(database, now)
    original_record = database.record_session_notification

    def restore_then_record(candidate_id: str, kind, notified_at: str, **kwargs) -> bool:
        if kind.value == "moved_to_trash":
            assert database.restore_session(candidate_id, now[0].isoformat())
        return original_record(candidate_id, kind, notified_at, **kwargs)

    monkeypatch.setattr(database, "record_session_notification", restore_then_record)
    assert service.run_session_maintenance() == ()
    restored = database.get_session(session_id)
    assert restored is not None
    assert restored.trashed_at_utc is None
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM session_notifications WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            == 0
        )


def test_final_delete_failure_notice_skips_session_restored_after_failure(
    tmp_path: Path, monkeypatch
) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    now = [ended_at + timedelta(days=37)]
    database = Database(tmp_path / "delete-failure-notice-race.sqlite3")
    session_id = _completed_session(database, "失败通知竞态", ended_at)
    assert database.move_session_to_trash(
        session_id,
        (ended_at + timedelta(days=30)).isoformat(),
        (ended_at + timedelta(days=37)).isoformat(),
    )
    service = _service(database, now)

    def restore_then_fail(candidate_id: str) -> bool:
        assert database.restore_session(candidate_id, (ended_at + timedelta(days=36)).isoformat())
        raise OSError("媒体正在使用")

    monkeypatch.setattr(database, "permanently_delete_session", restore_then_fail)
    assert service.run_session_maintenance() == ()
    restored = database.get_session(session_id)
    assert restored is not None
    assert restored.trashed_at_utc is None


def test_restore_rejects_expired_trash(tmp_path: Path) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    now = [ended_at + timedelta(days=30)]
    database = Database(tmp_path / "expired-trash.sqlite3")
    session_id = _completed_session(database, "过期回收", ended_at)
    service = _service(database, now)

    service.delete_session(session_id)
    now[0] += timedelta(days=7)

    with pytest.raises(KeyError, match="Unknown trashed session"):
        service.restore_session(session_id)
    assert service.list_sessions(status="trash")[0].id == session_id


def test_final_delete_preserves_session_restored_during_delete(tmp_path: Path, monkeypatch) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    database = Database(tmp_path / "restore-race.sqlite3")
    session_id = _completed_session(database, "恢复竞态", ended_at)
    media_dir = tmp_path / "sessions" / session_id
    media_dir.mkdir(parents=True)
    (media_dir / "frame.webp").write_bytes(b"frame")
    assert database.move_session_to_trash(
        session_id,
        (ended_at + timedelta(days=30)).isoformat(),
        (ended_at + timedelta(days=37)).isoformat(),
    )

    original_get_session = database.get_session

    def get_then_restore(candidate_id: str):
        session = original_get_session(candidate_id)
        if session is not None:
            database.restore_session(candidate_id, (ended_at + timedelta(days=31)).isoformat())
        return session

    monkeypatch.setattr(database, "get_session", get_then_restore)
    assert database.permanently_delete_session(session_id) is False

    reopened = Database(database.path)
    restored = reopened.get_session(session_id)
    assert restored is not None
    assert restored.trashed_at_utc is None
    assert media_dir.exists()


def test_maintenance_does_not_trash_session_completed_after_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    ended_at = datetime(2026, 8, 1, 9, tzinfo=UTC)
    now = [ended_at + timedelta(days=30)]
    database = Database(tmp_path / "completion-race.sqlite3")
    session_id = database.create_session("中断竞态", (ended_at - timedelta(hours=1)).isoformat())
    database.finish_session(session_id, ended_at.isoformat(), "interrupted")
    service = _service(database, now)
    original_move = database.move_session_to_trash

    def complete_then_move(candidate_id: str, trashed_at: str, expires_at: str, **kwargs) -> bool:
        assert database.complete_interrupted_session(candidate_id, now[0].isoformat())
        return original_move(candidate_id, trashed_at, expires_at, **kwargs)

    monkeypatch.setattr(database, "move_session_to_trash", complete_then_move)
    assert service.run_session_maintenance() == ()
    completed = database.get_session(session_id)
    assert completed is not None
    assert completed.status == "complete"
    assert completed.trashed_at_utc is None


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
    frame_id = database.add_frame(session_id, 1_000, frame_path, "hash", (10, 10))
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 2_000, audio_path)
    segment_id = database.add_transcript(
        session_id,
        chunk_id,
        "system",
        0,
        2_000,
        "必须从全文索引删除",
        "zh",
        1.0,
    )
    correction_version_id = database.add_transcript_version(
        segment_id, "correction", "必须从全文索引删除（校订）"
    )
    assert correction_version_id is not None
    question_id = database.create_question(session_id, 1_500, "为什么要删除？", 0, 2_000)
    database.record_answer_version(
        question_id,
        model="test-model",
        connection_json="{}",
        request_status="succeeded",
        request_id=None,
        answer="为了验证完整级联删除。",
        error=None,
        evidence_state="exact",
        evidence=[
            {
                "stable_id": "frame:1",
                "kind": "frame",
                "source": "display:primary",
                "start_ms": 1_000,
                "end_ms": 1_000,
                "frame_id": frame_id,
                "content_text": "frame",
                "resource_path": str(frame_path),
            },
            {
                "stable_id": f"transcript:{correction_version_id}",
                "kind": "transcript",
                "source": "system",
                "start_ms": 0,
                "end_ms": 2_000,
                "transcript_version_id": correction_version_id,
                "content_text": "必须从全文索引删除（校订）",
            },
        ],
    )
    database.configure_transcript_correction(session_id, enabled=True, window_ms=30_000)
    correction_run_id = database.start_correction_run(session_id, 0, 2_000, "test-model")
    database.finish_correction_run(correction_run_id, "corrected")
    database.record_whisper_run(
        session_id=session_id,
        requested=WhisperSettings(),
        actual=WhisperSettings(),
        fallback_advice="",
    )
    invocation_id = database.start_model_invocation(
        session_id=session_id,
        role="instant_answer",
        connection_id="test-connection",
        connection_name="测试连接",
        base_url="https://example.test",
        api_mode="responses",
        model="test-model",
        reasoning_level="fast",
        fallback_reason=None,
        evidence=(
            ModelInvocationEvidenceRecord(
                stable_id="frame:1",
                kind="frame",
                source="display:primary",
                start_ms=1_000,
                end_ms=1_000,
                frame_id=frame_id,
            ),
        ),
    )
    database.finish_model_invocation(invocation_id, "succeeded")
    database.add_artifact(session_id, "summary", ended_at.isoformat(), "{}", "test-model")
    service = _service(database, now)

    service.run_session_maintenance()
    now[0] += timedelta(days=7)
    notices = service.run_session_maintenance()

    assert [notice.kind for notice in notices] == ["permanently_deleted"]
    assert database.get_session(session_id) is None
    assert not media_dir.exists()
    with database.connect() as connection:
        counts = [
            connection.execute(
                "SELECT COUNT(*) FROM frames WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM audio_chunks WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM transcript_segments WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
            connection.execute(
                """SELECT COUNT(*) FROM transcript_versions
                   WHERE segment_id IN (
                       SELECT id FROM transcript_segments WHERE session_id = ?
                   )""",
                (session_id,),
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM transcript_fts WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM questions WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
            connection.execute(
                """SELECT COUNT(*) FROM answer_versions
                   WHERE question_id IN (
                       SELECT id FROM questions WHERE session_id = ?
                   )""",
                (session_id,),
            ).fetchone()[0],
            connection.execute(
                """SELECT COUNT(*) FROM answer_evidence
                   WHERE answer_version_id IN (
                       SELECT answer.id FROM answer_versions AS answer
                       JOIN questions ON questions.id = answer.question_id
                       WHERE questions.session_id = ?
                   )""",
                (session_id,),
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM model_invocations WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
            connection.execute(
                """SELECT COUNT(*) FROM model_invocation_evidence AS evidence
                   LEFT JOIN model_invocations AS invocation
                     ON invocation.id = evidence.invocation_id
                   WHERE invocation.id IS NULL"""
            ).fetchone()[0],
            connection.execute(
                """SELECT COUNT(*) FROM model_invocation_evidence
                   WHERE invocation_id IN (
                       SELECT id FROM model_invocations WHERE session_id = ?
                   )""",
                (session_id,),
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM transcript_correction_settings WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM transcript_correction_runs WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM whisper_runs WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM artifacts WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM session_notifications WHERE session_id = ?", (session_id,)
            ).fetchone()[0],
        ]
    assert counts == [0] * len(counts)


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
    with database.connect() as connection:
        stored_path = connection.execute(
            "SELECT path FROM pending_media_deletions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    assert Path(stored_path) == Path("sessions") / f".{session_id}.deleting"
    assert not media_dir.exists()
    assert _service(Database(database.path), now).run_session_maintenance() == ()
    monkeypatch.undo()

    restarted = _service(Database(database.path), now)
    assert [notice.kind for notice in restarted.run_session_maintenance()] == [
        "permanently_deleted"
    ]
    assert not database.pending_media_deletions()


def test_restart_recovers_stale_recording_session(tmp_path: Path) -> None:
    now = [datetime(2026, 8, 4, 12, tzinfo=UTC)]
    database = Database(tmp_path / "stale-recording.sqlite3")
    session_id = database.create_session("重启恢复", (now[0] - timedelta(hours=1)).isoformat())

    restarted = _service(Database(database.path), now)

    unfinished = restarted.list_sessions(status="unfinished")
    assert [item.id for item in unfinished] == [session_id]
    assert unfinished[0].status == "interrupted"
    restarted.complete_interrupted_session(session_id)
    assert database.get_session(session_id).status == "complete"  # type: ignore[union-attr]


def test_active_current_session_is_sorted_first_even_when_older(tmp_path: Path) -> None:
    now = [datetime(2026, 8, 4, 12, tzinfo=UTC)]
    database = Database(tmp_path / "active-current.sqlite3")
    current_id = database.create_session("当前会话", (now[0] - timedelta(days=3)).isoformat())
    newer_id = _completed_session(database, "更新会话", now[0] - timedelta(days=1))
    recorder = ActiveRecorder()
    recorder.session_id = current_id
    service = JingzhiApplicationService(database, recorder=recorder, now=lambda: now[0])

    assert [item.id for item in service.list_sessions()] == [current_id, newer_id]
    assert service.list_sessions()[0].status == "recording"


def test_busy_interrupted_session_cannot_be_completed_or_deleted(tmp_path: Path) -> None:
    now = [datetime(2026, 8, 4, 12, tzinfo=UTC)]
    database = Database(tmp_path / "busy-session.sqlite3")
    session_id = database.create_session("仍在写入", (now[0] - timedelta(days=31)).isoformat())
    recorder = BusyRecorder()
    recorder.session_id = session_id
    service = JingzhiApplicationService(database, recorder=recorder, now=lambda: now[0])

    assert database.get_session(session_id).status == "interrupted"  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="会话仍在写入"):
        service.complete_interrupted_session(session_id)
    with pytest.raises(RuntimeError, match="会话仍在写入"):
        service.delete_session(session_id)
    assert service.run_session_maintenance() == ()
    assert database.get_session(session_id).trashed_at_utc is None  # type: ignore[union-attr]


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
