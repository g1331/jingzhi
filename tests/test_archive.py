from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import jingzhi.archive as archive_module
from jingzhi.application import JingzhiApplicationService
from jingzhi.archive import ArchiveConflictError, ArchiveError, ArchiveManager
from jingzhi.database import Database


def _seed_session(data_dir: Path) -> tuple[Database, str, int, int]:
    database = Database(data_dir / "jingzhi.sqlite3")
    session_id = database.create_session("可归档会话", "2026-08-04T10:00:00+00:00")
    session_dir = data_dir / "sessions" / session_id
    frame_path = session_dir / "frames" / "display-01" / "frame.webp"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"frame-bytes")
    frame_id = database.add_frame(
        session_id,
        1_000,
        frame_path,
        "hash",
        (10, 10),
        source_id="display:primary",
    )
    audio_path = session_dir / "audio" / "system" / "chunk.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio-bytes")
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 2_000, audio_path)
    segment_id = database.add_transcript(
        session_id,
        chunk_id,
        "system",
        200,
        1_800,
        "原始字幕内容",
        "zh",
        0.95,
    )
    transcript_version_id = database.add_transcript_version(
        segment_id, "user_edit", "修订后的字幕内容"
    )
    assert transcript_version_id is not None
    question_id = database.add_question(
        session_id,
        1_500,
        "这段内容说明了什么？",
        None,
        0,
        1_500,
    )
    database.add_question_note(question_id, "需要在复习时再确认。")
    database.record_answer_version(
        question_id,
        model="answer-model",
        connection_json='{"connection_name":"本地测试连接"}',
        request_status="succeeded",
        request_id="answer-request",
        answer="这是回答。",
        error=None,
        evidence_state="exact",
        evidence=[
            {
                "stable_id": f"transcript-version:{transcript_version_id}",
                "kind": "transcript",
                "source": "system",
                "start_ms": 200,
                "end_ms": 1_800,
                "transcript_version_id": transcript_version_id,
                "content_text": "修订后的字幕内容",
            },
            {
                "stable_id": f"frame:{frame_id}",
                "kind": "frame",
                "source": "display:primary",
                "start_ms": 1_000,
                "end_ms": 1_000,
                "frame_id": frame_id,
                "resource_path": str(frame_path),
            },
        ],
    )
    material = database.record_material_version(
        session_id,
        kind="generated",
        content="# 会话材料\n\n这是可独立检查的 Markdown。",
        template_id="default",
        model="analysis-model",
        connection_json='{"connection_name":"本地测试连接"}',
        model_invocation_id=None,
        request_status="succeeded",
        request_id="material-request",
        error=None,
        evidence_state="exact",
        evidence=[
            {
                "stable_id": f"transcript-version:{transcript_version_id}",
                "kind": "transcript",
                "source": "system",
                "start_ms": 200,
                "end_ms": 1_800,
                "transcript_version_id": transcript_version_id,
                "content_text": "修订后的字幕内容",
            },
            {
                "stable_id": f"frame:{frame_id}",
                "kind": "frame",
                "source": "display:primary",
                "start_ms": 1_000,
                "end_ms": 1_000,
                "frame_id": frame_id,
                "resource_path": str(frame_path),
            },
        ],
    )
    assert material.session_id == session_id
    database.finish_session(session_id, "2026-08-04T10:05:00+00:00", "complete")
    (data_dir / "provider.json").write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": "default",
                        "name": "默认连接",
                        "base_url": "https://example.invalid/v1",
                        "api_key": "do-not-export-this-key",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (data_dir / "whisper.json").write_text(
        '{"version": 1, "profile": "balanced"}', encoding="utf-8"
    )
    return database, session_id, question_id, frame_id


def test_application_archive_rejects_active_capture_writers(tmp_path: Path) -> None:
    database = Database(tmp_path / "jingzhi.sqlite3")
    recorder = SimpleNamespace(
        is_recording=False,
        storage_busy_reason=lambda: "字幕转写仍在写入数据",
    )
    service = JingzhiApplicationService(database, recorder=recorder)

    with pytest.raises(RuntimeError, match="归档需要等待当前写入完成"):
        service.create_backup(tmp_path / "backup.zip")
    with pytest.raises(RuntimeError, match="归档需要等待当前写入完成"):
        service.export_session("missing-session", tmp_path / "session.zip")
    with pytest.raises(RuntimeError, match="归档需要等待当前写入完成"):
        service.preview_restore(tmp_path / "missing.zip", tmp_path / "restore")


def test_application_restore_rejects_active_capture_writers(tmp_path: Path) -> None:
    database, _session_id, _question_id, _frame_id = _seed_session(tmp_path)
    backup = tmp_path / "backup.zip"
    ArchiveManager(database).create_backup(backup)
    recorder = SimpleNamespace(
        is_recording=False,
        storage_busy_reason=lambda: "字幕转写仍在写入数据",
    )
    service = JingzhiApplicationService(database, recorder=recorder)

    with pytest.raises(RuntimeError, match="归档需要等待当前写入完成"):
        service.restore_backup(backup, tmp_path / "restored")


def test_archive_rejects_symlink_destination(tmp_path: Path) -> None:
    database, session_id, _question_id, _frame_id = _seed_session(tmp_path)
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"keep")
    destination = tmp_path / "archive.zip"
    try:
        destination.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ArchiveError, match="归档目标不能是符号链接"):
        ArchiveManager(database).export_session(session_id, destination)
    assert outside.read_bytes() == b"keep"


def test_session_export_rejects_source_symlinks(tmp_path: Path) -> None:
    database, session_id, _question_id, _frame_id = _seed_session(tmp_path)
    outside = tmp_path / "outside.webp"
    outside.write_bytes(b"outside")
    symlink = tmp_path / "sessions" / session_id / "linked.webp"
    try:
        symlink.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    database.add_frame(session_id, 2_000, symlink, "linked-hash", (10, 10))

    with pytest.raises(ArchiveError, match="不允许符号链接"):
        ArchiveManager(database).export_session(session_id, tmp_path / "export.zip")


def test_session_export_contains_markdown_media_versions_and_relative_evidence(
    tmp_path: Path,
) -> None:
    database, session_id, question_id, frame_id = _seed_session(tmp_path)
    archive = tmp_path / "session-export.zip"

    result = ArchiveManager(database).export_session(session_id, archive)

    assert result == archive
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        manifest = json.loads(package.read("manifest.json"))
        evidence = json.loads(package.read("evidence/index.json"))
        transcripts = json.loads(package.read("transcripts.json"))
        questions = json.loads(package.read("questions.json"))
        records = json.loads(package.read("records.json"))
        assert manifest["format_version"] == 1
        assert manifest["kind"] == "session_export"
        assert manifest["session_id"] == session_id
        assert "transcripts.json" in names
        assert "questions.json" in names
        assert any(name.startswith("materials/") and name.endswith(".md") for name in names)
        assert any(name.startswith("answers/") and name.endswith(".md") for name in names)
        assert any(name.startswith("media/frames/") for name in names)
        assert any(name.startswith("media/audio/") for name in names)
        entries = evidence["entries"]
        answer_entry = next(item for item in entries if item["owner"] == "answer")
        material_entry = next(item for item in entries if item["owner"] == "material")
        assert answer_entry["document_path"] == f"answers/question-{question_id}-v1.md"
        assert answer_entry["document_path"].startswith("answers/")
        assert material_entry["document_path"].startswith("materials/")
        assert any(item["frame_id"] == frame_id for item in entries)
        assert all(not Path(item["relative_path"]).is_absolute() for item in entries)
        assert all(not Path(item["document_path"]).is_absolute() for item in entries)
        assert {
            version["kind"]: version["text"] for version in transcripts["segments"][0]["versions"]
        } == {"original": "原始字幕内容", "user_edit": "修订后的字幕内容"}
        assert questions["questions"][0]["question"] == "这段内容说明了什么？"
        assert questions["notes"][0]["content"] == "需要在复习时再确认。"
        assert records["answer_versions"][0]["answer"] == "这是回答。"
        assert records["session_material_versions"][0]["content"].startswith("# 会话材料")
        assert {item["stable_id"] for item in entries if item["owner"] == "answer"} == {
            item["stable_id"] for item in records["answer_evidence"]
        }
        assert str(tmp_path) not in package.read("evidence/index.json").decode("utf-8")


def test_archive_destination_inside_session_media_tree_does_not_recurse(tmp_path: Path) -> None:
    database, session_id, _question_id, _frame_id = _seed_session(tmp_path)
    destination = tmp_path / "sessions" / session_id / "backup.zip"

    manager = ArchiveManager(database)
    manager.create_backup(destination)
    session_export = tmp_path / "sessions" / session_id / "session-export.zip"
    manager.export_session(session_id, session_export)
    manager.create_backup(destination)
    manager.export_session(session_id, session_export)

    assert destination.is_file()
    assert session_export.is_file()
    with zipfile.ZipFile(destination) as package:
        names = package.namelist()
        assert "database/jingzhi.sqlite3" in names
        assert f"media/sessions/{session_id}/backup.zip" not in names
        assert f"media/sessions/{session_id}/session-export.zip" in names
    with zipfile.ZipFile(session_export) as package:
        names = package.namelist()
        assert "session.json" in names
        assert "media/backup.zip" in names
        assert "media/session-export.zip" not in names


def test_failed_session_export_preserves_previous_destination_and_leaves_no_partial_zip(
    tmp_path: Path,
) -> None:
    database, session_id, _question_id, _frame_id = _seed_session(tmp_path)
    archive = tmp_path / "session-export.zip"
    archive.write_bytes(b"previous-complete-export")
    frame_path = tmp_path / "sessions" / session_id / "frames" / "display-01" / "frame.webp"
    frame_path.unlink()

    with pytest.raises(ArchiveError, match="媒体文件不存在"):
        ArchiveManager(database).export_session(session_id, archive)

    assert archive.read_bytes() == b"previous-complete-export"
    assert not any(tmp_path.glob(".session-export.zip.*.tmp"))


def test_full_backup_excludes_api_keys_and_restores_observable_data_and_index(
    tmp_path: Path,
) -> None:
    database, session_id, question_id, frame_id = _seed_session(tmp_path)
    backup = tmp_path / "full-backup.zip"

    ArchiveManager(database).create_backup(backup)

    with zipfile.ZipFile(backup) as package:
        names = set(package.namelist())
        raw = b"".join(package.read(name) for name in names)
        manifest = json.loads(package.read("manifest.json"))
        config_manifest = json.loads(package.read("config/manifest.json"))
        assert manifest["format_version"] == 1
        assert manifest["kind"] == "full_backup"
        assert "database/jingzhi.sqlite3" in names
        assert "config/manifest.json" in names
        assert any(name.startswith("media/sessions/") for name in names)
        assert b"do-not-export-this-key" not in raw
        assert config_manifest["secrets_excluded"] == ["api_key"]

    restored_dir = tmp_path / "restored-data"
    manager = ArchiveManager(database)
    preview = manager.preview_restore(backup, restored_dir)
    assert preview.can_restore is True
    assert preview.session_ids == (session_id,)

    manager.restore_backup(backup, restored_dir)

    restored = Database(restored_dir / "jingzhi.sqlite3")
    restored_session = restored.get_session(session_id)
    assert restored_session is not None
    assert restored_session.title == "可归档会话"
    assert restored.session_answers(session_id)[0].question_id == question_id
    answer = restored.session_answers(session_id)[0]
    answer_evidence = restored.answer_evidence(answer.id)
    assert {item.stable_id for item in answer_evidence} == {
        item.stable_id
        for item in database.answer_evidence(database.session_answers(session_id)[0].id)
    }
    assert {item.frame_id for item in answer_evidence if item.frame_id is not None} == {frame_id}
    assert restored.question_notes(question_id)[0].content == "需要在复习时再确认。"
    segment_id = restored.timeline_transcripts(session_id, 0, 10_000)[0].id
    assert [item.text for item in restored.transcript_versions(segment_id)] == [
        "原始字幕内容",
        "修订后的字幕内容",
    ]
    assert restored.session_material_versions(session_id)[0].content.startswith("# 会话材料")
    material_id = restored.session_material_versions(session_id)[0].id
    material_evidence = restored.material_evidence(material_id)
    assert {item.stable_id for item in material_evidence} == {
        item.stable_id
        for item in database.material_evidence(database.session_material_versions(session_id)[0].id)
    }
    assert {item.frame_id for item in material_evidence if item.frame_id is not None} == {frame_id}
    with sqlite3.connect(restored_dir / "jingzhi.sqlite3") as connection:
        frame_path = Path(connection.execute("SELECT path FROM frames").fetchone()[0])
        audio_path = Path(connection.execute("SELECT path FROM audio_chunks").fetchone()[0])
        fts_count = connection.execute("SELECT COUNT(*) FROM transcript_fts").fetchone()[0]
        segment_count = connection.execute("SELECT COUNT(*) FROM transcript_segments").fetchone()[0]
    assert (
        frame_path
        == restored_dir / "sessions" / session_id / "frames" / "display-01" / "frame.webp"
    )
    assert audio_path == restored_dir / "sessions" / session_id / "audio" / "system" / "chunk.flac"
    assert frame_path.read_bytes() == b"frame-bytes"
    assert audio_path.read_bytes() == b"audio-bytes"
    assert fts_count == segment_count == 1
    assert (
        json.loads((restored_dir / "provider.json").read_text(encoding="utf-8"))["connections"][
            0
        ].get("api_key")
        is None
    )


def test_full_backup_restores_referenced_media_outside_session_tree(tmp_path: Path) -> None:
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = database.create_session("外部媒体", "2026-08-04T10:00:00+00:00")
    frame_path = tmp_path / "external" / "frame.webp"
    frame_path.parent.mkdir()
    frame_path.write_bytes(b"external-frame")
    database.add_frame(session_id, 1_000, frame_path, "hash", (10, 10))
    audio_path = tmp_path / "external" / "chunk.flac"
    audio_path.write_bytes(b"external-audio")
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 2_000, audio_path)
    database.add_transcript(session_id, chunk_id, "system", 100, 1_900, "外部字幕", "zh", 0.9)
    database.finish_session(session_id, "2026-08-04T10:05:00+00:00", "complete")
    backup = tmp_path / "external-media-backup.zip"

    manager = ArchiveManager(database)
    manager.create_backup(backup)
    restored_dir = tmp_path / "restored-external"
    manager.restore_backup(backup, restored_dir)

    assert (restored_dir / "external" / "frame.webp").read_bytes() == b"external-frame"
    assert (restored_dir / "external" / "chunk.flac").read_bytes() == b"external-audio"


def test_full_backup_resolves_relative_media_paths_before_restoring(tmp_path: Path) -> None:
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = database.create_session("相对媒体路径", "2026-08-04T10:00:00+00:00")
    frame_path = tmp_path / "sessions" / session_id / "frame.webp"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"relative-frame")
    database.add_frame(
        session_id,
        1_000,
        Path("sessions") / session_id / "frame.webp",
        "hash",
        (10, 10),
    )
    database.finish_session(session_id, "2026-08-04T10:01:00+00:00", "complete")
    backup = tmp_path / "relative-backup.zip"

    manager = ArchiveManager(database)
    manager.create_backup(backup)
    restored_dir = tmp_path / "restored-relative"
    manager.restore_backup(backup, restored_dir)

    with sqlite3.connect(restored_dir / "jingzhi.sqlite3") as connection:
        restored_path = Path(connection.execute("SELECT path FROM frames").fetchone()[0])
    assert restored_path == restored_dir / "sessions" / session_id / "frame.webp"
    assert restored_path.read_bytes() == b"relative-frame"


def test_restore_preview_rejects_invalid_json_configuration(tmp_path: Path) -> None:
    database, _session_id, _question_id, _frame_id = _seed_session(tmp_path)
    source = tmp_path / "valid-backup.zip"
    ArchiveManager(database).create_backup(source)
    with zipfile.ZipFile(source) as package:
        entries = {name: package.read(name) for name in package.namelist()}
    entries["config/provider.json"] = b"{invalid"
    manifest = json.loads(entries["manifest.json"])
    manifest["files"]["config/provider.json"] = hashlib.sha256(
        entries["config/provider.json"]
    ).hexdigest()
    entries["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    corrupted = tmp_path / "invalid-config.zip"
    with zipfile.ZipFile(corrupted, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, content in entries.items():
            package.writestr(name, content)

    with pytest.raises(ArchiveError, match="配置文件格式无效：provider.json"):
        ArchiveManager(database).preview_restore(corrupted, tmp_path / "restored")


def test_restore_rejects_target_replaced_with_symlink_after_preview(tmp_path: Path) -> None:
    database, _session_id, _question_id, _frame_id = _seed_session(tmp_path)
    backup = tmp_path / "full-backup.zip"
    manager = ArchiveManager(database)
    manager.create_backup(backup)
    target = tmp_path / "restored-data"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_preview = manager._preview_restore

    def preview_with_symlink(archive_path: Path, target_path: Path):
        preview = original_preview(archive_path, target_path)
        target_path.symlink_to(outside, target_is_directory=True)
        return preview

    try:
        target.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    target.unlink()
    manager._preview_restore = preview_with_symlink  # type: ignore[method-assign]

    with pytest.raises(ArchiveError, match="符号链接"):
        manager.restore_backup(backup, target)


def test_restore_does_not_delete_data_created_during_media_copy(
    tmp_path: Path, monkeypatch
) -> None:
    database, _session_id, _question_id, _frame_id = _seed_session(tmp_path)
    backup = tmp_path / "full-backup.zip"
    manager = ArchiveManager(database)
    manager.create_backup(backup)
    target = tmp_path / "restored-data"
    original_copy = archive_module._copy_restore_path
    injected = False

    def copy_with_race(source: Path, destination: Path, root: Path, created_paths: list[Path]):
        nonlocal injected
        if not injected:
            destination.mkdir(parents=True)
            (destination / "created-by-other-process.txt").write_text("keep", encoding="utf-8")
            injected = True
        return original_copy(source, destination, root, created_paths)

    monkeypatch.setattr(archive_module, "_copy_restore_path", copy_with_race)

    with pytest.raises(ArchiveConflictError, match="恢复已取消"):
        manager.restore_backup(backup, target)
    assert (target / "sessions" / "created-by-other-process.txt").read_text(
        encoding="utf-8"
    ) == "keep"


def test_restore_preview_rejects_mismatched_fts_content(tmp_path: Path) -> None:
    database, _session_id, _question_id, _frame_id = _seed_session(tmp_path)
    source = tmp_path / "valid-backup.zip"
    ArchiveManager(database).create_backup(source)
    with zipfile.ZipFile(source) as package:
        entries = {name: package.read(name) for name in package.namelist()}
    database_copy = tmp_path / "mutated.sqlite3"
    database_copy.write_bytes(entries["database/jingzhi.sqlite3"])
    with sqlite3.connect(database_copy) as connection:
        connection.execute("UPDATE transcript_fts SET text = '损坏索引'")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    entries["database/jingzhi.sqlite3"] = database_copy.read_bytes()
    manifest = json.loads(entries["manifest.json"])
    manifest["files"]["database/jingzhi.sqlite3"] = hashlib.sha256(
        entries["database/jingzhi.sqlite3"]
    ).hexdigest()
    entries["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    corrupted = tmp_path / "corrupted-backup.zip"
    with zipfile.ZipFile(corrupted, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, content in entries.items():
            package.writestr(name, content)

    with pytest.raises(ArchiveError, match="全文索引与字幕片段不一致"):
        ArchiveManager(database).preview_restore(corrupted, tmp_path / "restored")


def test_restore_rejects_target_changed_after_preview_without_deleting_raced_data(
    tmp_path: Path, monkeypatch
) -> None:
    database, _session_id, _question_id, _frame_id = _seed_session(tmp_path)
    backup = tmp_path / "full-backup.zip"
    manager = ArchiveManager(database)
    manager.create_backup(backup)
    target = tmp_path / "restored-data"
    original_preview = manager._preview_restore

    def preview_with_race(archive_path: Path, target_path: Path):
        preview = original_preview(archive_path, target_path)
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "created-by-other-process.txt").write_text("keep", encoding="utf-8")
        return preview

    monkeypatch.setattr(manager, "_preview_restore", preview_with_race)

    with pytest.raises(ArchiveConflictError, match="预览后发生变化"):
        manager.restore_backup(backup, target)

    assert (target / "created-by-other-process.txt").read_text(encoding="utf-8") == "keep"
    assert not (target / "jingzhi.sqlite3").exists()


def test_full_backup_fails_when_existing_config_is_unreadable(tmp_path: Path) -> None:
    database, _session_id, _question_id, _frame_id = _seed_session(tmp_path)
    (tmp_path / "provider.json").write_text("{invalid", encoding="utf-8")

    with pytest.raises(ArchiveError, match="配置文件无法读取：provider.json"):
        ArchiveManager(database).create_backup(tmp_path / "backup.zip")

    assert not (tmp_path / "backup.zip").exists()


def test_restore_rejects_unlisted_zip_members(tmp_path: Path) -> None:
    database, _session_id, _question_id, _frame_id = _seed_session(tmp_path)
    source = tmp_path / "valid-backup.zip"
    ArchiveManager(database).create_backup(source)
    with zipfile.ZipFile(source) as package:
        entries = {name: package.read(name) for name in package.namelist()}
    entries["media/unverified.bin"] = b"not-in-manifest"
    corrupted = tmp_path / "unlisted-member.zip"
    with zipfile.ZipFile(corrupted, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, content in entries.items():
            package.writestr(name, content)

    with pytest.raises(ArchiveError, match="文件校验表与 ZIP 内容不一致"):
        ArchiveManager(database).preview_restore(corrupted, tmp_path / "restored")


def test_restore_preview_rejects_repeat_restore_and_existing_id_without_overwrite(
    tmp_path: Path,
) -> None:
    database, session_id, _question_id, _frame_id = _seed_session(tmp_path)
    backup = tmp_path / "full-backup.zip"
    manager = ArchiveManager(database)
    manager.create_backup(backup)
    restored_dir = tmp_path / "restored-data"
    manager.restore_backup(backup, restored_dir)

    preview = manager.preview_restore(backup, restored_dir)

    assert preview.can_restore is False
    assert preview.conflicting_session_ids == (session_id,)
    assert preview.strategy == "reject_existing_data"
    with pytest.raises(ArchiveConflictError, match="不会覆盖现有数据"):
        manager.restore_backup(backup, restored_dir)
    assert Database(restored_dir / "jingzhi.sqlite3").get_session(session_id) is not None


def test_restore_rejects_zip_path_traversal_before_writing_target(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    manifest = {"format_version": 1, "kind": "full_backup", "files": {}}
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("../outside.txt", "not allowed")
    target = tmp_path / "target"

    with pytest.raises(ArchiveError, match="非法归档路径"):
        ArchiveManager(Database(tmp_path / "source.sqlite3")).preview_restore(archive, target)

    assert not (tmp_path / "outside.txt").exists()
    assert not target.exists()
