from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from jingzhi.database import Database
from jingzhi.recording_settings import RecordingPreferences, RecordingSettingsStore
from jingzhi.storage import (
    StartupSettingsStore,
    StorageActivity,
    StorageManager,
    canonical_whisper_repository_id,
    storage_activity,
)


def _store(
    tmp_path: Path,
    *,
    environ: dict[str, str] | None = None,
    writable_probe=None,
) -> StartupSettingsStore:
    return StartupSettingsStore(
        legacy_dir=tmp_path / "missing-legacy-data",
        config_dir=tmp_path / "local" / "Jingzhi",
        install_dir=tmp_path / "install",
        local_app_data=tmp_path / "local",
        environ=environ or {},
        writable_probe=writable_probe,
    )


def test_first_start_uses_install_data_and_persists_fixed_startup_config(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = store.resolve()
    restarted = store.resolve()

    assert first.data_dir == (tmp_path / "install" / "Data").resolve()
    assert first.model_dir == (tmp_path / "local" / "Jingzhi" / "Models").resolve()
    assert restarted == first
    assert store.path == (tmp_path / "local" / "Jingzhi" / "settings.json")
    saved = json.loads(store.path.read_text(encoding="utf-8"))
    assert saved["data_dir"] == str(first.data_dir)
    assert saved["model_dir"] == str(first.model_dir)


def test_unwritable_install_data_falls_back_to_local_app_data(tmp_path: Path) -> None:
    install_data = (tmp_path / "install" / "Data").resolve()
    store = _store(tmp_path, writable_probe=lambda path: path != install_data)

    paths = store.resolve()

    assert paths.data_dir == (tmp_path / "local" / "Jingzhi" / "Data").resolve()


def test_existing_relative_data_wins_during_upgrade(tmp_path: Path, monkeypatch) -> None:
    legacy = tmp_path / "working" / "data"
    legacy.mkdir(parents=True)
    (legacy / "jingzhi.sqlite3").touch()
    monkeypatch.chdir(legacy.parent)
    store = StartupSettingsStore(
        config_dir=tmp_path / "local" / "Jingzhi",
        install_dir=tmp_path / "install",
        local_app_data=tmp_path / "local",
        environ={},
        legacy_dir=legacy,
    )

    paths = store.resolve()

    assert paths.data_dir == legacy.resolve()


def test_environment_paths_override_ui_configuration(tmp_path: Path) -> None:
    env_data = tmp_path / "managed-data"
    hf_home = tmp_path / "managed-hf"
    store = _store(
        tmp_path,
        environ={"STUDY_DATA_DIR": str(env_data), "HF_HOME": str(hf_home)},
    )

    paths = store.resolve()

    assert paths.data_dir == env_data.resolve()
    assert paths.model_dir == (hf_home / "hub").resolve()
    assert paths.data_managed_by_env
    assert paths.model_managed_by_env


def test_hf_hub_cache_environment_variable_overrides_saved_model_directory(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "explicit-cache"
    store = _store(tmp_path, environ={"HF_HUB_CACHE": str(cache)})

    paths = store.resolve()

    assert paths.model_dir == cache.resolve()
    assert paths.model_managed_by_env


def _seed_data_directory(data_dir: Path) -> str:
    database = Database(data_dir / "jingzhi.sqlite3")
    session_id = database.create_session("迁移测试", "2026-08-04T00:00:00+00:00")
    frame = data_dir / "sessions" / session_id / "frames" / "display-01" / "frame.webp"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    database.add_frame(session_id, 100, frame, "hash", (10, 10))
    audio = data_dir / "sessions" / session_id / "audio" / "system" / "chunk.flac"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    database.add_audio_chunk(session_id, "system", 0, 100, audio)
    (data_dir / "samples").mkdir()
    (data_dir / "samples" / "sample.wav").write_bytes(b"sample")
    (data_dir / "logs").mkdir()
    (data_dir / "logs" / "app.log").write_text("log", encoding="utf-8")
    (data_dir / "whisper.json").write_text('{"version": 1}', encoding="utf-8")
    return session_id


def test_existing_session_migration_copies_all_data_rewrites_paths_and_survives_restart(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    session_id = _seed_data_directory(paths.data_dir)
    staged = paths.data_dir / "sessions" / session_id / f".{session_id}.deleting"
    staged.mkdir(parents=True)
    (staged / "frame.webp").write_bytes(b"pending")
    with sqlite3.connect(paths.data_dir / "jingzhi.sqlite3") as connection:
        connection.execute(
            """INSERT INTO pending_media_deletions(session_id, title, path)
               VALUES (?, ?, ?)""",
            (session_id, "迁移中的删除", str(staged)),
        )
    manager = StorageManager(paths, store)
    target = tmp_path / "moved-data"

    result = manager.migrate_data(target)
    restarted = store.resolve()

    assert result.old_dir == paths.data_dir
    assert result.new_dir == target.resolve()
    assert restarted.data_dir == target.resolve()
    assert (target / "samples" / "sample.wav").read_bytes() == b"sample"
    assert (target / "logs" / "app.log").read_text(encoding="utf-8") == "log"
    assert not (target / "jingzhi.sqlite3-wal").exists()
    assert not (target / "jingzhi.sqlite3-shm").exists()
    with sqlite3.connect(target / "jingzhi.sqlite3") as connection:
        frame_path = Path(connection.execute("SELECT path FROM frames").fetchone()[0])
        audio_path = Path(connection.execute("SELECT path FROM audio_chunks").fetchone()[0])
        pending_path = Path(
            connection.execute(
                "SELECT path FROM pending_media_deletions WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        )
    assert frame_path == target / "sessions" / session_id / "frames" / "display-01" / "frame.webp"
    assert audio_path == target / "sessions" / session_id / "audio" / "system" / "chunk.flac"
    assert pending_path == target / "sessions" / session_id / f".{session_id}.deleting"
    assert (pending_path / "frame.webp").read_bytes() == b"pending"
    assert paths.data_dir.exists()


def test_empty_data_directory_migrates_and_switches_configuration(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "empty-target"
    target.mkdir()

    StorageManager(paths, store).migrate_data(target)

    assert target.is_dir()
    assert store.resolve().data_dir == target.resolve()


def test_data_migration_is_blocked_during_recording_or_background_writes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    manager = StorageManager(paths, store, busy_reason=lambda: "正在录制会话")

    with pytest.raises(RuntimeError, match="正在录制会话"):
        manager.migrate_data(tmp_path / "target")


def test_space_and_write_validation_leave_old_configuration_unchanged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    paths.data_dir.mkdir(parents=True)
    (paths.data_dir / "large.bin").write_bytes(b"1234")
    original = store.resolve()
    insufficient = StorageManager(
        paths,
        store,
        disk_usage=lambda _path: SimpleNamespace(total=10, used=9, free=1),
    )

    with pytest.raises(OSError, match="空间不足"):
        insufficient.migrate_data(tmp_path / "no-space")
    assert store.resolve() == original

    unwritable = StorageManager(paths, store, writable_probe=lambda _path: False)
    with pytest.raises(OSError, match="不可写"):
        unwritable.migrate_data(tmp_path / "not-writable")
    assert store.resolve() == original


def test_copy_failure_removes_staging_and_keeps_old_data_active(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    paths.data_dir.mkdir(parents=True)
    (paths.data_dir / "one.txt").write_text("one", encoding="utf-8")
    (paths.data_dir / "two.txt").write_text("two", encoding="utf-8")
    calls = 0

    def failing_copy(source, destination, *, follow_symlinks=True):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        return shutil.copy2(source, destination, follow_symlinks=follow_symlinks)

    target = tmp_path / "failed-target"
    manager = StorageManager(paths, store, copy_function=failing_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        manager.migrate_data(target)

    assert store.resolve().data_dir == paths.data_dir
    assert not target.exists()
    assert not any(tmp_path.glob(".failed-target.jingzhi-migration-*"))


def test_migration_rejects_target_nested_inside_source(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    paths.data_dir.mkdir(parents=True)
    manager = StorageManager(paths, store)

    with pytest.raises(ValueError, match="当前目录内部"):
        manager.migrate_data(paths.data_dir / "nested")


def test_storage_activity_prevents_writes_for_full_migration_scope() -> None:
    activity = StorageActivity()

    with (
        activity.migrating(lambda: None),
        pytest.raises(RuntimeError, match="存储迁移正在进行"),
        activity.writing("开始会话"),
    ):
        pass

    with (
        activity.writing("保存配置"),
        pytest.raises(RuntimeError, match="后台任务仍在写入"),
        activity.migrating(lambda: None),
    ):
        pass


def test_recording_preferences_cannot_write_during_migration(tmp_path: Path) -> None:
    store = RecordingSettingsStore(tmp_path)

    with (
        storage_activity.migrating(lambda: None),
        pytest.raises(RuntimeError, match="存储迁移正在进行"),
    ):
        store.save(RecordingPreferences())


def test_model_aliases_resolve_to_the_same_repository() -> None:
    assert canonical_whisper_repository_id("large") == canonical_whisper_repository_id("large-v3")
    assert canonical_whisper_repository_id("Systran/faster-whisper-small") == (
        "Systran/faster-whisper-small"
    )


def test_model_cache_lists_sizes_migrates_or_switches_to_empty_directory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    model_file = (
        paths.model_dir
        / "models--Systran--faster-whisper-small"
        / "snapshots"
        / "revision"
        / "model.bin"
    )
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"model")
    manager = StorageManager(paths, store)

    assert [(item.name, item.size_bytes) for item in manager.models()] == [("small", 5)]

    migrated = tmp_path / "migrated-models"
    manager.migrate_models(migrated, move_existing=True)
    assert (migrated / model_file.relative_to(paths.model_dir)).read_bytes() == b"model"
    assert store.resolve().model_dir == migrated.resolve()

    restarted_paths = store.resolve()
    empty = tmp_path / "redownload-models"
    StorageManager(restarted_paths, store).migrate_models(empty, move_existing=False)
    assert empty.is_dir()
    assert not any(empty.iterdir())
    assert store.resolve().model_dir == empty.resolve()


def test_model_migration_failure_keeps_old_cache_configured(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    model_file = paths.model_dir / "models--Systran--faster-whisper-small" / "model.bin"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"model")

    def failing_copy(_source, _destination, *, follow_symlinks=True):  # type: ignore[no-untyped-def]
        raise OSError("model copy failed")

    target = tmp_path / "failed-models"
    manager = StorageManager(paths, store, copy_function=failing_copy)

    with pytest.raises(OSError, match="model copy failed"):
        manager.migrate_models(target, move_existing=True)

    assert store.resolve().model_dir == paths.model_dir
    assert model_file.read_bytes() == b"model"
    assert not target.exists()


def test_model_delete_rejects_active_model_and_removes_idle_model(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    repository = paths.model_dir / "models--Systran--faster-whisper-small"
    repository.mkdir(parents=True)
    (repository / "model.bin").write_bytes(b"model")
    active = StorageManager(
        paths,
        store,
        model_in_use=lambda repository_id: repository_id.endswith("/faster-whisper-small"),
    )

    with pytest.raises(RuntimeError, match="正在使用"):
        active.delete_model("small")
    assert repository.exists()

    StorageManager(paths, store).delete_model("small")
    assert not repository.exists()


def test_confirmed_old_data_deletion_runs_only_after_restart(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    paths.data_dir.mkdir(parents=True)
    (paths.data_dir / "keep.txt").write_text("old", encoding="utf-8")
    manager = StorageManager(paths, store)
    target = tmp_path / "new-data"

    result = manager.migrate_data(target)
    manager.confirm_delete_old_data(result.old_dir)

    assert result.old_dir.exists()
    restarted = store.resolve()
    assert restarted.data_dir == target.resolve()
    assert result.old_dir.exists()

    store.complete_successful_startup(restarted.data_dir)

    assert not result.old_dir.exists()


def test_missing_migration_target_never_deletes_confirmed_old_data(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = store.resolve()
    paths.data_dir.mkdir(parents=True)
    (paths.data_dir / "keep.txt").write_text("old", encoding="utf-8")
    manager = StorageManager(paths, store)
    result = manager.migrate_data(tmp_path / "new-data")
    manager.confirm_delete_old_data(result.old_dir)
    shutil.rmtree(result.new_dir)

    restarted = store.resolve()
    restarted.data_dir.mkdir(parents=True)
    store.complete_successful_startup(restarted.data_dir)

    assert result.old_dir.exists()
