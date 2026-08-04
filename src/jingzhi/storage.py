from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StoragePaths:
    data_dir: Path
    model_dir: Path
    data_managed_by_env: bool = False
    model_managed_by_env: bool = False


@dataclass(frozen=True, slots=True)
class StorageUsage:
    path: Path
    used_bytes: int
    free_bytes: int


@dataclass(frozen=True, slots=True)
class StoredModel:
    name: str
    repository_id: str
    path: Path
    size_bytes: int


@dataclass(frozen=True, slots=True)
class MigrationResult:
    old_dir: Path
    new_dir: Path


class StorageActivity:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_writes = 0
        self._migration_active = False

    @contextmanager
    def writing(self, description: str) -> Iterator[None]:
        with self._lock:
            if self._migration_active:
                raise RuntimeError(f"存储迁移正在进行，不能{description}")
            self._active_writes += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_writes -= 1

    @contextmanager
    def migrating(self, busy_reason: Callable[[], str | None]) -> Iterator[None]:
        with self._lock:
            if self._migration_active:
                raise RuntimeError("已有存储迁移正在进行")
            if self._active_writes:
                raise RuntimeError("后台任务仍在写入应用数据或模型缓存")
            reason = busy_reason()
            if reason:
                raise RuntimeError(reason)
            self._migration_active = True
        try:
            yield
        finally:
            with self._lock:
                self._migration_active = False


storage_activity = StorageActivity()


def storage_writer(description: str):  # type: ignore[no-untyped-def]
    def decorate(function):  # type: ignore[no-untyped-def]
        @wraps(function)
        def guarded(*args, **kwargs):  # type: ignore[no-untyped-def]
            with storage_activity.writing(description):
                return function(*args, **kwargs)

        return guarded

    return decorate


def canonical_whisper_repository_id(model: str) -> str:
    if "/" in model:
        return model
    try:
        from faster_whisper.utils import _MODELS

        repository_id = _MODELS.get(model)
        if repository_id:
            return str(repository_id)
    except ImportError:
        pass
    return f"Systran/faster-whisper-{model}"


def application_install_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def local_app_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    configured = values.get("LOCALAPPDATA")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "AppData" / "Local").resolve()


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _is_writable_directory(path: Path) -> bool:
    probe_dir = path if path.is_dir() else path.parent
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_dir / f".jingzhi-write-test-{uuid.uuid4().hex}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        probe.unlink(missing_ok=True)
        return False


class StartupSettingsStore:
    def __init__(
        self,
        *,
        config_dir: Path | None = None,
        install_dir: Path | None = None,
        local_app_data: Path | None = None,
        environ: Mapping[str, str] | None = None,
        writable_probe: Callable[[Path], bool] | None = None,
        legacy_dir: Path | None = None,
    ) -> None:
        self.environ = os.environ if environ is None else environ
        self.local_app_data = (
            local_app_data_dir(self.environ) if local_app_data is None else local_app_data.resolve()
        )
        self.config_dir = (
            self.local_app_data / "Jingzhi" if config_dir is None else config_dir.resolve()
        )
        self.install_dir = (
            application_install_dir() if install_dir is None else install_dir.resolve()
        )
        self.legacy_dir = (
            (Path.cwd() / "data").resolve() if legacy_dir is None else legacy_dir.resolve()
        )
        self.path = self.config_dir / "settings.json"
        self.writable_probe = writable_probe or _is_writable_directory

    def _load(self) -> dict[str, object]:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) and loaded.get("version") == 1 else {}

    def _save_document(self, document: dict[str, object]) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def _default_data_dir(self) -> Path:
        if self.legacy_dir.is_dir() and any(self.legacy_dir.iterdir()):
            return self.legacy_dir
        installed = (self.install_dir / "Data").resolve()
        if self.writable_probe(installed):
            return installed
        return (self.local_app_data / "Jingzhi" / "Data").resolve()

    def resolve(self) -> StoragePaths:
        document = self._load()
        configured_data = document.get("data_dir")
        configured_models = document.get("model_dir")
        data_dir = (
            Path(str(configured_data)).expanduser().resolve()
            if configured_data
            else self._default_data_dir()
        )
        model_dir = (
            Path(str(configured_models)).expanduser().resolve()
            if configured_models
            else (self.local_app_data / "Jingzhi" / "Models").resolve()
        )
        if not document:
            self._save_document(
                {"version": 1, "data_dir": str(data_dir), "model_dir": str(model_dir)}
            )

        env_data = self.environ.get("STUDY_DATA_DIR")
        explicit_model_cache = self.environ.get("HF_HUB_CACHE") or self.environ.get(
            "HUGGINGFACE_HUB_CACHE"
        )
        hf_home = self.environ.get("HF_HOME")
        env_model_dir = (
            Path(explicit_model_cache).expanduser().resolve()
            if explicit_model_cache
            else (Path(hf_home).expanduser().resolve() / "hub" if hf_home else None)
        )
        paths = StoragePaths(
            data_dir=(Path(env_data).expanduser().resolve() if env_data else data_dir),
            model_dir=env_model_dir or model_dir,
            data_managed_by_env=bool(env_data),
            model_managed_by_env=env_model_dir is not None,
        )
        return paths

    def complete_successful_startup(self, active_data_dir: Path) -> None:
        document = self._load()
        token = document.get("data_migration_token")
        configured = document.get("data_dir")
        if not token or not configured:
            return
        if Path(str(configured)).resolve() != active_data_dir.resolve():
            return
        marker = active_data_dir / ".jingzhi-migration-complete"
        try:
            if marker.read_text(encoding="utf-8") != str(token):
                return
        except OSError:
            return
        pending = document.get("delete_data_dir_on_restart")
        if pending:
            old_dir = Path(str(pending)).resolve()
            if old_dir != active_data_dir.resolve():
                try:
                    if old_dir.exists():
                        shutil.rmtree(old_dir)
                except OSError:
                    return
        marker.unlink(missing_ok=True)
        document.pop("delete_data_dir_on_restart", None)
        document.pop("data_migration_token", None)
        self._save_document(document)

    def update(
        self,
        *,
        data_dir: Path | None = None,
        model_dir: Path | None = None,
        migration_token: str | None = None,
    ) -> None:
        document = self._load()
        document.update(
            {
                "version": 1,
                "data_dir": str((data_dir or Path(str(document["data_dir"]))).resolve()),
                "model_dir": str((model_dir or Path(str(document["model_dir"]))).resolve()),
            }
        )
        if migration_token is not None:
            document["data_migration_token"] = migration_token
        self._save_document(document)

    def schedule_old_data_deletion(self, path: Path) -> None:
        document = self._load()
        document["delete_data_dir_on_restart"] = str(path.resolve())
        self._save_document(document)


class StorageManager:
    def __init__(
        self,
        paths: StoragePaths,
        settings_store: StartupSettingsStore,
        *,
        busy_reason: Callable[[], str | None] | None = None,
        model_in_use: Callable[[str], bool] | None = None,
        disk_usage: Callable[[Path], object] = shutil.disk_usage,
        writable_probe: Callable[[Path], bool] | None = None,
        copy_function: Callable[..., object] = shutil.copy2,
        activity: StorageActivity = storage_activity,
    ) -> None:
        self.paths = paths
        self.settings_store = settings_store
        self.busy_reason = busy_reason or (lambda: None)
        self.model_in_use = model_in_use or (lambda _name: False)
        self.disk_usage = disk_usage
        self.writable_probe = writable_probe or _is_writable_directory
        self.copy_function = copy_function
        self.activity = activity

    def usage(self, path: Path) -> StorageUsage:
        existing = path if path.exists() else path.parent
        free = int(self.disk_usage(existing).free)
        return StorageUsage(path.resolve(), _path_size(path), free)

    def data_usage(self) -> StorageUsage:
        return self.usage(self.paths.data_dir)

    def model_usage(self) -> StorageUsage:
        return self.usage(self.paths.model_dir)

    def _validate_target(self, target: Path, required_bytes: int) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise OSError("目标目录必须为空")
        if not self.writable_probe(target):
            raise OSError("目标目录不可写")
        free = int(self.disk_usage(target.parent).free)
        if free < required_bytes:
            raise OSError(f"目标磁盘空间不足：需要 {required_bytes} 字节，可用 {free} 字节")

    @staticmethod
    def _reject_nested_target(source: Path, target: Path) -> None:
        try:
            target.relative_to(source)
        except ValueError:
            return
        raise ValueError("目标目录不能位于当前目录内部")

    def _copy_and_verify(
        self, source: Path, staging: Path, *, excluded: frozenset[Path] = frozenset()
    ) -> None:
        staging.mkdir(parents=True)
        source_items = tuple(source.rglob("*")) if source.exists() else ()
        source_files = {
            item.relative_to(source): item
            for item in source_items
            if item.is_file() and item.relative_to(source) not in excluded
        }
        for relative, item in source_files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.copy_function(item, destination, follow_symlinks=False)
        copied_files = {
            item.relative_to(staging): item for item in staging.rglob("*") if item.is_file()
        }
        if source_files.keys() != copied_files.keys():
            raise OSError("迁移文件清单校验失败")
        for relative, source_file in source_files.items():
            copied_file = copied_files[relative]
            if source_file.stat().st_size != copied_file.stat().st_size:
                raise OSError(f"迁移文件大小校验失败：{relative}")
            if self._digest(source_file) != self._digest(copied_file):
                raise OSError(f"迁移文件内容校验失败：{relative}")

    @staticmethod
    def _digest(path: Path) -> bytes:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest()

    @staticmethod
    def _checkpoint_database(data_dir: Path) -> None:
        database = data_dir / "jingzhi.sqlite3"
        if not database.is_file():
            return
        connection = sqlite3.connect(database, timeout=10)
        try:
            busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy:
                raise RuntimeError("SQLite WAL 正在使用，无法迁移应用数据")
        finally:
            connection.close()

    @staticmethod
    def _rewrite_database_paths(database_path: Path, old_dir: Path, new_dir: Path) -> None:
        if not database_path.is_file():
            return
        connection = sqlite3.connect(database_path)
        try:
            for table in ("frames", "audio_chunks"):
                rows = connection.execute(f"SELECT id, path FROM {table}").fetchall()
                for identifier, stored_path in rows:
                    path = Path(stored_path)
                    if not path.is_absolute():
                        continue
                    try:
                        relative = path.resolve().relative_to(old_dir.resolve())
                    except ValueError:
                        continue
                    connection.execute(
                        f"UPDATE {table} SET path = ? WHERE id = ?",
                        (str(new_dir / relative), identifier),
                    )
            rows = connection.execute(
                "SELECT session_id, path FROM pending_media_deletions"
            ).fetchall()
            for session_id, stored_path in rows:
                path = Path(stored_path)
                if not path.is_absolute():
                    continue
                try:
                    relative = path.resolve().relative_to(old_dir.resolve())
                except ValueError:
                    continue
                connection.execute(
                    "UPDATE pending_media_deletions SET path = ? WHERE session_id = ?",
                    (str(new_dir / relative), session_id),
                )
            connection.commit()
        finally:
            connection.close()

    def migrate_data(self, target: Path) -> MigrationResult:
        if self.paths.data_managed_by_env:
            raise RuntimeError("应用数据目录由环境变量 STUDY_DATA_DIR 管理")
        source = self.paths.data_dir.resolve()
        target = target.expanduser().resolve()
        if target == source:
            raise ValueError("新旧应用数据目录相同")
        self._reject_nested_target(source, target)
        with self.activity.migrating(self.busy_reason):
            self._checkpoint_database(source)
            self._validate_target(target, _path_size(source))
            staging = target.parent / f".{target.name}.jingzhi-migration-{uuid.uuid4().hex}"
            token = uuid.uuid4().hex
            try:
                self._copy_and_verify(
                    source,
                    staging,
                    excluded=frozenset({Path("jingzhi.sqlite3-wal"), Path("jingzhi.sqlite3-shm")}),
                )
                self._rewrite_database_paths(staging / "jingzhi.sqlite3", source, target)
                (staging / ".jingzhi-migration-complete").write_text(token, encoding="utf-8")
                if target.exists():
                    target.rmdir()
                os.replace(staging, target)
                try:
                    self.settings_store.update(data_dir=target, migration_token=token)
                except Exception:
                    shutil.rmtree(target, ignore_errors=True)
                    raise
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return MigrationResult(source, target)

    def models(self) -> tuple[StoredModel, ...]:
        if not self.paths.model_dir.is_dir():
            return ()
        models = []
        for path in self.paths.model_dir.glob("models--*--*"):
            if not path.is_dir():
                continue
            parts = path.name.split("--", 2)
            if len(parts) != 3:
                continue
            namespace, repository = parts[1], parts[2]
            repository_id = f"{namespace}/{repository}"
            name = (
                repository.removeprefix("faster-whisper-")
                if namespace == "Systran" and repository.startswith("faster-whisper-")
                else repository_id
            )
            models.append(StoredModel(name, repository_id, path, _path_size(path)))
        return tuple(sorted(models, key=lambda item: item.name))

    def migrate_models(self, target: Path, *, move_existing: bool) -> MigrationResult:
        if self.paths.model_managed_by_env:
            raise RuntimeError("Whisper 模型目录由环境变量 HF_HOME/HF_HUB_CACHE 管理")
        source = self.paths.model_dir.resolve()
        target = target.expanduser().resolve()
        if target == source:
            raise ValueError("新旧 Whisper 模型目录相同")
        self._reject_nested_target(source, target)
        with self.activity.migrating(self.busy_reason):
            required = _path_size(source) if move_existing else 0
            self._validate_target(target, required)
            staging = target.parent / f".{target.name}.jingzhi-migration-{uuid.uuid4().hex}"
            try:
                if move_existing:
                    self._copy_and_verify(source, staging)
                else:
                    staging.mkdir(parents=True)
                if target.exists():
                    target.rmdir()
                os.replace(staging, target)
                try:
                    self.settings_store.update(model_dir=target)
                except Exception:
                    shutil.rmtree(target, ignore_errors=True)
                    raise
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return MigrationResult(source, target)

    def delete_model(self, name: str) -> None:
        model = next((item for item in self.models() if item.name == name), None)
        if model is None:
            raise FileNotFoundError(f"未找到 Whisper 模型：{name}")
        if self.model_in_use(model.repository_id):
            raise RuntimeError(f"Whisper 模型“{name}”正在使用，不能删除")
        shutil.rmtree(model.path)

    def confirm_delete_old_data(self, old_dir: Path) -> None:
        if old_dir.resolve() == self.paths.data_dir.resolve():
            self.settings_store.schedule_old_data_deletion(old_dir)
        else:
            raise ValueError("只能删除本次迁移保留的旧应用数据目录")
