from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from jingzhi.database import CROSS_SESSION_FTS_CONTENT_QUERY, Database
from jingzhi.storage import storage_reader, storage_writer

ARCHIVE_FORMAT_VERSION = 1

_PATH_COLUMNS = (
    ("frames", "path"),
    ("audio_chunks", "path"),
    ("answer_evidence", "resource_path"),
    ("session_material_evidence", "resource_path"),
    ("model_invocation_content_evidence", "resource_path"),
    ("cross_session_synthesis_evidence", "resource_path"),
    ("pending_media_deletions", "path"),
)
_CONFIG_FILES = ("provider.json", "whisper.json", "recording.json", "material.json")
_SECRET_KEYS = {"apikey", "secret", "token", "password", "credential"}

_SESSION_TABLE_QUERIES = (
    ("sessions", "id = ?"),
    ("session_notifications", "session_id = ?"),
    ("source_events", "session_id = ?"),
    ("timeline_events", "session_id = ?"),
    ("frames", "session_id = ?"),
    ("audio_chunks", "session_id = ?"),
    ("transcript_segments", "session_id = ?"),
    (
        "transcript_versions",
        "segment_id IN (SELECT id FROM transcript_segments WHERE session_id = ?)",
    ),
    ("transcript_correction_settings", "session_id = ?"),
    ("transcript_correction_runs", "session_id = ?"),
    ("whisper_runs", "session_id = ?"),
    ("questions", "session_id = ?"),
    ("question_notes", "question_id IN (SELECT id FROM questions WHERE session_id = ?)"),
    (
        "answer_versions",
        "question_id IN (SELECT id FROM questions WHERE session_id = ?)",
    ),
    (
        "answer_evidence",
        """answer_version_id IN (
            SELECT id FROM answer_versions
            WHERE question_id IN (SELECT id FROM questions WHERE session_id = ?)
        )""",
    ),
    ("model_invocations", "session_id = ?"),
    (
        "model_invocation_evidence",
        "invocation_id IN (SELECT id FROM model_invocations WHERE session_id = ?)",
    ),
    (
        "model_invocation_content_evidence",
        "invocation_id IN (SELECT id FROM model_invocations WHERE session_id = ?)",
    ),
    ("session_material_versions", "session_id = ?"),
    (
        "session_material_evidence",
        """material_version_id IN (
            SELECT id FROM session_material_versions WHERE session_id = ?
        )""",
    ),
    ("artifacts", "session_id = ?"),
    ("pending_media_deletions", "session_id = ?"),
    ("transcript_fts", "session_id = ?"),
)


class ArchiveError(RuntimeError):
    """The archive could not be created, validated, or restored safely."""


class ArchiveConflictError(ArchiveError):
    """Restoring would target existing data, so no files were changed."""


@dataclass(frozen=True, slots=True)
class RestorePreview:
    archive: Path
    target_dir: Path
    session_ids: tuple[str, ...]
    conflicting_session_ids: tuple[str, ...]
    target_is_empty: bool
    can_restore: bool
    strategy: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RestoreResult:
    target_dir: Path
    session_ids: tuple[str, ...]


class ArchiveManager:
    """Builds portable, inspectable archives without mutating the live database."""

    def __init__(
        self,
        database: Database,
        *,
        source_busy_reason: Callable[[], str | None] | None = None,
    ) -> None:
        self.database = database
        self.data_dir = database.path.parent.resolve()
        self._source_busy_reason = source_busy_reason

    @storage_reader("导出会话")
    def export_session(self, session_id: str, destination: Path) -> Path:
        self._ensure_source_idle()
        destination = _prepare_archive_destination(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.database.connect() as connection:
            snapshot = self._session_snapshot(connection, session_id)
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}.") as temporary:
            package_dir = Path(temporary)
            path_map = self._copy_session_resources(
                package_dir, snapshot, session_id, excluded_paths={destination}
            )
            transformed = _transform_snapshot(snapshot, path_map, self.data_dir)
            self._write_session_files(package_dir, transformed)
            _write_manifest(
                package_dir,
                kind="session_export",
                extra={
                    "session_id": session_id,
                    "title": transformed["sessions"][0]["title"],
                    "counts": {
                        table: len(rows)
                        for table, rows in transformed.items()
                        if isinstance(rows, list)
                    },
                },
            )
            return _zip_directory_atomic(package_dir, destination)

    @storage_reader("创建完整备份")
    def create_backup(self, destination: Path) -> Path:
        self._ensure_source_idle()
        destination = _prepare_archive_destination(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}.") as temporary:
            package_dir = Path(temporary)
            database_path = package_dir / "database" / "jingzhi.sqlite3"
            database_path.parent.mkdir(parents=True)
            self._backup_database(database_path)
            self._validate_database(database_path, None)
            path_map = self._copy_full_resources(
                package_dir, database_path, excluded_paths={destination}
            )
            _rewrite_database_paths(database_path, self.data_dir, path_map)
            self._validate_database(database_path, package_dir)
            config_files = self._copy_safe_config(package_dir)
            _write_manifest(
                package_dir,
                kind="full_backup",
                extra={
                    "source_schema_version": _database_user_version(database_path),
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "config_files": config_files,
                },
            )
            return _zip_directory_atomic(package_dir, destination)

    @storage_reader("检查完整备份")
    def preview_restore(self, archive: Path, target_dir: Path) -> RestorePreview:
        self._ensure_source_idle()
        return self._preview_restore(archive, target_dir)

    def _preview_restore(self, archive: Path, target_dir: Path) -> RestorePreview:
        manifest = _read_verified_manifest(archive, expected_kind="full_backup")
        with tempfile.TemporaryDirectory(prefix="jingzhi-restore-preview-") as temporary:
            staging_dir = Path(temporary)
            _extract_archive(archive, staging_dir)
            _validate_config_manifest(staging_dir, manifest)
            database_path = staging_dir / "database" / "jingzhi.sqlite3"
            self._validate_database(database_path, staging_dir)
            session_ids = _database_session_ids(database_path)
        target_dir = _resolve_restore_target(target_dir)
        if target_dir.exists() and not target_dir.is_dir():
            raise ArchiveError("恢复目标不是目录")
        target_is_empty = not target_dir.exists() or not any(target_dir.iterdir())
        existing_ids = _existing_session_ids(target_dir)
        conflicts = tuple(session_id for session_id in session_ids if session_id in existing_ids)
        reason: str | None = None
        if not target_is_empty:
            reason = "目标目录不是空目录；恢复策略是拒绝已有数据，避免覆盖或隐式合并。"
        elif conflicts:
            reason = "发现重复会话 ID；恢复策略是拒绝冲突，避免覆盖现有数据。"
        return RestorePreview(
            archive=Path(archive).resolve(),
            target_dir=target_dir,
            session_ids=session_ids,
            conflicting_session_ids=conflicts,
            target_is_empty=target_is_empty,
            can_restore=target_is_empty and not conflicts,
            strategy="reject_existing_data",
            reason=reason,
        )

    @storage_writer("恢复完整备份")
    def restore_backup(self, archive: Path, target_dir: Path) -> RestoreResult:
        self._ensure_source_idle()
        target_input = Path(target_dir).expanduser()
        preview = self._preview_restore(archive, target_input)
        if not preview.can_restore:
            if preview.conflicting_session_ids:
                raise ArchiveConflictError("恢复已取消：发现重复会话 ID，不会覆盖现有数据。")
            raise ArchiveError(preview.reason or "恢复目标不可用")

        target_dir = _resolve_restore_target(target_input)
        if target_dir != preview.target_dir:
            raise ArchiveConflictError("恢复已取消：目标目录在预览后发生变化。")
        target_existed = target_dir.exists()
        restore_marker = target_dir / ".jingzhi-restore-in-progress"
        marker_created = False
        with tempfile.TemporaryDirectory(prefix="jingzhi-restore-") as temporary:
            staging_dir = Path(temporary)
            manifest = _read_verified_manifest(archive, expected_kind="full_backup")
            _extract_archive(archive, staging_dir)
            _validate_config_manifest(staging_dir, manifest)
            staged_database = staging_dir / "database" / "jingzhi.sqlite3"
            self._validate_database(staged_database, staging_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_dir = _resolve_restore_target(target_input)
            if target_dir != preview.target_dir:
                raise ArchiveConflictError("恢复已取消：目标目录在预览后发生变化。")
            restore_marker = target_dir / ".jingzhi-restore-in-progress"
            created_paths: list[Path] = []
            try:
                try:
                    marker_fd = os.open(restore_marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    marker_created = True
                    os.close(marker_fd)
                except FileExistsError as exc:
                    raise ArchiveConflictError("恢复已取消：目标目录正在进行另一次恢复。") from exc
                if any(path != restore_marker for path in target_dir.iterdir()):
                    raise ArchiveConflictError("恢复已取消：目标目录在预览后发生变化。")
                staged_media_root = staging_dir / "media"
                if staged_media_root.exists():
                    for child in staged_media_root.iterdir():
                        _copy_restore_path(
                            child, target_dir / child.name, target_dir, created_paths
                        )
                for name in manifest.get("config_files", []):
                    if not isinstance(name, str) or not _is_safe_relative_name(name):
                        raise ArchiveError("配置清单包含非法路径")
                    source = staging_dir / "config" / name
                    target = target_dir / name
                    _copy_restore_file(source, target, target_dir, created_paths)
                _rewrite_archive_paths_to_target(staged_database, target_dir)
                _copy_restore_file(
                    staged_database,
                    target_dir / "jingzhi.sqlite3",
                    target_dir,
                    created_paths,
                )
                self._validate_database(target_dir / "jingzhi.sqlite3", target_dir)
            except Exception:
                for path in reversed(created_paths):
                    try:
                        _assert_restore_path_unredirected(path)
                    except ArchiveConflictError:
                        continue
                    if path.is_symlink() or not path.is_dir():
                        path.unlink(missing_ok=True)
                    else:
                        try:
                            path.rmdir()
                        except OSError:
                            pass
                raise
            finally:
                if marker_created:
                    try:
                        _assert_restore_path_unredirected(restore_marker)
                    except ArchiveConflictError:
                        pass
                    else:
                        restore_marker.unlink(missing_ok=True)
                if not target_existed and target_dir.exists():
                    try:
                        _assert_restore_path_unredirected(target_dir)
                        target_dir.rmdir()
                    except (OSError, ArchiveConflictError):
                        pass
        return RestoreResult(target_dir, preview.session_ids)

    def _ensure_source_idle(self) -> None:
        if self._source_busy_reason is None:
            return
        reason = self._source_busy_reason()
        if reason:
            raise RuntimeError(f"归档需要等待当前写入完成：{reason}")

    def _session_snapshot(
        self, connection: sqlite3.Connection, session_id: str
    ) -> dict[str, list[dict[str, Any]]]:
        session = connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            raise KeyError(f"Unknown session: {session_id}")
        snapshot: dict[str, list[dict[str, Any]]] = {}
        for table, condition in _SESSION_TABLE_QUERIES:
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE {condition}", (session_id,)
            ).fetchall()
            snapshot[table] = [dict(row) for row in rows]
        return snapshot

    def _copy_session_resources(
        self,
        package_dir: Path,
        snapshot: dict[str, list[dict[str, Any]]],
        session_id: str,
        *,
        excluded_paths: set[Path],
    ) -> dict[str, str]:
        media_dir = package_dir / "media"
        media_dir.mkdir(parents=True)
        session_root = self.data_dir / "sessions" / session_id
        _reject_source_symlinks(session_root)
        path_map: dict[str, str] = {}
        if session_root.exists():
            _copy_tree_excluding(session_root, media_dir, excluded_paths)
        for source in _snapshot_paths(snapshot, self.data_dir):
            if not source.exists():
                raise ArchiveError(f"媒体文件不存在：{source}")
            key = str(source)
            if key in path_map:
                continue
            try:
                relative = source.relative_to(session_root)
                archive_relative = Path("media") / relative
            except ValueError:
                archive_relative = Path("media") / "resources" / _resource_name(source)
            destination = package_dir / archive_relative
            _copy_resource(source, destination)
            path_map[key] = archive_relative.as_posix()
        return path_map

    def _write_session_files(
        self, package_dir: Path, snapshot: dict[str, list[dict[str, Any]]]
    ) -> None:
        _write_json(package_dir / "session.json", snapshot["sessions"][0])
        _write_json(
            package_dir / "transcripts.json",
            _transcript_document(snapshot["transcript_segments"], snapshot["transcript_versions"]),
        )
        _write_json(
            package_dir / "questions.json",
            {
                "questions": snapshot["questions"],
                "notes": snapshot["question_notes"],
                "answers": snapshot["answer_versions"],
            },
        )
        _write_json(
            package_dir / "materials.json",
            {"versions": snapshot["session_material_versions"]},
        )
        _write_json(package_dir / "records.json", snapshot)
        for material in snapshot["session_material_versions"]:
            path = (
                package_dir
                / "materials"
                / f"material-{material['id']}-v{material['version_number']}.md"
            )
            _write_text(path, str(material["content"]))
        for answer in snapshot["answer_versions"]:
            answer_text = answer.get("answer") or ""
            if answer.get("error"):
                answer_text = f"## 生成失败\n\n{answer_text}\n\n错误：{answer['error']}"
            question_id = next(
                item["id"] for item in snapshot["questions"] if item["id"] == answer["question_id"]
            )
            path = (
                package_dir / "answers" / f"question-{question_id}-v{answer['version_number']}.md"
            )
            _write_text(
                path, f"# 问题\n\n{_question_text(snapshot, question_id)}\n\n{answer_text}\n"
            )
        answer_paths = {
            int(answer["id"]): (
                f"answers/question-{answer['question_id']}-v{answer['version_number']}.md"
            )
            for answer in snapshot["answer_versions"]
        }
        material_paths = {
            int(material["id"]): (
                f"materials/material-{material['id']}-v{material['version_number']}.md"
            )
            for material in snapshot["session_material_versions"]
        }
        _write_json(
            package_dir / "evidence" / "index.json",
            _evidence_document(snapshot, answer_paths, material_paths),
        )

    def _backup_database(self, destination: Path) -> None:
        source = sqlite3.connect(self.database.path, timeout=10)
        target = sqlite3.connect(destination, timeout=10)
        try:
            source.backup(target)
            target.commit()
        finally:
            target.close()
            source.close()

    def _copy_full_resources(
        self,
        package_dir: Path,
        database_path: Path,
        *,
        excluded_paths: set[Path],
    ) -> dict[str, str]:
        media_dir = package_dir / "media"
        media_dir.mkdir(parents=True)
        sessions_source = self.data_dir / "sessions"
        if sessions_source.exists():
            _copy_tree_excluding(sessions_source, media_dir / "sessions", excluded_paths)
        path_map: dict[str, str] = {}
        for source in _database_paths(database_path, self.data_dir):
            if not source.exists():
                raise ArchiveError(f"媒体文件不存在：{source}")
            key = str(source)
            if key in path_map:
                continue
            try:
                relative = source.relative_to(self.data_dir)
                archive_relative = Path("media") / relative
            except ValueError:
                archive_relative = Path("media") / "resources" / _resource_name(source)
            _copy_resource(source, package_dir / archive_relative)
            path_map[key] = archive_relative.as_posix()
        return path_map

    def _copy_safe_config(self, package_dir: Path) -> list[str]:
        config_dir = package_dir / "config"
        config_dir.mkdir(parents=True)
        included: list[str] = []
        for name in _CONFIG_FILES:
            source = self.data_dir / name
            if not source.is_file():
                continue
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArchiveError(f"配置文件无法读取：{name}") from exc
            _write_json(config_dir / name, _strip_secrets(value))
            included.append(name)
        _write_json(
            config_dir / "manifest.json",
            {
                "format_version": ARCHIVE_FORMAT_VERSION,
                "files": included,
                "secrets_excluded": ["api_key"],
            },
        )
        return included

    def _validate_database(self, database_path: Path, media_root: Path | None) -> None:
        connection = sqlite3.connect(database_path)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ArchiveError(f"数据库完整性校验失败：{integrity}")
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_keys:
                raise ArchiveError("数据库存在无效外键关系")
            segment_rows = connection.execute(
                "SELECT id, session_id, text FROM transcript_segments"
            ).fetchall()
            fts_rows = connection.execute(
                "SELECT segment_id, session_id, text FROM transcript_fts"
            ).fetchall()
            expected_fts = Counter((str(row[0]), str(row[1]), str(row[2])) for row in segment_rows)
            actual_fts = Counter((str(row[0]), str(row[1]), str(row[2])) for row in fts_rows)
            if actual_fts != expected_fts:
                raise ArchiveError("全文索引与字幕片段不一致")
            expected_cross_fts = Counter(
                tuple(str(value) if value is not None else "" for value in row)
                for row in connection.execute(CROSS_SESSION_FTS_CONTENT_QUERY)
            )
            actual_cross_fts = Counter(
                tuple(str(value) if value is not None else "" for value in row)
                for row in connection.execute(
                    """SELECT entry_id, session_id, kind, source_id, source, start_ms,
                              end_ms, version_kind, text
                       FROM cross_session_fts"""
                )
            )
            if actual_cross_fts != expected_cross_fts:
                raise ArchiveError("跨会话全文索引与内容不一致")
            if media_root is not None:
                _validate_database_paths(connection, media_root)
        except sqlite3.DatabaseError as exc:
            raise ArchiveError(f"数据库无法读取：{exc}") from exc
        finally:
            connection.close()


def _transcript_document(
    segments: list[dict[str, Any]], versions: list[dict[str, Any]]
) -> dict[str, Any]:
    by_segment: dict[int, list[dict[str, Any]]] = {}
    for version in versions:
        by_segment.setdefault(int(version["segment_id"]), []).append(version)
    return {
        "segments": [
            dict(segment, versions=by_segment.get(int(segment["id"]), [])) for segment in segments
        ]
    }


def _question_text(snapshot: dict[str, list[dict[str, Any]]], question_id: int) -> str:
    return next(item["question"] for item in snapshot["questions"] if item["id"] == question_id)


def _evidence_document(
    snapshot: dict[str, list[dict[str, Any]]],
    answer_paths: dict[int, str],
    material_paths: dict[int, str],
) -> dict[str, Any]:
    frame_paths = {int(item["id"]): str(item.get("path", "")) for item in snapshot["frames"]}
    entries: list[dict[str, Any]] = []
    for row in snapshot["answer_evidence"]:
        answer_id = int(row["answer_version_id"])
        entries.append(
            _evidence_entry("answer", answer_id, row, frame_paths, answer_paths[answer_id])
        )
    for row in snapshot["session_material_evidence"]:
        material_id = int(row["material_version_id"])
        entries.append(
            _evidence_entry("material", material_id, row, frame_paths, material_paths[material_id])
        )
    return {"entries": entries}


def _evidence_entry(
    owner: str,
    owner_id: int,
    row: dict[str, Any],
    frame_paths: dict[int, str],
    document_path: str,
) -> dict[str, Any]:
    entry = {
        "owner": owner,
        "owner_id": owner_id,
        "document_path": document_path,
        "ordinal": row["ordinal"],
        "stable_id": row["stable_id"],
        "kind": row["kind"],
        "source": row["source"],
        "start_ms": row["start_ms"],
        "end_ms": row["end_ms"],
        "transcript_version_id": row.get("transcript_version_id"),
        "frame_id": row.get("frame_id"),
        "relative_path": row.get("resource_path") or "transcripts.json",
    }
    if row.get("frame_id") is not None and not row.get("resource_path"):
        entry["relative_path"] = frame_paths.get(int(row["frame_id"]), "media/")
    return entry


def _transform_snapshot(
    snapshot: dict[str, list[dict[str, Any]]], path_map: dict[str, str], data_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    transformed = {table: [dict(row) for row in rows] for table, rows in snapshot.items()}
    for table, column in _PATH_COLUMNS:
        for row in transformed.get(table, []):
            value = row.get(column)
            if value:
                source = _resolve_stored_path(str(value), data_dir)
                try:
                    row[column] = path_map[str(source)]
                except KeyError as exc:
                    raise ArchiveError(f"媒体路径未纳入归档：{source}") from exc
    return transformed


def _snapshot_paths(snapshot: dict[str, list[dict[str, Any]]], data_dir: Path) -> set[Path]:
    paths: set[Path] = set()
    for table, column in _PATH_COLUMNS:
        for row in snapshot.get(table, []):
            value = row.get(column)
            if value:
                paths.add(_resolve_stored_path(str(value), data_dir))
    return paths


def _database_paths(database_path: Path, data_dir: Path) -> set[Path]:
    paths: set[Path] = set()
    connection = sqlite3.connect(database_path)
    try:
        for table, column in _PATH_COLUMNS:
            rows = connection.execute(_path_rows_query(table, column)).fetchall()
            for (value,) in rows:
                paths.add(_resolve_stored_path(str(value), data_dir))
    finally:
        connection.close()
    return paths


def _path_rows_query(table: str, column: str, *, with_rowid: bool = False) -> str:
    if table == "cross_session_synthesis_evidence":
        prefix = "evidence.rowid, " if with_rowid else ""
        return (
            f"SELECT {prefix}evidence.{column} "
            "FROM cross_session_synthesis_evidence AS evidence "
            "JOIN cross_session_syntheses AS synthesis "
            "ON synthesis.id = evidence.synthesis_id "
            "WHERE evidence.resource_path IS NOT NULL AND synthesis.evidence_state = 'exact'"
        )
    prefix = "rowid, " if with_rowid else ""
    return f"SELECT {prefix}{column} FROM {table} WHERE {column} IS NOT NULL"


def _prepare_archive_destination(destination: Path) -> Path:
    candidate = Path(destination).expanduser()
    current = candidate
    while True:
        if current.is_symlink():
            raise ArchiveError("归档目标不能是符号链接")
        parent = current.parent
        if parent == current:
            break
        current = parent
    return candidate.absolute()


def _resolve_stored_path(value: str, data_dir: Path) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else data_dir / path
    _reject_source_symlinks(candidate)
    return candidate.resolve()


def _resolve_restore_target(target_dir: Path) -> Path:
    candidate = Path(target_dir).expanduser()
    current = candidate
    while True:
        if current.is_symlink():
            raise ArchiveError("恢复目标不能是符号链接")
        parent = current.parent
        if parent == current:
            break
        current = parent
    return candidate.resolve()


def _reject_source_symlinks(source: Path) -> None:
    current = source
    while True:
        if current.is_symlink():
            raise ArchiveError(f"媒体归档不允许符号链接：{current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not source.is_dir():
        return
    for directory, dirnames, filenames in os.walk(source, followlinks=False):
        for name in (*dirnames, *filenames):
            candidate = Path(directory) / name
            if candidate.is_symlink():
                raise ArchiveError(f"媒体归档不允许符号链接：{candidate}")


def _resource_name(source: Path) -> str:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{source.name}"


def _copy_resource(source: Path, destination: Path) -> None:
    _reject_source_symlinks(source)
    if source.is_dir():
        _copy_source_tree(source, destination, excluded_paths=set())
    else:
        _copy_source_file(source, destination)


def _copy_tree_excluding(source: Path, destination: Path, excluded_paths: set[Path]) -> None:
    _copy_source_tree(source, destination, excluded_paths=excluded_paths)


def _copy_source_tree(source: Path, destination: Path, excluded_paths: set[Path]) -> None:
    _reject_source_symlinks(source)
    excluded = {path.resolve() for path in excluded_paths}
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        _reject_source_symlinks(child)
        if child.resolve() in excluded:
            continue
        target = destination / child.name
        if child.is_dir():
            _copy_source_tree(child, target, excluded)
        elif child.is_file():
            _copy_source_file(child, target)
        else:
            raise ArchiveError(f"无法归档媒体文件：{child}")


def _copy_source_file(source: Path, destination: Path) -> None:
    descriptor = _open_source_file(source)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = os.fdopen(descriptor, "rb")
        descriptor = -1
        with original, destination.open("wb") as target:
            shutil.copyfileobj(original, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_source_file(source: Path) -> int:
    _reject_source_symlinks(source)
    lexical_path = os.path.abspath(os.fspath(source))
    real_before = os.path.realpath(source)
    if os.path.normcase(real_before) != os.path.normcase(lexical_path):
        raise ArchiveError(f"媒体归档不允许符号链接：{source}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ArchiveError(f"媒体文件无法读取：{source}") from exc
    try:
        real_after = os.path.realpath(source)
        path_stat = os.stat(source, follow_symlinks=False)
        descriptor_stat = os.fstat(descriptor)
        if (
            os.path.normcase(real_after) != os.path.normcase(real_before)
            or stat.S_ISLNK(path_stat.st_mode)
            or (
                path_stat.st_ino
                and descriptor_stat.st_ino
                and path_stat.st_ino != descriptor_stat.st_ino
            )
        ):
            raise ArchiveError(f"媒体归档不允许符号链接：{source}")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _copy_restore_path(source: Path, target: Path, root: Path, created_paths: list[Path]) -> None:
    if source.is_symlink():
        raise ArchiveError(f"归档不允许符号链接：{source}")
    if source.is_dir():
        _ensure_restore_parent(target.parent, root, created_paths)
        if target.exists() or target.is_symlink():
            raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。")
        try:
            target.mkdir()
            _assert_restore_path_unredirected(target)
        except FileExistsError as exc:
            raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。") from exc
        except ArchiveConflictError:
            raise
        created_paths.append(target)
        for child in source.iterdir():
            _copy_restore_path(child, target / child.name, root, created_paths)
        return
    if source.is_file():
        _copy_restore_file(source, target, root, created_paths)
        return
    raise ArchiveError(f"归档资源不存在：{source}")


def _copy_restore_file(source: Path, target: Path, root: Path, created_paths: list[Path]) -> None:
    if source.is_symlink() or not source.is_file():
        raise ArchiveError(f"归档资源不存在：{source}")
    _ensure_restore_parent(target.parent, root, created_paths)
    if target.exists() or target.is_symlink():
        raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。")
    parent_real_before = _assert_restore_path_unredirected(target.parent)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(target, flags)
    except FileExistsError as exc:
        raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。") from exc
    try:
        if _assert_restore_path_unredirected(target.parent) != parent_real_before:
            os.close(descriptor)
            target.unlink(missing_ok=True)
            raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。")
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        target.unlink(missing_ok=True)
        raise
    created_paths.append(target)
    try:
        with os.fdopen(descriptor, "wb") as destination, source.open("rb") as original:
            shutil.copyfileobj(original, destination)
        shutil.copystat(source, target)
    except Exception:
        target.unlink(missing_ok=True)
        created_paths.remove(target)
        raise


def _assert_restore_path_unredirected(path: Path) -> str:
    lexical_path = os.path.abspath(os.fspath(path))
    real_path = os.path.realpath(path)
    if os.path.normcase(real_path) != os.path.normcase(lexical_path):
        raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。")
    return real_path


def _ensure_restore_parent(parent: Path, root: Path, created_paths: list[Path]) -> None:
    _assert_restore_path_unredirected(parent)
    relative = parent.relative_to(root)
    current = root
    for part in relative.parts:
        candidate = current / part
        _assert_restore_path_unredirected(candidate)
        if candidate.is_symlink():
            raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。")
        if candidate.exists():
            if not candidate.is_dir():
                raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。")
        else:
            try:
                candidate.mkdir()
            except FileExistsError:
                if candidate.is_symlink() or not candidate.is_dir():
                    raise ArchiveConflictError("恢复已取消：目标目录在恢复时发生变化。") from None
            else:
                created_paths.append(candidate)
        current = candidate


def _rewrite_database_paths(database_path: Path, data_dir: Path, path_map: dict[str, str]) -> None:
    connection = sqlite3.connect(database_path)
    try:
        for table, column in _PATH_COLUMNS:
            rows = connection.execute(_path_rows_query(table, column, with_rowid=True)).fetchall()
            for row_id, value in rows:
                source = _resolve_stored_path(str(value), data_dir)
                try:
                    relative = path_map[str(source)]
                except KeyError as exc:
                    raise ArchiveError(f"数据库媒体路径未纳入归档：{source}") from exc
                connection.execute(
                    f"UPDATE {table} SET {column} = ? WHERE rowid = ?", (relative, row_id)
                )
        connection.commit()
    finally:
        connection.close()


def _rewrite_archive_paths_to_target(database_path: Path, target_dir: Path) -> None:
    connection = sqlite3.connect(database_path)
    try:
        for table, column in _PATH_COLUMNS:
            rows = connection.execute(_path_rows_query(table, column, with_rowid=True)).fetchall()
            for row_id, value in rows:
                relative = _safe_archive_path(str(value))
                if not relative.parts or relative.parts[0] != "media":
                    raise ArchiveError(f"数据库包含非归档媒体路径：{value}")
                restored = target_dir.joinpath(*relative.parts[1:])
                connection.execute(
                    f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                    (str(restored), row_id),
                )
        connection.commit()
    finally:
        connection.close()


def _validate_database_paths(connection: sqlite3.Connection, media_root: Path) -> None:
    for table, column in _PATH_COLUMNS:
        rows = connection.execute(_path_rows_query(table, column)).fetchall()
        for (value,) in rows:
            path = Path(value)
            if not path.is_absolute():
                path = media_root / path
            if not path.exists():
                raise ArchiveError(f"媒体文件不存在：{path}")


def _database_session_ids(database_path: Path) -> tuple[str, ...]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute("SELECT id FROM sessions ORDER BY id").fetchall()
    finally:
        connection.close()
    return tuple(str(row[0]) for row in rows)


def _existing_session_ids(target_dir: Path) -> set[str]:
    database_path = target_dir / "jingzhi.sqlite3"
    if not database_path.is_file():
        return set()
    try:
        return set(_database_session_ids(database_path))
    except sqlite3.DatabaseError as exc:
        raise ArchiveError(f"目标目录中的数据库无法读取：{exc}") from exc


def _database_user_version(database_path: Path) -> int:
    connection = sqlite3.connect(database_path)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


def _strip_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized = "".join(character for character in str(key).lower() if character.isalnum())
            if normalized in _SECRET_KEYS or any(
                marker in normalized for marker in ("apikey", "secret", "token", "password")
            ):
                continue
            clean[str(key)] = _strip_secrets(item)
        return clean
    if isinstance(value, list):
        return [_strip_secrets(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_manifest(package_dir: Path, *, kind: str, extra: dict[str, Any]) -> None:
    files = {
        path.relative_to(package_dir).as_posix(): _sha256(path)
        for path in package_dir.rglob("*")
        if path.is_file() and path != package_dir / "manifest.json"
    }
    manifest = {
        "format_version": ARCHIVE_FORMAT_VERSION,
        "kind": kind,
        "files": dict(sorted(files.items())),
        **extra,
    }
    _write_json(package_dir / "manifest.json", manifest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_directory_atomic(package_dir: Path, destination: Path) -> Path:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as package:
            for path in sorted(package_dir.rglob("*")):
                if path.is_file():
                    package.write(path, path.relative_to(package_dir).as_posix())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _read_verified_manifest(archive: Path, *, expected_kind: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(archive) as package:
            _validate_zip_members(package)
            try:
                manifest = json.loads(package.read("manifest.json"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArchiveError("归档缺少有效 manifest.json") from exc
            if not isinstance(manifest, dict):
                raise ArchiveError("归档清单格式无效")
            if manifest.get("format_version") != ARCHIVE_FORMAT_VERSION:
                raise ArchiveError("归档格式版本不受支持")
            if manifest.get("kind") != expected_kind:
                raise ArchiveError("归档类型与操作不匹配")
            files = manifest.get("files")
            if not isinstance(files, dict):
                raise ArchiveError("归档清单缺少文件校验表")
            names = set(package.namelist())
            if set(files) != names - {"manifest.json"}:
                raise ArchiveError("归档文件校验表与 ZIP 内容不一致")
            for name, expected_hash in files.items():
                if not isinstance(name, str) or not isinstance(expected_hash, str):
                    raise ArchiveError("归档文件校验表格式无效")
                if name not in names:
                    raise ArchiveError(f"归档缺少文件：{name}")
                with package.open(name) as stream:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ArchiveError(f"归档文件校验失败：{name}")
            return manifest
    except zipfile.BadZipFile as exc:
        raise ArchiveError("归档不是有效 ZIP 文件") from exc


def _validate_config_manifest(staging_dir: Path, manifest: dict[str, Any]) -> None:
    path = staging_dir / "config" / "manifest.json"
    try:
        config_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveError("备份缺少有效配置清单") from exc
    if not isinstance(config_manifest, dict):
        raise ArchiveError("配置清单格式无效")
    files = config_manifest.get("files")
    expected_files = manifest.get("config_files")
    if (
        config_manifest.get("format_version") != ARCHIVE_FORMAT_VERSION
        or not isinstance(files, list)
        or not isinstance(expected_files, list)
        or files != expected_files
    ):
        raise ArchiveError("配置清单与备份清单不一致")
    for name in files:
        if not isinstance(name, str) or not _is_safe_relative_name(name):
            raise ArchiveError("配置清单包含非法路径")
        config_path = staging_dir / "config" / name
        if not config_path.is_file():
            raise ArchiveError(f"备份缺少配置文件：{name}")
        try:
            json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArchiveError(f"配置文件格式无效：{name}") from exc


def _extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as package:
        _validate_zip_members(package)
        package.extractall(destination)


def _validate_zip_members(package: zipfile.ZipFile) -> None:
    seen: set[str] = set()
    for info in package.infolist():
        name = _normalise_zip_name(info.filename)
        if name in seen:
            raise ArchiveError(f"归档包含重复路径：{name}")
        seen.add(name)
        if _zip_is_symlink(info):
            raise ArchiveError(f"归档不允许符号链接：{name}")
        if not _is_safe_relative_name(name):
            raise ArchiveError(f"非法归档路径：{info.filename}")


def _normalise_zip_name(name: str) -> str:
    return name.replace("\\", "/")


def _safe_archive_path(value: str) -> PurePosixPath:
    normalised = _normalise_zip_name(value)
    if not _is_safe_relative_name(normalised):
        raise ArchiveError(f"非法归档媒体路径：{value}")
    return PurePosixPath(normalised)


def _is_safe_relative_name(name: str) -> bool:
    normalised = name.replace("\\", "/")
    if not normalised or normalised.startswith("/"):
        return False
    if len(normalised) >= 2 and normalised[1] == ":":
        return False
    path = PurePosixPath(normalised.rstrip("/"))
    return (
        bool(path.parts)
        and ".." not in path.parts
        and all(part not in {"", "."} for part in path.parts)
    )


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000
