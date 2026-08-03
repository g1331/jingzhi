from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jingzhi.transcript_correction import CORRECTION_WINDOW_MS

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    ended_at_utc TEXT,
    status TEXT NOT NULL CHECK (status IN ('recording', 'complete', 'interrupted'))
);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL DEFAULT 'display:primary',
    ts_ms INTEGER NOT NULL CHECK (ts_ms >= 0),
    path TEXT NOT NULL,
    perceptual_hash TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_session_time ON frames(session_id, ts_ms);

CREATE TABLE IF NOT EXISTS audio_chunks (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('system', 'microphone')),
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    path TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'transcribed', 'failed')),
    error TEXT
);
CREATE INDEX IF NOT EXISTS audio_session_time ON audio_chunks(session_id, start_ms);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    audio_chunk_id INTEGER REFERENCES audio_chunks(id) ON DELETE SET NULL,
    source TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    language TEXT,
    confidence REAL
);
CREATE INDEX IF NOT EXISTS transcript_session_time
ON transcript_segments(session_id, start_ms, end_ms);

CREATE TABLE IF NOT EXISTS transcript_versions (
    id INTEGER PRIMARY KEY,
    segment_id INTEGER NOT NULL REFERENCES transcript_segments(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('original', 'correction', 'user_edit')),
    text TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    model TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
);
CREATE UNIQUE INDEX IF NOT EXISTS transcript_original_version
ON transcript_versions(segment_id) WHERE kind = 'original';
CREATE INDEX IF NOT EXISTS transcript_version_history
ON transcript_versions(segment_id, id);

CREATE VIEW IF NOT EXISTS effective_transcript_versions AS
SELECT id, segment_id, kind, text, created_at_utc, model, active
FROM (
    SELECT version.*,
           ROW_NUMBER() OVER (
               PARTITION BY segment_id
               ORDER BY CASE kind
                   WHEN 'user_edit' THEN 3
                   WHEN 'correction' THEN 2
                   ELSE 1 END DESC,
                   id DESC
           ) AS priority_rank
    FROM transcript_versions AS version
    WHERE active = 1
)
WHERE priority_rank = 1;

CREATE TABLE IF NOT EXISTS transcript_correction_settings (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    window_ms INTEGER NOT NULL CHECK (window_ms IN (15000, 30000, 60000))
);

CREATE TABLE IF NOT EXISTS transcript_correction_runs (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    window_start_ms INTEGER NOT NULL,
    window_end_ms INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('running', 'corrected', 'failed')),
    model TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    error_source TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS transcript_correction_run_window
ON transcript_correction_runs(session_id, window_start_ms, window_end_ms, id);

CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    segment_id UNINDEXED, session_id UNINDEXED, text, tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    asked_at_ms INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    context_start_ms INTEGER,
    context_end_ms INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('summary', 'knowledge_points', 'mistakes')),
    created_at_utc TEXT NOT NULL,
    content_json TEXT NOT NULL,
    model TEXT
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at_utc TEXT NOT NULL
);
"""

LATEST_SCHEMA_VERSION = 3

SESSION_SUMMARY_QUERY = """
SELECT
    sessions.id,
    sessions.title,
    sessions.started_at_utc,
    sessions.ended_at_utc,
    sessions.status,
    (SELECT COUNT(*) FROM frames WHERE session_id = sessions.id) AS frame_count,
    COALESCE((SELECT MAX(ts_ms) FROM frames WHERE session_id = sessions.id), 0)
        AS last_frame_ms,
    COALESCE(
        (SELECT MAX(end_ms) FROM transcript_segments WHERE session_id = sessions.id), 0
    ) AS last_transcript_ms,
    COALESCE((SELECT MAX(asked_at_ms) FROM questions WHERE session_id = sessions.id), 0)
        AS last_question_ms
FROM sessions
"""


@dataclass(frozen=True, slots=True)
class FrameRecord:
    id: int
    ts_ms: int
    path: Path


@dataclass(frozen=True, slots=True)
class TimelineFrameRecord:
    id: int
    session_id: str
    source_id: str
    ts_ms: int
    path: Path
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class TimelineTranscriptRecord:
    id: int
    source: str
    start_ms: int
    end_ms: int
    text: str
    correction_state: str | None = None
    version_id: int | None = None
    version_kind: str = "original"
    original_text: str = ""


@dataclass(frozen=True, slots=True)
class TimelineQuestionRecord:
    id: int
    asked_at_ms: int
    question: str


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    title: str
    started_at_utc: str
    ended_at_utc: str | None
    status: str
    duration_ms: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    start_ms: int
    end_ms: int
    source: str
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptVersionRecord:
    id: int
    segment_id: int
    kind: str
    text: str
    created_at_utc: str
    model: str | None
    active: bool


@dataclass(frozen=True, slots=True)
class CorrectionSegmentRecord:
    id: int
    start_ms: int
    end_ms: int
    source: str
    text: str


@dataclass(frozen=True, slots=True)
class CorrectionFrameRecord:
    id: int
    source_id: str
    ts_ms: int
    path: Path


@dataclass(frozen=True, slots=True)
class TranscriptCorrectionSettingsRecord:
    enabled: bool
    window_ms: int


@dataclass(frozen=True, slots=True)
class TranscriptCorrectionRunRecord:
    id: int
    state: str
    error_source: str | None
    error: str | None


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            self.migrate(connection)

    def migrate(self, connection: sqlite3.Connection) -> None:
        """Bring both fresh and pre-versioned MVP databases to the latest schema."""
        connection.executescript(SCHEMA)
        frame_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(frames)").fetchall()
        }
        if "source_id" not in frame_columns:
            connection.execute(
                "ALTER TABLE frames ADD COLUMN source_id TEXT NOT NULL DEFAULT 'display:primary'"
            )
        applied_at = datetime.now(UTC).isoformat()
        connection.execute(
            """INSERT OR IGNORE INTO transcript_versions(
                   segment_id, kind, text, created_at_utc, active
               )
               SELECT id, 'original', text, ?, 1 FROM transcript_segments""",
            (applied_at,),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
            [(version, applied_at) for version in range(1, LATEST_SCHEMA_VERSION + 1)],
        )
        connection.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION}")

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def create_session(self, title: str, started_at_utc: str) -> str:
        session_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO sessions(id, title, started_at_utc, status) VALUES (?, ?, ?, ?)",
                (session_id, title, started_at_utc, "recording"),
            )
        return session_id

    def finish_session(self, session_id: str, ended_at_utc: str, status: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE sessions SET ended_at_utc = ?, status = ? WHERE id = ?",
                (ended_at_utc, status, session_id),
            )

    def add_frame(
        self,
        session_id: str,
        ts_ms: int,
        path: Path,
        image_hash: str,
        size: tuple[int, int],
        *,
        source_id: str = "display:primary",
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO frames(
                       session_id, source_id, ts_ms, path, perceptual_hash, width, height
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, source_id, ts_ms, str(path), image_hash, size[0], size[1]),
            )
            return int(cursor.lastrowid)

    def list_sessions(self) -> list[SessionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                SESSION_SUMMARY_QUERY + " ORDER BY sessions.started_at_utc DESC, sessions.id"
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                SESSION_SUMMARY_QUERY + " WHERE sessions.id = ?",
                (session_id,),
            ).fetchone()
        return self._session_from_row(row) if row is not None else None

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> SessionRecord:
        stored_duration_ms = 0
        if row["ended_at_utc"]:
            started = datetime.fromisoformat(row["started_at_utc"])
            ended = datetime.fromisoformat(row["ended_at_utc"])
            stored_duration_ms = max(0, int((ended - started).total_seconds() * 1000))
        duration_ms = max(
            stored_duration_ms,
            int(row["last_frame_ms"]),
            int(row["last_transcript_ms"]),
            int(row["last_question_ms"]),
        )
        return SessionRecord(
            id=row["id"],
            title=row["title"],
            started_at_utc=row["started_at_utc"],
            ended_at_utc=row["ended_at_utc"],
            status=row["status"],
            duration_ms=duration_ms,
            frame_count=int(row["frame_count"]),
        )

    def timeline_frames(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[TimelineFrameRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, session_id, source_id, ts_ms, path, width, height
                   FROM frames
                   WHERE session_id = ? AND ts_ms BETWEEN ? AND ?
                   ORDER BY ts_ms, id""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        return [
            TimelineFrameRecord(
                id=row["id"],
                session_id=row["session_id"],
                source_id=row["source_id"],
                ts_ms=row["ts_ms"],
                path=Path(row["path"]),
                width=row["width"],
                height=row["height"],
            )
            for row in rows
        ]

    def timeline_transcripts(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[TimelineTranscriptRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT segment.id, segment.source, segment.start_ms, segment.end_ms,
                          effective.text, effective.id AS version_id,
                          effective.kind AS version_kind, original.text AS original_text,
                          CASE
                              WHEN settings.enabled IS NULL OR settings.enabled = 0 THEN NULL
                              WHEN effective.kind = 'user_edit' THEN 'edited'
                              WHEN effective.kind = 'correction' THEN 'corrected'
                              ELSE 'pending'
                          END AS correction_state
                   FROM transcript_segments AS segment
                   JOIN transcript_versions AS original
                     ON original.segment_id = segment.id AND original.kind = 'original'
                   JOIN effective_transcript_versions AS effective
                     ON effective.segment_id = segment.id
                   LEFT JOIN transcript_correction_settings AS settings
                     ON settings.session_id = segment.session_id
                   WHERE segment.session_id = ?
                     AND segment.end_ms >= ? AND segment.start_ms <= ?
                   ORDER BY segment.start_ms, segment.id""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        return [TimelineTranscriptRecord(**dict(row)) for row in rows]

    def recognizing_transcripts(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[TimelineTranscriptRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, source, start_ms, end_ms FROM audio_chunks
                   WHERE session_id = ? AND state = 'pending'
                     AND end_ms >= ? AND start_ms <= ?
                   ORDER BY start_ms, id""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        return [
            TimelineTranscriptRecord(
                id=-int(row["id"]),
                source=row["source"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                text="正在识别…",
                correction_state="recognizing",
                version_kind="recognizing",
            )
            for row in rows
        ]

    def timeline_questions(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[TimelineQuestionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, asked_at_ms, question
                   FROM questions
                   WHERE session_id = ? AND asked_at_ms BETWEEN ? AND ?
                   ORDER BY asked_at_ms, id""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        return [TimelineQuestionRecord(**dict(row)) for row in rows]

    def add_audio_chunk(
        self, session_id: str, source: str, start_ms: int, end_ms: int, path: Path
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO audio_chunks(session_id, source, start_ms, end_ms, path, state)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                (session_id, source, start_ms, end_ms, str(path)),
            )
            return int(cursor.lastrowid)

    def add_transcript(
        self,
        session_id: str,
        chunk_id: int,
        source: str,
        start_ms: int,
        end_ms: int,
        text: str,
        language: str | None,
        confidence: float | None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO transcript_segments(
                       session_id, audio_chunk_id, source, start_ms, end_ms,
                       text, language, confidence
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, chunk_id, source, start_ms, end_ms, text, language, confidence),
            )
            segment_id = int(cursor.lastrowid)
            connection.execute(
                """INSERT INTO transcript_versions(
                       segment_id, kind, text, created_at_utc, active
                   ) VALUES (?, 'original', ?, ?, 1)""",
                (segment_id, text, datetime.now(UTC).isoformat()),
            )
            connection.execute(
                "INSERT INTO transcript_fts(segment_id, session_id, text) VALUES (?, ?, ?)",
                (segment_id, session_id, text),
            )
            return segment_id

    def set_chunk_state(self, chunk_id: int, state: str, error: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE audio_chunks SET state = ?, error = ? WHERE id = ?",
                (state, error, chunk_id),
            )

    def transcripts_between(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[TranscriptRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT start_ms, end_ms, source, text FROM transcript_segments
                   WHERE session_id = ? AND end_ms >= ? AND start_ms <= ?
                   ORDER BY start_ms""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        return [TranscriptRecord(**dict(row)) for row in rows]

    def nearest_frames(
        self, session_id: str, center_ms: int, start_ms: int, end_ms: int, limit: int = 4
    ) -> list[FrameRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, ts_ms, path FROM frames
                   WHERE session_id = ? AND ts_ms BETWEEN ? AND ?
                   ORDER BY abs(ts_ms - ?) LIMIT ?""",
                (session_id, start_ms, end_ms, center_ms, limit),
            ).fetchall()
        return [
            FrameRecord(id=row["id"], ts_ms=row["ts_ms"], path=Path(row["path"])) for row in rows
        ]

    def recent_transcripts(self, session_id: str, limit: int = 80) -> list[TranscriptRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT start_ms, end_ms, source, text FROM transcript_segments
                   WHERE session_id = ? ORDER BY start_ms DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [TranscriptRecord(**dict(row)) for row in reversed(rows)]

    def all_transcripts(self, session_id: str) -> list[TranscriptRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT start_ms, end_ms, source, text FROM transcript_segments
                   WHERE session_id = ? ORDER BY start_ms""",
                (session_id,),
            ).fetchall()
        return [TranscriptRecord(**dict(row)) for row in rows]

    def transcript_versions(self, segment_id: int) -> list[TranscriptVersionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, segment_id, kind, text, created_at_utc, model, active
                   FROM transcript_versions WHERE segment_id = ? ORDER BY id""",
                (segment_id,),
            ).fetchall()
        return [
            TranscriptVersionRecord(
                id=row["id"],
                segment_id=row["segment_id"],
                kind=row["kind"],
                text=row["text"],
                created_at_utc=row["created_at_utc"],
                model=row["model"],
                active=bool(row["active"]),
            )
            for row in rows
        ]

    def add_transcript_version(
        self, segment_id: int, kind: str, text: str, *, model: str | None = None
    ) -> int | None:
        if kind not in {"correction", "user_edit"}:
            raise ValueError(f"Unsupported transcript version kind: {kind}")
        text = text.strip()
        if not text:
            raise ValueError("Transcript text is required")
        with self.connect() as connection:
            if kind == "correction":
                user_edit = connection.execute(
                    """SELECT 1 FROM transcript_versions
                       WHERE segment_id = ? AND kind = 'user_edit' AND active = 1""",
                    (segment_id,),
                ).fetchone()
                if user_edit is not None:
                    return None
                duplicate = connection.execute(
                    """SELECT id FROM transcript_versions
                       WHERE segment_id = ? AND kind = 'correction'
                         AND text = ? AND active = 1
                       ORDER BY id DESC LIMIT 1""",
                    (segment_id, text),
                ).fetchone()
                if duplicate is not None:
                    return int(duplicate["id"])
            cursor = connection.execute(
                """INSERT INTO transcript_versions(
                       segment_id, kind, text, created_at_utc, model, active
                   ) VALUES (?, ?, ?, ?, ?, 1)""",
                (segment_id, kind, text, datetime.now(UTC).isoformat(), model),
            )
            return int(cursor.lastrowid)

    def undo_transcript_correction(self, segment_id: int) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE transcript_versions SET active = 0
                   WHERE segment_id = ? AND kind = 'correction'""",
                (segment_id,),
            )

    def configure_transcript_correction(
        self, session_id: str, *, enabled: bool, window_ms: int
    ) -> None:
        if window_ms not in CORRECTION_WINDOW_MS:
            raise ValueError("Correction window must be 15, 30, or 60 seconds")
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO transcript_correction_settings(session_id, enabled, window_ms)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       enabled = excluded.enabled, window_ms = excluded.window_ms""",
                (session_id, int(enabled), window_ms),
            )

    def transcript_correction_settings(self, session_id: str) -> TranscriptCorrectionSettingsRecord:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT enabled, window_ms FROM transcript_correction_settings
                   WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
        if row is None:
            return TranscriptCorrectionSettingsRecord(False, 30_000)
        return TranscriptCorrectionSettingsRecord(bool(row["enabled"]), int(row["window_ms"]))

    def correction_segments(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[CorrectionSegmentRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT segment.id, segment.start_ms, segment.end_ms, segment.source,
                          version.text
                   FROM transcript_segments AS segment
                   JOIN effective_transcript_versions AS version
                     ON version.segment_id = segment.id
                   WHERE segment.session_id = ?
                     AND segment.end_ms >= ? AND segment.start_ms < ?
                   ORDER BY segment.start_ms, segment.id""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        return [CorrectionSegmentRecord(**dict(row)) for row in rows]

    def representative_frames(
        self, session_id: str, start_ms: int, end_ms: int, *, limit: int = 6
    ) -> list[CorrectionFrameRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, source_id, ts_ms, path, perceptual_hash
                   FROM frames
                   WHERE session_id = ? AND ts_ms BETWEEN ? AND ?
                   ORDER BY source_id, ts_ms, id""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        if not rows:
            return []

        latest_by_source: dict[str, sqlite3.Row] = {}
        change_candidates: list[tuple[int, sqlite3.Row]] = []
        previous_hashes: dict[str, int] = {}
        for row in rows:
            source_id = str(row["source_id"])
            latest_by_source[source_id] = row
            try:
                image_hash = int(row["perceptual_hash"], 16)
            except (TypeError, ValueError):
                image_hash = 0
            previous_hash = previous_hashes.get(source_id)
            if previous_hash is not None:
                change_candidates.append(((image_hash ^ previous_hash).bit_count(), row))
            previous_hashes[source_id] = image_hash

        selected = {int(row["id"]): row for row in latest_by_source.values()}
        for _distance, row in sorted(
            change_candidates,
            key=lambda item: (item[0], item[1]["ts_ms"], item[1]["id"]),
            reverse=True,
        ):
            if len(selected) >= min(limit, len(latest_by_source) + 2):
                break
            selected.setdefault(int(row["id"]), row)
        selected_rows = sorted(selected.values(), key=lambda row: (row["ts_ms"], row["id"]))[:limit]
        return [
            CorrectionFrameRecord(
                id=row["id"],
                source_id=row["source_id"],
                ts_ms=row["ts_ms"],
                path=Path(row["path"]),
            )
            for row in selected_rows
        ]

    def start_correction_run(self, session_id: str, start_ms: int, end_ms: int, model: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO transcript_correction_runs(
                       session_id, window_start_ms, window_end_ms, state, model, started_at_utc
                   ) VALUES (?, ?, ?, 'running', ?, ?)""",
                (session_id, start_ms, end_ms, model, datetime.now(UTC).isoformat()),
            )
            return int(cursor.lastrowid)

    def finish_correction_run(
        self,
        run_id: int,
        state: str,
        *,
        error_source: str | None = None,
        error: str | None = None,
    ) -> TranscriptCorrectionRunRecord:
        if state not in {"corrected", "failed"}:
            raise ValueError(f"Unsupported correction run state: {state}")
        with self.connect() as connection:
            connection.execute(
                """UPDATE transcript_correction_runs
                   SET state = ?, completed_at_utc = ?, error_source = ?, error = ?
                   WHERE id = ?""",
                (state, datetime.now(UTC).isoformat(), error_source, error, run_id),
            )
        return TranscriptCorrectionRunRecord(run_id, state, error_source, error)

    def add_question(
        self,
        session_id: str,
        asked_at_ms: int,
        question: str,
        answer: str | None,
        context_start_ms: int,
        context_end_ms: int,
        error: str | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO questions(
                       session_id, asked_at_ms, question, answer,
                       context_start_ms, context_end_ms, error
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    asked_at_ms,
                    question,
                    answer,
                    context_start_ms,
                    context_end_ms,
                    error,
                ),
            )
            return int(cursor.lastrowid)

    def add_artifact(
        self, session_id: str, kind: str, created_at_utc: str, content_json: str, model: str
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO artifacts(session_id, kind, created_at_utc, content_json, model)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, kind, created_at_utc, content_json, model),
            )
            return int(cursor.lastrowid)
