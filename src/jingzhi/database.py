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
    error TEXT,
    state TEXT NOT NULL DEFAULT 'submitted' CHECK (state IN ('draft', 'submitted'))
);

CREATE TABLE IF NOT EXISTS answer_versions (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    model TEXT,
    connection_json TEXT,
    request_status TEXT NOT NULL CHECK (request_status IN ('succeeded', 'failed')),
    upstream_request_id TEXT,
    answer TEXT,
    error TEXT,
    evidence_state TEXT NOT NULL CHECK (evidence_state IN ('exact', 'unavailable')),
    created_at_utc TEXT NOT NULL,
    UNIQUE(question_id, version_number)
);
CREATE INDEX IF NOT EXISTS answer_version_history
ON answer_versions(question_id, version_number);

CREATE TABLE IF NOT EXISTS answer_evidence (
    answer_version_id INTEGER NOT NULL REFERENCES answer_versions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    stable_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('transcript', 'frame')),
    source TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    transcript_version_id INTEGER REFERENCES transcript_versions(id),
    frame_id INTEGER REFERENCES frames(id),
    content_text TEXT,
    resource_path TEXT,
    PRIMARY KEY(answer_version_id, ordinal),
    UNIQUE(answer_version_id, stable_id)
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

LATEST_SCHEMA_VERSION = 5

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


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    id: int
    session_id: str
    asked_at_ms: int
    question: str
    context_start_ms: int | None
    context_end_ms: int | None
    state: str


@dataclass(frozen=True, slots=True)
class AnswerVersionRecord:
    id: int
    question_id: int
    version_number: int
    model: str | None
    connection_json: str | None
    request_status: str
    request_id: str | None
    answer: str | None
    error: str | None
    evidence_state: str
    created_at_utc: str


@dataclass(frozen=True, slots=True)
class AnswerEvidenceRecord:
    ordinal: int
    stable_id: str
    kind: str
    source: str
    start_ms: int
    end_ms: int
    transcript_version_id: int | None
    frame_id: int | None
    content_text: str | None
    resource_path: Path | None


@dataclass(frozen=True, slots=True)
class AnswerTranscriptRecord:
    segment_id: int
    version_id: int
    source: str
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class AnswerFrameRecord:
    frame_id: int
    source: str
    ts_ms: int
    path: Path


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            self.migrate(connection)

    def migrate(self, connection: sqlite3.Connection) -> None:
        """Bring both fresh and pre-versioned MVP databases to the latest schema."""
        previous_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        connection.executescript(SCHEMA)
        frame_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(frames)").fetchall()
        }
        if "source_id" not in frame_columns:
            connection.execute(
                "ALTER TABLE frames ADD COLUMN source_id TEXT NOT NULL DEFAULT 'display:primary'"
            )
        question_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(questions)").fetchall()
        }
        if "state" not in question_columns:
            connection.execute(
                "ALTER TABLE questions ADD COLUMN state TEXT NOT NULL DEFAULT 'submitted'"
            )
        applied_at = datetime.now(UTC).isoformat()
        connection.execute(
            """INSERT OR IGNORE INTO transcript_versions(
                   segment_id, kind, text, created_at_utc, active
               )
               SELECT id, 'original', text, ?, 1 FROM transcript_segments""",
            (applied_at,),
        )
        if previous_version < 4:
            connection.execute(
                """INSERT OR IGNORE INTO answer_versions(
                       question_id, version_number, request_status, answer, error,
                       evidence_state, created_at_utc
                   )
                   SELECT id, 1,
                          CASE WHEN answer IS NOT NULL THEN 'succeeded' ELSE 'failed' END,
                          answer, error, 'unavailable', ?
                   FROM questions
                   WHERE answer IS NOT NULL OR error IS NOT NULL""",
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
                   WHERE session_id = ? AND state = 'submitted'
                     AND asked_at_ms BETWEEN ? AND ?
                   ORDER BY asked_at_ms, id""",
                (session_id, start_ms, end_ms),
            )
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

    def answer_transcripts_between(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[AnswerTranscriptRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT segment.id AS segment_id, version.id AS version_id,
                          segment.source, segment.start_ms, segment.end_ms, version.text
                   FROM transcript_segments AS segment
                   JOIN effective_transcript_versions AS version
                     ON version.segment_id = segment.id
                   WHERE segment.session_id = ?
                     AND segment.end_ms >= ? AND segment.start_ms <= ?
                   ORDER BY segment.start_ms, segment.id""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        return [AnswerTranscriptRecord(**dict(row)) for row in rows]

    def answer_frames_near(
        self, session_id: str, center_ms: int, start_ms: int, end_ms: int, limit: int = 4
    ) -> list[AnswerFrameRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id AS frame_id, source_id AS source, ts_ms, path FROM frames
                   WHERE session_id = ? AND ts_ms BETWEEN ? AND ?
                   ORDER BY abs(ts_ms - ?), id LIMIT ?""",
                (session_id, start_ms, end_ms, center_ms, limit),
            ).fetchall()
        return [
            AnswerFrameRecord(
                frame_id=row["frame_id"],
                source=row["source"],
                ts_ms=row["ts_ms"],
                path=Path(row["path"]),
            )
            for row in rows
        ]

    def create_question(
        self,
        session_id: str,
        asked_at_ms: int,
        question: str,
        context_start_ms: int,
        context_end_ms: int,
        *,
        state: str = "submitted",
    ) -> int:
        if state not in {"draft", "submitted"}:
            raise ValueError(f"Unsupported question state: {state}")
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO questions(
                       session_id, asked_at_ms, question, context_start_ms, context_end_ms, state
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, asked_at_ms, question, context_start_ms, context_end_ms, state),
            )
            return int(cursor.lastrowid)

    def update_question_range(self, question_id: int, start_ms: int, end_ms: int) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE questions SET context_start_ms = ?, context_end_ms = ?
                   WHERE id = ? AND state = 'draft'""",
                (start_ms, end_ms, question_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("The pending question anchor is unavailable")

    def submit_question(self, question_id: int, question: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE questions SET question = ?, state = 'submitted'
                   WHERE id = ? AND state = 'draft'""",
                (question, question_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("The pending question anchor is unavailable")

    def delete_pending_question(self, question_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM questions WHERE id = ? AND state = 'draft'",
                (question_id,),
            )
            return cursor.rowcount == 1

    def question(self, question_id: int) -> QuestionRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id, session_id, asked_at_ms, question,
                          context_start_ms, context_end_ms, state
                   FROM questions WHERE id = ?""",
                (question_id,),
            ).fetchone()
        return QuestionRecord(**dict(row)) if row is not None else None

    def latest_question_id(self, session_id: str) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id FROM questions
                   WHERE session_id = ? AND state = 'submitted'
                   ORDER BY asked_at_ms DESC, id DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        return int(row["id"]) if row is not None else None

    def record_answer_version(
        self,
        question_id: int,
        *,
        model: str | None,
        connection_json: str | None,
        request_status: str,
        request_id: str | None,
        answer: str | None,
        error: str | None,
        evidence_state: str,
        evidence: list[dict[str, object]],
    ) -> AnswerVersionRecord:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version_number = int(
                connection.execute(
                    """SELECT COALESCE(MAX(version_number), 0) + 1
                       FROM answer_versions WHERE question_id = ?""",
                    (question_id,),
                ).fetchone()[0]
            )
            created_at = datetime.now(UTC).isoformat()
            cursor = connection.execute(
                """INSERT INTO answer_versions(
                       question_id, version_number, model, connection_json, request_status,
                       upstream_request_id, answer, error, evidence_state, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    question_id,
                    version_number,
                    model,
                    connection_json,
                    request_status,
                    request_id,
                    answer,
                    error,
                    evidence_state,
                    created_at,
                ),
            )
            answer_version_id = int(cursor.lastrowid)
            connection.executemany(
                """INSERT INTO answer_evidence(
                       answer_version_id, ordinal, stable_id, kind, source,
                       start_ms, end_ms, transcript_version_id, frame_id,
                       content_text, resource_path
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        answer_version_id,
                        ordinal,
                        item["stable_id"],
                        item["kind"],
                        item["source"],
                        item["start_ms"],
                        item["end_ms"],
                        item.get("transcript_version_id"),
                        item.get("frame_id"),
                        item.get("content_text"),
                        item.get("resource_path"),
                    )
                    for ordinal, item in enumerate(evidence)
                ],
            )
        return AnswerVersionRecord(
            answer_version_id,
            question_id,
            version_number,
            model,
            connection_json,
            request_status,
            request_id,
            answer,
            error,
            evidence_state,
            created_at,
        )

    def answer_versions(self, question_id: int) -> list[AnswerVersionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, question_id, version_number, model, connection_json,
                          request_status, upstream_request_id AS request_id, answer, error,
                          evidence_state, created_at_utc
                   FROM answer_versions WHERE question_id = ? ORDER BY version_number""",
                (question_id,),
            ).fetchall()
        return [AnswerVersionRecord(**dict(row)) for row in rows]

    def answer_evidence(self, answer_version_id: int) -> list[AnswerEvidenceRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT ordinal, stable_id, kind, source, start_ms, end_ms,
                          transcript_version_id, frame_id, content_text, resource_path
                   FROM answer_evidence WHERE answer_version_id = ? ORDER BY ordinal""",
                (answer_version_id,),
            ).fetchall()
        return [
            AnswerEvidenceRecord(
                ordinal=row["ordinal"],
                stable_id=row["stable_id"],
                kind=row["kind"],
                source=row["source"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                transcript_version_id=row["transcript_version_id"],
                frame_id=row["frame_id"],
                content_text=row["content_text"],
                resource_path=Path(row["resource_path"]) if row["resource_path"] else None,
            )
            for row in rows
        ]

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
