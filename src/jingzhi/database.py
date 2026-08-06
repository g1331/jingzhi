from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from jingzhi.transcript_correction import CORRECTION_WINDOW_MS
from jingzhi.whisper_settings import WhisperSettings

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    ended_at_utc TEXT,
    status TEXT NOT NULL CHECK (status IN ('recording', 'complete', 'interrupted')),
    pinned_at_utc TEXT,
    retention_started_at_utc TEXT,
    trashed_at_utc TEXT,
    trash_expires_at_utc TEXT
);

CREATE TABLE IF NOT EXISTS session_notifications (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN (
        'retention_7d', 'retention_1d', 'moved_to_trash',
        'permanently_deleted', 'final_delete_failed'
    )),
    notified_at_utc TEXT NOT NULL,
    PRIMARY KEY(session_id, kind)
);

CREATE TABLE IF NOT EXISTS source_events (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN (
        'failure', 'device_unavailable', 'stream_stopped', 'overflow'
    )),
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms >= start_ms),
    message TEXT NOT NULL,
    data_loss_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (data_loss_confirmed IN (0, 1)),
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS source_event_session_time
ON source_events(session_id, start_ms, end_ms, id);

CREATE TABLE IF NOT EXISTS timeline_events (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('pause', 'data_gap')),
    source TEXT,
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms >= start_ms),
    message TEXT NOT NULL,
    source_event_id INTEGER REFERENCES source_events(id) ON DELETE SET NULL,
    UNIQUE(kind, source_event_id)
);
CREATE INDEX IF NOT EXISTS timeline_event_session_time
ON timeline_events(session_id, start_ms, end_ms, id);
CREATE TABLE IF NOT EXISTS pending_media_deletions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    failure_notified_at_utc TEXT
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

CREATE TABLE IF NOT EXISTS whisper_runs (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    profile TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    requested_device TEXT NOT NULL,
    requested_compute_type TEXT NOT NULL,
    actual_model TEXT NOT NULL,
    actual_device TEXT NOT NULL,
    actual_compute_type TEXT NOT NULL,
    language TEXT NOT NULL,
    vad_enabled INTEGER NOT NULL CHECK (vad_enabled IN (0, 1)),
    vad_min_silence_ms INTEGER NOT NULL,
    fallback_advice TEXT NOT NULL,
    started_at_utc TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    segment_id UNINDEXED, session_id UNINDEXED, text, tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS cross_session_fts USING fts5(
    entry_id UNINDEXED,
    session_id UNINDEXED,
    kind UNINDEXED,
    source_id UNINDEXED,
    source UNINDEXED,
    start_ms UNINDEXED,
    end_ms UNINDEXED,
    version_kind UNINDEXED,
    text,
    tokenize='unicode61'
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

CREATE TABLE IF NOT EXISTS question_notes (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS question_note_history
ON question_notes(question_id, id);

CREATE TABLE IF NOT EXISTS answer_versions (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    model TEXT,
    connection_json TEXT,
    request_status TEXT NOT NULL CHECK (request_status IN ('succeeded', 'failed')),
    model_invocation_id INTEGER REFERENCES model_invocations(id),
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

CREATE TABLE IF NOT EXISTS model_invocations (
    id INTEGER PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN (
        'utility', 'transcript_correction', 'instant_answer', 'deep_analysis'
    )),
    connection_id TEXT NOT NULL,
    connection_name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_mode TEXT NOT NULL CHECK (api_mode IN ('responses', 'chat_completions')),
    model TEXT NOT NULL,
    reasoning_level TEXT NOT NULL CHECK (reasoning_level IN ('fast', 'balanced', 'deep')),
    fallback_reason TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    retryable INTEGER NOT NULL DEFAULT 1,
    task_type TEXT,
    task_payload_json TEXT,
    upstream_request_id TEXT,
    error TEXT,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT
);
CREATE INDEX IF NOT EXISTS model_invocation_session
ON model_invocations(session_id, id);

CREATE TABLE IF NOT EXISTS model_invocation_evidence (
    invocation_id INTEGER NOT NULL REFERENCES model_invocations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    stable_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('transcript', 'frame')),
    source TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    transcript_version_id INTEGER,
    frame_id INTEGER,
    PRIMARY KEY(invocation_id, ordinal),
    UNIQUE(invocation_id, stable_id)
);

CREATE TABLE IF NOT EXISTS model_invocation_content_evidence (
    invocation_id INTEGER NOT NULL REFERENCES model_invocations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    stable_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('answer', 'material')),
    source TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    answer_version_id INTEGER,
    material_version_id INTEGER,
    content_text TEXT,
    resource_path TEXT,
    PRIMARY KEY(invocation_id, ordinal),
    UNIQUE(invocation_id, stable_id)
);

CREATE TABLE IF NOT EXISTS session_material_versions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('generated', 'user_edit')),
    content TEXT NOT NULL,
    template_id TEXT,
    model TEXT,
    connection_json TEXT,
    model_invocation_id INTEGER REFERENCES model_invocations(id) ON DELETE SET NULL,
    request_status TEXT NOT NULL CHECK (request_status IN ('succeeded', 'failed')),
    upstream_request_id TEXT,
    error TEXT,
    evidence_state TEXT NOT NULL CHECK (evidence_state IN ('exact', 'unavailable')),
    created_at_utc TEXT NOT NULL,
    UNIQUE(session_id, version_number)
);
CREATE INDEX IF NOT EXISTS session_material_version_history
ON session_material_versions(session_id, version_number);

CREATE TABLE IF NOT EXISTS session_material_evidence (
    material_version_id INTEGER NOT NULL REFERENCES session_material_versions(id) ON DELETE CASCADE,
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
    PRIMARY KEY(material_version_id, ordinal),
    UNIQUE(material_version_id, stable_id)
);

CREATE TABLE IF NOT EXISTS cross_session_syntheses (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT,
    model TEXT,
    connection_json TEXT,
    model_invocation_id INTEGER REFERENCES model_invocations(id) ON DELETE SET NULL,
    request_status TEXT NOT NULL CHECK (request_status IN ('succeeded', 'failed')),
    upstream_request_id TEXT,
    error TEXT,
    evidence_state TEXT NOT NULL CHECK (evidence_state IN ('exact', 'unavailable')),
    created_at_utc TEXT NOT NULL,
    retry_of_id INTEGER REFERENCES cross_session_syntheses(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS cross_session_synthesis_history
ON cross_session_syntheses(created_at_utc, id);

CREATE TABLE IF NOT EXISTS cross_session_retry_claims (
    synthesis_id INTEGER PRIMARY KEY REFERENCES cross_session_syntheses(id) ON DELETE CASCADE,
    claimed_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cross_session_synthesis_evidence (
    synthesis_id INTEGER NOT NULL REFERENCES cross_session_syntheses(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    stable_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    session_title TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('transcript', 'frame', 'answer', 'material')),
    source TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    transcript_version_id INTEGER REFERENCES transcript_versions(id) ON DELETE SET NULL,
    frame_id INTEGER REFERENCES frames(id) ON DELETE SET NULL,
    answer_version_id INTEGER REFERENCES answer_versions(id) ON DELETE SET NULL,
    material_version_id INTEGER REFERENCES session_material_versions(id) ON DELETE SET NULL,
    content_text TEXT,
    resource_path TEXT,
    PRIMARY KEY(synthesis_id, ordinal),
    UNIQUE(synthesis_id, stable_id)
);
CREATE INDEX IF NOT EXISTS cross_session_synthesis_evidence_session
ON cross_session_synthesis_evidence(session_id, synthesis_id);

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

LATEST_SCHEMA_VERSION = 17

CROSS_SESSION_FTS_CONTENT_QUERY = """
SELECT 'transcript-version:' || version.id, segment.session_id, 'transcript',
       version.id, segment.source, segment.start_ms, segment.end_ms,
       version.kind, version.text
FROM transcript_segments AS segment
JOIN effective_transcript_versions AS version
  ON version.segment_id = segment.id
UNION ALL
SELECT 'answer-version:' || answer.id, question.session_id, 'answer',
       answer.id, '问答', question.asked_at_ms, question.asked_at_ms,
       NULL, COALESCE(question.question, '') || char(10) || COALESCE(answer.answer, '')
FROM answer_versions AS answer
JOIN questions AS question ON question.id = answer.question_id
WHERE question.state = 'submitted'
  AND answer.request_status = 'succeeded'
  AND answer.answer IS NOT NULL
UNION ALL
SELECT 'material-version:' || material.id, material.session_id, 'material',
       material.id, '材料',
       COALESCE((SELECT MIN(start_ms) FROM session_material_evidence
                 WHERE material_version_id = material.id), 0),
       COALESCE((SELECT MAX(end_ms) FROM session_material_evidence
                 WHERE material_version_id = material.id), 0),
       material.kind, material.content
FROM session_material_versions AS material
WHERE material.request_status = 'succeeded'
"""


class SessionNotificationKind(StrEnum):
    RETENTION_7D = "retention_7d"
    RETENTION_1D = "retention_1d"
    MOVED_TO_TRASH = "moved_to_trash"
    PERMANENTLY_DELETED = "permanently_deleted"
    FINAL_DELETE_FAILED = "final_delete_failed"


class SourceEventKind(StrEnum):
    FAILURE = "failure"
    DEVICE_UNAVAILABLE = "device_unavailable"
    STREAM_STOPPED = "stream_stopped"
    OVERFLOW = "overflow"


class TimelineEventKind(StrEnum):
    PAUSE = "pause"
    DATA_GAP = "data_gap"


SESSION_SUMMARY_QUERY = """
SELECT
    sessions.id,
    sessions.title,
    sessions.started_at_utc,
    sessions.ended_at_utc,
    sessions.status,
    sessions.pinned_at_utc,
    sessions.retention_started_at_utc,
    sessions.trashed_at_utc,
    sessions.trash_expires_at_utc,
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
class TimelineEventRecord:
    id: int
    session_id: str
    kind: str
    source: str | None
    start_ms: int
    end_ms: int
    message: str
    source_event_id: int | None = None


@dataclass(frozen=True, slots=True)
class SourceEventRecord:
    id: int
    session_id: str
    source: str
    kind: str
    start_ms: int
    end_ms: int
    message: str
    data_loss_confirmed: bool


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    title: str
    started_at_utc: str
    ended_at_utc: str | None
    status: str
    duration_ms: int
    frame_count: int
    pinned: bool
    retention_started_at_utc: str | None
    trashed_at_utc: str | None
    trash_expires_at_utc: str | None


@dataclass(frozen=True, slots=True)
class SessionRuntimeCounts:
    frame_count: int
    transcribed_audio_ms: int
    transcript_count: int
    correction_backlog: int
    pending_audio_chunks: int
    failed_audio_chunks: int


@dataclass(frozen=True, slots=True)
class PendingAudioChunk:
    id: int
    session_id: str
    source: str
    start_ms: int
    end_ms: int
    path: Path


@dataclass(frozen=True, slots=True)
class PendingMediaDeletion:
    session_id: str
    title: str
    path: Path


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
    version_id: int


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
    model_invocation_id: int | None = None


@dataclass(frozen=True, slots=True)
class SessionAnswerRecord:
    id: int
    question_id: int
    version_number: int
    asked_at_ms: int
    question: str
    model: str | None
    request_status: str
    answer: str | None
    error: str | None
    evidence_state: str
    created_at_utc: str


@dataclass(frozen=True, slots=True)
class QuestionNoteRecord:
    id: int
    question_id: int
    content: str
    created_at_utc: str


@dataclass(frozen=True, slots=True)
class SessionMaterialVersionRecord:
    id: int
    session_id: str
    version_number: int
    kind: str
    content: str
    template_id: str | None
    model: str | None
    connection_json: str | None
    model_invocation_id: int | None
    request_status: str
    request_id: str | None
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
class MaterialEvidenceRecord:
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


@dataclass(frozen=True, slots=True)
class ModelInvocationEvidenceRecord:
    stable_id: str
    kind: str
    source: str
    start_ms: int
    end_ms: int
    transcript_version_id: int | None = None
    frame_id: int | None = None
    answer_version_id: int | None = None
    material_version_id: int | None = None
    content_text: str | None = None
    resource_path: Path | None = None


@dataclass(frozen=True, slots=True)
class WhisperRunRecord:
    session_id: str
    profile: str
    requested_model: str
    requested_device: str
    requested_compute_type: str
    actual_model: str
    actual_device: str
    actual_compute_type: str
    language: str
    vad_enabled: bool
    vad_min_silence_ms: int
    fallback_advice: str
    started_at_utc: str


@dataclass(frozen=True, slots=True)
class ModelInvocationRecord:
    id: int
    session_id: str | None
    role: str
    connection_id: str
    connection_name: str
    base_url: str
    api_mode: str
    model: str
    reasoning_level: str
    fallback_reason: str | None
    status: str
    request_id: str | None
    error: str | None
    started_at_utc: str
    completed_at_utc: str | None
    evidence_ids: tuple[str, ...]
    task_type: str | None = None
    task_payload_json: str | None = None


@dataclass(frozen=True, slots=True)
class CrossSessionSearchResult:
    stable_id: str
    session_id: str
    session_title: str
    kind: str
    source_id: int
    source: str
    start_ms: int
    end_ms: int
    version_kind: str | None
    text: str
    snippet: str


@dataclass(frozen=True, slots=True)
class CrossSessionEvidenceRecord:
    stable_id: str
    session_id: str
    session_title: str
    kind: str
    source: str
    start_ms: int
    end_ms: int
    content_text: str | None
    resource_path: Path | None
    transcript_version_id: int | None = None
    frame_id: int | None = None
    answer_version_id: int | None = None
    material_version_id: int | None = None


@dataclass(frozen=True, slots=True)
class CrossSessionSynthesisRecord:
    id: int
    question: str
    answer: str | None
    model: str | None
    connection_json: str | None
    model_invocation_id: int | None
    request_status: str
    request_id: str | None
    error: str | None
    evidence_state: str
    created_at_utc: str
    retry_of_id: int | None = None


@dataclass(frozen=True, slots=True)
class CrossSessionSynthesisEvidenceRecord:
    synthesis_id: int
    ordinal: int
    stable_id: str
    session_id: str
    session_title: str
    kind: str
    source: str
    start_ms: int
    end_ms: int
    content_text: str | None
    resource_path: Path | None
    transcript_version_id: int | None
    frame_id: int | None
    answer_version_id: int | None
    material_version_id: int | None


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
        session_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        }
        for name in (
            "pinned_at_utc",
            "retention_started_at_utc",
            "trashed_at_utc",
            "trash_expires_at_utc",
        ):
            if name not in session_columns:
                connection.execute(f"ALTER TABLE sessions ADD COLUMN {name} TEXT")
        pending_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(pending_media_deletions)").fetchall()
        }
        if "failure_notified_at_utc" not in pending_columns:
            connection.execute(
                "ALTER TABLE pending_media_deletions ADD COLUMN failure_notified_at_utc TEXT"
            )
        connection.execute(
            """UPDATE sessions
               SET retention_started_at_utc = COALESCE(ended_at_utc, started_at_utc)
               WHERE status != 'recording' AND retention_started_at_utc IS NULL"""
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
        if previous_version < 14:
            self._migrate_cross_session_evidence_snapshots(connection)
        if previous_version < 16:
            model_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(model_invocations)").fetchall()
            }
            if "retryable" not in model_columns:
                connection.execute(
                    "ALTER TABLE model_invocations ADD COLUMN retryable INTEGER NOT NULL DEFAULT 1"
                )
            model_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(model_invocations)").fetchall()
            }
            for name in ("task_type", "task_payload_json"):
                if name not in model_columns:
                    connection.execute(f"ALTER TABLE model_invocations ADD COLUMN {name} TEXT")
            connection.execute(
                "UPDATE model_invocations SET retryable = 0 WHERE status = 'succeeded'"
            )
        if previous_version < 17:
            answer_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(answer_versions)").fetchall()
            }
            if "model_invocation_id" not in answer_columns:
                connection.execute(
                    "ALTER TABLE answer_versions ADD COLUMN model_invocation_id INTEGER "
                    "REFERENCES model_invocations(id)"
                )
            model_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(model_invocations)").fetchall()
            }
            for name in ("task_type", "task_payload_json"):
                if name not in model_columns:
                    connection.execute(f"ALTER TABLE model_invocations ADD COLUMN {name} TEXT")
        if previous_version < 15:
            synthesis_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(cross_session_syntheses)"
                ).fetchall()
            }
            if "retry_of_id" not in synthesis_columns:
                connection.execute(
                    "ALTER TABLE cross_session_syntheses ADD COLUMN retry_of_id INTEGER "
                    "REFERENCES cross_session_syntheses(id) ON DELETE SET NULL"
                )
        if previous_version < 13:
            self._rebuild_cross_session_fts(connection)
        connection.executemany(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
            [(version, applied_at) for version in range(1, LATEST_SCHEMA_VERSION + 1)],
        )
        connection.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION}")

    @staticmethod
    def _migrate_cross_session_evidence_snapshots(connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(cross_session_synthesis_evidence)"
            ).fetchall()
        }
        if "session_title" in columns:
            return
        connection.execute(
            """CREATE TABLE cross_session_synthesis_evidence_v14 (
                   synthesis_id INTEGER NOT NULL REFERENCES cross_session_syntheses(id)
                       ON DELETE CASCADE,
                   ordinal INTEGER NOT NULL,
                   stable_id TEXT NOT NULL,
                   session_id TEXT NOT NULL,
                   session_title TEXT NOT NULL,
                   kind TEXT NOT NULL CHECK (kind IN (
                       'transcript', 'frame', 'answer', 'material'
                   )),
                   source TEXT NOT NULL,
                   start_ms INTEGER NOT NULL,
                   end_ms INTEGER NOT NULL,
                   transcript_version_id INTEGER REFERENCES transcript_versions(id)
                       ON DELETE SET NULL,
                   frame_id INTEGER REFERENCES frames(id) ON DELETE SET NULL,
                   answer_version_id INTEGER REFERENCES answer_versions(id)
                       ON DELETE SET NULL,
                   material_version_id INTEGER REFERENCES session_material_versions(id)
                       ON DELETE SET NULL,
                   content_text TEXT,
                   resource_path TEXT,
                   PRIMARY KEY(synthesis_id, ordinal),
                   UNIQUE(synthesis_id, stable_id)
               )"""
        )
        connection.execute(
            """INSERT INTO cross_session_synthesis_evidence_v14(
                   synthesis_id, ordinal, stable_id, session_id, session_title, kind, source,
                   start_ms, end_ms, transcript_version_id, frame_id, answer_version_id,
                   material_version_id, content_text, resource_path
               )
               SELECT evidence.synthesis_id, evidence.ordinal, evidence.stable_id,
                      evidence.session_id, COALESCE(sessions.title, '已删除会话'),
                      evidence.kind, evidence.source,
                      evidence.start_ms, evidence.end_ms, evidence.transcript_version_id,
                      evidence.frame_id, evidence.answer_version_id, evidence.material_version_id,
                      evidence.content_text, evidence.resource_path
               FROM cross_session_synthesis_evidence AS evidence
               LEFT JOIN sessions ON sessions.id = evidence.session_id"""
        )
        connection.execute("DROP TABLE cross_session_synthesis_evidence")
        connection.execute(
            "ALTER TABLE cross_session_synthesis_evidence_v14 RENAME TO cross_session_synthesis_evidence"
        )
        connection.execute(
            """CREATE INDEX cross_session_synthesis_evidence_session
               ON cross_session_synthesis_evidence(session_id, synthesis_id)"""
        )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @staticmethod
    def _fts_match_query(query: str) -> str:
        terms = re.findall(r"[\w\u0080-\uffff]+", query, flags=re.UNICODE)
        escaped = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
        return " OR ".join(escaped)

    @staticmethod
    def _fts_result(row: sqlite3.Row) -> CrossSessionSearchResult:
        text = str(row["text"])
        return CrossSessionSearchResult(
            stable_id=str(row["entry_id"]),
            session_id=str(row["session_id"]),
            session_title=str(row["session_title"]),
            kind=str(row["kind"]),
            source_id=int(row["source_id"]),
            source=str(row["source"]),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            version_kind=str(row["version_kind"]) if row["version_kind"] else None,
            text=text,
            snippet=text[:240],
        )

    def _rebuild_cross_session_fts(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM cross_session_fts")
        connection.execute(
            """INSERT INTO cross_session_fts(
                   entry_id, session_id, kind, source_id, source, start_ms, end_ms,
                   version_kind, text
               ) """
            + CROSS_SESSION_FTS_CONTENT_QUERY
        )

    @staticmethod
    def _replace_cross_session_fts_entry(
        connection: sqlite3.Connection,
        *,
        stable_id: str,
        session_id: str,
        kind: str,
        source_id: int,
        source: str,
        start_ms: int,
        end_ms: int,
        version_kind: str | None,
        text: str,
    ) -> None:
        connection.execute("DELETE FROM cross_session_fts WHERE entry_id = ?", (stable_id,))
        if text:
            connection.execute(
                """INSERT INTO cross_session_fts(
                       entry_id, session_id, kind, source_id, source, start_ms, end_ms,
                       version_kind, text
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stable_id,
                    session_id,
                    kind,
                    source_id,
                    source,
                    start_ms,
                    end_ms,
                    version_kind,
                    text,
                ),
            )

    def _reindex_transcript_segment(self, connection: sqlite3.Connection, segment_id: int) -> None:
        version_rows = connection.execute(
            "SELECT id FROM transcript_versions WHERE segment_id = ?", (segment_id,)
        ).fetchall()
        for row in version_rows:
            connection.execute(
                "DELETE FROM cross_session_fts WHERE entry_id = ?",
                (f"transcript-version:{row['id']}",),
            )
        row = connection.execute(
            """SELECT segment.session_id, segment.source, segment.start_ms, segment.end_ms,
                      version.id, version.kind, version.text
               FROM transcript_segments AS segment
               JOIN effective_transcript_versions AS version
                 ON version.segment_id = segment.id
               WHERE segment.id = ?""",
            (segment_id,),
        ).fetchone()
        if row is not None:
            self._replace_cross_session_fts_entry(
                connection,
                stable_id=f"transcript-version:{row['id']}",
                session_id=row["session_id"],
                kind="transcript",
                source_id=int(row["id"]),
                source=row["source"],
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                version_kind=row["kind"],
                text=row["text"],
            )

    def _index_answer_version(self, connection: sqlite3.Connection, answer_id: int) -> None:
        stable_id = f"answer-version:{answer_id}"
        connection.execute("DELETE FROM cross_session_fts WHERE entry_id = ?", (stable_id,))
        row = connection.execute(
            """SELECT answer.id, question.session_id, question.question,
                      question.asked_at_ms, answer.answer
               FROM answer_versions AS answer
               JOIN questions AS question ON question.id = answer.question_id
               WHERE answer.id = ? AND question.state = 'submitted'
                 AND answer.request_status = 'succeeded' AND answer.answer IS NOT NULL""",
            (answer_id,),
        ).fetchone()
        if row is not None:
            self._replace_cross_session_fts_entry(
                connection,
                stable_id=stable_id,
                session_id=row["session_id"],
                kind="answer",
                source_id=int(row["id"]),
                source="问答",
                start_ms=int(row["asked_at_ms"]),
                end_ms=int(row["asked_at_ms"]),
                version_kind=None,
                text=f"{row['question']}\n{row['answer']}".strip(),
            )

    def _index_material_version(self, connection: sqlite3.Connection, material_id: int) -> None:
        stable_id = f"material-version:{material_id}"
        connection.execute("DELETE FROM cross_session_fts WHERE entry_id = ?", (stable_id,))
        row = connection.execute(
            """SELECT material.id, material.session_id, material.kind, material.content,
                      COALESCE((SELECT MIN(start_ms) FROM session_material_evidence
                                WHERE material_version_id = material.id), 0) AS start_ms,
                      COALESCE((SELECT MAX(end_ms) FROM session_material_evidence
                                WHERE material_version_id = material.id), 0) AS end_ms
               FROM session_material_versions AS material
               WHERE material.id = ? AND material.request_status = 'succeeded'""",
            (material_id,),
        ).fetchone()
        if row is not None:
            self._replace_cross_session_fts_entry(
                connection,
                stable_id=stable_id,
                session_id=row["session_id"],
                kind="material",
                source_id=int(row["id"]),
                source="材料",
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                version_kind=row["kind"],
                text=row["content"],
            )

    def cross_session_search(
        self, query: str, *, limit: int = 50
    ) -> list[CrossSessionSearchResult]:
        query = query.strip()
        if not query or limit <= 0:
            return []
        limit = min(limit, 100)
        match_query = self._fts_match_query(query)
        if not match_query:
            return []
        select = """SELECT entry.entry_id, entry.session_id, sessions.title AS session_title,
                          entry.kind, entry.source_id, entry.source, entry.start_ms, entry.end_ms,
                          entry.version_kind, entry.text
                   FROM cross_session_fts AS entry
                   JOIN sessions ON sessions.id = entry.session_id
                   WHERE sessions.trashed_at_utc IS NULL
                     AND cross_session_fts MATCH ?
                   ORDER BY bm25(cross_session_fts), entry.entry_id
                   LIMIT ?"""
        with self.connect() as connection:
            try:
                rows = connection.execute(select, (match_query, limit)).fetchall()
            except sqlite3.OperationalError:
                rows = []
            like_rows = connection.execute(
                """SELECT entry.entry_id, entry.session_id, sessions.title AS session_title,
                          entry.kind, entry.source_id, entry.source, entry.start_ms,
                          entry.end_ms, entry.version_kind, entry.text
                   FROM cross_session_fts AS entry
                   JOIN sessions ON sessions.id = entry.session_id
                   WHERE sessions.trashed_at_utc IS NULL AND entry.text LIKE ?
                   ORDER BY entry.entry_id LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()
        merged: list[sqlite3.Row] = []
        seen: set[str] = set()
        for row in (*rows, *like_rows):
            stable_id = str(row["entry_id"])
            if stable_id not in seen:
                merged.append(row)
                seen.add(stable_id)
            if len(merged) >= limit:
                break
        return [self._fts_result(row) for row in merged]

    def cross_session_selected_evidence(
        self, stable_ids: tuple[str, ...]
    ) -> list[CrossSessionEvidenceRecord]:
        selected: list[CrossSessionEvidenceRecord] = []
        seen: set[str] = set()
        for stable_id in stable_ids:
            candidates = self._cross_session_evidence_for_id(stable_id)
            if candidates and candidates[0].stable_id not in seen:
                selected.append(candidates[0])
                seen.add(candidates[0].stable_id)
        return selected

    def cross_session_evidence_candidates(
        self, stable_ids: tuple[str, ...]
    ) -> list[CrossSessionEvidenceRecord]:
        candidates: list[CrossSessionEvidenceRecord] = []
        seen: set[str] = set()
        for stable_id in stable_ids:
            for candidate in self._cross_session_evidence_for_id(stable_id):
                if candidate.stable_id not in seen:
                    candidates.append(candidate)
                    seen.add(candidate.stable_id)
        return candidates

    def _cross_session_evidence_for_id(self, stable_id: str) -> list[CrossSessionEvidenceRecord]:
        prefix, _, raw_id = stable_id.partition(":")
        if not raw_id.isdigit():
            return []
        source_id = int(raw_id)
        with self.connect() as connection:
            if prefix == "transcript-version":
                row = connection.execute(
                    """SELECT version.id, segment.session_id, sessions.title, segment.source,
                              segment.start_ms, segment.end_ms, version.text
                       FROM transcript_versions AS version
                       JOIN transcript_segments AS segment ON segment.id = version.segment_id
                       JOIN sessions ON sessions.id = segment.session_id
                       WHERE version.id = ? AND sessions.trashed_at_utc IS NULL""",
                    (source_id,),
                ).fetchone()
                if row is None:
                    return []
                candidates = [
                    CrossSessionEvidenceRecord(
                        stable_id=stable_id,
                        session_id=row["session_id"],
                        session_title=row["title"],
                        kind="transcript",
                        source=row["source"],
                        start_ms=row["start_ms"],
                        end_ms=row["end_ms"],
                        content_text=row["text"],
                        resource_path=None,
                        transcript_version_id=row["id"],
                    )
                ]
                frame_rows = connection.execute(
                    """SELECT frame.id, frame.session_id, frame.source_id, frame.ts_ms, frame.path,
                              sessions.title
                       FROM frames AS frame JOIN sessions ON sessions.id = frame.session_id
                       WHERE frame.session_id = ? AND sessions.trashed_at_utc IS NULL
                         AND frame.ts_ms BETWEEN ? AND ?
                       ORDER BY abs(frame.ts_ms - ?), frame.id LIMIT 4""",
                    (
                        row["session_id"],
                        max(0, row["start_ms"] - 30_000),
                        row["end_ms"] + 30_000,
                        row["start_ms"],
                    ),
                ).fetchall()
                candidates.extend(
                    CrossSessionEvidenceRecord(
                        stable_id=f"frame:{frame['id']}",
                        session_id=frame["session_id"],
                        session_title=frame["title"],
                        kind="frame",
                        source=frame["source_id"],
                        start_ms=frame["ts_ms"],
                        end_ms=frame["ts_ms"],
                        content_text=None,
                        resource_path=Path(frame["path"]),
                        frame_id=frame["id"],
                    )
                    for frame in frame_rows
                )
                return candidates
            if prefix == "answer-version":
                row = connection.execute(
                    """SELECT answer.id, answer.question_id, answer.answer,
                              question.session_id, sessions.title, question.asked_at_ms,
                              question.context_start_ms, question.context_end_ms
                       FROM answer_versions AS answer
                       JOIN questions AS question ON question.id = answer.question_id
                       JOIN sessions ON sessions.id = question.session_id
                       WHERE answer.id = ? AND answer.request_status = 'succeeded'
                         AND answer.answer IS NOT NULL AND question.state = 'submitted'
                         AND sessions.trashed_at_utc IS NULL""",
                    (source_id,),
                ).fetchone()
                if row is None:
                    return []
                result = [
                    CrossSessionEvidenceRecord(
                        stable_id=stable_id,
                        session_id=row["session_id"],
                        session_title=row["title"],
                        kind="answer",
                        source="问答",
                        start_ms=row["context_start_ms"] or row["asked_at_ms"],
                        end_ms=row["context_end_ms"] or row["asked_at_ms"],
                        content_text=row["answer"],
                        resource_path=None,
                        answer_version_id=row["id"],
                    )
                ]
                result.extend(self._answer_or_material_evidence(connection, "answer", source_id))
                return result
            if prefix == "material-version":
                row = connection.execute(
                    """SELECT material.id, material.session_id, sessions.title, material.content,
                              COALESCE((SELECT MIN(start_ms) FROM session_material_evidence
                                        WHERE material_version_id = material.id), 0) AS start_ms,
                              COALESCE((SELECT MAX(end_ms) FROM session_material_evidence
                                        WHERE material_version_id = material.id), 0) AS end_ms
                       FROM session_material_versions AS material
                       JOIN sessions ON sessions.id = material.session_id
                       WHERE material.id = ? AND material.request_status = 'succeeded'
                         AND sessions.trashed_at_utc IS NULL""",
                    (source_id,),
                ).fetchone()
                if row is None:
                    return []
                result = [
                    CrossSessionEvidenceRecord(
                        stable_id=stable_id,
                        session_id=row["session_id"],
                        session_title=row["title"],
                        kind="material",
                        source="材料",
                        start_ms=row["start_ms"],
                        end_ms=row["end_ms"],
                        content_text=row["content"],
                        resource_path=None,
                        material_version_id=row["id"],
                    )
                ]
                result.extend(self._answer_or_material_evidence(connection, "material", source_id))
                return result
            if prefix == "frame":
                row = connection.execute(
                    """SELECT frame.id, frame.session_id, sessions.title, frame.source_id,
                              frame.ts_ms, frame.path
                       FROM frames AS frame JOIN sessions ON sessions.id = frame.session_id
                       WHERE frame.id = ? AND sessions.trashed_at_utc IS NULL""",
                    (source_id,),
                ).fetchone()
                if row is None:
                    return []
                return [
                    CrossSessionEvidenceRecord(
                        stable_id=stable_id,
                        session_id=row["session_id"],
                        session_title=row["title"],
                        kind="frame",
                        source=row["source_id"],
                        start_ms=row["ts_ms"],
                        end_ms=row["ts_ms"],
                        content_text=None,
                        resource_path=Path(row["path"]),
                        frame_id=row["id"],
                    )
                ]
        return []

    @staticmethod
    def _answer_or_material_evidence(
        connection: sqlite3.Connection, owner: str, owner_id: int
    ) -> list[CrossSessionEvidenceRecord]:
        if owner == "answer":
            rows = connection.execute(
                """SELECT evidence.*, question.session_id, sessions.title
                   FROM answer_evidence AS evidence
                   JOIN answer_versions AS answer ON answer.id = evidence.answer_version_id
                   JOIN questions AS question ON question.id = answer.question_id
                   JOIN sessions ON sessions.id = question.session_id
                   WHERE evidence.answer_version_id = ? AND sessions.trashed_at_utc IS NULL
                   ORDER BY evidence.ordinal""",
                (owner_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """SELECT evidence.*, material.session_id, sessions.title
                   FROM session_material_evidence AS evidence
                   JOIN session_material_versions AS material
                     ON material.id = evidence.material_version_id
                   JOIN sessions ON sessions.id = material.session_id
                   WHERE evidence.material_version_id = ? AND sessions.trashed_at_utc IS NULL
                   ORDER BY evidence.ordinal""",
                (owner_id,),
            ).fetchall()
        return [
            CrossSessionEvidenceRecord(
                stable_id=row["stable_id"],
                session_id=row["session_id"],
                session_title=row["title"],
                kind=row["kind"],
                source=row["source"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                content_text=row["content_text"],
                resource_path=Path(row["resource_path"]) if row["resource_path"] else None,
                transcript_version_id=row["transcript_version_id"],
                frame_id=row["frame_id"],
                answer_version_id=row["answer_version_id"] if owner == "answer" else None,
                material_version_id=row["material_version_id"] if owner == "material" else None,
            )
            for row in rows
        ]

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
                """UPDATE sessions
                   SET ended_at_utc = ?, status = ?, retention_started_at_utc = ?
                   WHERE id = ?""",
                (ended_at_utc, status, ended_at_utc, session_id),
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

    def list_sessions(
        self,
        *,
        query: str = "",
        status: str = "all",
        newest_first: bool = True,
        current_session_id: str | None = None,
    ) -> list[SessionRecord]:
        clauses: list[str] = []
        parameters: list[object] = []
        if status == "trash":
            clauses.append("sessions.trashed_at_utc IS NOT NULL")
        else:
            clauses.append("sessions.trashed_at_utc IS NULL")
            if status != "all":
                clauses.append("sessions.status = ?")
                parameters.append(status)
        if query.strip():
            pattern = f"%{query.strip()}%"
            clauses.append(
                """(sessions.title LIKE ? OR EXISTS (
                       SELECT 1
                       FROM transcript_segments AS searchable_segment
                       JOIN effective_transcript_versions AS searchable_version
                         ON searchable_version.segment_id = searchable_segment.id
                       WHERE searchable_segment.session_id = sessions.id
                         AND searchable_version.text LIKE ?
                   ))"""
            )
            parameters.extend((pattern, pattern))
        direction = "DESC" if newest_first else "ASC"
        order = (
            " ORDER BY CASE WHEN sessions.id = ? THEN 0 ELSE 1 END, "
            f"CASE WHEN sessions.pinned_at_utc IS NOT NULL THEN 0 ELSE 1 END, "
            f"sessions.started_at_utc {direction}, sessions.id"
        )
        parameters.append(current_session_id)
        with self.connect() as connection:
            rows = connection.execute(
                SESSION_SUMMARY_QUERY + " WHERE " + " AND ".join(clauses) + order,
                parameters,
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
            pinned=row["pinned_at_utc"] is not None,
            retention_started_at_utc=row["retention_started_at_utc"],
            trashed_at_utc=row["trashed_at_utc"],
            trash_expires_at_utc=row["trash_expires_at_utc"],
        )

    def interrupt_recording_sessions(
        self, ended_at_utc: str, *, exclude_session_id: str | None = None
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE sessions
                   SET status = 'interrupted', ended_at_utc = ?, retention_started_at_utc = ?
                   WHERE status = 'recording' AND id IS NOT ?""",
                (ended_at_utc, ended_at_utc, exclude_session_id),
            )

    def complete_interrupted_session(self, session_id: str, completed_at_utc: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE sessions SET status = 'complete', ended_at_utc = ?,
                          retention_started_at_utc = ?
                   WHERE id = ? AND status = 'interrupted' AND trashed_at_utc IS NULL""",
                (completed_at_utc, completed_at_utc, session_id),
            )
        return cursor.rowcount == 1

    def set_session_pinned(
        self,
        session_id: str,
        pinned_at_utc: str | None,
        retention_started_at_utc: str,
    ) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE sessions
                   SET pinned_at_utc = ?,
                       retention_started_at_utc = CASE WHEN ? IS NULL THEN ?
                                                     ELSE retention_started_at_utc END
                   WHERE id = ? AND trashed_at_utc IS NULL""",
                (pinned_at_utc, pinned_at_utc, retention_started_at_utc, session_id),
            )
            if cursor.rowcount and pinned_at_utc is None:
                connection.execute(
                    "DELETE FROM session_notifications WHERE session_id = ?", (session_id,)
                )
        return cursor.rowcount == 1

    def move_session_to_trash(
        self,
        session_id: str,
        trashed_at_utc: str,
        expires_at_utc: str,
        *,
        allow_pinned: bool = True,
        expected_session: SessionRecord | None = None,
    ) -> bool:
        conditions = ["id = ?", "trashed_at_utc IS NULL", "status != 'recording'"]
        parameters: list[object] = [session_id]
        if not allow_pinned:
            conditions.append("pinned_at_utc IS NULL")
        if expected_session is not None:
            conditions.append("status = ?")
            parameters.append(expected_session.status)
            if expected_session.retention_started_at_utc is None:
                conditions.append("retention_started_at_utc IS NULL")
            else:
                conditions.append("retention_started_at_utc = ?")
                parameters.append(expected_session.retention_started_at_utc)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""UPDATE sessions SET trashed_at_utc = ?, trash_expires_at_utc = ?
                   WHERE {" AND ".join(conditions)}""",
                (trashed_at_utc, expires_at_utc, *parameters),
            )
        return cursor.rowcount == 1

    def restore_session(self, session_id: str, restored_at_utc: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE sessions
                   SET trashed_at_utc = NULL, trash_expires_at_utc = NULL,
                       retention_started_at_utc = ?
                   WHERE id = ? AND trashed_at_utc IS NOT NULL
                     AND trash_expires_at_utc > ?""",
                (restored_at_utc, session_id, restored_at_utc),
            )
            if cursor.rowcount:
                connection.execute(
                    "DELETE FROM session_notifications WHERE session_id = ?", (session_id,)
                )
        return cursor.rowcount == 1

    def record_session_notification(
        self,
        session_id: str,
        kind: SessionNotificationKind,
        notified_at_utc: str,
        *,
        expected_session: SessionRecord | None = None,
        expected_trashed: bool | None = None,
    ) -> bool:
        with self.connect() as connection:
            if expected_session is None and expected_trashed is None:
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO session_notifications(session_id, kind, notified_at_utc)
                       VALUES (?, ?, ?)""",
                    (session_id, kind.value, notified_at_utc),
                )
            else:
                conditions = ["id = ?"]
                parameters: list[object] = [session_id]
                if expected_session is not None:
                    conditions.extend(
                        [
                            "status = ?",
                            "status != 'recording'",
                            "pinned_at_utc IS NULL",
                        ]
                    )
                    parameters.append(expected_session.status)
                    if expected_session.retention_started_at_utc is None:
                        conditions.append("retention_started_at_utc IS NULL")
                    else:
                        conditions.append("retention_started_at_utc = ?")
                        parameters.append(expected_session.retention_started_at_utc)
                if expected_trashed is True:
                    conditions.append("trashed_at_utc IS NOT NULL")
                elif expected_trashed is False or expected_session is not None:
                    conditions.append("trashed_at_utc IS NULL")
                cursor = connection.execute(
                    f"""INSERT OR IGNORE INTO session_notifications(session_id, kind, notified_at_utc)
                       SELECT ?, ?, ?
                       WHERE EXISTS (SELECT 1 FROM sessions WHERE {" AND ".join(conditions)})""",
                    (session_id, kind.value, notified_at_utc, *parameters),
                )
        return cursor.rowcount == 1

    def record_final_delete_failure(self, session_id: str, notified_at_utc: str) -> bool:
        with self.connect() as connection:
            pending = connection.execute(
                "SELECT 1 FROM pending_media_deletions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if pending is not None:
                cursor = connection.execute(
                    """UPDATE pending_media_deletions
                       SET failure_notified_at_utc = ?
                       WHERE session_id = ? AND failure_notified_at_utc IS NULL""",
                    (notified_at_utc, session_id),
                )
                return cursor.rowcount == 1
            session = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND trashed_at_utc IS NOT NULL",
                (session_id,),
            ).fetchone()
            if session is None:
                return False
            cursor = connection.execute(
                """INSERT OR IGNORE INTO session_notifications(
                       session_id, kind, notified_at_utc
                   ) VALUES (?, ?, ?)""",
                (session_id, SessionNotificationKind.FINAL_DELETE_FAILED.value, notified_at_utc),
            )
        return cursor.rowcount == 1

    def pending_media_deletions(self) -> tuple[PendingMediaDeletion, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT session_id, title, path FROM pending_media_deletions ORDER BY session_id"
            ).fetchall()
        pending: list[PendingMediaDeletion] = []
        for row in rows:
            path = Path(row["path"])
            if not path.is_absolute():
                path = self.path.parent / path
            pending.append(PendingMediaDeletion(row["session_id"], row["title"], path))
        return tuple(pending)

    def finalize_pending_media_deletion(self, session_id: str) -> bool:
        pending = next(
            (item for item in self.pending_media_deletions() if item.session_id == session_id),
            None,
        )
        if pending is None:
            return False
        if pending.path.exists():
            shutil.rmtree(pending.path)
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM pending_media_deletions WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount == 1

    def permanently_delete_session(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if session is None or session.trashed_at_utc is None:
            return False
        media_dir = self.path.parent / "sessions" / session_id
        staged_dir = media_dir.with_name(f".{session_id}.deleting")
        if media_dir.exists():
            if staged_dir.exists():
                raise OSError("会话媒体正在等待删除")
            os.replace(media_dir, staged_dir)
        elif not staged_dir.exists():
            staged_dir = None
        try:
            with self.connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM sessions WHERE id = ? AND trashed_at_utc IS NOT NULL",
                    (session_id,),
                )
                if cursor.rowcount != 1:
                    deleted = False
                else:
                    connection.execute(
                        """UPDATE cross_session_syntheses
                           SET evidence_state = 'unavailable'
                           WHERE id IN (
                               SELECT synthesis_id FROM cross_session_synthesis_evidence
                               WHERE session_id = ?
                           )""",
                        (session_id,),
                    )
                    connection.execute(
                        "DELETE FROM transcript_fts WHERE session_id = ?", (session_id,)
                    )
                    connection.execute(
                        "DELETE FROM cross_session_fts WHERE session_id = ?", (session_id,)
                    )
                    if staged_dir is not None:
                        connection.execute(
                            """INSERT OR REPLACE INTO pending_media_deletions(session_id, title, path)
                               VALUES (?, ?, ?)""",
                            (
                                session_id,
                                session.title,
                                str(staged_dir.relative_to(self.path.parent)),
                            ),
                        )
                    deleted = True
        except Exception:
            if staged_dir is not None and staged_dir.exists() and not media_dir.exists():
                os.replace(staged_dir, media_dir)
            raise
        if not deleted:
            if staged_dir is not None and staged_dir.exists() and not media_dir.exists():
                os.replace(staged_dir, media_dir)
            return False
        if staged_dir is not None:
            self.finalize_pending_media_deletion(session_id)
        return True

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

    def timeline_transcript_versions(
        self,
        session_id: str,
        version_ids: tuple[int, ...],
        start_ms: int,
        end_ms: int,
    ) -> list[TimelineTranscriptRecord]:
        if not version_ids:
            return []
        placeholders = ", ".join("?" for _ in version_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT segment.id, segment.source, segment.start_ms, segment.end_ms,
                           version.text, version.id AS version_id,
                           version.kind AS version_kind, original.text AS original_text,
                           CASE
                               WHEN settings.enabled IS NULL OR settings.enabled = 0 THEN NULL
                               WHEN version.kind = 'user_edit' THEN 'edited'
                               WHEN version.kind = 'correction' THEN 'corrected'
                               ELSE 'pending'
                           END AS correction_state
                    FROM transcript_versions AS version
                    JOIN transcript_segments AS segment ON segment.id = version.segment_id
                    JOIN transcript_versions AS original
                      ON original.segment_id = segment.id AND original.kind = 'original'
                    LEFT JOIN transcript_correction_settings AS settings
                      ON settings.session_id = segment.session_id
                    WHERE segment.session_id = ?
                      AND segment.end_ms >= ? AND segment.start_ms <= ?
                      AND version.id IN ({placeholders})
                    ORDER BY segment.start_ms, segment.id""",
                (session_id, start_ms, end_ms, *version_ids),
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

    def record_source_event(
        self,
        session_id: str,
        source: str,
        kind: str | SourceEventKind,
        start_ms: int,
        end_ms: int,
        message: str,
    ) -> int:
        kind_value = kind.value if isinstance(kind, SourceEventKind) else kind
        if kind_value not in {item.value for item in SourceEventKind}:
            raise ValueError(f"Unsupported source event kind: {kind_value}")
        if start_ms < 0 or end_ms < start_ms:
            raise ValueError("Source event range is invalid")
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO source_events(
                       session_id, source, kind, start_ms, end_ms, message, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source,
                    kind_value,
                    start_ms,
                    end_ms,
                    message,
                    datetime.now(UTC).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def source_event(self, event_id: int) -> SourceEventRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id, session_id, source, kind, start_ms, end_ms, message,
                          data_loss_confirmed
                   FROM source_events WHERE id = ?""",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return SourceEventRecord(
            id=row["id"],
            session_id=row["session_id"],
            source=row["source"],
            kind=row["kind"],
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            message=row["message"],
            data_loss_confirmed=bool(row["data_loss_confirmed"]),
        )

    def source_events(
        self, session_id: str, start_ms: int = 0, end_ms: int | None = None
    ) -> list[SourceEventRecord]:
        with self.connect() as connection:
            if end_ms is None:
                rows = connection.execute(
                    """SELECT id, session_id, source, kind, start_ms, end_ms, message,
                              data_loss_confirmed
                       FROM source_events
                       WHERE session_id = ? AND end_ms >= ?
                       ORDER BY start_ms, id""",
                    (session_id, start_ms),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT id, session_id, source, kind, start_ms, end_ms, message,
                              data_loss_confirmed
                       FROM source_events
                       WHERE session_id = ? AND end_ms >= ? AND start_ms <= ?
                       ORDER BY start_ms, id""",
                    (session_id, start_ms, end_ms),
                ).fetchall()
        return [
            SourceEventRecord(
                id=row["id"],
                session_id=row["session_id"],
                source=row["source"],
                kind=row["kind"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                message=row["message"],
                data_loss_confirmed=bool(row["data_loss_confirmed"]),
            )
            for row in rows
        ]

    def add_timeline_event(
        self,
        session_id: str,
        kind: str | TimelineEventKind,
        source: str | None,
        start_ms: int,
        end_ms: int,
        message: str,
        *,
        source_event_id: int | None = None,
    ) -> int:
        kind_value = kind.value if isinstance(kind, TimelineEventKind) else kind
        if kind_value not in {item.value for item in TimelineEventKind}:
            raise ValueError(f"Unsupported timeline event kind: {kind_value}")
        if start_ms < 0 or end_ms < start_ms:
            raise ValueError("Timeline event range is invalid")
        with self.connect() as connection:
            if source_event_id is not None:
                existing = connection.execute(
                    """SELECT id FROM timeline_events
                       WHERE kind = ? AND source_event_id = ?""",
                    (kind_value, source_event_id),
                ).fetchone()
                if existing is not None:
                    return int(existing["id"])
            cursor = connection.execute(
                """INSERT INTO timeline_events(
                       session_id, kind, source, start_ms, end_ms, message, source_event_id
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    kind_value,
                    source,
                    start_ms,
                    end_ms,
                    message,
                    source_event_id,
                ),
            )
            return int(cursor.lastrowid)

    def finish_timeline_event(self, event_id: int, end_ms: int) -> bool:
        if end_ms < 0:
            raise ValueError("Timeline event end is invalid")
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE timeline_events
                   SET end_ms = ?
                   WHERE id = ? AND kind = 'pause' AND end_ms = start_ms AND start_ms <= ?""",
                (end_ms, event_id, end_ms),
            )
        return cursor.rowcount == 1

    def timeline_events(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[TimelineEventRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, session_id, kind, source, start_ms, end_ms, message,
                          source_event_id
                   FROM timeline_events
                   WHERE session_id = ? AND end_ms >= ? AND start_ms <= ?
                   ORDER BY start_ms, id""",
                (session_id, start_ms, end_ms),
            ).fetchall()
        return [TimelineEventRecord(**dict(row)) for row in rows]

    def confirm_data_gap(self, source_event_id: int) -> int:
        with self.connect() as connection:
            source = connection.execute(
                """SELECT id, session_id, source, start_ms, end_ms, message
                   FROM source_events WHERE id = ?""",
                (source_event_id,),
            ).fetchone()
            if source is None:
                raise KeyError(f"Unknown source event: {source_event_id}")
            existing = connection.execute(
                """SELECT id FROM timeline_events
                   WHERE kind = 'data_gap' AND source_event_id = ?""",
                (source_event_id,),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            cursor = connection.execute(
                """INSERT INTO timeline_events(
                       session_id, kind, source, start_ms, end_ms, message, source_event_id
                   ) VALUES (?, 'data_gap', ?, ?, ?, ?, ?)""",
                (
                    source["session_id"],
                    source["source"],
                    source["start_ms"],
                    source["end_ms"],
                    source["message"],
                    source_event_id,
                ),
            )
            connection.execute(
                "UPDATE source_events SET data_loss_confirmed = 1 WHERE id = ?",
                (source_event_id,),
            )
            return int(cursor.lastrowid)

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

    def persist_transcript_segments(
        self,
        session_id: str,
        chunk_id: int,
        source: str,
        records: list[tuple[int, int, str, str | None, float | None]],
    ) -> list[int]:
        with self.connect() as connection:
            segment_ids: list[int] = []
            for start_ms, end_ms, text, language, confidence in records:
                cursor = connection.execute(
                    """INSERT INTO transcript_segments(
                           session_id, audio_chunk_id, source, start_ms, end_ms,
                           text, language, confidence
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (session_id, chunk_id, source, start_ms, end_ms, text, language, confidence),
                )
                segment_id = int(cursor.lastrowid)
                segment_ids.append(segment_id)
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
                self._reindex_transcript_segment(connection, segment_id)
            connection.execute(
                "UPDATE audio_chunks SET state = 'transcribed', error = NULL WHERE id = ?",
                (chunk_id,),
            )
            return segment_ids

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
            self._reindex_transcript_segment(connection, segment_id)
            return segment_id

    def set_chunk_state(self, chunk_id: int, state: str, error: str | None = None) -> None:
        if state not in {"pending", "transcribed", "failed"}:
            raise ValueError(f"Unsupported audio chunk state: {state}")
        with self.connect() as connection:
            connection.execute(
                "UPDATE audio_chunks SET state = ?, error = ? WHERE id = ?",
                (state, error, chunk_id),
            )

    def audio_chunk_state(self, chunk_id: int) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state FROM audio_chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
        return str(row["state"]) if row is not None else None

    def reconcile_transcribed_audio_chunks(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE audio_chunks
                   SET state = 'transcribed', error = NULL
                   WHERE state = 'pending'
                     AND EXISTS (
                         SELECT 1 FROM transcript_segments
                         WHERE transcript_segments.audio_chunk_id = audio_chunks.id
                     )"""
            )
        return int(cursor.rowcount)

    def pending_audio_chunks(
        self, *, include_failed: bool = False
    ) -> tuple[PendingAudioChunk, ...]:
        states = ("pending", "failed") if include_failed else ("pending",)
        placeholders = ", ".join("?" for _ in states)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT chunks.id, chunks.session_id, chunks.source,
                              chunks.start_ms, chunks.end_ms, chunks.path
                       FROM audio_chunks AS chunks
                       JOIN sessions ON sessions.id = chunks.session_id
                       WHERE chunks.state IN ({placeholders})
                         AND sessions.trashed_at_utc IS NULL
                       ORDER BY chunks.session_id, chunks.start_ms, chunks.id""",
                states,
            ).fetchall()
        return tuple(
            PendingAudioChunk(
                id=int(row["id"]),
                session_id=str(row["session_id"]),
                source=str(row["source"]),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                path=Path(row["path"]),
            )
            for row in rows
        )

    def retry_failed_audio_chunks(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE audio_chunks SET state = 'pending', error = NULL
                   WHERE state = 'failed'
                     AND session_id IN (
                         SELECT id FROM sessions WHERE trashed_at_utc IS NULL
                     )"""
            )
        return int(cursor.rowcount)

    def recovery_audio_counts(self) -> tuple[int, int]:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT
                       (SELECT COUNT(*) FROM audio_chunks AS chunk
                          JOIN sessions AS session ON session.id = chunk.session_id
                         WHERE session.trashed_at_utc IS NULL AND chunk.state = 'pending')
                           AS pending_count,
                       (SELECT COUNT(*) FROM audio_chunks AS chunk
                          JOIN sessions AS session ON session.id = chunk.session_id
                         WHERE session.trashed_at_utc IS NULL AND chunk.state = 'failed')
                           AS failed_count"""
            ).fetchone()
        assert row is not None
        return int(row["pending_count"]), int(row["failed_count"])

    def session_runtime_counts(self, session_id: str) -> SessionRuntimeCounts:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT
                       (SELECT COUNT(*) FROM frames WHERE session_id = ?) AS frame_count,
                       (SELECT COALESCE(SUM(end_ms - start_ms), 0)
                          FROM audio_chunks
                         WHERE session_id = ? AND state = 'transcribed')
                           AS transcribed_audio_ms,
                       (SELECT COUNT(*) FROM transcript_segments WHERE session_id = ?)
                           AS transcript_count,
                       (SELECT COUNT(*) FROM transcript_correction_runs
                         WHERE session_id = ? AND state = 'running')
                           AS correction_backlog,
                       (SELECT COUNT(*) FROM audio_chunks
                         WHERE session_id = ? AND state = 'pending')
                           AS pending_audio_chunks,
                       (SELECT COUNT(*) FROM audio_chunks
                         WHERE session_id = ? AND state = 'failed')
                           AS failed_audio_chunks""",
                (session_id,) * 6,
            ).fetchone()
        assert row is not None
        return SessionRuntimeCounts(
            frame_count=int(row["frame_count"]),
            transcribed_audio_ms=int(row["transcribed_audio_ms"]),
            transcript_count=int(row["transcript_count"]),
            correction_backlog=int(row["correction_backlog"]),
            pending_audio_chunks=int(row["pending_audio_chunks"]),
            failed_audio_chunks=int(row["failed_audio_chunks"]),
        )

    def recover_running_model_invocations(self, completed_at_utc: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE model_invocations
                   SET status = 'failed',
                       retryable = CASE WHEN task_type IS NULL THEN 0 ELSE 1 END,
                       error = COALESCE(error, '应用异常退出，模型任务可重试'),
                       completed_at_utc = ?
                   WHERE status = 'running'""",
                (completed_at_utc,),
            )
        return int(cursor.rowcount)

    def failed_correction_windows(self) -> tuple[tuple[str, int], ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT session_id, window_start_ms
                   FROM transcript_correction_runs AS failed
                   WHERE failed.state = 'failed'
                     AND NOT EXISTS (
                         SELECT 1 FROM transcript_correction_runs AS newer
                         WHERE newer.session_id = failed.session_id
                           AND newer.window_start_ms = failed.window_start_ms
                           AND newer.id > failed.id
                           AND newer.state IN ('running', 'corrected', 'failed')
                     )
                   ORDER BY failed.id"""
            ).fetchall()
        return tuple((str(row["session_id"]), int(row["window_start_ms"])) for row in rows)

    def failed_correction_run_count(self) -> int:
        return len(self.failed_correction_windows())

    def recover_running_correction_runs(self, completed_at_utc: str) -> list[tuple[str, int]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT session_id, window_start_ms
                   FROM transcript_correction_runs
                   WHERE state = 'running'
                   ORDER BY id"""
            ).fetchall()
            if rows:
                connection.execute(
                    """UPDATE transcript_correction_runs
                       SET state = 'failed',
                           completed_at_utc = ?,
                           error_source = 'application',
                           error = COALESCE(error, '应用异常退出，字幕校订将重试')
                       WHERE state = 'running'""",
                    (completed_at_utc,),
                )
        return [(str(row["session_id"]), int(row["window_start_ms"])) for row in rows]

    def resolve_retryable_model_tasks(
        self,
        task_type: str,
        session_id: str | None,
        *,
        payload_key: str | None = None,
        payload_value: object | None = None,
    ) -> None:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, task_payload_json
                   FROM model_invocations
                   WHERE status = 'failed' AND retryable = 1
                     AND task_type = ? AND session_id IS ?""",
                (task_type, session_id),
            ).fetchall()
            ids: list[int] = []
            for row in rows:
                if payload_key is not None:
                    try:
                        payload = json.loads(row["task_payload_json"] or "{}")
                    except (TypeError, ValueError):
                        continue
                    if payload.get(payload_key) != payload_value:
                        continue
                ids.append(int(row["id"]))
            if ids:
                placeholders = ", ".join("?" for _ in ids)
                connection.execute(
                    f"UPDATE model_invocations SET retryable = 0 WHERE id IN ({placeholders})",
                    ids,
                )

    def resolve_model_fallbacks(self, invocation_ids: list[int]) -> None:
        if not invocation_ids:
            return
        placeholders = ", ".join("?" for _ in invocation_ids)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE model_invocations SET retryable = 0 WHERE id IN ({placeholders})",
                invocation_ids,
            )

    def retryable_model_task_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS count FROM model_invocations
                   WHERE status = 'failed' AND retryable = 1
                     AND task_type IN (
                         'answer', 'material', 'cross_session', 'transcript_correction'
                     )"""
            ).fetchone()
        return int(row["count"] if row is not None else 0)

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

    def all_effective_transcripts(self, session_id: str) -> list[AnswerTranscriptRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT segment.id AS segment_id, version.id AS version_id,
                          segment.source, segment.start_ms, segment.end_ms, version.text
                   FROM transcript_segments AS segment
                   JOIN effective_transcript_versions AS version
                     ON version.segment_id = segment.id
                   WHERE segment.session_id = ?
                   ORDER BY segment.start_ms, segment.id""",
                (session_id,),
            ).fetchall()
        return [AnswerTranscriptRecord(**dict(row)) for row in rows]

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
            version_id = int(cursor.lastrowid)
            self._reindex_transcript_segment(connection, segment_id)
            return version_id

    def undo_transcript_correction(self, segment_id: int) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE transcript_versions SET active = 0
                   WHERE segment_id = ? AND kind = 'correction'""",
                (segment_id,),
            )
            self._reindex_transcript_segment(connection, segment_id)

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
                """SELECT segment.id, version.id AS version_id,
                          segment.start_ms, segment.end_ms, segment.source, version.text
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

    def add_question_note(self, question_id: int, content: str) -> QuestionNoteRecord:
        content = content.strip()
        if not content:
            raise ValueError("Question note is required")
        created_at = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO question_notes(question_id, content, created_at_utc)
                   VALUES (?, ?, ?)""",
                (question_id, content, created_at),
            )
            note_id = int(cursor.lastrowid)
        return QuestionNoteRecord(note_id, question_id, content, created_at)

    def question_notes(self, question_id: int) -> list[QuestionNoteRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, question_id, content, created_at_utc
                   FROM question_notes WHERE question_id = ? ORDER BY id""",
                (question_id,),
            ).fetchall()
        return [QuestionNoteRecord(**dict(row)) for row in rows]

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
        model_invocation_id: int | None = None,
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
                       model_invocation_id, upstream_request_id, answer, error,
                       evidence_state, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    question_id,
                    version_number,
                    model,
                    connection_json,
                    request_status,
                    model_invocation_id,
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
            self._index_answer_version(connection, answer_version_id)
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
            model_invocation_id,
        )

    def answer_versions(self, question_id: int) -> list[AnswerVersionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, question_id, version_number, model, connection_json,
                          request_status, upstream_request_id AS request_id, answer, error,
                          evidence_state, created_at_utc, model_invocation_id
                   FROM answer_versions WHERE question_id = ? ORDER BY version_number""",
                (question_id,),
            ).fetchall()
        return [AnswerVersionRecord(**dict(row)) for row in rows]

    def session_answers(self, session_id: str) -> list[SessionAnswerRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT answer.id, answer.question_id, answer.version_number,
                          question.asked_at_ms, question.question, answer.model,
                          answer.request_status, answer.answer, answer.error,
                          answer.evidence_state, answer.created_at_utc
                   FROM answer_versions AS answer
                   JOIN questions AS question ON question.id = answer.question_id
                   WHERE question.session_id = ? AND question.state = 'submitted'
                   ORDER BY question.asked_at_ms, question.id, answer.version_number""",
                (session_id,),
            ).fetchall()
        return [SessionAnswerRecord(**dict(row)) for row in rows]

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

    def session_material_versions(self, session_id: str) -> list[SessionMaterialVersionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id, session_id, version_number, kind, content, template_id,
                          model, connection_json, model_invocation_id, request_status,
                          upstream_request_id AS request_id, error, evidence_state, created_at_utc
                   FROM session_material_versions
                   WHERE session_id = ? ORDER BY version_number""",
                (session_id,),
            ).fetchall()
        return [SessionMaterialVersionRecord(**dict(row)) for row in rows]

    def material_version(self, material_version_id: int) -> SessionMaterialVersionRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id, session_id, version_number, kind, content, template_id,
                          model, connection_json, model_invocation_id, request_status,
                          upstream_request_id AS request_id, error, evidence_state, created_at_utc
                   FROM session_material_versions WHERE id = ?""",
                (material_version_id,),
            ).fetchone()
        return SessionMaterialVersionRecord(**dict(row)) if row is not None else None

    def material_evidence(self, material_version_id: int) -> list[MaterialEvidenceRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT ordinal, stable_id, kind, source, start_ms, end_ms,
                          transcript_version_id, frame_id, content_text, resource_path
                   FROM session_material_evidence
                   WHERE material_version_id = ? ORDER BY ordinal""",
                (material_version_id,),
            ).fetchall()
        return [
            MaterialEvidenceRecord(
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

    def record_material_version(
        self,
        session_id: str,
        *,
        kind: str,
        content: str,
        template_id: str | None,
        model: str | None,
        connection_json: str | None,
        model_invocation_id: int | None,
        request_status: str,
        request_id: str | None,
        error: str | None,
        evidence_state: str,
        evidence: list[dict[str, object]],
    ) -> SessionMaterialVersionRecord:
        if kind not in {"generated", "user_edit"}:
            raise ValueError(f"Unsupported material version kind: {kind}")
        if request_status not in {"succeeded", "failed"}:
            raise ValueError(f"Unsupported material request status: {request_status}")
        if evidence_state not in {"exact", "unavailable"}:
            raise ValueError(f"Unsupported material evidence state: {evidence_state}")
        content = content.strip()
        if not content:
            raise ValueError("Material content is required")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version_number = int(
                connection.execute(
                    """SELECT COALESCE(MAX(version_number), 0) + 1
                       FROM session_material_versions WHERE session_id = ?""",
                    (session_id,),
                ).fetchone()[0]
            )
            created_at = datetime.now(UTC).isoformat()
            cursor = connection.execute(
                """INSERT INTO session_material_versions(
                       session_id, version_number, kind, content, template_id, model,
                       connection_json, model_invocation_id, request_status,
                       upstream_request_id, error, evidence_state, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    version_number,
                    kind,
                    content,
                    template_id,
                    model,
                    connection_json,
                    model_invocation_id,
                    request_status,
                    request_id,
                    error,
                    evidence_state,
                    created_at,
                ),
            )
            material_id = int(cursor.lastrowid)
            connection.executemany(
                """INSERT INTO session_material_evidence(
                       material_version_id, ordinal, stable_id, kind, source,
                       start_ms, end_ms, transcript_version_id, frame_id,
                       content_text, resource_path
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        material_id,
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
            self._index_material_version(connection, material_id)
        return SessionMaterialVersionRecord(
            material_id,
            session_id,
            version_number,
            kind,
            content,
            template_id,
            model,
            connection_json,
            model_invocation_id,
            request_status,
            request_id,
            error,
            evidence_state,
            created_at,
        )

    def record_material_edit(
        self, material_version_id: int, content: str
    ) -> SessionMaterialVersionRecord:
        parent = self.material_version(material_version_id)
        if parent is None:
            raise KeyError(f"Unknown material version: {material_version_id}")
        evidence = [
            {
                "stable_id": item.stable_id,
                "kind": item.kind,
                "source": item.source,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "transcript_version_id": item.transcript_version_id,
                "frame_id": item.frame_id,
                "content_text": item.content_text,
                "resource_path": str(item.resource_path) if item.resource_path else None,
            }
            for item in self.material_evidence(material_version_id)
        ]
        # Keep the original invocation as provenance; `user_edit` records that no new
        # model request produced this version while preserving its evidence lineage.
        return self.record_material_version(
            parent.session_id,
            kind="user_edit",
            content=content,
            template_id=parent.template_id,
            model=parent.model,
            connection_json=parent.connection_json,
            model_invocation_id=parent.model_invocation_id,
            request_status="succeeded",
            request_id=parent.request_id,
            error=None,
            evidence_state=parent.evidence_state,
            evidence=evidence,
        )

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

    def record_whisper_run(
        self,
        *,
        session_id: str,
        requested: WhisperSettings,
        actual: WhisperSettings,
        fallback_advice: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO whisper_runs(
                       session_id, profile, requested_model, requested_device,
                       requested_compute_type, actual_model, actual_device,
                       actual_compute_type, language, vad_enabled, vad_min_silence_ms,
                       fallback_advice, started_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    requested.profile.value,
                    requested.model,
                    requested.device,
                    requested.compute_type,
                    actual.model,
                    actual.device,
                    actual.compute_type,
                    actual.language,
                    int(actual.vad_enabled),
                    actual.vad_min_silence_ms,
                    fallback_advice,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def whisper_run(self, session_id: str) -> WhisperRunRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM whisper_runs WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        values = dict(row)
        values["vad_enabled"] = bool(values["vad_enabled"])
        return WhisperRunRecord(**values)

    def start_model_invocation(
        self,
        *,
        session_id: str | None,
        role: str,
        connection_id: str,
        connection_name: str,
        base_url: str,
        api_mode: str,
        model: str,
        reasoning_level: str,
        fallback_reason: str | None,
        evidence: tuple[ModelInvocationEvidenceRecord, ...] = (),
        task_type: str | None = None,
        task_payload_json: str | None = None,
    ) -> int:
        started_at = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO model_invocations(
                       session_id, role, connection_id, connection_name, base_url, api_mode,
                       model, reasoning_level, fallback_reason, status, retryable,
                       task_type, task_payload_json, started_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 1, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    connection_id,
                    connection_name,
                    base_url,
                    api_mode,
                    model,
                    reasoning_level,
                    fallback_reason,
                    task_type,
                    task_payload_json,
                    started_at,
                ),
            )
            invocation_id = int(cursor.lastrowid)
            connection.executemany(
                """INSERT INTO model_invocation_evidence(
                       invocation_id, ordinal, stable_id, kind, source, start_ms, end_ms,
                       transcript_version_id, frame_id
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        invocation_id,
                        ordinal,
                        item.stable_id,
                        item.kind,
                        item.source,
                        item.start_ms,
                        item.end_ms,
                        item.transcript_version_id,
                        item.frame_id,
                    )
                    for ordinal, item in enumerate(evidence)
                    if item.kind in {"transcript", "frame"}
                ],
            )
            connection.executemany(
                """INSERT INTO model_invocation_content_evidence(
                       invocation_id, ordinal, stable_id, kind, source, start_ms, end_ms,
                       answer_version_id, material_version_id, content_text, resource_path
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        invocation_id,
                        ordinal,
                        item.stable_id,
                        item.kind,
                        item.source,
                        item.start_ms,
                        item.end_ms,
                        item.answer_version_id,
                        item.material_version_id,
                        item.content_text,
                        str(item.resource_path) if item.resource_path else None,
                    )
                    for ordinal, item in enumerate(evidence)
                    if item.kind in {"answer", "material"}
                ],
            )
            return invocation_id

    def finish_model_invocation(
        self,
        invocation_id: int,
        status: str,
        *,
        request_id: str | None = None,
        error: str | None = None,
    ) -> ModelInvocationRecord:
        if status not in {"succeeded", "failed"}:
            raise ValueError(f"Invalid invocation status: {status}")
        completed_at = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE model_invocations
                   SET status = ?, retryable = ?, upstream_request_id = ?,
                       error = ?, completed_at_utc = ?
                   WHERE id = ? AND status = 'running'""",
                (status, int(status == "failed"), request_id, error, completed_at, invocation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Model invocation is unavailable or already finished")
        record = self.model_invocation(invocation_id)
        if record is None:
            raise RuntimeError("Finished model invocation is unavailable")
        return record

    def model_invocation(self, invocation_id: int) -> ModelInvocationRecord | None:
        records = self._model_invocations("WHERE invocation.id = ?", (invocation_id,))
        return records[0] if records else None

    def model_invocations(self, session_id: str | None = None) -> tuple[ModelInvocationRecord, ...]:
        if session_id is None:
            return self._model_invocations("", ())
        return self._model_invocations("WHERE invocation.session_id = ?", (session_id,))

    def running_model_invocations(self) -> tuple[ModelInvocationRecord, ...]:
        return self._model_invocations("WHERE invocation.status = 'running'", ())

    def _model_invocations(
        self, where: str, parameters: tuple[object, ...]
    ) -> tuple[ModelInvocationRecord, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT invocation.*,
                           (SELECT GROUP_CONCAT(stable_id, char(31))
                            FROM (
                                SELECT stable_id, ordinal, 0 AS source_order
                                FROM model_invocation_evidence
                                WHERE invocation_id = invocation.id
                                UNION ALL
                                SELECT stable_id, ordinal, 1 AS source_order
                                FROM model_invocation_content_evidence
                                WHERE invocation_id = invocation.id
                                ORDER BY ordinal, source_order
                            )) AS evidence_ids
                    FROM model_invocations AS invocation
                    {where}
                    ORDER BY invocation.id""",
                parameters,
            ).fetchall()
        return tuple(
            ModelInvocationRecord(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                connection_id=row["connection_id"],
                connection_name=row["connection_name"],
                base_url=row["base_url"],
                api_mode=row["api_mode"],
                model=row["model"],
                reasoning_level=row["reasoning_level"],
                fallback_reason=row["fallback_reason"],
                status=row["status"],
                request_id=row["upstream_request_id"],
                error=row["error"],
                started_at_utc=row["started_at_utc"],
                completed_at_utc=row["completed_at_utc"],
                evidence_ids=tuple((row["evidence_ids"] or "").split(chr(31)))
                if row["evidence_ids"]
                else (),
                task_type=row["task_type"],
                task_payload_json=row["task_payload_json"],
            )
            for row in rows
        )

    def record_cross_session_synthesis(
        self,
        *,
        question: str,
        answer: str | None,
        model: str | None,
        connection_json: str | None,
        model_invocation_id: int | None,
        request_status: str,
        request_id: str | None,
        error: str | None,
        evidence_state: str,
        evidence: tuple[CrossSessionEvidenceRecord, ...],
        retry_of_id: int | None = None,
    ) -> CrossSessionSynthesisRecord:
        if request_status not in {"succeeded", "failed"}:
            raise ValueError(f"Invalid synthesis status: {request_status}")
        if evidence_state not in {"exact", "unavailable"}:
            raise ValueError(f"Invalid synthesis evidence state: {evidence_state}")
        created_at = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO cross_session_syntheses(
                       question, answer, model, connection_json, model_invocation_id,
                       request_status, upstream_request_id, error, evidence_state,
                       created_at_utc, retry_of_id
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    question,
                    answer,
                    model,
                    connection_json,
                    model_invocation_id,
                    request_status,
                    request_id,
                    error,
                    evidence_state,
                    created_at,
                    retry_of_id,
                ),
            )
            synthesis_id = int(cursor.lastrowid)
            connection.executemany(
                """INSERT INTO cross_session_synthesis_evidence(
                       synthesis_id, ordinal, stable_id, session_id, session_title, kind, source,
                       start_ms, end_ms, transcript_version_id, frame_id,
                       answer_version_id, material_version_id, content_text, resource_path
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        synthesis_id,
                        ordinal,
                        item.stable_id,
                        item.session_id,
                        item.session_title,
                        item.kind,
                        item.source,
                        item.start_ms,
                        item.end_ms,
                        item.transcript_version_id,
                        item.frame_id,
                        item.answer_version_id,
                        item.material_version_id,
                        item.content_text,
                        str(item.resource_path) if item.resource_path else None,
                    )
                    for ordinal, item in enumerate(evidence)
                ],
            )
        record = self.cross_session_synthesis(synthesis_id)
        if record is None:
            raise RuntimeError("Cross-session synthesis was not persisted")
        return record

    def cross_session_synthesis(self, synthesis_id: int) -> CrossSessionSynthesisRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM cross_session_syntheses WHERE id = ?", (synthesis_id,)
            ).fetchone()
        if row is None:
            return None
        return CrossSessionSynthesisRecord(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            model=row["model"],
            connection_json=row["connection_json"],
            model_invocation_id=row["model_invocation_id"],
            request_status=row["request_status"],
            request_id=row["upstream_request_id"],
            error=row["error"],
            evidence_state=row["evidence_state"],
            created_at_utc=row["created_at_utc"],
            retry_of_id=row["retry_of_id"],
        )

    def reclaim_orphan_cross_session_retry_claims(self) -> int:
        cutoff = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """DELETE FROM cross_session_retry_claims AS claim
                   WHERE claim.claimed_at_utc < ?
                     AND NOT EXISTS (
                         SELECT 1 FROM cross_session_syntheses AS successor
                         WHERE successor.retry_of_id = claim.synthesis_id
                     )""",
                (cutoff,),
            )
        return int(cursor.rowcount)

    def claim_cross_session_retry(self, synthesis_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO cross_session_retry_claims(
                       synthesis_id, claimed_at_utc
                   ) VALUES (?, ?)""",
                (synthesis_id, datetime.now(UTC).isoformat()),
            )
        return cursor.rowcount == 1

    def release_cross_session_retry_claim(self, synthesis_id: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM cross_session_retry_claims WHERE synthesis_id = ?",
                (synthesis_id,),
            )

    def cross_session_successor_exists(self, synthesis_id: int) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM cross_session_syntheses WHERE retry_of_id = ? LIMIT 1",
                (synthesis_id,),
            ).fetchone()
        return row is not None

    def cross_session_synthesis_for_invocation(
        self, invocation_id: int
    ) -> CrossSessionSynthesisRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM cross_session_syntheses WHERE model_invocation_id = ?",
                (invocation_id,),
            ).fetchone()
        return self.cross_session_synthesis(int(row["id"])) if row is not None else None

    def cross_session_syntheses(
        self, *, limit: int = 50, request_status: str | None = None
    ) -> tuple[CrossSessionSynthesisRecord, ...]:
        bounded_limit = min(max(limit, 0), 100)
        with self.connect() as connection:
            if request_status is None:
                rows = connection.execute(
                    """SELECT id FROM cross_session_syntheses
                       ORDER BY id LIMIT ?""",
                    (bounded_limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT id FROM cross_session_syntheses
                       WHERE request_status = ?
                     AND evidence_state = 'exact'
                     AND NOT EXISTS (
                         SELECT 1 FROM cross_session_syntheses AS successor
                         WHERE successor.retry_of_id = cross_session_syntheses.id
                     )
                     AND NOT EXISTS (
                         SELECT 1 FROM cross_session_synthesis_evidence AS evidence
                         LEFT JOIN sessions ON sessions.id = evidence.session_id
                         WHERE evidence.synthesis_id = cross_session_syntheses.id
                           AND (sessions.id IS NULL OR sessions.trashed_at_utc IS NOT NULL)
                     )
                   ORDER BY id DESC LIMIT ?""",
                    (request_status, bounded_limit),
                ).fetchall()
        return tuple(
            record
            for row in rows
            if (record := self.cross_session_synthesis(int(row["id"]))) is not None
        )

    def cross_session_synthesis_evidence(
        self, synthesis_id: int
    ) -> tuple[CrossSessionSynthesisEvidenceRecord, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT evidence.*
                   FROM cross_session_synthesis_evidence AS evidence
                   WHERE evidence.synthesis_id = ?
                   ORDER BY evidence.ordinal""",
                (synthesis_id,),
            ).fetchall()
        return tuple(
            CrossSessionSynthesisEvidenceRecord(
                synthesis_id=row["synthesis_id"],
                ordinal=row["ordinal"],
                stable_id=row["stable_id"],
                session_id=row["session_id"],
                session_title=row["session_title"],
                kind=row["kind"],
                source=row["source"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                content_text=row["content_text"],
                resource_path=Path(row["resource_path"]) if row["resource_path"] else None,
                transcript_version_id=row["transcript_version_id"],
                frame_id=row["frame_id"],
                answer_version_id=row["answer_version_id"],
                material_version_id=row["material_version_id"],
            )
            for row in rows
        )
