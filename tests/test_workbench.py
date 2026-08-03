from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from jingzhi.application import JingzhiApplicationService
from jingzhi.database import Database


class FakeRecorder:
    def __init__(self, database: Database, now: datetime) -> None:
        self.database = database
        self.now = now
        self.session_id: str | None = None

    @property
    def is_recording(self) -> bool:
        return self.session_id is not None

    def start(
        self,
        title: str,
        *,
        capture_system_audio: bool | None = None,
        capture_microphone: bool | None = None,
    ) -> str:
        del capture_system_audio, capture_microphone
        self.session_id = self.database.create_session(title, self.now.isoformat())
        return self.session_id

    def stop(self) -> str | None:
        if self.session_id is None:
            return None
        session_id = self.session_id
        self.database.finish_session(session_id, self.now.isoformat(), "complete")
        self.session_id = None
        return session_id


def test_legacy_database_migrates_frames_without_losing_data(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                status TEXT NOT NULL
            );
            CREATE TABLE frames (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                ts_ms INTEGER NOT NULL,
                path TEXT NOT NULL,
                perceptual_hash TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            );
            INSERT INTO sessions VALUES (
                'legacy-session', '旧会话', '2026-08-01T08:00:00+00:00',
                '2026-08-01T08:01:00+00:00', 'complete'
            );
            INSERT INTO frames VALUES (
                17, 'legacy-session', 12000, 'legacy.webp', 'hash', 1280, 720
            );
            """
        )

    database = Database(path)

    sessions = database.list_sessions()
    frames = database.timeline_frames("legacy-session", 0, 60_000)
    assert [(item.id, item.title, item.frame_count) for item in sessions] == [
        ("legacy-session", "旧会话", 1)
    ]
    assert [(item.id, item.source_id, item.ts_ms) for item in frames] == [
        (17, "display:primary", 12_000)
    ]
    with database.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert [row["version"] for row in versions] == [1, 2, 3]


def test_application_service_browses_sessions_and_scaled_keyframes_without_hardware(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 2, 9, 30, tzinfo=UTC)
    database = Database(tmp_path / "test.sqlite3")
    recorder = FakeRecorder(database, now)
    service = JingzhiApplicationService(database, recorder=recorder, now=lambda: now)

    session_id = service.start_session("多来源会话", capture_system_audio=False)
    first_path = tmp_path / "display-1.webp"
    second_path = tmp_path / "display-2.webp"
    Image.new("RGB", (320, 180), "white").save(first_path)
    Image.new("RGB", (320, 180), "navy").save(second_path)
    first_id = database.add_frame(
        session_id,
        15_000,
        first_path,
        "first",
        (320, 180),
        source_id="display:1",
    )
    database.add_frame(
        session_id,
        320_000,
        second_path,
        "second",
        (320, 180),
        source_id="display:2",
    )
    service.stop_session()

    summaries = service.list_sessions()
    timeline = service.open_session(session_id, window_start_ms=0, window_duration_ms=60_000)

    assert summaries[0].id == session_id
    assert summaries[0].frame_count == 2
    assert timeline.session.id == session_id
    assert timeline.duration_ms == 320_000
    assert [(frame.id, frame.source_id) for frame in timeline.frames] == [
        (first_id, "display:1")
    ]

    clamped = service.open_session(
        session_id, window_start_ms=999_000, window_duration_ms=60_000
    )
    assert clamped.window_start_ms == 260_000
    assert clamped.window_end_ms == 320_000
