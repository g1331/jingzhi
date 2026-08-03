from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jingzhi.application import ModelConnectionSnapshot, QuestionAnsweringService
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.llm import AnswerModelResult
from jingzhi.session import SessionManager


class RecordingAnswerModel:
    def __init__(self) -> None:
        self.contexts = []

    def answer(self, question, context):
        self.contexts.append((question, context))
        return AnswerModelResult(
            f"answer {len(self.contexts)}",
            f"request-{len(self.contexts)}",
            "resolved-answer-model",
        )


def add_transcript(database: Database, session_id: str, path: Path, text: str) -> int:
    chunk_id = database.add_audio_chunk(session_id, "system", 1_000, 2_000, path)
    return database.add_transcript(session_id, chunk_id, "system", 1_000, 2_000, text, "zh", 0.9)


def test_answer_persists_exact_model_evidence_and_reanswer_uses_latest_version(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("test", "2026-01-01T00:00:00+00:00")
    existing_frame = tmp_path / "frame.webp"
    existing_frame.write_bytes(b"frame")
    frame_id = database.add_frame(
        session_id, 1_500, existing_frame, "0" * 64, (100, 100), source_id="display:2"
    )
    database.add_frame(session_id, 1_700, tmp_path / "missing.webp", "1" * 64, (100, 100))
    segment_id = add_transcript(database, session_id, tmp_path / "audio.wav", "原始字幕")
    original_version_id = database.transcript_versions(segment_id)[0].id
    model = RecordingAnswerModel()
    service = QuestionAnsweringService(
        database,
        model,
        ModelConnectionSnapshot("answer-model", "https://example.test/v1", "responses"),
    )

    first = service.ask(session_id, 3_000, "发生了什么？", lookback_ms=3_000)

    assert first.answer == "answer 1"
    assert first.version_number == 1
    assert first.model == "resolved-answer-model"
    assert first.request_id == "request-1"
    sent_context = model.contexts[0][1]
    assert [item.frame_id for item in sent_context.frames] == [frame_id]
    assert [item.version_id for item in sent_context.transcripts] == [original_version_id]
    persisted = database.answer_evidence(first.id)
    persisted_items = [
        {
            "stable_id": item.stable_id,
            "kind": item.kind,
            "source": item.source,
            "start_ms": item.start_ms,
            "end_ms": item.end_ms,
            **(
                {
                    "transcript_version_id": item.transcript_version_id,
                    "content_text": item.content_text,
                }
                if item.kind == "transcript"
                else {"frame_id": item.frame_id, "resource_path": str(item.resource_path)}
            ),
        }
        for item in persisted
    ]
    assert persisted_items == sent_context.persistence_items()

    correction_id = database.add_transcript_version(
        segment_id, "correction", "校订字幕", model="correction-model"
    )
    assert correction_id is not None
    second = service.reanswer(first.question_id)

    assert second.version_number == 2
    assert [item.version_id for item in model.contexts[1][1].transcripts] == [correction_id]
    assert [item.transcript_version_id for item in database.answer_evidence(first.id)] == [
        original_version_id,
        None,
    ]
    assert [item.transcript_version_id for item in database.answer_evidence(second.id)] == [
        correction_id,
        None,
    ]

    user_edit_id = database.add_transcript_version(segment_id, "user_edit", "用户编辑字幕")
    assert user_edit_id is not None
    third = service.reanswer(first.question_id)
    assert third.version_number == 3
    assert [item.version_id for item in model.contexts[2][1].transcripts] == [user_edit_id]
    assert database.answer_evidence(first.id)[0].content_text == "原始字幕"


def test_question_anchor_is_fixed_before_user_finishes_typing(monkeypatch, tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session_id = manager.database.create_session("test", "2026-01-01T00:00:00+00:00")
    times = iter((1_000, 9_000))
    manager.session_id = session_id
    manager.clock = SimpleNamespace(now_ms=lambda: next(times))
    service = QuestionAnsweringService(
        manager.database,
        RecordingAnswerModel(),
        ModelConnectionSnapshot("answer-model", "", "responses"),
    )
    monkeypatch.setattr(manager, "_question_service", lambda: service)

    manager.capture_question_anchor()
    manager.answer("输入了较长时间的问题")

    assert manager.last_question_id is not None
    question = manager.database.question(manager.last_question_id)
    assert question is not None
    assert question.asked_at_ms == 1_000


def test_failed_request_remains_the_current_reanswer_target(monkeypatch, tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session_id = manager.database.create_session("test", "2026-01-01T00:00:00+00:00")
    manager.session_id = session_id
    manager.clock = SimpleNamespace(now_ms=lambda: 1_000)

    class FailingModel:
        def answer(self, question, context):
            raise RuntimeError("upstream failed")

    service = QuestionAnsweringService(
        manager.database,
        FailingModel(),
        ModelConnectionSnapshot("answer-model", "", "responses"),
    )
    monkeypatch.setattr(manager, "_question_service", lambda: service)

    with pytest.raises(RuntimeError, match="upstream failed"):
        manager.answer("失败问题")

    assert manager.last_question_id == manager.database.latest_question_id(session_id)


def test_reanswer_persisted_question_after_application_restart(monkeypatch, tmp_path: Path) -> None:
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = database.create_session("test", "2026-01-01T00:00:00+00:00")
    first_service = QuestionAnsweringService(
        database,
        RecordingAnswerModel(),
        ModelConnectionSnapshot("answer-model", "", "responses"),
    )
    first = first_service.ask(session_id, 1_000, "持久化问题")

    restarted = SessionManager(Settings(data_dir=tmp_path))
    second_service = QuestionAnsweringService(
        restarted.database,
        RecordingAnswerModel(),
        ModelConnectionSnapshot("answer-model", "", "responses"),
    )
    monkeypatch.setattr(restarted, "_question_service", lambda: second_service)

    answer = restarted.reanswer_question(first.question_id)

    assert answer == "answer 1"
    assert [
        item.version_number for item in restarted.database.answer_versions(first.question_id)
    ] == [
        1,
        2,
    ]


def test_failed_request_persists_connection_error_and_request_id(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("test", "2026-01-01T00:00:00+00:00")

    class RequestFailure(RuntimeError):
        request_id = "failed-request"

    class FailingModel:
        def answer(self, question, context):
            raise RequestFailure("upstream failed")

    service = QuestionAnsweringService(
        database,
        FailingModel(),
        ModelConnectionSnapshot("answer-model", "https://example.test/v1", "responses"),
    )

    try:
        service.ask(session_id, 1_000, "失败问题")
    except RequestFailure:
        pass
    else:
        raise AssertionError("request failure was not raised")

    with database.connect() as connection:
        question_id = int(connection.execute("SELECT id FROM questions").fetchone()[0])
    version = database.answer_versions(question_id)[0]
    assert version.request_status == "failed"
    assert version.request_id == "failed-request"
    assert version.error == "upstream failed"
    assert "api_key" not in (version.connection_json or "")


def test_legacy_answer_is_migrated_without_invented_evidence(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite3"
    database = Database(path)
    session_id = database.create_session("legacy", "2026-01-01T00:00:00+00:00")
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO questions(
                   session_id, asked_at_ms, question, answer, context_start_ms, context_end_ms
               ) VALUES (?, 1000, '旧问题', '旧回答', 0, 1000)""",
            (session_id,),
        )
        question_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
        connection.execute("PRAGMA user_version = 3")

    migrated = Database(path)
    versions = migrated.answer_versions(question_id)
    assert len(versions) == 1
    assert versions[0].answer == "旧回答"
    assert versions[0].evidence_state == "unavailable"
    assert migrated.answer_evidence(versions[0].id) == []

    reopened = Database(path)
    assert len(reopened.answer_versions(question_id)) == 1


def test_reopening_current_database_does_not_turn_pending_question_into_legacy_answer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test.sqlite3"
    database = Database(path)
    session_id = database.create_session("current", "2026-01-01T00:00:00+00:00")
    question_id = database.create_question(session_id, 1_000, "处理中", 0, 1_000)

    reopened = Database(path)

    assert reopened.answer_versions(question_id) == []
