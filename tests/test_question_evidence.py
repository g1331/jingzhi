from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jingzhi.application import (
    AnswerEvidenceSummary,
    JingzhiApplicationService,
    QuestionAnsweringService,
    present_answer,
)
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.llm import AnswerModelResult
from jingzhi.model_routing import ModelRouter
from jingzhi.provider_settings import default_saved_settings
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


def answer_service(database: Database, model) -> QuestionAnsweringService:
    router = ModelRouter(
        database,
        default_saved_settings("answer-model"),
        adapter_factory=lambda _connection, _model, _reasoning: model,
    )
    return QuestionAnsweringService(database, router)


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
    service = answer_service(database, model)

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

    application = JingzhiApplicationService(database, recorder=SimpleNamespace(is_recording=False))
    timeline = application.open_session(session_id, answer_version_id=first.id)
    summary = timeline.answer_evidence_summary
    assert summary is not None
    assert summary.frame_count == 1
    assert summary.transcript_count == 1
    assert (summary.start_ms, summary.end_ms) == (1_000, 2_000)
    assert summary.stable_ids == (
        f"transcript-version:{original_version_id}",
        f"frame:{frame_id}",
    )

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


def test_answer_presentation_requires_complete_boundary_sections() -> None:
    summary = AnswerEvidenceSummary(
        state="exact",
        frame_count=1,
        transcript_count=0,
        start_ms=1_000,
        end_ms=1_000,
        stable_ids=("frame:1",),
    )
    compliant = "## 会话证据确认\n\n证据支持 $x^2$。"

    assert present_answer(compliant, summary) == compliant
    assert present_answer("请按 ## 会话证据确认 格式回答", summary).startswith(
        "## 无法确认\n\n模型回答未标明依据边界"
    )
    assert present_answer("前言\n\n" + compliant, summary).startswith("## 无法确认")


@pytest.mark.parametrize(
    ("evidence_kind", "expected_counts", "expected_range"),
    [
        ("transcript", (0, 1), (1_000, 2_000)),
        ("frame", (1, 0), (1_500, 1_500)),
        (None, (0, 0), (None, None)),
    ],
)
def test_answer_evidence_summary_handles_single_source_and_empty_context(
    tmp_path: Path,
    evidence_kind: str | None,
    expected_counts: tuple[int, int],
    expected_range: tuple[int | None, int | None],
) -> None:
    database = Database(tmp_path / "summary.sqlite3")
    session_id = database.create_session("test", "2026-01-01T00:00:00+00:00")
    frame_id = database.add_frame(session_id, 1_500, tmp_path / "frame.webp", "0" * 64, (100, 100))
    segment_id = add_transcript(database, session_id, tmp_path / "audio.wav", "字幕")
    version_id = database.transcript_versions(segment_id)[0].id
    evidence_by_kind = {
        "transcript": {
            "stable_id": f"transcript-version:{version_id}",
            "kind": "transcript",
            "source": "system",
            "start_ms": 1_000,
            "end_ms": 2_000,
            "transcript_version_id": version_id,
            "content_text": "字幕",
        },
        "frame": {
            "stable_id": f"frame:{frame_id}",
            "kind": "frame",
            "source": "display:primary",
            "start_ms": 1_500,
            "end_ms": 1_500,
            "frame_id": frame_id,
            "resource_path": str(tmp_path / "frame.webp"),
        },
    }
    question_id = database.create_question(session_id, 3_000, "问题", 0, 3_000)
    answer = database.record_answer_version(
        question_id,
        model="answer-model",
        connection_json=None,
        request_status="succeeded",
        request_id=None,
        answer="回答",
        error=None,
        evidence_state="exact",
        evidence=[] if evidence_kind is None else [evidence_by_kind[evidence_kind]],
    )
    application = JingzhiApplicationService(database, recorder=SimpleNamespace(is_recording=False))

    summary = application.open_session(
        session_id, answer_version_id=answer.id
    ).answer_evidence_summary

    assert summary is not None
    assert (summary.frame_count, summary.transcript_count) == expected_counts
    assert (summary.start_ms, summary.end_ms) == expected_range


def test_application_service_persists_anchor_before_slow_input_and_applies_selected_range(
    monkeypatch, tmp_path: Path
) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session_id = manager.database.create_session("test", "2026-01-01T00:00:00+00:00")
    times = iter((400_000, 900_000))
    manager.session_id = session_id
    manager.clock = SimpleNamespace(now_ms=lambda: next(times))
    model = RecordingAnswerModel()
    question_service = answer_service(manager.database, model)
    monkeypatch.setattr(manager, "_question_service", lambda: question_service)
    service = JingzhiApplicationService(manager.database, recorder=manager)

    question_id = service.begin_question()
    anchor = manager.database.question(question_id)
    assert anchor is not None
    assert anchor.asked_at_ms == 400_000
    assert anchor.question == ""
    assert anchor.state == "draft"
    assert (anchor.context_start_ms, anchor.context_end_ms) == (280_000, 400_000)

    service.set_question_range(30_000)
    answer = service.submit_question("输入了较长时间的问题")

    assert answer == "answer 1"
    question = manager.database.question(question_id)
    assert question is not None
    assert question.asked_at_ms == 400_000
    assert (question.context_start_ms, question.context_end_ms) == (370_000, 400_000)
    assert question.state == "submitted"
    assert model.contexts[0][0] == "输入了较长时间的问题"


def test_repeated_trigger_reuses_pending_anchor_and_cancel_deletes_it_without_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session_id = manager.database.create_session("test", "2026-01-01T00:00:00+00:00")
    times = iter((400_000, 490_000))
    manager.session_id = session_id
    manager.clock = SimpleNamespace(now_ms=lambda: next(times))
    model = RecordingAnswerModel()
    monkeypatch.setattr(
        manager,
        "_question_service",
        lambda: answer_service(manager.database, model),
    )
    service = JingzhiApplicationService(manager.database, recorder=manager)

    first_id = service.begin_question(5 * 60_000)
    second_id = service.begin_question(30_000)

    assert second_id == first_id
    pending = manager.database.question(first_id)
    assert pending is not None
    assert (pending.context_start_ms, pending.context_end_ms) == (100_000, 400_000)
    assert manager.database.latest_question_id(session_id) is None
    assert manager.database.timeline_questions(session_id, 0, 500_000) == []
    assert pending.asked_at_ms == 400_000
    assert service.cancel_question() is True
    assert manager.database.question(first_id) is None
    assert model.contexts == []
    assert service.cancel_question() is False


def test_question_voice_release_transcribes_once_and_returns_editable_text(
    monkeypatch, tmp_path: Path
) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session_id = manager.database.create_session("test", "2026-01-01T00:00:00+00:00")
    manager.session_id = session_id
    manager.clock = SimpleNamespace(now_ms=lambda: 5_000)
    recording_path = tmp_path / "question.flac"
    voice_recorders = []

    class FakeVoiceRecorder:
        def __init__(self, *_args, **_kwargs) -> None:
            self.started_paths: list[Path] = []
            self.stop_calls = 0
            self.cancel_calls = 0
            voice_recorders.append(self)

        def start(self, path: Path) -> None:
            self.started_paths.append(path)

        def stop(self) -> Path:
            self.stop_calls += 1
            return recording_path

        def cancel(self) -> None:
            self.cancel_calls += 1

    class FakeQuestionTranscriber:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def transcribe(self, path: Path) -> str:
            assert path == recording_path
            return "可编辑的语音问题"

    monkeypatch.setattr("jingzhi.session.QuestionVoiceRecorder", FakeVoiceRecorder)
    monkeypatch.setattr("jingzhi.session.WhisperQuestionTranscriber", FakeQuestionTranscriber)
    service = JingzhiApplicationService(manager.database, recorder=manager)

    service.begin_question()
    service.start_question_voice()
    assert service.finish_question_voice() == "可编辑的语音问题"
    with pytest.raises(RuntimeError, match="No question voice recording is active"):
        service.finish_question_voice()

    service.start_question_voice()
    pending_question_id = manager.pending_question_id
    assert pending_question_id is not None
    assert manager.stop() is None
    assert manager.database.question(pending_question_id) is None
    assert voice_recorders[-1].cancel_calls == 1


def test_failed_request_remains_the_current_reanswer_target(monkeypatch, tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session_id = manager.database.create_session("test", "2026-01-01T00:00:00+00:00")
    manager.session_id = session_id
    manager.clock = SimpleNamespace(now_ms=lambda: 1_000)

    class FailingModel:
        def answer(self, question, context):
            raise RuntimeError("upstream failed")

    service = answer_service(manager.database, FailingModel())
    monkeypatch.setattr(manager, "_question_service", lambda: service)

    with pytest.raises(RuntimeError, match="upstream failed"):
        manager.answer("失败问题")

    assert manager.last_question_id == manager.database.latest_question_id(session_id)


def test_reanswer_persisted_question_after_application_restart(monkeypatch, tmp_path: Path) -> None:
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = database.create_session("test", "2026-01-01T00:00:00+00:00")
    first_service = answer_service(database, RecordingAnswerModel())
    first = first_service.ask(session_id, 1_000, "持久化问题")

    restarted = SessionManager(Settings(data_dir=tmp_path))
    second_service = answer_service(restarted.database, RecordingAnswerModel())
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

    service = answer_service(database, FailingModel())

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
