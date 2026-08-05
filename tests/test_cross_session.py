from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jingzhi.application import JingzhiApplicationService
from jingzhi.config import Settings
from jingzhi.cross_session import MAX_SYNTHESIS_CHARACTERS, CrossSessionSynthesisError
from jingzhi.database import CrossSessionEvidenceRecord, Database
from jingzhi.llm import SynthesisModelResult
from jingzhi.model_roles import ModelConnection, ModelRole, ReasoningLevel, RoleName
from jingzhi.model_routing import ModelRouter
from jingzhi.provider_settings import SavedProviderSettings
from jingzhi.session import SessionManager


class RecordingSynthesisModel:
    def __init__(self, result: object = SynthesisModelResult("综合结果", "synthesis-1")) -> None:
        self.result = result
        self.contexts: list[tuple[str, object]] = []

    def synthesize(self, question: str, context: object) -> SynthesisModelResult:
        self.contexts.append((question, context))
        if isinstance(self.result, Exception):
            raise self.result
        assert isinstance(self.result, SynthesisModelResult)
        return self.result


def synthesis_settings() -> SavedProviderSettings:
    return SavedProviderSettings(
        connections=(ModelConnection("primary", "主连接", api_key="secret"),),
        roles=(
            ModelRole(
                RoleName.DEEP_ANALYSIS,
                "primary",
                "analysis-model",
                ReasoningLevel.DEEP,
            ),
        ),
    )


def synthesis_manager(
    tmp_path: Path, model: RecordingSynthesisModel
) -> tuple[SessionManager, str, str]:
    settings = Settings(data_dir=tmp_path, provider_settings=synthesis_settings())
    manager = SessionManager(settings)
    first_id = manager.database.create_session("第一会话", "2026-08-01T00:00:00+00:00")
    second_id = manager.database.create_session("第二会话", "2026-08-02T00:00:00+00:00")
    router = ModelRouter(
        manager.database,
        settings.provider_settings,
        adapter_factory=lambda _connection, _model, _reasoning: model,
    )
    manager._model_router = lambda: router
    return manager, first_id, second_id


def add_transcript(
    database: Database,
    session_id: str,
    path: Path,
    text: str,
    *,
    start_ms: int = 1_000,
    end_ms: int = 2_000,
) -> tuple[int, int]:
    chunk_id = database.add_audio_chunk(session_id, "system", start_ms, end_ms, path)
    segment_id = database.add_transcript(
        session_id, chunk_id, "system", start_ms, end_ms, text, "zh", 0.9
    )
    return segment_id, database.transcript_versions(segment_id)[0].id


def test_schema_migrates_cross_session_evidence_to_an_independent_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "migration.sqlite3"
    database = Database(path)
    session_id = database.create_session("迁移会话", "2026-08-01T00:00:00+00:00")
    synthesis = database.record_cross_session_synthesis(
        question="迁移问题",
        answer="迁移答案",
        model="analysis-model",
        connection_json="{}",
        model_invocation_id=None,
        request_status="succeeded",
        request_id=None,
        error=None,
        evidence_state="exact",
        evidence=(
            CrossSessionEvidenceRecord(
                "answer-version:1",
                session_id,
                "迁移会话",
                "answer",
                "问答",
                0,
                1,
                "迁移证据",
                None,
            ),
        ),
    )
    connection = database.connect()
    connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    connection.execute(
        """CREATE TABLE cross_session_synthesis_evidence_old AS
           SELECT synthesis_id, ordinal, stable_id, session_id, kind, source,
                  start_ms, end_ms, transcript_version_id, frame_id,
                  answer_version_id, material_version_id, content_text, resource_path
           FROM cross_session_synthesis_evidence"""
    )
    connection.execute("DROP TABLE cross_session_synthesis_evidence")
    connection.execute(
        "ALTER TABLE cross_session_synthesis_evidence_old RENAME TO cross_session_synthesis_evidence"
    )
    connection.execute("DELETE FROM schema_migrations WHERE version > 13")
    connection.execute("PRAGMA user_version = 13")
    connection.commit()
    connection.close()

    migrated = Database(path)

    evidence = migrated.cross_session_synthesis_evidence(synthesis.id)
    assert evidence[0].session_title == "已删除会话"
    assert evidence[0].content_text == "迁移证据"


def test_search_covers_transcripts_answers_and_materials_but_excludes_trashed_sessions(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "search.sqlite3")
    active_id = database.create_session("主动会话", "2026-08-01T00:00:00+00:00")
    trashed_id = database.create_session("回收会话", "2026-08-02T00:00:00+00:00")
    _segment_id, _version_id = add_transcript(
        database, active_id, tmp_path / "active.flac", "跨会话检索的字幕证据"
    )
    add_transcript(database, trashed_id, tmp_path / "trash.flac", "跨会话检索的回收内容")

    question_id = database.create_question(active_id, 3_000, "检索问题", 0, 3_000)
    database.record_answer_version(
        question_id,
        model="answer-model",
        connection_json=None,
        request_status="succeeded",
        request_id="answer-request",
        answer="跨会话检索的问答答案",
        error=None,
        evidence_state="exact",
        evidence=[],
    )
    database.record_material_version(
        active_id,
        kind="generated",
        content="# 跨会话检索的材料",
        template_id=None,
        model="analysis-model",
        connection_json=None,
        model_invocation_id=None,
        request_status="succeeded",
        request_id="material-request",
        error=None,
        evidence_state="exact",
        evidence=[],
    )
    database.finish_session(trashed_id, "2026-08-03T00:00:00+00:00", "complete")
    database.move_session_to_trash(
        trashed_id,
        "2026-08-03T00:00:00+00:00",
        "2026-08-10T00:00:00+00:00",
    )

    results = database.cross_session_search("跨会话检索", limit=20)

    assert {item.kind for item in results} == {"transcript", "answer", "material"}
    assert {item.session_id for item in results} == {active_id}
    assert all(item.session_title == "主动会话" for item in results)
    assert all(item.start_ms <= item.end_ms for item in results)


def test_search_index_follows_effective_transcript_version(tmp_path: Path) -> None:
    database = Database(tmp_path / "versions.sqlite3")
    session_id = database.create_session("版本会话", "2026-08-01T00:00:00+00:00")
    segment_id, original_id = add_transcript(
        database, session_id, tmp_path / "audio.flac", "原始检索词"
    )

    correction_id = database.add_transcript_version(
        segment_id, "correction", "校订检索词", model="correction-model"
    )

    assert correction_id is not None
    assert [item.stable_id for item in database.cross_session_search("校订检索词")] == [
        f"transcript-version:{correction_id}"
    ]
    assert database.cross_session_search("原始检索词") == []

    database.undo_transcript_correction(segment_id)

    assert [item.stable_id for item in database.cross_session_search("原始检索词")] == [
        f"transcript-version:{original_id}"
    ]
    assert database.cross_session_search("校订检索词") == []


def test_search_result_expands_to_authorized_answer_material_transcript_and_frame_evidence(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "evidence.sqlite3")
    session_id = database.create_session("证据会话", "2026-08-01T00:00:00+00:00")
    media_dir = tmp_path / "sessions" / session_id
    media_dir.mkdir(parents=True)
    frame_path = media_dir / "frame.webp"
    frame_path.write_bytes(b"frame")
    frame_id = database.add_frame(
        session_id, 1_500, frame_path, "0" * 64, (100, 100), source_id="display:2"
    )
    _segment_id, version_id = add_transcript(
        database, session_id, tmp_path / "audio.flac", "关联证据检索词"
    )
    question_id = database.create_question(session_id, 3_000, "关联问题", 0, 3_000)
    answer = database.record_answer_version(
        question_id,
        model="answer-model",
        connection_json=None,
        request_status="succeeded",
        request_id=None,
        answer="关联证据检索词的回答",
        error=None,
        evidence=[
            {
                "stable_id": f"transcript-version:{version_id}",
                "kind": "transcript",
                "source": "system",
                "start_ms": 1_000,
                "end_ms": 2_000,
                "transcript_version_id": version_id,
                "content_text": "关联证据检索词",
            },
            {
                "stable_id": f"frame:{frame_id}",
                "kind": "frame",
                "source": "display:2",
                "start_ms": 1_500,
                "end_ms": 1_500,
                "frame_id": frame_id,
                "resource_path": str(frame_path),
            },
        ],
        evidence_state="exact",
    )
    material = database.record_material_version(
        session_id,
        kind="generated",
        content="# 关联材料检索词",
        template_id=None,
        model="analysis-model",
        connection_json=None,
        model_invocation_id=None,
        request_status="succeeded",
        request_id=None,
        error=None,
        evidence_state="exact",
        evidence=[
            {
                "stable_id": f"transcript-version:{version_id}",
                "kind": "transcript",
                "source": "system",
                "start_ms": 1_000,
                "end_ms": 2_000,
                "transcript_version_id": version_id,
                "content_text": "关联证据检索词",
            }
        ],
    )

    answer_result = next(
        item for item in database.cross_session_search("关联证据检索词") if item.kind == "answer"
    )
    candidates = database.cross_session_evidence_candidates((answer_result.stable_id,))

    assert [item.stable_id for item in candidates] == [
        f"answer-version:{answer.id}",
        f"transcript-version:{version_id}",
        f"frame:{frame_id}",
    ]
    assert candidates[0].answer_version_id == answer.id
    assert candidates[1].content_text == "关联证据检索词"
    assert candidates[2].resource_path == frame_path

    material_result = next(
        item for item in database.cross_session_search("关联材料检索词") if item.kind == "material"
    )
    material_candidates = database.cross_session_evidence_candidates((material_result.stable_id,))
    assert material_candidates[0].material_version_id == material.id
    assert (material_candidates[0].start_ms, material_candidates[0].end_ms) == (1_000, 2_000)
    assert material_candidates[1].transcript_version_id == version_id


def test_permanent_delete_marks_synthesis_evidence_unavailable_before_cascade(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "delete-synthesis.sqlite3")
    session_id = database.create_session("待删除会话", "2026-08-01T00:00:00+00:00")
    _segment_id, version_id = add_transcript(
        database, session_id, tmp_path / "delete.flac", "待删除的综合证据"
    )
    database.finish_session(session_id, "2026-08-01T00:01:00+00:00", "complete")
    database.move_session_to_trash(
        session_id,
        "2026-08-02T00:00:00+00:00",
        "2026-08-09T00:00:00+00:00",
    )
    synthesis = database.record_cross_session_synthesis(
        question="删除后仍可解释吗？",
        answer="结果",
        model="analysis-model",
        connection_json="{}",
        model_invocation_id=None,
        request_status="succeeded",
        request_id=None,
        error=None,
        evidence_state="exact",
        evidence=(
            CrossSessionEvidenceRecord(
                f"transcript-version:{version_id}",
                session_id,
                "待删除会话",
                "transcript",
                "system",
                1_000,
                2_000,
                "待删除的综合证据",
                None,
                transcript_version_id=version_id,
            ),
        ),
    )

    assert database.permanently_delete_session(session_id) is True
    persisted = database.cross_session_synthesis(synthesis.id)
    assert persisted is not None
    assert persisted.evidence_state == "unavailable"
    evidence = database.cross_session_synthesis_evidence(synthesis.id)
    assert len(evidence) == 1
    assert evidence[0].stable_id == f"transcript-version:{version_id}"
    assert evidence[0].content_text == "待删除的综合证据"
    assert evidence[0].session_title == "待删除会话"


def test_cross_session_synthesis_sends_only_selected_evidence_and_persists_audit(
    tmp_path: Path,
) -> None:
    model = RecordingSynthesisModel()
    manager, first_id, second_id = synthesis_manager(tmp_path, model)
    _segment_id, first_version_id = add_transcript(
        manager.database, first_id, tmp_path / "first.flac", "第一份授权证据"
    )
    _segment_id, _second_version_id = add_transcript(
        manager.database, second_id, tmp_path / "second.flac", "第二份未授权证据"
    )

    selected = (f"transcript-version:{first_version_id}",)
    preview = manager.cross_session_synthesis_preview("比较两个会话", selected)

    assert preview.can_synthesize is True
    assert preview.evidence_count == 1
    assert preview.character_count == len("第一份授权证据")

    result = manager.synthesize_cross_session("比较两个会话", selected)

    assert result.request_status == "succeeded"
    assert result.answer == "综合结果"
    question, context = model.contexts[0]
    assert question == "比较两个会话"
    assert tuple(item.stable_id for item in context.evidence) == selected
    assert "第一份授权证据" in context.prompt_text
    assert "第二份未授权证据" not in context.prompt_text
    persisted = manager.database.cross_session_synthesis_evidence(result.id)
    assert [item.stable_id for item in persisted] == list(selected)
    invocation = manager.database.model_invocations(None)[-1]
    assert invocation.session_id is None
    assert invocation.role == "deep_analysis"
    assert invocation.evidence_ids == selected
    assert "api_key" not in (result.connection_json or "")


def test_cross_session_synthesis_rejects_empty_or_oversized_authorization_before_model_call(
    tmp_path: Path,
) -> None:
    model = RecordingSynthesisModel()
    manager, first_id, _second_id = synthesis_manager(tmp_path, model)
    _segment_id, version_id = add_transcript(
        manager.database,
        first_id,
        tmp_path / "first.flac",
        "x" * (MAX_SYNTHESIS_CHARACTERS + 1),
    )
    stable_id = f"transcript-version:{version_id}"

    empty = manager.cross_session_synthesis_preview("问题", ())
    oversized = manager.cross_session_synthesis_preview("问题", (stable_id,))

    assert empty.can_synthesize is False
    assert "选择" in (empty.reason or "")
    assert oversized.can_synthesize is False
    assert "过长" in (oversized.reason or "")
    with pytest.raises(RuntimeError, match="过长"):
        manager.synthesize_cross_session("问题", (stable_id,))
    assert model.contexts == []
    assert manager.database.model_invocations(None) == ()


def test_failed_cross_session_synthesis_enters_retry_queue(tmp_path: Path) -> None:
    model = RecordingSynthesisModel(RuntimeError("上下文窗口超限"))
    manager, first_id, _second_id = synthesis_manager(tmp_path, model)
    _segment_id, version_id = add_transcript(
        manager.database, first_id, tmp_path / "retry.flac", "可重试证据"
    )
    stable_id = f"transcript-version:{version_id}"

    with pytest.raises(CrossSessionSynthesisError) as error:
        manager.synthesize_cross_session("请重试", (stable_id,))

    failed_id = error.value.synthesis_id
    failed = manager.database.cross_session_synthesis(failed_id)
    assert failed is not None and failed.request_status == "failed"
    assert [
        item.stable_id for item in manager.database.cross_session_synthesis_evidence(failed_id)
    ] == [stable_id]

    model.result = SynthesisModelResult("重试成功", "retry-request")
    retried = manager.retry_cross_session_synthesis(failed_id)

    assert retried.request_status == "succeeded"
    assert retried.answer == "重试成功"
    assert retried.retry_of_id == failed_id
    assert manager.database.cross_session_syntheses(request_status="failed") == ()


def test_failed_cross_session_synthesis_is_not_retryable_after_source_is_trashed(
    tmp_path: Path,
) -> None:
    model = RecordingSynthesisModel(RuntimeError("模型不可用"))
    manager, first_id, _second_id = synthesis_manager(tmp_path, model)
    _segment_id, version_id = add_transcript(
        manager.database, first_id, tmp_path / "trashed-retry.flac", "回收后不可重试"
    )
    with pytest.raises(CrossSessionSynthesisError):
        manager.synthesize_cross_session("请不要重试", (f"transcript-version:{version_id}",))

    manager.database.finish_session(first_id, "2026-08-02T00:00:00+00:00", "complete")
    manager.database.move_session_to_trash(
        first_id,
        "2026-08-03T00:00:00+00:00",
        "2026-08-10T00:00:00+00:00",
    )

    assert manager.database.cross_session_syntheses(request_status="failed") == ()


def test_failed_cross_session_synthesis_persists_failure_and_keeps_evidence(
    tmp_path: Path,
) -> None:
    model = RecordingSynthesisModel(RuntimeError("上下文窗口超限"))
    manager, first_id, _second_id = synthesis_manager(tmp_path, model)
    _segment_id, version_id = add_transcript(
        manager.database, first_id, tmp_path / "first.flac", "失败也要保留授权证据"
    )
    stable_id = f"transcript-version:{version_id}"

    with pytest.raises(RuntimeError, match="上下文"):
        manager.synthesize_cross_session("问题", (stable_id,))

    synthesis = manager.database.cross_session_syntheses()[0]
    assert synthesis.request_status == "failed"
    assert synthesis.answer is None
    assert synthesis.error == "上下文窗口超限"
    assert [
        item.stable_id for item in manager.database.cross_session_synthesis_evidence(synthesis.id)
    ] == [stable_id]


def test_application_service_exposes_cross_session_search_and_synthesis(tmp_path: Path) -> None:
    database = Database(tmp_path / "application.sqlite3")
    session_id = database.create_session("应用边界", "2026-08-01T00:00:00+00:00")
    add_transcript(database, session_id, tmp_path / "audio.flac", "应用搜索词")
    service = JingzhiApplicationService(database, recorder=SimpleNamespace(is_recording=False))

    results = service.cross_session_search("应用搜索词")

    assert results[0].session_id == session_id
    assert (
        service.cross_session_evidence_candidates((results[0].stable_id,))[0].session_id
        == session_id
    )
