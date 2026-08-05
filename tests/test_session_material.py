from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jingzhi.application import JingzhiApplicationService
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.llm import MaterialModelResult, OpenAIContextModel
from jingzhi.material_settings import MaterialGenerationMode, MaterialGenerationSettingsStore
from jingzhi.model_roles import ModelConnection, ModelRole, ReasoningLevel, RoleName
from jingzhi.model_routing import ModelRouter
from jingzhi.provider_settings import SavedProviderSettings
from jingzhi.session import SessionManager


class RecordingMaterialModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.contexts = []

    def generate_material(self, context, template_id=None):
        self.contexts.append((context, template_id))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def material_settings(model: str = "analysis-model") -> SavedProviderSettings:
    return SavedProviderSettings(
        (ModelConnection("primary", "主连接", api_key="secret"),),
        (
            ModelRole(
                RoleName.DEEP_ANALYSIS,
                "primary",
                model,
                ReasoningLevel.DEEP,
            ),
        ),
    )


def material_manager(tmp_path: Path, model: RecordingMaterialModel) -> tuple[SessionManager, str]:
    settings = Settings(data_dir=tmp_path, provider_settings=material_settings())
    manager = SessionManager(settings)
    session_id = manager.database.create_session("材料会话", "2026-08-05T00:00:00+00:00")
    manager.session_id = session_id
    router = ModelRouter(
        manager.database,
        settings.provider_settings,
        adapter_factory=lambda _connection, _model, _reasoning: model,
    )
    manager._model_router = lambda: router
    chunk_id = manager.database.add_audio_chunk(
        session_id, "system", 0, 2_000, tmp_path / "audio.flac"
    )
    manager.database.add_transcript(
        session_id, chunk_id, "system", 500, 1_500, "这是可核验的会话证据。", "zh", 0.9
    )
    return manager, session_id


def test_material_generation_persists_free_markdown_provenance_and_evidence(tmp_path: Path) -> None:
    model = RecordingMaterialModel(
        [MaterialModelResult("# 自由材料\n\n公式 $x^2$。", "request-1", "resolved-model")]
    )
    manager, session_id = material_manager(tmp_path, model)

    material = manager.generate_material(session_id, template_id="meeting-v1")

    assert material.version_number == 1
    assert material.kind == "generated"
    assert material.content == "# 自由材料\n\n公式 $x^2$。"
    assert material.template_id == "meeting-v1"
    assert material.model == "resolved-model"
    assert material.request_status == "succeeded"
    assert material.request_id == "request-1"
    context, template_id = model.contexts[0]
    assert template_id == "meeting-v1"
    assert "这是可核验的会话证据。" in context.transcript

    evidence = manager.database.material_evidence(material.id)
    assert len(evidence) == 1
    assert evidence[0].stable_id.startswith("transcript-version:")
    assert evidence[0].content_text == "这是可核验的会话证据。"
    invocation = manager.database.model_invocations(session_id)[0]
    assert invocation.role == "deep_analysis"
    assert invocation.status == "succeeded"
    assert invocation.evidence_ids == (evidence[0].stable_id,)


def test_material_edit_and_regeneration_create_versions_without_overwriting_original(
    tmp_path: Path,
) -> None:
    model = RecordingMaterialModel(
        [
            MaterialModelResult("# 原始材料", "request-1", "analysis-model"),
            MaterialModelResult("# 再生成材料", "request-2", "analysis-model"),
        ]
    )
    manager, session_id = material_manager(tmp_path, model)

    original = manager.generate_material(session_id)
    edited = manager.edit_material(original.id, "# 用户编辑材料\n\n独立附注")
    regenerated = manager.generate_material(session_id)

    assert original.version_number == 1
    assert original.content == "# 原始材料"
    assert edited.version_number == 2
    assert edited.kind == "user_edit"
    assert edited.content == "# 用户编辑材料\n\n独立附注"
    assert regenerated.version_number == 3
    assert regenerated.kind == "generated"
    assert [item.version_number for item in manager.material_versions(session_id)] == [1, 2, 3]
    assert [item.content for item in manager.material_versions(session_id)] == [
        "# 原始材料",
        "# 用户编辑材料\n\n独立附注",
        "# 再生成材料",
    ]
    assert [item.stable_id for item in manager.database.material_evidence(edited.id)] == [
        item.stable_id for item in manager.database.material_evidence(original.id)
    ]


def test_failed_material_generation_can_retry_without_partial_version(tmp_path: Path) -> None:
    model = RecordingMaterialModel(
        [RuntimeError("analysis unavailable"), MaterialModelResult("# 重试成功", "request-2")]
    )
    manager, session_id = material_manager(tmp_path, model)

    with pytest.raises(RuntimeError, match="analysis unavailable"):
        manager.generate_material(session_id)

    assert manager.material_versions(session_id) == []
    assert manager.database.model_invocations(session_id)[0].status == "failed"

    material = manager.generate_material(session_id)

    assert material.version_number == 1
    assert material.content == "# 重试成功"
    assert len(manager.database.model_invocations(session_id)) == 2


def test_material_generation_preference_is_persisted(tmp_path: Path) -> None:
    store = MaterialGenerationSettingsStore(tmp_path)

    assert store.load() is None
    store.save(MaterialGenerationMode.ALWAYS)

    assert store.load() == MaterialGenerationMode.ALWAYS


def test_question_original_is_unchanged_when_user_adds_independent_note(tmp_path: Path) -> None:
    database = Database(tmp_path / "notes.sqlite3")
    session_id = database.create_session("问答会话", "2026-08-05T00:00:00+00:00")
    question_id = database.create_question(session_id, 1_000, "原始问题", 0, 1_000)
    recorder = SimpleNamespace(is_recording=False)
    service = JingzhiApplicationService(database, recorder=recorder)

    note = service.add_question_note(session_id, question_id, "用户补充的独立附注")

    assert note.content == "用户补充的独立附注"
    assert database.question(question_id).question == "原始问题"
    assert [item.content for item in service.question_notes(session_id, question_id)] == [
        "用户补充的独立附注"
    ]


def test_provider_material_call_returns_markdown_without_json_parsing(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text="# 标题\n\n$\\epsilon$", id="material-1", model="analysis"
            )

    model = OpenAIContextModel("analysis", api_key="secret")
    monkeypatch.setattr(model, "_client", lambda: SimpleNamespace(responses=FakeResponses()))

    from jingzhi.context import QuestionContext, TranscriptEvidence

    context = QuestionContext(
        0,
        2_000,
        (TranscriptEvidence("transcript-version:1", 1, 1, "system", 500, 1_500, "会话证据"),),
        (),
    )
    result = model.generate_material(context, template_id="notes")

    assert result == MaterialModelResult("# 标题\n\n$\\epsilon$", "material-1", "analysis")
    prompt = calls[0]["input"]
    assert "自由 Markdown" in prompt
    assert "严格 JSON" not in prompt
    assert "transcript-version:1" in prompt
