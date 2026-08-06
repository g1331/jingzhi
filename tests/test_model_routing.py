import json
from types import SimpleNamespace

import keyring
import pytest

from jingzhi.config import Settings
from jingzhi.context import SynthesisContext, SynthesisEvidence
from jingzhi.database import Database
from jingzhi.llm import OpenAIContextModel
from jingzhi.model_roles import (
    ModelConnection,
    ModelFallback,
    ModelRole,
    ReasoningLevel,
    RoleName,
)
from jingzhi.model_routing import (
    InvocationEvidence,
    ModelRouter,
    RoutedTranscriptCorrectionModel,
)
from jingzhi.provider_settings import ProviderSettingsStore, SavedProviderSettings
from jingzhi.session import SessionManager
from jingzhi.transcript_correction import TranscriptCorrectionProcessor


def test_model_connections_and_roles_round_trip_without_public_credentials(
    tmp_path, monkeypatch
) -> None:
    secrets: dict[str, str] = {}
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda service, username, password: secrets.__setitem__(f"{service}:{username}", password),
    )
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda service, username: secrets.get(f"{service}:{username}"),
    )
    monkeypatch.setattr(
        keyring,
        "delete_password",
        lambda service, username: secrets.pop(f"{service}:{username}", None),
    )
    settings = SavedProviderSettings(
        connections=(
            ModelConnection(
                id="primary",
                name="主连接",
                base_url="https://primary.example/v1",
                api_key="primary-secret",
                api_mode="responses",
            ),
            ModelConnection(
                id="backup",
                name="后备连接",
                base_url="https://backup.example/v1",
                api_key="backup-secret",
                api_mode="chat_completions",
            ),
        ),
        roles=(
            ModelRole(
                name=RoleName.INSTANT_ANSWER,
                connection_id="primary",
                model="answer-main",
                reasoning=ReasoningLevel.BALANCED,
                fallbacks=(
                    ModelFallback("primary", "answer-small"),
                    ModelFallback("backup", "answer-backup", cross_connection_authorized=True),
                ),
            ),
        ),
    )

    store = ProviderSettingsStore(tmp_path)
    store.save(settings)
    loaded = store.load()

    public_text = (tmp_path / "provider.json").read_text(encoding="utf-8")
    public = json.loads(public_text)
    assert public["version"] == 2
    assert "primary-secret" not in public_text
    assert "backup-secret" not in public_text
    assert loaded == settings


def test_reasoning_effort_is_only_sent_to_responses_protocol(monkeypatch) -> None:
    calls: list[dict] = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="OK")

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    client = SimpleNamespace(
        responses=Responses(),
        chat=SimpleNamespace(completions=Completions()),
    )
    responses_model = OpenAIContextModel(
        "answer-model",
        api_key="secret",
        api_mode="responses",
        reasoning_effort="medium",
    )
    monkeypatch.setattr(responses_model, "_client", lambda: client)
    responses_model.test_connection()
    chat_model = OpenAIContextModel(
        "answer-model",
        api_key="secret",
        api_mode="chat_completions",
        reasoning_effort="medium",
    )
    monkeypatch.setattr(chat_model, "_client", lambda: client)
    chat_model.test_connection()

    assert calls[0]["reasoning"] == {"effort": "medium"}
    assert "reasoning" not in calls[1]


def test_synthesis_builds_provider_payload_from_only_the_authorized_context(monkeypatch) -> None:
    calls: list[dict] = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="综合结论", id="request-1", model="analysis-model")

    model = OpenAIContextModel(
        "analysis-model",
        api_key="secret",
        api_mode="responses",
        reasoning_effort="high",
    )
    monkeypatch.setattr(model, "_client", lambda: SimpleNamespace(responses=Responses()))
    context = SynthesisContext(
        (
            SynthesisEvidence(
                "answer-version:1",
                "session-1",
                "会话一",
                "answer",
                "问答",
                1_000,
                1_000,
                "已授权答案",
                None,
            ),
            SynthesisEvidence(
                "frame:2",
                "session-2",
                "会话二",
                "frame",
                "display:1",
                2_000,
                2_000,
                None,
                "data:image/png;base64,ZmFrZQ==",
            ),
        )
    )

    result = model.synthesize("比较已选证据", context)

    assert result.text == "综合结论"
    payload = calls[0]
    content = payload["input"][0]["content"]
    text = "\n".join(item["text"] for item in content if item["type"] == "input_text")
    assert "已授权答案" in text
    assert "未授权" not in text
    assert sum(item["type"] == "input_image" for item in content) == 1
    assert payload["reasoning"] == {"effort": "high"}


def test_router_executes_authorized_fallbacks_and_records_actual_source(tmp_path) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("路由测试", "2026-08-04T00:00:00+00:00")
    settings = SavedProviderSettings(
        connections=(
            ModelConnection("primary", "主连接", api_key="one"),
            ModelConnection("backup", "后备连接", api_key="two"),
        ),
        roles=(
            ModelRole(
                RoleName.INSTANT_ANSWER,
                "primary",
                "answer-main",
                ReasoningLevel.BALANCED,
                (
                    ModelFallback("primary", "answer-small"),
                    ModelFallback("backup", "answer-backup", True),
                ),
            ),
        ),
    )
    attempts: list[str] = []

    class Adapter:
        def __init__(self, model: str) -> None:
            self.model = model

    def factory(connection, model, reasoning_effort):
        assert reasoning_effort == "medium"
        return Adapter(model)

    def operation(adapter):
        attempts.append(adapter.model)
        if adapter.model != "answer-backup":
            raise RuntimeError(f"{adapter.model} unavailable")
        return "answer"

    result = ModelRouter(database, settings, adapter_factory=factory).invoke(
        RoleName.INSTANT_ANSWER,
        operation,
        session_id=session_id,
        evidence=(
            InvocationEvidence(
                stable_id="transcript:7:v2",
                kind="transcript",
                source="microphone",
                start_ms=1000,
                end_ms=2000,
                transcript_version_id=2,
            ),
        ),
    )

    assert result.value == "answer"
    assert attempts == ["answer-main", "answer-small", "answer-backup"]
    assert result.invocation.connection_id == "backup"
    assert result.invocation.model == "answer-backup"
    assert result.invocation.fallback_reason == "answer-small unavailable"
    records = database.model_invocations(session_id)
    assert [record.status for record in records] == ["failed", "failed", "succeeded"]
    assert records[-1].role == "instant_answer"
    assert records[-1].reasoning_level == "balanced"
    assert records[-1].evidence_ids == ("transcript:7:v2",)
    assert database.retryable_model_task_count() == 0


def test_router_does_not_use_unauthorized_cross_connection_fallback(tmp_path) -> None:
    database = Database(tmp_path / "test.sqlite3")
    settings = SavedProviderSettings(
        connections=(
            ModelConnection("primary", "主连接", api_key="one"),
            ModelConnection("backup", "后备连接", api_key="two"),
        ),
        roles=(
            ModelRole(
                RoleName.UTILITY,
                "primary",
                "utility-main",
                ReasoningLevel.FAST,
                (ModelFallback("backup", "utility-backup"),),
            ),
        ),
    )
    attempts: list[str] = []

    def factory(_connection, model, _reasoning_effort):
        attempts.append(model)
        return object()

    router = ModelRouter(database, settings, adapter_factory=factory)
    try:
        router.invoke(
            RoleName.UTILITY, lambda _adapter: (_ for _ in ()).throw(RuntimeError("down"))
        )
    except RuntimeError as exc:
        assert str(exc) == "down"
    else:
        raise AssertionError("Routing failure was not raised")

    assert attempts == ["utility-main"]


def test_transcript_correction_failure_keeps_original_transcript(tmp_path) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("校订降级", "2026-08-04T00:00:00+00:00")
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 2_000, tmp_path / "audio.wav")
    segment_id = database.add_transcript(
        session_id, chunk_id, "system", 500, 1_500, "原始字幕", "zh", 0.8
    )
    version_id = database.transcript_versions(segment_id)[0].id
    database.configure_transcript_correction(session_id, enabled=True, window_ms=15_000)
    settings = SavedProviderSettings(
        (ModelConnection("primary", "主连接", api_key="one"),),
        (
            ModelRole(
                RoleName.TRANSCRIPT_CORRECTION,
                "primary",
                "correction-main",
                ReasoningLevel.FAST,
            ),
        ),
    )

    class FailingAdapter:
        def correct(self, _request):
            raise RuntimeError("correction unavailable")

    router = ModelRouter(
        database,
        settings,
        adapter_factory=lambda _connection, _model, _reasoning: FailingAdapter(),
    )
    result = TranscriptCorrectionProcessor(database, RoutedTranscriptCorrectionModel(router)).run(
        session_id, window_start_ms=0
    )

    assert result.state == "failed"
    assert database.transcript_versions(segment_id)[0].text == "原始字幕"
    invocations = database.model_invocations(session_id)
    assert len(invocations) == 1
    assert invocations[0].role == "transcript_correction"
    assert invocations[0].status == "failed"
    assert invocations[0].evidence_ids == (f"transcript-version:{version_id}",)


def test_deep_analysis_failure_creates_no_partial_artifacts(tmp_path, monkeypatch) -> None:
    settings = SavedProviderSettings(
        (ModelConnection("primary", "主连接", api_key="one"),),
        (
            ModelRole(
                RoleName.DEEP_ANALYSIS,
                "primary",
                "analysis-main",
                ReasoningLevel.DEEP,
            ),
        ),
    )
    manager = SessionManager(Settings(data_dir=tmp_path, provider_settings=settings))
    session_id = manager.database.create_session("分析降级", "2026-08-04T00:00:00+00:00")
    manager.session_id = session_id
    chunk_id = manager.database.add_audio_chunk(
        session_id, "system", 0, 2_000, tmp_path / "audio.wav"
    )
    segment_id = manager.database.add_transcript(
        session_id, chunk_id, "system", 500, 1_500, "会话字幕", "zh", 0.8
    )
    version_id = manager.database.transcript_versions(segment_id)[0].id

    class FailingAnalysisAdapter:
        def summarize(self, _transcript):
            raise RuntimeError("analysis unavailable")

    router = ModelRouter(
        manager.database,
        settings,
        adapter_factory=lambda _connection, _model, _reasoning: FailingAnalysisAdapter(),
    )
    monkeypatch.setattr(manager, "_model_router", lambda: router)

    with pytest.raises(RuntimeError, match="analysis unavailable"):
        manager.summarize()

    with manager.database.connect() as connection:
        artifact_count = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    assert artifact_count == 0
    invocation = manager.database.model_invocations(session_id)[0]
    assert invocation.role == "deep_analysis"
    assert invocation.status == "failed"
    assert invocation.evidence_ids == (f"transcript-version:{version_id}",)
