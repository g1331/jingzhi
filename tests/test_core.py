from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from jingzhi.capture.audio import _prepare_mono_audio
from jingzhi.capture.screen import average_hash
from jingzhi.context import ContextAssembler, QuestionContext
from jingzhi.database import Database
from jingzhi.llm import OpenAIContextModel, ProviderRequestError


def test_average_hash_distinguishes_opposite_images() -> None:
    black = Image.new("RGB", (32, 32), "black")
    split = Image.new("RGB", (32, 32), "black")
    for x in range(16):
        for y in range(32):
            split.putpixel((x, y), (255, 255, 255))
    assert (average_hash(black) ^ average_hash(split)).bit_count() > 0


def test_audio_is_mixed_to_mono_and_downsampled() -> None:
    import numpy as np

    stereo = np.ones((48_000, 2), dtype=np.float32)
    output = _prepare_mono_audio(stereo, 48_000, 16_000)
    assert output.shape == (16_000,)
    assert np.allclose(output, 1.0)

    audio_44k = np.ones((44_100, 1), dtype=np.float32)
    output_44k = _prepare_mono_audio(audio_44k, 44_100, 16_000)
    assert output_44k.shape == (16_000,)
    assert np.allclose(output_44k, 1.0)


def test_context_aligns_transcript_and_frames(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("test", "2026-01-01T00:00:00+00:00")
    frame = tmp_path / "frame.webp"
    frame.touch()
    database.add_frame(session_id, 9_500, frame, "0" * 64, (100, 100))
    chunk_id = database.add_audio_chunk(session_id, "system", 8_000, 12_000, tmp_path / "audio.wav")
    database.add_transcript(
        session_id,
        chunk_id,
        "system",
        9_000,
        10_000,
        "单纯形法的换入变量",
        "zh",
        -0.1,
    )

    context = ContextAssembler(database).around_question(session_id, 11_000, lookback_ms=5_000)

    assert "单纯形法" in context.transcript
    assert context.frame_paths == (frame,)


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
def test_provider_api_modes_route_without_network(monkeypatch, api_mode: str) -> None:
    calls: list[dict] = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="response answer")

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content="chat answer")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        responses=FakeResponses(),
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    model = OpenAIContextModel(
        "vision-model",
        api_key="secret",
        base_url="https://provider.example/v1/",
        api_mode=api_mode,
    )
    monkeypatch.setattr(model, "_client", lambda: client)
    context = QuestionContext(0, 1_000, "[0.1s] lecture", ())

    answer = model.answer("why?", context)

    assert answer in {"response answer", "chat answer"}
    assert model.base_url == "https://provider.example/v1"
    assert calls[0]["model"] == "vision-model"


def test_provider_html_response_becomes_concise_error(monkeypatch) -> None:
    class HtmlResponses:
        def create(self, **_kwargs):
            raise RuntimeError("<!DOCTYPE html><html>" + "wide-content" * 1_000 + "</html>")

    client = SimpleNamespace(responses=HtmlResponses())
    model = OpenAIContextModel(
        "vision-model",
        api_key="secret",
        base_url="https://provider.example/api/proxy",
        api_mode="responses",
    )
    monkeypatch.setattr(model, "_client", lambda: client)

    with pytest.raises(ProviderRequestError) as captured:
        model.test_connection()

    message = str(captured.value)
    assert "HTML 页面" in message
    assert "wide-content" not in message
    assert len(message) < 240
