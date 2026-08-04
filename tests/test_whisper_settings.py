from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from jingzhi.whisper_settings import (
    PROFILE_PRESETS,
    DownloadState,
    WhisperBenchmark,
    WhisperCapabilities,
    WhisperModelDownloader,
    WhisperProfile,
    WhisperSettings,
    WhisperSettingsStore,
    detect_whisper_capabilities,
    resolve_whisper_settings,
)


def test_product_profiles_map_to_distinct_whisper_configurations() -> None:
    assert PROFILE_PRESETS[WhisperProfile.LIGHTWEIGHT].model == "base"
    assert PROFILE_PRESETS[WhisperProfile.BALANCED].model == "small"
    assert PROFILE_PRESETS[WhisperProfile.ACCURATE].model == "medium"
    assert PROFILE_PRESETS[WhisperProfile.LIGHTWEIGHT].hardware_impact
    assert PROFILE_PRESETS[WhisperProfile.BALANCED].hardware_impact
    assert PROFILE_PRESETS[WhisperProfile.ACCURATE].hardware_impact


def test_advanced_settings_survive_restart(tmp_path: Path) -> None:
    store = WhisperSettingsStore(tmp_path)
    settings = WhisperSettings(
        profile=WhisperProfile.ACCURATE,
        model="large-v3-turbo",
        device="cuda",
        compute_type="float16",
        language="zh",
        vad_enabled=True,
        vad_min_silence_ms=650,
        first_run_completed=True,
    )

    store.save(settings)

    assert store.load() == settings
    public = json.loads((tmp_path / "whisper.json").read_text(encoding="utf-8"))
    assert public["version"] == 1
    assert public["profile"] == "accurate"


def test_unsupported_device_and_precision_get_actionable_fallback() -> None:
    capabilities = WhisperCapabilities(
        devices=("cpu",),
        compute_types={"cpu": ("int8", "float32")},
        gpu_name=None,
    )
    requested = replace(
        PROFILE_PRESETS[WhisperProfile.ACCURATE].settings,
        device="cuda",
        compute_type="float16",
    )

    resolved = resolve_whisper_settings(requested, capabilities)

    assert resolved.settings.device == "cpu"
    assert resolved.settings.compute_type == "int8"
    assert "CUDA" in resolved.fallback_advice
    assert "CPU" in resolved.fallback_advice
    assert "int8" in resolved.fallback_advice


def test_auto_device_uses_available_gpu_with_supported_precision() -> None:
    capabilities = WhisperCapabilities(
        devices=("cpu", "cuda"),
        compute_types={"cpu": ("int8", "float32"), "cuda": ("float16", "int8_float16")},
        gpu_name="Test GPU",
    )
    requested = PROFILE_PRESETS[WhisperProfile.BALANCED].settings

    resolved = resolve_whisper_settings(requested, capabilities)

    assert resolved.settings.device == "cuda"
    assert resolved.settings.compute_type == "float16"
    assert resolved.fallback_advice == ""


def test_fake_model_benchmark_reports_text_latency_realtime_factor_and_resources(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"sample")
    calls: list[dict[str, object]] = []

    class FakeModel:
        def transcribe(self, path: str, **kwargs):
            calls.append({"path": path, **kwargs})
            return [SimpleNamespace(text=" 境织样本 ")], SimpleNamespace(language="zh")

    benchmark = WhisperBenchmark(
        model_factory=lambda _settings: FakeModel(),
        duration_reader=lambda _path: 2.0,
        clock=iter((10.0, 10.5)).__next__,
        resource_reader=lambda: {"process_cpu_seconds": 0.25, "peak_memory_mb": 96.0},
    )

    result = benchmark.run(PROFILE_PRESETS[WhisperProfile.LIGHTWEIGHT].settings, sample)

    assert result.text == "境织样本"
    assert result.elapsed_seconds == 0.5
    assert result.realtime_factor == 0.25
    assert result.language == "zh"
    assert result.resources == {"process_cpu_seconds": 0.25, "peak_memory_mb": 96.0}
    assert calls == [
        {
            "path": str(sample),
            "beam_size": 1,
            "language": None,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 400},
        }
    ]


def test_fake_model_receives_all_three_profile_mappings(tmp_path: Path) -> None:
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"sample")
    received = []

    class FakeModel:
        def transcribe(self, _path: str, **_kwargs):
            return [SimpleNamespace(text="ok")], SimpleNamespace(language="en")

    for profile in WhisperProfile:
        benchmark = WhisperBenchmark(
            model_factory=lambda settings: (received.append(settings), FakeModel())[1],
            duration_reader=lambda _path: 1.0,
            clock=iter((0.0, 0.1)).__next__,
            resource_reader=dict,
        )
        benchmark.run(PROFILE_PRESETS[profile].settings, sample)

    assert received == [PROFILE_PRESETS[profile].settings for profile in WhisperProfile]


def test_download_state_exposes_disk_progress_cancel_failure_and_retry() -> None:
    state = DownloadState(model="small", required_bytes=100)
    state.mark_progress(40)
    assert state.progress_percent == 40
    assert state.status == "downloading"

    state.cancel()
    assert state.status == "cancelled"
    assert state.can_retry

    state.retry()
    state.fail("network unavailable")
    assert state.status == "failed"
    assert state.error == "network unavailable"
    assert state.can_retry


def test_model_download_can_cancel_and_retry_between_repository_files() -> None:
    cancel_event = threading.Event()
    downloaded: list[str] = []
    states: list[str] = []
    should_cancel = True

    def download(_model: str, filename: str) -> None:
        nonlocal should_cancel
        downloaded.append(filename)
        if filename == "model.bin" and should_cancel:
            should_cancel = False
            cancel_event.set()

    downloader = WhisperModelDownloader(
        repository_files=lambda _model: (("model.bin", 60), ("config.json", 40)),
        download_file=download,
    )
    cancelled = downloader.prepare(
        "small",
        on_progress=lambda state: states.append(state.status),
        cancel_event=cancel_event,
    )

    assert cancelled.status == "cancelled"
    assert downloaded == ["model.bin"]
    assert states[-1] == "cancelled"

    cancel_event.clear()
    completed = downloader.prepare(
        "small", on_progress=lambda _state: None, cancel_event=cancel_event
    )
    assert completed.status == "complete"
    assert completed.progress_percent == 100


def test_model_metadata_failure_is_retryable() -> None:
    downloader = WhisperModelDownloader(
        repository_files=lambda _model: (_ for _ in ()).throw(RuntimeError("offline"))
    )

    failed = downloader.prepare(
        "small", on_progress=lambda _state: None, cancel_event=threading.Event()
    )

    assert failed.status == "failed"
    assert failed.error == "offline"
    assert failed.can_retry


def test_windows_detected_capabilities_resolve_cpu_and_available_gpu(monkeypatch) -> None:
    class FakeCTranslate2:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 1

        @staticmethod
        def get_supported_compute_types(device: str) -> set[str]:
            return {"int8", "float32"} if device == "cpu" else {"float16", "float32"}

    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", FakeCTranslate2)
    capabilities = detect_whisper_capabilities()

    assert capabilities.devices == ("cpu", "cuda")
    assert capabilities.compute_types["cpu"] == ("int8", "float32")
    assert capabilities.compute_types["cuda"] == ("float16", "float32")
    assert (
        resolve_whisper_settings(
            replace(PROFILE_PRESETS[WhisperProfile.LIGHTWEIGHT].settings, device="cpu"),
            capabilities,
        ).settings.compute_type
        == "int8"
    )
    assert (
        resolve_whisper_settings(
            PROFILE_PRESETS[WhisperProfile.BALANCED].settings, capabilities
        ).settings.device
        == "cuda"
    )
