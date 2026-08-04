from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from jingzhi.storage import canonical_whisper_repository_id


class WhisperProfile(StrEnum):
    LIGHTWEIGHT = "lightweight"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    profile: WhisperProfile = WhisperProfile.BALANCED
    model: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    language: str = "auto"
    vad_enabled: bool = True
    vad_min_silence_ms: int = 400
    first_run_completed: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Whisper model is required")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Whisper device must be auto, cpu, or cuda")
        if self.compute_type not in {"default", "int8", "float32", "float16", "int8_float16"}:
            raise ValueError("Unsupported Whisper compute type")
        if self.vad_min_silence_ms < 0:
            raise ValueError("VAD minimum silence must not be negative")


def whisper_transcribe_options(settings: WhisperSettings) -> dict[str, Any]:
    return {
        "beam_size": 1,
        "language": None if settings.language == "auto" else settings.language,
        "vad_filter": settings.vad_enabled,
        "vad_parameters": {"min_silence_duration_ms": settings.vad_min_silence_ms},
    }


@dataclass(frozen=True, slots=True)
class WhisperProfilePreset:
    label: str
    hardware_impact: str
    settings: WhisperSettings

    @property
    def model(self) -> str:
        return self.settings.model


PROFILE_PRESETS = {
    WhisperProfile.LIGHTWEIGHT: WhisperProfilePreset(
        "轻量",
        "内存和处理器占用最低，适合无独立显卡或希望减少录制期间资源占用的设备。",
        WhisperSettings(
            profile=WhisperProfile.LIGHTWEIGHT,
            model="base",
            device="cpu",
            compute_type="int8",
        ),
    ),
    WhisperProfile.BALANCED: WhisperProfilePreset(
        "均衡",
        "优先使用可用显卡，兼顾中文识别效果与延迟；仅有处理器时自动回退到 int8。",
        WhisperSettings(profile=WhisperProfile.BALANCED, model="small"),
    ),
    WhisperProfile.ACCURATE: WhisperProfilePreset(
        "准确",
        "模型下载和内存占用较高，优先使用可用显卡；处理器运行可能明显慢于实时速度。",
        WhisperSettings(profile=WhisperProfile.ACCURATE, model="medium"),
    ),
}


class WhisperSettingsStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "whisper.json"

    def load(self) -> WhisperSettings:
        if not self.path.is_file():
            return PROFILE_PRESETS[WhisperProfile.BALANCED].settings
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                return PROFILE_PRESETS[WhisperProfile.BALANCED].settings
            if loaded.get("version") != 1:
                return PROFILE_PRESETS[WhisperProfile.BALANCED].settings
            return WhisperSettings(
                profile=WhisperProfile(str(loaded["profile"])),
                model=str(loaded["model"]),
                device=str(loaded["device"]),
                compute_type=str(loaded["compute_type"]),
                language=str(loaded.get("language", "auto")),
                vad_enabled=bool(loaded.get("vad_enabled", True)),
                vad_min_silence_ms=int(loaded.get("vad_min_silence_ms", 400)),
                first_run_completed=bool(loaded.get("first_run_completed", False)),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return PROFILE_PRESETS[WhisperProfile.BALANCED].settings

    def save(self, settings: WhisperSettings) -> None:
        public = asdict(settings)
        public["profile"] = settings.profile.value
        public["version"] = 1
        temporary = self.path.with_suffix(".json.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)


@dataclass(frozen=True, slots=True)
class WhisperCapabilities:
    devices: tuple[str, ...]
    compute_types: dict[str, tuple[str, ...]]
    gpu_name: str | None = None


def detect_whisper_capabilities() -> WhisperCapabilities:
    devices = ["cpu"]
    compute_types: dict[str, tuple[str, ...]] = {"cpu": ("int8", "float32")}
    gpu_name = None
    try:
        import ctranslate2

        cpu_supported = ctranslate2.get_supported_compute_types("cpu")
        compute_types["cpu"] = tuple(
            value for value in ("int8", "float32") if value in cpu_supported
        ) or ("float32",)
        if ctranslate2.get_cuda_device_count() > 0:
            cuda_supported = ctranslate2.get_supported_compute_types("cuda")
            supported = tuple(
                value
                for value in ("float16", "int8_float16", "int8", "float32")
                if value in cuda_supported
            )
            if supported:
                devices.append("cuda")
                compute_types["cuda"] = supported
                gpu_name = "CUDA GPU"
    except (ImportError, RuntimeError, ValueError):
        pass
    return WhisperCapabilities(tuple(devices), compute_types, gpu_name)


@dataclass(frozen=True, slots=True)
class ResolvedWhisperSettings:
    settings: WhisperSettings
    fallback_advice: str = ""


def resolve_whisper_settings(
    requested: WhisperSettings, capabilities: WhisperCapabilities
) -> ResolvedWhisperSettings:
    device = requested.device
    advice: list[str] = []
    if device == "auto":
        device = "cuda" if "cuda" in capabilities.devices else "cpu"
    elif device not in capabilities.devices:
        advice.append("当前环境不支持 CUDA；已回退到 CPU。可安装兼容的 NVIDIA 驱动后重试。")
        device = "cpu"

    supported = capabilities.compute_types[device]
    compute_type = requested.compute_type
    if compute_type == "default":
        compute_type = "float16" if device == "cuda" and "float16" in supported else supported[0]
    elif compute_type not in supported:
        fallback = "float16" if device == "cuda" and "float16" in supported else supported[0]
        advice.append(f"{device.upper()} 不支持 {compute_type}；已改用 {fallback}。")
        compute_type = fallback

    return ResolvedWhisperSettings(
        replace(requested, device=device, compute_type=compute_type), " ".join(advice)
    )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    text: str
    elapsed_seconds: float
    realtime_factor: float
    language: str | None
    resources: dict[str, float]
    actual_settings: WhisperSettings


class WhisperModel(Protocol):
    def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, Any]: ...


class WhisperBenchmark:
    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        model_factory: Callable[[WhisperSettings], WhisperModel] | None = None,
        duration_reader: Callable[[Path], float] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        resource_reader: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.model_factory = model_factory or self._create_model
        self.duration_reader = duration_reader or self._read_duration
        self.clock = clock
        self.resource_reader = resource_reader

    def _create_model(self, settings: WhisperSettings) -> WhisperModel:
        from faster_whisper import WhisperModel as FasterWhisperModel

        return FasterWhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
            download_root=str(self.model_dir) if self.model_dir is not None else None,
        )

    @staticmethod
    def _read_duration(path: Path) -> float:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()

    @staticmethod
    def _peak_process_memory_mb() -> float | None:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("page_fault_count", wintypes.DWORD),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            success = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                ctypes.byref(counters),
                counters.cb,
            )
            if success:
                return counters.peak_working_set_size / 1024 / 1024
            return None
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 1024**2 if sys.platform == "darwin" else 1024
            return peak / divisor
        except (ImportError, OSError):
            return None

    @classmethod
    def _read_resources(cls, cpu_started: float) -> dict[str, float]:
        resources = {"process_cpu_seconds": time.process_time() - cpu_started}
        peak_memory = cls._peak_process_memory_mb()
        if peak_memory is not None:
            resources["peak_memory_mb"] = peak_memory
        return resources

    def run(self, settings: WhisperSettings, sample_path: Path) -> BenchmarkResult:
        duration = self.duration_reader(sample_path)
        started = self.clock()
        cpu_started = time.process_time()
        model = self.model_factory(settings)
        segments, info = model.transcribe(str(sample_path), **whisper_transcribe_options(settings))
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        elapsed = self.clock() - started
        resources = (
            self.resource_reader()
            if self.resource_reader is not None
            else self._read_resources(cpu_started)
        )
        return BenchmarkResult(
            text=text,
            elapsed_seconds=elapsed,
            realtime_factor=elapsed / duration if duration else 0.0,
            language=getattr(info, "language", None),
            resources=resources,
            actual_settings=settings,
        )


@dataclass(slots=True)
class DownloadState:
    model: str
    required_bytes: int
    downloaded_bytes: int = 0
    status: str = "pending"
    error: str | None = None
    cancel_requested: bool = False

    @property
    def progress_percent(self) -> int:
        if self.required_bytes <= 0:
            return 0
        return min(100, round(self.downloaded_bytes * 100 / self.required_bytes))

    @property
    def can_retry(self) -> bool:
        return self.status in {"cancelled", "failed"}

    def mark_progress(self, downloaded_bytes: int) -> None:
        self.downloaded_bytes = max(0, downloaded_bytes)
        self.status = "downloading"
        self.error = None

    def cancel(self) -> None:
        self.cancel_requested = True
        self.status = "cancelled"

    def retry(self) -> None:
        self.downloaded_bytes = 0
        self.status = "pending"
        self.error = None
        self.cancel_requested = False

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.error = error


class _DownloadCancelled(Exception):
    pass


class WhisperModelDownloader:
    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        repository_files: Callable[[str], tuple[tuple[str, int], ...]] | None = None,
        download_file: Callable[[str, str], None] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.repository_files = repository_files or self._repository_files
        self.download_file = download_file

    @staticmethod
    def repository_id(model: str) -> str:
        return canonical_whisper_repository_id(model)

    @classmethod
    def _repository_files(cls, model: str) -> tuple[tuple[str, int], ...]:
        from huggingface_hub import HfApi

        info = HfApi().model_info(cls.repository_id(model), files_metadata=True)
        return tuple(
            (sibling.rfilename, int(sibling.size or 0))
            for sibling in info.siblings
            if sibling.rfilename
        )

    def _download_file(
        self,
        model: str,
        filename: str,
        *,
        file_size: int,
        downloaded: int,
        state: DownloadState,
        on_progress: Callable[[DownloadState], None],
        cancel_event: threading.Event,
    ) -> None:
        from huggingface_hub import hf_hub_download
        from tqdm.auto import tqdm

        class CancellableProgress(tqdm):
            def update(self, amount: float | None = 1) -> bool | None:
                if cancel_event.is_set():
                    raise _DownloadCancelled
                result = super().update(amount)
                state.mark_progress(min(downloaded + int(self.n), downloaded + file_size))
                on_progress(state)
                return result

        hf_hub_download(
            repo_id=self.repository_id(model),
            filename=filename,
            cache_dir=str(self.model_dir) if self.model_dir is not None else None,
            tqdm_class=CancellableProgress,
        )

    def prepare(
        self,
        model: str,
        *,
        on_progress: Callable[[DownloadState], None],
        cancel_event: threading.Event,
    ) -> DownloadState:
        state = DownloadState(model, 0)
        on_progress(state)
        downloaded = 0
        try:
            files = self.repository_files(model)
            state.required_bytes = sum(size for _filename, size in files)
            for filename, size in files:
                if cancel_event.is_set():
                    state.cancel()
                    on_progress(state)
                    return state
                if self.download_file is None:
                    self._download_file(
                        model,
                        filename,
                        file_size=size,
                        downloaded=downloaded,
                        state=state,
                        on_progress=on_progress,
                        cancel_event=cancel_event,
                    )
                else:
                    self.download_file(model, filename)
                if cancel_event.is_set():
                    raise _DownloadCancelled
                downloaded += size
                state.mark_progress(downloaded)
                on_progress(state)
        except _DownloadCancelled:
            state.cancel()
            on_progress(state)
            return state
        except Exception as exc:  # noqa: BLE001 - download boundary exposes retry state
            state.fail(str(exc).strip() or type(exc).__name__)
            on_progress(state)
            return state
        state.status = "complete"
        on_progress(state)
        return state


def create_builtin_whisper_sample(data_dir: Path) -> Path:
    if sys.platform != "win32":
        raise RuntimeError("内置语音样本目前仅支持 Windows；请在 Windows 真机运行测试。")
    path = data_dir / "samples" / "whisper-first-run.wav"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = str(path).replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$voice.SetOutputToWaveFile('{escaped}'); "
        "$voice.Speak('境织正在测试本地语音识别的速度和准确度。'); "
        "$voice.Dispose()"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or not path.is_file():
        detail = completed.stderr.strip() or "Windows speech synthesis failed"
        raise RuntimeError(f"无法生成内置语音样本：{detail}")
    return path
