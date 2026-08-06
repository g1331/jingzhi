from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResourceMetrics:
    cpu_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    gpu_utilization_percent: float | None
    gpu_memory_used_bytes: int | None
    gpu_memory_total_bytes: int | None


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    session_id: str | None
    duration_ms: int
    frame_count: int
    storage_bytes: int
    free_bytes: int
    transcribed_audio_ms: int
    transcription_realtime_factor: float | None
    correction_backlog: int
    pending_audio_chunks: int
    failed_audio_chunks: int
    retryable_model_tasks: int
    cpu_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    gpu_utilization_percent: float | None
    gpu_memory_used_bytes: int | None
    gpu_memory_total_bytes: int | None
    sampled_at_utc: str


@dataclass(frozen=True, slots=True)
class AudioRecoveryReport:
    queued_chunks: int
    missing_chunks: int


@dataclass(frozen=True, slots=True)
class _CpuTimes:
    idle: int
    total: int


class SystemMetricsSampler:
    """Best-effort local resource metrics with explicit unavailable values.

    Windows APIs are used for CPU and memory so the desktop build does not need a
    third-party monitoring dependency. GPU values are read from nvidia-smi when it
    is installed; an unavailable GPU is represented by ``None`` rather than a
    fabricated zero.
    """

    def __init__(self) -> None:
        self._previous_cpu: _CpuTimes | None = None
        self._gpu_cache: tuple[float, ResourceMetrics] | None = None

    def sample(self) -> ResourceMetrics:
        cpu_times = self._read_cpu_times()
        cpu_percent: float | None = None
        if cpu_times is not None and self._previous_cpu is not None:
            total_delta = cpu_times.total - self._previous_cpu.total
            idle_delta = cpu_times.idle - self._previous_cpu.idle
            if total_delta > 0:
                cpu_percent = max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100))
        self._previous_cpu = cpu_times
        memory_used, memory_total = self._read_memory()
        gpu = self._read_gpu()
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_used_bytes=memory_used,
            memory_total_bytes=memory_total,
            gpu_utilization_percent=gpu[0] if gpu is not None else None,
            gpu_memory_used_bytes=gpu[1] if gpu is not None else None,
            gpu_memory_total_bytes=gpu[2] if gpu is not None else None,
        )

    @staticmethod
    def _read_cpu_times() -> _CpuTimes | None:
        if os.name != "nt":
            try:
                load = os.getloadavg()[0]
            except (AttributeError, OSError):
                return None
            total = int(time.monotonic_ns())
            cpu_count = max(1, os.cpu_count() or 1)
            idle = int(total * max(0.0, 1.0 - load / cpu_count))
            return _CpuTimes(idle=idle, total=total)

        class FileTime(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

        idle = FileTime()
        kernel = FileTime()
        user = FileTime()
        try:
            success = ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            )
        except (AttributeError, OSError):
            return None
        if not success:
            return None

        def value(file_time: FileTime) -> int:
            return (int(file_time.high) << 32) | int(file_time.low)

        idle_value = value(idle)
        total = value(kernel) + value(user)
        return _CpuTimes(idle=idle_value, total=total)

    @staticmethod
    def _read_memory() -> tuple[int | None, int | None]:
        if os.name == "nt":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint32),
                    ("memory_load", ctypes.c_uint32),
                    ("total_physical", ctypes.c_uint64),
                    ("available_physical", ctypes.c_uint64),
                    ("total_page_file", ctypes.c_uint64),
                    ("available_page_file", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64),
                    ("available_virtual", ctypes.c_uint64),
                    ("available_extended_virtual", ctypes.c_uint64),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            try:
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return (
                        int(status.total_physical - status.available_physical),
                        int(status.total_physical),
                    )
            except (AttributeError, OSError):
                pass
            return None, None

        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                name, value = line.split(":", 1)
                values[name] = int(value.strip().split()[0]) * 1024
            total = values["MemTotal"]
            available = values.get("MemAvailable", values.get("MemFree", 0))
            return total - available, total
        except (KeyError, OSError, ValueError):
            return None, None

    def _read_gpu(self) -> tuple[float, int, int] | None:
        now = time.monotonic()
        if self._gpu_cache is not None and now - self._gpu_cache[0] < 2.0:
            metrics = self._gpu_cache[1]
            if (
                metrics.gpu_utilization_percent is not None
                and metrics.gpu_memory_used_bytes is not None
                and metrics.gpu_memory_total_bytes is not None
            ):
                return (
                    metrics.gpu_utilization_percent,
                    metrics.gpu_memory_used_bytes,
                    metrics.gpu_memory_total_bytes,
                )
            return None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=0.5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            first = completed.stdout.strip().splitlines()[0]
            utilization, used_mb, total_mb = (float(item.strip()) for item in first.split(","))
            result = (utilization, int(used_mb * 1024 * 1024), int(total_mb * 1024 * 1024))
        except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError, ValueError):
            self._gpu_cache = (now, ResourceMetrics(None, None, None, None, None, None))
            return None
        self._gpu_cache = (
            now,
            ResourceMetrics(
                None,
                None,
                None,
                result[0],
                result[1],
                result[2],
            ),
        )
        return result


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file() and not item.is_symlink():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def disk_free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    try:
        return int(shutil.disk_usage(target).free)
    except OSError:
        return 0


def format_bytes(value: int | None) -> str:
    if value is None:
        return "未知"
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(max(0, value))
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f}{unit}" if unit != "B" else f"{int(amount)}B"
        amount /= 1024
    return "未知"


def format_runtime_metrics(metrics: RuntimeMetrics) -> str:
    cpu = "未知" if metrics.cpu_percent is None else f"{metrics.cpu_percent:.0f}%"
    memory = (
        "未知"
        if metrics.memory_used_bytes is None or metrics.memory_total_bytes is None
        else f"{format_bytes(metrics.memory_used_bytes)}/{format_bytes(metrics.memory_total_bytes)}"
    )
    gpu = (
        "未检测"
        if metrics.gpu_utilization_percent is None
        else (
            f"{metrics.gpu_utilization_percent:.0f}%"
            f"/{format_bytes(metrics.gpu_memory_used_bytes)}"
            f"/{format_bytes(metrics.gpu_memory_total_bytes)}"
        )
    )
    rtf = (
        "等待转写"
        if metrics.transcription_realtime_factor is None
        else f"{metrics.transcription_realtime_factor:.2f}x"
    )
    return (
        f"关键帧 {metrics.frame_count} · 存储 {format_bytes(metrics.storage_bytes)}"
        f" · 空闲磁盘 {format_bytes(metrics.free_bytes)} · 转写实时系数 {rtf}"
        f" · 校订积压 {metrics.correction_backlog}"
        f" · 待转写 {metrics.pending_audio_chunks} · 失败转写 {metrics.failed_audio_chunks}"
        f" · 待重试模型 {metrics.retryable_model_tasks}"
        f" · CPU {cpu} · GPU {gpu} · 内存 {memory}"
    )


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()
