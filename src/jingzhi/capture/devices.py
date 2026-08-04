from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class DisplayDevice:
    id: str
    name: str
    monitor: dict[str, int]
    preview: Image.Image


@dataclass(frozen=True, slots=True)
class AudioDevice:
    id: str
    name: str
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class DeviceSnapshot:
    displays: tuple[DisplayDevice, ...]
    system_audio: tuple[AudioDevice, ...]
    microphones: tuple[AudioDevice, ...]


@dataclass(frozen=True, slots=True)
class RecordingSelection:
    display_ids: tuple[str, ...]
    system_audio_id: str | None
    microphone_id: str | None
    estimated_duration_minutes: int


@dataclass(frozen=True, slots=True)
class ResolvedRecordingSelection:
    displays: tuple[DisplayDevice, ...]
    system_audio: AudioDevice | None
    microphone: AudioDevice | None
    estimated_duration_minutes: int


class DeviceCatalog(Protocol):
    def snapshot(self) -> DeviceSnapshot: ...

    def microphone_level(self, device: AudioDevice | None) -> float: ...

    def audio_locator(self, identifier: str) -> tuple[Any, int | None]: ...


class WindowsDeviceCatalog:
    """Enumerates current Windows capture sources and samples recognizable previews."""

    def __init__(self) -> None:
        self._audio_locators: dict[str, tuple[Any, int | None]] = {}

    def snapshot(self) -> DeviceSnapshot:
        return DeviceSnapshot(
            displays=self._displays(),
            system_audio=self._system_audio_devices(),
            microphones=self._microphones(),
        )

    @staticmethod
    def _displays() -> tuple[DisplayDevice, ...]:
        from mss import MSS

        metadata = _windows_display_metadata()
        devices: list[DisplayDevice] = []
        with MSS() as capture:
            for index, monitor in enumerate(capture.monitors[1:], start=1):
                bounds = {key: int(monitor[key]) for key in ("left", "top", "width", "height")}
                rectangle = (
                    bounds["left"],
                    bounds["top"],
                    bounds["left"] + bounds["width"],
                    bounds["top"] + bounds["height"],
                )
                stable_id, device_name = metadata.get(
                    rectangle, (f"windows-display-{index}", f"显示器 {index}")
                )
                shot = capture.grab(bounds)
                preview = Image.frombytes("RGB", shot.size, shot.rgb)
                preview.thumbnail((480, 270), Image.Resampling.LANCZOS)
                devices.append(DisplayDevice(f"display:{stable_id}", device_name, bounds, preview))
        return tuple(devices)

    def _system_audio_devices(self) -> tuple[AudioDevice, ...]:
        import soundcard as sc

        default = sc.default_speaker()
        default_id = str(default.id) if default is not None else None
        devices: list[AudioDevice] = []
        for speaker in sc.all_speakers():
            identifier = f"speaker:{speaker.id}"
            self._audio_locators[identifier] = (speaker.id, None)
            devices.append(
                AudioDevice(
                    id=identifier,
                    name=str(speaker.name),
                    is_default=str(speaker.id) == default_id,
                )
            )
        return tuple(devices)

    def _microphones(self) -> tuple[AudioDevice, ...]:
        import soundcard as sc
        import sounddevice as sd

        default = sc.default_microphone()
        default_id = str(default.id) if default is not None else None
        host_apis = sd.query_hostapis()
        portaudio_inputs = [
            (index, str(item["name"]))
            for index, item in enumerate(sd.query_devices())
            if int(item["max_input_channels"]) >= 1
            and "WASAPI" in str(host_apis[int(item["hostapi"])]["name"])
        ]
        devices: list[AudioDevice] = []
        for endpoint in sc.all_microphones(include_loopback=False):
            endpoint_id = str(endpoint.id)
            endpoint_name = str(endpoint.name)
            matching_indices = [
                index
                for index, portaudio_name in portaudio_inputs
                if endpoint_name == portaudio_name or endpoint_name.startswith(portaudio_name)
            ]
            fallback_index = matching_indices[0] if len(matching_indices) == 1 else None
            identifier = f"microphone:{endpoint_id}"
            self._audio_locators[identifier] = (endpoint.id, fallback_index)
            devices.append(
                AudioDevice(
                    id=identifier,
                    name=endpoint_name,
                    is_default=endpoint_id == default_id,
                )
            )
        return tuple(devices)

    def audio_locator(self, identifier: str) -> tuple[Any, int | None]:
        try:
            return self._audio_locators[identifier]
        except KeyError as exc:
            raise RuntimeError(f"录制来源已不可用：{identifier}") from exc

    def microphone_level(self, device: AudioDevice | None) -> float:
        if device is None:
            return 0.0
        import soundcard as sc
        import sounddevice as sd

        endpoint_id, portaudio_index = self.audio_locator(device.id)
        try:
            if portaudio_index is not None:
                samples = sd.rec(
                    2_400,
                    samplerate=48_000,
                    channels=1,
                    dtype="float32",
                    device=portaudio_index,
                    blocking=True,
                )
            else:
                microphone = sc.get_microphone(endpoint_id)
                with microphone.recorder(samplerate=48_000, blocksize=2_400) as recorder:
                    samples = recorder.record(numframes=2_400)
        except (AssertionError, RuntimeError, sd.PortAudioError, ValueError):
            return 0.0
        rms = math.sqrt(float(np.mean(np.square(samples, dtype=np.float32))))
        return min(1.0, rms * 4)


def _windows_display_metadata() -> dict[tuple[int, int, int, int], tuple[str, str]]:
    if os.name != "nt":
        return {}
    from ctypes import wintypes

    class Rect(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", Rect),
            ("rcWork", Rect),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    class Win32DisplayDeviceInfo(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("DeviceName", wintypes.WCHAR * 32),
            ("DeviceString", wintypes.WCHAR * 128),
            ("StateFlags", wintypes.DWORD),
            ("DeviceID", wintypes.WCHAR * 128),
            ("DeviceKey", wintypes.WCHAR * 128),
        ]

    user32 = ctypes.windll.user32
    metadata: dict[tuple[int, int, int, int], tuple[str, str]] = {}
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(Rect),
        wintypes.LPARAM,
    )

    def collect(hmonitor, _hdc, _rect, _data):  # type: ignore[no-untyped-def]
        monitor = MonitorInfo()
        monitor.cbSize = ctypes.sizeof(MonitorInfo)
        if not user32.GetMonitorInfoW(hmonitor, ctypes.byref(monitor)):
            return True
        device = Win32DisplayDeviceInfo()
        device.cb = ctypes.sizeof(Win32DisplayDeviceInfo)
        user32.EnumDisplayDevicesW(monitor.szDevice, 0, ctypes.byref(device), 1)
        rectangle = (
            monitor.rcMonitor.left,
            monitor.rcMonitor.top,
            monitor.rcMonitor.right,
            monitor.rcMonitor.bottom,
        )
        stable_id = device.DeviceID or device.DeviceKey or monitor.szDevice
        name = device.DeviceString or monitor.szDevice
        metadata[rectangle] = (stable_id, name)
        return True

    callback = callback_type(collect)
    user32.EnumDisplayMonitors(None, None, callback, 0)
    return metadata
