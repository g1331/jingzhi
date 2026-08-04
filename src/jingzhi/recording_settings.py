from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from jingzhi.capture.devices import (
    DeviceSnapshot,
    ResolvedRecordingSelection,
)


@dataclass(frozen=True, slots=True)
class RecordingPreferences:
    display_ids: tuple[str, ...] = ()
    system_audio_id: str | None = None
    microphone_id: str | None = None
    system_audio_enabled: bool = True
    microphone_enabled: bool = True
    estimated_duration_minutes: int = 60


class RecordingSettingsStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "recording.json"

    def load(self) -> RecordingPreferences:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return RecordingPreferences()
        if not isinstance(loaded, dict):
            return RecordingPreferences()
        display_ids = loaded.get("display_ids", [])
        if not isinstance(display_ids, list):
            display_ids = []
        duration = loaded.get("estimated_duration_minutes", 60)
        try:
            duration_minutes = int(duration)
        except (TypeError, ValueError):
            duration_minutes = 60
        return RecordingPreferences(
            display_ids=tuple(item for item in display_ids if isinstance(item, str)),
            system_audio_id=_optional_string(loaded.get("system_audio_id")),
            microphone_id=_optional_string(loaded.get("microphone_id")),
            system_audio_enabled=_boolean(loaded.get("system_audio_enabled"), True),
            microphone_enabled=_boolean(loaded.get("microphone_enabled"), True),
            estimated_duration_minutes=max(1, min(24 * 60, duration_minutes)),
        )

    def save(self, preferences: RecordingPreferences) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(preferences), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _boolean(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _selected_audio(identifier: str | None, devices):  # type: ignore[no-untyped-def]
    if identifier is not None:
        selected = next((device for device in devices if device.id == identifier), None)
        if selected is not None:
            return selected
    return next(
        (device for device in devices if device.is_default), devices[0] if devices else None
    )


def resolve_recording_selection(
    preferences: RecordingPreferences, snapshot: DeviceSnapshot
) -> ResolvedRecordingSelection:
    available = {display.id: display for display in snapshot.displays}
    displays = tuple(
        available[identifier] for identifier in preferences.display_ids if identifier in available
    )
    if not displays:
        displays = snapshot.displays
    return ResolvedRecordingSelection(
        displays=displays,
        system_audio=(
            _selected_audio(preferences.system_audio_id, snapshot.system_audio)
            if preferences.system_audio_enabled
            else None
        ),
        microphone=(
            _selected_audio(preferences.microphone_id, snapshot.microphones)
            if preferences.microphone_enabled
            else None
        ),
        estimated_duration_minutes=preferences.estimated_duration_minutes,
    )


def estimate_storage_bytes(
    selection: ResolvedRecordingSelection,
    *,
    screen_interval_s: float,
    audio_storage_rate: int,
) -> int:
    seconds = selection.estimated_duration_minutes * 60
    frames_per_display = seconds / max(0.1, screen_interval_s)
    display_bytes = len(selection.displays) * frames_per_display * 250_000
    audio_sources = int(selection.system_audio is not None) + int(selection.microphone is not None)
    audio_bytes = audio_sources * seconds * audio_storage_rate * 2
    return round(display_bytes + audio_bytes)
