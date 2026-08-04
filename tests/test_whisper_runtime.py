from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from jingzhi.capture.devices import DeviceSnapshot, RecordingSelection
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.session import SessionManager
from jingzhi.whisper_settings import (
    PROFILE_PRESETS,
    WhisperCapabilities,
    WhisperProfile,
    WhisperSettingsStore,
)


class EmptyDeviceCatalog:
    def snapshot(self) -> DeviceSnapshot:
        return DeviceSnapshot((), (), ())

    def microphone_level(self, _device) -> float:
        return 0.0

    def audio_locator(self, _identifier: str):
        raise LookupError("No audio devices")


class FakeTranscriptionWorker:
    instances: ClassVar[list[FakeTranscriptionWorker]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return False


def test_saved_whisper_settings_are_loaded_by_environment_settings(
    tmp_path: Path, monkeypatch
) -> None:
    selected = PROFILE_PRESETS[WhisperProfile.ACCURATE].settings
    WhisperSettingsStore(tmp_path).save(selected)
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)

    restarted = Settings.from_env()

    assert restarted.whisper == selected


def test_session_records_requested_and_actual_whisper_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    FakeTranscriptionWorker.instances.clear()
    monkeypatch.setattr("jingzhi.session.TranscriptionWorker", FakeTranscriptionWorker)
    requested = PROFILE_PRESETS[WhisperProfile.BALANCED].settings
    settings = Settings(data_dir=tmp_path, whisper=requested)
    manager = SessionManager(
        settings,
        device_catalog=EmptyDeviceCatalog(),
        whisper_capabilities=WhisperCapabilities(
            devices=("cpu",), compute_types={"cpu": ("int8", "float32")}
        ),
    )

    session_id = manager.start(
        "Whisper 审计",
        selection=RecordingSelection((), None, None, 15),
    )

    record = manager.database.whisper_run(session_id)
    assert record is not None
    assert record.profile == "balanced"
    assert record.requested_device == "auto"
    assert record.actual_model == "small"
    assert record.actual_device == "cpu"
    assert record.actual_compute_type == "int8"
    assert FakeTranscriptionWorker.instances[0].kwargs["settings"].device == "cpu"
    assert FakeTranscriptionWorker.instances[0].started
    manager.stop()


def test_manager_saves_advanced_whisper_settings_for_restart(tmp_path: Path) -> None:
    manager = SessionManager(
        Settings(data_dir=tmp_path),
        device_catalog=EmptyDeviceCatalog(),
        whisper_capabilities=WhisperCapabilities(
            devices=("cpu",), compute_types={"cpu": ("int8", "float32")}
        ),
    )
    selected = PROFILE_PRESETS[WhisperProfile.LIGHTWEIGHT].settings

    manager.configure_whisper(selected)
    manager.save_whisper()

    assert WhisperSettingsStore(tmp_path).load() == selected
    assert Database(tmp_path / "jingzhi.sqlite3").whisper_run("missing") is None
