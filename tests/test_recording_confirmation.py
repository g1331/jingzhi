from __future__ import annotations

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog

from jingzhi.application import JingzhiApplicationService
from jingzhi.capture.devices import (
    AudioDevice,
    DeviceSnapshot,
    DisplayDevice,
    RecordingSelection,
)
from jingzhi.capture.screen import ScreenCaptureWorker
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.recording_settings import (
    RecordingPreferences,
    RecordingSettingsStore,
    estimate_storage_bytes,
    resolve_recording_selection,
)
from jingzhi.ui import MainWindow, RecordingConfirmationDialog


def _display(identifier: str, name: str, left: int, color: str) -> DisplayDevice:
    return DisplayDevice(
        id=identifier,
        name=name,
        monitor={"left": left, "top": 0, "width": 320, "height": 180},
        preview=Image.new("RGB", (320, 180), color),
    )


def _process_until(application: QApplication, predicate, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    application.processEvents()
    assert predicate()


def test_recording_preferences_restore_devices_and_fall_back_safely(tmp_path: Path) -> None:
    first = _display("display:left", "左侧显示器", 0, "navy")
    second = _display("display:right", "右侧显示器", 320, "white")
    speakers = (AudioDevice("speaker:desk", "桌面扬声器", True),)
    microphones = (AudioDevice("microphone:usb", "USB 麦克风", True),)
    snapshot = DeviceSnapshot((first, second), speakers, microphones)
    store = RecordingSettingsStore(tmp_path)
    store.save(
        RecordingPreferences(
            display_ids=("display:right",),
            system_audio_id="speaker:missing",
            microphone_id="microphone:missing",
            estimated_duration_minutes=90,
        )
    )

    selection = resolve_recording_selection(store.load(), snapshot)

    assert selection.displays == (second,)
    assert selection.system_audio == speakers[0]
    assert selection.microphone == microphones[0]
    assert selection.estimated_duration_minutes == 90

    all_displays = resolve_recording_selection(
        RecordingPreferences(estimated_duration_minutes=60), snapshot
    )
    assert all_displays.displays == (first, second)
    assert (
        estimate_storage_bytes(all_displays, screen_interval_s=1.0, audio_storage_rate=16_000) > 0
    )


def test_malformed_recording_preferences_fall_back_to_defaults(tmp_path: Path) -> None:
    store = RecordingSettingsStore(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        '{"display_ids": "not-a-list", "estimated_duration_minutes": "invalid"}',
        encoding="utf-8",
    )

    assert store.load() == RecordingPreferences()


class FakeDeviceCatalog:
    def __init__(self, snapshots: list[DeviceSnapshot]) -> None:
        self.snapshots = snapshots
        self.index = 0

    def snapshot(self) -> DeviceSnapshot:
        return self.snapshots[min(self.index, len(self.snapshots) - 1)]

    def microphone_level(self, _device: AudioDevice | None) -> float:
        return 0.42


def test_confirmation_dialog_handles_hotplug_and_empty_display_selection(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    first = _display("display:first", "主显示器", 0, "navy")
    second = _display("display:second", "副显示器", 320, "white")
    speaker = AudioDevice("speaker:default", "默认扬声器", True)
    replacement_speaker = AudioDevice("speaker:replacement", "新默认扬声器", True)
    microphone = AudioDevice("microphone:default", "默认麦克风", True)
    catalog = FakeDeviceCatalog(
        [
            DeviceSnapshot((first,), (speaker,), (microphone,)),
            DeviceSnapshot((first, second), (replacement_speaker,), (microphone,)),
        ]
    )
    store = RecordingSettingsStore(tmp_path)
    dialog = RecordingConfirmationDialog(
        catalog,
        store,
        screen_interval_s=1.0,
        audio_storage_rate=16_000,
    )
    dialog.show()
    _process_until(application, lambda: set(dialog.display_checks) == {"display:first"})
    _process_until(application, lambda: dialog.microphone_level.value() == 42)

    assert dialog.display_checks["display:first"].isChecked()
    assert "估算" in dialog.storage_estimate.text()

    dialog.display_checks["display:first"].setChecked(False)
    dialog.microphone_combo.setCurrentIndex(0)
    catalog.index = 1
    dialog.refresh_devices()
    _process_until(
        application,
        lambda: set(dialog.display_checks) == {"display:first", "display:second"},
    )

    assert dialog.system_audio_combo.currentData() == "speaker:replacement"
    assert dialog.microphone_combo.currentData() is None
    selection = dialog.recording_selection()
    assert selection.display_ids == ()
    assert selection.system_audio_id == "speaker:replacement"
    assert selection.microphone_id is None
    dialog.close()


@pytest.mark.skipif(
    os.name != "nt" or os.getenv("JINGZHI_WINDOWS_HOTPLUG_CHECK") != "1",
    reason="Set JINGZHI_WINDOWS_HOTPLUG_CHECK=1 for the interactive Windows hardware check",
)
def test_windows_confirmation_refreshes_after_real_hotplug(tmp_path: Path) -> None:
    from jingzhi.capture.devices import WindowsDeviceCatalog

    application = QApplication.instance() or QApplication([])
    catalog = WindowsDeviceCatalog()
    before = catalog.snapshot()
    assert before.displays
    store = RecordingSettingsStore(tmp_path)
    store.save(
        RecordingPreferences(
            display_ids=(before.displays[0].id,),
            system_audio_id=before.system_audio[0].id if before.system_audio else None,
            microphone_id=before.microphones[0].id if before.microphones else None,
        )
    )
    dialog = RecordingConfirmationDialog(
        catalog,
        store,
        screen_interval_s=1.0,
        audio_storage_rate=16_000,
    )
    dialog.show()
    _process_until(application, lambda: not dialog._refresh_in_progress, timeout=5)

    wait_seconds = int(os.getenv("JINGZHI_WINDOWS_HOTPLUG_WAIT_SECONDS", "15"))
    print(f"Hot-plug a display or audio device within {wait_seconds} seconds...")
    time.sleep(wait_seconds)
    after = catalog.snapshot()
    before_ids = {
        *(item.id for item in before.displays),
        *(item.id for item in before.system_audio),
        *(item.id for item in before.microphones),
    }
    after_ids = {
        *(item.id for item in after.displays),
        *(item.id for item in after.system_audio),
        *(item.id for item in after.microphones),
    }
    assert before_ids != after_ids, "No Windows device hot-plug was detected"

    dialog.refresh_devices()
    _process_until(application, lambda: not dialog._refresh_in_progress, timeout=5)
    assert set(dialog.display_checks) == {item.id for item in after.displays}
    dialog.recording_selection()
    dialog.close()


def test_main_window_enters_recording_state_after_confirmed_sources(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    class Recorder:
        is_recording = False
        selection: RecordingSelection | None = None

        def start(self, _title: str, *, selection: RecordingSelection | None = None) -> str:
            self.selection = selection
            self.is_recording = True
            return "confirmed-session"

        def stop(self) -> str | None:
            self.is_recording = False
            return "confirmed-session"

    selection = RecordingSelection(("display:stable",), "speaker:stable", None, 60)

    class AcceptedDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

        def recording_selection(self) -> RecordingSelection:
            return selection

    monkeypatch.setattr("jingzhi.ui.RecordingConfirmationDialog", AcceptedDialog)
    application = QApplication.instance() or QApplication([])
    recorder = Recorder()
    service = JingzhiApplicationService(Database(tmp_path / "ui-start.sqlite3"), recorder=recorder)
    window = MainWindow(Settings(data_dir=tmp_path), service=service)

    window._start()
    application.processEvents()

    assert recorder.selection == selection
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()
    assert window.system_audio_check.isChecked()
    assert not window.microphone_check.isChecked()
    window.close()


class FixedClock:
    def now_ms(self) -> int:
        return 1_000


class FakeShot:
    size = (2, 1)
    rgb = bytes([255, 0, 0, 0, 255, 0])


class FakeCapture:
    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def grab(self, _monitor):
        return FakeShot()


def test_each_display_writes_an_independent_source_and_path(tmp_path: Path) -> None:
    database = Database(tmp_path / "capture.sqlite3")
    session_id = database.create_session("多显示器", "2026-08-04T00:00:00+00:00")
    displays = (
        _display("display:left", "左侧显示器", 0, "navy"),
        _display("display:right", "右侧显示器", 320, "white"),
    )

    for display in displays:
        stopped = threading.Event()
        worker = ScreenCaptureWorker(
            database=database,
            session_id=session_id,
            clock=FixedClock(),  # type: ignore[arg-type]
            display=display,
            output_dir=tmp_path / "frames" / display.id.replace(":", "-"),
            stop_event=stopped,
            interval_s=0,
            hash_distance=0,
            on_frame=lambda _ts, _path, event=stopped: event.set(),
            capture_factory=FakeCapture,
        )
        worker.run()

    frames = database.timeline_frames(session_id, 0, 2_000)
    assert [frame.source_id for frame in frames] == ["display:left", "display:right"]
    assert frames[0].path != frames[1].path
    assert all(frame.path.is_file() for frame in frames)
