from __future__ import annotations

import os
import queue
import threading
import time
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog

from jingzhi.application import JingzhiApplicationService
from jingzhi.capture.audio import AudioCaptureWorker
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


def test_screen_frame_captured_at_pause_boundary_is_not_persisted(tmp_path: Path) -> None:
    database = Database(tmp_path / "screen-pause.sqlite3")
    session_id = database.create_session("暂停边界屏幕", "2026-08-04T00:00:00+00:00")
    stop_event = threading.Event()
    pause_event = threading.Event()

    class PauseAfterGrabCapture(FakeCapture):
        def grab(self, monitor):
            pause_event.set()
            stop_event.set()
            return super().grab(monitor)

    worker = ScreenCaptureWorker(
        database=database,
        session_id=session_id,
        clock=FixedClock(),  # type: ignore[arg-type]
        display=_display("display:pause", "暂停显示器", 0, "navy"),
        output_dir=tmp_path / "frames" / "pause",
        stop_event=stop_event,
        pause_event=pause_event,
        interval_s=0,
        hash_distance=0,
        capture_factory=PauseAfterGrabCapture,
    )

    worker.run()

    assert database.timeline_frames(session_id, 0, 2_000) == []


class AdvancingClock:
    def __init__(self) -> None:
        self.value = 0

    def now_ms(self) -> int:
        self.value += 1_000
        return self.value


class SoundcardRuntimeWarning(RuntimeWarning):
    pass


class FakeAudioRecorder:
    sample_rate = 10

    def __init__(self, *, mode: str, stop_event: threading.Event | None = None) -> None:
        self.mode = mode
        self.stop_event = stop_event
        self.calls = 0
        self.last_overflowed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def record(self, numframes: int) -> np.ndarray:
        self.calls += 1
        self.last_overflowed = self.mode == "overflow"
        if self.mode == "warning":
            warnings.warn("data discontinuity in recording", SoundcardRuntimeWarning)
        if self.mode == "empty":
            return np.empty((0, 1), dtype=np.float32)
        samples = np.zeros((numframes, 1), dtype=np.float32)
        if self.stop_event is not None and self.calls >= 3:
            self.stop_event.set()
        return samples


def _audio_worker(
    tmp_path: Path,
    *,
    mode: str,
    on_failure=None,  # type: ignore[no-untyped-def]
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> tuple[AudioCaptureWorker, FakeAudioRecorder]:
    database = Database(tmp_path / f"{mode}.sqlite3")
    session_id = database.create_session("音频采集", "2026-08-04T00:00:00+00:00")
    actual_stop_event = stop_event or threading.Event()
    recorder = FakeAudioRecorder(
        mode=mode, stop_event=actual_stop_event if mode == "silence" else None
    )
    worker = AudioCaptureWorker(
        database=database,
        session_id=session_id,
        clock=AdvancingClock(),  # type: ignore[arg-type]
        source="microphone",
        device=None,
        device_catalog=object(),  # type: ignore[arg-type]
        output_dir=tmp_path / "audio",
        stop_event=actual_stop_event,
        pause_event=pause_event,
        chunk_queue=queue.Queue(),
        sample_rate=10,
        storage_sample_rate=10,
        chunk_s=0.1,
        on_failure=on_failure,
    )
    worker._open_recorder = lambda: recorder  # type: ignore[method-assign]
    return worker, recorder


def test_sustained_empty_audio_reports_without_treating_silence_as_failure(tmp_path: Path) -> None:
    failures: list[tuple[str, str, int, int, str]] = []
    worker, recorder = _audio_worker(
        tmp_path, mode="empty", on_failure=lambda *args: failures.append(args)
    )

    worker.run()

    assert recorder.calls >= 3
    assert failures and failures[0][1] == "stream_stopped"


def test_sustained_audio_overflow_reports_source_failure(tmp_path: Path) -> None:
    failures: list[tuple[str, str, int, int, str]] = []
    worker, recorder = _audio_worker(
        tmp_path, mode="overflow", on_failure=lambda *args: failures.append(args)
    )

    worker.run()

    assert recorder.calls >= 3
    assert failures and failures[0][1] == "overflow"


def test_sustained_soundcard_warning_reports_source_failure(tmp_path: Path) -> None:
    failures: list[tuple[str, str, int, int, str]] = []
    worker, recorder = _audio_worker(
        tmp_path, mode="warning", on_failure=lambda *args: failures.append(args)
    )

    worker.run()

    assert recorder.calls >= 3
    assert failures and failures[0][1] == "overflow"


def test_normal_silent_audio_does_not_report_failure(tmp_path: Path) -> None:
    stop_event = threading.Event()
    failures: list[tuple[str, str, int, int, str]] = []
    worker, recorder = _audio_worker(
        tmp_path,
        mode="silence",
        on_failure=lambda *args: failures.append(args),
        stop_event=stop_event,
    )

    worker.run()

    assert recorder.calls == 3
    assert failures == []


def test_pause_event_blocks_audio_until_resumed(tmp_path: Path) -> None:
    class PauseAwareStopEvent(threading.Event):
        def __init__(self) -> None:
            super().__init__()
            self.waiting = threading.Event()

        def wait(self, timeout=None):  # type: ignore[no-untyped-def]
            self.waiting.set()
            return super().wait(timeout)

    stop_event = PauseAwareStopEvent()
    pause_event = threading.Event()
    pause_event.set()
    worker, recorder = _audio_worker(
        tmp_path,
        mode="silence",
        stop_event=stop_event,
        pause_event=pause_event,
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    assert stop_event.waiting.wait(timeout=2)
    assert recorder.calls == 0
    pause_event.clear()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert recorder.calls == 3


def test_pause_persists_audio_captured_before_pause(tmp_path: Path) -> None:
    database = Database(tmp_path / "pause-partial.sqlite3")
    session_id = database.create_session("暂停前音频", "2026-08-04T00:00:00+00:00")
    stop_event = threading.Event()
    pause_event = threading.Event()

    class PauseRecorder(FakeAudioRecorder):
        def record(self, numframes: int) -> np.ndarray:
            block = super().record(numframes)
            pause_event.set()
            stop_event.set()
            return block

    recorder = PauseRecorder(mode="silence")
    worker = AudioCaptureWorker(
        database=database,
        session_id=session_id,
        clock=AdvancingClock(),  # type: ignore[arg-type]
        source="microphone",
        device=None,
        device_catalog=object(),  # type: ignore[arg-type]
        output_dir=tmp_path / "audio",
        stop_event=stop_event,
        pause_event=pause_event,
        chunk_queue=queue.Queue(),
        sample_rate=10,
        storage_sample_rate=10,
        chunk_s=0.3,
    )
    worker._open_recorder = lambda: recorder  # type: ignore[method-assign]

    worker.run()

    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM audio_chunks WHERE session_id = ?",
            (session_id,),
        ).fetchone()["count"]
    assert count == 1


def test_pause_closes_and_reopens_audio_recorder(tmp_path: Path) -> None:
    database = Database(tmp_path / "pause-reopen.sqlite3")
    session_id = database.create_session("暂停重开音频", "2026-08-04T00:00:00+00:00")
    stop_event = threading.Event()
    pause_event = threading.Event()
    first_exit = threading.Event()

    class LifecycleRecorder(FakeAudioRecorder):
        def __init__(self) -> None:
            super().__init__(mode="silence")
            self.enters = 0
            self.exits = 0

        def __enter__(self):
            self.enters += 1
            return self

        def __exit__(self, *_args) -> None:
            self.exits += 1
            if self.exits == 1:
                first_exit.set()

        def record(self, numframes: int) -> np.ndarray:
            block = super().record(numframes)
            if self.enters == 1:
                pause_event.set()
            else:
                stop_event.set()
            return block

    recorder = LifecycleRecorder()
    worker = AudioCaptureWorker(
        database=database,
        session_id=session_id,
        clock=AdvancingClock(),  # type: ignore[arg-type]
        source="microphone",
        device=None,
        device_catalog=object(),  # type: ignore[arg-type]
        output_dir=tmp_path / "audio",
        stop_event=stop_event,
        pause_event=pause_event,
        chunk_queue=queue.Queue(),
        sample_rate=10,
        storage_sample_rate=10,
        chunk_s=0.1,
    )
    worker._open_recorder = lambda: recorder  # type: ignore[method-assign]

    thread = threading.Thread(target=worker.run)
    thread.start()
    assert first_exit.wait(timeout=2)
    pause_event.clear()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert (recorder.enters, recorder.exits) == (2, 2)
    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM audio_chunks WHERE session_id = ?",
            (session_id,),
        ).fetchone()["count"]
    assert count == 2


def test_reopen_failure_is_classified_as_device_unavailable(tmp_path: Path) -> None:
    database = Database(tmp_path / "pause-device-loss.sqlite3")
    session_id = database.create_session("暂停后设备消失", "2026-08-04T00:00:00+00:00")
    stop_event = threading.Event()
    pause_event = threading.Event()
    first_exit = threading.Event()
    failures: list[tuple[str, str, int, int, str]] = []

    class ReopenFailureRecorder(FakeAudioRecorder):
        def __init__(self) -> None:
            super().__init__(mode="silence")
            self.enters = 0

        def __enter__(self):
            self.enters += 1
            if self.enters == 2:
                raise RuntimeError("设备已拔出")
            return self

        def __exit__(self, *_args) -> None:
            first_exit.set()

        def record(self, numframes: int) -> np.ndarray:
            block = super().record(numframes)
            pause_event.set()
            return block

    recorder = ReopenFailureRecorder()
    worker = AudioCaptureWorker(
        database=database,
        session_id=session_id,
        clock=AdvancingClock(),  # type: ignore[arg-type]
        source="microphone",
        device=None,
        device_catalog=object(),  # type: ignore[arg-type]
        output_dir=tmp_path / "audio",
        stop_event=stop_event,
        pause_event=pause_event,
        chunk_queue=queue.Queue(),
        sample_rate=10,
        storage_sample_rate=10,
        chunk_s=0.1,
        on_failure=lambda *args: failures.append(args),
    )
    worker._open_recorder = lambda: recorder  # type: ignore[method-assign]

    thread = threading.Thread(target=worker.run)
    thread.start()
    assert first_exit.wait(timeout=2)
    pause_event.clear()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert failures and failures[0][1] == "device_unavailable"
