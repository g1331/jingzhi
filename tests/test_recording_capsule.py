from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from jingzhi.application import JingzhiApplicationService
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.ui import MainWindow, RecordingCapsule, RecordingCapsulePositionStore


class MutableRecorder:
    def __init__(self) -> None:
        self.is_recording = False

    def start(self, _title: str, **_kwargs: object) -> str:
        self.is_recording = True
        return "recording-session"

    def stop(self) -> str:
        self.is_recording = False
        return "recording-session"


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_floating_capsule_uses_independent_top_level_flags(tmp_path) -> None:
    _application()
    capsule = RecordingCapsule(
        default_system_audio_enabled=True,
        default_microphone_enabled=True,
        pause_enabled=False,
        floating=True,
        position_store=RecordingCapsulePositionStore(tmp_path),
    )

    flags = capsule.windowFlags()
    assert capsule.isWindow()
    assert flags & Qt.WindowType.Tool
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert capsule.parentWidget() is None
    assert capsule.hide_button is not None
    assert capsule.drag_handle is not None

    capsule.shutdown()


def test_capsule_close_hides_without_stopping_recording(tmp_path) -> None:
    application = _application()
    recorder = MutableRecorder()
    service = JingzhiApplicationService(Database(tmp_path / "capsule.sqlite3"), recorder=recorder)
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.show()
    application.processEvents()
    recorder.is_recording = True
    window._show_recording_capsule()
    application.processEvents()

    window.capsule.close()
    application.processEvents()

    assert recorder.is_recording
    assert not window.capsule.isVisible()

    recorder.is_recording = False
    window.close()
    application.processEvents()


def test_capsule_position_is_clamped_and_restored(tmp_path) -> None:
    application = _application()
    store = RecordingCapsulePositionStore(tmp_path)
    store.save(QPoint(100_000, 100_000))
    capsule = RecordingCapsule(
        default_system_audio_enabled=True,
        default_microphone_enabled=False,
        pause_enabled=False,
        floating=True,
        position_store=store,
    )
    capsule.show()
    application.processEvents()

    screen = application.primaryScreen()
    assert screen is not None
    available = screen.availableGeometry()
    assert (
        available.left()
        <= capsule.x()
        <= max(available.left(), available.right() - capsule.width() + 1)
    )
    assert (
        available.top()
        <= capsule.y()
        <= max(available.top(), available.bottom() - capsule.height() + 1)
    )

    capsule.move(40, 40)
    capsule.close()
    application.processEvents()
    assert store.load() == (40, 40)

    restored = RecordingCapsule(
        default_system_audio_enabled=True,
        default_microphone_enabled=False,
        pause_enabled=False,
        floating=True,
        position_store=store,
    )
    restored.show()
    application.processEvents()
    assert restored.pos() == QPoint(40, 40)

    restored.shutdown()
    application.processEvents()


def test_main_window_close_does_not_stop_active_recording(tmp_path) -> None:
    application = _application()
    recorder = MutableRecorder()
    service = JingzhiApplicationService(Database(tmp_path / "close.sqlite3"), recorder=recorder)
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.show()
    application.processEvents()
    recorder.is_recording = True
    window._show_recording_capsule()

    window.close()
    application.processEvents()

    assert recorder.is_recording
    assert window.isVisible()
    assert window.capsule.isVisible()

    recorder.is_recording = False
    window.close()
    application.processEvents()
