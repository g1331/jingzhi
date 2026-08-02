import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QListWidget, QPushButton, QWidget

from jingzhi.application import JingzhiApplicationService
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.ui import MainWindow


class NoHardwareRecorder:
    is_recording = False

    def start(self, title, **_kwargs):
        raise AssertionError(f"Unexpected hardware start for {title}")

    def stop(self):
        return None


def _frame_buttons(window: MainWindow) -> list[QPushButton]:
    return [
        button
        for button in window.findChildren(QPushButton)
        if button.objectName().startswith("keyframe-")
    ]


def test_html_warning_is_compacted_and_does_not_change_window_width(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(Settings(data_dir=tmp_path))
    initial_width = window.width()

    window._show_worker_warning("<!DOCTYPE html><html>" + "unbroken" * 2_000 + "</html>")
    application.processEvents()

    assert "网页源码" in window.notice_text.text()
    assert len(window.notice_text.text()) < 120
    assert window.width() == initial_width
    window.manager.save_provider = lambda: None
    window.close()


def test_session_selection_thumbnail_zoom_and_detail_switching(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("界面验收会话", "2026-08-02T09:00:00+00:00")
    other_id = database.create_session("较早会话", "2026-08-01T09:00:00+00:00")
    database.finish_session(other_id, "2026-08-01T09:01:00+00:00", "complete")
    for index, (ts_ms, source, color) in enumerate(
        [(15_000, "display:1", "white"), (320_000, "display:2", "navy")], start=1
    ):
        image_path = tmp_path / f"frame-{index}.webp"
        Image.new("RGB", (640, 360), color).save(image_path)
        database.add_frame(
            session_id,
            ts_ms,
            image_path,
            f"hash-{index}",
            (640, 360),
            source_id=source,
        )
    database.finish_session(session_id, "2026-08-02T09:06:00+00:00", "complete")
    service = JingzhiApplicationService(
        database,
        recorder=NoHardwareRecorder(),
        now=lambda: datetime(2026, 8, 2, 9, 6, tzinfo=UTC),
    )
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.resize(1280, 720)
    window.show()
    application.processEvents()

    library = window.findChild(QListWidget, "sessionLibrary")
    assert library is not None
    assert library.count() == 2
    matching_items = [
        library.item(index)
        for index in range(library.count())
        if library.item(index).data(Qt.ItemDataRole.UserRole) == session_id
    ]
    library.setCurrentItem(matching_items[0])
    application.processEvents()
    assert len(_frame_buttons(window)) == 2

    zoom = window.findChild(QPushButton, "zoom-1-minute")
    assert zoom is not None
    zoom.click()
    application.processEvents()
    assert zoom.isChecked()
    assert len(_frame_buttons(window)) == 1

    _frame_buttons(window)[0].click()
    application.processEvents()
    detail_image = window.findChild(QLabel, "evidenceImage")
    detail_metadata = window.findChild(QLabel, "evidenceMetadata")
    assert detail_image is not None and detail_image.pixmap() is not None
    assert detail_metadata is not None
    assert "display:1" in detail_metadata.text()
    assert "00:15" in detail_metadata.text()

    library_panel = window.findChild(QWidget, "libraryPanel")
    detail_panel = window.findChild(QWidget, "detailPanel")
    keyframe_track = window.findChild(QWidget, "keyframeTrack")
    transcript_track = window.findChild(QWidget, "transcriptTrack")
    event_track = window.findChild(QWidget, "eventTrack")
    assert library_panel is not None and 190 <= library_panel.width() <= 240
    assert detail_panel is not None and 250 <= detail_panel.width() <= 310
    assert all(
        widget is not None and widget.isVisible()
        for widget in (keyframe_track, transcript_track, event_track)
    )

    window.close()
