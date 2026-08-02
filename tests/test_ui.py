import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from jingzhi.config import Settings
from jingzhi.ui import MainWindow


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
