from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QPlainTextEdit,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

ASSET_ROOT = Path(__file__).resolve().parent / "static" / "markdown"
RENDERER_PATH = ASSET_ROOT / "index.html"


class LocalOnlyPage(QWebEnginePage):
    """Web page that can load only the bundled renderer and its local assets."""

    def acceptNavigationRequest(
        self,
        url: QUrl,
        navigation_type: QWebEnginePage.NavigationType,
        is_main_frame: bool,
    ) -> bool:
        del navigation_type, is_main_frame
        if not url.isLocalFile():
            return False
        try:
            Path(url.toLocalFile()).resolve().relative_to(ASSET_ROOT)
        except ValueError:
            return False
        return True

    def createWindow(self, window_type: QWebEnginePage.WebWindowType) -> QWebEnginePage | None:
        del window_type
        return None


class MarkdownWebView(QWebEngineView):
    render_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ready = False
        self._pending_source = ""

        page = LocalOnlyPage(self)
        self.setPage(page)
        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False
        )
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, False)
        self.loadFinished.connect(self._on_loaded)
        self.setUrl(QUrl.fromLocalFile(str(RENDERER_PATH)))

    def set_markdown(self, source: str) -> None:
        self._pending_source = source
        if self._ready:
            self._render_pending()

    def _on_loaded(self, succeeded: bool) -> None:
        self._ready = succeeded
        if not succeeded:
            self.render_failed.emit("富文本渲染器加载失败，可切换到“原文”查看回答。")
            return
        self._render_pending()

    def _render_pending(self) -> None:
        encoded = json.dumps(self._pending_source, ensure_ascii=False)
        self.page().runJavaScript(f"window.renderMarkdown({encoded});")


class MarkdownDocument(QWidget):
    """A rendered Markdown document with a raw-source fallback."""

    render_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = ""
        self.rendered = MarkdownWebView()
        self.rendered.render_failed.connect(self.render_failed)
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setPlaceholderText("回答原文会显示在这里。")

        self.stack = QStackedWidget()
        self.stack.addWidget(self.rendered)
        self.stack.addWidget(self.raw)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

    def set_markdown(self, source: str) -> None:
        self._source = source
        self.raw.setPlainText(source)
        self.rendered.set_markdown(source)

    def set_source_visible(self, visible: bool) -> None:
        self.stack.setCurrentWidget(self.raw if visible else self.rendered)

    def source_visible(self) -> bool:
        return self.stack.currentWidget() is self.raw

    def copy_source(self) -> None:
        QApplication.clipboard().setText(self._source)
