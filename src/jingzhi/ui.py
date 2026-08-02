from __future__ import annotations

import logging
import re
import threading

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from jingzhi.config import Settings
from jingzhi.rich_text import MarkdownDocument
from jingzhi.session import SessionManager

logger = logging.getLogger(__name__)

APP_STYLE = """
QWidget {
    background: #111719;
    color: #e7ece9;
    font-size: 13px;
}
QMainWindow { background: #0b1012; }
QLabel#appTitle {
    color: #f2f7f4;
    font-size: 24px;
    font-weight: 700;
}
QLabel#subtitle { color: #8fa09a; font-size: 12px; }
QLabel#sectionTitle { color: #b9c8c2; font-weight: 600; }
QLabel#hint { color: #71827c; font-size: 11px; }
QLabel#statusPill {
    color: #b7c4bf;
    background: #1b2426;
    border: 1px solid #2a3638;
    border-radius: 10px;
    padding: 4px 10px;
}
QLabel#statusPill[state="recording"] {
    color: #ffd59a;
    background: #302416;
    border-color: #6d4d24;
}
QLabel#statusPill[state="success"] {
    color: #9ee7ca;
    background: #142a23;
    border-color: #295b4b;
}
QLabel#statusPill[state="error"] {
    color: #ffb4aa;
    background: #321d1c;
    border-color: #713a36;
}
QGroupBox {
    background: #141b1d;
    border: 1px solid #273235;
    border-radius: 10px;
    margin-top: 12px;
    padding: 14px 12px 10px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #aab9b4;
}
QLineEdit, QComboBox, QPlainTextEdit {
    background: #0d1315;
    color: #eef3f0;
    border: 1px solid #2b373a;
    border-radius: 7px;
    selection-background-color: #2f806d;
}
QLineEdit, QComboBox { min-height: 32px; padding: 0 9px; }
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border-color: #51a88f; }
QLineEdit:disabled, QComboBox:disabled { color: #687671; background: #121719; }
QPlainTextEdit {
    padding: 10px;
    font-size: 12px;
}
QPushButton {
    min-height: 32px;
    padding: 0 14px;
    color: #dce6e2;
    background: #222d2f;
    border: 1px solid #334145;
    border-radius: 7px;
    font-weight: 600;
}
QPushButton:hover { background: #2a383a; border-color: #4b6064; }
QPushButton:pressed { background: #172022; }
QPushButton:disabled { color: #5e6a66; background: #171d1f; border-color: #242d2f; }
QPushButton[role="primary"] { color: #071612; background: #76cdb3; border-color: #76cdb3; }
QPushButton[role="primary"]:hover { background: #91ddc6; }
QPushButton[role="danger"] { color: #ffd4cf; background: #39201e; border-color: #6d3834; }
QPushButton[role="quiet"] { min-height: 27px; padding: 0 10px; font-size: 11px; }
QCheckBox { spacing: 7px; color: #b7c4bf; }
QFrame#notice {
    background: #2b2316;
    border: 1px solid #765522;
    border-radius: 8px;
}
QLabel#noticeText { color: #f2d49d; background: transparent; }
QPushButton#noticeClose {
    min-width: 26px; max-width: 26px; min-height: 26px; padding: 0;
    background: transparent; border: none; color: #d9bc86;
}
QFrame#contentPanel { background: #141b1d; border: 1px solid #273235; border-radius: 10px; }
QSplitter::handle { background: #1f292b; height: 6px; }
"""


class UiBridge(QObject):
    segment = Signal(int, int, str, str)
    worker_warning = Signal(str)
    action_error = Signal(str)
    answer = Signal(str)
    summary = Signal(str)
    stopped = Signal(str)
    provider_tested = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.setWindowTitle("境织")
        self.setMinimumSize(860, 640)
        self.resize(1120, 780)
        self.bridge = UiBridge()
        self.manager = SessionManager(
            settings,
            on_segment=lambda start, end, source, text: self.bridge.segment.emit(
                start, end, source, text
            ),
            on_error=self.bridge.worker_warning.emit,
        )
        self._build_ui()
        self._connect_signals()
        self.setStyleSheet(APP_STYLE)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title_column = QVBoxLayout()
        title = QLabel("境织")
        title.setObjectName("appTitle")
        subtitle = QLabel("让刚刚发生的一切，随时可问")
        subtitle.setObjectName("subtitle")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header.addLayout(title_column)
        header.addStretch(1)
        self.status = QLabel("空闲")
        self.status.setObjectName("statusPill")
        self.status.setProperty("state", "idle")
        header.addWidget(self.status, alignment=Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(header)

        provider_group = QGroupBox("模型连接")
        provider_layout = QGridLayout(provider_group)
        provider_layout.setHorizontalSpacing(10)
        provider_layout.setVerticalSpacing(8)
        provider_layout.setColumnStretch(1, 4)
        provider_layout.setColumnStretch(3, 2)

        self.base_url_input = QLineEdit(self.manager.llm_base_url)
        self.base_url_input.setPlaceholderText(
            "例如 https://provider.example/v1；官方 OpenAI 可留空"
        )
        self.base_url_input.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.api_mode_input = QComboBox()
        self.api_mode_input.addItem("Responses API", "responses")
        self.api_mode_input.addItem("Chat Completions", "chat_completions")
        mode_index = self.api_mode_input.findData(self.manager.llm_api_mode)
        self.api_mode_input.setCurrentIndex(max(0, mode_index))
        self.api_key_input = QLineEdit(self.manager.llm_api_key)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("只保存在本次运行的内存中")
        self.model_input = QLineEdit(self.manager.llm_model)
        self.model_input.setPlaceholderText("支持图片输入的模型名称")
        self.test_provider_button = QPushButton("测试连接")
        self.save_provider_button = QPushButton("保存配置")

        provider_layout.addWidget(QLabel("Base URL"), 0, 0)
        provider_layout.addWidget(self.base_url_input, 0, 1)
        provider_layout.addWidget(QLabel("接口类型"), 0, 2)
        provider_layout.addWidget(self.api_mode_input, 0, 3)
        provider_layout.addWidget(QLabel("API Key"), 1, 0)
        provider_layout.addWidget(self.api_key_input, 1, 1)
        provider_layout.addWidget(QLabel("模型"), 1, 2)
        provider_layout.addWidget(self.model_input, 1, 3)
        provider_buttons = QVBoxLayout()
        provider_buttons.setSpacing(6)
        provider_buttons.addWidget(self.test_provider_button)
        provider_buttons.addWidget(self.save_provider_button)
        provider_layout.addLayout(provider_buttons, 0, 4, 2, 1)
        hint = QLabel(
            "返回网页源码通常表示 Base URL 或接口类型不匹配。API Key 保存到 Windows 凭据管理器。"
        )
        hint.setObjectName("hint")
        provider_layout.addWidget(hint, 2, 1, 1, 3)
        layout.addWidget(provider_group)

        session_group = QGroupBox("上下文会话")
        session_layout = QHBoxLayout(session_group)
        self.title_input = QLineEdit("新会话")
        self.title_input.setPlaceholderText("为这次记录命名")
        self.system_audio_check = QCheckBox("系统声音")
        self.system_audio_check.setChecked(self.manager.settings.capture_system_audio)
        self.microphone_check = QCheckBox("麦克风")
        self.microphone_check.setChecked(self.manager.settings.capture_microphone)
        self.start_button = QPushButton("开始记录")
        self.start_button.setProperty("role", "primary")
        self.stop_button = QPushButton("结束")
        self.stop_button.setProperty("role", "danger")
        self.stop_button.setEnabled(False)
        self.summary_button = QPushButton("生成会话总结")
        session_layout.addWidget(QLabel("标题"))
        session_layout.addWidget(self.title_input, 1)
        session_layout.addWidget(self.system_audio_check)
        session_layout.addWidget(self.microphone_check)
        session_layout.addWidget(self.start_button)
        session_layout.addWidget(self.stop_button)
        session_layout.addWidget(self.summary_button)
        layout.addWidget(session_group)

        self.notice = QFrame()
        self.notice.setObjectName("notice")
        notice_layout = QHBoxLayout(self.notice)
        notice_layout.setContentsMargins(10, 6, 6, 6)
        self.notice_text = QLabel()
        self.notice_text.setObjectName("noticeText")
        self.notice_text.setTextFormat(Qt.TextFormat.PlainText)
        self.notice_text.setWordWrap(True)
        self.notice_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        notice_close = QPushButton("×")
        notice_close.setObjectName("noticeClose")
        notice_close.clicked.connect(self.notice.hide)
        notice_layout.addWidget(self.notice_text, 1)
        notice_layout.addWidget(notice_close, alignment=Qt.AlignmentFlag.AlignTop)
        self.notice.hide()
        layout.addWidget(self.notice)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_transcript_panel())
        splitter.addWidget(self._build_answer_panel())
        splitter.setSizes([390, 260])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _build_transcript_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("contentPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 10, 12, 12)
        heading = QLabel("实时字幕")
        heading.setObjectName("sectionTitle")
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("开始记录后，字幕会按时间出现在这里。")
        panel_layout.addWidget(heading)
        panel_layout.addWidget(self.transcript, 1)
        return panel

    def _build_answer_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("contentPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 10, 12, 12)
        question_row = QHBoxLayout()
        self.question = QLineEdit()
        self.question.setPlaceholderText("例如：刚才这个结论是怎么得到的？")
        self.ask_button = QPushButton("提问")
        self.ask_button.setProperty("role", "primary")
        question_row.addWidget(self.question, 1)
        question_row.addWidget(self.ask_button)
        answer_header = QHBoxLayout()
        heading = QLabel("回答与会话材料")
        heading.setObjectName("sectionTitle")
        self.output_source_button = QPushButton("查看原文")
        self.output_source_button.setProperty("role", "quiet")
        self.copy_output_button = QPushButton("复制原文")
        self.copy_output_button.setProperty("role", "quiet")
        answer_header.addWidget(heading)
        answer_header.addStretch(1)
        answer_header.addWidget(self.output_source_button)
        answer_header.addWidget(self.copy_output_button)
        self.output = MarkdownDocument()
        panel_layout.addLayout(question_row)
        panel_layout.addLayout(answer_header)
        panel_layout.addWidget(self.output, 1)
        return panel

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.ask_button.clicked.connect(self._ask)
        self.question.returnPressed.connect(self._ask)
        self.summary_button.clicked.connect(self._summarize)
        self.test_provider_button.clicked.connect(self._test_provider)
        self.save_provider_button.clicked.connect(self._save_provider)
        self.output_source_button.clicked.connect(self._toggle_output_source)
        self.copy_output_button.clicked.connect(self._copy_output_source)
        self.output.render_failed.connect(self._show_worker_warning)
        self.bridge.segment.connect(self._append_segment)
        self.bridge.worker_warning.connect(self._show_worker_warning)
        self.bridge.action_error.connect(self._show_action_error)
        self.bridge.answer.connect(self._show_answer)
        self.bridge.summary.connect(self._show_summary)
        self.bridge.stopped.connect(self._recording_stopped)
        self.bridge.provider_tested.connect(self._provider_tested)

    @staticmethod
    def _compact_message(message: str) -> str:
        compact = re.sub(r"\s+", " ", message).strip()
        if "<!DOCTYPE html" in compact or "<html" in compact.lower():
            return "服务端返回了网页源码。请检查 Base URL 和接口类型，然后使用“测试连接”。"
        return compact[:420] or "发生了未提供详细信息的错误，请查看 data/logs/app.log。"

    def _set_status(self, text: str, state: str) -> None:
        self.status.setText(text)
        self.status.setProperty("state", state)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    @staticmethod
    def _format_summary(result: dict) -> str:
        lines = ["# 会话总结", "", str(result.get("summary") or "暂无会话摘要。")]
        knowledge_points = result.get("knowledge_points")
        lines.extend(["", "## 知识点"])
        if isinstance(knowledge_points, list) and knowledge_points:
            for index, item in enumerate(knowledge_points, 1):
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or f"知识点 {index}"
                lines.extend(["", f"### {index}. {name}", "", str(item.get("explanation") or "")])
                if item.get("evidence_time_s") is not None:
                    lines.extend(["", f"> 字幕证据：{item['evidence_time_s']} 秒附近"])
        else:
            lines.extend(["", "暂未提取到明确的知识点。"])

        mistakes = result.get("mistakes")
        lines.extend(["", "## 疑问与错题"])
        if isinstance(mistakes, list) and mistakes:
            for index, item in enumerate(mistakes, 1):
                if not isinstance(item, dict):
                    continue
                issue = item.get("issue") or f"问题 {index}"
                lines.extend(
                    [
                        "",
                        f"### {index}. {issue}",
                        "",
                        f"**订正：** {item.get('correction') or '暂无订正内容。'}",
                    ]
                )
                metadata = []
                if item.get("evidence_time_s") is not None:
                    metadata.append(f"字幕证据：{item['evidence_time_s']} 秒附近")
                if item.get("confidence"):
                    metadata.append(f"置信度：{item['confidence']}")
                if metadata:
                    lines.extend(["", "> " + "；".join(metadata)])
        else:
            lines.extend(["", "本次字幕中未确认到可提取的错题或错误。"])
        return "\n".join(lines)

    @Slot()
    def _toggle_output_source(self) -> None:
        show_source = not self.output.source_visible()
        self.output.set_source_visible(show_source)
        self.output_source_button.setText("查看渲染" if show_source else "查看原文")

    @Slot()
    def _copy_output_source(self) -> None:
        self.output.copy_source()
        self._set_status("回答原文已复制", "success")

    @Slot()
    def _start(self) -> None:
        try:
            session_id = self.manager.start(
                self.title_input.text(),
                capture_system_audio=self.system_audio_check.isChecked(),
                capture_microphone=self.microphone_check.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001 - UI boundary must surface worker failures
            self._show_action_error(str(exc))
            return
        self._set_status(f"记录中 · {session_id[:8]}", "recording")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.system_audio_check.setEnabled(False)
        self.microphone_check.setEnabled(False)

    @Slot()
    def _stop(self) -> None:
        self.stop_button.setEnabled(False)
        self._set_status("正在结束并处理剩余字幕…", "idle")

        def work() -> None:
            try:
                session_id = self.manager.stop()
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.action_error.emit(str(exc))
            else:
                self.bridge.stopped.emit(session_id or "")

        threading.Thread(target=work, name="stop-session", daemon=True).start()

    @Slot()
    def _test_provider(self) -> None:
        if not self._configure_provider_from_form():
            return
        self.test_provider_button.setEnabled(False)
        self._set_status("正在测试模型连接…", "idle")

        def work() -> None:
            try:
                result = self.manager.test_provider()
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.action_error.emit(str(exc))
            else:
                self.bridge.provider_tested.emit(result)

        threading.Thread(target=work, name="test-provider", daemon=True).start()

    @Slot()
    def _save_provider(self) -> None:
        if not self._configure_provider_from_form():
            return
        try:
            self.manager.save_provider()
        except Exception as exc:  # noqa: BLE001 - OS credential store boundary
            self._show_action_error(str(exc))
            return
        self._set_status("模型配置已保存", "success")

    @Slot()
    def _ask(self) -> None:
        question = self.question.text().strip()
        if not question or not self._configure_provider_from_form():
            return
        self.ask_button.setEnabled(False)
        self.output.set_markdown("_正在结合字幕和关键画面回答…_")

        def work() -> None:
            try:
                result = self.manager.answer(question)
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.action_error.emit(str(exc))
            else:
                self.bridge.answer.emit(result)

        threading.Thread(target=work, name="answer-question", daemon=True).start()

    @Slot()
    def _summarize(self) -> None:
        if not self._configure_provider_from_form():
            return
        self.summary_button.setEnabled(False)
        self.output.set_markdown("_正在生成会话总结…_")

        def work() -> None:
            try:
                result = self.manager.summarize()
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.action_error.emit(str(exc))
            else:
                self.bridge.summary.emit(self._format_summary(result))

        threading.Thread(target=work, name="summarize-session", daemon=True).start()

    def _configure_provider_from_form(self) -> bool:
        try:
            self.manager.configure_provider(
                model=self.model_input.text(),
                base_url=self.base_url_input.text(),
                api_key=self.api_key_input.text(),
                api_mode=str(self.api_mode_input.currentData()),
            )
        except Exception as exc:  # noqa: BLE001 - form validation boundary
            self._show_action_error(str(exc))
            return False
        return True

    @Slot(int, int, str, str)
    def _append_segment(self, start_ms: int, _end_ms: int, source: str, text: str) -> None:
        source_name = "系统" if source == "system" else "麦克风"
        self.transcript.appendPlainText(f"[{start_ms / 1000:7.1f}s][{source_name}] {text}")

    @Slot(str)
    def _show_worker_warning(self, message: str) -> None:
        self.notice_text.setText(self._compact_message(message))
        self.notice.show()

    @Slot(str)
    def _show_action_error(self, message: str) -> None:
        compact = self._compact_message(message)
        self._set_status("操作失败", "error")
        self.notice_text.setText(compact)
        self.notice.show()
        self.output.set_markdown(f"## 请求未完成\n\n{compact}")
        self.ask_button.setEnabled(True)
        self.summary_button.setEnabled(True)
        self.test_provider_button.setEnabled(True)

    @Slot(str)
    def _show_answer(self, answer: str) -> None:
        self.output.set_markdown(answer)
        self.ask_button.setEnabled(True)
        self._set_status("回答完成", "success")

    @Slot(str)
    def _show_summary(self, summary: str) -> None:
        self.output.set_markdown(summary)
        self.summary_button.setEnabled(True)
        self._set_status("总结已生成", "success")

    @Slot(str)
    def _provider_tested(self, result: str) -> None:
        self.test_provider_button.setEnabled(True)
        try:
            self.manager.save_provider()
        except Exception as exc:  # noqa: BLE001 - OS credential store boundary
            self._show_action_error(str(exc))
            return
        self._set_status("模型连接可用 · 配置已保存", "success")
        preview = self._compact_message(result)
        self.output.set_markdown(f"## 模型连接测试成功\n\n模型回复：{preview}")

    @Slot(str)
    def _recording_stopped(self, session_id: str) -> None:
        self._set_status(f"已结束 · {session_id[:8]}", "success")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.system_audio_check.setEnabled(True)
        self.microphone_check.setEnabled(True)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        try:
            self.manager.configure_provider(
                model=self.model_input.text(),
                base_url=self.base_url_input.text(),
                api_key=self.api_key_input.text(),
                api_mode=str(self.api_mode_input.currentData()),
            )
            self.manager.save_provider()
        except Exception:
            logger.exception("Could not save provider settings while closing")
        if self.manager.is_recording:
            self.manager.stop()
        event.accept()


def run_app(settings: Settings) -> int:
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    window = MainWindow(settings)
    window.show()
    return application.exec()
