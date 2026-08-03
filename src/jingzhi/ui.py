from __future__ import annotations

import logging
import os
import re
import sys
import threading
from typing import ClassVar

from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QSize, Qt, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from jingzhi.application import JingzhiApplicationService, SessionTimeline
from jingzhi.config import Settings
from jingzhi.database import TimelineFrameRecord, TimelineTranscriptRecord
from jingzhi.rich_text import MarkdownDocument
from jingzhi.session import SessionManager

logger = logging.getLogger(__name__)


def motion_enabled() -> bool:
    if os.environ.get("JINGZHI_REDUCE_MOTION") == "1":
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        animations_enabled = ctypes.c_int()
        success = ctypes.windll.user32.SystemParametersInfoW(  # type: ignore[attr-defined]
            0x1042, 0, ctypes.byref(animations_enabled), 0
        )
    except (AttributeError, OSError):
        return True
    return not success or bool(animations_enabled.value)


class EvidenceButton(QPushButton):
    HOVER_DURATION_MS = 145

    def __init__(self, text: str, *, animations: bool) -> None:
        super().__init__(text)
        self._animations = animations
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.92 if animations else 1.0)
        self.setGraphicsEffect(self._opacity)
        self._hover_animation = QPropertyAnimation(self._opacity, b"opacity", self)
        self._hover_animation.setDuration(self.HOVER_DURATION_MS)
        self._hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def enterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._animate_hover(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._animate_hover(0.92)
        super().leaveEvent(event)

    def _animate_hover(self, target: float) -> None:
        if not self._animations:
            self._opacity.setOpacity(1.0)
            return
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._opacity.opacity())
        self._hover_animation.setEndValue(target)
        self._hover_animation.start()

APP_STYLE = """
QWidget {
    background: #111719;
    color: #e7ece9;
    font-family: "Microsoft YaHei UI", "Segoe UI";
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
QFrame#libraryPanel { background: #0b1112; border-right: 1px solid #273235; }
QFrame#detailPanel { background: #11191a; border-left: 1px solid #273235; }
QListWidget#sessionLibrary {
    background: transparent; border: none; outline: none; padding: 2px;
}
QListWidget#sessionLibrary::item {
    min-height: 52px; padding: 7px 9px; border-radius: 6px; color: #b7c4bf;
}
QListWidget#sessionLibrary::item:hover { background: #151f20; }
QListWidget#sessionLibrary::item:selected { background: #173329; color: #dcf5eb; }
QFrame#timelinePanel { background: #0e1516; border: 1px solid #273235; border-radius: 8px; }
QFrame[timelineTrack="true"] { background: #121b1c; border-top: 1px solid #243032; }
QLabel#trackLabel { color: #71827c; font-size: 10px; font-weight: 700; }
QPushButton[zoom="true"] { min-height: 25px; padding: 0 9px; font-size: 10px; }
QPushButton[zoom="true"]:checked { color: #081612; background: #79d3b4; border-color: #79d3b4; }
QPushButton[keyframe="true"] {
    min-width: 132px; max-width: 132px; min-height: 88px; padding: 5px;
    background: #edf0e9; color: #17201d; border: 1px solid #43514d;
    text-align: left; font-size: 9px;
}
QPushButton[keyframe="true"]:hover { border-color: #79d3b4; }
QPushButton[keyframe="true"][cited="true"] { border: 2px solid #79d3b4; }
QPushButton[keyframe="true"][selected="true"] { border: 2px solid #e7b36a; }
QPushButton[keyframe="true"][cited="true"][selected="true"] {
    border: 2px solid #e7b36a; background: #dff5e9;
}
QPushButton[transcript="true"] {
    min-width: 190px; max-width: 240px; min-height: 48px; padding: 6px 9px;
    background: #182221; color: #bdcac5; border: 1px solid #33413e;
    text-align: left; font-size: 9px;
}
QPushButton[transcript="true"][cited="true"] { border: 2px solid #79d3b4; }
QPushButton[transcript="true"][selected="true"] { border: 2px solid #e7b36a; }
QPushButton[transcript="true"][cited="true"][selected="true"] {
    border: 2px solid #e7b36a; background: #17352d;
}
QPushButton[correctionState="corrected"] { color: #9ee7ca; }
QPushButton[correctionState="pending"] { color: #ffd59a; }
QLabel#transcriptChip { background: #182221; border: 1px solid #33413e; padding: 7px; }
QLabel#eventChip { color: #a7b5b0; background: #171f20; padding: 7px; }
QFrame#recordingCapsule {
    background: #111817; border: 1px solid #604927; border-radius: 9px;
}
QLabel#evidenceImage { background: #edeae1; border: 1px solid #374442; }
QLabel#evidenceMetadata { color: #92a29c; font-size: 11px; }
QLabel#emptyState { color: #71827c; font-size: 12px; }
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
    ZOOM_WINDOWS: ClassVar[dict[str, int | None]] = {
        "whole": None,
        "5-minutes": 5 * 60_000,
        "1-minute": 60_000,
        "seconds": 10_000,
    }

    def __init__(
        self,
        settings: Settings,
        *,
        service: JingzhiApplicationService | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("境织")
        self.setMinimumSize(1080, 640)
        self.resize(1280, 720)
        self.bridge = UiBridge()
        if service is None:
            manager = SessionManager(
                settings,
                on_segment=lambda start, end, source, text: self.bridge.segment.emit(
                    start, end, source, text
                ),
                on_error=self.bridge.worker_warning.emit,
            )
            service = JingzhiApplicationService(manager.database, recorder=manager)
        self.service = service
        self.manager = service.recorder
        self.settings = settings
        self._selected_session_id: str | None = None
        self._selected_frame: TimelineFrameRecord | None = None
        self._timeline: SessionTimeline | None = None
        self._zoom_key = "whole"
        self._window_start_ms = 0
        self._paused = False
        self._animations_enabled = motion_enabled()
        self._build_ui()
        self._connect_signals()
        self.setStyleSheet(APP_STYLE)
        self._refresh_sessions()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        capsule_row = QHBoxLayout()
        capsule_row.setContentsMargins(220, 8, 280, 6)
        capsule_row.addStretch(1)
        capsule = QFrame()
        capsule.setObjectName("recordingCapsule")
        capsule_layout = QHBoxLayout(capsule)
        capsule_layout.setContentsMargins(10, 6, 8, 6)
        capsule_layout.setSpacing(7)
        self.status = QLabel("空闲")
        self.status.setObjectName("statusPill")
        self.status.setProperty("state", "idle")
        self.title_input = QLineEdit("新会话")
        self.title_input.setPlaceholderText("会话标题")
        self.title_input.setMaximumWidth(150)
        self.system_audio_check = QCheckBox("系统声音")
        self.system_audio_check.setChecked(self.settings.capture_system_audio)
        self.microphone_check = QCheckBox("麦克风")
        self.microphone_check.setChecked(self.settings.capture_microphone)
        self.start_button = QPushButton("开始记录")
        self.start_button.setProperty("role", "primary")
        self.stop_button = QPushButton("结束")
        self.stop_button.setProperty("role", "danger")
        self.stop_button.setEnabled(False)
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(callable(getattr(self.manager, "pause", None)))
        if not self.pause_button.isEnabled():
            self.pause_button.setToolTip("当前采集适配器尚未提供暂停能力")
        self.capsule_ask_button = QPushButton("提问")
        capsule_layout.addWidget(self.status)
        capsule_layout.addWidget(self.title_input)
        capsule_layout.addWidget(self.system_audio_check)
        capsule_layout.addWidget(self.microphone_check)
        capsule_layout.addWidget(self.pause_button)
        capsule_layout.addWidget(self.capsule_ask_button)
        capsule_layout.addWidget(self.start_button)
        capsule_layout.addWidget(self.stop_button)
        capsule_row.addWidget(capsule)
        capsule_row.addStretch(1)
        layout.addLayout(capsule_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_library_panel())
        splitter.addWidget(self._build_workspace_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setSizes([220, 780, 280])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _build_library_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("libraryPanel")
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(230)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 10, 12)
        title = QLabel("境织")
        title.setObjectName("appTitle")
        subtitle = QLabel("本地会话库")
        subtitle.setObjectName("subtitle")
        self.session_library = QListWidget()
        self.session_library.setObjectName("sessionLibrary")
        self.session_library.setSpacing(3)
        self.session_library.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        panel_layout.addWidget(title)
        panel_layout.addWidget(subtitle)
        panel_layout.addSpacing(10)
        panel_layout.addWidget(self.session_library, 1)
        library_state = QLabel("会话与关键帧均保存在本机")
        library_state.setObjectName("hint")
        library_state.setWordWrap(True)
        panel_layout.addWidget(library_state)
        return panel

    def _build_workspace_panel(self) -> QWidget:
        panel = QFrame()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 8, 14, 12)
        panel_layout.setSpacing(8)

        header = QHBoxLayout()
        title_column = QVBoxLayout()
        self.workspace_breadcrumb = QLabel("会话 / 请选择会话")
        self.workspace_breadcrumb.setObjectName("hint")
        self.workspace_title = QLabel("关键帧时间线")
        self.workspace_title.setObjectName("appTitle")
        self.workspace_meta = QLabel("从左侧打开已有会话，沿统一时间线核对证据。")
        self.workspace_meta.setObjectName("subtitle")
        title_column.addWidget(self.workspace_breadcrumb)
        title_column.addWidget(self.workspace_title)
        title_column.addWidget(self.workspace_meta)
        header.addLayout(title_column, 1)
        self.summary_button = QPushButton("生成会话材料")
        self.summary_button.setProperty("role", "primary")
        self.provider_toggle_button = QPushButton("模型连接")
        self.provider_toggle_button.setProperty("role", "quiet")
        header.addWidget(self.provider_toggle_button, alignment=Qt.AlignmentFlag.AlignBottom)
        header.addWidget(self.summary_button, alignment=Qt.AlignmentFlag.AlignBottom)
        panel_layout.addLayout(header)

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
        panel_layout.addWidget(self.notice)

        panel_layout.addWidget(self._build_timeline_panel(), 3)
        panel_layout.addWidget(self._build_answer_panel(), 2)
        self.provider_group = self._build_provider_panel()
        self.provider_group.hide()
        panel_layout.addWidget(self.provider_group)
        return panel

    def _build_provider_panel(self) -> QWidget:
        provider_group = QGroupBox("模型连接")
        provider_layout = QGridLayout(provider_group)
        provider_layout.setHorizontalSpacing(10)
        provider_layout.setVerticalSpacing(8)
        provider_layout.setColumnStretch(1, 4)
        provider_layout.setColumnStretch(3, 2)

        self.base_url_input = QLineEdit(getattr(self.manager, "llm_base_url", ""))
        self.base_url_input.setPlaceholderText(
            "例如 https://provider.example/v1；官方 OpenAI 可留空"
        )
        self.base_url_input.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.api_mode_input = QComboBox()
        self.api_mode_input.addItem("Responses API", "responses")
        self.api_mode_input.addItem("Chat Completions", "chat_completions")
        mode_index = self.api_mode_input.findData(
            getattr(self.manager, "llm_api_mode", "responses")
        )
        self.api_mode_input.setCurrentIndex(max(0, mode_index))
        self.api_key_input = QLineEdit(getattr(self.manager, "llm_api_key", ""))
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("只保存在本次运行的内存中")
        self.model_input = QLineEdit(getattr(self.manager, "llm_model", ""))
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
        return provider_group

    def _build_timeline_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("timelinePanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(12, 7, 9, 7)
        timeline_title = QLabel("统一时间线")
        timeline_title.setObjectName("sectionTitle")
        self.timeline_range = QLabel("00:00 — 00:00")
        self.timeline_range.setObjectName("hint")
        toolbar.addWidget(timeline_title)
        toolbar.addWidget(self.timeline_range)
        toolbar.addStretch(1)
        self.zoom_group = QButtonGroup(self)
        self.zoom_group.setExclusive(True)
        for key, text in (
            ("whole", "整段"),
            ("5-minutes", "5 分钟"),
            ("1-minute", "1 分钟"),
            ("seconds", "秒级"),
        ):
            button = QPushButton(text)
            button.setObjectName(f"zoom-{key}")
            button.setProperty("zoom", True)
            button.setCheckable(True)
            button.setChecked(key == "whole")
            button.clicked.connect(lambda _checked=False, zoom_key=key: self._set_zoom(zoom_key))
            self.zoom_group.addButton(button)
            toolbar.addWidget(button)
        panel_layout.addLayout(toolbar)

        navigator_row = QHBoxLayout()
        navigator_row.setContentsMargins(12, 0, 10, 5)
        navigator_label = QLabel("窗口起点")
        navigator_label.setObjectName("trackLabel")
        self.timeline_navigator = QSlider(Qt.Orientation.Horizontal)
        self.timeline_navigator.setObjectName("timelineNavigator")
        self.timeline_navigator.setTracking(False)
        self.timeline_navigator.setEnabled(False)
        self.timeline_navigator.valueChanged.connect(self._navigate_timeline)
        navigator_row.addWidget(navigator_label)
        navigator_row.addWidget(self.timeline_navigator, 1)
        panel_layout.addLayout(navigator_row)

        self.keyframe_track, self.keyframe_content, self.keyframe_layout = self._make_track(
            "keyframeTrack", "关键帧", 106
        )
        panel_layout.addWidget(self.keyframe_track)

        self.transcript_track, self.transcript_content, self.transcript_layout = self._make_track(
            "transcriptTrack", "字幕", 68
        )
        panel_layout.addWidget(self.transcript_track)

        event_frame = QFrame()
        event_frame.setObjectName("eventTrack")
        event_frame.setProperty("timelineTrack", True)
        event_layout = QHBoxLayout(event_frame)
        event_layout.setContentsMargins(10, 4, 8, 4)
        event_label = QLabel("事件")
        event_label.setObjectName("trackLabel")
        event_label.setFixedWidth(48)
        self.event_text = QLabel("选择会话后显示会话边界与状态")
        self.event_text.setObjectName("eventChip")
        event_layout.addWidget(event_label)
        event_layout.addWidget(self.event_text, 1)
        panel_layout.addWidget(event_frame)
        return panel

    @staticmethod
    def _make_track(
        object_name: str, label_text: str, height: int
    ) -> tuple[QWidget, QWidget, QHBoxLayout]:
        frame = QFrame()
        frame.setObjectName(object_name)
        frame.setProperty("timelineTrack", True)
        frame.setMinimumHeight(height)
        track_layout = QHBoxLayout(frame)
        track_layout.setContentsMargins(10, 4, 8, 4)
        label = QLabel(label_text)
        label.setObjectName("trackLabel")
        label.setFixedWidth(48)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        track_layout.addWidget(label)
        track_layout.addWidget(scroll, 1)
        return frame, content, content_layout

    def _build_detail_panel(self) -> QWidget:
        panel = QFrame()
        self.detail_panel = panel
        panel.setObjectName("detailPanel")
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(300)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        heading = QLabel("证据详情")
        heading.setObjectName("sectionTitle")
        self.evidence_image = QLabel("选择关键帧查看大图")
        self.evidence_image.setObjectName("evidenceImage")
        self.evidence_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evidence_image.setMinimumHeight(180)
        self.evidence_image.setWordWrap(True)
        self.evidence_title = QLabel("尚未选择证据")
        self.evidence_title.setObjectName("appTitle")
        self.evidence_title.setWordWrap(True)
        self.evidence_metadata = QLabel("来源与会话相对时间将在这里显示。")
        self.evidence_metadata.setObjectName("evidenceMetadata")
        self.evidence_metadata.setWordWrap(True)
        self.evidence_version = QLabel()
        self.evidence_version.setObjectName("hint")
        self.evidence_version.setWordWrap(True)
        self.evidence_version.hide()
        panel_layout.addWidget(heading)
        panel_layout.addSpacing(6)
        panel_layout.addWidget(self.evidence_image)
        panel_layout.addWidget(self.evidence_title)
        panel_layout.addWidget(self.evidence_metadata)
        panel_layout.addWidget(self.evidence_version)
        panel_layout.addStretch(1)
        self._detail_opacity = QGraphicsOpacityEffect(panel)
        self._detail_opacity.setOpacity(1.0)
        panel.setGraphicsEffect(self._detail_opacity)
        self._detail_animation = QPropertyAnimation(self._detail_opacity, b"opacity", panel)
        self._detail_animation.setDuration(220)
        self._detail_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        return panel

    @staticmethod
    def _format_time(milliseconds: int) -> str:
        total_seconds = max(0, milliseconds // 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _refresh_sessions(self, select_session_id: str | None = None) -> None:
        selected_id = select_session_id or self._selected_session_id
        self.session_library.blockSignals(True)
        self.session_library.clear()
        selected_item: QListWidgetItem | None = None
        for session in self.service.list_sessions():
            state = "记录中" if session.status == "recording" else "已完成"
            item = QListWidgetItem(
                f"{session.title}\n{state} · {self._format_time(session.duration_ms)}"
                f" · {session.frame_count} 帧"
            )
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            item.setToolTip(session.started_at_utc)
            self.session_library.addItem(item)
            if session.id == selected_id:
                selected_item = item
        self.session_library.blockSignals(False)
        if selected_item is None and self.session_library.count():
            selected_item = self.session_library.item(0)
        if selected_item is not None:
            self.session_library.setCurrentItem(selected_item)
            self._open_session_item(selected_item)
        else:
            self._show_empty_timeline()

    def _open_session_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        session_id = str(item.data(Qt.ItemDataRole.UserRole))
        self._selected_session_id = session_id
        window_duration = self.ZOOM_WINDOWS[self._zoom_key]
        try:
            timeline = self.service.open_session(
                session_id,
                window_start_ms=self._window_start_ms,
                window_duration_ms=window_duration,
            )
        except Exception as exc:  # noqa: BLE001 - UI boundary reports persistence failures
            self._show_action_error(str(exc))
            return
        self._timeline = timeline
        self._selected_frame = None
        self._render_timeline(timeline)

    def _select_session_item(self, item: QListWidgetItem | None) -> None:
        self._window_start_ms = 0
        self._open_session_item(item)

    def _set_zoom(self, zoom_key: str) -> None:
        self._zoom_key = zoom_key
        self._window_start_ms = 0
        current = self.session_library.currentItem()
        if current is not None:
            self._open_session_item(current)

    @Slot(int)
    def _navigate_timeline(self, start_seconds: int) -> None:
        next_start_ms = start_seconds * 1000
        if next_start_ms == self._window_start_ms:
            return
        self._window_start_ms = next_start_ms
        current = self.session_library.currentItem()
        if current is not None:
            self._open_session_item(current)

    def _render_timeline(self, timeline: SessionTimeline) -> None:
        session = timeline.session
        self.workspace_breadcrumb.setText(f"会话 / {session.started_at_utc[:10]}")
        self.workspace_title.setText(session.title)
        state = "记录中" if session.status == "recording" else "已完成"
        self.workspace_meta.setText(
            f"{state} · {len(timeline.frames)} 张关键帧位于当前缩放窗口"
        )
        self.timeline_range.setText(
            f"{self._format_time(timeline.window_start_ms)} — "
            f"{self._format_time(min(timeline.window_end_ms, timeline.duration_ms))}"
        )
        window_duration = self.ZOOM_WINDOWS[self._zoom_key]
        maximum_start_ms = (
            0 if window_duration is None else max(0, timeline.duration_ms - window_duration)
        )
        self.timeline_navigator.blockSignals(True)
        self.timeline_navigator.setRange(0, maximum_start_ms // 1000)
        self.timeline_navigator.setValue(timeline.window_start_ms // 1000)
        self.timeline_navigator.setEnabled(maximum_start_ms > 0)
        self.timeline_navigator.blockSignals(False)

        self._clear_layout(self.keyframe_layout)
        if timeline.frames:
            cursor_ms = timeline.window_start_ms
            for frame in timeline.frames:
                gap_seconds = max(0, (frame.ts_ms - cursor_ms) // 1000)
                if gap_seconds:
                    self.keyframe_layout.addStretch(gap_seconds)
                button = EvidenceButton(
                    f"{self._format_time(frame.ts_ms)}\n{frame.source_id}",
                    animations=self._animations_enabled,
                )
                button.setObjectName(f"keyframe-{frame.id}")
                button.setProperty("keyframe", True)
                button.setProperty("selected", False)
                button.setProperty("cited", frame.id in timeline.answer_frame_ids)
                button.setToolTip(
                    f"关键帧 #{frame.id} · {frame.source_id} · "
                    f"{self._format_time(frame.ts_ms)}"
                )
                pixmap = QPixmap(str(frame.path))
                if not pixmap.isNull():
                    thumbnail = pixmap.scaled(
                        QSize(82, 58),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    button.setIcon(QIcon(thumbnail))
                    button.setIconSize(QSize(82, 58))
                button.clicked.connect(
                    lambda _checked=False, selected_frame=frame: self._select_frame(selected_frame)
                )
                self.keyframe_layout.addWidget(button)
                cursor_ms = frame.ts_ms
            tail_seconds = max(1, (timeline.window_end_ms - cursor_ms) // 1000)
            self.keyframe_layout.addStretch(tail_seconds)
        else:
            empty = QLabel("当前缩放窗口内没有关键帧")
            empty.setObjectName("emptyState")
            self.keyframe_layout.addWidget(empty)
            self.keyframe_layout.addStretch(1)

        self._clear_layout(self.transcript_layout)
        if timeline.transcripts:
            cursor_ms = timeline.window_start_ms
            for transcript in timeline.transcripts:
                gap_seconds = max(0, (transcript.start_ms - cursor_ms) // 1000)
                if gap_seconds:
                    self.transcript_layout.addStretch(gap_seconds)
                state_labels = {"pending": "待校订", "corrected": "已校订"}
                state_label = state_labels.get(transcript.correction_state or "")
                prefix = f"{state_label} · " if state_label else ""
                button = EvidenceButton(
                    f"{prefix}{self._format_time(transcript.start_ms)}–"
                    f"{self._format_time(transcript.end_ms)}\n{transcript.text}",
                    animations=self._animations_enabled,
                )
                button.setObjectName(f"transcript-{transcript.id}")
                button.setProperty("transcript", True)
                button.setProperty("selected", False)
                button.setProperty(
                    "cited", transcript.id in timeline.answer_transcript_ids
                )
                button.setProperty("correctionState", transcript.correction_state or "")
                button.setToolTip(f"{transcript.source} · {transcript.text}")
                button.clicked.connect(
                    lambda _checked=False, item=transcript: self._select_transcript(item)
                )
                self.transcript_layout.addWidget(button)
                cursor_ms = transcript.end_ms
            tail_seconds = max(1, (timeline.window_end_ms - cursor_ms) // 1000)
            self.transcript_layout.addStretch(tail_seconds)
        else:
            empty = QLabel("当前缩放窗口内没有字幕片段")
            empty.setObjectName("emptyState")
            self.transcript_layout.addWidget(empty)
            self.transcript_layout.addStretch(1)

        question_events = "  ·  ".join(
            f"Q {self._format_time(question.asked_at_ms)} {question.question}"
            for question in timeline.questions
        )
        self.event_text.setText(
            f"会话开始 00:00  ·  当前窗口 "
            f"{self._format_time(timeline.window_start_ms)}–"
            f"{self._format_time(min(timeline.window_end_ms, timeline.duration_ms))}  ·  "
            f"{question_events or '无问题锚点'}  ·  {state}"
        )
        self.evidence_image.clear()
        self.evidence_image.setText("选择关键帧查看大图")
        self.evidence_title.setText("尚未选择证据")
        self.evidence_metadata.setText("来源与会话相对时间将在这里显示。")
        self.evidence_version.hide()
        self._animate_detail_change()

    @staticmethod
    def _clear_layout(layout: QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    @Slot()
    def _select_frame(self, frame: TimelineFrameRecord) -> None:
        self._selected_frame = frame
        self._set_selected_evidence_button(f"keyframe-{frame.id}")
        self.evidence_image.setStyleSheet("background: #edeae1; color: #17201d;")
        pixmap = QPixmap(str(frame.path))
        if pixmap.isNull():
            self.evidence_image.setText("关键帧文件不可读取")
        else:
            self.evidence_image.setPixmap(
                pixmap.scaled(
                    self.evidence_image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.evidence_title.setText(f"关键帧 #{frame.id}")
        self.evidence_metadata.setText(
            f"来源：{frame.source_id}\n"
            f"会话时间：{self._format_time(frame.ts_ms)}\n"
            f"尺寸：{frame.width} × {frame.height}\n"
            f"稳定 ID：{frame.id}"
        )
        self.evidence_version.hide()
        self._animate_detail_change()

    @Slot()
    def _select_transcript(self, transcript: TimelineTranscriptRecord) -> None:
        self._selected_frame = None
        self._set_selected_evidence_button(f"transcript-{transcript.id}")
        self.evidence_image.setPixmap(QPixmap())
        self.evidence_image.setStyleSheet(
            "background: #172221; color: #dce8e3; padding: 18px; font-size: 15px;"
        )
        self.evidence_image.setText(f"“{transcript.text}”")
        self.evidence_title.setText(f"字幕片段 #{transcript.id}")
        self.evidence_metadata.setText(
            f"来源：{transcript.source}\n"
            f"会话时间：{self._format_time(transcript.start_ms)}–"
            f"{self._format_time(transcript.end_ms)}\n"
            f"稳定 ID：{transcript.id}"
        )
        state_labels = {"pending": "待校订", "corrected": "已校订"}
        state_label = state_labels.get(transcript.correction_state or "")
        if state_label:
            self.evidence_version.setText(
                f"字幕校订已启用 · 当前状态：{state_label}\nWhisper 原文保留为独立版本。"
            )
            self.evidence_version.show()
        else:
            self.evidence_version.hide()
        self._animate_detail_change()

    def _set_selected_evidence_button(self, object_name: str) -> None:
        for button in self.findChildren(QPushButton):
            if not (
                button.objectName().startswith("keyframe-")
                or button.objectName().startswith("transcript-")
            ):
                continue
            button.setProperty("selected", button.objectName() == object_name)
            button.style().unpolish(button)
            button.style().polish(button)

    def _animate_detail_change(self) -> None:
        if not self._animations_enabled:
            self._detail_opacity.setOpacity(1.0)
            return
        self._detail_animation.stop()
        self._detail_animation.setStartValue(0.72)
        self._detail_animation.setEndValue(1.0)
        self._detail_animation.start()

    def _show_empty_timeline(self) -> None:
        self.workspace_breadcrumb.setText("会话 / 暂无会话")
        self.workspace_title.setText("关键帧时间线")
        self.workspace_meta.setText("开始一段会话后，关键帧会出现在这里。")
        self.timeline_range.setText("00:00 — 00:00")
        self._clear_layout(self.keyframe_layout)
        empty = QLabel("暂无可浏览的会话")
        empty.setObjectName("emptyState")
        self.keyframe_layout.addWidget(empty)
        self.keyframe_layout.addStretch(1)
        self._clear_layout(self.transcript_layout)
        transcript_empty = QLabel("暂无字幕片段")
        transcript_empty.setObjectName("emptyState")
        self.transcript_layout.addWidget(transcript_empty)
        self.transcript_layout.addStretch(1)
        self.timeline_navigator.setRange(0, 0)
        self.timeline_navigator.setEnabled(False)
        self.event_text.setText("尚未开始会话")

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
        self.session_library.currentItemChanged.connect(
            lambda current, _previous: self._select_session_item(current)
        )
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.capsule_ask_button.clicked.connect(self._focus_question)
        self.ask_button.clicked.connect(self._ask)
        self.question.returnPressed.connect(self._ask)
        self.summary_button.clicked.connect(self._summarize)
        self.test_provider_button.clicked.connect(self._test_provider)
        self.save_provider_button.clicked.connect(self._save_provider)
        self.output_source_button.clicked.connect(self._toggle_output_source)
        self.copy_output_button.clicked.connect(self._copy_output_source)
        self.provider_toggle_button.clicked.connect(
            lambda: self.provider_group.setVisible(not self.provider_group.isVisible())
        )
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
            session_id = self.service.start_session(
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
        self.pause_button.setEnabled(callable(getattr(self.manager, "pause", None)))
        self.system_audio_check.setEnabled(False)
        self.microphone_check.setEnabled(False)
        self._refresh_sessions(session_id)

    @Slot()
    def _focus_question(self) -> None:
        self.question.setFocus(Qt.FocusReason.ShortcutFocusReason)

    @Slot()
    def _toggle_pause(self) -> None:
        method_name = "resume" if self._paused else "pause"
        method = getattr(self.manager, method_name, None)
        if not callable(method):
            return
        try:
            method()
        except Exception as exc:  # noqa: BLE001 - UI boundary surfaces adapter failures
            self._show_action_error(str(exc))
            return
        self._paused = not self._paused
        self.pause_button.setText("继续" if self._paused else "暂停")
        self._set_status("已暂停" if self._paused else "记录中", "recording")

    @Slot()
    def _stop(self) -> None:
        self.stop_button.setEnabled(False)
        self._set_status("正在结束并处理剩余字幕…", "idle")

        def work() -> None:
            try:
                session_id = self.service.stop_session()
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
        del start_ms, source, text
        current = self.session_library.currentItem()
        if current is not None:
            self._open_session_item(current)

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
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self._paused = False
        self.system_audio_check.setEnabled(True)
        self.microphone_check.setEnabled(True)
        self._refresh_sessions(session_id)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        configure_provider = getattr(self.manager, "configure_provider", None)
        save_provider = getattr(self.manager, "save_provider", None)
        if callable(configure_provider) and callable(save_provider):
            try:
                configure_provider(
                    model=self.model_input.text(),
                    base_url=self.base_url_input.text(),
                    api_key=self.api_key_input.text(),
                    api_mode=str(self.api_mode_input.currentData()),
                )
                save_provider()
            except Exception:
                logger.exception("Could not save provider settings while closing")
        if self.service.is_recording:
            self.service.stop_session()
        event.accept()


def run_app(settings: Settings) -> int:
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    window = MainWindow(settings)
    window.show()
    return application.exec()
