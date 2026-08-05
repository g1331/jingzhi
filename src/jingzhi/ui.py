from __future__ import annotations

import logging
import os
import re
import sys
import threading
import uuid
from dataclasses import replace
from difflib import SequenceMatcher
from typing import ClassVar

from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtTextToSpeech import QTextToSpeech
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from jingzhi.application import JingzhiApplicationService, SessionTimeline, present_answer
from jingzhi.capture.devices import (
    AudioDevice,
    DeviceCatalog,
    DeviceSnapshot,
    RecordingSelection,
    WindowsDeviceCatalog,
)
from jingzhi.config import Settings
from jingzhi.database import (
    AnswerEvidenceRecord,
    MaterialEvidenceRecord,
    SessionAnswerRecord,
    SessionMaterialVersionRecord,
    SessionNotificationKind,
    SessionRecord,
    SourceEventRecord,
    TimelineFrameRecord,
    TimelineTranscriptRecord,
)
from jingzhi.material_settings import MaterialGenerationMode
from jingzhi.model_roles import (
    ModelConnection,
    ModelFallback,
    ModelRole,
    ReasoningLevel,
    RoleName,
)
from jingzhi.provider_settings import SavedProviderSettings
from jingzhi.recording_settings import (
    RecordingPreferences,
    RecordingSettingsStore,
    estimate_storage_bytes,
    resolve_recording_selection,
)
from jingzhi.rich_text import MarkdownDocument
from jingzhi.session import SessionManager
from jingzhi.storage_ui import StorageSettingsDialog
from jingzhi.transcript_correction import CORRECTION_WINDOW_SECONDS
from jingzhi.whisper_ui import WhisperSettingsDialog

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


class QuestionInput(QLineEdit):
    focused = Signal()

    def focusInEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.focused.emit()
        super().focusInEvent(event)


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
QPushButton[correctionState="recognizing"] { color: #a9c7ff; }
QPushButton[correctionState="edited"] { color: #e7b36a; }
QLabel#transcriptChip { background: #182221; border: 1px solid #33413e; padding: 7px; }
QLabel#eventChip { color: #a7b5b0; background: #171f20; padding: 7px; }
QFrame#recordingCapsule {
    background: #111817; border: 1px solid #604927; border-radius: 9px;
}
QLabel#evidenceImage { background: #edeae1; border: 1px solid #374442; }
QLabel#evidenceMetadata { color: #92a29c; font-size: 11px; }
QLabel#answerEvidenceStatus {
    color: #9ee7ca; background: #142a23; border: 1px solid #295b4b;
    border-radius: 6px; padding: 5px 8px; font-size: 11px;
}
QLabel#answerEvidenceStatus[state="unavailable"] {
    color: #f2d49d; background: #2b2316; border-color: #765522;
}
QLabel#answerEvidenceStatus[state="insufficient"] {
    color: #f2d49d; background: #2b2316; border-color: #765522;
}
QLabel#questionNotes { color: #b9c8c2; background: #182221; border: 1px solid #33413e; padding: 7px; }

QLabel#emptyState { color: #71827c; font-size: 12px; }
"""


class UiBridge(QObject):
    segment = Signal(int, int, str, str)
    worker_warning = Signal(str)
    source_event = Signal(object)
    action_error = Signal(str)
    answer = Signal(int, str, str)

    voice_transcript = Signal(int, str)
    voice_error = Signal(int, str)
    summary = Signal(str)
    material = Signal(object)
    stopped = Signal(str)
    stop_failed = Signal(str)
    pause_finished = Signal(bool)
    provider_tested = Signal(str)
    maintenance_finished = Signal(object)
    maintenance_failed = Signal(str)


class RecordingConfirmationDialog(QDialog):
    snapshot_ready = Signal(object)
    snapshot_failed = Signal(str)
    level_ready = Signal(object)

    def __init__(
        self,
        catalog: DeviceCatalog,
        store: RecordingSettingsStore,
        *,
        screen_interval_s: float,
        audio_storage_rate: int,
        default_system_audio_enabled: bool = True,
        default_microphone_enabled: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("确认录制来源")
        self.resize(760, 620)
        self.catalog = catalog
        self.store = store
        self.screen_interval_s = screen_interval_s
        self.audio_storage_rate = audio_storage_rate
        self.preferences = store.load()
        if not store.path.is_file():
            self.preferences = RecordingPreferences(
                system_audio_enabled=default_system_audio_enabled,
                microphone_enabled=default_microphone_enabled,
            )
        self.snapshot = DeviceSnapshot((), (), ())
        self.display_checks: dict[str, QCheckBox] = {}
        self._display_widgets: list[QWidget] = []
        self._display_selection_changed = False
        self._audio_initialized = False
        self._refresh_in_progress = False
        self._level_in_progress = False

        root = QVBoxLayout(self)
        heading = QLabel("开始会话前确认来源")
        heading.setObjectName("sectionTitle")
        root.addWidget(heading)
        hint = QLabel(
            "打开此窗口会临时采样缩略图和麦克风电平，仅用于来源确认；"
            "取消后不保存。未选择显示器时按全部显示器处理。"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.device_error = QLabel()
        self.device_error.setWordWrap(True)
        self.device_error.setVisible(False)
        root.addWidget(self.device_error)

        display_scroll = QScrollArea()
        display_scroll.setWidgetResizable(True)
        display_scroll.setMinimumHeight(270)
        display_container = QWidget()
        self.display_grid = QGridLayout(display_container)
        display_scroll.setWidget(display_container)
        root.addWidget(display_scroll, 1)

        audio_group = QGroupBox("音频来源")
        audio_layout = QGridLayout(audio_group)
        self.system_audio_combo = QComboBox()
        self.microphone_combo = QComboBox()
        self.microphone_level = QProgressBar()
        self.microphone_level.setRange(0, 100)
        self.microphone_level.setFormat("麦克风电平 %p%")
        audio_layout.addWidget(QLabel("系统声音"), 0, 0)
        audio_layout.addWidget(self.system_audio_combo, 0, 1)
        audio_layout.addWidget(QLabel("麦克风"), 1, 0)
        audio_layout.addWidget(self.microphone_combo, 1, 1)
        audio_layout.addWidget(self.microphone_level, 2, 0, 1, 2)
        root.addWidget(audio_group)

        estimate_row = QHBoxLayout()
        estimate_row.addWidget(QLabel("预计时长"))
        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 24 * 60)
        self.duration_input.setSingleStep(15)
        self.duration_input.setSuffix(" 分钟")
        self.duration_input.setValue(self.preferences.estimated_duration_minutes)
        estimate_row.addWidget(self.duration_input)
        self.storage_estimate = QLabel()
        estimate_row.addWidget(self.storage_estimate, 1)
        root.addLayout(estimate_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认并开始")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.confirm_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.confirm_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.system_audio_combo.currentIndexChanged.connect(self._update_estimate)
        self.microphone_combo.currentIndexChanged.connect(self._update_level_and_estimate)
        self.duration_input.valueChanged.connect(self._update_estimate)
        self.snapshot_ready.connect(self._apply_snapshot)
        self.snapshot_failed.connect(self._show_snapshot_error)
        self.level_ready.connect(self._apply_microphone_level)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(1_000)
        self.refresh_timer.timeout.connect(self.refresh_devices)
        self.level_timer = QTimer(self)
        self.level_timer.setInterval(250)
        self.level_timer.timeout.connect(self._update_microphone_level)
        self.refresh_devices()
        self.refresh_timer.start()
        self.level_timer.start()

    def refresh_devices(self) -> None:
        if self._refresh_in_progress:
            return
        self._refresh_in_progress = True

        def sample() -> None:
            try:
                self.snapshot_ready.emit(self.catalog.snapshot())
            except Exception as exc:  # noqa: BLE001 - hardware enumeration boundary
                self.snapshot_failed.emit(str(exc))

        threading.Thread(target=sample, name="recording-device-preview", daemon=True).start()

    @Slot(object)
    def _apply_snapshot(self, snapshot: DeviceSnapshot) -> None:
        self._refresh_in_progress = False
        checked = {
            identifier: checkbox.isChecked() for identifier, checkbox in self.display_checks.items()
        }
        system_id = self.system_audio_combo.currentData()
        microphone_id = self.microphone_combo.currentData()
        self.device_error.setVisible(False)
        self.snapshot = snapshot
        resolved = resolve_recording_selection(self.preferences, snapshot)
        preferred_displays = {display.id for display in resolved.displays}

        for widget in self._display_widgets:
            self.display_grid.removeWidget(widget)
            widget.deleteLater()
        self._display_widgets.clear()
        self.display_checks.clear()
        for index, display in enumerate(snapshot.displays):
            panel = QGroupBox(display.name)
            panel_layout = QVBoxLayout(panel)
            preview = QLabel()
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(220, 124)
            pixmap = QPixmap.fromImage(ImageQt(display.preview))
            preview.setPixmap(
                pixmap.scaled(
                    QSize(260, 146),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            checkbox = QCheckBox("录制此显示器")
            if display.id in checked:
                checkbox.setChecked(checked[display.id])
            elif self._display_selection_changed:
                checkbox.setChecked(False)
            else:
                checkbox.setChecked(display.id in preferred_displays)
            checkbox.toggled.connect(self._display_toggled)
            panel_layout.addWidget(preview)
            panel_layout.addWidget(checkbox)
            self.display_grid.addWidget(panel, index // 2, index % 2)
            self._display_widgets.append(panel)
            self.display_checks[display.id] = checkbox
        if not snapshot.displays:
            empty = QLabel("当前未检测到显示器，请连接显示器后刷新。")
            self.display_grid.addWidget(empty, 0, 0)
            self._display_widgets.append(empty)

        preferred_system_id = (
            system_id
            if self._audio_initialized
            else (resolved.system_audio.id if resolved.system_audio is not None else None)
        )
        preferred_microphone_id = (
            microphone_id
            if self._audio_initialized
            else (resolved.microphone.id if resolved.microphone is not None else None)
        )
        self._populate_audio_combo(
            self.system_audio_combo, snapshot.system_audio, preferred_system_id
        )
        self._populate_audio_combo(
            self.microphone_combo, snapshot.microphones, preferred_microphone_id
        )
        self._audio_initialized = True
        self.confirm_button.setEnabled(bool(snapshot.displays))
        self._update_level_and_estimate()

    @Slot(str)
    def _show_snapshot_error(self, message: str) -> None:
        self._refresh_in_progress = False
        self.device_error.setText(f"设备刷新失败：{message}")
        self.device_error.setVisible(True)

    @staticmethod
    def _populate_audio_combo(
        combo: QComboBox, devices: tuple[AudioDevice, ...], selected_id: object
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("不录制", None)
        for device in devices:
            suffix = "（默认）" if device.is_default else ""
            combo.addItem(f"{device.name}{suffix}", device.id)
        selected_index = combo.findData(selected_id)
        if selected_index < 0 and selected_id is not None:
            fallback = next((device.id for device in devices if device.is_default), None)
            if fallback is None and devices:
                fallback = devices[0].id
            selected_index = combo.findData(fallback)
        combo.setCurrentIndex(max(selected_index, 0))
        combo.blockSignals(False)

    def _display_toggled(self) -> None:
        self._display_selection_changed = True
        self._update_estimate()

    def _audio_device(self, identifier: object, devices: tuple[AudioDevice, ...]):
        return next((device for device in devices if device.id == identifier), None)

    def _current_preferences(self) -> RecordingPreferences:
        return RecordingPreferences(
            display_ids=tuple(
                identifier
                for identifier, checkbox in self.display_checks.items()
                if checkbox.isChecked()
            ),
            system_audio_id=self.system_audio_combo.currentData(),
            microphone_id=self.microphone_combo.currentData(),
            system_audio_enabled=self.system_audio_combo.currentData() is not None,
            microphone_enabled=self.microphone_combo.currentData() is not None,
            estimated_duration_minutes=self.duration_input.value(),
        )

    def recording_selection(self) -> RecordingSelection:
        preferences = self._current_preferences()
        self.store.save(preferences)
        return RecordingSelection(
            display_ids=preferences.display_ids,
            system_audio_id=(
                preferences.system_audio_id if preferences.system_audio_enabled else None
            ),
            microphone_id=(preferences.microphone_id if preferences.microphone_enabled else None),
            estimated_duration_minutes=preferences.estimated_duration_minutes,
        )

    def _update_microphone_level(self) -> None:
        if self._level_in_progress:
            return
        device = self._audio_device(self.microphone_combo.currentData(), self.snapshot.microphones)
        if device is None:
            self.microphone_level.setValue(0)
            return
        self._level_in_progress = True

        def sample() -> None:
            self.level_ready.emit((device.id, self.catalog.microphone_level(device)))

        threading.Thread(target=sample, name="recording-microphone-level", daemon=True).start()

    @Slot(object)
    def _apply_microphone_level(self, result: tuple[str, float]) -> None:
        self._level_in_progress = False
        device_id, level = result
        if self.microphone_combo.currentData() == device_id:
            self.microphone_level.setValue(round(level * 100))

    def _update_level_and_estimate(self) -> None:
        self._update_microphone_level()
        self._update_estimate()

    def _update_estimate(self) -> None:
        selection = resolve_recording_selection(self._current_preferences(), self.snapshot)
        estimate = estimate_storage_bytes(
            selection,
            screen_interval_s=self.screen_interval_s,
            audio_storage_rate=self.audio_storage_rate,
        )
        if estimate >= 1024**3:
            formatted = f"{estimate / 1024**3:.1f} GiB"
        else:
            formatted = f"{estimate / 1024**2:.0f} MiB"
        self.storage_estimate.setText(f"预计占用（估算）：{formatted}")

    def done(self, result: int) -> None:
        self.refresh_timer.stop()
        self.level_timer.stop()
        super().done(result)


class MainWindow(QMainWindow):
    CORRECTION_STATE_LABELS: ClassVar[dict[str, str]] = {
        "recognizing": "识别中",
        "pending": "待校订",
        "corrected": "已校订",
        "edited": "用户已编辑",
    }
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
                on_source_event=self.bridge.source_event.emit,
            )
            service = JingzhiApplicationService(manager.database, recorder=manager)
        self.service = service
        self.manager = service.recorder
        if hasattr(self.manager, "on_source_event"):
            self.manager.on_source_event = self.bridge.source_event.emit
        self.settings = settings
        self._selected_session_id: str | None = None
        self._reanswer_question_id: int | None = None
        self._selected_answer_version_id: int | None = None
        self._answers_by_id: dict[int, SessionAnswerRecord] = {}
        self._selected_material_version_id: int | None = None
        self._materials_by_id: dict[int, SessionMaterialVersionRecord] = {}
        self._content_kind = "answer"
        self._material_generation_in_flight = False
        provider_settings = getattr(self.manager, "provider_settings", settings.provider_settings)
        self._provider_connections = list(provider_settings.connections)
        self._provider_roles = {role.name: role for role in provider_settings.roles}
        self._active_connection_index = 0

        self._selected_frame: TimelineFrameRecord | None = None
        self._selected_transcript: TimelineTranscriptRecord | None = None
        self._timeline: SessionTimeline | None = None
        self._zoom_key = "whole"
        self._window_start_ms = 0
        self._paused = False
        self._stop_in_flight = False
        self._question_active = False
        self._question_generation = 0
        self._active_question_id: int | None = None
        self._prompted_source_event_ids: set[int] = set()

        self._last_answer = ""
        self._speech: QTextToSpeech | None = None
        self._animations_enabled = motion_enabled()
        self._storage_dialog: StorageSettingsDialog | None = None
        self._session_sort_newest = True
        self._maintenance_thread: threading.Thread | None = None
        self._build_ui()
        self._connect_signals()
        self.setStyleSheet(APP_STYLE)
        self._maintenance_timer = QTimer(self)
        self._maintenance_timer.setInterval(60_000)
        self._maintenance_timer.timeout.connect(self._run_session_maintenance)
        self._maintenance_timer.start()
        self._recording_status_timer = QTimer(self)
        self._recording_status_timer.setInterval(500)
        self._recording_status_timer.timeout.connect(self._refresh_recording_status)
        self._recording_status_timer.start()
        self._refresh_recording_status()
        self._refresh_sessions()
        QTimer.singleShot(0, self._run_session_maintenance)
        self._whisper_dialog: WhisperSettingsDialog | None = None
        if (
            callable(getattr(self.manager, "configure_whisper", None))
            and not self.manager.whisper_settings.first_run_completed
        ):
            QTimer.singleShot(0, self._show_whisper_settings)

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
        self.session_search = QLineEdit()
        self.session_search.setObjectName("sessionSearch")
        self.session_search.setPlaceholderText("搜索标题或字幕")
        filter_row = QHBoxLayout()
        self.session_filter = QComboBox()
        self.session_filter.setObjectName("sessionFilter")
        for label, value in (
            ("全部", "all"),
            ("未完成", "unfinished"),
            ("已完成", "complete"),
            ("回收区", "trash"),
        ):
            self.session_filter.addItem(label, value)
        self.session_sort_button = QPushButton("新→旧")
        self.session_sort_button.setObjectName("sessionSort")
        filter_row.addWidget(self.session_filter, 1)
        filter_row.addWidget(self.session_sort_button)
        self.session_library = QListWidget()
        self.session_library.setObjectName("sessionLibrary")
        self.session_library.setSpacing(3)
        self.session_library.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        action_row = QHBoxLayout()
        self.session_pin_button = QPushButton("固定")
        self.session_delete_button = QPushButton("删除")
        self.session_restore_button = QPushButton("恢复")
        self.session_complete_button = QPushButton("标记完成")
        action_row.addWidget(self.session_pin_button)
        action_row.addWidget(self.session_delete_button)
        action_row.addWidget(self.session_restore_button)
        panel_layout.addWidget(title)
        panel_layout.addWidget(subtitle)
        panel_layout.addSpacing(6)
        panel_layout.addWidget(self.session_search)
        panel_layout.addLayout(filter_row)
        panel_layout.addWidget(self.session_library, 1)
        panel_layout.addLayout(action_row)
        panel_layout.addWidget(self.session_complete_button)
        library_state = QLabel("未固定会话保留 30 天 · 回收区保留 7 天")
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
        self.whisper_settings_button = QPushButton("本地 Whisper")
        self.whisper_settings_button.setProperty("role", "quiet")
        self.storage_settings_button = QPushButton("存储")
        self.storage_settings_button.setProperty("role", "quiet")
        header.addWidget(self.storage_settings_button, alignment=Qt.AlignmentFlag.AlignBottom)
        header.addWidget(self.whisper_settings_button, alignment=Qt.AlignmentFlag.AlignBottom)
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(300)
        scroll.setMaximumHeight(360)
        provider_group = QGroupBox("模型连接与角色")
        provider_group.setMinimumHeight(700)
        provider_layout = QGridLayout(provider_group)
        provider_layout.setHorizontalSpacing(8)
        provider_layout.setVerticalSpacing(7)
        provider_layout.setColumnStretch(1, 2)
        provider_layout.setColumnStretch(2, 3)

        self.connection_selector = QComboBox()
        for connection in self._provider_connections:
            self.connection_selector.addItem(connection.name, connection.id)
        self.add_connection_button = QPushButton("新增连接")
        self.remove_connection_button = QPushButton("删除连接")
        active = self._provider_connections[0]
        self.connection_name_input = QLineEdit(active.name)
        self.base_url_input = QLineEdit(active.base_url)
        self.base_url_input.setPlaceholderText("例如 https://api.example/v1；官方 OpenAI 可留空")
        self.base_url_input.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.api_mode_input = QComboBox()
        self.api_mode_input.addItem("Responses API", "responses")
        self.api_mode_input.addItem("Chat Completions", "chat_completions")
        self.api_mode_input.setCurrentIndex(max(0, self.api_mode_input.findData(active.api_mode)))
        self.api_key_input = QLineEdit(active.api_key)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("保存配置时写入 Windows 凭据管理器")
        self.test_provider_button = QPushButton("测试实用角色")
        self.save_provider_button = QPushButton("保存配置")
        self.correction_check = QCheckBox("启用字幕校订")
        self.correction_check.setChecked(self.settings.transcript_correction_enabled)
        self.correction_window_input = QComboBox()
        for seconds in CORRECTION_WINDOW_SECONDS:
            self.correction_window_input.addItem(f"{seconds} 秒", seconds)
        correction_index = self.correction_window_input.findData(
            self.settings.transcript_correction_window_seconds
        )
        self.correction_window_input.setCurrentIndex(max(0, correction_index))

        provider_layout.addWidget(QLabel("当前连接"), 0, 0)
        provider_layout.addWidget(self.connection_selector, 0, 1)
        provider_layout.addWidget(self.add_connection_button, 0, 2)
        provider_layout.addWidget(self.remove_connection_button, 0, 3)
        provider_layout.addWidget(QLabel("连接名称"), 1, 0)
        provider_layout.addWidget(self.connection_name_input, 1, 1)
        provider_layout.addWidget(QLabel("Base URL"), 1, 2)
        provider_layout.addWidget(self.base_url_input, 1, 3)
        provider_layout.addWidget(QLabel("API Key"), 2, 0)
        provider_layout.addWidget(self.api_key_input, 2, 1)
        provider_layout.addWidget(QLabel("接口类型"), 2, 2)
        provider_layout.addWidget(self.api_mode_input, 2, 3)
        provider_layout.addWidget(self.correction_check, 3, 0)
        provider_layout.addWidget(self.correction_window_input, 3, 1)
        provider_layout.addWidget(self.test_provider_button, 3, 2)
        provider_layout.addWidget(self.save_provider_button, 3, 3)

        headers = ("角色", "主连接", "模型", "思考档位")
        for column, text in enumerate(headers):
            label = QLabel(text)
            label.setObjectName("hint")
            provider_layout.addWidget(label, 4, column)

        self.role_connection_inputs: dict[RoleName, QComboBox] = {}
        self.role_model_inputs: dict[RoleName, QLineEdit] = {}
        self.role_reasoning_inputs: dict[RoleName, QComboBox] = {}
        self.role_fallback_connection_inputs: dict[RoleName, QComboBox] = {}
        self.role_fallback_model_inputs: dict[RoleName, QLineEdit] = {}
        self.role_cross_auth_checks: dict[RoleName, QCheckBox] = {}
        self.role_second_fallback_connection_inputs: dict[RoleName, QComboBox] = {}
        self.role_second_fallback_model_inputs: dict[RoleName, QLineEdit] = {}
        self.role_second_cross_auth_checks: dict[RoleName, QCheckBox] = {}
        role_labels = {
            RoleName.UTILITY: "实用",
            RoleName.TRANSCRIPT_CORRECTION: "字幕校订",
            RoleName.INSTANT_ANSWER: "即时问答",
            RoleName.DEEP_ANALYSIS: "深度分析",
        }
        for index, role_name in enumerate(RoleName):
            row = 5 + index * 3
            role = self._provider_roles[role_name]
            connection_input = QComboBox()
            for connection in self._provider_connections:
                connection_input.addItem(connection.name, connection.id)
            connection_input.setCurrentIndex(max(0, connection_input.findData(role.connection_id)))
            model_input = QLineEdit(role.model)
            reasoning_input = QComboBox()
            reasoning_input.addItem("快速", ReasoningLevel.FAST.value)
            reasoning_input.addItem("均衡", ReasoningLevel.BALANCED.value)
            reasoning_input.addItem("深入", ReasoningLevel.DEEP.value)
            reasoning_input.setCurrentIndex(max(0, reasoning_input.findData(role.reasoning.value)))
            fallback_controls: list[tuple[QComboBox, QLineEdit, QCheckBox]] = []
            for fallback in (*role.fallbacks, None, None)[:2]:
                fallback_connection = QComboBox()
                fallback_connection.addItem("无", "")
                for connection in self._provider_connections:
                    fallback_connection.addItem(connection.name, connection.id)
                if fallback is not None:
                    fallback_connection.setCurrentIndex(
                        max(0, fallback_connection.findData(fallback.connection_id))
                    )
                fallback_model = QLineEdit(fallback.model if fallback else "")
                cross_auth = QCheckBox("已授权")
                cross_auth.setChecked(bool(fallback and fallback.cross_connection_authorized))
                fallback_controls.append((fallback_connection, fallback_model, cross_auth))
            first_fallback, second_fallback = fallback_controls
            self.role_connection_inputs[role_name] = connection_input
            self.role_model_inputs[role_name] = model_input
            self.role_reasoning_inputs[role_name] = reasoning_input
            self.role_fallback_connection_inputs[role_name] = first_fallback[0]
            self.role_fallback_model_inputs[role_name] = first_fallback[1]
            self.role_cross_auth_checks[role_name] = first_fallback[2]
            self.role_second_fallback_connection_inputs[role_name] = second_fallback[0]
            self.role_second_fallback_model_inputs[role_name] = second_fallback[1]
            self.role_second_cross_auth_checks[role_name] = second_fallback[2]
            provider_layout.addWidget(QLabel(role_labels[role_name]), row, 0)
            provider_layout.addWidget(connection_input, row, 1)
            provider_layout.addWidget(model_input, row, 2)
            provider_layout.addWidget(reasoning_input, row, 3)
            for offset, controls in enumerate(fallback_controls, start=1):
                fallback_label = QLabel(f"后备 {offset}")
                fallback_label.setObjectName("hint")
                provider_layout.addWidget(fallback_label, row + offset, 0)
                provider_layout.addWidget(controls[0], row + offset, 1)
                provider_layout.addWidget(controls[1], row + offset, 2)
                provider_layout.addWidget(controls[2], row + offset, 3)

        hint = QLabel(
            "同连接后备自动执行；跨连接后备只有勾选“已授权”才会执行。API Key 只写入 "
            "Windows 凭据管理器。字幕校订失败时继续保留本地 Whisper 原文。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        provider_layout.addWidget(hint, 17, 0, 1, 4)
        self.invocation_audit = QLabel("尚无模型调用记录")
        self.invocation_audit.setObjectName("hint")
        self.invocation_audit.setWordWrap(True)
        provider_layout.addWidget(self.invocation_audit, 18, 0, 1, 4)
        self._refresh_invocation_audit()
        scroll.setWidget(provider_group)
        return scroll

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
        transcript_actions = QHBoxLayout()
        self.transcript_diff_button = QPushButton("查看差异")
        self.transcript_edit_button = QPushButton("手动编辑")
        self.transcript_undo_button = QPushButton("撤销校订")
        for button in (
            self.transcript_diff_button,
            self.transcript_edit_button,
            self.transcript_undo_button,
        ):
            button.setProperty("role", "quiet")
            button.hide()
            transcript_actions.addWidget(button)
        panel_layout.addWidget(heading)
        panel_layout.addSpacing(6)
        panel_layout.addWidget(self.evidence_image)
        panel_layout.addWidget(self.evidence_title)
        panel_layout.addWidget(self.evidence_metadata)
        panel_layout.addWidget(self.evidence_version)
        panel_layout.addLayout(transcript_actions)
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
        status = str(self.session_filter.currentData())
        current_session_id = self._active_session_id()
        sessions = self.service.list_sessions(
            query=self.session_search.text(),
            status=status,
            newest_first=self._session_sort_newest,
        )
        state_labels = {
            "recording": "记录中",
            "interrupted": "已中断 · 可标记完成",
            "complete": "已完成",
        }
        for session in sessions:
            prefixes = []
            if session.id == current_session_id:
                prefixes.append("当前")
            if session.pinned:
                prefixes.append("已固定")
            if session.trashed_at_utc:
                prefixes.append("回收区")
            state = state_labels[session.status]
            prefix = f"{' · '.join(prefixes)} · " if prefixes else ""
            item = QListWidgetItem(
                f"{session.title}\n{prefix}{state} · {self._format_time(session.duration_ms)}"
                f" · {session.frame_count} 帧"
            )
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, session)
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
            self._selected_session_id = None
            self._show_empty_timeline()
        self._update_session_actions(selected_item)

    def _open_session_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        session_id = str(item.data(Qt.ItemDataRole.UserRole))
        session_changed = session_id != self._selected_session_id
        self._selected_session_id = session_id
        correction_settings = self.service.transcript_correction_settings(session_id)
        self.correction_check.blockSignals(True)
        self.correction_window_input.blockSignals(True)
        self.correction_check.setChecked(correction_settings.enabled)
        correction_index = self.correction_window_input.findData(
            correction_settings.window_ms // 1000
        )
        self.correction_window_input.setCurrentIndex(max(0, correction_index))
        self.correction_check.blockSignals(False)
        self.correction_window_input.blockSignals(False)

        answers = self.service.session_answers(session_id)
        self._answers_by_id = {answer.id: answer for answer in answers}
        if session_changed or self._selected_answer_version_id not in self._answers_by_id:
            self._selected_answer_version_id = answers[-1].id if answers else None
        self._populate_answer_selector(answers)
        materials = self.service.session_materials(session_id)
        self._materials_by_id = {material.id: material for material in materials}
        if session_changed or self._selected_material_version_id not in self._materials_by_id:
            self._selected_material_version_id = materials[-1].id if materials else None
        self._populate_material_selector(materials)
        if session_changed:
            self._content_kind = "answer" if answers else "material"

        window_duration = self.ZOOM_WINDOWS[self._zoom_key]
        try:
            timeline = self.service.open_session(
                session_id,
                window_start_ms=self._window_start_ms,
                window_duration_ms=window_duration,
                answer_version_id=self._selected_answer_version_id,
            )
        except Exception as exc:  # noqa: BLE001 - UI boundary reports persistence failures
            self._show_action_error(str(exc))
            return
        self._timeline = timeline
        self._selected_frame = None
        self._selected_transcript = None
        self._render_timeline(timeline)
        if self._content_kind == "material" and self._selected_material_version_id is not None:
            self._show_selected_material()
        else:
            self._content_kind = "answer"
            self._show_selected_answer(timeline)
        self._refresh_reanswer_target()
        self._refresh_material_controls()
        self._refresh_invocation_audit()
        self._prompt_pending_source_events(session_id)

    def _prompt_pending_source_events(self, session_id: str) -> None:
        try:
            events = self.service.source_events(session_id)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports persistence failures
            self._show_action_error(str(exc))
            return
        for event in events:
            if event.data_loss_confirmed or event.id in self._prompted_source_event_ids:
                continue
            self._source_event_reported(event)

    def _populate_answer_selector(self, answers: list[SessionAnswerRecord]) -> None:
        self.answer_selector.blockSignals(True)
        self.answer_selector.clear()
        for answer in answers:
            self.answer_selector.addItem(
                f"{self._format_time(answer.asked_at_ms)} · {answer.question}"
                f" · 回答 {answer.version_number}",
                answer.id,
            )
        selected_index = self.answer_selector.findData(self._selected_answer_version_id)
        self.answer_selector.setCurrentIndex(selected_index)
        self.answer_selector.setEnabled(bool(answers))
        self.answer_selector.blockSignals(False)

    def _populate_material_selector(self, materials: list[SessionMaterialVersionRecord]) -> None:
        self.material_selector.blockSignals(True)
        self.material_selector.clear()
        for material in materials:
            kind = "生成" if material.kind == "generated" else "编辑"
            self.material_selector.addItem(
                f"版本 {material.version_number} · {kind} · {self._format_time_string(material.created_at_utc)}",
                material.id,
            )
        selected_index = self.material_selector.findData(self._selected_material_version_id)
        self.material_selector.setCurrentIndex(selected_index)
        self.material_selector.setEnabled(bool(materials))
        self.material_selector.blockSignals(False)

    @staticmethod
    def _format_time_string(value: str) -> str:
        return value.replace("T", " ").split("+", 1)[0][:16]

    def _refresh_material_controls(self) -> None:
        has_materials = bool(self._materials_by_id)
        self.material_edit_button.setEnabled(
            has_materials
            and self._content_kind == "material"
            and not self._material_generation_in_flight
        )
        self.summary_button.setText("重新生成材料" if has_materials else "生成会话材料")
        self.summary_button.setEnabled(
            self._selected_session_id is not None and not self._material_generation_in_flight
        )

    def _show_selected_material(self) -> None:
        material = self._materials_by_id.get(self._selected_material_version_id or -1)
        if material is None:
            self._show_selected_answer(self._timeline) if self._timeline is not None else None
            return
        self._content_kind = "material"
        self.question_notes_label.hide()
        self.answer_evidence_status.setText(
            f"材料证据 · 版本 {material.version_number} · {material.evidence_state}"
        )
        material_evidence_state = (
            material.evidence_state
            if material.evidence_state in {"exact", "unavailable"}
            else "unavailable"
        )
        self.answer_evidence_status.setProperty("state", material_evidence_state)
        self.answer_evidence_status.setToolTip(
            "模型来源："
            + (material.connection_json or "未记录")
            + "\n\n证据标识：\n"
            + "\n".join(
                item.stable_id
                for item in self.service.material_evidence_entries(
                    self._selected_session_id or "", material.id
                )
            )
        )
        self.answer_evidence_status.style().unpolish(self.answer_evidence_status)
        self.answer_evidence_status.style().polish(self.answer_evidence_status)
        self.answer_evidence_status.show()
        self._render_material_evidence_entries(material.id)
        self.output.set_markdown(material.content)
        self._last_answer = material.content
        self.speak_button.setEnabled(False)
        self.add_note_button.setEnabled(False)
        self._refresh_reanswer_target()

    def _render_material_evidence_entries(self, material_version_id: int) -> None:
        self._clear_layout(self.answer_evidence_layout)
        if self._selected_session_id is None:
            self.answer_evidence_entries.hide()
            return
        evidence = self.service.material_evidence_entries(
            self._selected_session_id, material_version_id
        )
        for item in evidence:
            button = QPushButton(self._material_evidence_label(item))
            button.setObjectName(f"material-evidence-{item.ordinal}")
            button.setProperty("role", "quiet")
            button.setProperty("stableId", item.stable_id)
            button.setToolTip(f"定位证据：{item.stable_id}")
            button.clicked.connect(
                lambda _checked=False, material_id=material_version_id, stable_id=item.stable_id: (
                    self._navigate_to_material_evidence(material_id, stable_id)
                )
            )
            self.answer_evidence_layout.addWidget(button)
        self.answer_evidence_entries.setVisible(bool(evidence))

    def _material_evidence_label(self, evidence: MaterialEvidenceRecord) -> str:
        kind = "关键帧" if evidence.kind == "frame" else "字幕"
        return f"{kind} · {self._format_time(evidence.start_ms)} · {evidence.source}"

    def _navigate_to_material_evidence(self, material_version_id: int, stable_id: str) -> None:
        if self._selected_session_id is None or self._timeline is None:
            return
        evidence = next(
            (
                item
                for item in self.service.material_evidence_entries(
                    self._selected_session_id, material_version_id
                )
                if item.stable_id == stable_id
            ),
            None,
        )
        if evidence is None:
            self._show_evidence_navigation_error("材料证据目标不存在")
            return
        target_ms = evidence.start_ms
        self._zoom_key = "1-minute"
        zoom_button = self.findChild(QPushButton, "zoom-1-minute")
        if zoom_button is not None:
            zoom_button.setChecked(True)
        self._window_start_ms = max(0, target_ms - 30_000)
        current = self.session_library.currentItem()
        if current is None:
            return
        self._content_kind = "material"
        self._open_session_item(current)
        if self._timeline is None:
            return
        visible = next(
            (
                item
                for item in self._timeline.transcripts
                if item.version_id == evidence.transcript_version_id
            ),
            None,
        )
        if visible is None:
            self._show_evidence_navigation_error("材料证据不属于当前会话")
            return
        self._select_transcript(visible)
        button = self.findChild(QPushButton, f"transcript-{visible.id}")
        scroll = self.transcript_track.findChild(QScrollArea)
        if button is not None and scroll is not None:
            scroll.ensureWidgetVisible(button)

    def _show_selected_answer(self, timeline: SessionTimeline) -> None:
        self._content_kind = "answer"
        self.material_edit_button.setEnabled(False)
        self.question_notes_label.hide()
        answer = self._answers_by_id.get(timeline.selected_answer_id or -1)
        if answer is None:
            self.answer_evidence_status.hide()
            self._clear_layout(self.answer_evidence_layout)
            self.answer_evidence_entries.hide()
            self.output.set_markdown("")
            self._last_answer = ""
            self.speak_button.setEnabled(False)
            self.add_note_button.setEnabled(False)
            return

        summary = timeline.answer_evidence_summary
        if summary is None or summary.state == "unavailable":
            self.answer_evidence_status.setText(
                "此历史回答的精确证据不可恢复；未按问题时间范围推测引用。"
            )
            self.answer_evidence_status.setToolTip("")
            evidence_state = "unavailable"
        elif summary.frame_count or summary.transcript_count:
            assert summary.start_ms is not None and summary.end_ms is not None
            self.answer_evidence_status.setText(
                f"会话证据 · {summary.frame_count} 张关键帧 · "
                f"{summary.transcript_count} 条字幕 · "
                f"{self._format_time(summary.start_ms)}–{self._format_time(summary.end_ms)}"
            )
            self.answer_evidence_status.setToolTip(
                "稳定证据标识：\n" + "\n".join(summary.stable_ids)
            )
            evidence_state = "exact"
        else:
            self.answer_evidence_status.setText("会话证据不足 · 0 张关键帧 · 0 条字幕")
            self.answer_evidence_status.setToolTip("当前回答没有稳定证据标识。")
            evidence_state = "insufficient"
        self.answer_evidence_status.setProperty("state", evidence_state)
        self.answer_evidence_status.style().unpolish(self.answer_evidence_status)
        self.answer_evidence_status.style().polish(self.answer_evidence_status)
        self.answer_evidence_status.show()
        self._render_answer_evidence_entries(answer.id)

        if answer.answer:
            content = present_answer(answer.answer, summary)
        else:
            content = f"请求失败：{answer.error}" if answer.error else "此回答没有内容。"
        self._last_answer = content
        self.output.set_markdown(content)
        self.speak_button.setEnabled(bool(answer.answer and answer.answer.strip()))
        self.add_note_button.setEnabled(True)
        self._render_question_notes(answer.question_id)

    def _render_question_notes(self, question_id: int) -> None:
        if self._selected_session_id is None:
            self.question_notes_label.hide()
            return
        notes = self.service.question_notes(self._selected_session_id, question_id)
        if not notes:
            self.question_notes_label.hide()
            return
        self.question_notes_label.setText(
            "用户附注：\n" + "\n".join(f"- {note.content}" for note in notes)
        )
        self.question_notes_label.show()

    def _render_answer_evidence_entries(self, answer_version_id: int) -> None:
        self._clear_layout(self.answer_evidence_layout)
        if self._selected_session_id is None:
            self.answer_evidence_entries.hide()
            return
        evidence = self.service.answer_evidence_entries(
            self._selected_session_id, answer_version_id
        )
        for item in evidence:
            button = QPushButton(self._answer_evidence_label(item))
            button.setObjectName(f"answer-evidence-{item.ordinal}")
            button.setProperty("role", "quiet")
            button.setProperty("stableId", item.stable_id)
            button.setToolTip(f"定位证据：{item.stable_id}")
            button.clicked.connect(
                lambda _checked=False, answer_id=answer_version_id, stable_id=item.stable_id: (
                    self._navigate_to_answer_evidence(answer_id, stable_id)
                )
            )
            self.answer_evidence_layout.addWidget(button)
        self.answer_evidence_entries.setVisible(bool(evidence))

    def _answer_evidence_label(self, evidence: AnswerEvidenceRecord) -> str:
        kind = "关键帧" if evidence.kind == "frame" else "字幕"
        return f"{kind} · {self._format_time(evidence.start_ms)} · {evidence.source}"

    def _navigate_to_answer_evidence(self, answer_version_id: int, stable_id: str) -> None:
        if self._selected_session_id is None:
            return
        try:
            target = self.service.resolve_answer_evidence(
                self._selected_session_id, answer_version_id, stable_id
            )
        except (KeyError, LookupError, PermissionError, ValueError) as exc:
            self._show_evidence_navigation_error(str(exc))
            return

        target_ms = target.ts_ms if isinstance(target, TimelineFrameRecord) else target.start_ms
        self._zoom_key = "1-minute"
        zoom_button = self.findChild(QPushButton, "zoom-1-minute")
        if zoom_button is not None:
            zoom_button.setChecked(True)
        self._window_start_ms = max(0, target_ms - 30_000)
        current = self.session_library.currentItem()
        if current is None:
            return
        self._open_session_item(current)

        if self._timeline is None:
            return
        if isinstance(target, TimelineFrameRecord):
            visible = next((item for item in self._timeline.frames if item.id == target.id), None)
            object_name = f"keyframe-{target.id}"
        else:
            visible = next(
                (
                    item
                    for item in self._timeline.transcripts
                    if item.version_id == target.version_id
                ),
                None,
            )
            object_name = f"transcript-{target.id}"
        if visible is None:
            self._show_evidence_navigation_error("证据目标不属于当前会话")
            return
        if isinstance(visible, TimelineFrameRecord):
            self._select_frame(visible)
            track = self.keyframe_track
        else:
            self._select_transcript(visible)
            track = self.transcript_track
        button = self.findChild(QPushButton, object_name)
        scroll = track.findChild(QScrollArea)
        if button is not None and scroll is not None:
            scroll.ensureWidgetVisible(button)

    def _show_evidence_navigation_error(self, message: str) -> None:
        self._selected_frame = None
        self._selected_transcript = None
        self.evidence_image.setPixmap(QPixmap())
        self.evidence_image.setStyleSheet(
            "background: #172221; color: #dce8e3; padding: 18px; font-size: 15px;"
        )
        self.evidence_image.setText(message)
        self.evidence_title.setText("无法打开证据")
        self.evidence_metadata.setText("当前会话和时间线位置保持不变。")
        self.evidence_version.hide()
        self._show_transcript_actions(False)
        self._animate_detail_change()

    @Slot()
    def _select_answer(self) -> None:
        answer_id = self.answer_selector.currentData()
        self._selected_answer_version_id = int(answer_id) if answer_id is not None else None
        self._content_kind = "answer"
        current = self.session_library.currentItem()
        if current is not None:
            self._open_session_item(current)

    @Slot()
    def _select_material(self) -> None:
        material_id = self.material_selector.currentData()
        self._selected_material_version_id = int(material_id) if material_id is not None else None
        if material_id is None:
            return
        self._content_kind = "material"
        self._show_selected_material()
        self._refresh_material_controls()

    @Slot()
    def _add_question_note(self) -> None:
        if self._selected_session_id is None or self._selected_answer_version_id is None:
            return
        answer = self._answers_by_id.get(self._selected_answer_version_id)
        if answer is None:
            return
        content, accepted = QInputDialog.getMultiLineText(self, "添加问题附注", "附注内容", "")
        if not accepted or not content.strip():
            return
        try:
            self.service.add_question_note(self._selected_session_id, answer.question_id, content)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports validation failures
            self._show_action_error(str(exc))
            return
        self._render_question_notes(answer.question_id)
        self._set_status("附注已保存", "success")

    @Slot()
    def _edit_selected_material(self) -> None:
        if self._selected_session_id is None or self._selected_material_version_id is None:
            return
        material = self._materials_by_id.get(self._selected_material_version_id)
        if material is None:
            return
        content, accepted = QInputDialog.getMultiLineText(
            self, "编辑会话材料", "Markdown 内容", material.content
        )
        if not accepted or not content.strip() or content == material.content:
            return
        try:
            edited = self.service.edit_material(self._selected_session_id, material.id, content)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports validation failures
            self._show_action_error(str(exc))
            return
        self._materials_by_id[edited.id] = edited
        self._selected_material_version_id = edited.id
        self._content_kind = "material"
        self._populate_material_selector(list(self._materials_by_id.values()))
        self._show_selected_material()
        self._set_status(f"材料版本 {edited.version_number} 已保存", "success")

    def _select_session_item(self, item: QListWidgetItem | None) -> None:
        self._window_start_ms = 0
        self._update_session_actions(item)
        self._open_session_item(item)

    def _selected_session_record(self) -> SessionRecord | None:
        item = self.session_library.currentItem()
        if item is None:
            return None
        record = item.data(Qt.ItemDataRole.UserRole + 1)
        return record if isinstance(record, SessionRecord) else None

    def _update_session_actions(self, item: QListWidgetItem | None) -> None:
        record = item.data(Qt.ItemDataRole.UserRole + 1) if item is not None else None
        if not isinstance(record, SessionRecord):
            self.session_pin_button.setEnabled(False)
            self.session_delete_button.setEnabled(False)
            self.session_restore_button.hide()
            self.session_complete_button.hide()
            return
        current = record.id == self._active_session_id()
        busy = self.service.session_storage_busy_reason(record.id) is not None
        self.session_pin_button.setEnabled(
            not current and not busy and record.trashed_at_utc is None
        )
        self.session_pin_button.setText("取消固定" if record.pinned else "固定")
        self.session_delete_button.setEnabled(
            not current and not busy and record.trashed_at_utc is None
        )
        self.session_delete_button.setVisible(record.trashed_at_utc is None)
        self.session_restore_button.setVisible(record.trashed_at_utc is not None)
        interrupted = record.status == "interrupted" and record.trashed_at_utc is None
        self.session_complete_button.setVisible(interrupted)
        self.session_complete_button.setEnabled(interrupted and not busy)

    def _toggle_session_sort(self) -> None:
        self._session_sort_newest = not self._session_sort_newest
        self.session_sort_button.setText("新→旧" if self._session_sort_newest else "旧→新")
        self._refresh_sessions()

    def _toggle_selected_session_pin(self) -> None:
        record = self._selected_session_record()
        if record is None:
            return
        try:
            self.service.pin_session(record.id, not record.pinned)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports persistence failures
            self._show_action_error(str(exc))
            return
        self._refresh_sessions(record.id)

    def _delete_selected_session(self) -> None:
        record = self._selected_session_record()
        if record is None:
            return
        answer = QMessageBox.question(
            self,
            "删除完整会话",
            f"“{record.title}”将移入本地回收区并保留 7 天。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.delete_session(record.id)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports persistence failures
            self._show_action_error(str(exc))
            return
        self._selected_session_id = None
        self._refresh_sessions()

    def _restore_selected_session(self) -> None:
        record = self._selected_session_record()
        if record is None:
            return
        try:
            self.service.restore_session(record.id)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports persistence failures
            self._show_action_error(str(exc))
            return
        self.session_filter.setCurrentIndex(0)
        self._refresh_sessions(record.id)

    def _complete_selected_session(self) -> None:
        record = self._selected_session_record()
        if record is None:
            return
        try:
            self.service.complete_interrupted_session(record.id)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports persistence failures
            self._show_action_error(str(exc))
            return
        self._refresh_sessions(record.id)

    def _refresh_recording_status(self) -> None:
        if not getattr(self.service, "is_recording", False):
            if self._stop_in_flight:
                self.start_button.setEnabled(False)
            else:
                session_id = getattr(self.manager, "session_id", None)
                busy_reason = None
                storage_busy = getattr(self.service, "session_storage_busy_reason", None)
                if session_id is not None and callable(storage_busy):
                    busy_reason = storage_busy(session_id)
                self.start_button.setEnabled(busy_reason is None)
            self._paused = False
            self.pause_button.setEnabled(False)
            self.pause_button.setText("暂停")
            return
        if not self.stop_button.isEnabled():
            self.pause_button.setEnabled(False)
            return
        status = self.service.recording_status()
        self._paused = status.state == "paused"
        self.pause_button.setEnabled(self.stop_button.isEnabled() and self.service.supports_pause)
        self.pause_button.setText("继续" if self._paused else "暂停")
        details = [self._format_time(status.duration_ms)]
        details.append(f"{status.display_count} 屏")
        if status.system_audio:
            details.append("系统声音")
        if status.microphone:
            details.append("麦克风")
        if status.failed_sources:
            details.append("来源故障：" + "、".join(sorted(status.failed_sources)))
        prefix = "已暂停" if self._paused else "记录中"
        self._set_status(f"{prefix} · " + " · ".join(details), "recording")

    def _active_session_id(self) -> str | None:
        if not getattr(self.manager, "is_recording", False):
            return None
        return getattr(self.manager, "session_id", None)

    def _run_session_maintenance(self) -> None:
        if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
            return

        def work() -> None:
            try:
                notices = self.service.run_session_maintenance()
            except Exception as exc:
                logger.exception("Session maintenance failed")
                self.bridge.maintenance_failed.emit(str(exc))
            else:
                self.bridge.maintenance_finished.emit(notices)

        self._maintenance_thread = threading.Thread(
            target=work, name="session-maintenance", daemon=True
        )
        self._maintenance_thread.start()

    @Slot(object)
    def _maintenance_finished(self, notices) -> None:  # type: ignore[no-untyped-def]
        self._update_session_actions(self.session_library.currentItem())
        if not notices:
            return
        self._refresh_sessions(self._selected_session_id)
        self._show_lifecycle_notices(notices)

    @Slot(str)
    def _maintenance_failed(self, message: str) -> None:
        self._update_session_actions(self.session_library.currentItem())
        self._show_action_error(message)

    def _show_lifecycle_notices(self, notices) -> None:  # type: ignore[no-untyped-def]
        if not notices:
            return
        labels = {
            SessionNotificationKind.RETENTION_7D: "将在 7 天内进入回收区",
            SessionNotificationKind.RETENTION_1D: "将在 1 天内进入回收区",
            SessionNotificationKind.MOVED_TO_TRASH: "已移入回收区，可在 7 天内恢复",
            SessionNotificationKind.PERMANENTLY_DELETED: "已从本机最终删除",
            SessionNotificationKind.FINAL_DELETE_FAILED: "媒体删除失败，完整会话仍保留在回收区并将在下次重试",
        }
        self.notice_text.setText(
            "\n".join(f"{notice.title}：{labels[notice.kind]}" for notice in notices)
        )
        self.notice.show()

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
        state = {
            "recording": "记录中",
            "interrupted": "已中断 · 可恢复处理",
            "complete": "已完成",
        }.get(session.status, session.status)
        self.workspace_meta.setText(f"{state} · {len(timeline.frames)} 张关键帧位于当前缩放窗口")
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
                    f"关键帧 #{frame.id} · {frame.source_id} · {self._format_time(frame.ts_ms)}"
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
                state_label = self.CORRECTION_STATE_LABELS.get(transcript.correction_state or "")
                prefix = f"{state_label} · " if state_label else ""
                button = EvidenceButton(
                    f"{prefix}{self._format_time(transcript.start_ms)}–"
                    f"{self._format_time(transcript.end_ms)}\n{transcript.text}",
                    animations=self._animations_enabled,
                )
                button.setObjectName(f"transcript-{transcript.id}")
                button.setProperty("transcript", True)
                button.setProperty("selected", False)
                button.setProperty("cited", transcript.id in timeline.answer_transcript_ids)
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
        timeline_event_items = []
        for event in timeline.events:
            label = "暂停" if event.kind == "pause" else "数据缺口"
            source = f" · {event.source}" if event.source else ""
            message = f"：{event.message}" if event.kind == "data_gap" else ""
            timeline_event_items.append(
                f"{label} {self._format_time(event.start_ms)}–"
                f"{self._format_time(event.end_ms)}{source}{message}"
            )
        event_summary = (
            "  ·  ".join(item for item in (question_events, *timeline_event_items) if item)
            or "无事件"
        )
        self.event_text.setText(
            f"会话开始 00:00  ·  当前窗口 "
            f"{self._format_time(timeline.window_start_ms)}–"
            f"{self._format_time(min(timeline.window_end_ms, timeline.duration_ms))}  ·  "
            f"{event_summary}  ·  {state}"
        )
        self.evidence_image.clear()
        self.evidence_image.setText("选择关键帧查看大图")
        self.evidence_title.setText("尚未选择证据")
        self.evidence_metadata.setText("来源与会话相对时间将在这里显示。")
        self.evidence_version.hide()
        self._show_transcript_actions(False)
        self._animate_detail_change()

    @staticmethod
    def _clear_layout(layout: QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    @Slot()
    def _select_frame(self, frame: TimelineFrameRecord) -> None:
        self._selected_frame = frame
        self._selected_transcript = None
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
        self._show_transcript_actions(False)
        self._animate_detail_change()

    @Slot()
    def _select_transcript(self, transcript: TimelineTranscriptRecord) -> None:
        self._selected_frame = None
        self._selected_transcript = transcript
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
        state_label = self.CORRECTION_STATE_LABELS.get(transcript.correction_state or "")
        if transcript.version_kind == "correction":
            version_label = "校订文"
        elif transcript.version_kind == "user_edit":
            version_label = "用户编辑"
        else:
            version_label = "当前版本"
        state_line = f"字幕校订状态：{state_label}\n" if state_label else ""
        self.evidence_version.setText(
            f"{state_line}Whisper 原文：{transcript.original_text}\n"
            f"{version_label}：{transcript.text}"
        )
        self.evidence_version.show()
        editable = transcript.version_id is not None and transcript.version_kind != "recognizing"
        self._show_transcript_actions(editable)
        self.transcript_undo_button.setVisible(editable and transcript.version_kind == "correction")
        self._animate_detail_change()

    def _show_transcript_actions(self, visible: bool) -> None:
        self.transcript_diff_button.setVisible(visible)
        self.transcript_edit_button.setVisible(visible)
        if not visible:
            self.transcript_undo_button.hide()

    @staticmethod
    def _transcript_diff(original: str, current: str) -> str:
        parts: list[str] = []
        for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
            None, original, current
        ).get_opcodes():
            if tag == "equal":
                parts.append(original[old_start:old_end])
            elif tag == "delete":
                parts.append(f"[-{original[old_start:old_end]}-]")
            elif tag == "insert":
                parts.append(f"{{+{current[new_start:new_end]}+}}")
            else:
                parts.append(f"[-{original[old_start:old_end]}-]")
                parts.append(f"{{+{current[new_start:new_end]}+}}")
        return "".join(parts)

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
        self._answers_by_id = {}
        self._materials_by_id = {}
        self._selected_answer_version_id = None
        self._selected_material_version_id = None
        self._content_kind = "answer"
        self._populate_answer_selector([])
        self._populate_material_selector([])
        self.question_notes_label.hide()
        self._refresh_material_controls()

    def _build_answer_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("contentPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 10, 12, 12)
        question_row = QHBoxLayout()
        self.question = QuestionInput()
        self.question.setPlaceholderText("例如：刚才这个结论是怎么得到的？")
        self.question_range = QComboBox()
        self.question_range.setObjectName("questionRange")
        for label, lookback_ms in (
            ("30 秒", 30_000),
            ("2 分钟", 2 * 60_000),
            ("5 分钟", 5 * 60_000),
        ):
            self.question_range.addItem(label, lookback_ms)
        self.question_range.setCurrentIndex(1)
        self.voice_button = QPushButton("按住说话")
        self.cancel_question_button = QPushButton("取消")
        self.cancel_question_button.setProperty("role", "quiet")
        self.ask_button = QPushButton("提问")
        self.ask_button.setProperty("role", "primary")
        question_row.addWidget(self.question, 1)
        question_row.addWidget(self.question_range)
        question_row.addWidget(self.voice_button)
        question_row.addWidget(self.cancel_question_button)
        question_row.addWidget(self.ask_button)
        answer_selection_row = QHBoxLayout()
        answer_selection_label = QLabel("选择问答")
        answer_selection_label.setObjectName("trackLabel")
        self.answer_selector = QComboBox()
        self.answer_selector.setObjectName("answerSelector")
        self.answer_selector.setEnabled(False)
        self.answer_evidence_status = QLabel()
        self.answer_evidence_status.setObjectName("answerEvidenceStatus")
        self.answer_evidence_status.hide()
        answer_selection_row.addWidget(answer_selection_label)
        answer_selection_row.addWidget(self.answer_selector, 1)
        answer_selection_row.addWidget(self.answer_evidence_status, 2)
        self.answer_evidence_entries = QWidget()
        material_selection_row = QHBoxLayout()
        material_selection_label = QLabel("选择材料")
        material_selection_label.setObjectName("trackLabel")
        self.material_selector = QComboBox()
        self.material_selector.setObjectName("materialSelector")
        self.material_selector.setEnabled(False)
        self.material_edit_button = QPushButton("编辑材料")
        self.material_edit_button.setObjectName("materialEditButton")
        self.material_edit_button.setProperty("role", "quiet")
        self.material_edit_button.setEnabled(False)
        material_selection_row.addWidget(material_selection_label)
        material_selection_row.addWidget(self.material_selector, 1)
        material_selection_row.addWidget(self.material_edit_button)

        self.answer_evidence_entries.setObjectName("answerEvidenceEntries")
        self.answer_evidence_layout = QHBoxLayout(self.answer_evidence_entries)
        self.answer_evidence_layout.setContentsMargins(0, 0, 0, 0)
        self.answer_evidence_layout.setSpacing(6)
        self.answer_evidence_entries.hide()
        self.question_notes_label = QLabel()
        self.question_notes_label.setObjectName("questionNotes")
        self.question_notes_label.setWordWrap(True)
        self.question_notes_label.hide()

        answer_header = QHBoxLayout()
        heading = QLabel("回答与会话材料")
        heading.setObjectName("sectionTitle")
        self.reanswer_button = QPushButton("基于最新字幕重新回答")
        self.reanswer_button.setProperty("role", "quiet")
        self.reanswer_button.setEnabled(False)
        self.speak_button = QPushButton("朗读回答")
        self.speak_button.setProperty("role", "quiet")
        self.speak_button.setEnabled(False)
        self.add_note_button = QPushButton("添加附注")
        self.add_note_button.setProperty("role", "quiet")
        self.add_note_button.setEnabled(False)
        self.output_source_button = QPushButton("查看原文")
        self.output_source_button.setProperty("role", "quiet")
        self.copy_output_button = QPushButton("复制原文")
        self.copy_output_button.setProperty("role", "quiet")
        answer_header.addWidget(heading)
        answer_header.addStretch(1)
        answer_header.addWidget(self.speak_button)
        answer_header.addWidget(self.add_note_button)
        answer_header.addWidget(self.reanswer_button)
        answer_header.addWidget(self.output_source_button)
        answer_header.addWidget(self.copy_output_button)
        self.output = MarkdownDocument()
        panel_layout.addLayout(question_row)
        panel_layout.addLayout(answer_selection_row)
        panel_layout.addLayout(material_selection_row)
        panel_layout.addWidget(self.answer_evidence_entries)
        panel_layout.addWidget(self.question_notes_label)

        panel_layout.addLayout(answer_header)
        panel_layout.addWidget(self.output, 1)
        return panel

    @Slot()
    def _show_whisper_settings(self) -> None:
        if not callable(getattr(self.manager, "configure_whisper", None)):
            return
        if self._whisper_dialog is None:
            self._whisper_dialog = WhisperSettingsDialog(self.manager, self)
        self._whisper_dialog.show()
        self._whisper_dialog.raise_()
        self._whisper_dialog.activateWindow()

    def _storage_busy_reason(self) -> str | None:
        manager_reason = getattr(self.manager, "storage_busy_reason", None)
        if callable(manager_reason):
            reason = manager_reason()
            if reason:
                return reason
        write_thread_names = {
            "question-voice",
            "stop-session",
            "session-maintenance",
            "test-provider",
            "answer-question",
            "reanswer-question",
            "generate-material",
            "edit-material",
            "summarize-session",
            "whisper-model-download",
            "whisper-benchmark",
        }
        if any(thread.name in write_thread_names for thread in threading.enumerate()):
            return "后台任务仍在写入应用数据或模型缓存"
        return None

    @Slot()
    def _show_storage_settings(self) -> None:
        if self._storage_dialog is None:
            model_in_use = getattr(self.manager, "whisper_model_in_use", None)
            self._storage_dialog = StorageSettingsDialog(
                self.settings,
                busy_reason=self._storage_busy_reason,
                model_in_use=model_in_use if callable(model_in_use) else None,
                parent=self,
            )
        self._storage_dialog.refresh()
        self._storage_dialog.show()
        self._storage_dialog.raise_()
        self._storage_dialog.activateWindow()

    def _connect_signals(self) -> None:
        self.session_library.currentItemChanged.connect(
            lambda current, _previous: self._select_session_item(current)
        )
        self.session_search.textChanged.connect(self._refresh_sessions)
        self.session_filter.currentIndexChanged.connect(self._refresh_sessions)
        self.session_sort_button.clicked.connect(self._toggle_session_sort)
        self.session_pin_button.clicked.connect(self._toggle_selected_session_pin)
        self.session_delete_button.clicked.connect(self._delete_selected_session)
        self.session_restore_button.clicked.connect(self._restore_selected_session)
        self.session_complete_button.clicked.connect(self._complete_selected_session)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.capsule_ask_button.clicked.connect(self._focus_question)
        self.ask_button.clicked.connect(self._ask)
        self.reanswer_button.clicked.connect(self._reanswer)
        self.bridge.segment.connect(self._append_segment)
        self.bridge.worker_warning.connect(self._show_worker_warning)
        self.bridge.source_event.connect(self._source_event_reported)
        self.answer_selector.currentIndexChanged.connect(self._select_answer)
        self.material_selector.currentIndexChanged.connect(self._select_material)
        self.material_edit_button.clicked.connect(self._edit_selected_material)
        self.add_note_button.clicked.connect(self._add_question_note)

        self.question.focused.connect(self._capture_question_anchor)
        self.question.returnPressed.connect(self._ask)
        self.question_range.currentIndexChanged.connect(self._change_question_range)
        self.cancel_question_button.clicked.connect(self._cancel_question)
        self.voice_button.pressed.connect(self._start_question_voice)
        self.voice_button.released.connect(self._finish_question_voice)
        self.speak_button.clicked.connect(self._speak_answer)
        self.ask_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
        self.ask_shortcut.activated.connect(self._focus_question)
        self.ask_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.summary_button.clicked.connect(self._summarize)
        self.test_provider_button.clicked.connect(self._test_provider)
        self.save_provider_button.clicked.connect(self._save_provider)
        self.connection_selector.currentIndexChanged.connect(self._select_provider_connection)
        self.add_connection_button.clicked.connect(self._add_provider_connection)
        self.remove_connection_button.clicked.connect(self._remove_provider_connection)
        self.correction_check.toggled.connect(self._configure_correction)
        self.correction_window_input.currentIndexChanged.connect(self._configure_correction)
        self.transcript_diff_button.clicked.connect(self._show_transcript_diff)
        self.transcript_edit_button.clicked.connect(self._edit_selected_transcript)
        self.transcript_undo_button.clicked.connect(self._undo_selected_correction)
        self.output_source_button.clicked.connect(self._toggle_output_source)
        self.copy_output_button.clicked.connect(self._copy_output_source)
        self.provider_toggle_button.clicked.connect(
            lambda: self.provider_group.setVisible(not self.provider_group.isVisible())
        )
        self.whisper_settings_button.clicked.connect(self._show_whisper_settings)
        self.storage_settings_button.clicked.connect(self._show_storage_settings)
        self.output.render_failed.connect(self._show_worker_warning)
        self.bridge.action_error.connect(self._show_action_error)
        self.bridge.answer.connect(self._show_answer)
        self.bridge.voice_transcript.connect(self._show_voice_transcript)
        self.bridge.voice_error.connect(self._show_voice_error)
        self.bridge.summary.connect(self._show_summary)
        self.bridge.material.connect(self._show_material)
        self.bridge.stopped.connect(self._recording_stopped)
        self.bridge.stop_failed.connect(self._stop_failed)
        self.bridge.pause_finished.connect(self._pause_operation_finished)
        self.bridge.provider_tested.connect(self._provider_tested)
        self.bridge.maintenance_finished.connect(self._maintenance_finished)
        self.bridge.maintenance_failed.connect(self._maintenance_failed)

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
        catalog = getattr(self.manager, "device_catalog", None) or WindowsDeviceCatalog()
        dialog = RecordingConfirmationDialog(
            catalog,
            RecordingSettingsStore(self.settings.data_dir),
            screen_interval_s=self.settings.screen_interval_s,
            audio_storage_rate=self.settings.audio_storage_rate,
            default_system_audio_enabled=self.system_audio_check.isChecked(),
            default_microphone_enabled=self.microphone_check.isChecked(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            selection = dialog.recording_selection()
            self._configure_correction()
            session_id = self.service.start_session(self.title_input.text(), selection=selection)
        except Exception as exc:  # noqa: BLE001 - UI boundary must surface worker failures
            self._show_action_error(str(exc))
            return
        self.system_audio_check.setChecked(selection.system_audio_id is not None)
        self.microphone_check.setChecked(selection.microphone_id is not None)
        self._set_status(f"记录中 · {session_id[:8]}", "recording")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(callable(getattr(self.manager, "pause", None)))
        self.system_audio_check.setEnabled(False)
        self.microphone_check.setEnabled(False)
        self.correction_check.setEnabled(False)
        self.correction_window_input.setEnabled(False)
        self._refresh_recording_status()
        self._refresh_sessions(session_id)

    @Slot()
    def _configure_correction(self) -> None:
        enabled = self.correction_check.isChecked()
        window_seconds = int(self.correction_window_input.currentData())
        configure_manager = getattr(self.manager, "configure_transcript_correction", None)
        if callable(configure_manager):
            configure_manager(enabled=enabled, window_seconds=window_seconds)
        if self._selected_session_id is not None:
            self.service.configure_transcript_correction(
                self._selected_session_id,
                enabled=enabled,
                window_seconds=window_seconds,
            )
            current = self.session_library.currentItem()
            if current is not None:
                self._open_session_item(current)

    @Slot()
    def _show_transcript_diff(self) -> None:
        transcript = self._selected_transcript
        if transcript is None:
            return
        self.evidence_version.setText(
            "原文与当前版本差异：\n"
            + self._transcript_diff(transcript.original_text, transcript.text)
        )
        self.evidence_version.show()

    @Slot()
    def _edit_selected_transcript(self) -> None:
        transcript = self._selected_transcript
        if transcript is None:
            return
        text, accepted = QInputDialog.getMultiLineText(
            self, "手动编辑字幕", "字幕内容", transcript.text
        )
        if not accepted:
            return
        try:
            self.service.edit_transcript(transcript.id, text)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports validation failures
            self._show_action_error(str(exc))
            return
        current = self.session_library.currentItem()
        if current is not None:
            self._open_session_item(current)

    @Slot()
    def _undo_selected_correction(self) -> None:
        transcript = self._selected_transcript
        if transcript is None:
            return
        self.service.undo_transcript_correction(transcript.id)
        current = self.session_library.currentItem()
        if current is not None:
            self._open_session_item(current)

    def _refresh_reanswer_target(self) -> None:
        selected_answer = self._answers_by_id.get(self._selected_answer_version_id or -1)
        if selected_answer is not None:
            self._reanswer_question_id = selected_answer.question_id
        elif self._selected_session_id is not None:
            self._reanswer_question_id = self.service.latest_question_id(self._selected_session_id)
        else:
            self._reanswer_question_id = None
        self.reanswer_button.setEnabled(
            self._content_kind == "answer"
            and self._reanswer_question_id is not None
            and callable(getattr(self.manager, "reanswer_question", None))
        )

    @Slot()
    def _capture_question_anchor(self) -> None:
        was_active = self._question_active
        try:
            self._active_question_id = self.service.begin_question(
                int(self.question_range.currentData())
            )
        except Exception as exc:  # noqa: BLE001 - UI boundary surfaces invalid session state
            self._show_action_error(str(exc))
            return
        self._question_active = True
        if not was_active:
            self._question_generation += 1

    @Slot()
    def _focus_question(self) -> None:
        if self.question.hasFocus():
            self._capture_question_anchor()
        else:
            self.question.setFocus(Qt.FocusReason.ShortcutFocusReason)

    @Slot()
    def _change_question_range(self) -> None:
        if not self._question_active:
            return
        try:
            self.service.set_question_range(int(self.question_range.currentData()))
        except Exception as exc:  # noqa: BLE001 - UI boundary surfaces persistence failures
            self._show_action_error(str(exc))

    @Slot()
    def _cancel_question(self) -> None:
        try:
            self.service.cancel_question()
        except Exception as exc:  # noqa: BLE001 - UI boundary surfaces persistence failures
            self._show_action_error(str(exc))
            return
        self._question_active = False
        self._active_question_id = None

        self._question_generation += 1
        self.question.clear()
        self.voice_button.setEnabled(True)
        self.voice_button.setText("按住说话")

    @Slot()
    def _start_question_voice(self) -> None:
        if not self._question_active:
            self._capture_question_anchor()
        if not self._question_active:
            return
        try:
            self.service.start_question_voice()
        except Exception as exc:  # noqa: BLE001 - UI boundary surfaces microphone failures
            self._show_action_error(str(exc))
            return
        self.voice_button.setText("松开结束")

    @Slot()
    def _finish_question_voice(self) -> None:
        if self.voice_button.text() != "松开结束":
            return
        self.voice_button.setEnabled(False)
        self.voice_button.setText("正在转写…")
        generation = self._question_generation

        def work() -> None:
            try:
                transcript = self.service.finish_question_voice()
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.voice_error.emit(generation, str(exc))
            else:
                self.bridge.voice_transcript.emit(generation, transcript)

        threading.Thread(target=work, name="question-voice", daemon=True).start()

    @Slot(int, str)
    def _show_voice_transcript(self, generation: int, transcript: str) -> None:
        self.voice_button.setEnabled(True)
        self.voice_button.setText("按住说话")
        if generation != self._question_generation or not self._question_active:
            return
        self.question.setText(transcript)
        self.question.setFocus(Qt.FocusReason.OtherFocusReason)

    @Slot(int, str)
    def _show_voice_error(self, generation: int, message: str) -> None:
        if generation != self._question_generation:
            return
        self.voice_button.setEnabled(True)
        self.voice_button.setText("按住说话")
        self._show_action_error(message)

    @Slot()
    def _toggle_pause(self) -> None:
        if not self.service.supports_pause:
            return
        method = self.service.resume_session if self._paused else self.service.pause_session
        self.pause_button.setEnabled(False)

        def work() -> None:
            try:
                changed = bool(method())
            except Exception as exc:  # noqa: BLE001 - UI boundary surfaces adapter failures
                self.bridge.action_error.emit(str(exc))
                changed = False
            self.bridge.pause_finished.emit(changed)

        threading.Thread(target=work, name="pause-session", daemon=True).start()

    @Slot(bool)
    def _pause_operation_finished(self, changed: bool) -> None:
        if changed:
            self._refresh_recording_status()
        if self.stop_button.isEnabled() and self.service.is_recording:
            self.pause_button.setEnabled(self.service.supports_pause)
        else:
            self.pause_button.setEnabled(False)

    @Slot()
    def _stop(self) -> None:
        self._stop_in_flight = True
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self._set_status("正在结束并处理剩余字幕…", "idle")

        def work() -> None:
            try:
                session_id = self.service.stop_session()
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.stop_failed.emit(str(exc))
            else:
                self.bridge.stopped.emit(session_id or "")

        threading.Thread(target=work, name="stop-session", daemon=True).start()

    def _store_active_connection_form(self) -> None:
        current = self._provider_connections[self._active_connection_index]
        self._provider_connections[self._active_connection_index] = replace(
            current,
            name=self.connection_name_input.text().strip(),
            base_url=self.base_url_input.text().strip(),
            api_key=self.api_key_input.text().strip(),
            api_mode=str(self.api_mode_input.currentData()),
        )

    @Slot(int)
    def _select_provider_connection(self, index: int) -> None:
        if index < 0 or index == self._active_connection_index:
            return
        previous = self._active_connection_index
        try:
            self._store_active_connection_form()
        except Exception as exc:  # noqa: BLE001 - form validation boundary
            self.connection_selector.blockSignals(True)
            self.connection_selector.setCurrentIndex(previous)
            self.connection_selector.blockSignals(False)
            self._show_action_error(str(exc))
            return
        self._active_connection_index = index
        self._load_active_connection_form()

    def _load_active_connection_form(self) -> None:
        connection = self._provider_connections[self._active_connection_index]
        self.connection_name_input.setText(connection.name)
        self.base_url_input.setText(connection.base_url)
        self.api_key_input.setText(connection.api_key)
        self.api_mode_input.setCurrentIndex(
            max(0, self.api_mode_input.findData(connection.api_mode))
        )

    @Slot()
    def _add_provider_connection(self) -> None:
        try:
            self._store_active_connection_form()
        except Exception as exc:  # noqa: BLE001 - form validation boundary
            self._show_action_error(str(exc))
            return
        self._provider_connections.append(ModelConnection(uuid.uuid4().hex, "新连接"))
        self._refresh_provider_connection_choices(len(self._provider_connections) - 1)

    @Slot()
    def _remove_provider_connection(self) -> None:
        if len(self._provider_connections) == 1:
            self._show_action_error("至少保留一个模型连接")
            return
        connection_id = self._provider_connections[self._active_connection_index].id
        used = any(
            combo.currentData() == connection_id
            for combo in (
                *self.role_connection_inputs.values(),
                *self.role_fallback_connection_inputs.values(),
                *self.role_second_fallback_connection_inputs.values(),
            )
        )
        if used:
            self._show_action_error("请先把使用该连接的角色和后备策略切换到其他连接")
            return
        self._provider_connections.pop(self._active_connection_index)
        self._refresh_provider_connection_choices(
            min(self._active_connection_index, len(self._provider_connections) - 1)
        )

    def _refresh_provider_connection_choices(self, selected_index: int) -> None:
        role_connections = {
            name: combo.currentData() for name, combo in self.role_connection_inputs.items()
        }
        fallback_groups = (
            self.role_fallback_connection_inputs,
            self.role_second_fallback_connection_inputs,
        )
        fallback_connections = tuple(
            {name: combo.currentData() for name, combo in group.items()}
            for group in fallback_groups
        )
        self.connection_selector.blockSignals(True)
        self.connection_selector.clear()
        for connection in self._provider_connections:
            self.connection_selector.addItem(connection.name, connection.id)
        self.connection_selector.setCurrentIndex(selected_index)
        self.connection_selector.blockSignals(False)
        for name, combo in self.role_connection_inputs.items():
            combo.clear()
            for connection in self._provider_connections:
                combo.addItem(connection.name, connection.id)
            combo.setCurrentIndex(max(0, combo.findData(role_connections[name])))
        for group, selections in zip(fallback_groups, fallback_connections, strict=True):
            for name, combo in group.items():
                combo.clear()
                combo.addItem("无", "")
                for connection in self._provider_connections:
                    combo.addItem(connection.name, connection.id)
                combo.setCurrentIndex(max(0, combo.findData(selections[name])))
        self._active_connection_index = selected_index
        self._load_active_connection_form()

    def _provider_settings_from_form(self) -> SavedProviderSettings:
        self._store_active_connection_form()
        roles: list[ModelRole] = []
        for name in RoleName:
            connection_id = str(self.role_connection_inputs[name].currentData())
            model = self.role_model_inputs[name].text().strip()
            if not model:
                raise ValueError(f"{name.value} 角色必须配置模型")
            fallbacks: list[ModelFallback] = []
            fallback_controls = (
                (
                    self.role_fallback_connection_inputs[name],
                    self.role_fallback_model_inputs[name],
                    self.role_cross_auth_checks[name],
                ),
                (
                    self.role_second_fallback_connection_inputs[name],
                    self.role_second_fallback_model_inputs[name],
                    self.role_second_cross_auth_checks[name],
                ),
            )
            for position, (connection_input, model_input, authorization) in enumerate(
                fallback_controls, start=1
            ):
                fallback_connection_id = str(connection_input.currentData() or "")
                fallback_model = model_input.text().strip()
                if bool(fallback_connection_id) != bool(fallback_model):
                    raise ValueError(f"{name.value} 角色的后备 {position} 连接和模型必须同时填写")
                if fallback_connection_id:
                    fallbacks.append(
                        ModelFallback(
                            fallback_connection_id,
                            fallback_model,
                            authorization.isChecked() and fallback_connection_id != connection_id,
                        )
                    )
            roles.append(
                ModelRole(
                    name,
                    connection_id,
                    model,
                    ReasoningLevel(str(self.role_reasoning_inputs[name].currentData())),
                    tuple(fallbacks),
                )
            )
        settings = SavedProviderSettings(tuple(self._provider_connections), tuple(roles))
        self._provider_roles = {role.name: role for role in settings.roles}
        return settings

    def _refresh_invocation_audit(self) -> None:
        database = getattr(self.manager, "database", None)
        if database is None or not hasattr(self, "invocation_audit"):
            return
        records = database.model_invocations(self._selected_session_id)
        if not records:
            self.invocation_audit.setText("尚无模型调用记录")
            return
        summaries = []
        for record in records[-4:]:
            fallback = f"，后备原因：{record.fallback_reason}" if record.fallback_reason else ""
            evidence = "、".join(record.evidence_ids) if record.evidence_ids else "无"
            summaries.append(
                f"{record.role} · {record.connection_name}/{record.model} · "
                f"{record.reasoning_level} · {record.status}{fallback} · 证据：{evidence}"
            )
        self.invocation_audit.setText("最近调用：" + "；".join(summaries))

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
            configure_correction = getattr(self.manager, "configure_transcript_correction", None)
            if callable(configure_correction):
                configure_correction(
                    enabled=self.correction_check.isChecked(),
                    window_seconds=int(self.correction_window_input.currentData()),
                )
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
        if not self._question_active:
            self._capture_question_anchor()
        question_id = self._active_question_id
        if not self._question_active or question_id is None:
            return
        self._question_active = False
        self._active_question_id = None
        self.ask_button.setEnabled(False)
        self.output.set_markdown("_正在结合字幕和关键画面回答…_")

        question_record = self.service.database.question(question_id)
        origin_session_id = question_record.session_id if question_record is not None else ""

        def work() -> None:
            try:
                result = self.service.submit_question(question)
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.action_error.emit(str(exc))
            else:
                self.bridge.answer.emit(question_id, result, origin_session_id)

        threading.Thread(target=work, name="answer-question", daemon=True).start()

    @Slot()
    def _reanswer(self) -> None:
        question_id = self._reanswer_question_id
        if question_id is None or not self._configure_provider_from_form():
            return
        self.ask_button.setEnabled(False)
        self.reanswer_button.setEnabled(False)
        self.output.set_markdown("_正在基于最新有效字幕重新回答…_")
        question_record = self.service.database.question(question_id)
        origin_session_id = question_record.session_id if question_record is not None else ""

        def work() -> None:
            try:
                result = self.manager.reanswer_question(question_id)
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.action_error.emit(str(exc))
            else:
                self.bridge.answer.emit(question_id, result, origin_session_id)

        threading.Thread(target=work, name="reanswer-question", daemon=True).start()

    @Slot()
    def _summarize(self) -> None:
        session_id = self._selected_session_id
        if session_id is None:
            self._show_action_error("请先选择一个会话")
            return
        if not self._configure_provider_from_form():
            return
        if not self._confirm_material_generation(session_id, automatic=False):
            return
        self._start_material_generation(session_id)

    def _confirm_material_generation(self, session_id: str, *, automatic: bool) -> bool:
        try:
            preview = self.service.material_generation_preview(session_id)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports configuration failures
            self._show_action_error(str(exc))
            return False
        if preview.transcript_count == 0:
            self._show_action_error("当前会话没有可发送的有效字幕")
            return False
        title = "确认自动生成会话材料" if automatic else "确认生成会话材料"
        message = (
            f"将发送 {preview.transcript_count} 条完整会话字幕 "
            f"（约 {preview.character_count} 字）到深度分析模型。\n\n"
            f"连接：{preview.connection_name}\n"
            f"模型：{preview.model}\n"
            f"地址：{preview.base_url}\n"
            f"推理级别：{preview.reasoning_level}\n\n"
            "可能产生模型连接服务费用，费用与配额由所选连接决定；应用不会显示或发送密钥。"
        )
        if automatic:
            QMessageBox.information(self, title, message + "\n\n将按已保存策略开始生成。")
            return True
        response = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def _start_material_generation(self, session_id: str) -> None:
        self._material_generation_in_flight = True
        self.summary_button.setEnabled(False)
        self.material_edit_button.setEnabled(False)
        self.output.set_markdown("_正在生成会话材料…_")

        def work() -> None:
            try:
                result = self.service.generate_material(session_id)
            except Exception as exc:  # noqa: BLE001 - background task reports through Qt
                self.bridge.action_error.emit(str(exc))
            else:
                self.bridge.material.emit(result)

        threading.Thread(target=work, name="generate-material", daemon=True).start()

    def _configure_provider_from_form(self) -> bool:
        try:
            self.manager.configure_provider(self._provider_settings_from_form())
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

    @Slot(object)
    def _source_event_reported(self, event: SourceEventRecord) -> None:
        if event.data_loss_confirmed or event.id in self._prompted_source_event_ids:
            return
        self._prompted_source_event_ids.add(event.id)
        response = QMessageBox.question(
            self,
            "确认数据缺口",
            f"{event.source} 在 {self._format_time(event.start_ms)}–"
            f"{self._format_time(event.end_ms)} 报告：{event.message}\n\n"
            "是否确认这段时间存在数据缺失？确认后会写入统一时间线。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            self._prompted_source_event_ids.discard(event.id)
            return
        try:
            self.service.confirm_data_gap(event.session_id, event.id)
        except Exception as exc:  # noqa: BLE001 - UI boundary surfaces persistence failures
            self._show_action_error(str(exc))
            self._prompted_source_event_ids.discard(event.id)
            return
        current = self.session_library.currentItem()
        if current is not None and current.data(Qt.ItemDataRole.UserRole) == event.session_id:
            self._open_session_item(current)
        self._show_worker_warning(
            f"已确认数据缺口：{event.source} · {self._format_time(event.start_ms)}–"
            f"{self._format_time(event.end_ms)}"
        )

    @Slot(str)
    def _show_action_error(self, message: str) -> None:
        self._material_generation_in_flight = False
        compact = self._compact_message(message)
        self._set_status("操作失败", "error")
        self.notice_text.setText(compact)
        self.notice.show()
        self.output.set_markdown(f"## 请求未完成\n\n{compact}")
        self.ask_button.setEnabled(True)
        self._refresh_reanswer_target()
        self.summary_button.setEnabled(True)
        self._refresh_material_controls()
        self.test_provider_button.setEnabled(True)
        self.voice_button.setEnabled(True)
        self.voice_button.setText("按住说话")
        if self.service.is_recording:
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(self.service.supports_pause)
        self._refresh_invocation_audit()

    @Slot(int, str, str)
    def _show_answer(
        self, question_id: int, answer: str, origin_session_id: str | None = None
    ) -> None:
        if origin_session_id and origin_session_id != self._selected_session_id:
            self.ask_button.setEnabled(True)
            self._refresh_reanswer_target()
            self._show_worker_warning("回答已保存到原会话；当前浏览位置未切换。")
            return
        self._content_kind = "answer"
        self._last_answer = answer
        self.output.set_markdown(answer)
        self.ask_button.setEnabled(True)
        self.speak_button.setEnabled(bool(answer.strip()))
        current = self.session_library.currentItem()
        if self._selected_session_id is not None:
            completed_answers = [
                item
                for item in self.service.session_answers(self._selected_session_id)
                if item.question_id == question_id
            ]
            if completed_answers:
                self._selected_answer_version_id = max(
                    completed_answers, key=lambda item: (item.version_number, item.id)
                ).id
                if current is not None:
                    self._open_session_item(current)
            else:
                self._refresh_reanswer_target()
        else:
            self._refresh_reanswer_target()
        self._set_status("回答完成", "success")
        self._refresh_invocation_audit()

    @Slot()
    def _speak_answer(self) -> None:
        if not self._last_answer:
            return
        if self._speech is None:
            self._speech = QTextToSpeech()
        self._speech.say(self._last_answer)

    @Slot(str)
    def _show_summary(self, summary: str) -> None:
        self.output.set_markdown(summary)
        self.summary_button.setEnabled(True)
        self._set_status("材料已生成", "success")
        self._refresh_invocation_audit()

    @Slot(object)
    def _show_material(self, material: SessionMaterialVersionRecord) -> None:
        self._material_generation_in_flight = False
        if material.session_id != self._selected_session_id:
            self._set_status(f"材料已保存 · {material.session_id[:8]}", "success")
            self._refresh_material_controls()
            self._refresh_invocation_audit()
            return
        self._materials_by_id[material.id] = material
        self._selected_material_version_id = material.id
        self._content_kind = "material"
        self._populate_material_selector(list(self._materials_by_id.values()))
        self._show_selected_material()
        self._refresh_material_controls()
        self._set_status(f"材料版本 {material.version_number} 已生成", "success")
        self._refresh_invocation_audit()

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
        self._refresh_invocation_audit()

    @Slot(str)
    def _stop_failed(self, message: str) -> None:
        self._stop_in_flight = False
        self._show_action_error(message)
        if self.service.is_recording:
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(self.service.supports_pause)
        else:
            self._refresh_recording_status()

    def _choose_material_generation_mode(self) -> MaterialGenerationMode | None:
        labels = ["始终自动生成", "每次结束时询问", "仅手动生成"]
        choice, accepted = QInputDialog.getItem(
            self,
            "选择会话材料生成策略",
            "以后结束会话时：",
            labels,
            1,
            False,
        )
        if not accepted:
            return None
        mode = {
            labels[0]: MaterialGenerationMode.ALWAYS,
            labels[1]: MaterialGenerationMode.ASK,
            labels[2]: MaterialGenerationMode.MANUAL,
        }[choice]
        setter = getattr(self.manager, "set_material_generation_mode", None)
        if callable(setter):
            setter(mode)
        return mode

    def _maybe_generate_material_after_stop(self, session_id: str) -> None:
        getter = getattr(self.manager, "material_generation_mode", None)
        setter = getattr(self.manager, "set_material_generation_mode", None)
        generator = getattr(self.service, "generate_material", None)
        if not callable(getter) or not callable(setter) or not callable(generator):
            return
        mode = getter()
        if mode is None:
            mode = self._choose_material_generation_mode()
        if mode is None or mode == MaterialGenerationMode.MANUAL:
            return
        if not self._confirm_material_generation(
            session_id, automatic=mode == MaterialGenerationMode.ALWAYS
        ):
            return
        self._start_material_generation(session_id)

    @Slot(str)
    def _recording_stopped(self, session_id: str) -> None:
        self._stop_in_flight = False
        self._question_active = False
        self._question_generation += 1
        self.question.clear()
        self.voice_button.setEnabled(True)
        self.voice_button.setText("按住说话")
        session = self.service.database.get_session(session_id)
        interrupted = session is not None and session.status == "interrupted"
        self._set_status(
            f"{'已结束但仍有线程运行' if interrupted else '已结束'} · {session_id[:8]}",
            "error" if interrupted else "success",
        )
        if interrupted:
            self._show_worker_warning("会话已标记为中断；仍运行的线程将自行退出后再允许清理。")
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self._paused = False
        self.system_audio_check.setEnabled(True)
        self.microphone_check.setEnabled(True)
        self.correction_check.setEnabled(True)
        self.correction_window_input.setEnabled(True)
        self._refresh_recording_status()
        self._refresh_sessions(session_id)
        if not interrupted:
            self._maybe_generate_material_after_stop(session_id)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._storage_dialog is not None and self._storage_dialog.operation_active:
            self._storage_dialog.show()
            self._storage_dialog.raise_()
            self._show_action_error("存储迁移正在进行，请等待完成后再退出境织。")
            event.ignore()
            return
        if self._question_active:
            self.service.cancel_question()
        configure_provider = getattr(self.manager, "configure_provider", None)
        save_provider = getattr(self.manager, "save_provider", None)
        if callable(configure_provider) and callable(save_provider):
            try:
                configure_provider(self._provider_settings_from_form())
                save_provider()
            except Exception:
                logger.exception("Could not save provider settings while closing")
        save_whisper = getattr(self.manager, "save_whisper", None)
        if callable(save_whisper):
            try:
                save_whisper()
            except Exception:
                logger.exception("Could not save Whisper settings while closing")
        if self.service.is_recording:
            self.service.stop_session()
        event.accept()


def run_app(settings: Settings) -> int:
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    window = MainWindow(settings)
    window.show()
    if settings.startup_settings_store is not None:
        settings.startup_settings_store.complete_successful_startup(settings.data_dir)
    return application.exec()
