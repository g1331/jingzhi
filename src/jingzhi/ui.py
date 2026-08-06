from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import ClassVar

from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPoint,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QPixmapCache, QShortcut
from PySide6.QtTextToSpeech import QTextToSpeech
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QKeySequenceEdit,
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
    QStackedWidget,
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
from jingzhi.cross_session import CrossSessionSynthesisError
from jingzhi.database import (
    AnswerEvidenceRecord,
    CrossSessionEvidenceRecord,
    CrossSessionSearchResult,
    MaterialEvidenceRecord,
    SessionAnswerRecord,
    SessionMaterialVersionRecord,
    SessionNotificationKind,
    SessionRecord,
    SourceEventRecord,
    TimelineFrameRecord,
    TimelineTranscriptRecord,
)
from jingzhi.diagnostics import format_runtime_metrics
from jingzhi.material_settings import MaterialGenerationMode
from jingzhi.model_roles import (
    ModelConnection,
    ModelFallback,
    ModelRole,
    ReasoningLevel,
    RoleName,
)
from jingzhi.onboarding import (
    DEFAULT_QUESTION_SHORTCUT,
    ONBOARDING_STEPS,
    OnboardingSettingsStore,
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
from jingzhi.whisper_settings import (
    PROFILE_PRESETS,
    WhisperProfile,
    create_builtin_whisper_sample,
)
from jingzhi.whisper_ui import WhisperSettingsDialog

logger = logging.getLogger(__name__)


def _confirm_recording_selection(
    parent: QWidget,
    manager: object,
    settings: Settings,
    *,
    default_system_audio_enabled: bool,
    default_microphone_enabled: bool,
) -> RecordingSelection | None:
    catalog = getattr(manager, "device_catalog", None) or WindowsDeviceCatalog()
    dialog = RecordingConfirmationDialog(
        catalog,
        RecordingSettingsStore(settings.data_dir),
        screen_interval_s=settings.screen_interval_s,
        audio_storage_rate=settings.audio_storage_rate,
        default_system_audio_enabled=default_system_audio_enabled,
        default_microphone_enabled=default_microphone_enabled,
        parent=parent,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.recording_selection()


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
QLabel#statusPill[state="busy"] {
    color: #c9d7d2;
    background: #1d292b;
    border-color: #3d5758;
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
QLabel#capsuleDragHandle {
    color: #9ee7ca; font-weight: 700; padding: 0 2px;
}
QLabel#capsuleDragHandle:hover { color: #dff5e9; }
QPushButton#capsuleHideButton {
    min-width: 26px; max-width: 26px; min-height: 27px; padding: 0;
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
    archive_finished = Signal(str, str)
    archive_failed = Signal(str, str)
    archive_preview_finished = Signal(object)
    archive_preview_failed = Signal(str)
    audio_recovery_finished = Signal(object)
    audio_recovery_failed = Signal(str)
    correction_retry_finished = Signal(int)
    correction_retry_failed = Signal(str)
    runtime_metrics_ready = Signal(object)
    runtime_metrics_failed = Signal(str)


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


class CrossSessionSynthesisDialog(QDialog):
    synthesis_finished = Signal(object)
    synthesis_failed = Signal(object)

    KIND_LABELS: ClassVar[dict[str, str]] = {
        "transcript": "字幕版本",
        "frame": "关键帧",
        "answer": "问答版本",
        "material": "材料版本",
    }

    def __init__(
        self,
        service: JingzhiApplicationService,
        *,
        navigate_callback: Callable[[CrossSessionEvidenceRecord], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.navigate_callback = navigate_callback
        self.setWindowTitle("跨会话综合")
        self.resize(980, 700)
        self._results: dict[str, CrossSessionSearchResult] = {}
        self._selected_ids: set[str] = set()
        self._selection_order: list[str] = []
        self._candidate_by_id: dict[str, CrossSessionEvidenceRecord] = {}
        self._failed_synthesis_id: int | None = None
        self._busy = False
        self._thread: threading.Thread | None = None

        root = QVBoxLayout(self)
        heading = QLabel("跨会话搜索与综合")
        heading.setObjectName("sectionTitle")
        root.addWidget(heading)
        hint = QLabel(
            "只会把你在下方明确勾选的字幕、关键帧、问答或材料发送给深度分析模型；"
            "回收区会话不会出现在搜索结果中。"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        search_row = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setObjectName("crossSessionQuery")
        self.query_input.setPlaceholderText("搜索字幕、问答或材料中的关键词")
        self.search_button = QPushButton("搜索")
        search_row.addWidget(self.query_input, 1)
        search_row.addWidget(self.search_button)
        root.addLayout(search_row)

        lists = QSplitter(Qt.Orientation.Horizontal)
        self.result_list = QListWidget()
        self.result_list.setObjectName("crossSessionResults")
        self.result_list.setToolTip("选择搜索结果后查看可授权证据")
        self.evidence_list = QListWidget()
        self.evidence_list.setObjectName("crossSessionEvidence")
        self.evidence_list.setToolTip("勾选后才会授权给模型")
        lists.addWidget(self.result_list)
        lists.addWidget(self.evidence_list)
        lists.setSizes([430, 510])
        root.addWidget(lists, 1)

        self.result_hint = QLabel("输入关键词后搜索。")
        self.result_hint.setObjectName("hint")
        self.result_hint.setWordWrap(True)
        root.addWidget(self.result_hint)

        question_row = QHBoxLayout()
        question_row.addWidget(QLabel("综合问题"))
        self.question_input = QLineEdit()
        self.question_input.setObjectName("crossSessionQuestion")
        self.question_input.setPlaceholderText("例如：比较这些会话中反复出现的问题")
        question_row.addWidget(self.question_input, 1)
        root.addLayout(question_row)

        self.authorization_status = QLabel("尚未选择证据")
        self.authorization_status.setObjectName("answerEvidenceStatus")
        self.authorization_status.setWordWrap(True)
        root.addWidget(self.authorization_status)

        self.synthesize_button = QPushButton("确认证据并开始综合")
        self.synthesize_button.setProperty("role", "primary")
        self.synthesize_button.setEnabled(False)
        root.addWidget(self.synthesize_button)
        self.retry_button = QPushButton("重试上次失败的综合")
        self.retry_button.setObjectName("crossSessionRetry")
        self.retry_button.clicked.connect(self._retry_failed_synthesis)
        self.retry_button.hide()
        root.addWidget(self.retry_button)

        self.output = MarkdownDocument()
        self.output.setObjectName("crossSessionOutput")
        self.output.setMinimumHeight(180)
        root.addWidget(self.output, 1)
        self.output.hide()

        self.navigate_button = QPushButton("跳转当前证据")
        self.navigate_button.setEnabled(False)
        root.addWidget(self.navigate_button)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.search_button.clicked.connect(self._search)
        self.query_input.returnPressed.connect(self._search)
        self.result_list.currentItemChanged.connect(
            lambda current, _previous: self._show_result_evidence(current)
        )
        self.evidence_list.itemChanged.connect(self._evidence_changed)
        self.evidence_list.itemDoubleClicked.connect(lambda _item: self._navigate_current())
        self.question_input.textChanged.connect(self._update_preview)
        self.synthesize_button.clicked.connect(self._synthesize)
        self.navigate_button.clicked.connect(self._navigate_current)
        self.synthesis_finished.connect(self._show_synthesis)
        self.synthesis_failed.connect(self._show_synthesis_error)
        self._load_retry_queue()

    def _load_retry_queue(self) -> None:
        method = getattr(self.service, "failed_cross_session_syntheses", None)
        if not callable(method):
            return
        try:
            failed = method(limit=1)
        except Exception:  # noqa: BLE001 - optional retry queue boundary
            return
        if failed:
            self._failed_synthesis_id = failed[0].id
            self.retry_button.setEnabled(True)
            self.retry_button.show()
            evidence_method = getattr(self.service, "cross_session_synthesis_evidence", None)
            if callable(evidence_method):
                try:
                    saved = evidence_method(failed[0].id)
                except Exception:  # noqa: BLE001 - optional retry queue boundary
                    saved = ()
                self._selection_order = [item.stable_id for item in saved]
                self._selected_ids = set(self._selection_order)
                self._populate_saved_evidence()
            self.authorization_status.setText("有一次失败的跨会话综合等待重试。")

    @Slot()
    def _search(self) -> None:
        self._failed_synthesis_id = None
        self.retry_button.hide()
        self.output.set_markdown("")
        self.output.hide()
        self.navigate_button.setEnabled(False)
        query = self.query_input.text().strip()
        try:
            results = self.service.cross_session_search(query, limit=50)
        except Exception as exc:  # noqa: BLE001 - application boundary
            self._selected_ids.clear()
            self._selection_order.clear()
            self.result_list.clear()
            self.evidence_list.clear()
            self.result_hint.setText(f"搜索失败：{exc}")
            self._update_preview()
            return
        self._selected_ids.clear()
        self._selection_order.clear()
        self._results = {item.stable_id: item for item in results}
        self.result_list.clear()
        self.evidence_list.clear()
        if not results:
            self._selected_ids.clear()
            self._selection_order.clear()
            self._candidate_by_id.clear()
            self.result_hint.setText("没有找到未回收会话中的匹配内容。")
            self._update_preview()
            return
        self.result_hint.setText(f"找到 {len(results)} 条结果；选择一条结果查看可授权证据。")
        for result in results:
            item = QListWidgetItem(self._result_text(result))
            item.setData(Qt.ItemDataRole.UserRole, result)
            self.result_list.addItem(item)
        self.result_list.setCurrentRow(0)

    def _result_text(self, result: CrossSessionSearchResult) -> str:
        version = f" · {result.version_kind}" if result.version_kind else ""
        return (
            f"[{self.KIND_LABELS.get(result.kind, result.kind)}{version}] "
            f"{result.session_title} · {result.start_ms / 1000:.1f}s · {result.stable_id}\n"
            f"{result.snippet}"
        )

    @Slot(QListWidgetItem, QListWidgetItem)
    def _show_result_evidence(
        self, item: QListWidgetItem | None, _previous: QListWidgetItem | None = None
    ) -> None:
        self.evidence_list.blockSignals(True)
        self.evidence_list.clear()
        self._candidate_by_id.clear()
        if item is None:
            self.evidence_list.blockSignals(False)
            self._update_preview()
            return
        result = item.data(Qt.ItemDataRole.UserRole)
        try:
            candidates = self.service.cross_session_evidence_candidates((result.stable_id,))
        except Exception as exc:  # noqa: BLE001 - application boundary
            self.result_hint.setText(f"证据预览失败：{exc}")
            candidates = []
        for candidate in candidates:
            self._candidate_by_id[candidate.stable_id] = candidate
            evidence_item = QListWidgetItem(self._evidence_text(candidate))
            evidence_item.setData(Qt.ItemDataRole.UserRole, candidate)
            evidence_item.setFlags(
                evidence_item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
            )
            evidence_item.setCheckState(
                Qt.CheckState.Checked
                if candidate.stable_id in self._selected_ids
                else Qt.CheckState.Unchecked
            )
            self.evidence_list.addItem(evidence_item)
        self.evidence_list.blockSignals(False)
        self._update_preview()

    def _evidence_text(self, candidate: CrossSessionEvidenceRecord) -> str:
        detail = candidate.content_text or "（关键帧图像）"
        return (
            f"[{self.KIND_LABELS.get(candidate.kind, candidate.kind)}] "
            f"{candidate.session_title} · {candidate.start_ms / 1000:.1f}s · {candidate.source} · "
            f"{candidate.stable_id}\n{detail[:180]}"
        )

    @Slot(QListWidgetItem)
    def _evidence_changed(self, item: QListWidgetItem) -> None:
        stable_id = item.data(Qt.ItemDataRole.UserRole).stable_id
        if item.checkState() == Qt.CheckState.Checked:
            self._selected_ids.add(stable_id)
            if stable_id not in self._selection_order:
                self._selection_order.append(stable_id)
        else:
            self._selected_ids.discard(stable_id)
            self._selection_order = [
                selected_id for selected_id in self._selection_order if selected_id != stable_id
            ]
        self._update_preview()

    def _selected_stable_ids(self) -> tuple[str, ...]:
        return tuple(
            stable_id for stable_id in self._selection_order if stable_id in self._selected_ids
        )

    @Slot()
    def _update_preview(self) -> None:
        try:
            preview = self.service.cross_session_synthesis_preview(
                self.question_input.text(), self._selected_stable_ids()
            )
        except Exception as exc:  # noqa: BLE001 - application boundary
            self.authorization_status.setText(f"无法检查综合条件：{exc}")
            self.synthesize_button.setEnabled(False)
            return
        if preview.can_synthesize:
            self.authorization_status.setText(
                f"已授权 {preview.evidence_count} 条证据，共 {preview.character_count} 字；"
                f"关键帧 {preview.frame_count} 个。模型：{preview.model} · 连接：{preview.connection_name}"
            )
        else:
            self.authorization_status.setText(preview.reason or "当前不能开始综合。")
        self.synthesize_button.setEnabled(preview.can_synthesize and not self._busy)

    @Slot()
    def _synthesize(self) -> None:
        if self._busy:
            return
        question = self.question_input.text().strip()
        stable_ids = self._selected_stable_ids()
        preview = self.service.cross_session_synthesis_preview(question, stable_ids)
        if not preview.can_synthesize:
            self._update_preview()
            return
        confirmation = QMessageBox.question(
            self,
            "确认综合范围",
            (
                f"将把 {preview.evidence_count} 条已选证据（{preview.character_count} 字、"
                f"{preview.frame_count} 个关键帧）发送给深度分析模型。\n"
                f"模型：{preview.model}\n连接：{preview.connection_name}\n\n继续吗？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return
        self._failed_synthesis_id = None
        self.retry_button.hide()
        self._start_synthesis(lambda: self.service.synthesize_cross_session(question, stable_ids))

    def _start_synthesis(self, operation: Callable[[], object]) -> None:
        self._busy = True
        self.synthesize_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        self.authorization_status.setText("正在综合已授权证据……")

        def run() -> None:
            try:
                self.synthesis_finished.emit(operation())
            except Exception as exc:  # noqa: BLE001 - model boundary
                self.synthesis_failed.emit(exc)

        self._thread = threading.Thread(target=run, name="cross-session-synthesis", daemon=True)
        self._thread.start()

    @Slot()
    def _retry_failed_synthesis(self) -> None:
        if self._busy or self._failed_synthesis_id is None:
            return
        synthesis_id = self._failed_synthesis_id
        self._start_synthesis(lambda: self.service.retry_cross_session_synthesis(synthesis_id))

    @Slot(object)
    def _show_synthesis(self, result: object) -> None:
        self._busy = False
        self._failed_synthesis_id = None
        self.retry_button.hide()
        self.output.set_markdown(result.answer or "")
        self.output.show()
        self.authorization_status.setText(
            f"综合已保存：模型 {result.model or '未知'} · 证据状态 {result.evidence_state}"
        )
        self._update_preview()
        self._populate_saved_evidence()

    @Slot(object)
    def _show_synthesis_error(self, error: object) -> None:
        self._busy = False
        message = str(error)
        lowered = message.lower()
        if any(
            marker in lowered
            for marker in (
                "context length",
                "maximum context",
                "context_length",
                "too many tokens",
                "token limit",
                "上下文",
                "超限",
            )
        ):
            message = f"模型上下文超限：{message}。请减少已授权证据后重试。"
        self.authorization_status.setText(f"综合失败：{message}")
        if isinstance(error, CrossSessionSynthesisError):
            self._failed_synthesis_id = error.synthesis_id
            self.retry_button.setEnabled(True)
            self.retry_button.show()
        else:
            self._failed_synthesis_id = None
            self.retry_button.hide()
        self._update_preview()

    def _populate_saved_evidence(self) -> None:
        selected: list[CrossSessionEvidenceRecord] = []
        for stable_id in self._selected_stable_ids():
            candidates = self.service.cross_session_evidence_candidates((stable_id,))
            candidate = next((item for item in candidates if item.stable_id == stable_id), None)
            if candidate is not None:
                selected.append(candidate)
        self._candidate_by_id = {item.stable_id: item for item in selected}
        self.evidence_list.blockSignals(True)
        self.evidence_list.clear()
        for candidate in selected:
            item = QListWidgetItem(self._evidence_text(candidate))
            item.setData(Qt.ItemDataRole.UserRole, candidate)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(Qt.CheckState.Checked)
            self.evidence_list.addItem(item)
        self.evidence_list.blockSignals(False)
        if selected:
            self.evidence_list.setCurrentRow(0)
        self.navigate_button.setEnabled(bool(selected) and self.navigate_callback is not None)

    @Slot()
    def _navigate_current(self) -> None:
        if self.navigate_callback is None:
            return
        item = self.evidence_list.currentItem()
        if item is None:
            return
        candidate = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(candidate, CrossSessionEvidenceRecord):
            self.navigate_callback(candidate)


class RecordingCapsulePositionStore:
    VERSION = 1

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "recording-capsule.json"

    def load(self) -> tuple[int, int] | None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        if not isinstance(document, dict) or document.get("version") != self.VERSION:
            return None
        x = document.get("x")
        y = document.get("y")
        if type(x) is not int or type(y) is not int:
            return None
        return x, y

    def save(self, position: QPoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"version": self.VERSION, "x": position.x(), "y": position.y()},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class CapsuleDragHandle(QLabel):
    drag_finished = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__("境织", parent)
        self.setObjectName("capsuleDragHandle")
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip("拖动以移动录制胶囊")
        self._drag_offset: QPoint | None = None

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        window = self.window()
        self._drag_offset = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            self.drag_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class RecordingCapsule(QFrame):
    def __init__(
        self,
        *,
        default_system_audio_enabled: bool,
        default_microphone_enabled: bool,
        pause_enabled: bool,
        floating: bool = False,
        position_store: RecordingCapsulePositionStore | None = None,
        parent: QWidget | None = None,
        object_name: str = "recordingCapsule",
    ) -> None:
        super().__init__(parent)
        self._floating = floating
        self._position_store = position_store
        self._position_restored = False
        self._allow_close = False
        self.setObjectName(object_name)
        self.setProperty("floating", floating)
        if self._floating:
            self.setWindowFlags(
                Qt.WindowType.Tool
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
            )
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.setWindowTitle("境织 · 录制胶囊")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 6)
        layout.setSpacing(7)
        self.drag_handle: CapsuleDragHandle | None = None
        self.hide_button: QPushButton | None = None
        if self._floating:
            self.drag_handle = CapsuleDragHandle(self)
            layout.addWidget(self.drag_handle)
        self.status = QLabel("空闲")
        self.status.setObjectName("statusPill")
        self.status.setProperty("state", "idle")
        self.title_input = QLineEdit("新会话")
        self.title_input.setPlaceholderText("会话标题")
        self.title_input.setMaximumWidth(150)
        self.system_audio_check = QCheckBox("系统声音")
        self.system_audio_check.setChecked(default_system_audio_enabled)
        self.microphone_check = QCheckBox("麦克风")
        self.microphone_check.setChecked(default_microphone_enabled)
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(pause_enabled)
        if not pause_enabled:
            self.pause_button.setToolTip("当前采集适配器尚未提供暂停能力")
        self.capsule_ask_button = QPushButton("提问")
        self.start_button = QPushButton("开始记录")
        self.start_button.setProperty("role", "primary")
        self.stop_button = QPushButton("结束")
        self.stop_button.setProperty("role", "danger")
        self.stop_button.setEnabled(False)
        if self._floating:
            self.hide_button = QPushButton("×")
            self.hide_button.setObjectName("capsuleHideButton")
            self.hide_button.setProperty("role", "quiet")
            self.hide_button.setToolTip("隐藏胶囊，不会结束当前会话")
            self.hide_button.setVisible(False)
            self.hide_button.clicked.connect(self._hide)
        for widget in (
            self.status,
            self.title_input,
            self.system_audio_check,
            self.microphone_check,
            self.pause_button,
            self.capsule_ask_button,
            self.start_button,
            self.stop_button,
        ):
            layout.addWidget(widget)
        if self.hide_button is not None:
            layout.addWidget(self.hide_button)
            self.drag_handle.drag_finished.connect(self._save_position)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if not self._floating or self._position_restored:
            return
        self.adjustSize()
        saved = self._position_store.load() if self._position_store is not None else None
        if saved is None:
            screen = QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                self.move(
                    available.left() + max(0, (available.width() - self.width()) // 2),
                    available.top() + 18,
                )
        else:
            self.move(self._clamp_position(QPoint(*saved)))
        self._position_restored = True

    def _clamp_position(self, position: QPoint) -> QPoint:
        available = None
        for screen in QApplication.screens():
            geometry = screen.availableGeometry()
            if geometry.contains(position):
                available = geometry
                break
        if available is None:
            screen = QApplication.primaryScreen()
            available = screen.availableGeometry() if screen is not None else None
        if available is None:
            return position
        maximum_x = max(available.left(), available.right() - self.width() + 1)
        maximum_y = max(available.top(), available.bottom() - self.height() + 1)
        return QPoint(
            min(max(position.x(), available.left()), maximum_x),
            min(max(position.y(), available.top()), maximum_y),
        )

    def _hide(self) -> None:
        self._save_position()
        self.hide()

    def _save_position(self) -> None:
        if not self._floating or self._position_store is None:
            return
        try:
            self._position_store.save(self.pos())
        except OSError:
            logger.warning("Could not save recording capsule position", exc_info=True)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._floating and not self._allow_close:
            self._save_position()
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)

    def shutdown(self) -> None:
        if not self._floating:
            return
        self._allow_close = True
        self._save_position()
        self.close()


class OnboardingDialog(QDialog):
    provider_tested = Signal(str)
    whisper_benchmarked = Signal(object)
    task_failed = Signal(str, str)

    def __init__(
        self,
        manager: object,
        settings: Settings,
        *,
        parent: QWidget | None = None,
        state_store: OnboardingSettingsStore | None = None,
        question_callback: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.manager = manager
        self.settings = settings
        self.state_store = state_store or OnboardingSettingsStore(settings.data_dir)
        self.state = self.state_store.load()
        self._question_callback = question_callback
        self._recording_selection: RecordingSelection | None = None
        self._provider_test_creating = False
        self._onboarding_task: str | None = None
        self._provider_settings_before_test: SavedProviderSettings | None = None
        self._whisper_settings_before_test = None
        self._whisper_dialog: WhisperSettingsDialog | None = None
        self._shortcut = QShortcut(QKeySequence(self.state.question_shortcut), self)
        self._shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._shortcut.activated.connect(self._shortcut_triggered)
        self.setModal(True)
        self.setWindowTitle("境织 · 首次使用引导")
        self.setMinimumSize(760, 560)
        self.resize(820, 640)

        root = QVBoxLayout(self)
        heading = QLabel("先把境织设置成适合你的工作台")
        heading.setObjectName("appTitle")
        root.addWidget(heading)
        self.progress_label = QLabel()
        self.progress_label.setObjectName("hint")
        root.addWidget(self.progress_label)

        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self._build_privacy_page()
        self._build_provider_page()
        self._build_whisper_page()
        self._build_recording_page()
        self._build_shortcuts_page()

        navigation = QHBoxLayout()
        self.later_button = QPushButton("稍后继续")
        self.back_button = QPushButton("上一步")
        self.skip_button = QPushButton("跳过此步")
        self.next_button = QPushButton("下一步")
        self.next_button.setProperty("role", "primary")
        navigation.addWidget(self.later_button)
        navigation.addStretch(1)
        navigation.addWidget(self.back_button)
        navigation.addWidget(self.skip_button)
        navigation.addWidget(self.next_button)
        root.addLayout(navigation)

        self.later_button.clicked.connect(self.reject)
        self.back_button.clicked.connect(self._previous_page)
        self.skip_button.clicked.connect(self._skip_page)
        self.next_button.clicked.connect(self._next_page)
        self.provider_tested.connect(self._provider_test_succeeded)
        self.whisper_benchmarked.connect(self._whisper_benchmark_succeeded)
        self.task_failed.connect(self._task_failed)

        self._hydrate_existing_configuration()
        self._set_page(self.state.step_index)

    @property
    def recording_selection(self) -> RecordingSelection | None:
        if self._recording_selection is not None:
            return self._recording_selection
        if self.state.recording_confirmed:
            preferences = RecordingSettingsStore(self.settings.data_dir).load()
            self._recording_selection = RecordingSelection(
                display_ids=preferences.display_ids,
                system_audio_id=(
                    preferences.system_audio_id if preferences.system_audio_enabled else None
                ),
                microphone_id=(
                    preferences.microphone_id if preferences.microphone_enabled else None
                ),
                estimated_duration_minutes=preferences.estimated_duration_minutes,
            )
        return self._recording_selection

    def _save_state(self, **changes: object) -> None:
        self.state = replace(self.state, **changes)
        self.state_store.save(self.state)
        self._update_navigation()

    def _hydrate_existing_configuration(self) -> None:
        provider_settings = getattr(self.manager, "provider_settings", None)
        connections = getattr(provider_settings, "connections", ())
        utility_role = next(
            (
                role
                for role in getattr(provider_settings, "roles", ())
                if role.name == RoleName.UTILITY
            ),
            None,
        )
        utility_connection = next(
            (
                connection
                for connection in connections
                if utility_role is None or connection.id == utility_role.connection_id
            ),
            None,
        )
        if utility_connection is not None and (
            bool(getattr(utility_connection, "base_url", "").strip())
            or bool(getattr(utility_connection, "api_key", ""))
        ):
            self.state = replace(self.state, provider_completed=True, provider_skipped=False)
        whisper_settings = getattr(self.manager, "whisper_settings", None)
        if getattr(whisper_settings, "first_run_completed", False):
            self.state = replace(self.state, whisper_completed=True, whisper_skipped=False)
        recording_store = RecordingSettingsStore(self.settings.data_dir)
        if recording_store.path.is_file():
            self.state = replace(self.state, recording_confirmed=True)
        self.state_store.save(self.state)
        if self.state.recording_confirmed:
            self.recording_status.setText("已保存录制来源配置；重新打开可以修改全部选择。")

    def _new_page(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("hint")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
        self.pages.addWidget(page)
        return page, layout

    def _build_privacy_page(self) -> None:
        _page, layout = self._new_page(
            "隐私与数据生命周期",
            "境织默认把会话保存在本机；只有你确认发送的模型任务才会上传完成任务所需的最小上下文。",
        )
        notice = QLabel(
            "• 会话、字幕、关键帧、音频和模型调用记录默认保存在本机。\n"
            "• 发送模型任务前会显示发送范围；应用不会无边界读取整个会话库。\n"
            "• 未固定会话默认保留 30 天，通知后进入回收区，再保留 7 天后最终删除。\n"
            "• 你可以固定重要会话、导出单个会话或创建完整备份。"
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        layout.addStretch(1)
        self.privacy_check = QCheckBox("我已了解本地保存、最小上传、30 天保留和 7 天回收规则")
        self.privacy_check.setObjectName("onboardingPrivacyAcknowledgement")
        self.privacy_check.setChecked(self.state.privacy_acknowledged)
        self.privacy_check.toggled.connect(
            lambda checked: self._save_state(privacy_acknowledged=checked)
        )
        layout.addWidget(self.privacy_check)

    def _build_provider_page(self) -> None:
        _page, layout = self._new_page(
            "模型连接",
            "模型连接用于问答和材料任务。你可以现在测试，也可以先跳过，只使用本地采集与转写。",
        )
        provider_settings = getattr(self.manager, "provider_settings", None)
        connections = list(getattr(provider_settings, "connections", ()))
        utility_role = next(
            (
                role
                for role in getattr(provider_settings, "roles", ())
                if role.name == RoleName.UTILITY
            ),
            None,
        )
        utility_connection_id = getattr(utility_role, "connection_id", None)
        connection = next(
            (item for item in connections if item.id == utility_connection_id),
            connections[0] if connections else ModelConnection("default", "默认连接"),
        )
        self._provider_target_connection_id = connection.id
        group = QGroupBox("Utility 角色连接")
        form = QGridLayout(group)
        self.onboarding_provider_name = QLineEdit(connection.name)
        self.onboarding_provider_name.setObjectName("onboardingProviderName")
        self.onboarding_provider_url = QLineEdit(connection.base_url)
        self.onboarding_provider_url.setObjectName("onboardingProviderUrl")
        self.onboarding_provider_api_mode = QComboBox()
        self.onboarding_provider_api_mode.addItem("Responses API", "responses")
        self.onboarding_provider_api_mode.addItem("Chat Completions", "chat_completions")
        self.onboarding_provider_api_mode.setCurrentIndex(
            max(0, self.onboarding_provider_api_mode.findData(connection.api_mode))
        )
        self.onboarding_provider_key = QLineEdit(connection.api_key)
        self.onboarding_provider_key.setObjectName("onboardingProviderKey")
        self.onboarding_provider_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.onboarding_provider_model = QLineEdit(
            utility_role.model if utility_role is not None else "gpt-5.5"
        )
        self.onboarding_provider_model.setObjectName("onboardingProviderModel")
        fields = (
            ("连接名称", self.onboarding_provider_name),
            ("Base URL", self.onboarding_provider_url),
            ("接口类型", self.onboarding_provider_api_mode),
            ("API Key", self.onboarding_provider_key),
            ("测试模型", self.onboarding_provider_model),
        )
        for row, (label, field) in enumerate(fields):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(field, row, 1)
        layout.addWidget(group)
        self.provider_status = QLabel()
        self.provider_status.setObjectName("onboardingProviderStatus")
        self.provider_status.setWordWrap(True)
        layout.addWidget(self.provider_status)
        provider_buttons = QHBoxLayout()
        self.provider_test_button = QPushButton("保存并测试连接")
        self.provider_test_button.setObjectName("onboardingTestProvider")
        self.provider_create_button = QPushButton("创建为新连接并测试")
        self.provider_create_button.setObjectName("onboardingCreateProvider")
        self.provider_test_button.clicked.connect(self._test_provider)
        self.provider_create_button.clicked.connect(lambda: self._test_provider(create_new=True))
        provider_buttons.addWidget(self.provider_test_button)
        provider_buttons.addWidget(self.provider_create_button)
        provider_buttons.addStretch(1)
        layout.addLayout(provider_buttons)
        layout.addStretch(1)

    def _build_whisper_page(self) -> None:
        _page, layout = self._new_page(
            "本地 Whisper",
            "选择本地转写档位。样本测试会下载或加载模型并报告识别、速度和资源结果；没有网络或模型不可用时也可以跳过。",
        )
        group = QGroupBox("转写档位")
        form = QGridLayout(group)
        self.onboarding_whisper_profile = QComboBox()
        self.onboarding_whisper_profile.setObjectName("onboardingWhisperProfile")
        for profile in WhisperProfile:
            self.onboarding_whisper_profile.addItem(PROFILE_PRESETS[profile].label, profile.value)
        current_whisper = getattr(self.manager, "whisper_settings", None)
        current_profile = getattr(current_whisper, "profile", WhisperProfile.BALANCED)
        self.onboarding_whisper_profile.setCurrentIndex(
            max(0, self.onboarding_whisper_profile.findData(current_profile.value))
        )
        self.whisper_impact = QLabel()
        self.whisper_impact.setObjectName("onboardingWhisperImpact")
        self.whisper_impact.setWordWrap(True)
        form.addWidget(QLabel("档位"), 0, 0)
        form.addWidget(self.onboarding_whisper_profile, 0, 1)
        form.addWidget(self.whisper_impact, 1, 0, 1, 2)
        layout.addWidget(group)
        self.whisper_status = QLabel()
        self.whisper_status.setObjectName("onboardingWhisperStatus")
        self.whisper_status.setWordWrap(True)
        layout.addWidget(self.whisper_status)
        buttons = QHBoxLayout()
        self.whisper_benchmark_button = QPushButton("运行内置中文样本")
        self.whisper_benchmark_button.setObjectName("onboardingBenchmarkWhisper")
        self.whisper_advanced_button = QPushButton("打开高级设置")
        self.whisper_advanced_button.setObjectName("onboardingAdvancedWhisper")
        buttons.addWidget(self.whisper_benchmark_button)
        buttons.addWidget(self.whisper_advanced_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(1)
        self.onboarding_whisper_profile.currentIndexChanged.connect(self._whisper_profile_changed)
        self.whisper_benchmark_button.clicked.connect(self._run_whisper_benchmark)
        self.whisper_advanced_button.clicked.connect(self._show_advanced_whisper)
        self._whisper_profile_changed()

    def _build_recording_page(self) -> None:
        _page, layout = self._new_page(
            "录制前确认",
            "开始首个会话前，境织会真实枚举显示器、系统声音和麦克风，显示缩略图、电平和预计存储占用。",
        )
        self.capsule_preview = RecordingCapsule(
            default_system_audio_enabled=self.settings.capture_system_audio,
            default_microphone_enabled=self.settings.capture_microphone,
            pause_enabled=False,
            object_name="recordingCapsulePreview",
        )
        self.capsule_preview_question = self.capsule_preview.capsule_ask_button
        self.capsule_preview_question.setObjectName("onboardingCapsuleQuestion")
        self.capsule_preview_start = self.capsule_preview.start_button
        self.capsule_preview_start.setObjectName("onboardingCapsuleStart")
        self.capsule_preview_status = QLabel("显示器 · 系统声音 · 麦克风")
        self.capsule_preview_status.setObjectName("onboardingCapsuleStatus")
        self.capsule_preview_question.clicked.connect(self._preview_question)
        self.capsule_preview_start.clicked.connect(self._open_recording_confirmation)
        layout.addWidget(self.capsule_preview)
        layout.addWidget(self.capsule_preview_status)
        capsule_hint = QLabel(
            "这里的开始按钮会打开真实录制确认；完成引导后，主窗口会显示同样的录制胶囊并直接进入首个会话。"
        )
        capsule_hint.setObjectName("hint")
        capsule_hint.setWordWrap(True)
        layout.addWidget(capsule_hint)
        self.recording_status = QLabel()
        self.recording_status.setObjectName("onboardingRecordingStatus")
        self.recording_status.setWordWrap(True)
        layout.addWidget(self.recording_status)
        self.open_recording_button = QPushButton("打开真实录制确认")
        self.open_recording_button.setObjectName("onboardingOpenRecordingConfirmation")
        self.open_recording_button.clicked.connect(self._open_recording_confirmation)
        layout.addWidget(self.open_recording_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        if self.state.recording_confirmed:
            self.recording_status.setText("已保存录制来源配置；重新打开可以修改全部选择。")
        else:
            self.recording_status.setText("尚未确认录制来源。")

    def _build_shortcuts_page(self) -> None:
        _page, layout = self._new_page(
            "快捷键与首个会话",
            "提问快捷键和录制胶囊是录制期间的即时控制入口。完成后会直接开始第一个会话。",
        )
        shortcut_group = QGroupBox("提问快捷键")
        shortcut_layout = QGridLayout(shortcut_group)
        shortcut_layout.addWidget(QLabel("当前快捷键"), 0, 0)
        self.onboarding_shortcut = QKeySequenceEdit(QKeySequence(self.state.question_shortcut))
        self.onboarding_shortcut.setObjectName("onboardingQuestionShortcut")
        self.onboarding_shortcut.keySequenceChanged.connect(self._shortcut_changed)
        shortcut_layout.addWidget(self.onboarding_shortcut, 0, 1)
        self.test_shortcut_button = QPushButton("按下快捷键测试")
        self.test_shortcut_button.setObjectName("onboardingTestShortcut")
        self.test_shortcut_button.clicked.connect(
            lambda: self.shortcut_status.setText(
                f"请在当前窗口按 {self.state.question_shortcut}；也可以直接点击“我已了解”。"
            )
        )
        shortcut_layout.addWidget(self.test_shortcut_button, 1, 0, 1, 2)
        layout.addWidget(shortcut_group)
        self.shortcut_status = QLabel()
        self.shortcut_status.setObjectName("onboardingShortcutStatus")
        self.shortcut_status.setWordWrap(True)
        layout.addWidget(self.shortcut_status)
        self.confirm_shortcut_button = QPushButton("我已了解并继续")
        self.confirm_shortcut_button.setObjectName("onboardingConfirmShortcut")
        self.confirm_shortcut_button.clicked.connect(self._confirm_shortcut)
        layout.addWidget(self.confirm_shortcut_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        self.shortcut_status.setText(f"完成引导后，{self.state.question_shortcut} 会聚焦提问框。")

    def _set_page(self, index: int) -> None:
        index = max(0, min(index, len(ONBOARDING_STEPS) - 1))
        self.state_store.save(replace(self.state, step=ONBOARDING_STEPS[index]))
        self.state = replace(self.state, step=ONBOARDING_STEPS[index])
        is_shortcut_page = index == len(ONBOARDING_STEPS) - 1
        self.setModal(not is_shortcut_page)
        self.setWindowModality(
            Qt.WindowModality.NonModal if is_shortcut_page else Qt.WindowModality.WindowModal
        )
        self.pages.setCurrentIndex(index)
        self.progress_label.setText(f"第 {index + 1} 步，共 {len(ONBOARDING_STEPS)} 步")
        self._update_navigation()

    def _step_ready(self) -> bool:
        index = self.pages.currentIndex()
        return (
            self.state.privacy_acknowledged,
            self.state.provider_ready,
            self.state.whisper_ready,
            self.state.recording_confirmed,
            self.state.shortcuts_completed,
        )[index]

    def _update_navigation(self) -> None:
        if not hasattr(self, "back_button"):
            return
        index = self.pages.currentIndex()
        task_running = self._onboarding_task is not None
        self.later_button.setEnabled(not task_running)
        self.back_button.setEnabled(index > 0 and not task_running)
        self.skip_button.setVisible(index in {1, 2})
        self.skip_button.setEnabled(not task_running)
        self.next_button.setEnabled(self._step_ready() and not task_running)
        self.next_button.setText(
            "完成并开始第一个会话" if index == len(ONBOARDING_STEPS) - 1 else "下一步"
        )

    def _next_page(self) -> None:
        if not self._step_ready():
            return
        index = self.pages.currentIndex()
        if index == 2:
            self._whisper_settings_before_test = self.manager.whisper_settings
            self._onboarding_task = "whisper"
            try:
                self._save_whisper_settings()
            except Exception as exc:  # noqa: BLE001 - Whisper settings boundary
                self._task_failed("whisper", str(exc))
                return
            self._onboarding_task = None
            self._whisper_settings_before_test = None
        if index == len(ONBOARDING_STEPS) - 1:
            self.state = replace(self.state, completed=False, step="shortcuts")
            self.state_store.save(self.state)
            self.accept()
            return
        self._set_page(index + 1)

    def _previous_page(self) -> None:
        self._set_page(self.pages.currentIndex() - 1)

    def mark_completed(self) -> None:
        if not self.state.ready_to_finish:
            raise RuntimeError("Onboarding is not ready to finish")
        self.state = replace(self.state, completed=True, step="shortcuts")
        self.state_store.save(self.state)

    def _skip_page(self) -> None:
        index = self.pages.currentIndex()
        if index == 1:
            self._save_state(provider_completed=False, provider_skipped=True)
        elif index == 2:
            self._whisper_settings_before_test = self.manager.whisper_settings
            self._onboarding_task = "whisper"
            try:
                self._save_whisper_settings(first_run_completed=True)
            except Exception as exc:  # noqa: BLE001 - settings boundary
                self._task_failed("whisper", str(exc))
                return
            self._onboarding_task = None
            self._whisper_settings_before_test = None
            self._save_state(whisper_completed=False, whisper_skipped=True)
        self._set_page(index + 1)

    def _provider_settings_from_form(self, *, create_new: bool = False) -> SavedProviderSettings:
        current = getattr(self.manager, "provider_settings", None)
        connections = list(getattr(current, "connections", ()))
        if not connections:
            connections = [ModelConnection("default", "默认连接")]
        target_index = next(
            (
                index
                for index, connection in enumerate(connections)
                if connection.id == self._provider_target_connection_id
            ),
            0,
        )
        primary = replace(
            connections[target_index],
            id=uuid.uuid4().hex if create_new else connections[target_index].id,
            name=self.onboarding_provider_name.text().strip() or "默认连接",
            base_url=self.onboarding_provider_url.text().strip(),
            api_key=self.onboarding_provider_key.text().strip(),
            api_mode=str(self.onboarding_provider_api_mode.currentData()),
        )
        if create_new:
            connections.insert(0, primary)
            self._provider_target_connection_id = primary.id
        else:
            connections[target_index] = primary
        roles = []
        model = self.onboarding_provider_model.text().strip()
        for role in getattr(current, "roles", ()):
            if role.name == RoleName.UTILITY:
                roles.append(
                    replace(
                        role,
                        connection_id=primary.id if create_new else role.connection_id,
                        model=model or role.model,
                    )
                )
            else:
                roles.append(role)
        return SavedProviderSettings(tuple(connections), tuple(roles))

    @Slot()
    def _test_provider(self, *, create_new: bool = False) -> None:
        if self._onboarding_task is not None:
            return
        self._provider_settings_before_test = getattr(self.manager, "provider_settings", None)
        self._provider_test_creating = create_new
        self.provider_test_button.setEnabled(False)
        self.provider_create_button.setEnabled(False)
        self._update_navigation()
        self.provider_status.setText("正在测试模型连接…")
        settings = self._provider_settings_from_form(create_new=create_new)
        self._onboarding_task = "provider"
        self._update_navigation()

        def work() -> None:
            try:
                self.manager.configure_provider(settings)
                result = self.manager.test_provider()
            except Exception as exc:  # noqa: BLE001 - provider boundary
                self.task_failed.emit("provider", str(exc))
            else:
                self.provider_tested.emit(result)

        threading.Thread(target=work, name="onboarding-test-provider", daemon=True).start()

    @Slot(str)
    def _provider_test_succeeded(self, result: str) -> None:
        if self._onboarding_task != "provider":
            return
        self.provider_test_button.setEnabled(True)
        self.provider_create_button.setEnabled(True)
        self._update_navigation()
        try:
            self.manager.save_provider()
        except Exception as exc:  # noqa: BLE001 - credential boundary
            self._task_failed("provider", str(exc))
            return
        self._save_state(provider_completed=True, provider_skipped=False)
        self._onboarding_task = None
        self._provider_settings_before_test = None
        self._update_navigation()
        action = (
            "新模型连接已创建并测试成功" if self._provider_test_creating else "模型连接测试成功"
        )
        self.provider_status.setText(f"{action}：{self._compact_text(result)}")

    def _whisper_profile_changed(self) -> None:
        profile = WhisperProfile(str(self.onboarding_whisper_profile.currentData()))
        self.whisper_impact.setText(PROFILE_PRESETS[profile].hardware_impact)
        self._update_navigation()

    def _save_whisper_settings(self, *, first_run_completed: bool = False) -> None:
        profile = WhisperProfile(str(self.onboarding_whisper_profile.currentData()))
        current = self.manager.whisper_settings
        selected = current if current.profile == profile else PROFILE_PRESETS[profile].settings
        self.manager.configure_whisper(
            replace(
                selected,
                first_run_completed=first_run_completed or current.first_run_completed,
            )
        )
        self.manager.save_whisper()

    @Slot()
    def _run_whisper_benchmark(self) -> None:
        if self._onboarding_task is not None:
            return
        self._whisper_settings_before_test = self.manager.whisper_settings
        self._onboarding_task = "whisper"
        try:
            self._save_whisper_settings()
        except Exception as exc:  # noqa: BLE001 - Whisper settings boundary
            self._task_failed("whisper", str(exc))
            return
        self.whisper_benchmark_button.setEnabled(False)
        self._update_navigation()
        self.whisper_status.setText("正在准备本地模型并运行样本…")

        def work() -> None:
            try:
                sample = create_builtin_whisper_sample(self.settings.data_dir)
                result = self.manager.benchmark_whisper(sample)
            except Exception as exc:  # noqa: BLE001 - benchmark boundary
                self.task_failed.emit("whisper", str(exc))
            else:
                self.whisper_benchmarked.emit(result)

        threading.Thread(target=work, name="onboarding-whisper-benchmark", daemon=True).start()

    @Slot(object)
    def _whisper_benchmark_succeeded(self, result: object) -> None:
        if self._onboarding_task != "whisper":
            return
        self.whisper_benchmark_button.setEnabled(True)
        self._update_navigation()
        elapsed = getattr(result, "elapsed_seconds", 0.0)
        factor = getattr(result, "realtime_factor", 0.0)
        self._save_state(whisper_completed=True, whisper_skipped=False)
        self._onboarding_task = None
        self._whisper_settings_before_test = None
        self._update_navigation()
        self.whisper_status.setText(f"样本测试完成：耗时 {elapsed:.2f} 秒，实时系数 {factor:.2f}。")

    @Slot()
    def _show_advanced_whisper(self) -> None:
        if self._onboarding_task is not None:
            return
        if self._whisper_dialog is None:
            self._whisper_dialog = WhisperSettingsDialog(self.manager, self)
        self._whisper_dialog.show()
        self._whisper_dialog.raise_()
        self._whisper_dialog.activateWindow()

    @Slot()
    def _open_recording_confirmation(self) -> None:
        if self._onboarding_task is not None:
            return
        self._onboarding_task = "recording"
        self._update_navigation()
        try:
            selection = _confirm_recording_selection(
                self,
                self.manager,
                self.settings,
                default_system_audio_enabled=self.capsule_preview.system_audio_check.isChecked(),
                default_microphone_enabled=self.capsule_preview.microphone_check.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001 - recording settings boundary
            self._task_failed("recording", str(exc))
            return
        if selection is None:
            self._onboarding_task = None
            self._update_navigation()
            return
        self._recording_selection = selection
        self._save_state(recording_confirmed=True)
        self._onboarding_task = None
        self._update_navigation()
        self.recording_status.setText("录制来源已确认并保存；可以重新打开修改。")

    @Slot()
    def _preview_question(self) -> None:
        self._shortcut_triggered()
        self.capsule_preview_status.setText("提问入口已触发；正式会话中会聚焦真实提问框。")

    @Slot()
    def _shortcut_triggered(self) -> None:
        if self._onboarding_task is not None:
            return
        if self._question_callback is not None:
            self._question_callback()
        self._confirm_shortcut()

    @Slot(QKeySequence)
    def _shortcut_changed(self, sequence: QKeySequence) -> None:
        value = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        if not value:
            value = DEFAULT_QUESTION_SHORTCUT
            self.onboarding_shortcut.setKeySequence(QKeySequence(value))
        self._shortcut.setKey(QKeySequence(value))
        parent = self.parentWidget()
        parent_shortcut = getattr(parent, "ask_shortcut", None)
        if parent_shortcut is not None:
            parent_shortcut.setKey(QKeySequence(value))
        self._save_state(question_shortcut=value)
        self.shortcut_status.setText(f"当前快捷键：{value}。按下它测试提问入口。")

    @Slot()
    def _confirm_shortcut(self) -> None:
        self._save_state(shortcuts_completed=True)
        self.shortcut_status.setText(
            f"快捷键测试完成。完成引导后仍可按 {self.state.question_shortcut} 聚焦提问框。"
        )

    def show_start_failure(self) -> None:
        self._set_page(len(ONBOARDING_STEPS) - 1)
        self.shortcut_status.setText("首个会话启动失败；请检查录制设备后重试，当前引导进度已保留。")
        self.show()
        self.raise_()
        self.activateWindow()

    @Slot(str, str)
    def _task_failed(self, task: str, message: str) -> None:
        if self._onboarding_task is not None and self._onboarding_task != task:
            return
        rollback_error = ""
        try:
            if task == "provider" and self._provider_settings_before_test is not None:
                configure_provider = getattr(self.manager, "configure_provider", None)
                if callable(configure_provider):
                    configure_provider(self._provider_settings_before_test)
                utility_role = next(
                    (
                        role
                        for role in self._provider_settings_before_test.roles
                        if role.name == RoleName.UTILITY
                    ),
                    None,
                )
                if utility_role is not None:
                    self._provider_target_connection_id = utility_role.connection_id
            elif task == "whisper" and self._whisper_settings_before_test is not None:
                configure_whisper = getattr(self.manager, "configure_whisper", None)
                if callable(configure_whisper):
                    configure_whisper(self._whisper_settings_before_test)
                save_whisper = getattr(self.manager, "save_whisper", None)
                if callable(save_whisper):
                    save_whisper()
        except Exception as exc:  # noqa: BLE001 - best-effort rollback boundary
            rollback_error = f"；恢复旧配置失败：{exc}"
        finally:
            self._provider_settings_before_test = None
            self._whisper_settings_before_test = None
            self._onboarding_task = None
            self._provider_test_creating = False
            self.provider_test_button.setEnabled(True)
            self.provider_create_button.setEnabled(True)
            self.whisper_benchmark_button.setEnabled(True)
            self._update_navigation()
        if task == "whisper":
            self.whisper_status.setText(f"样本测试失败：{message}{rollback_error}")
        elif task == "recording":
            self.recording_status.setText(f"录制确认失败：{message}{rollback_error}")
        else:
            self.provider_status.setText(f"模型连接失败：{message}{rollback_error}")

    @staticmethod
    def _compact_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()[:240] or "服务已响应。"

    def accept(self) -> None:
        if self._onboarding_task is not None:
            return
        self.state_store.save(self.state)
        super().accept()

    def reject(self) -> None:
        if self._onboarding_task is not None:
            return
        self.state_store.save(self.state)
        super().reject()


class MainWindow(QMainWindow):
    LIVE_TIMELINE_REFRESH_MS = 1_000
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
        show_onboarding: bool = False,
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
        self._onboarding_store = OnboardingSettingsStore(settings.data_dir)
        self._capsule_position_store = RecordingCapsulePositionStore(settings.data_dir)
        self._capsule_shown_once = False
        self._onboarding_dialog: OnboardingDialog | None = None
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
        self._cross_session_dialog: CrossSessionSynthesisDialog | None = None
        self._session_sort_newest = True
        self._maintenance_thread: threading.Thread | None = None
        self._archive_operation: str | None = None
        self._runtime_metrics_snapshot = None
        self._runtime_metrics_in_flight = False
        self._runtime_metrics_generation = 0
        self._timeline_refresh_session_id: str | None = None
        self._timeline_refresh_timer = QTimer(self)
        self._timeline_refresh_timer.setSingleShot(True)
        self._timeline_refresh_timer.setInterval(self.LIVE_TIMELINE_REFRESH_MS)
        self._timeline_refresh_timer.timeout.connect(self._refresh_active_timeline)
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
        QTimer.singleShot(0, self._recover_pending_audio)
        self._whisper_dialog: WhisperSettingsDialog | None = None
        if show_onboarding:
            QTimer.singleShot(0, self._maybe_show_onboarding)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if not self._capsule_shown_once:
            self._capsule_shown_once = True
            self._show_recording_capsule()

    def _show_recording_capsule(self) -> None:
        if not self.capsule.isVisible():
            self.capsule.show()
        self.capsule.raise_()
        if self.capsule_hide_button is not None:
            self.capsule_hide_button.setVisible(bool(getattr(self.service, "is_recording", False)))

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        capsule = RecordingCapsule(
            default_system_audio_enabled=self.settings.capture_system_audio,
            default_microphone_enabled=self.settings.capture_microphone,
            pause_enabled=callable(getattr(self.manager, "pause", None)),
            floating=True,
            position_store=self._capsule_position_store,
        )
        capsule.setStyleSheet(APP_STYLE)
        self.capsule = capsule
        self.status = capsule.status
        self.title_input = capsule.title_input
        self.system_audio_check = capsule.system_audio_check
        self.microphone_check = capsule.microphone_check
        self.pause_button = capsule.pause_button
        self.capsule_ask_button = capsule.capsule_ask_button
        self.start_button = capsule.start_button
        self.stop_button = capsule.stop_button
        self.capsule_hide_button = capsule.hide_button

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
        self.session_export_button = QPushButton("导出会话")
        action_row.addWidget(self.session_pin_button)
        action_row.addWidget(self.session_delete_button)
        action_row.addWidget(self.session_restore_button)
        panel_layout.addWidget(title)
        panel_layout.addWidget(subtitle)
        panel_layout.addSpacing(6)
        panel_layout.addWidget(self.session_search)
        self.cross_session_button = QPushButton("跨会话综合")
        self.cross_session_button.setObjectName("crossSessionButton")
        self.cross_session_button.setToolTip("搜索并选择多个会话的证据后进行深度综合")
        panel_layout.addWidget(self.cross_session_button)
        panel_layout.addLayout(filter_row)
        panel_layout.addWidget(self.session_library, 1)
        panel_layout.addLayout(action_row)
        panel_layout.addWidget(self.session_complete_button)
        panel_layout.addWidget(self.session_export_button)
        archive_row = QHBoxLayout()
        self.backup_button = QPushButton("完整备份")
        self.restore_backup_button = QPushButton("恢复备份")
        archive_row.addWidget(self.backup_button)
        archive_row.addWidget(self.restore_backup_button)
        panel_layout.addLayout(archive_row)
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
        self.onboarding_button = QPushButton("首次引导")
        self.onboarding_button.setProperty("role", "quiet")
        header.addWidget(self.storage_settings_button, alignment=Qt.AlignmentFlag.AlignBottom)
        header.addWidget(self.onboarding_button, alignment=Qt.AlignmentFlag.AlignBottom)
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
        self.retry_audio_button = QPushButton("重试音频")
        self.retry_audio_button.setObjectName("retryAudio")
        self.retry_audio_button.clicked.connect(self._retry_failed_audio)
        self.retry_audio_button.hide()
        self.retry_correction_button = QPushButton("重试校订")
        self.retry_correction_button.setObjectName("retryCorrection")
        self.retry_correction_button.clicked.connect(self._retry_failed_corrections)
        self.retry_correction_button.hide()
        notice_close = QPushButton("×")
        notice_close.setObjectName("noticeClose")
        notice_close.clicked.connect(self.notice.hide)
        notice_layout.addWidget(self.notice_text, 1)
        notice_layout.addWidget(self.retry_audio_button, alignment=Qt.AlignmentFlag.AlignTop)
        notice_layout.addWidget(self.retry_correction_button, alignment=Qt.AlignmentFlag.AlignTop)
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
            self.session_export_button.setEnabled(False)
            self.session_restore_button.hide()
            self.session_complete_button.hide()
            return
        current = record.id == self._active_session_id()
        busy = self.service.session_storage_busy_reason(record.id) is not None
        archive_busy = self._archive_operation is not None
        self.session_pin_button.setEnabled(
            not archive_busy and not current and not busy and record.trashed_at_utc is None
        )
        self.session_pin_button.setText("取消固定" if record.pinned else "固定")
        self.session_delete_button.setEnabled(
            not archive_busy and not current and not busy and record.trashed_at_utc is None
        )
        self.session_export_button.setEnabled(not archive_busy and not busy)
        self.session_delete_button.setVisible(record.trashed_at_utc is None)
        self.session_restore_button.setVisible(record.trashed_at_utc is not None)
        interrupted = record.status == "interrupted" and record.trashed_at_utc is None
        self.session_complete_button.setVisible(interrupted)
        self.session_complete_button.setEnabled(interrupted and not busy and not archive_busy)

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

    def _archive_active(self) -> bool:
        return self._archive_operation is not None

    def _set_archive_buttons_enabled(self, enabled: bool) -> None:
        self.backup_button.setEnabled(enabled)
        self.restore_backup_button.setEnabled(enabled)
        self._update_session_actions(self.session_library.currentItem())
        self._refresh_recording_status()
        if not enabled:
            self.session_export_button.setEnabled(False)

    def _run_archive_operation(self, operation: str, work: Callable[[], object]) -> None:
        if self._archive_active():
            return
        self._archive_operation = operation
        self._set_archive_buttons_enabled(False)
        labels = {
            "export": "正在导出会话",
            "backup": "正在创建完整备份",
            "restore": "正在恢复完整备份",
        }
        self._set_status(labels[operation], "busy")

        def run() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - transferred to the UI thread
                self.bridge.archive_failed.emit(operation, str(exc))
                return
            target = getattr(result, "target_dir", result)
            self.bridge.archive_finished.emit(operation, str(target))

        threading.Thread(target=run, name=f"archive-{operation}", daemon=True).start()

    def _run_restore_preview(self, archive: Path, target: Path) -> None:
        if self._archive_active():
            return
        self._archive_operation = "restore-preview"
        self._set_archive_buttons_enabled(False)
        self._set_status("正在校验完整备份", "busy")

        def run() -> None:
            try:
                preview = self.service.preview_restore(archive, target)
            except Exception as exc:  # noqa: BLE001 - transferred to the UI thread
                self.bridge.archive_preview_failed.emit(str(exc))
                return
            self.bridge.archive_preview_finished.emit(preview)

        threading.Thread(target=run, name="archive-restore-preview", daemon=True).start()

    @Slot()
    def _export_selected_session(self) -> None:
        record = self._selected_session_record()
        if record is None:
            return
        busy_reason = self._storage_busy_reason()
        if busy_reason:
            self._show_action_error(f"当前不能导出会话：{busy_reason}")
            return
        busy_reason = self.service.session_storage_busy_reason(record.id)
        if busy_reason:
            self._show_action_error(f"会话仍在写入：{busy_reason}")
            return
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "导出会话 ZIP",
            str(self.settings.data_dir.parent / f"{record.id}.zip"),
            "ZIP 归档 (*.zip)",
        )
        if not selected:
            return
        destination = self._zip_destination(selected)
        self._run_archive_operation(
            "export", lambda: self.service.export_session(record.id, destination)
        )

    @Slot()
    def _create_full_backup(self) -> None:
        busy_reason = self._storage_busy_reason()
        if busy_reason:
            self._show_action_error(f"当前不能创建完整备份：{busy_reason}")
            return
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "创建完整备份",
            str(self.settings.data_dir.parent / "jingzhi-backup.zip"),
            "ZIP 备份 (*.zip)",
        )
        if not selected:
            return
        destination = self._zip_destination(selected)
        self._run_archive_operation("backup", lambda: self.service.create_backup(destination))

    @staticmethod
    def _zip_destination(selected: str) -> Path:
        destination = Path(selected)
        return (
            destination if destination.suffix.lower() == ".zip" else destination.with_suffix(".zip")
        )

    @Slot()
    def _restore_full_backup(self) -> None:
        busy_reason = self._storage_busy_reason()
        if busy_reason:
            self._show_action_error(f"当前不能恢复完整备份：{busy_reason}")
            return
        selected, _filter = QFileDialog.getOpenFileName(
            self, "选择完整备份", str(self.settings.data_dir.parent), "ZIP 备份 (*.zip)"
        )
        if not selected:
            return
        archive = Path(selected)
        selected_target = QFileDialog.getExistingDirectory(
            self, "选择空的数据目录", str(self.settings.data_dir.parent)
        )
        if not selected_target:
            return
        self._run_restore_preview(archive, Path(selected_target))

    @Slot(object)
    def _restore_preview_finished(self, preview) -> None:  # type: ignore[no-untyped-def]
        self._archive_operation = None
        if not preview.can_restore:
            reason = preview.reason or "恢复目标不可用。"
            if preview.conflicting_session_ids:
                reason += "\n重复会话：" + "、".join(preview.conflicting_session_ids)
            self._set_archive_buttons_enabled(True)
            self._show_action_error(reason)
            return
        confirmation = QMessageBox.question(
            self,
            "确认恢复完整备份",
            f"将把 {len(preview.session_ids)} 个会话恢复到：\n{preview.target_dir}\n\n"
            "目标目录必须保持为空；恢复策略为拒绝已有数据，不覆盖或合并。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        self._set_archive_buttons_enabled(True)
        if confirmation != QMessageBox.StandardButton.Yes:
            self._set_status("恢复已取消", "idle")
            return
        self._run_archive_operation(
            "restore",
            lambda: self.service.restore_backup(preview.archive, preview.target_dir),
        )

    @Slot(str)
    def _restore_preview_failed(self, message: str) -> None:
        self._archive_failed("restore-preview", message)

    @Slot(str, str)
    def _archive_finished(self, operation: str, target: str) -> None:
        self._archive_operation = None
        labels = {
            "export": "会话导出完成",
            "backup": "完整备份完成",
            "restore": "完整备份恢复完成",
        }
        message = f"{labels.get(operation, '归档操作完成')}：{target}"
        if operation == "restore":
            restored_dir = Path(target)
            startup_store = self.settings.startup_settings_store
            if self.settings.data_dir_managed_by_env:
                message += "\n当前数据目录由 STUDY_DATA_DIR 管理；本窗口未切换数据源。"
            elif startup_store is None:
                message += "\n已恢复到新目录；请重启境织后使用该目录。"
            else:
                try:
                    startup_store.update(data_dir=restored_dir)
                except Exception as exc:  # noqa: BLE001 - UI boundary reports persistence failures
                    message += (
                        f"\n已恢复，但未能设置下次启动目录：{self._compact_message(str(exc))}"
                    )
                else:
                    message += "\n已设置为下次启动的数据目录；当前窗口将在重启后切换。"
        self._set_archive_buttons_enabled(True)
        self._set_status(labels.get(operation, "归档操作完成"), "success")
        self.notice_text.setText(message)
        self.notice.show()

    @Slot(str, str)
    def _archive_failed(self, operation: str, message: str) -> None:
        self._archive_operation = None
        self._set_archive_buttons_enabled(True)
        self._set_status("归档操作失败", "error")
        self.notice_text.setText(self._compact_message(message))
        self.notice.show()

    def _refresh_recording_status(self) -> None:
        if self.capsule_hide_button is not None:
            self.capsule_hide_button.setVisible(bool(getattr(self.service, "is_recording", False)))
        if not getattr(self.service, "is_recording", False):
            if self._archive_active() or self._stop_in_flight:
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
            self._runtime_metrics_generation += 1
            self._runtime_metrics_snapshot = None
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
        metrics = self._runtime_metrics_snapshot
        if metrics is not None:
            details.append(format_runtime_metrics(metrics))
            if metrics.free_bytes < 512 * 1024 * 1024:
                details.append("磁盘空间告警")
        self._start_runtime_metrics_sample()
        prefix = "已暂停" if self._paused else "记录中"
        self._set_status(f"{prefix} · " + " · ".join(details), "recording")

    def _start_runtime_metrics_sample(self) -> None:
        if self._runtime_metrics_in_flight:
            return
        method = getattr(self.service, "runtime_metrics", None)
        if not callable(method):
            return
        self._runtime_metrics_in_flight = True
        generation = self._runtime_metrics_generation

        def work() -> None:
            try:
                metrics = method()
            except Exception as exc:  # noqa: BLE001 - optional diagnostics boundary
                self.bridge.runtime_metrics_failed.emit(str(exc))
            else:
                self.bridge.runtime_metrics_ready.emit((generation, metrics))

        threading.Thread(target=work, name="runtime-metrics", daemon=True).start()

    @Slot(object)
    def _runtime_metrics_ready(self, payload) -> None:  # type: ignore[no-untyped-def]
        generation, metrics = payload
        if generation == self._runtime_metrics_generation:
            self._runtime_metrics_snapshot = metrics
        self._runtime_metrics_in_flight = False

    @Slot(str)
    def _runtime_metrics_failed(self, message: str) -> None:
        self._runtime_metrics_in_flight = False
        logger.debug("Runtime metrics sample failed: %s", message)

    def _recover_pending_audio(self) -> None:
        method = getattr(self.service, "recover_pending_audio", None)
        if not callable(method):
            return
        try:
            report = method()
        except Exception as exc:  # noqa: BLE001 - recovery boundary reports to UI
            self.bridge.audio_recovery_failed.emit(str(exc))
        else:
            self.bridge.audio_recovery_finished.emit(report)

    @Slot(object)
    def _audio_recovery_finished(self, report) -> None:  # type: ignore[no-untyped-def]
        self.retry_audio_button.setEnabled(True)
        try:
            unfinished = self.service.list_sessions(status="unfinished")
        except Exception:  # noqa: BLE001 - optional recovery notice boundary
            unfinished = []
        try:
            retryable_model_tasks = self.service.retryable_model_task_count()
            failed_audio_chunks = self.service.failed_audio_chunk_count()
        except Exception:  # noqa: BLE001 - optional recovery notice boundary
            retryable_model_tasks = 0
            failed_audio_chunks = 0
        try:
            failed_correction_runs = self.service.failed_correction_run_count()
        except Exception:  # noqa: BLE001 - optional retry affordance
            failed_correction_runs = 0
        self.retry_audio_button.setVisible(failed_audio_chunks > 0)
        self.retry_correction_button.setVisible(failed_correction_runs > 0)
        if failed_correction_runs:
            self.retry_correction_button.setEnabled(True)
        if (
            report.queued_chunks
            or report.missing_chunks
            or unfinished
            or retryable_model_tasks
            or failed_audio_chunks
            or failed_correction_runs
        ):
            message = f"已恢复 {report.queued_chunks} 个待转写音频片段"
            if unfinished:
                message += f"；发现 {len(unfinished)} 个未完成会话，时间线已保留"
            if retryable_model_tasks:
                message += f"；有 {retryable_model_tasks} 个模型任务可重试"
            if failed_audio_chunks:
                message += f"；有 {failed_audio_chunks} 个失败音频可重试"
            if failed_correction_runs:
                message += f"；有 {failed_correction_runs} 个字幕校订窗口可重试"
            if report.missing_chunks:
                message += f"；{report.missing_chunks} 个音频文件缺失，已标记失败"
            status_state = (
                "warning"
                if (
                    report.queued_chunks
                    or report.missing_chunks
                    or failed_audio_chunks
                    or failed_correction_runs
                )
                else "success"
            )
            status_label = "后台转写已排队" if report.queued_chunks else "后台转写恢复"
            self._set_status(status_label, status_state)
            self.notice_text.setText(message)
            self.notice.show()

    def _retry_failed_audio(self) -> None:
        method = getattr(self.service, "retry_failed_audio", None)
        if not callable(method):
            return
        self.retry_audio_button.setEnabled(False)

        def work() -> None:
            try:
                report = method()
            except Exception as exc:  # noqa: BLE001 - recovery boundary reports to UI
                self.bridge.audio_recovery_failed.emit(str(exc))
            else:
                self.bridge.audio_recovery_finished.emit(report)

        threading.Thread(target=work, name="retry-audio", daemon=True).start()

    def _retry_failed_corrections(self) -> None:
        if not self.retry_correction_button.isEnabled():
            return
        self.retry_correction_button.setEnabled(False)
        self._set_status("正在重新排队字幕校订…", "warning")

        def work() -> None:
            try:
                count = self.service.retry_failed_correction_runs()
            except Exception as exc:  # noqa: BLE001 - retry boundary reports to UI
                self.bridge.correction_retry_failed.emit(str(exc))
            else:
                self.bridge.correction_retry_finished.emit(count)

        threading.Thread(target=work, name="retry-corrections", daemon=True).start()

    @Slot(int)
    def _correction_retry_finished(self, count: int) -> None:
        self.retry_correction_button.setVisible(False)
        self.retry_correction_button.setEnabled(True)
        self._set_status("字幕校订已排队", "warning")
        self.notice_text.setText(f"已重新排队 {count} 个字幕校订窗口")
        self.notice.show()

    @Slot(str)
    def _correction_retry_failed(self, message: str) -> None:
        self.retry_correction_button.setEnabled(True)
        self._set_status("字幕校订重试失败", "error")
        self.notice_text.setText(self._compact_message(message))
        self.notice.show()

    @Slot(str)
    def _audio_recovery_failed(self, message: str) -> None:
        self.retry_audio_button.setEnabled(True)
        self._set_status("后台转写恢复失败", "error")
        self.notice_text.setText(self._compact_message(message))
        self.notice.show()

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
                cache_key = f"jingzhi-frame-thumbnail:{frame.path}"
                thumbnail = QPixmap()
                if not QPixmapCache.find(cache_key, thumbnail):
                    pixmap = QPixmap(str(frame.path))
                    if not pixmap.isNull():
                        thumbnail = pixmap.scaled(
                            QSize(82, 58),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        QPixmapCache.insert(cache_key, thumbnail)
                if not thumbnail.isNull():
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
    def _maybe_show_onboarding(self) -> None:
        if self._onboarding_store.load().completed:
            if (
                callable(getattr(self.manager, "configure_whisper", None))
                and not self.manager.whisper_settings.first_run_completed
            ):
                self._show_whisper_settings()
            return
        self._show_onboarding()

    @Slot()
    def _show_onboarding(self, *, reset: bool = False) -> None:
        if self._onboarding_dialog is not None:
            self._onboarding_dialog.raise_()
            self._onboarding_dialog.activateWindow()
            return
        if self.service.is_recording:
            self._show_action_error("录制期间不能打开首次使用引导。")
            return
        if reset:
            self._onboarding_store.reset()
        self._onboarding_dialog = OnboardingDialog(
            self.manager,
            self.settings,
            parent=self,
            state_store=self._onboarding_store,
            question_callback=self._focus_question,
        )
        self._onboarding_dialog.finished.connect(self._onboarding_finished)
        self._onboarding_dialog.show()
        self._onboarding_dialog.raise_()
        self._onboarding_dialog.activateWindow()

    @Slot(int)
    def _onboarding_finished(self, result: int) -> None:
        dialog = self._onboarding_dialog
        if dialog is None:
            return
        selection = dialog.recording_selection if result == QDialog.DialogCode.Accepted else None
        if result == QDialog.DialogCode.Accepted:
            self._reload_provider_settings_from_manager()
            if self._start(selection=selection):
                self.ask_shortcut.setKey(QKeySequence(dialog.state.question_shortcut))
                dialog.mark_completed()
            else:
                dialog.show_start_failure()
                return
        dialog.deleteLater()
        self._onboarding_dialog = None

    def _reload_provider_settings_from_manager(self) -> None:
        provider_settings = getattr(self.manager, "provider_settings", None)
        connections = getattr(provider_settings, "connections", ())
        roles = getattr(provider_settings, "roles", ())
        if not connections:
            return
        self._provider_connections = list(connections)
        self._provider_roles = {role.name: role for role in roles}
        self._active_connection_index = min(
            self._active_connection_index, len(self._provider_connections) - 1
        )
        if hasattr(self, "connection_selector"):
            self._refresh_provider_connection_choices(self._active_connection_index)
            self._load_provider_role_form_values()

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
            "archive-export",
            "archive-backup",
            "archive-restore",
            "archive-restore-preview",
            "cross-session-synthesis",
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

    @Slot()
    def _open_cross_session_synthesis(self) -> None:
        if self._cross_session_dialog is None:
            self._cross_session_dialog = CrossSessionSynthesisDialog(
                self.service,
                navigate_callback=self._navigate_cross_session_evidence,
                parent=self,
            )
        self._cross_session_dialog.show()
        self._cross_session_dialog.raise_()
        self._cross_session_dialog.activateWindow()

    def _navigate_cross_session_evidence(self, candidate: CrossSessionEvidenceRecord) -> None:
        if candidate.session_id != self._selected_session_id:
            self.session_search.clear()
            all_index = self.session_filter.findData("all")
            if all_index >= 0:
                self.session_filter.setCurrentIndex(all_index)
            self._refresh_sessions(candidate.session_id)
        current = self.session_library.currentItem()
        if current is None or str(current.data(Qt.ItemDataRole.UserRole)) != candidate.session_id:
            self._show_evidence_navigation_error("目标会话当前不可用")
            return
        self._zoom_key = "1-minute"
        zoom_button = self.findChild(QPushButton, "zoom-1-minute")
        if zoom_button is not None:
            zoom_button.setChecked(True)
        self._window_start_ms = max(0, candidate.start_ms - 30_000)
        target_answer_id = candidate.answer_version_id if candidate.kind == "answer" else None
        target_material_id = candidate.material_version_id if candidate.kind == "material" else None
        self._open_session_item(current)
        if target_answer_id is not None:
            self._selected_answer_version_id = target_answer_id
            self._content_kind = "answer"
            self._open_session_item(current)
        elif target_material_id is not None:
            self._selected_material_version_id = target_material_id
            self._content_kind = "material"
            self._open_session_item(current)
        if self._timeline is None:
            return
        if candidate.kind == "frame" and candidate.frame_id is not None:
            visible = next(
                (item for item in self._timeline.frames if item.id == candidate.frame_id), None
            )
            object_name = f"keyframe-{candidate.frame_id}"
            track = self.keyframe_track
        elif candidate.kind == "transcript" and candidate.transcript_version_id is not None:
            visible = next(
                (
                    item
                    for item in self._timeline.transcripts
                    if item.version_id == candidate.transcript_version_id
                ),
                None,
            )
            object_name = ""
            track = self.transcript_track
        else:
            visible = None
            object_name = ""
            track = None
        if visible is None and candidate.kind == "transcript":
            historical = self.service.timeline_transcript_version(
                candidate.session_id,
                candidate.transcript_version_id or 0,
                candidate.start_ms,
                candidate.end_ms,
            )
            if historical is not None and self._timeline is not None:
                transcripts = tuple(
                    historical if item.id == historical.id else item
                    for item in self._timeline.transcripts
                )
                self._timeline = replace(self._timeline, transcripts=transcripts)
                self._render_timeline(self._timeline)
                visible = historical
        if visible is not None:
            if candidate.kind == "frame":
                self._select_frame(visible)
                object_name = f"keyframe-{visible.id}"
            else:
                self._select_transcript(visible)
                object_name = f"transcript-{visible.id}"
            if track is not None:
                button = self.findChild(QPushButton, object_name)
                scroll = track.findChild(QScrollArea)
                if button is not None and scroll is not None:
                    scroll.ensureWidgetVisible(button)
        if self._cross_session_dialog is not None:
            self._cross_session_dialog.hide()

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
        self.session_export_button.clicked.connect(self._export_selected_session)
        self.backup_button.clicked.connect(self._create_full_backup)
        self.restore_backup_button.clicked.connect(self._restore_full_backup)
        self.cross_session_button.clicked.connect(self._open_cross_session_synthesis)
        self.start_button.clicked.connect(lambda: self._start())
        self.stop_button.clicked.connect(self._stop)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.capsule_ask_button.clicked.connect(self._focus_question)
        self.ask_button.clicked.connect(self._ask)
        self.reanswer_button.clicked.connect(self._reanswer)
        self.bridge.segment.connect(self._append_segment)
        self.bridge.worker_warning.connect(self._show_worker_warning)
        self.bridge.source_event.connect(self._source_event_reported)
        self.bridge.audio_recovery_finished.connect(self._audio_recovery_finished)
        self.bridge.audio_recovery_failed.connect(self._audio_recovery_failed)
        self.bridge.correction_retry_finished.connect(self._correction_retry_finished)
        self.bridge.correction_retry_failed.connect(self._correction_retry_failed)
        self.bridge.runtime_metrics_ready.connect(self._runtime_metrics_ready)
        self.bridge.runtime_metrics_failed.connect(self._runtime_metrics_failed)
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
        shortcut = self._onboarding_store.load().question_shortcut
        self.ask_shortcut = QShortcut(QKeySequence(shortcut), self)
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
        self.onboarding_button.clicked.connect(lambda: self._show_onboarding(reset=True))
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
        self.bridge.archive_finished.connect(self._archive_finished)
        self.bridge.archive_failed.connect(self._archive_failed)
        self.bridge.archive_preview_finished.connect(self._restore_preview_finished)
        self.bridge.archive_preview_failed.connect(self._restore_preview_failed)

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
    def _start(self, selection: RecordingSelection | None = None) -> bool:
        if self._archive_active():
            self._show_action_error("归档操作正在进行，请等待完成后再开始会话。")
            return False
        if selection is None:
            try:
                selection = _confirm_recording_selection(
                    self,
                    self.manager,
                    self.settings,
                    default_system_audio_enabled=self.system_audio_check.isChecked(),
                    default_microphone_enabled=self.microphone_check.isChecked(),
                )
            except Exception as exc:  # noqa: BLE001 - UI boundary must surface selection failures
                self._show_action_error(str(exc))
                return False
            if selection is None:
                return False
        try:
            self._configure_correction()
            session_id = self.service.start_session(self.title_input.text(), selection=selection)
        except Exception as exc:  # noqa: BLE001 - UI boundary must surface worker failures
            self._show_action_error(str(exc))
            return False
        self._show_recording_capsule()
        self.system_audio_check.setChecked(selection.system_audio_id is not None)
        self.microphone_check.setChecked(selection.microphone_id is not None)
        self._runtime_metrics_generation += 1
        self._runtime_metrics_snapshot = None
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
        return True

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
        if self.service.is_recording and not self.capsule.isVisible():
            self._show_recording_capsule()
        self.activateWindow()
        self.raise_()
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

    def _load_provider_role_form_values(self) -> None:
        for name, role in self._provider_roles.items():
            connection_input = self.role_connection_inputs[name]
            connection_input.setCurrentIndex(max(0, connection_input.findData(role.connection_id)))
            self.role_model_inputs[name].setText(role.model)
            self.role_reasoning_inputs[name].setCurrentIndex(
                max(0, self.role_reasoning_inputs[name].findData(role.reasoning.value))
            )
            fallback_groups = (
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
            for controls, fallback in zip(
                fallback_groups, (*role.fallbacks, None, None)[:2], strict=True
            ):
                connection_input, model_input, authorization = controls
                connection_input.setCurrentIndex(
                    max(0, connection_input.findData(fallback.connection_id if fallback else ""))
                )
                model_input.setText(fallback.model if fallback else "")
                authorization.setChecked(bool(fallback and fallback.cross_connection_authorized))

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
        active_session_id = self._active_session_id()
        if active_session_id is None or active_session_id != self._selected_session_id:
            return
        self._timeline_refresh_session_id = active_session_id
        if not self._timeline_refresh_timer.isActive():
            self._timeline_refresh_timer.start()

    @Slot()
    def _refresh_active_timeline(self) -> None:
        session_id = self._timeline_refresh_session_id
        self._timeline_refresh_session_id = None
        if session_id is None or session_id != self._active_session_id():
            return
        current = self.session_library.currentItem()
        if current is None or current.data(Qt.ItemDataRole.UserRole) != session_id:
            return
        self._open_session_item(current)

    @Slot(str)
    def _show_worker_warning(self, message: str) -> None:
        try:
            failed_audio_chunks = self.service.failed_audio_chunk_count()
            self.retry_audio_button.setVisible(failed_audio_chunks > 0)
            failed_correction_runs = self.service.failed_correction_run_count()
            self.retry_correction_button.setVisible(failed_correction_runs > 0)
            self.retry_correction_button.setEnabled(failed_correction_runs > 0)
        except Exception as exc:  # noqa: BLE001 - warning display must not mask the worker error
            logger.debug("Could not refresh retryable task counts: %s", exc)
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
        if self._archive_active():
            self._show_action_error("归档操作正在进行，请等待完成后再退出境织。")
            event.ignore()
            return
        if self._storage_dialog is not None and self._storage_dialog.operation_active:
            self._storage_dialog.show()
            self._storage_dialog.raise_()
            self._show_action_error("存储迁移正在进行，请等待完成后再退出境织。")
            event.ignore()
            return
        if self.service.is_recording:
            self._show_recording_capsule()
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
        self.capsule.shutdown()
        event.accept()


def run_app(settings: Settings) -> int:
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    window = MainWindow(settings, show_onboarding=True)
    window.show()
    if settings.startup_settings_store is not None:
        settings.startup_settings_store.complete_successful_startup(settings.data_dir)
    return application.exec()
