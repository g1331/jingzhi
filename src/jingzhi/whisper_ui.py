from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from jingzhi.whisper_settings import (
    PROFILE_PRESETS,
    BenchmarkResult,
    DownloadState,
    WhisperProfile,
    WhisperSettings,
    create_builtin_whisper_sample,
)


class WhisperSettingsDialog(QDialog):
    benchmark_ready = Signal(object)
    download_changed = Signal(object)
    task_failed = Signal(str)

    def __init__(self, manager: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.manager = manager
        self._cancel_download = threading.Event()
        self.setWindowTitle("本地 Whisper 设置")
        self.setMinimumWidth(680)
        layout = QVBoxLayout(self)

        profile_group = QGroupBox("产品档位")
        profile_layout = QGridLayout(profile_group)
        self.profile_input = QComboBox()
        self.profile_input.setObjectName("whisperProfile")
        for profile in WhisperProfile:
            preset = PROFILE_PRESETS[profile]
            self.profile_input.addItem(preset.label, profile.value)
        self.hardware_impact = QLabel()
        self.hardware_impact.setObjectName("whisperHardwareImpact")
        self.hardware_impact.setWordWrap(True)
        profile_layout.addWidget(QLabel("档位"), 0, 0)
        profile_layout.addWidget(self.profile_input, 0, 1)
        profile_layout.addWidget(self.hardware_impact, 1, 0, 1, 2)
        layout.addWidget(profile_group)

        advanced_group = QGroupBox("高级设置")
        advanced_layout = QGridLayout(advanced_group)
        self.model_input = QLineEdit()
        self.model_input.setObjectName("whisperModel")
        self.device_input = QComboBox()
        self.device_input.setObjectName("whisperDevice")
        for label, value in (("自动", "auto"), ("CPU", "cpu"), ("CUDA GPU", "cuda")):
            self.device_input.addItem(label, value)
        self.compute_input = QComboBox()
        self.compute_input.setObjectName("whisperComputeType")
        for value in ("default", "int8", "float32", "float16", "int8_float16"):
            self.compute_input.addItem(value, value)
        self.language_input = QComboBox()
        self.language_input.setObjectName("whisperLanguage")
        for label, value in (
            ("自动检测", "auto"),
            ("中文", "zh"),
            ("English", "en"),
            ("日本語", "ja"),
        ):
            self.language_input.addItem(label, value)
        self.vad_check = QCheckBox("启用语音活动检测（VAD）")
        self.vad_silence = QSpinBox()
        self.vad_silence.setObjectName("whisperVadSilence")
        self.vad_silence.setRange(0, 5_000)
        self.vad_silence.setSuffix(" ms")
        controls = (
            ("模型", self.model_input),
            ("设备", self.device_input),
            ("计算精度", self.compute_input),
            ("语言", self.language_input),
            ("VAD 最短静音", self.vad_silence),
        )
        for row, (label, control) in enumerate(controls):
            advanced_layout.addWidget(QLabel(label), row, 0)
            advanced_layout.addWidget(control, row, 1)
        advanced_layout.addWidget(self.vad_check, len(controls), 0, 1, 2)
        layout.addWidget(advanced_group)

        self.fallback_advice = QLabel()
        self.fallback_advice.setObjectName("whisperFallbackAdvice")
        self.fallback_advice.setWordWrap(True)
        layout.addWidget(self.fallback_advice)

        download_group = QGroupBox("模型下载")
        download_layout = QGridLayout(download_group)
        self.download_progress = QProgressBar()
        self.download_progress.setObjectName("whisperDownloadProgress")
        self.disk_requirement = QLabel("尚未检查磁盘需求")
        self.disk_requirement.setObjectName("whisperDiskRequirement")
        self.prepare_button = QPushButton("下载或检查模型")
        self.prepare_button.setObjectName("prepareWhisperModel")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("cancelWhisperDownload")
        self.retry_button = QPushButton("重试")
        self.retry_button.setObjectName("retryWhisperDownload")
        self.cancel_button.hide()
        self.retry_button.hide()
        download_layout.addWidget(self.download_progress, 0, 0, 1, 3)
        download_layout.addWidget(self.disk_requirement, 1, 0, 1, 3)
        download_layout.addWidget(self.prepare_button, 2, 0)
        download_layout.addWidget(self.cancel_button, 2, 1)
        download_layout.addWidget(self.retry_button, 2, 2)
        layout.addWidget(download_group)

        benchmark_group = QGroupBox("首次样本测试")
        benchmark_layout = QVBoxLayout(benchmark_group)
        self.benchmark_button = QPushButton("运行内置中文样本")
        self.benchmark_result = QLabel("尚未运行样本测试")
        self.benchmark_result.setObjectName("whisperBenchmarkResult")
        self.benchmark_result.setWordWrap(True)
        benchmark_layout.addWidget(self.benchmark_button)
        benchmark_layout.addWidget(self.benchmark_result)
        layout.addWidget(benchmark_group)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.save_button = QPushButton("保存 Whisper 设置")
        self.save_button.setObjectName("saveWhisper")
        self.save_button.setProperty("role", "primary")
        buttons.addButton(self.save_button, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.profile_input.currentIndexChanged.connect(self._apply_profile)
        self.save_button.clicked.connect(self._save)
        self.prepare_button.clicked.connect(self._prepare_model)
        self.retry_button.clicked.connect(self._prepare_model)
        self.cancel_button.clicked.connect(self._cancel_model_download)
        self.benchmark_button.clicked.connect(self._run_benchmark)
        self.benchmark_ready.connect(self.show_benchmark_result)
        self.download_changed.connect(self.show_download_state)
        self.task_failed.connect(self._show_task_error)
        self._load_settings(self.manager.whisper_settings)

    def _load_settings(self, settings: WhisperSettings) -> None:
        self.profile_input.blockSignals(True)
        self.profile_input.setCurrentIndex(
            max(0, self.profile_input.findData(settings.profile.value))
        )
        self.profile_input.blockSignals(False)
        self.model_input.setText(settings.model)
        self.device_input.setCurrentIndex(max(0, self.device_input.findData(settings.device)))
        self.compute_input.setCurrentIndex(
            max(0, self.compute_input.findData(settings.compute_type))
        )
        self.language_input.setCurrentIndex(max(0, self.language_input.findData(settings.language)))
        self.vad_check.setChecked(settings.vad_enabled)
        self.vad_silence.setValue(settings.vad_min_silence_ms)
        self.hardware_impact.setText(PROFILE_PRESETS[settings.profile].hardware_impact)

    @Slot()
    def _apply_profile(self) -> None:
        profile = WhisperProfile(str(self.profile_input.currentData()))
        current = self.manager.whisper_settings
        self._load_settings(
            replace(
                PROFILE_PRESETS[profile].settings, first_run_completed=current.first_run_completed
            )
        )

    def _settings_from_form(self) -> WhisperSettings:
        return WhisperSettings(
            profile=WhisperProfile(str(self.profile_input.currentData())),
            model=self.model_input.text().strip(),
            device=str(self.device_input.currentData()),
            compute_type=str(self.compute_input.currentData()),
            language=str(self.language_input.currentData()),
            vad_enabled=self.vad_check.isChecked(),
            vad_min_silence_ms=self.vad_silence.value(),
            first_run_completed=self.manager.whisper_settings.first_run_completed,
        )

    @Slot()
    def _save(self) -> bool:
        try:
            advice = self.manager.configure_whisper(self._settings_from_form())
            self.manager.save_whisper()
        except Exception as exc:  # noqa: BLE001 - settings form boundary
            self._show_task_error(str(exc))
            return False
        self.fallback_advice.setText(advice or "当前设备与计算精度组合可用。")
        return True

    @Slot()
    def _prepare_model(self) -> None:
        if not self._save():
            return
        self._cancel_download = threading.Event()
        self.prepare_button.setEnabled(False)
        self.retry_button.hide()
        self.cancel_button.show()

        def work() -> None:
            self.manager.prepare_whisper_model(
                on_progress=self.download_changed.emit,
                cancel_event=self._cancel_download,
            )

        threading.Thread(target=work, name="whisper-model-download", daemon=True).start()

    @Slot()
    def _cancel_model_download(self) -> None:
        self._cancel_download.set()
        self.cancel_button.setEnabled(False)

    @Slot()
    def _run_benchmark(self) -> None:
        if not self._save():
            return
        self.benchmark_button.setEnabled(False)
        self.benchmark_result.setText("正在生成样本并测试…")

        def work() -> None:
            try:
                sample = create_builtin_whisper_sample(self.manager.settings.data_dir)
                result = self.manager.benchmark_whisper(sample)
            except Exception as exc:  # noqa: BLE001 - background benchmark boundary
                self.task_failed.emit(str(exc))
            else:
                self.benchmark_ready.emit(result)

        threading.Thread(target=work, name="whisper-benchmark", daemon=True).start()

    @Slot(object)
    def show_benchmark_result(self, result: BenchmarkResult) -> None:
        resources = result.resources
        peak_memory = resources.get("peak_memory_mb")
        memory_text = "不可用" if peak_memory is None else f"{peak_memory:.1f} MiB"
        self.benchmark_result.setText(
            f"识别文字：{result.text or '未识别到语音'}\n"
            f"处理耗时：{result.elapsed_seconds:.2f} 秒 · 实时系数：{result.realtime_factor:.2f}\n"
            f"语言：{result.language or '未知'} · CPU 时间："
            f"{resources.get('process_cpu_seconds', 0.0):.2f} 秒 · 进程峰值内存：{memory_text}"
        )
        self.benchmark_button.setEnabled(True)

    @Slot(object)
    def show_download_state(self, state: DownloadState) -> None:
        self.download_progress.setValue(state.progress_percent)
        detail = f"磁盘需求：{state.required_bytes} 字节 · 状态：{state.status}"
        if state.error:
            detail += f" · {state.error}"
        self.disk_requirement.setText(detail)
        active = state.status in {"pending", "downloading"}
        self.cancel_button.setVisible(active)
        self.cancel_button.setEnabled(active)
        self.retry_button.setVisible(state.can_retry)
        self.prepare_button.setEnabled(not active)

    @Slot(str)
    def _show_task_error(self, message: str) -> None:
        self.benchmark_button.setEnabled(True)
        self.fallback_advice.setText(message)
