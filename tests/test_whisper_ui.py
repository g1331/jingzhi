from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QWidget,
)

from jingzhi.ui import WhisperSettingsDialog
from jingzhi.whisper_settings import (
    PROFILE_PRESETS,
    BenchmarkResult,
    DownloadState,
    WhisperCapabilities,
    WhisperProfile,
)


class FakeWhisperManager:
    def __init__(self, data_dir: Path) -> None:
        self.settings = type("Settings", (), {"data_dir": data_dir})()
        self.whisper_settings = PROFILE_PRESETS[WhisperProfile.BALANCED].settings
        self.whisper_capabilities = WhisperCapabilities(
            devices=("cpu",), compute_types={"cpu": ("int8", "float32")}
        )
        self.configured = []
        self.saved = 0

    def configure_whisper(self, settings):
        self.whisper_settings = settings
        self.configured.append(settings)
        if settings.device == "cuda":
            return "当前环境不支持 CUDA；已回退到 CPU 和 int8。"
        return ""

    def save_whisper(self) -> None:
        self.saved += 1


def test_whisper_dialog_exposes_profiles_hardware_impact_and_advanced_settings(
    tmp_path: Path,
) -> None:
    _application = QApplication.instance() or QApplication([])
    manager = FakeWhisperManager(tmp_path)
    dialog = WhisperSettingsDialog(manager)

    profile = dialog.findChild(QComboBox, "whisperProfile")
    assert profile is not None
    assert [profile.itemData(index) for index in range(profile.count())] == [
        "lightweight",
        "balanced",
        "accurate",
    ]
    impact = dialog.findChild(QLabel, "whisperHardwareImpact")
    assert impact is not None
    assert "显卡" in impact.text() or "处理器" in impact.text()
    for object_name in (
        "whisperModel",
        "whisperDevice",
        "whisperComputeType",
        "whisperLanguage",
        "whisperVadSilence",
    ):
        assert dialog.findChild(QWidget, object_name) is not None

    profile.setCurrentIndex(profile.findData("accurate"))
    model = dialog.findChild(QLineEdit, "whisperModel")
    assert model is not None
    assert model.text() == "medium"
    dialog.close()


def test_whisper_dialog_saves_and_shows_actionable_fallback(tmp_path: Path) -> None:
    _application = QApplication.instance() or QApplication([])
    manager = FakeWhisperManager(tmp_path)
    dialog = WhisperSettingsDialog(manager)
    device = dialog.findChild(QComboBox, "whisperDevice")
    assert device is not None
    device.setCurrentIndex(device.findData("cuda"))

    save = dialog.findChild(QPushButton, "saveWhisper")
    assert save is not None
    save.click()

    fallback = dialog.findChild(QLabel, "whisperFallbackAdvice")
    assert fallback is not None
    assert "CUDA" in fallback.text()
    assert "CPU" in fallback.text()
    assert manager.saved == 1
    dialog.close()


def test_whisper_dialog_renders_benchmark_and_download_states(tmp_path: Path) -> None:
    _application = QApplication.instance() or QApplication([])
    dialog = WhisperSettingsDialog(FakeWhisperManager(tmp_path))
    result = BenchmarkResult(
        text="境织样本",
        elapsed_seconds=0.75,
        realtime_factor=0.38,
        language="zh",
        resources={"process_cpu_seconds": 0.3, "peak_memory_mb": 88.0},
        actual_settings=PROFILE_PRESETS[WhisperProfile.BALANCED].settings,
    )

    dialog.show_benchmark_result(result)
    benchmark = dialog.findChild(QLabel, "whisperBenchmarkResult")
    assert benchmark is not None
    assert "境织样本" in benchmark.text()
    assert "0.75" in benchmark.text()
    assert "0.38" in benchmark.text()
    assert "88.0" in benchmark.text()

    downloading = DownloadState("small", required_bytes=1_000, downloaded_bytes=400)
    downloading.status = "downloading"
    dialog.show_download_state(downloading)
    progress = dialog.findChild(QProgressBar, "whisperDownloadProgress")
    assert progress is not None
    assert progress.value() == 40
    disk = dialog.findChild(QLabel, "whisperDiskRequirement")
    assert disk is not None
    assert "1000" in disk.text()

    downloading.fail("network unavailable")
    dialog.show_download_state(downloading)
    retry = dialog.findChild(QPushButton, "retryWhisperDownload")
    assert retry is not None and not retry.isHidden()
    assert "network unavailable" in disk.text()
    dialog.close()
