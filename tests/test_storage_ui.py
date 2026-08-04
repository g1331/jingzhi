from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from jingzhi.application import JingzhiApplicationService
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.storage import StartupSettingsStore
from jingzhi.storage_ui import StorageSettingsDialog
from jingzhi.ui import MainWindow


def _settings(tmp_path: Path) -> Settings:
    store = StartupSettingsStore(
        config_dir=tmp_path / "config",
        install_dir=tmp_path / "install",
        local_app_data=tmp_path / "local",
        environ={},
        legacy_dir=tmp_path / "missing-legacy",
    )
    paths = store.resolve()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=paths.data_dir,
        model_dir=paths.model_dir,
        startup_settings_store=store,
    )


def test_storage_dialog_shows_absolute_paths_usage_free_space_and_environment_state(
    tmp_path: Path,
) -> None:
    application = QApplication.instance() or QApplication([])
    settings = _settings(tmp_path)
    (settings.data_dir / "content.bin").write_bytes(b"data")
    assert settings.model_dir is not None
    (settings.model_dir / "model.bin").write_bytes(b"model")
    dialog = StorageSettingsDialog(settings, busy_reason=lambda: None)

    dialog.refresh()
    application.processEvents()

    assert dialog.data_path.text() == str(settings.data_dir.resolve())
    assert dialog.model_path.text() == str(settings.model_dir.resolve())
    assert "4 B" in dialog.data_usage.text()
    assert "5 B" in dialog.model_usage.text()
    assert "剩余" in dialog.data_usage.text()
    assert dialog.open_data_button.isEnabled()
    assert dialog.open_model_button.isEnabled()
    dialog.close()

    managed = Settings(
        data_dir=settings.data_dir,
        model_dir=settings.model_dir,
        data_dir_managed_by_env=True,
        model_dir_managed_by_env=True,
        startup_settings_store=settings.startup_settings_store,
    )
    managed_dialog = StorageSettingsDialog(managed, busy_reason=lambda: None)
    assert "STUDY_DATA_DIR" in managed_dialog.data_management.text()
    assert "HF_HOME" in managed_dialog.model_management.text()
    assert not managed_dialog.change_data_button.isEnabled()
    assert not managed_dialog.change_model_button.isEnabled()
    managed_dialog.close()


def test_storage_dialog_exposes_models_and_blocks_busy_migration(
    tmp_path: Path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    settings = _settings(tmp_path)
    assert settings.model_dir is not None
    repository = settings.model_dir / "models--Systran--faster-whisper-small"
    repository.mkdir(parents=True)
    (repository / "model.bin").write_bytes(b"model")
    dialog = StorageSettingsDialog(settings, busy_reason=lambda: "正在录制会话")
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path / "new-data"),
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )

    dialog._change_data_directory()
    application.processEvents()

    assert dialog.models.count() == 1
    assert "small" in dialog.models.item(0).text()
    assert warnings and "正在录制会话" in warnings[0]
    dialog.close()


def test_main_window_opens_storage_settings(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])

    class IdleRecorder:
        is_recording = False

        def start(self, _title: str, **_kwargs) -> str:
            return "session"

        def stop(self) -> None:
            return None

    settings = _settings(tmp_path)
    service = JingzhiApplicationService(
        Database(tmp_path / "window.sqlite3"), recorder=IdleRecorder()
    )
    window = MainWindow(settings, service=service)

    window.storage_settings_button.click()
    application.processEvents()

    assert window._storage_dialog is not None
    assert window._storage_dialog.isVisible()
    window.close()
