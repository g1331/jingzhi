from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jingzhi.config import Settings
from jingzhi.storage import MigrationResult, StartupSettingsStore, StorageManager


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


class StorageSettingsDialog(QDialog):
    operation_finished = Signal(str, object)
    operation_failed = Signal(str)

    def __init__(
        self,
        settings: Settings,
        *,
        busy_reason: Callable[[], str | None],
        model_in_use: Callable[[str], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.busy_reason = busy_reason
        self._worker: threading.Thread | None = None
        store = settings.startup_settings_store or StartupSettingsStore()
        self.manager = StorageManager(
            settings.storage_paths,
            store,
            busy_reason=busy_reason,
            model_in_use=model_in_use,
        )
        self.setWindowTitle("存储")
        self.setMinimumSize(760, 520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        intro = QLabel("会话数据是你的本地资产；Whisper 模型是可以重新下载的缓存。两者独立管理。")
        intro.setWordWrap(True)
        intro.setObjectName("subtitle")
        layout.addWidget(intro)

        data_group = QGroupBox("应用数据")
        data_layout = QGridLayout(data_group)
        self.data_path = QLabel()
        self.data_path.setTextInteractionFlags(self.data_path.textInteractionFlags())
        self.data_path.setWordWrap(True)
        self.data_usage = QLabel()
        self.data_management = QLabel()
        self.data_management.setObjectName("hint")
        self.open_data_button = QPushButton("打开目录")
        self.change_data_button = QPushButton("迁移数据")
        data_actions = QHBoxLayout()
        data_actions.addWidget(self.open_data_button)
        data_actions.addWidget(self.change_data_button)
        data_actions.addStretch(1)
        data_layout.addWidget(QLabel("位置"), 0, 0)
        data_layout.addWidget(self.data_path, 0, 1)
        data_layout.addWidget(QLabel("空间"), 1, 0)
        data_layout.addWidget(self.data_usage, 1, 1)
        data_layout.addWidget(self.data_management, 2, 0, 1, 2)
        data_layout.addLayout(data_actions, 3, 0, 1, 2)
        layout.addWidget(data_group)

        model_group = QGroupBox("Whisper 模型缓存")
        model_layout = QGridLayout(model_group)
        self.model_path = QLabel()
        self.model_path.setWordWrap(True)
        self.model_usage = QLabel()
        self.model_management = QLabel()
        self.model_management.setObjectName("hint")
        self.models = QListWidget()
        self.models.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.models.setMinimumHeight(110)
        self.open_model_button = QPushButton("打开目录")
        self.change_model_button = QPushButton("更改目录")
        self.delete_model_button = QPushButton("删除所选模型")
        model_actions = QHBoxLayout()
        model_actions.addWidget(self.open_model_button)
        model_actions.addWidget(self.change_model_button)
        model_actions.addWidget(self.delete_model_button)
        model_actions.addStretch(1)
        model_layout.addWidget(QLabel("位置"), 0, 0)
        model_layout.addWidget(self.model_path, 0, 1)
        model_layout.addWidget(QLabel("空间"), 1, 0)
        model_layout.addWidget(self.model_usage, 1, 1)
        model_layout.addWidget(self.model_management, 2, 0, 1, 2)
        model_layout.addWidget(self.models, 3, 0, 1, 2)
        model_layout.addLayout(model_actions, 4, 0, 1, 2)
        layout.addWidget(model_group, 1)

        self.status = QLabel()
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=self.layout().alignment())

        self.open_data_button.clicked.connect(lambda: self._open_directory(self.settings.data_dir))
        self.open_model_button.clicked.connect(
            lambda: self._open_directory(self.settings.model_dir)
        )
        self.change_data_button.clicked.connect(self._change_data_directory)
        self.change_model_button.clicked.connect(self._change_model_directory)
        self.delete_model_button.clicked.connect(self._delete_selected_model)
        self.operation_finished.connect(self._operation_completed)
        self.operation_failed.connect(self._operation_failed)
        self.refresh()

    @property
    def operation_active(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def refresh(self) -> None:
        data = self.manager.data_usage()
        models = self.manager.model_usage()
        self.data_path.setText(str(data.path))
        self.model_path.setText(str(models.path))
        self.data_usage.setText(
            f"当前占用 {format_bytes(data.used_bytes)} · 磁盘剩余 {format_bytes(data.free_bytes)}"
        )
        self.model_usage.setText(
            f"当前占用 {format_bytes(models.used_bytes)} · 磁盘剩余 {format_bytes(models.free_bytes)}"
        )
        self.data_management.setText(
            "由环境变量 STUDY_DATA_DIR 管理，界面不能修改。"
            if self.settings.data_dir_managed_by_env
            else "迁移成功后将在下次启动使用新目录；旧目录不会自动删除。"
        )
        self.model_management.setText(
            "由环境变量 HF_HOME/HF_HUB_CACHE 管理，界面不能修改。"
            if self.settings.model_dir_managed_by_env
            else "更改目录时可以迁移已有模型，也可以在需要时重新下载。"
        )
        self.change_data_button.setEnabled(not self.settings.data_dir_managed_by_env)
        self.change_model_button.setEnabled(not self.settings.model_dir_managed_by_env)
        self.models.clear()
        for model in self.manager.models():
            self.models.addItem(f"{model.name} · {format_bytes(model.size_bytes)}")
        self.delete_model_button.setEnabled(self.models.count() > 0)

    def _open_directory(self, path: Path | None) -> None:
        if path is None:
            return
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _check_idle(self) -> bool:
        reason = self.busy_reason()
        if not reason:
            return True
        QMessageBox.warning(self, "当前不能更改存储目录", reason)
        return False

    @Slot()
    def _change_data_directory(self) -> None:
        if not self._check_idle():
            return
        selected = QFileDialog.getExistingDirectory(
            self, "选择新的应用数据目录", str(self.settings.data_dir.parent)
        )
        if not selected:
            return
        self._run_operation("data", lambda: self.manager.migrate_data(Path(selected)))

    @Slot()
    def _change_model_directory(self) -> None:
        if not self._check_idle():
            return
        model_dir = self.settings.model_dir
        assert model_dir is not None
        selected = QFileDialog.getExistingDirectory(
            self, "选择新的 Whisper 模型目录", str(model_dir.parent)
        )
        if not selected:
            return
        choice = QMessageBox.question(
            self,
            "处理已有模型",
            "要把已下载模型迁移到新目录吗？选择“否”将在需要时重新下载。",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return
        move_existing = choice == QMessageBox.StandardButton.Yes
        self._run_operation(
            "models",
            lambda: self.manager.migrate_models(Path(selected), move_existing=move_existing),
        )

    def _run_operation(self, kind: str, operation: Callable[[], MigrationResult]) -> None:
        if self.operation_active:
            return
        self.change_data_button.setEnabled(False)
        self.change_model_button.setEnabled(False)
        self.status.setText("正在校验并复制，请勿关闭应用……")

        def work() -> None:
            try:
                result = operation()
            except Exception as exc:  # noqa: BLE001 - transferred to the UI thread
                self.operation_failed.emit(str(exc).strip() or type(exc).__name__)
                return
            self.operation_finished.emit(kind, result)

        self._worker = threading.Thread(target=work, name=f"storage-migration-{kind}")
        self._worker.start()

    @Slot(str, object)
    def _operation_completed(self, kind: str, result: MigrationResult) -> None:
        self._worker = None
        if kind == "data":
            self.data_path.setText(str(result.new_dir))
            self.status.setText("应用数据迁移完成。重启境织后使用新目录。")
            confirm = QMessageBox.question(
                self,
                "保留旧数据",
                "已完整保留旧目录。是否确认在下次启动新目录成功后删除旧目录？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self.manager.confirm_delete_old_data(result.old_dir)
        else:
            self.model_path.setText(str(result.new_dir))
            self.status.setText("Whisper 模型目录已更新。重启境织后使用新目录。")
        self._restore_buttons()

    @Slot(str)
    def _operation_failed(self, message: str) -> None:
        self._worker = None
        self.status.setText("更改失败；境织继续使用原目录。")
        self._restore_buttons()
        QMessageBox.warning(self, "存储目录未更改", message)

    def _restore_buttons(self) -> None:
        self.change_data_button.setEnabled(not self.settings.data_dir_managed_by_env)
        self.change_model_button.setEnabled(not self.settings.model_dir_managed_by_env)

    @Slot()
    def _delete_selected_model(self) -> None:
        row = self.models.currentRow()
        available = self.manager.models()
        if row < 0 or row >= len(available):
            return
        model = available[row]
        confirm = QMessageBox.question(
            self,
            "删除 Whisper 模型",
            f"删除“{model.name}”后，下次使用时需要重新下载。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self.manager.delete_model(model.name)
        except Exception as exc:  # noqa: BLE001 - user action boundary
            QMessageBox.warning(self, "不能删除模型", str(exc))
            return
        self.refresh()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.operation_active:
            QMessageBox.warning(self, "存储迁移正在进行", "请等待迁移完成后再关闭窗口。")
            event.ignore()
            return
        event.accept()
