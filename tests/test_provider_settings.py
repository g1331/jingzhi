import json
from dataclasses import replace

import keyring
import pytest
from keyring.errors import KeyringError

from jingzhi.config import Settings
from jingzhi.model_roles import ModelConnection, RoleName, default_roles
from jingzhi.provider_settings import (
    KEYRING_SERVICE,
    LEGACY_KEYRING_SERVICE,
    ProviderSettingsStore,
    SavedProviderSettings,
    default_saved_settings,
)
from jingzhi.session import SessionManager


def test_provider_settings_keep_api_key_out_of_json(tmp_path, monkeypatch) -> None:
    stored_secret: dict[str, str] = {}
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda _service, _username, password: stored_secret.update(value=password),
    )
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda _service, _username: stored_secret.get("value"),
    )
    monkeypatch.setattr(keyring, "delete_password", lambda _service, _username: None)
    store = ProviderSettingsStore(tmp_path)
    defaults = default_saved_settings("vision-model")
    connection = replace(
        defaults.connections[0],
        base_url="https://provider.example/v1",
        api_key="secret-key",
        api_mode="chat_completions",
    )

    store.save(SavedProviderSettings((connection,), defaults.roles))

    public_text = (tmp_path / "provider.json").read_text(encoding="utf-8")
    public = json.loads(public_text)
    assert "api_key" not in public["connections"][0]
    assert "secret-key" not in public_text
    assert store.load().connections[0].api_key == "secret-key"
    assert store.load().roles[0].model == "vision-model"


def test_provider_settings_fall_back_to_legacy_keyring_entry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda service, _username: "legacy-key" if service == LEGACY_KEYRING_SERVICE else None,
    )

    settings = ProviderSettingsStore(tmp_path).load()

    assert KEYRING_SERVICE != LEGACY_KEYRING_SERVICE
    assert settings.connections[0].api_key == "legacy-key"


def test_provider_settings_use_main_model_for_legacy_file(tmp_path, monkeypatch) -> None:
    (tmp_path / "provider.json").write_text(
        json.dumps({"version": 1, "model": "legacy-main"}), encoding="utf-8"
    )
    monkeypatch.setattr(keyring, "get_password", lambda _service, _username: None)

    settings = ProviderSettingsStore(tmp_path).load()
    roles = {role.name: role for role in settings.roles}

    assert roles[RoleName.INSTANT_ANSWER].model == "legacy-main"
    assert roles[RoleName.TRANSCRIPT_CORRECTION].model == "legacy-main"


def test_saved_role_models_survive_restart(tmp_path, monkeypatch) -> None:
    stored_secret: dict[str, str] = {}
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda _service, _username, password: stored_secret.update(value=password),
    )
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda _service, _username: stored_secret.get("value"),
    )
    monkeypatch.setattr(keyring, "delete_password", lambda _service, _username: None)
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("TRANSCRIPT_CORRECTION_MODEL", raising=False)
    defaults = default_saved_settings("main-model")
    roles = tuple(
        replace(role, model="luna") if role.name == RoleName.TRANSCRIPT_CORRECTION else role
        for role in defaults.roles
    )
    manager = SessionManager(
        Settings(
            data_dir=tmp_path, provider_settings=SavedProviderSettings(defaults.connections, roles)
        )
    )
    manager.save_provider()

    restarted = SessionManager(Settings.from_env())

    assert restarted.model_role(RoleName.INSTANT_ANSWER).model == "main-model"
    assert restarted.model_role(RoleName.TRANSCRIPT_CORRECTION).model == "luna"


def test_multi_connection_save_rolls_back_credentials_before_public_config(
    tmp_path, monkeypatch
) -> None:
    secrets: dict[str, str] = {}

    def get_password(_service, username):
        return secrets.get(username)

    def set_password(_service, username, password):
        secrets[username] = password

    def delete_password(_service, username):
        secrets.pop(username, None)

    monkeypatch.setattr(keyring, "get_password", get_password)
    monkeypatch.setattr(keyring, "set_password", set_password)
    monkeypatch.setattr(keyring, "delete_password", delete_password)
    store = ProviderSettingsStore(tmp_path)
    roles = tuple(replace(role, connection_id="primary") for role in default_roles())
    old_settings = SavedProviderSettings(
        (
            ModelConnection("primary", "主连接", api_key="old-primary"),
            ModelConnection("backup", "后备连接", api_key="old-backup"),
        ),
        roles,
    )
    store.save(old_settings)
    public_before = store.path.read_text(encoding="utf-8")
    secrets_before = secrets.copy()
    failed_once = False

    def fail_second_write(_service, username, password):
        nonlocal failed_once
        if username.endswith("backup") and password == "new-backup" and not failed_once:
            failed_once = True
            raise KeyringError("credential write failed")
        secrets[username] = password

    monkeypatch.setattr(keyring, "set_password", fail_second_write)
    new_settings = SavedProviderSettings(
        (
            ModelConnection("primary", "主连接", api_key="new-primary"),
            ModelConnection("backup", "后备连接", api_key="new-backup"),
        ),
        roles,
    )

    with pytest.raises(RuntimeError, match="原配置已恢复"):
        store.save(new_settings)

    assert store.path.read_text(encoding="utf-8") == public_before
    assert secrets == secrets_before
