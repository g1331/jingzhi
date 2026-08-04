import json

import keyring

from jingzhi.config import Settings
from jingzhi.provider_settings import (
    KEYRING_SERVICE,
    LEGACY_KEYRING_SERVICE,
    ProviderSettingsStore,
    SavedProviderSettings,
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

    store.save(
        SavedProviderSettings(
            base_url="https://provider.example/v1",
            api_key="secret-key",
            model="vision-model",
            api_mode="chat_completions",
        )
    )

    public = json.loads((tmp_path / "provider.json").read_text(encoding="utf-8"))
    assert "api_key" not in public
    assert "secret-key" not in json.dumps(public)
    assert store.load().api_key == "secret-key"
    assert store.load().model == "vision-model"


def test_provider_settings_fall_back_to_legacy_keyring_entry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda service, _username: "legacy-key" if service == LEGACY_KEYRING_SERVICE else None,
    )

    settings = ProviderSettingsStore(tmp_path).load()

    assert KEYRING_SERVICE != LEGACY_KEYRING_SERVICE
    assert settings.api_key == "legacy-key"


def test_provider_settings_use_main_model_for_legacy_file(tmp_path, monkeypatch) -> None:
    (tmp_path / "provider.json").write_text(
        json.dumps({"version": 1, "model": "legacy-main"}), encoding="utf-8"
    )
    monkeypatch.setattr(keyring, "get_password", lambda _service, _username: None)

    settings = ProviderSettingsStore(tmp_path).load()

    assert settings.correction_model == "legacy-main"


def test_saved_correction_model_survives_restart(tmp_path, monkeypatch) -> None:
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

    manager = SessionManager(
        Settings(
            data_dir=tmp_path,
            llm_model="main-model",
            transcript_correction_model="main-model",
        )
    )
    manager.configure_transcript_correction(enabled=True, window_seconds=30, model="luna")
    manager.save_provider()

    restarted = SessionManager(Settings.from_env())

    assert restarted.llm_model == "main-model"
    assert restarted.correction_model == "luna"
