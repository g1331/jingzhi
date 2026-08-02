import json

import keyring

from jingzhi.provider_settings import (
    KEYRING_SERVICE,
    LEGACY_KEYRING_SERVICE,
    ProviderSettingsStore,
    SavedProviderSettings,
)


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
