from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from jingzhi.model_roles import (
    ModelConnection,
    ModelFallback,
    ModelRole,
    ReasoningLevel,
    RoleName,
    default_roles,
)

KEYRING_SERVICE = "JINGZHI"
LEGACY_KEYRING_SERVICE = "Study Companion"
KEYRING_USERNAME = "openai-compatible-api-key"
CONNECTION_KEYRING_PREFIX = "model-connection:"


@dataclass(frozen=True, slots=True)
class SavedProviderSettings:
    connections: tuple[ModelConnection, ...]
    roles: tuple[ModelRole, ...]


def default_saved_settings(
    model: str = "gpt-5.5", correction_model: str | None = None
) -> SavedProviderSettings:
    return SavedProviderSettings(
        connections=(ModelConnection("default", "默认连接"),),
        roles=default_roles(model, correction_model),
    )


class ProviderSettingsStore:
    """Persists reusable connection metadata while keeping credentials in the OS keyring."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "provider.json"

    @staticmethod
    def _keyring_username(connection_id: str) -> str:
        return f"{CONNECTION_KEYRING_PREFIX}{connection_id}"

    def _public_settings(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _read_secret(connection_id: str, *, legacy: bool = False) -> str:
        import keyring
        from keyring.errors import KeyringError

        try:
            secret = keyring.get_password(
                KEYRING_SERVICE,
                ProviderSettingsStore._keyring_username(connection_id),
            )
            if secret or not legacy:
                return secret or ""
            return (
                keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
                or keyring.get_password(LEGACY_KEYRING_SERVICE, KEYRING_USERNAME)
                or ""
            )
        except KeyringError:
            return ""

    def load(self) -> SavedProviderSettings:
        public = self._public_settings()
        if public.get("version") == 2 and isinstance(public.get("connections"), list):
            connections = tuple(self._connection_from_json(item) for item in public["connections"])
            connection_ids = {connection.id for connection in connections}
            roles = tuple(
                role
                for item in public.get("roles", [])
                if isinstance(item, dict)
                for role in (self._role_from_json(item),)
                if role.connection_id in connection_ids
            )
            return SavedProviderSettings(connections, roles)

        api_mode = str(public.get("api_mode", "responses"))
        if api_mode not in {"responses", "chat_completions"}:
            api_mode = "responses"
        model = str(public.get("model", "gpt-5.5")).strip() or "gpt-5.5"
        correction_model = str(public.get("correction_model", "")).strip() or model
        connection = ModelConnection(
            id="default",
            name="默认连接",
            base_url=str(public.get("base_url", "")).strip(),
            api_key=self._read_secret("default", legacy=True),
            api_mode=api_mode,
        )
        return SavedProviderSettings((connection,), default_roles(model, correction_model))

    def _connection_from_json(self, item: Any) -> ModelConnection:
        if not isinstance(item, dict):
            raise TypeError("Invalid model connection settings")
        connection_id = str(item.get("id", "")).strip()
        return ModelConnection(
            id=connection_id,
            name=str(item.get("name", connection_id)).strip(),
            base_url=str(item.get("base_url", "")).strip(),
            api_key=self._read_secret(connection_id),
            api_mode=str(item.get("api_mode", "responses")),
        )

    @staticmethod
    def _role_from_json(item: dict[str, Any]) -> ModelRole:
        fallbacks = tuple(
            ModelFallback(
                connection_id=str(fallback.get("connection_id", "")).strip(),
                model=str(fallback.get("model", "")).strip(),
                cross_connection_authorized=bool(
                    fallback.get("cross_connection_authorized", False)
                ),
            )
            for fallback in item.get("fallbacks", [])
            if isinstance(fallback, dict)
        )
        return ModelRole(
            name=RoleName(str(item["name"])),
            connection_id=str(item["connection_id"]).strip(),
            model=str(item["model"]).strip(),
            reasoning=ReasoningLevel(str(item["reasoning"])),
            fallbacks=fallbacks,
        )

    def save(self, settings: SavedProviderSettings) -> None:
        connection_ids = {connection.id for connection in settings.connections}
        if len(connection_ids) != len(settings.connections):
            raise ValueError("Connection IDs must be unique")
        if not connection_ids:
            raise ValueError("At least one model connection is required")
        for role in settings.roles:
            if role.connection_id not in connection_ids:
                raise ValueError(f"Unknown connection for role {role.name}: {role.connection_id}")
            if not role.model.strip():
                raise ValueError(f"Model is required for role {role.name}")
            for fallback in role.fallbacks:
                if fallback.connection_id not in connection_ids:
                    raise ValueError(f"Unknown fallback connection: {fallback.connection_id}")
                if not fallback.model.strip():
                    raise ValueError(f"Fallback model is required for role {role.name}")

        previous_ids = {
            str(item.get("id", ""))
            for item in self._public_settings().get("connections", [])
            if isinstance(item, dict)
        }
        public = {
            "version": 2,
            "connections": [
                asdict(replace(connection, api_key="")) | {"api_key": None}
                for connection in settings.connections
            ],
            "roles": [asdict(role) for role in settings.roles],
        }
        for connection in public["connections"]:
            connection.pop("api_key")
        import keyring
        from keyring.errors import KeyringError

        affected_ids = connection_ids | previous_ids
        previous_secrets: dict[str, str | None] = {}
        temporary = self.path.with_suffix(".json.tmp")
        try:
            for connection_id in affected_ids:
                previous_secrets[connection_id] = keyring.get_password(
                    KEYRING_SERVICE, self._keyring_username(connection_id)
                )
            for connection in settings.connections:
                username = self._keyring_username(connection.id)
                if connection.api_key:
                    keyring.set_password(KEYRING_SERVICE, username, connection.api_key)
                elif previous_secrets[connection.id] is not None:
                    keyring.delete_password(KEYRING_SERVICE, username)
            for removed_id in previous_ids - connection_ids:
                if previous_secrets[removed_id] is not None:
                    keyring.delete_password(KEYRING_SERVICE, self._keyring_username(removed_id))

            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, self.path)
        except (KeyringError, OSError) as exc:
            rollback_error: KeyringError | None = None
            for connection_id, secret in previous_secrets.items():
                username = self._keyring_username(connection_id)
                try:
                    current = keyring.get_password(KEYRING_SERVICE, username)
                    if secret is None:
                        if current is not None:
                            keyring.delete_password(KEYRING_SERVICE, username)
                    else:
                        keyring.set_password(KEYRING_SERVICE, username, secret)
                except KeyringError as rollback_exc:
                    rollback_error = rollback_exc
            temporary.unlink(missing_ok=True)
            message = "无法完整保存模型连接，原配置已恢复"
            if rollback_error is not None:
                message = "模型连接保存失败，且 Windows 凭据回滚未完整完成"
            raise RuntimeError(message) from exc
