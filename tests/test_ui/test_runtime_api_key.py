"""Tests for build_runtime auth failure handling."""

from __future__ import annotations

import pytest

from openharness.config.settings import Settings
from openharness.ui.runtime import _resolve_api_client_from_settings
from openharness.ui.runtime import build_runtime


@pytest.mark.asyncio
async def test_build_runtime_exits_cleanly_when_auth_resolution_fails(monkeypatch):
    """build_runtime should raise SystemExit(1) — not ValueError — when auth resolution fails."""

    def fake_resolve_auth(self):
        raise ValueError("No credentials found")

    monkeypatch.setattr("openharness.config.settings.Settings.resolve_auth", fake_resolve_auth)

    with pytest.raises(SystemExit, match="1"):
        await build_runtime(active_profile="claude-api")


@pytest.mark.asyncio
async def test_build_runtime_exits_cleanly_for_openai_format(monkeypatch):
    """Same check for the openai-compatible path."""

    def fake_resolve_auth(self):
        raise ValueError("No credentials found")

    monkeypatch.setattr("openharness.config.settings.Settings.resolve_auth", fake_resolve_auth)

    with pytest.raises(SystemExit, match="1"):
        await build_runtime(active_profile="openai-compatible", api_format="openai")


def test_resolve_runtime_client_for_azure_anthropic(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAnthropicClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openharness.ui.runtime.AnthropicApiClient", _FakeAnthropicClient)
    monkeypatch.setattr(
        "openharness.auth.azure_entra.build_token_provider",
        lambda config: (lambda: "entra-token"),
    )

    settings = Settings(active_profile="azure-anthropic")
    settings = settings.sync_active_profile_from_flat_fields().model_copy(
        update={
            "profiles": {
                **settings.profiles,
                "azure-anthropic": settings.profiles["azure-anthropic"].model_copy(
                    update={
                        "base_url": "https://example.services.ai.azure.com/anthropic/v1/messages",
                        "tenant_id": "contoso.onmicrosoft.com",
                    }
                ),
            }
        }
    )

    _resolve_api_client_from_settings(settings)

    assert captured["auth_token"] == "entra-token"
    assert captured["base_url"] == "https://example.services.ai.azure.com/anthropic"
    assert captured["include_auth_token_beta_header"] is False
