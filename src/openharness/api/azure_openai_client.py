"""Azure OpenAI Service client with Microsoft Entra ID authentication.

This client reuses the streaming / tool-call / retry machinery from
:class:`openharness.api.openai_client.OpenAICompatibleClient` and only
swaps out the underlying SDK constructor: ``AsyncAzureOpenAI`` accepts
an ``azure_ad_token_provider`` callable which we wire to the cached
Entra ID token provider built in :mod:`openharness.auth.azure_entra`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from openai import AsyncAzureOpenAI

from openharness.api.openai_client import OpenAICompatibleClient

log = logging.getLogger(__name__)

# Newest stable Azure OpenAI API version that supports tool/function calling
# and streaming as of this writing.  Profiles may override on a per-deployment
# basis (e.g. ``2025-01-01-preview`` for newest preview features).
DEFAULT_AZURE_API_VERSION = "2024-10-21"


class AzureOpenAIClient(OpenAICompatibleClient):
    """OpenAI-compatible client backed by Azure OpenAI + Entra ID.

    Args:
        azure_endpoint: Azure OpenAI resource URL,
            e.g. ``https://my-aoai.openai.azure.com/``.
        api_version: Azure OpenAI API version (e.g. ``2024-10-21``).
        token_provider: ``() -> str`` callable returning a fresh Bearer
            token.  Typically built via
            :func:`openharness.auth.azure_entra.build_token_provider`.
        api_key: Optional Azure OpenAI key (used when Entra ID is
            unavailable).  When supplied, ``token_provider`` is ignored.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        azure_endpoint: str,
        api_version: str | None = None,
        token_provider: Callable[[], str] | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint (base_url) is required.")
        if not (token_provider or api_key):
            raise ValueError(
                "Azure OpenAI client requires either a token_provider "
                "(Entra ID) or an api_key."
            )

        kwargs: dict[str, Any] = {
            "azure_endpoint": azure_endpoint.rstrip("/"),
            "api_version": api_version or DEFAULT_AZURE_API_VERSION,
        }
        if token_provider is not None:
            kwargs["azure_ad_token_provider"] = token_provider
        else:
            kwargs["api_key"] = api_key
        if timeout is not None:
            kwargs["timeout"] = timeout

        # Bypass the parent constructor — it builds AsyncOpenAI which is
        # the wrong underlying SDK for Azure routing.
        self._client = AsyncAzureOpenAI(**kwargs)
        log.debug(
            "Initialized AzureOpenAIClient (endpoint=%s, api_version=%s, "
            "auth=%s)",
            azure_endpoint,
            kwargs["api_version"],
            "entra_id" if token_provider else "api_key",
        )


__all__ = ["AzureOpenAIClient", "DEFAULT_AZURE_API_VERSION"]
