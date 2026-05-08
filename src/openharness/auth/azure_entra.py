"""Microsoft Entra ID (Azure AD) authentication for Azure OpenAI provider.

This module is a thin layer on top of the ``azure-identity`` SDK.  The active
auth chain is :class:`azure.identity.DefaultAzureCredential`, which
automatically picks up credentials from any of the following, in order:

* Environment variables (``AZURE_CLIENT_ID`` / ``AZURE_TENANT_ID`` /
  ``AZURE_CLIENT_SECRET`` …)
* Workload Identity (federated tokens for AKS / GitHub Actions OIDC)
* Managed Identity (system- or user-assigned, on Azure compute)
* Shared developer credentials (``az login``, Azure PowerShell,
  Azure Developer CLI, IntelliJ / VS Code Azure account)
* Interactive browser as a last resort (when running locally with a TTY)

Usage example::

    provider = build_token_provider(tenant_id="contoso.onmicrosoft.com")
    token = provider()  # blocking call returning a Bearer token (string)

The ``azure-identity`` package is an *optional* dependency — install it via::

    pip install openharness-ai[azure]
    # or
    pip install azure-identity
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

# Default scope for the Azure OpenAI cognitive services data plane.
# (``.default`` asks AAD for whatever scopes the resource is configured for.)
AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"

# The Azure CLI's first-party application ID — used as a sane default
# ``client_id`` for ``InteractiveBrowserCredential`` so that local devs do not
# need to register their own AAD app.
AZURE_CLI_CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"

# Refresh tokens proactively this many seconds before expiry.
_TOKEN_REFRESH_SKEW_SECONDS = 300


class AzureIdentityNotInstalled(ImportError):
    """Raised when the ``azure-identity`` package is not available."""


def _import_azure_identity() -> Any:
    """Import :mod:`azure.identity` lazily so the rest of OpenHarness still
    works in environments where the optional dependency is missing.
    """
    try:
        import azure.identity as az_identity  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise AzureIdentityNotInstalled(
            "Azure Entra ID auth requires the 'azure-identity' package.\n"
            "Install it with:\n"
            "    pip install azure-identity\n"
            "  or\n"
            "    pip install openharness-ai[azure]"
        ) from exc
    return az_identity


@dataclass
class AzureEntraConfig:
    """Resolved configuration for the Azure OpenAI Entra ID auth flow."""

    tenant_id: str | None = None
    client_id: str | None = None
    azure_endpoint: str = ""
    api_version: str = ""
    deployment: str = ""
    allow_interactive_browser: bool = True

    def credential_kwargs(self) -> dict[str, Any]:
        """Translate this config into kwargs for ``DefaultAzureCredential``."""
        kwargs: dict[str, Any] = {
            # Disable noisy authority validation when tenant_id is not
            # explicitly set (DefaultAzureCredential probes "common" otherwise).
            "exclude_interactive_browser_credential": not self.allow_interactive_browser,
        }
        if self.tenant_id:
            # Restrict the credential chain to a specific tenant.
            kwargs["tenant_id"] = self.tenant_id
        if self.client_id:
            # Use a non-default client ID for the interactive flow.
            kwargs["interactive_browser_client_id"] = self.client_id
        return kwargs


def build_credential(config: AzureEntraConfig | None = None) -> Any:
    """Construct a long-lived ``DefaultAzureCredential`` instance."""
    az_identity = _import_azure_identity()
    cfg = config or AzureEntraConfig()
    log.debug("Building DefaultAzureCredential (tenant=%s, interactive=%s)",
              cfg.tenant_id or "<auto>", cfg.allow_interactive_browser)
    return az_identity.DefaultAzureCredential(**cfg.credential_kwargs())


class _CachedTokenProvider:
    """Callable that lazily acquires & caches an Entra ID access token.

    Implements the signature expected by the ``openai`` SDK's
    ``azure_ad_token_provider`` parameter: ``() -> str``.

    The token is refreshed proactively a few minutes before its declared
    expiry so the underlying HTTPS call rarely sees a fresh-auth latency
    spike.
    """

    def __init__(
        self,
        credential: Any,
        scope: str = AZURE_OPENAI_SCOPE,
    ) -> None:
        self._credential = credential
        self._scope = scope
        self._lock = threading.Lock()
        self._cached_token: str | None = None
        self._expires_at_epoch: float = 0.0

    def __call__(self) -> str:
        now = time.time()
        # Fast path: cached token is still comfortably valid.
        if self._cached_token and now < self._expires_at_epoch - _TOKEN_REFRESH_SKEW_SECONDS:
            return self._cached_token

        with self._lock:
            # Re-check inside the lock (another thread may have refreshed).
            if self._cached_token and time.time() < self._expires_at_epoch - _TOKEN_REFRESH_SKEW_SECONDS:
                return self._cached_token
            log.debug("Acquiring Azure Entra ID token (scope=%s)", self._scope)
            token = self._credential.get_token(self._scope)
            self._cached_token = token.token
            self._expires_at_epoch = float(token.expires_on)
            return self._cached_token

    def force_refresh(self) -> str:
        """Force a fresh token acquisition (e.g. after a 401)."""
        with self._lock:
            self._cached_token = None
            self._expires_at_epoch = 0.0
        return self()


def build_token_provider(
    config: AzureEntraConfig | None = None,
    *,
    credential: Any | None = None,
    scope: str = AZURE_OPENAI_SCOPE,
) -> Callable[[], str]:
    """Return a ``() -> str`` callable suitable for ``azure_ad_token_provider``.

    Args:
        config: Optional :class:`AzureEntraConfig` controlling the credential
            chain.  If ``credential`` is supplied, ``config`` is ignored.
        credential: Pre-built credential to wrap.  When omitted, a new
            ``DefaultAzureCredential`` is built from ``config``.
        scope: Token audience.  Defaults to the Azure OpenAI data plane.
    """
    cred = credential or build_credential(config)
    return _CachedTokenProvider(cred, scope=scope)


def probe_credential(config: AzureEntraConfig | None = None) -> tuple[bool, str]:
    """Try to acquire a token to validate the user's credential chain.

    Returns ``(True, "")`` on success or ``(False, error_message)`` on
    failure.  Used by the ``oh auth azure-login`` CLI to give the user
    immediate feedback during setup.
    """
    try:
        provider = build_token_provider(config)
        provider()
    except AzureIdentityNotInstalled as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - depends on local Azure setup
        return False, f"Azure Entra ID token acquisition failed: {exc}"
    return True, ""


__all__ = [
    "AZURE_CLI_CLIENT_ID",
    "AZURE_OPENAI_SCOPE",
    "AzureEntraConfig",
    "AzureIdentityNotInstalled",
    "build_credential",
    "build_token_provider",
    "probe_credential",
]
