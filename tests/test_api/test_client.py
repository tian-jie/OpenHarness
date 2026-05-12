from __future__ import annotations

from openharness.engine.messages import ConversationMessage, ImageBlock, TextBlock
from openharness.api.client import AnthropicApiClient, OAUTH_BETA_HEADER


def test_anthropic_client_adds_oauth_beta_header(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openharness.api.client.AsyncAnthropic", _FakeAsyncAnthropic)

    AnthropicApiClient(auth_token="oauth-token")

    assert captured["auth_token"] == "oauth-token"
    assert captured["default_headers"] == {"anthropic-beta": OAUTH_BETA_HEADER}


def test_anthropic_client_uses_api_key_without_oauth_beta(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openharness.api.client.AsyncAnthropic", _FakeAsyncAnthropic)

    AnthropicApiClient(api_key="api-key")

    assert captured["api_key"] == "api-key"
    assert "default_headers" not in captured


def test_anthropic_client_adds_claude_oauth_identity_headers(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openharness.api.client.AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setattr(
        "openharness.auth.external.get_claude_code_version",
        lambda: "2.1.92",
    )
    monkeypatch.setattr(
        "openharness.auth.external.get_claude_code_session_id",
        lambda: "session-123",
    )
    monkeypatch.setattr(
        "openharness.api.client.get_claude_code_session_id",
        lambda: "session-123",
    )
    monkeypatch.setattr(
        "openharness.api.client.claude_attribution_header",
        lambda: "x-anthropic-billing-header: cc_version=2.1.92; cc_entrypoint=cli;",
    )

    AnthropicApiClient(auth_token="oauth-token", claude_oauth=True)

    headers = captured["default_headers"]
    assert captured["auth_token"] == "oauth-token"
    assert headers["x-app"] == "cli"
    assert headers["user-agent"] == "claude-cli/2.1.92 (external, cli)"
    assert headers["X-Claude-Code-Session-Id"] == "session-123"
    assert "oauth-2025-04-20" in headers["anthropic-beta"]
    assert "claude-code-20250219" in headers["anthropic-beta"]


def test_anthropic_client_can_disable_auth_token_beta_header(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openharness.api.client.AsyncAnthropic", _FakeAsyncAnthropic)

    AnthropicApiClient(
        auth_token="bearer-token",
        include_auth_token_beta_header=False,
    )

    assert captured["auth_token"] == "bearer-token"
    assert "default_headers" not in captured


def test_anthropic_client_refreshes_auth_with_resolver(monkeypatch):
    calls: list[dict[str, object]] = []

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))

    monkeypatch.setattr("openharness.api.client.AsyncAnthropic", _FakeAsyncAnthropic)

    current_token = {"value": "token-1"}

    client = AnthropicApiClient(
        auth_token="token-1",
        auth_token_resolver=lambda: current_token["value"],
    )
    assert calls[-1]["auth_token"] == "token-1"

    current_token["value"] = "token-2"
    client._refresh_client_auth()
    assert calls[-1]["auth_token"] == "token-2"


def test_anthropic_client_can_skip_tls_verification_with_env_flag(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeHttpClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setenv("OPENHARNESS_INSECURE_SKIP_TLS_VERIFY", "1")
    monkeypatch.setattr("openharness.api.client.AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setattr("openharness.api.client.httpx.AsyncClient", _FakeHttpClient)

    AnthropicApiClient(auth_token="oauth-token")

    http_client = captured["http_client"]
    assert isinstance(http_client, _FakeHttpClient)
    assert http_client.kwargs["verify"] is False


def test_conversation_message_serializes_image_block_for_anthropic():
    message = ConversationMessage(
        role="user",
        content=[
            TextBlock(text="Describe this."),
            ImageBlock(media_type="image/png", data="YWJj", source_path="/tmp/example.png"),
        ],
    )

    assert message.to_api_param() == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this."},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "YWJj",
                },
            },
        ],
    }


def test_anthropic_client_refreshes_claude_token_on_request(monkeypatch):
    captured_tokens: list[str] = []

    class _FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def __aiter__(self):
            if False:
                yield None
            return

        async def get_final_message(self):
            class _Usage:
                input_tokens = 1
                output_tokens = 1

            class _Message:
                usage = _Usage()
                stop_reason = "end_turn"
                role = "assistant"
                content = []

            return _Message()

    class _FakeMessages:
        def __init__(self):
            self.last_params = None

        def stream(self, **params):
            self.last_params = params
            return _FakeStream()

    class _FakeBeta:
        def __init__(self):
            self.messages = _FakeMessages()

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured_tokens.append(kwargs["auth_token"])
            self.beta = _FakeBeta()
            self.messages = _FakeMessages()

    monkeypatch.setattr("openharness.api.client.AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setattr(
        "openharness.auth.external.get_claude_code_session_id",
        lambda: "session-123",
    )
    monkeypatch.setattr(
        "openharness.api.client.get_claude_code_session_id",
        lambda: "session-123",
    )
    monkeypatch.setattr(
        "openharness.api.client.claude_attribution_header",
        lambda: "x-anthropic-billing-header: cc_version=2.1.92; cc_entrypoint=cli;",
    )

    current_token = {"value": "initial-token"}

    client = AnthropicApiClient(
        auth_token="initial-token",
        claude_oauth=True,
        auth_token_resolver=lambda: current_token["value"],
    )
    current_token["value"] = "refreshed-token"

    from openharness.api.client import ApiMessageRequest

    async def _run():
        events = []
        async for event in client.stream_message(
            ApiMessageRequest(
                model="claude-sonnet-4-6",
                messages=[],
                system_prompt="system prompt",
            )
        ):
            events.append(event)
        return events

    import asyncio

    events = asyncio.run(_run())

    assert captured_tokens == ["initial-token", "refreshed-token"]
    assert events
    assert client._client.beta.messages.last_params["metadata"] == {
        "user_id": '{"device_id":"openharness","session_id":"session-123","account_uuid":""}'
    }
    assert "oauth-2025-04-20" in client._client.beta.messages.last_params["betas"]
    assert client._client.beta.messages.last_params["system"].startswith(
        "x-anthropic-billing-header: cc_version=2.1.92; cc_entrypoint=cli;\n"
    )
