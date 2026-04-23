from __future__ import annotations

from scripts.feishu_oauth_flow import _build_auth_url, _redirect_server_config, _scopes_from_args


class _Args:
    def __init__(self, scope):
        self.scope = scope


def test_scopes_from_args_dedupes_and_splits_tokens():
    args = _Args(["im:message,im:message:send_as_bot", "im:message", "im:chat:readonly"])
    scopes = _scopes_from_args(args)
    assert scopes == ["im:message", "im:message:send_as_bot", "im:chat:readonly"]


def test_build_auth_url_contains_expected_fields():
    url = _build_auth_url(
        "cli_test",
        "http://localhost:3000/callback",
        ["im:message", "im:message:send_as_bot"],
    )
    assert "app_id=cli_test" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A3000%2Fcallback" in url
    assert "scope=im%3Amessage+im%3Amessage%3Asend_as_bot" in url


def test_redirect_server_config_parses_defaults():
    host, port, path = _redirect_server_config("http://localhost:3000/callback")
    assert host == "localhost"
    assert port == 3000
    assert path == "/callback"
