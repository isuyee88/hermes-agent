"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None

    def test_single_dict_backwards_compat(self):
        fb = {"provider": "openai", "model": "gpt-4o"}
        agent = _make_agent(fallback_model=fb)
        assert agent._fallback_chain == [fb]
        assert agent._fallback_model == fb

    def test_list_of_providers(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 2
        assert agent._fallback_model == fbs[0]

    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"

    def test_empty_list(self):
        agent = _make_agent(fallback_model=[])
        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    def test_second_fallback_works(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._try_activate_fallback() is True
            assert agent.model == "glm-4.7"
            assert agent._fallback_index == 2

    def test_all_exhausted_returns_false(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False

    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_feishu_limits_fallback_activations_to_one(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.platform = "feishu"
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False

    def test_feishu_retry_policy_prefers_fast_fail(self):
        agent = _make_agent()
        agent.platform = "feishu"

        assert agent._max_retry_attempts_for_request() == 2
        assert agent._cap_retry_wait_for_request(12.0) == 1.5

    def test_feishu_timeout_profile_uses_fast_limits(self):
        agent = _make_agent()
        agent.platform = "feishu"

        assert agent._request_api_timeout_seconds() == 45.0
        assert agent._stream_read_timeout_seconds() == 20.0
        assert agent._stream_stale_timeout_base_seconds() == 25.0
        assert agent._should_fail_fast_provider_errors_for_request() is True

    def test_cloudflare_gateway_disables_local_fallback_by_default(self):
        fbs = [{"provider": "custom", "model": "moonshotai/kimi-k2.5"}]
        agent = _make_agent(fallback_model=fbs)
        agent.base_url = "https://gateway.ai.cloudflare.com/v1/account/gateway/compat"
        agent._base_url_lower = agent.base_url.lower()

        assert agent._cloudflare_gateway_delegates_failover() is True
        assert agent._try_activate_fallback() is False
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False

    def test_cloudflare_gateway_can_opt_back_into_local_fallback(self, monkeypatch):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        agent.base_url = "https://gateway.ai.cloudflare.com/v1/account/gateway/compat"
        agent._base_url_lower = agent.base_url.lower()
        monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_DELEGATE_FAILOVER", "false")

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(), "gpt-4o")):
            assert agent._cloudflare_gateway_delegates_failover() is False
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_cloudflare_gateway_allows_terminal_local_fallback_after_retries(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        agent.base_url = "https://gateway.ai.cloudflare.com/v1/account/gateway/compat"
        agent._base_url_lower = agent.base_url.lower()

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(), "gpt-4o")):
            assert agent._attempt_local_fallback(fallback_status="Max retries exhausted") is True
            assert agent.model == "gpt-4o"
