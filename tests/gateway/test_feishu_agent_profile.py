from gateway.run import GatewayRunner


def test_resolve_agent_execution_profile_for_feishu(monkeypatch):
    monkeypatch.setenv("HERMES_FEISHU_AGENT_MAX_ITERATIONS", "10")
    monkeypatch.setenv("HERMES_FEISHU_AGENT_MAX_TOKENS", "700")
    monkeypatch.setenv("HERMES_FEISHU_REASONING_EFFORT", "none")

    profile = GatewayRunner._resolve_agent_execution_profile(
        "feishu",
        max_iterations=90,
        reasoning_config={"enabled": True, "effort": "high"},
        max_tokens=None,
        ephemeral_prompt="base prompt",
    )

    assert profile["max_iterations"] == 10
    assert profile["max_tokens"] == 700
    assert profile["reasoning_config"] == {"enabled": False}
    assert "base prompt" in profile["ephemeral_prompt"]
    assert "Feishu chat profile" in profile["ephemeral_prompt"]


def test_resolve_agent_execution_profile_non_feishu_unchanged():
    profile = GatewayRunner._resolve_agent_execution_profile(
        "telegram",
        max_iterations=90,
        reasoning_config={"enabled": True, "effort": "medium"},
        max_tokens=1200,
        ephemeral_prompt="base prompt",
    )

    assert profile == {
        "max_iterations": 90,
        "reasoning_config": {"enabled": True, "effort": "medium"},
        "max_tokens": 1200,
        "ephemeral_prompt": "base prompt",
    }
