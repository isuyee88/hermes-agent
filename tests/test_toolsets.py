"""Tests for toolsets.py."""

from tools.registry import ToolRegistry
from toolsets import (
    TOOLSETS,
    create_custom_toolset,
    get_all_toolsets,
    get_toolset,
    get_toolset_info,
    resolve_multiple_toolsets,
    resolve_toolset,
    validate_toolset,
)


def _dummy_handler(args, **kwargs):
    return "{}"


def _make_schema(name: str, description: str = "test tool"):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


class TestGetToolset:
    def test_known_toolset(self):
        ts = get_toolset("web")
        assert ts is not None
        assert "web_search" in ts["tools"]

    def test_unknown_returns_none(self):
        assert get_toolset("nonexistent") is None


class TestResolveToolset:
    def test_leaf_toolset(self):
        tools = resolve_toolset("web")
        assert set(tools) == {"web_search", "web_extract"}

    def test_composite_toolset(self):
        tools = resolve_toolset("debugging")
        assert "terminal" in tools
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_cycle_detection(self):
        TOOLSETS["_cycle_a"] = {"description": "test", "tools": ["t1"], "includes": ["_cycle_b"]}
        TOOLSETS["_cycle_b"] = {"description": "test", "tools": ["t2"], "includes": ["_cycle_a"]}
        try:
            tools = resolve_toolset("_cycle_a")
            assert "t1" in tools
            assert "t2" in tools
        finally:
            del TOOLSETS["_cycle_a"]
            del TOOLSETS["_cycle_b"]

    def test_unknown_toolset_returns_empty(self):
        assert resolve_toolset("nonexistent") == []

    def test_all_alias(self):
        tools = resolve_toolset("all")
        assert len(tools) > 10

    def test_star_alias(self):
        tools = resolve_toolset("*")
        assert len(tools) > 10

    def test_founder_max_includes_browser_and_execution_tools(self):
        tools = resolve_toolset("founder-max")
        for tool in ("browser_navigate", "terminal", "execute_code", "skills_list", "send_message"):
            assert tool in tools

    def test_collab_safe_keeps_browser_but_blocks_mutation_surfaces(self):
        tools = resolve_toolset("collab-safe")
        assert "browser_navigate" in tools
        assert "send_message" in tools
        assert "terminal" not in tools
        assert "write_file" not in tools
        assert "patch" not in tools

    def test_finance_max_includes_analysis_tools_without_browser_requirement(self):
        tools = resolve_toolset("finance-max")
        for tool in ("read_file", "execute_code", "skills_list", "session_search"):
            assert tool in tools

    def test_plugin_toolset_uses_registry_snapshot(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="plugin_b",
            toolset="plugin_example",
            schema=_make_schema("plugin_b", "B"),
            handler=_dummy_handler,
        )
        reg.register(
            name="plugin_a",
            toolset="plugin_example",
            schema=_make_schema("plugin_a", "A"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        assert resolve_toolset("plugin_example") == ["plugin_a", "plugin_b"]


class TestResolveMultipleToolsets:
    def test_combines_and_deduplicates(self):
        tools = resolve_multiple_toolsets(["web", "terminal"])
        assert "web_search" in tools
        assert "web_extract" in tools
        assert "terminal" in tools
        assert len(tools) == len(set(tools))

    def test_empty_list(self):
        assert resolve_multiple_toolsets([]) == []


class TestValidateToolset:
    def test_valid(self):
        assert validate_toolset("web") is True
        assert validate_toolset("terminal") is True

    def test_all_alias_valid(self):
        assert validate_toolset("all") is True
        assert validate_toolset("*") is True

    def test_invalid(self):
        assert validate_toolset("nonexistent") is False


class TestGetToolsetInfo:
    def test_leaf(self):
        info = get_toolset_info("web")
        assert info["name"] == "web"
        assert info["is_composite"] is False
        assert info["tool_count"] == 2

    def test_composite(self):
        info = get_toolset_info("debugging")
        assert info["is_composite"] is True
        assert info["tool_count"] > len(info["direct_tools"])

    def test_unknown_returns_none(self):
        assert get_toolset_info("nonexistent") is None


class TestCreateCustomToolset:
    def test_runtime_creation(self):
        create_custom_toolset(
            name="_test_custom",
            description="Test toolset",
            tools=["web_search"],
            includes=["terminal"],
        )
        try:
            tools = resolve_toolset("_test_custom")
            assert "web_search" in tools
            assert "terminal" in tools
            assert validate_toolset("_test_custom") is True
        finally:
            del TOOLSETS["_test_custom"]


class TestToolsetConsistency:
    def test_all_toolsets_have_required_keys(self):
        for name, ts in TOOLSETS.items():
            assert "description" in ts, f"{name} missing description"
            assert "tools" in ts, f"{name} missing tools"
            assert "includes" in ts, f"{name} missing includes"

    def test_all_includes_reference_existing_toolsets(self):
        for name, ts in TOOLSETS.items():
            for inc in ts["includes"]:
                assert inc in TOOLSETS, f"{name} includes unknown toolset '{inc}'"

    def test_hermes_platforms_share_core_tools(self):
        platforms = [
            "hermes-cli",
            "hermes-telegram",
            "hermes-discord",
            "hermes-whatsapp",
            "hermes-slack",
            "hermes-signal",
            "hermes-homeassistant",
        ]
        tool_sets = [set(TOOLSETS[p]["tools"]) for p in platforms]
        for ts in tool_sets[1:]:
            assert ts == tool_sets[0]


class TestPluginToolsets:
    def test_get_all_toolsets_includes_plugin_toolset(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="plugin_tool",
            toolset="plugin_bundle",
            schema=_make_schema("plugin_tool", "Plugin tool"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        all_toolsets = get_all_toolsets()
        assert "plugin_bundle" in all_toolsets
        assert all_toolsets["plugin_bundle"]["tools"] == ["plugin_tool"]
        assert all_toolsets["plugin_bundle"]["includes"] == []
