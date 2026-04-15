"""Tests for /personality none — clearing personality overlay."""
import pytest
from unittest.mock import MagicMock, patch, mock_open
import yaml


# ── CLI tests ──────────────────────────────────────────────────────────────

class TestCLIPersonalityNone:

    def _make_cli(self, personalities=None):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.personalities = personalities or {
            "helpful": "You are helpful.",
            "concise": "You are concise.",
        }
        cli.system_prompt = "You are kawaii~"
        cli.agent = MagicMock()
        cli.console = MagicMock()
        return cli

    def test_none_clears_system_prompt(self):
        cli = self._make_cli()
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality none")
        assert cli.system_prompt == ""

    def test_default_clears_system_prompt(self):
        cli = self._make_cli()
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality default")
        assert cli.system_prompt == ""

    def test_neutral_clears_system_prompt(self):
        cli = self._make_cli()
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality neutral")
        assert cli.system_prompt == ""

    def test_none_forces_agent_reinit(self):
        cli = self._make_cli()
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality none")
        assert cli.agent is None

    def test_none_saves_to_config(self):
        cli = self._make_cli()
        with patch("cli.save_config_value", return_value=True) as mock_save:
            cli._handle_personality_command("/personality none")
        mock_save.assert_called_once_with("agent.system_prompt", "")

    def test_known_personality_still_works(self):
        cli = self._make_cli()
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality helpful")
        assert cli.system_prompt == "You are helpful."

    def test_unknown_personality_shows_none_in_available(self, capsys):
        cli = self._make_cli()
        cli._handle_personality_command("/personality nonexistent")
        output = capsys.readouterr().out
        assert "none" in output.lower()

    def test_list_shows_none_option(self):
        cli = self._make_cli()
        with patch("builtins.print") as mock_print:
            cli._handle_personality_command("/personality")
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "none" in output.lower()


# ── Gateway tests ──────────────────────────────────────────────────────────

class TestGatewayPersonalityNone:

    def _make_event(self, args=""):
        event = MagicMock()
        event.get_command.return_value = "personality"
        event.get_command_args.return_value = args
        return event

    def _make_runner(self, personalities=None):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._ephemeral_system_prompt = "You are kawaii~"
        runner.config = {
            "agent": {
                "personalities": personalities or {"helpful": "You are helpful."}
            }
        }
        return runner

    @pytest.mark.asyncio
    async def test_none_clears_ephemeral_prompt(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}, "system_prompt": "kawaii"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("none")
            result = await runner._handle_personality_command(event)

        assert runner._ephemeral_system_prompt == ""
        assert "cleared" in result.lower()

    @pytest.mark.asyncio
    async def test_default_clears_ephemeral_prompt(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("default")
            result = await runner._handle_personality_command(event)

        assert runner._ephemeral_system_prompt == ""

    @pytest.mark.asyncio
    async def test_list_includes_none(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("")
            result = await runner._handle_personality_command(event)

        assert "none" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_shows_none_in_available(self, tmp_path):
        runner = self._make_runner()
        config_data = {"agent": {"personalities": {"helpful": "You are helpful."}}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("nonexistent")
            result = await runner._handle_personality_command(event)

        assert "none" in result.lower()

    @pytest.mark.asyncio
    async def test_list_uses_builtin_personalities_without_config_file(self, tmp_path):
        runner = self._make_runner()

        with patch("gateway.run._hermes_home", tmp_path):
            event = self._make_event("")
            result = await runner._handle_personality_command(event)

        assert "ceo" in result
        assert "cto" in result


class TestPersonalityDictFormat:
    """Test dict-format custom personalities with description, tone, style."""

    def _make_cli(self, personalities):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.personalities = personalities
        cli.system_prompt = ""
        cli.agent = None
        cli.console = MagicMock()
        return cli

    def test_dict_personality_uses_system_prompt(self):
        cli = self._make_cli({
            "coder": {
                "description": "Expert programmer",
                "system_prompt": "You are an expert programmer.",
                "tone": "technical",
                "style": "concise",
            }
        })
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality coder")
        assert "You are an expert programmer." in cli.system_prompt

    def test_dict_personality_includes_tone(self):
        cli = self._make_cli({
            "coder": {
                "system_prompt": "You are an expert programmer.",
                "tone": "technical and precise",
            }
        })
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality coder")
        assert "Tone: technical and precise" in cli.system_prompt

    def test_dict_personality_includes_style(self):
        cli = self._make_cli({
            "coder": {
                "system_prompt": "You are an expert programmer.",
                "style": "use code examples",
            }
        })
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality coder")
        assert "Style: use code examples" in cli.system_prompt

    def test_string_personality_still_works(self):
        cli = self._make_cli({"helper": "You are helpful."})
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality helper")
        assert cli.system_prompt == "You are helpful."

    def test_resolve_prompt_dict_no_tone_no_style(self):
        from cli import HermesCLI
        result = HermesCLI._resolve_personality_prompt({
            "description": "A helper",
            "system_prompt": "You are helpful.",
        })
        assert result == "You are helpful."

    def test_resolve_prompt_string(self):
        from cli import HermesCLI
        result = HermesCLI._resolve_personality_prompt("You are helpful.")
        assert result == "You are helpful."


class TestOrganizationPersonalities:

    def test_cli_config_includes_org_personalities(self):
        from cli import CLI_CONFIG

        personalities = CLI_CONFIG["agent"]["personalities"]

        for name in ("board", "ceo", "grow", "content", "seo", "ads", "bd", "ops", "finance", "cto", "staff", "sev"):
            assert name in personalities

    def test_org_personality_uses_dict_prompt_parts(self):
        from cli import HermesCLI, CLI_CONFIG

        ceo = CLI_CONFIG["agent"]["personalities"]["ceo"]
        result = HermesCLI._resolve_personality_prompt(ceo)

        assert "互联网初创公司 CEO" in result
        assert "Tone:" in result
        assert "Style:" in result

    @pytest.mark.asyncio
    async def test_gateway_lists_org_personalities(self, tmp_path):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._ephemeral_system_prompt = ""
        config_data = {
            "agent": {
                "personalities": {
                    "ceo": {
                        "description": "CEO 主操盘人格",
                        "system_prompt": "你以互联网初创公司 CEO 视角工作。",
                    },
                    "sev": {
                        "description": "事故指挥人格",
                        "system_prompt": "你以事故指挥官视角工作。",
                    },
                }
            }
        }
        (tmp_path / "config.yaml").write_text(yaml.dump(config_data, allow_unicode=True), encoding="utf-8")

        event = MagicMock()
        event.get_command.return_value = "personality"
        event.get_command_args.return_value = ""

        with patch("gateway.run._hermes_home", tmp_path):
            result = await runner._handle_personality_command(event)

        assert "ceo" in result
        assert "sev" in result
