import importlib
import sys


def test_load_gateway_config_expands_env_placeholders(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: ${DEFAULT_MODEL}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("DEFAULT_MODEL", "openrouter/free")

    sys.modules.pop("gateway.run", None)
    module = importlib.import_module("gateway.run")

    cfg = module._load_gateway_config()

    assert cfg["model"]["default"] == "openrouter/free"
