from __future__ import annotations

from typing import Any


def build_modal_image(*, modal_module: Any, include_local_plugins: bool) -> Any:
    image = (
        modal_module.Image.debian_slim(python_version="3.11")
        .apt_install(
            "curl",
            "ca-certificates",
            "gnupg",
            "libsecret-1-0",
            "libnspr4",
            "libnss3",
            "libatk1.0-0",
            "libatk-bridge2.0-0",
            "libcups2",
            "libdrm2",
            "libdbus-1-3",
            "libxcb1",
            "libxkbcommon0",
            "libx11-6",
            "libx11-xcb1",
            "libxcomposite1",
            "libxdamage1",
            "libxext6",
            "libxfixes3",
            "libxrandr2",
            "libgbm1",
            "libasound2",
            "libatspi2.0-0",
            "libgtk-3-0",
            "fonts-liberation",
        )
        .run_commands(
            "install -d /etc/apt/keyrings",
            "curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg",
            'echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list',
            "apt-get update",
            "apt-get install -y nodejs",
            "npm install -g agent-browser @askjo/camofox-browser",
            "agent-browser install --with-deps || agent-browser install || true",
            "python -c \"from pathlib import Path; import uuid; mid = uuid.uuid4().hex; Path('/etc/machine-id').write_text(mid + '\\n', encoding='utf-8'); Path('/var/lib/dbus').mkdir(parents=True, exist_ok=True); Path('/var/lib/dbus/machine-id').write_text(mid + '\\n', encoding='utf-8')\"",
        )
        .pip_install_from_pyproject(
            "pyproject.toml",
            optional_dependencies=[
                "modal",
                "messaging",
                "qq",
                "cron",
                "mcp",
                "pty",
                "feishu",
                "honcho",
                "homeassistant",
            ],
        )
        .pip_install(["fastapi[standard]", "supermemory>=3.33.0,<4", "uv>=0.7.0,<1"])
        .env(
            {
                "HERMES_HOME": "/data/hermes-home",
                "HERMES_BUNDLED_SKILLS": "/root/skills",
            }
        )
        .add_local_python_source(
            "acp_adapter",
            "agent",
            "cron",
            "environments",
            "gateway",
            "hermes_cli",
            "plugins",
            "tools",
            copy=True,
        )
        .add_local_dir("acp_registry", remote_path="/root/acp_registry", copy=True)
        .add_local_dir("skills", remote_path="/root/skills", copy=True)
        .add_local_dir("optional-skills", remote_path="/root/optional-skills", copy=True)
        .add_local_file("run_agent.py", remote_path="/root/run_agent.py", copy=True)
        .add_local_file("batch_runner.py", remote_path="/root/batch_runner.py", copy=True)
        .add_local_file("model_tools.py", remote_path="/root/model_tools.py", copy=True)
        .add_local_file("toolsets.py", remote_path="/root/toolsets.py", copy=True)
        .add_local_file(
            "toolset_distributions.py",
            remote_path="/root/toolset_distributions.py",
            copy=True,
        )
        .add_local_file(
            "trajectory_compressor.py",
            remote_path="/root/trajectory_compressor.py",
            copy=True,
        )
        .add_local_file("cli.py", remote_path="/root/cli.py", copy=True)
        .add_local_file("rl_cli.py", remote_path="/root/rl_cli.py", copy=True)
        .add_local_file("hermes_constants.py", remote_path="/root/hermes_constants.py", copy=True)
        .add_local_file("hermes_logging.py", remote_path="/root/hermes_logging.py", copy=True)
        .add_local_file("hermes_state.py", remote_path="/root/hermes_state.py", copy=True)
        .add_local_file("hermes_time.py", remote_path="/root/hermes_time.py", copy=True)
        .add_local_file("utils.py", remote_path="/root/utils.py", copy=True)
        .add_local_file("README.md", remote_path="/root/README.md", copy=True)
        .add_local_file("MANIFEST.in", remote_path="/root/MANIFEST.in", copy=True)
        .add_local_file("config.modal.yaml", remote_path="/root/config.modal.yaml", copy=True)
        .add_local_file(".env.modal.example", remote_path="/root/.env.modal.example", copy=True)
        .add_local_file(
            "supermemory.modal.json",
            remote_path="/root/supermemory.modal.json",
            copy=True,
        )
    )
    if include_local_plugins:
        image = image.add_local_dir(
            ".hermes/plugins",
            remote_path="/root/.hermes/plugins",
            copy=True,
        )
    return image


def build_modal_volume(*, modal_module: Any, default_volume_name: str) -> Any:
    return modal_module.Volume.from_name(default_volume_name, create_if_missing=True)


def build_modal_secrets(*, modal_module: Any, default_secret_name: str) -> list[Any]:
    return [modal_module.Secret.from_name(default_secret_name)]


def build_maintenance_schedule(
    *,
    modal_module: Any,
    enabled: bool,
    default_maintenance_heartbeat_minutes: int,
) -> Any:
    if not enabled:
        return None
    return modal_module.Period(minutes=default_maintenance_heartbeat_minutes)
