from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

DIRECT_RUNTIME_PROVIDER_BYPASS = frozenset({"custom", "local", "copilot-acp"})


def sync_runtime_config(
    *,
    default_config_source: Path,
    hermes_home_dir: Path,
    is_truthy: Callable[..., bool],
    materialize_dynamic_free_model_config: Callable[[dict[str, Any]], None],
    logger: Any,
) -> str | None:
    source = Path(os.getenv("HERMES_MODAL_CONFIG_SOURCE", str(default_config_source)))
    if not source.exists():
        return None

    target = hermes_home_dir / "config.yaml"
    force_sync = is_truthy(os.getenv("HERMES_MODAL_SYNC_CONFIG"), default=True)
    source_text = source.read_text(encoding="utf-8")

    resolved_text = source_text
    try:
        import yaml
        from hermes_cli.config import _expand_env_vars

        config_payload = yaml.safe_load(source_text) or {}
        raw_default_model = ""
        if isinstance(config_payload, dict):
            raw_model_config = config_payload.get("model")
            if isinstance(raw_model_config, dict):
                raw_default_model = str(raw_model_config.get("default") or "").strip()
        if isinstance(config_payload, dict):
            config_payload = _expand_env_vars(config_payload)
            model_config = config_payload.get("model")
            if isinstance(model_config, dict):
                default_model = str(model_config.get("default") or "").strip()
                if not default_model:
                    model_config["default"] = os.getenv("DEFAULT_MODEL", "openrouter/free")
                    default_model = str(model_config.get("default") or "").strip()
                should_materialize_dynamic_free_route = raw_default_model in {"openrouter/free", "free"}
            else:
                should_materialize_dynamic_free_route = False
            if should_materialize_dynamic_free_route:
                restore_dynamic_alias = str(model_config.get("default") or "").strip() if isinstance(model_config, dict) else ""
                if raw_default_model == "openrouter/free" and isinstance(model_config, dict):
                    model_config["default"] = "free"
                materialize_dynamic_free_model_config(config_payload)
                if (
                    raw_default_model == "openrouter/free"
                    and isinstance(model_config, dict)
                    and str(model_config.get("default") or "").strip() == "free"
                ):
                    model_config["default"] = restore_dynamic_alias or "openrouter/free"
            resolved_text = yaml.safe_dump(
                config_payload,
                allow_unicode=True,
                sort_keys=False,
            )
    except Exception as exc:
        logger.warning("Falling back to raw Modal config sync without env expansion: %s", exc)

    if target.exists() and not force_sync:
        return str(target)

    if not target.exists() or target.read_text(encoding="utf-8") != resolved_text:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(resolved_text, encoding="utf-8")
    return str(target)


def sync_runtime_support_file(
    *,
    env_var_name: str,
    default_source: Path,
    target_name: str,
    hermes_home_dir: Path,
    is_truthy: Callable[..., bool],
) -> str | None:
    source = Path(os.getenv(env_var_name, str(default_source)))
    if not source.exists():
        return None

    target = hermes_home_dir / target_name
    force_sync = is_truthy(os.getenv("HERMES_MODAL_SYNC_CONFIG"), default=True)
    source_text = source.read_text(encoding="utf-8")

    if target.exists() and not force_sync:
        return str(target)

    if not target.exists() or target.read_text(encoding="utf-8") != source_text:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source_text, encoding="utf-8")
    return str(target)


def sync_supermemory_config(
    *,
    default_supermemory_config_source: Path,
    hermes_home_dir: Path,
    is_truthy: Callable[..., bool],
) -> str | None:
    return sync_runtime_support_file(
        env_var_name="HERMES_MODAL_SUPERMEMORY_CONFIG_SOURCE",
        default_source=default_supermemory_config_source,
        target_name="supermemory.json",
        hermes_home_dir=hermes_home_dir,
        is_truthy=is_truthy,
    )


def pick_runtime_api_config(
    *,
    env_flag: Callable[..., bool],
    normalize_cloudflare_ai_gateway_runtime_base_url: Callable[[str], str],
    default_cloudflare_ai_gateway_base_url: str,
) -> tuple[str | None, str | None, str | None]:
    provider = os.getenv("HERMES_PROVIDER") or None
    provider_name = str(provider or "").strip().lower()
    cloudflare_base_url = normalize_cloudflare_ai_gateway_runtime_base_url(
        os.getenv("CLOUDFLARE_AI_GATEWAY_BASE_URL", default_cloudflare_ai_gateway_base_url)
    )
    use_cloudflare_gateway = (
        env_flag("HERMES_INFERENCE_USE_CLOUDFLARE_AI_GATEWAY", default=True)
        and bool(cloudflare_base_url)
        and provider_name not in DIRECT_RUNTIME_PROVIDER_BYPASS
        and not provider_name.startswith("custom:")
    )
    explicit_base_url = os.getenv("HERMES_BASE_URL")
    using_cloudflare_gateway = use_cloudflare_gateway
    base_url = (
        (cloudflare_base_url if use_cloudflare_gateway else None)
        or explicit_base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or None
    )
    if using_cloudflare_gateway:
        api_key_candidates = [
            os.getenv("CLOUDFLARE_API_TOKEN"),
            os.getenv("CLOUDFLARE_AI_GATEWAY_API_KEY"),
            os.getenv("AI_GATEWAY_API_KEY"),
            os.getenv("HERMES_API_KEY"),
        ]
    else:
        api_key_candidates = [os.getenv("HERMES_API_KEY")]
    api_key_candidates.extend(
        [
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
        ]
    )
    api_key = next((candidate for candidate in api_key_candidates if candidate), None)
    if not provider and not base_url and os.getenv("OPENROUTER_API_KEY"):
        provider = "openrouter"
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    elif not provider and using_cloudflare_gateway:
        provider = "openrouter"
    return provider, base_url, api_key


def get_memory_provider_status(*, hermes_home_dir: Path) -> dict[str, Any]:
    provider_name = ""
    config_error = None
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        provider_name = str(((config.get("memory") or {}).get("provider")) or "").strip()
    except Exception as exc:
        config_error = str(exc)

    status = {
        "provider": provider_name,
        "configured": bool(provider_name),
    }
    if config_error:
        status["config_error"] = config_error

    if provider_name == "supermemory":
        config_path = Path(os.getenv("HERMES_HOME", str(hermes_home_dir))) / "supermemory.json"
        status["api_key_configured"] = bool(os.getenv("SUPERMEMORY_API_KEY", "").strip())
        status["sdk_available"] = importlib.util.find_spec("supermemory") is not None
        status["container_tag_override"] = os.getenv("SUPERMEMORY_CONTAINER_TAG", "").strip()
        status["config_path"] = str(config_path)
        status["config_present"] = config_path.exists()

    return status


def get_camofox_url() -> str:
    return str(os.getenv("CAMOFOX_URL") or "").strip().rstrip("/")


def is_local_camofox_url(url: str) -> bool:
    normalized = str(url or "").strip()
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def is_camofox_healthcheck_ready(url: str) -> bool:
    normalized = str(url or "").strip().rstrip("/")
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    request_path = (parsed.path.rstrip("/") or "") + "/health"
    try:
        with socket.create_connection((host, port), timeout=2.0) as sock:
            request = (
                f"GET {request_path} HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Connection: close\r\n\r\n"
            )
            sock.sendall(request.encode("ascii", "ignore"))
            response = sock.recv(256).decode("latin-1", "ignore")
        return (
            " 200 " in response
            or response.startswith("HTTP/1.1 200")
            or response.startswith("HTTP/1.0 200")
        )
    except OSError:
        return False


def resolve_camofox_launch_command(*, which: Callable[[str], str | None]) -> list[str]:
    for candidate in ("camofox-browser", "camoufox-browser"):
        binary = which(candidate)
        if binary:
            return [binary]
    return ["npx", "--yes", "@askjo/camofox-browser"]


def ensure_camofox_server(
    *,
    get_process: Callable[[], Any],
    set_process: Callable[[Any], None],
    camofox_server_lock: Any,
    get_camofox_url: Callable[[], str],
    is_local_camofox_url: Callable[[str], bool],
    is_camofox_healthcheck_ready: Callable[[str], bool],
    resolve_camofox_launch_command: Callable[[], list[str]],
    data_root: Path,
    default_camofox_boot_timeout_seconds: float,
    default_camofox_boot_poll_interval_seconds: float,
    logger: Any,
) -> None:
    camofox_url = get_camofox_url()
    if not camofox_url or not is_local_camofox_url(camofox_url):
        return
    if is_camofox_healthcheck_ready(camofox_url):
        return

    with camofox_server_lock:
        if is_camofox_healthcheck_ready(camofox_url):
            return

        process = get_process()
        if process is None or process.poll() is not None:
            parsed = urlparse(camofox_url)
            port = parsed.port or 9377
            log_dir = data_root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "camofox.log"
            launch_env = dict(os.environ)
            launch_env.setdefault("CAMOFOX_PORT", str(port))
            launch_env.setdefault("PORT", str(port))
            launch_command = resolve_camofox_launch_command()
            with log_path.open("ab") as log_file:
                process = subprocess.Popen(
                    launch_command,
                    cwd="/root",
                    env=launch_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            set_process(process)
            logger.info(
                "Started local Camofox server for Modal runtime: url=%s pid=%s cmd=%s",
                camofox_url,
                process.pid,
                " ".join(launch_command),
            )

        deadline = time.time() + default_camofox_boot_timeout_seconds
        while time.time() < deadline:
            if is_camofox_healthcheck_ready(camofox_url):
                return
            process = get_process()
            if process is not None and process.poll() is not None:
                break
            time.sleep(default_camofox_boot_poll_interval_seconds)

        process = get_process()
        process_status = process.poll() if process is not None else None
        raise RuntimeError(
            "Timed out waiting for local Camofox server to become ready "
            f"at {camofox_url} (process_status={process_status}, log={data_root / 'logs' / 'camofox.log'})"
        )


def prepare_runtime_environment(
    *,
    hermes_home_dir: Path,
    ensure_runtime_dirs: Callable[[], None],
    ensure_camofox_server: Callable[[], None],
    sync_runtime_config: Callable[[], str | None],
    sync_supermemory_config: Callable[[], str | None],
    get_runtime_skills_synced: Callable[[], bool],
    set_runtime_skills_synced: Callable[[bool], None],
    runtime_skills_sync_lock: Any,
    logger: Any,
) -> None:
    os.environ.setdefault("HERMES_HOME", str(hermes_home_dir))
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ["TERMINAL_ENV"] = os.getenv("HERMES_MODAL_TERMINAL_ENV", "local")
    os.environ.setdefault("HERMES_ENABLE_PROJECT_PLUGINS", "true")
    os.environ.setdefault("HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS", "0")
    os.environ.setdefault("HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES", "1")
    os.environ.setdefault("HERMES_FEISHU_RESOLVE_SENDER_NAMES", "false")
    os.environ.setdefault("HERMES_FEISHU_MENU_OPEN_BY_OPEN_ID", "true")
    os.environ.setdefault(
        "HERMES_FEISHU_DISABLED_TOOLSETS",
        "browser,terminal,code_execution,delegation,tts,messaging,rl",
    )
    if "NGC_API_KEY" in os.environ and "NVIDIA_API_KEY" not in os.environ:
        os.environ["NVIDIA_API_KEY"] = os.environ["NGC_API_KEY"]
    os.environ.setdefault("HERMES_STREAM_READ_TIMEOUT", "25")
    if "HERMES_MAX_TURNS" in os.environ and "HERMES_MAX_ITERATIONS" not in os.environ:
        os.environ["HERMES_MAX_ITERATIONS"] = os.environ["HERMES_MAX_TURNS"]
    ensure_runtime_dirs()
    ensure_camofox_server()
    sync_runtime_config()
    sync_supermemory_config()
    if not get_runtime_skills_synced():
        with runtime_skills_sync_lock:
            if not get_runtime_skills_synced():
                try:
                    from tools.skills_sync import sync_skills

                    sync_skills(quiet=True)
                    set_runtime_skills_synced(True)
                except Exception:
                    logger.warning("Failed to sync bundled skills into HERMES_HOME", exc_info=True)


def maintenance_heartbeat_is_enabled(*, is_truthy: Callable[..., bool]) -> bool:
    return is_truthy(
        os.getenv("HERMES_MODAL_MAINTENANCE_HEARTBEAT_ENABLED"),
        default=False,
    )
