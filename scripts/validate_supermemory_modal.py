#!/usr/bin/env python3
"""Live Supermemory regression validation for the Modal deployment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid


def _maybe_reexec_with_supported_python() -> None:
    if sys.version_info < (3, 13):
        return

    override = os.getenv("HERMES_MODAL_VALIDATOR_PYTHON", "").strip()
    candidates: list[list[str]] = []
    if override:
        candidates.append([override])
    py_launcher = shutil.which("py")
    if py_launcher:
        candidates.append([py_launcher, "-3.11"])
    explicit_py311 = shutil.which("python3.11")
    if explicit_py311:
        candidates.append([explicit_py311])

    for command in candidates:
        try:
            completed = subprocess.run(command + [__file__, *sys.argv[1:]], check=False)
        except OSError:
            continue
        raise SystemExit(completed.returncode)

    raise SystemExit(
        "Modal validation requires Python 3.11/3.12 locally. "
        "Set HERMES_MODAL_VALIDATOR_PYTHON to a supported interpreter."
    )


_maybe_reexec_with_supported_python()

import modal


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _run_agent(function, prompt: str, session_key: str) -> dict:
    result = function.remote(prompt, session_key=session_key)
    if not isinstance(result, dict):
        raise RuntimeError(f"run_agent_task returned unexpected payload: {result!r}")
    return result


def _poll_agent_output(
    function,
    *,
    prompt: str,
    session_key: str,
    expected_output: str,
    required_tool: str,
    attempts: int = 6,
    delay_seconds: float = 5.0,
) -> tuple[dict, list[dict]]:
    history: list[dict] = []
    for index in range(attempts):
        result = _run_agent(function, prompt, session_key)
        history.append(result)
        _assert(required_tool in (result.get("tool_summary") or []), f"{required_tool} was not called")
        if str(result.get("output") or "").strip() == expected_output:
            return result, history
        if index < attempts - 1:
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"Expected output {expected_output!r} after {attempts} attempts, got "
        f"{str(history[-1].get('output') or '').strip()!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Supermemory in the deployed Modal Hermes app.")
    parser.add_argument("--app-name", default="hermes-agent", help="Modal app name")
    parser.add_argument("--health-function", default="health_check", help="Health function name")
    parser.add_argument("--task-function", default="run_agent_task", help="Agent task function name")
    args = parser.parse_args()

    health_function = modal.Function.from_name(args.app_name, args.health_function)
    task_function = modal.Function.from_name(args.app_name, args.task_function)

    summary: dict[str, object] = {
        "app_name": args.app_name,
        "health_function": args.health_function,
        "task_function": args.task_function,
    }

    health = health_function.remote()
    summary["health"] = health

    memory_provider = (health or {}).get("memory_provider") or {}
    _assert(memory_provider.get("provider") == "supermemory", "memory.provider is not supermemory")
    _assert(memory_provider.get("configured") is True, "supermemory is not marked configured")
    _assert(memory_provider.get("api_key_configured") is True, "SUPERMEMORY_API_KEY is not available at runtime")
    _assert(memory_provider.get("sdk_available") is True, "supermemory SDK is missing in the image")

    probe_id = f"modal-supermemory-probe-{uuid.uuid4().hex[:12]}"
    session_key = f"validation:{probe_id}"
    summary["probe_id"] = probe_id

    store_result = _run_agent(
        task_function,
        (
            "Use the supermemory_store tool to save this exact memory: "
            f"{probe_id}. Then reply with STORE_OK only."
        ),
        session_key,
    )
    summary["store_result"] = store_result
    _assert("supermemory_store" in (store_result.get("tool_summary") or []), "supermemory_store was not called")
    _assert(str(store_result.get("output") or "").strip() == "STORE_OK", "store probe did not acknowledge correctly")

    search_result, search_attempts = _poll_agent_output(
        task_function,
        prompt=(
            "Use the supermemory_search tool to look up this exact text: "
            f"{probe_id}. Reply with FOUND only if it exists, otherwise reply with MISSING only."
        ),
        session_key=session_key,
        expected_output="FOUND",
        required_tool="supermemory_search",
    )
    summary["search_result"] = search_result
    summary["search_attempts"] = search_attempts

    cleanup_result = _run_agent(
        task_function,
        (
            "Use the supermemory_forget tool to remove the memory containing this exact text: "
            f"{probe_id}. Then reply with CLEANED only."
        ),
        session_key,
    )
    summary["cleanup_result"] = cleanup_result
    _assert("supermemory_forget" in (cleanup_result.get("tool_summary") or []), "supermemory_forget was not called")
    _assert(str(cleanup_result.get("output") or "").strip() == "CLEANED", "cleanup probe did not confirm removal")

    verify_cleanup, cleanup_attempts = _poll_agent_output(
        task_function,
        prompt=(
            "Use the supermemory_search tool to look up this exact text: "
            f"{probe_id}. Reply with MISSING only if it no longer exists, otherwise reply with FOUND only."
        ),
        session_key=session_key,
        expected_output="MISSING",
        required_tool="supermemory_search",
    )
    summary["verify_cleanup"] = verify_cleanup
    summary["cleanup_attempts"] = cleanup_attempts

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)
