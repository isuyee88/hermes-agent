#!/usr/bin/env python
"""Send a Feishu probe message and build a recent triparty report snapshot."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.config import PlatformConfig
from gateway.platforms.feishu import FEISHU_AVAILABLE, FEISHU_DOMAIN, LARK_DOMAIN, FeishuAdapter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = REPO_ROOT / ".tmp-feishu-triparty-probe-report.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / ".tmp-feishu-triparty-probe-report.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a Feishu probe message and generate a recent triparty report.")
    parser.add_argument("--chat-id", default=os.getenv("FEISHU_HOME_CHANNEL", "").strip())
    parser.add_argument("--message", default="Hermes triparty probe")
    parser.add_argument("--wait-seconds", type=float, default=45.0)
    parser.add_argument("--recent-hours", type=int, default=3)
    parser.add_argument("--recent-min-sessions", type=int, default=20)
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT))
    parser.add_argument("--strict-goal-metric", default="p90")
    return parser


def _ensure_env(chat_id: str) -> None:
    if not chat_id:
        raise RuntimeError("No target chat_id provided. Pass --chat-id or set FEISHU_HOME_CHANNEL.")
    missing = [name for name in ("FEISHU_APP_ID", "FEISHU_APP_SECRET") if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if not FEISHU_AVAILABLE:
        raise RuntimeError("Feishu dependencies are not installed.")


def _build_adapter() -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    domain = FEISHU_DOMAIN if adapter._domain_name != "lark" else LARK_DOMAIN
    adapter._client = adapter._build_lark_client(domain)
    adapter._running = True
    return adapter


async def _send_probe(chat_id: str, probe_text: str) -> dict[str, object]:
    adapter = _build_adapter()
    result = await adapter.send(chat_id=chat_id, content=probe_text)
    return {
        "success": bool(getattr(result, "success", False)),
        "message_id": str(getattr(result, "message_id", "") or ""),
        "error": str(getattr(result, "error", "") or ""),
    }


def _run_recent_report(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "feishu_triparty_pk_report.py"),
        "--hours",
        str(max(int(args.recent_hours), 1)),
        "--recent-hours",
        str(max(int(args.recent_hours), 1)),
        "--recent-min-sessions",
        str(max(int(args.recent_min_sessions), 1)),
        "--strict-goal-metric",
        str(args.strict_goal_metric or "p90"),
        "--json-out",
        str(Path(args.json_out).resolve()),
        "--markdown-out",
        str(Path(args.markdown_out).resolve()),
    ]
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _ensure_env(str(args.chat_id or "").strip())
        probe_id = f"probe-{uuid.uuid4().hex[:12]}"
        probe_text = f"[{probe_id}] {str(args.message or '').strip()} @hermes"
        send_started_at = time.time()
        send_result = asyncio.run(_send_probe(str(args.chat_id or "").strip(), probe_text))
        if not send_result["success"]:
            raise RuntimeError(f"Feishu probe send failed: {send_result['error']}")
        time.sleep(max(float(args.wait_seconds or 0.0), 0.0))
        _run_recent_report(args)
        report = json.loads(Path(args.json_out).read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": "ok",
                    "probe_id": probe_id,
                    "probe_text": probe_text,
                    "chat_id": str(args.chat_id or "").strip(),
                    "feishu_send": send_result,
                    "wait_seconds": float(args.wait_seconds or 0.0),
                    "sent_at_epoch_ms": int(send_started_at * 1000),
                    "recent_window_eval": report.get("recent_window_eval"),
                    "goal_check_recent": report.get("goal_check_recent"),
                    "json_report": str(Path(args.json_out).resolve()),
                    "markdown_report": str(Path(args.markdown_out).resolve()),
                    "note": "This probe sends a real Feishu message and then regenerates the recent triparty report snapshot. Use feishu_triparty_tail.py for per-event live tailing.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"feishu_triparty_probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
