import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
MODAL_CLI = REPO_ROOT / ".modal-py311" / "Scripts" / "modal.exe"
CPU_COST_PER_CORE_SECOND = Decimal(os.getenv("HERMES_MODAL_CPU_COST_PER_CORE_SECOND", "0.0000131"))
MEMORY_COST_PER_GIB_SECOND = Decimal(os.getenv("HERMES_MODAL_MEMORY_COST_PER_GIB_SECOND", "0.00000222"))
PRICING_SOURCE = "https://modal.com/pricing"


def _load_modal_module():
    spec = spec_from_file_location("modal_cost_model", REPO_ROOT / "modal_.py")
    module = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _decimal_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value.quantize(Decimal("0.00000001")), "f")


def _estimate_function_cost(duration_ms: int | float | None, *, cpu: float, memory_mb: int | float) -> Decimal:
    if duration_ms is None:
        return Decimal("0")
    seconds = Decimal(str(max(float(duration_ms), 0.0))) / Decimal("1000")
    memory_gib = Decimal(str(max(float(memory_mb), 0.0))) / Decimal("1024")
    cpu_decimal = Decimal(str(max(float(cpu), 0.0)))
    return seconds * ((cpu_decimal * CPU_COST_PER_CORE_SECOND) + (memory_gib * MEMORY_COST_PER_GIB_SECOND))


def _build_function_cost_breakdown(perf_summary: dict, modal_module) -> dict:
    resources = {
        "web_app": {"cpu": 1.0, "memory_mb": 2048, "duration_field": "response_elapsed_ms"},
        "feishu_inline_worker": {"cpu": 0.5, "memory_mb": 1024, "duration_field": "inline_elapsed_ms"},
        "feishu_background_exec_worker": {
            "cpu": float(modal_module.DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU),
            "memory_mb": int(modal_module.DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB),
            "duration_field": "background_exec_elapsed_ms",
        },
        "feishu_ack_reaction_worker": {
            "cpu": float(modal_module.DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU),
            "memory_mb": int(modal_module.DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB),
            "duration_field": "ack_reaction_total_elapsed_ms",
            "include_if": lambda event: not isinstance(event.get("ack_reaction_inline_elapsed_ms"), (int, float)),
        },
        "chat_queue_worker": {
            "cpu": float(modal_module.DEFAULT_CHAT_QUEUE_WORKER_CPU),
            "memory_mb": int(modal_module.DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB),
            "duration_field": "worker_elapsed_ms",
        },
    }

    totals: dict[str, Decimal] = {name: Decimal("0") for name in resources}
    counts: dict[str, int] = {name: 0 for name in resources}
    events = perf_summary.get("events") or []
    for event in events:
        if not isinstance(event, dict):
            continue
        for function_name, config in resources.items():
            include_if = config.get("include_if")
            if callable(include_if) and not include_if(event):
                continue
            duration_value = event.get(config["duration_field"])
            if not isinstance(duration_value, (int, float)):
                continue
            estimated = _estimate_function_cost(
                duration_value,
                cpu=float(config["cpu"]),
                memory_mb=float(config["memory_mb"]),
            )
            if estimated <= 0:
                continue
            totals[function_name] += estimated
            counts[function_name] += 1

    raw_total = sum(totals.values(), Decimal("0"))
    official_total = Decimal(str((perf_summary.get("cost_summary") or {}).get("total_cost_usd") or "0"))
    calibration_ratio = (official_total / raw_total) if raw_total > 0 else None

    by_function: dict[str, dict] = {}
    for function_name, raw_cost in totals.items():
        invocation_count = counts[function_name]
        calibrated_cost = (raw_cost * calibration_ratio) if calibration_ratio is not None else None
        by_function[function_name] = {
            "resource_shape": {
                "cpu": resources[function_name]["cpu"],
                "memory_mb": resources[function_name]["memory_mb"],
            },
            "duration_field": resources[function_name]["duration_field"],
            "invocation_count": invocation_count,
            "raw_estimated_cost_usd": _decimal_str(raw_cost),
            "raw_estimated_cost_per_invocation_usd": _decimal_str(raw_cost / invocation_count) if invocation_count else None,
            "calibrated_cost_usd": _decimal_str(calibrated_cost),
            "calibrated_cost_per_invocation_usd": _decimal_str(calibrated_cost / invocation_count)
            if calibrated_cost is not None and invocation_count
            else None,
        }

    return {
        "model": {
            "cpu_cost_per_core_second_usd": _decimal_str(CPU_COST_PER_CORE_SECOND),
            "memory_cost_per_gib_second_usd": _decimal_str(MEMORY_COST_PER_GIB_SECOND),
            "pricing_source": PRICING_SOURCE,
            "notes": [
                "Raw estimates use reserved CPU/memory shape times observed function duration.",
                "Calibrated costs scale raw estimates to match the official app billing total for the selected window.",
                "If the app window contains unrelated workloads outside the Feishu event set, calibrated per-function costs are allocation estimates, not direct official function invoices.",
                "Inline ack reactions are counted inside web_app response time and excluded from feishu_ack_reaction_worker to avoid double counting.",
            ],
        },
        "raw_total_estimated_cost_usd": _decimal_str(raw_total),
        "official_app_total_cost_usd": _decimal_str(official_total),
        "calibration_ratio": _decimal_str(calibration_ratio) if calibration_ratio is not None else None,
        "by_function": by_function,
    }


def _iso_local(dt: datetime) -> str:
    return dt.astimezone().replace(microsecond=0).isoformat()


def _run_modal_billing_report(start: datetime, end: datetime) -> list[dict]:
    cmd = [
        str(MODAL_CLI if MODAL_CLI.exists() else "modal"),
        "billing",
        "report",
        "--start",
        _iso_local(start),
        "--end",
        _iso_local(end),
        "--resolution",
        "h",
        "--tz",
        "local",
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        timeout=180,
        check=True,
    )
    payload = json.loads(proc.stdout or "[]")
    return payload if isinstance(payload, list) else []


def _build_billing_summary(rows: list[dict], descriptions: set[str]) -> dict:
    filtered = [row for row in rows if str(row.get("Description") or "").strip() in descriptions]
    total_cost = sum(Decimal(str(row.get("Cost") or "0")) for row in filtered)
    by_interval: dict[str, Decimal] = {}
    for row in filtered:
        interval = str(row.get("Interval Start") or "").strip()
        by_interval[interval] = by_interval.get(interval, Decimal("0")) + Decimal(str(row.get("Cost") or "0"))
    return {
        "matched_rows": len(filtered),
        "matched_descriptions": sorted(descriptions),
        "total_cost_usd": str(total_cost),
        "hourly_costs": [
            {"interval_start": interval, "cost_usd": str(cost)}
            for interval, cost in sorted(by_interval.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Feishu performance phases and Modal billing for an A/B window.")
    parser.add_argument("--app", default="hermes-agent", help="Modal app name to query for debug_feishu_perf_summary.")
    parser.add_argument(
        "--billing-description",
        action="append",
        default=[],
        help="Modal billing Description values to include. Defaults to --app if omitted.",
    )
    parser.add_argument("--since-hours", type=float, default=24.0, help="Summary window size in hours.")
    parser.add_argument("--limit", type=int, default=20000, help="Trace rows to read from debug_feishu_perf_summary.")
    parser.add_argument("--event-type", default="im.message.receive_v1", help="Feishu event type filter.")
    parser.add_argument("--experiment-label", default="", help="Optional experiment label filter.")
    parser.add_argument("--snapshot-profile", default="", help="Optional snapshot profile filter.")
    parser.add_argument("--app-name-filter", default="", help="Optional trace app_name filter.")
    parser.add_argument("--include-duplicates", action="store_true", help="Include duplicate webhook retries in event stats.")
    args = parser.parse_args()
    modal_module = _load_modal_module()

    since_seconds = max(int(args.since_hours * 3600), 0)
    descriptions = {item.strip() for item in args.billing_description if item.strip()}
    if not descriptions:
        descriptions.add(args.app)

    perf_summary = modal.Function.from_name(args.app, "debug_feishu_perf_summary").remote(
        limit=max(args.limit, 1),
        since_seconds=since_seconds,
        event_type=args.event_type,
        experiment_label=args.experiment_label,
        app_name_filter=args.app_name_filter,
        snapshot_profile=args.snapshot_profile,
        include_duplicates=args.include_duplicates,
    )

    end = datetime.now().astimezone()
    start = end - timedelta(seconds=since_seconds)
    billing_rows = _run_modal_billing_report(start, end)
    billing_summary = _build_billing_summary(billing_rows, descriptions)

    event_count = int(perf_summary.get("event_count") or 0)
    total_cost_decimal = Decimal(billing_summary["total_cost_usd"])
    perf_summary["cost_summary"] = {
        **billing_summary,
        "window_start": _iso_local(start),
        "window_end": _iso_local(end),
        "cost_per_event_usd": (str(total_cost_decimal / event_count) if event_count > 0 else None),
    }
    perf_summary["function_cost_summary"] = _build_function_cost_breakdown(perf_summary, modal_module)

    print(json.dumps(perf_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
