import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover - optional dependency in local/report-only environments
    modal = None


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


def _load_feishu_perf_module():
    spec = spec_from_file_location("feishu_perf_summary_module", REPO_ROOT / "internal" / "feishu_perf.py")
    module = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _modal_python_candidates() -> list[Path]:
    candidates: list[Path] = []

    explicit = str(os.getenv("HERMES_MODAL_PYTHON") or "").strip()
    if explicit:
        candidates.append(Path(explicit))

    sibling_python = MODAL_CLI.with_name("python.exe")
    candidates.append(sibling_python)

    modal_cli_on_path = shutil.which("modal")
    if modal_cli_on_path:
        candidates.append(Path(modal_cli_on_path).with_name("python.exe"))

    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.exists() and resolved not in deduped:
            deduped.append(resolved)
    return deduped


def _run_modal_remote_subprocess(app: str, function_name: str, payload: dict) -> dict:
    inline = """
import json
import modal
import sys

app_name = sys.argv[1]
fn_name = sys.argv[2]
kwargs = json.loads(sys.argv[3])
result = modal.Function.from_name(app_name, fn_name).remote(**kwargs)
print(json.dumps(result, ensure_ascii=False))
""".strip()

    errors: list[str] = []
    for python_path in _modal_python_candidates():
        command = [str(python_path), "-c", inline, app, function_name, json.dumps(payload, ensure_ascii=False)]
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            timeout=180,
        )
        if completed.returncode != 0:
            errors.append(f"{python_path}: {completed.stderr.strip() or completed.stdout.strip() or 'unknown error'}")
            continue
        try:
            decoded = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            errors.append(f"{python_path}: json_decode_failed:{exc}")
            continue
        if isinstance(decoded, dict):
            return decoded
        errors.append(f"{python_path}: unexpected_payload_type={type(decoded).__name__}")
    raise RuntimeError("modal_remote_subprocess_failed: " + " | ".join(errors or ["no_modal_python_found"]))


def _normalize_phase_timings(value) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, raw in value.items():
        try:
            normalized[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return normalized


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
            "cpu": float(getattr(modal_module, "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_CPU", 1.0)),
            "memory_mb": int(getattr(modal_module, "DEFAULT_FEISHU_BACKGROUND_EXEC_WORKER_MEMORY_MB", 1024)),
            "duration_field": "background_exec_elapsed_ms",
        },
        "feishu_ack_reaction_worker": {
            "cpu": float(getattr(modal_module, "DEFAULT_FEISHU_ACK_REACTION_WORKER_CPU", 0.5)),
            "memory_mb": int(getattr(modal_module, "DEFAULT_FEISHU_ACK_REACTION_WORKER_MEMORY_MB", 512)),
            "duration_field": "ack_reaction_total_elapsed_ms",
            "include_if": lambda event: not isinstance(event.get("ack_reaction_inline_elapsed_ms"), (int, float)),
        },
        "chat_queue_worker": {
            "cpu": float(getattr(modal_module, "DEFAULT_CHAT_QUEUE_WORKER_CPU", 1.0)),
            "memory_mb": int(getattr(modal_module, "DEFAULT_CHAT_QUEUE_WORKER_MEMORY_MB", 1024)),
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


def _emit_blocked_report(*, blocker: str, message: str, app: str) -> int:
    payload = {
        "status": "blocked",
        "blocker": blocker,
        "app": app,
        "message": message,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2


def _build_perf_summary_from_trace_rows(
    *,
    app: str,
    limit: int,
    since_seconds: int,
    event_type: str,
    experiment_label: str,
    app_name_filter: str,
    snapshot_profile: str,
    include_duplicates: bool,
) -> dict:
    feishu_perf_module = _load_feishu_perf_module()
    if modal is None:
        rows_payload = _run_modal_remote_subprocess(app, "debug_feishu_trace", {"limit": max(limit, 1)})
    else:
        rows_payload = modal.Function.from_name(app, "debug_feishu_trace").remote(limit=max(limit, 1))
    rows = rows_payload.get("rows") if isinstance(rows_payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    return feishu_perf_module.build_feishu_perf_summary_from_rows(
        rows,
        since_seconds=since_seconds,
        event_type=event_type,
        experiment_label=experiment_label,
        app_name_filter=app_name_filter,
        snapshot_profile=snapshot_profile,
        include_duplicates=include_duplicates,
        normalize_phase_timings_fn=_normalize_phase_timings,
    )


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

    try:
        request_payload = {
            "limit": max(args.limit, 1),
            "since_seconds": since_seconds,
            "event_type": args.event_type,
            "experiment_label": args.experiment_label,
            "app_name_filter": args.app_name_filter,
            "snapshot_profile": args.snapshot_profile,
            "include_duplicates": args.include_duplicates,
        }
        if modal is None:
            perf_summary = _run_modal_remote_subprocess(args.app, "debug_feishu_perf_summary", request_payload)
        else:
            perf_summary = modal.Function.from_name(args.app, "debug_feishu_perf_summary").remote(**request_payload)
    except Exception:
        try:
            perf_summary = _build_perf_summary_from_trace_rows(
                app=args.app,
                limit=args.limit,
                since_seconds=since_seconds,
                event_type=args.event_type,
                experiment_label=args.experiment_label,
                app_name_filter=args.app_name_filter,
                snapshot_profile=args.snapshot_profile,
                include_duplicates=args.include_duplicates,
            )
        except Exception as exc:
            return _emit_blocked_report(
                blocker="modal_debug_function_missing",
                message=(
                    "Unable to query Modal debug_feishu_perf_summary or rebuild the summary from debug_feishu_trace. "
                    "The local repo snapshot and deployed Modal app may still be out of sync. "
                    f"Original error: {exc}"
                ),
                app=args.app,
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
