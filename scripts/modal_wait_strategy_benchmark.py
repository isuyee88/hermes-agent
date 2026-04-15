from __future__ import annotations

import argparse
import json
import time
from decimal import Decimal
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
CPU_COST_PER_CORE_SECOND = Decimal("0.00001310")
MEMORY_COST_PER_GIB_SECOND = Decimal("0.00000222")
PRICING_SOURCE = "https://modal.com/pricing"


def _decimal_str(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")), "f")


def _estimate_function_cost_usd(duration_ms: int | float, *, cpu: float, memory_mb: int) -> Decimal:
    seconds = Decimal(str(max(duration_ms, 0))) / Decimal("1000")
    cpu_cost = Decimal(str(cpu)) * CPU_COST_PER_CORE_SECOND * seconds
    memory_cost = (Decimal(str(memory_mb)) / Decimal("1024")) * MEMORY_COST_PER_GIB_SECOND * seconds
    return cpu_cost + memory_cost


def _invoke_modal_function(app_name: str, function_name: str, **kwargs) -> tuple[dict, int]:
    fn = modal.Function.from_name(app_name, function_name)
    started_at = time.perf_counter()
    result = fn.remote(**kwargs)
    wall_elapsed_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    if not isinstance(result, dict):
        raise RuntimeError(f"{function_name} returned non-dict payload: {type(result)!r}")
    return result, wall_elapsed_ms


def _build_hold_summary(label: str, remote_result: dict, wall_elapsed_ms: int) -> dict:
    resource = remote_result.get("resource_shape") or {}
    cpu = float(resource.get("cpu") or 0)
    memory_mb = int(resource.get("memory_mb") or 0)
    modal_compute_ms = int(remote_result.get("elapsed_ms") or wall_elapsed_ms)
    estimated_cost = _estimate_function_cost_usd(modal_compute_ms, cpu=cpu, memory_mb=memory_mb)
    return {
        "strategy": label,
        "pattern": remote_result.get("pattern"),
        "wall_elapsed_ms": wall_elapsed_ms,
        "modal_compute_elapsed_ms": modal_compute_ms,
        "resource_shape": {
            "cpu": cpu,
            "memory_mb": memory_mb,
        },
        "estimated_cost_usd": _decimal_str(estimated_cost),
        "notes": remote_result.get("notes") or "",
    }


def _poll_offsets(wait_seconds: float, poll_interval_seconds: float) -> list[float]:
    offsets = [0.0]
    normalized_wait = max(0.0, float(wait_seconds))
    normalized_interval = max(0.1, float(poll_interval_seconds))
    while offsets[-1] < normalized_wait:
        offsets.append(min(normalized_wait, offsets[-1] + normalized_interval))
    return offsets


def _run_external_poll_summary(
    *,
    app_name: str,
    wait_seconds: float,
    poll_interval_seconds: float,
    work_ms: int,
) -> dict:
    offsets = _poll_offsets(wait_seconds, poll_interval_seconds)
    tick_rows: list[dict] = []
    started_at = time.perf_counter()
    for index, offset_seconds in enumerate(offsets):
        target_at = started_at + offset_seconds
        remaining = target_at - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        remote_result, wall_elapsed_ms = _invoke_modal_function(
            app_name,
            "benchmark_modal_wait_poll_tick",
            work_ms=work_ms,
        )
        tick_rows.append(
            {
                "index": index,
                "scheduled_offset_ms": int(offset_seconds * 1000),
                "wall_elapsed_ms": wall_elapsed_ms,
                "modal_compute_elapsed_ms": int(remote_result.get("elapsed_ms") or wall_elapsed_ms),
                "resource_shape": remote_result.get("resource_shape") or {},
            }
        )
    total_wall_elapsed_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    total_modal_compute_ms = sum(int(item["modal_compute_elapsed_ms"]) for item in tick_rows)
    cpu = float((tick_rows[0].get("resource_shape") or {}).get("cpu") or 0.125)
    memory_mb = int((tick_rows[0].get("resource_shape") or {}).get("memory_mb") or 256)
    estimated_cost = _estimate_function_cost_usd(total_modal_compute_ms, cpu=cpu, memory_mb=memory_mb)
    provider_ready_at_ms = int(max(wait_seconds, 0) * 1000)
    detection_delay_ms = max(0, total_wall_elapsed_ms - provider_ready_at_ms)
    return {
        "strategy": f"external_poll_{poll_interval_seconds:g}s",
        "pattern": "external_poll",
        "wall_elapsed_ms": total_wall_elapsed_ms,
        "modal_compute_elapsed_ms": total_modal_compute_ms,
        "resource_shape": {
            "cpu": cpu,
            "memory_mb": memory_mb,
        },
        "estimated_cost_usd": _decimal_str(estimated_cost),
        "poll_interval_seconds": poll_interval_seconds,
        "poll_invocation_count": len(tick_rows),
        "provider_ready_detection_delay_ms": detection_delay_ms,
        "notes": "Approximates a provider async callback/poll design where waiting happens outside Modal and Modal only pays for short poll ticks.",
        "ticks": tick_rows,
    }


def _build_recommendation(strategies: list[dict], *, provider_async_supported: bool) -> dict:
    by_cost = sorted(strategies, key=lambda item: Decimal(str(item.get("estimated_cost_usd") or "0")))
    by_wall = sorted(strategies, key=lambda item: int(item.get("wall_elapsed_ms") or 0))
    lowest_cost = by_cost[0] if by_cost else None
    fastest = by_wall[0] if by_wall else None
    recommendation = {
        "provider_async_supported": provider_async_supported,
        "lowest_cost_strategy": lowest_cost.get("strategy") if lowest_cost else None,
        "fastest_strategy": fastest.get("strategy") if fastest else None,
        "recommended_strategy": None,
        "reason": "",
    }
    if provider_async_supported:
        recommendation["recommended_strategy"] = recommendation["lowest_cost_strategy"]
        recommendation["reason"] = (
            "If the provider exposes a real async job/callback/poll API, external poll/callback minimizes Modal billed wait time."
        )
    else:
        background = next((item for item in strategies if item.get("strategy") == "background_hold"), None)
        light = next((item for item in strategies if item.get("strategy") == "light_hold"), None)
        recommendation["recommended_strategy"] = "background_hold"
        if background and light:
            recommendation["reason"] = (
                "Current OpenRouter/NVIDIA probes did not expose a usable async request_id/poll_url path. "
                "That means pure Modal synchronous provider waiting is still required today; low-resource hold only becomes viable after splitting provider wait from generation."
            )
        else:
            recommendation["reason"] = "Current providers appear synchronous in practice, so a background hold remains the only directly usable pattern today."
    return recommendation


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Modal wait patterns for provider-result return paths.")
    parser.add_argument("--app", default="hermes-agent", help="Modal app name.")
    parser.add_argument("--wait-seconds", type=float, default=15.0, help="Target provider wait window to simulate.")
    parser.add_argument("--hold-work-ms", type=int, default=250, help="Extra work time after the wait for hold strategies.")
    parser.add_argument("--poll-work-ms", type=int, default=150, help="Work time per external poll tick.")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        action="append",
        default=[],
        help="External poll intervals to test. May be passed multiple times. Defaults to 1s and 3s.",
    )
    parser.add_argument(
        "--provider-async-supported",
        action="store_true",
        help="Set if a provider-native async request/callback path has been verified for the target flow.",
    )
    args = parser.parse_args()

    poll_intervals = args.poll_interval_seconds or [1.0, 3.0]
    strategies: list[dict] = []

    background_result, background_wall_ms = _invoke_modal_function(
        args.app,
        "benchmark_modal_wait_background_hold",
        wait_seconds=args.wait_seconds,
        work_ms=args.hold_work_ms,
    )
    strategies.append(_build_hold_summary("background_hold", background_result, background_wall_ms))

    light_result, light_wall_ms = _invoke_modal_function(
        args.app,
        "benchmark_modal_wait_light_hold",
        wait_seconds=args.wait_seconds,
        work_ms=args.hold_work_ms,
    )
    strategies.append(_build_hold_summary("light_hold", light_result, light_wall_ms))

    for poll_interval_seconds in poll_intervals:
        strategies.append(
            _run_external_poll_summary(
                app_name=args.app,
                wait_seconds=args.wait_seconds,
                poll_interval_seconds=poll_interval_seconds,
                work_ms=args.poll_work_ms,
            )
        )

    output = {
        "status": "ok",
        "pricing_source": PRICING_SOURCE,
        "wait_seconds": args.wait_seconds,
        "hold_work_ms": args.hold_work_ms,
        "poll_work_ms": args.poll_work_ms,
        "strategies": strategies,
        "recommendation": _build_recommendation(
            strategies,
            provider_async_supported=bool(args.provider_async_supported),
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
