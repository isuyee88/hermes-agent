from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.feishu_kpi_tools import (  # noqa: E402
    DEFAULT_TIMEZONE,
    format_feishu_kpi_execution_summary,
    generate_feishu_kpi_execution_snapshot,
)


DEFAULT_JSON_OUT = REPO_ROOT / ".tmp-feishu-kpi-execution-snapshot.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / ".tmp-feishu-kpi-execution-snapshot.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a unified Feishu KPI execution snapshot.")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--compare-days", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--recent-hours", type=int, default=3)
    parser.add_argument("--recent-min-sessions", type=int, default=20)
    parser.add_argument("--strict-goal-metric", default="p90")
    parser.add_argument("--session-cost-source-policy", default="strict")
    parser.add_argument("--artifacts-dir", default="")
    parser.add_argument("--skip-analytics-snapshot", action="store_true")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT))
    args = parser.parse_args()

    snapshot = generate_feishu_kpi_execution_snapshot(
        hours=args.hours,
        compare_days=args.compare_days,
        timezone=args.timezone,
        recent_hours=args.recent_hours,
        recent_min_sessions=args.recent_min_sessions,
        strict_goal_metric=args.strict_goal_metric,
        session_cost_source_policy=args.session_cost_source_policy,
        include_analytics_snapshot=not args.skip_analytics_snapshot,
        artifacts_dir=(args.artifacts_dir or None),
    )
    summary_text = format_feishu_kpi_execution_summary(snapshot)

    json_out = Path(args.json_out)
    json_out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_out = Path(args.markdown_out)
    markdown_out.write_text(summary_text + "\n", encoding="utf-8")

    print(summary_text)
    return 0 if snapshot.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
