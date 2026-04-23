from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

from tools.feishu_kpi_tools import DEFAULT_TIMEZONE, generate_feishu_kpi_execution_snapshot  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / ".tmp-feishu-execution-task-checklist.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / ".tmp-feishu-execution-task-checklist.md"


def _tail(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_delivery_check(
    *,
    target_chat_id: str,
    expected_webhook_url: str,
    recent_message_window_minutes: int,
    include_all_env_apps: bool = True,
) -> dict[str, Any]:
    script_path = REPO_ROOT / "scripts" / "check_feishu_delivery_path.py"
    command = [sys.executable, str(script_path)]
    if include_all_env_apps:
        command.append("--all-env-apps")
    if target_chat_id:
        command.extend(["--target-chat-id", target_chat_id])
    if expected_webhook_url:
        command.extend(["--expected-webhook-url", expected_webhook_url])
    if recent_message_window_minutes > 0:
        command.extend(["--recent-message-window-minutes", str(recent_message_window_minutes)])
    env = os.environ.copy()
    user_token, user_token_source = _discover_user_access_token()
    if user_token:
        env["FEISHU_USER_ACCESS_TOKEN"] = user_token
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
    )
    payload: dict[str, Any] | None = None
    try:
        decoded = json.loads(completed.stdout or "{}")
        if isinstance(decoded, dict):
            payload = decoded
    except Exception:
        payload = None
    return {
        "success": payload is not None,
        "returncode": completed.returncode,
        "command": " ".join(shlex.quote(part) for part in command),
        "user_token_source": user_token_source,
        "stdout_tail": _tail(completed.stdout or ""),
        "stderr_tail": _tail(completed.stderr or ""),
        "payload": payload or {},
    }


def _discover_user_access_token() -> tuple[str, str]:
    latest_token_files = sorted(
        REPO_ROOT.glob("feishu_tokens_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for token_file in latest_token_files:
        try:
            payload = json.loads(token_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        token = str(payload.get("user_access_token") or "").strip()
        if token:
            return token, str(token_file)
    env_token = str(os.getenv("FEISHU_USER_ACCESS_TOKEN") or "").strip()
    if env_token:
        return env_token, "env:FEISHU_USER_ACCESS_TOKEN"
    return "", ""


def _parse_user_send_error(text: str) -> dict[str, Any]:
    code_match = re.search(r"['\"]code['\"]:\s*(\d+)", text)
    msg_match = re.search(r"['\"]msg['\"]:\s*['\"]([^'\"]+)['\"]", text)
    code = int(code_match.group(1)) if code_match else 0
    message = msg_match.group(1) if msg_match else ""
    required_scope = ""
    scope_match = re.search(r"requires ([A-Za-z0-9:._-]+) scope", message)
    if scope_match:
        required_scope = scope_match.group(1)
    return {
        "code": code,
        "message": message,
        "required_scope": required_scope,
    }


def _load_latest_user_send_observation() -> dict[str, Any] | None:
    report_files = sorted(
        REPO_ROOT.glob("feishu_bot_test_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not report_files:
        return None
    latest = report_files[0]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        return None
    error_text = str(first.get("error") or "")
    return {
        "success": bool(first.get("success")),
        "source_file": str(latest),
        "parsed_error": _parse_user_send_error(error_text),
        "raw_result": first,
    }


def _task(
    task_id: str,
    title: str,
    *,
    category: str,
    status: str,
    rationale: str,
    evidence: list[str] | None = None,
    commands: list[str] | None = None,
    depends_on: list[str] | None = None,
    current_conclusion: str = "",
    verification: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "title": title,
        "category": category,
        "status": status,
        "rationale": rationale,
        "evidence": evidence or [],
        "commands": commands or [],
        "depends_on": depends_on or [],
        "current_conclusion": current_conclusion,
        "verification": verification or [],
    }


def _delivery_apps(delivery_payload: dict[str, Any]) -> list[dict[str, Any]]:
    apps = delivery_payload.get("apps")
    if isinstance(apps, list):
        return [item for item in apps if isinstance(item, dict)]
    if isinstance(delivery_payload, dict) and delivery_payload:
        return [delivery_payload]
    return []


def _active_delivery_apps(apps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in apps if int(item.get("visible_chat_count") or 0) > 0]


def _preferred_delivery_app_ids() -> list[str]:
    ids: list[str] = []

    def _append(value: str) -> None:
        normalized = str(value or "").strip()
        if normalized and normalized not in ids:
            ids.append(normalized)

    preferred_suffix = str(
        os.getenv("HERMES_FEISHU_APP_SUFFIX")
        or os.getenv("FEISHU_APP_SUFFIX")
        or ""
    ).strip()
    if preferred_suffix in {"", "1", "default", "primary"}:
        _append(str(os.getenv("FEISHU_APP_ID") or ""))
    elif preferred_suffix in {"2", "3"}:
        _append(str(os.getenv(f"FEISHU_APP_ID{preferred_suffix}") or ""))

    # Fall back to commonly used retained app order for this workspace.
    _append(str(os.getenv("FEISHU_APP_ID3") or ""))
    _append(str(os.getenv("FEISHU_APP_ID2") or ""))
    _append(str(os.getenv("FEISHU_APP_ID") or ""))
    return ids


def _pick_by_preferred_ids(
    candidates: list[dict[str, Any]],
    preferred_ids: list[str],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    for app_id in preferred_ids:
        for item in candidates:
            if str(item.get("app_id") or "").strip() == app_id:
                return item
    return candidates[0]


def _canonical_delivery_app(apps: list[dict[str, Any]]) -> dict[str, Any]:
    preferred_ids = _preferred_delivery_app_ids()
    active_apps = _active_delivery_apps(apps)
    ready_active_apps = [item for item in active_apps if bool(item.get("delivery_ready"))]
    ready_apps = [item for item in apps if bool(item.get("delivery_ready"))]
    picked = _pick_by_preferred_ids(ready_active_apps, preferred_ids)
    if picked:
        return picked
    picked = _pick_by_preferred_ids(ready_apps, preferred_ids)
    if picked:
        return picked
    picked = _pick_by_preferred_ids(active_apps, preferred_ids)
    if picked:
        return picked
    picked = _pick_by_preferred_ids(apps, preferred_ids)
    if picked:
        return picked
    if active_apps:
        return active_apps[0]
    if apps:
        return apps[0]
    return {}


def _task_ids(tasks: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("id")) for item in tasks]


def _task_done(tasks: list[dict[str, Any]], task_id: str) -> bool:
    for item in tasks:
        if item.get("id") == task_id:
            return item.get("status") == "done"
    return False


def _status_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"done": 0, "blocked": 0, "todo": 0, "attention": 0}
    for item in tasks:
        status = str(item.get("status") or "todo")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _render_status_label(status: str) -> str:
    return {
        "done": "已完成",
        "blocked": "阻塞",
        "attention": "需关注",
        "todo": "待执行",
    }.get(str(status), str(status))


def _derive_execution_order(tasks: list[dict[str, Any]]) -> list[list[str]]:
    ordered_groups = [
        ["FX001", "FX002"],
        ["FX003", "FX004"],
        ["FX005", "FX006"],
    ]
    present = set(_task_ids(tasks))
    return [[task_id for task_id in group if task_id in present] for group in ordered_groups if any(task_id in present for task_id in group)]


def _derive_next_focus(tasks: list[dict[str, Any]], execution_order: list[list[str]]) -> list[str]:
    ordered_ids = [task_id for group in execution_order for task_id in group]
    if not ordered_ids:
        ordered_ids = _task_ids(tasks)
    next_focus: list[str] = []
    for task_id in ordered_ids:
        if not _task_done(tasks, task_id):
            next_focus.append(task_id)
    deduped: list[str] = []
    for task_id in next_focus:
        if task_id not in deduped:
            deduped.append(task_id)
    return deduped[:5]


def _derive_checklist(
    *,
    snapshot: dict[str, Any],
    delivery_result: dict[str, Any] | None,
    user_send_result: dict[str, Any] | None,
    target_chat_id: str,
) -> dict[str, Any]:
    summary = snapshot.get("summary") or {}
    statuses = summary.get("statuses") or {}
    blockers = list(snapshot.get("priority_blockers") or [])
    next_actions = list(snapshot.get("next_actions") or [])

    delivery_payload = (delivery_result or {}).get("payload") or {}
    delivery_status = str(delivery_payload.get("status") or "").strip() or "unknown"
    delivery_summary = str(delivery_payload.get("summary") or "").strip()
    apps = _delivery_apps(delivery_payload)
    canonical_app = _canonical_delivery_app(apps)
    canonical_app_id = str(canonical_app.get("app_id") or "").strip()
    canonical_published = canonical_app.get("published_version_summary") or {}
    canonical_topology = canonical_app.get("chat_topology") or {}
    canonical_recent = canonical_app.get("recent_chat_activity") or {}
    canonical_missing_callbacks = [
        str(item) for item in (canonical_published.get("missing_required_callbacks") or [])
    ]
    primary_blockers: list[str] = []
    for app in apps:
        blocker = str((app or {}).get("primary_blocker") or "").strip()
        if blocker and blocker not in primary_blockers:
            primary_blockers.append(blocker)

    canonical_has_multiple_bots = int(canonical_topology.get("chats_with_multiple_bots") or 0) > 0
    canonical_user_token_expired = int(canonical_topology.get("chats_with_expired_user_token") or 0) > 0
    canonical_user_access_token_missing = int(canonical_topology.get("chats_with_missing_user_access_token") or 0) > 0
    canonical_user_scope_missing_chat_members = (
        int(canonical_topology.get("chats_with_user_scope_missing_chat_members_read") or 0) > 0
    )
    canonical_member_audit_incomplete = int(canonical_topology.get("chats_with_incomplete_members") or 0) > 0
    canonical_recent_other_app_replies = bool(canonical_recent.get("total_recent_app_messages_from_other_app_ids"))
    message_read_missing_on_canonical = "im.message.message_read_v1" in canonical_missing_callbacks
    user_send_error = ((user_send_result or {}).get("parsed_error") or {}) if isinstance(user_send_result, dict) else {}
    user_send_error_code = int(user_send_error.get("code") or 0)
    user_send_required_scope = str(user_send_error.get("required_scope") or "").strip()
    user_send_missing_send_as_user = (
        user_send_error_code == 230027 and user_send_required_scope == "im:message.send_as_user"
    )
    read_receipt_met = statuses.get("read_receipt_under_5s") == "met"
    session_cost_met = statuses.get("session_cost_under_0_0045") == "met"
    idle_cost_met = statuses.get("modal_idle_hourly_cost_under_0_005") == "met"
    cache_hit_met = statuses.get("cache_hit_rate_over_0_30") == "met"
    browser_single_call_met = statuses.get("browser_single_ai_call_completion_rate_over_0_50") == "met"
    capability_match_met = statuses.get("capability_match_rate_equals_1_00") == "met"
    preferred_model_met = statuses.get("preferred_model_selection_accuracy_over_0_95") == "met"
    chain_blocker = str((snapshot.get("chain_status") or {}).get("blocker") or "").strip()
    chain_ready = chain_blocker in {"", "none"}
    non_kpi_observations: list[str] = []
    if canonical_has_multiple_bots:
        non_kpi_observations.append("目标聊天仍存在多 bot，但这不再作为 KPI 主线 blocker。")
    if canonical_recent_other_app_replies:
        non_kpi_observations.append("最近同一聊天里仍有其他 app 回复，但只要 app3 样本可入报表，仍可先推进 KPI。")
    if canonical_user_scope_missing_chat_members:
        non_kpi_observations.append("用户态聊天读取权限仍缺，但这只影响成员审计，不作为 KPI 主线 blocker。")
    if canonical_user_token_expired or canonical_user_access_token_missing:
        non_kpi_observations.append("当前 user token 不可用，但如果直接在飞书里与机器人对话，仍可继续积累 KPI 样本。")
    if user_send_missing_send_as_user:
        non_kpi_observations.append("缺少 `im:message.send_as_user` 只影响 API 模拟用户发消息，不影响你直接在飞书客户端与机器人聊天取样。")

    tasks: list[dict[str, Any]] = []

    if message_read_missing_on_canonical or not read_receipt_met:
        if message_read_missing_on_canonical:
            fx001_title = f"补齐 `{canonical_app_id or 'app3'}` 的 `message_read` 观测并闭合 read_receipt KPI"
            fx001_rationale = "读回执 KPI 只看是否有真实样本和是否低于 5 秒；如果 `message_read` 订阅缺失，就先补观测。"
            fx001_status = "blocked" if not chain_ready else "todo"
            fx001_conclusion = "当前 `message_read` 观测仍不完整，先补齐订阅或直接确认读回执采样链路。"
        else:
            fx001_title = "补采任意可达 Hermes 的真实读回执样本并核实 read_receipt KPI"
            fx001_rationale = "这里不要求特定聊天对象，只要求真实样本能进入 KPI 统计，验证 `read_receipt_p90_ms < 5000`。"
            fx001_status = "blocked" if not chain_ready else "todo"
            fx001_conclusion = "当前不是链路定义问题，而是真实读回执样本还不够稳定。"
        tasks.append(
            _task(
                "FX001",
                fx001_title,
                category="kpi-sampling",
                status=fx001_status,
                rationale=fx001_rationale,
                evidence=([delivery_summary] if delivery_summary else [])
                + ([f"canonical_app_id={canonical_app_id}"] if canonical_app_id else [])
                + (
                    [f"canonical_missing_callbacks={','.join(canonical_missing_callbacks)}"]
                    if canonical_missing_callbacks
                    else []
                ),
                verification=[
                    "`read_receipt_p90_ms` 不再是 `null`",
                    "`read_receipt_p90_ms < 5000`",
                ],
                commands=[
                    "python scripts/feishu_read_receipt_probe.py --chat-id <target_chat_id_or_direct_dm> --window-minutes 180 --page-size 20 --limit 10",
                    "python scripts/feishu_read_receipt_probe.py --chat-id <target_chat_id> --window-minutes 180 --page-size 20 --limit 10",
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\\suyee\\github\\hermesagent",
                ],
                current_conclusion=fx001_conclusion,
            )
        )

    if not cache_hit_met:
        tasks.append(
            _task(
                "FX002",
                "采集重复无状态文本样本并提升 AI Gateway cache eligible hit rate",
                category="kpi-sampling",
                status="blocked" if not chain_ready else "todo",
                rationale="缓存 KPI 只取决于重复无状态请求是否命中同一 cache key，不依赖你和谁聊。",
                evidence=[f"cache_eligible_hit_rate={summary.get('cache_eligible_hit_rate')}"]
                + ([delivery_summary] if delivery_summary else []),
                verification=[
                    "`cache_eligible_hit_rate > 0.30`",
                ],
                commands=[
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\\suyee\\github\\hermesagent",
                    "python scripts/feishu_triparty_pk_report.py --hours 24 --compare-days 1 2 --recent-hours 3 --recent-min-sessions 20",
                ],
                current_conclusion=(
                    "最新 Worker cache policy 已上线，但统计窗口里还没有足够的重复无状态命中样本。"
                ),
            )
        )

    fx003_needed = (
        "gap:official_cost_without_matching_function_trace" in blockers
        or "cost_report:modal_debug_function_missing" in blockers
        or not session_cost_met
    )
    if fx003_needed:
        fx003_status = "blocked" if "cost_report:modal_debug_function_missing" in blockers else "todo"
        tasks.append(
            _task(
                "FX003",
                "固定 session-cost 真相源并闭合成本口径",
                category="cost-truth",
                status=fx003_status,
                rationale="成本 KPI 只取决于是否有可信真相源，不取决于聊天对象。当前主矛盾是官方计费与 function trace 还没稳定对齐。",
                evidence=blockers,
                verification=[
                    "默认 gatekeeping 保持 `strict`",
                    "能明确说明 `strict / blended_total / official_average` 各自用途",
                ],
                commands=[
                    "python scripts/feishu_perf_cost_report.py --since-hours 24",
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\\suyee\\github\\hermesagent",
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\\suyee\\github\\hermesagent --session-cost-source-policy blended_total",
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\\suyee\\github\\hermesagent --session-cost-source-policy official_average",
                ],
                current_conclusion="当前仍缺少可稳定复原的 Modal cross-function trace 对齐，`strict` 需要继续作为唯一门禁口径。",
            )
        )

    if not idle_cost_met:
        tasks.append(
            _task(
                "FX004",
                "复测 24h/72h 空闲成本窗口并推动 idle cost 达标",
                category="cost-validation",
                status="todo",
                rationale="空闲成本 KPI 是纯计费窗口问题，和聊天对象无关，重点是让 downsizing 完整进入 24h/72h 统计。",
                evidence=[f"idle_hourly_p90_cost_usd={summary.get('idle_hourly_p90_cost_usd')}"],
                verification=[
                    "`idle_hourly_cost_p90_usd < 0.005`",
                ],
                commands=[
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\\suyee\\github\\hermesagent",
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 72 --artifacts-dir D:\\suyee\\github\\hermesagent",
                ],
                current_conclusion="近期短窗已经接近目标，但 24h/72h 成本窗还没完全成熟。",
            )
        )

    if not browser_single_call_met or not capability_match_met or not preferred_model_met:
        tasks.append(
            _task(
                "FX005",
                "补齐浏览器/路由样本并验证剩余路由类 KPI",
                category="routing-validation",
                status="blocked" if not chain_ready else "todo",
                rationale="浏览器单次调用率、能力匹配率、优选模型准确率都只看样本统计，不要求特定聊天对象。",
                evidence=[
                    f"browser_single_ai_call_completion_rate={summary.get('browser_single_ai_call_completion_rate')}",
                    f"capability_match_rate={summary.get('capability_match_rate')}",
                    f"preferred_model_selection_accuracy={summary.get('preferred_model_selection_accuracy')}",
                ],
                verification=[
                    "`browser_single_ai_call_completion_rate > 0.50`",
                    "`capability_match_rate = 1.00`",
                    "`preferred_model_selection_accuracy >= 0.95`",
                ],
                commands=[
                    "python scripts/feishu_triparty_pk_report.py --hours 24 --compare-days 1 2 --recent-hours 3 --recent-min-sessions 20",
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\\suyee\\github\\hermesagent",
                ],
                current_conclusion="当前剩余的是样本密度问题，而不是路由定义问题。",
            )
        )

    if next_actions:
        tasks.append(
            _task(
                "FX006",
                "保持每日 KPI 快照复测并吸收新的主矛盾",
                category="pdca",
                status="todo",
                rationale="清单只服务于 KPI 闭合，日更快照是最小闭环。",
                evidence=next_actions[:2],
                verification=["新快照能明确显示未达标项是否收敛"],
                commands=[
                    "python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\\suyee\\github\\hermesagent"
                ],
                current_conclusion="后续执行按 KPI 新快照滚动，不再把聊天拓扑作为主线 gate。",
            )
        )

    task_counts = _status_counts(tasks)
    execution_order = _derive_execution_order(tasks)
    next_focus = _derive_next_focus(tasks, execution_order)
    return {
        "status": "ok" if task_counts.get("blocked", 0) == 0 and task_counts.get("todo", 0) == 0 and task_counts.get("attention", 0) == 0 else "attention",
        "summary": {
            "snapshot_status": snapshot.get("status"),
            "delivery_status": delivery_status,
            "target_chat_id": target_chat_id,
            "priority_blockers": blockers,
            "delivery_primary_blockers": primary_blockers,
            "canonical_app_id": canonical_app_id,
            "user_send_result": user_send_result or {},
            "non_kpi_observations": non_kpi_observations,
            "task_counts": task_counts,
            "next_focus": next_focus,
        },
        "execution_order": execution_order,
        "tasks": tasks,
    }


def _format_markdown(checklist: dict[str, Any]) -> str:
    summary = checklist.get("summary") or {}
    task_counts = summary.get("task_counts") or {}
    lines = [
        "# Feishu 执行任务清单",
        "",
        f"- 快照状态：`{summary.get('snapshot_status')}`",
        f"- 投递状态：`{summary.get('delivery_status')}`",
        f"- 目标会话：`{summary.get('target_chat_id') or 'n/a'}`",
        f"- canonical app：`{summary.get('canonical_app_id') or 'n/a'}`",
        f"- 任务统计：已完成 `{task_counts.get('done', 0)}` / 阻塞 `{task_counts.get('blocked', 0)}` / 待执行 `{task_counts.get('todo', 0)}` / 需关注 `{task_counts.get('attention', 0)}`",
    ]
    blockers = summary.get("priority_blockers") or []
    if blockers:
        lines.append(f"- 优先阻塞：`{', '.join(str(item) for item in blockers)}`")
    observations = summary.get("non_kpi_observations") or []
    if observations:
        lines.append(f"- 非 KPI 观察：`{' | '.join(str(item) for item in observations)}`")
    next_focus = summary.get("next_focus") or []
    if next_focus:
        lines.append(f"- 当前优先推进：`{', '.join(str(item) for item in next_focus)}`")
    lines.append("")
    execution_order = checklist.get("execution_order") or []
    if execution_order:
        lines.append("## 推荐执行顺序")
        lines.append("")
        for idx, group in enumerate(execution_order, start=1):
            lines.append(f"{idx}. `{', '.join(group)}`")
        lines.append("")
    lines.append("## 任务明细")
    lines.append("")
    for task in checklist.get("tasks") or []:
        lines.append(f"- [ ] {task['id']} {task['title']}")
        lines.append(f"  - 类别：`{task['category']}`")
        lines.append(f"  - 状态：`{_render_status_label(str(task['status']))}`")
        lines.append(f"  - 原因：{task['rationale']}")
        if task.get("depends_on"):
            lines.append(f"  - 依赖：`{', '.join(task['depends_on'])}`")
        if task.get("current_conclusion"):
            lines.append(f"  - 当前结论：{task['current_conclusion']}")
        if task.get("verification"):
            lines.append("  - 验证标准：")
            for item in task["verification"]:
                lines.append(f"    - {item}")
        if task.get("commands"):
            lines.append("  - 命令：")
            for command in task["commands"]:
                lines.append(f"    - `{command}`")
        if task.get("evidence"):
            lines.append("  - 证据：")
            for item in task["evidence"]:
                lines.append(f"    - `{item}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a step-by-step Feishu execution task checklist.")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--compare-days", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--recent-hours", type=int, default=3)
    parser.add_argument("--recent-min-sessions", type=int, default=20)
    parser.add_argument("--strict-goal-metric", default="p90")
    parser.add_argument("--artifacts-dir", default="")
    parser.add_argument("--target-chat-id", default=os.getenv("FEISHU_HOME_CHANNEL", ""))
    parser.add_argument("--expected-webhook-url", default=os.getenv("HERMES_PUBLIC_BASE_URL", ""))
    parser.add_argument("--delivery-recent-message-window-minutes", type=int, default=180)
    parser.add_argument("--skip-analytics-snapshot", action="store_true")
    parser.add_argument("--skip-delivery-check", action="store_true")
    parser.add_argument("--skip-user-send-observation", action="store_true")
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
        include_analytics_snapshot=not args.skip_analytics_snapshot,
        artifacts_dir=(args.artifacts_dir or None),
    )
    if not snapshot.get("success"):
        payload = {
            "status": "error",
            "message": snapshot.get("error") or "feishu_kpi_execution_snapshot_failed",
            "snapshot": snapshot,
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(args.markdown_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    delivery_result = None
    if not args.skip_delivery_check:
        delivery_result = _run_delivery_check(
            target_chat_id=args.target_chat_id,
            expected_webhook_url=args.expected_webhook_url,
            recent_message_window_minutes=args.delivery_recent_message_window_minutes,
        )
    user_send_result = None if args.skip_user_send_observation else _load_latest_user_send_observation()

    checklist = _derive_checklist(
        snapshot=snapshot,
        delivery_result=delivery_result,
        user_send_result=user_send_result,
        target_chat_id=args.target_chat_id,
    )
    payload = {
        "status": checklist.get("status"),
        "snapshot": snapshot,
        "delivery_check": delivery_result,
        "user_send_observation": user_send_result,
        "checklist": checklist,
    }
    Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_out).write_text(_format_markdown(checklist), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "task_count": len((checklist.get("tasks") or [])),
                "target_chat_id": args.target_chat_id,
                "json_out": str(Path(args.json_out)),
                "markdown_out": str(Path(args.markdown_out)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
