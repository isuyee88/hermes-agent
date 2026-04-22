"""拉取三方日志：Cloudflare + Modal + 飞书"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime

import httpx

CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "d1215a30b84b673ef0367010b0e78c10")
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID3", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET3", "")
CHAT_ID = "oc_33edcba53ad086b50f175352947750eb"
SESSION_ID = "feishu_oc_33edcba53ad086b50f175352947750eb_1e4c486455cbb81273878"


async def get_feishu_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    async with httpx.AsyncClient() as c:
        r = await c.post(url, json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}, timeout=30)
        d = r.json()
        return d.get("tenant_access_token", "")


async def pull_cf_logs():
    print("=" * 60)
    print("[1/3] 拉取 Cloudflare Worker 日志")
    print("=" * 60)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 2 * 60 * 60 * 1000
    body = {
        "queryId": "feishu-trace",
        "view": "events",
        "limit": 500,
        "timeframe": {"from": start_ms, "to": end_ms},
        "parameters": {
            "filterCombination": "and",
            "filters": [
                {"kind": "filter", "key": "$workers.scriptName", "operation": "eq", "type": "string", "value": "hermes-feishu-gateway"}
            ],
            "orderBy": {"value": "$metadata.id", "order": "desc"},
        },
    }
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/workers/observability/telemetry/query"
    async with httpx.AsyncClient() as c:
        r = await c.post(url, json=body, headers={"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}, timeout=30)
        data = r.json()
    result = data.get("result", {}) if isinstance(data, dict) else {}
    ed = result.get("events", {}) if isinstance(result, dict) else {}
    evts = ed.get("events", []) if isinstance(ed, dict) else []
    print(f"  总事件数: {len(evts)}")
    relevant = []
    for item in evts:
        src = item.get("source", {})
        ev_str = json.dumps(src)
        if CHAT_ID in ev_str or "agent_exec" in ev_str or "send_plan" in ev_str or "reconcile" in ev_str:
            relevant.append({"ts": item.get("timestamp", ""), "event": src.get("event", ""), "error": src.get("error", ""), "keys": list(src.keys())[:15]})
    print(f"  相关事件数: {len(relevant)}")
    for e in relevant[:30]:
        err_part = f" | ERR={e['error'][:80]}" if e.get("error") else ""
        print(f"  {e['ts']} | {e['event'][:80]}{err_part}")
    with open("cf_feishu_trace.json", "w", encoding="utf-8") as f:
        json.dump(relevant, f, ensure_ascii=False, indent=2)
    return relevant


async def pull_cf_workflow_instances():
    print("\n" + "=" * 60)
    print("[2/3] 拉取 Cloudflare Workflow 实例")
    print("=" * 60)
    wf_name = "hermes-feishu-agent-workflow"
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/workflows/{wf_name}/instances"
    async with httpx.AsyncClient() as c:
        r = await c.get(url, headers={"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}, timeout=30)
        data = r.json()
    if not data.get("success"):
        print(f"  ❌ API错误: {data.get('errors')}")
        return []
    instances = data.get("result", [])
    print(f"  总实例数: {len(instances)}")
    matched = [i for i in instances if CHAT_ID in str(i.get("id", ""))]
    print(f"  匹配实例数: {len(matched)}")
    for inst in matched[:5]:
        print(f"  ID: {inst.get('id')}")
        print(f"  状态: {inst.get('status')}")
        print(f"  创建: {inst.get('created_at', 'N/A')}")
        print(f"  错误: {inst.get('error', 'N/A')}")
        print()
    return matched


async def pull_feishu_chat_messages(token):
    print("\n" + "=" * 60)
    print("[3/3] 拉取飞书群组消息")
    print("=" * 60)
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {"container_id_type": "chat", "container_id": CHAT_ID, "page_size": 50}
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(url, params=params, headers=headers, timeout=30)
        data = r.json()
    if data.get("code") != 0:
        print(f"  ❌ API错误: code={data.get('code')} msg={data.get('msg')}")
        return []
    items = data.get("data", {}).get("items", [])
    print(f"  消息数: {len(items)}")
    for i, msg in enumerate(items[:15]):
        msg_id = msg.get("message_id", "")
        msg_type = msg.get("msg_type", "")
        ct_ms = msg.get("create_time", 0)
        try:
            ct_str = datetime.fromtimestamp(int(ct_ms) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        except:
            ct_str = str(ct_ms)
        sender_type = msg.get("sender", {}).get("sender_type", "")
        content = msg.get("body", {}).get("content", "")[:60]
        icon = "🤖" if sender_type == "app" else "👤"
        print(f"  {i+1}. {icon} [{ct_str}] {msg_type} | {content}")
    return items


async def main():
    print("三方日志拉取工具")
    print(f"Chat ID: {CHAT_ID}")
    print(f"Session ID: {SESSION_ID}")
    print()
    cf_logs = await pull_cf_logs()
    wf_instances = await pull_cf_workflow_instances()
    token = await get_feishu_token()
    if token:
        feishu_msgs = await pull_feishu_chat_messages(token)
    else:
        print("❌ 无法获取飞书token")
        feishu_msgs = []
    print("\n" + "=" * 60)
    print("综合分析")
    print("=" * 60)
    print(f"  CF相关事件: {len(cf_logs)}")
    print(f"  WF匹配实例: {len(wf_instances)}")
    print(f"  飞书消息数: {len(feishu_msgs)}")
    bot_msgs = [m for m in feishu_msgs if m.get("sender", {}).get("sender_type") == "app"]
    user_msgs = [m for m in feishu_msgs if m.get("sender", {}).get("sender_type") != "app"]
    print(f"  机器人消息: {len(bot_msgs)}")
    print(f"  用户消息: {len(user_msgs)}")
    if not bot_msgs:
        print("\n  ⚠️ 没有找到机器人发送的消息！")
        print("  这意味着回复消息没有成功发送到飞书群组")


if __name__ == "__main__":
    asyncio.run(main())
