"""拉取飞书+CF+Modal三方日志"""
import asyncio
import json
import os
import time
from datetime import datetime
import httpx

CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "d1215a30b84b673ef0367010b0e78c10")
CHAT_ID = "oc_33edcba53ad086b50f175352947750eb"


async def get_tenant_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    async with httpx.AsyncClient() as c:
        r = await c.post(url, json={"app_id": os.getenv("FEISHU_APP_ID3", ""), "app_secret": os.getenv("FEISHU_APP_SECRET3", "")}, timeout=30)
        return r.json().get("tenant_access_token", "")


async def pull_cf_logs():
    print("\n" + "=" * 60)
    print("[1] Cloudflare Worker 日志")
    print("=" * 60)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 30 * 60 * 1000  # 最近30分钟
    body = {
        "queryId": "feishu-trace",
        "view": "events",
        "limit": 100,
        "timeframe": {"from": start_ms, "to": end_ms},
        "parameters": {
            "filterCombination": "and",
            "filters": [
                {"kind": "filter", "key": "$workers.scriptName", "operation": "eq", "type": "string", "value": "hermes-feishu-gateway"}
            ],
            "orderBy": {"value": "$metadata.id", "order": "desc"},
        },
    }
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/workflows/hermes-feishu-agent-workflow/instances"
    async with httpx.AsyncClient() as c:
        r = await c.get(url, headers={"Authorization": f"Bearer {CF_API_TOKEN}"}, timeout=30)
        wf_data = r.json() if r.status_code == 200 else {}
    
    print(f"  Workflow实例数: {len(wf_data.get('result', []))}")
    
    # 获取最近的实例
    instances = wf_data.get("result", [])
    for inst in instances[:3]:
        inst_id = inst.get("id", "")
        print(f"\n  实例: {inst_id}")
        print(f"  状态: {inst.get('status')}")
        print(f"  创建: {inst.get('created_at')}")
        
        # 获取实例详情
        detail_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/workflows/hermes-feishu-agent-workflow/instances/{inst_id}"
        async with httpx.AsyncClient() as c:
            r = await c.get(detail_url, headers={"Authorization": f"Bearer {CF_API_TOKEN}"}, timeout=30)
            if r.status_code == 200:
                detail = r.json().get("result", {})
                steps = detail.get("steps", [])
                for step in steps:
                    step_name = step.get("name", "N/A")
                    step_status = step.get("status", "")
                    icon = "OK" if step_status == "completed" else "ERR" if step_status == "errored" else "..."
                    print(f"    {icon} {step_name}")
                    if step.get("error"):
                        print(f"       Error: {str(step.get('error'))[:100]}")
    
    return instances


async def pull_feishu_messages(token):
    print("\n" + "=" * 60)
    print("[2] 飞书群组消息")
    print("=" * 60)
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {"container_id_type": "chat", "container_id": CHAT_ID, "page_size": 20}
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(url, params=params, headers=headers, timeout=30)
        data = r.json()
    
    if data.get("code") != 0:
        print(f"  API错误: {data.get('msg')}")
        return []
    
    items = data.get("data", {}).get("items", [])
    print(f"  消息数: {len(items)}")
    
    for i, msg in enumerate(items[:10]):
        msg_id = msg.get("message_id", "")
        sender_type = msg.get("sender", {}).get("sender_type", "")
        ct_ms = msg.get("create_time", 0)
        try:
            ct_str = datetime.fromtimestamp(int(ct_ms) / 1000).strftime("%m-%d %H:%M:%S")
        except:
            ct_str = str(ct_ms)
        content = msg.get("body", {}).get("content", "")[:50]
        icon = "BOT" if sender_type == "app" else "USER"
        print(f"  {icon} [{ct_str}] {content}")
    
    return items


async def main():
    print("=" * 60)
    print("三方日志分析 - 最近30分钟")
    print("=" * 60)
    
    # CF日志
    await pull_cf_logs()
    
    # 飞书消息
    token = await get_tenant_token()
    if token:
        await pull_feishu_messages(token)
    else:
        print("\n无法获取飞书token")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
