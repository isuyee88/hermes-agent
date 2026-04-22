"""发送测试消息到飞书群组，验证机器人是否回复"""
import asyncio
import os
import httpx

FEISHU_APP_ID = os.getenv("FEISHU_APP_ID3", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET3", "")
CHAT_ID = "oc_33edcba53ad086b50f175352947750eb"


async def get_tenant_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    async with httpx.AsyncClient() as c:
        r = await c.post(url, json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}, timeout=30)
        return r.json().get("tenant_access_token", "")


async def send_test_message(token):
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {"receive_id_type": "chat_id"}

    # 发送测试消息
    message = {
        "receive_id": CHAT_ID,
        "msg_type": "text",
        "content": {"text": "TEST: Robot reply check " + __import__('time').strftime("%H:%M:%S")}
    }

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as c:
        r = await c.post(url, params=params, json=message, headers=headers, timeout=30)
        return r.json()


async def get_recent_messages(token, limit=10):
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {"container_id_type": "chat", "container_id": CHAT_ID, "page_size": limit}
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(url, params=params, headers=headers, timeout=30)
        return r.json()


async def main():
    print("=" * 60)
    print("发送测试消息并检查回复")
    print("=" * 60)

    token = await get_tenant_token()
    if not token:
        print("无法获取token")
        return

    print(f"Token获取成功: {token[:20]}...")

    # 发送测试消息
    result = await send_test_message(token)
    if result.get("code") == 0:
        print(f"消息发送成功: {result.get('data', {}).get('message_id', 'N/A')}")
    else:
        print(f"消息发送失败: code={result.get('code')} msg={result.get('msg')}")

    # 等待几秒让消息处理
    print("\n等待5秒让消息处理...")
    await asyncio.sleep(5)

    # 获取最近消息
    print("\n获取最近消息:")
    msgs = await get_recent_messages(token, limit=5)
    items = msgs.get("data", {}).get("items", [])

    for i, msg in enumerate(items[:5]):
        sender_type = msg.get("sender", {}).get("sender_type", "")
        content = msg.get("body", {}).get("content", "")[:60]
        ct = msg.get("create_time", "0")
        from datetime import datetime
        try:
            ct_str = datetime.fromtimestamp(int(ct)/1000).strftime("%H:%M:%S")
        except:
            ct_str = ct
        icon = "BOT" if sender_type == "app" else "USER"
        print(f"  {icon} [{ct_str}] {content}")


if __name__ == "__main__":
    asyncio.run(main())
