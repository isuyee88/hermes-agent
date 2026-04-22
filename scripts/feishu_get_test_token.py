"""
飞书测试令牌获取 - 通过开放平台开发者工具

无需配置回调URL，直接使用应用凭证获取测试用的 user_access_token
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import aiohttp


# 飞书应用配置
APP_ID = os.getenv("FEISHU_APP_ID3", "cli_a9525a47e4f99bc2c")
APP_SECRET = os.getenv("FEISHU_APP_SECRET3", "")


async def get_tenant_token() -> str:
    """获取 tenant_access_token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    
    data = {
        "app_id": APP_ID,
        "app_secret": APP_SECRET
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            result = await resp.json()
            if result.get("code") == 0:
                return result["tenant_access_token"]
            raise Exception(f"获取 tenant_token 失败: {result}")


async def get_user_access_token_direct(user_id: str = None) -> dict:
    """
    通过应用身份直接获取用户访问令牌（仅用于测试）
    
    注意：此方法需要应用有相应的权限，且仅适用于测试企业
    """
    tenant_token = await get_tenant_token()
    
    # 方法1: 使用 identity/access_token 接口
    url = "https://open.feishu.cn/open-apis/auth/v3/identity/access_token"
    
    headers = {
        "Authorization": f"Bearer {tenant_token}",
        "Content-Type": "application/json"
    }
    
    # 如果不指定用户ID，则获取应用自己的令牌
    data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:identity",
        "scope": "im:message im:message:send"
    }
    
    if user_id:
        data["user_id"] = user_id
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            result = await resp.json()
            return result


async def send_message_as_bot(tenant_token: str, chat_id: str, message: str) -> dict:
    """使用 tenant_access_token (应用身份) 发送消息"""
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    
    headers = {
        "Authorization": f"Bearer {tenant_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": message})
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            result = await resp.json()
            return result


async def get_bot_info(tenant_token: str) -> dict:
    """获取机器人信息"""
    url = "https://open.feishu.cn/open-apis/bot/v3/info"
    
    headers = {
        "Authorization": f"Bearer {tenant_token}"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            result = await resp.json()
            return result


async def get_chat_members(tenant_token: str, chat_id: str) -> list:
    """获取群成员列表"""
    url = f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}/members"
    
    headers = {
        "Authorization": f"Bearer {tenant_token}"
    }
    
    params = {
        "member_id_type": "open_id"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as resp:
            result = await resp.json()
            if result.get("code") == 0:
                return result.get("data", {}).get("items", [])
            return []


async def main():
    print("=" * 60)
    print("飞书测试令牌获取工具")
    print("=" * 60)
    print()
    
    if not APP_SECRET:
        print("❌ 错误: 需要设置 FEISHU_APP_SECRET3 环境变量")
        print()
        print("设置方法:")
        print("  $env:FEISHU_APP_SECRET3=\"your_secret\"")
        sys.exit(1)
    
    print(f"应用 ID: {APP_ID}")
    print()
    
    # 获取 tenant_access_token
    print("🔄 正在获取 tenant_access_token...")
    try:
        tenant_token = await get_tenant_token()
        print(f"✅ 获取成功: {tenant_token[:30]}...")
        print()
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        sys.exit(1)
    
    # 获取机器人信息
    print("🤖 正在获取机器人信息...")
    bot_info = await get_bot_info(tenant_token)
    if bot_info.get("code") == 0:
        bot_data = bot_info.get("data", {})
        print(f"   机器人名称: {bot_data.get('bot_name')}")
        print(f"   机器人 ID: {bot_data.get('open_id', 'N/A')[:20]}...")
        print()
    else:
        print(f"   获取失败: {bot_info}")
        print()
    
    # 测试群组
    chat_id = "oc_ec86c28e66596c25377aff2ee028901c"
    
    # 获取群成员
    print(f"👥 正在获取群成员列表 (Chat ID: {chat_id[:20]}...)...")
    members = await get_chat_members(tenant_token, chat_id)
    if members:
        print(f"   群成员数: {len(members)}")
        for i, member in enumerate(members[:3], 1):
            member_id = member.get("member_id", "N/A")
            print(f"   成员 {i}: {member_id[:20]}...")
        if len(members) > 3:
            print(f"   ... 还有 {len(members) - 3} 个成员")
    else:
        print("   无法获取群成员列表（可能需要权限）")
    print()
    
    # 发送测试消息
    print("📤 正在发送测试消息...")
    test_message = f"🧪 测试消息 - 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    result = await send_message_as_bot(tenant_token, chat_id, test_message)
    
    if result.get("code") == 0:
        print("✅ 测试消息发送成功！")
        print()
        print("=" * 60)
        print("说明:")
        print("=" * 60)
        print()
        print("由于 OAuth 回调 URL 需要预先配置，我们使用应用身份")
        print("(tenant_access_token) 来发送消息。")
        print()
        print("如果您需要以用户身份发送消息，请:")
        print("1. 访问飞书开放平台: https://open.feishu.cn/app/")
        print("2. 进入您的应用 -> 安全设置")
        print("3. 添加重定向 URL: http://localhost:8080/callback")
        print("4. 然后运行: python scripts/feishu_oauth_flow.py")
        print()
        print("当前 tenant_access_token 可以直接用于:")
        print("  - 发送应用消息")
        print("  - 获取群信息")
        print("  - 机器人相关操作")
        print()
        print("环境变量:")
        print("-" * 60)
        print(f"$env:FEISHU_TENANT_TOKEN=\"{tenant_token}\"")
        print("-" * 60)
        
        # 保存令牌
        token_file = f"feishu_tenant_token_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(f"Tenant Token: {tenant_token}\n")
            f.write(f"Created at: {datetime.now().isoformat()}\n")
            f.write(f"Expires in: 2 hours\n")
        print()
        print(f"💾 令牌已保存到: {token_file}")
        
    else:
        print(f"❌ 发送失败: {result}")
        print()
        print("可能的原因:")
        print("  1. 机器人不在该群组中")
        print("  2. 应用没有 im:message:send 权限")
        print("  3. 群组 ID 不正确")
        print()
        print("建议:")
        print("  1. 确认机器人已添加到群组 'CEO 工作汇报'")
        print("  2. 在开放平台检查应用权限")


if __name__ == "__main__":
    asyncio.run(main())
