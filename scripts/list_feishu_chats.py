"""
获取飞书群组列表

用于查找可用的测试群组Chat ID
"""

import asyncio
import json
import os
import sys

import aiohttp


async def get_tenant_access_token(app_id: str, app_secret: str) -> str:
    """获取租户访问令牌"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json"}
    data = {
        "app_id": app_id,
        "app_secret": app_secret
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            result = await resp.json()
            if result.get("code") == 0:
                return result["tenant_access_token"]
            else:
                raise RuntimeError(f"获取token失败: {result}")


async def list_chats(token: str) -> list[dict]:
    """获取群组列表"""
    url = "https://open.feishu.cn/open-apis/im/v1/chats"
    headers = {"Authorization": f"Bearer {token}"}
    
    chats = []
    page_token = None
    
    async with aiohttp.ClientSession() as session:
        while True:
            params = {"page_size": 100}
            if page_token:
                params["page_token"] = page_token
            
            async with session.get(url, headers=headers, params=params) as resp:
                result = await resp.json()
                
                if result.get("code") != 0:
                    print(f"获取群组列表失败: {result}")
                    break
                
                items = result.get("data", {}).get("items", [])
                chats.extend(items)
                
                # 检查是否还有更多
                has_more = result.get("data", {}).get("has_more", False)
                if not has_more:
                    break
                
                page_token = result.get("data", {}).get("page_token")
    
    return chats


async def main():
    # 从环境变量获取配置
    app_id = os.getenv("FEISHU_APP_ID3")
    app_secret = os.getenv("FEISHU_APP_SECRET3")
    
    if not app_id or not app_secret:
        print("错误: 需要设置 FEISHU_APP_ID3 和 FEISHU_APP_SECRET3 环境变量")
        sys.exit(1)
    
    print("=" * 60)
    print("飞书群组列表")
    print("=" * 60)
    print(f"App ID: {app_id}")
    print()
    
    try:
        # 获取访问令牌
        print("正在获取访问令牌...")
        token = await get_tenant_access_token(app_id, app_secret)
        print("✓ 访问令牌获取成功")
        print()
        
        # 获取群组列表
        print("正在获取群组列表...")
        chats = await list_chats(token)
        
        if not chats:
            print("未找到任何群组")
            print()
            print("可能的原因：")
            print("1. 机器人尚未被添加到任何群组")
            print("2. 应用权限不足")
            print()
            print("建议操作：")
            print("1. 在飞书中创建一个测试群组")
            print("2. 将机器人添加到群组")
            print("3. 确保应用有 im:chat:readonly 权限")
            return
        
        print(f"找到 {len(chats)} 个群组：")
        print()
        
        # 显示群组信息
        for i, chat in enumerate(chats, 1):
            chat_id = chat.get("chat_id", "N/A")
            chat_name = chat.get("name", "未命名")
            chat_type = chat.get("chat_type", "unknown")
            member_count = chat.get("member_count", 0)
            
            print(f"[{i}] {chat_name}")
            print(f"    Chat ID: {chat_id}")
            print(f"    类型: {chat_type}")
            print(f"    成员数: {member_count}")
            print()
        
        print("=" * 60)
        print("使用说明")
        print("=" * 60)
        print()
        print("选择一个群组作为测试目标，记录其 Chat ID")
        print()
        print("然后执行测试：")
        print(f"  python scripts/feishu_kpi_e2e_test.py --chat-id <CHAT_ID>")
        print()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
