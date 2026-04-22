"""
简化版飞书机器人测试 - 使用应用身份调用API

由于用户身份权限需要审核，我们使用应用身份(tenant_access_token)调用API
这种方式可以免审核直接调试
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

import aiohttp

TEST_CHAT_ID = os.getenv("FEISHU_TEST_CHAT_ID") or os.getenv("FEISHU_HOME_CHANNEL") or ""


async def get_tenant_token(app_id: str, app_secret: str) -> str:
    """获取应用级别的tenant_access_token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json"}
    data = {"app_id": app_id, "app_secret": app_secret}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            result = await resp.json()
            if result.get("code") == 0:
                return result["tenant_access_token"]
            raise RuntimeError(f"获取token失败: {result}")


async def send_message_as_bot(token: str, chat_id: str, message: str) -> dict:
    """以应用身份发送消息到群组"""
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": message})
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            return await resp.json()


async def main():
    print("=" * 60)
    print("飞书机器人测试 - 应用身份模式")
    print("=" * 60)
    
    # 使用已配置的环境变量
    app_id = os.getenv("FEISHU_APP_ID3")
    app_secret = os.getenv("FEISHU_APP_SECRET3")
    chat_id = TEST_CHAT_ID
    
    if not app_id or not app_secret or not chat_id:
        print("错误: 缺少 FEISHU_APP_ID3 / FEISHU_APP_SECRET3 / FEISHU_TEST_CHAT_ID(或 FEISHU_HOME_CHANNEL)")
        sys.exit(1)
    
    print(f"App ID: {app_id}")
    print(f"Chat ID: {chat_id}")
    print()
    
    try:
        # 获取tenant token
        print("获取tenant_access_token...")
        token = await get_tenant_token(app_id, app_secret)
        print(f"✓ Token获取成功: {token[:20]}...")
        print()
        
        # 测试消息列表
        test_messages = [
            ("基础问候", "你好，这是一条测试消息"),
            ("浏览器任务", "请打开浏览器访问 https://www.baidu.com"),
            ("代码任务", "写一个Python函数计算1到100的和"),
            ("知识问答", "什么是人工智能？"),
        ]
        
        results = []
        
        for test_name, message in test_messages:
            print(f"[{test_name}] 发送: {message[:40]}...")
            
            start_time = time.time()
            result = await send_message_as_bot(token, chat_id, message)
            duration = (time.time() - start_time) * 1000
            
            if result.get("code") == 0:
                print(f"  ✓ 成功 (耗时: {duration:.0f}ms)")
                print(f"    Message ID: {result['data']['message_id']}")
                results.append({
                    "test": test_name,
                    "success": True,
                    "duration_ms": duration,
                    "message_id": result['data']['message_id']
                })
            else:
                print(f"  ✗ 失败: {result.get('msg')}")
                results.append({
                    "test": test_name,
                    "success": False,
                    "error": result.get('msg'),
                    "code": result.get('code')
                })
            
            print()
            await asyncio.sleep(1)  # 避免发送太快
        
        # 生成报告
        print("=" * 60)
        print("测试报告")
        print("=" * 60)
        
        success_count = sum(1 for r in results if r['success'])
        print(f"总测试: {len(results)}")
        print(f"成功: {success_count}")
        print(f"失败: {len(results) - success_count}")
        print()
        
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"{status} {r['test']}: {r.get('duration_ms', 0):.0f}ms")
        
        print()
        print("=" * 60)
        print("说明:")
        print("- 消息已通过应用身份发送到飞书群组")
        print("- 请检查飞书群组查看消息")
        print("- 如果群组中有Hermes机器人，它会自动回复")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
