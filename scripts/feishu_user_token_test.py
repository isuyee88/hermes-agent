"""
使用用户访问令牌驱动机器人工作 - 真实测试

通过用户身份发送消息给机器人，触发机器人的AI处理流程
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

import aiohttp


class FeishuUserDriver:
    """使用用户令牌驱动机器人"""
    
    def __init__(self, user_access_token: str, bot_chat_id: str):
        self.user_access_token = user_access_token
        self.bot_chat_id = bot_chat_id
        self.results = []
        
    async def send_message_to_bot(self, message: str, msg_type: str = "text") -> dict:
        """以用户身份发送消息给机器人"""
        url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
        headers = {
            "Authorization": f"Bearer {self.user_access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "receive_id": self.bot_chat_id,
            "msg_type": msg_type,
            "content": json.dumps({"text": message}) if msg_type == "text" else message
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 发送消息: {message[:50]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as resp:
                result = await resp.json()
                return result
    
    async def get_messages(self, container_id: str, page_size: int = 20) -> list:
        """获取会话消息列表"""
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {"Authorization": f"Bearer {self.user_access_token}"}
        params = {
            "container_id_type": "chat",
            "container_id": container_id,
            "page_size": page_size
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                result = await resp.json()
                if result.get("code") == 0:
                    return result.get("data", {}).get("items", [])
                return []
    
    async def wait_for_bot_reply(self, sent_message_id: str, timeout: int = 60) -> dict | None:
        """等待机器人回复"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待机器人回复 (最多{timeout}秒)...")
        
        start_time = time.time()
        check_count = 0
        
        while time.time() - start_time < timeout:
            await asyncio.sleep(2)
            check_count += 1
            
            messages = await self.get_messages(self.bot_chat_id, page_size=10)
            
            # 查找机器人回复（非用户发送的消息）
            for msg in messages:
                msg_id = msg.get("message_id")
                sender_type = msg.get("sender", {}).get("sender_type")
                
                # 跳过我们发送的消息
                if msg_id == sent_message_id:
                    continue
                
                # 找到机器人回复
                if sender_type == "app":  # 机器人类型
                    create_time = int(msg.get("create_time", 0)) / 1000
                    if create_time > start_time:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 收到机器人回复")
                        return msg
            
            if check_count % 5 == 0:
                print(f"  已等待 {int(time.time() - start_time)} 秒...")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 等待超时，未收到回复")
        return None
    
    async def test_scenario(self, test_name: str, message: str, expected_type: str) -> dict:
        """测试场景"""
        print(f"\n{'='*60}")
        print(f"测试: {test_name}")
        print(f"{'='*60}")
        
        result = {
            "test_name": test_name,
            "message": message,
            "expected_type": expected_type,
            "start_time": time.time(),
            "success": False
        }
        
        try:
            # T0: 发送消息
            send_result = await self.send_message_to_bot(message)
            
            if send_result.get("code") != 0:
                result["error"] = f"发送失败: {send_result}"
                result["end_time"] = time.time()
                return result
            
            sent_message_id = send_result["data"]["message_id"]
            result["message_id"] = sent_message_id
            result["send_time"] = time.time()
            
            # T1-T4: 等待机器人回复
            bot_reply = await self.wait_for_bot_reply(sent_message_id, timeout=60)
            result["end_time"] = time.time()
            
            if bot_reply:
                result["success"] = True
                result["bot_reply"] = {
                    "message_id": bot_reply.get("message_id"),
                    "msg_type": bot_reply.get("msg_type"),
                    "content": bot_reply.get("body", {}).get("content", "")[:200]
                }
                result["total_latency_ms"] = (result["end_time"] - result["start_time"]) * 1000
            else:
                result["error"] = "等待机器人回复超时"
                
        except Exception as e:
            result["error"] = str(e)
            result["end_time"] = time.time()
            import traceback
            traceback.print_exc()
        
        return result
    
    async def run_all_tests(self):
        """运行所有测试场景"""
        print("=" * 60)
        print("飞书机器人真实工作测试")
        print("=" * 60)
        print(f"目标Chat ID: {self.bot_chat_id}")
        print(f"开始时间: {datetime.now().isoformat()}")
        print()
        
        # 测试场景1: 纯文本问候
        result1 = await self.test_scenario(
            "纯文本问候",
            "你好，请介绍一下你自己",
            "text_plain"
        )
        self.results.append(result1)
        
        # 测试场景2: 浏览器任务
        result2 = await self.test_scenario(
            "浏览器任务",
            "打开浏览器访问 https://github.com/microsoft/vscode 查看最近的issues",
            "browser_heavy"
        )
        self.results.append(result2)
        
        # 测试场景3: 代码任务
        result3 = await self.test_scenario(
            "代码任务",
            "帮我写一个Python函数，计算斐波那契数列的前n项",
            "text_coding"
        )
        self.results.append(result3)
        
        # 测试场景4: 知识问答
        result4 = await self.test_scenario(
            "知识问答",
            "什么是微服务架构？有什么优缺点？",
            "text_general"
        )
        self.results.append(result4)
        
        return self.results
    
    def generate_report(self) -> str:
        """生成测试报告"""
        lines = [
            "\n" + "=" * 60,
            "测试报告",
            "=" * 60,
            f"总测试数: {len(self.results)}",
            f"成功数: {sum(1 for r in self.results if r['success'])}",
            f"失败数: {sum(1 for r in self.results if not r['success'])}",
            ""
        ]
        
        for i, result in enumerate(self.results, 1):
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            lines.append(f"\n[{i}] {status} {result['test_name']}")
            lines.append(f"    消息: {result['message'][:60]}...")
            lines.append(f"    预期类型: {result['expected_type']}")
            
            if result.get("total_latency_ms"):
                lines.append(f"    总时延: {result['total_latency_ms']:.0f}ms")
            
            if result.get("bot_reply"):
                reply_content = result["bot_reply"]["content"][:100]
                lines.append(f"    机器人回复: {reply_content}...")
            
            if result.get("error"):
                lines.append(f"    错误: {result['error']}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


async def main():
    # 从环境变量获取用户访问令牌
    user_access_token = os.getenv("FEISHU_USER_ACCESS_TOKEN")
    
    if not user_access_token:
        print("错误: 需要设置 FEISHU_USER_ACCESS_TOKEN 环境变量")
        print()
        print("获取用户访问令牌的方法:")
        print("1. 访问飞书开放平台: https://open.feishu.cn/app/")
        print("2. 进入你的应用 -> 凭证与基础信息")
        print("3. 使用'获取 user_access_token'功能")
        print("4. 或者通过OAuth2流程获取")
        print()
        print("设置环境变量:")
        print("  Windows: set FEISHU_USER_ACCESS_TOKEN=your_token")
        print("  Mac/Linux: export FEISHU_USER_ACCESS_TOKEN=your_token")
        sys.exit(1)
    
    # 使用之前发现的群组
    bot_chat_id = "oc_ec86c28e66596c25377aff2ee028901c"
    
    driver = FeishuUserDriver(user_access_token, bot_chat_id)
    
    try:
        await driver.run_all_tests()
        report = driver.generate_report()
        print(report)
        
        # 保存详细结果
        result_file = f"feishu_bot_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(driver.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
