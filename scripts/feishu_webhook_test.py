"""
使用飞书Webhook驱动机器人工作 - 真实测试

通过Webhook发送消息到群组，触发机器人的AI处理流程
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

import aiohttp


class FeishuWebhookDriver:
    """使用Webhook驱动机器人"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.results = []
        
    async def send_message(self, message: str) -> dict:
        """通过Webhook发送消息"""
        headers = {"Content-Type": "application/json"}
        data = {
            "msg_type": "text",
            "content": {
                "text": message
            }
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 发送消息: {message[:50]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, headers=headers, json=data) as resp:
                result = await resp.json()
                return result
    
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
            # 发送消息
            send_result = await self.send_message(message)
            
            if send_result.get("code") == 0:
                result["success"] = True
                result["response"] = send_result
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 消息发送成功")
            else:
                result["error"] = f"发送失败: {send_result}"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 发送失败: {send_result}")
                
        except Exception as e:
            result["error"] = str(e)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
        
        result["end_time"] = time.time()
        result["duration_ms"] = (result["end_time"] - result["start_time"]) * 1000
        
        return result
    
    async def run_all_tests(self):
        """运行所有测试场景"""
        print("=" * 60)
        print("飞书Webhook机器人测试")
        print("=" * 60)
        print(f"Webhook: {self.webhook_url[:50]}...")
        print(f"开始时间: {datetime.now().isoformat()}")
        print()
        
        # 测试场景1: 纯文本问候
        result1 = await self.test_scenario(
            "纯文本问候",
            "你好，请介绍一下你自己",
            "text_plain"
        )
        self.results.append(result1)
        
        # 等待一段时间，避免消息太快
        await asyncio.sleep(2)
        
        # 测试场景2: 浏览器任务
        result2 = await self.test_scenario(
            "浏览器任务",
            "打开浏览器访问 https://github.com/microsoft/vscode 查看最近的issues",
            "browser_heavy"
        )
        self.results.append(result2)
        
        await asyncio.sleep(2)
        
        # 测试场景3: 代码任务
        result3 = await self.test_scenario(
            "代码任务",
            "帮我写一个Python函数，计算斐波那契数列的前n项",
            "text_coding"
        )
        self.results.append(result3)
        
        await asyncio.sleep(2)
        
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
            lines.append(f"    时延: {result.get('duration_ms', 0):.0f}ms")
            
            if result.get("error"):
                lines.append(f"    错误: {result['error']}")
        
        lines.append("\n" + "=" * 60)
        lines.append("\n说明:")
        lines.append("- 消息已通过Webhook发送到飞书群组")
        lines.append("- 请检查飞书群组查看机器人的实际回复")
        lines.append("- 观察机器人的响应时延和回复质量")
        lines.append("=" * 60)
        
        return "\n".join(lines)


async def main():
    # 从环境变量获取Webhook URL
    webhook_url = os.getenv("FEISHU_WEBHOOK_URL")
    
    if not webhook_url:
        print("错误: 需要设置 FEISHU_WEBHOOK_URL 环境变量")
        print()
        print("获取Webhook URL的方法:")
        print("1. 在飞书群组中，点击群组设置 -> 群机器人")
        print("2. 添加自定义机器人")
        print("3. 复制Webhook地址")
        print()
        print("设置环境变量:")
        print("  Windows: $env:FEISHU_WEBHOOK_URL='https://open.feishu.cn/...'")
        print("  Mac/Linux: export FEISHU_WEBHOOK_URL='https://open.feishu.cn/...'")
        sys.exit(1)
    
    driver = FeishuWebhookDriver(webhook_url)
    
    try:
        await driver.run_all_tests()
        report = driver.generate_report()
        print(report)
        
        # 保存详细结果
        result_file = f"feishu_webhook_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(driver.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
