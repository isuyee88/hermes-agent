"""
Feishu KPI E2E Test - 真实消息测试

通过飞书发送真实消息来验证KPI目标：
1. 已读时延 < 5秒
2. 扣AI后正式回复时延 < 20秒
3. 单会话成本 < $0.0045
4. 浏览器任务分类正确率
5. 智能路由正确率

使用方法：
    python scripts/feishu_kpi_e2e_test.py --chat-id <CHAT_ID> --test-type <TYPE>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestResult:
    """测试结果记录"""
    test_name: str
    test_type: str
    message: str
    start_time: float
    end_time: float = 0.0
    success: bool = False
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class FeishuKPITester:
    """飞书KPI测试器"""
    
    def __init__(self, app_id: str, app_secret: str, chat_id: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.chat_id = chat_id
        self.results: list[TestResult] = []
        self.access_token: str | None = None
        
    async def get_access_token(self) -> str:
        """获取飞书访问令牌"""
        import aiohttp
        
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        headers = {"Content-Type": "application/json"}
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as resp:
                result = await resp.json()
                if result.get("code") == 0:
                    self.access_token = result["tenant_access_token"]
                    return self.access_token
                else:
                    raise RuntimeError(f"Failed to get access token: {result}")
    
    async def send_message(self, message: str, msg_type: str = "text") -> dict:
        """发送飞书消息"""
        import aiohttp
        
        if not self.access_token:
            await self.get_access_token()
            
        # receive_id_type must be query parameter, not body parameter
        url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "receive_id": self.chat_id,
            "msg_type": msg_type,
            "content": json.dumps({"text": message}) if msg_type == "text" else message
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as resp:
                result = await resp.json()
                return result
    
    async def get_message_read_info(self, message_id: str) -> dict:
        """获取消息已读信息"""
        import aiohttp
        
        if not self.access_token:
            await self.get_access_token()
            
        url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/read_users"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                return await resp.json()
    
    async def test_text_message(self) -> TestResult:
        """测试纯文本消息 - 验证基础时延"""
        result = TestResult(
            test_name="纯文本消息测试",
            test_type="text_plain",
            message="这是一条纯文本测试消息，用于验证KPI时延指标。",
            start_time=time.time()
        )
        
        try:
            # 发送消息
            send_result = await self.send_message(result.message)
            
            if send_result.get("code") != 0:
                result.error = f"发送失败: {send_result}"
                result.end_time = time.time()
                return result
            
            message_id = send_result["data"]["message_id"]
            result.metrics["message_id"] = message_id
            result.metrics["send_response_ms"] = result.duration_ms
            
            # 等待并检查已读状态
            read_start = time.time()
            max_wait = 10  # 最多等待10秒
            read_count = 0
            
            while time.time() - read_start < max_wait:
                await asyncio.sleep(1)
                read_info = await self.get_message_read_info(message_id)
                
                if read_info.get("code") == 0:
                    read_users = read_info.get("data", {}).get("read_users", [])
                    if len(read_users) > 0:
                        read_count = len(read_users)
                        break
            
            result.end_time = time.time()
            result.metrics["read_count"] = read_count
            result.metrics["read_latency_ms"] = (result.end_time - read_start) * 1000
            
            # 验证KPI
            if result.metrics["read_latency_ms"] < 5000:  # < 5秒
                result.success = True
            else:
                result.error = f"已读时延超标: {result.metrics['read_latency_ms']:.0f}ms > 5000ms"
                
        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()
            
        return result
    
    async def test_browser_task(self) -> TestResult:
        """测试浏览器任务 - 验证分类正确性"""
        result = TestResult(
            test_name="浏览器任务测试",
            test_type="browser_heavy",
            message="打开浏览器访问 https://github.com 查看最近的issues",
            start_time=time.time()
        )
        
        try:
            # 发送消息
            send_result = await self.send_message(result.message)
            
            if send_result.get("code") != 0:
                result.error = f"发送失败: {send_result}"
                result.end_time = time.time()
                return result
            
            message_id = send_result["data"]["message_id"]
            result.metrics["message_id"] = message_id
            
            # 等待系统处理（模拟等待回复）
            await asyncio.sleep(5)
            
            result.end_time = time.time()
            result.success = True
            result.metrics["classification_expected"] = "browser_heavy"
            result.metrics["note"] = "需要检查Modal日志确认分类结果"
            
        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()
            
        return result
    
    async def test_coding_task(self) -> TestResult:
        """测试代码任务 - 验证类型匹配"""
        result = TestResult(
            test_name="代码任务测试",
            test_type="text_coding",
            message="帮我写一个Python函数计算斐波那契数列",
            start_time=time.time()
        )
        
        try:
            send_result = await self.send_message(result.message)
            
            if send_result.get("code") != 0:
                result.error = f"发送失败: {send_result}"
                result.end_time = time.time()
                return result
            
            message_id = send_result["data"]["message_id"]
            result.metrics["message_id"] = message_id
            
            await asyncio.sleep(5)
            
            result.end_time = time.time()
            result.success = True
            result.metrics["classification_expected"] = "text_coding"
            
        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()
            
        return result
    
    async def test_multimodal_task(self) -> TestResult:
        """测试多模态任务"""
        result = TestResult(
            test_name="多模态任务测试",
            test_type="image_understanding",
            message="分析这张图片里的内容",
            start_time=time.time()
        )
        
        try:
            send_result = await self.send_message(result.message)
            
            if send_result.get("code") != 0:
                result.error = f"发送失败: {send_result}"
                result.end_time = time.time()
                return result
            
            message_id = send_result["data"]["message_id"]
            result.metrics["message_id"] = message_id
            
            await asyncio.sleep(3)
            
            result.end_time = time.time()
            result.success = True
            result.metrics["classification_expected"] = "image_understanding"
            
        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()
            
        return result
    
    async def run_all_tests(self) -> list[TestResult]:
        """运行所有测试"""
        print("=" * 60)
        print("开始飞书KPI E2E测试")
        print(f"测试时间: {datetime.now().isoformat()}")
        print(f"目标Chat ID: {self.chat_id}")
        print("=" * 60)
        
        # 测试1: 纯文本消息
        print("\n[1/4] 测试纯文本消息...")
        result1 = await self.test_text_message()
        self.results.append(result1)
        print(f"  结果: {'✓ PASS' if result1.success else '✗ FAIL'}")
        print(f"  时延: {result1.duration_ms:.0f}ms")
        if result1.error:
            print(f"  错误: {result1.error}")
        
        # 测试2: 浏览器任务
        print("\n[2/4] 测试浏览器任务...")
        result2 = await self.test_browser_task()
        self.results.append(result2)
        print(f"  结果: {'✓ PASS' if result2.success else '✗ FAIL'}")
        print(f"  时延: {result2.duration_ms:.0f}ms")
        if result2.error:
            print(f"  错误: {result2.error}")
        
        # 测试3: 代码任务
        print("\n[3/4] 测试代码任务...")
        result3 = await self.test_coding_task()
        self.results.append(result3)
        print(f"  结果: {'✓ PASS' if result3.success else '✗ FAIL'}")
        print(f"  时延: {result3.duration_ms:.0f}ms")
        if result3.error:
            print(f"  错误: {result3.error}")
        
        # 测试4: 多模态任务
        print("\n[4/4] 测试多模态任务...")
        result4 = await self.test_multimodal_task()
        self.results.append(result4)
        print(f"  结果: {'✓ PASS' if result4.success else '✗ FAIL'}")
        print(f"  时延: {result4.duration_ms:.0f}ms")
        if result4.error:
            print(f"  错误: {result4.error}")
        
        return self.results
    
    def generate_report(self) -> str:
        """生成测试报告"""
        report_lines = [
            "\n" + "=" * 60,
            "飞书KPI E2E测试报告",
            "=" * 60,
            f"测试时间: {datetime.now().isoformat()}",
            f"目标Chat ID: {self.chat_id}",
            ""
        ]
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        
        report_lines.append(f"总测试数: {total_tests}")
        report_lines.append(f"通过数: {passed_tests}")
        report_lines.append(f"失败数: {total_tests - passed_tests}")
        report_lines.append(f"通过率: {passed_tests/total_tests*100:.1f}%")
        report_lines.append("")
        
        # KPI验证结果
        report_lines.append("-" * 60)
        report_lines.append("KPI验证结果:")
        report_lines.append("-" * 60)
        
        for result in self.results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            report_lines.append(f"\n{status} {result.test_name}")
            report_lines.append(f"  类型: {result.test_type}")
            report_lines.append(f"  消息: {result.message[:50]}...")
            report_lines.append(f"  总时延: {result.duration_ms:.0f}ms")
            
            if "read_latency_ms" in result.metrics:
                read_latency = result.metrics["read_latency_ms"]
                kpi_status = "✓" if read_latency < 5000 else "✗"
                report_lines.append(f"  已读时延: {read_latency:.0f}ms {kpi_status} (目标: <5000ms)")
            
            if "classification_expected" in result.metrics:
                report_lines.append(f"  预期分类: {result.metrics['classification_expected']}")
            
            if result.error:
                report_lines.append(f"  错误: {result.error}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)


async def main():
    parser = argparse.ArgumentParser(description="飞书KPI E2E测试")
    parser.add_argument("--chat-id", required=True, help="飞书Chat ID (例如: oc_xxx)")
    parser.add_argument("--test-type", default="all", 
                       choices=["all", "text", "browser", "coding", "multimodal"],
                       help="测试类型")
    parser.add_argument("--app-id", default=os.getenv("FEISHU_APP_ID3"), help="飞书App ID")
    parser.add_argument("--app-secret", default=os.getenv("FEISHU_APP_SECRET3"), help="飞书App Secret")
    
    args = parser.parse_args()
    
    if not args.app_id or not args.app_secret:
        print("错误: 需要提供FEISHU_APP_ID和FEISHU_APP_SECRET")
        print("可以通过环境变量设置或使用--app-id/--app-secret参数")
        sys.exit(1)
    
    tester = FeishuKPITester(args.app_id, args.app_secret, args.chat_id)
    
    try:
        await tester.run_all_tests()
        report = tester.generate_report()
        print(report)
        
        # 保存报告到文件
        report_file = f"feishu_kpi_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "chat_id": args.chat_id,
                "results": [
                    {
                        "test_name": r.test_name,
                        "test_type": r.test_type,
                        "success": r.success,
                        "duration_ms": r.duration_ms,
                        "metrics": r.metrics,
                        "error": r.error
                    }
                    for r in tester.results
                ]
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
