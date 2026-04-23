from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_CHAT_ID = os.getenv("FEISHU_TEST_CHAT_ID") or os.getenv("FEISHU_HOME_CHANNEL") or "oc_ec86c28e66596c25377aff2ee028901c"


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


def _trim(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class TestScenario:
    name: str
    message: str
    expected_type: str


DEFAULT_SCENARIOS = [
    TestScenario("纯文本问候", "你好，请介绍一下你自己", "text_plain"),
    TestScenario("浏览器任务", "打开浏览器访问 https://github.com/microsoft/vscode 查看最近的 issues", "browser_heavy"),
    TestScenario("代码任务", "帮我写一个 Python 函数，计算斐波那契数列的前 n 项", "text_coding"),
    TestScenario("知识问答", "什么是微服务架构？有什么优缺点？", "text_general"),
]


class FeishuUserDriver:
    def __init__(self, user_access_token: str, bot_chat_id: str):
        self.user_access_token = user_access_token
        self.bot_chat_id = bot_chat_id
        self.results: list[dict[str, Any]] = []

    async def send_message_to_bot(self, message: str, msg_type: str = "text") -> dict[str, Any]:
        url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
        headers = {
            "Authorization": f"Bearer {self.user_access_token}",
            "Content-Type": "application/json",
        }
        data = {
            "receive_id": self.bot_chat_id,
            "msg_type": msg_type,
            "content": json.dumps({"text": message}, ensure_ascii=False) if msg_type == "text" else message,
        }

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 发送消息: {message[:60]}...")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(url, headers=headers, json=data) as resp:
                return await resp.json()

    async def get_messages(self, container_id: str, page_size: int = 20) -> list[dict[str, Any]]:
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {"Authorization": f"Bearer {self.user_access_token}"}
        params = {
            "container_id_type": "chat",
            "container_id": container_id,
            "page_size": page_size,
            "sort_type": "ByCreateTimeDesc",
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(url, headers=headers, params=params) as resp:
                payload = await resp.json()
                if payload.get("code") == 0:
                    items = ((payload.get("data") or {}).get("items") or [])
                    return [item for item in items if isinstance(item, dict)]
                return []

    async def wait_for_bot_reply(self, sent_message_id: str, timeout: int = 60) -> dict[str, Any] | None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待机器人回复，最长 {timeout} 秒...")
        start_time = time.time()
        check_count = 0

        while time.time() - start_time < timeout:
            await asyncio.sleep(2)
            check_count += 1
            messages = await self.get_messages(self.bot_chat_id, page_size=10)
            for msg in messages:
                msg_id = _trim(msg.get("message_id"))
                sender_type = _trim((msg.get("sender") or {}).get("sender_type"))
                if not msg_id or msg_id == sent_message_id:
                    continue
                if sender_type != "app":
                    continue
                create_time = int(msg.get("create_time") or 0) / 1000
                if create_time > start_time:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 收到机器人回复。")
                    return msg
            if check_count % 5 == 0:
                print(f"  已等待 {int(time.time() - start_time)} 秒...")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 超时，未收到机器人回复。")
        return None

    async def test_scenario(self, scenario: TestScenario, timeout: int) -> dict[str, Any]:
        print("\n" + "=" * 60)
        print(f"测试: {scenario.name}")
        print("=" * 60)
        result: dict[str, Any] = {
            "test_name": scenario.name,
            "message": scenario.message,
            "expected_type": scenario.expected_type,
            "start_time": time.time(),
            "success": False,
        }

        try:
            send_result = await self.send_message_to_bot(scenario.message)
            if send_result.get("code") != 0:
                result["error"] = f"发送失败: {send_result}"
                result["end_time"] = time.time()
                return result

            sent_message_id = _trim(((send_result.get("data") or {}).get("message_id")))
            result["message_id"] = sent_message_id
            result["send_time"] = time.time()

            bot_reply = await self.wait_for_bot_reply(sent_message_id, timeout=timeout)
            result["end_time"] = time.time()

            if bot_reply:
                result["success"] = True
                result["bot_reply"] = {
                    "message_id": _trim(bot_reply.get("message_id")),
                    "msg_type": _trim(bot_reply.get("msg_type")),
                    "content": _trim(((bot_reply.get("body") or {}).get("content")))[:200],
                }
                result["total_latency_ms"] = (result["end_time"] - result["start_time"]) * 1000
            else:
                result["error"] = "等待机器人回复超时"
        except Exception as exc:
            result["error"] = str(exc)
            result["end_time"] = time.time()
        return result

    async def run(self, scenarios: list[TestScenario], timeout: int) -> list[dict[str, Any]]:
        print("=" * 60)
        print("飞书用户态真实工作测试")
        print("=" * 60)
        print(f"目标 Chat ID: {self.bot_chat_id}")
        print(f"开始时间: {datetime.now().isoformat()}")
        print()

        for scenario in scenarios:
            self.results.append(await self.test_scenario(scenario, timeout))
        return self.results

    def generate_report(self) -> str:
        lines = [
            "",
            "=" * 60,
            "测试报告",
            "=" * 60,
            f"总测试数: {len(self.results)}",
            f"成功数: {sum(1 for item in self.results if item['success'])}",
            f"失败数: {sum(1 for item in self.results if not item['success'])}",
            "",
        ]
        for idx, result in enumerate(self.results, start=1):
            status = "PASS" if result["success"] else "FAIL"
            lines.append(f"[{idx}] {status} {result['test_name']}")
            lines.append(f"    消息: {result['message'][:80]}...")
            lines.append(f"    预期类型: {result['expected_type']}")
            if result.get("total_latency_ms") is not None:
                lines.append(f"    总时延: {result['total_latency_ms']:.0f}ms")
            if result.get("bot_reply"):
                lines.append(f"    机器人回复: {result['bot_reply']['content'][:120]}...")
            if result.get("error"):
                lines.append(f"    错误: {result['error']}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send user-originated Feishu messages and wait for real bot replies.")
    parser.add_argument("--chat-id", default=DEFAULT_CHAT_ID)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--single-message", default="")
    parser.add_argument("--report-file", default="")
    return parser


async def main() -> int:
    args = _build_parser().parse_args()
    user_access_token = _trim(os.getenv("FEISHU_USER_ACCESS_TOKEN"))
    if not user_access_token:
        print("错误: 需要设置 FEISHU_USER_ACCESS_TOKEN 环境变量。")
        print("获取方式:")
        print("1. 运行 python scripts/feishu_oauth_flow.py")
        print("2. 或在飞书开放平台手动获取测试 user_access_token")
        print("设置环境变量示例:")
        print('  PowerShell: $env:FEISHU_USER_ACCESS_TOKEN="u-xxxx"')
        return 1

    chat_id = _trim(args.chat_id)
    if not chat_id:
        print("错误: 缺少 chat_id。")
        return 1

    scenarios = [TestScenario("单条消息", args.single_message, "custom")] if _trim(args.single_message) else list(DEFAULT_SCENARIOS)
    driver = FeishuUserDriver(user_access_token, chat_id)
    await driver.run(scenarios, timeout=max(1, int(args.timeout)))
    report = driver.generate_report()
    print(report)

    report_path = Path(args.report_file) if _trim(args.report_file) else Path(f"feishu_bot_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report_path.write_text(json.dumps(driver.results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果已保存到: {report_path}")
    return 0 if all(item.get("success") for item in driver.results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
