from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_APP_ID = os.getenv("FEISHU_APP_ID3", "cli_a9525a47e4f99bc2")
DEFAULT_APP_SECRET = os.getenv("FEISHU_APP_SECRET3", "")
DEFAULT_CHAT_ID = os.getenv("FEISHU_TEST_CHAT_ID") or os.getenv("FEISHU_HOME_CHANNEL") or "oc_ec86c28e66596c25377aff2ee028901c"


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


def _trim(value: Any) -> str:
    return str(value or "").strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch a Feishu tenant token, verify bot visibility, and send a test app message.")
    parser.add_argument("--app-id", default=DEFAULT_APP_ID)
    parser.add_argument("--app-secret", default=DEFAULT_APP_SECRET)
    parser.add_argument("--chat-id", default=DEFAULT_CHAT_ID)
    parser.add_argument("--token-file", default="")
    parser.add_argument("--skip-send-test", action="store_true")
    return parser


async def _post_json(url: str, *, headers: dict[str, str] | None = None, payload: dict[str, object] | None = None) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            return await resp.json()


async def _get_json(url: str, *, headers: dict[str, str] | None = None, params: dict[str, object] | None = None) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.get(url, headers=headers, params=params) as resp:
            return await resp.json()


async def get_tenant_token(app_id: str, app_secret: str) -> str:
    payload = await _post_json(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        payload={"app_id": app_id, "app_secret": app_secret},
    )
    if payload.get("code") != 0:
        raise RuntimeError(f"tenant_access_token_failed: {payload}")
    token = _trim(payload.get("tenant_access_token"))
    if not token:
        raise RuntimeError(f"tenant_access_token_missing: {payload}")
    return token


async def get_bot_info(tenant_token: str) -> dict:
    return await _get_json(
        "https://open.feishu.cn/open-apis/bot/v3/info",
        headers={"Authorization": f"Bearer {tenant_token}"},
    )


async def get_chat_members(tenant_token: str, chat_id: str) -> dict:
    return await _get_json(
        f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}/members",
        headers={"Authorization": f"Bearer {tenant_token}"},
        params={"member_id_type": "open_id", "page_size": 100},
    )


async def send_message_as_bot(tenant_token: str, chat_id: str, message: str) -> dict:
    return await _post_json(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        headers={
            "Authorization": f"Bearer {tenant_token}",
            "Content-Type": "application/json",
        },
        payload={
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": message}, ensure_ascii=False),
        },
    )


async def main() -> int:
    args = _build_parser().parse_args()
    app_id = _trim(args.app_id)
    app_secret = _trim(args.app_secret)
    chat_id = _trim(args.chat_id)

    print("=" * 60)
    print("飞书 tenant token 检查工具")
    print("=" * 60)
    print(f"App ID: {app_id}")
    print(f"Target chat: {chat_id or 'n/a'}")
    print()

    if not app_secret:
        print("错误: 需要设置 FEISHU_APP_SECRET3。")
        return 1

    tenant_token = await get_tenant_token(app_id, app_secret)
    print(f"tenant_access_token 获取成功: {tenant_token[:30]}...")
    print()

    bot_info = await get_bot_info(tenant_token)
    if bot_info.get("code") == 0:
        bot = bot_info.get("bot") or {}
        print("机器人信息:")
        print(f"  app_name: {_trim(bot.get('app_name'))}")
        print(f"  open_id: {_trim(bot.get('open_id'))[:30]}...")
        print()
    else:
        print(f"机器人信息获取失败: {json.dumps(bot_info, ensure_ascii=False)}")
        print()

    if chat_id:
        members_payload = await get_chat_members(tenant_token, chat_id)
        if members_payload.get("code") == 0:
            items = ((members_payload.get("data") or {}).get("items") or [])
            print(f"群成员接口返回成功: {len(items)} 条")
            for idx, item in enumerate(items[:5], start=1):
                print(f"  member[{idx}]: {_trim(item.get('member_id'))[:30]}...")
            print()
        else:
            print(f"群成员获取失败: {json.dumps(members_payload, ensure_ascii=False)}")
            print()

    if not args.skip_send_test and chat_id:
        test_message = f"tenant token 测试消息 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_result = await send_message_as_bot(tenant_token, chat_id, test_message)
        if send_result.get("code") == 0:
            print("应用身份测试消息发送成功。")
        else:
            print(f"应用身份测试消息发送失败: {json.dumps(send_result, ensure_ascii=False)}")
        print()

    token_path = Path(args.token_file) if _trim(args.token_file) else Path(f"feishu_tenant_token_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    token_path.write_text(
        "\n".join(
            [
                f"Tenant Token: {tenant_token}",
                f"App ID: {app_id}",
                f"Created at: {datetime.now().isoformat()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"tenant token 已保存到: {token_path}")
    print(f"$env:FEISHU_TENANT_TOKEN=\"{tenant_token}\"")
    print()
    print("说明:")
    print("1. 这是 tenant_access_token，不是 user_access_token。")
    print("2. 它可用于应用身份发消息和机器人侧接口验证。")
    print("3. 若要打通真实用户态链路，请继续运行 python scripts/feishu_oauth_flow.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
