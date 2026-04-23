from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import aiohttp


DEFAULT_APP_ID = os.getenv("FEISHU_APP_ID3", "cli_a9525a47e4f99bc2")
DEFAULT_APP_SECRET = os.getenv("FEISHU_APP_SECRET3", "")
DEFAULT_REDIRECT_URI = os.getenv("FEISHU_OAUTH_REDIRECT_URI", "http://localhost:3000/callback")
DEFAULT_CHAT_ID = os.getenv("FEISHU_TEST_CHAT_ID") or os.getenv("FEISHU_HOME_CHANNEL") or "oc_ec86c28e66596c25377aff2ee028901c"
DEFAULT_SCOPES = [
    "im:message",
    "im:message:send_as_bot",
    "im:chat:readonly",
    "im:chat.members:read",
]


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


@dataclass
class OAuthCallbackState:
    redirect_path: str
    event: threading.Event
    auth_code: str = ""
    error: str = ""


def _trim(value: object) -> str:
    return str(value or "").strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Feishu OAuth flow and exchange a user access token.")
    parser.add_argument("--app-id", default=DEFAULT_APP_ID)
    parser.add_argument("--app-secret", default=DEFAULT_APP_SECRET)
    parser.add_argument("--redirect-uri", default=DEFAULT_REDIRECT_URI)
    parser.add_argument("--scope", action="append", default=[])
    parser.add_argument("--chat-id", default=DEFAULT_CHAT_ID)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--print-auth-url-only", action="store_true")
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--skip-existing-token-check", action="store_true")
    parser.add_argument("--skip-send-test", action="store_true")
    parser.add_argument("--token-file", default="")
    return parser


def _scopes_from_args(args: argparse.Namespace) -> list[str]:
    raw_scopes = list(args.scope or [])
    if not raw_scopes:
        raw_scopes = list(DEFAULT_SCOPES)
    result: list[str] = []
    for item in raw_scopes:
        for token in str(item).replace(",", " ").split():
            normalized = token.strip()
            if normalized and normalized not in result:
                result.append(normalized)
    return result


def _build_auth_url(app_id: str, redirect_uri: str, scopes: list[str]) -> str:
    params = {
        "app_id": app_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
    }
    return f"https://open.feishu.cn/open-apis/authen/v1/index?{urlencode(params)}"


def _redirect_server_config(redirect_uri: str) -> tuple[str, int, str]:
    parsed = urlparse(redirect_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    return host, port, path


def _callback_handler_factory(state: OAuthCallbackState):
    class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != state.redirect_path:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Unexpected callback path.")
                return

            query_params = parse_qs(parsed.query)
            if "code" in query_params:
                state.auth_code = query_params["code"][0]
                state.event.set()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    (
                        "<html><body style='font-family:Arial;padding:40px;'>"
                        "<h2>Feishu authorization succeeded.</h2>"
                        "<p>You can close this page and return to the terminal.</p>"
                        "</body></html>"
                    ).encode("utf-8")
                )
                return

            if "error" in query_params:
                state.error = query_params["error"][0]
                state.event.set()
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    (
                        "<html><body style='font-family:Arial;padding:40px;'>"
                        "<h2>Feishu authorization failed.</h2>"
                        f"<p>Error: {state.error}</p>"
                        "</body></html>"
                    ).encode("utf-8")
                )
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Waiting for authorization...")

        def log_message(self, format: str, *args) -> None:
            return

    return OAuthCallbackHandler


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def _start_callback_server(redirect_uri: str, state: OAuthCallbackState) -> threading.Thread:
    host, port, _ = _redirect_server_config(redirect_uri)
    bind_host = "127.0.0.1" if host in {"localhost", "127.0.0.1"} else host
    handler_cls = _callback_handler_factory(state)

    def _serve() -> None:
        with _ReusableTCPServer((bind_host, port), handler_cls) as server:
            server.timeout = 1
            while not state.event.is_set():
                server.handle_request()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return thread


async def _post_json(url: str, *, headers: dict[str, str] | None = None, payload: dict[str, object] | None = None) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
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


async def get_user_access_token(app_id: str, app_secret: str, code: str) -> dict:
    tenant_token = await get_tenant_token(app_id, app_secret)
    return await _post_json(
        "https://open.feishu.cn/open-apis/authen/v1/access_token",
        headers={
            "Authorization": f"Bearer {tenant_token}",
            "Content-Type": "application/json",
        },
        payload={"grant_type": "authorization_code", "code": code},
    )


async def send_message_with_user_token(user_token: str, chat_id: str, message: str) -> dict:
    return await _post_json(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        headers={
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/json",
        },
        payload={
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": message}, ensure_ascii=False),
        },
    )


async def _validate_existing_token(user_token: str, chat_id: str) -> dict:
    message = f"用户令牌有效性检测 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return await send_message_with_user_token(user_token, chat_id, message)


def _save_token_file(path_value: str, payload: dict[str, object]) -> Path:
    target = Path(path_value) if path_value else Path(f"feishu_tokens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


async def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    app_id = _trim(args.app_id)
    app_secret = _trim(args.app_secret)
    redirect_uri = _trim(args.redirect_uri)
    chat_id = _trim(args.chat_id)
    scopes = _scopes_from_args(args)

    if not app_id:
        print("错误: 缺少 Feishu app_id。")
        return 1
    if not app_secret:
        print("错误: 缺少 FEISHU_APP_SECRET3。")
        return 1

    auth_url = _build_auth_url(app_id, redirect_uri, scopes)

    print("=" * 60)
    print("飞书 OAuth 授权流程")
    print("=" * 60)
    print(f"App ID: {app_id}")
    print(f"Redirect URI: {redirect_uri}")
    print(f"Scopes: {' '.join(scopes)}")
    print(f"Target chat: {chat_id or 'n/a'}")
    print()
    print("授权 URL:")
    print(auth_url)
    print()

    if args.print_auth_url_only:
        return 0

    existing_token = _trim(os.getenv("FEISHU_USER_ACCESS_TOKEN"))
    if existing_token and not args.skip_existing_token_check and chat_id:
        print("检测到现有 FEISHU_USER_ACCESS_TOKEN，先做有效性验证...")
        validation = await _validate_existing_token(existing_token, chat_id)
        if validation.get("code") == 0:
            print("现有用户令牌仍然有效，无需重新授权。")
            print("下一步可直接运行: python scripts/feishu_user_token_test.py")
            return 0
        print(f"现有用户令牌无效: {validation.get('msg')}")
        print()

    host, port, callback_path = _redirect_server_config(redirect_uri)
    if host not in {"localhost", "127.0.0.1"}:
        print("当前 redirect URI 不是本地回调地址。请先手动完成授权，或改用 localhost/127.0.0.1。")
        return 1

    callback_state = OAuthCallbackState(redirect_path=callback_path, event=threading.Event())
    print(f"启动本地回调服务: {host}:{port}{callback_path}")
    _start_callback_server(redirect_uri, callback_state)

    if args.no_open_browser:
        print("已禁用自动打开浏览器，请手动访问上面的授权 URL。")
    else:
        opened = webbrowser.open(auth_url)
        if opened:
            print("已尝试自动打开浏览器。")
        else:
            print("无法自动打开浏览器，请手动访问上面的授权 URL。")

    print(f"等待授权回调，最长 {args.timeout_seconds} 秒...")
    started = time.time()
    callback_state.event.wait(timeout=max(1, int(args.timeout_seconds)))

    if callback_state.error:
        print(f"授权失败: {callback_state.error}")
        return 1
    if not callback_state.auth_code:
        waited = int(time.time() - started)
        print(f"超时: {waited} 秒内未收到授权回调。")
        print("如需仅打印授权 URL，可运行: python scripts/feishu_oauth_flow.py --print-auth-url-only")
        return 1

    print(f"收到授权码: {callback_state.auth_code[:24]}...")
    print("正在交换 user_access_token...")
    token_result = await get_user_access_token(app_id, app_secret, callback_state.auth_code)
    if token_result.get("code") != 0:
        print(f"获取 user_access_token 失败: {json.dumps(token_result, ensure_ascii=False)}")
        return 1

    data = token_result.get("data") or {}
    user_access_token = _trim(data.get("access_token"))
    refresh_token = _trim(data.get("refresh_token"))
    expires_in = data.get("expires_in")
    if not user_access_token:
        print(f"授权返回中缺少 access_token: {json.dumps(token_result, ensure_ascii=False)}")
        return 1

    saved_path = _save_token_file(
        args.token_file,
        {
            "user_access_token": user_access_token,
            "refresh_token": refresh_token,
            "expires_in": expires_in,
            "created_at": datetime.now().isoformat(),
            "app_id": app_id,
            "redirect_uri": redirect_uri,
            "scopes": scopes,
        },
    )

    print("授权成功。")
    print(f"Token 文件: {saved_path}")
    print("环境变量设置命令:")
    print(f"$env:FEISHU_USER_ACCESS_TOKEN=\"{user_access_token}\"")
    if refresh_token:
        print(f"$env:FEISHU_USER_REFRESH_TOKEN=\"{refresh_token}\"")
    print()

    if not args.skip_send_test and chat_id:
        print("使用新令牌发送一条验证消息...")
        test_result = await send_message_with_user_token(
            user_access_token,
            chat_id,
            f"OAuth 授权成功验证消息 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        if test_result.get("code") == 0:
            print("用户令牌验证消息发送成功。")
        else:
            print(f"用户令牌已获取，但验证消息发送失败: {json.dumps(test_result, ensure_ascii=False)}")

    print("下一步建议:")
    print("1. 设置 FEISHU_USER_ACCESS_TOKEN 环境变量。")
    print("2. 运行 python scripts/feishu_user_token_test.py")
    print("3. 重新运行 python scripts/check_feishu_delivery_path.py --all-env-apps --target-chat-id <target_chat_id> --recent-message-window-minutes 180")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
