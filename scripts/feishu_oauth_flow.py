"""
飞书 OAuth 授权流程 - 获取 user_access_token

引导用户完成授权，获取用户访问令牌
"""

import asyncio
import json
import os
import sys
import time
import webbrowser
from datetime import datetime
from urllib.parse import urlencode, parse_qs, urlparse
import http.server
import socketserver
import threading

import aiohttp


# 飞书应用配置
APP_ID = os.getenv("FEISHU_APP_ID3", "cli_a9525a47e4f99bc2c")
APP_SECRET = os.getenv("FEISHU_APP_SECRET3", "")

# OAuth 配置
REDIRECT_URI = "http://localhost:8080/callback"
SCOPE = "im:message im:message:send"

# 全局变量存储授权码
auth_code = None
auth_received = threading.Event()


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """处理 OAuth 回调"""
    
    def do_GET(self):
        global auth_code
        
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)
        
        if "code" in query_params:
            auth_code = query_params["code"][0]
            auth_received.set()
            
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """
            <html>
            <head><title>Authorization Success</title></head>
            <body style="text-align: center; padding: 50px; font-family: Arial;">
                <h1 style="color: green;">Authorization Successful</h1>
                <p>You have successfully authorized. You can close this page and return to the terminal.</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif "error" in query_params:
            error = query_params["error"][0]
            auth_received.set()
            
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = f"""
            <html>
            <head><title>Authorization Failed</title></head>
            <body style="text-align: center; padding: 50px; font-family: Arial;">
                <h1 style="color: red;">Authorization Failed</h1>
                <p>Error: {error}</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Waiting for authorization...")
    
    def log_message(self, format, *args):
        # 静默日志
        pass


def start_callback_server():
    """启动回调服务器"""
    with socketserver.TCPServer(("", 8080), OAuthCallbackHandler) as httpd:
        httpd.timeout = 1
        while not auth_received.is_set():
            httpd.handle_request()


async def get_user_access_token(code: str) -> dict:
    """用授权码换取用户访问令牌"""
    url = "https://open.feishu.cn/open-apis/authen/v1/access_token"
    
    # 先获取 tenant_access_token
    tenant_token = await get_tenant_token()
    
    headers = {
        "Authorization": f"Bearer {tenant_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "grant_type": "authorization_code",
        "code": code
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            result = await resp.json()
            return result


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


async def refresh_user_token(refresh_token: str) -> dict:
    """刷新用户访问令牌"""
    url = "https://open.feishu.cn/open-apis/authen/v1/refresh_access_token"
    
    tenant_token = await get_tenant_token()
    
    headers = {
        "Authorization": f"Bearer {tenant_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            result = await resp.json()
            return result


async def send_message_with_user_token(user_token: str, chat_id: str, message: str) -> dict:
    """使用用户令牌发送消息"""
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    
    headers = {
        "Authorization": f"Bearer {user_token}",
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


async def main():
    print("=" * 60)
    print("飞书 OAuth 授权流程 - 获取 user_access_token")
    print("=" * 60)
    print()
    
    if not APP_SECRET:
        print("❌ 错误: 需要设置 FEISHU_APP_SECRET3 环境变量")
        sys.exit(1)
    
    # 检查是否已有有效的 user_access_token
    existing_token = os.getenv("FEISHU_USER_ACCESS_TOKEN")
    if existing_token:
        print(f"�� 检测到现有 USER_ACCESS_TOKEN: {existing_token[:20]}...")
        print("正在验证令牌有效性...")
        
        # 测试发送一条消息验证
        chat_id = "oc_ec86c28e66596c25377aff2ee028901c"
        test_result = await send_message_with_user_token(
            existing_token, 
            chat_id, 
            "测试消息 - 验证令牌有效性"
        )
        
        if test_result.get("code") == 0:
            print("✅ 现有令牌有效，无需重新授权")
            print()
            print("您可以直接运行: python scripts/feishu_user_token_test.py")
            return
        else:
            print(f"⚠️ 现有令牌无效: {test_result.get('msg')}")
            print("需要重新授权...")
            print()
    
    # 构建授权 URL
    auth_params = {
        "app_id": APP_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE
    }
    auth_url = f"https://open.feishu.cn/open-apis/authen/v1/index?{urlencode(auth_params)}"
    
    print("�� 授权流程:")
    print()
    print("步骤 1: 打开以下 URL 进行授权:")
    print(f"   {auth_url}")
    print()
    print("步骤 2: 登录您的飞书账号")
    print("步骤 3: 点击'确认授权'")
    print()
    
    # 启动回调服务器
    print("�� 正在启动本地回调服务器 (localhost:8080)...")
    server_thread = threading.Thread(target=start_callback_server)
    server_thread.daemon = True
    server_thread.start()
    
    # 尝试自动打开浏览器
    try:
        webbrowser.open(auth_url)
        print("�� 已自动打开浏览器")
    except Exception:
        print("⚠️ 无法自动打开浏览器，请手动访问上面的 URL")
    
    print()
    print("⏳ 等待授权完成 (最多等待 5 分钟)...")
    
    # 等待授权码
    auth_received.wait(timeout=300)
    
    if not auth_code:
        print("❌ 错误: 未收到授权码，请重试")
        sys.exit(1)
    
    print(f"�� 收到授权码: {auth_code[:20]}...")
    print()
    
    # 用授权码换取访问令牌
    print("�� 正在用授权码换取用户访问令牌...")
    token_result = await get_user_access_token(auth_code)
    
    if token_result.get("code") != 0:
        print(f"❌ 获取令牌失败: {token_result}")
        sys.exit(1)
    
    data = token_result.get("data", {})
    user_access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    expires_in = data.get("expires_in")
    
    print("✅ 授权成功！")
    print()
    print("=" * 60)
    print("授权信息:")
    print("=" * 60)
    print(f"用户 Access Token:  {user_access_token[:30]}...")
    print(f"Refresh Token:      {refresh_token[:30]}...")
    print(f"有效期:              {expires_in} 秒 (约 {expires_in//3600} 小时)")
    print()
    
    # 保存到文件
    token_file = f"feishu_tokens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(token_file, "w", encoding="utf-8") as f:
        json.dump({
            "user_access_token": user_access_token,
            "refresh_token": refresh_token,
            "expires_in": expires_in,
            "created_at": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    
    print(f"�� 令牌已保存到: {token_file}")
    print()
    print("环境变量设置命令:")
    print("-" * 60)
    print(f"$env:FEISHU_USER_ACCESS_TOKEN=\"{user_access_token}\"")
    print("-" * 60)
    print()
    
    # 测试发送消息
    print("�� 正在测试使用用户令牌发送消息...")
    chat_id = "oc_ec86c28e66596c25377aff2ee028901c"
    test_result = await send_message_with_user_token(
        user_access_token,
        chat_id,
        "�� OAuth 授权成功！这是从用户身份发送的测试消息"
    )
    
    if test_result.get("code") == 0:
        print("✅ 测试消息发送成功！")
        print()
        print("下一步:")
        print("  1. 设置环境变量: $env:FEISHU_USER_ACCESS_TOKEN=\"<token>\"")
        print("  2. 运行测试: python scripts/feishu_user_token_test.py")
    else:
        print(f"❌ 测试消息发送失败: {test_result}")


if __name__ == "__main__":
    asyncio.run(main())