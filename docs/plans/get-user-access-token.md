# 获取飞书用户访问令牌 (user_access_token)

## 方法一：通过飞书开放平台获取测试令牌（推荐用于测试）

### 步骤1: 访问飞书开放平台
1. 打开 https://open.feishu.cn/app/
2. 登录您的飞书账号
3. 找到您的应用（App ID: cli_a9525a47e4f99bc2）

### 步骤2: 获取测试令牌
1. 进入应用详情页
2. 点击左侧菜单 "凭证与基础信息"
3. 找到 "用户访问令牌" 部分
4. 点击 "获取 user_access_token"
5. 选择权限范围（需要 `im:message:send` 和 `im:message`）
6. 复制生成的令牌

### 步骤3: 设置环境变量

**Windows PowerShell:**
```powershell
$env:FEISHU_USER_ACCESS_TOKEN="u-xxxxxxxxxxxxxxxx"
```

**Windows CMD:**
```cmd
set FEISHU_USER_ACCESS_TOKEN=u-xxxxxxxxxxxxxxxx
```

**Mac/Linux:**
```bash
export FEISHU_USER_ACCESS_TOKEN=u-xxxxxxxxxxxxxxxx
```

## 方法二：通过OAuth2流程获取（生产环境）

如果您需要长期有效的令牌，需要通过OAuth2授权流程：

### 1. 配置OAuth回调地址
在飞书开放平台 -> 您的应用 -> 安全设置中配置重定向URL

### 2. 引导用户授权
```
https://open.feishu.cn/open-apis/authen/v1/index?app_id=cli_a9525a47e4f99bc2&redirect_uri=https://your-domain.com/callback
```

### 3. 获取授权码并交换令牌
用户授权后，飞书会重定向到您的回调地址并附带 `code` 参数，然后用code换取access_token。

## 方法三：使用机器人Webhook（最简单）

如果您只是想测试机器人是否能正常工作，可以直接使用Webhook：

### 1. 在飞书群组中添加机器人
1. 打开目标群组
2. 点击群组设置 -> 群机器人
3. 添加自定义机器人
4. 复制Webhook地址

### 2. 使用Webhook发送消息
```bash
curl -X POST https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx \
  -H 'Content-Type: application/json' \
  -d '{
    "msg_type": "text",
    "content": {
      "text": "测试消息"
    }
  }'
```

## 测试脚本使用方法

获取令牌后，运行测试：

```bash
# 设置令牌
$env:FEISHU_USER_ACCESS_TOKEN="u-your-token"

# 运行测试
python scripts/feishu_user_token_test.py
```

## 注意事项

1. **令牌有效期**：测试令牌通常有效期为2小时，过期需要重新获取
2. **权限范围**：确保令牌有发送消息的权限
3. **安全性**：不要将令牌提交到代码仓库
4. **频率限制**：注意飞书API的调用频率限制

## 当前环境状态

- FEISHU_APP_ID3: ✅ 已设置 (cli_a9525a47e4f99bc2)
- FEISHU_APP_SECRET3: ✅ 已设置
- FEISHU_USER_ACCESS_TOKEN: ❌ 未设置

请先获取用户访问令牌，然后重新运行测试脚本。
