# 修复飞书401问题：更新Cloudflare Worker Bearer Token

## 问题诊断

- **现象**：飞书请求到达Modal但返回401 Unauthorized
- **根因**：Cloudflare Worker的 `MODAL_INTERNAL_BEARER_TOKEN` 与Modal期望的token不匹配
- **日志**：`provided_fp=3a4a385423d3 expected_fp=fe08791a3282`

## 正确的Token值

```
Token: fi_b00129d62967080376fb238f80d7687b0c2ab3de5c062f1a4d9f4cb5990ca59a
```

基于：FEISHU_APP_ID3 (`cli_a9525a47e4f99bc2`) + FEISHU_APP_SECRET3 (`OMWiq2XeayEC6MXBxCYaNfA7qv0kre7h`)

## 更新方法

### 方法1：Cloudflare Dashboard（推荐）

1. 登录 https://dash.cloudflare.com
2. 进入 Workers & Pages > hermes-feishu-gateway
3. 点击 Settings > Variables
4. 找到 `MODAL_INTERNAL_BEARER_TOKEN` 或点击 Add variable
5. 类型选择 **Secret**（不是Variable）
6. 粘贴上面的Token值
7. 点击 Save and deploy

### 方法2：Wrangler CLI（网络恢复后）

```powershell
cd d:\suyee\github\hermesagent\hermes-agent\cloudflare\feishu-gateway
npx wrangler secret put MODAL_INTERNAL_BEARER_TOKEN
# 粘贴: fi_b00129d62967080376fb238f80d7687b0c2ab3de5c062f1a4d9f4cb5990ca59a
```

## 验证步骤

更新后运行：

```powershell
py -3.11 -c "
import httpx
url = 'https://isuyee88--hermes-agent-web-app.modal.run/healthz'
r = httpx.get(url, timeout=30)
print(r.json())
"

# 从飞书发送一条测试消息，检查Modal日志是否还有401错误
py -3.11 -m modal app logs ap-aZqKQTpiCfJj5hXwF1sTx9 --since 2026-04-19T00:00:00 | Select-String "401"
```

期望结果：不再有 `401 Unauthorized` 错误
