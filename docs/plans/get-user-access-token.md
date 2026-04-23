# 获取飞书用户访问令牌 (`user_access_token`)

## 当前结论

- `app3` 的应用身份链路已经可用，机器人身份发消息和 webhook 回调都正常。
- 真用户闭环现在分成三类独立阻塞，不能再只看“有没有 token”：
  - `99991677`：当前 shell 里的 `FEISHU_USER_ACCESS_TOKEN` 已过期。
  - `99991672`：用户态成员接口缺少聊天读取权限，至少要有 `im:chat.members:read`、`im:chat.group_info:readonly`、`im:chat:readonly`、`im:chat` 之一。
  - `230027`：用户态发消息缺少 `im:message.send_as_user` 权限。
- 所以 `FX003` 的真实目标已经变成“补齐用户身份权限并重新授权完成成员审计”，`FX004` 的真实目标变成“补齐 `send_as_user` 后再做真实用户消息闭环”。

## 推荐授权方式

优先使用仓库内脚本打印授权地址：

```powershell
python scripts/feishu_oauth_flow.py --print-auth-url-only
```

如果希望脚本直接拉起本地回调并等待浏览器回跳：

```powershell
python scripts/feishu_oauth_flow.py
```

如果要显式追加用户态发消息权限，可直接传 scope：

```powershell
python scripts/feishu_oauth_flow.py --scope im:message --scope im:message:send_as_bot --scope im:chat:readonly --scope im:chat.members:read --scope im:message.send_as_user
```

说明：

- API 错误里显示的是 `im:message.send_as_user`。
- 飞书后台权限名有时会显示成 `im:message:send_as_user`。
- 以后台实际可勾选权限为准，开通后再重新授权。

## 当前默认参数

- App ID: `cli_a9525a47e4f99bc2`
- Redirect URI: `http://localhost:3000/callback`
- 默认 scopes:
  - `im:message`
  - `im:message:send_as_bot`
  - `im:chat:readonly`
  - `im:chat.members:read`

默认脚本还没有强制带上 `send_as_user`，因为这项权限是否已在后台开通需要先由管理员确认。

## 管理后台需要确认的权限

至少确认 `app3` 已开通并发布以下用户身份权限：

- 成员审计相关：`im:chat.members:read` 或 `im:chat.group_info:readonly` 或 `im:chat:readonly` 或 `im:chat`
- 用户态发消息相关：`im:message.send_as_user`

如果权限刚开通，还需要重新发布应用并重新做一次 OAuth 授权。

## 授权完成后如何落地到当前环境

脚本成功后会生成 `feishu_tokens_*.json`，也会打印环境变量示例。PowerShell 可直接执行：

```powershell
$env:FEISHU_USER_ACCESS_TOKEN="u-xxxxxxxxxxxxxxxx"
$env:FEISHU_USER_REFRESH_TOKEN="ur-xxxxxxxxxxxxxxxx"
```

如果终端里还是旧 token，脚本会继续报 `99991677`。当前仓库里的执行清单脚本已经会优先读取最新的 `feishu_tokens_*.json`，但手工命令仍建议显式更新环境变量。

## 重新授权后的验证顺序

1. 先验证用户态发消息权限是否齐全：

```powershell
python scripts/feishu_user_token_test.py --single-message "用户态 smoke test"
```

2. 再验证用户态成员读取是否齐全：

```powershell
python scripts/check_feishu_delivery_path.py --all-env-apps --target-chat-id oc_ec86c28e66596c25377aff2ee028901c --recent-message-window-minutes 180
```

3. 成员和发消息都通过后，再补采真实 read receipt：

```powershell
python scripts/feishu_read_receipt_probe.py --chat-id oc_ec86c28e66596c25377aff2ee028901c --window-minutes 180 --page-size 20 --limit 10
```

4. 最后重跑统一 KPI 快照和执行清单：

```powershell
python scripts/feishu_kpi_execution_snapshot.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent
python scripts/feishu_execution_task_checklist.py --hours 24 --artifacts-dir D:\suyee\github\hermesagent --target-chat-id oc_ec86c28e66596c25377aff2ee028901c
```

## 如果只想验证应用身份

如果当前只是确认 `app3` 的应用身份、机器人身份和群可见性，可运行：

```powershell
python scripts/feishu_get_test_token.py --skip-send-test
```

这个脚本只验证 tenant token 侧链路，不会生成 `user_access_token`。

## 常见问题

### 1. `99991677 Authentication token expired`

说明当前终端中的 `FEISHU_USER_ACCESS_TOKEN` 还是旧值。重新授权后要么重新导出环境变量，要么改用最新生成的 `feishu_tokens_*.json`。

### 2. `99991672 Access denied`

说明 token 本身可能是新的，但用户态成员读取权限没有开通。先去飞书后台给 `app3` 开通聊天读取相关用户权限，再重新授权。

### 3. `230027 Lack of necessary permissions, ext=requires im:message.send_as_user scope.`

说明用户态发消息权限没开通。必须先补齐 `send_as_user`，否则 `FX004` 无法形成真实用户消息闭环。

### 4. 授权成功但终端里还是失败

优先排查：

- 当前 shell 是否仍在使用旧的 `FEISHU_USER_ACCESS_TOKEN`
- 最新 `feishu_tokens_*.json` 是否已经生成
- 飞书后台权限是否已经发布而不仅仅是勾选

### 5. 本地回调始终收不到

检查：

- 飞书开放平台是否已配置 `http://localhost:3000/callback`
- 本机 `3000` 端口是否被占用
- 浏览器里是否真正完成了授权确认
