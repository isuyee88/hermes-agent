# 消息管道断裂点分析 (2026-04-20)

## 根因: `open_id cross app`

39分钟前 (18:33:54) 的 Workflow 实例在最后一步 `send-feishu-response` 失败:
```
send_message_failed:400:open_id cross app
```

这意味着:
1. 飞书 webhook ✅ 触发了 CF Worker
2. CF Worker ✅ 触发了 Workflow
3. Workflow ✅ 执行了 plan → modal-exec (返回 `[Hermes] Message received, processing...`)
4. send-feishu-response ❌ **发送失败** — `open_id cross app`

### `open_id cross app` 错误含义

飞书 API 的 `send_message_failed:400:open_id cross app` 表示:
- 机器人试图用 `open_id` 发送消息给一个它不在的聊天
- 机器人被加入了群，但没有该群的发送权限
- 或者 `receiveIdType` 设置错误 (应该用 `chat_id` 但用了 `open_id`)

### 代码路径: resolveDefaultSendTarget

```typescript
// control-handlers.ts:109-136
export function resolveDefaultSendTarget(normalized, deps) {
  const rawChatType = deps.readString(message, "chat_type") || ...;
  if (rawChatType.toLowerCase() === "p2p") {
    // P2P 聊天用 open_id
  }
  // 群聊: 用 chat_id
  return {
    receiveId: normalized.chat_id,
    receiveIdType: "chat_id",
  };
}
```

**问题**: `rawChatType` 可能不是 "p2p"，所以走群聊路径返回 `chat_id`，但 `chat_id` 值可能为空或错误。

### 为什么飞书群里看不到消息

即使 `send_message_failed` 报错，`send-feishu-response` 步骤的 output 仍显示 `delivered: true` — 说明**错误被吞掉了**，没有抛出异常让 Workflow 终止。

## 修复方案

1. 修复 `sendLoggedFeishuOperation` 的错误处理，让发送失败时抛出异常
2. 添加 `receive_id_type` 日志确认实际使用的 ID 类型
3. 确保群消息使用 `chat_id` 而非 `open_id`
4. 添加 webhook 入口层日志，确认请求是否到达 CF Worker

## 待排查

- 用户最新发的消息是否触发了 webhook? (看 CF Worker 日志)
- `chat_id` 值在 `normalized` 中是否正确传递?
- 飞书 webhook 配置的 URL 是否正确指向 CF Worker?
