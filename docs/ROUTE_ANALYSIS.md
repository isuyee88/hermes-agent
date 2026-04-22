# 消息路由决策分析

## 用户消息: "请介绍一下你自己，你的核心竞争力是什么？"

### 分类逻辑 (routing.ts)

1. **siteCategory 推断**: 
   - 无URL → `{ siteCategory: "none", siteIntent: "general" }`
   
2. **requestClass 推断**:
   - 不匹配 coding patterns → `text_plain`
   
3. **routeFamily 推断**:
   - `isTextGatewayRequestClass("text_plain")` = true → `routeFamily: "gateway_text"`
   
4. **gatewayEligible**:
   - `gateway_eligible: gatewayEligible && !requiresModalRuntime`
   - `requiresModalRuntime` = lane !== "agent" || routeFamily === "modal_runtime" || requiresMediaHydration || requiresTools
   - 对于纯文本agent消息: requiresModalRuntime = false
   - → gateway_eligible = true

5. **route_hint**:
   - `inferRouteHintFromRequestClass("agent", "text_plain")` = "modal_heavy_exec"

### 问题: route_hint = "modal_heavy_exec"

这意味着即使 gateway_eligible = true，**route_hint 仍然是 "modal_heavy_exec"**，所以消息会走 Modal 路径而不是 AI 网关！

## 管道断裂点

**关键发现**: 
- `route_hint` 决定实际路由，但 `inferRouteHintFromRequestClass` 对于 `text_plain` 返回 `"modal_heavy_exec"`
- AI 网关只有在 `route_hint` 匹配特定值时才会被使用
- 当前逻辑中，`gateway_eligible` 为 true 但 `route_hint` 为 "modal_heavy_exec"，导致消息总是走 Modal

## 修复方案

需要修改 `inferRouteHintFromRequestClass` 或使用 `gateway_eligible` 来决定是否走 AI 网关
