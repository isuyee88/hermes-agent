# 错误修复清单

## 错误1: /feishu/webhook -> 500 (Invalid verification token)
**根因**: 飞书webhook verification token验证失败
**修复**: 检查Modal中FEISHU_VERIFICATION_TOKEN配置

## 错误2: Unauthorized user: (feishu-user) on feishu
**根因**: FEISHU_ALLOW_ALL_USERS未配置或为false
**修复**: 设置FEISHU_ALLOW_ALL_USERS=true

## 错误3: /telegram/webhook -> 401
**根因**: Telegram webhook secret不匹配
**修复**: 检查TELEGRAM_WEBHOOK_SECRET配置

## 修复步骤
1. 通过Modal API更新Secret配置
2. 重新部署Modal应用
3. 验证日志无错误
