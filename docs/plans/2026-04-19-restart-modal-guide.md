# 重启Modal应用使FEISHU_ALLOW_ALL_USERS=true生效

## 当前状态
- ✅ custom-secret已更新：FEISHU_ALLOW_ALL_USERS=true
- ❌ 应用处于stopped状态，需要重启

## 重启方法

### 方法1：Modal Dashboard（推荐）
1. 访问 https://modal.com/apps/isuyee88/
2. 找到 hermes-agent 应用（ap-DLnXelDnPk8ujXSOek7MPz）
3. 点击进入应用详情
4. 点击 **Stop** 然后再次点击 **Deploy**（或找到Restart按钮）
5. 等待应用启动完成

### 方法2：通过Webhook触发冷启动
Modal的serverless架构会在收到请求时自动启动应用。但由于应用完全停止，可能需要Dashboard操作。

## 验证步骤
重启后运行：
```powershell
py -3.11 test_feishu_fix.py
```

期望结果：
- healthz返回200
- FEISHU_ALLOW_ALL_USERS: true
- 飞书API不再返回"Unauthorized user"错误
