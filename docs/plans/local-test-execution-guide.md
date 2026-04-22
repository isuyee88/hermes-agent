# 本地环境飞书KPI测试执行指南

## 前置条件

1. **可以访问外网的机器**（Windows/Mac/Linux）
2. **Python 3.11+**（推荐3.11，避免3.14的Modal问题）
3. **飞书应用配置**：
   - App ID: `cli_a96babfc89b8dcd1`
   - App Secret: 已配置在环境变量
   - 测试群组 Chat ID: 需要您提供

## 快速开始

### 步骤1: 克隆代码库

```bash
# 在您的本地机器上
git clone <repository-url>
cd hermes-agent
```

### 步骤2: 创建Python虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 步骤3: 安装依赖

```bash
pip install aiohttp
```

### 步骤4: 配置环境变量

```bash
# Windows PowerShell
$env:FEISHU_APP_ID="cli_a96babfc89b8dcd1"
$env:FEISHU_APP_SECRET="<您的App Secret>"

# Mac/Linux
export FEISHU_APP_ID="cli_a96babfc89b8dcd1"
export FEISHU_APP_SECRET="<您的App Secret>"
```

### 步骤5: 获取测试群组Chat ID

1. 打开飞书，进入测试群组
2. 点击群组设置 -> 群机器人
3. 添加 Hermes Agent 机器人
4. 在浏览器中打开飞书开发者工具，查看群组信息的 `open_chat_id`

或者使用以下API获取：

```bash
curl -X POST https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "cli_a96babfc89b8dcd1",
    "app_secret": "<您的App Secret>"
  }'
```

### 步骤6: 执行测试

```bash
# 基础测试
python scripts/feishu_kpi_e2e_test.py --chat-id oc_xxxxxx

# 指定测试类型
python scripts/feishu_kpi_e2e_test.py --chat-id oc_xxxxxx --test-type text
python scripts/feishu_kpi_e2e_test.py --chat-id oc_xxxxxx --test-type browser
python scripts/feishu_kpi_e2e_test.py --chat-id oc_xxxxxx --test-type coding
```

## 测试场景详解

### 场景1: 纯文本消息测试

**消息内容**: "今天天气怎么样？"

**验证指标**:
- 发送响应时间 < 1秒
- 已读时延 < 5秒
- AI回复时延 < 20秒

**执行命令**:
```bash
python scripts/feishu_kpi_e2e_test.py \
  --chat-id oc_xxxxxx \
  --test-type text
```

### 场景2: 浏览器任务测试

**消息内容**: "打开浏览器访问github.com查看最近的issues"

**验证指标**:
- 任务分类为 `browser_heavy`
- 触发浏览器工具调用
- 单次AI调用完成

**执行命令**:
```bash
python scripts/feishu_kpi_e2e_test.py \
  --chat-id oc_xxxxxx \
  --test-type browser
```

### 场景3: 代码任务测试

**消息内容**: "帮我写一个Python函数计算斐波那契数列"

**验证指标**:
- 任务分类为 `text_coding`
- 选择高性能代码模型
- 模型选择率 >= 95%

**执行命令**:
```bash
python scripts/feishu_kpi_e2e_test.py \
  --chat-id oc_xxxxxx \
  --test-type coding
```

### 场景4: 批量压力测试

**场景**: 快速发送10条消息

**验证指标**:
- 无429错误触发
- 限流在queue/worker层吸收
- CF网关缓存率 > 30%

**执行脚本**:
```python
import asyncio
from scripts.feishu_kpi_e2e_test import FeishuKPITester

async def stress_test():
    tester = FeishuKPITester(
        app_id="cli_a96babfc89b8dcd1",
        app_secret="<secret>",
        chat_id="oc_xxxxxx"
    )
    
    # 快速发送10条消息
    tasks = []
    for i in range(10):
        tasks.append(tester.send_message(f"压力测试消息 #{i}"))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 分析结果
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("code") == 0)
    print(f"成功率: {success_count}/10")

asyncio.run(stress_test())
```

## 数据收集与分析

### 收集的指标

测试脚本会自动收集以下指标：

1. **时间指标**:
   - `send_response_ms`: 发送响应时间
   - `read_latency_ms`: 已读时延
   - `total_duration_ms`: 总处理时延

2. **分类指标**:
   - `classification_expected`: 预期分类
   - `classification_actual`: 实际分类（需要查看Modal日志）

3. **成本指标**:
   - `session_cost_usd`: 单会话成本
   - `idle_cost_per_hour`: 空闲时成本

4. **成功率指标**:
   - `browser_task_success_rate`: 浏览器任务成功率
   - `single_ai_call_achievement_rate`: 单次AI调用达成率
   - `routing_accuracy`: 路由正确率

### 查看Modal日志

在测试执行期间，查看Modal日志获取详细指标：

```bash
# 查看最新日志
modal app logs hermes-agent --tail

# 查看特定函数日志
modal app logs hermes-agent --function process_feishu_event
```

### 生成测试报告

测试完成后，会生成JSON格式的详细报告：

```json
{
  "timestamp": "2026-04-20T10:00:00",
  "chat_id": "oc_xxxxxx",
  "results": [
    {
      "test_name": "纯文本消息测试",
      "test_type": "text_plain",
      "success": true,
      "duration_ms": 450,
      "metrics": {
        "message_id": "om_xxxxxx",
        "send_response_ms": 120,
        "read_latency_ms": 3200,
        "read_count": 1
      }
    }
  ]
}
```

## KPI验证标准

| KPI | 目标值 | 验证方法 |
|-----|--------|----------|
| 单会话成本 | < $0.0045 | 查看Modal成本报表 |
| 空闲耗费 | < $0.005/小时 | Modal成本API |
| 已读时延 | < 5秒 | 测试脚本测量 |
| 扣AI后正式回复时延 | < 20秒 | 测试脚本测量 |
| CF网关速率限制触发fallback | = 0 | 网关日志审计 |
| 单次fallback成功率 | = 100% | 回归测试 |
| CF网关缓存率 | > 30% | 网关日志分析 |
| 浏览器任务分类正确率 | = 100% | 测试验证 |
| 浏览器单次AI调用达成率 | > 50% | 测试验证 |
| CF智能路由正确率 | = 100% | 测试验证 |
| 类型能力匹配正确率 | = 100% | 测试验证 |
| 高性能类型匹配模型选择率 | >= 95% | 测试验证 |

## 故障排查

### 问题1: 无法连接飞书API

**症状**: `Cannot connect to host open.feishu.cn:443`

**解决**: 
- 检查网络连接
- 确认可以访问外网
- 检查防火墙设置

### 问题2: 发送消息失败

**症状**: 返回错误码非0

**解决**:
- 检查App ID和App Secret
- 确认机器人已添加到群组
- 检查群组权限设置

### 问题3: 获取不到已读信息

**症状**: `read_count` 始终为0

**解决**:
- 确认消息已发送成功
- 等待更长时间（最多10秒）
- 检查飞书应用权限

## 下一步

1. 在您的本地环境执行上述步骤
2. 提供测试群组的Chat ID
3. 运行测试脚本收集真实数据
4. 分析结果并验证KPI达标情况

准备好开始测试了吗？请提供：
1. 您的测试群组Chat ID（格式：oc_xxxxxx）
2. 确认可以访问外网的环境
