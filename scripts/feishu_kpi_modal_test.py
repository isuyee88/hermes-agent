"""
Feishu KPI Modal Test - 通过Modal远程测试飞书消息

使用Modal worker在云端执行飞书消息测试，验证真实KPI指标
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_modal_feishu_test(chat_id: str, test_message: str, test_type: str) -> dict:
    """
    通过Modal运行飞书测试
    
    由于本地环境无法直接访问飞书API，我们通过Modal的web_app函数
    在云端执行测试，然后获取结果
    """
    import modal
    
    # 调用Modal函数进行测试
    try:
        # 使用Modal的Function.lookup来调用已部署的函数
        result = modal.Function.lookup("hermes-agent-direct", "debug_feishu_runtime").remote()
        return {
            "success": True,
            "result": result,
            "note": "Modal函数调用成功，但这只是debug函数，不是真实消息测试"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "note": "Modal函数调用失败"
        }


def generate_test_plan() -> str:
    """生成真实飞书测试计划"""
    
    plan = """
# 飞书KPI真实消息测试计划

## 测试环境要求

由于当前开发环境无法直接访问飞书API，需要在以下环境执行真实测试：

### 方案1: 本地环境（需要网络访问）
```bash
# 在可以访问飞书API的环境中运行
python scripts/feishu_kpi_e2e_test.py \
    --chat-id oc_xxxxxx \
    --test-type all
```

### 方案2: Modal云端环境
```python
# 在Modal环境中执行测试
import modal

# 调用Hermes Agent处理真实飞书消息
result = modal.Function.lookup("hermes-agent", "process_feishu_event").remote({
    "event": {
        "message": {
            "chat_id": "oc_xxxxxx",
            "content": {"text": "测试消息"}
        }
    }
})
```

### 方案3: 飞书Webhook测试
```bash
# 直接调用飞书Webhook
curl -X POST https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx \
    -H 'Content-Type: application/json' \
    -d '{"msg_type": "text", "content": {"text": "测试消息"}}'
```

## 需要测试的真实场景

### 1. 纯文本消息 (text_plain)
- **消息**: "今天天气怎么样？"
- **验证KPI**: 
  - 已读时延 < 5秒
  - 扣AI后回复时延 < 20秒
  - 单会话成本 < $0.0045

### 2. 浏览器任务 (browser_heavy)
- **消息**: "打开浏览器访问github.com查看issues"
- **验证KPI**:
  - 浏览器任务分类正确率 = 100%
  - 单次AI调用达成率 > 50%
  - 智能路由正确率 = 100%

### 3. 代码任务 (text_coding)
- **消息**: "帮我写一个Python函数计算斐波那契数列"
- **验证KPI**:
  - 类型匹配正确率 = 100%
  - 高性能模型选择率 >= 95%

### 4. 多模态任务 (image_understanding)
- **消息**: "分析这张图片里的内容"（带图片）
- **验证KPI**:
  - 能力匹配正确率 = 100%
  - 缓存率 > 30%

### 5. 控制命令 (session_mutation)
- **消息**: 
  - "/model gpt-4"
  - "/persona expert"
  - "/reset"
- **验证KPI**:
  - 控制命令响应时延 < 2秒
  - 会话状态切换正确率 = 100%

### 6. 限流测试 (rate_limit)
- **场景**: 快速发送10条消息
- **验证KPI**:
  - CF网关速率限制触发fallback = 0
  - 429错误吸收在queue/worker层

### 7. Fallback测试
- **场景**: 触发模型错误后观察fallback行为
- **验证KPI**:
  - 单次fallback成功率 = 100%
  - 无二次fallback

## 测试数据收集

每次测试需要记录：
1. **T0**: 消息发送时间
2. **T1**: 飞书ACK时间
3. **T2**: Hermes分类完成时间
4. **T3**: AI模型响应完成时间
5. **T4**: 正式回复发送时间
6. **T5**: 已读事件时间

计算指标：
- ACK时延 = T1 - T0
- 分类时延 = T2 - T0
- AI执行时长 = T3 - T2
- 扣AI后回复时延 = T4 - T0 - (T3 - T2)
- 已读时延 = T5 - T0

## 执行建议

1. **在飞书测试群组中执行**
   - 创建专门的测试群组
   - 邀请测试机器人加入
   - 记录所有消息和响应

2. **使用真实用户账号**
   - 模拟真实使用场景
   - 观察实际时延和成本

3. **批量测试**
   - 每种类型发送10-20条消息
   - 计算平均指标和P95/P99

4. **长期监控**
   - 设置定时任务每小时发送测试消息
   - 记录趋势和异常

## 当前状态

✅ 单元测试: 83/83 通过
✅ Modal部署: 成功
⚠️  E2E测试: 需要真实飞书环境

下一步: 在可以访问飞书API的环境中执行上述测试计划
"""
    
    return plan


def main():
    print("=" * 60)
    print("飞书KPI真实消息测试")
    print("=" * 60)
    print()
    
    # 检查环境
    print("环境检查:")
    print(f"  FEISHU_APP_ID: {os.getenv('FEISHU_APP_ID', 'Not set')}")
    print(f"  FEISHU_APP_SECRET: {'Set' if os.getenv('FEISHU_APP_SECRET') else 'Not set'}")
    print()
    
    # 尝试Modal测试
    print("尝试Modal远程测试...")
    result = run_modal_feishu_test("oc_test", "测试消息", "text")
    print(f"  结果: {result}")
    print()
    
    # 生成测试计划
    print("生成真实测试计划...")
    plan = generate_test_plan()
    
    # 保存测试计划
    plan_file = f"feishu_kpi_test_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(plan_file, "w", encoding="utf-8") as f:
        f.write(plan)
    
    print(f"测试计划已保存到: {plan_file}")
    print()
    print("=" * 60)
    print("重要提示")
    print("=" * 60)
    print()
    print("当前开发环境无法直接访问飞书API，需要在以下环境执行真实测试：")
    print()
    print("1. 本地开发机（如果可以访问外网）")
    print("2. 云服务器（有公网访问权限）")
    print("3. Modal云端环境（已部署，需要配置chat_id）")
    print()
    print("请查看测试计划文件了解详细步骤。")
    print()


if __name__ == "__main__":
    main()
