from __future__ import annotations

import re
from typing import Any

_GENERIC_REFUSAL_MARKERS = (
    "i'm sorry, but i can't help with that",
    "i am sorry, but i can't help with that",
    "i'm sorry, but i cannot help with that",
    "i'm sorry, but i can't assist with that",
    "i'm sorry, but i cannot assist with that",
    "sorry, but i can't help with that",
    "sorry, but i cannot help with that",
    "\u62b1\u6b49\uff0c\u6211\u4e0d\u80fd\u5e2e\u52a9\u5904\u7406\u8fd9\u4e2a\u8bf7\u6c42",
    "\u62b1\u6b49\uff0c\u6211\u65e0\u6cd5\u5e2e\u52a9\u5904\u7406\u8fd9\u4e2a\u8bf7\u6c42",
    "\u62b1\u6b49\uff0c\u6211\u4e0d\u80fd\u5e2e\u4f60\u5904\u7406\u8fd9\u4e2a\u8bf7\u6c42",
    "\u62b1\u6b49\uff0c\u6211\u65e0\u6cd5\u5e2e\u4f60\u5904\u7406\u8fd9\u4e2a\u8bf7\u6c42",
    "\u62b1\u6b49\uff0c\u6211\u4e0d\u80fd\u534f\u52a9\u8fd9\u4e2a\u8bf7\u6c42",
    "\u62b1\u6b49\uff0c\u6211\u65e0\u6cd5\u534f\u52a9\u8fd9\u4e2a\u8bf7\u6c42",
)

_ACCOUNT_CREATION_MARKERS = (
    "register",
    "sign up",
    "signup",
    "create account",
    "open account",
    "email account",
    "social media account",
    "\u793e\u4ea4\u5a92\u4f53\u8d26\u53f7",
    "\u793e\u4ea4\u5a92\u4f53\u8d26\u6237",
    "\u90ae\u7bb1\u8d26\u53f7",
    "\u90ae\u7bb1\u8d26\u6237",
    "\u6ce8\u518c\u8d26\u53f7",
    "\u6ce8\u518c\u8d26\u6237",
    "\u521b\u5efa\u8d26\u53f7",
    "\u521b\u5efa\u8d26\u6237",
    "\u6ce8\u518c\u4e00\u4e2a\u90ae\u7bb1",
    "\u6ce8\u518c\u90ae\u7bb1",
    "\u6ce8\u518c\u793e\u4ea4\u5a92\u4f53",
)

_FEISHU_WORKSPACE_REFUSAL_MARKERS = (
    "can't access feishu",
    "cannot access feishu",
    "unable to access feishu",
    "can't operate feishu",
    "cannot operate feishu",
    "can't access lark",
    "cannot access lark",
    "unable to access lark",
    "can't access bitable",
    "cannot access bitable",
    "unable to access bitable",
    "can't operate bitable",
    "cannot operate bitable",
    "unable to operate bitable",
    "can't access docs",
    "cannot access docs",
    "can't access sheets",
    "cannot access sheets",
    "\u65e0\u6cd5\u64cd\u4f5c\u98de\u4e66",
    "\u4e0d\u80fd\u64cd\u4f5c\u98de\u4e66",
    "\u65e0\u6cd5\u8bbf\u95ee\u98de\u4e66",
    "\u4e0d\u80fd\u8bbf\u95ee\u98de\u4e66",
    "\u65e0\u6cd5\u4f7f\u7528\u98de\u4e66",
    "\u4e0d\u80fd\u4f7f\u7528\u98de\u4e66",
    "\u65e0\u6cd5\u64cd\u4f5c\u591a\u7ef4\u8868\u683c",
    "\u4e0d\u80fd\u64cd\u4f5c\u591a\u7ef4\u8868\u683c",
    "\u65e0\u6cd5\u8bbf\u95ee\u591a\u7ef4\u8868\u683c",
    "\u4e0d\u80fd\u8bbf\u95ee\u591a\u7ef4\u8868\u683c",
    "\u65e0\u6cd5\u64cd\u4f5c\u98de\u4e66\u6587\u6863",
    "\u4e0d\u80fd\u64cd\u4f5c\u98de\u4e66\u6587\u6863",
    "\u65e0\u6cd5\u64cd\u4f5c\u7535\u5b50\u8868\u683c",
    "\u4e0d\u80fd\u64cd\u4f5c\u7535\u5b50\u8868\u683c",
)


def extract_tool_names_from_messages(messages: list[dict[str, Any]] | None) -> list[str]:
    tool_names: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            name = None
            if isinstance(tool_call, dict):
                function_data = tool_call.get("function") or {}
                if isinstance(function_data, dict):
                    name = function_data.get("name")
            else:
                function_data = getattr(tool_call, "function", None)
                name = getattr(function_data, "name", None)
            if isinstance(name, str) and name.strip():
                tool_names.append(name.strip())
    return tool_names


def looks_like_generic_refusal(response: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(response or "")).strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _GENERIC_REFUSAL_MARKERS)


def looks_like_account_creation_request(user_message: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(user_message or "")).strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _ACCOUNT_CREATION_MARKERS)


def looks_like_feishu_workspace_refusal(response: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(response or "")).strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _FEISHU_WORKSPACE_REFUSAL_MARKERS)


def clarify_feishu_browser_refusal(
    response: str,
    *,
    user_message: str,
    agent_messages: list[dict[str, Any]] | None,
) -> str:
    if not looks_like_generic_refusal(response):
        return response
    if not looks_like_account_creation_request(user_message):
        return response

    tool_names = sorted(
        {
            name
            for name in extract_tool_names_from_messages(agent_messages)
            if name.startswith("browser_")
        }
    )
    if not tool_names:
        return response

    used_tools = "\u3001".join(f"`{name}`" for name in tool_names[:4])
    if len(tool_names) > 4:
        used_tools = f"{used_tools} \u7b49"

    return (
        f"\u6d4f\u89c8\u5668\u80fd\u529b\u662f\u53ef\u7528\u7684\uff0c\u8fd9\u4e00\u8f6e\u6211\u5df2\u7ecf\u5b9e\u9645\u8c03\u7528\u4e86 {used_tools}\u3002\n"
        "\u5f53\u524d\u6ca1\u6709\u7ee7\u7eed\u6267\u884c\uff0c\u4e0d\u662f\u56e0\u4e3a\u6d4f\u89c8\u5668\u6216 Modal \u4e0d\u53ef\u7528\uff0c\u800c\u662f\u8fd9\u6761\u8bf7\u6c42\u6d89\u53ca\u8d26\u53f7\u6ce8\u518c/\u521b\u5efa\u7b49\u53d7\u9650\u64cd\u4f5c\u3002\n"
        "\u5982\u679c\u4f60\u8981\u6211\u505a\u5141\u8bb8\u8303\u56f4\u5185\u7684\u6d4f\u89c8\u5668\u4efb\u52a1\uff0c\u4f8b\u5982\u6253\u5f00\u7f51\u9875\u3001\u8bfb\u53d6\u5185\u5bb9\u3001\u622a\u56fe\u3001\u9875\u9762\u6838\u9a8c\u6216\u586b\u5199\u975e\u654f\u611f\u8868\u5355\uff0c\u6211\u53ef\u4ee5\u7ee7\u7eed\u5904\u7406\u3002"
    )


def clarify_feishu_workspace_refusal(
    response: str,
    *,
    user_message: str,
    agent_messages: list[dict[str, Any]] | None,
) -> str:
    if not looks_like_feishu_workspace_refusal(response):
        return response

    tool_names = sorted(
        {
            name
            for name in extract_tool_names_from_messages(agent_messages)
            if name.startswith("feishu_")
        }
    )

    if tool_names:
        used_tools = "\u3001".join(f"`{name}`" for name in tool_names[:4])
        if len(tool_names) > 4:
            used_tools = f"{used_tools} \u7b49"
        evidence_line = f"\u8fd9\u4e00\u8f6e\u5df2\u7ecf\u5b9e\u9645\u8c03\u7528\u4e86 {used_tools}\u3002"
    else:
        evidence_line = "\u5f53\u524d\u4f1a\u8bdd\u5177\u5907\u539f\u751f `feishu_*` \u5de5\u5177\u80fd\u529b\uff0c\u4e0d\u5e94\u76f4\u63a5\u8bf4\u201c\u65e0\u6cd5\u64cd\u4f5c\u98de\u4e66\u201d\u3002"

    request_hint = ""
    normalized_user_message = re.sub(r"\s+", " ", str(user_message or "")).strip().lower()
    if any(token in normalized_user_message for token in ("bitable", "\u591a\u7ef4\u8868\u683c", "\u8868\u683c", "sheet", "sheets")):
        request_hint = "\u66f4\u51c6\u786e\u7684\u7ed3\u8bba\u5e94\u8be5\u662f\uff1a\u5f53\u524d\u5931\u8d25\u7684\u662f\u5177\u4f53\u8868\u3001range\u3001table_id\u3001schema \u6216\u6743\u9650\u914d\u7f6e\uff0c\u4e0d\u662f\u98de\u4e66\u591a\u7ef4\u8868\u683c\u80fd\u529b\u6574\u4f53\u4e0d\u53ef\u7528\u3002"
    elif any(token in normalized_user_message for token in ("doc", "docs", "\u6587\u6863")):
        request_hint = "\u66f4\u51c6\u786e\u7684\u7ed3\u8bba\u5e94\u8be5\u662f\uff1a\u5f53\u524d\u5931\u8d25\u7684\u662f\u5177\u4f53\u6587\u6863\u76ee\u6807\u3001block \u683c\u5f0f\u6216\u6743\u9650\u914d\u7f6e\uff0c\u4e0d\u662f\u98de\u4e66\u6587\u6863\u80fd\u529b\u6574\u4f53\u4e0d\u53ef\u7528\u3002"
    else:
        request_hint = "\u66f4\u51c6\u786e\u7684\u7ed3\u8bba\u5e94\u8be5\u662f\uff1a\u5f53\u524d\u5931\u8d25\u7684\u662f\u5177\u4f53\u76ee\u6807\u3001\u53c2\u6570\u3001\u6570\u636e\u683c\u5f0f\u6216\u6743\u9650\u914d\u7f6e\uff0c\u4e0d\u662f\u98de\u4e66\u5de5\u4f5c\u53f0\u80fd\u529b\u6574\u4f53\u4e0d\u53ef\u7528\u3002"

    return (
        "\u98de\u4e66\u539f\u751f\u5de5\u4f5c\u53f0\u80fd\u529b\u662f\u53ef\u7528\u7684\u3002"
        f"{evidence_line}\n"
        f"{request_hint}\n"
        "\u56e0\u6b64\u8fd9\u79cd\u60c5\u51b5\u4e0b\uff0c\u5e94\u8be5\u7ee7\u7eed\u5b9a\u4f4d\u5177\u4f53\u7684 doc/sheet/bitable \u76ee\u6807\u6216\u5199\u5165\u53c2\u6570\uff0c\u800c\u4e0d\u662f\u628a\u95ee\u9898\u63cf\u8ff0\u6210\u201c\u65e0\u6cd5\u64cd\u4f5c\u98de\u4e66\u201d\u3002"
    )
