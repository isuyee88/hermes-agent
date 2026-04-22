from __future__ import annotations

from typing import Any, Callable, Mapping

from internal.domain_skills import extract_target_url_and_domain

from .planner import (
    build_internal_action_plan as _planner_build_internal_action_plan,
    build_internal_plan as _planner_build_internal_plan,
    build_internal_result as _planner_build_internal_result,
    build_provider_plan as _planner_build_provider_plan,
    infer_route_hint as _planner_infer_route_hint,
    is_external_exec_candidate as _planner_is_external_exec_candidate,
)


def normalize_route_hint(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"fast_control", "native_io", "cf_browser_first", "modal_heavy_exec"}:
        return normalized
    return "modal_heavy_exec"


def message_explicitly_requests_browser_tools(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    target_url, _ = extract_target_url_and_domain(text)
    patterns = (
        "browser",
        "playwright",
        "camofox",
        "browserbase",
        "browser use",
        "visit ",
        "open ",
        "navigate",
        "click",
        "scroll",
        "screenshot",
        "fill form",
        "submit",
        "upload",
        "captcha",
        "login to",
        "sign in to",
        "sign up on",
        "register on",
        "打开",
        "访问",
        "导航到",
        "点击",
        "滚动",
        "截图",
        "填写",
        "提交",
        "上传",
        "验证码",
        "用浏览器",
        "帮我登录",
        "登录到",
        "注册到",
        "控制台",
        "后台",
    )
    if any(pattern in text for pattern in patterns):
        return True
    if target_url:
        return any(
            pattern in text
            for pattern in (
                "看一下首页并截图",
                "打开这个网站",
                "访问这个网站",
                "open this url",
                "open this site",
                "go to this site",
            )
        )
    return False


def message_explicitly_requests_browser_tools(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False

    target_url, _ = extract_target_url_and_domain(text)
    patterns = (
        "browser",
        "playwright",
        "camofox",
        "browserbase",
        "browser use",
        "visit ",
        "open ",
        "navigate",
        "click",
        "scroll",
        "screenshot",
        "fill form",
        "fill out",
        "submit",
        "upload",
        "captcha",
        "login to",
        "log in to",
        "sign in to",
        "sign up on",
        "register on",
        "\u6253\u5f00",
        "\u8bbf\u95ee",
        "\u5bfc\u822a\u5230",
        "\u70b9\u51fb",
        "\u6eda\u52a8",
        "\u622a\u56fe",
        "\u586b\u5199",
        "\u63d0\u4ea4",
        "\u4e0a\u4f20",
        "\u9a8c\u8bc1\u7801",
        "\u7528\u6d4f\u89c8\u5668",
        "\u5e2e\u6211\u767b\u5f55",
        "\u767b\u5f55\u5230",
        "\u6ce8\u518c\u5230",
        "\u63a7\u5236\u53f0",
        "\u540e\u53f0",
    )
    if any(pattern in text for pattern in patterns):
        return True

    if target_url:
        return any(
            pattern in text
            for pattern in (
                "\u770b\u4e00\u4e0b\u9996\u9875\u5e76\u622a\u56fe",
                "\u6253\u5f00\u8fd9\u4e2a\u7f51\u7ad9",
                "\u8bbf\u95ee\u8fd9\u4e2a\u7f51\u7ad9",
                "open this url",
                "open this site",
                "go to this site",
            )
        )
    return False


def infer_internal_route_hint(payload: Mapping[str, Any]) -> str:
    explicit = normalize_route_hint(payload.get("route_hint"))
    if explicit != "modal_heavy_exec":
        return explicit
    requires_browser = payload.get("requires_browser")
    if isinstance(requires_browser, bool):
        return "cf_browser_first" if requires_browser else "modal_heavy_exec"
    return _planner_infer_route_hint(
        payload,
        normalize_route_hint=normalize_route_hint,
        message_requests_browser=message_explicitly_requests_browser_tools,
    )


def build_internal_action_plan(send_plan: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return _planner_build_internal_action_plan(send_plan)


def is_external_exec_candidate(payload: Mapping[str, Any], route_hint: str) -> bool:
    return _planner_is_external_exec_candidate(
        payload,
        route_hint,
        normalize_route_hint=normalize_route_hint,
        message_requests_browser=message_explicitly_requests_browser_tools,
    )


def build_provider_plan(
    session_state: Mapping[str, Any] | None,
    *,
    is_freeish_model_name: Callable[[str | None], bool],
) -> dict[str, Any]:
    return _planner_build_provider_plan(
        session_state,
        is_freeish_model_name=is_freeish_model_name,
    )


def build_internal_plan(
    payload: Mapping[str, Any],
    *,
    session_state_before: Mapping[str, Any] | None = None,
    llm_request: Mapping[str, Any] | None = None,
    is_freeish_model_name: Callable[[str | None], bool],
) -> dict[str, Any]:
    return _planner_build_internal_plan(
        payload,
        session_state_before=session_state_before,
        llm_request=llm_request,
        normalize_route_hint=normalize_route_hint,
        message_requests_browser=message_explicitly_requests_browser_tools,
        is_freeish_model_name=is_freeish_model_name,
    )


def build_internal_result(
    *,
    status: str,
    route_hint: str,
    execution_mode: str,
    session_state_before: Mapping[str, Any] | None,
    session_state_after: Mapping[str, Any] | None,
    send_plan: list[dict[str, Any]] | None = None,
    final_response: str | None = None,
    provider_usage: Mapping[str, Any] | None = None,
    reconcile_required: bool = False,
    browser_fallback_allowed: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return _planner_build_internal_result(
        status=status,
        route_hint=route_hint,
        execution_mode=execution_mode,
        session_state_before=session_state_before,
        session_state_after=session_state_after,
        normalize_route_hint=normalize_route_hint,
        send_plan=send_plan,
        final_response=final_response,
        provider_usage=provider_usage,
        reconcile_required=reconcile_required,
        browser_fallback_allowed=browser_fallback_allowed,
        **extra,
    )
