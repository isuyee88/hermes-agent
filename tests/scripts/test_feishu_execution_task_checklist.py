from __future__ import annotations

from scripts.feishu_execution_task_checklist import _derive_checklist


def test_derive_checklist_prioritizes_kpi_tasks_over_chat_topology():
    snapshot = {
        "status": "attention",
        "summary": {
            "statuses": {
                "read_receipt_under_5s": "not_met",
                "session_cost_under_0_0045": "not_met",
                "modal_idle_hourly_cost_under_0_005": "not_met",
                "cache_hit_rate_over_0_30": "not_met",
                "browser_single_ai_call_completion_rate_over_0_50": "met",
                "capability_match_rate_equals_1_00": "met",
                "preferred_model_selection_accuracy_over_0_95": "met",
            },
            "read_receipt_p90_ms": None,
            "session_cost_p90_usd": None,
            "idle_hourly_p90_cost_usd": "0.01023596",
            "cache_eligible_hit_rate": "0.00000000",
        },
        "priority_blockers": [
            "kpi:read_receipt",
            "gap:official_cost_without_matching_function_trace",
        ],
        "next_actions": ["补采真实读回执样本"],
    }
    delivery_result = {
        "payload": {
            "status": "error",
            "summary": "Target chat still contains multiple bots and other app replies.",
            "apps": [
                {
                    "app_id": "cli_app3",
                    "visible_chat_count": 1,
                    "delivery_ready": True,
                    "published_version_summary": {"missing_required_callbacks": []},
                    "chat_topology": {
                        "chats_with_multiple_bots": 1,
                        "chats_with_user_scope_missing_chat_members_read": 1,
                    },
                    "recent_chat_activity": {
                        "total_recent_app_messages_from_other_app_ids": [{"app_id": "cli_other", "count": 1}],
                    },
                }
            ],
        }
    }
    user_send_result = {
        "success": False,
        "parsed_error": {
            "code": 230027,
            "message": "Lack of necessary permissions, ext=requires im:message.send_as_user scope.",
            "required_scope": "im:message.send_as_user",
        },
    }

    checklist = _derive_checklist(
        snapshot=snapshot,
        delivery_result=delivery_result,
        user_send_result=user_send_result,
        target_chat_id="oc_test",
    )

    task_ids = [item["id"] for item in checklist["tasks"]]
    assert task_ids[:4] == ["FX001", "FX002", "FX003", "FX004"]
    assert "FX006" in task_ids
    assert "FX005" not in task_ids
    assert checklist["summary"]["task_counts"]["blocked"] == 0
    assert checklist["summary"]["non_kpi_observations"]
    assert any("send_as_user" in item for item in checklist["summary"]["non_kpi_observations"])


def test_derive_checklist_can_be_empty_when_all_kpis_are_met():
    snapshot = {
        "status": "ok",
        "summary": {
            "statuses": {
                "read_receipt_under_5s": "met",
                "session_cost_under_0_0045": "met",
                "modal_idle_hourly_cost_under_0_005": "met",
                "cache_hit_rate_over_0_30": "met",
                "browser_single_ai_call_completion_rate_over_0_50": "met",
                "capability_match_rate_equals_1_00": "met",
                "preferred_model_selection_accuracy_over_0_95": "met",
            }
        },
        "priority_blockers": [],
        "next_actions": [],
    }

    checklist = _derive_checklist(snapshot=snapshot, delivery_result=None, user_send_result=None, target_chat_id="")

    assert checklist["tasks"] == []
    assert checklist["summary"]["task_counts"]["todo"] == 0
    assert checklist["summary"]["task_counts"]["blocked"] == 0


def test_derive_checklist_uses_canonical_app_for_read_receipt_sampling_title():
    snapshot = {
        "status": "attention",
        "summary": {
            "statuses": {
                "read_receipt_under_5s": "not_met",
                "session_cost_under_0_0045": "met",
                "modal_idle_hourly_cost_under_0_005": "met",
                "cache_hit_rate_over_0_30": "met",
                "browser_single_ai_call_completion_rate_over_0_50": "met",
                "capability_match_rate_equals_1_00": "met",
                "preferred_model_selection_accuracy_over_0_95": "met",
            },
            "read_receipt_p90_ms": None,
        },
        "priority_blockers": ["kpi:read_receipt"],
        "next_actions": [],
    }
    delivery_result = {
        "payload": {
            "status": "ok",
            "summary": "Canonical app is healthy.",
            "apps": [
                {
                    "app_id": "cli_legacy",
                    "visible_chat_count": 1,
                    "delivery_ready": False,
                    "published_version_summary": {
                        "missing_required_callbacks": ["im.message.message_read_v1"],
                    },
                    "chat_topology": {},
                    "recent_chat_activity": {},
                },
                {
                    "app_id": "cli_app3",
                    "visible_chat_count": 1,
                    "delivery_ready": True,
                    "published_version_summary": {"missing_required_callbacks": []},
                    "chat_topology": {},
                    "recent_chat_activity": {},
                },
            ],
        }
    }

    checklist = _derive_checklist(
        snapshot=snapshot,
        delivery_result=delivery_result,
        user_send_result=None,
        target_chat_id="oc_test",
    )

    fx001 = next(item for item in checklist["tasks"] if item["id"] == "FX001")
    assert "cli_app3" not in fx001["title"]
    assert "真实读回执样本" in fx001["title"]
    assert checklist["summary"]["canonical_app_id"] == "cli_app3"


def test_derive_checklist_creates_routing_validation_task_when_related_kpis_are_not_met():
    snapshot = {
        "status": "attention",
        "summary": {
            "statuses": {
                "read_receipt_under_5s": "met",
                "session_cost_under_0_0045": "met",
                "modal_idle_hourly_cost_under_0_005": "met",
                "cache_hit_rate_over_0_30": "met",
                "browser_single_ai_call_completion_rate_over_0_50": "not_met",
                "capability_match_rate_equals_1_00": "not_met",
                "preferred_model_selection_accuracy_over_0_95": "not_met",
            },
            "browser_single_ai_call_completion_rate": "0.25",
            "capability_match_rate": "0.75",
            "preferred_model_selection_accuracy": "0.80",
        },
        "priority_blockers": [],
        "next_actions": [],
    }

    checklist = _derive_checklist(
        snapshot=snapshot,
        delivery_result=None,
        user_send_result=None,
        target_chat_id="",
    )

    fx005 = next(item for item in checklist["tasks"] if item["id"] == "FX005")
    assert fx005["category"] == "routing-validation"
    assert fx005["status"] == "todo"
    assert "browser_single_ai_call_completion_rate" in "".join(fx005["evidence"])


def test_derive_checklist_prefers_delivery_ready_app_even_when_visible_chat_count_is_zero(monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID3", "cli_app3")
    snapshot = {
        "status": "attention",
        "summary": {
            "statuses": {
                "read_receipt_under_5s": "not_met",
                "session_cost_under_0_0045": "met",
                "modal_idle_hourly_cost_under_0_005": "met",
                "cache_hit_rate_over_0_30": "met",
                "browser_single_ai_call_completion_rate_over_0_50": "met",
                "capability_match_rate_equals_1_00": "met",
                "preferred_model_selection_accuracy_over_0_95": "met",
            },
            "read_receipt_p90_ms": None,
        },
        "priority_blockers": ["kpi:read_receipt"],
        "next_actions": [],
    }
    delivery_result = {
        "payload": {
            "status": "error",
            "summary": "No audited Feishu app can currently see any target chats, so live delivery cannot succeed.",
            "apps": [
                {
                    "app_id": "cli_legacy",
                    "visible_chat_count": 0,
                    "delivery_ready": False,
                    "published_version_summary": {
                        "missing_required_callbacks": ["im.message.message_read_v1", "card.action.trigger"],
                    },
                    "chat_topology": {},
                    "recent_chat_activity": {},
                },
                {
                    "app_id": "cli_app3",
                    "visible_chat_count": 0,
                    "delivery_ready": True,
                    "published_version_summary": {"missing_required_callbacks": ["card.action.trigger"]},
                    "chat_topology": {},
                    "recent_chat_activity": {},
                },
            ],
        }
    }

    checklist = _derive_checklist(
        snapshot=snapshot,
        delivery_result=delivery_result,
        user_send_result=None,
        target_chat_id="oc_test",
    )

    assert checklist["summary"]["canonical_app_id"] == "cli_app3"
