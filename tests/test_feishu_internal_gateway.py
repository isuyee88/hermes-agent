from __future__ import annotations

from pathlib import Path

import modal_


def _collect_action_names(card: dict) -> set[str]:
    actions: set[str] = set()
    for element in card.get("elements", []):
        if not isinstance(element, dict):
            continue
        for action in element.get("actions", []):
            value = action.get("value", {}) if isinstance(action, dict) else {}
            if isinstance(value, dict):
                name = str(value.get("hermes_action") or "").strip()
                if name:
                    actions.add(name)
    return actions


def test_model_picker_local_menu_card_exposes_navigation_buttons() -> None:
    card = modal_._build_feishu_local_menu_card("model_picker")
    assert card is not None
    assert card["header"]["title"]["content"] == "Hermes Model Hub"
    actions = _collect_action_names(card)
    assert "registry_switch_model" in actions
    assert "registry_close_card" in actions


def test_internal_result_file_registry_round_trip(tmp_path: Path) -> None:
    sample = tmp_path / "artifact.txt"
    sample.write_text("hello", encoding="utf-8")

    ticket = modal_._register_feishu_internal_result_file(str(sample), kind="document_file")
    assert ticket is not None
    looked_up = modal_._lookup_feishu_internal_result_file(str(ticket["token"]))
    assert looked_up is not None
    assert looked_up["path"] == str(sample)
    assert looked_up["filename"] == "artifact.txt"
    assert looked_up["kind"] == "document_file"
