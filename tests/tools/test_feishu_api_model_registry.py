from __future__ import annotations

import importlib


def test_load_feishu_model_registry_prefers_bitable(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_MODAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret-123")
    monkeypatch.setenv("FEISHU_BITABLE_APP_TOKEN", "app_token_123")
    monkeypatch.setenv("FEISHU_BITABLE_TABLE_ID", "tbl_123")

    feishu_api = importlib.import_module("tools.feishu_api")
    feishu_api = importlib.reload(feishu_api)

    class _FakeClient:
        def request_json(self, method, path, *, params=None, json_body=None, headers=None, retries=3):
            del method, path, params, json_body, headers, retries
            return {
                "items": [
                    {
                        "record_id": "rec_1",
                        "fields": {
                            "Provider": "nvidia",
                            "Model": "qwen/qwq-32b",
                            "Display Name": "QwQ 32B",
                            "Introduction": "Reasoning model sourced from Feishu Bitable.",
                            "Selection Hint": "recommended",
                            "Rank": 1,
                        },
                    }
                ],
                "has_more": False,
            }

    monkeypatch.setattr(feishu_api, "build_feishu_client", lambda timeout=None: _FakeClient())
    monkeypatch.setattr(feishu_api, "resolve_bitable_target", lambda _args, _client: ("app_token_123", "tbl_123"))
    monkeypatch.setattr(
        feishu_api,
        "build_model_registry",
        lambda force_refresh=False: (_ for _ in ()).throw(AssertionError("fallback should not be used")),
    )

    payload = feishu_api.load_feishu_model_registry(force_refresh=True)

    assert payload["source"] == "bitable"
    assert payload["entries"][0]["provider"] == "nvidia"
    assert payload["entries"][0]["model"] == "qwen/qwq-32b"
    assert payload["entries"][0]["introduction"] == "Reasoning model sourced from Feishu Bitable."
