"""Tests for Feishu business tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx

from tools.feishu_api import (
    build_model_registry,
    check_feishu_available,
    extract_document_id,
    get_model_registry_path,
    mirror_model_registry_to_bitable,
    normalize_document_summary,
)
from tools.feishu_tools import (
    feishu_bitable_get_schema_tool,
    feishu_bitable_list_records_tool,
    feishu_bitable_upsert_records_tool,
    feishu_chat_lookup_tool,
    feishu_doc_append_markdown_tool,
    feishu_doc_create_tool,
    feishu_doc_get_tool,
    feishu_file_download_tool,
    feishu_file_send_tool,
    feishu_file_upload_tool,
    feishu_lookup_user_tool,
    feishu_message_send_tool,
    feishu_model_registry_publish_card_tool,
    feishu_model_registry_sync_tool,
    feishu_sheet_create_tool,
    feishu_sheet_read_range_tool,
    feishu_sheet_write_range_tool,
)


class FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []

    def request_json(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        raise NotImplementedError

    def request_bytes(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        raise NotImplementedError

    def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))
        raise NotImplementedError

    def upload_im_file(self, **kwargs):
        self.calls.append(("upload_im_file", kwargs))
        raise NotImplementedError

    def send_uploaded_file_message(self, **kwargs):
        self.calls.append(("send_uploaded_file_message", kwargs))
        raise NotImplementedError


class TestFeishuAvailability:
    @patch.dict("os.environ", {}, clear=True)
    def test_check_feishu_available_false_without_credentials(self):
        assert check_feishu_available() is False

    @patch.dict(
        "os.environ",
        {"FEISHU_APP_ID": "cli_test", "FEISHU_APP_SECRET": "secret_test"},
        clear=True,
    )
    def test_check_feishu_available_true_with_credentials(self):
        assert check_feishu_available() is True


class TestDocumentHelpers:
    def test_extract_document_id_from_url(self):
        assert (
            extract_document_id("https://example.feishu.cn/docx/doxcnAbCdEf12345")
            == "doxcnAbCdEf12345"
        )

    def test_extract_document_id_from_token(self):
        assert extract_document_id("doxcnAbCdEf12345") == "doxcnAbCdEf12345"

    def test_normalize_document_summary_truncates_raw_content(self):
        result = normalize_document_summary(
            "doxcnAbCdEf12345",
            {"document": {"title": "Demo", "url": "https://example/docx/doxcnAbCdEf12345"}},
            "x" * 12_500,
        )
        assert result["title"] == "Demo"
        assert result["raw_content_truncated"] is True
        assert len(result["raw_content"]) == 12_000


class TestFeishuDocTools:
    def test_doc_create_returns_document_metadata(self):
        client = FakeClient()
        client.request_json = lambda *_args, **_kwargs: {
            "document": {
                "document_id": "doxcnCreated123",
                "title": "Test Doc",
                "url": "https://open.feishu.cn/docx/doxcnCreated123",
            }
        }
        with patch("tools.feishu_tools._client", return_value=client):
            result = json.loads(feishu_doc_create_tool({"title": "Test Doc"}))

        assert result["success"] is True
        assert result["document_id"] == "doxcnCreated123"

    def test_doc_get_reads_info_and_raw_content(self):
        client = FakeClient()
        responses = iter(
            [
                {"document": {"title": "Project Notes", "url": "https://open.feishu.cn/docx/doxcnProject123"}},
                {"content": "Line 1\nLine 2"},
            ]
        )
        client.request_json = lambda *_args, **_kwargs: next(responses)
        with patch("tools.feishu_tools._client", return_value=client):
            result = json.loads(feishu_doc_get_tool({"document_id_or_url": "https://open.feishu.cn/docx/doxcnProject123"}))

        assert result["success"] is True
        assert result["document_id"] == "doxcnProject123"
        assert result["raw_content"] == "Line 1\nLine 2"

    def test_doc_append_markdown_reports_inserted_blocks(self):
        client = FakeClient()
        client.request_json = lambda *_args, **_kwargs: {"document_revision_id": 7}
        with patch("tools.feishu_tools._client", return_value=client):
            result = json.loads(
                feishu_doc_append_markdown_tool(
                    {"document_id_or_url": "doxcnProject123", "markdown": "First line\nSecond line"}
                )
            )

        assert result["success"] is True
        assert result["inserted_blocks"] == 2


class TestFeishuLookupAndSheets:
    def test_lookup_user_by_open_id(self):
        client = FakeClient()
        client.request_json = lambda *_args, **_kwargs: {
            "user": {
                "open_id": "ou_abc",
                "user_id": "ou_abc",
                "name": "Alice",
                "enterprise_email": "alice@example.com",
            }
        }
        with patch("tools.feishu_tools._client", return_value=client), patch(
            "tools.feishu_tools.resolve_user_identifier",
            return_value=("ou_abc", "open_id"),
        ):
            result = json.loads(feishu_lookup_user_tool({"open_id": "ou_abc"}))

        assert result["success"] is True
        assert result["user"]["resolved_via"] == "open_id"

    def test_sheet_read_and_write_range(self):
        client = FakeClient()
        responses = iter(
            [
                {"valueRange": {"values": [["A1", "B1"]]}},
                {"updatedRange": "Sheet1!A1:B1", "updatedRows": 1},
            ]
        )
        client.request_json = lambda *_args, **_kwargs: next(responses)
        with patch("tools.feishu_tools._client", return_value=client):
            read_result = json.loads(
                feishu_sheet_read_range_tool(
                    {"spreadsheet_token_or_url": "shtcn123", "range": "Sheet1!A1:B1"}
                )
            )
            write_result = json.loads(
                feishu_sheet_write_range_tool(
                    {
                        "spreadsheet_token_or_url": "shtcn123",
                        "range": "Sheet1!A1:B1",
                        "values": [["A1", "B1"]],
                    }
                )
            )

        assert read_result["values"] == [["A1", "B1"]]
        assert write_result["updated_rows"] == 1

    def test_sheet_create_returns_token(self):
        client = FakeClient()
        client.request_json = lambda *_args, **_kwargs: {
            "spreadsheet": {
                "spreadsheet_token": "shtcnCreated123",
                "title": "Sheet Title",
            }
        }
        with patch("tools.feishu_tools._client", return_value=client):
            result = json.loads(feishu_sheet_create_tool({"title": "Sheet Title"}))

        assert result["success"] is True
        assert result["spreadsheet_token"] == "shtcnCreated123"


class TestFeishuBitableAndMessages:
    def test_bitable_get_list_and_upsert(self):
        client = FakeClient()
        responses = iter(
            [
                {"table": {"table_id": "tbl1", "name": "Registry"}},
                {"items": [{"field_name": "Provider"}, {"field_name": "Model"}]},
                {"items": [{"record_id": "rec1", "fields": {"Provider": "openrouter", "Model": "x"}}], "has_more": False},
                {"record": {"record_id": "rec2", "fields": {"Provider": "nvidia"}}},
                {"record": {"record_id": "rec1", "fields": {"Provider": "openrouter"}}},
            ]
        )
        client.request_json = lambda *_args, **_kwargs: next(responses)
        with patch("tools.feishu_tools._client", return_value=client), patch.dict(
            "os.environ",
            {"FEISHU_BITABLE_APP_TOKEN": "app_token", "FEISHU_BITABLE_TABLE_ID": "tbl1"},
            clear=False,
        ):
            schema = json.loads(feishu_bitable_get_schema_tool({}))
            listed = json.loads(feishu_bitable_list_records_tool({}))
            upserted = json.loads(
                feishu_bitable_upsert_records_tool(
                    {
                        "records": [
                            {"fields": {"Provider": "nvidia"}},
                            {"record_id": "rec1", "fields": {"Provider": "openrouter"}},
                        ]
                    }
                )
            )

        assert schema["success"] is True
        assert listed["items"][0]["record_id"] == "rec1"
        assert upserted["created"] == 1
        assert upserted["updated"] == 1

    def test_message_send_and_chat_lookup(self):
        client = FakeClient()
        client.send_message = lambda **_kwargs: {"message_id": "om_123"}
        client.request_json = lambda *_args, **_kwargs: {"chat": {"chat_id": "oc_123", "name": "Demo"}}
        with patch("tools.feishu_tools._client", return_value=client):
            sent = json.loads(
                feishu_message_send_tool(
                    {"chat_id": "oc_123", "message": "Hello", "msg_type": "post", "title": "Greeting"}
                )
            )
            chat = json.loads(feishu_chat_lookup_tool({"chat_id": "oc_123"}))

        assert sent["success"] is True
        assert sent["message_id"] == "om_123"
        assert chat["chat"]["chat_id"] == "oc_123"


class TestFeishuFilesAndRegistry:
    def test_build_model_registry_enriches_commands_and_recent_usage(self, tmp_path):
        routing_state = {
            "refreshed_at": 1700000000,
            "providers": {
                "openrouter": {"candidates": ["openai/gpt-oss-120b", "google/gemma-3-27b-it:free"]},
                "nvidia": {"candidates": ["qwen/qwq-32b"]},
            },
        }
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "s1.json").write_text(
            json.dumps(
                {
                    "updated_at": 1700000100,
                    "route_lease": {
                        "provider": "nvidia",
                        "model": "qwen/qwq-32b",
                        "fail_count": 0,
                    },
                    "route_debug": {"last_error": "", "last_failure_reason": ""},
                }
            ),
            encoding="utf-8",
        )
        (sessions_dir / "s2.json").write_text(
            json.dumps(
                {
                    "updated_at": 1700000200,
                    "route_lease": {
                        "provider": "openrouter",
                        "model": "google/gemma-3-27b-it:free",
                        "fail_count": 2,
                    },
                    "route_debug": {
                        "last_error": "invalid model returned by upstream",
                        "last_failure_reason": "invalid_model",
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch("tools.feishu_api.get_routing_state_path", return_value=tmp_path / "free_model_routing.json"), patch(
            "tools.feishu_api.get_model_registry_path", return_value=tmp_path / "feishu_model_registry.json"
        ), patch("tools.feishu_api.get_sessions_dir", return_value=sessions_dir), patch(
            "tools.feishu_api.load_json",
            side_effect=lambda path, default: routing_state if str(path).endswith("free_model_routing.json") else json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default,
        ):
            payload = build_model_registry(force_refresh=True)

        assert payload["schema_version"] >= 2
        by_key = {(item["provider"], item["model"]): item for item in payload["entries"]}
        nvidia = by_key[("nvidia", "qwen/qwq-32b")]
        assert nvidia["generated_command"] == "/model qwen/qwq-32b --provider nvidia"
        assert nvidia["recent_used"] is True
        assert nvidia["recent_used_count"] == 1
        assert nvidia["status"] == "active"

        invalid = by_key[("openrouter", "google/gemma-3-27b-it:free")]
        assert invalid["hidden"] is True
        assert invalid["is_available"] is False
        assert invalid["failure_kind"] == "invalid_model"
        assert invalid["status"] == "invalid"

    def test_mirror_model_registry_to_bitable_marks_missing_records_hidden(self):
        registry_payload = {
            "generated_at": 1700000001,
            "entries": [
                {
                    "provider": "openrouter",
                    "model": "openai/gpt-oss-120b",
                    "display_name": "GPT OSS 120B",
                    "is_free": False,
                    "is_available": True,
                    "rank": 1,
                    "last_probe_at": 1700000000,
                    "latency_ms": None,
                    "context_window": 128000,
                    "reasoning": True,
                    "manual_pinned": True,
                    "selection_hint": "recommended",
                    "status": "active",
                    "hidden": False,
                    "recent_used": True,
                    "recent_used_count": 3,
                    "recent_used_at": 1700000001,
                    "generated_command": "/model openai/gpt-oss-120b --provider openrouter",
                    "last_error_code": "",
                    "last_error_message": "",
                    "last_failed_at": None,
                    "consecutive_failures": 0,
                    "failure_kind": None,
                }
            ],
        }

        client = FakeClient()
        responses = iter(
            [
                {
                    "items": [
                        {"field_name": "Provider"},
                        {"field_name": "Model"},
                        {"field_name": "Display Name"},
                        {"field_name": "Status"},
                        {"field_name": "Hidden"},
                        {"field_name": "Is Available"},
                        {"field_name": "Generated Command"},
                        {"field_name": "Last Error Message"},
                        {"field_name": "Failure Kind"},
                        {"field_name": "Last Sync At"},
                    ]
                },
                {
                    "items": [
                        {"record_id": "rec_keep", "fields": {"Provider": "openrouter", "Model": "openai/gpt-oss-120b"}},
                        {"record_id": "rec_old", "fields": {"Provider": "nvidia", "Model": "old/model"}},
                    ],
                    "has_more": False,
                },
                {"record": {"record_id": "rec_keep"}},
                {"record": {"record_id": "rec_old"}},
            ]
        )
        def _request_json(method, path, **kwargs):
            client.calls.append((method, path, kwargs))
            return next(responses)

        client.request_json = _request_json

        result = mirror_model_registry_to_bitable(
            client,
            registry_payload,
            app_token="app_token",
            table_id="tbl1",
        )

        assert result["mirrored"] is True
        assert result["updated"] == 1
        assert result["hidden"] == 1
        put_calls = [call for call in client.calls if call[0] == "PUT"]
        assert len(put_calls) == 2
        hidden_payload = put_calls[-1][2]["json_body"]["fields"]
        assert hidden_payload["Status"] == "inactive"
        assert hidden_payload["Hidden"] is True
        assert hidden_payload["Failure Kind"] == "not_in_snapshot"

    def test_file_upload_send_and_download(self, tmp_path):
        source = tmp_path / "report.txt"
        source.write_text("demo", encoding="utf-8")
        target = tmp_path / "downloaded.txt"

        client = FakeClient()
        client.upload_im_file = lambda **_kwargs: {"file_key": "file_123"}
        client.send_uploaded_file_message = lambda **_kwargs: {"message_id": "om_file_123"}
        client.request_bytes = lambda *_args, **_kwargs: (b"downloaded", httpx.Headers({"content-type": "text/plain"}))

        with patch("tools.feishu_tools._client", return_value=client), patch(
            "tools.feishu_tools.build_download_target_path",
            return_value=target,
        ):
            uploaded = json.loads(feishu_file_upload_tool({"file_path": str(source)}))
            sent = json.loads(feishu_file_send_tool({"chat_id": "oc_123", "file_path": str(source)}))
            downloaded = json.loads(
                feishu_file_download_tool({"message_id": "om_file_123", "file_key": "file_123"})
            )

        assert uploaded["file_key"] == "file_123"
        assert sent["message_id"] == "om_file_123"
        assert downloaded["local_path"] == str(target)
        assert target.read_bytes() == b"downloaded"

    def test_model_registry_sync_and_publish_card(self, tmp_path):
        registry_payload = {
            "status": "ok",
            "generated_at": 123,
            "refreshed_at": 123,
            "source": "routing_state",
            "entries": [
                {"provider": "openrouter", "model": "m1", "is_available": True, "is_free": True, "selection_hint": "recommended"},
                {"provider": "nvidia", "model": "m2", "is_available": True, "is_free": True, "selection_hint": "fallback"},
            ],
        }
        client = FakeClient()
        client.send_message = lambda **_kwargs: {"message_id": "om_card_123"}
        with patch("tools.feishu_tools._client", return_value=client), patch(
            "tools.feishu_tools.build_model_registry",
            return_value=registry_payload,
        ), patch(
            "tools.feishu_tools.get_model_registry_path",
            return_value=tmp_path / "feishu_model_registry.json",
        ), patch(
            "tools.feishu_tools.mirror_model_registry_to_bitable",
            return_value={"success": True, "updated": 2, "hidden": 0},
        ), patch.dict(
            "os.environ",
            {"FEISHU_BITABLE_APP_TOKEN": "app_token", "FEISHU_BITABLE_TABLE_ID": "tbl1"},
            clear=False,
        ):
            synced = json.loads(
                feishu_model_registry_sync_tool({"force_refresh": True, "mirror_to_bitable": True})
            )
            published = json.loads(
                feishu_model_registry_publish_card_tool({"chat_id": "oc_123", "top_n": 3})
            )

        assert synced["success"] is True
        assert synced["registry_path"] == str(tmp_path / "feishu_model_registry.json")
        assert synced["bitable_mirror"]["updated"] == 2
        assert published["message_id"] == "om_card_123"
        assert published["card_providers"] == ["nvidia", "openrouter"]
