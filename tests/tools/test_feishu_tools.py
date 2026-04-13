"""Tests for Feishu business tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx

from tools.feishu_api import (
    FeishuOpenApiClient,
    build_model_registry,
    bootstrap_model_registry_bitable,
    check_feishu_available,
    ensure_model_registry_bitable_schema,
    extract_bitable_reference,
    extract_document_id,
    get_model_registry_path,
    mirror_model_registry_to_bitable,
    normalize_document_summary,
    resolve_bitable_target,
)
from tools.feishu_tools import (
    feishu_bitable_get_schema_tool,
    feishu_bitable_list_records_tool,
    feishu_bitable_upsert_records_tool,
    feishu_chat_lookup_tool,
    feishu_doc_append_markdown_tool,
    feishu_doc_create_tool,
    feishu_doc_get_tool,
    feishu_doc_replace_markdown_tool,
    feishu_audio_send_tool,
    feishu_file_download_tool,
    feishu_image_send_tool,
    feishu_image_upload_tool,
    feishu_file_send_tool,
    feishu_file_upload_tool,
    feishu_lookup_user_tool,
    feishu_message_send_tool,
    feishu_model_registry_list_tool,
    feishu_model_registry_bootstrap_bitable_tool,
    feishu_model_registry_publish_card_tool,
    feishu_model_registry_prepare_bitable_tool,
    feishu_model_registry_sync_tool,
    feishu_sheet_create_tool,
    feishu_sheet_read_range_tool,
    feishu_sheet_write_range_tool,
    feishu_video_send_tool,
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

    def test_extract_bitable_reference_from_wiki_url(self):
        assert extract_bitable_reference(
            "wiki/B5tywV8uLiSGZckFCNvckOpMn6g?table=tblUtsdxN5HTFztl&view=vewMiw6ttA"
        ) == {
            "wiki_token": "B5tywV8uLiSGZckFCNvckOpMn6g",
            "table_id": "tblUtsdxN5HTFztl",
            "view_id": "vewMiw6ttA",
        }

    def test_extract_bitable_reference_from_base_url(self):
        assert extract_bitable_reference("https://example.feishu.cn/base/appAbc123?table=tbl001") == {
            "app_token": "appAbc123",
            "table_id": "tbl001",
        }

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

    def test_doc_replace_markdown_clears_existing_children_then_inserts(self):
        client = FakeClient()
        responses = iter(
            [
                {"block": {"block_id": "doxcnProject123", "children": ["blk_1", "blk_2"]}},
                {"document_revision_id": 8},
                {"document_revision_id": 9},
            ]
        )

        def _request_json(method, path, **kwargs):
            client.calls.append((method, path, kwargs))
            return next(responses)

        client.request_json = _request_json
        with patch("tools.feishu_tools._client", return_value=client):
            result = json.loads(
                feishu_doc_replace_markdown_tool(
                    {"document_id_or_url": "doxcnProject123", "markdown": "Replacement line 1\nReplacement line 2"}
                )
            )

        assert result["success"] is True
        assert result["update_mode"] == "replace"
        assert result["cleared_blocks"] == 2
        assert result["inserted_blocks"] == 2
        assert client.calls == [
            (
                "GET",
                "/open-apis/docx/v1/documents/doxcnProject123/blocks/doxcnProject123",
                {},
            ),
            (
                "DELETE",
                "/open-apis/docx/v1/documents/doxcnProject123/blocks/doxcnProject123/children/batch_delete",
                {"json_body": {"start_index": 0, "end_index": 1}},
            ),
            (
                "POST",
                "/open-apis/docx/v1/documents/doxcnProject123/blocks/doxcnProject123/children",
                {"json_body": {"children": [
                    {"block_type": 2, "paragraph": {"elements": [{"text_run": {"content": "Replacement line 1"}, "type": "text_run"}]}},
                    {"block_type": 2, "paragraph": {"elements": [{"text_run": {"content": "Replacement line 2"}, "type": "text_run"}]}},
                ], "index": 0}},
            ),
        ]

    def test_doc_replace_markdown_supports_clear_only(self):
        client = FakeClient()
        responses = iter(
            [
                {"block": {"block_id": "doxcnProject123", "children": ["blk_1"]}},
                {"document_revision_id": 11},
            ]
        )

        def _request_json(method, path, **kwargs):
            client.calls.append((method, path, kwargs))
            return next(responses)

        client.request_json = _request_json
        with patch("tools.feishu_tools._client", return_value=client):
            result = json.loads(
                feishu_doc_replace_markdown_tool(
                    {"document_id_or_url": "doxcnProject123", "markdown": ""}
                )
            )

        assert result["success"] is True
        assert result["cleared_only"] is True
        assert result["inserted_blocks"] == 0


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
    @patch.dict(
        "os.environ",
        {"FEISHU_BITABLE_WIKI_TOKEN": "wiki_123", "FEISHU_BITABLE_TABLE_ID": "tbl_from_env"},
        clear=False,
    )
    def test_resolve_bitable_target_from_wiki_token(self):
        client = FakeClient()

        def _request_json(method, path, **kwargs):
            client.calls.append((method, path, kwargs))
            return {"node": {"obj_type": "bitable", "obj_token": "app_resolved_123"}}

        client.request_json = _request_json

        app_token, table_id = resolve_bitable_target({}, client)

        assert app_token == "app_resolved_123"
        assert table_id == "tbl_from_env"
        assert client.calls == [
            (
                "GET",
                "/open-apis/wiki/v2/spaces/get_node",
                {"params": {"token": "wiki_123"}},
            )
        ]

    def test_resolve_bitable_target_prefers_direct_app_token(self):
        client = FakeClient()

        app_token, table_id = resolve_bitable_target(
            {
                "app_token": "https://example.feishu.cn/base/app_direct_123?table=tbl_inline",
                "table_id": "tbl_override",
            },
            client,
        )

        assert app_token == "app_direct_123"
        assert table_id == "tbl_override"
        assert client.calls == []

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

    def test_ensure_model_registry_bitable_schema_creates_table_fields_and_views(self):
        client = FakeClient()
        responses = iter(
            [
                {"items": [], "has_more": False},
                {"table": {"table_id": "tbl_new", "name": "Hermes Model Registry"}},
                {"items": [{"field_name": "Model"}, {"field_name": "Provider"}]},
                {"field": {"field_id": "fld_display", "field_name": "Display Name"}},
                {"field": {"field_id": "fld_status", "field_name": "Status"}},
                {"field": {"field_id": "fld_hidden", "field_name": "Hidden"}},
                {"field": {"field_id": "fld_available", "field_name": "Is Available"}},
                {"field": {"field_id": "fld_free", "field_name": "Is Free"}},
                {"field": {"field_id": "fld_rank", "field_name": "Rank"}},
                {"field": {"field_id": "fld_hint", "field_name": "Selection Hint"}},
                {"field": {"field_id": "fld_pinned", "field_name": "Manual Pinned"}},
                {"field": {"field_id": "fld_recent", "field_name": "Recent Used"}},
                {"field": {"field_id": "fld_recent_count", "field_name": "Recent Used Count"}},
                {"field": {"field_id": "fld_command", "field_name": "Generated Command"}},
                {"field": {"field_id": "fld_probe", "field_name": "Last Probe At"}},
                {"field": {"field_id": "fld_recent_at", "field_name": "Recent Used At"}},
                {"field": {"field_id": "fld_sync", "field_name": "Last Sync At"}},
                {"field": {"field_id": "fld_latency", "field_name": "Latency Ms"}},
                {"field": {"field_id": "fld_context", "field_name": "Context Window"}},
                {"field": {"field_id": "fld_reasoning", "field_name": "Reasoning"}},
                {"field": {"field_id": "fld_failures", "field_name": "Consecutive Failures"}},
                {"field": {"field_id": "fld_failure_kind", "field_name": "Failure Kind"}},
                {"field": {"field_id": "fld_error_code", "field_name": "Last Error Code"}},
                {"field": {"field_id": "fld_error_message", "field_name": "Last Error Message"}},
                {"field": {"field_id": "fld_failed_at", "field_name": "Last Failed At"}},
                {"field": {"field_id": "fld_source", "field_name": "Source"}},
                {"items": [], "has_more": False},
                {"view": {"view_id": "vew_all", "view_name": "All Models"}},
                {"view": {"view_id": "vew_recommended", "view_name": "Recommended"}},
                {"view": {"view_id": "vew_recent", "view_name": "Recent Used"}},
                {"view": {"view_id": "vew_hidden", "view_name": "Hidden or Inactive"}},
            ]
        )

        def _request_json(method, path, **kwargs):
            client.calls.append((method, path, kwargs))
            return next(responses)

        client.request_json = _request_json

        result = ensure_model_registry_bitable_schema(
            client,
            app_token="app_token",
            table_name="Hermes Model Registry",
            create_missing_table=True,
            create_missing_fields=True,
            create_missing_views=True,
        )

        assert result["status"] == "ok"
        assert result["created_table"] is True
        assert result["table_id"] == "tbl_new"
        assert len(result["created_fields"]) >= 1
        assert [item["view_name"] for item in result["created_views"]] == [
            "All Models",
            "Recommended",
            "Recent Used",
            "Hidden or Inactive",
        ]

    def test_bootstrap_model_registry_bitable_creates_dedicated_app_and_schema(self):
        client = FakeClient()
        responses = iter(
            [
                {
                    "app": {
                        "app_token": "app_bootstrap_123",
                        "default_table_id": "tbl_default_123",
                        "url": "https://example.feishu.cn/base/app_bootstrap_123",
                    }
                },
                {"items": [{"table_id": "tbl_default_123", "name": "数据表"}], "has_more": False},
                {"items": [{"field_name": "文本"}]},
                {"field": {"field_id": "fld_model", "field_name": "Model"}},
                {"field": {"field_id": "fld_provider", "field_name": "Provider"}},
                {"field": {"field_id": "fld_display", "field_name": "Display Name"}},
                {"field": {"field_id": "fld_status", "field_name": "Status"}},
                {"field": {"field_id": "fld_hidden", "field_name": "Hidden"}},
                {"field": {"field_id": "fld_available", "field_name": "Is Available"}},
                {"field": {"field_id": "fld_free", "field_name": "Is Free"}},
                {"field": {"field_id": "fld_rank", "field_name": "Rank"}},
                {"field": {"field_id": "fld_hint", "field_name": "Selection Hint"}},
                {"field": {"field_id": "fld_pinned", "field_name": "Manual Pinned"}},
                {"field": {"field_id": "fld_recent", "field_name": "Recent Used"}},
                {"field": {"field_id": "fld_recent_count", "field_name": "Recent Used Count"}},
                {"field": {"field_id": "fld_command", "field_name": "Generated Command"}},
                {"field": {"field_id": "fld_probe", "field_name": "Last Probe At"}},
                {"field": {"field_id": "fld_recent_at", "field_name": "Recent Used At"}},
                {"field": {"field_id": "fld_sync", "field_name": "Last Sync At"}},
                {"field": {"field_id": "fld_latency", "field_name": "Latency Ms"}},
                {"field": {"field_id": "fld_context", "field_name": "Context Window"}},
                {"field": {"field_id": "fld_reasoning", "field_name": "Reasoning"}},
                {"field": {"field_id": "fld_failures", "field_name": "Consecutive Failures"}},
                {"field": {"field_id": "fld_failure_kind", "field_name": "Failure Kind"}},
                {"field": {"field_id": "fld_error_code", "field_name": "Last Error Code"}},
                {"field": {"field_id": "fld_error_message", "field_name": "Last Error Message"}},
                {"field": {"field_id": "fld_failed_at", "field_name": "Last Failed At"}},
                {"field": {"field_id": "fld_source", "field_name": "Source"}},
                {"items": [], "has_more": False},
                {"view": {"view_id": "vew_all", "view_name": "All Models"}},
                {"view": {"view_id": "vew_recommended", "view_name": "Recommended"}},
                {"view": {"view_id": "vew_recent", "view_name": "Recent Used"}},
                {"view": {"view_id": "vew_hidden", "view_name": "Hidden or Inactive"}},
            ]
        )

        def _request_json(method, path, **kwargs):
            client.calls.append((method, path, kwargs))
            return next(responses)

        client.request_json = _request_json

        result = bootstrap_model_registry_bitable(
            client,
            app_name="Hermes Dedicated Mirror",
            table_name="Hermes Model Registry",
        )

        assert result["status"] == "ok"
        assert result["env"] == {
            "FEISHU_BITABLE_APP_TOKEN": "app_bootstrap_123",
            "FEISHU_BITABLE_TABLE_ID": "tbl_default_123",
        }
        assert result["schema"]["table_id"] == "tbl_default_123"
        assert result["app"]["app_token"] == "app_bootstrap_123"

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

    def test_message_send_auto_detects_open_id_target(self):
        client = FakeClient()

        def _send_message(**kwargs):
            client.calls.append(("send_message", kwargs))
            return {"message_id": "om_open_123"}

        client.send_message = _send_message

        with patch("tools.feishu_tools._client", return_value=client):
            sent = json.loads(
                feishu_message_send_tool(
                    {"chat_id": "ou_1234567890", "message": "Hello from menu fallback", "msg_type": "text"}
                )
            )

        assert sent["success"] is True
        assert sent["receive_id"] == "ou_1234567890"
        assert sent["receive_id_type"] == "open_id"
        assert client.calls == [
            (
                "send_message",
                {
                    "receive_id": "ou_1234567890",
                    "receive_id_type": "open_id",
                    "msg_type": "text",
                    "content": json.dumps({"text": "Hello from menu fallback"}, ensure_ascii=False),
                },
            )
        ]

    def test_message_send_supports_resource_messages(self):
        client = FakeClient()

        def _send_message(**kwargs):
            client.calls.append(("send_message", kwargs))
            return {"message_id": "om_resource_123"}

        client.send_message = _send_message

        with patch("tools.feishu_tools._client", return_value=client):
            image_result = json.loads(
                feishu_message_send_tool(
                    {
                        "receive_id": "oc_123",
                        "msg_type": "image",
                        "image_key": "img_123",
                    }
                )
            )
            audio_result = json.loads(
                feishu_message_send_tool(
                    {
                        "receive_id": "u_123",
                        "msg_type": "audio",
                        "file_key": "file_audio_123",
                    }
                )
            )

        assert image_result["success"] is True
        assert image_result["receive_id_type"] == "chat_id"
        assert audio_result["success"] is True
        assert audio_result["receive_id_type"] == "user_id"
        assert client.calls == [
            (
                "send_message",
                {
                    "receive_id": "oc_123",
                    "receive_id_type": "chat_id",
                    "msg_type": "image",
                    "content": json.dumps({"image_key": "img_123"}, ensure_ascii=False),
                },
            ),
            (
                "send_message",
                {
                    "receive_id": "u_123",
                    "receive_id_type": "user_id",
                    "msg_type": "audio",
                    "content": json.dumps({"file_key": "file_audio_123"}, ensure_ascii=False),
                },
            ),
        ]

    def test_api_client_send_message_passes_receive_id_type(self):
        client = object.__new__(FeishuOpenApiClient)
        captured = {}

        def _request_json(method, path, **kwargs):
            captured["method"] = method
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {"message_id": "om_api_123"}

        client.request_json = _request_json

        result = client.send_message(
            receive_id="ou_1234567890",
            msg_type="text",
            content=json.dumps({"text": "hello"}, ensure_ascii=False),
        )

        assert result["message_id"] == "om_api_123"
        assert captured["method"] == "POST"
        assert captured["path"] == "/open-apis/im/v1/messages"
        assert captured["kwargs"]["params"] == {"receive_id_type": "open_id"}
        assert captured["kwargs"]["json_body"]["receive_id"] == "ou_1234567890"


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
        ), patch(
            "agent.models_dev.list_agentic_models",
            side_effect=lambda provider: ["openai/gpt-4.1-mini"] if provider == "openrouter" else ["meta/llama-3.1-70b-instruct"] if provider == "nvidia" else [],
        ):
            payload = build_model_registry(force_refresh=True)

        assert payload["schema_version"] >= 2
        by_key = {(item["provider"], item["model"]): item for item in payload["entries"]}
        nvidia = by_key[("nvidia", "qwen/qwq-32b")]
        assert nvidia["generated_command"] == "/model qwen/qwq-32b --provider nvidia"
        assert nvidia["recent_used"] is True
        assert nvidia["recent_used_count"] == 1
        assert nvidia["status"] == "active"
        assert ("openrouter", "openai/gpt-4.1-mini") in by_key
        assert ("nvidia", "meta/llama-3.1-70b-instruct") in by_key

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

    def test_image_upload_and_send(self, tmp_path):
        source = tmp_path / "chart.png"
        source.write_bytes(b"pngdata")

        client = FakeClient()
        client.upload_im_image = lambda **_kwargs: {"image_key": "img_123"}
        client.send_uploaded_image_message = lambda **_kwargs: {"message_id": "om_img_123"}

        with patch("tools.feishu_tools._client", return_value=client):
            uploaded = json.loads(feishu_image_upload_tool({"file_path": str(source)}))
            sent = json.loads(
                feishu_image_send_tool({"chat_id": "oc_123", "file_path": str(source), "caption": "Latest chart"})
            )

        assert uploaded["image_key"] == "img_123"
        assert sent["image_key"] == "img_123"
        assert sent["message_id"] == "om_img_123"

    def test_audio_and_video_send(self, tmp_path):
        audio = tmp_path / "sample.ogg"
        video = tmp_path / "demo.mp4"
        audio.write_bytes(b"audiodata")
        video.write_bytes(b"videodata")

        client = FakeClient()
        sent_calls = []
        client.upload_im_file = lambda **kwargs: {"file_key": f"file_{kwargs['file_name']}"}

        def _send_uploaded_file_message(**kwargs):
            sent_calls.append(kwargs)
            return {"message_id": f"om_{kwargs['outbound_message_type']}"}

        client.send_uploaded_file_message = _send_uploaded_file_message

        with patch("tools.feishu_tools._client", return_value=client):
            audio_result = json.loads(
                feishu_audio_send_tool({"chat_id": "oc_123", "file_path": str(audio), "caption": "Voice note"})
            )
            video_result = json.loads(
                feishu_video_send_tool({"chat_id": "oc_123", "file_path": str(video), "caption": "Demo clip"})
            )

        assert audio_result["message_id"] == "om_audio"
        assert video_result["message_id"] == "om_media"
        assert sent_calls[0]["outbound_message_type"] == "audio"
        assert sent_calls[1]["outbound_message_type"] == "media"

    def test_model_registry_sync_and_publish_card(self, tmp_path):
        registry_payload = {
            "status": "ok",
            "generated_at": 123,
            "refreshed_at": 123,
            "source": "routing_state",
            "entries": [
                {"provider": "openrouter", "model": "m1", "is_available": True, "is_free": True, "selection_hint": "recommended", "recent_used": True},
                {"provider": "nvidia", "model": "m2", "is_available": True, "is_free": True, "selection_hint": "fallback", "recent_used": False},
            ],
        }
        client = FakeClient()

        def _send_message(**kwargs):
            client.calls.append(("send_message", kwargs))
            return {"message_id": "om_card_123"}

        client.send_message = _send_message
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
        assert published["interaction_mode"] == "registry_switch_model"
        sent = next(call for call in client.calls if call[0] == "send_message")
        card = json.loads(sent[1]["content"])
        action_rows = [item for item in card["elements"] if item.get("tag") == "action"]
        button_values = [action["value"] for row in action_rows for action in row.get("actions", [])]
        assert any(value.get("hermes_action") == "registry_switch_model" for value in button_values)
        assert any(value.get("provider") == "openrouter" and value.get("model") == "m1" for value in button_values)
        assert any(value.get("provider") == "nvidia" and value.get("model") == "m2" for value in button_values)

    def test_model_registry_list_filters_provider_and_recent(self):
        registry_payload = {
            "status": "ok",
            "generated_at": 123,
            "source": "routing_state",
            "entries": [
                {
                    "provider": "openrouter",
                    "model": "m1",
                    "display_name": "Model 1",
                    "status": "active",
                    "is_available": True,
                    "is_free": True,
                    "selection_hint": "recommended",
                    "recent_used": False,
                    "recent_used_count": 0,
                    "generated_command": "/model m1 --provider openrouter",
                },
                {
                    "provider": "nvidia",
                    "model": "m2",
                    "display_name": "Model 2",
                    "status": "active",
                    "is_available": True,
                    "is_free": True,
                    "selection_hint": "candidate",
                    "recent_used": True,
                    "recent_used_count": 4,
                    "generated_command": "/model m2 --provider nvidia",
                },
            ],
        }

        with patch("tools.feishu_tools.build_model_registry", return_value=registry_payload):
            result = json.loads(
                feishu_model_registry_list_tool(
                    {"provider": "nvidia", "recent_only": True, "available_only": True, "limit": 10}
                )
            )

        assert result["success"] is True
        assert result["count"] == 1
        assert result["entries"][0]["provider"] == "nvidia"
        assert result["entries"][0]["model"] == "m2"
        assert result["entries"][0]["generated_command"] == "/model m2 --provider nvidia"

    def test_model_registry_prepare_bitable_accepts_wiki_url(self):
        client = FakeClient()

        def _request_json(method, path, **kwargs):
            client.calls.append((method, path, kwargs))
            return {"node": {"obj_type": "bitable", "obj_token": "app_from_wiki"}}

        client.request_json = _request_json

        with patch("tools.feishu_tools._client", return_value=client), patch(
            "tools.feishu_tools.ensure_model_registry_bitable_schema",
            return_value={"status": "ok", "table_id": "tbl_target"},
        ):
            result = json.loads(
                feishu_model_registry_prepare_bitable_tool(
                    {
                        "bitable_url": "wiki/B5tywV8uLiSGZckFCNvckOpMn6g?table=tblUtsdxN5HTFztl",
                        "table_name": "Hermes Model Registry",
                    }
                )
            )

        assert result["status"] == "ok"
        assert client.calls == [
            (
                "GET",
                "/open-apis/wiki/v2/spaces/get_node",
                {"params": {"token": "B5tywV8uLiSGZckFCNvckOpMn6g"}},
            )
        ]

    def test_model_registry_bootstrap_bitable_tool_returns_env_suggestions(self):
        client = FakeClient()

        with patch("tools.feishu_tools._client", return_value=client), patch(
            "tools.feishu_tools.bootstrap_model_registry_bitable",
            return_value={
                "status": "ok",
                "app": {"app_token": "app_bootstrap_123"},
                "schema": {"table_id": "tbl_bootstrap_123"},
                "env": {
                    "FEISHU_BITABLE_APP_TOKEN": "app_bootstrap_123",
                    "FEISHU_BITABLE_TABLE_ID": "tbl_bootstrap_123",
                },
            },
        ):
            result = json.loads(
                feishu_model_registry_bootstrap_bitable_tool(
                    {
                        "app_name": "Hermes Workbench",
                        "table_name": "Hermes Model Registry",
                        "reuse_default_table": True,
                    }
                )
            )

        assert result["status"] == "ok"
        assert result["env"]["FEISHU_BITABLE_APP_TOKEN"] == "app_bootstrap_123"
        assert result["schema"]["table_id"] == "tbl_bootstrap_123"
