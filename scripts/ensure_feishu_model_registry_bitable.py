from __future__ import annotations

import argparse
import json
import os

from tools.feishu_api import build_feishu_client, ensure_model_registry_bitable_schema, resolve_bitable_target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or validate the recommended Feishu Bitable table, fields, and views for Hermes model registry mirroring."
    )
    parser.add_argument("--app-token", default="", help="Feishu Bitable app token, or a wiki/base URL that resolves into one. Falls back to FEISHU_BITABLE_APP_TOKEN.")
    parser.add_argument("--wiki-token", default="", help="Optional wiki token when the Bitable is mounted under wiki. Falls back to FEISHU_BITABLE_WIKI_TOKEN.")
    parser.add_argument("--bitable-url", default="", help="Optional wiki/base URL. Hermes will extract app_token and table_id when possible.")
    parser.add_argument("--table-id", default="", help="Existing table id. Falls back to FEISHU_BITABLE_TABLE_ID. If omitted, the script finds or creates by table name.")
    parser.add_argument("--table-name", default="Hermes Model Registry", help="Target table name.")
    parser.add_argument("--no-create-table", action="store_true", help="Validate only; do not create the table.")
    parser.add_argument("--no-create-fields", action="store_true", help="Validate only; do not create missing fields.")
    parser.add_argument("--no-create-views", action="store_true", help="Validate only; do not create missing views.")
    args = parser.parse_args()
    client = build_feishu_client()
    try:
        app_token, table_id = resolve_bitable_target(
            {
                "app_token": args.app_token,
                "wiki_token": args.wiki_token,
                "bitable_url": args.bitable_url,
                "table_id": args.table_id,
            },
            client,
            require_table_id=False,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    result = ensure_model_registry_bitable_schema(
        client,
        app_token=app_token,
        table_id=table_id or None,
        table_name=args.table_name,
        create_missing_table=not args.no_create_table,
        create_missing_fields=not args.no_create_fields,
        create_missing_views=not args.no_create_views,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
