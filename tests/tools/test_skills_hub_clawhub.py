#!/usr/bin/env python3

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.skills_hub import (
    ClawHubSource,
    SkillMeta,
    build_clawhub_migration_queue,
    load_marketplace_migration_queue,
    save_clawhub_migration_queue,
)


class _MockResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class TestClawHubSource(unittest.TestCase):
    def setUp(self):
        self.src = ClawHubSource()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch.object(ClawHubSource, "_load_catalog_index", return_value=[])
    @patch("tools.skills_hub.httpx.get")
    def test_search_uses_listing_endpoint_as_fallback(
        self, mock_get, _mock_load_catalog, _mock_read_cache, _mock_write_cache
    ):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "items": [
                            {
                                "slug": "caldav-calendar",
                                "displayName": "CalDAV Calendar",
                                "summary": "Calendar integration",
                                "tags": ["calendar", "productivity"],
                            }
                        ]
                    },
                )
            if url.endswith("/skills/caldav"):
                return _MockResponse(status_code=404, json_data={})
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        results = self.src.search("caldav", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "caldav-calendar")
        self.assertEqual(results[0].name, "CalDAV Calendar")
        self.assertEqual(results[0].description, "Calendar integration")

        self.assertGreaterEqual(mock_get.call_count, 2)
        args, kwargs = mock_get.call_args_list[0]
        self.assertTrue(args[0].endswith("/skills"))
        self.assertEqual(kwargs["params"], {"search": "caldav", "limit": 5})

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch.object(
        ClawHubSource,
        "_load_catalog_index",
        return_value=[],
    )
    @patch("tools.skills_hub.httpx.get")
    def test_search_falls_back_to_exact_slug_when_search_results_are_irrelevant(
        self, mock_get, _mock_load_catalog, _mock_read_cache, _mock_write_cache
    ):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "items": [
                            {
                                "slug": "apple-music-dj",
                                "displayName": "Apple Music DJ",
                                "summary": "Unrelated result",
                            }
                        ]
                    },
                )
            if url.endswith("/skills/self-improving-agent"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "skill": {
                            "slug": "self-improving-agent",
                            "displayName": "self-improving-agent",
                            "summary": "Captures learnings and errors for continuous improvement.",
                            "tags": {"latest": "3.0.2", "automation": "3.0.2"},
                        },
                        "latestVersion": {"version": "3.0.2"},
                    },
                )
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        results = self.src.search("self-improving-agent", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "self-improving-agent")
        self.assertEqual(results[0].name, "self-improving-agent")
        self.assertIn("continuous improvement", results[0].description)

    def test_top_by_category_groups_and_ranks_by_popularity(self):
        catalog = [
            SkillMeta(
                name="Feishu Sync Pro",
                description="Feishu bi-directional sync",
                source="clawhub",
                identifier="feishu-sync-pro",
                trust_level="community",
                tags=["communication", "productivity"],
                extra={"installs": 2200, "downloads": 1500},
            ),
            SkillMeta(
                name="Feishu Inbox",
                description="Feishu inbox actions",
                source="clawhub",
                identifier="feishu-inbox",
                trust_level="community",
                tags=["communication"],
                extra={"installs": 3100, "downloads": 500},
            ),
            SkillMeta(
                name="Code Review Bot",
                description="PR review workflows",
                source="clawhub",
                identifier="code-review-bot",
                trust_level="community",
                tags=["coding", "productivity"],
                extra={"installs": 1900, "downloads": 1200},
            ),
        ]

        grouped = self.src.top_by_category(limit_per_category=2, catalog=catalog)

        self.assertEqual([item.identifier for item in grouped["communication"]], [
            "feishu-inbox",
            "feishu-sync-pro",
        ])
        self.assertEqual([item.identifier for item in grouped["productivity"]], [
            "feishu-sync-pro",
            "code-review-bot",
        ])
        self.assertEqual([item.identifier for item in grouped["coding"]], [
            "code-review-bot",
        ])

    def test_meta_from_item_preserves_popularity_metrics(self):
        meta = self.src._meta_from_item(
            {
                "slug": "feishu-sync-pro",
                "displayName": "Feishu Sync Pro",
                "summary": "Feishu bi-directional sync",
                "tags": ["communication"],
                "installs": 2200,
                "downloads": 1500,
                "weeklyInstalls": 180,
                "category": "communication",
            }
        )

        self.assertIsNotNone(meta)
        assert meta is not None
        self.assertEqual(meta.extra["installs"], 2200)
        self.assertEqual(meta.extra["downloads"], 1500)
        self.assertEqual(meta.extra["weeklyInstalls"], 180)
        self.assertEqual(meta.extra["category"], "communication")

    def test_build_clawhub_migration_queue_marks_covered_items_done(self):
        catalog = [
            SkillMeta(
                name="Feishu Inbox",
                description="Feishu inbox actions",
                source="clawhub",
                identifier="feishu-inbox",
                trust_level="community",
                tags=["communication"],
                extra={"installs": 3100, "downloads": 500},
            )
        ]
        local_catalog = [
            SkillMeta(
                name="Feishu Inbox",
                description="Inbox actions for Feishu chats",
                source="builtin",
                identifier="skills/productivity/feishu-inbox",
                trust_level="builtin",
                path="productivity/feishu-inbox",
                tags=["feishu", "communication"],
                extra={"category": "communication"},
            )
        ]

        queue = build_clawhub_migration_queue(
            limit_per_category=10,
            catalog=catalog,
            local_catalog=local_catalog,
        )

        self.assertEqual(queue["summary"]["covered_items"], 1)
        item = queue["categories"]["communication"][0]
        self.assertEqual(item["coverage_status"], "covered")
        self.assertEqual(item["task_status"], "done")
        self.assertEqual(item["local_matches"][0]["name"], "Feishu Inbox")

    def test_build_clawhub_migration_queue_preserves_gap_status_and_notes(self):
        catalog = [
            SkillMeta(
                name="Code Review Bot",
                description="PR review workflows",
                source="clawhub",
                identifier="code-review-bot",
                trust_level="community",
                tags=["coding", "productivity"],
                extra={"installs": 1900, "downloads": 1200},
            )
        ]
        existing_queue = {
            "items": [
                {
                    "category": "coding",
                    "identifier": "code-review-bot",
                    "task_status": "blocked",
                    "notes": "Waiting for human review of migration scope.",
                }
            ]
        }

        queue = build_clawhub_migration_queue(
            limit_per_category=10,
            catalog=catalog,
            local_catalog=[],
            existing_queue=existing_queue,
        )

        item = queue["categories"]["coding"][0]
        self.assertEqual(item["coverage_status"], "gap")
        self.assertEqual(item["task_status"], "blocked")
        self.assertEqual(item["notes"], "Waiting for human review of migration scope.")

    def test_save_clawhub_migration_queue_persists_and_reuses_state(self):
        catalog = [
            SkillMeta(
                name="Code Review Bot",
                description="PR review workflows",
                source="clawhub",
                identifier="code-review-bot",
                trust_level="community",
                tags=["coding", "productivity"],
                extra={"installs": 1900, "downloads": 1200},
            )
        ]

        temp_root = Path(os.getcwd()) / ".tmp-pytest" / "skills-hub-queue-state"
        temp_root.mkdir(parents=True, exist_ok=True)
        queue_path = temp_root / f"clawhub-migration-queue-{os.getpid()}.json"
        temp_queue_path = queue_path.with_suffix(f"{queue_path.suffix}.tmp")
        for path in (queue_path, temp_queue_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

        first = save_clawhub_migration_queue(
            queue_path,
            limit_per_category=10,
            catalog=catalog,
            local_catalog=[],
            existing_queue={
                "items": [
                    {
                        "category": "coding",
                        "identifier": "code-review-bot",
                        "task_status": "in_progress",
                        "notes": "Migration scaffolding started.",
                    }
                ]
            },
        )
        self.assertTrue(queue_path.exists())
        self.assertEqual(first["categories"]["coding"][0]["task_status"], "in_progress")

        loaded = load_marketplace_migration_queue(queue_path)
        self.assertEqual(loaded["categories"]["coding"][0]["notes"], "Migration scaffolding started.")

        second = save_clawhub_migration_queue(
            queue_path,
            limit_per_category=10,
            catalog=catalog,
            local_catalog=[],
        )
        self.assertEqual(second["categories"]["coding"][0]["task_status"], "in_progress")
        self.assertEqual(second["categories"]["coding"][0]["notes"], "Migration scaffolding started.")

    def test_write_index_cache_skips_unwritable_cache_dir(self):
        import tools.skills_hub as hub_mod
        original_mkdir = type(hub_mod.INDEX_CACHE_DIR).mkdir

        def _guarded_mkdir(path_self, *args, **kwargs):
            if path_self == hub_mod.INDEX_CACHE_DIR:
                raise PermissionError("denied")
            return original_mkdir(path_self, *args, **kwargs)

        with patch.object(type(hub_mod.INDEX_CACHE_DIR), "mkdir", _guarded_mkdir):
            hub_mod._write_index_cache("test_key", {"data": "test"})

    @patch("tools.skills_hub.httpx.get")
    def test_search_repairs_poisoned_cache_with_exact_slug_lookup(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            json_data={
                "skill": {
                    "slug": "self-improving-agent",
                    "displayName": "self-improving-agent",
                    "summary": "Captures learnings and errors for continuous improvement.",
                    "tags": {"latest": "3.0.2", "automation": "3.0.2"},
                },
                "latestVersion": {"version": "3.0.2"},
            },
        )

        poisoned = [
            SkillMeta(
                name="Apple Music DJ",
                description="Unrelated cached result",
                source="clawhub",
                identifier="apple-music-dj",
                trust_level="community",
                tags=[],
            )
        ]
        results = self.src._finalize_search_results("self-improving-agent", poisoned, 5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "self-improving-agent")
        mock_get.assert_called_once()
        self.assertTrue(mock_get.call_args.args[0].endswith("/skills/self-improving-agent"))

    @patch.object(
        ClawHubSource,
        "_exact_slug_meta",
        return_value=SkillMeta(
            name="self-improving-agent",
            description="Captures learnings and errors for continuous improvement.",
            source="clawhub",
            identifier="self-improving-agent",
            trust_level="community",
            tags=["automation"],
        ),
    )
    def test_search_matches_space_separated_query_to_hyphenated_slug(
        self, _mock_exact_slug
    ):
        results = self.src.search("self improving", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "self-improving-agent")

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_maps_display_name_and_summary(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            json_data={
                "slug": "caldav-calendar",
                "displayName": "CalDAV Calendar",
                "summary": "Calendar integration",
                "tags": ["calendar"],
            },
        )

        meta = self.src.inspect("caldav-calendar")

        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "CalDAV Calendar")
        self.assertEqual(meta.description, "Calendar integration")
        self.assertEqual(meta.identifier, "caldav-calendar")

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_handles_nested_skill_payload(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            json_data={
                "skill": {
                    "slug": "self-improving-agent",
                    "displayName": "self-improving-agent",
                    "summary": "Captures learnings and errors for continuous improvement.",
                    "tags": {"latest": "3.0.2", "automation": "3.0.2"},
                },
                "latestVersion": {"version": "3.0.2"},
            },
        )

        meta = self.src.inspect("self-improving-agent")

        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "self-improving-agent")
        self.assertIn("continuous improvement", meta.description)
        self.assertEqual(meta.identifier, "self-improving-agent")
        self.assertEqual(meta.tags, ["automation"])

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_resolves_latest_version_and_downloads_raw_files(self, mock_get):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills/caldav-calendar"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "slug": "caldav-calendar",
                        "latestVersion": {"version": "1.0.1"},
                    },
                )
            if url.endswith("/skills/caldav-calendar/versions/1.0.1"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "files": [
                            {"path": "SKILL.md", "rawUrl": "https://files.example/skill-md"},
                            {"path": "README.md", "content": "hello"},
                        ]
                    },
                )
            if url == "https://files.example/skill-md":
                return _MockResponse(status_code=200, text="# Skill")
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("caldav-calendar")

        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.name, "caldav-calendar")
        self.assertIn("SKILL.md", bundle.files)
        self.assertEqual(bundle.files["SKILL.md"], "# Skill")
        self.assertEqual(bundle.files["README.md"], "hello")

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_falls_back_to_versions_list(self, mock_get):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills/caldav-calendar"):
                return _MockResponse(status_code=200, json_data={"slug": "caldav-calendar"})
            if url.endswith("/skills/caldav-calendar/versions"):
                return _MockResponse(status_code=200, json_data=[{"version": "2.0.0"}])
            if url.endswith("/skills/caldav-calendar/versions/2.0.0"):
                return _MockResponse(status_code=200, json_data={"files": {"SKILL.md": "# Skill"}})
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("caldav-calendar")
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.files["SKILL.md"], "# Skill")


if __name__ == "__main__":
    unittest.main()
