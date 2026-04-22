import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_domain_skill_registry.py"


def test_build_domain_skill_registry_script_generates_tiered_registry(tmp_path):
    source = tmp_path / "sites.json"
    source.write_text(
        json.dumps(
            [
                {
                    "domain": "docs.example.com",
                    "url": "https://docs.example.com/guide",
                    "category": "docs",
                    "description": "Developer docs",
                },
                {
                    "domain": "dashboard.example.com",
                    "url": "https://dashboard.example.com/login",
                    "category": "dashboard login",
                    "description": "Account console",
                },
                {
                    "domain": "pricing.example.com",
                    "url": "https://pricing.example.com/plans",
                    "category": "pricing",
                    "description": "Plans and billing",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "registry.json"

    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--input", str(source), "--output", str(output)],
        check=True,
        cwd=str(REPO_ROOT),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["templates"]["content_docs"]["kind"] == "content"
    assert payload["domains"]["docs.example.com"]["template"] == "content_docs"
    assert payload["domains"]["dashboard.example.com"]["template"] == "console_dashboard"
    assert payload["domains"]["pricing.example.com"]["template"] == "content_pricing"
    assert payload["domains"]["dashboard.example.com"]["site_intents"] == ["login", "navigation"]
