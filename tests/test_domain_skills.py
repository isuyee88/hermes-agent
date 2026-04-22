from internal.domain_skills import (
    build_site_skill_runtime_note,
    extract_target_url_and_domain,
    resolve_browser_domain_strategy,
    resolve_domain_skill,
    resolve_domain_skill_from_payload,
)


def test_extract_target_url_and_domain_handles_docs_urls():
    url, domain = extract_target_url_and_domain("请看这个文档站 https://developers.cloudflare.com/workers/")

    assert url == "https://developers.cloudflare.com/workers/"
    assert domain == "developers.cloudflare.com"


def test_resolve_domain_skill_uses_registry_entry_for_known_docs_site():
    resolved = resolve_domain_skill(
        target_domain="developers.cloudflare.com",
        target_url="https://developers.cloudflare.com/workers/",
        site_category="site_content",
        site_intent="docs",
    )

    assert resolved["skill_name"] == "site.cloudflare-developers-docs"
    assert resolved["browser_strategy"] == "local_preferred"
    assert resolved["entry_url"] == "https://developers.cloudflare.com/"


def test_build_site_skill_runtime_note_includes_domain_specific_hint():
    note = build_site_skill_runtime_note(
        {
            "target_url": "https://dashboard.stripe.com/login",
            "target_domain": "dashboard.stripe.com",
            "site_category": "site_interactive_heavy",
            "site_intent": "login",
        },
        config={"skills": {"domain_autoload_enabled": True}},
    )

    assert "site_skill_name=site.stripe-dashboard" in note
    assert "browser_strategy=cloud_required" in note


def test_resolve_domain_skill_from_payload_reads_ingress_site_prefetch():
    resolved = resolve_domain_skill_from_payload(
        {
            "_hermes_ingress": {
                "site_prefetch": {
                    "domain": "docs.browserbase.com",
                    "final_url": "https://docs.browserbase.com/introduction",
                    "category": "site_content",
                    "intent": "docs",
                }
            }
        }
    )

    assert resolved["skill_name"] == "site.browserbase-docs"
    assert resolved["target_domain"] == "docs.browserbase.com"


def test_resolve_browser_domain_strategy_prefers_explicit_domain_override():
    resolved = resolve_browser_domain_strategy(
        target_domain="dashboard.stripe.com",
        target_url="https://dashboard.stripe.com/login",
        config={
            "browser": {
                "domain_strategy": {
                    "default": "local_preferred",
                    "domains": {"dashboard.stripe.com": "cloud_required"},
                }
            }
        },
    )

    assert resolved["strategy"] == "cloud_required"
    assert resolved["override_strategy"] == "cloud_required"
    assert resolved["site_skill_name"] == "site.stripe-dashboard"
