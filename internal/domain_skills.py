from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "domain-skills.registry.json"
_DOMAIN_STRATEGIES = {"local_only", "local_preferred", "cloud_required"}


def _read_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import read_raw_config

        config = read_raw_config()
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def get_domain_registry_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config if isinstance(config, dict) else _read_config()
    skills_cfg = cfg.get("skills", {}) if isinstance(cfg, dict) else {}
    raw_path = ""
    if isinstance(skills_cfg, dict):
        raw_path = str(skills_cfg.get("domain_registry_path") or "").strip()
    if raw_path:
        expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
        return expanded if expanded.is_absolute() else (Path(__file__).resolve().parents[1] / expanded).resolve()
    return _DEFAULT_REGISTRY_PATH


def domain_autoload_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = config if isinstance(config, dict) else _read_config()
    skills_cfg = cfg.get("skills", {}) if isinstance(cfg, dict) else {}
    if isinstance(skills_cfg, dict) and "domain_autoload_enabled" in skills_cfg:
        return bool(skills_cfg.get("domain_autoload_enabled"))
    return True


@lru_cache(maxsize=4)
def _load_registry_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        logger.debug("Domain skill registry does not exist: %s", path)
        return {"templates": {}, "domains": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.setdefault("templates", {})
            payload.setdefault("domains", {})
            return payload
    except Exception as exc:
        logger.warning("Failed to load domain skill registry %s: %s", path, exc)
    return {"templates": {}, "domains": {}}


def load_domain_registry(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return _load_registry_cached(str(get_domain_registry_path(config)))


def normalize_domain(value: Any) -> str:
    domain = str(value or "").strip().lower().strip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def normalize_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("www."):
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    if parsed.scheme not in {"http", "https"}:
        return ""
    return parsed._replace(fragment="").geturl()


def extract_target_url_and_domain(message: str) -> tuple[str, str]:
    text = str(message or "").strip()
    if not text:
        return "", ""
    import re

    candidates = re.findall(r"(?:https?://|www\.)[^\s<>'\"`)]+", text, flags=re.IGNORECASE)
    if not candidates:
        candidates = re.findall(
            r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:/[^\s<>'\"`)]+)?",
            text,
            flags=re.IGNORECASE,
        )
    for candidate in candidates:
        normalized = normalize_url(candidate.rstrip("),.;!?"))
        if not normalized:
            continue
        try:
            domain = normalize_domain(urlparse(normalized).hostname)
        except Exception:
            domain = ""
        if domain:
            return normalized, domain
    return "", ""


def _iter_domain_candidates(domain: str) -> list[str]:
    normalized = normalize_domain(domain)
    if not normalized:
        return []
    parts = normalized.split(".")
    return [".".join(parts[index:]) for index in range(len(parts) - 1)]


def resolve_domain_skill(
    *,
    target_domain: str = "",
    target_url: str = "",
    site_category: str = "",
    site_intent: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = load_domain_registry(config)
    domains = registry.get("domains", {})
    templates = registry.get("templates", {})
    if not isinstance(domains, dict):
        return {}

    candidate_domain = normalize_domain(target_domain)
    if not candidate_domain and target_url:
        try:
            candidate_domain = normalize_domain(urlparse(normalize_url(target_url)).hostname)
        except Exception:
            candidate_domain = ""

    matched_key = ""
    matched_entry: dict[str, Any] = {}
    for domain_key in _iter_domain_candidates(candidate_domain):
        raw_entry = domains.get(domain_key)
        if isinstance(raw_entry, dict):
            matched_key = domain_key
            matched_entry = dict(raw_entry)
            break
    if not matched_entry:
        return {}

    template_name = str(matched_entry.get("template") or "").strip()
    template = templates.get(template_name) if isinstance(templates, dict) else {}
    template = dict(template) if isinstance(template, dict) else {}
    resolved = {
        **template,
        **matched_entry,
        "match_domain": matched_key,
        "target_domain": candidate_domain,
        "target_url": normalize_url(target_url),
        "site_category": str(site_category or "").strip(),
        "site_intent": str(site_intent or "").strip(),
    }
    browser_strategy = str(resolved.get("browser_strategy") or "").strip().lower() or "local_preferred"
    resolved["browser_strategy"] = browser_strategy if browser_strategy in _DOMAIN_STRATEGIES else "local_preferred"
    resolved["skill_name"] = str(resolved.get("skill_name") or f"site.{matched_key.replace('.', '-')}").strip()
    resolved["entry_url"] = normalize_url(resolved.get("entry_url")) or resolved["target_url"]
    for key in ("preferred_actions", "avoid_actions", "common_actions", "disabled_actions", "notes", "site_intents"):
        value = resolved.get(key)
        if not isinstance(value, list):
            resolved[key] = []
        else:
            resolved[key] = [str(item).strip() for item in value if str(item or "").strip()]
    return resolved


def resolve_domain_skill_from_payload(payload: dict[str, Any] | None, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    ingress = raw.get("_hermes_ingress") if isinstance(raw.get("_hermes_ingress"), dict) else {}
    site_prefetch = raw.get("site_prefetch") if isinstance(raw.get("site_prefetch"), dict) else {}
    if not site_prefetch and isinstance(ingress, dict):
        site_prefetch = ingress.get("site_prefetch") if isinstance(ingress.get("site_prefetch"), dict) else {}
    target_url = str(raw.get("target_url") or site_prefetch.get("final_url") or site_prefetch.get("target_url") or "").strip()
    target_domain = str(raw.get("target_domain") or site_prefetch.get("domain") or "").strip()
    if not target_url and not target_domain:
        target_url, target_domain = extract_target_url_and_domain(str(raw.get("text") or ""))
    return resolve_domain_skill(
        target_domain=target_domain,
        target_url=target_url,
        site_category=str(raw.get("site_category") or site_prefetch.get("category") or "").strip(),
        site_intent=str(raw.get("site_intent") or site_prefetch.get("intent") or "").strip(),
        config=config,
    )


def build_site_skill_runtime_note(payload: dict[str, Any] | None, *, config: dict[str, Any] | None = None) -> str:
    if not domain_autoload_enabled(config):
        return ""
    resolved = resolve_domain_skill_from_payload(payload, config=config)
    if not resolved:
        return ""
    parts = [
        "Domain skill hint (internal, not user-authored):",
        f"site_skill_name={resolved['skill_name']}",
        f"match_domain={resolved.get('match_domain') or resolved.get('target_domain') or ''}",
        f"browser_strategy={resolved.get('browser_strategy') or 'local_preferred'}",
    ]
    if resolved.get("description"):
        parts.append(f"description={resolved['description']}")
    if resolved.get("entry_url"):
        parts.append(f"entry_url={resolved['entry_url']}")
    if resolved.get("notes"):
        parts.append(f"notes={'; '.join(resolved['notes'][:4])}")
    if resolved.get("preferred_actions"):
        parts.append(f"preferred_actions={'; '.join(resolved['preferred_actions'][:4])}")
    if resolved.get("common_actions"):
        parts.append(f"common_actions={'; '.join(resolved['common_actions'][:4])}")
    if resolved.get("avoid_actions"):
        parts.append(f"avoid_actions={'; '.join(resolved['avoid_actions'][:4])}")
    if resolved.get("disabled_actions"):
        parts.append(f"disabled_actions={'; '.join(resolved['disabled_actions'][:4])}")
    return "[" + " | ".join(part for part in parts if part) + "]"


def _normalize_domain_strategy(value: Any) -> str:
    strategy = str(value or "").strip().lower()
    return strategy if strategy in _DOMAIN_STRATEGIES else "local_preferred"


def resolve_browser_domain_strategy(
    *,
    target_domain: str = "",
    target_url: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, dict) else _read_config()
    browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
    browser_cfg = browser_cfg if isinstance(browser_cfg, dict) else {}
    domain_strategy_cfg = browser_cfg.get("domain_strategy", {})
    domain_strategy_cfg = domain_strategy_cfg if isinstance(domain_strategy_cfg, dict) else {}
    default_strategy = _normalize_domain_strategy(domain_strategy_cfg.get("default"))
    domain_overrides = domain_strategy_cfg.get("domains", {})
    domain_overrides = domain_overrides if isinstance(domain_overrides, dict) else {}
    resolved = resolve_domain_skill(target_domain=target_domain, target_url=target_url, config=cfg)
    matched_domain = normalize_domain(target_domain) or normalize_domain(resolved.get("match_domain"))
    override_value = ""
    for candidate in _iter_domain_candidates(matched_domain):
        if candidate in domain_overrides:
            override_value = str(domain_overrides.get(candidate) or "").strip()
            break
    registry_strategy = _normalize_domain_strategy(resolved.get("browser_strategy"))
    strategy = _normalize_domain_strategy(override_value) if override_value else registry_strategy or default_strategy
    return {
        "strategy": strategy or default_strategy,
        "default_strategy": default_strategy,
        "override_strategy": _normalize_domain_strategy(override_value) if override_value else "",
        "registry_strategy": registry_strategy,
        "site_skill_name": str(resolved.get("skill_name") or "").strip(),
        "target_domain": matched_domain,
        "target_url": normalize_url(target_url),
    }


def cloud_escalation_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = config if isinstance(config, dict) else _read_config()
    browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
    if isinstance(browser_cfg, dict) and "cloud_escalation_enabled" in browser_cfg:
        return bool(browser_cfg.get("cloud_escalation_enabled"))
    return True


def resolve_cloud_escalation_rules(config: dict[str, Any] | None = None) -> dict[str, bool]:
    cfg = config if isinstance(config, dict) else _read_config()
    browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
    rules = browser_cfg.get("cloud_escalation_rules", {}) if isinstance(browser_cfg, dict) else {}
    rules = rules if isinstance(rules, dict) else {}
    return {
        "local_failure": bool(rules.get("local_failure", True)),
        "bot_detection": bool(rules.get("bot_detection", True)),
        "login_failure": bool(rules.get("login_failure", True)),
        "redirect_anomaly": bool(rules.get("redirect_anomaly", True)),
        "complex_interaction_failure": bool(rules.get("complex_interaction_failure", True)),
    }
