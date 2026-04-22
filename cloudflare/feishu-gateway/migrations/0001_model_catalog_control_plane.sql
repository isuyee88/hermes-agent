PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS model_catalog (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  provider_model_key TEXT NOT NULL,
  display_name TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  hidden INTEGER NOT NULL DEFAULT 0,
  is_available INTEGER NOT NULL DEFAULT 1,
  is_free INTEGER NOT NULL DEFAULT 0,
  rank INTEGER,
  selection_hint TEXT,
  manual_pinned INTEGER NOT NULL DEFAULT 0,
  recent_used INTEGER NOT NULL DEFAULT 0,
  recent_used_count INTEGER NOT NULL DEFAULT 0,
  generated_command TEXT,
  last_probe_at INTEGER,
  recent_used_at INTEGER,
  last_sync_at INTEGER,
  latency_ms REAL,
  context_window INTEGER,
  reasoning INTEGER,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  failure_kind TEXT,
  last_error_code TEXT,
  last_error_message TEXT,
  last_failed_at INTEGER,
  source TEXT,
  function_type TEXT,
  task_kinds_json TEXT NOT NULL DEFAULT '[]',
  modalities_json TEXT NOT NULL DEFAULT '[]',
  model_family TEXT,
  model_version TEXT,
  vision INTEGER,
  tool_calling INTEGER,
  structured_output INTEGER,
  streaming INTEGER,
  max_output_tokens INTEGER,
  input_price_per_million REAL,
  output_price_per_million REAL,
  cache_read_price_per_million REAL,
  cache_write_price_per_million REAL,
  temperature_supported INTEGER,
  top_p_supported INTEGER,
  json_mode_supported INTEGER,
  parameter_schema_json TEXT,
  api_family TEXT,
  provider_region TEXT,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
  UNIQUE (provider, model),
  UNIQUE (provider_model_key)
);

CREATE INDEX IF NOT EXISTS idx_model_catalog_status ON model_catalog(status, hidden, is_available);
CREATE INDEX IF NOT EXISTS idx_model_catalog_selection ON model_catalog(selection_hint, manual_pinned, rank);

CREATE TABLE IF NOT EXISTS model_runtime_policy (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  task_kind TEXT NOT NULL DEFAULT 'general',
  route_family TEXT,
  gateway_route_name TEXT,
  priority_rank INTEGER NOT NULL DEFAULT 100,
  enabled INTEGER NOT NULL DEFAULT 1,
  hidden_reason TEXT,
  fallback_allowed INTEGER NOT NULL DEFAULT 1,
  rate_limit_rpm INTEGER,
  rate_limit_tpm INTEGER,
  rate_limit_rpd INTEGER,
  burst_limit INTEGER,
  rate_limit_window_seconds INTEGER,
  cooldown_until INTEGER,
  cache_mode TEXT NOT NULL DEFAULT 'default',
  cache_ttl_seconds INTEGER,
  cache_scope TEXT,
  cache_key_template TEXT,
  request_timeout_ms INTEGER,
  max_attempts INTEGER,
  retry_delay_ms INTEGER,
  backoff TEXT,
  notes TEXT,
  last_route_publish_at INTEGER,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
  UNIQUE (provider, model, task_kind),
  FOREIGN KEY (provider, model) REFERENCES model_catalog(provider, model) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_model_runtime_policy_enabled
  ON model_runtime_policy(enabled, task_kind, priority_rank, provider, model);
CREATE INDEX IF NOT EXISTS idx_model_runtime_policy_route
  ON model_runtime_policy(route_family, gateway_route_name, task_kind);

CREATE TABLE IF NOT EXISTS model_health_stats (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  task_kind TEXT NOT NULL DEFAULT 'general',
  window_granularity TEXT NOT NULL DEFAULT 'hour',
  window_start INTEGER NOT NULL,
  window_end INTEGER NOT NULL,
  request_count INTEGER NOT NULL DEFAULT 0,
  success_count INTEGER NOT NULL DEFAULT 0,
  error_count INTEGER NOT NULL DEFAULT 0,
  auth_error_count INTEGER NOT NULL DEFAULT 0,
  model_not_found_count INTEGER NOT NULL DEFAULT 0,
  provider_failure_count INTEGER NOT NULL DEFAULT 0,
  rate_limit_count INTEGER NOT NULL DEFAULT 0,
  timeout_count INTEGER NOT NULL DEFAULT 0,
  cache_hit_count INTEGER NOT NULL DEFAULT 0,
  cache_miss_count INTEGER NOT NULL DEFAULT 0,
  positive_feedback_count INTEGER NOT NULL DEFAULT 0,
  negative_feedback_count INTEGER NOT NULL DEFAULT 0,
  score_avg REAL,
  health_score REAL,
  success_rate REAL,
  error_rate REAL,
  cache_hit_rate REAL,
  latency_p50_ms REAL,
  latency_p95_ms REAL,
  latency_p99_ms REAL,
  ttfb_p50_ms REAL,
  ttfb_p95_ms REAL,
  ttfb_p99_ms REAL,
  input_tokens_sum INTEGER NOT NULL DEFAULT 0,
  output_tokens_sum INTEGER NOT NULL DEFAULT 0,
  estimated_cost_sum REAL NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
  UNIQUE (provider, model, task_kind, window_granularity, window_start),
  FOREIGN KEY (provider, model) REFERENCES model_catalog(provider, model) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_model_health_window
  ON model_health_stats(provider, model, task_kind, window_granularity, window_end DESC);
CREATE INDEX IF NOT EXISTS idx_model_health_score
  ON model_health_stats(health_score DESC, error_rate ASC, cache_hit_rate DESC);

CREATE TABLE IF NOT EXISTS model_feedback_events (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  task_kind TEXT NOT NULL DEFAULT 'general',
  route_family TEXT,
  gateway_route_name TEXT,
  gateway_log_id TEXT,
  correlation_id TEXT,
  feedback INTEGER,
  score INTEGER,
  error_kind TEXT,
  error_code TEXT,
  error_message TEXT,
  latency_ms REAL,
  ttfb_ms REAL,
  cache_status TEXT,
  cache_hit INTEGER,
  request_tokens INTEGER,
  response_tokens INTEGER,
  estimated_cost REAL,
  metadata_json TEXT,
  created_at INTEGER NOT NULL DEFAULT (unixepoch()),
  FOREIGN KEY (provider, model) REFERENCES model_catalog(provider, model) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_model_feedback_created
  ON model_feedback_events(provider, model, task_kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_feedback_gateway_log
  ON model_feedback_events(gateway_log_id, correlation_id);

CREATE TABLE IF NOT EXISTS provider_sync_runs (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'scheduled',
  run_kind TEXT NOT NULL DEFAULT 'catalog_refresh',
  status TEXT NOT NULL,
  discovered_model_count INTEGER NOT NULL DEFAULT 0,
  upserted_model_count INTEGER NOT NULL DEFAULT 0,
  hidden_model_count INTEGER NOT NULL DEFAULT 0,
  error_count INTEGER NOT NULL DEFAULT 0,
  error_summary TEXT,
  started_at INTEGER,
  completed_at INTEGER,
  created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_provider_sync_runs_recent
  ON provider_sync_runs(provider, run_kind, created_at DESC);

CREATE VIEW IF NOT EXISTS model_registry_sync_view AS
WITH latest_general_policy AS (
  SELECT p.*
  FROM model_runtime_policy p
  WHERE p.task_kind = 'general'
),
latest_general_health AS (
  SELECT h.*
  FROM model_health_stats h
  WHERE h.task_kind = 'general'
    AND h.window_end = (
      SELECT MAX(h2.window_end)
      FROM model_health_stats h2
      WHERE h2.provider = h.provider
        AND h2.model = h.model
        AND h2.task_kind = h.task_kind
    )
)
SELECT
  c.provider,
  c.model,
  c.display_name,
  c.status,
  c.hidden,
  c.is_available,
  c.is_free,
  c.rank,
  c.selection_hint,
  c.manual_pinned,
  c.recent_used,
  c.recent_used_count,
  c.generated_command,
  c.last_probe_at,
  c.recent_used_at,
  c.last_sync_at,
  c.latency_ms,
  c.context_window,
  c.reasoning,
  c.consecutive_failures,
  c.failure_kind,
  c.last_error_code,
  c.last_error_message,
  c.last_failed_at,
  c.source,
  c.function_type,
  c.task_kinds_json,
  c.modalities_json,
  c.vision,
  c.tool_calling,
  c.structured_output,
  c.streaming,
  c.max_output_tokens,
  c.input_price_per_million,
  c.output_price_per_million,
  c.cache_read_price_per_million,
  c.cache_write_price_per_million,
  c.temperature_supported,
  c.top_p_supported,
  c.json_mode_supported,
  c.parameter_schema_json,
  p.gateway_route_name,
  p.route_family,
  p.hidden_reason,
  p.rate_limit_rpm,
  p.rate_limit_tpm,
  p.rate_limit_rpd,
  p.burst_limit,
  p.rate_limit_window_seconds,
  p.cooldown_until,
  h.health_score,
  h.success_rate,
  h.error_rate,
  h.cache_hit_rate,
  h.latency_p50_ms,
  h.latency_p95_ms,
  h.ttfb_p50_ms,
  h.ttfb_p95_ms,
  h.positive_feedback_count,
  h.negative_feedback_count,
  h.provider_failure_count,
  h.rate_limit_count,
  h.auth_error_count,
  h.model_not_found_count
FROM model_catalog c
LEFT JOIN latest_general_policy p
  ON p.provider = c.provider
 AND p.model = c.model
LEFT JOIN latest_general_health h
  ON h.provider = c.provider
 AND h.model = c.model;
