export const MODEL_CATALOG_DATABASE_BINDING = "MODEL_CATALOG_DB";
export const MODEL_CATALOG_MIGRATION_TAG = "0001_model_catalog_control_plane";
export const MODEL_CATALOG_SYNC_VIEW = "model_registry_sync_view";
export const MODEL_CATALOG_SYNC_ENABLED_VAR = "HERMES_MODEL_CATALOG_SYNC_ENABLED";
export const MODEL_CATALOG_SYNC_PROVIDERS_VAR = "HERMES_MODEL_CATALOG_SYNC_PROVIDERS";
export const MODEL_CATALOG_DEFAULT_SYNC_PROVIDERS = ["openrouter", "nvidia"] as const;
export const MODEL_CATALOG_TABLES = [
  "model_catalog",
  "model_runtime_policy",
  "model_health_stats",
  "model_feedback_events",
  "provider_sync_runs",
] as const;
