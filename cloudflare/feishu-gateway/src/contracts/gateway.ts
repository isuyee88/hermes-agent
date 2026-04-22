export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type RouteHint = "fast_control" | "native_io" | "cf_browser_first" | "modal_heavy_exec";
export type SiteCategory = "none" | "site_content" | "site_interactive_light" | "site_interactive_heavy";
export type SiteIntent = "general" | "docs" | "api" | "pricing" | "help" | "blog" | "login" | "signup" | "navigation";
export type SitePrefetchMode =
  | "none"
  | "markdown_for_agents"
  | "browser_markdown"
  | "browser_crawl"
  | "readability_fallback"
  | "playwright_preflight";
export type RequestClass =
  | "text_plain"
  | "text_coding"
  | "image_understanding"
  | "image_generation"
  | "media_hydration"
  | "tool_browser"
  | "tool_non_browser"
  | "file_or_attachment"
  | "session_mutation_heavy";
export type RouteFamily = "modal_control" | "modal_runtime" | "modal_tools" | "gateway_text" | "gateway_image";
export type DfmeaFailureMode =
  | "nominal_path"
  | "modal_runtime_required"
  | "request_classification_error"
  | "gateway_misroute"
  | "provider_config_invalid"
  | "provider_auth_invalid"
  | "provider_rate_exhausted"
  | "provider_sync_wait_budget_exceeded"
  | "provider_timeout"
  | "provider_http_error";
export type DfmeaControlPhase = "classification" | "gateway_preflight" | "gateway_execution" | "fallback" | "modal_runtime";
export type ExecutionMode =
  | "control_complete"
  | "native_io_complete"
  | "cf_browser_first"
  | "modal_heavy_exec"
  | "deferred_reconcile";

export type WorkflowBinding<Params> = {
  create(options: { id?: string; params: Params }): Promise<unknown>;
};

export type FeishuAttachmentRef = {
  resource_type: "image" | "file" | "audio" | "media";
  message_id: string;
  file_key?: string;
  image_key?: string;
  file_name?: string;
};

export type FeishuNormalizedPayload = {
  correlation_id: string;
  session_key: string;
  lane: "control" | "agent" | "ignore";
  route_hint: RouteHint;
  site_category: SiteCategory;
  site_intent: SiteIntent;
  target_url: string;
  target_domain: string;
  task_kind: string;
  request_class: RequestClass;
  content_modalities: string[];
  route_family: RouteFamily;
  gateway_route_name: string;
  gateway_eligible: boolean;
  requires_tools: boolean;
  requires_browser: boolean;
  requires_media_hydration: boolean;
  requires_modal_runtime: boolean;
  modality_profile: string;
  toolset: string[];
  reason_code: string;
  event_id: string;
  event_type: string;
  chat_id: string;
  chat_type: "dm" | "group";
  chat_name: string;
  user_id: string;
  user_name: string;
  message_id: string;
  message_type: "text" | "photo" | "audio" | "video" | "document" | "command";
  text: string;
  attachment_refs: FeishuAttachmentRef[];
  raw_payload: Record<string, JsonValue>;
};

export type PendingReconcileItem = {
  correlation_id: string;
  assistant_text: string;
  final_response: string;
  operation_kinds: string[];
  delivered_at: string;
};

export type OutboundMessageCorrelationRef = {
  send_message_id: string;
  correlation_id: string;
  session_key: string;
  event_id: string;
  source_message_id: string;
  kind: string;
  created_at: string;
};

export type SessionProfileState = {
  current_model?: string;
  current_provider?: string;
  current_personality?: string;
  route_status_lines?: string[];
  updated_at?: string;
};

export type SessionHistoryEntry = {
  id: string;
  role: "system" | "user" | "assistant";
  content: string;
  created_at: string;
};

export type SessionStateCache = {
  profile?: SessionProfileState | null;
  history?: SessionHistoryEntry[];
};

export type ModalInternalResponse = {
  status: string;
  action?: string;
  error?: string;
  card?: Record<string, JsonValue>;
  final_response?: string;
  send_plan?: Array<Record<string, JsonValue>>;
  action_plan?: Array<Record<string, JsonValue>>;
  execution_mode?: ExecutionMode;
  route_hint?: RouteHint;
  reconcile_required?: boolean;
  provider_usage?: Record<string, JsonValue>;
  provider_plan?: Record<string, JsonValue>;
  llm_request?: Record<string, JsonValue>;
  browser_fallback_allowed?: boolean;
  retry_class?: string;
  retry_attempt_count?: number;
  external_exec_candidate?: boolean;
  requires_modal_runtime?: boolean;
  requires_tools?: boolean;
  requires_browser?: boolean;
  requires_media_hydration?: boolean;
  request_class?: RequestClass;
  content_modalities?: string[];
  route_family?: RouteFamily;
  gateway_route_name?: string;
  gateway_eligible?: boolean;
  modality_profile?: string;
  toolset?: string[];
  reason_code?: string;
  misroute_detected?: boolean;
  gateway_error_class?: string;
  dfmea_failure_mode?: DfmeaFailureMode;
  dfmea_control_phase?: DfmeaControlPhase;
  dfmea_detection_signal?: string;
  dfmea_control_action?: string;
  dfmea_severity?: number;
  provider_async_eligible?: boolean;
  provider_async_observed?: boolean;
  route_decision_reason?: string;
  fallback_reason?: string;
  route_version?: string;
  provider_alias?: string;
  cache_eligible?: boolean;
  cache_status?: string;
  ai_call_count?: number;
  capability_match?: boolean;
  preferred_model_selected?: boolean;
  model_catalog_version?: string;
  feedback_score_before?: number;
  feedback_score_after?: number;
  provider_wait_measurement_mode?: string;
  site_prefetch?: SitePrefetchManifest;
  site_skill_name?: string;
  modal_avoided?: boolean;
  browser_backend_selected?: string;
  browser_escalation_reason?: string;
  session_state_after?: {
    current_model?: string;
    current_provider?: string;
    current_personality?: string;
    route_status_lines?: string[];
  };
};

export type SitePrefetchManifest = {
  key: string;
  category: SiteCategory;
  mode: SitePrefetchMode;
  domain: string;
  intent: SiteIntent;
  target_url: string;
  final_url: string;
  page_title: string;
  candidate_urls: string[];
  top_nav_links: string[];
  summary: string;
  sections: string[];
  page_kind: string;
  primary_actions: string[];
  forms_summary: string[];
  input_fields: string[];
  dialog_or_banner: string[];
  auth_required_guess: boolean;
  a11y_snapshot_summary: string;
  confidence: number;
  source_evidence: string[];
  status: "pending" | "completed" | "rate_limited" | "empty" | "error";
  fetched_at: string;
  expires_at: string;
  error_class?: string;
  token_estimate?: number;
  browser_ms_used?: number;
  crawl_job_id?: string;
};
