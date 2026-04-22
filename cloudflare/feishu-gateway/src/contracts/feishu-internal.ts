type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

type NormalizedPayloadLike = {
  correlation_id?: string;
  session_key?: string;
  event_id?: string;
  event_type?: string;
  chat_id?: string;
  chat_type?: string;
  chat_name?: string;
  user_id?: string;
  user_name?: string;
  message_id?: string;
  message_type?: string;
  text?: string;
  lane?: string;
  task_kind?: string;
  request_class?: string;
  route_hint?: string;
  route_family?: string;
  gateway_route_name?: string;
  gateway_eligible?: boolean;
  modality_profile?: string;
  requires_tools?: boolean;
  requires_browser?: boolean;
  requires_media_hydration?: boolean;
  requires_modal_runtime?: boolean;
  reason_code?: string;
  site_category?: string;
  site_intent?: string;
  target_domain?: string;
  target_url?: string;
  content_modalities?: string[];
  toolset?: string[];
};

export type ModalRequestEnvelope = {
  contract_version: "feishu_internal.v1";
  ingress: Record<string, JsonValue>;
  route: Record<string, JsonValue>;
  site_prefetch: JsonValue | null;
  session_hints: Record<string, JsonValue>;
  gateway_meta: Record<string, JsonValue>;
};

export type ModalResponseEnvelope = {
  contract_version: "feishu_internal.v1";
  result: Record<string, JsonValue>;
  send_plan: Record<string, JsonValue>;
  session_patch: Record<string, JsonValue>;
  reconcile: Record<string, JsonValue>;
  provider_metrics: Record<string, JsonValue>;
  legacy_response?: Record<string, JsonValue>;
};

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function pickDefined(record: Record<string, unknown>): Record<string, JsonValue> {
  const out: Record<string, JsonValue> = {};
  for (const [key, value] of Object.entries(record)) {
    if (value === undefined) {
      continue;
    }
    if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      out[key] = value;
      continue;
    }
    if (Array.isArray(value)) {
      out[key] = value as JsonValue;
      continue;
    }
    if (typeof value === "object") {
      out[key] = value as JsonValue;
    }
  }
  return out;
}

function normalizeRecord(value: unknown): Record<string, JsonValue> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, JsonValue>;
}

export function buildModalRequestEnvelope(
  path: string,
  normalized: NormalizedPayloadLike | undefined,
  body: Record<string, JsonValue>,
  pendingReconcile: JsonValue,
): ModalRequestEnvelope {
  const legacyPayload: Record<string, JsonValue> = {
    ...body,
    session_key: trim(body.session_key ?? normalized?.session_key),
    pending_reconcile: pendingReconcile,
  };
  return {
    contract_version: "feishu_internal.v1",
    ingress: pickDefined({
      correlation_id: normalized?.correlation_id,
      session_key: legacyPayload.session_key,
      event_id: normalized?.event_id,
      event_type: normalized?.event_type,
      chat_id: normalized?.chat_id,
      chat_type: normalized?.chat_type,
      chat_name: normalized?.chat_name,
      user_id: normalized?.user_id,
      user_name: normalized?.user_name,
      message_id: normalized?.message_id,
      message_type: normalized?.message_type,
      text: normalized?.text,
      lane: normalized?.lane,
      task_kind: normalized?.task_kind,
    }),
    route: pickDefined({
      request_class: normalized?.request_class,
      route_hint: body.route_hint ?? normalized?.route_hint,
      route_family: body.route_family ?? normalized?.route_family,
      gateway_route_name: body.gateway_route_name ?? normalized?.gateway_route_name,
      gateway_eligible: body.gateway_eligible ?? normalized?.gateway_eligible,
      modality_profile: body.modality_profile ?? normalized?.modality_profile,
      requires_tools: body.requires_tools ?? normalized?.requires_tools,
      requires_browser: body.requires_browser ?? normalized?.requires_browser,
      requires_media_hydration: body.requires_media_hydration ?? normalized?.requires_media_hydration,
      requires_modal_runtime: body.requires_modal_runtime ?? normalized?.requires_modal_runtime,
      reason_code: body.reason_code ?? normalized?.reason_code,
      content_modalities: body.content_modalities ?? normalized?.content_modalities,
      toolset: body.toolset ?? normalized?.toolset,
    }),
    site_prefetch: body.site_prefetch ?? null,
    session_hints: pickDefined({
      session_key: legacyPayload.session_key,
      pending_reconcile: pendingReconcile,
    }),
    gateway_meta: pickDefined({
      endpoint: path,
      gateway_hop: "cloudflare-worker",
      gateway_script: "hermes-feishu-gateway",
      sent_at: new Date().toISOString(),
      site_category: normalized?.site_category,
      site_intent: normalized?.site_intent,
      target_domain: normalized?.target_domain,
      target_url: normalized?.target_url,
      legacy_payload: legacyPayload,
    }),
  };
}

export function unwrapModalResponseEnvelope<T>(value: unknown): T {
  const record = normalizeRecord(value);
  if (trim(record.contract_version) !== "feishu_internal.v1") {
    return record as T;
  }
  const legacy = normalizeRecord(record.legacy_response);
  if (Object.keys(legacy).length > 0) {
    return legacy as T;
  }
  const result = normalizeRecord(record.result);
  const sendPlan = normalizeRecord(record.send_plan);
  const sessionPatch = normalizeRecord(record.session_patch);
  const reconcile = normalizeRecord(record.reconcile);
  const providerMetrics = normalizeRecord(record.provider_metrics);
  return {
    ...result,
    ...sendPlan,
    ...sessionPatch,
    ...reconcile,
    ...providerMetrics,
  } as T;
}
