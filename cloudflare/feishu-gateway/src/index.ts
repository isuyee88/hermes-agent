import { DurableObject, WorkflowEntrypoint } from "cloudflare:workers";

type JsonPrimitive = string | number | boolean | null;
type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
type RouteHint = "fast_control" | "native_io" | "cf_browser_first" | "modal_heavy_exec";
type ExecutionMode =
  | "control_complete"
  | "native_io_complete"
  | "cf_browser_first"
  | "modal_heavy_exec"
  | "deferred_reconcile";

type WorkflowBinding<Params> = {
  create(options: { id?: string; params: Params }): Promise<unknown>;
};

type FeishuAttachmentRef = {
  resource_type: "image" | "file" | "audio" | "media";
  message_id: string;
  file_key?: string;
  image_key?: string;
  file_name?: string;
};

type FeishuNormalizedPayload = {
  correlation_id: string;
  session_key: string;
  lane: "control" | "agent" | "ignore";
  route_hint: RouteHint;
  task_kind: string;
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

type PendingReconcileItem = {
  correlation_id: string;
  assistant_text: string;
  final_response: string;
  operation_kinds: string[];
  delivered_at: string;
};

type ModalInternalResponse = {
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
  session_state_after?: {
    current_model?: string;
    current_provider?: string;
    current_personality?: string;
    route_status_lines?: string[];
  };
};

const MAX_PENDING_RECONCILE_ITEMS = 50;
const MAX_PENDING_RECONCILE_BYTES = 64 * 1024;
const MAX_PENDING_RECONCILE_TEXT_LENGTH = 4000;
const MODAL_AGENT_EXEC_TIMEOUT_SECONDS = 150;
const MODAL_AGENT_EXEC_RETRY_LIMIT = 1;
const DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL =
  "https://gateway.ai.cloudflare.com/v1/d1215a30b84b673ef0367010b0e78c10/affiliate-manager/compat";

function buildTextSendPlan(content: string): Array<Record<string, JsonValue>> {
  const text = trim(content);
  return text ? [{ kind: "text", content: text }] : [];
}

function buildInteractiveCardSendPlan(
  card: Record<string, JsonValue>,
  receiveId: string,
  receiveIdType: string,
): Array<Record<string, JsonValue>> {
  return [
    {
      kind: "interactive_card",
      card,
      receive_id: trim(receiveId),
      receive_id_type: trim(receiveIdType) || "chat_id",
    },
  ];
}

function buildDeleteMessageSendPlan(messageId: string): Array<Record<string, JsonValue>> {
  const normalizedMessageId = trim(messageId);
  return normalizedMessageId ? [{ kind: "delete_message", message_id: normalizedMessageId }] : [];
}

type Env = {
  FEISHU_AGENT_WORKFLOW: WorkflowBinding<FeishuNormalizedPayload>;
  FEISHU_RECONCILE_QUEUE: DurableObjectNamespace;
  FEISHU_API_BASE?: string;
  FEISHU_APP_ID: string;
  FEISHU_APP_SECRET: string;
  FEISHU_VERIFICATION_TOKEN?: string;
  FEISHU_ENCRYPT_KEY?: string;
  FEISHU_ACK_REACTION_EMOJI?: string;
  CLOUDFLARE_API_TOKEN?: string;
  CLOUDFLARE_AI_GATEWAY_API_KEY?: string;
  CLOUDFLARE_AI_GATEWAY_BASE_URL?: string;
  MODAL_INTERNAL_BASE_URL: string;
  MODAL_INTERNAL_BEARER_TOKEN: string;
};

type TenantTokenCache = {
  token: string;
  expiresAt: number;
};

let tenantTokenCache: TenantTokenCache | null = null;

function jsonResponse(body: JsonValue, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "content-type": "application/json; charset=utf-8",
      ...(init.headers ?? {}),
    },
  });
}

function log(event: string, extra: Record<string, unknown>): void {
  console.log(JSON.stringify({ event, ...extra }));
}

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = trim(value);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function getApiBase(env: Env): string {
  return trim(env.FEISHU_API_BASE) || "https://open.feishu.cn";
}

function getCloudflareAiGatewayBase(env: Env): string {
  const configured = trim(env.CLOUDFLARE_AI_GATEWAY_BASE_URL) || DEFAULT_CLOUDFLARE_AI_GATEWAY_BASE_URL;
  return configured.replace(/\/chat\/completions$/i, "").replace(/\/$/, "");
}

function getCloudflareAiGatewayRoot(env: Env): string {
  return getCloudflareAiGatewayBase(env).replace(/\/compat$/i, "").replace(/\/$/, "");
}

function parsePositiveInt(value: unknown, fallback: number, min = 1, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = Number.parseInt(trim(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, parsed));
}

function parseJsonSafely(raw: string): Record<string, JsonValue> {
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? (parsed as Record<string, JsonValue>) : {};
  } catch {
    return {};
  }
}

function toWorkflowInstanceId(correlationId: string): string {
  const normalized = trim(correlationId).replace(/[^a-zA-Z0-9_-]/g, "_");
  if (normalized) {
    return normalized.slice(0, 64);
  }
  return `feishu_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

function readEvent(payload: Record<string, JsonValue>): Record<string, JsonValue> {
  const event = payload.event;
  return event && typeof event === "object" ? (event as Record<string, JsonValue>) : {};
}

function readRecord(parent: Record<string, JsonValue>, key: string): Record<string, JsonValue> {
  const value = parent[key];
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, JsonValue>)
    : {};
}

function readString(parent: Record<string, JsonValue>, key: string): string {
  return trim(parent[key]);
}

async function sha256Hex(input: string): Promise<string> {
  const data = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", data);
  const bytes = new Uint8Array(digest);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function isSignatureValid(request: Request, rawBody: string, env: Env): Promise<boolean> {
  const encryptKey = trim(env.FEISHU_ENCRYPT_KEY);
  if (!encryptKey) {
    return true;
  }
  const timestamp = trim(request.headers.get("x-lark-request-timestamp"));
  const nonce = trim(request.headers.get("x-lark-request-nonce"));
  const signature = trim(request.headers.get("x-lark-signature")).toLowerCase();
  if (!timestamp || !nonce || !signature) {
    return false;
  }
  const computed = await sha256Hex(`${timestamp}${nonce}${encryptKey}${rawBody}`);
  return computed === signature;
}

function isVerificationTokenValid(payload: Record<string, JsonValue>, env: Env): boolean {
  const expected = trim(env.FEISHU_VERIFICATION_TOKEN);
  if (!expected) {
    return true;
  }
  const header = readRecord(payload, "header");
  const token = readString(header, "token") || readString(payload, "token");
  return !token || token === expected;
}

function extractEventType(payload: Record<string, JsonValue>): string {
  const header = readRecord(payload, "header");
  return readString(header, "event_type") || readString(payload, "event_type");
}

function extractEventId(payload: Record<string, JsonValue>): string {
  const header = readRecord(payload, "header");
  return readString(header, "event_id") || readString(payload, "event_id");
}

function resolveControlLane(eventType: string): boolean {
  const normalized = eventType.toLowerCase();
  return normalized === "application.bot.menu_v6" || normalized === "card.action.trigger";
}

function parseFeishuMessageContent(rawContent: string): Record<string, JsonValue> {
  if (!rawContent) {
    return {};
  }
  return parseJsonSafely(rawContent);
}

function flattenPostContent(node: JsonValue): string[] {
  if (typeof node === "string") {
    return [node];
  }
  if (Array.isArray(node)) {
    return node.flatMap((item) => flattenPostContent(item));
  }
  if (!node || typeof node !== "object") {
    return [];
  }
  const record = node as Record<string, JsonValue>;
  const parts: string[] = [];
  for (const key of ["text", "title", "content"]) {
    if (record[key] !== undefined) {
      parts.push(...flattenPostContent(record[key] as JsonValue));
    }
  }
  return parts;
}

function messageExplicitlyRequestsBrowserTools(message: string): boolean {
  const text = trim(message).toLowerCase();
  if (!text) {
    return false;
  }
  const patterns = [
    /https?:\/\//,
    /\bbrowser\b/,
    /\bopen\b/,
    /\bvisit\b/,
    /\bnavigate\b/,
    /\bwebsite\b/,
    /浏览器/,
    /打开/,
    /访问/,
    /网页/,
    /网站/,
    /截图/,
    /注册/,
    /登录/,
    /cloudflare/,
    /google/,
  ];
  return patterns.some((pattern) => pattern.test(text));
}

function inferRouteHint(input: {
  lane: "control" | "agent" | "ignore";
  taskKind: string;
  messageType: FeishuNormalizedPayload["message_type"];
  text: string;
  attachmentCount: number;
}): RouteHint {
  if (input.lane === "control") {
    return "fast_control";
  }
  if (input.lane !== "agent") {
    return "modal_heavy_exec";
  }
  if (input.taskKind === "command" || input.messageType === "command") {
    return "fast_control";
  }
  if (input.attachmentCount > 0) {
    return "modal_heavy_exec";
  }
  if (messageExplicitlyRequestsBrowserTools(input.text)) {
    return "cf_browser_first";
  }
  return "modal_heavy_exec";
}

function buildSessionKey(
  platform: string,
  chatType: string,
  chatId: string,
  userId: string,
): string {
  const normalizedPlatform = trim(platform) || "feishu";
  const normalizedChatType = trim(chatType).toLowerCase() || "dm";
  const normalizedChatId = trim(chatId);
  const normalizedUserId = trim(userId);
  if (normalizedChatType === "dm") {
    if (normalizedChatId) {
      return `agent:main:${normalizedPlatform}:dm:${normalizedChatId}`;
    }
    return `agent:main:${normalizedPlatform}:dm`;
  }
  const parts = ["agent:main", normalizedPlatform, normalizedChatType];
  if (normalizedChatId) {
    parts.push(normalizedChatId);
  }
  if (normalizedUserId) {
    parts.push(normalizedUserId);
  }
  return parts.join(":");
}

function normalizeMessage(payload: Record<string, JsonValue>): FeishuNormalizedPayload {
  const event = readEvent(payload);
  const message = readRecord(event, "message");
  const sender = readRecord(event, "sender");
  const senderId = readRecord(sender, "sender_id");
  const chat = readRecord(event, "chat");
  const context = readRecord(event, "context");
  const operator = readRecord(event, "operator");
  const operatorId = readRecord(operator, "operator_id");
  const user = readRecord(event, "user");
  const userIdRecord = readRecord(event, "user_id");
  const messageTypeRaw = readString(message, "message_type").toLowerCase() || "text";
  const rawContent = readString(message, "content");
  const parsedContent = parseFeishuMessageContent(rawContent);
  const chatId =
    readString(message, "chat_id") ||
    readString(chat, "chat_id") ||
    readString(context, "open_chat_id") ||
    readString(context, "chat_id") ||
    readString(chat, "open_chat_id") ||
    readString(event, "open_chat_id") ||
    readString(event, "chat_id");
  const eventId = extractEventId(payload);
  const eventType = extractEventType(payload);
  const messageId =
    readString(message, "message_id") ||
    readString(context, "open_message_id") ||
    readString(context, "message_id");
  const chatTypeRaw = readString(chat, "chat_type") || readString(context, "chat_type");
  const chatType = chatTypeRaw.toLowerCase() === "p2p" ? "dm" : "group";
  const userId =
    readString(senderId, "open_id") ||
    readString(senderId, "user_id") ||
    readString(senderId, "union_id") ||
    readString(operatorId, "open_id") ||
    readString(operatorId, "user_id") ||
    readString(operatorId, "union_id") ||
    readString(operator, "open_id") ||
    readString(operator, "user_id") ||
    readString(operator, "union_id") ||
    readString(userIdRecord, "open_id") ||
    readString(userIdRecord, "user_id") ||
    readString(userIdRecord, "union_id") ||
    readString(user, "open_id") ||
    readString(user, "user_id") ||
    readString(user, "union_id") ||
    readString(event, "open_id") ||
    readString(event, "user_id") ||
    readString(event, "union_id");
  const userName =
    readString(sender, "name") ||
    readString(operator, "name") ||
    readString(user, "name") ||
    readString(sender, "sender_type") ||
    readString(operator, "operator_type") ||
    readString(user, "user_type") ||
    userId ||
    "feishu-user";
  const chatName = readString(chat, "name") || chatId || "Feishu Chat";
  const attachmentRefs: FeishuAttachmentRef[] = [];
  let text = "";
  let messageType: FeishuNormalizedPayload["message_type"] = "text";

  if (messageTypeRaw === "text") {
    text = trim(parsedContent.text);
    messageType = text.startsWith("/") ? "command" : "text";
  } else if (messageTypeRaw === "post") {
    text = flattenPostContent(parsedContent).join("\n").trim();
    messageType = "text";
  } else if (messageTypeRaw === "image") {
    const imageKey = trim(parsedContent.image_key);
    if (imageKey) {
      attachmentRefs.push({ resource_type: "image", message_id: messageId, image_key: imageKey });
    }
    messageType = "photo";
  } else if (messageTypeRaw === "audio") {
    const fileKey = trim(parsedContent.file_key);
    if (fileKey) {
      attachmentRefs.push({ resource_type: "audio", message_id: messageId, file_key: fileKey });
    }
    messageType = "audio";
  } else if (messageTypeRaw === "media") {
    const fileKey = trim(parsedContent.file_key);
    if (fileKey) {
      attachmentRefs.push({
        resource_type: "media",
        message_id: messageId,
        file_key: fileKey,
        file_name: trim(parsedContent.file_name),
      });
    }
    messageType = "video";
  } else {
    const fileKey = trim(parsedContent.file_key);
    if (fileKey) {
      attachmentRefs.push({
        resource_type: "file",
        message_id: messageId,
        file_key: fileKey,
        file_name: trim(parsedContent.file_name),
      });
    }
    messageType = "document";
  }

  const lane =
    resolveControlLane(eventType) ? "control" : eventType === "im.message.receive_v1" ? "agent" : "ignore";
  const routeHint = inferRouteHint({
    lane,
    taskKind: messageType,
    messageType,
    text,
    attachmentCount: attachmentRefs.length,
  });

  return {
    correlation_id: `feishu:${chatId}:${eventId || messageId || crypto.randomUUID()}`,
    session_key: buildSessionKey("feishu", chatType, chatId, userId),
    lane,
    route_hint: routeHint,
    task_kind: messageType,
    event_id: eventId,
    event_type: eventType,
    chat_id: chatId,
    chat_type: chatType,
    chat_name: chatName,
    user_id: userId,
    user_name: userName,
    message_id: messageId,
    message_type: messageType,
    text,
    attachment_refs: attachmentRefs,
    raw_payload: payload,
  };
}

async function getTenantAccessToken(env: Env): Promise<string> {
  const now = Date.now();
  if (tenantTokenCache && tenantTokenCache.expiresAt > now + 30_000) {
    return tenantTokenCache.token;
  }
  const response = await fetch(`${getApiBase(env)}/open-apis/auth/v3/tenant_access_token/internal`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      app_id: env.FEISHU_APP_ID,
      app_secret: env.FEISHU_APP_SECRET,
    }),
  });
  const payload = (await response.json()) as Record<string, JsonValue>;
  if (!response.ok || Number(payload.code ?? 0) !== 0) {
    throw new Error(`tenant_access_token_failed:${response.status}:${trim(payload.msg)}`);
  }
  const token = trim(payload.tenant_access_token);
  const expire = Number(payload.expire ?? 7200);
  tenantTokenCache = {
    token,
    expiresAt: now + Math.max(60, expire - 60) * 1000,
  };
  return token;
}

async function feishuApi(
  env: Env,
  method: string,
  path: string,
  init: {
    query?: URLSearchParams;
    body?: BodyInit | null;
    headers?: HeadersInit;
  } = {},
): Promise<Response> {
  const token = await getTenantAccessToken(env);
  const url = new URL(`${getApiBase(env)}${path}`);
  if (init.query) {
    url.search = init.query.toString();
  }
  return fetch(url.toString(), {
    method,
    headers: {
      authorization: `Bearer ${token}`,
      ...(init.headers ?? {}),
    },
    body: init.body ?? null,
  });
}

async function sendFeishuMessage(
  env: Env,
  options: {
    receiveId: string;
    receiveIdType?: string;
    msgType: "text" | "interactive" | "image" | "file" | "audio" | "media" | "post";
    content: Record<string, JsonValue> | string;
  },
): Promise<Record<string, JsonValue>> {
  const query = new URLSearchParams({
    receive_id_type: options.receiveIdType ?? "chat_id",
  });
  const payload = {
    receive_id: options.receiveId,
    msg_type: options.msgType,
    content: typeof options.content === "string" ? options.content : JSON.stringify(options.content),
    uuid: crypto.randomUUID(),
  };
  const response = await feishuApi(env, "POST", "/open-apis/im/v1/messages", {
    query,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = (await response.json()) as Record<string, JsonValue>;
  if (!response.ok || Number(body.code ?? 0) !== 0) {
    throw new Error(`send_message_failed:${response.status}:${trim(body.msg)}`);
  }
  return body;
}

async function deleteFeishuMessage(env: Env, messageId: string): Promise<void> {
  if (!trim(messageId)) {
    return;
  }
  const response = await feishuApi(env, "DELETE", `/open-apis/im/v1/messages/${messageId}`);
  if (!response.ok) {
    const body = (await response.text()).slice(0, 400);
    throw new Error(`delete_message_failed:${response.status}:${body}`);
  }
}

async function addAckReaction(env: Env, messageId: string): Promise<void> {
  if (!trim(messageId)) {
    return;
  }
  const response = await feishuApi(env, "POST", `/open-apis/im/v1/messages/${messageId}/reactions`, {
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      reaction_type: {
        emoji_type: trim(env.FEISHU_ACK_REACTION_EMOJI) || "OK",
      },
    }),
  });
  if (!response.ok) {
    const body = (await response.text()).slice(0, 300);
    throw new Error(`ack_reaction_failed:${response.status}:${body}`);
  }
}

async function uploadImageFromBytes(env: Env, bytes: ArrayBuffer, fileName: string): Promise<string> {
  const form = new FormData();
  form.set("image_type", "stream");
  form.set("image", new File([bytes], fileName, { type: "application/octet-stream" }));
  const response = await feishuApi(env, "POST", "/open-apis/im/v1/images", { body: form });
  const payload = (await response.json()) as Record<string, JsonValue>;
  if (!response.ok || Number(payload.code ?? 0) !== 0) {
    throw new Error(`upload_image_failed:${response.status}:${trim(payload.msg)}`);
  }
  const data = (payload.data ?? {}) as Record<string, JsonValue>;
  return trim(data.image_key);
}

function detectUploadFileType(fileName: string): "stream" | "opus" | "mp4" | "pdf" | "doc" | "xls" | "ppt" {
  const lower = fileName.toLowerCase();
  if (lower.endsWith(".opus") || lower.endsWith(".ogg") || lower.endsWith(".mp3") || lower.endsWith(".wav") || lower.endsWith(".m4a")) {
    return "opus";
  }
  if (lower.endsWith(".mp4") || lower.endsWith(".mov") || lower.endsWith(".webm")) {
    return "mp4";
  }
  if (lower.endsWith(".pdf")) {
    return "pdf";
  }
  if (lower.endsWith(".doc") || lower.endsWith(".docx")) {
    return "doc";
  }
  if (lower.endsWith(".xls") || lower.endsWith(".xlsx") || lower.endsWith(".csv")) {
    return "xls";
  }
  if (lower.endsWith(".ppt") || lower.endsWith(".pptx")) {
    return "ppt";
  }
  return "stream";
}

async function uploadFileFromBytes(env: Env, bytes: ArrayBuffer, fileName: string): Promise<string> {
  const form = new FormData();
  form.set("file_type", detectUploadFileType(fileName));
  form.set("file_name", fileName);
  form.set("file", new File([bytes], fileName, { type: "application/octet-stream" }));
  const response = await feishuApi(env, "POST", "/open-apis/im/v1/files", { body: form });
  const payload = (await response.json()) as Record<string, JsonValue>;
  if (!response.ok || Number(payload.code ?? 0) !== 0) {
    throw new Error(`upload_file_failed:${response.status}:${trim(payload.msg)}`);
  }
  const data = (payload.data ?? {}) as Record<string, JsonValue>;
  return trim(data.file_key);
}

async function fetchModalInternal<T>(env: Env, path: string, body: Record<string, JsonValue>): Promise<T> {
  const response = await fetch(`${trim(env.MODAL_INTERNAL_BASE_URL)}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.MODAL_INTERNAL_BEARER_TOKEN}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = (await response.text()).slice(0, 800);
    throw new Error(`modal_internal_failed:${response.status}:${text}`);
  }
  return (await response.json()) as T;
}

function normalizeChatMessages(raw: unknown): Array<Record<string, JsonValue>> {
  if (!Array.isArray(raw)) {
    return [];
  }
  const out: Array<Record<string, JsonValue>> = [];
  for (const item of raw) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }
    const role = trim((item as Record<string, JsonValue>).role);
    const content = trim((item as Record<string, JsonValue>).content);
    if (!role || !content) {
      continue;
    }
    out.push({ role, content });
  }
  return out;
}

function extractAssistantTextFromChatCompletion(payload: Record<string, JsonValue>): string {
  const choices = payload.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    return "";
  }
  const first = choices[0];
  if (!first || typeof first !== "object" || Array.isArray(first)) {
    return "";
  }
  const message = (first as Record<string, JsonValue>).message;
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return "";
  }
  const content = (message as Record<string, JsonValue>).content;
  if (typeof content === "string") {
    return trim(content);
  }
  if (Array.isArray(content)) {
    return content
      .map((entry) => {
        if (typeof entry === "string") {
          return trim(entry);
        }
        if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
          return "";
        }
        return trim((entry as Record<string, JsonValue>).text);
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  return "";
}

async function executeCloudflareAiExec(
  env: Env,
  normalized: FeishuNormalizedPayload,
  planned: ModalInternalResponse,
): Promise<ModalInternalResponse> {
  const token = trim(env.CLOUDFLARE_AI_GATEWAY_API_KEY) || trim(env.CLOUDFLARE_API_TOKEN);
  const llmRequest = (planned.llm_request ?? {}) as Record<string, JsonValue>;
  const providerPlan = (planned.provider_plan ?? {}) as Record<string, JsonValue>;
  const model = trim(llmRequest.model) || trim(providerPlan.model) || "openrouter/free";
  const messages = normalizeChatMessages(llmRequest.messages);
  if (!token || messages.length === 0) {
    throw new Error(`cf_ai_exec_unavailable:${token ? "missing_messages" : "missing_token"}`);
  }
  const startedAt = Date.now();
  const gatewayBase = trim(providerPlan.base_url) || getCloudflareAiGatewayBase(env);
  const provider = trim(providerPlan.provider).toLowerCase();
  const byokAlias = trim(providerPlan.byok_alias) || "default";
  const requestTimeoutMs = parsePositiveInt(providerPlan.request_timeout_ms, 4500, 1000, 30000);
  const maxAttempts = parsePositiveInt(providerPlan.max_attempts, 1, 1, 5);
  const retryDelayMs = parsePositiveInt(providerPlan.retry_delay_ms, 250, 0, 5000);
  const backoff = (() => {
    const candidate = trim(providerPlan.backoff).toLowerCase();
    return candidate === "constant" || candidate === "exponential" ? candidate : "linear";
  })();
  const cacheMode = (() => {
    const candidate = trim(providerPlan.cache_mode).toLowerCase();
    return candidate === "ttl" ? "ttl" : "skip";
  })();
  const cacheScope = (() => {
    const candidate = trim(providerPlan.cache_scope).toLowerCase();
    return candidate === "chat" || candidate === "global" ? candidate : "user";
  })();
  const cacheTtlSeconds = parsePositiveInt(providerPlan.cache_ttl_seconds, 300, 60, 31 * 24 * 3600);
  const fallbackModel = trim(providerPlan.fallback_model);
  const fallbackProvider = trim(providerPlan.fallback_provider).toLowerCase() || provider;

  const requestOnce = async (
    targetProvider: string,
    targetModel: string,
    reason: "primary" | "fallback",
  ): Promise<{
    response: Response;
    payload: Record<string, JsonValue>;
    requestUrl: string;
    cacheStatus: string;
    targetProvider: string;
    targetModel: string;
  }> => {
    const requestUrl =
      targetProvider === "openrouter"
        ? `${getCloudflareAiGatewayRoot(env)}/openrouter/chat/completions`
        : `${gatewayBase}/chat/completions`;
    const headers: Record<string, string> = {
      "content-type": "application/json",
      "cf-aig-authorization": `Bearer ${token}`,
      "cf-aig-metadata": JSON.stringify({
        correlation_id: normalized.correlation_id,
        feishu_event_id: normalized.event_id,
        chat_id: normalized.chat_id,
        session_key: normalized.session_key,
        task_kind: normalized.task_kind,
      }),
      "cf-aig-request-timeout": String(requestTimeoutMs),
      "cf-aig-max-attempts": String(maxAttempts),
      "cf-aig-retry-delay": String(retryDelayMs),
      "cf-aig-backoff": backoff,
    };
    if (cacheMode === "ttl") {
      headers["cf-aig-cache-ttl"] = String(cacheTtlSeconds);
      const cacheIdentity =
        cacheScope === "global"
          ? "global"
          : cacheScope === "chat"
            ? normalized.chat_id || normalized.session_key || "anon-chat"
            : normalized.user_id || normalized.session_key || "anon-user";
      const requestFingerprint = await sha256Hex(
        JSON.stringify({
          provider: targetProvider,
          model: targetModel,
          messages,
          personality: trim(providerPlan.personality),
          task_kind: normalized.task_kind,
        }),
      );
      const identityFingerprint = await sha256Hex(cacheIdentity);
      headers["cf-aig-cache-key"] = `feishu:${cacheScope}:${identityFingerprint}:${requestFingerprint}`;
    } else {
      headers["cf-aig-skip-cache"] = "true";
    }
    if (targetProvider === "openrouter" && byokAlias) {
      headers["cf-aig-byok-alias"] = byokAlias;
    }
    const response = await fetch(requestUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: targetModel,
        messages,
        stream: false,
      }),
    });
    const payload = (await response.json()) as Record<string, JsonValue>;
    const cacheStatus = trim(response.headers.get("cf-aig-cache-status"));
    if (!response.ok) {
      log("feishu.cf_ai_exec.attempt_failed", {
        correlation_id: normalized.correlation_id,
        reason,
        provider: targetProvider || "openrouter",
        model: targetModel,
        status_code: response.status,
        request_url: requestUrl,
        cf_cache_status: cacheStatus || "none",
      });
    }
    return { response, payload, requestUrl, cacheStatus, targetProvider, targetModel };
  };

  let result = await requestOnce(provider || "openrouter", model, "primary");
  if (
    !result.response.ok &&
    fallbackModel &&
    fallbackModel !== model &&
    [408, 429, 500, 502, 503, 504].includes(result.response.status)
  ) {
    log("feishu.cf_ai_exec.retrying_with_fallback_model", {
      correlation_id: normalized.correlation_id,
      primary_provider: result.targetProvider || "openrouter",
      primary_model: model,
      fallback_provider: fallbackProvider || result.targetProvider || "openrouter",
      fallback_model: fallbackModel,
      status_code: result.response.status,
    });
    result = await requestOnce(fallbackProvider || result.targetProvider || "openrouter", fallbackModel, "fallback");
  }

  if (!result.response.ok) {
    throw new Error(`cf_ai_exec_failed:${result.response.status}:${JSON.stringify(result.payload).slice(0, 500)}`);
  }

  const assistantText = extractAssistantTextFromChatCompletion(result.payload);
  if (!assistantText) {
    throw new Error("cf_ai_exec_empty_response");
  }
  const usage =
    result.payload.usage && typeof result.payload.usage === "object" && !Array.isArray(result.payload.usage)
      ? (result.payload.usage as Record<string, JsonValue>)
      : {};
  log("feishu.cf_ai_exec.done", {
    correlation_id: normalized.correlation_id,
    route_hint: planned.route_hint || normalized.route_hint,
    model: result.targetModel,
    provider: result.targetProvider || "openrouter",
    request_url: result.requestUrl,
    cf_cache_status: result.cacheStatus || "none",
    cf_ai_exec_elapsed_ms: Date.now() - startedAt,
  });
  return {
    status: "ok",
    route_hint: (planned.route_hint || normalized.route_hint) as RouteHint,
    execution_mode: "deferred_reconcile",
    final_response: assistantText,
    send_plan: buildTextSendPlan(assistantText),
    action_plan: buildTextSendPlan(assistantText),
    reconcile_required: true,
    provider_usage: {
      provider: result.targetProvider || "openrouter",
      response_model: result.targetModel,
      gateway_base_url: gatewayBase,
      request_url: result.requestUrl,
      cf_cache_status: result.cacheStatus || "none",
      completion_usage: usage,
    },
  };
}

function summarizeOperationsForHermes(sendPlan: Array<Record<string, JsonValue>>): {
  assistantText: string;
  operationKinds: string[];
} {
  const operationKinds = uniqueStrings(
    sendPlan.map((item) => trim(item.kind).toLowerCase()).filter(Boolean),
  );
  const textParts = uniqueStrings(
    sendPlan.map((item) => trim(item.content)).filter(Boolean),
  );
  const placeholderMap: Record<string, string> = {
    image: "[Sent image]",
    image_url: "[Sent image]",
    image_file: "[Sent image]",
    file: "[Sent file]",
    document_file: "[Sent file]",
    audio: "[Sent audio]",
    audio_file: "[Sent audio]",
    media: "[Sent media]",
    video_file: "[Sent media]",
    interactive: "[Sent card]",
  };
  const placeholders = uniqueStrings(
    operationKinds.map((kind) => placeholderMap[kind] ?? "").filter(Boolean),
  );
  return {
    assistantText: uniqueStrings([...textParts, ...placeholders]).join("\n").trim(),
    operationKinds,
  };
}

function buildPendingReconcile(
  normalized: FeishuNormalizedPayload,
  internal: ModalInternalResponse,
): PendingReconcileItem | null {
  const sendPlan = Array.isArray(internal.action_plan)
    ? internal.action_plan
    : Array.isArray(internal.send_plan)
      ? internal.send_plan
      : [];
  const summary = summarizeOperationsForHermes(sendPlan);
  const finalResponse = trim(internal.final_response);
  const assistantText = trim(summary.assistantText || finalResponse);
  if (!assistantText && summary.operationKinds.length === 0) {
    return null;
  }
  return {
    correlation_id: normalized.correlation_id,
    assistant_text: assistantText,
    final_response: finalResponse,
    operation_kinds: summary.operationKinds,
    delivered_at: new Date().toISOString(),
  };
}

function trimPendingReconcileItem(item: PendingReconcileItem): PendingReconcileItem {
  return {
    ...item,
    assistant_text: trim(item.assistant_text).slice(0, MAX_PENDING_RECONCILE_TEXT_LENGTH),
    final_response: trim(item.final_response).slice(0, MAX_PENDING_RECONCILE_TEXT_LENGTH),
    operation_kinds: uniqueStrings(Array.isArray(item.operation_kinds) ? item.operation_kinds : []),
    delivered_at: trim(item.delivered_at) || new Date().toISOString(),
  };
}

function estimatePendingReconcileBytes(items: PendingReconcileItem[]): number {
  return new TextEncoder().encode(JSON.stringify(items)).length;
}

function buildPendingReconcileSummary(items: PendingReconcileItem[]): PendingReconcileItem | null {
  if (items.length === 0) {
    return null;
  }
  const operationKinds = uniqueStrings(items.flatMap((item) => item.operation_kinds || []));
  const references = uniqueStrings(items.map((item) => trim(item.correlation_id)).filter(Boolean)).slice(0, 5);
  const assistantText = [
    `[Pending reconcile summary] ${items.length} earlier Cloudflare-delivered results were compacted.`,
    operationKinds.length > 0 ? `Kinds: ${operationKinds.join(", ")}` : "",
    references.length > 0 ? `Refs: ${references.join(", ")}` : "",
  ]
    .filter(Boolean)
    .join("\n");
  return {
    correlation_id: `summary:${Date.now()}`,
    assistant_text: assistantText.slice(0, MAX_PENDING_RECONCILE_TEXT_LENGTH),
    final_response: "",
    operation_kinds: operationKinds,
    delivered_at: new Date().toISOString(),
  };
}

function compactPendingReconcileItems(items: PendingReconcileItem[]): {
  items: PendingReconcileItem[];
  compacted: number;
} {
  const nextItems = items.map((item) => trimPendingReconcileItem(item));
  const removed: PendingReconcileItem[] = [];
  while (
    nextItems.length > MAX_PENDING_RECONCILE_ITEMS ||
    estimatePendingReconcileBytes(nextItems) > MAX_PENDING_RECONCILE_BYTES
  ) {
    if (nextItems.length <= 1) {
      break;
    }
    const dropped = nextItems.shift();
    if (dropped) {
      removed.push(dropped);
    }
  }
  if (removed.length > 0) {
    const summary = buildPendingReconcileSummary(removed);
    if (summary) {
      nextItems.unshift(summary);
    }
    while (
      nextItems.length > MAX_PENDING_RECONCILE_ITEMS ||
      estimatePendingReconcileBytes(nextItems) > MAX_PENDING_RECONCILE_BYTES
    ) {
      nextItems.pop();
    }
  }
  return {
    items: nextItems,
    compacted: removed.length,
  };
}

function classifyModalInternalError(error: unknown): { retryClass: string; retryable: boolean } {
  const message = error instanceof Error ? error.message : String(error);
  if (/modal_internal_failed:(429|502|503|504|524):/i.test(message) || /timeout/i.test(message)) {
    return { retryClass: "retryable_upstream", retryable: true };
  }
  if (/modal_internal_failed:401:/i.test(message)) {
    return { retryClass: "auth_error", retryable: false };
  }
  if (/modal_internal_failed:400:/i.test(message)) {
    return { retryClass: "bad_request", retryable: false };
  }
  return { retryClass: "fatal", retryable: false };
}

async function reconcileQueueRequest<T extends JsonValue>(
  env: Env,
  sessionKey: string,
  path: "/peek" | "/enqueue" | "/ack",
  body?: Record<string, JsonValue>,
): Promise<T> {
  const stub = env.FEISHU_RECONCILE_QUEUE.getByName(trim(sessionKey) || "agent:main:feishu:unknown");
  const response = await stub.fetch(`https://reconcile${path}`, {
    method: "POST",
    headers: { "content-type": "application/json; charset=utf-8" },
    body: JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    const text = (await response.text()).slice(0, 400);
    throw new Error(`reconcile_queue_failed:${path}:${response.status}:${text}`);
  }
  return (await response.json()) as T;
}

async function peekPendingReconciles(env: Env, sessionKey: string): Promise<PendingReconcileItem[]> {
  const payload = await reconcileQueueRequest<Record<string, JsonValue>>(env, sessionKey, "/peek");
  const items = Array.isArray(payload.items) ? payload.items : [];
  return items.filter((item): item is PendingReconcileItem => Boolean(item && typeof item === "object"));
}

async function enqueuePendingReconcile(env: Env, sessionKey: string, item: PendingReconcileItem): Promise<void> {
  await reconcileQueueRequest(env, sessionKey, "/enqueue", { item: item as unknown as JsonValue });
}

async function ackPendingReconciles(env: Env, sessionKey: string, correlationIds: string[]): Promise<void> {
  const ids = uniqueStrings(correlationIds);
  if (ids.length === 0) {
    return;
  }
  await reconcileQueueRequest(env, sessionKey, "/ack", {
    correlation_ids: ids as unknown as JsonValue,
  });
}

async function fetchModalInternalWithPendingReconciles<T>(
  env: Env,
  path: string,
  normalized: FeishuNormalizedPayload,
  body: Record<string, JsonValue>,
): Promise<T> {
  const pending = await peekPendingReconciles(env, normalized.session_key);
  if (pending.length > 0) {
    log("feishu.reconcile.peek", {
      session_key: normalized.session_key,
      correlation_id: normalized.correlation_id,
      pending_count: pending.length,
    });
  }
  const response = await fetchModalInternal<T>(env, path, {
    ...body,
    session_key: normalized.session_key,
    pending_reconcile: pending as unknown as JsonValue,
  });
  if (pending.length > 0) {
    await ackPendingReconciles(
      env,
      normalized.session_key,
      pending.map((item) => item.correlation_id),
    );
    log("feishu.reconcile.ack", {
      session_key: normalized.session_key,
      correlation_id: normalized.correlation_id,
      ack_count: pending.length,
    });
  }
  return response;
}

async function fetchModalResultFile(env: Env, token: string): Promise<{ bytes: ArrayBuffer; fileName: string }> {
  const response = await fetch(`${trim(env.MODAL_INTERNAL_BASE_URL)}/internal/feishu/result-file/${encodeURIComponent(token)}`, {
    headers: {
      authorization: `Bearer ${env.MODAL_INTERNAL_BEARER_TOKEN}`,
    },
  });
  if (!response.ok) {
    const text = (await response.text()).slice(0, 400);
    throw new Error(`modal_result_file_failed:${response.status}:${text}`);
  }
  const disposition = response.headers.get("content-disposition") ?? "";
  const filenameMatch = disposition.match(/filename=\"?([^\";]+)\"?/i);
  return {
    bytes: await response.arrayBuffer(),
    fileName: filenameMatch?.[1] ?? "artifact.bin",
  };
}

function resolveMenuTarget(payload: Record<string, JsonValue>): { receiveId: string; receiveIdType: string } | null {
  const event = readEvent(payload);
  const context = readRecord(event, "context");
  const chat = readRecord(event, "chat");
  const operator = readRecord(event, "operator");
  const operatorId = readRecord(operator, "operator_id");
  const openChatId = readString(context, "open_chat_id");
  const chatId = readString(chat, "chat_id") || openChatId;
  const openId = readString(operatorId, "open_id");
  if (chatId.startsWith("oc_")) {
    return { receiveId: chatId, receiveIdType: "chat_id" };
  }
  if (openId) {
    return { receiveId: openId, receiveIdType: "open_id" };
  }
  if (chatId) {
    return { receiveId: chatId, receiveIdType: "chat_id" };
  }
  return null;
}

function controlPayloadBase(normalized: FeishuNormalizedPayload): Record<string, JsonValue> {
  return {
    correlation_id: normalized.correlation_id,
    session_key: normalized.session_key,
    lane: normalized.lane,
    route_hint: normalized.route_hint,
    task_kind: normalized.task_kind,
    chat_id: normalized.chat_id,
    chat_type: normalized.chat_type,
    chat_name: normalized.chat_name,
    user_id: normalized.user_id,
    user_name: normalized.user_name,
    message_id: normalized.message_id,
  };
}

async function sendStateText(env: Env, normalized: FeishuNormalizedPayload, internal: ModalInternalResponse): Promise<void> {
  const lines = internal.session_state_after?.route_status_lines ?? [];
  const content =
    lines.length > 0
      ? lines.join("\n")
      : [
          `Current model: ${internal.session_state_after?.current_model ?? "unknown"}`,
          `Current provider: ${internal.session_state_after?.current_provider ?? "unknown"}`,
          `Current personality: ${internal.session_state_after?.current_personality ?? "none"}`,
        ].join("\n");
  await sendFeishuMessage(env, {
    receiveId: normalized.chat_id,
    receiveIdType: "chat_id",
    msgType: "text",
    content: { text: content },
  });
}

async function executeFeishuOperations(
  env: Env,
  normalized: FeishuNormalizedPayload,
  operations: Array<Record<string, JsonValue>>,
): Promise<void> {
  for (const operation of operations) {
    const kind = trim(operation.kind);
    if (!kind) {
      continue;
    }
    if (kind === "delete_message") {
      await deleteFeishuMessage(env, trim(operation.message_id));
      continue;
    }
    if (kind === "interactive_card") {
      const card = operation.card;
      if (!card || typeof card !== "object" || Array.isArray(card)) {
        continue;
      }
      await sendFeishuMessage(env, {
        receiveId: trim(operation.receive_id) || normalized.chat_id,
        receiveIdType: trim(operation.receive_id_type) || "chat_id",
        msgType: "interactive",
        content: card as Record<string, JsonValue>,
      });
      continue;
    }
    if (kind === "text" || kind === "edit") {
      const content = trim(operation.content);
      if (!content) {
        continue;
      }
      await sendFeishuMessage(env, {
        receiveId: normalized.chat_id,
        receiveIdType: "chat_id",
        msgType: "text",
        content: { text: content },
      });
      continue;
    }
    if (kind === "image_url") {
      const imageUrl = trim(operation.image_url);
      if (!imageUrl) {
        continue;
      }
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`image_fetch_failed:${response.status}:${imageUrl}`);
      }
      const imageKey = await uploadImageFromBytes(env, await response.arrayBuffer(), "image.bin");
      await sendFeishuMessage(env, {
        receiveId: normalized.chat_id,
        receiveIdType: "chat_id",
        msgType: "image",
        content: { image_key: imageKey },
      });
      continue;
    }
    const file = operation.file;
    if (!file || typeof file !== "object") {
      continue;
    }
    const token = trim((file as Record<string, JsonValue>).token);
    if (!token) {
      continue;
    }
    const modalFile = await fetchModalResultFile(env, token);
    if (kind === "image_file") {
      const imageKey = await uploadImageFromBytes(env, modalFile.bytes, modalFile.fileName);
      await sendFeishuMessage(env, {
        receiveId: normalized.chat_id,
        receiveIdType: "chat_id",
        msgType: "image",
        content: { image_key: imageKey },
      });
      continue;
    }
    const fileKey = await uploadFileFromBytes(env, modalFile.bytes, modalFile.fileName);
    const msgType = kind === "audio_file" ? "audio" : kind === "video_file" ? "media" : "file";
    await sendFeishuMessage(env, {
      receiveId: normalized.chat_id,
      receiveIdType: "chat_id",
      msgType,
      content: { file_key: fileKey },
    });
  }
}

async function sendPlanToFeishu(
  env: Env,
  normalized: FeishuNormalizedPayload,
  sendPlan: Array<Record<string, JsonValue>>,
): Promise<void> {
  await executeFeishuOperations(env, normalized, sendPlan);
}

async function handleMenuEvent(env: Env, normalized: FeishuNormalizedPayload): Promise<ModalInternalResponse> {
  const event = readEvent(normalized.raw_payload);
  const eventKey = readString(event, "event_key");
  const target = resolveMenuTarget(normalized.raw_payload);
  if (!target) {
    throw new Error(`menu_target_missing:${eventKey}`);
  }
  if (eventKey === "model_status" || eventKey === "route_status") {
    const internal = await fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
      ...controlPayloadBase(normalized),
      action: "get_session_state",
    });
    const lines = internal.session_state_after?.route_status_lines ?? [];
    const content =
      lines.length > 0
        ? lines.join("\n")
        : [
            `Current model: ${internal.session_state_after?.current_model ?? "unknown"}`,
            `Current provider: ${internal.session_state_after?.current_provider ?? "unknown"}`,
            `Current personality: ${internal.session_state_after?.current_personality ?? "none"}`,
          ].join("\n");
    return {
      ...internal,
      send_plan: buildTextSendPlan(content),
      action_plan: buildTextSendPlan(content),
    };
  }
  if (eventKey === "provider_status") {
    return fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
      ...controlPayloadBase(normalized),
      action: "dispatch_command",
      command_text: "/provider",
    });
  }
  const internal = await fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
    ...controlPayloadBase(normalized),
    action: "render_card",
    event_key: eventKey,
  });
  if (!internal.card) {
    throw new Error(internal.error || `render_card_failed:${eventKey}`);
  }
  return {
    ...internal,
    send_plan: buildInteractiveCardSendPlan(internal.card, target.receiveId, target.receiveIdType),
    action_plan: buildInteractiveCardSendPlan(internal.card, target.receiveId, target.receiveIdType),
  };
}

async function handleCardAction(env: Env, normalized: FeishuNormalizedPayload): Promise<ModalInternalResponse> {
  const event = readEvent(normalized.raw_payload);
  const action = readRecord(event, "action");
  const actionValue = readRecord(action, "value");
  const hermesAction = trim(actionValue.hermes_action);
  if (hermesAction === "registry_close_card") {
    const context = readRecord(event, "context");
    const messageId = readString(context, "open_message_id") || readString(context, "message_id");
    const deletePlan = buildDeleteMessageSendPlan(messageId);
    return {
      status: "ok",
      route_hint: "fast_control",
      execution_mode: "control_complete",
      send_plan: deletePlan,
      action_plan: deletePlan,
    };
  }
  if (hermesAction === "open_menu_card") {
    const internal = await fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
      ...controlPayloadBase(normalized),
      action: "render_card",
      event_key: trim(actionValue.event_key),
    });
    if (!internal.card) {
      throw new Error(internal.error || "open_menu_card_failed");
    }
    return {
      ...internal,
      send_plan: buildInteractiveCardSendPlan(internal.card, normalized.chat_id, "chat_id"),
      action_plan: buildInteractiveCardSendPlan(internal.card, normalized.chat_id, "chat_id"),
    };
  }

  let internal: ModalInternalResponse;
  if (hermesAction === "skill_combo_apply") {
    internal = await fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
      ...controlPayloadBase(normalized),
      action: "activate_skill_combo",
      combo_id: trim(actionValue.combo_id),
    });
  } else {
    let commandText = "";
    if (hermesAction === "registry_switch_model") {
      commandText = `/model ${trim(actionValue.model)} --provider ${trim(actionValue.provider)}`;
    } else if (hermesAction === "personality_set") {
      commandText = `/personality ${trim(actionValue.personality) || "none"}`;
    } else if (hermesAction === "command_run") {
      commandText = trim(actionValue.command_text);
    }
    if (!commandText) {
      throw new Error(`unsupported_card_action:${hermesAction}`);
    }
    internal = await fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
      ...controlPayloadBase(normalized),
      action: "dispatch_command",
      command_text: commandText,
    });
  }
  return internal;
}

async function handleControlEvent(env: Env, normalized: FeishuNormalizedPayload): Promise<ModalInternalResponse> {
  if (normalized.event_type === "application.bot.menu_v6") {
    return handleMenuEvent(env, normalized);
  }
  if (normalized.event_type === "card.action.trigger") {
    return handleCardAction(env, normalized);
  }
  return { status: "ignored", send_plan: [] };
}

async function invokeAgentExec(
  env: Env,
  normalized: FeishuNormalizedPayload,
): Promise<ModalInternalResponse> {
  let attempt = 0;
  let lastRetryClass = "";
  while (true) {
    attempt += 1;
    try {
      const internal = await fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/agent-exec", normalized, {
        correlation_id: normalized.correlation_id,
        session_key: normalized.session_key,
        lane: normalized.lane,
        route_hint: normalized.route_hint,
        task_kind: normalized.task_kind,
        event_id: normalized.event_id,
        event_type: normalized.event_type,
        chat_id: normalized.chat_id,
        chat_type: normalized.chat_type,
        chat_name: normalized.chat_name,
        user_id: normalized.user_id,
        user_name: normalized.user_name,
        message_id: normalized.message_id,
        message_type: normalized.message_type,
        text: normalized.text,
        attachment_refs: normalized.attachment_refs as unknown as JsonValue,
        raw_message: normalized.raw_payload,
      });
      return {
        ...internal,
        retry_class: trim(internal.retry_class) || lastRetryClass,
        retry_attempt_count: attempt,
      };
    } catch (error) {
      const retryInfo = classifyModalInternalError(error);
      lastRetryClass = retryInfo.retryClass;
      log("feishu.modal_internal.error", {
        correlation_id: normalized.correlation_id,
        route_hint: normalized.route_hint,
        retry_class: retryInfo.retryClass,
        retryable: retryInfo.retryable,
        retry_attempt_count: attempt,
        error: error instanceof Error ? error.message : String(error),
      });
      if (!retryInfo.retryable || attempt > MODAL_AGENT_EXEC_RETRY_LIMIT) {
        throw error;
      }
      log("feishu.modal_internal.retry", {
        correlation_id: normalized.correlation_id,
        route_hint: normalized.route_hint,
        retry_class: retryInfo.retryClass,
        retry_attempt_count: attempt,
      });
    }
  }
}

async function invokeAgentPlan(
  env: Env,
  normalized: FeishuNormalizedPayload,
): Promise<ModalInternalResponse> {
  return fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/agent-plan", normalized, {
    correlation_id: normalized.correlation_id,
    session_key: normalized.session_key,
    lane: normalized.lane,
    route_hint: normalized.route_hint,
    task_kind: normalized.task_kind,
    event_id: normalized.event_id,
    event_type: normalized.event_type,
    chat_id: normalized.chat_id,
    chat_type: normalized.chat_type,
    chat_name: normalized.chat_name,
    user_id: normalized.user_id,
    user_name: normalized.user_name,
    message_id: normalized.message_id,
    message_type: normalized.message_type,
    text: normalized.text,
    attachment_refs: normalized.attachment_refs as unknown as JsonValue,
    raw_message: normalized.raw_payload,
  });
}

async function handleAgentCommand(env: Env, normalized: FeishuNormalizedPayload): Promise<ModalInternalResponse> {
  return fetchModalInternalWithPendingReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
    ...controlPayloadBase(normalized),
    action: "dispatch_command",
    command_text: normalized.text,
  });
}

function getExecutablePlan(internal: ModalInternalResponse): Array<Record<string, JsonValue>> {
  if (Array.isArray(internal.action_plan) && internal.action_plan.length > 0) {
    return internal.action_plan;
  }
  if (Array.isArray(internal.send_plan)) {
    return internal.send_plan;
  }
  return [];
}

function shouldAckReactionInline(normalized: FeishuNormalizedPayload): boolean {
  return normalized.event_type === "im.message.receive_v1" && !!trim(normalized.message_id);
}

function shouldUseDirectWorkerFastPath(normalized: FeishuNormalizedPayload): boolean {
  return normalized.lane === "control" || (normalized.lane === "agent" && normalized.route_hint === "fast_control");
}

function shouldUseDirectWorkerPlannedPath(normalized: FeishuNormalizedPayload): boolean {
  return (
    normalized.lane === "agent" &&
    normalized.route_hint === "modal_heavy_exec" &&
    normalized.task_kind === "text" &&
    normalized.attachment_refs.length === 0 &&
    !trim(normalized.text).startsWith("/")
  );
}

async function enqueueWorkflowInstance(env: Env, normalized: FeishuNormalizedPayload): Promise<void> {
  await env.FEISHU_AGENT_WORKFLOW.create({
    id: toWorkflowInstanceId(normalized.correlation_id),
    params: normalized,
  });
}

async function runAckReactionFastPath(env: Env, normalized: FeishuNormalizedPayload): Promise<void> {
  if (!shouldAckReactionInline(normalized)) {
    return;
  }
  const startedAt = Date.now();
  try {
    await addAckReaction(env, normalized.message_id);
    log("feishu.ack_reaction.done", {
      correlation_id: normalized.correlation_id,
      message_id: normalized.message_id,
      ack_reaction_elapsed_ms: Date.now() - startedAt,
      execution_path: "worker_direct",
    });
  } catch (error) {
    log("feishu.ack_reaction.error", {
      correlation_id: normalized.correlation_id,
      message_id: normalized.message_id,
      ack_reaction_elapsed_ms: Date.now() - startedAt,
      execution_path: "worker_direct",
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

async function runDirectWorkerFastPath(env: Env, normalized: FeishuNormalizedPayload): Promise<void> {
  const directStartedAt = Date.now();
  log("feishu.direct_fast.start", {
    correlation_id: normalized.correlation_id,
    lane: normalized.lane,
    route_hint: normalized.route_hint,
    task_kind: normalized.task_kind,
  });
  const internal =
    normalized.lane === "control"
      ? ({
          ...(await handleControlEvent(env, normalized)),
          route_hint: "fast_control",
          execution_mode: "control_complete",
        } satisfies ModalInternalResponse)
      : ({
          ...(await handleAgentCommand(env, normalized)),
          route_hint: "fast_control",
          execution_mode: "control_complete",
        } satisfies ModalInternalResponse);

  const executablePlan = getExecutablePlan(internal);
  const sendStartedAt = Date.now();
  await executeFeishuOperations(env, normalized, executablePlan);
  const sendElapsedMs = Date.now() - sendStartedAt;

  if (normalized.lane === "agent" && internal.reconcile_required !== false) {
    const pending = buildPendingReconcile(normalized, internal);
    if (pending) {
      await enqueuePendingReconcile(env, normalized.session_key, pending);
      log("feishu.reconcile.enqueue", {
        session_key: normalized.session_key,
        correlation_id: pending.correlation_id,
        operation_count: pending.operation_kinds.length,
        execution_path: "worker_direct",
      });
    }
  }

  log("feishu.direct_fast.done", {
    correlation_id: normalized.correlation_id,
    lane: normalized.lane,
    route_hint: internal.route_hint || normalized.route_hint,
    execution_mode: internal.execution_mode || "control_complete",
    direct_exec_elapsed_ms: Date.now() - directStartedAt,
    direct_send_elapsed_ms: sendElapsedMs,
    operation_count: executablePlan.length,
  });
}

async function runDirectWorkerPlannedPath(env: Env, normalized: FeishuNormalizedPayload): Promise<void> {
  const directStartedAt = Date.now();
  log("feishu.direct_planned.start", {
    correlation_id: normalized.correlation_id,
    lane: normalized.lane,
    route_hint: normalized.route_hint,
    task_kind: normalized.task_kind,
  });

  const planned = await invokeAgentPlan(env, normalized);
  const effectiveRouteHint = trim(planned.route_hint) || normalized.route_hint;
  log("feishu.direct_planned.plan.done", {
    correlation_id: normalized.correlation_id,
    route_hint: effectiveRouteHint,
    execution_mode: trim(planned.execution_mode) || "modal_heavy_exec",
    external_exec_candidate: !!planned.external_exec_candidate,
  });

  if (!planned.external_exec_candidate) {
    await enqueueWorkflowInstance(env, normalized);
    log("feishu.direct_planned.workflow_enqueued", {
      correlation_id: normalized.correlation_id,
      route_hint: effectiveRouteHint,
      enqueue_elapsed_ms: Date.now() - directStartedAt,
      reason: "planner_not_external_candidate",
    });
    return;
  }

  let internal: ModalInternalResponse;
  try {
    internal = await executeCloudflareAiExec(env, normalized, planned);
  } catch (error) {
    log("feishu.direct_planned.cf_ai_exec.fallback", {
      correlation_id: normalized.correlation_id,
      route_hint: effectiveRouteHint,
      error: error instanceof Error ? error.message : String(error),
    });
    await enqueueWorkflowInstance(env, normalized);
    log("feishu.direct_planned.workflow_enqueued", {
      correlation_id: normalized.correlation_id,
      route_hint: effectiveRouteHint,
      enqueue_elapsed_ms: Date.now() - directStartedAt,
      reason: "cf_ai_exec_fallback",
    });
    return;
  }

  const executablePlan = getExecutablePlan(internal);
  const sendStartedAt = Date.now();
  await executeFeishuOperations(env, normalized, executablePlan);
  const sendElapsedMs = Date.now() - sendStartedAt;

  if (internal.reconcile_required !== false) {
    const pending = buildPendingReconcile(normalized, internal);
    if (pending) {
      await enqueuePendingReconcile(env, normalized.session_key, pending);
      log("feishu.reconcile.enqueue", {
        session_key: normalized.session_key,
        correlation_id: pending.correlation_id,
        operation_count: pending.operation_kinds.length,
        execution_path: "worker_direct_planned",
      });
    }
  }

  log("feishu.direct_planned.done", {
    correlation_id: normalized.correlation_id,
    route_hint: internal.route_hint || effectiveRouteHint,
    execution_mode: internal.execution_mode || "deferred_reconcile",
    direct_exec_elapsed_ms: Date.now() - directStartedAt,
    direct_send_elapsed_ms: sendElapsedMs,
    operation_count: executablePlan.length,
  });
}

export class FeishuAgentWorkflow extends WorkflowEntrypoint<Env, FeishuNormalizedPayload> {
  async run(event: any, step: any): Promise<Record<string, JsonValue>> {
    const normalized = event.payload as FeishuNormalizedPayload;
    await step.do("emit-ingress-log", async () => {
      log("feishu.workflow.start", {
        correlation_id: normalized.correlation_id,
        lane: normalized.lane,
        route_hint: normalized.route_hint,
        task_kind: normalized.task_kind,
        workflow_instance_id: event.instanceId,
      });
      return { ok: true };
    });

    const planned =
      normalized.lane === "agent" && normalized.route_hint !== "fast_control"
        ? await step.do("modal-agent-plan", async () => invokeAgentPlan(this.env, normalized))
        : null;
    const effectiveRouteHint = trim(planned?.route_hint) || normalized.route_hint;
    if (planned) {
      log("feishu.workflow.plan.done", {
        correlation_id: normalized.correlation_id,
        route_hint: effectiveRouteHint,
        execution_mode: trim(planned.execution_mode) || "modal_heavy_exec",
        external_exec_candidate: !!planned.external_exec_candidate,
      });
    }

    const internal =
      normalized.lane === "control"
        ? await step.do("fast-control-exec", async () => {
            return {
              ...(await handleControlEvent(this.env, normalized)),
              route_hint: "fast_control",
              execution_mode: "control_complete",
            } satisfies ModalInternalResponse;
          })
        : normalized.lane === "agent" && normalized.route_hint === "fast_control"
          ? await step.do("command-control-exec", async () => {
              return {
                ...(await handleAgentCommand(this.env, normalized)),
                route_hint: "fast_control",
                execution_mode: "control_complete",
              } satisfies ModalInternalResponse;
            })
          : normalized.lane === "agent" && effectiveRouteHint === "cf_browser_first"
            ? await step.do("cf-browser-first-route", async () => {
                log("feishu.workflow.browser_route", {
                  correlation_id: normalized.correlation_id,
                  route_hint: effectiveRouteHint,
                  browser_provider: "modal_fallback_pending_cf_browser",
                });
                return invokeAgentExec(this.env, normalized);
              })
          : normalized.lane === "agent"
              ? await step.do(
                  planned?.external_exec_candidate ? "cf-ai-exec" : "modal-agent-exec",
                  { timeout: `${MODAL_AGENT_EXEC_TIMEOUT_SECONDS} seconds` },
                  async () => {
                    if (planned?.external_exec_candidate) {
                      log("feishu.workflow.exec_delegation_candidate", {
                        correlation_id: normalized.correlation_id,
                        route_hint: effectiveRouteHint,
                        execution_mode: trim(planned.execution_mode) || "deferred_reconcile",
                        provider_plan: planned.provider_plan || {},
                      });
                      try {
                        return await executeCloudflareAiExec(this.env, normalized, planned);
                      } catch (error) {
                        log("feishu.cf_ai_exec.fallback", {
                          correlation_id: normalized.correlation_id,
                          error: error instanceof Error ? error.message : String(error),
                        });
                      }
                    }
                    return invokeAgentExec(this.env, normalized);
                  },
                )
              : { status: "ignored", send_plan: [], route_hint: effectiveRouteHint };

    const executablePlan = getExecutablePlan(internal);
    await step.do("send-feishu-response", async () => {
      const sendStartedAt = Date.now();
      await executeFeishuOperations(this.env, normalized, executablePlan);
      const cfSendElapsedMs = Date.now() - sendStartedAt;
      log("feishu.workflow.send.done", {
        correlation_id: normalized.correlation_id,
        route_hint: internal.route_hint || normalized.route_hint,
        execution_mode: internal.execution_mode || "modal_heavy_exec",
        cf_send_elapsed_ms: cfSendElapsedMs,
        operation_count: executablePlan.length,
      });
      if (normalized.lane === "agent" && internal.reconcile_required !== false) {
        const pending = buildPendingReconcile(normalized, internal);
        if (pending) {
          await enqueuePendingReconcile(this.env, normalized.session_key, pending);
          log("feishu.reconcile.enqueue", {
            session_key: normalized.session_key,
            correlation_id: pending.correlation_id,
            operation_count: pending.operation_kinds.length,
          });
        }
      }
      return { delivered: true };
    });
    await step.do("emit-final-log", async () => {
      log("feishu.workflow.done", {
        correlation_id: normalized.correlation_id,
        lane: normalized.lane,
        route_hint: internal.route_hint || normalized.route_hint,
        execution_mode: internal.execution_mode || "modal_heavy_exec",
        retry_attempt_count: internal.retry_attempt_count || 1,
        retry_class: trim(internal.retry_class) || "none",
      });
      return { ok: true };
    });
    return {
      status: internal.status,
      correlation_id: normalized.correlation_id,
      route_hint: (internal.route_hint || normalized.route_hint) as JsonValue,
      execution_mode: (internal.execution_mode || "modal_heavy_exec") as JsonValue,
    };
  }
}

export class FeishuReconcileQueue extends DurableObject<Env> {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const payload = request.method === "POST" ? parseJsonSafely(await request.text()) : {};
    const stored = (await this.ctx.storage.get<PendingReconcileItem[]>("items")) ?? [];
    const items = Array.isArray(stored) ? stored : [];

    if (url.pathname === "/peek") {
      return jsonResponse({ items: items as unknown as JsonValue });
    }

    if (url.pathname === "/enqueue") {
      const candidate = payload.item;
      if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
        return jsonResponse({ ok: false, error: "invalid_item" }, { status: 400 });
      }
      const item = trimPendingReconcileItem(candidate as unknown as PendingReconcileItem);
      const correlationId = trim(item.correlation_id);
      if (!correlationId) {
        return jsonResponse({ ok: false, error: "missing_correlation_id" }, { status: 400 });
      }
      const nextItems = items.filter((entry) => trim(entry.correlation_id) !== correlationId);
      nextItems.push({
        correlation_id: correlationId,
        assistant_text: item.assistant_text,
        final_response: item.final_response,
        operation_kinds: item.operation_kinds,
        delivered_at: item.delivered_at,
      });
      const compacted = compactPendingReconcileItems(nextItems);
      await this.ctx.storage.put("items", compacted.items);
      return jsonResponse({ ok: true, count: compacted.items.length, compacted: compacted.compacted });
    }

    if (url.pathname === "/ack") {
      const rawIds = Array.isArray(payload.correlation_ids) ? payload.correlation_ids : [];
      const correlationIds = uniqueStrings(rawIds.map((value) => trim(value)));
      if (correlationIds.length === 0) {
        return jsonResponse({ ok: true, count: items.length });
      }
      const nextItems = items.filter((entry) => !correlationIds.includes(trim(entry.correlation_id)));
      await this.ctx.storage.put("items", nextItems);
      return jsonResponse({ ok: true, count: nextItems.length });
    }

    return jsonResponse({ ok: false, error: "not_found" }, { status: 404 });
  }
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method !== "POST") {
      return jsonResponse({ ok: true, service: "hermes-feishu-gateway" });
    }

    const rawBody = await request.text();
    const payload = parseJsonSafely(rawBody);
    if (payload.type === "url_verification") {
      return jsonResponse({ challenge: trim(payload.challenge) });
    }
    if (!(await isSignatureValid(request, rawBody, env)) || !isVerificationTokenValid(payload, env)) {
      return jsonResponse({ code: 401, msg: "invalid signature" }, { status: 401 });
    }

    const normalized = normalizeMessage(payload);
    log("feishu.webhook.accepted", {
      event_type: normalized.event_type,
      event_id: normalized.event_id,
      lane: normalized.lane,
      route_hint: normalized.route_hint,
      correlation_id: normalized.correlation_id,
    });

    if (shouldAckReactionInline(normalized)) {
      ctx.waitUntil(runAckReactionFastPath(env, normalized));
    }

    if (normalized.event_type === "im.message.message_read_v1") {
      log("feishu.message_read.accepted", {
        event_id: normalized.event_id,
        correlation_id: normalized.correlation_id,
        execution_path: "worker_direct",
      });
      return jsonResponse({ code: 0, msg: "accepted" });
    }

    if (shouldUseDirectWorkerFastPath(normalized)) {
      ctx.waitUntil(
        runDirectWorkerFastPath(env, normalized).catch((error) => {
          log("feishu.direct_fast.error", {
            correlation_id: normalized.correlation_id,
            lane: normalized.lane,
            route_hint: normalized.route_hint,
            error: error instanceof Error ? error.message : String(error),
          });
        }),
      );
      return jsonResponse({ code: 0, msg: "accepted" });
    }

    if (shouldUseDirectWorkerPlannedPath(normalized)) {
      ctx.waitUntil(
        runDirectWorkerPlannedPath(env, normalized).catch((error) => {
          log("feishu.direct_planned.error", {
            correlation_id: normalized.correlation_id,
            lane: normalized.lane,
            route_hint: normalized.route_hint,
            error: error instanceof Error ? error.message : String(error),
          });
        }),
      );
      return jsonResponse({ code: 0, msg: "accepted" });
    }

    if (normalized.lane === "control" || normalized.lane === "agent") {
      ctx.waitUntil(
        enqueueWorkflowInstance(env, normalized).catch((error) => {
          log("feishu.workflow.error", {
            correlation_id: normalized.correlation_id,
            error: error instanceof Error ? error.message : String(error),
          });
        }),
      );
      return jsonResponse({ code: 0, msg: "accepted" });
    }

    return jsonResponse({ code: 0, msg: "accepted" });
  },
};
