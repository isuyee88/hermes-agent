import { classifyNormalizedRequest, extractTargetUrlAndDomain } from "./routing";
import type { Env, FeishuNormalizedPayload, JsonValue } from "../runtime";

type AttachmentRef = FeishuNormalizedPayload["attachment_refs"][number];

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function parseJsonSafely(raw: string): Record<string, JsonValue> {
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? (parsed as Record<string, JsonValue>) : {};
  } catch {
    return {};
  }
}

export function readEvent(payload: Record<string, JsonValue>): Record<string, JsonValue> {
  const event = payload.event;
  return event && typeof event === "object" ? (event as Record<string, JsonValue>) : {};
}

export function readRecord(parent: Record<string, JsonValue>, key: string): Record<string, JsonValue> {
  const value = parent[key];
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, JsonValue>)
    : {};
}

export function readString(parent: Record<string, JsonValue>, key: string): string {
  return trim(parent[key]);
}

export function extractEventType(payload: Record<string, JsonValue>): string {
  const header = readRecord(payload, "header");
  return readString(header, "event_type") || readString(payload, "event_type");
}

export function extractEventId(payload: Record<string, JsonValue>): string {
  const header = readRecord(payload, "header");
  return readString(header, "event_id") || readString(payload, "event_id");
}

export function resolveControlLane(eventType: string): boolean {
  const normalized = eventType.toLowerCase();
  return normalized === "application.bot.menu_v6" || normalized === "card.action.trigger";
}

function parseFeishuMessageContent(rawContent: string): Record<string, JsonValue> {
  return rawContent ? parseJsonSafely(rawContent) : {};
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

function buildSessionKey(platform: string, chatType: string, chatId: string, userId: string): string {
  const normalizedPlatform = trim(platform) || "feishu";
  const normalizedChatType = trim(chatType).toLowerCase() || "dm";
  const normalizedChatId = trim(chatId);
  const normalizedUserId = trim(userId);
  if (normalizedChatType === "dm") {
    return normalizedChatId ? `agent:main:${normalizedPlatform}:dm:${normalizedChatId}` : `agent:main:${normalizedPlatform}:dm`;
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

export function normalizeMessage(env: Env, payload: Record<string, JsonValue>): FeishuNormalizedPayload {
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
  const parsedContent = parseFeishuMessageContent(readString(message, "content"));
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
  const chatType = (readString(chat, "chat_type") || readString(context, "chat_type")).toLowerCase() === "p2p" ? "dm" : "group";
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
  const attachmentRefs: AttachmentRef[] = [];
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

  const { targetUrl, targetDomain } = extractTargetUrlAndDomain(text);
  const lane = resolveControlLane(eventType) ? "control" : eventType === "im.message.receive_v1" ? "agent" : "ignore";
  const classified = classifyNormalizedRequest(env, {
    lane,
    task_kind: messageType,
    message_type: messageType,
    text,
    attachment_refs: attachmentRefs,
    target_url: targetUrl,
    target_domain: targetDomain,
  });

  return {
    correlation_id: `feishu:${chatId}:${eventId || messageId || crypto.randomUUID()}`,
    session_key: buildSessionKey("feishu", chatType, chatId, userId),
    lane,
    route_hint: classified.route_hint,
    site_category: classified.site_category,
    site_intent: classified.site_intent,
    target_url: targetUrl,
    target_domain: targetDomain,
    task_kind: messageType,
    request_class: classified.request_class,
    content_modalities: classified.content_modalities,
    route_family: classified.route_family,
    gateway_route_name: classified.gateway_route_name,
    gateway_eligible: classified.gateway_eligible,
    requires_tools: classified.requires_tools,
    requires_browser: classified.requires_browser,
    requires_media_hydration: classified.requires_media_hydration,
    requires_modal_runtime: classified.requires_modal_runtime,
    modality_profile: classified.modality_profile,
    toolset: classified.toolset,
    reason_code: classified.reason_code,
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
