import type { JsonValue } from "../contracts/gateway";
import { readEvent, readRecord, readString } from "../gateway/message-normalize";
import { trim } from "./value";

function normalizeEpochMsString(value: unknown): string {
  const raw = trim(value);
  if (!raw) {
    return "";
  }
  if (/^\d{10}$/.test(raw)) {
    return String(Number(raw) * 1000);
  }
  if (/^\d{13}$/.test(raw)) {
    return raw;
  }
  return "";
}

export function extractMessageReadInfo(payload: Record<string, JsonValue>): {
  readerOpenId: string;
  readerUserId: string;
  readerUnionId: string;
  tenantKey: string;
  readTime: string;
  messageIdList: string[];
} {
  const event = readEvent(payload);
  const user = readRecord(event, "user");
  const userId = readRecord(event, "user_id");
  const header = readRecord(payload, "header");
  const rawMessageIdList = event.message_id_list;
  const messageIdList = Array.isArray(rawMessageIdList)
    ? Array.from(
        new Set(
          rawMessageIdList
            .map((item) => trim(item))
            .filter(Boolean),
        ),
      )
    : [];

  return {
    readerOpenId: readString(userId, "open_id") || readString(user, "open_id"),
    readerUserId: readString(userId, "user_id") || readString(user, "user_id"),
    readerUnionId: readString(userId, "union_id") || readString(user, "union_id"),
    tenantKey: readString(event, "tenant_key"),
    readTime:
      normalizeEpochMsString(event.read_time) ||
      normalizeEpochMsString(event.readTime) ||
      normalizeEpochMsString(event.event_time) ||
      normalizeEpochMsString(event.eventTime) ||
      normalizeEpochMsString(header.create_time) ||
      normalizeEpochMsString(payload.event_time) ||
      normalizeEpochMsString(payload.eventTime) ||
      normalizeEpochMsString(payload.create_time),
    messageIdList,
  };
}

export async function sha256Hex(input: string): Promise<string> {
  const data = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", data);
  const bytes = new Uint8Array(digest);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function isVerificationTokenValid(
  payload: Record<string, JsonValue>,
  env: { FEISHU_VERIFICATION_TOKEN?: string },
): boolean {
  const expected = trim(env.FEISHU_VERIFICATION_TOKEN);
  if (!expected) {
    return true;
  }
  const header = readRecord(payload, "header");
  const token = readString(header, "token") || readString(payload, "token");
  return !token || token === expected;
}
