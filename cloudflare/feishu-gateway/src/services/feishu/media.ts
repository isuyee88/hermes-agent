import type { Env, JsonValue } from "../../runtime";
import { feishuApi } from "./client";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

export async function uploadImageFromBytes(env: Env, bytes: ArrayBuffer, fileName: string): Promise<string> {
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

export async function uploadFileFromBytes(env: Env, bytes: ArrayBuffer, fileName: string): Promise<string> {
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
