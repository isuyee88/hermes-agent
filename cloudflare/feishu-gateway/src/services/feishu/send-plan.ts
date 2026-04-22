import type { Env, FeishuNormalizedPayload, JsonValue } from "../../runtime";

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

export async function executeFeishuOperations(
  env: Env,
  normalized: FeishuNormalizedPayload,
  operations: Array<Record<string, JsonValue>>,
  deps: {
    resolveDefaultSendTarget: (
      normalized: FeishuNormalizedPayload,
    ) => { receiveId: string; receiveIdType: "chat_id" | "open_id" | "user_id" | "union_id" };
    deleteFeishuMessage: (env: Env, messageId: string) => Promise<void>;
    sendLoggedFeishuOperation: (
      env: Env,
      normalized: FeishuNormalizedPayload,
      kind: string,
      options: {
        receiveId: string;
        receiveIdType?: string;
        msgType: "text" | "interactive" | "image" | "file" | "audio" | "media" | "post";
        content: Record<string, JsonValue> | string;
      },
    ) => Promise<Record<string, JsonValue>>;
    uploadImageFromBytes: (env: Env, bytes: ArrayBuffer, fileName: string) => Promise<string>;
    uploadFileFromBytes: (env: Env, bytes: ArrayBuffer, fileName: string) => Promise<string>;
    fetchModalResultFile: (env: Env, token: string) => Promise<{ bytes: ArrayBuffer; fileName: string }>;
    log?: (event: string, extra: Record<string, unknown>) => void;
  },
): Promise<void> {
  const defaultTarget = deps.resolveDefaultSendTarget(normalized);
  const startedAt = Date.now();
  
  if (deps.log && operations.length > 0) {
    deps.log("feishu.operations.start", {
      correlation_id: normalized.correlation_id,
      event_id: normalized.event_id,
      operation_count: operations.length,
      operation_kinds: operations.map(op => trim(op.kind)).filter(Boolean),
      receive_id: defaultTarget.receiveId,
      receive_id_type: defaultTarget.receiveIdType,
    });
  }
  
  for (let i = 0; i < operations.length; i++) {
    const operation = operations[i];
    const kind = trim(operation.kind);
    if (!kind) {
      if (deps.log) {
        deps.log("feishu.operations.skip_no_kind", {
          correlation_id: normalized.correlation_id,
          operation_index: i,
          operation_keys: Object.keys(operation),
        });
      }
      continue;
    }
    try {
      if (kind === "delete_message") {
        await deps.deleteFeishuMessage(env, trim(operation.message_id));
        if (deps.log) {
          deps.log("feishu.operations.done.delete", {
            correlation_id: normalized.correlation_id,
            operation_index: i,
            kind,
            message_id: trim(operation.message_id),
          });
        }
        continue;
      }
      if (kind === "interactive_card") {
        const card = operation.card;
        if (!card || typeof card !== "object" || Array.isArray(card)) {
          if (deps.log) {
            deps.log("feishu.operations.skip.invalid_card", {
              correlation_id: normalized.correlation_id,
              operation_index: i,
              kind,
            });
          }
          continue;
        }
        await deps.sendLoggedFeishuOperation(env, normalized, kind, {
          receiveId: trim(operation.receive_id) || defaultTarget.receiveId,
          receiveIdType: trim(operation.receive_id_type) || defaultTarget.receiveIdType,
          msgType: "interactive",
          content: card as Record<string, JsonValue>,
        });
        continue;
      }
      if (kind === "text" || kind === "edit") {
        const content = trim(operation.content);
        if (!content) {
          if (deps.log) {
            deps.log("feishu.operations.skip.empty_content", {
              correlation_id: normalized.correlation_id,
              operation_index: i,
              kind,
              content_keys: Object.keys(operation),
            });
          }
          continue;
        }
        await deps.sendLoggedFeishuOperation(env, normalized, kind, {
          receiveId: defaultTarget.receiveId,
          receiveIdType: defaultTarget.receiveIdType,
          msgType: "text",
          content: { text: content },
        });
        continue;
      }
      if (kind === "image_url") {
        const imageUrl = trim(operation.image_url);
        if (!imageUrl) {
          if (deps.log) {
            deps.log("feishu.operations.skip.empty_image_url", {
              correlation_id: normalized.correlation_id,
              operation_index: i,
              kind,
            });
          }
          continue;
        }
        const response = await fetch(imageUrl);
        if (!response.ok) {
          throw new Error(`image_fetch_failed:${response.status}:${imageUrl}`);
        }
        const imageKey = await deps.uploadImageFromBytes(env, await response.arrayBuffer(), "image.bin");
        await deps.sendLoggedFeishuOperation(env, normalized, kind, {
          receiveId: defaultTarget.receiveId,
          receiveIdType: defaultTarget.receiveIdType,
          msgType: "image",
          content: { image_key: imageKey },
        });
        continue;
      }
      const file = operation.file;
      if (!file || typeof file !== "object") {
        if (deps.log) {
          deps.log("feishu.operations.skip.no_file", {
            correlation_id: normalized.correlation_id,
            operation_index: i,
            kind,
          });
        }
        continue;
      }
      const token = trim((file as Record<string, JsonValue>).token);
      if (!token) {
        if (deps.log) {
          deps.log("feishu.operations.skip.no_file_token", {
            correlation_id: normalized.correlation_id,
            operation_index: i,
            kind,
          });
        }
        continue;
      }
      const modalFile = await deps.fetchModalResultFile(env, token);
      if (kind === "image_file") {
        const imageKey = await deps.uploadImageFromBytes(env, modalFile.bytes, modalFile.fileName);
        await deps.sendLoggedFeishuOperation(env, normalized, kind, {
          receiveId: defaultTarget.receiveId,
          receiveIdType: defaultTarget.receiveIdType,
          msgType: "image",
          content: { image_key: imageKey },
        });
        continue;
      }
      const fileKey = await deps.uploadFileFromBytes(env, modalFile.bytes, modalFile.fileName);
      const msgType = kind === "audio_file" ? "audio" : kind === "video_file" ? "media" : "file";
      await deps.sendLoggedFeishuOperation(env, normalized, kind, {
        receiveId: defaultTarget.receiveId,
        receiveIdType: defaultTarget.receiveIdType,
        msgType,
        content: { file_key: fileKey },
      });
    } catch (error) {
      if (deps.log) {
        deps.log("feishu.operations.error", {
          correlation_id: normalized.correlation_id,
          operation_index: i,
          kind,
          error: error instanceof Error ? error.message : String(error),
        });
      }
      throw error;
    }
  }
  
  if (deps.log) {
    deps.log("feishu.operations.done", {
      correlation_id: normalized.correlation_id,
      operation_count: operations.length,
      elapsed_ms: Date.now() - startedAt,
    });
  }
}
