import { executeFeishuOperations as executeFeishuSendPlan } from "../services/feishu/send-plan";
import type {
  Env,
  FeishuNormalizedPayload,
  JsonValue,
  ModalInternalResponse,
  SessionHistoryEntry,
  SessionProfileState,
} from "../runtime";

type SessionCacheDeps = {
  trim: (value: unknown) => string;
  log: (event: string, extra: Record<string, unknown>) => void;
  resolveDefaultSendTarget: (payload: FeishuNormalizedPayload) => { receiveId: string; receiveIdType: string };
  deleteFeishuMessage: typeof import("../services/feishu/messages").deleteFeishuMessage;
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
  uploadImageFromBytes: typeof import("../services/feishu/media").uploadImageFromBytes;
  uploadFileFromBytes: typeof import("../services/feishu/media").uploadFileFromBytes;
  fetchModalResultFile: typeof import("../services/modal/client").fetchModalResultFile;
  updateSessionStateCache: (
    env: Env,
    sessionKey: string,
    patch: { profile?: SessionProfileState | null; historyEntries?: SessionHistoryEntry[] },
  ) => Promise<void>;
  summarizeOperationsForHermes: (sendPlan: Array<Record<string, JsonValue>>) => {
    assistantText: string;
    operationKinds: string[];
  };
};

async function executeFeishuOperations(
  env: Env,
  normalized: FeishuNormalizedPayload,
  operations: Array<Record<string, JsonValue>>,
  deps: SessionCacheDeps,
): Promise<void> {
  await executeFeishuSendPlan(env, normalized, operations, {
    resolveDefaultSendTarget: deps.resolveDefaultSendTarget,
    deleteFeishuMessage: deps.deleteFeishuMessage,
    sendLoggedFeishuOperation: (workerEnv, payload, kind, options) =>
      deps.sendLoggedFeishuOperation(workerEnv, payload, kind, options),
    uploadImageFromBytes: deps.uploadImageFromBytes,
    uploadFileFromBytes: deps.uploadFileFromBytes,
    fetchModalResultFile: deps.fetchModalResultFile,
    log: deps.log,
  });
}

function summarizeAssistantTextForHistory(
  internal: ModalInternalResponse,
  deps: Pick<SessionCacheDeps, "summarizeOperationsForHermes" | "trim">,
): string {
  const summary = deps.summarizeOperationsForHermes(
    Array.isArray(internal.action_plan)
      ? internal.action_plan
      : Array.isArray(internal.send_plan)
        ? internal.send_plan
        : [],
  );
  return deps.trim(summary.assistantText || internal.final_response);
}

function buildSessionProfilePatch(
  internal: ModalInternalResponse,
  deps: Pick<SessionCacheDeps, "trim">,
): SessionProfileState | null {
  const state = internal.session_state_after;
  if (!state) {
    return null;
  }
  const routeStatusLines = Array.isArray(state.route_status_lines)
    ? state.route_status_lines.map((line) => deps.trim(line)).filter(Boolean)
    : [];
  const currentModel = deps.trim(state.current_model);
  const currentProvider = deps.trim(state.current_provider);
  const currentPersonality = deps.trim(state.current_personality);
  if (!currentModel && !currentProvider && !currentPersonality && routeStatusLines.length === 0) {
    return null;
  }
  return {
    current_model: currentModel,
    current_provider: currentProvider,
    current_personality: currentPersonality,
    route_status_lines: routeStatusLines,
    updated_at: new Date().toISOString(),
  };
}

function buildConversationHistoryPatch(
  normalized: FeishuNormalizedPayload,
  internal: ModalInternalResponse,
  deps: Pick<SessionCacheDeps, "trim" | "summarizeOperationsForHermes">,
): SessionHistoryEntry[] {
  if (normalized.lane !== "agent") {
    return [];
  }
  const text = deps.trim(normalized.text);
  const entries: SessionHistoryEntry[] = [];
  if (text && !text.startsWith("/")) {
    entries.push({
      id: `user:${normalized.correlation_id}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    });
  }
  const assistantText = summarizeAssistantTextForHistory(internal, deps);
  if (assistantText) {
    entries.push({
      id: `assistant:${normalized.correlation_id}`,
      role: "assistant",
      content: assistantText,
      created_at: new Date().toISOString(),
    });
  }
  return entries;
}

export async function persistSessionCacheAfterResponse(
  env: Env,
  normalized: FeishuNormalizedPayload,
  internal: ModalInternalResponse,
  deps: SessionCacheDeps,
): Promise<void> {
  const profile = buildSessionProfilePatch(internal, deps);
  const historyEntries = buildConversationHistoryPatch(normalized, internal, deps);
  if (!profile && historyEntries.length === 0) {
    return;
  }
  try {
    await deps.updateSessionStateCache(env, normalized.session_key, { profile, historyEntries });
  } catch (error) {
    deps.log("feishu.session_cache.update_failed", {
      correlation_id: normalized.correlation_id,
      session_key: normalized.session_key,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

export function createResponseHandlers(deps: SessionCacheDeps): {
  executeFeishuOperations: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    operations: Array<Record<string, JsonValue>>,
  ) => Promise<void>;
  persistSessionCacheAfterResponse: (
    env: Env,
    normalized: FeishuNormalizedPayload,
    internal: ModalInternalResponse,
  ) => Promise<void>;
} {
  return {
    executeFeishuOperations: (env, normalized, operations) => executeFeishuOperations(env, normalized, operations, deps),
    persistSessionCacheAfterResponse: (env, normalized, internal) =>
      persistSessionCacheAfterResponse(env, normalized, internal, deps),
  };
}
