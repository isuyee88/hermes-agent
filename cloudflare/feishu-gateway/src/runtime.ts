import { WorkflowEntrypoint } from "cloudflare:workers";
import { maybeBuildSitePrefetch as maybeBuildSitePrefetchOrchestrator } from "./browser/prefetch/orchestrator";
import { buildSitePrefetchErrorManifest } from "./browser/prefetch/shared";
import {
  extractMessageReadInfo,
  isVerificationTokenValid,
  sha256Hex,
} from "./core/feishu-webhook";
import {
  getCloudflareAiGatewayBase,
  getCloudflareAiGatewayRoot,
  isFreeishModelName,
  parseBoolean,
  parsePositiveInt,
  toWorkflowInstanceId,
} from "./core/runtime-utils";
import { SITE_PREFETCH_ERROR_TTL_MS, jsonResponse, log, parseJsonSafely, trim } from "./core/value";
import {
  buildPendingReconcile,
  summarizeOperationsForHermes,
} from "./durable/pending-reconcile";
import {
  enqueuePendingReconcile,
  fetchModalInternalWithPendingReconciles,
  peekSessionStateCache,
  resolveOutboundMessageCorrelationForRead,
  upsertOutboundMessageCorrelationRef,
  updateSessionStateCache,
} from "./durable/reconcile-client";
import {
  compactSessionHistory,
  normalizeSessionHistoryEntries,
  normalizeSessionProfileState,
} from "./durable/session-state";
import {
  buildEdgeDirectPlan as buildEdgeDirectPlanGateway,
  invokeAgentExec as invokeAgentExecGateway,
  invokeAgentPlan as invokeAgentPlanGateway,
} from "./gateway/agent-runtime";
import {
  handleAgentCommand as handleAgentCommandGateway,
  handleControlEvent as handleControlEventGateway,
  resolveDefaultSendTarget,
} from "./gateway/control-handlers";
import { executeCloudflareAiExec as executeCloudflareAiExecGateway } from "./gateway/cf-ai-exec";
import {
  normalizeMessage as normalizeMessageGateway,
  readEvent as readEventGateway,
  readRecord as readRecordGateway,
  readString as readStringGateway,
} from "./gateway/message-normalize";
import {
  isTextGatewayRequestClass as isTextGatewayRequestClassGateway,
  messageExplicitlyRequestsBrowserTools as messageExplicitlyRequestsBrowserToolsGateway,
} from "./gateway/routing";
import { buildDirectConversationMessages as buildDirectConversationMessagesGateway } from "./gateway/conversation-messages";
import { createResponseHandlers } from "./gateway/session-cache";
import {
  buildAiGatewayMetadata as buildAiGatewayMetadataGateway,
  buildProviderPlanFromSessionProfile as buildProviderPlanFromSessionProfileGateway,
  buildRoutingLogFields as buildRoutingLogFieldsGateway,
  classifyCfAiExecFallbackReason as classifyCfAiExecFallbackReasonGateway,
  inferRouteDecisionReason as inferRouteDecisionReasonGateway,
  inferSiteExecutionStage as inferSiteExecutionStageGateway,
  isExternalExecCandidate as isExternalExecCandidateGateway,
} from "./gateway/route-policy";
import {
  enqueueWorkflowInstance as enqueueWorkflowRunner,
  runAckReactionFastPath as runAckReactionFastPathRunner,
  runDirectWorkerFastPath as runDirectWorkerFastPathRunner,
  runDirectWorkerPlannedPath as runDirectWorkerPlannedPathRunner,
} from "./gateway/direct-runners";
import type {
  DfmeaControlPhase,
  DfmeaFailureMode,
  ExecutionMode,
  FeishuAttachmentRef,
  FeishuNormalizedPayload,
  JsonValue,
  ModalInternalResponse,
  OutboundMessageCorrelationRef,
  RequestClass,
  RouteFamily,
  RouteHint,
  SessionHistoryEntry,
  SessionProfileState,
  SessionStateCache,
  SitePrefetchManifest,
  WorkflowBinding,
} from "./contracts/gateway";
import { addAckReaction, deleteFeishuMessage, sendLoggedFeishuOperation } from "./services/feishu/messages";
import { uploadFileFromBytes, uploadImageFromBytes } from "./services/feishu/media";
import { classifyModalInternalError, fetchModalInternal, fetchModalResultFile } from "./services/modal/client";
import { runFeishuWorkflow } from "./workflow/runner";
export type {
  DfmeaControlPhase,
  DfmeaFailureMode,
  ExecutionMode,
  FeishuAttachmentRef,
  FeishuNormalizedPayload,
  JsonValue,
  ModalInternalResponse,
  OutboundMessageCorrelationRef,
  PendingReconcileItem,
  RequestClass,
  RouteFamily,
  RouteHint,
  SessionHistoryEntry,
  SessionProfileState,
  SessionStateCache,
  SitePrefetchManifest,
  WorkflowBinding,
} from "./contracts/gateway";

const MAX_PENDING_RECONCILE_ITEMS = 50;
const MAX_PENDING_RECONCILE_BYTES = 64 * 1024;
const MAX_PENDING_RECONCILE_TEXT_LENGTH = 4000;
const MAX_SESSION_HISTORY_ITEMS = 16;
const MAX_SESSION_HISTORY_CHARS = 12000;
const MODAL_AGENT_EXEC_TIMEOUT_SECONDS = 150;
const MODAL_AGENT_EXEC_RETRY_LIMIT = 1;
export { SITE_PREFETCH_ERROR_TTL_MS, jsonResponse, log, parseJsonSafely, trim } from "./core/value";
export {
  extractMessageReadInfo,
  isVerificationTokenValid,
} from "./core/feishu-webhook";

function readEvent(payload: Record<string, JsonValue>): Record<string, JsonValue> {
  return readEventGateway(payload);
}

function readRecord(parent: Record<string, JsonValue>, key: string): Record<string, JsonValue> {
  return readRecordGateway(parent, key);
}

function readString(parent: Record<string, JsonValue>, key: string): string {
  return readStringGateway(parent, key);
}

function messageExplicitlyRequestsBrowserTools(message: string): boolean {
  return messageExplicitlyRequestsBrowserToolsGateway(message);
}

function isTextGatewayRequestClass(value: string): boolean {
  return isTextGatewayRequestClassGateway(value);
}
export function buildRoutingLogFields(
  normalized: FeishuNormalizedPayload,
  overrides: Partial<ModalInternalResponse> = {},
): Record<string, unknown> {
  return buildRoutingLogFieldsGateway(normalized, overrides);
}

function buildAiGatewayMetadata(
  normalized: FeishuNormalizedPayload,
  planned: Partial<ModalInternalResponse>,
): Record<string, string | number | boolean> {
  return buildAiGatewayMetadataGateway(normalized, planned);
}

function isExternalExecCandidate(normalized: FeishuNormalizedPayload, routeHint: RouteHint): boolean {
  return isExternalExecCandidateGateway(
    normalized,
    routeHint,
    isTextGatewayRequestClass,
    messageExplicitlyRequestsBrowserTools,
  );
}

function inferRouteDecisionReason(normalized: FeishuNormalizedPayload, routeHint: RouteHint): string {
  return inferRouteDecisionReasonGateway(normalized, routeHint, messageExplicitlyRequestsBrowserTools);
}

function classifyCfAiExecFallbackReason(error: unknown, planned?: Partial<ModalInternalResponse>): string {
  return classifyCfAiExecFallbackReasonGateway(error, planned, isTextGatewayRequestClass);
}

function buildProviderPlanFromSessionProfile(
  env: Env,
  profile?: SessionProfileState | null,
  normalized?: Pick<FeishuNormalizedPayload, "request_class" | "gateway_route_name" | "route_family" | "modality_profile">,
): Record<string, JsonValue> {
  return buildProviderPlanFromSessionProfileGateway(env, profile, normalized, getCloudflareAiGatewayBase);
}

export function normalizeMessage(env: Env, payload: Record<string, JsonValue>): FeishuNormalizedPayload {
  return normalizeMessageGateway(env, payload);
}

export async function resolveMessageReadCorrelation(
  env: Env,
  sessionKey: string,
  messageIds: string[],
): Promise<{
  refs: OutboundMessageCorrelationRef[];
  unmatched_message_ids: string[];
}> {
  return resolveOutboundMessageCorrelationForRead(env, sessionKey, messageIds);
}

async function executeCloudflareAiExec(
  env: Env,
  normalized: FeishuNormalizedPayload,
  planned: ModalInternalResponse,
): Promise<ModalInternalResponse> {
  return executeCloudflareAiExecGateway(env, normalized, planned, {
    trim,
    log,
    parseBoolean,
    parsePositiveInt,
    isFreeishModelName,
    isTextGatewayRequestClass,
    getCloudflareAiGatewayBase,
    getCloudflareAiGatewayRoot,
    sha256Hex,
    buildAiGatewayMetadata,
    buildRoutingLogFields,
    buildTextSendPlan,
    inferRouteDecisionReason,
    classifyCfAiExecFallbackReason,
  });
}

const responseHandlers = createResponseHandlers({
  trim,
  log,
  resolveDefaultSendTarget: (payload) => resolveDefaultSendTarget(payload, { readEvent, readRecord, readString, trim }),
  deleteFeishuMessage,
  sendLoggedFeishuOperation: (workerEnv, payload, kind, options) =>
    sendLoggedFeishuOperation(workerEnv, payload, kind, options, log, upsertOutboundMessageCorrelationRef),
  uploadImageFromBytes,
  uploadFileFromBytes,
  fetchModalResultFile,
  updateSessionStateCache,
  summarizeOperationsForHermes,
});

const { executeFeishuOperations, persistSessionCacheAfterResponse } = responseHandlers;

async function handleControlEvent(env: Env, normalized: FeishuNormalizedPayload): Promise<ModalInternalResponse> {
  return handleControlEventGateway(env, normalized, {
    trim,
    readEvent,
    readRecord,
    readString,
    buildTextSendPlan,
    buildInteractiveCardSendPlan,
    buildDeleteMessageSendPlan,
    callModalWithReconciles,
  });
}

async function invokeAgentExec(
  env: Env,
  normalized: FeishuNormalizedPayload,
  metadata: Partial<ModalInternalResponse> = {},
  sitePrefetch?: SitePrefetchManifest | null,
): Promise<ModalInternalResponse> {
  return invokeAgentExecGateway(env, normalized, metadata, sitePrefetch, {
    trim,
    log,
    parseBoolean,
    callModalWithReconciles,
    classifyModalInternalError,
    buildRoutingLogFields,
    peekSessionStateCache,
    inferRouteDecisionReason,
    isExternalExecCandidate,
    buildProviderPlanFromSessionProfile,
    buildConversationMessages: (history, latestUserText, sitePrefetch) =>
      buildDirectConversationMessagesGateway(history, latestUserText, sitePrefetch, {
        trim,
        compactSessionHistory,
      }),
    modalAgentExecRetryLimit: MODAL_AGENT_EXEC_RETRY_LIMIT,
  });
}

async function invokeAgentPlan(
  env: Env,
  normalized: FeishuNormalizedPayload,
  sitePrefetch?: SitePrefetchManifest | null,
): Promise<ModalInternalResponse> {
  return invokeAgentPlanGateway(env, normalized, sitePrefetch, {
    callModalWithReconciles,
  });
}

async function buildEdgeDirectPlan(
  env: Env,
  normalized: FeishuNormalizedPayload,
  sitePrefetch?: SitePrefetchManifest | null,
): Promise<ModalInternalResponse> {
  return buildEdgeDirectPlanGateway(env, normalized, sitePrefetch, {
    trim,
    log,
    parseBoolean,
    peekSessionStateCache,
    inferRouteDecisionReason,
    isExternalExecCandidate,
    buildProviderPlanFromSessionProfile,
    buildConversationMessages: (history, latestUserText, sitePrefetch) =>
      buildDirectConversationMessagesGateway(history, latestUserText, sitePrefetch, {
        trim,
        compactSessionHistory,
      }),
    buildRoutingLogFields,
  });
}

async function handleAgentCommand(env: Env, normalized: FeishuNormalizedPayload): Promise<ModalInternalResponse> {
  return handleAgentCommandGateway(env, normalized, {
    callModalWithReconciles: (workerEnv, path, payload, body) =>
      fetchModalInternalWithPendingReconciles<ModalInternalResponse>(
        workerEnv,
        path,
        payload,
        body,
        fetchModalInternal,
        log,
      ),
  });
}

function getExecutablePlan(internal: ModalInternalResponse): Array<Record<string, JsonValue>> {
  if (Array.isArray(internal.action_plan) && internal.action_plan.length > 0) {
    return internal.action_plan;
  }
  if (Array.isArray(internal.send_plan) && internal.send_plan.length > 0) {
    return internal.send_plan;
  }
  // Fallback: if send_plan/action_plan are empty but final_response exists, build text send plan
  const finalResponse = trim(internal.final_response);
  if (finalResponse) {
    return [{ kind: "text", content: finalResponse }];
  }
  return [];
}

function buildTextSendPlan(content: string): Array<Record<string, JsonValue>> {
  const normalizedContent = trim(content);
  if (!normalizedContent) {
    return [];
  }
  return [{ kind: "text", content: normalizedContent }];
}

function buildInteractiveCardSendPlan(
  card: Record<string, JsonValue>,
  receiveId: string,
  receiveIdType: string,
): Array<Record<string, JsonValue>> {
  if (!card || typeof card !== "object" || Array.isArray(card)) {
    return [];
  }
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
  if (!normalizedMessageId) {
    return [];
  }
  return [{ kind: "delete_message", message_id: normalizedMessageId }];
}

async function callModalWithReconciles<T>(
  env: Env,
  path: string,
  normalized: FeishuNormalizedPayload,
  body: Record<string, JsonValue>,
): Promise<T> {
  return fetchModalInternalWithPendingReconciles(env, path, normalized, body, fetchModalInternal, log);
}

export function shouldAckReactionInline(normalized: FeishuNormalizedPayload): boolean {
  return normalized.event_type === "im.message.receive_v1" && !!trim(normalized.message_id);
}

export function shouldUseDirectWorkerFastPath(normalized: FeishuNormalizedPayload): boolean {
  return normalized.lane === "control" || (normalized.lane === "agent" && normalized.route_hint === "fast_control");
}

export function shouldUseDirectWorkerPlannedPath(normalized: FeishuNormalizedPayload): boolean {
  return (
    normalized.lane === "agent" &&
    normalized.route_hint === "modal_heavy_exec" &&
    isTextGatewayRequestClass(normalized.request_class) &&
    normalized.gateway_eligible &&
    !trim(normalized.text).startsWith("/")
  );
}

function shouldEnqueueWorkflowForPlannedPath(planned: ModalInternalResponse): boolean {
  return !planned.external_exec_candidate;
}

export type PrefetchBudgetState = {
  day: string;
  browser_ms_used: number;
  crawl_jobs: number;
  last_quick_action_at: number;
};

function inferSiteExecutionStage(normalized: FeishuNormalizedPayload): string {
  return inferSiteExecutionStageGateway(normalized);
}

async function maybeBuildSitePrefetch(
  env: Env,
  normalized: FeishuNormalizedPayload,
  options: { allowCrawl: boolean } = { allowCrawl: false },
): Promise<SitePrefetchManifest | null> {
  return maybeBuildSitePrefetchOrchestrator(
    env,
    normalized,
    {
      trim,
      log,
      buildRoutingLogFields,
      inferSiteExecutionStage,
      buildSitePrefetchErrorManifest,
      getCloudflareAiGatewayBase,
    },
    options,
  );
}

export async function enqueueWorkflowInstance(env: Env, normalized: FeishuNormalizedPayload): Promise<void> {
  return enqueueWorkflowRunner(env, normalized, {
    log,
    toWorkflowInstanceId,
    trim,
  });
}

export async function runAckReactionFastPath(env: Env, normalized: FeishuNormalizedPayload): Promise<void> {
  return runAckReactionFastPathRunner(env, normalized, {
    addAckReaction,
    log,
    shouldAckReactionInline,
  });
}

export async function runDirectWorkerFastPath(env: Env, normalized: FeishuNormalizedPayload): Promise<void> {
  return runDirectWorkerFastPathRunner(env, normalized, {
    log,
    handleControlEvent,
    handleAgentCommand,
    getExecutablePlan,
    executeFeishuOperations,
    persistSessionCacheAfterResponse,
    buildPendingReconcile,
    enqueuePendingReconcile,
  });
}

export async function runDirectWorkerPlannedPath(env: Env, normalized: FeishuNormalizedPayload): Promise<void> {
  return runDirectWorkerPlannedPathRunner(env, normalized, {
    trim,
    log,
    toWorkflowInstanceId,
    shouldAckReactionInline,
    addAckReaction,
    handleControlEvent,
    handleAgentCommand,
    getExecutablePlan,
    executeFeishuOperations,
    persistSessionCacheAfterResponse,
    buildPendingReconcile,
    enqueuePendingReconcile,
    buildRoutingLogFields,
    maybeBuildSitePrefetch,
    buildEdgeDirectPlan,
    shouldEnqueueWorkflowForPlannedPath,
    executeCloudflareAiExec,
    classifyCfAiExecFallbackReason,
    invokeAgentExec,
  });
}

export class FeishuAgentWorkflow extends WorkflowEntrypoint<Env, FeishuNormalizedPayload> {
  async run(event: any, step: any): Promise<Record<string, JsonValue>> {
    return runFeishuWorkflow({
      env: this.env,
      event,
      step,
      deps: {
        log,
        buildRoutingLogFields,
        inferSiteExecutionStage,
        maybeBuildSitePrefetch,
        buildEdgeDirectPlan,
        trim,
        handleControlEvent,
        handleAgentCommand,
        invokeAgentExec,
        executeCloudflareAiExec,
        classifyCfAiExecFallbackReason,
        getExecutablePlan,
        executeFeishuOperations,
        persistSessionCacheAfterResponse,
        buildPendingReconcile,
        enqueuePendingReconcile,
        modalAgentExecTimeoutSeconds: MODAL_AGENT_EXEC_TIMEOUT_SECONDS,
      },
    });
  }
}
