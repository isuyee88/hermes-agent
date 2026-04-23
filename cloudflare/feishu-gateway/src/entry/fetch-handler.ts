import type { Env, JsonValue } from "../runtime";
import {
  bootstrapModelCatalogHeartbeat,
  buildModelCatalogQueueBootstrapResponse,
  handleModelCatalogQueueMessage,
  isInternalModelCatalogRequest,
  isInternalModelCatalogRequestAuthorized,
} from "../model-catalog/queue-heartbeat";
import { runModelCatalogScheduledTick } from "../model-catalog/scheduled";
import {
  KPI_ROLLUP_JOB_NAME,
  bootstrapKpiRollupHeartbeat,
  buildKpiRollupQueueBootstrapResponse,
  handleKpiRollupQueueMessage,
  isInternalKpiRollupRequest,
} from "../observability/kpi-rollup-queue";
import {
  buildRoutingLogFields,
  enqueueWorkflowInstance,
  extractMessageReadInfo,
  isVerificationTokenValid,
  jsonResponse,
  log,
  normalizeMessage,
  parseJsonSafely,
  resolveMessageReadCorrelation,
  runAckReactionFastPath,
  runDirectWorkerFastPath,
  shouldAckReactionInline,
  shouldUseDirectWorkerFastPath,
  shouldUseDirectWorkerPlannedPath,
  trim,
} from "../runtime";
import { writeFeishuAnalyticsEvent } from "../observability/analytics-engine";
import { runFeishuKpiRollupScheduledTick } from "../observability/kpi-rollups";
import { createHoneycombLogger } from "../services/honeycomb/logger";

const FEISHU_CARD_ACTION_ACK_CONTENT = "\u5df2\u6536\u5230\uff0c\u6b63\u5728\u5904\u7406";
const FEISHU_CARD_CLOSE_ACK_CONTENT = "\u5df2\u5173\u95ed";
const TEN_MINUTE_CRON = "*/10 * * * *";

function isTopOfHourScheduledTick(controller: ScheduledController): boolean {
  return trim(controller.cron) === TEN_MINUTE_CRON && controller.scheduledTime % (60 * 60 * 1000) === 0;
}

function resolveRequestTimestampMs(request: Request): string {
  const rawTimestamp = trim(request.headers.get("x-lark-request-timestamp"));
  if (/^\d{10}$/.test(rawTimestamp)) {
    return String(Number(rawTimestamp) * 1000);
  }
  if (/^\d{13}$/.test(rawTimestamp)) {
    return rawTimestamp;
  }
  return "";
}

function readRecord(parent: Record<string, JsonValue>, key: string): Record<string, JsonValue> {
  const value = parent[key];
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, JsonValue>) : {};
}

export function normalizeQueueBody(body: unknown): Record<string, JsonValue> {
  if (body && typeof body === "object" && !Array.isArray(body)) {
    return body as Record<string, JsonValue>;
  }
  if (typeof body === "string") {
    const parsed = parseJsonSafely(body);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, JsonValue>;
    }
  }
  return {};
}

function buildFeishuCardActionAck(content: string): JsonValue {
  return {
    toast: {
      type: "info",
      content,
    },
  };
}

export function extractFeishuControlLogFields(payload: Record<string, JsonValue>): Record<string, string | undefined> {
  const header = readRecord(payload, "header");
  const event = readRecord(payload, "event");
  const action = readRecord(event, "action");
  const value = readRecord(action, "value");
  const context = readRecord(event, "context");
  const commandText = trim(value.command_text);
  const eventKey = trim(event.event_key) || trim(header.event_key) || trim(value.event_key);
  const hermesAction = trim(value.hermes_action);

  return {
    control_event_key: eventKey || undefined,
    control_hermes_action: hermesAction || undefined,
    control_command_text_preview: commandText ? commandText.slice(0, 80) : undefined,
    control_message_id: trim(context.open_message_id) || trim(context.message_id) || undefined,
  };
}

export function buildFeishuAcceptedBody(
  normalized: Pick<{ event_type: string }, "event_type">,
  payload: Record<string, JsonValue>,
): JsonValue {
  if (trim(normalized.event_type).toLowerCase() !== "card.action.trigger") {
    return { code: 0, msg: "accepted" };
  }
  const event = readRecord(payload, "event");
  const action = readRecord(event, "action");
  const value = readRecord(action, "value");
  const hermesAction = trim(value.hermes_action);
  if (hermesAction === "registry_close_card") {
    return buildFeishuCardActionAck(FEISHU_CARD_CLOSE_ACK_CONTENT);
  }
  return buildFeishuCardActionAck(FEISHU_CARD_ACTION_ACK_CONTENT);
}

const workerFetchHandler = {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (isInternalModelCatalogRequest(request)) {
      if (!isInternalModelCatalogRequestAuthorized(env, request)) {
        return jsonResponse({ ok: false, error: "unauthorized" }, { status: 401 });
      }
      const message = await bootstrapModelCatalogHeartbeat(env);
      return jsonResponse(buildModelCatalogQueueBootstrapResponse(message));
    }
    if (isInternalKpiRollupRequest(request)) {
      if (!isInternalModelCatalogRequestAuthorized(env, request)) {
        return jsonResponse({ ok: false, error: "unauthorized" }, { status: 401 });
      }
      const message = await bootstrapKpiRollupHeartbeat(env);
      return jsonResponse(buildKpiRollupQueueBootstrapResponse(message));
    }

    if (request.method !== "POST") {
      return jsonResponse({ ok: true, service: "hermes-feishu-gateway" });
    }

    const rawBody = await request.text();
    const payload = parseJsonSafely(rawBody);
    if (payload.type === "url_verification") {
      return jsonResponse({ challenge: trim(payload.challenge) });
    }
    if (trim(payload.encrypt)) {
      log("feishu.webhook.encryption_not_supported", {
        content_length: rawBody.length,
        has_encrypt: true,
      });
      return jsonResponse(
        { code: 400, msg: "encrypted payload is not supported; disable Feishu event encryption" },
        { status: 400 },
      );
    }
    if (!isVerificationTokenValid(payload, env)) {
      return jsonResponse({ code: 401, msg: "invalid verification token" }, { status: 401 });
    }

    const normalized = normalizeMessage(env, payload);
    const honeycomb = createHoneycombLogger(env);
    const webhookAcceptedPayload = {
      event_type: normalized.event_type,
      event_id: normalized.event_id,
      correlation_id: normalized.correlation_id,
      session_key: normalized.session_key,
      lane: normalized.lane,
      chat_id: normalized.chat_id,
      user_id: normalized.user_id,
      message_id: normalized.message_id,
      text_preview: normalized.text ? normalized.text.substring(0, 80) : undefined,
      ...(normalized.lane === "control" ? extractFeishuControlLogFields(payload) : {}),
      ...buildRoutingLogFields(normalized),
    };
    log("feishu.webhook.accepted", webhookAcceptedPayload);
    writeFeishuAnalyticsEvent(env, "feishu.webhook.accepted", webhookAcceptedPayload);

    ctx.waitUntil(
      honeycomb.logFeishuWebhook(
        normalized.correlation_id,
        normalized.event_type,
        normalized.chat_id,
        normalized.user_id,
        "received",
      ),
    );

    const fastPath = shouldUseDirectWorkerFastPath(normalized);
    const plannedPath = shouldUseDirectWorkerPlannedPath(normalized);
    const willEnqueueWorkflow = normalized.lane === "agent" || normalized.lane === "control";
    log("feishu.routing_decision", {
      correlation_id: normalized.correlation_id,
      lane: normalized.lane,
      route_hint: normalized.route_hint,
      request_class: normalized.request_class,
      gateway_eligible: normalized.gateway_eligible,
      fast_path: fastPath,
      planned_path: plannedPath,
      will_enqueue_workflow: willEnqueueWorkflow,
      route_family: normalized.route_family,
      task_kind: normalized.task_kind,
    });

    if (shouldAckReactionInline(normalized)) {
      ctx.waitUntil(runAckReactionFastPath(env, normalized));
    }

    log("feishu.routing.path_selected", {
      correlation_id: normalized.correlation_id,
      path: fastPath ? "direct_fast" : plannedPath ? "planned_workflow" : willEnqueueWorkflow ? "workflow" : "unknown",
      lane: normalized.lane,
    });
    const acceptedBody = buildFeishuAcceptedBody(normalized, payload);

    if (normalized.event_type === "im.message.message_read_v1") {
      const readInfo = extractMessageReadInfo(payload);
      const resolvedReadTime = readInfo.readTime || resolveRequestTimestampMs(request) || String(Date.now());
      const resolution =
        readInfo.messageIdList.length > 0
          ? await resolveMessageReadCorrelation(env, normalized.session_key, readInfo.messageIdList)
          : { refs: [], unmatched_message_ids: [] };
      const resolvedCorrelationIds = Array.from(
        new Set(resolution.refs.map((item) => trim(item.correlation_id)).filter(Boolean)),
      );
      const resolvedSessionKeys = Array.from(
        new Set(resolution.refs.map((item) => trim(item.session_key)).filter(Boolean)),
      );
      const resolvedEventIds = Array.from(
        new Set(resolution.refs.map((item) => trim(item.event_id)).filter(Boolean)),
      );
      const primaryRef = resolution.refs[0];
      const messageReadPayload = {
        event_id: normalized.event_id,
        correlation_id: resolvedCorrelationIds.length === 1 ? resolvedCorrelationIds[0] : normalized.correlation_id,
        session_key: resolvedSessionKeys.length === 1 ? resolvedSessionKeys[0] : normalized.session_key,
        chat_id: normalized.chat_id,
        message_id: trim(primaryRef?.send_message_id) || readInfo.messageIdList[0] || normalized.message_id,
        message_id_list: readInfo.messageIdList,
        message_count: readInfo.messageIdList.length,
        read_time: resolvedReadTime,
        reader_open_id: readInfo.readerOpenId,
        reader_user_id: readInfo.readerUserId,
        reader_union_id: readInfo.readerUnionId,
        tenant_key: readInfo.tenantKey,
        correlation_resolution_status:
          resolution.refs.length === 0
            ? "miss"
            : resolution.unmatched_message_ids.length > 0
              ? "partial"
              : "matched",
        resolved_correlation_ids: resolvedCorrelationIds,
        resolved_session_keys: resolvedSessionKeys,
        resolved_event_ids: resolvedEventIds,
        resolved_send_message_ids: resolution.refs.map((item) => trim(item.send_message_id)).filter(Boolean),
        unmatched_message_ids: resolution.unmatched_message_ids,
        execution_path: "worker_direct",
        ...buildRoutingLogFields(normalized),
      };
      log("feishu.message_read.accepted", messageReadPayload);
      writeFeishuAnalyticsEvent(env, "feishu.message_read.accepted", messageReadPayload);
      return jsonResponse({ code: 0, msg: "accepted" });
    }

    if (shouldUseDirectWorkerFastPath(normalized)) {
      ctx.waitUntil(
        runDirectWorkerFastPath(env, normalized)
          .then(() => {
            ctx.waitUntil(
              honeycomb.logFeishuWebhook(
                normalized.correlation_id,
                normalized.event_type,
                normalized.chat_id,
                normalized.user_id,
                "processed",
              ),
            );
          })
          .catch((error) => {
            log("feishu.direct_fast.error", {
              correlation_id: normalized.correlation_id,
              lane: normalized.lane,
              error: error instanceof Error ? error.message : String(error),
              ...buildRoutingLogFields(normalized),
            });
            ctx.waitUntil(
              honeycomb.logError(
                normalized.correlation_id,
                "direct_fast_error",
                error instanceof Error ? error.message : String(error),
                error instanceof Error ? error.stack : undefined,
              ),
            );
          }),
      );
      return jsonResponse(acceptedBody);
    }

    if (shouldUseDirectWorkerPlannedPath(normalized)) {
      ctx.waitUntil(
        enqueueWorkflowInstance(env, normalized)
          .then(() => {
            log("feishu.direct_planned.workflow_delegated", {
              correlation_id: normalized.correlation_id,
              lane: normalized.lane,
              route_hint: normalized.route_hint,
              reason: "waituntil_budget_guard",
              ...buildRoutingLogFields(normalized),
            });
            ctx.waitUntil(
              honeycomb.logFeishuWebhook(
                normalized.correlation_id,
                normalized.event_type,
                normalized.chat_id,
                normalized.user_id,
                "processed",
              ),
            );
          })
          .catch((error) => {
            log("feishu.direct_planned.workflow_delegate_error", {
              correlation_id: normalized.correlation_id,
              lane: normalized.lane,
              error: error instanceof Error ? error.message : String(error),
              ...buildRoutingLogFields(normalized),
            });
            ctx.waitUntil(
              honeycomb.logError(
                normalized.correlation_id,
                "direct_planned_error",
                error instanceof Error ? error.message : String(error),
                error instanceof Error ? error.stack : undefined,
              ),
            );
          }),
      );
      return jsonResponse(acceptedBody);
    }

    if (normalized.lane === "control" || normalized.lane === "agent") {
      ctx.waitUntil(
        enqueueWorkflowInstance(env, normalized)
          .then(() => {
            ctx.waitUntil(
              honeycomb.logFeishuWebhook(
                normalized.correlation_id,
                normalized.event_type,
                normalized.chat_id,
                normalized.user_id,
                "processed",
              ),
            );
          })
          .catch((error) => {
            log("feishu.workflow.error", {
              correlation_id: normalized.correlation_id,
              error: error instanceof Error ? error.message : String(error),
              ...buildRoutingLogFields(normalized),
            });
            ctx.waitUntil(
              honeycomb.logError(
                normalized.correlation_id,
                "workflow_error",
                error instanceof Error ? error.message : String(error),
                error instanceof Error ? error.stack : undefined,
              ),
            );
          }),
      );
      return jsonResponse(acceptedBody);
    }

    return jsonResponse(acceptedBody);
  },

  async scheduled(controller: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(
      runFeishuKpiRollupScheduledTick(env, controller).catch((error) => {
        console.log(
          JSON.stringify({
            event: "feishu.kpi_rollup.unhandled_error",
            cron: controller.cron,
            scheduled_time: controller.scheduledTime,
            error: error instanceof Error ? error.message : String(error),
          }),
        );
      }),
    );

    if (!isTopOfHourScheduledTick(controller) && trim(controller.cron) !== "") {
      return;
    }

    ctx.waitUntil(
      runModelCatalogScheduledTick(env, controller).catch((error) => {
        console.log(
          JSON.stringify({
            event: "model_catalog.tick.unhandled_error",
            cron: controller.cron,
            scheduled_time: controller.scheduledTime,
            error: error instanceof Error ? error.message : String(error),
          }),
        );
      }),
    );
  },

  async queue(batch: MessageBatch, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(
      (async () => {
        for (const message of batch.messages) {
          try {
            const body = normalizeQueueBody(message.body);
            const queueJob = trim(body.job);
            const result =
              queueJob === KPI_ROLLUP_JOB_NAME
                ? await handleKpiRollupQueueMessage(env, body, Math.trunc(Date.now() / 1000))
                : await handleModelCatalogQueueMessage(env, body, Math.trunc(Date.now() / 1000));
            console.log(
              JSON.stringify({
                event: queueJob === KPI_ROLLUP_JOB_NAME ? "feishu.kpi_rollup.queue.processed" : "model_catalog.queue.processed",
                message_id: message.id,
                result,
              }),
            );
          } catch (error) {
            const body = normalizeQueueBody(message.body);
            const queueJob = trim(body.job);
            console.log(
              JSON.stringify({
                event:
                  queueJob === KPI_ROLLUP_JOB_NAME
                    ? "feishu.kpi_rollup.queue.unhandled_error"
                    : "model_catalog.queue.unhandled_error",
                message_id: message.id,
                error: error instanceof Error ? error.message : String(error),
              }),
            );
            throw error;
          }
        }
      })(),
    );
  },
};

export default workerFetchHandler;
