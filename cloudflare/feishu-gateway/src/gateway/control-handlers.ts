import type { Env, FeishuNormalizedPayload, JsonValue, ModalInternalResponse } from "../runtime";

type ReceiveIdType = "chat_id" | "open_id" | "user_id" | "union_id";

type ControlDeps = {
  trim: (value: unknown) => string;
  readEvent: (payload: Record<string, JsonValue>) => Record<string, JsonValue>;
  readRecord: (parent: Record<string, JsonValue>, key: string) => Record<string, JsonValue>;
  readString: (parent: Record<string, JsonValue>, key: string) => string;
  buildTextSendPlan: (content: string) => Array<Record<string, JsonValue>>;
  buildInteractiveCardSendPlan: (
    card: Record<string, JsonValue>,
    receiveId: string,
    receiveIdType: string,
  ) => Array<Record<string, JsonValue>>;
  buildDeleteMessageSendPlan: (messageId: string) => Array<Record<string, JsonValue>>;
  callModalWithReconciles: <T>(
    env: Env,
    path: string,
    normalized: FeishuNormalizedPayload,
    body: Record<string, JsonValue>,
  ) => Promise<T>;
};

export function controlPayloadBase(normalized: FeishuNormalizedPayload): Record<string, JsonValue> {
  return {
    correlation_id: normalized.correlation_id,
    session_key: normalized.session_key,
    lane: normalized.lane,
    route_hint: normalized.route_hint,
    task_kind: normalized.task_kind,
    request_class: normalized.request_class,
    content_modalities: normalized.content_modalities as unknown as JsonValue,
    route_family: normalized.route_family,
    gateway_route_name: normalized.gateway_route_name,
    gateway_eligible: normalized.gateway_eligible,
    requires_tools: normalized.requires_tools,
    requires_browser: normalized.requires_browser,
    requires_media_hydration: normalized.requires_media_hydration,
    requires_modal_runtime: normalized.requires_modal_runtime,
    modality_profile: normalized.modality_profile,
    toolset: normalized.toolset as unknown as JsonValue,
    reason_code: normalized.reason_code,
    chat_id: normalized.chat_id,
    chat_type: normalized.chat_type,
    chat_name: normalized.chat_name,
    user_id: normalized.user_id,
    user_name: normalized.user_name,
    message_id: normalized.message_id,
  };
}

function buildAgentExecCommandPayload(
  normalized: FeishuNormalizedPayload,
  commandText: string,
): Record<string, JsonValue> {
  return {
    ...controlPayloadBase(normalized),
    text: commandText,
    message_type: "command",
    internal: true,
    raw_message: normalized.raw_payload as unknown as JsonValue,
    fallback_reason: "session_control_dispatch_failed",
    route_decision_reason: "worker_control_fallback_agent_exec",
  };
}

function buildSkillComboInstruction(
  comboLabel: string,
  skills: string[],
  suggestedPersonality: string,
): string {
  const normalizedLabel = comboLabel.trim() || "技能组合";
  const normalizedSkills = skills.map((item) => item.trim()).filter(Boolean);
  let instruction = `请切换到“${normalizedLabel}”工作模式。`;
  if (normalizedSkills.length > 0) {
    instruction += ` 本次请优先加载这些技能：${normalizedSkills.join(", ")}。`;
  }
  instruction += "先用中文在 3 行内确认已加载的技能、适用场景，以及接下来准备如何协作。";
  const personality = suggestedPersonality.trim().toLowerCase();
  if (personality) {
    instruction += ` 如需匹配风格，建议同步执行 \`/personality ${personality}\`。`;
  }
  return instruction;
}

function buildAgentExecTextPayload(
  normalized: FeishuNormalizedPayload,
  text: string,
): Record<string, JsonValue> {
  return {
    ...controlPayloadBase(normalized),
    text,
    message_type: "text",
    internal: true,
    raw_message: normalized.raw_payload as unknown as JsonValue,
    fallback_reason: "session_control_skill_combo_failed",
    route_decision_reason: "worker_control_fallback_agent_exec_text",
  };
}

async function dispatchCommandViaAgentExec(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: Pick<ControlDeps, "callModalWithReconciles">,
  commandText: string,
): Promise<ModalInternalResponse> {
  return deps.callModalWithReconciles<ModalInternalResponse>(env, "/internal/feishu/agent-exec", normalized, {
    ...buildAgentExecCommandPayload(normalized, commandText),
  });
}

async function dispatchTextViaAgentExec(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: Pick<ControlDeps, "callModalWithReconciles">,
  text: string,
): Promise<ModalInternalResponse> {
  return deps.callModalWithReconciles<ModalInternalResponse>(env, "/internal/feishu/agent-exec", normalized, {
    ...buildAgentExecTextPayload(normalized, text),
  });
}

function inferFeishuReceiveIdType(trim: ControlDeps["trim"], receiveId: string): ReceiveIdType {
  const normalizedReceiveId = trim(receiveId);
  if (!normalizedReceiveId) {
    return "chat_id";
  }
  if (normalizedReceiveId.startsWith("oc_")) {
    return "chat_id";
  }
  if (normalizedReceiveId.startsWith("ou_")) {
    return "open_id";
  }
  if (normalizedReceiveId.startsWith("on_")) {
    return "union_id";
  }
  if (/^[A-Za-z0-9_\-]+$/.test(normalizedReceiveId)) {
    return "user_id";
  }
  return "chat_id";
}

function resolveDirectUserTarget(
  normalized: FeishuNormalizedPayload,
  deps: Pick<ControlDeps, "readEvent" | "readRecord" | "readString" | "trim">,
): { receiveId: string; receiveIdType: "open_id" | "user_id" | "union_id" } | null {
  const event = deps.readEvent(normalized.raw_payload);
  const sender = deps.readRecord(event, "sender");
  const senderId = deps.readRecord(sender, "sender_id");
  const operator = deps.readRecord(event, "operator");
  const operatorId = deps.readRecord(operator, "operator_id");
  const user = deps.readRecord(event, "user");
  const userIdRecord = deps.readRecord(event, "user_id");
  const candidates: Array<{ receiveId: string; receiveIdType: "open_id" | "user_id" | "union_id" }> = [
    { receiveId: deps.readString(senderId, "open_id"), receiveIdType: "open_id" },
    { receiveId: deps.readString(senderId, "user_id"), receiveIdType: "user_id" },
    { receiveId: deps.readString(senderId, "union_id"), receiveIdType: "union_id" },
    { receiveId: deps.readString(operatorId, "open_id"), receiveIdType: "open_id" },
    { receiveId: deps.readString(operatorId, "user_id"), receiveIdType: "user_id" },
    { receiveId: deps.readString(operatorId, "union_id"), receiveIdType: "union_id" },
    { receiveId: deps.readString(userIdRecord, "open_id"), receiveIdType: "open_id" },
    { receiveId: deps.readString(userIdRecord, "user_id"), receiveIdType: "user_id" },
    { receiveId: deps.readString(userIdRecord, "union_id"), receiveIdType: "union_id" },
    { receiveId: deps.readString(user, "open_id"), receiveIdType: "open_id" },
    { receiveId: deps.readString(user, "user_id"), receiveIdType: "user_id" },
    { receiveId: deps.readString(user, "union_id"), receiveIdType: "union_id" },
  ];
  for (const candidate of candidates) {
    if (deps.trim(candidate.receiveId)) {
      return {
        receiveId: deps.trim(candidate.receiveId),
        receiveIdType: candidate.receiveIdType,
      };
    }
  }
  return null;
}

export function resolveDefaultSendTarget(
  normalized: FeishuNormalizedPayload,
  deps: Pick<ControlDeps, "readEvent" | "readRecord" | "readString" | "trim">,
): { receiveId: string; receiveIdType: ReceiveIdType } {
  const event = deps.readEvent(normalized.raw_payload);
  const message = deps.readRecord(event, "message");
  const chat = deps.readRecord(event, "chat");
  const context = deps.readRecord(event, "context");
  const rawChatType =
    deps.readString(message, "chat_type") || deps.readString(chat, "chat_type") || deps.readString(context, "chat_type");
  if (rawChatType.toLowerCase() === "p2p") {
    const directTarget = resolveDirectUserTarget(normalized, deps);
    if (directTarget) {
      return directTarget;
    }
    const userTarget = deps.trim(normalized.user_id);
    if (userTarget) {
      return {
        receiveId: userTarget,
        receiveIdType: inferFeishuReceiveIdType(deps.trim, userTarget),
      };
    }
  }
  return {
    receiveId: normalized.chat_id,
    receiveIdType: "chat_id",
  };
}

function resolveMenuTarget(
  payload: Record<string, JsonValue>,
  deps: Pick<ControlDeps, "readEvent" | "readRecord" | "readString">,
): { receiveId: string; receiveIdType: string } | null {
  const event = deps.readEvent(payload);
  const context = deps.readRecord(event, "context");
  const chat = deps.readRecord(event, "chat");
  const operator = deps.readRecord(event, "operator");
  const operatorId = deps.readRecord(operator, "operator_id");
  const openChatId = deps.readString(context, "open_chat_id");
  const chatId = deps.readString(chat, "chat_id") || openChatId;
  const openId = deps.readString(operatorId, "open_id");
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

export async function handleMenuEvent(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: ControlDeps,
): Promise<ModalInternalResponse> {
  const event = deps.readEvent(normalized.raw_payload);
  const eventKey = deps.readString(event, "event_key");
  const target = resolveMenuTarget(normalized.raw_payload, deps);
  if (!target) {
    throw new Error(`menu_target_missing:${eventKey}`);
  }
  if (eventKey === "model_status" || eventKey === "route_status") {
    const internal = await deps.callModalWithReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
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
      send_plan: deps.buildTextSendPlan(content),
      action_plan: deps.buildTextSendPlan(content),
    };
  }
  if (eventKey === "provider_status") {
    return dispatchCommandViaAgentExec(env, normalized, deps, "/provider");
  }
  const internal = await deps.callModalWithReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
    ...controlPayloadBase(normalized),
    action: "render_card",
    event_key: eventKey,
  });
  if (!internal.card) {
    throw new Error(internal.error || `render_card_failed:${eventKey}`);
  }
  return {
    ...internal,
    send_plan: deps.buildInteractiveCardSendPlan(internal.card, target.receiveId, target.receiveIdType),
    action_plan: deps.buildInteractiveCardSendPlan(internal.card, target.receiveId, target.receiveIdType),
  };
}

export async function handleCardAction(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: ControlDeps,
): Promise<ModalInternalResponse> {
  const event = deps.readEvent(normalized.raw_payload);
  const action = deps.readRecord(event, "action");
  const actionValue = deps.readRecord(action, "value");
  const hermesAction = deps.trim(actionValue.hermes_action);
  if (hermesAction === "registry_close_card") {
    const context = deps.readRecord(event, "context");
    const messageId = deps.readString(context, "open_message_id") || deps.readString(context, "message_id");
    const deletePlan = deps.buildDeleteMessageSendPlan(messageId);
    return {
      status: "ok",
      route_hint: "fast_control",
      execution_mode: "control_complete",
      send_plan: deletePlan,
      action_plan: deletePlan,
    };
  }
  if (hermesAction === "open_menu_card") {
    const internal = await deps.callModalWithReconciles<ModalInternalResponse>(env, "/internal/feishu/session-control", normalized, {
      ...controlPayloadBase(normalized),
      action: "render_card",
      event_key: deps.trim(actionValue.event_key),
    });
    if (!internal.card) {
      throw new Error(internal.error || "open_menu_card_failed");
    }
    const target = resolveDefaultSendTarget(normalized, deps);
    return {
      ...internal,
      send_plan: deps.buildInteractiveCardSendPlan(internal.card, target.receiveId, target.receiveIdType),
      action_plan: deps.buildInteractiveCardSendPlan(internal.card, target.receiveId, target.receiveIdType),
    };
  }

  let internal: ModalInternalResponse;
  if (hermesAction === "skill_combo_apply") {
    const comboId = deps.trim(actionValue.combo_id);
    const comboLabel = deps.trim(actionValue.combo_label) || comboId;
    const suggestedPersonality = deps.trim(actionValue.suggested_personality);
    const skills = Array.isArray(actionValue.skills)
      ? actionValue.skills.map((item) => deps.trim(item)).filter(Boolean)
      : [];
    const skillInstruction = buildSkillComboInstruction(comboLabel, skills, suggestedPersonality);
    internal = await dispatchTextViaAgentExec(env, normalized, deps, skillInstruction);
  } else {
    let commandText = "";
    if (hermesAction === "registry_switch_model") {
      commandText = `/model ${deps.trim(actionValue.model)} --provider ${deps.trim(actionValue.provider)}`;
    } else if (hermesAction === "personality_set") {
      commandText = `/personality ${deps.trim(actionValue.personality) || "none"}`;
    } else if (hermesAction === "command_run") {
      commandText = deps.trim(actionValue.command_text);
    }
    if (!commandText) {
      throw new Error(`unsupported_card_action:${hermesAction}`);
    }
    internal = await dispatchCommandViaAgentExec(env, normalized, deps, commandText);
  }
  return internal;
}

export async function handleControlEvent(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: ControlDeps,
): Promise<ModalInternalResponse> {
  if (normalized.event_type === "application.bot.menu_v6") {
    return handleMenuEvent(env, normalized, deps);
  }
  if (normalized.event_type === "card.action.trigger") {
    return handleCardAction(env, normalized, deps);
  }
  return { status: "ignored", send_plan: [] };
}

export async function handleAgentCommand(
  env: Env,
  normalized: FeishuNormalizedPayload,
  deps: Pick<ControlDeps, "callModalWithReconciles">,
): Promise<ModalInternalResponse> {
  return dispatchCommandViaAgentExec(env, normalized, deps, normalized.text);
}
