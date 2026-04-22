export interface HoneycombEvent {
  timestamp?: string;
  dataset?: string;
  data: Record<string, unknown>;
  trace?: {
    trace_id: string;
    span_id?: string;
    parent_id?: string;
  };
}

export class HoneycombLogger {
  private readonly apiKey: string | undefined;
  private readonly dataset: string;
  private readonly enabled: boolean;

  constructor(apiKey?: string, dataset: string = "hermes-feishu-gateway") {
    this.apiKey = apiKey;
    this.dataset = dataset;
    this.enabled = !!apiKey && apiKey.length > 0;
  }

  async sendEvent(event: HoneycombEvent): Promise<void> {
    if (!this.enabled || !this.apiKey) {
      console.log("[HONEYCOMB DISABLED]", event);
      return;
    }

    const payload = {
      timestamp: event.timestamp || new Date().toISOString(),
      dataset: event.dataset || this.dataset,
      data: {
        ...event.data,
        service_name: "hermes-feishu-gateway",
        environment: "production",
      },
      trace: event.trace,
    };

    try {
      const response = await fetch("https://api.honeycomb.io/v1/events", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Honeycomb-Team": this.apiKey,
          "X-Honeycomb-Dataset": this.dataset,
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[HONEYCOMB ERROR] ${response.status}: ${errorText}`);
      } else {
        console.log(`[HONEYCOMB] Event sent: ${event.data["event_type"] || "unknown"}`);
      }
    } catch (error) {
      console.error("[HONEYCOMB SEND FAILED]", error);
    }
  }

  async logRequest(
    correlationId: string,
    path: string,
    method: string,
    statusCode: number,
    durationMs: number,
    metadata: Record<string, unknown> = {}
  ): Promise<void> {
    await this.sendEvent({
      data: {
        event_type: "http_request",
        correlation_id: correlationId,
        path,
        method,
        status_code: statusCode,
        duration_ms: durationMs,
        ...metadata,
      },
      trace: {
        trace_id: correlationId,
      },
    });
  }

  async logFeishuWebhook(
    correlationId: string,
    eventType: string,
    chatId: string,
    userId: string,
    status: "received" | "processed" | "sent" | "error",
    error?: string
  ): Promise<void> {
    await this.sendEvent({
      data: {
        event_type: "feishu_webhook",
        correlation_id: correlationId,
        webhook_event_type: eventType,
        chat_id: chatId,
        user_id: userId,
        status,
        error,
      },
      trace: {
        trace_id: correlationId,
      },
    });
  }

  async logModalCall(
    correlationId: string,
    endpoint: string,
    status: "started" | "completed" | "error",
    durationMs?: number,
    error?: string
  ): Promise<void> {
    await this.sendEvent({
      data: {
        event_type: "modal_call",
        correlation_id: correlationId,
        endpoint,
        status,
        duration_ms: durationMs,
        error,
      },
      trace: {
        trace_id: correlationId,
      },
    });
  }

  async logFeishuSend(
    correlationId: string,
    chatId: string,
    operationKind: string,
    status: "started" | "completed" | "error",
    error?: string
  ): Promise<void> {
    await this.sendEvent({
      data: {
        event_type: "feishu_send",
        correlation_id: correlationId,
        chat_id: chatId,
        operation_kind: operationKind,
        status,
        error,
      },
      trace: {
        trace_id: correlationId,
      },
    });
  }

  async logError(
    correlationId: string,
    errorType: string,
    message: string,
    stack?: string,
    context?: Record<string, unknown>
  ): Promise<void> {
    await this.sendEvent({
      data: {
        event_type: "error",
        correlation_id: correlationId,
        error_type: errorType,
        message,
        stack,
        ...context,
      },
      trace: {
        trace_id: correlationId,
      },
    });
  }
}

export function createHoneycombLogger(env: Record<string, string | undefined>): HoneycombLogger {
  return new HoneycombLogger(
    env.HONEYCOMB_API_KEY,
    env.HONEYCOMB_DATASET || "hermes-feishu-gateway"
  );
}
