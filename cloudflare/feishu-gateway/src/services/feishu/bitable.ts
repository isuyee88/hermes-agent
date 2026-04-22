import type { Env, JsonValue } from "../../runtime";
import { getTenantAccessToken } from "./auth";

type FeishuApiPayload = Record<string, JsonValue>;

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function getApiBase(env: Env): string {
  return trim(env.FEISHU_API_BASE) || "https://open.feishu.cn";
}

function buildUrl(path: string, params?: Record<string, string | number | boolean | undefined>): string {
  const url = new URL(path, "https://feishu.local");
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value === undefined) {
      continue;
    }
    url.searchParams.set(key, String(value));
  }
  return `${url.pathname}${url.search}`;
}

async function feishuRequest<T extends FeishuApiPayload = FeishuApiPayload>(
  env: Env,
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
  options: {
    params?: Record<string, string | number | boolean | undefined>;
    body?: Record<string, JsonValue>;
  } = {},
): Promise<T> {
  const token = await getTenantAccessToken(env);
  const response = await fetch(`${getApiBase(env)}${buildUrl(path, options.params)}`, {
    method,
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: method === "GET" ? undefined : JSON.stringify(options.body ?? {}),
  });
  const payload = (await response.json()) as FeishuApiPayload;
  if (!response.ok || Number(payload.code ?? 0) !== 0) {
    throw new Error(`feishu_api_failed:${response.status}:${trim(payload.msg)}:${path}`);
  }
  return payload as T;
}

function readPayloadData(payload: FeishuApiPayload): FeishuApiPayload {
  const data = payload.data;
  return data && typeof data === "object" && !Array.isArray(data) ? (data as FeishuApiPayload) : payload;
}

function readNestedObject(payload: FeishuApiPayload, key: string): Record<string, JsonValue> {
  const data = readPayloadData(payload);
  const value = data[key];
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, JsonValue>) : data;
}

async function paginateItems(
  env: Env,
  path: string,
  pageSize: number,
): Promise<Array<Record<string, JsonValue>>> {
  const items: Array<Record<string, JsonValue>> = [];
  let pageToken = "";
  while (true) {
    const payload = await feishuRequest(env, "GET", path, {
      params: {
        page_size: pageSize,
        page_token: pageToken || undefined,
      },
    });
    const data = readPayloadData(payload);
    for (const item of Array.isArray(data.items) ? data.items : []) {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        items.push(item as Record<string, JsonValue>);
      }
    }
    if (data.has_more !== true) {
      break;
    }
    pageToken = trim(data.page_token);
    if (!pageToken) {
      break;
    }
  }
  return items;
}

export async function listBitableTables(env: Env, appToken: string): Promise<Array<Record<string, JsonValue>>> {
  return paginateItems(env, `/open-apis/bitable/v1/apps/${appToken}/tables`, 100);
}

export async function listBitableFields(
  env: Env,
  appToken: string,
  tableId: string,
): Promise<Array<Record<string, JsonValue>>> {
  return paginateItems(env, `/open-apis/bitable/v1/apps/${appToken}/tables/${tableId}/fields`, 200);
}

export async function listBitableViews(
  env: Env,
  appToken: string,
  tableId: string,
): Promise<Array<Record<string, JsonValue>>> {
  return paginateItems(env, `/open-apis/bitable/v1/apps/${appToken}/tables/${tableId}/views`, 100);
}

export async function listBitableRecords(
  env: Env,
  appToken: string,
  tableId: string,
): Promise<Array<Record<string, JsonValue>>> {
  return paginateItems(env, `/open-apis/bitable/v1/apps/${appToken}/tables/${tableId}/records`, 500);
}

export async function createBitableTable(
  env: Env,
  args: {
    appToken: string;
    tableName: string;
    fields: Array<{ name: string; type: number }>;
    defaultViewName?: string;
  },
): Promise<Record<string, JsonValue>> {
  const payload = await feishuRequest(env, "POST", `/open-apis/bitable/v1/apps/${args.appToken}/tables`, {
    body: {
      table: {
        name: args.tableName,
        fields: args.fields.map((field) => ({
          field_name: field.name,
          type: field.type,
        })) as unknown as JsonValue,
        default_view_name: args.defaultViewName || null,
      } as unknown as JsonValue,
    },
  });
  return readNestedObject(payload, "table");
}

export async function createBitableField(
  env: Env,
  args: {
    appToken: string;
    tableId: string;
    fieldName: string;
    fieldType: number;
  },
): Promise<Record<string, JsonValue>> {
  const payload = await feishuRequest(
    env,
    "POST",
    `/open-apis/bitable/v1/apps/${args.appToken}/tables/${args.tableId}/fields`,
    {
      params: { client_token: crypto.randomUUID() },
      body: {
        field_name: args.fieldName,
        type: args.fieldType,
      },
    },
  );
  return readNestedObject(payload, "field");
}

export async function createBitableView(
  env: Env,
  args: {
    appToken: string;
    tableId: string;
    viewName: string;
    viewType?: "grid";
  },
): Promise<Record<string, JsonValue>> {
  const payload = await feishuRequest(
    env,
    "POST",
    `/open-apis/bitable/v1/apps/${args.appToken}/tables/${args.tableId}/views`,
    {
      body: {
        view_name: args.viewName,
        view_type: args.viewType || "grid",
      },
    },
  );
  return readNestedObject(payload, "view");
}

export async function createBitableRecord(
  env: Env,
  args: {
    appToken: string;
    tableId: string;
    fields: Record<string, JsonValue>;
  },
): Promise<Record<string, JsonValue>> {
  const payload = await feishuRequest(
    env,
    "POST",
    `/open-apis/bitable/v1/apps/${args.appToken}/tables/${args.tableId}/records`,
    {
      body: {
        fields: args.fields,
      },
    },
  );
  return readNestedObject(payload, "record");
}

export async function updateBitableRecord(
  env: Env,
  args: {
    appToken: string;
    tableId: string;
    recordId: string;
    fields: Record<string, JsonValue>;
  },
): Promise<Record<string, JsonValue>> {
  const payload = await feishuRequest(
    env,
    "PUT",
    `/open-apis/bitable/v1/apps/${args.appToken}/tables/${args.tableId}/records/${args.recordId}`,
    {
      body: {
        fields: args.fields,
      },
    },
  );
  return readNestedObject(payload, "record");
}

export async function deleteBitableRecord(
  env: Env,
  args: {
    appToken: string;
    tableId: string;
    recordId: string;
  },
): Promise<void> {
  await feishuRequest(
    env,
    "DELETE",
    `/open-apis/bitable/v1/apps/${args.appToken}/tables/${args.tableId}/records/${args.recordId}`,
  );
}
