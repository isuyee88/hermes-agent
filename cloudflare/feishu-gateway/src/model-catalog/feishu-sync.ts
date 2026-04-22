import type { Env, JsonValue } from "../runtime";
import {
  buildModelRegistryFeishuFields,
  getModelRegistryBitableBlueprint,
  MODEL_REGISTRY_FIELD_SPECS,
} from "./feishu-mapping";
import type { ModelRegistrySyncEntry } from "./types";
import {
  createBitableField,
  createBitableRecord,
  createBitableTable,
  createBitableView,
  deleteBitableRecord,
  listBitableFields,
  listBitableRecords,
  listBitableTables,
  listBitableViews,
  updateBitableRecord,
} from "../services/feishu/bitable";

type ModelRegistrySyncViewRow = {
  provider?: string | null;
  model?: string | null;
  display_name?: string | null;
  status?: string | null;
  hidden?: number | null;
  is_available?: number | null;
  is_free?: number | null;
  rank?: number | null;
  selection_hint?: string | null;
  manual_pinned?: number | null;
  recent_used?: number | null;
  recent_used_count?: number | null;
  generated_command?: string | null;
  last_probe_at?: number | null;
  recent_used_at?: number | null;
  last_sync_at?: number | null;
  latency_ms?: number | null;
  context_window?: number | null;
  reasoning?: number | null;
  consecutive_failures?: number | null;
  failure_kind?: string | null;
  last_error_code?: string | null;
  last_error_message?: string | null;
  last_failed_at?: number | null;
  source?: string | null;
  function_type?: string | null;
  task_kinds_json?: string | null;
  modalities_json?: string | null;
  vision?: number | null;
  tool_calling?: number | null;
  structured_output?: number | null;
  streaming?: number | null;
  max_output_tokens?: number | null;
  input_price_per_million?: number | null;
  output_price_per_million?: number | null;
  cache_read_price_per_million?: number | null;
  cache_write_price_per_million?: number | null;
  temperature_supported?: number | null;
  top_p_supported?: number | null;
  json_mode_supported?: number | null;
  parameter_schema_json?: string | null;
  gateway_route_name?: string | null;
  route_family?: string | null;
  hidden_reason?: string | null;
  rate_limit_rpm?: number | null;
  rate_limit_tpm?: number | null;
  rate_limit_rpd?: number | null;
  burst_limit?: number | null;
  rate_limit_window_seconds?: number | null;
  cooldown_until?: number | null;
  health_score?: number | null;
  success_rate?: number | null;
  error_rate?: number | null;
  cache_hit_rate?: number | null;
  latency_p50_ms?: number | null;
  latency_p95_ms?: number | null;
  ttfb_p50_ms?: number | null;
  ttfb_p95_ms?: number | null;
  positive_feedback_count?: number | null;
  negative_feedback_count?: number | null;
  provider_failure_count?: number | null;
  rate_limit_count?: number | null;
  auth_error_count?: number | null;
  model_not_found_count?: number | null;
};

type EnsureSchemaSummary = {
  status: "ok" | "partial";
  tableId: string;
  tableName: string;
  createdTable: boolean;
  createdFields: string[];
  createdViews: string[];
  fieldNames: string[];
};

export type FeishuModelRegistrySyncSummary = {
  mirrored: boolean;
  skippedReason?: string;
  appToken?: string;
  tableId?: string;
  tableName?: string;
  rowCount?: number;
  existingRowCount?: number;
  blankRowCount?: number;
  historicalOnlyCount?: number;
  staleDeleteCandidateCount?: number;
  staleHideCandidateCount?: number;
  entryCreateCandidateCount?: number;
  entryUpdateCandidateCount?: number;
  created?: number;
  updated?: number;
  hidden?: number;
  createdTable?: boolean;
  createdFields?: string[];
  createdViews?: string[];
  missingOptionalFields?: string[];
  pending?: boolean;
  remainingEstimate?: number;
  mutationBudget?: number;
  mutationsApplied?: number;
  deleted?: number;
};

function trim(value: unknown): string {
  return String(value ?? "").trim();
}

function parseBoolean(value: unknown, fallback = false): boolean {
  const normalized = trim(value).toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return fallback;
}

function parsePositiveInt(value: unknown, fallback: number, min = 1, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = Number.parseInt(trim(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, parsed));
}

function parseNonNegativeInt(value: unknown, fallback: number, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = Number.parseInt(trim(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(max, Math.max(0, parsed));
}

function normalizeNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const normalized = trim(value);
  if (!normalized) {
    return null;
  }
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function toBoolean(value: unknown): boolean | null {
  const normalized = normalizeNumber(value);
  if (normalized === null) {
    return null;
  }
  return normalized === 1;
}

function parseStringArray(value: unknown): string[] | null {
  const normalized = trim(value);
  if (!normalized) {
    return null;
  }
  try {
    const parsed = JSON.parse(normalized) as unknown;
    if (!Array.isArray(parsed)) {
      return null;
    }
    const values = parsed.map((item) => trim(item)).filter(Boolean);
    return values.length > 0 ? values : null;
  } catch {
    return null;
  }
}

function buildRecordKey(provider: string, model: string): string {
  return `${provider.toLowerCase()}::${model}`;
}

function buildFieldsForSchema(
  entry: ModelRegistrySyncEntry,
  fieldNames: Set<string>,
  lastSyncAt: number,
): Record<string, JsonValue> {
  const base = buildModelRegistryFeishuFields({ ...entry, lastSyncAt });
  const fields: Record<string, JsonValue> = {};
  for (const spec of MODEL_REGISTRY_FIELD_SPECS) {
    if (!fieldNames.has(spec.name)) {
      continue;
    }
    if (spec.name in base) {
      fields[spec.name] = base[spec.name] as JsonValue;
      continue;
    }
    if (spec.name === "Provider") {
      fields[spec.name] = entry.provider;
      continue;
    }
    if (spec.name === "Model") {
      fields[spec.name] = entry.model;
      continue;
    }
    fields[spec.name] = null;
  }
  return fields;
}

function normalizeComparableFieldValue(value: JsonValue | undefined): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "";
  }
  return trim(value);
}

function shouldUpdateBitableRecord(
  existingFields: Record<string, JsonValue>,
  desiredFields: Record<string, JsonValue>,
): boolean {
  const keys = new Set([...Object.keys(existingFields), ...Object.keys(desiredFields)]);
  for (const key of keys) {
    // Avoid spending a write just to refresh the sync timestamp on otherwise unchanged rows.
    if (key === "Last Sync At") {
      continue;
    }
    if (normalizeComparableFieldValue(existingFields[key]) !== normalizeComparableFieldValue(desiredFields[key])) {
      return true;
    }
  }
  return false;
}

export function mapModelRegistrySyncRows(
  rows: ModelRegistrySyncViewRow[],
  lastSyncAt?: number,
): ModelRegistrySyncEntry[] {
  return rows
    .map((row) => {
      const provider = trim(row.provider).toLowerCase();
      const model = trim(row.model);
      if (!provider || !model) {
        return null;
      }
      return {
        provider,
        model,
        displayName: trim(row.display_name) || null,
        status: trim(row.status) || null,
        hidden: toBoolean(row.hidden),
        isAvailable: toBoolean(row.is_available),
        isFree: toBoolean(row.is_free),
        rank: normalizeNumber(row.rank),
        selectionHint: trim(row.selection_hint) || null,
        manualPinned: toBoolean(row.manual_pinned),
        recentUsed: toBoolean(row.recent_used),
        recentUsedCount: normalizeNumber(row.recent_used_count),
        generatedCommand: trim(row.generated_command) || null,
        lastProbeAt: normalizeNumber(row.last_probe_at),
        recentUsedAt: normalizeNumber(row.recent_used_at),
        lastSyncAt: lastSyncAt ?? normalizeNumber(row.last_sync_at),
        latencyMs: normalizeNumber(row.latency_ms),
        contextWindow: normalizeNumber(row.context_window),
        reasoning: toBoolean(row.reasoning),
        consecutiveFailures: normalizeNumber(row.consecutive_failures),
        failureKind: trim(row.failure_kind) || null,
        lastErrorCode: trim(row.last_error_code) || null,
        lastErrorMessage: trim(row.last_error_message) || null,
        lastFailedAt: normalizeNumber(row.last_failed_at),
        source: trim(row.source) || null,
        functionType: trim(row.function_type) || null,
        taskKinds: parseStringArray(row.task_kinds_json),
        modalities: parseStringArray(row.modalities_json),
        toolCalling: toBoolean(row.tool_calling),
        streaming: toBoolean(row.streaming),
        structuredOutput: toBoolean(row.structured_output),
        vision: toBoolean(row.vision),
        maxOutputTokens: normalizeNumber(row.max_output_tokens),
        inputPricePerMillion: normalizeNumber(row.input_price_per_million),
        outputPricePerMillion: normalizeNumber(row.output_price_per_million),
        cacheReadPricePerMillion: normalizeNumber(row.cache_read_price_per_million),
        cacheWritePricePerMillion: normalizeNumber(row.cache_write_price_per_million),
        rateLimitRpm: normalizeNumber(row.rate_limit_rpm),
        rateLimitTpm: normalizeNumber(row.rate_limit_tpm),
        rateLimitRpd: normalizeNumber(row.rate_limit_rpd),
        burstLimit: normalizeNumber(row.burst_limit),
        rateLimitWindowSeconds: normalizeNumber(row.rate_limit_window_seconds),
        cooldownUntil: normalizeNumber(row.cooldown_until),
        healthScore: normalizeNumber(row.health_score),
        successRate: normalizeNumber(row.success_rate),
        errorRate: normalizeNumber(row.error_rate),
        cacheHitRate: normalizeNumber(row.cache_hit_rate),
        latencyP50Ms: normalizeNumber(row.latency_p50_ms),
        latencyP95Ms: normalizeNumber(row.latency_p95_ms),
        ttfbP50Ms: normalizeNumber(row.ttfb_p50_ms),
        ttfbP95Ms: normalizeNumber(row.ttfb_p95_ms),
        positiveFeedbackCount: normalizeNumber(row.positive_feedback_count),
        negativeFeedbackCount: normalizeNumber(row.negative_feedback_count),
        providerFailureCount: normalizeNumber(row.provider_failure_count),
        rateLimitCount: normalizeNumber(row.rate_limit_count),
        authErrorCount: normalizeNumber(row.auth_error_count),
        modelNotFoundCount: normalizeNumber(row.model_not_found_count),
        gatewayRouteName: trim(row.gateway_route_name) || null,
        routeFamily: trim(row.route_family) || null,
        hiddenReason: trim(row.hidden_reason) || null,
        temperatureSupported: toBoolean(row.temperature_supported),
        topPSupported: toBoolean(row.top_p_supported),
        jsonModeSupported: toBoolean(row.json_mode_supported),
        parameterSchemaJson: trim(row.parameter_schema_json) || null,
      } satisfies ModelRegistrySyncEntry;
    })
    .filter((item): item is ModelRegistrySyncEntry => item !== null);
}

export async function loadModelRegistrySyncEntries(
  env: Env,
  lastSyncAt = Math.trunc(Date.now() / 1000),
): Promise<ModelRegistrySyncEntry[]> {
  const db = env.MODEL_CATALOG_DB;
  if (!db) {
    return [];
  }
  const result = await db.prepare(`SELECT * FROM model_registry_sync_view ORDER BY rank ASC, provider ASC, model ASC`).all<ModelRegistrySyncViewRow>();
  const rows = Array.isArray(result.results) ? result.results : [];
  return mapModelRegistrySyncRows(rows, lastSyncAt);
}

async function ensureModelRegistryBitableSchema(
  env: Env,
  args: {
    appToken: string;
    tableId?: string;
    tableName: string;
  },
): Promise<EnsureSchemaSummary> {
  const blueprint = getModelRegistryBitableBlueprint(args.tableName);
  const tables = await listBitableTables(env, args.appToken);
  let selectedTable =
    tables.find((item) => trim(item.table_id) === trim(args.tableId)) ??
    tables.find((item) => trim(item.name) === args.tableName) ??
    null;
  let tableId = trim(selectedTable?.table_id);
  let createdTable = false;
  const createdFields: string[] = [];
  const createdViews: string[] = [];

  if (!selectedTable) {
    selectedTable = await createBitableTable(env, {
      appToken: args.appToken,
      tableName: args.tableName,
      fields: blueprint.fields.map((field) => ({ name: field.name, type: field.type })),
      defaultViewName: blueprint.views[0]?.name,
    });
    tableId = trim(selectedTable.table_id);
    createdTable = true;
  }

  if (!tableId) {
    throw new Error("feishu_bitable_table_id_missing");
  }

  const fieldItems = await listBitableFields(env, args.appToken, tableId);
  const existingFieldNames = new Set(fieldItems.map((item) => trim(item.field_name)).filter(Boolean));
  for (const spec of blueprint.fields) {
    if (existingFieldNames.has(spec.name)) {
      continue;
    }
    await createBitableField(env, {
      appToken: args.appToken,
      tableId,
      fieldName: spec.name,
      fieldType: spec.type,
    });
    existingFieldNames.add(spec.name);
    createdFields.push(spec.name);
  }

  const views = await listBitableViews(env, args.appToken, tableId);
  const existingViewNames = new Set(views.map((item) => trim(item.view_name)).filter(Boolean));
  for (const view of blueprint.views) {
    if (existingViewNames.has(view.name)) {
      continue;
    }
    await createBitableView(env, {
      appToken: args.appToken,
      tableId,
      viewName: view.name,
      viewType: view.view_type,
    });
    existingViewNames.add(view.name);
    createdViews.push(view.name);
  }

  const missingRequiredFields = blueprint.required_field_names.filter((name) => !existingFieldNames.has(name));

  return {
    status: missingRequiredFields.length === 0 ? "ok" : "partial",
    tableId,
    tableName: trim(selectedTable?.name) || args.tableName,
    createdTable,
    createdFields,
    createdViews,
    fieldNames: Array.from(existingFieldNames).sort(),
  };
}

export async function syncModelRegistryToFeishuBitable(
  env: Env,
  nowSeconds = Math.trunc(Date.now() / 1000),
): Promise<FeishuModelRegistrySyncSummary> {
  const enabled = parseBoolean(env.FEISHU_MODEL_REGISTRY_MIRROR_ENABLED, false);
  if (!enabled) {
    return { mirrored: false, skippedReason: "mirror_disabled" };
  }
  const appToken = trim(env.FEISHU_BITABLE_APP_TOKEN);
  if (!appToken) {
    return { mirrored: false, skippedReason: "missing_app_token" };
  }

  const tableName = trim(env.FEISHU_BITABLE_TABLE_NAME) || "Hermes Model Registry";
  const schema = await ensureModelRegistryBitableSchema(env, {
    appToken,
    tableId: trim(env.FEISHU_BITABLE_TABLE_ID) || undefined,
    tableName,
  });
  const fieldNameSet = new Set(schema.fieldNames);
  if (!fieldNameSet.has("Provider") || !fieldNameSet.has("Model")) {
    throw new Error("feishu_bitable_required_fields_missing");
  }

  const entries = await loadModelRegistrySyncEntries(env, nowSeconds);
  const existingRecords = await listBitableRecords(env, appToken, schema.tableId);
  const mutationBudget = parsePositiveInt(
    (env as Env & Record<string, unknown>).FEISHU_MODEL_REGISTRY_MAX_MUTATIONS_PER_RUN,
    25,
    1,
    200,
  );
  const staleDeleteAfterSeconds = parseNonNegativeInt(
    (env as Env & Record<string, unknown>).FEISHU_MODEL_REGISTRY_STALE_DELETE_AFTER_SECONDS,
    86400,
  );
  const existingLookup = new Map<string, { recordId: string; fields: Record<string, JsonValue> }>();
  const blankRecordIds: string[] = [];
  for (const record of existingRecords) {
    const recordId = trim(record.record_id);
    const fields =
      record.fields && typeof record.fields === "object" && !Array.isArray(record.fields)
        ? (record.fields as Record<string, JsonValue>)
        : {};
    const provider = trim(fields.Provider).toLowerCase();
    const model = trim(fields.Model);
    if (recordId && (!provider || !model)) {
      blankRecordIds.push(recordId);
      continue;
    }
    if (recordId && provider && model) {
      existingLookup.set(buildRecordKey(provider, model), { recordId, fields });
    }
  }

  let created = 0;
  let updated = 0;
  let hidden = 0;
  let deleted = 0;
  let mutationsApplied = 0;
  let pending = false;
  const orderedEntries = [...entries].sort((left, right) => {
    const leftHasExisting = existingLookup.has(buildRecordKey(left.provider, left.model));
    const rightHasExisting = existingLookup.has(buildRecordKey(right.provider, right.model));
    if (leftHasExisting === rightHasExisting) {
      return 0;
    }
    return leftHasExisting ? 1 : -1;
  });
  const desiredKeys = new Set(entries.map((entry) => buildRecordKey(entry.provider, entry.model)));

  const staleCandidates = [...existingLookup.entries()]
    .filter(([key]) => !desiredKeys.has(key))
    .map(([key, existingRecord]) => {
      const lastSyncAt = normalizeNumber(existingRecord.fields["Last Sync At"]);
      const eligibleForDelete =
        staleDeleteAfterSeconds === 0 ||
        (typeof lastSyncAt === "number" &&
          Number.isFinite(lastSyncAt) &&
          nowSeconds - lastSyncAt >= staleDeleteAfterSeconds);
      const staleFields: Record<string, JsonValue> = {};
      if (fieldNameSet.has("Status")) staleFields.Status = "inactive";
      if (fieldNameSet.has("Hidden")) staleFields.Hidden = true;
      if (fieldNameSet.has("Is Available")) staleFields["Is Available"] = false;
      if (fieldNameSet.has("Last Error Message")) staleFields["Last Error Message"] = "Model missing from current Hermes registry snapshot";
      if (fieldNameSet.has("Failure Kind")) staleFields["Failure Kind"] = "not_in_snapshot";
      if (fieldNameSet.has("Hidden Reason")) staleFields["Hidden Reason"] = "missing_from_snapshot";
      return {
        key,
        existingRecord,
        eligibleForDelete,
        staleFields,
        needsStaleUpdate: !eligibleForDelete && shouldUpdateBitableRecord(existingRecord.fields, staleFields),
      };
    });

  const desiredCandidates = orderedEntries.map((entry) => {
    const key = buildRecordKey(entry.provider, entry.model);
    const fields = buildFieldsForSchema(entry, fieldNameSet, nowSeconds);
    const existingRecord = existingLookup.get(key) ?? null;
    return {
      key,
      entry,
      fields,
      existingRecord,
      needsUpdate: existingRecord ? shouldUpdateBitableRecord(existingRecord.fields, fields) : false,
    };
  });

  const staleDeleteCandidateCount = staleCandidates.filter((candidate) => candidate.eligibleForDelete).length;
  const staleHideCandidateCount = staleCandidates.filter((candidate) => candidate.needsStaleUpdate).length;
  const entryCreateCandidateCount = desiredCandidates.filter((candidate) => candidate.existingRecord === null).length;
  const entryUpdateCandidateCount = desiredCandidates.filter((candidate) => candidate.needsUpdate).length;
  const totalMutationCandidates =
    blankRecordIds.length +
    staleDeleteCandidateCount +
    staleHideCandidateCount +
    entryCreateCandidateCount +
    entryUpdateCandidateCount;

  for (const recordId of blankRecordIds) {
    if (mutationsApplied >= mutationBudget) {
      pending = true;
      break;
    }
    await deleteBitableRecord(env, {
      appToken,
      tableId: schema.tableId,
      recordId,
    });
    deleted += 1;
    mutationsApplied += 1;
  }

  if (!pending) {
    for (const candidate of staleCandidates) {
      if (candidate.eligibleForDelete) {
        if (mutationsApplied >= mutationBudget) {
          pending = true;
          break;
        }
        await deleteBitableRecord(env, {
          appToken,
          tableId: schema.tableId,
          recordId: candidate.existingRecord.recordId,
        });
        deleted += 1;
        mutationsApplied += 1;
        continue;
      }
      if (!candidate.needsStaleUpdate || Object.keys(candidate.staleFields).length === 0) {
        continue;
      }
      if (mutationsApplied >= mutationBudget) {
        pending = true;
        break;
      }
      await updateBitableRecord(env, {
        appToken,
        tableId: schema.tableId,
        recordId: candidate.existingRecord.recordId,
        fields: candidate.staleFields,
      });
      hidden += 1;
      mutationsApplied += 1;
    }
  }

  for (const candidate of desiredCandidates) {
    if (pending) {
      continue;
    }
    if (candidate.existingRecord) {
      if (!candidate.needsUpdate) {
        continue;
      }
      if (mutationsApplied >= mutationBudget) {
        pending = true;
        continue;
      }
      await updateBitableRecord(env, {
        appToken,
        tableId: schema.tableId,
        recordId: candidate.existingRecord.recordId,
        fields: candidate.fields,
      });
      updated += 1;
      mutationsApplied += 1;
      continue;
    }
    if (mutationsApplied >= mutationBudget) {
      pending = true;
      continue;
    }
    await createBitableRecord(env, {
      appToken,
      tableId: schema.tableId,
      fields: candidate.fields,
    });
    created += 1;
    mutationsApplied += 1;
  }

  pending = pending || totalMutationCandidates > mutationsApplied;

  return {
    mirrored: true,
    appToken,
    tableId: schema.tableId,
    tableName: schema.tableName,
    rowCount: entries.length,
    existingRowCount: existingRecords.length,
    blankRowCount: blankRecordIds.length,
    historicalOnlyCount: staleCandidates.length,
    staleDeleteCandidateCount,
    staleHideCandidateCount,
    entryCreateCandidateCount,
    entryUpdateCandidateCount,
    created,
    updated,
    hidden,
    deleted,
    createdTable: schema.createdTable,
    createdFields: schema.createdFields,
    createdViews: schema.createdViews,
    missingOptionalFields: MODEL_REGISTRY_FIELD_SPECS.map((item) => item.name).filter((name) => !fieldNameSet.has(name)),
    pending,
    remainingEstimate: Math.max(0, totalMutationCandidates - mutationsApplied),
    mutationBudget,
    mutationsApplied,
  };
}
