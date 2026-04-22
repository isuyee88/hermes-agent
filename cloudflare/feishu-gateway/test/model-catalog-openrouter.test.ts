import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchOpenRouterModels, mapOpenRouterModelsToSyncBundles } from "../src/model-catalog/openrouter";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("model catalog openrouter sync mapping", () => {
  it("filters non-zero-cost models at fetch time", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          data: [
            {
              id: "openrouter/free-model",
              pricing: {
                prompt: "0",
                completion: "0",
              },
            },
            {
              id: "openrouter/paid-input",
              pricing: {
                prompt: "0.000001",
                completion: "0",
              },
            },
            {
              id: "openrouter/paid-output",
              pricing: {
                prompt: "0",
                completion: "0.000001",
              },
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    vi.stubGlobal("fetch", fetchMock);

    const models = await fetchOpenRouterModels();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(models.map((item) => item.id)).toEqual(["openrouter/free-model"]);
  });

  it("maps OpenRouter models into catalog and policy bundles", () => {
    const bundles = mapOpenRouterModelsToSyncBundles(
      [
        {
          id: "openrouter/elephant-alpha",
          canonical_slug: "openrouter/elephant-alpha",
          name: "Elephant",
          created: 1776052598,
          context_length: 262144,
          architecture: {
            modality: "text->text",
            input_modalities: ["text"],
            output_modalities: ["text"],
            tokenizer: "Other",
          },
          pricing: {
            prompt: "0",
            completion: "0",
            input_cache_read: "0.0000001",
            input_cache_write: "0.0000002",
          },
          top_provider: {
            context_length: 262144,
            max_completion_tokens: 32768,
          },
          supported_parameters: ["max_tokens", "temperature", "tools", "tool_choice", "response_format", "top_p"],
          default_parameters: {
            temperature: null,
            top_p: null,
          },
        },
        {
          id: "openrouter/paid-model",
          canonical_slug: "openrouter/paid-model",
          name: "Paid Model",
          created: 1776052599,
          architecture: {
            modality: "text->text",
            input_modalities: ["text"],
            output_modalities: ["text"],
          },
          pricing: {
            prompt: "0.000001",
            completion: "0",
          },
        },
        {
          id: "openrouter/not-really-free:free",
          canonical_slug: "openrouter/not-really-free:free",
          name: "Fake Free",
          created: 1776052600,
          architecture: {
            modality: "text->text",
            input_modalities: ["text"],
            output_modalities: ["text"],
          },
          pricing: {
            prompt: "0",
            completion: "0.000001",
          },
        },
      ],
      {
        HERMES_CF_TEXT_PLAIN_ROUTE_NAME: "text-general",
        HERMES_CF_TEXT_CODING_ROUTE_NAME: "text-coding",
      },
      1_776_111_111,
    );

    expect(bundles).toHaveLength(1);
    expect(bundles[0]?.catalog.provider).toBe("openrouter");
    expect(bundles[0]?.catalog.isFree).toBe(true);
    expect(bundles[0]?.catalog.contextWindow).toBe(262144);
    expect(bundles[0]?.catalog.inputPricePerMillion).toBe(0);
    expect(bundles[0]?.catalog.cacheWritePricePerMillion).toBe(0.2);
    expect(bundles[0]?.syncEntry.taskKinds).toContain("long_context");
    expect(bundles[0]?.syncEntry.toolCalling).toBe(true);
    expect(bundles[0]?.syncEntry.structuredOutput).toBe(true);
    expect(bundles[0]?.policies.some((item) => item.gatewayRouteName === "text-general")).toBe(true);
    expect(bundles.map((item) => item.catalog.model)).toEqual(["openrouter/elephant-alpha"]);
  });
});
