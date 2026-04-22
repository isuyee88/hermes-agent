import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchNvidiaModels, mapNvidiaModelsToSyncBundles } from "../src/model-catalog/nvidia";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("model catalog nvidia sync mapping", () => {
  it("deduplicates model ids at fetch time", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          data: [
            { id: "meta/llama-3.1-405b-instruct", owned_by: "meta" },
            { id: "meta/llama-3.1-405b-instruct", owned_by: "meta" },
            { id: "nvidia/llama-nemotron-embed-1b-v2", owned_by: "nvidia" },
            { id: "   ", owned_by: "nvidia" },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    vi.stubGlobal("fetch", fetchMock);

    const models = await fetchNvidiaModels();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(models.map((item) => item.id)).toEqual([
      "meta/llama-3.1-405b-instruct",
      "nvidia/llama-nemotron-embed-1b-v2",
    ]);
  });

  it("maps NVIDIA models into catalog and policy bundles", () => {
    const bundles = mapNvidiaModelsToSyncBundles(
      [
        {
          id: "meta/llama-3.2-90b-vision-instruct",
          created: 1_776_052_598,
          owned_by: "meta",
        },
        {
          id: "nvidia/llama-nemotron-embed-1b-v2",
          created: 1_776_052_599,
          owned_by: "nvidia",
        },
        {
          id: "qwen/qwen3-coder-480b-a35b-instruct",
          created: 1_776_052_600,
          owned_by: "qwen",
        },
      ],
      {
        HERMES_CF_TEXT_PLAIN_ROUTE_NAME: "text-general",
        HERMES_CF_TEXT_CODING_ROUTE_NAME: "text-coding",
      },
      1_776_111_111,
    );

    expect(bundles).toHaveLength(3);
    expect(bundles.every((bundle) => bundle.catalog.provider === "nvidia")).toBe(true);

    const codingBundle = bundles.find((bundle) => bundle.catalog.model === "qwen/qwen3-coder-480b-a35b-instruct");
    expect(codingBundle?.catalog.functionType).toBe("coding");
    expect(codingBundle?.syncEntry.taskKinds).toContain("coding");
    expect(codingBundle?.policies.some((policy) => policy.gatewayRouteName === "text-coding")).toBe(true);

    const embeddingBundle = bundles.find((bundle) => bundle.catalog.model === "nvidia/llama-nemotron-embed-1b-v2");
    expect(embeddingBundle?.catalog.functionType).toBe("embedding");
    expect(embeddingBundle?.syncEntry.taskKinds).toContain("embedding");
    expect(embeddingBundle?.policies.find((policy) => policy.taskKind === "embedding")?.enabled).toBe(false);

    const visionBundle = bundles.find((bundle) => bundle.catalog.model === "meta/llama-3.2-90b-vision-instruct");
    expect(visionBundle?.catalog.vision).toBe(true);
    expect(visionBundle?.syncEntry.taskKinds).toContain("vision");
    expect(visionBundle?.syncEntry.modalities).toEqual(["text", "image"]);
  });
});
