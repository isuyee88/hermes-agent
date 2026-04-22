import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/services/feishu/auth", () => ({
  getTenantAccessToken: vi.fn(async () => "tenant-token"),
}));

const { createBitableRecord, listBitableRecords } = await import("../src/services/feishu/bitable");

describe("feishu bitable client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("reads paginated record lists from nested data payloads", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            code: 0,
            msg: "success",
            data: {
              items: [{ record_id: "rec-1" }],
              has_more: true,
              page_token: "next-page",
            },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            code: 0,
            msg: "success",
            data: {
              items: [{ record_id: "rec-2" }],
              has_more: false,
            },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      );

    vi.stubGlobal("fetch", fetchMock);

    const records = await listBitableRecords(
      {
        FEISHU_API_BASE: "https://open.feishu.cn",
        FEISHU_APP_ID: "app-id",
        FEISHU_APP_SECRET: "app-secret",
      } as never,
      "app-token",
      "table-id",
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(records.map((item) => item.record_id)).toEqual(["rec-1", "rec-2"]);
    expect(String(fetchMock.mock.calls[1]?.[0])).toContain("page_token=next-page");
  });

  it("unwraps created records from nested data payloads", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          code: 0,
          msg: "success",
          data: {
            record: {
              record_id: "rec-created",
              fields: {
                Model: "openai/gpt-4.1-mini",
              },
            },
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    vi.stubGlobal("fetch", fetchMock);

    const record = await createBitableRecord(
      {
        FEISHU_API_BASE: "https://open.feishu.cn",
        FEISHU_APP_ID: "app-id",
        FEISHU_APP_SECRET: "app-secret",
      } as never,
      {
        appToken: "app-token",
        tableId: "table-id",
        fields: {
          Model: "openai/gpt-4.1-mini",
        },
      },
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(record.record_id).toBe("rec-created");
    expect(record.fields).toEqual({
      Model: "openai/gpt-4.1-mini",
    });
  });
});
