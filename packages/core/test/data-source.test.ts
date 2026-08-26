import { describe, expect, it, vi } from "vitest";
import {
  fetchRemoteOnnx,
  HttpRangeSource,
  ParseError
} from "../src";

function rangeResponse(
  bytes: Uint8Array,
  range: string,
  total = bytes.byteLength
): Response {
  return new Response(bytes.slice().buffer, {
    status: 206,
    headers: { "content-range": `${range}/${total}` }
  });
}

describe("HttpRangeSource", () => {
  it("probes the length and validates ranged reads", async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(rangeResponse(new Uint8Array([0]), "bytes 0-0", 10))
      .mockResolvedValueOnce(
        rangeResponse(new Uint8Array([2, 3, 4]), "bytes 2-4", 10)
      );
    const source = await HttpRangeSource.create("https://example.test/model.gguf", {
      fetch: fetcher
    });

    expect(source.size).toBe(10n);
    expect(await source.read(2n, 3)).toEqual(new Uint8Array([2, 3, 4]));
    expect(fetcher).toHaveBeenLastCalledWith(
      "https://example.test/model.gguf",
      expect.objectContaining({ headers: { Range: "bytes=2-4" } })
    );
  });

  it("rejects servers that ignore Range", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response("all"));
    await expect(
      HttpRangeSource.create("https://example.test/model.gguf", { fetch: fetcher })
    ).rejects.toThrow(/must support CORS HTTP Range/);
  });

  it("rejects inconsistent Content-Range responses", async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(rangeResponse(new Uint8Array([0]), "bytes 0-0", 10))
      .mockResolvedValueOnce(
        rangeResponse(new Uint8Array([2, 3]), "bytes 1-2", 10)
      );
    const source = await HttpRangeSource.create("https://example.test/model.gguf", {
      fetch: fetcher
    });

    await expect(source.read(2n, 2)).rejects.toThrow(/Unexpected Content-Range/);
  });

  it("explains gated Hugging Face files", async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response("denied", { status: 403 }));
    await expect(
      HttpRangeSource.create(
        "https://huggingface.co/org/private/resolve/main/model.gguf",
        { fetch: fetcher }
      )
    ).rejects.toThrow(/Gated and private/);
  });
});

describe("fetchRemoteOnnx", () => {
  it("enforces the declared size limit before reading", async () => {
    const response = new Response(new Uint8Array(5).buffer, {
      headers: { "content-length": "100" }
    });
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(response);
    await expect(
      fetchRemoteOnnx("https://example.test/model.onnx", 50, undefined, fetcher)
    ).rejects.toBeInstanceOf(ParseError);
  });

  it("enforces the streamed size limit", async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(new Uint8Array(51).buffer));
    await expect(
      fetchRemoteOnnx("https://example.test/model.onnx", 50, undefined, fetcher)
    ).rejects.toThrow(/exceeds/);
  });
});
