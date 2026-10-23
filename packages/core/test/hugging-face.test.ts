import { describe, expect, it, vi } from "vitest";
import {
  HuggingFaceHubSource,
  huggingFaceFileUrl,
  listHuggingFaceModelFiles,
  parseHuggingFaceFileUrl,
  parseHuggingFaceRepository
} from "../src";

describe("Hugging Face integration", () => {
  it("parses model repository identifiers and URLs", () => {
    expect(parseHuggingFaceRepository("unsloth/Qwen3.8-27B-GGUF")).toBe(
      "unsloth/Qwen3.8-27B-GGUF"
    );
    expect(
      parseHuggingFaceRepository(
        "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main"
      )
    ).toBe("unsloth/Qwen3.8-27B-GGUF");
    expect(() => parseHuggingFaceRepository("not-a-repository")).toThrow(
      /owner\/repo/
    );
  });

  it("parses file URLs for the Hub SDK", () => {
    expect(
      parseHuggingFaceFileUrl(
        "https://huggingface.co/org/model/resolve/main/subdir/model.gguf"
      )
    ).toEqual({
      repo: "org/model",
      revision: "main",
      path: "subdir/model.gguf"
    });
  });

  it("reads exact ranges from the Hub blob abstraction", async () => {
    const source = new HuggingFaceHubSource(
      "https://huggingface.co/org/model/resolve/main/model.gguf",
      new Blob([new Uint8Array([0, 1, 2, 3, 4])]),
      "model.gguf"
    );

    expect(source.size).toBe(5n);
    expect(await source.read(1n, 3)).toEqual(new Uint8Array([1, 2, 3]));
  });

  it("lists only supported model files", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify([
          { type: "file", path: "model.gguf", size: 100 },
          { type: "file", path: "model.safetensors.index.json", size: 20 },
          { type: "file", path: "notes.md", size: 10 },
          { type: "directory", path: "subdir", size: 0 }
        ]),
        { headers: { "content-type": "application/json" } }
      )
    );

    await expect(
      listHuggingFaceModelFiles("org/model", undefined, fetcher)
    ).resolves.toEqual([
      {
        path: "model.gguf",
        size: 100,
        format: "gguf",
        url: "https://huggingface.co/org/model/resolve/main/model.gguf"
      },
      {
        path: "model.safetensors.index.json",
        size: 20,
        format: "safetensors",
        url: "https://huggingface.co/org/model/resolve/main/model.safetensors.index.json"
      }
    ]);
    expect(fetcher).toHaveBeenCalledWith(
      "https://huggingface.co/api/models/org/model/tree/main?recursive=true&expand=false",
      expect.objectContaining({
        headers: expect.objectContaining({ accept: "application/json" })
      })
    );
  });

  it("preserves a repository URL revision when listing files", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify([{ type: "file", path: "model.gguf", size: 1 }]))
    );

    const files = await listHuggingFaceModelFiles(
      "https://huggingface.co/org/model/tree/dev",
      undefined,
      fetcher
    );

    expect(files[0]?.url).toBe(
      "https://huggingface.co/org/model/resolve/dev/model.gguf"
    );
    expect(fetcher).toHaveBeenCalledWith(
      "https://huggingface.co/api/models/org/model/tree/dev?recursive=true&expand=false",
      expect.anything()
    );
  });

  it("builds encoded resolve URLs", () => {
    expect(huggingFaceFileUrl("org/model", "quantized/model v1.gguf")).toBe(
      "https://huggingface.co/org/model/resolve/main/quantized/model%20v1.gguf"
    );
  });

  it("round-trips revisions with slashes and literal percent signs in filenames", () => {
    const url = huggingFaceFileUrl("org/model", "权重/model 100%.safetensors", "refs/pr/42");
    expect(parseHuggingFaceFileUrl(url)).toEqual({
      repo: "org/model",
      path: "权重/model 100%.safetensors",
      revision: "refs/pr/42"
    });
    expect(new HuggingFaceHubSource(url, new Blob(["data"]), "权重/model 100%.safetensors").name).toBe("model 100%.safetensors");
  });

  it("passes encoded branch revisions through the actual SDK listing", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify([{ type: "file", path: "权重/model 100%.gguf", size: 5 }]))
    );
    const files = await listHuggingFaceModelFiles(
      "https://huggingface.co/org/model/tree/refs%2Fpr%2F42", undefined, fetcher
    );
    expect(fetcher).toHaveBeenCalledWith(
      "https://huggingface.co/api/models/org/model/tree/refs%2Fpr%2F42?recursive=true&expand=false",
      expect.anything()
    );
    expect(parseHuggingFaceFileUrl(files[0]!.url)).toMatchObject({
      revision: "refs/pr/42", path: "权重/model 100%.gguf"
    });
  });

  it("opens encoded paths with the SDK and preserves slice ranges and abort signals", async () => {
    const url = huggingFaceFileUrl("org/model", "sub dir/model 100%.gguf", "refs/pr/42");
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(new Uint8Array([0]), {
        status: 206,
        headers: { "content-range": "bytes 0-0/5", etag: '"test"' }
      }))
      .mockResolvedValueOnce(new Response(new Uint8Array([1, 2, 3]), { status: 206 }));
    const controller = new AbortController();
    const source = await HuggingFaceHubSource.create(url, { fetch: fetcher, signal: controller.signal });
    expect(source.name).toBe("model 100%.gguf");
    expect(source.size).toBe(5n);
    expect(await source.read(1n, 3)).toEqual(new Uint8Array([1, 2, 3]));
    expect(fetcher).toHaveBeenNthCalledWith(1, url, expect.objectContaining({
      signal: controller.signal,
      headers: expect.objectContaining({ Range: "bytes=0-0" })
    }));
    expect(String(fetcher.mock.calls[1]?.[0])).toBe(url);
    expect(fetcher.mock.calls[1]?.[1]).toMatchObject({
      signal: controller.signal, headers: { Range: "bytes=1-3" }
    });
    await expect(source.read(4n, 2)).rejects.toThrow();
    controller.abort();
    await expect(source.read(0n, 1, controller.signal)).rejects.toThrow(/abort/i);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("does not accept a short SDK response as a complete slice", async () => {
    const source = new HuggingFaceHubSource(
      "https://huggingface.co/org/model/resolve/main/model.gguf",
      new class extends Blob {
        override slice() { return new Blob([new Uint8Array([0])]); }
      }([new Uint8Array(5)])
    );
    await expect(source.read(0n, 3)).rejects.toThrow("Expected 3 bytes");
  });

  it("can retry an Xet slice after a transient token-refresh failure", async () => {
    const refreshUrl = "https://huggingface.co/api/models/org/retry-test/xet-read-token/main";
    const reconstructionUrl = "https://cas.example.test/v2/reconstructions/retry-test";
    const chunkUrl = "https://cas.example.test/retry-test.xorb";
    let refreshes = 0;
    const fetcher = vi.fn<typeof fetch>(async (input) => {
      const url = String(input);
      if (url.includes("/resolve/")) {
        return new Response("{}", { headers: {
          "content-type": "application/vnd.xet-fileinfo+json",
          "x-linked-size": "5",
          "x-linked-etag": '"retry-test"',
          "x-xet-hash": "retry-test",
          link: `<${refreshUrl}>; rel="xet-auth", <${reconstructionUrl}>; rel="xet-reconstruction-info"`
        } });
      }
      if (url === refreshUrl) {
        refreshes += 1;
        return refreshes === 1
          ? new Response("temporary failure", { status: 503 })
          : Response.json({ accessToken: "test-only-token", casUrl: "https://cas.example.test", exp: Math.floor(Date.now() / 1000) + 3600 });
      }
      if (url === reconstructionUrl) {
        return Response.json({
          terms: [{ hash: "retry-test-xorb", unpacked_length: 5, range: { start: 0, end: 1 } }],
          xorbs: { "retry-test-xorb": [{
            url: chunkUrl,
            ranges: [{ chunks: { start: 0, end: 1 }, bytes: { start: 0, end: 12 } }]
          }] },
          offset_into_first_range: 1
        });
      }
      if (url === chunkUrl) {
        // Xet uncompressed chunk: eight-byte header followed by five data bytes.
        return new Response(new Uint8Array([0, 5, 0, 0, 0, 5, 0, 0, 0, 1, 2, 3, 4]), { status: 206 });
      }
      throw new Error(`Unexpected test request: ${url}`);
    });
    const source = await HuggingFaceHubSource.create(
      "https://huggingface.co/org/retry-test/resolve/main/model.safetensors",
      { fetch: fetcher }
    );
    await expect(source.read(1n, 3)).rejects.toThrow(/503/);
    await expect(source.read(1n, 3)).resolves.toEqual(new Uint8Array([1, 2, 3]));
    expect(refreshes).toBe(2);
  });
});
