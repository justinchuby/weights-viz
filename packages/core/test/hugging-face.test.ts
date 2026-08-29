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
});
