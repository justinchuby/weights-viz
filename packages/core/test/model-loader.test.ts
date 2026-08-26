import { describe, expect, it } from "vitest";
import { loadSources, MemorySource, normalizeModelUrl } from "../src";

function safeTensor(name: string): MemorySource {
  const header = new TextEncoder().encode(
    JSON.stringify({
      weight: { dtype: "F32", shape: [1], data_offsets: [0, 4] }
    })
  );
  const bytes = new Uint8Array(8 + header.byteLength + 4);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  bytes.set(header, 8);
  new DataView(bytes.buffer).setFloat32(8 + header.byteLength, 1, true);
  return new MemorySource(name, bytes);
}

describe("loadSources", () => {
  it("joins shards named by a SafeTensors index", async () => {
    const index = new MemorySource(
      "model.safetensors.index.json",
      new TextEncoder().encode(
        JSON.stringify({
          metadata: { total_size: 8 },
          weight_map: {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors"
          }
        })
      )
    );
    const models = await loadSources([
      index,
      safeTensor("model-00001-of-00002.safetensors"),
      safeTensor("model-00002-of-00002.safetensors")
    ]);

    expect(models).toHaveLength(1);
    expect(models[0]?.files).toHaveLength(2);
    expect(models[0]?.diagnostics).toEqual([]);
  });

  describe("normalizeModelUrl", () => {
    it("turns Hugging Face blob pages into ranged download URLs", () => {
      expect(
        normalizeModelUrl(
          "https://huggingface.co/org/model/blob/main/weights/model.Q4_K_M.gguf?download=true"
        )
      ).toBe(
        "https://huggingface.co/org/model/resolve/main/weights/model.Q4_K_M.gguf"
      );
    });

    it("preserves Hugging Face resolve URLs", () => {
      expect(
        normalizeModelUrl(
          "https://huggingface.co/org/model/resolve/main/model.safetensors"
        )
      ).toBe(
        "https://huggingface.co/org/model/resolve/main/model.safetensors"
      );
    });

    it("rejects repository pages that do not identify a file", () => {
      expect(() =>
        normalizeModelUrl("https://huggingface.co/org/model")
      ).toThrow(/repository page/);
    });
  });

  it("reports a missing shard without hiding available shards", async () => {
    const index = new MemorySource(
      "model.safetensors.index.json",
      new TextEncoder().encode(
        JSON.stringify({
          weight_map: {
            present: "present.safetensors",
            missing: "missing.safetensors"
          }
        })
      )
    );
    const models = await loadSources([index, safeTensor("present.safetensors")]);

    expect(models[0]?.files).toHaveLength(1);
    expect(models[0]?.diagnostics[0]?.message).toContain("missing.safetensors");
  });
});
