import { describe, expect, it } from "vitest";
import {
  loadSources,
  MemorySource,
  normalizeModelUrl,
  validateOnnxExternalLocation
} from "../src";

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

  it("maps ONNX external initializers into the referenced data file", async () => {
    const manifest = externalOnnx("model.onnx.data", 16n, 16n);
    const data = new MemorySource("model.onnx.data", new Uint8Array(64));
    const models = await loadSources([manifest, data]);

    expect(models).toHaveLength(1);
    expect(models[0]?.files).toHaveLength(1);
    expect(models[0]?.files[0]).toMatchObject({
      id: data.id,
      name: "model.onnx.data",
      size: 64n
    });
    expect(models[0]?.files[0]?.tensors[0]).toMatchObject({
      name: "weight",
      fileId: data.id,
      byteOffset: 16n,
      byteLength: 16n,
      storage: "external"
    });
    expect(models[0]?.diagnostics).toEqual([]);
  });

  it("reports a missing ONNX external data file", async () => {
    const models = await loadSources([
      externalOnnx("model.onnx.data", 0n, 16n)
    ]);

    expect(models[0]?.files).toEqual([]);
    expect(models[0]?.diagnostics[0]?.message).toContain(
      "Missing ONNX external data file"
    );
  });

  it("rejects unsafe ONNX external locations", () => {
    expect(() => validateOnnxExternalLocation("../weights.data")).toThrow(
      /Unsafe/
    );
    expect(() => validateOnnxExternalLocation("https://example.com/data")).toThrow(
      /relative/
    );
    expect(validateOnnxExternalLocation("shards/model.onnx.data")).toBe(
      "shards/model.onnx.data"
    );
  });
});

function externalOnnx(
  location: string,
  offset: bigint,
  length: bigint
): MemorySource {
  const tensor = protoMessage(
    5,
    concatBytes(
      protoVarint(1, 4n),
      protoVarint(2, 1n),
      protoBytes(8, new TextEncoder().encode("weight")),
      protoMessage(
        13,
        concatBytes(
          protoBytes(1, new TextEncoder().encode("location")),
          protoBytes(2, new TextEncoder().encode(location))
        )
      ),
      protoMessage(
        13,
        concatBytes(
          protoBytes(1, new TextEncoder().encode("offset")),
          protoBytes(2, new TextEncoder().encode(offset.toString()))
        )
      ),
      protoMessage(
        13,
        concatBytes(
          protoBytes(1, new TextEncoder().encode("length")),
          protoBytes(2, new TextEncoder().encode(length.toString()))
        )
      ),
      protoVarint(14, 1n)
    )
  );
  return new MemorySource(
    "model.onnx",
    Uint8Array.from(protoMessage(7, tensor))
  );
}

function protoVarint(field: number, value: bigint): number[] {
  return [...encodeVarint(BigInt(field << 3)), ...encodeVarint(value)];
}

function protoBytes(field: number, value: Uint8Array): number[] {
  return [
    ...encodeVarint(BigInt((field << 3) | 2)),
    ...encodeVarint(BigInt(value.byteLength)),
    ...value
  ];
}

function protoMessage(field: number, value: number[]): number[] {
  return [
    ...encodeVarint(BigInt((field << 3) | 2)),
    ...encodeVarint(BigInt(value.length)),
    ...value
  ];
}

function concatBytes(...parts: number[][]): number[] {
  return parts.flat();
}

function encodeVarint(value: bigint): number[] {
  const bytes: number[] = [];
  let remaining = value;
  while (remaining >= 0x80n) {
    bytes.push(Number((remaining & 0x7fn) | 0x80n));
    remaining >>= 7n;
  }
  bytes.push(Number(remaining));
  return bytes;
}
