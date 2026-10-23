import { afterEach, describe, expect, it, vi } from "vitest";
import {
  loadModelUrl,
  loadSources,
  MemorySource,
  normalizeModelUrl,
  parseGgufShardName,
  validateOnnxExternalLocation
} from "../src";

afterEach(() => vi.unstubAllGlobals());

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
  it("groups conventionally named GGUF shards in numeric order", async () => {
    const models = await loadSources([
      emptyGguf("model-00002-of-00002.gguf"),
      emptyGguf("model-00001-of-00002.gguf")
    ]);

    expect(models).toHaveLength(1);
    expect(models[0]?.name).toBe("model");
    expect(models[0]?.files.map((file) => file.name)).toEqual([
      "model-00001-of-00002.gguf",
      "model-00002-of-00002.gguf"
    ]);
    expect(models[0]?.diagnostics).toEqual([]);
  });

  it("reports missing GGUF shards while retaining available files", async () => {
    const models = await loadSources([
      emptyGguf("model-00001-of-00003.gguf"),
      emptyGguf("model-00003-of-00003.gguf")
    ]);

    expect(models[0]?.files).toHaveLength(2);
    expect(models[0]?.diagnostics).toContainEqual(
      expect.objectContaining({ message: expect.stringContaining("2 of 3") })
    );
  });

  it("keeps GGUF families with different declared totals separate", async () => {
    const models = await loadSources([
      emptyGguf("model-00001-of-00001.gguf"),
      emptyGguf("model-00001-of-00002.gguf"),
      emptyGguf("model-00002-of-00002.gguf")
    ]);

    expect(models).toHaveLength(2);
    expect(models.map((model) => model.files.length).sort()).toEqual([1, 2]);
  });

  it("bounds malicious local GGUF shard counts", async () => {
    const models = await loadSources([
      emptyGguf("model-00001-of-4294967296.gguf")
    ]);

    expect(models[0]?.files).toHaveLength(1);
    expect(models[0]?.diagnostics).toContainEqual(
      expect.objectContaining({ message: expect.stringContaining("1024 shard limit") })
    );
  });

  it("parses only valid GGUF shard names", () => {
    expect(parseGgufShardName("model-00002-of-00010.gguf")).toEqual({
      prefix: "model",
      index: 2,
      total: 10,
      width: 5
    });
    expect(parseGgufShardName("model.gguf")).toBeUndefined();
    expect(parseGgufShardName("model-00011-of-00010.gguf")).toBeUndefined();
  });

  it("joins shards named by a SafeTensors index", async () => {
    const index = new MemorySource(
      "model.safetensors.index.json",
      new TextEncoder().encode(
        JSON.stringify({
          metadata: { total_size: 8 },
          weight_map: {
            "layer.0": "model-00002-of-00002.safetensors",
            "layer.1": "model-00001-of-00002.safetensors"
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
    expect(models[0]?.files.map((file) => file.name)).toEqual([
      "model-00001-of-00002.safetensors",
      "model-00002-of-00002.safetensors"
    ]);
    expect(models[0]?.diagnostics).toEqual([]);
  });

  it("loads Hugging Face shards through the SDK after streaming a sizeless index", async () => {
    const shard = safeTensor("shard.safetensors");
    const bytes = await shard.read(0n, Number(shard.size));
    const fetcher = vi.fn<typeof fetch>(async (input, init) => {
      const url = String(input);
      if (url.endsWith(".index.json")) {
        return Response.json({
          weight_map: { first: "shard-1.safetensors", second: "shard-2.safetensors" }
        });
      }
      if (!/\/shard-[12]\.safetensors$/.test(url)) {
        throw new Error(`Unexpected test request: ${url}`);
      }
      const range = new Headers(init?.headers).get("range")?.match(/^bytes=(\d+)-(\d+)$/);
      if (!range) throw new Error("SDK request must use a byte range");
      const start = Number(range[1]);
      const end = Number(range[2]);
      return new Response(bytes.slice(start, end + 1), {
        status: 206,
        headers: { "content-range": `bytes ${start}-${end}/${bytes.length}`, etag: '"test-shard"' }
      });
    });
    vi.stubGlobal("fetch", fetcher);
    const models = await loadModelUrl(
      "https://huggingface.co/org/sharded/resolve/main/model.safetensors.index.json"
    );
    expect(models[0]?.name).toBe("org/sharded");
    expect(models[0]?.files.map((file) => file.name)).toEqual([
      "shard-1.safetensors", "shard-2.safetensors"
    ]);
    expect(models[0]?.files.every((file) => file.tensors.length === 1)).toBe(true);
    expect(models[0]?.diagnostics).toEqual([]);
  });

  function emptyGguf(name: string): MemorySource {
    const bytes = new Uint8Array(24);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x46554747, true);
    view.setUint32(4, 3, true);
    view.setBigUint64(8, 0n, true);
    view.setBigUint64(16, 0n, true);
    return new MemorySource(name, bytes);
  }

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

    it("streams Hugging Face SafeTensors indexes without requiring size headers", async () => {
      const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
        new Response(JSON.stringify({ weight_map: {} }))
      );
      vi.stubGlobal("fetch", fetcher);

      const models = await loadModelUrl(
        "https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/model.safetensors.index.json"
      );

      expect(models).toHaveLength(1);
      expect(models[0]?.name).toBe("zai-org/GLM-5.3-Flash");
      expect(fetcher).toHaveBeenCalledWith(
        "https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/model.safetensors.index.json",
        expect.objectContaining({
          cache: "no-store",
          credentials: "omit",
          mode: "cors"
        })
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

  it("infers ONNX external layouts without opening the data file", async () => {
    const models = await loadSources([
      externalOnnx("model.onnx.data", 0n, 16n)
    ]);

    expect(models[0]?.files).toHaveLength(1);
    expect(models[0]?.files[0]).toMatchObject({
      name: "model.onnx.data",
      size: 16n
    });
    expect(models[0]?.files[0]?.tensors[0]).toMatchObject({
      name: "weight",
      byteOffset: 0n,
      byteLength: 16n,
      storage: "external"
    });
    expect(models[0]?.diagnostics).toEqual([]);
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

  it.each([27, 28])("infers packed external FP6 type %s ranges, including a partial byte", async (dtype) => {
    const models = await loadSources([
      externalOnnx("fp6.data", 16n, undefined, dtype, 5n)
    ]);
    expect(models[0]?.files[0]).toMatchObject({ name: "fp6.data", size: 20n });
    expect(models[0]?.files[0]?.tensors[0]).toMatchObject({
      dtype: dtype === 27 ? "FLOAT6E2M3" : "FLOAT6E3M2",
      byteOffset: 16n,
      byteLength: 4n,
      encoding: { onnxDataField: "external_data" },
      sampleSupport: "metadata-only"
    });
    expect(models[0]?.diagnostics).toEqual([]);
    const explicit = await loadSources([
      externalOnnx("fp6.data", 16n, 7n, dtype, 5n)
    ]);
    expect(explicit[0]?.files[0]?.tensors[0]?.byteLength).toBe(7n);
  });
});

function externalOnnx(
  location: string,
  offset: bigint,
  length: bigint | undefined,
  dtype = 1,
  elements = 4n
): MemorySource {
  const tensor = protoMessage(
    5,
    concatBytes(
      protoVarint(1, elements),
      protoVarint(2, BigInt(dtype)),
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
      ...(length === undefined ? [] : [protoMessage(
        13,
        concatBytes(
          protoBytes(1, new TextEncoder().encode("length")),
          protoBytes(2, new TextEncoder().encode(length.toString()))
        )
      )]),
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
