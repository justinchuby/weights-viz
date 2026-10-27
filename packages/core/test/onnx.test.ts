import { describe, expect, it } from "vitest";

import { MemorySource } from "../src/data-source";
import { ParseError } from "../src/errors";
import { OnnxParser } from "../src/parsers/onnx";
import type { RandomAccessSource } from "../src/types";

interface Encoded {
  bytes: number[];
  marks: Record<string, number>;
}

interface EncodedBytes {
  bytes: Uint8Array;
  marks: Record<string, number>;
}

describe("OnnxParser", () => {
  it.each([4, 5, 6, 7, 9, 10, 11])(
    "skips large field %s payloads without counting weights as metadata",
    async (field) => {
      const { source, reads, dataOffset, payloadLength } = virtualTensor(field);
      const parsed = await new OnnxParser().parse(source, { maxMetadataBytes: 128 });
      expect(parsed.tensors[0]).toMatchObject({
        name: "large",
        byteOffset: BigInt(dataOffset),
        byteLength: BigInt(payloadLength),
        byteSegments: [{ byteOffset: BigInt(dataOffset), byteLength: BigInt(payloadLength) }]
      });
      expect(reads.every((read) => read.length <= 64 * 1024)).toBe(true);
      expect(reads.reduce((sum, read) => sum + read.length, 0)).toBeLessThan(128 * 1024);
    }
  );

  it("preserves offsets beyond 4 GiB without allocating the tensor payload", async () => {
    const { source, dataOffset, payloadLength } = virtualTensor(9, [7, 5], 2 ** 32 + 128);
    const parsed = await new OnnxParser().parse(source, { maxMetadataBytes: 128 });
    expect(parsed.tensors[0]?.byteOffset).toBe(BigInt(dataOffset));
    expect(parsed.tensors[0]?.byteLength).toBe(BigInt(payloadLength));
    expect(parsed.tensors[0]?.name).toBe("large");
  });

  it.each([
    [7, 1, 5, 5], // Graph -> Node -> Attribute -> Tensor
    [7, 1, 5, 10], // Repeated tensor attributes
    [7, 1, 5, 6, 5], // Subgraph initializer
    [7, 1, 5, 11, 5], // Repeated subgraph attributes
    [7, 15, 1], // Sparse initializer values
    [7, 15, 2], // Sparse initializer indices
    [7, 1, 5, 22, 1], // Sparse tensor attribute
    [7, 1, 5, 23, 2], // Repeated sparse tensor attributes
    [25, 7, 5, 5], // Function node
    [25, 11, 5], // Function default attribute
    [20, 1, 5], // Training initialization graph
    [20, 2, 5] // Training algorithm graph
  ])("finds large tensors in nested path %j", async (...path) => {
    const { source, dataOffset, payloadLength } = virtualTensor(9, path);
    const parsed = await new OnnxParser().parse(source, { maxMetadataBytes: 256 });
    expect(parsed.tensors).toHaveLength(1);
    expect(parsed.tensors[0]).toMatchObject({
      name: "large", byteOffset: BigInt(dataOffset), byteLength: BigInt(payloadLength)
    });
  });

  it("keeps a metadata budget independent of file size", async () => {
    const bytes = toUint8Array(stringField(2, "a".repeat(256))).bytes;
    await expect(new OnnxParser().parse(new MemorySource("metadata.onnx", bytes), {
      maxMetadataBytes: 128
    })).rejects.toThrow(/metadata exceeds/);
    const unknown = toUint8Array(bytesField(99, new Uint8Array(256))).bytes;
    await expect(new OnnxParser().parse(new MemorySource("unknown.onnx", unknown), {
      maxMetadataBytes: 128
    })).rejects.toThrow(/metadata exceeds/);
  });

  it("counts metadata cumulatively across separate fields", async () => {
    const bytes = toUint8Array(concat(
      stringField(2, "a".repeat(80)), stringField(3, "b".repeat(80))
    )).bytes;
    await expect(new OnnxParser().parse(new MemorySource("metadata.onnx", bytes), {
      maxMetadataBytes: 128
    })).rejects.toThrow(/metadata exceeds/);
  });

  it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])("rejects invalid metadata limit %s", async (limit) => {
    await expect(new OnnxParser().parse(new MemorySource("empty.onnx", new Uint8Array()), {
      maxMetadataBytes: limit
    })).rejects.toThrow(/Invalid ONNX metadata byte limit/);
  });

  it("honors cancellation before and during reads", async () => {
    const controller = new AbortController();
    controller.abort();
    const { source } = virtualTensor(9);
    await expect(new OnnxParser().parse(source, { signal: controller.signal })).rejects.toThrow(/abort/i);
    await expect(new OnnxParser().parse(new MemorySource("empty.onnx", new Uint8Array()), {
      signal: controller.signal
    })).rejects.toThrow(/abort/i);
    const during = new AbortController();
    const abortingSource: RandomAccessSource = {
      ...source,
      async read(offset, length) {
        const bytes = await source.read(offset, length);
        during.abort();
        return bytes;
      }
    };
    await expect(new OnnxParser().parse(abortingSource, { signal: during.signal })).rejects.toThrow(/abort/i);
  });

  it("rejects short random-access reads", async () => {
    const { source } = virtualTensor(9);
    await expect(new OnnxParser().parse({
      ...source, read: async () => new Uint8Array()
    })).rejects.toThrow(/Truncated ONNX source read/);
  });

  it("bounds nested graph recursion", async () => {
    const path = [7, ...Array.from({ length: 12 }, () => [1, 5, 6]).flat(), 5];
    await expect(new OnnxParser().parse(virtualTensor(9, path).source)).rejects.toThrow(/nesting exceeds/);
  });

  it("uses distinct physical IDs for tensors in different subgraphs", async () => {
    const tensor = messageField(5, concat(
      varintField(1, 1n), varintField(2, 2n), stringField(8, "weight"), bytesField(9, [1])
    ));
    const nested = messageField(1, messageField(5, messageField(6, tensor)));
    const bytes = toUint8Array(messageField(7, concat(tensor, nested, nested))).bytes;
    const parsed = await new OnnxParser().parse(new MemorySource("nested.onnx", bytes));
    expect(parsed.tensors).toHaveLength(3);
    expect(new Set(parsed.tensors.map((tensor) => tensor.id)).size).toBe(3);
  });

  it.each([
    { field: 4, wire: 5, dtype: 1, values: [0, 0, 0x80, 0x3f] },
    { field: 10, wire: 1, dtype: 11, values: [0, 0, 0, 0, 0, 0, 0xf0, 0x3f] },
    { field: 5, wire: 0, dtype: 2, values: [0xff, 0x01] },
    { field: 7, wire: 0, dtype: 7, values: [0xff, 0x01] },
    { field: 11, wire: 0, dtype: 13, values: [0xff, 0x01] }
  ])("maps unpacked typed data field $field", async ({ field, wire, dtype, values }) => {
    const value = { bytes: [...encodeVarint((field << 3) | wire), ...values], marks: { value: 1 } };
    const bytes = toUint8Array(messageField(7, messageField(5, concat(
      varintField(1, 1n), varintField(2, BigInt(dtype)), value
    ))));
    const parsed = await new OnnxParser().parse(new MemorySource("unpacked.onnx", bytes.bytes));
    expect(parsed.tensors[0]?.byteSegments).toEqual([{
      byteOffset: BigInt(requireMark(bytes.marks, "value")), byteLength: BigInt(values.length)
    }]);
  });

  it.each([
    [0],
    [0x80],
    [...Array<number>(9).fill(0xff), 2],
    [...encodeVarint(1n << 32n), 0],
    [0x0b],
    [0x3a, 2, 0x2a, 127],
    [0x3a, 3, 0x2a, 1, 0x25],
    [0x3a, 5, 0x2a, 3, 0x22, 1, 0],
    [0x3a, 5, 0x2a, 3, 0x52, 1, 0]
  ])("rejects malformed protobuf bytes %j", async (...encoded) => {
    await expect(new OnnxParser().parse(new MemorySource(
      "bad.onnx", Uint8Array.from(encoded)
    ))).rejects.toBeInstanceOf(ParseError);
  });

  it.each(["-1", "abc", "1.5", ""])("rejects malformed external byte ranges %j", async (value) => {
    const bytes = toUint8Array(messageField(7, messageField(5,
      messageField(13, stringStringEntry("offset", value))
    ))).bytes;
    await expect(new OnnxParser().parse(new MemorySource("bad.onnx", bytes))).rejects.toThrow(/external data byte range/);
  });

  it("parses inline raw_data initializer metadata and file offsets", async () => {
    const rawData = [1, 2, 3, 4, 5, 6];
    const tensor = messageField(
      5,
      withMark(
        concat(
          packedVarintField(1, [2n, 3n]),
          varintField(2, 1n),
          stringField(8, "weight"),
          bytesField(9, rawData, "rawData")
        ),
        "tensorStart"
      )
    );
    const bytes = toUint8Array(
      concat(
        varintField(1, 9n),
        stringField(2, "synthetic"),
        stringField(3, "1.2.3"),
        stringField(4, "ai.weights-viz"),
        varintField(5, 42n),
        stringField(6, "model documentation"),
        messageField(7, concat(stringField(2, "main"), tensor))
      )
    );
    const rawDataOffset = requireMark(bytes.marks, "rawData");

    const parsed = await new OnnxParser().parse(new MemorySource("model.onnx", bytes.bytes));
    expect(parsed.format).toBe("onnx");
    expect(parsed.tensors).toHaveLength(1);

    const tensorRecord = parsed.tensors[0];
    expect(tensorRecord).toBeDefined();
    if (!tensorRecord) throw new Error("missing tensor");
    expect(tensorRecord).toMatchObject({
      name: "weight",
      dtype: "FLOAT",
      shape: [2n, 3n],
      sampleSupport: "metadata-only",
      storage: "inline",
      byteLength: 6n,
      dataOffset: BigInt(rawDataOffset)
    });
    expect(tensorRecord.byteOffset).toBe(BigInt(rawDataOffset));

    const metadata = parsed.metadata as {
      graph_name: string;
      initializers: Array<{
        data_location: number;
        data_location_name: string;
        data_type: number;
        data_type_name: string;
        dims: bigint[];
        raw_data: { present: boolean; offset?: bigint; length: bigint };
      }>;
      ir_version: bigint;
      model_version: bigint;
      producer_name: string;
      producer_version: string;
      domain: string;
      doc_string: string;
    };
    expect(metadata.graph_name).toBe("main");
    expect(metadata.ir_version).toBe(9n);
    expect(metadata.model_version).toBe(42n);
    expect(metadata.producer_name).toBe("synthetic");
    expect(metadata.producer_version).toBe("1.2.3");
    expect(metadata.domain).toBe("ai.weights-viz");
    expect(metadata.doc_string).toBe("model documentation");
    expect(metadata.initializers[0]).toMatchObject({
      data_location: 0,
      data_location_name: "DEFAULT",
      data_type: 1,
      data_type_name: "FLOAT",
      dims: [2n, 3n],
      raw_data: {
        present: true,
        offset: BigInt(rawDataOffset),
        length: 6n
      }
    });
  });

  function virtualTensor(field: number, path = [7, 5], payloadLength = 128 * 1024 * 1024): {
    source: RandomAccessSource;
    reads: Array<{ offset: number; length: number }>;
    dataOffset: number;
    payloadLength: number;
  } {
    const dtype = ({ 4: 1, 5: 2, 6: 8, 7: 7, 9: 2, 10: 11, 11: 13 } as Record<number, number>)[field]!;
    const elements = field === 4 ? payloadLength / 4 : field === 10 ? payloadLength / 8 : field === 6 ? 1 : payloadLength;
    const suffix = toUint8Array(concat(
      varintField(1, BigInt(elements)),
      varintField(2, BigInt(dtype)),
      stringField(8, "large")
    )).bytes;
    let prefix = [...encodeVarint((field << 3) | 2), ...encodeVarint(payloadLength)];
    for (const wrapper of [...path].reverse()) {
      const length = prefix.length + payloadLength + suffix.length;
      prefix = [...encodeVarint((wrapper << 3) | 2), ...encodeVarint(length), ...prefix];
    }
    const size = prefix.length + payloadLength + suffix.length;
    const reads: Array<{ offset: number; length: number }> = [];
    return {
      source: {
        id: "virtual", name: "large.onnx", size: BigInt(size),
        async read(offset, length) {
          const start = Number(offset);
          if (start < 0 || start + length > size) throw new Error("Read exceeds virtual source");
          reads.push({ offset: start, length });
          const result = new Uint8Array(length);
          for (let index = 0; index < length; index++) {
            const position = start + index;
            if (position < prefix.length) result[index] = prefix[position]!;
            else if (position >= prefix.length + payloadLength) {
              result[index] = suffix[position - prefix.length - payloadLength]!;
            }
          }
          return result;
        }
      },
      reads, dataOffset: prefix.length, payloadLength
    };
  }

  it("parses external_data metadata without reading external tensor bytes", async () => {
    const externalTensor = messageField(
      5,
      withMark(
        concat(
          varintField(1, 4n),
          varintField(2, 7n),
          stringField(8, "external_weight"),
          messageField(13, stringStringEntry("location", "weights.bin")),
          messageField(13, stringStringEntry("offset", "128")),
          messageField(13, stringStringEntry("length", "64")),
          messageField(13, stringStringEntry("checksum", "abc123")),
          messageField(13, stringStringEntry("basepath", "shards")),
          varintField(14, 1n)
        ),
        "externalTensor"
      )
    );
    const bytes = toUint8Array(concat(messageField(7, externalTensor)));
    const tensorOffset = requireMark(bytes.marks, "externalTensor");

    const parsed = await new OnnxParser().parse(new MemorySource("external.onnx", bytes.bytes));
    expect(parsed.tensors).toHaveLength(1);

    const tensorRecord = parsed.tensors[0];
    expect(tensorRecord).toBeDefined();
    if (!tensorRecord) throw new Error("missing tensor");
    expect(tensorRecord).toMatchObject({
      name: "external_weight",
      dtype: "INT64",
      shape: [4n],
      storage: "external",
      byteLength: 64n,
      externalLocation: "weights.bin",
      externalOffset: 128n,
      externalLength: 64n,
      sampleSupport: "metadata-only"
    });
    expect(tensorRecord.byteOffset).toBe(128n);

    const metadata = parsed.metadata as {
      initializers: Array<{
        external_data: {
          basepath?: string;
          checksum?: string;
          length?: bigint;
          location?: string;
          offset?: bigint;
        };
      }>;
    };
    expect(metadata.initializers[0]?.external_data).toEqual({
      location: "weights.bin",
      offset: 128n,
      length: 64n,
      checksum: "abc123",
      basepath: "shards"
    });
  });

  it("accounts for initializers stored in typed repeated fields", async () => {
    const firstValue = new Uint8Array(4);
    const secondValue = new Uint8Array(4);
    new DataView(firstValue.buffer).setFloat32(0, 1.5, true);
    new DataView(secondValue.buffer).setFloat32(0, -2, true);
    const tensor = messageField(
      5,
      concat(
        varintField(1, 2n),
        varintField(2, 1n),
        bytesField(4, firstValue, "floatData"),
        stringField(12, "separates repeated fields"),
        bytesField(4, secondValue, "floatData2"),
        stringField(8, "typed_weight")
      )
    );
    const bytes = toUint8Array(concat(messageField(7, tensor)));

    const parsed = await new OnnxParser().parse(new MemorySource("typed.onnx", bytes.bytes));
    expect(parsed.tensors[0]).toMatchObject({
      name: "typed_weight",
      dtype: "FLOAT",
      byteOffset: BigInt(requireMark(bytes.marks, "floatData")),
      byteLength: 8n,
      dataOffset: BigInt(requireMark(bytes.marks, "floatData")),
      byteSegments: [
        {
          byteOffset: BigInt(requireMark(bytes.marks, "floatData")),
          byteLength: 4n
        },
        {
          byteOffset: BigInt(requireMark(bytes.marks, "floatData2")),
          byteLength: 4n
        }
      ],
      storage: "inline",
      sampleSupport: "metadata-only"
    });
  });

  it("recognizes every current low-precision TensorProto data type", async () => {
      const tensors = [
        { id: 24, name: "FLOAT8E8M0" },
        { id: 25, name: "UINT2" },
        { id: 26, name: "INT2" },
        { id: 27, name: "FLOAT6E2M3" },
        { id: 28, name: "FLOAT6E3M2" }
      ].map(({ id, name }) =>
        messageField(
          5,
          concat(
            varintField(1, 1n),
            varintField(2, BigInt(id)),
            stringField(8, name),
            bytesField(9, [0])
          )
        )
      );
      const bytes = toUint8Array(
        concat(messageField(7, concat(...tensors)))
      );

      const parsed = await new OnnxParser().parse(
        new MemorySource("low-precision.onnx", bytes.bytes)
      );

      expect(parsed.tensors.map((tensor) => tensor.dtype)).toEqual([
        "FLOAT8E8M0",
        "UINT2",
        "INT2",
        "FLOAT6E2M3",
        "FLOAT6E3M2"
      ]);
  });

  it.each([27, 28])("preserves packed FP6 type %s payload offsets and partial groups", async (id) => {
    for (const [elements, payload] of [
      [0, []],
      [1, [0x01]],
      [2, [0x81, 0x00]],
      [3, [0x81, 0x30, 0x00]],
      [4, [0x81, 0x30, 0x10]],
      [5, [0x81, 0x30, 0x10, 0x05]]
    ] as const) {
      const bytes = toUint8Array(messageField(7, messageField(5, concat(
        varintField(1, BigInt(elements)),
        varintField(2, BigInt(id)),
        bytesField(9, payload, "fp6")
      ))));
      const parsed = await new OnnxParser().parse(new MemorySource("fp6.onnx", bytes.bytes));
      const offset = BigInt(requireMark(bytes.marks, "fp6"));
      expect(parsed.tensors[0]).toMatchObject({
        dtype: id === 27 ? "FLOAT6E2M3" : "FLOAT6E3M2",
        byteOffset: offset,
        byteLength: BigInt(payload.length),
        byteSegments: [{ byteOffset: offset, byteLength: BigInt(payload.length) }],
        encoding: { onnxDataField: "raw_data" },
        sampleSupport: "metadata-only"
      });
    }
  });

  it.each([27, 28])("maps packed and unpacked int32_data for FP6 type %s", async (id) => {
    const bytes = toUint8Array(messageField(7, messageField(5, concat(
      varintField(1, 4n),
      varintField(2, BigInt(id)),
      bytesField(5, [1, 2], "packedCodes"),
      stringField(8, "fp6_typed"),
      withMark(varintField(5, 3n), "unpackedCode"),
      bytesField(5, [63], "lastCode")
    ))));
    const parsed = await new OnnxParser().parse(new MemorySource("typed-fp6.onnx", bytes.bytes));
    expect(parsed.tensors[0]).toMatchObject({
      byteLength: 4n,
      encoding: { onnxDataField: "int32_data" },
      byteSegments: [
        { byteOffset: BigInt(requireMark(bytes.marks, "packedCodes")), byteLength: 2n },
        { byteOffset: BigInt(requireMark(bytes.marks, "unpackedCode") + 1), byteLength: 1n },
        { byteOffset: BigInt(requireMark(bytes.marks, "lastCode")), byteLength: 1n }
      ]
    });
  });

  it.each([
    { elements: 1, field: 9, payload: [0x40], error: /padding/ },
    { elements: 2, field: 9, payload: [0, 0x10], error: /padding/ },
    { elements: 3, field: 9, payload: [0, 0, 0x04], error: /padding/ },
    { elements: 4, field: 9, payload: [0, 0], error: /requires 3 bytes/ },
    { elements: 1, field: 5, payload: [64], error: /unsigned 6-bit/ },
    { elements: 1, field: 5, payload: [0xff, 0xff, 0xff, 0xff, 0x0f], error: /unsigned 6-bit/ },
    { elements: 2, field: 5, payload: [1], error: /requires 2 elements/ }
  ])("rejects non-canonical FP6 encodings: $field / $payload", async ({ elements, field, payload, error }) => {
    for (const id of [27, 28]) {
      const bytes = toUint8Array(messageField(7, messageField(5, concat(
        varintField(1, BigInt(elements)),
        varintField(2, BigInt(id)),
        bytesField(field, payload)
      ))));
      await expect(new OnnxParser().parse(new MemorySource("invalid-fp6.onnx", bytes.bytes))).rejects.toThrow(error);
    }
  });

  it("rejects malformed length-delimited protobuf fields", async () => {
    const malformed = Uint8Array.from([
      ...encodeVarint((7 << 3) | 2),
      4,
      ...encodeVarint((2 << 3) | 2),
      10,
      97
    ]);

    await expect(new OnnxParser().parse(new MemorySource("bad.onnx", malformed))).rejects.toBeInstanceOf(
      ParseError
    );
  });
});

function toUint8Array(encoded: Encoded): EncodedBytes {
  return { ...encoded, bytes: Uint8Array.from(encoded.bytes) };
}

function withMark(encoded: Encoded, name: string): Encoded {
  return {
    bytes: encoded.bytes,
    marks: { ...encoded.marks, [name]: 0 }
  };
}

function concat(...parts: Encoded[]): Encoded {
  const bytes: number[] = [];
  const marks: Record<string, number> = {};

  for (const part of parts) {
    const base = bytes.length;
    bytes.push(...part.bytes);
    for (const [name, offset] of Object.entries(part.marks)) {
      marks[name] = base + offset;
    }
  }

  return { bytes, marks };
}

function varintField(fieldNumber: number, value: bigint): Encoded {
  return {
    bytes: [...encodeVarint((fieldNumber << 3) | 0), ...encodeVarint(value)],
    marks: {}
  };
}

function packedVarintField(fieldNumber: number, values: bigint[]): Encoded {
  const payload = values.flatMap((value) => encodeVarint(value));
  return {
    bytes: [...encodeVarint((fieldNumber << 3) | 2), ...encodeVarint(payload.length), ...payload],
    marks: {}
  };
}

function stringField(fieldNumber: number, value: string): Encoded {
  return bytesField(fieldNumber, Array.from(new TextEncoder().encode(value)));
}

function bytesField(fieldNumber: number, value: ArrayLike<number>, markName?: string): Encoded {
  const payload = Array.from(value);
  const key = encodeVarint((fieldNumber << 3) | 2);
  const length = encodeVarint(payload.length);
  const marks = markName ? { [markName]: key.length + length.length } : {};
  return {
    bytes: [...key, ...length, ...payload],
    marks
  };
}

function messageField(fieldNumber: number, payload: Encoded): Encoded {
  const key = encodeVarint((fieldNumber << 3) | 2);
  const length = encodeVarint(payload.bytes.length);
  const marks: Record<string, number> = {};
  for (const [name, offset] of Object.entries(payload.marks)) {
    marks[name] = key.length + length.length + offset;
  }
  return {
    bytes: [...key, ...length, ...payload.bytes],
    marks
  };
}

function stringStringEntry(key: string, value: string): Encoded {
  return concat(stringField(1, key), stringField(2, value));
}

function encodeVarint(value: number | bigint): number[] {
  let remaining = typeof value === "bigint" ? value : BigInt(value);
  if (remaining < 0n) {
    remaining = BigInt.asUintN(64, remaining);
  }

  const bytes: number[] = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining !== 0n) byte |= 0x80;
    bytes.push(byte);
  } while (remaining !== 0n);
  return bytes;
}

function requireMark(marks: Record<string, number>, key: string): number {
  const value = marks[key];
  if (value === undefined) {
    throw new Error(`Missing mark ${key}`);
  }
  return value;
}
