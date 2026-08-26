import { describe, expect, it } from "vitest";

import { MemorySource } from "../src/data-source";
import { ParseError } from "../src/errors";
import { OnnxParser } from "../src/parsers/onnx";

interface Encoded {
  bytes: number[];
  marks: Record<string, number>;
}

interface EncodedBytes {
  bytes: Uint8Array;
  marks: Record<string, number>;
}

describe("OnnxParser", () => {
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
      graphName: string;
      initializers: Array<{
        dataLocation: number;
        dataLocationName: string;
        dataType: number;
        dataTypeName: string;
        dims: bigint[];
        rawData: { present: boolean; offset?: bigint; length: bigint };
      }>;
      irVersion: bigint;
      modelVersion: bigint;
      producerName: string;
      producerVersion: string;
      domain: string;
      docString: string;
    };
    expect(metadata.graphName).toBe("main");
    expect(metadata.irVersion).toBe(9n);
    expect(metadata.modelVersion).toBe(42n);
    expect(metadata.producerName).toBe("synthetic");
    expect(metadata.producerVersion).toBe("1.2.3");
    expect(metadata.domain).toBe("ai.weights-viz");
    expect(metadata.docString).toBe("model documentation");
    expect(metadata.initializers[0]).toMatchObject({
      dataLocation: 0,
      dataLocationName: "DEFAULT",
      dataType: 1,
      dataTypeName: "FLOAT",
      dims: [2n, 3n],
      rawData: {
        present: true,
        offset: BigInt(rawDataOffset),
        length: 6n
      }
    });
  });

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
        externalData: {
          basepath?: string;
          checksum?: string;
          length?: bigint;
          location?: string;
          offset?: bigint;
        };
      }>;
    };
    expect(metadata.initializers[0]?.externalData).toEqual({
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
        { id: 26, name: "INT2" }
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
        "INT2"
      ]);
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
