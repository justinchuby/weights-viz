import { describe, expect, it } from "vitest";

import { MemorySource } from "../src/data-source";
import { SafeTensorsParser } from "../src/parsers/safetensors";

const parser = new SafeTensorsParser();

describe("SafeTensorsParser", () => {
  it("parses metadata, absolute offsets, and only reads the header", async () => {
    const file = createSafeTensorsFile(
      [
        {
          name: "weights",
          dtype: "F32",
          shape: [2, 2],
          bytes: encodeF32([1, 2, 3, 4])
        },
        {
          name: "flags",
          dtype: "BOOL",
          shape: [3],
          bytes: Uint8Array.from([1, 0, 1])
        }
      ],
      { format: "synthetic", author: "tests" }
    );
    const source = new TrackingSource("model.safetensors", file.bytes);

    const parsed = await parser.parse(source);
    const weights = parsed.tensors[0]!;
    const flags = parsed.tensors[1]!;

    expect(parsed.metadata).toEqual({ format: "synthetic", author: "tests" });
    expect(parsed.diagnostics).toEqual([]);
    expect(parsed.tensors).toHaveLength(2);
    expect(weights).toMatchObject({
      name: "weights",
      dtype: "F32",
      shape: [2n, 2n],
      byteOffset: 8n + BigInt(file.headerLength),
      byteLength: 16n,
      dataOffset: 8n + BigInt(file.headerLength),
      storage: "inline",
      sampleSupport: "values"
    });
    expect(flags).toMatchObject({
      name: "flags",
      dtype: "BOOL",
      shape: [3n],
      byteOffset: 8n + BigInt(file.headerLength) + 16n,
      byteLength: 3n,
      dataOffset: 8n + BigInt(file.headerLength) + 16n,
      storage: "inline",
      sampleSupport: "values"
    });
    expect(source.reads).toEqual([
      { offset: 0n, length: 8 },
      { offset: 8n, length: file.headerLength }
    ]);
  });

  it("samples supported scalar dtypes with correct stats", async () => {
    for (const testCase of scalarDtypeCases) {
      const file = createSafeTensorsFile([
        {
          name: "tensor",
          dtype: testCase.dtype,
          shape: [testCase.expected.length],
          bytes: testCase.bytes
        }
      ]);
      const source = new MemorySource(`${testCase.dtype}.safetensors`, file.bytes);
      const parsed = await parser.parse(source);
      const tensor = parsed.tensors[0]!;

      const sample = await parser.sample(source, tensor, 32);

      expect(sample.values, testCase.dtype).toEqual(testCase.expected);
      expect(sample.sampledElements, testCase.dtype).toBe(testCase.expected.length);
      expect(sample.totalElements, testCase.dtype).toBe(BigInt(testCase.expected.length));
      expect(sample.min, testCase.dtype).toBe(Math.min(...testCase.expected));
      expect(sample.max, testCase.dtype).toBe(Math.max(...testCase.expected));
      expect(sample.mean, testCase.dtype).toBe(
        testCase.expected.reduce((sum, value) => sum + value, 0) / testCase.expected.length
      );
    }
  });

  it("samples evenly distributed values without reading the whole tensor", async () => {
    const values = Array.from({ length: 100 }, (_, index) => index);
    const file = createSafeTensorsFile([
      {
        name: "large",
        dtype: "F32",
        shape: [values.length],
        bytes: encodeF32(values)
      }
    ]);
    const source = new TrackingSource("large.safetensors", file.bytes);
    const parsed = await parser.parse(source);
    const tensor = parsed.tensors[0]!;
    source.reads.length = 0;

    const sample = await parser.sample(source, tensor, 5);

    expect(sample.values).toEqual([0, 24, 49, 74, 99]);
    expect(sample.sampledElements).toBe(5);
    expect(sample.totalElements).toBe(100n);
    expect(source.reads).toEqual([
      { offset: tensor.byteOffset, length: 4 },
      { offset: tensor.byteOffset + 24n * 4n, length: 4 },
      { offset: tensor.byteOffset + 49n * 4n, length: 4 },
      { offset: tensor.byteOffset + 74n * 4n, length: 4 },
      { offset: tensor.byteOffset + 99n * 4n, length: 4 }
    ]);
  });

  it("recognizes every official dtype and validates packed bit lengths", async () => {
    const dtypeCases = [
      ["BOOL", 8, 1],
      ["F4", 4, 2],
      ["F6_E2M3", 6, 4],
      ["F6_E3M2", 6, 4],
      ["U8", 8, 1],
      ["I8", 8, 1],
      ["F8_E5M2", 8, 1],
      ["F8_E4M3", 8, 1],
      ["F8_E8M0", 8, 1],
      ["F8_E4M3FNUZ", 8, 1],
      ["F8_E5M2FNUZ", 8, 1],
      ["I16", 16, 1],
      ["U16", 16, 1],
      ["F16", 16, 1],
      ["BF16", 16, 1],
      ["I32", 32, 1],
      ["U32", 32, 1],
      ["F32", 32, 1],
      ["C64", 64, 1],
      ["F64", 64, 1],
      ["I64", 64, 1],
      ["U64", 64, 1]
    ] as const;
    const file = createSafeTensorsFile(
      dtypeCases.map(([dtype, bits, elements]) => ({
        name: dtype,
        dtype,
        shape: [elements],
        bytes: new Uint8Array((bits * elements) / 8)
      }))
    );

    const parsed = await parser.parse(
      new MemorySource("all-dtypes.safetensors", file.bytes)
    );

    expect(parsed.diagnostics).toEqual([]);
    for (const [dtype, bits, elements] of dtypeCases) {
      const tensor = parsed.tensors.find((candidate) => candidate.name === dtype);
      expect(tensor?.dtype).toBe(dtype);
      expect(tensor?.byteLength).toBe(BigInt((bits * elements) / 8));
    }
  });

  it("rejects non-byte-aligned packed tensors", async () => {
    const file = createSafeTensorsFile([
      {
        name: "packed",
        dtype: "F6_E2M3",
        shape: [1],
        bytes: new Uint8Array(1)
      }
    ]);

    await expect(
      parser.parse(new MemorySource("misaligned.safetensors", file.bytes))
    ).rejects.toThrow(/require 6 bits, which is not byte-aligned/);
  });

  it("diagnoses unknown dtype identifiers without failing metadata parsing", async () => {
    const file = createSafeTensorsFile([
      {
        name: "unknown",
        dtype: "X99",
        shape: [4],
        bytes: Uint8Array.from([1, 2, 3, 4])
      }
    ]);

    const parsed = await parser.parse(
      new MemorySource("unknown.safetensors", file.bytes)
    );
    const tensor = parsed.tensors[0]!;

    expect(tensor.sampleSupport).toBe("unsupported");
    expect(parsed.diagnostics).toEqual([
      {
        severity: "warning",
        message: "Tensor unknown uses unknown SafeTensors dtype X99",
        offset: tensor.byteOffset
      }
    ]);
  });

  it("rejects oversized headers and out-of-bounds tensors with useful errors", async () => {
    const valid = createSafeTensorsFile([
      {
        name: "weights",
        dtype: "F32",
        shape: [2],
        bytes: encodeF32([1, 2])
      }
    ]);
    const cappedSource = new MemorySource("capped.safetensors", valid.bytes);

    await expect(parser.parse(cappedSource, { maxMetadataBytes: valid.headerLength - 1 })).rejects
      .toMatchObject({
        name: "ParseError",
        message: `SafeTensors header length ${valid.headerLength} exceeds maxMetadataBytes ${valid.headerLength - 1}`,
        offset: 0n
      });

    const invalid = createSafeTensorsFile(
      [
        {
          name: "bad",
          dtype: "F32",
          shape: [2],
          bytes: encodeF32([1, 2])
        }
      ],
      undefined,
      {
        bad: {
          dtype: "F32",
          shape: [2],
          data_offsets: [0, 12]
        }
      }
    );

    await expect(
      parser.parse(new MemorySource("invalid.safetensors", invalid.bytes))
    ).rejects.toThrow(/SafeTensors tensor bad range 0\.\.12 exceeds file size 8/);
  });
});

class TrackingSource extends MemorySource {
  readonly reads: Array<{ offset: bigint; length: number }> = [];

  override async read(offset: bigint, length: number, signal?: AbortSignal): Promise<Uint8Array> {
    this.reads.push({ offset, length });
    signal?.throwIfAborted();
    return super.read(offset, length);
  }
}

function createSafeTensorsFile(
  tensors: Array<{
    name: string;
    dtype: string;
    shape: number[];
    bytes: Uint8Array;
  }>,
  metadata?: Record<string, unknown>,
  overrides?: Record<string, unknown>
): { bytes: Uint8Array; headerLength: number } {
  const header: Record<string, unknown> = {};
  if (metadata) {
    header.__metadata__ = metadata;
  }

  const data: Uint8Array[] = [];
  let relativeOffset = 0;
  for (const tensor of tensors) {
    const start = relativeOffset;
    relativeOffset += tensor.bytes.byteLength;
    header[tensor.name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [start, relativeOffset]
    };
    data.push(tensor.bytes);
  }

  if (overrides) {
    Object.assign(header, overrides);
  }

  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const file = new Uint8Array(
    8 + headerBytes.byteLength + data.reduce((total, chunk) => total + chunk.byteLength, 0)
  );
  const view = new DataView(file.buffer);
  view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
  file.set(headerBytes, 8);

  let offset = 8 + headerBytes.byteLength;
  for (const chunk of data) {
    file.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return { bytes: file, headerLength: headerBytes.byteLength };
}

function encodeF64(values: number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 8);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat64(index * 8, value, true));
  return bytes;
}

function encodeF32(values: number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return bytes;
}

function encodeU16(values: number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 2);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setUint16(index * 2, value, true));
  return bytes;
}

function encodeI64(values: bigint[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 8);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setBigInt64(index * 8, value, true));
  return bytes;
}

function encodeU64(values: bigint[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 8);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setBigUint64(index * 8, value, true));
  return bytes;
}

function encodeI32(values: number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setInt32(index * 4, value, true));
  return bytes;
}

function encodeU32(values: number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setUint32(index * 4, value, true));
  return bytes;
}

function encodeI16(values: number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 2);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setInt16(index * 2, value, true));
  return bytes;
}

function encodeI8(values: number[]): Uint8Array {
  return Uint8Array.from(values.map((value) => value & 0xff));
}

const scalarDtypeCases: Array<{
  dtype: string;
  expected: number[];
  bytes: Uint8Array;
}> = [
  {
    dtype: "F64",
    expected: [1.5, -2.25, 7.75],
    bytes: encodeF64([1.5, -2.25, 7.75])
  },
  {
    dtype: "F32",
    expected: [1.25, -2.5, 4.75],
    bytes: encodeF32([1.25, -2.5, 4.75])
  },
  {
    dtype: "F16",
    expected: [1, -2, 0.5, 3.5],
    bytes: encodeU16([0x3c00, 0xc000, 0x3800, 0x4300])
  },
  {
    dtype: "BF16",
    expected: [1, -2, 0.5, 3.5],
    bytes: encodeU16([0x3f80, 0xc000, 0x3f00, 0x4060])
  },
  {
    dtype: "I64",
    expected: [-3, 0, 11],
    bytes: encodeI64([-3n, 0n, 11n])
  },
  {
    dtype: "U64",
    expected: [0, 9, 12],
    bytes: encodeU64([0n, 9n, 12n])
  },
  {
    dtype: "I32",
    expected: [-11, 7, 3],
    bytes: encodeI32([-11, 7, 3])
  },
  {
    dtype: "U32",
    expected: [0, 7, 15],
    bytes: encodeU32([0, 7, 15])
  },
  {
    dtype: "I16",
    expected: [-5, 2, 12],
    bytes: encodeI16([-5, 2, 12])
  },
  {
    dtype: "U16",
    expected: [1, 5, 9],
    bytes: encodeU16([1, 5, 9])
  },
  {
    dtype: "I8",
    expected: [-8, 0, 9],
    bytes: encodeI8([-8, 0, 9])
  },
  {
    dtype: "U8",
    expected: [1, 4, 9],
    bytes: Uint8Array.from([1, 4, 9])
  },
  {
    dtype: "BOOL",
    expected: [0, 1, 1, 0],
    bytes: Uint8Array.from([0, 1, 1, 0])
  }
];
