import { describe, expect, it } from "vitest";

import { MemorySource } from "../src/data-source";
import { ParseError } from "../src/errors";
import { GgufParser } from "../src/parsers/gguf";

type MetadataValue =
  | { type: "uint32"; value: number }
  | { type: "bool"; value: boolean }
  | { type: "string"; value: string }
  | { type: "array"; elementType: "string"; value: string[] };

interface TensorSpec {
  name: string;
  shape: bigint[];
  typeId: number;
  relativeOffset: number;
  data: Uint8Array;
}

class ByteWriter {
  private readonly chunks: number[] = [];

  get length(): number {
    return this.chunks.length;
  }

  u8(value: number): void {
    this.chunks.push(value & 0xff);
  }

  u16(value: number, littleEndian: boolean): void {
    const buffer = new ArrayBuffer(2);
    new DataView(buffer).setUint16(0, value, littleEndian);
    this.bytes(new Uint8Array(buffer));
  }

  u32(value: number, littleEndian: boolean): void {
    const buffer = new ArrayBuffer(4);
    new DataView(buffer).setUint32(0, value, littleEndian);
    this.bytes(new Uint8Array(buffer));
  }

  u64(value: bigint, littleEndian: boolean): void {
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setBigUint64(0, value, littleEndian);
    this.bytes(new Uint8Array(buffer));
  }

  f32(value: number, littleEndian: boolean): void {
    const buffer = new ArrayBuffer(4);
    new DataView(buffer).setFloat32(0, value, littleEndian);
    this.bytes(new Uint8Array(buffer));
  }

  string(value: string, littleEndian: boolean): void {
    const bytes = new TextEncoder().encode(value);
    this.u64(BigInt(bytes.length), littleEndian);
    this.bytes(bytes);
  }

  bytes(value: Uint8Array): void {
    for (const byte of value) this.chunks.push(byte);
  }

  toUint8Array(): Uint8Array {
    return Uint8Array.from(this.chunks);
  }
}

class TrackingSource extends MemorySource {
  readonly reads: Array<{ offset: bigint; length: number }> = [];

  override async read(offset: bigint, length: number, signal?: AbortSignal): Promise<Uint8Array> {
    this.reads.push({ offset, length });
    signal?.throwIfAborted();
    return super.read(offset, length);
  }
}

const VALUE_TYPES = {
  uint32: 4,
  bool: 7,
  string: 8,
  array: 9
} as const;

function encodeFloat16(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Number.POSITIVE_INFINITY) return 0x7c00;
  if (value === Number.NEGATIVE_INFINITY) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const absolute = Math.abs(value);
  if (absolute === 0) return sign;
  const exponent = Math.floor(Math.log2(absolute));
  if (exponent < -14) {
    const fraction = Math.round((absolute / 2 ** -14) * 1024);
    return sign | Math.min(fraction, 0x03ff);
  }
  if (exponent > 15) {
    return sign | 0x7c00;
  }
  const base = absolute / 2 ** exponent;
  const fraction = Math.round((base - 1) * 1024);
  if (fraction === 1024) {
    return sign | ((exponent + 16) << 10);
  }
  return sign | ((exponent + 15) << 10) | (fraction & 0x03ff);
}

function writeHalf(writer: ByteWriter, value: number, littleEndian: boolean): void {
  writer.u16(encodeFloat16(value), littleEndian);
}

function encodeF32(values: number[], littleEndian: boolean): Uint8Array {
  const writer = new ByteWriter();
  for (const value of values) writer.f32(value, littleEndian);
  return writer.toUint8Array();
}

function encodeF16(values: number[], littleEndian: boolean): Uint8Array {
  const writer = new ByteWriter();
  for (const value of values) writeHalf(writer, value, littleEndian);
  return writer.toUint8Array();
}

function encodeQ4_0(scale: number, firstHalf: number[]): Uint8Array {
  const writer = new ByteWriter();
  writeHalf(writer, scale, true);
  for (let index = 0; index < 16; index += 1) {
    const low = firstHalf[index] ?? 8;
    const high = 8;
    writer.u8(((high & 0x0f) << 4) | (low & 0x0f));
  }
  return writer.toUint8Array();
}

function encodeQ4_1(scale: number, minimum: number, firstHalf: number[]): Uint8Array {
  const writer = new ByteWriter();
  writeHalf(writer, scale, true);
  writeHalf(writer, minimum, true);
  for (let index = 0; index < 16; index += 1) {
    const low = firstHalf[index] ?? 0;
    const high = 0;
    writer.u8(((high & 0x0f) << 4) | (low & 0x0f));
  }
  return writer.toUint8Array();
}

function encodeQ5(
  scale: number,
  minimum: number,
  firstHalf: number[],
  highBitIndexes: number[],
  withMinimum: boolean
): Uint8Array {
  const writer = new ByteWriter();
  writeHalf(writer, scale, true);
  if (withMinimum) writeHalf(writer, minimum, true);
  const highBits = new Uint8Array(4);
  for (const index of highBitIndexes) {
    highBits[Math.floor(index / 8)] = (highBits[Math.floor(index / 8)] ?? 0) | (1 << (index % 8));
  }
  writer.bytes(highBits);
  for (let index = 0; index < 16; index += 1) {
    const low = firstHalf[index] ?? 0;
    const high = 0;
    writer.u8(((high & 0x0f) << 4) | (low & 0x0f));
  }
  return writer.toUint8Array();
}

function encodeQ8_0(scale: number, values: number[]): Uint8Array {
  const writer = new ByteWriter();
  writeHalf(writer, scale, true);
  for (let index = 0; index < 32; index += 1) {
    writer.u8((values[index] ?? 0) & 0xff);
  }
  return writer.toUint8Array();
}

function writeMetadataValue(
  writer: ByteWriter,
  value: MetadataValue,
  littleEndian: boolean,
  nested = false
): void {
  if (!nested) writer.u32(VALUE_TYPES[value.type], littleEndian);

  switch (value.type) {
    case "uint32":
      writer.u32(value.value, littleEndian);
      break;
    case "bool":
      writer.u8(value.value ? 1 : 0);
      break;
    case "string":
      writer.string(value.value, littleEndian);
      break;
    case "array":
      writer.u32(VALUE_TYPES[value.elementType], littleEndian);
      writer.u64(BigInt(value.value.length), littleEndian);
      for (const item of value.value) {
        writeMetadataValue(writer, { type: value.elementType, value: item }, littleEndian, true);
      }
      break;
  }
}

function buildGguf(options: {
  littleEndian: boolean;
  version: 2 | 3;
  metadata: Array<[string, MetadataValue]>;
  tensors: TensorSpec[];
}): { bytes: Uint8Array; dataRegionStart: number } {
  const writer = new ByteWriter();
  writer.u32(0x46554747, options.littleEndian);
  writer.u32(options.version, options.littleEndian);
  writer.u64(BigInt(options.tensors.length), options.littleEndian);
  writer.u64(BigInt(options.metadata.length), options.littleEndian);

  for (const [key, value] of options.metadata) {
    writer.string(key, options.littleEndian);
    writeMetadataValue(writer, value, options.littleEndian);
  }

  for (const tensor of options.tensors) {
    writer.string(tensor.name, options.littleEndian);
    writer.u32(tensor.shape.length, options.littleEndian);
    for (const dimension of tensor.shape) writer.u64(dimension, options.littleEndian);
    writer.u32(tensor.typeId, options.littleEndian);
    writer.u64(BigInt(tensor.relativeOffset), options.littleEndian);
  }

  const alignmentEntry = options.metadata.find(([key]) => key === "general.alignment")?.[1];
  const alignment = alignmentEntry?.type === "uint32" ? alignmentEntry.value : 32;
  const dataRegionStart = Math.ceil(writer.length / alignment) * alignment;
  const dataEnd = Math.max(0, ...options.tensors.map((tensor) => tensor.relativeOffset + tensor.data.length));
  const bytes = new Uint8Array(dataRegionStart + dataEnd);
  bytes.set(writer.toUint8Array(), 0);
  for (const tensor of options.tensors) {
    bytes.set(tensor.data, dataRegionStart + tensor.relativeOffset);
  }
  return { bytes, dataRegionStart };
}

function getTensor(parsed: Awaited<ReturnType<GgufParser["parse"]>>, name: string) {
  const tensor = parsed.tensors.find((candidate) => candidate.name === name);
  if (!tensor) throw new Error(`Missing tensor ${name}`);
  return tensor;
}

function expectCloseArray(actual: number[], expected: number[]): void {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, index) => {
    expect(value).toBeCloseTo(expected[index]!, 6);
  });
}

describe("GgufParser", () => {
  it("parses little-endian metadata progressively and derives aligned tensor ranges", async () => {
    const template = "x".repeat(300);
    const { bytes, dataRegionStart } = buildGguf({
      littleEndian: true,
      version: 3,
      metadata: [
        ["general.alignment", { type: "uint32", value: 64 }],
        ["general.name", { type: "string", value: "synthetic" }],
        ["tokenizer.ggml.tokens", { type: "array", elementType: "string", value: ["<s>", "</s>"] }],
        ["general.description", { type: "string", value: template }],
        ["general.has_encoder", { type: "bool", value: true }]
      ],
      tensors: [
        {
          name: "tok_embeddings.weight",
          shape: [2n, 2n],
          typeId: 0,
          relativeOffset: 64,
          data: encodeF32([1, 2, 3, 4], true)
        },
        {
          name: "output.weight",
          shape: [32n],
          typeId: 8,
          relativeOffset: 128,
          data: encodeQ8_0(0.25, Array.from({ length: 32 }, (_, index) => index - 4))
        },
        {
          name: "unsupported.weight",
          shape: [1n],
          typeId: 99,
          relativeOffset: 768,
          data: new Uint8Array([0xde, 0xad, 0xbe, 0xef])
        }
      ]
    });
    const source = new TrackingSource("synthetic.gguf", bytes);
    const parser = new GgufParser();

    const parsed = await parser.parse(source);

    expect(parsed.metadata["general.name"]).toBe("synthetic");
    expect(parsed.metadata["general.has_encoder"]).toBe(true);
    expect(parsed.metadata["tokenizer.ggml.tokens"]).toEqual(["<s>", "</s>"]);
    expect(source.reads.length).toBeGreaterThan(1);
    expect(source.reads.every((read) => read.offset === 0n)).toBe(true);
    expect(source.reads.every((read, index) => index === 0 || source.reads[index - 1]!.length < read.length)).toBe(true);
    expect(source.reads.at(-1)!.length).toBeLessThan(bytes.length);

    const embeddings = getTensor(parsed, "tok_embeddings.weight");
    const output = getTensor(parsed, "output.weight");
    const unsupported = getTensor(parsed, "unsupported.weight");

    expect(embeddings.dtype).toBe("F32");
    expect(embeddings.byteOffset).toBe(BigInt(dataRegionStart + 64));
    expect(embeddings.dataOffset).toBe(BigInt(dataRegionStart + 64));
    expect(embeddings.byteLength).toBe(16n);
    expect(output.dtype).toBe("Q8_0");
    expect(output.byteOffset).toBe(BigInt(dataRegionStart + 128));
    expect(output.dataOffset).toBe(BigInt(dataRegionStart + 128));
    expect(output.byteLength).toBe(34n);
    expect(unsupported.dtype).toBe("GGML_TYPE_99 (unsupported)");
    expect(unsupported.sampleSupport).toBe("unsupported");
    expect(unsupported.dataOffset).toBe(BigInt(dataRegionStart + 768));
    expect(unsupported.byteLength).toBe(4n);
    expect(parsed.diagnostics).toContainEqual(
      expect.objectContaining({ message: expect.stringContaining("unsupported GGML type id 99") })
    );
  });

  it("samples F32 and common quantized GGML tensor blocks without reading full files", async () => {
    const { bytes } = buildGguf({
      littleEndian: true,
      version: 3,
      metadata: [["general.alignment", { type: "uint32", value: 32 }]],
      tensors: [
        { name: "f32", shape: [4n], typeId: 0, relativeOffset: 0, data: encodeF32([1, 2, 3, 4], true) },
        {
          name: "q4_0",
          shape: [32n],
          typeId: 2,
          relativeOffset: 32,
          data: encodeQ4_0(0.5, [8, 9, 10, 11, 12, 13])
        },
        {
          name: "q4_1",
          shape: [32n],
          typeId: 3,
          relativeOffset: 64,
          data: encodeQ4_1(0.5, 1, [0, 1, 2, 3, 4, 5])
        },
        {
          name: "q5_0",
          shape: [32n],
          typeId: 6,
          relativeOffset: 96,
          data: encodeQ5(0.5, 0, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], false)
        },
        {
          name: "q5_1",
          shape: [32n],
          typeId: 7,
          relativeOffset: 128,
          data: encodeQ5(0.5, -1, [0, 1, 2, 3, 4, 5], [], true)
        },
        {
          name: "q8_0",
          shape: [32n],
          typeId: 8,
          relativeOffset: 160,
          data: encodeQ8_0(0.25, Array.from({ length: 32 }, (_, index) => index - 4))
        }
      ]
    });
    const source = new TrackingSource("sample.gguf", bytes);
    const parser = new GgufParser();
    const parsed = await parser.parse(source);

    const clonedF32 = structuredClone(getTensor(parsed, "f32"));
    expectCloseArray((await new GgufParser().sample(source, clonedF32, 4)).values, [1, 2, 3, 4]);
    expectCloseArray((await parser.sample(source, getTensor(parsed, "q4_0"), 6)).values, [0, 0.5, 1, 1.5, 2, 2.5]);
    expectCloseArray((await parser.sample(source, getTensor(parsed, "q4_1"), 6)).values, [1, 1.5, 2, 2.5, 3, 3.5]);
    expectCloseArray((await parser.sample(source, getTensor(parsed, "q5_0"), 6)).values, [0, 0.5, 1, 1.5, 2, 2.5]);
    expectCloseArray((await parser.sample(source, getTensor(parsed, "q5_1"), 6)).values, [-1, -0.5, 0, 0.5, 1, 1.5]);
    const q8 = await parser.sample(source, getTensor(parsed, "q8_0"), 6);
    expectCloseArray(q8.values, [-1, -0.75, -0.5, -0.25, 0, 0.25]);
    expect(q8.sampledElements).toBe(6);
    expect(q8.totalElements).toBe(32n);
    expect(source.reads.at(-1)!.length).toBeLessThan(bytes.length);
  });

  it("parses swapped big-endian GGUF v2 files and samples F16 tensors", async () => {
    const { bytes, dataRegionStart } = buildGguf({
      littleEndian: false,
      version: 2,
      metadata: [],
      tensors: [
        {
          name: "f16",
          shape: [3n],
          typeId: 1,
          relativeOffset: 0,
          data: encodeF16([1.5, -2, 0.25], false)
        }
      ]
    });
    const source = new MemorySource("be.gguf", bytes);
    const parser = new GgufParser();

    const parsed = await parser.parse(source);
    const tensor = getTensor(parsed, "f16");
    const sample = await parser.sample(source, tensor, 3);

    expect(tensor.dataOffset).toBe(BigInt(dataRegionStart));
    expectCloseArray(sample.values, [1.5, -2, 0.25]);
  });

  it("enforces the configured metadata prefix cap", async () => {
    const { bytes } = buildGguf({
      littleEndian: true,
      version: 3,
      metadata: [["general.description", { type: "string", value: "y".repeat(400) }]],
      tensors: []
    });
    const source = new MemorySource("too-large.gguf", bytes);
    const parser = new GgufParser();

    await expect(parser.parse(source, { maxMetadataBytes: 128 })).rejects.toThrowError(ParseError);
    await expect(parser.parse(source, { maxMetadataBytes: 128 })).rejects.toThrow(
      /GGUF metadata string length 400 exceeds limit 128/
    );
  });
});
