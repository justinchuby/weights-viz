import {
  BinaryReader,
  bigintToSafeNumber,
  product,
  sampleStats
} from "../binary";
import { assertRange, ParseError } from "../errors";
import {
  DEFAULT_MAX_METADATA_BYTES,
  type DtypeCatalogEntry,
  type Diagnostic,
  type ParseOptions,
  type ParsedFile,
  type Parser,
  type RandomAccessSource,
  type TensorRecord,
  type TensorSample
} from "../types";

const HEADER_PREFIX_BYTES = 8;
const HEADER_PREFIX_BYTES_BIGINT = 8n;
const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
const MIN_SAFE_BIGINT = BigInt(Number.MIN_SAFE_INTEGER);

interface DTypeInfo {
  readonly byteSize: number;
  read(reader: BinaryReader, label: string, offset: bigint): number;
}

const DTYPE_BITS: Record<string, number> = {
  BOOL: 8,
  F4: 4,
  F6_E2M3: 6,
  F6_E3M2: 6,
  U8: 8,
  I8: 8,
  F8_E5M2: 8,
  F8_E4M3: 8,
  F8_E8M0: 8,
  F8_E4M3FNUZ: 8,
  F8_E5M2FNUZ: 8,
  I16: 16,
  U16: 16,
  F16: 16,
  BF16: 16,
  I32: 32,
  U32: 32,
  F32: 32,
  C64: 64,
  F64: 64,
  I64: 64,
  U64: 64
};

const DTYPE_INFO: Record<string, DTypeInfo> = {
  F64: {
    byteSize: 8,
    read(reader) {
      return reader.f64();
    }
  },
  F32: {
    byteSize: 4,
    read(reader) {
      return reader.f32();
    }
  },
  F16: {
    byteSize: 2,
    read(reader) {
      return decodeFloat16(reader.u16());
    }
  },
  BF16: {
    byteSize: 2,
    read(reader) {
      return decodeBfloat16(reader.u16());
    }
  },
  I64: {
    byteSize: 8,
    read(reader, label, offset) {
      return bigintScalarToNumber(reader.i64(), label, offset);
    }
  },
  U64: {
    byteSize: 8,
    read(reader, label, offset) {
      return bigintScalarToNumber(reader.u64(), label, offset);
    }
  },
  I32: {
    byteSize: 4,
    read(reader) {
      return reader.i32();
    }
  },
  U32: {
    byteSize: 4,
    read(reader) {
      return reader.u32();
    }
  },
  I16: {
    byteSize: 2,
    read(reader) {
      return reader.i16();
    }
  },
  U16: {
    byteSize: 2,
    read(reader) {
      return reader.u16();
    }
  },
  I8: {
    byteSize: 1,
    read(reader) {
      return reader.i8();
    }
  },
  U8: {
    byteSize: 1,
    read(reader) {
      return reader.u8();
    }
  },
  BOOL: {
    byteSize: 1,
    read(reader, label, offset) {
      const value = reader.u8();
      if (value !== 0 && value !== 1) {
        throw new ParseError(
          `${label} contains invalid BOOL value ${value}; expected 0 or 1`,
          offset
        );
      }
      return value;
    }
  }
};

export const SAFETENSORS_DTYPE_CATALOG: readonly DtypeCatalogEntry[] =
  Object.freeze(
    Object.entries(DTYPE_BITS).map(([dtype, bitsPerValue]) => ({
      format: "safetensors" as const,
      dtype,
      bitsPerValue,
      sampleSupport: DTYPE_INFO[dtype] ? "values" as const : "unsupported" as const
    }))
  );

export class SafeTensorsParser implements Parser {
  readonly format = "safetensors" as const;

  async parse(
    source: RandomAccessSource,
    options: ParseOptions = {}
  ): Promise<ParsedFile> {
    options.signal?.throwIfAborted();

    if (source.size < HEADER_PREFIX_BYTES_BIGINT) {
      throw new ParseError(
        `SafeTensors file is too small to contain an 8-byte header length (${source.size} bytes)`,
        0n
      );
    }

    const maxMetadataBytes = resolveMaxMetadataBytes(options.maxMetadataBytes);
    const headerPrefix = await source.read(
      0n,
      HEADER_PREFIX_BYTES,
      options.signal
    );
    const headerLength = new BinaryReader(headerPrefix).u64();

    if (headerLength > BigInt(maxMetadataBytes)) {
      throw new ParseError(
        `SafeTensors header length ${headerLength} exceeds maxMetadataBytes ${maxMetadataBytes}`,
        0n
      );
    }

    assertRange(HEADER_PREFIX_BYTES_BIGINT, headerLength, source.size, "SafeTensors header");
    const headerBytes = await source.read(
      HEADER_PREFIX_BYTES_BIGINT,
      bigintToSafeNumber(headerLength, "SafeTensors header length"),
      options.signal
    );
    const header = parseHeader(headerBytes);
    const dataStart = HEADER_PREFIX_BYTES_BIGINT + headerLength;
    const dataLength = source.size - dataStart;
    const diagnostics: Diagnostic[] = [];
    const tensors: TensorRecord[] = [];
    let metadata: Record<string, unknown> = {};

    for (const [name, value] of Object.entries(header)) {
      if (name === "__metadata__") {
        metadata = parseMetadata(value);
        continue;
      }
      const tensor = parseTensorRecord(name, value, source.id, dataStart, dataLength);
      if (DTYPE_BITS[tensor.dtype] === undefined) {
        diagnostics.push({
          severity: "warning",
          message: `Tensor ${tensor.name} uses unknown SafeTensors dtype ${tensor.dtype}`,
          offset: tensor.byteOffset
        });
      }
      tensors.push(tensor);
    }

    assertNoTensorOverlaps(tensors);

    return {
      id: source.id,
      name: source.name,
      format: this.format,
      size: source.size,
      metadata,
      tensors,
      diagnostics
    };
  }

  async sample(
    source: RandomAccessSource,
    tensor: TensorRecord,
    maxValues: number,
    signal?: AbortSignal
  ): Promise<TensorSample> {
    signal?.throwIfAborted();

    if (!Number.isSafeInteger(maxValues) || maxValues < 0) {
      throw new ParseError(`maxValues must be a non-negative safe integer (received ${maxValues})`);
    }

    const dtype = DTYPE_INFO[tensor.dtype];
    if (!dtype) {
      throw new ParseError(`SafeTensors sampling does not support dtype ${tensor.dtype}`);
    }

    const totalElements = product(tensor.shape);
    const byteSize = BigInt(dtype.byteSize);
    const expectedByteLength = totalElements * byteSize;
    const dataOffset = tensor.dataOffset ?? tensor.byteOffset;

    if (tensor.byteLength !== expectedByteLength) {
      throw new ParseError(
        `Tensor ${tensor.name} byte length ${tensor.byteLength} does not match dtype ${tensor.dtype} and shape ${formatShape(tensor.shape)} (${expectedByteLength} bytes expected)`,
        dataOffset
      );
    }

    assertRange(dataOffset, tensor.byteLength, source.size, `Tensor ${tensor.name}`);

    if (maxValues === 0 || totalElements === 0n) {
      return sampleStats([], totalElements);
    }

    if (totalElements <= BigInt(maxValues)) {
      const bytes = await source.read(
        dataOffset,
        bigintToSafeNumber(tensor.byteLength, `Tensor ${tensor.name} byte length`),
        signal
      );
      const reader = new BinaryReader(bytes);
      const values: number[] = [];
      for (let index = 0n; index < totalElements; index += 1n) {
        values.push(
          dtype.read(
            reader,
            `Tensor ${tensor.name}`,
            dataOffset + index * byteSize
          )
        );
      }
      return sampleStats(values, totalElements);
    }

    const indices = sampleIndices(totalElements, maxValues);
    const values: number[] = [];

    for (const index of indices) {
      signal?.throwIfAborted();
      const offset = dataOffset + index * byteSize;
      const bytes = await source.read(offset, dtype.byteSize, signal);
      values.push(dtype.read(new BinaryReader(bytes), `Tensor ${tensor.name}`, offset));
    }

    return sampleStats(values, totalElements);
  }
}

function resolveMaxMetadataBytes(maxMetadataBytes: number | undefined): number {
  const value = maxMetadataBytes ?? DEFAULT_MAX_METADATA_BYTES;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new ParseError(
      `maxMetadataBytes must be a non-negative safe integer (received ${value})`
    );
  }
  return value;
}

function parseHeader(bytes: Uint8Array): Record<string, unknown> {
  let header: unknown;
  try {
    header = JSON.parse(new BinaryReader(bytes).text(bytes.byteLength));
  } catch (error) {
    throw new ParseError(
      `Invalid SafeTensors header JSON: ${error instanceof Error ? error.message : String(error)}`,
      HEADER_PREFIX_BYTES_BIGINT
    );
  }
  if (!isRecord(header)) {
    throw new ParseError("SafeTensors header must be a JSON object", HEADER_PREFIX_BYTES_BIGINT);
  }
  return header;
}

function parseMetadata(value: unknown): Record<string, unknown> {
  if (value === undefined) return {};
  if (!isRecord(value)) {
    throw new ParseError("SafeTensors __metadata__ entry must be an object");
  }
  return value;
}

function parseTensorRecord(
  name: string,
  value: unknown,
  fileId: string,
  dataStart: bigint,
  dataLength: bigint
): TensorRecord {
  if (!isRecord(value)) {
    throw new ParseError(`SafeTensors tensor ${name} metadata must be an object`);
  }

  const dtype = value.dtype;
  if (typeof dtype !== "string" || !dtype) {
    throw new ParseError(`SafeTensors tensor ${name} is missing a valid dtype`);
  }

  const shape = parseShape(name, value.shape);
  const [relativeStart, relativeEnd] = parseDataOffsets(name, value.data_offsets);
  const byteLength = relativeEnd - relativeStart;
  if (relativeEnd < relativeStart) {
    throw new ParseError(
      `SafeTensors tensor ${name} data_offsets end ${relativeEnd} is before start ${relativeStart}`
    );
  }

  assertRange(relativeStart, byteLength, dataLength, `SafeTensors tensor ${name}`);

  const dtypeBits = DTYPE_BITS[dtype];
  if (dtypeBits !== undefined) {
    const expectedBits = product(shape) * BigInt(dtypeBits);
    if (expectedBits % 8n !== 0n) {
      throw new ParseError(
        `SafeTensors tensor ${name} dtype ${dtype} and shape ${formatShape(shape)} require ${expectedBits} bits, which is not byte-aligned`,
        dataStart + relativeStart
      );
    }
    const expectedByteLength = expectedBits / 8n;
    if (expectedByteLength !== byteLength) {
      throw new ParseError(
        `SafeTensors tensor ${name} declares ${byteLength} bytes but dtype ${dtype} and shape ${formatShape(shape)} require ${expectedByteLength}`,
        dataStart + relativeStart
      );
    }
  }

  const absoluteOffset = dataStart + relativeStart;
  return {
    id: `${fileId}:${name}`,
    name,
    fileId,
    dtype,
    shape,
    byteOffset: absoluteOffset,
    byteLength,
    dataOffset: absoluteOffset,
    storage: "inline",
    sampleSupport: DTYPE_INFO[dtype] ? "values" : "unsupported"
  };
}

function parseShape(name: string, value: unknown): bigint[] {
  if (!Array.isArray(value)) {
    throw new ParseError(`SafeTensors tensor ${name} is missing a valid shape array`);
  }
  return value.map((dimension, index) =>
    parseJsonUint(
      dimension,
      `SafeTensors tensor ${name} shape[${index}]`
    )
  );
}

function parseDataOffsets(name: string, value: unknown): [bigint, bigint] {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new ParseError(
      `SafeTensors tensor ${name} must define data_offsets as [start, end]`
    );
  }
  return [
    parseJsonUint(value[0], `SafeTensors tensor ${name} data_offsets[0]`),
    parseJsonUint(value[1], `SafeTensors tensor ${name} data_offsets[1]`)
  ];
}

function parseJsonUint(value: unknown, label: string): bigint {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new ParseError(`${label} must be a non-negative safe integer`);
  }
  return BigInt(value);
}

function assertNoTensorOverlaps(tensors: TensorRecord[]): void {
  const sorted = [...tensors].sort((left, right) => {
    if (left.byteOffset === right.byteOffset) return left.name.localeCompare(right.name);
    return left.byteOffset < right.byteOffset ? -1 : 1;
  });

  for (let index = 1; index < sorted.length; index += 1) {
    const previous = sorted[index - 1]!;
    const current = sorted[index]!;
    if (previous.byteOffset + previous.byteLength > current.byteOffset) {
      throw new ParseError(
        `SafeTensors tensor ${current.name} overlaps tensor ${previous.name}`,
        current.byteOffset
      );
    }
  }
}

function sampleIndices(totalElements: bigint, count: number): bigint[] {
  if (count <= 0) return [];
  if (count === 1) return [0n];
  const lastIndex = totalElements - 1n;
  const denominator = BigInt(count - 1);
  const indices: bigint[] = [];

  for (let index = 0; index < count; index += 1) {
    indices.push((BigInt(index) * lastIndex) / denominator);
  }

  return indices;
}

function bigintScalarToNumber(value: bigint, label: string, offset: bigint): number {
  if (value < MIN_SAFE_BIGINT || value > MAX_SAFE_BIGINT) {
    throw new ParseError(
      `${label} contains ${value}, which cannot be represented accurately as a JavaScript number`,
      offset
    );
  }
  return Number(value);
}

function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) return sign * 0;
    return sign * 2 ** -14 * (fraction / 0x400);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 0x400);
}

function decodeBfloat16(bits: number): number {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setUint32(0, bits << 16, true);
  return new DataView(bytes.buffer).getFloat32(0, true);
}

function formatShape(shape: bigint[]): string {
  return `[${shape.map((dimension) => dimension.toString()).join(", ")}]`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
