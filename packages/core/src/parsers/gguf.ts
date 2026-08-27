import { BinaryReader, bigintToSafeNumber, product, sampleStats } from "../binary";
import { ParseError } from "../errors";
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

const GGUF_MAGIC = 0x46554747;
const GGUF_VERSION_V2 = 2;
const GGUF_VERSION_V3 = 3;
const INITIAL_PREFIX_BYTES = 256;
const MAX_COLLECTION_ENTRIES = 1_000_000;
const MAX_TENSOR_DIMS = 16;
const MAX_NESTED_ARRAY_DEPTH = 4;
const MAX_KEY_BYTES = 16 * 1024;

const enum GgufValueType {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12
}

type SampleSupport = TensorRecord["sampleSupport"];

interface GgmlTypeSpec {
  name: string;
  sampleSupport: SampleSupport;
  scalarBytes?: number;
  blockBytes?: number;
  blockElements?: number;
}

interface RawTensorInfo {
  name: string;
  shape: bigint[];
  typeId: number;
  relativeOffset: bigint;
}

interface ParsedTensorInfo {
  tensor: TensorRecord;
  typeId: number;
}

const GGML_TYPES = new Map<number, GgmlTypeSpec>([
  [0, { name: "F32", sampleSupport: "values", scalarBytes: 4 }],
  [1, { name: "F16", sampleSupport: "values", scalarBytes: 2 }],
  [2, { name: "Q4_0", sampleSupport: "values", blockBytes: 18, blockElements: 32 }],
  [3, { name: "Q4_1", sampleSupport: "values", blockBytes: 20, blockElements: 32 }],
  [4, { name: "Q4_2", sampleSupport: "unsupported" }],
  [5, { name: "Q4_3", sampleSupport: "unsupported" }],
  [6, { name: "Q5_0", sampleSupport: "values", blockBytes: 22, blockElements: 32 }],
  [7, { name: "Q5_1", sampleSupport: "values", blockBytes: 24, blockElements: 32 }],
  [8, { name: "Q8_0", sampleSupport: "values", blockBytes: 34, blockElements: 32 }],
  [9, { name: "Q8_1", sampleSupport: "unsupported", blockBytes: 36, blockElements: 32 }],
  [10, { name: "Q2_K", sampleSupport: "unsupported", blockBytes: 84, blockElements: 256 }],
  [11, { name: "Q3_K", sampleSupport: "unsupported", blockBytes: 110, blockElements: 256 }],
  [12, { name: "Q4_K", sampleSupport: "unsupported", blockBytes: 144, blockElements: 256 }],
  [13, { name: "Q5_K", sampleSupport: "unsupported", blockBytes: 176, blockElements: 256 }],
  [14, { name: "Q6_K", sampleSupport: "unsupported", blockBytes: 210, blockElements: 256 }],
  [15, { name: "Q8_K", sampleSupport: "unsupported", blockBytes: 292, blockElements: 256 }],
  [16, { name: "IQ2_XXS", sampleSupport: "unsupported", blockBytes: 66, blockElements: 256 }],
  [17, { name: "IQ2_XS", sampleSupport: "unsupported", blockBytes: 74, blockElements: 256 }],
  [18, { name: "IQ3_XXS", sampleSupport: "unsupported", blockBytes: 98, blockElements: 256 }],
  [19, { name: "IQ1_S", sampleSupport: "unsupported", blockBytes: 50, blockElements: 256 }],
  [20, { name: "IQ4_NL", sampleSupport: "unsupported", blockBytes: 18, blockElements: 32 }],
  [21, { name: "IQ3_S", sampleSupport: "unsupported", blockBytes: 110, blockElements: 256 }],
  [22, { name: "IQ2_S", sampleSupport: "unsupported", blockBytes: 82, blockElements: 256 }],
  [23, { name: "IQ4_XS", sampleSupport: "unsupported", blockBytes: 136, blockElements: 256 }],
  [24, { name: "I8", sampleSupport: "metadata-only", scalarBytes: 1 }],
  [25, { name: "I16", sampleSupport: "metadata-only", scalarBytes: 2 }],
  [26, { name: "I32", sampleSupport: "metadata-only", scalarBytes: 4 }],
  [27, { name: "I64", sampleSupport: "metadata-only", scalarBytes: 8 }],
  [28, { name: "F64", sampleSupport: "metadata-only", scalarBytes: 8 }],
  [29, { name: "IQ1_M", sampleSupport: "unsupported", blockBytes: 56, blockElements: 256 }],
  [30, { name: "BF16", sampleSupport: "metadata-only", scalarBytes: 2 }],
  [31, { name: "Q4_0_4_4", sampleSupport: "unsupported" }],
  [32, { name: "Q4_0_4_8", sampleSupport: "unsupported" }],
  [33, { name: "Q4_0_8_8", sampleSupport: "unsupported" }],
  [34, { name: "TQ1_0", sampleSupport: "unsupported", blockBytes: 54, blockElements: 256 }],
  [35, { name: "TQ2_0", sampleSupport: "unsupported", blockBytes: 66, blockElements: 256 }],
  [36, { name: "IQ4_NL_4_4", sampleSupport: "unsupported" }],
  [37, { name: "IQ4_NL_4_8", sampleSupport: "unsupported" }],
  [38, { name: "IQ4_NL_8_8", sampleSupport: "unsupported" }],
  [39, { name: "MXFP4", sampleSupport: "unsupported", blockBytes: 17, blockElements: 32 }],
  [40, { name: "NVFP4", sampleSupport: "unsupported", blockBytes: 36, blockElements: 64 }],
  [41, { name: "Q1_0", sampleSupport: "unsupported", blockBytes: 18, blockElements: 128 }],
  [42, { name: "Q2_0", sampleSupport: "unsupported", blockBytes: 18, blockElements: 64 }]
]);

export const GGUF_DTYPE_CATALOG: readonly DtypeCatalogEntry[] = Object.freeze(
  [...GGML_TYPES.entries()].map(([typeId, spec]) => ({
    format: "gguf" as const,
    dtype: spec.name,
    typeId,
    sampleSupport: spec.sampleSupport,
    ...(spec.scalarBytes !== undefined ? { scalarBytes: spec.scalarBytes } : {}),
    ...(spec.blockBytes !== undefined ? { blockBytes: spec.blockBytes } : {}),
    ...(spec.blockElements !== undefined
      ? { blockElements: spec.blockElements }
      : {})
  }))
);

function needMore(length: number, offset: number, remaining: number): ParseError {
  return new ParseError(`Need ${length} bytes at buffer offset ${offset}, only ${remaining} remain`);
}

function isNeedMoreError(error: unknown): error is ParseError {
  return error instanceof ParseError && error.message.startsWith("Need ");
}

function alignUp(value: bigint, alignment: bigint): bigint {
  if (alignment <= 0n) {
    throw new ParseError(`Invalid GGUF alignment ${alignment}`);
  }
  const remainder = value % alignment;
  return remainder === 0n ? value : value + (alignment - remainder);
}

function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    if (fraction === 0) return sign * 0;
    return sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function byteAt(bytes: Uint8Array, index: number): number {
  const value = bytes[index];
  if (value === undefined) {
    throw new ParseError(`Missing byte at index ${index}`);
  }
  return value;
}

function getTypeSpec(typeId: number): GgmlTypeSpec | undefined {
  return GGML_TYPES.get(typeId);
}

function getTypeName(typeId: number): string {
  const spec = getTypeSpec(typeId);
  return spec ? spec.name : `GGML_TYPE_${typeId} (unsupported)`;
}

function getAlignment(metadata: Record<string, unknown>): bigint {
  const raw = metadata["general.alignment"];
  if (raw === undefined) return 32n;
  if (typeof raw === "number" && Number.isSafeInteger(raw) && raw > 0) {
    return BigInt(raw);
  }
  if (typeof raw === "bigint" && raw > 0n) {
    return raw;
  }
  throw new ParseError("GGUF metadata general.alignment must be a positive integer");
}

function getSampleSupport(typeId: number): SampleSupport {
  return getTypeSpec(typeId)?.sampleSupport ?? "unsupported";
}

function readString(reader: BinaryReader, maxBytes: number, label: string): string {
  const length = reader.u64();
  if (length > BigInt(maxBytes)) {
    throw new ParseError(`${label} length ${length} exceeds limit ${maxBytes}`);
  }
  return reader.text(bigintToSafeNumber(length, `${label} length`));
}

function readTypedValue(
  reader: BinaryReader,
  typeId: number,
  maxMetadataBytes: number,
  depth = 0
): unknown {
  if (depth > MAX_NESTED_ARRAY_DEPTH) {
    throw new ParseError(`GGUF metadata array nesting exceeds ${MAX_NESTED_ARRAY_DEPTH}`);
  }

  switch (typeId) {
    case GgufValueType.Uint8:
      return reader.u8();
    case GgufValueType.Int8:
      return reader.i8();
    case GgufValueType.Uint16:
      return reader.u16();
    case GgufValueType.Int16:
      return reader.i16();
    case GgufValueType.Uint32:
      return reader.u32();
    case GgufValueType.Int32:
      return reader.i32();
    case GgufValueType.Float32:
      return reader.f32();
    case GgufValueType.Bool: {
      const value = reader.u8();
      if (value !== 0 && value !== 1) {
        throw new ParseError(`Invalid GGUF boolean value ${value}`);
      }
      return value === 1;
    }
    case GgufValueType.String:
      return readString(reader, maxMetadataBytes, "GGUF metadata string");
    case GgufValueType.Array: {
      const elementType = reader.u32();
      const count = reader.u64();
      const size = bigintToSafeNumber(count, "GGUF metadata array length");
      if (size > MAX_COLLECTION_ENTRIES) {
        throw new ParseError(
          `GGUF metadata array length ${size} exceeds limit ${MAX_COLLECTION_ENTRIES}`
        );
      }
      const values = new Array<unknown>(size);
      for (let index = 0; index < size; index += 1) {
        values[index] = readTypedValue(reader, elementType, maxMetadataBytes, depth + 1);
      }
      return values;
    }
    case GgufValueType.Uint64:
      return reader.u64();
    case GgufValueType.Int64:
      return reader.i64();
    case GgufValueType.Float64:
      return reader.f64();
    default:
      throw new ParseError(`Unsupported GGUF metadata value type ${typeId}`);
  }
}

function detectLittleEndian(bytes: Uint8Array): boolean {
  if (bytes.byteLength < 4) {
    throw needMore(4, 0, bytes.byteLength);
  }
  if (bytes[0] === 0x47 && bytes[1] === 0x47 && bytes[2] === 0x55 && bytes[3] === 0x46) {
    return true;
  }
  if (bytes[0] === 0x46 && bytes[1] === 0x55 && bytes[2] === 0x47 && bytes[3] === 0x47) {
    return false;
  }
  throw new ParseError("Invalid GGUF magic");
}

function computeScalarValues(
  reader: BinaryReader,
  typeId: number,
  count: number,
  totalElements: bigint
): TensorSample {
  const values: number[] = [];
  for (let index = 0; index < count; index += 1) {
    switch (typeId) {
      case 0:
        values.push(reader.f32());
        break;
      case 1:
        values.push(decodeFloat16(reader.u16()));
        break;
      default:
        throw new ParseError(`Tensor type ${getTypeName(typeId)} cannot be sampled as scalar`);
    }
  }
  return sampleStats(values, totalElements);
}

function pushQ4Values(values: number[], scale: number, quantized: Uint8Array, offset: number): void {
  for (let index = 0; index < 16; index += 1) {
    values.push(scale * ((byteAt(quantized, index) & 0x0f) - offset));
  }
  for (let index = 0; index < 16; index += 1) {
    values.push(scale * ((byteAt(quantized, index) >> 4) - offset));
  }
}

function pushQ4_1Values(
  values: number[],
  scale: number,
  minimum: number,
  quantized: Uint8Array
): void {
  for (let index = 0; index < 16; index += 1) {
    values.push(scale * (byteAt(quantized, index) & 0x0f) + minimum);
  }
  for (let index = 0; index < 16; index += 1) {
    values.push(scale * (byteAt(quantized, index) >> 4) + minimum);
  }
}

function pushQ5Values(
  values: number[],
  scale: number,
  minimum: number,
  highBits: Uint8Array,
  quantized: Uint8Array,
  subtract: number
): void {
  for (let index = 0; index < 32; index += 1) {
    const byte = byteAt(quantized, index % 16);
    const low = index < 16 ? byte & 0x0f : byte >> 4;
    const high = (byteAt(highBits, Math.floor(index / 8)) >> (index % 8)) & 0x01;
    values.push(scale * (((high << 4) | low) - subtract) + minimum);
  }
}

function pushQ8_0Values(values: number[], scale: number, quantized: Uint8Array): void {
  for (let index = 0; index < 32; index += 1) {
    const byte = byteAt(quantized, index);
    const signed = byte >= 128 ? byte - 256 : byte;
    values.push(scale * signed);
  }
}

function toByteReader(bytes: Uint8Array, littleEndian: boolean): BinaryReader {
  return new BinaryReader(bytes, littleEndian);
}

export class GgufParser implements Parser {
  readonly format = "gguf" as const;

  async parse(source: RandomAccessSource, options: ParseOptions = {}): Promise<ParsedFile> {
    const configuredCap = Math.max(1, options.maxMetadataBytes ?? DEFAULT_MAX_METADATA_BYTES);
    const fileCap = source.size < BigInt(configuredCap)
      ? bigintToSafeNumber(source.size, "GGUF file size")
      : configuredCap;
    let prefixLength = Math.min(fileCap, INITIAL_PREFIX_BYTES);
    if (prefixLength < 24 && fileCap >= 24) {
      prefixLength = 24;
    }

    while (true) {
      const prefix = await source.read(0n, prefixLength, options.signal);
      try {
        return this.parsePrefix(source, prefix, configuredCap);
      } catch (error) {
        if (!isNeedMoreError(error)) {
          throw error;
        }
        if (prefixLength < fileCap) {
          prefixLength = Math.min(fileCap, Math.max(prefixLength * 2, prefixLength + 256));
          continue;
        }
        if (BigInt(fileCap) < source.size) {
          throw new ParseError(`GGUF metadata exceeds the ${fileCap} byte limit`);
        }
        throw error;
      }
    }
  }

  async sample(
    source: RandomAccessSource,
    tensor: TensorRecord,
    maxValues: number,
    signal?: AbortSignal
  ): Promise<TensorSample> {
    if (!Number.isSafeInteger(maxValues) || maxValues < 0) {
      throw new ParseError(`Invalid maxValues ${maxValues}`);
    }
    const typeId = tensor.encoding?.ggmlTypeId;
    const littleEndian = tensor.encoding?.littleEndian;
    if (typeof typeId !== "number" || typeof littleEndian !== "boolean") {
      throw new ParseError(`Tensor ${tensor.name} is missing GGUF decoding metadata`);
    }
    if (tensor.dataOffset === undefined) {
      throw new ParseError(`Tensor ${tensor.name} is missing an absolute data offset`);
    }

    const totalElements = product(tensor.shape);
    const requested = Number(totalElements < BigInt(maxValues) ? totalElements : BigInt(maxValues));
    if (requested === 0) {
      return sampleStats([], totalElements);
    }

    const spec = getTypeSpec(typeId);
    if (!spec || spec.sampleSupport !== "values") {
      throw new ParseError(`Sampling is not supported for tensor type ${tensor.dtype}`);
    }

    if (spec.scalarBytes) {
      const available = tensor.byteLength / BigInt(spec.scalarBytes);
      const count = Number(available < BigInt(requested) ? available : BigInt(requested));
      const bytes = await source.read(
        tensor.dataOffset,
        count * spec.scalarBytes,
        signal
      );
      return computeScalarValues(toByteReader(bytes, littleEndian), typeId, count, totalElements);
    }

    if (!spec.blockBytes || !spec.blockElements) {
      throw new ParseError(`Tensor type ${tensor.dtype} is missing block information`);
    }

    const blocksAvailable = tensor.byteLength / BigInt(spec.blockBytes);
    const blocksNeeded = Math.ceil(requested / spec.blockElements);
    const blocksToRead = Number(
      blocksAvailable < BigInt(blocksNeeded) ? blocksAvailable : BigInt(blocksNeeded)
    );
    const bytes = await source.read(
      tensor.dataOffset,
      blocksToRead * spec.blockBytes,
      signal
    );
    const reader = toByteReader(bytes, littleEndian);
    const values: number[] = [];

    for (let blockIndex = 0; blockIndex < blocksToRead && values.length < requested; blockIndex += 1) {
      switch (typeId) {
        case 2: {
          const scale = decodeFloat16(reader.u16());
          const quantized = bytes.subarray(reader.offset, reader.offset + 16);
          reader.skip(16);
          pushQ4Values(values, scale, quantized, 8);
          break;
        }
        case 3: {
          const scale = decodeFloat16(reader.u16());
          const minimum = decodeFloat16(reader.u16());
          const quantized = bytes.subarray(reader.offset, reader.offset + 16);
          reader.skip(16);
          pushQ4_1Values(values, scale, minimum, quantized);
          break;
        }
        case 6: {
          const scale = decodeFloat16(reader.u16());
          const highBits = bytes.subarray(reader.offset, reader.offset + 4);
          reader.skip(4);
          const quantized = bytes.subarray(reader.offset, reader.offset + 16);
          reader.skip(16);
          pushQ5Values(values, scale, 0, highBits, quantized, 16);
          break;
        }
        case 7: {
          const scale = decodeFloat16(reader.u16());
          const minimum = decodeFloat16(reader.u16());
          const highBits = bytes.subarray(reader.offset, reader.offset + 4);
          reader.skip(4);
          const quantized = bytes.subarray(reader.offset, reader.offset + 16);
          reader.skip(16);
          pushQ5Values(values, scale, minimum, highBits, quantized, 0);
          break;
        }
        case 8: {
          const scale = decodeFloat16(reader.u16());
          const quantized = bytes.subarray(reader.offset, reader.offset + 32);
          reader.skip(32);
          pushQ8_0Values(values, scale, quantized);
          break;
        }
        default:
          throw new ParseError(`Sampling is not implemented for tensor type ${tensor.dtype}`);
      }
    }

    if (values.length > requested) {
      values.length = requested;
    }
    return sampleStats(values, totalElements);
  }

  private parsePrefix(
    source: RandomAccessSource,
    bytes: Uint8Array,
    maxMetadataBytes: number
  ): ParsedFile {
    const littleEndian = detectLittleEndian(bytes);
    const reader = new BinaryReader(bytes, littleEndian);
    const magic = reader.u32();
    if (magic !== GGUF_MAGIC) {
      throw new ParseError("Invalid GGUF magic");
    }

    const version = reader.u32();
    if (version !== GGUF_VERSION_V2 && version !== GGUF_VERSION_V3) {
      throw new ParseError(`Unsupported GGUF version ${version}`);
    }

    const tensorCount = bigintToSafeNumber(reader.u64(), "GGUF tensor count");
    const metadataCount = bigintToSafeNumber(reader.u64(), "GGUF metadata count");
    if (tensorCount > MAX_COLLECTION_ENTRIES) {
      throw new ParseError(
        `GGUF tensor count ${tensorCount} exceeds limit ${MAX_COLLECTION_ENTRIES}`
      );
    }
    if (metadataCount > MAX_COLLECTION_ENTRIES) {
      throw new ParseError(
        `GGUF metadata count ${metadataCount} exceeds limit ${MAX_COLLECTION_ENTRIES}`
      );
    }

    const metadata: Record<string, unknown> = {};
    for (let index = 0; index < metadataCount; index += 1) {
      const key = readString(reader, MAX_KEY_BYTES, "GGUF metadata key");
      const typeId = reader.u32();
      metadata[key] = readTypedValue(reader, typeId, maxMetadataBytes);
    }

    const rawTensors: RawTensorInfo[] = [];
    for (let index = 0; index < tensorCount; index += 1) {
      const name = readString(reader, MAX_KEY_BYTES, "GGUF tensor name");
      const dimensionCount = reader.u32();
      if (dimensionCount > MAX_TENSOR_DIMS) {
        throw new ParseError(
          `GGUF tensor ${name} has ${dimensionCount} dimensions, limit is ${MAX_TENSOR_DIMS}`
        );
      }
      const shape: bigint[] = [];
      for (let dimension = 0; dimension < dimensionCount; dimension += 1) {
        shape.push(reader.u64());
      }
      const typeId = reader.u32();
      const relativeOffset = reader.u64();
      rawTensors.push({ name, shape, typeId, relativeOffset });
    }

    const alignment = getAlignment(metadata);
    const dataRegionStart = alignUp(BigInt(reader.offset), alignment);
    const diagnostics: Diagnostic[] = [];
    const tensors = rawTensors.map((info) => {
      const typeSpec = getTypeSpec(info.typeId);
      if (!typeSpec) {
        diagnostics.push({
          severity: "warning",
          message: `Tensor ${info.name} uses unsupported GGML type id ${info.typeId}`
        });
      }
      const tensor: TensorRecord = {
        id: `${source.id}:${info.name}`,
        name: info.name,
        fileId: source.id,
        dtype: getTypeName(info.typeId),
        shape: info.shape,
        byteOffset: dataRegionStart + info.relativeOffset,
        byteLength: 0n,
        dataOffset: dataRegionStart + info.relativeOffset,
        encoding: {
          ggmlTypeId: info.typeId,
          littleEndian,
          ...(typeSpec?.scalarBytes
            ? { scalarBytes: typeSpec.scalarBytes }
            : {}),
          ...(typeSpec?.blockBytes
            ? { blockBytes: typeSpec.blockBytes }
            : {}),
          ...(typeSpec?.blockElements
            ? { blockElements: typeSpec.blockElements }
            : {})
        },
        sampleSupport: getSampleSupport(info.typeId)
      };
      return { tensor, typeId: info.typeId } satisfies ParsedTensorInfo;
    });

    const sortedByOffset = [...tensors].sort((left, right) =>
      left.tensor.byteOffset < right.tensor.byteOffset ? -1 : left.tensor.byteOffset > right.tensor.byteOffset ? 1 : 0
    );
    for (let index = 0; index < sortedByOffset.length; index += 1) {
      const current = sortedByOffset[index]!.tensor;
      if (current.dataOffset! > source.size) {
        throw new ParseError(`Tensor ${current.name} data starts beyond the end of the file`, current.dataOffset);
      }
      const next = sortedByOffset[index + 1]?.tensor;
      const end = next ? next.byteOffset : source.size;
      if (end < current.dataOffset!) {
        throw new ParseError(`Tensor ${current.name} has an invalid byte range`, current.dataOffset);
      }
      const parsedTensor = sortedByOffset[index]!;
      const spec = getTypeSpec(parsedTensor.typeId);
      const calculatedLength = spec
        ? calculateTensorByteLength(
            current.name,
            current.shape,
            spec,
            current.dataOffset
          )
        : undefined;
      const availableLength = end - current.dataOffset!;
      if (calculatedLength !== undefined && calculatedLength > availableLength) {
        throw new ParseError(
          `Tensor ${current.name} needs ${calculatedLength} bytes but only ${availableLength} are available`,
          current.dataOffset
        );
      }

      current.byteLength = calculatedLength ?? availableLength;
    }

    return {
      id: source.id,
      name: source.name,
      format: "gguf",
      size: source.size,
      metadata,
      tensors: tensors.map(({ tensor }) => tensor),
      diagnostics
    };
  }
}

function calculateTensorByteLength(
  name: string,
  shape: bigint[],
  spec: GgmlTypeSpec,
  offset: bigint | undefined
): bigint | undefined {
  if (spec.scalarBytes) {
    return product(shape) * BigInt(spec.scalarBytes);
  }
  if (!spec.blockBytes || !spec.blockElements) return undefined;

  const rowElements = shape[0] ?? 1n;
  const blockElements = BigInt(spec.blockElements);
  if (rowElements % blockElements !== 0n) {
    throw new ParseError(
      `Tensor ${name} row length ${rowElements} is not divisible by block size ${blockElements}`,
      offset
    );
  }
  const rowCount = product(shape.slice(1));
  return (
    (rowElements / blockElements) *
    BigInt(spec.blockBytes) *
    rowCount
  );
}
