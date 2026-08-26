import { bigintToSafeNumber } from "../binary";
import { ParseError } from "../errors";
import {
  DEFAULT_MAX_METADATA_BYTES,
  type ParseOptions,
  type ParsedFile,
  type Parser,
  type RandomAccessSource,
  type TensorRecord
} from "../types";

const decoder = new TextDecoder();

const MAX_RECURSION_DEPTH = 32;

const ONNX_DATA_TYPE_NAMES: Record<number, string> = {
  0: "UNDEFINED",
  1: "FLOAT",
  2: "UINT8",
  3: "INT8",
  4: "UINT16",
  5: "INT16",
  6: "INT32",
  7: "INT64",
  8: "STRING",
  9: "BOOL",
  10: "FLOAT16",
  11: "DOUBLE",
  12: "UINT32",
  13: "UINT64",
  14: "COMPLEX64",
  15: "COMPLEX128",
  16: "BFLOAT16",
  17: "FLOAT8E4M3FN",
  18: "FLOAT8E4M3FNUZ",
  19: "FLOAT8E5M2",
  20: "FLOAT8E5M2FNUZ",
  21: "UINT4",
  22: "INT4",
  23: "FLOAT4E2M1",
  24: "FLOAT8E8M0",
  25: "UINT2",
  26: "INT2"
};

const ONNX_DATA_LOCATION_NAMES: Record<number, string> = {
  0: "DEFAULT",
  1: "EXTERNAL"
};

interface ReadVarintResult {
  next: number;
  value: bigint;
}

interface LengthDelimitedRange {
  next: number;
  valueEnd: number;
  valueStart: number;
}

interface OnnxExternalData {
  basepath?: string;
  checksum?: string;
  length?: bigint;
  location?: string;
  offset?: bigint;
}

interface OnnxInitializerMetadata {
  dataLocation: number;
  dataLocationName: string;
  dataType: number;
  dataTypeName: string;
  dims: bigint[];
  externalData: OnnxExternalData;
  name: string;
  rawData: {
    length: bigint;
    offset?: bigint;
    present: boolean;
  };
  tensorOffset: bigint;
}

interface ParsedOnnxModel {
  graphName?: string;
  initializerMetadata: OnnxInitializerMetadata[];
  irVersion?: bigint;
  producerName?: string;
  producerVersion?: string;
  modelVersion?: bigint;
  domain?: string;
  docString?: string;
  tensors: TensorRecord[];
}

export class OnnxParser implements Parser {
  readonly format = "onnx" as const;

  async parse(source: RandomAccessSource, options: ParseOptions = {}): Promise<ParsedFile> {
    const maxMetadataBytes = resolveOnnxMetadataLimit(options.maxMetadataBytes);
    if (source.size > BigInt(maxMetadataBytes)) {
      throw new ParseError(`ONNX metadata exceeds the ${maxMetadataBytes} byte limit`);
    }

    options.signal?.throwIfAborted();
    const bytes = await source.read(
      0n,
      bigintToSafeNumber(source.size, "ONNX file size"),
      options.signal
    );
    options.signal?.throwIfAborted();

    const parsed = parseModelProto(bytes, source.id);
    return {
      id: source.id,
      name: source.name,
      format: this.format,
      size: source.size,
      metadata: buildFileMetadata(parsed),
      tensors: parsed.tensors,
      diagnostics: []
    };
  }
}

function resolveOnnxMetadataLimit(value?: number): number {
  const candidate = value ?? DEFAULT_MAX_METADATA_BYTES;
  if (!Number.isSafeInteger(candidate) || candidate <= 0) {
    throw new ParseError(`Invalid ONNX metadata byte limit: ${value}`);
  }
  return candidate;
}

function buildFileMetadata(parsed: ParsedOnnxModel): Record<string, unknown> {
  const metadata: Record<string, unknown> = {
    initializers: parsed.initializerMetadata,
    initializerCount: parsed.initializerMetadata.length
  };
  if (parsed.graphName !== undefined) metadata.graphName = parsed.graphName;
  if (parsed.irVersion !== undefined) metadata.irVersion = parsed.irVersion;
  if (parsed.producerName !== undefined) metadata.producerName = parsed.producerName;
  if (parsed.producerVersion !== undefined) metadata.producerVersion = parsed.producerVersion;
  if (parsed.modelVersion !== undefined) metadata.modelVersion = parsed.modelVersion;
  if (parsed.domain !== undefined) metadata.domain = parsed.domain;
  if (parsed.docString !== undefined) metadata.docString = parsed.docString;
  return metadata;
}

function parseModelProto(bytes: Uint8Array, fileId: string): ParsedOnnxModel {
  let offset = 0;
  const model: ParsedOnnxModel = {
    initializerMetadata: [],
    tensors: []
  };

  while (offset < bytes.byteLength) {
    const tag = readVarint(bytes, offset, bytes.byteLength);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 0x7n);
    ensureTag(fieldNumber, wireType, offset);

    switch (fieldNumber) {
      case 1: {
        expectWireType(wireType, 0, offset);
        const value = readVarint(bytes, offset, bytes.byteLength);
        model.irVersion = asSignedInt64(value.value);
        offset = value.next;
        break;
      }
      case 2: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        model.producerName = decodeUtf8(bytes, range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 3: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        model.producerVersion = decodeUtf8(bytes, range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 4: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        model.domain = decodeUtf8(bytes, range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 5: {
        expectWireType(wireType, 0, offset);
        const value = readVarint(bytes, offset, bytes.byteLength);
        model.modelVersion = asSignedInt64(value.value);
        offset = value.next;
        break;
      }
      case 6: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        model.docString = decodeUtf8(bytes, range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 7: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        const graph = parseGraphProto(bytes, range.valueStart, range.valueEnd, fileId, 1);
        if (graph.graphName !== undefined) model.graphName = graph.graphName;
        model.initializerMetadata = graph.initializerMetadata;
        model.tensors = graph.tensors;
        offset = range.next;
        break;
      }
      default:
        offset = skipField(bytes, offset, bytes.byteLength, wireType);
        break;
    }
  }

  return model;
}

function parseGraphProto(
  bytes: Uint8Array,
  start: number,
  end: number,
  fileId: string,
  depth: number
): {
  graphName?: string;
  initializerMetadata: OnnxInitializerMetadata[];
  tensors: TensorRecord[];
} {
  ensureDepth(depth, start);
  let offset = start;
  let graphName: string | undefined;
  const initializerMetadata: OnnxInitializerMetadata[] = [];
  const tensors: TensorRecord[] = [];
  let tensorIndex = 0;

  while (offset < end) {
    const tag = readVarint(bytes, offset, end);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 0x7n);
    ensureTag(fieldNumber, wireType, offset);

    switch (fieldNumber) {
      case 2: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, end);
        graphName = decodeUtf8(bytes, range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 5: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, end);
        const parsedTensor = parseTensorProto(
          bytes,
          range.valueStart,
          range.valueEnd,
          fileId,
          tensorIndex++,
          depth + 1
        );
        initializerMetadata.push(parsedTensor.metadata);
        tensors.push(parsedTensor.tensor);
        offset = range.next;
        break;
      }
      default:
        offset = skipField(bytes, offset, end, wireType);
        break;
    }
  }

  const result: {
    graphName?: string;
    initializerMetadata: OnnxInitializerMetadata[];
    tensors: TensorRecord[];
  } = {
    initializerMetadata,
    tensors
  };
  if (graphName !== undefined) result.graphName = graphName;
  return result;
}

function parseTensorProto(
  bytes: Uint8Array,
  start: number,
  end: number,
  fileId: string,
  tensorIndex: number,
  depth: number
): {
  metadata: OnnxInitializerMetadata;
  tensor: TensorRecord;
} {
  ensureDepth(depth, start);
  let offset = start;

  const dims: bigint[] = [];
  let dataType = 0;
  let dataLocation = 0;
  let name = "";
  let rawDataOffset: bigint | undefined;
  let rawDataLength = 0n;
  const typedDataSegments: Array<{ byteOffset: bigint; byteLength: bigint }> = [];
  const externalData: OnnxExternalData = {};

  const markTypedData = (startOffset: number, endOffset: number) => {
    typedDataSegments.push({
      byteOffset: BigInt(startOffset),
      byteLength: BigInt(endOffset - startOffset)
    });
  };

  while (offset < end) {
    const tag = readVarint(bytes, offset, end);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 0x7n);
    ensureTag(fieldNumber, wireType, offset);

    switch (fieldNumber) {
      case 1:
        if (wireType === 0) {
          const value = readVarint(bytes, offset, end);
          dims.push(asSignedInt64(value.value));
          offset = value.next;
        } else if (wireType === 2) {
          const range = readLengthDelimitedRange(bytes, offset, end);
          let packedOffset = range.valueStart;
          while (packedOffset < range.valueEnd) {
            const value = readVarint(bytes, packedOffset, range.valueEnd);
            dims.push(asSignedInt64(value.value));
            packedOffset = value.next;
          }
          offset = range.next;
        } else {
          throw new ParseError(`Unexpected wire type ${wireType} for TensorProto.dims`, BigInt(offset));
        }
        break;
      case 2: {
        expectWireType(wireType, 0, offset);
        const value = readVarint(bytes, offset, end);
        dataType = Number(value.value);
        offset = value.next;
        break;
      }
      case 4:
      case 5:
      case 6:
      case 7: {
        if (wireType === 2) {
          const range = readLengthDelimitedRange(bytes, offset, end);
          markTypedData(range.valueStart, range.valueEnd);
          offset = range.next;
        } else if (wireType === 0 && (fieldNumber === 5 || fieldNumber === 7)) {
          const valueStart = offset;
          const value = readVarint(bytes, offset, end);
          markTypedData(valueStart, value.next);
          offset = value.next;
        } else if (wireType === 5 && fieldNumber === 4) {
          markTypedData(offset, offset + 4);
          offset = skipField(bytes, offset, end, wireType);
        } else {
          throw new ParseError(
            `Unexpected wire type ${wireType} for TensorProto typed data`,
            BigInt(offset)
          );
        }
        break;
      }
      case 8: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, end);
        name = decodeUtf8(bytes, range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 9: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, end);
        rawDataOffset = BigInt(range.valueStart);
        rawDataLength = BigInt(range.valueEnd - range.valueStart);
        offset = range.next;
        break;
      }
      case 10:
      case 11: {
        if (wireType === 2) {
          const range = readLengthDelimitedRange(bytes, offset, end);
          markTypedData(range.valueStart, range.valueEnd);
          offset = range.next;
        } else if (wireType === 1 && fieldNumber === 10) {
          markTypedData(offset, offset + 8);
          offset = skipField(bytes, offset, end, wireType);
        } else if (wireType === 0 && fieldNumber === 11) {
          const valueStart = offset;
          const value = readVarint(bytes, offset, end);
          markTypedData(valueStart, value.next);
          offset = value.next;
        } else {
          throw new ParseError(
            `Unexpected wire type ${wireType} for TensorProto typed data`,
            BigInt(offset)
          );
        }
        break;
      }
      case 13: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, end);
        mergeExternalData(externalData, parseStringStringEntry(bytes, range.valueStart, range.valueEnd, depth + 1));
        offset = range.next;
        break;
      }
      case 14: {
        expectWireType(wireType, 0, offset);
        const value = readVarint(bytes, offset, end);
        dataLocation = Number(value.value);
        offset = value.next;
        break;
      }
      default:
        offset = skipField(bytes, offset, end, wireType);
        break;
    }
  }

  const inferredExternal =
    dataLocation === 1 ||
    externalData.location !== undefined ||
    externalData.offset !== undefined ||
    externalData.length !== undefined;
  const dataTypeName = ONNX_DATA_TYPE_NAMES[dataType] ?? `UNKNOWN(${dataType})`;
  const dataLocationName =
    ONNX_DATA_LOCATION_NAMES[dataLocation] ?? (inferredExternal ? "EXTERNAL" : `UNKNOWN(${dataLocation})`);
  const firstTypedSegment = typedDataSegments[0];
  const inlineOffset = rawDataOffset ?? firstTypedSegment?.byteOffset;
  const typedDataLength = typedDataSegments.reduce(
    (sum, segment) => sum + segment.byteLength,
    0n
  );
  const byteOffset = inferredExternal
    ? externalData.offset ?? 0n
    : inlineOffset ?? BigInt(start);
  const byteLength =
    inferredExternal
      ? externalData.length ?? 0n
      : rawDataOffset !== undefined
      ? rawDataLength
      : typedDataSegments.length > 0
        ? typedDataLength
        : 0n;

  const tensor: TensorRecord = {
    id: `${fileId}:initializer:${tensorIndex}`,
    name,
    fileId,
    dtype: dataTypeName,
    shape: dims,
    byteOffset,
    byteLength,
    sampleSupport: "metadata-only",
    storage: inferredExternal ? "external" : "inline"
  };
  if (!inferredExternal && rawDataOffset !== undefined) {
    tensor.byteSegments = [{ byteOffset: rawDataOffset, byteLength: rawDataLength }];
  } else if (!inferredExternal && typedDataSegments.length > 0) {
    tensor.byteSegments = typedDataSegments;
  }
  if (inlineOffset !== undefined) tensor.dataOffset = inlineOffset;
  if (externalData.location !== undefined) tensor.externalLocation = externalData.location;
  if (externalData.offset !== undefined) tensor.externalOffset = externalData.offset;
  if (externalData.length !== undefined) tensor.externalLength = externalData.length;

  return {
    metadata: {
      name,
      dims,
      dataType,
      dataTypeName,
      dataLocation,
      dataLocationName,
      tensorOffset: BigInt(start),
      rawData:
        rawDataOffset === undefined
          ? {
              present: false,
              length: rawDataLength
            }
          : {
              present: true,
              offset: rawDataOffset,
              length: rawDataLength
            },
      externalData
    },
    tensor
  };
}

function parseStringStringEntry(
  bytes: Uint8Array,
  start: number,
  end: number,
  depth: number
): Partial<Record<keyof OnnxExternalData, string | bigint>> {
  ensureDepth(depth, start);
  let offset = start;
  let key = "";
  let value = "";

  while (offset < end) {
    const tag = readVarint(bytes, offset, end);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 0x7n);
    ensureTag(fieldNumber, wireType, offset);

    switch (fieldNumber) {
      case 1: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, end);
        key = decodeUtf8(bytes, range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 2: {
        expectWireType(wireType, 2, offset);
        const range = readLengthDelimitedRange(bytes, offset, end);
        value = decodeUtf8(bytes, range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      default:
        offset = skipField(bytes, offset, end, wireType);
        break;
    }
  }

  switch (key) {
    case "location":
      return { location: value };
    case "offset": {
      const offsetValue = parseDecimalBigInt(value);
      return offsetValue === undefined ? {} : { offset: offsetValue };
    }
    case "length": {
      const lengthValue = parseDecimalBigInt(value);
      return lengthValue === undefined ? {} : { length: lengthValue };
    }
    case "checksum":
      return { checksum: value };
    case "basepath":
      return { basepath: value };
    default:
      return {};
  }
}

function mergeExternalData(
  target: OnnxExternalData,
  update: Partial<Record<keyof OnnxExternalData, string | bigint>>
): void {
  if (typeof update.location === "string") target.location = update.location;
  if (typeof update.offset === "bigint") target.offset = update.offset;
  if (typeof update.length === "bigint") target.length = update.length;
  if (typeof update.checksum === "string") target.checksum = update.checksum;
  if (typeof update.basepath === "string") target.basepath = update.basepath;
}

function parseDecimalBigInt(value: string): bigint | undefined {
  if (!/^\d+$/.test(value)) return undefined;
  return BigInt(value);
}

function decodeUtf8(bytes: Uint8Array, start: number, end: number): string {
  return decoder.decode(bytes.subarray(start, end));
}

function readVarint(bytes: Uint8Array, start: number, end: number): ReadVarintResult {
  let value = 0n;
  let shift = 0n;
  let offset = start;

  for (let index = 0; index < 10; index += 1) {
    if (offset >= end) {
      throw new ParseError("Truncated protobuf varint", BigInt(start));
    }
    const byte = bytes[offset++];
    if (byte === undefined) {
      throw new ParseError("Unexpected end of protobuf varint", BigInt(offset));
    }
    if (index === 9 && byte > 1) {
      throw new ParseError("Protobuf varint exceeds 64 bits", BigInt(start));
    }
    value |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return { next: offset, value };
    }
    shift += 7n;
  }

  throw new ParseError("Protobuf varint exceeds 10 bytes", BigInt(start));
}

function readLengthDelimitedRange(
  bytes: Uint8Array,
  start: number,
  end: number
): LengthDelimitedRange {
  const length = readVarint(bytes, start, end);
  const remaining = end - length.next;
  if (length.value > BigInt(remaining)) {
    throw new ParseError(
      `Length-delimited field exceeds remaining bytes (${length.value} > ${remaining})`,
      BigInt(start)
    );
  }
  const byteLength = Number(length.value);
  if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
    throw new ParseError(`Invalid length-delimited field size ${length.value}`, BigInt(start));
  }
  return {
    valueStart: length.next,
    valueEnd: length.next + byteLength,
    next: length.next + byteLength
  };
}

function skipField(bytes: Uint8Array, start: number, end: number, wireType: number): number {
  switch (wireType) {
    case 0:
      return readVarint(bytes, start, end).next;
    case 1:
      ensureFixedWidth(start, end, 8);
      return start + 8;
    case 2:
      return readLengthDelimitedRange(bytes, start, end).next;
    case 5:
      ensureFixedWidth(start, end, 4);
      return start + 4;
    default:
      throw new ParseError(`Unsupported protobuf wire type ${wireType}`, BigInt(start));
  }
}

function ensureFixedWidth(start: number, end: number, width: number): void {
  if (start + width > end) {
    throw new ParseError(`Fixed-width protobuf field overruns message bounds`, BigInt(start));
  }
}

function ensureTag(fieldNumber: number, wireType: number, offset: number): void {
  if (!Number.isSafeInteger(fieldNumber) || fieldNumber <= 0) {
    throw new ParseError(`Invalid protobuf field number ${fieldNumber}`, BigInt(offset));
  }
  if (![0, 1, 2, 5].includes(wireType)) {
    throw new ParseError(`Unsupported protobuf wire type ${wireType}`, BigInt(offset));
  }
}

function expectWireType(actual: number, expected: number, offset: number): void {
  if (actual !== expected) {
    throw new ParseError(
      `Unexpected protobuf wire type ${actual}, expected ${expected}`,
      BigInt(offset)
    );
  }
}

function ensureDepth(depth: number, offset: number): void {
  if (depth > MAX_RECURSION_DEPTH) {
    throw new ParseError(`ONNX protobuf nesting exceeds depth ${MAX_RECURSION_DEPTH}`, BigInt(offset));
  }
}

function asSignedInt64(value: bigint): bigint {
  return value >= 0x8000000000000000n ? value - 0x10000000000000000n : value;
}
