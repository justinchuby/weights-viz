import { bigintToSafeNumber } from "../binary";
import { ParseError } from "../errors";
import {
  DEFAULT_MAX_METADATA_BYTES,
  type DtypeCatalogEntry,
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
  26: "INT2",
  27: "FLOAT6E2M3",
  28: "FLOAT6E3M2"
};

export const ONNX_DTYPE_CATALOG: readonly DtypeCatalogEntry[] = Object.freeze(
  Object.entries(ONNX_DATA_TYPE_NAMES).map(([typeId, dtype]) => ({
    format: "onnx" as const,
    dtype,
    typeId: Number(typeId),
    sampleSupport: "metadata-only" as const
  }))
);

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
  data_location: number;
  data_location_name: string;
  data_type: number;
  data_type_name: string;
  dims: bigint[];
  external_data: OnnxExternalData;
  name: string;
  raw_data: {
    length: bigint;
    offset?: bigint;
    present: boolean;
  };
  tensor_offset: bigint;
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
    options.signal?.throwIfAborted();
    const maxMetadataBytes = resolveOnnxMetadataLimit(options.maxMetadataBytes);
    const reader = new OnnxReader(source, maxMetadataBytes, options.signal);
    const parsed = await parseModelProto(reader, source.id);
    options.signal?.throwIfAborted();
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

// Cache bounded windows, not the protobuf container: tensor payloads may be gigabytes.
class OnnxReader {
  readonly byteLength: number;
  private cache: Uint8Array = new Uint8Array();
  private cacheStart = 0;
  private metadataBytes = 0;

  constructor(
    private readonly source: RandomAccessSource,
    private readonly limit: number,
    private readonly signal?: AbortSignal
  ) {
    this.byteLength = bigintToSafeNumber(source.size, "ONNX file size");
  }

  charge(length: number): void {
    this.signal?.throwIfAborted();
    this.metadataBytes += length;
    if (this.metadataBytes > this.limit) {
      throw new ParseError(`ONNX metadata exceeds the ${this.limit} byte limit`);
    }
  }

  async byte(offset: number): Promise<number> {
    this.signal?.throwIfAborted();
    if (offset < this.cacheStart || offset >= this.cacheStart + this.cache.length) {
      this.cacheStart = offset;
      const length = Math.min(64 * 1024, this.byteLength - offset);
      this.cache = await this.source.read(BigInt(offset), length, this.signal);
      if (this.cache.length !== length) {
        throw new ParseError("Truncated ONNX source read", BigInt(offset));
      }
    }
    this.signal?.throwIfAborted();
    return this.cache[offset - this.cacheStart]!;
  }

  async string(start: number, end: number): Promise<string> {
    this.charge(end - start);
    const bytes = new Uint8Array(end - start);
    let offset = start;
    while (offset < end) {
      await this.byte(offset);
      const length = Math.min(end - offset, this.cacheStart + this.cache.length - offset);
      bytes.set(this.cache.subarray(offset - this.cacheStart, offset - this.cacheStart + length), offset - start);
      offset += length;
    }
    return decoder.decode(bytes);
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
    initializer_count: parsed.initializerMetadata.length
  };
  if (parsed.graphName !== undefined) metadata.graph_name = parsed.graphName;
  if (parsed.irVersion !== undefined) metadata.ir_version = parsed.irVersion;
  if (parsed.producerName !== undefined) metadata.producer_name = parsed.producerName;
  if (parsed.producerVersion !== undefined) metadata.producer_version = parsed.producerVersion;
  if (parsed.modelVersion !== undefined) metadata.model_version = parsed.modelVersion;
  if (parsed.domain !== undefined) metadata.domain = parsed.domain;
  if (parsed.docString !== undefined) metadata.doc_string = parsed.docString;
  return metadata;
}

async function parseModelProto(bytes: OnnxReader, fileId: string): Promise<ParsedOnnxModel> {
  let offset = 0;
  const model: ParsedOnnxModel = {
    initializerMetadata: [],
    tensors: []
  };

  while (offset < bytes.byteLength) {
    const tag = await readVarint(bytes, offset, bytes.byteLength);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 0x7n);
    ensureTag(fieldNumber, wireType, offset);

    switch (fieldNumber) {
      case 1: {
        expectWireType(wireType, 0, offset);
        const value = await readVarint(bytes, offset, bytes.byteLength);
        model.irVersion = asSignedInt64(value.value);
        offset = value.next;
        break;
      }
      case 2: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        model.producerName = await bytes.string(range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 3: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        model.producerVersion = await bytes.string(range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 4: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        model.domain = await bytes.string(range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 5: {
        expectWireType(wireType, 0, offset);
        const value = await readVarint(bytes, offset, bytes.byteLength);
        model.modelVersion = asSignedInt64(value.value);
        offset = value.next;
        break;
      }
      case 6: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        model.docString = await bytes.string(range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 7: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        const graph = await parseGraphProto(bytes, range.valueStart, range.valueEnd, fileId, 1);
        if (graph.graphName !== undefined) model.graphName = graph.graphName;
        for (const metadata of graph.initializerMetadata) model.initializerMetadata.push(metadata);
        for (const tensor of graph.tensors) model.tensors.push(tensor);
        offset = range.next;
        break;
      }
      case 20:
      case 25: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, bytes.byteLength);
        await parseTensorContainers(
          bytes, range.valueStart, range.valueEnd, fileId, 1,
          fieldNumber === 20 ? "training" : "function", model.tensors
        );
        offset = range.next;
        break;
      }
      default:
        offset = await skipField(bytes, offset, bytes.byteLength, wireType);
        break;
    }
  }

  return model;
}

async function parseGraphProto(
  bytes: OnnxReader,
  start: number,
  end: number,
  fileId: string,
  depth: number
): Promise<{
  graphName?: string;
  initializerMetadata: OnnxInitializerMetadata[];
  tensors: TensorRecord[];
}> {
  ensureDepth(depth, start);
  let offset = start;
  let graphName: string | undefined;
  const initializerMetadata: OnnxInitializerMetadata[] = [];
  const tensors: TensorRecord[] = [];

  while (offset < end) {
    const tag = await readVarint(bytes, offset, end);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 0x7n);
    ensureTag(fieldNumber, wireType, offset);

    switch (fieldNumber) {
      case 1:
      case 15: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, end);
        await parseTensorContainers(
          bytes, range.valueStart, range.valueEnd, fileId, depth + 1,
          fieldNumber === 1 ? "node" : "sparse", tensors
        );
        offset = range.next;
        break;
      }
      case 2: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, end);
        graphName = await bytes.string(range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 5: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, end);
        const parsedTensor = await parseTensorProto(
          bytes,
          range.valueStart,
          range.valueEnd,
          fileId,
          depth + 1
        );
        initializerMetadata.push(parsedTensor.metadata);
        tensors.push(parsedTensor.tensor);
        offset = range.next;
        break;
      }
      default:
        offset = await skipField(bytes, offset, end, wireType);
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

type TensorContainer = "node" | "attribute" | "sparse" | "function" | "training";

const TENSOR_CONTAINERS: Record<TensorContainer, Readonly<Record<number, TensorContainer | "tensor" | "graph">>> = {
  node: { 5: "attribute" },
  attribute: { 5: "tensor", 6: "graph", 10: "tensor", 11: "graph", 22: "sparse", 23: "sparse" },
  sparse: { 1: "tensor", 2: "tensor" },
  function: { 7: "node", 11: "attribute" },
  training: { 1: "graph", 2: "graph" }
};

async function parseTensorContainers(
  bytes: OnnxReader,
  start: number,
  end: number,
  fileId: string,
  depth: number,
  kind: TensorContainer,
  tensors: TensorRecord[]
): Promise<void> {
  ensureDepth(depth, start);
  let offset = start;
  while (offset < end) {
    const tag = await readVarint(bytes, offset, end);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 7n);
    ensureTag(fieldNumber, wireType, offset);
    const child = TENSOR_CONTAINERS[kind][fieldNumber];
    if (!child) {
      offset = await skipField(bytes, offset, end, wireType);
      continue;
    }
    expectWireType(wireType, 2, offset);
    const range = await readLengthDelimitedRange(bytes, offset, end);
    if (child === "tensor") {
      const parsed = await parseTensorProto(bytes, range.valueStart, range.valueEnd, fileId, depth + 1);
      tensors.push(parsed.tensor);
    } else if (child === "graph") {
      const parsed = await parseGraphProto(bytes, range.valueStart, range.valueEnd, fileId, depth + 1);
      for (const tensor of parsed.tensors) tensors.push(tensor);
    } else {
      await parseTensorContainers(bytes, range.valueStart, range.valueEnd, fileId, depth + 1, child, tensors);
    }
    offset = range.next;
  }
}

async function parseTensorProto(
  bytes: OnnxReader,
  start: number,
  end: number,
  fileId: string,
  depth: number
): Promise<{
  metadata: OnnxInitializerMetadata;
  tensor: TensorRecord;
}> {
  ensureDepth(depth, start);
  let offset = start;

  const dims: bigint[] = [];
  let dataType = 0;
  let dataLocation = 0;
  let name = "";
  let rawDataOffset: bigint | undefined;
  let rawDataLength = 0n;
  const typedDataSegments: Array<{ byteOffset: bigint; byteLength: bigint }> = [];
  const int32DataRanges: Array<{ start: number; end: number }> = [];
  const externalData: OnnxExternalData = {};

  const markTypedData = (startOffset: number, endOffset: number) => {
    typedDataSegments.push({
      byteOffset: BigInt(startOffset),
      byteLength: BigInt(endOffset - startOffset)
    });
  };

  while (offset < end) {
    const tag = await readVarint(bytes, offset, end);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 0x7n);
    ensureTag(fieldNumber, wireType, offset);

    switch (fieldNumber) {
      case 1:
        if (wireType === 0) {
          const value = await readVarint(bytes, offset, end);
          dims.push(asSignedInt64(value.value));
          offset = value.next;
        } else if (wireType === 2) {
          const range = await readLengthDelimitedRange(bytes, offset, end);
          let packedOffset = range.valueStart;
          while (packedOffset < range.valueEnd) {
            const value = await readVarint(bytes, packedOffset, range.valueEnd);
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
        const value = await readVarint(bytes, offset, end);
        dataType = Number(value.value);
        offset = value.next;
        break;
      }
      case 4:
      case 5:
      case 6:
      case 7: {
        if (wireType === 2) {
          const range = await readLengthDelimitedRange(bytes, offset, end);
          if (fieldNumber === 4 && (range.valueEnd - range.valueStart) % 4 !== 0) {
            throw new ParseError("Packed float_data length must be a multiple of 4", BigInt(offset));
          }
          markTypedData(range.valueStart, range.valueEnd);
          if (fieldNumber === 5) {
            int32DataRanges.push({ start: range.valueStart, end: range.valueEnd });
          }
          offset = range.next;
        } else if (wireType === 0 && (fieldNumber === 5 || fieldNumber === 7)) {
          const valueStart = offset;
          const value = await readVarint(bytes, offset, end, false);
          markTypedData(valueStart, value.next);
          if (fieldNumber === 5) {
            int32DataRanges.push({ start: valueStart, end: value.next });
          }
          offset = value.next;
        } else if (wireType === 5 && fieldNumber === 4) {
          markTypedData(offset, offset + 4);
          ensureFixedWidth(offset, end, 4);
          offset += 4;
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
        const range = await readLengthDelimitedRange(bytes, offset, end);
        name = await bytes.string(range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 9: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, end);
        rawDataOffset = BigInt(range.valueStart);
        rawDataLength = BigInt(range.valueEnd - range.valueStart);
        offset = range.next;
        break;
      }
      case 10:
      case 11: {
        if (wireType === 2) {
          const range = await readLengthDelimitedRange(bytes, offset, end);
          if (fieldNumber === 10 && (range.valueEnd - range.valueStart) % 8 !== 0) {
            throw new ParseError("Packed double_data length must be a multiple of 8", BigInt(offset));
          }
          markTypedData(range.valueStart, range.valueEnd);
          offset = range.next;
        } else if (wireType === 1 && fieldNumber === 10) {
          markTypedData(offset, offset + 8);
          ensureFixedWidth(offset, end, 8);
          offset += 8;
        } else if (wireType === 0 && fieldNumber === 11) {
          const valueStart = offset;
          const value = await readVarint(bytes, offset, end, false);
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
        const range = await readLengthDelimitedRange(bytes, offset, end);
        mergeExternalData(externalData, await parseStringStringEntry(bytes, range.valueStart, range.valueEnd, depth + 1));
        offset = range.next;
        break;
      }
      case 14: {
        expectWireType(wireType, 0, offset);
        const value = await readVarint(bytes, offset, end);
        dataLocation = Number(value.value);
        offset = value.next;
        break;
      }
      default:
        offset = await skipField(bytes, offset, end, wireType);
        break;
    }
  }

  const inferredExternal =
    dataLocation === 1 ||
    externalData.location !== undefined ||
    externalData.offset !== undefined ||
    externalData.length !== undefined;
  const dataTypeName = ONNX_DATA_TYPE_NAMES[dataType] ?? `UNKNOWN(${dataType})`;
  const float6 = dataType === 27 || dataType === 28;
  if (float6 && !inferredExternal) {
    await validateFloat6Data(bytes, dims, rawDataOffset, rawDataLength, int32DataRanges);
  }
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
    id: `${fileId}:tensor:${start}`,
    name,
    fileId,
    dtype: dataTypeName,
    shape: dims,
    byteOffset,
    byteLength,
    sampleSupport: "metadata-only",
    ...(float6
      ? {
          encoding: {
            onnxDataField: inferredExternal
              ? "external_data"
              : rawDataOffset !== undefined
                ? "raw_data"
                : "int32_data"
          }
        }
      : {}),
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
      data_type: dataType,
      data_type_name: dataTypeName,
      data_location: dataLocation,
      data_location_name: dataLocationName,
      tensor_offset: BigInt(start),
      raw_data:
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
      external_data: externalData
    },
    tensor
  };
}

async function validateFloat6Data(
  bytes: OnnxReader,
  dims: bigint[],
  rawOffset: bigint | undefined,
  rawLength: bigint,
  int32Ranges: Array<{ start: number; end: number }>
): Promise<void> {
  if (dims.some((dimension) => dimension < 0n)) {
    throw new ParseError("ONNX FLOAT6 dimensions must be non-negative");
  }
  const elements = dims.reduce((count, dimension) => count * dimension, 1n);
  if (rawOffset !== undefined) {
    const expectedBytes = (elements * 6n + 7n) / 8n;
    if (rawLength !== expectedBytes) {
      throw new ParseError(`ONNX FLOAT6 raw_data requires ${expectedBytes} bytes, received ${rawLength}`);
    }
    const usedBits = Number((elements * 6n) % 8n);
    if (usedBits !== 0) {
      const lastByte = await bytes.byte(Number(rawOffset + rawLength - 1n));
      if ((lastByte >> usedBits) !== 0) {
        throw new ParseError("ONNX FLOAT6 raw_data has non-zero padding bits");
      }
    }
    return;
  }
  let count = 0n;
  for (const range of int32Ranges) {
    let offset = range.start;
    while (offset < range.end) {
      const value = await readVarint(bytes, offset, range.end, false);
      if (value.value > 63n) {
        throw new ParseError("ONNX FLOAT6 int32_data must contain unsigned 6-bit codes (0–63)", BigInt(offset));
      }
      count += 1n;
      offset = value.next;
    }
  }
  if (count !== elements) {
    throw new ParseError(`ONNX FLOAT6 int32_data requires ${elements} elements, received ${count}`);
  }
}

async function parseStringStringEntry(
  bytes: OnnxReader,
  start: number,
  end: number,
  depth: number
): Promise<Partial<Record<keyof OnnxExternalData, string | bigint>>> {
  ensureDepth(depth, start);
  let offset = start;
  let key = "";
  let value = "";

  while (offset < end) {
    const tag = await readVarint(bytes, offset, end);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 0x7n);
    ensureTag(fieldNumber, wireType, offset);

    switch (fieldNumber) {
      case 1: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, end);
        key = await bytes.string(range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      case 2: {
        expectWireType(wireType, 2, offset);
        const range = await readLengthDelimitedRange(bytes, offset, end);
        value = await bytes.string(range.valueStart, range.valueEnd);
        offset = range.next;
        break;
      }
      default:
        offset = await skipField(bytes, offset, end, wireType);
        break;
    }
  }

  switch (key) {
    case "location":
      return { location: value };
    case "offset": {
      const offsetValue = parseDecimalBigInt(value);
      return { offset: offsetValue };
    }
    case "length": {
      const lengthValue = parseDecimalBigInt(value);
      return { length: lengthValue };
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

function parseDecimalBigInt(value: string): bigint {
  if (!/^\d+$/.test(value)) throw new ParseError(`Invalid ONNX external data byte range: ${value}`);
  return BigInt(value);
}

async function readVarint(bytes: OnnxReader, start: number, end: number, metadata = true): Promise<ReadVarintResult> {
  let value = 0n;
  let shift = 0n;
  let offset = start;

  for (let index = 0; index < 10; index += 1) {
    if (offset >= end) {
      throw new ParseError("Truncated protobuf varint", BigInt(start));
    }
    const byte = await bytes.byte(offset++);
    if (metadata) bytes.charge(1);
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

async function readLengthDelimitedRange(
  bytes: OnnxReader,
  start: number,
  end: number
): Promise<LengthDelimitedRange> {
  const length = await readVarint(bytes, start, end);
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

async function skipField(bytes: OnnxReader, start: number, end: number, wireType: number): Promise<number> {
  switch (wireType) {
    case 0:
      return (await readVarint(bytes, start, end)).next;
    case 1:
      ensureFixedWidth(start, end, 8);
      bytes.charge(8);
      return start + 8;
    case 2: {
      const range = await readLengthDelimitedRange(bytes, start, end);
      bytes.charge(range.valueEnd - range.valueStart);
      return range.next;
    }
    case 5:
      ensureFixedWidth(start, end, 4);
      bytes.charge(4);
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
  if (!Number.isSafeInteger(fieldNumber) || fieldNumber <= 0 || fieldNumber > 0x1fffffff) {
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
