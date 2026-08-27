export type WeightFormat = "safetensors" | "gguf" | "onnx";

export interface DtypeCatalogEntry {
  format: WeightFormat;
  dtype: string;
  typeId?: number;
  bitsPerValue?: number;
  scalarBytes?: number;
  blockBytes?: number;
  blockElements?: number;
  sampleSupport: TensorRecord["sampleSupport"];
}

export interface Diagnostic {
  severity: "info" | "warning" | "error";
  message: string;
  offset?: bigint;
}

export interface TensorRecord {
  id: string;
  name: string;
  fileId: string;
  dtype: string;
  shape: bigint[];
  byteOffset: bigint;
  byteLength: bigint;
  byteSegments?: Array<{ byteOffset: bigint; byteLength: bigint }>;
  dataOffset?: bigint;
  storage?: "inline" | "external";
  externalLocation?: string;
  externalOffset?: bigint;
  externalLength?: bigint;
  encoding?: Record<string, string | number | boolean>;
  sampleSupport: "values" | "metadata-only" | "unsupported";
}

export interface ParsedFile {
  id: string;
  name: string;
  format: WeightFormat;
  size: bigint;
  metadata: Record<string, unknown>;
  tensors: TensorRecord[];
  diagnostics: Diagnostic[];
}

export interface ParsedModel {
  id: string;
  name: string;
  files: ParsedFile[];
  diagnostics: Diagnostic[];
}

export interface TensorSample {
  values: number[];
  sampledElements: number;
  totalElements: bigint;
  min: number;
  max: number;
  mean: number;
}

export interface ParseOptions {
  signal?: AbortSignal;
  maxMetadataBytes?: number;
}

export interface Parser {
  readonly format: WeightFormat;
  parse(source: RandomAccessSource, options?: ParseOptions): Promise<ParsedFile>;
  sample?(
    source: RandomAccessSource,
    tensor: TensorRecord,
    maxValues: number,
    signal?: AbortSignal
  ): Promise<TensorSample>;
}

export interface RandomAccessSource {
  readonly id: string;
  readonly name: string;
  readonly size: bigint;
  readonly url?: string;
  read(offset: bigint, length: number, signal?: AbortSignal): Promise<Uint8Array>;
}

export const DEFAULT_MAX_METADATA_BYTES = 64 * 1024 * 1024;
export const REMOTE_ONNX_MAX_BYTES = 50 * 1024 * 1024;
export const SAFETENSORS_INDEX_MAX_BYTES = 64 * 1024 * 1024;
