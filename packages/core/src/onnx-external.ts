import { ParseError } from "./errors";
import type {
  Diagnostic,
  ParsedFile,
  ParsedModel,
  RandomAccessSource,
  TensorRecord
} from "./types";

export function onnxExternalLocations(file: ParsedFile): string[] {
  return [
    ...new Set(
      file.tensors
        .filter((tensor) => tensor.storage === "external")
        .map((tensor) => tensor.externalLocation)
        .filter((location): location is string => location !== undefined)
        .map(validateOnnxExternalLocation)
    )
  ];
}

export function validateOnnxExternalLocation(location: string): string {
  if (!location || /[\u0000-\u001f\u007f]/.test(location)) {
    throw new ParseError(
      "ONNX external data location is empty or contains control characters"
    );
  }
  let decoded: string;
  try {
    decoded = decodeURIComponent(location);
  } catch {
    throw new ParseError(`Invalid ONNX external data location: ${location}`);
  }
  if (
    decoded.startsWith("/") ||
    decoded.startsWith("\\") ||
    decoded.includes("\\") ||
    decoded.includes("?") ||
    decoded.includes("#") ||
    /^[a-z][a-z0-9+.-]*:/i.test(decoded)
  ) {
    throw new ParseError(
      `ONNX external data location must be a relative file path: ${location}`
    );
  }
  const segments = decoded.split("/");
  if (
    segments.some((segment) => !segment || segment === "." || segment === "..")
  ) {
    throw new ParseError(`Unsafe ONNX external data location: ${location}`);
  }
  return segments.join("/");
}

export function materializeOnnxModel(
  manifest: ParsedFile,
  sources: RandomAccessSource[]
): { model: ParsedModel; usedSourceIds: Set<string> } {
  const diagnostics: Diagnostic[] = [...manifest.diagnostics];
  const files: ParsedFile[] = [];
  const usedSourceIds = new Set<string>();
  const hasExternalTensors = manifest.tensors.some(
    (tensor) => tensor.storage === "external"
  );
  const inlineTensors = manifest.tensors.filter(
    (tensor) => tensor.storage !== "external"
  );
  if (!hasExternalTensors && inlineTensors.length > 0) {
    files.push({ ...manifest, tensors: inlineTensors });
  }

  const groups = new Map<string, TensorRecord[]>();
  for (const tensor of manifest.tensors) {
    if (tensor.storage !== "external") continue;
    if (!tensor.externalLocation) {
      diagnostics.push({
        severity: "error",
        message: `External ONNX tensor ${tensor.name || tensor.id} has no data location`
      });
      continue;
    }
    let location: string;
    try {
      location = validateOnnxExternalLocation(tensor.externalLocation);
    } catch (error) {
      diagnostics.push({
        severity: "error",
        message: error instanceof Error ? error.message : String(error)
      });
      continue;
    }
    const tensors = groups.get(location) ?? [];
    tensors.push(tensor);
    groups.set(location, tensors);
  }

  for (const [location, tensors] of groups) {
    const source = findOnnxExternalSource(location, sources);
    if (!source) {
      diagnostics.push({
        severity: "error",
        message: `Missing ONNX external data file: ${location}`
      });
      continue;
    }
    usedSourceIds.add(source.id);
    files.push({
      id: source.id,
      name: location,
      format: "onnx",
      size: source.size,
      metadata: {
        ...manifest.metadata,
        onnxContainer: manifest.name,
        onnxContainerSize: manifest.size,
        externalDataFile: location
      },
      tensors: mapExternalTensors(tensors, source, diagnostics),
      diagnostics: []
    });
  }

  return {
    model: {
      id: `model-${manifest.id}`,
      name: manifest.name,
      files,
      diagnostics
    },
    usedSourceIds
  };
}

function findOnnxExternalSource(
  location: string,
  sources: RandomAccessSource[]
): RandomAccessSource | undefined {
  const basename = location.split("/").pop() ?? location;
  const exact = sources.filter((source) => source.name === location);
  if (exact.length === 1) return exact[0];
  const byBasename = sources.filter((source) => source.name === basename);
  return byBasename.length === 1 ? byBasename[0] : undefined;
}

function mapExternalTensors(
  tensors: TensorRecord[],
  source: RandomAccessSource,
  diagnostics: Diagnostic[]
): TensorRecord[] {
  const sorted = [...tensors].sort((left, right) =>
    compareBigInt(left.externalOffset ?? 0n, right.externalOffset ?? 0n)
  );
  const mapped: TensorRecord[] = [];
  for (let index = 0; index < sorted.length; index += 1) {
    const tensor = sorted[index]!;
    const byteOffset = tensor.externalOffset ?? 0n;
    const nextOffset = sorted[index + 1]?.externalOffset;
    const inferred = inferOnnxTensorByteLength(tensor);
    const byteLength =
      tensor.externalLength ??
      inferred ??
      (nextOffset !== undefined && nextOffset >= byteOffset
        ? nextOffset - byteOffset
        : source.size - byteOffset);
    if (
      byteOffset < 0n ||
      byteLength < 0n ||
      byteOffset + byteLength > source.size
    ) {
      diagnostics.push({
        severity: "error",
        message: `ONNX tensor ${tensor.name || tensor.id} exceeds ${source.name}`
      });
      continue;
    }
    mapped.push({
      ...tensor,
      fileId: source.id,
      byteOffset,
      byteLength,
      dataOffset: byteOffset
    });
  }
  return mapped;
}

function inferOnnxTensorByteLength(tensor: TensorRecord): bigint | undefined {
  const bits = ONNX_DTYPE_BITS[tensor.dtype];
  if (bits === undefined || tensor.shape.some((dimension) => dimension < 0n)) {
    return undefined;
  }
  const elements = tensor.shape.reduce(
    (product, dimension) => product * dimension,
    1n
  );
  return (elements * BigInt(bits) + 7n) / 8n;
}

const ONNX_DTYPE_BITS: Readonly<Record<string, number>> = {
  FLOAT: 32,
  UINT8: 8,
  INT8: 8,
  UINT16: 16,
  INT16: 16,
  INT32: 32,
  INT64: 64,
  BOOL: 8,
  FLOAT16: 16,
  DOUBLE: 64,
  UINT32: 32,
  UINT64: 64,
  COMPLEX64: 64,
  COMPLEX128: 128,
  BFLOAT16: 16,
  FLOAT8E4M3FN: 8,
  FLOAT8E4M3FNUZ: 8,
  FLOAT8E5M2: 8,
  FLOAT8E5M2FNUZ: 8,
  UINT4: 4,
  INT4: 4,
  FLOAT4E2M1: 4,
  FLOAT8E8M0: 8,
  UINT2: 2,
  INT2: 2
};

function compareBigInt(left: bigint, right: bigint): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
