import { bigintToSafeNumber } from "./binary";
import {
  fetchRemoteOnnx,
  HttpRangeSource,
  MemorySource
} from "./data-source";
import { detectFormat } from "./detect";
import { ParseError } from "./errors";
import { GgufParser } from "./parsers/gguf";
import { OnnxParser } from "./parsers/onnx";
import { SafeTensorsParser } from "./parsers/safetensors";
import {
  REMOTE_ONNX_MAX_BYTES,
  type ParsedFile,
  type ParsedModel,
  type Parser,
  type RandomAccessSource
} from "./types";

const parsers = {
  safetensors: new SafeTensorsParser(),
  gguf: new GgufParser(),
  onnx: new OnnxParser()
} satisfies Record<string, Parser>;

export async function parseSource(
  source: RandomAccessSource,
  signal?: AbortSignal
): Promise<ParsedFile> {
  const format = await detectFormat(source);
  return parsers[format].parse(source, signal ? { signal } : {});
}

export async function loadSources(
  sources: RandomAccessSource[],
  signal?: AbortSignal
): Promise<ParsedModel[]> {
  const indexSources = sources.filter((source) =>
    source.name.toLowerCase().endsWith(".safetensors.index.json")
  );
  const consumed = new Set<string>();
  const models: ParsedModel[] = [];

  for (const index of indexSources) {
    const model = await loadSafeTensorsIndex(
      index,
      new Map(sources.map((source) => [source.name, source])),
      signal
    );
    models.push(model);
    consumed.add(index.id);
    for (const file of model.files) consumed.add(file.id);
  }

  for (const source of sources) {
    if (consumed.has(source.id)) continue;
    const file = await parseSource(source, signal);
    models.push({
      id: `model-${source.id}`,
      name: source.name,
      files: [file],
      diagnostics: [...file.diagnostics]
    });
  }
  return models;
}

export async function loadModelUrl(
  rawUrl: string,
  signal?: AbortSignal,
  onSource?: (source: RandomAccessSource) => void
): Promise<ParsedModel[]> {
  const url = new URL(rawUrl).toString();
  const lowerPath = new URL(url).pathname.toLowerCase();
  if (lowerPath.endsWith(".safetensors.index.json")) {
    return [await loadRemoteSafeTensorsIndex(url, signal, onSource)];
  }
  if (lowerPath.endsWith(".onnx")) {
    const source = await fetchRemoteOnnx(url, REMOTE_ONNX_MAX_BYTES, signal);
    onSource?.(source);
    return loadSources([source], signal);
  }
  const source = await HttpRangeSource.create(url, signal ? { signal } : {});
  onSource?.(source);
  return loadSources([source], signal);
}

async function loadSafeTensorsIndex(
  indexSource: RandomAccessSource,
  candidates: Map<string, RandomAccessSource>,
  signal?: AbortSignal
): Promise<ParsedModel> {
  if (indexSource.size > 16n * 1024n * 1024n) {
    throw new ParseError("SafeTensors index exceeds the 16 MiB limit");
  }
  const bytes = await indexSource.read(
    0n,
    bigintToSafeNumber(indexSource.size, "Index size"),
    signal
  );
  const index = parseSafeTensorsIndex(bytes);
  const shardNames = [...new Set(Object.values(index.weight_map))];
  const files: ParsedFile[] = [];
  const diagnostics = [];
  for (const shardName of shardNames) {
    const source =
      candidates.get(shardName) ??
      candidates.get(shardName.split("/").pop() ?? shardName);
    if (!source) {
      diagnostics.push({
        severity: "error" as const,
        message: `Missing SafeTensors shard: ${shardName}`
      });
      continue;
    }
    files.push(await parsers.safetensors.parse(source, signal ? { signal } : {}));
  }
  return {
    id: `model-${indexSource.id}`,
    name: indexSource.name.replace(/\.safetensors\.index\.json$/i, ""),
    files,
    diagnostics
  };
}

async function loadRemoteSafeTensorsIndex(
  url: string,
  signal?: AbortSignal,
  onSource?: (source: RandomAccessSource) => void
): Promise<ParsedModel> {
  const response = await fetch(url, {
    mode: "cors",
    credentials: "omit",
    ...(signal ? { signal } : {})
  });
  if (!response.ok) {
    throw new ParseError(`Unable to fetch SafeTensors index: HTTP ${response.status}`);
  }
  const bytes = await readResponseCapped(response, 16 * 1024 * 1024, "SafeTensors index");
  const index = parseSafeTensorsIndex(bytes);
  const shardNames = [...new Set(Object.values(index.weight_map))];
  const files: ParsedFile[] = [];
  for (const shardName of shardNames) {
    const source = await HttpRangeSource.create(
      new URL(shardName, url).toString(),
      signal ? { signal } : {}
    );
    onSource?.(source);
    files.push(await parsers.safetensors.parse(source, signal ? { signal } : {}));
  }
  return {
    id: `model-url-${url}`,
    name: new URL(url).pathname.split("/").pop()?.replace(/\.safetensors\.index\.json$/i, "") ||
      "Remote SafeTensors",
    files,
    diagnostics: []
  };
}

async function readResponseCapped(
  response: Response,
  maxBytes: number,
  label: string
): Promise<Uint8Array> {
  const declared = response.headers.get("content-length");
  if (declared && Number(declared) > maxBytes) {
    await response.body?.cancel();
    throw new ParseError(`${label} exceeds the ${maxBytes} byte limit`);
  }
  if (!response.body) throw new ParseError(`${label} response has no body`);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  while (true) {
    const result = await reader.read();
    if (result.done) break;
    length += result.value.byteLength;
    if (length > maxBytes) {
      await reader.cancel();
      throw new ParseError(`${label} exceeds the ${maxBytes} byte limit`);
    }
    chunks.push(result.value);
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function parseSafeTensorsIndex(bytes: Uint8Array): {
  metadata?: Record<string, unknown>;
  weight_map: Record<string, string>;
} {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder().decode(bytes));
  } catch (error) {
    throw new ParseError(
      `Invalid SafeTensors index JSON: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  if (!value || typeof value !== "object") {
    throw new ParseError("SafeTensors index must be an object");
  }
  const weightMap = (value as Record<string, unknown>).weight_map;
  if (!weightMap || typeof weightMap !== "object" || Array.isArray(weightMap)) {
    throw new ParseError("SafeTensors index is missing weight_map");
  }
  for (const [tensor, file] of Object.entries(weightMap)) {
    if (!tensor || typeof file !== "string" || !file) {
      throw new ParseError("SafeTensors weight_map entries must map names to files");
    }
  }
  return value as {
    metadata?: Record<string, unknown>;
    weight_map: Record<string, string>;
  };
}

export function sourceFromBytes(name: string, bytes: Uint8Array): RandomAccessSource {
  return new MemorySource(name, bytes);
}
