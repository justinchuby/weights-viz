import { bigintToSafeNumber } from "./binary";
import {
  fetchRemoteOnnx,
  HttpRangeSource,
  MemorySource
} from "./data-source";
import { detectFormat } from "./detect";
import { ParseError } from "./errors";
import { materializeOnnxModel } from "./onnx-external";
import { GgufParser } from "./parsers/gguf";
import { OnnxParser } from "./parsers/onnx";
import { SafeTensorsParser } from "./parsers/safetensors";
import {
  REMOTE_ONNX_MAX_BYTES,
  SAFETENSORS_INDEX_MAX_BYTES,
  type ParsedFile,
  type ParsedModel,
  type Parser,
  type RandomAccessSource
} from "./types";

export {
  onnxExternalLocations,
  validateOnnxExternalLocation
} from "./onnx-external";

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
    if (consumed.has(source.id) || !source.name.toLowerCase().endsWith(".onnx")) {
      continue;
    }
    const manifest = await parsers.onnx.parse(source, signal ? { signal } : {});
    const result = materializeOnnxModel(manifest, sources);
    models.push(result.model);
    consumed.add(source.id);
    result.usedSourceIds.forEach((id) => consumed.add(id));
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
  const url = normalizeModelUrl(rawUrl);
  const lowerPath = new URL(url).pathname.toLowerCase();
  if (lowerPath.endsWith(".safetensors.index.json")) {
    return [await loadRemoteSafeTensorsIndex(url, signal, onSource)];
  }
  if (lowerPath.endsWith(".onnx")) {
    return [
      await loadRemoteOnnx(url, signal, onSource)
    ];
  }
  const source = await HttpRangeSource.create(url, signal ? { signal } : {});
  onSource?.(source);
  return loadSources([source], signal);
}

async function loadRemoteOnnx(
  url: string,
  signal?: AbortSignal,
  onSource?: (source: RandomAccessSource) => void
): Promise<ParsedModel> {
  const manifestSource = await fetchRemoteOnnx(
    url,
    REMOTE_ONNX_MAX_BYTES,
    signal
  );
  onSource?.(manifestSource);
  const manifest = await parsers.onnx.parse(
    manifestSource,
    signal ? { signal } : {}
  );
  return materializeOnnxModel(manifest, []).model;
}

export function normalizeModelUrl(rawUrl: string): string {
  const url = new URL(rawUrl);
  if (url.hostname !== "huggingface.co" && url.hostname !== "www.huggingface.co") {
    return url.toString();
  }
  url.hostname = "huggingface.co";
  const segments = url.pathname.split("/").filter(Boolean);
  const routeIndex = segments.findIndex(
    (segment) => segment === "blob" || segment === "resolve"
  );
  if (routeIndex === -1) {
    throw new ParseError(
      "This is a Hugging Face repository page, not a model file. Open Files and versions, choose a .gguf, .safetensors, .safetensors.index.json, or .onnx file, then paste that file URL."
    );
  }
  if (routeIndex < 2 || routeIndex + 2 >= segments.length) {
    throw new ParseError("The Hugging Face file URL is incomplete");
  }
  if (segments[routeIndex] === "blob") segments[routeIndex] = "resolve";
  url.pathname = `/${segments.map(encodeURIComponentPreservingEscapes).join("/")}`;
  url.searchParams.delete("download");
  return url.toString();
}

function encodeURIComponentPreservingEscapes(segment: string): string {
  try {
    return encodeURIComponent(decodeURIComponent(segment));
  } catch {
    return encodeURIComponent(segment);
  }
}

async function loadSafeTensorsIndex(
  indexSource: RandomAccessSource,
  candidates: Map<string, RandomAccessSource>,
  signal?: AbortSignal
): Promise<ParsedModel> {
  if (indexSource.size > BigInt(SAFETENSORS_INDEX_MAX_BYTES)) {
    throw new ParseError("SafeTensors index exceeds the 64 MiB limit");
  }
  const bytes = await indexSource.read(
    0n,
    bigintToSafeNumber(indexSource.size, "Index size"),
    signal
  );
  const index = parseSafeTensorsIndex(bytes);
  const shardNames = orderedShardNames(index.weight_map);
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
  const bytes = await readResponseCapped(
    response,
    SAFETENSORS_INDEX_MAX_BYTES,
    "SafeTensors index"
  );
  const index = parseSafeTensorsIndex(bytes);
  const shardNames = orderedShardNames(index.weight_map);
  const files: ParsedFile[] = [];
  const concurrency = 8;
  for (let index = 0; index < shardNames.length; index += concurrency) {
    const outcomes = await Promise.allSettled(
      shardNames.slice(index, index + concurrency).map(async (shardName) => {
        const source = await HttpRangeSource.create(
          new URL(shardName, url).toString(),
          signal ? { signal } : {}
        );
        onSource?.(source);
        return parsers.safetensors.parse(source, signal ? { signal } : {});
      })
    );
    const failure = outcomes.find(
      (outcome): outcome is PromiseRejectedResult => outcome.status === "rejected"
    );
    if (failure) throw failure.reason;
    for (const outcome of outcomes) {
      if (outcome.status === "fulfilled") files.push(outcome.value);
    }
  }
  return {
    id: `model-url-${url}`,
    name: new URL(url).pathname.split("/").pop()?.replace(/\.safetensors\.index\.json$/i, "") ||
      "Remote SafeTensors",
    files,
    diagnostics: []
  };
}

function orderedShardNames(weightMap: Record<string, string>): string[] {
  return [...new Set(Object.values(weightMap))].sort((left, right) =>
    left.localeCompare(right, undefined, { numeric: true })
  );
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
