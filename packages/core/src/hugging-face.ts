import { downloadFile, listFiles } from "@huggingface/hub";
import { bigintToSafeNumber } from "./binary";
import { assertRange, ParseError } from "./errors";
import type { RandomAccessSource, WeightFormat } from "./types";

export interface HuggingFaceFileLocation {
  repo: string;
  revision: string;
  path: string;
}

export interface HuggingFaceModelFile {
  path: string;
  size: number;
  format: WeightFormat;
  url: string;
}

interface HuggingFaceSourceOptions {
  signal?: AbortSignal;
  fetch?: typeof globalThis.fetch;
}

let nextHuggingFaceSourceId = 1;

export class HuggingFaceHubSource implements RandomAccessSource {
  readonly id = `hf-${nextHuggingFaceSourceId++}`;
  readonly size: bigint;
  readonly name: string;

  constructor(
    readonly url: string,
    private readonly blob: Blob,
    path?: string
  ) {
    this.size = BigInt(blob.size);
    this.name = decodeURIComponent(path?.split("/").pop() || "remote");
  }

  static async create(
    url: string,
    options: HuggingFaceSourceOptions = {}
  ): Promise<HuggingFaceHubSource> {
    const location = parseHuggingFaceFileUrl(url);
    if (!location) throw new ParseError("Not a Hugging Face model file URL");
    const fetcher = withAbortSignal(
      options.fetch ?? globalThis.fetch.bind(globalThis),
      options.signal
    );
    try {
      const blob = await downloadFile({
        repo: { type: "model", name: location.repo },
        path: location.path,
        revision: location.revision,
        fetch: fetcher
      });
      if (!blob) throw new ParseError("Hugging Face could not find this file");
      return new HuggingFaceHubSource(url, blob, location.path);
    } catch (error) {
      if (error instanceof ParseError) throw error;
      const detail = error instanceof Error ? error.message : String(error);
      throw new ParseError(`Unable to open this Hugging Face file. ${detail}`);
    }
  }

  async read(
    offset: bigint,
    length: number,
    signal?: AbortSignal
  ): Promise<Uint8Array> {
    signal?.throwIfAborted();
    assertRange(offset, BigInt(length), this.size, "Hugging Face read");
    if (length === 0) return new Uint8Array();
    const start = bigintToSafeNumber(offset, "Hugging Face file offset");
    const bytes = await this.blob.slice(start, start + length).arrayBuffer();
    signal?.throwIfAborted();
    if (bytes.byteLength !== length) {
      throw new ParseError(
        `Expected ${length} bytes from Hugging Face, received ${bytes.byteLength}`,
        offset
      );
    }
    return new Uint8Array(bytes);
  }
}

export function parseHuggingFaceFileUrl(
  rawUrl: string
): HuggingFaceFileLocation | undefined {
  const url = new URL(rawUrl);
  if (!isHuggingFaceHost(url.hostname)) return undefined;
  const segments = url.pathname.split("/").filter(Boolean).map(decodeUrlSegment);
  const routeIndex = segments.findIndex(
    (segment) => segment === "blob" || segment === "resolve"
  );
  if (routeIndex !== 2 || segments.length < 5) return undefined;
  return {
    repo: `${segments[0]}/${segments[1]}`,
    revision: segments[3]!,
    path: segments.slice(4).join("/")
  };
}

export function parseHuggingFaceRepository(input: string): string {
  return parseHuggingFaceRepositoryLocation(input).repo;
}

function parseHuggingFaceRepositoryLocation(input: string): {
  repo: string;
  revision: string;
} {
  const trimmed = input.trim().replace(/\/+$/, "");
  if (/^[^/\s]+\/[^/\s]+$/.test(trimmed)) {
    return { repo: trimmed, revision: "main" };
  }
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new ParseError("Enter a Hugging Face repository as owner/repo");
  }
  if (!isHuggingFaceHost(url.hostname)) {
    throw new ParseError("Enter a huggingface.co model repository");
  }
  const segments = url.pathname.split("/").filter(Boolean).map(decodeUrlSegment);
  if (segments.length < 2 || segments[0] === "datasets" || segments[0] === "spaces") {
    throw new ParseError("Enter a Hugging Face model repository");
  }
  return {
    repo: `${segments[0]}/${segments[1]}`,
    revision: segments[2] === "tree" && segments[3] ? segments[3] : "main"
  };
}

export async function listHuggingFaceModelFiles(
  input: string,
  signal?: AbortSignal,
  fetcher: typeof globalThis.fetch = globalThis.fetch.bind(globalThis)
): Promise<HuggingFaceModelFile[]> {
  const { repo, revision } = parseHuggingFaceRepositoryLocation(input);
  const files: HuggingFaceModelFile[] = [];
  try {
    for await (const entry of listFiles({
      repo: { type: "model", name: repo },
      recursive: true,
      revision,
      fetch: withAbortSignal(fetcher, signal)
    })) {
      if (entry.type !== "file") continue;
      const format = supportedFormat(entry.path);
      if (!format) continue;
      files.push({
        path: entry.path,
        size: entry.size,
        format,
        url: huggingFaceFileUrl(repo, entry.path, revision)
      });
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new ParseError(`Unable to list ${repo}. ${detail}`);
  }
  return files.sort((left, right) =>
    left.path.localeCompare(right.path, undefined, { numeric: true })
  );
}

export function huggingFaceFileUrl(
  repo: string,
  path: string,
  revision = "main"
): string {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/");
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${encodedPath}`;
}

function supportedFormat(path: string): WeightFormat | undefined {
  const lower = path.toLowerCase();
  if (lower.endsWith(".gguf")) return "gguf";
  if (lower.endsWith(".safetensors") || lower.endsWith(".safetensors.index.json")) {
    return "safetensors";
  }
  if (lower.endsWith(".onnx")) return "onnx";
  return undefined;
}

function isHuggingFaceHost(hostname: string): boolean {
  return hostname === "huggingface.co" || hostname === "www.huggingface.co";
}

function decodeUrlSegment(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function withAbortSignal(
  fetcher: typeof globalThis.fetch,
  signal?: AbortSignal
): typeof globalThis.fetch {
  if (!signal) return fetcher;
  return (input, init = {}) =>
    fetcher(input, {
      ...init,
      signal: init.signal ?? signal
    });
}
