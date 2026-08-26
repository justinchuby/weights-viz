import { assertRange, ParseError } from "./errors";
import type { RandomAccessSource } from "./types";

let nextSourceId = 1;

export class MemorySource implements RandomAccessSource {
  readonly id: string;
  readonly size: bigint;

  constructor(
    readonly name: string,
    private readonly bytes: Uint8Array,
    id?: string
  ) {
    this.id = id ?? `memory-${nextSourceId++}`;
    this.size = BigInt(bytes.byteLength);
  }

  async read(offset: bigint, length: number): Promise<Uint8Array> {
    assertRange(offset, BigInt(length), this.size, "Read");
    const start = Number(offset);
    return this.bytes.slice(start, start + length);
  }
}

export class BrowserFileSource implements RandomAccessSource {
  readonly id: string;
  readonly size: bigint;

  constructor(readonly file: File) {
    this.id = `file-${nextSourceId++}`;
    this.size = BigInt(file.size);
  }

  get name(): string {
    return this.file.name;
  }

  async read(
    offset: bigint,
    length: number,
    signal?: AbortSignal
  ): Promise<Uint8Array> {
    signal?.throwIfAborted();
    assertRange(offset, BigInt(length), this.size, "Read");
    const start = Number(offset);
    const buffer = await this.file.slice(start, start + length).arrayBuffer();
    signal?.throwIfAborted();
    return new Uint8Array(buffer);
  }
}

interface HttpSourceOptions {
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
}

export class HttpRangeSource implements RandomAccessSource {
  readonly id: string;
  readonly name: string;
  readonly size: bigint;
  readonly url: string;
  private readonly fetcher: typeof globalThis.fetch;
  private readonly cache = new Map<string, Uint8Array>();

  private constructor(url: string, size: bigint, fetcher: typeof globalThis.fetch) {
    this.id = `url-${nextSourceId++}`;
    this.url = url;
    this.name = decodeURIComponent(new URL(url).pathname.split("/").pop() || "remote");
    this.size = size;
    this.fetcher = fetcher;
  }

  static async create(url: string, options: HttpSourceOptions = {}): Promise<HttpRangeSource> {
    const fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
    let response: Response;
    try {
      response = await fetcher(url, {
        headers: { Range: "bytes=0-0" },
        mode: "cors",
        credentials: "omit",
        ...(options.signal ? { signal: options.signal } : {})
      });
    } catch (error) {
      throw remoteFetchError(url, error);
    }
    if ((response.status === 401 || response.status === 403) && isHuggingFaceUrl(url)) {
      throw new ParseError(
        "Hugging Face denied this file. Gated and private repositories require authentication, which is not sent by this static app. Download the file locally and open it instead."
      );
    }
    if (response.status === 404 && isHuggingFaceUrl(url)) {
      throw new ParseError(
        "Hugging Face could not find this file. Check the repository, revision, and filename."
      );
    }
    if (response.status !== 206) {
      throw new ParseError(
        `Server must support CORS HTTP Range requests (expected 206, received ${response.status})`
      );
    }
    const contentRange = response.headers.get("content-range");
    const match = /^bytes 0-0\/(\d+)$/.exec(contentRange ?? "");
    if (!match?.[1]) {
      throw new ParseError("Server returned an invalid or hidden Content-Range header");
    }
    return new HttpRangeSource(url, BigInt(match[1]), fetcher);
  }

  async read(
    offset: bigint,
    length: number,
    signal?: AbortSignal
  ): Promise<Uint8Array> {
    assertRange(offset, BigInt(length), this.size, "Remote read");
    if (length === 0) return new Uint8Array();
    const key = `${offset}:${length}`;
    const cached = this.cache.get(key);
    if (cached) return cached.slice();
    const end = offset + BigInt(length) - 1n;
    let response: Response;
    try {
      response = await this.fetcher(this.url, {
        headers: { Range: `bytes=${offset}-${end}` },
        mode: "cors",
        credentials: "omit",
        ...(signal ? { signal } : {})
      });
    } catch (error) {
      throw remoteFetchError(this.url, error);
    }
    if (response.status !== 206) {
      throw new ParseError(`Range request failed with HTTP ${response.status}`, offset);
    }

    const expected = `bytes ${offset}-${end}/${this.size}`;
    if (response.headers.get("content-range") !== expected) {
      throw new ParseError(
        `Unexpected Content-Range: ${response.headers.get("content-range") ?? "missing"}`,
        offset
      );
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength !== length) {
      throw new ParseError(`Expected ${length} bytes, received ${bytes.byteLength}`, offset);
    }
    if (this.cache.size >= 24) this.cache.delete(this.cache.keys().next().value as string);
    this.cache.set(key, bytes);
    return bytes.slice();
  }
}

function isHuggingFaceUrl(url: string): boolean {
  return new URL(url).hostname === "huggingface.co";
}

function remoteFetchError(url: string, cause: unknown): ParseError {
  const detail = cause instanceof Error ? cause.message : String(cause);
  if (isHuggingFaceUrl(url)) {
    return new ParseError(
      `The browser blocked the Hugging Face request. Public /resolve/ file URLs normally support CORS and byte ranges; gated/private files and restrictive embedded browsers do not. Download the file locally or open this app in a normal browser. (${detail})`
    );
  }
  return new ParseError(
    `The browser blocked the remote request. The server must allow CORS and HTTP Range requests. (${detail})`
  );
}

export async function fetchRemoteOnnx(
  url: string,
  maxBytes: number,
  signal?: AbortSignal,
  fetcher: typeof globalThis.fetch = globalThis.fetch.bind(globalThis)
): Promise<MemorySource> {
  const response = await fetcher(url, {
    mode: "cors",
    credentials: "omit",
    ...(signal ? { signal } : {})
  });
  if (!response.ok || !response.body) {
    throw new ParseError(`Unable to download ONNX file: HTTP ${response.status}`);
  }
  const declared = response.headers.get("content-length");
  if (declared && Number(declared) > maxBytes) {
    await response.body.cancel();
    throw new ParseError(`Remote ONNX exceeds the ${maxBytes} byte limit`);
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  while (true) {
    const result = await reader.read();
    if (result.done) break;
    length += result.value.byteLength;
    if (length > maxBytes) {
      await reader.cancel();
      throw new ParseError(`Remote ONNX exceeds the ${maxBytes} byte limit`);
    }
    chunks.push(result.value);
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const name = decodeURIComponent(new URL(url).pathname.split("/").pop() || "remote.onnx");
  return new MemorySource(name, bytes);
}
