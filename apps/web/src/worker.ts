/// <reference lib="webworker" />

import {
  BrowserFileSource,
  detectFormat,
  GgufParser,
  loadModelUrl,
  loadSources,
  OnnxParser,
  SafeTensorsParser,
  type Parser,
  type RandomAccessSource,
  type TensorRecord,
  type WeightFormat
} from "@weights-viz/core";

const sources = new Map<string, RandomAccessSource>();
const parsers: Record<WeightFormat, Parser> = {
  safetensors: new SafeTensorsParser(),
  gguf: new GgufParser(),
  onnx: new OnnxParser()
};

self.onmessage = async (
  event: MessageEvent<
    | { id: number; type: "files"; files: File[] }
    | { id: number; type: "url"; url: string }
    | { id: number; type: "sample"; tensor: TensorRecord; maxValues: number }
  >
) => {
  const { id } = event.data;
  try {
    if (event.data.type === "files") {
      const incoming = event.data.files.map((file) => new BrowserFileSource(file));
      incoming.forEach((source) => sources.set(source.id, source));
      postMessage({ id, ok: true, result: await loadSources(incoming) });
      return;
    }
    if (event.data.type === "url") {
      const result = await loadModelUrl(event.data.url, undefined, (source) => {
        sources.set(source.id, source);
      }, (progress) => {
        postMessage({ id, type: "progress", progress });
      });
      postMessage({ id, ok: true, result });
      return;
    }
    const source = sources.get(event.data.tensor.fileId);
    if (!source) throw new Error("The source file is no longer available");
    const format = await detectFormat(source);
    const parser = parsers[format];
    if (!parser.sample) throw new Error("This format does not support value sampling");
    postMessage({
      id,
      ok: true,
      result: await parser.sample(source, event.data.tensor, event.data.maxValues)
    });
  } catch (error) {
    postMessage({
      id,
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

export {};
