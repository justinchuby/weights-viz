import type { RandomAccessSource, WeightFormat } from "./types";

export async function detectFormat(source: RandomAccessSource): Promise<WeightFormat> {
  const lower = source.name.toLowerCase();
  if (lower.endsWith(".safetensors")) return "safetensors";
  if (lower.endsWith(".gguf")) return "gguf";
  if (lower.endsWith(".onnx")) return "onnx";
  const prefix = await source.read(0n, Number(source.size < 8n ? source.size : 8n));
  if (
    prefix.length >= 4 &&
    prefix[0] === 0x47 &&
    prefix[1] === 0x47 &&
    prefix[2] === 0x55 &&
    prefix[3] === 0x46
  ) {
    return "gguf";
  }
  if (prefix.length >= 8) {
    const headerLength = new DataView(prefix.buffer, prefix.byteOffset, 8).getBigUint64(0, true);
    if (headerLength > 1n && headerLength < source.size) return "safetensors";
  }
  return "onnx";
}
