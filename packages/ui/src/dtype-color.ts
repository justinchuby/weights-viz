const FALLBACK_PALETTE = [
  "#6ee7ff",
  "#9b8cff",
  "#ff7ab6",
  "#ffb45e",
  "#69e6a6",
  "#e6df69",
  "#7aa7ff",
  "#de8cff"
];

export function colorForDtype(dtype: string): string {
  const normalized = dtype.toUpperCase().replaceAll("-", "_");
  if (normalized === "BF16" || normalized === "BFLOAT16") return "#6ee7ff";
  if (["F16", "FLOAT16", "HALF"].includes(normalized)) return "#7aa7ff";
  if (["F32", "FLOAT", "FLOAT32"].includes(normalized)) return "#69e6a6";
  if (["F64", "DOUBLE", "FLOAT64"].includes(normalized)) return "#e6df69";
  if (normalized.startsWith("F8_") || normalized.startsWith("FLOAT8")) {
    return "#9b8cff";
  }
  if (["I8", "U8", "INT8", "UINT8"].includes(normalized)) return "#ff7ab6";
  if (["I16", "U16", "INT16", "UINT16"].includes(normalized)) return "#ffb45e";
  if (["I32", "U32", "INT32", "UINT32"].includes(normalized)) return "#de8cff";
  if (["I64", "U64", "INT64", "UINT64"].includes(normalized)) return "#9b8cff";
  if (normalized === "BOOL") return "#708897";
  return FALLBACK_PALETTE[stableHash(normalized) % FALLBACK_PALETTE.length]!;
}

function stableHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
