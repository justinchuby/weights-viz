import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  BookOpen,
  Box,
  Cpu,
  Search,
  Sparkles
} from "lucide-react";
import {
  GGUF_DTYPE_CATALOG,
  ONNX_DTYPE_CATALOG,
  SAFETENSORS_DTYPE_CATALOG,
  type DtypeCatalogEntry,
  type TensorRecord,
  type WeightFormat
} from "@weights-viz/core";
import { colorForDtype } from "./dtype-color";
import { createDtypeEducation, formatDecimal } from "./dtype-education";

interface DtypeAtlasProps {
  onBack: () => void;
  syncUrlState?: boolean;
  onExplain: (
    format: WeightFormat,
    tensor: TensorRecord,
    catalogMode: boolean
  ) => void;
}

type FormatFilter = "all" | WeightFormat;

const CATALOG = [
  ...SAFETENSORS_DTYPE_CATALOG,
  ...GGUF_DTYPE_CATALOG,
  ...ONNX_DTYPE_CATALOG
];

const FORMAT_DETAILS: Array<{
  format: WeightFormat;
  title: string;
  description: string;
}> = [
  {
    format: "safetensors",
    title: "SafeTensors",
    description:
      "A safe tensor container: dtype, shape, and byte ranges are explicit; quantization conventions may be carried by separate tensors."
  },
  {
    format: "gguf",
    title: "GGUF / GGML",
    description:
      "Inference-oriented scalar and block formats with scales, minima, codebooks, or shared exponents embedded directly beside packed weights."
  },
  {
    format: "onnx",
    title: "ONNX",
    description:
      "Graph tensor types from full precision through sub-byte values; scale and zero-point usually live in initializers or Q/DQ operators."
  }
];

export function DtypeAtlas({
  onBack,
  onExplain,
  syncUrlState = false
}: DtypeAtlasProps) {
  const [query, setQuery] = useState(() =>
    syncUrlState ? new URLSearchParams(window.location.search).get("dtypeQuery") ?? "" : ""
  );
  const [format, setFormat] = useState<FormatFilter>(() => {
    if (!syncUrlState) return "all";
    const value = new URLSearchParams(window.location.search).get("dtypeFormat");
    return value === "safetensors" || value === "gguf" || value === "onnx"
      ? value
      : "all";
  });
  const entries = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return CATALOG.filter((entry) => {
      if (format !== "all" && entry.format !== format) return false;
      if (!normalized) return true;
      const tensor = catalogTensor(entry);
      const lesson = createDtypeEducation(entry.format, tensor);
      return `${entry.dtype} ${lesson.family} ${lesson.summary}`
        .toLowerCase()
        .includes(normalized);
    });
  }, [format, query]);

  useEffect(() => {
    if (!syncUrlState) return;
    const pageUrl = new URL(window.location.href);
    if (query) pageUrl.searchParams.set("dtypeQuery", query);
    else pageUrl.searchParams.delete("dtypeQuery");
    if (format !== "all") pageUrl.searchParams.set("dtypeFormat", format);
    else pageUrl.searchParams.delete("dtypeFormat");
    window.history.replaceState(window.history.state, "", pageUrl);
  }, [format, query, syncUrlState]);

  return (
    <section className="wv-atlas">
      <header className="wv-atlas-header">
        <button className="wv-button wv-icon-label" type="button" onClick={onBack}>
          <ArrowLeft className="wv-icon" aria-hidden="true" />
          <span>Model map</span>
        </button>
        <div>
          <span className="wv-kicker">Interactive reference</span>
          <h1>Dtype Atlas</h1>
          <p>
            See how every supported weight type is encoded, quantized, and used
            during inference—no model file required.
          </p>
        </div>
        <div className="wv-atlas-count">
          <strong>{CATALOG.length}</strong>
          <span>format entries</span>
        </div>
      </header>

      <div className="wv-atlas-body">
        <div className="wv-atlas-primer">
          {FORMAT_DETAILS.map((detail) => (
            <button
              key={detail.format}
              type="button"
              className={format === detail.format ? "active" : ""}
              aria-pressed={format === detail.format}
              onClick={() => setFormat(detail.format)}
            >
              {detail.format === "gguf" ? (
                <Cpu aria-hidden="true" />
              ) : detail.format === "onnx" ? (
                <Sparkles aria-hidden="true" />
              ) : (
                <Box aria-hidden="true" />
              )}
              <span>
                <strong>{detail.title}</strong>
                <small>{detail.description}</small>
              </span>
              <b>
                {CATALOG.filter((entry) => entry.format === detail.format).length}
              </b>
            </button>
          ))}
        </div>

        <div className="wv-atlas-toolbar">
          <label>
            <Search aria-hidden="true" />
            <input
              type="search"
              name="dtype-query"
              autoComplete="off"
              value={query}
              placeholder="Search dtype, family, or concept…"
              aria-label="Search dtype atlas"
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <div className="wv-atlas-filters" aria-label="Filter dtype format">
            {(["all", "safetensors", "gguf", "onnx"] as const).map((value) => (
              <button
                key={value}
                type="button"
                className={format === value ? "active" : ""}
                aria-pressed={format === value}
                onClick={() => setFormat(value)}
              >
                {value === "all" ? "All" : formatLabel(value)}
              </button>
            ))}
          </div>
          <span>{entries.length} shown</span>
        </div>

        {FORMAT_DETAILS.map((detail) => {
          const formatEntries = entries.filter(
            (entry) => entry.format === detail.format
          );
          if (!formatEntries.length) return null;
          return (
            <section className="wv-atlas-group" key={detail.format}>
              <header>
                <div>
                  <span>{detail.title}</span>
                  <p>{formatSubtitle(detail.format)}</p>
                </div>
                <b>{formatEntries.length}</b>
              </header>
              <div className="wv-atlas-grid">
                {formatEntries.map((entry) => (
                  <DtypeCard
                    key={`${entry.format}:${entry.dtype}`}
                    entry={entry}
                    onOpen={() =>
                      onExplain(entry.format, catalogTensor(entry), true)
                    }
                  />
                ))}
              </div>
            </section>
          );
        })}

        {!entries.length && (
          <div className="wv-atlas-empty">
            No dtype matches “{query.trim()}”.
          </div>
        )}
      </div>
    </section>
  );
}

function DtypeCard({
  entry,
  onOpen
}: {
  entry: DtypeCatalogEntry;
  onOpen: () => void;
}) {
  const lesson = createDtypeEducation(entry.format, catalogTensor(entry));
  const geometry = lesson.block
    ? `${lesson.block.elements} values / ${lesson.block.bytes} B`
    : lesson.bitsPerValue
      ? `${formatDecimal(lesson.bitsPerValue)} bits / value`
      : "format-defined layout";
  return (
    <button className="wv-atlas-card" type="button" onClick={onOpen}>
      <span
        className="wv-atlas-swatch"
        style={{ background: colorForDtype(entry.dtype) }}
      />
      <span className="wv-atlas-card-main">
        <span>
          <strong>{entry.dtype}</strong>
          {entry.typeId !== undefined && <small>ID {entry.typeId}</small>}
        </span>
        <b>{lesson.family}</b>
        <small>{geometry}</small>
      </span>
      <BookOpen aria-hidden="true" />
    </button>
  );
}

function catalogTensor(entry: DtypeCatalogEntry): TensorRecord {
  const elements = BigInt(entry.blockElements ?? 1);
  const byteLength = BigInt(
    entry.blockBytes ??
      entry.scalarBytes ??
      Math.max(1, Math.ceil((entry.bitsPerValue ?? 8) / 8))
  );
  return {
    id: `catalog:${entry.format}:${entry.dtype}`,
    name: `${entry.dtype} reference`,
    fileId: `catalog:${entry.format}`,
    dtype: entry.dtype,
    shape: [elements],
    byteOffset: 0n,
    byteLength,
    encoding: {
      ...(entry.typeId !== undefined ? { typeId: entry.typeId } : {}),
      ...(entry.scalarBytes !== undefined
        ? { scalarBytes: entry.scalarBytes }
        : {}),
      ...(entry.blockBytes !== undefined ? { blockBytes: entry.blockBytes } : {}),
      ...(entry.blockElements !== undefined
        ? { blockElements: entry.blockElements }
        : {})
    },
    sampleSupport: entry.sampleSupport
  };
}

function formatLabel(format: WeightFormat): string {
  if (format === "safetensors") return "SafeTensors";
  return format.toUpperCase();
}

function formatSubtitle(format: WeightFormat): string {
  if (format === "gguf") return "Scalar, block, codebook, ternary, and microscaled types";
  if (format === "onnx") return "TensorProto data types and graph-level quantization";
  return "Contiguous tensor payload dtypes";
}
