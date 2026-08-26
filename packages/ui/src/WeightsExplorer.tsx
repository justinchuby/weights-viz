import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ParsedFile,
  ParsedModel,
  TensorRecord,
  TensorSample
} from "@weights-viz/core";
import { formatAddress, formatBytes, formatShape } from "./format";

interface WeightsExplorerProps {
  models: ParsedModel[];
  busy?: boolean;
  error?: string;
  onChooseFiles?: () => void;
  onOpenUrl?: (url: string) => void;
  onSample?: (tensor: TensorRecord) => Promise<TensorSample>;
  intro?: string;
}

interface LayoutCell {
  file: ParsedFile;
  tensor: TensorRecord;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface HoverInfo {
  cell: LayoutCell;
  clientX: number;
  clientY: number;
  chunk?: { start: bigint; end: bigint };
}

const PALETTE = [
  "#6ee7ff",
  "#9b8cff",
  "#ff7ab6",
  "#ffb45e",
  "#69e6a6",
  "#e6df69",
  "#7aa7ff",
  "#de8cff"
];

export function WeightsExplorer({
  models,
  busy = false,
  error,
  onChooseFiles,
  onOpenUrl,
  onSample,
  intro
}: WeightsExplorerProps) {
  const [activeModelId, setActiveModelId] = useState<string>();
  const [selected, setSelected] = useState<TensorRecord>();
  const [sample, setSample] = useState<TensorSample>();
  const [sampleError, setSampleError] = useState<string>();
  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const activeModel =
    models.find((model) => model.id === activeModelId) ?? models[0];

  useEffect(() => {
    if (activeModel && activeModel.id !== activeModelId) {
      setActiveModelId(activeModel.id);
      setSelected(undefined);
    }
  }, [activeModel, activeModelId]);

  const visibleModel = useMemo(() => {
    if (!activeModel || !query.trim()) return activeModel;
    const needle = query.trim().toLowerCase();
    return {
      ...activeModel,
      files: activeModel.files.map((file) => ({
        ...file,
        tensors: file.tensors.filter(
          (tensor) =>
            tensor.name.toLowerCase().includes(needle) ||
            tensor.dtype.toLowerCase().includes(needle)
        )
      }))
    };
  }, [activeModel, query]);

  const requestSample = async () => {
    if (!selected || !onSample) return;
    setSample(undefined);
    setSampleError(undefined);
    try {
      setSample(await onSample(selected));
    } catch (reason) {
      setSampleError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  return (
    <main className="wv-shell">
      <header className="wv-header">
        <div>
          <div className="wv-kicker">LOCAL-FIRST MODEL INSPECTOR</div>
          <h1>Weights <span>Viz</span></h1>
        </div>
        <div className="wv-actions">
          {onChooseFiles && (
            <button className="wv-button primary" onClick={onChooseFiles}>
              Open files
            </button>
          )}
          {onOpenUrl && (
            <form
              className="wv-url"
              onSubmit={(event) => {
                event.preventDefault();
                if (url.trim()) onOpenUrl(url.trim());
              }}
            >
              <input
                aria-label="Model URL"
                type="url"
                placeholder="https://…/model.gguf"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
              />
              <button className="wv-button" type="submit">Load URL</button>
            </form>
          )}
        </div>
      </header>

      {error && <div className="wv-alert error">{error}</div>}
      {busy && <div className="wv-progress"><span /></div>}

      {!activeModel ? (
        <section className="wv-empty">
          <div className="wv-empty-grid" />
          <div className="wv-empty-content">
            <div className="wv-file-mark">01</div>
            <h2>See what your model is made of.</h2>
            <p>
              {intro ??
                "Drop GGUF, SafeTensors, or ONNX files here. Files stay on this device; remote models use byte-range requests."}
            </p>
            {onChooseFiles && (
              <button className="wv-button primary large" onClick={onChooseFiles}>
                Choose model files
              </button>
            )}
          </div>
        </section>
      ) : (
        <section className="wv-workspace">
          <aside className="wv-sidebar">
            <label className="wv-label">Models</label>
            <div className="wv-model-list">
              {models.map((model) => (
                <button
                  key={model.id}
                  className={model.id === activeModel.id ? "active" : ""}
                  onClick={() => setActiveModelId(model.id)}
                >
                  <span>{model.name}</span>
                  <small>{model.files.length} file{model.files.length === 1 ? "" : "s"}</small>
                </button>
              ))}
            </div>
            <label className="wv-label" htmlFor="tensor-filter">Filter tensors</label>
            <input
              id="tensor-filter"
              className="wv-filter"
              placeholder="name or dtype"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <div className="wv-summary">
              <SummaryRow label="Files" value={String(activeModel.files.length)} />
              <SummaryRow
                label="Tensors"
                value={String(activeModel.files.reduce((sum, file) => sum + file.tensors.length, 0))}
              />
              <SummaryRow
                label="Total size"
                value={formatBytes(activeModel.files.reduce((sum, file) => sum + file.size, 0n))}
              />
            </div>
            <div className="wv-legend">
              <label className="wv-label">Data types</label>
              {[...new Set(activeModel.files.flatMap((file) => file.tensors.map((tensor) => tensor.dtype)))]
                .slice(0, 8)
                .map((dtype, index) => (
                  <span key={dtype}>
                    <i style={{ background: PALETTE[index % PALETTE.length] }} />
                    {dtype}
                  </span>
                ))}
            </div>
          </aside>

          <div className="wv-map-panel">
            <div className="wv-panel-title">
              <div>
                <span>BYTE MAP</span>
                <h2>{activeModel.name}</h2>
              </div>
              <p>Area represents on-disk bytes · click a tensor to inspect</p>
            </div>
            {visibleModel && (
              <WeightMap
                model={visibleModel}
                {...(selected ? { selected } : {})}
                onSelect={(tensor) => {
                  setSelected(tensor);
                  setSample(undefined);
                  setSampleError(undefined);
                }}
              />
            )}
          </div>

          <aside className="wv-inspector">
            {selected ? (
              <>
                <div className="wv-tensor-type">{selected.dtype}</div>
                <h2>{selected.name}</h2>
                <dl>
                  <Detail label="Shape" value={formatShape(selected.shape)} />
                  <Detail label="Size" value={formatBytes(selected.byteLength)} />
                  <Detail
                    label={selected.storage === "external" ? "External start" : "Start"}
                    value={formatAddress(selected.byteOffset)}
                    mono
                  />
                  <Detail
                    label={selected.storage === "external" ? "External end" : "End"}
                    value={formatAddress(tensorEnd(selected))}
                    mono
                  />
                  <Detail label="Storage" value={selected.storage ?? "inline"} />
                  {selected.byteSegments && selected.byteSegments.length > 1 && (
                    <Detail
                      label="Payload ranges"
                      value={selected.byteSegments
                        .map(
                          (segment) =>
                            `${formatAddress(segment.byteOffset)}–${formatAddress(
                              segment.byteOffset + segment.byteLength
                            )}`
                        )
                        .join(", ")}
                      mono
                    />
                  )}
                  {selected.externalLocation && (
                    <Detail label="External file" value={selected.externalLocation} />
                  )}
                </dl>
                {selected.sampleSupport === "values" && onSample && (
                  <button className="wv-button primary full" onClick={() => void requestSample()}>
                    Sample values
                  </button>
                )}
                {selected.sampleSupport !== "values" && (
                  <div className="wv-alert">
                    {selected.sampleSupport === "metadata-only"
                      ? "This format is displayed as metadata only."
                      : "Value decoding is not available for this data type."}
                  </div>
                )}
                {sampleError && <div className="wv-alert error">{sampleError}</div>}
                {sample && (
                  <div className="wv-sample">
                    <label className="wv-label">Sample statistics</label>
                    <div className="wv-stat-grid">
                      <b>{sample.min.toPrecision(5)}<small>MIN</small></b>
                      <b>{sample.max.toPrecision(5)}<small>MAX</small></b>
                      <b>{sample.mean.toPrecision(5)}<small>MEAN</small></b>
                    </div>
                    <pre>{sample.values.slice(0, 32).map((value) => value.toPrecision(5)).join("\n")}</pre>
                  </div>
                )}
              </>
            ) : (
              <div className="wv-inspector-empty">
                <span>↖</span>
                Select a tensor to see its exact range, shape, storage, and values.
              </div>
            )}
          </aside>
        </section>
      )}
    </main>
  );
}

function WeightMap({
  model,
  selected,
  onSelect
}: {
  model: ParsedModel;
  selected?: TensorRecord;
  onSelect: (tensor: TensorRecord) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const [hover, setHover] = useState<HoverInfo>();
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const cells = useMemo(
    () => layoutModel(model, size.width, size.height),
    [model, size]
  );

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) {
        setSize({
          width: Math.max(320, Math.floor(entry.contentRect.width)),
          height: Math.max(360, Math.floor(entry.contentRect.height))
        });
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size.width * ratio);
    canvas.height = Math.floor(size.height * ratio);
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, size.width, size.height);
    context.save();
    context.translate(offset.x, offset.y);
    context.scale(zoom, zoom);
    drawCells(context, cells, selected, zoom);
    context.restore();
  }, [cells, offset, selected, size, zoom]);

  const hitTest = (clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return undefined;
    const x = (clientX - rect.left - offset.x) / zoom;
    const y = (clientY - rect.top - offset.y) / zoom;
    return cells.find(
      (cell) =>
        x >= cell.x &&
        x <= cell.x + cell.width &&
        y >= cell.y &&
        y <= cell.y + cell.height
    );
  };

  return (
    <div className="wv-canvas-wrap" ref={containerRef}>
      <canvas
        ref={canvasRef}
        onClick={(event) => {
          const cell = hitTest(event.clientX, event.clientY);
          if (cell) onSelect(cell.tensor);
        }}
        onMouseMove={(event) => {
          const cell = hitTest(event.clientX, event.clientY);
          if (!cell) return setHover(undefined);
          let chunk: HoverInfo["chunk"];
          if (zoom >= 2.5 && (cell.tensor.byteSegments?.length ?? 0) <= 1) {
            const rect = event.currentTarget.getBoundingClientRect();
            const localX = (event.clientX - rect.left - offset.x) / zoom - cell.x;
            const columns = Math.max(1, Math.floor(cell.width / 18));
            const rows = Math.max(1, Math.floor(cell.height / 18));
            const column = Math.min(columns - 1, Math.max(0, Math.floor(localX / (cell.width / columns))));
            const localY = (event.clientY - rect.top - offset.y) / zoom - cell.y;
            const row = Math.min(rows - 1, Math.max(0, Math.floor(localY / (cell.height / rows))));
            const index = row * columns + column;
            const count = BigInt(columns * rows);
            const start = cell.tensor.byteOffset + (cell.tensor.byteLength * BigInt(index)) / count;
            const end = cell.tensor.byteOffset + (cell.tensor.byteLength * BigInt(index + 1)) / count;
            chunk = { start, end };
          }
          setHover({
            cell,
            clientX: event.clientX,
            clientY: event.clientY,
            ...(chunk ? { chunk } : {})
          });
        }}
        onMouseLeave={() => setHover(undefined)}
        onWheel={(event) => {
          event.preventDefault();
          const next = Math.min(8, Math.max(1, zoom * (event.deltaY < 0 ? 1.18 : 0.85)));
          const rect = event.currentTarget.getBoundingClientRect();
          const px = event.clientX - rect.left;
          const py = event.clientY - rect.top;
          setOffset({
            x: px - ((px - offset.x) * next) / zoom,
            y: py - ((py - offset.y) * next) / zoom
          });
          setZoom(next);
        }}
      />
      <div className="wv-zoom">
        <button onClick={() => setZoom((value) => Math.min(8, value * 1.4))}>+</button>
        <span>{Math.round(zoom * 100)}%</span>
        <button
          onClick={() => {
            setZoom(1);
            setOffset({ x: 0, y: 0 });
          }}
        >↺</button>
      </div>
      {hover && (
        <div
          className="wv-tooltip"
          style={{
            left: Math.min(size.width - 280, Math.max(8, hover.clientX - (canvasRef.current?.getBoundingClientRect().left ?? 0) + 12)),
            top: Math.min(size.height - 150, Math.max(8, hover.clientY - (canvasRef.current?.getBoundingClientRect().top ?? 0) + 12))
          }}
        >
          <b>{hover.cell.tensor.name}</b>
          <span>{hover.cell.tensor.dtype} · {formatShape(hover.cell.tensor.shape)}</span>
          <span>{formatBytes(hover.cell.tensor.byteLength)}</span>
          {hover.cell.tensor.byteSegments && hover.cell.tensor.byteSegments.length > 1 ? (
            <code>{hover.cell.tensor.byteSegments.length} non-contiguous payload ranges</code>
          ) : (
            <code>
              {formatAddress(hover.chunk?.start ?? hover.cell.tensor.byteOffset)}
              {" → "}
              {formatAddress(hover.chunk?.end ?? tensorEnd(hover.cell.tensor))}
            </code>
          )}
        </div>
      )}
    </div>
  );
}

function layoutModel(model: ParsedModel, width: number, height: number): LayoutCell[] {
  const padding = 8;
  const fileGap = 12;
  const availableHeight = height - padding * 2;
  const totalSize = model.files.reduce((sum, file) => sum + file.size, 0n) || 1n;
  const cells: LayoutCell[] = [];
  let fileX = padding;
  for (const file of model.files) {
    const fileWidth = Math.max(
      80,
      ((width - padding * 2 - fileGap * (model.files.length - 1)) *
        ratio(file.size, totalSize))
    );
    layoutTensors(
      file,
      [...file.tensors].sort((a, b) =>
        a.byteLength === b.byteLength ? 0 : a.byteLength > b.byteLength ? -1 : 1
      ),
      { x: fileX, y: padding + 24, width: fileWidth, height: Math.max(1, availableHeight - 24) },
      cells,
      false
    );
    fileX += fileWidth + fileGap;
  }
  return cells;
}

function layoutTensors(
  file: ParsedFile,
  tensors: TensorRecord[],
  rect: { x: number; y: number; width: number; height: number },
  output: LayoutCell[],
  vertical: boolean
): void {
  if (!tensors.length) return;
  if (tensors.length === 1) {
    const tensor = tensors[0];
    if (tensor) output.push({ file, tensor, ...rect });
    return;
  }
  const total = tensors.reduce((sum, tensor) => sum + tensor.byteLength, 0n) || 1n;
  let split = 1;
  let leftSize = tensors[0]?.byteLength ?? 0n;
  while (
    split < tensors.length - 1 &&
    leftSize + (tensors[split]?.byteLength ?? 0n) <= total / 2n
  ) {
    leftSize += tensors[split]?.byteLength ?? 0n;
    split++;
  }
  const fraction = Math.min(0.95, Math.max(0.05, ratio(leftSize, total)));
  if (vertical) {
    const firstWidth = rect.width * fraction;
    layoutTensors(file, tensors.slice(0, split), { ...rect, width: firstWidth }, output, !vertical);
    layoutTensors(
      file,
      tensors.slice(split),
      { ...rect, x: rect.x + firstWidth, width: rect.width - firstWidth },
      output,
      !vertical
    );
  } else {
    const firstHeight = rect.height * fraction;
    layoutTensors(file, tensors.slice(0, split), { ...rect, height: firstHeight }, output, !vertical);
    layoutTensors(
      file,
      tensors.slice(split),
      { ...rect, y: rect.y + firstHeight, height: rect.height - firstHeight },
      output,
      !vertical
    );
  }
}

function ratio(part: bigint, whole: bigint): number {
  if (whole <= 0n) return 0;
  return Number((part * 1_000_000n) / whole) / 1_000_000;
}

function tensorEnd(tensor: TensorRecord): bigint {
  const last = tensor.byteSegments?.at(-1);
  return last ? last.byteOffset + last.byteLength : tensor.byteOffset + tensor.byteLength;
}

function drawCells(
  context: CanvasRenderingContext2D,
  cells: LayoutCell[],
  selected: TensorRecord | undefined,
  zoom: number
) {
  const dtypeColors = new Map<string, string>();
  let colorIndex = 0;
  for (const cell of cells) {
    let color = dtypeColors.get(cell.tensor.dtype);
    if (!color) {
      color = PALETTE[colorIndex++ % PALETTE.length] ?? "#6ee7ff";
      dtypeColors.set(cell.tensor.dtype, color);
    }
    context.fillStyle = `${color}b8`;
    context.fillRect(cell.x + 1, cell.y + 1, Math.max(0, cell.width - 2), Math.max(0, cell.height - 2));
    if (zoom >= 2.5 && cell.height > 15) {
      context.strokeStyle = "rgba(5, 10, 20, .35)";
      context.lineWidth = 0.5 / zoom;
      for (let x = cell.x + 18; x < cell.x + cell.width; x += 18) {
        context.beginPath();
        context.moveTo(x, cell.y);
        context.lineTo(x, cell.y + cell.height);
        context.stroke();
      }
      for (let y = cell.y + 18; y < cell.y + cell.height; y += 18) {
        context.beginPath();
        context.moveTo(cell.x, y);
        context.lineTo(cell.x + cell.width, y);
        context.stroke();
      }
    }
    if (selected?.id === cell.tensor.id) {
      context.strokeStyle = "#ffffff";
      context.lineWidth = 2 / zoom;
      context.strokeRect(cell.x, cell.y, cell.width, cell.height);
    }
    if (cell.height > 28 && cell.width > 100) {
      context.fillStyle = "#071019";
      context.font = `600 ${Math.max(8, 11 / Math.sqrt(zoom))}px ui-monospace, monospace`;
      context.fillText(
        cell.tensor.name.length > 34 ? `${cell.tensor.name.slice(0, 31)}…` : cell.tensor.name,
        cell.x + 7,
        cell.y + 16,
        cell.width - 12
      );
      context.font = `${Math.max(7, 9 / Math.sqrt(zoom))}px ui-monospace, monospace`;
      context.fillText(
        `${cell.tensor.dtype} · ${formatBytes(cell.tensor.byteLength)}`,
        cell.x + 7,
        cell.y + 29,
        cell.width - 12
      );
    }
  }
  const files = new Map<string, LayoutCell[]>();
  for (const cell of cells) {
    const group = files.get(cell.file.id) ?? [];
    group.push(cell);
    files.set(cell.file.id, group);
  }
  context.fillStyle = "#d8e3ec";
  context.font = "600 10px ui-monospace, monospace";
  for (const group of files.values()) {
    const first = group[0];
    if (first) context.fillText(first.file.name, first.x + 2, 18, first.width - 4);
  }
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><b>{value}</b></div>;
}

function Detail({
  label,
  value,
  mono = false
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return <div><dt>{label}</dt><dd className={mono ? "mono" : ""}>{value}</dd></div>;
}
