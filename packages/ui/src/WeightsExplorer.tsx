import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ParsedFile,
  ParsedModel,
  TensorRecord,
  TensorSample
} from "@weights-viz/core";
import {
  createAddressMapLayout,
  hitTestAddressMap,
  isClickGesture,
  type AddressHit,
  type AddressMapLayout,
  type AddressRect
} from "./address-map";
import { formatAddress, formatBytes, formatShape } from "./format";

interface WeightsExplorerProps {
  models: ParsedModel[];
  busy?: boolean;
  error?: string;
  onChooseFiles?: () => void;
  onFilesSelected?: (files: File[]) => void;
  onOpenUrl?: (url: string) => void;
  onSample?: (tensor: TensorRecord) => Promise<TensorSample>;
  intro?: string;
  compact?: boolean;
}

interface HoverInfo {
  hit: AddressHit;
  clientX: number;
  clientY: number;
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

const CHOOSER_GRACE_MS = 1200;

function embeddedHostLabel(): string | undefined {
  if (typeof navigator === "undefined") return undefined;
  const agent = navigator.userAgent;
  if (/\bCode\/|VSCode|Electron\//i.test(agent)) return "The VS Code built-in browser";
  if (/\bwv\b|; wv\)/.test(agent)) return "This Android in-app browser";
  if (/FBAN|FBAV|Instagram|Line\/|Twitter|MicroMessenger/i.test(agent)) {
    return "This in-app browser";
  }
  return undefined;
}

export function WeightsExplorer({
  models,
  busy = false,
  error,
  onChooseFiles,
  onFilesSelected,
  onOpenUrl,
  onSample,
  intro,
  compact = false
}: WeightsExplorerProps) {
  const [activeModelId, setActiveModelId] = useState<string>();
  const [selected, setSelected] = useState<TensorRecord>();
  const [sample, setSample] = useState<TensorSample>();
  const [sampleError, setSampleError] = useState<string>();
  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const [inspectorMode, setInspectorMode] = useState<"metadata" | "tensor">("metadata");
  const [metadataFileId, setMetadataFileId] = useState<string>();
  const [metadataQuery, setMetadataQuery] = useState("");
  const [pickerBlocked, setPickerBlocked] = useState(false);
  const activeModel =
    models.find((model) => model.id === activeModelId) ?? models[0];
  const metadataFile =
    activeModel?.files.find((file) => file.id === metadataFileId) ??
    activeModel?.files[0];

  useEffect(() => {
    if (activeModel && activeModel.id !== activeModelId) {
      setActiveModelId(activeModel.id);
      setSelected(undefined);
      setMetadataFileId(activeModel.files[0]?.id);
      setInspectorMode("metadata");
      setMetadataQuery("");
    }
  }, [activeModel, activeModelId]);

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
    <main className={`wv-shell${compact ? " compact" : ""}`}>
      {!compact && <header className="wv-header">
        <div className="wv-brand">
          <div className="wv-kicker">LOCAL-FIRST MODEL INSPECTOR</div>
          <h1>Weights <span>Viz</span></h1>
          <a
            className="wv-repo-link"
            href="https://github.com/justinchuby/weights-viz"
            target="_blank"
            rel="noreferrer"
          >
            Open source by justinchuby · GitHub ↗
          </a>
        </div>
        <div className="wv-actions">
          {onFilesSelected ? (
            <FilePicker onFilesSelected={onFilesSelected} onPickerResult={setPickerBlocked}>
              Open files
            </FilePicker>
          ) : onChooseFiles ? (
            <button className="wv-button primary" onClick={onChooseFiles}>
              Open files
            </button>
          ) : null}
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
      </header>}

      {error && <div className="wv-alert error">{error}</div>}
      {pickerBlocked && (
        <div className="wv-alert warning" role="status">
          <div>
            <b>The file chooser did not open.</b>{" "}
            {embeddedHostLabel() ?? "This browser or embedded webview"} does not allow web pages
            to open the system file dialog. Drag model files onto this window instead, or paste a
            model URL into the field above. Opening this page in a normal browser tab also works.
          </div>
          <button
            type="button"
            className="wv-alert-dismiss"
            aria-label="Dismiss"
            onClick={() => setPickerBlocked(false)}
          >
            ×
          </button>
        </div>
      )}
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
            {onFilesSelected ? (
              <FilePicker large onFilesSelected={onFilesSelected} onPickerResult={setPickerBlocked}>
                Choose model files
              </FilePicker>
            ) : onChooseFiles ? (
              <button className="wv-button primary large" onClick={onChooseFiles}>
                Choose model files
              </button>
            ) : null}
          </div>
        </section>
      ) : (
        <section className="wv-workspace">
          <div className="wv-map-panel">
            <div className="wv-panel-title">
              <div>
                <span>BYTE MAP</span>
                {models.length > 1 ? (
                  <select
                    className="wv-model-select"
                    aria-label="Active model"
                    value={activeModel.id}
                    onChange={(event) => setActiveModelId(event.target.value)}
                  >
                    {models.map((model) => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </select>
                ) : (
                  <h2>{activeModel.name}</h2>
                )}
              </div>
              <p>Addresses run left → right, top → bottom · drag to pan</p>
            </div>
            <div className="wv-map-toolbar">
              <input
                id="tensor-filter"
                className="wv-filter"
                aria-label="Filter tensors"
                placeholder="Filter tensors by name or dtype"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <div className="wv-legend" aria-label="Data type legend">
                {[...new Set(activeModel.files.flatMap((file) => file.tensors.map((tensor) => tensor.dtype)))]
                  .slice(0, 8)
                  .map((dtype, index) => (
                    <span key={dtype}>
                      <i style={{ background: PALETTE[index % PALETTE.length] }} />
                      {dtype}
                    </span>
                  ))}
              </div>
            </div>
            <WeightMap
              model={activeModel}
              query={query}
              {...(selected ? { selected } : {})}
              onSelect={(tensor) => {
                setSelected(tensor);
                setInspectorMode("tensor");
                setSample(undefined);
                setSampleError(undefined);
              }}
            />
          </div>

          <aside className="wv-inspector">
            <div className="wv-inspector-head">
              <div>
                <label className="wv-label">Model</label>
                <h1>{activeModel.name}</h1>
              </div>
              {compact && (onChooseFiles || onFilesSelected) && (
                onFilesSelected ? (
                  <FilePicker onFilesSelected={onFilesSelected} onPickerResult={setPickerBlocked}>
                    Open files
                  </FilePicker>
                ) : (
                  <button className="wv-button" onClick={onChooseFiles}>Open files</button>
                )
              )}
            </div>
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
            <div className="wv-inspector-tabs">
              <button
                className={inspectorMode === "metadata" ? "active" : ""}
                onClick={() => setInspectorMode("metadata")}
              >
                Metadata
              </button>
              <button
                className={inspectorMode === "tensor" ? "active" : ""}
                disabled={!selected}
                onClick={() => setInspectorMode("tensor")}
              >
                Tensor
              </button>
            </div>
            {inspectorMode === "metadata" && metadataFile ? (
              <MetadataInspector
                file={metadataFile}
                files={activeModel.files}
                query={metadataQuery}
                onQueryChange={setMetadataQuery}
                onFileChange={setMetadataFileId}
              />
            ) : selected ? (
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
  query,
  selected,
  onSelect
}: {
  model: ParsedModel;
  query: string;
  selected?: TensorRecord;
  onSelect: (tensor: TensorRecord) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
    moved: boolean;
  } | undefined>(undefined);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const [hover, setHover] = useState<HoverInfo>();
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const themeRevision = useThemeRevision();
  const layout = useMemo(
    () => {
      const needle = query.trim().toLowerCase();
      return createAddressMapLayout(
        model,
        size.width,
        needle
          ? (tensor) =>
              tensor.name.toLowerCase().includes(needle) ||
              tensor.dtype.toLowerCase().includes(needle)
          : undefined
      );
    },
    [model, query, size.width]
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
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  }, [model.id]);

  useEffect(() => {
    setOffset((current) => clampOffset(current, layout, size, zoom));
  }, [layout, size, zoom]);

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
    const theme = readCanvasTheme(canvas);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, size.width, size.height);
    context.save();
    context.translate(offset.x, offset.y);
    context.scale(zoom, zoom);
    drawAddressMap(context, layout, selected, zoom, offset, size, theme);
    context.restore();
    drawAddressRuler(context, layout, zoom, offset, size, theme);
  }, [layout, offset, selected, size, themeRevision, zoom]);

  const hitTest = (clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return undefined;
    const x = (clientX - rect.left - offset.x) / zoom;
    const y = (clientY - rect.top - offset.y) / zoom;
    return hitTestAddressMap(layout, x, y);
  };

  const setZoomAt = (nextZoom: number, px: number, py: number) => {
    const next = Math.min(128, Math.max(1, nextZoom));
    const nextOffset = {
      x: px - ((px - offset.x) * next) / zoom,
      y: py - ((py - offset.y) * next) / zoom
    };
    setZoom(next);
    setOffset(clampOffset(nextOffset, layout, size, next));
    setHover(undefined);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      setZoomAt(
        zoom * (event.deltaY < 0 ? 1.18 : 0.85),
        event.clientX - rect.left,
        event.clientY - rect.top
      );
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [layout, offset, size, zoom]);

  return (
    <div
      className={`wv-canvas-wrap${dragging ? " dragging" : ""}`}
      ref={containerRef}
    >
      <canvas
        ref={canvasRef}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = {
            pointerId: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
            originX: offset.x,
            originY: offset.y,
            moved: false
          };
          setDragging(true);
          setHover(undefined);
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (drag?.pointerId === event.pointerId) {
            const dx = event.clientX - drag.startX;
            const dy = event.clientY - drag.startY;
            if (!isClickGesture(0, 0, dx, dy)) drag.moved = true;
            if (drag.moved) {
              setOffset(
                clampOffset(
                  { x: drag.originX + dx, y: drag.originY + dy },
                  layout,
                  size,
                  zoom
                )
              );
            }
            return;
          }
          const hit = hitTest(event.clientX, event.clientY);
          if (!hit) return setHover(undefined);
          setHover({
            hit,
            clientX: event.clientX,
            clientY: event.clientY
          });
        }}
        onPointerUp={(event) => {
          const drag = dragRef.current;
          if (!drag || drag.pointerId !== event.pointerId) return;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
          if (!drag.moved) {
            const hit = hitTest(event.clientX, event.clientY);
            if (hit?.kind === "tensor" && hit.tensor) onSelect(hit.tensor);
          }
          dragRef.current = undefined;
          setDragging(false);
        }}
        onPointerCancel={() => {
          dragRef.current = undefined;
          setDragging(false);
        }}
        onPointerLeave={() => {
          if (!dragRef.current) setHover(undefined);
        }}
      />
      <div className="wv-zoom">
        <button
          aria-label="Zoom out"
          onClick={() => setZoomAt(zoom / 1.4, size.width / 2, size.height / 2)}
        >
          −
        </button>
        <span>{Math.round(zoom * 100)}%</span>
        <button
          aria-label="Zoom in"
          onClick={() => setZoomAt(zoom * 1.4, size.width / 2, size.height / 2)}
        >
          +
        </button>
        <button
          aria-label="Reset view"
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
            top: Math.min(size.height - 230, Math.max(8, hover.clientY - (canvasRef.current?.getBoundingClientRect().top ?? 0) + 12))
          }}
        >
          {hover.hit.kind === "tensor" && hover.hit.tensor ? (
            <>
              <b>{hover.hit.tensor.name}</b>
              <span>{hover.hit.tensor.dtype} · {formatShape(hover.hit.tensor.shape)}</span>
              <span>{formatBytes(hover.hit.tensor.byteLength)}</span>
              <code>Pointer: {formatAddress(hover.hit.address)}</code>
              <code>
                Tensor: {formatAddress(hover.hit.start)} → {formatAddress(hover.hit.end)}
              </code>
            </>
          ) : hover.hit.kind === "filtered" && hover.hit.tensor ? (
            <>
              <b>{hover.hit.tensor.name} · filtered</b>
              <span>{hover.hit.tensor.dtype} · {formatShape(hover.hit.tensor.shape)}</span>
              <code>Pointer: {formatAddress(hover.hit.address)}</code>
              <code>
                Tensor: {formatAddress(hover.hit.start)} → {formatAddress(hover.hit.end)}
              </code>
            </>
          ) : hover.hit.kind === "metadata" ? (
            <>
              <b>{hover.hit.file.name} · header / metadata</b>
              <code>Pointer: {formatAddress(hover.hit.address)}</code>
              <code>
                Range: {formatAddress(hover.hit.start)} → {formatAddress(hover.hit.end)}
              </code>
            </>
          ) : (
            <>
              <b>Unmapped / alignment gap</b>
              <span>{hover.hit.file.name}</span>
              <span>Padding: {formatBytes(hover.hit.end - hover.hit.start)}</span>
              {formattedAlignment(hover.hit.file) && (
                <span>
                  Declared {hover.hit.file.format.toUpperCase()} alignment:{" "}
                  {formattedAlignment(hover.hit.file)}
                </span>
              )}
              <code>Pointer: {formatAddress(hover.hit.address)}</code>
              <code>
                Gap: {formatAddress(hover.hit.start)} → {formatAddress(hover.hit.end)}
              </code>
            </>
          )}
          <span>Grid cell: {formatBytes(hover.hit.bytesPerCell)}</span>
          <code>
            Cell: {formatAddress(hover.hit.cellStart)} → {formatAddress(hover.hit.cellEnd)}
          </code>
        </div>
      )}
    </div>
  );
}

function tensorEnd(tensor: TensorRecord): bigint {
  const last = tensor.byteSegments?.at(-1);
  return last ? last.byteOffset + last.byteLength : tensor.byteOffset + tensor.byteLength;
}

function declaredAlignment(file: ParsedFile): bigint | undefined {
  if (file.format !== "gguf") return undefined;
  const value = file.metadata["general.alignment"];
  if (typeof value === "bigint" && value > 0n) return value;
  if (typeof value === "number" && Number.isSafeInteger(value) && value > 0) {
    return BigInt(value);
  }
  return 32n;
}

function formattedAlignment(file: ParsedFile): string | undefined {
  const alignment = declaredAlignment(file);
  return alignment === undefined ? undefined : formatBytes(alignment);
}

function drawAddressMap(
  context: CanvasRenderingContext2D,
  layout: AddressMapLayout,
  selected: TensorRecord | undefined,
  zoom: number,
  offset: { x: number; y: number },
  viewport: { width: number; height: number },
  theme: CanvasTheme
) {
  const dtypeColors = new Map<string, string>();
  let colorIndex = 0;
  const visibleTop = -offset.y / zoom;
  const visibleBottom = (viewport.height - offset.y) / zoom;

  for (const fileLayout of layout.files) {
    const gridBottom =
      fileLayout.gridY + fileLayout.rowCount * fileLayout.rowHeight;
    if (gridBottom < visibleTop || fileLayout.gridY > visibleBottom) continue;

    context.fillStyle = theme.mapBackground;
    context.fillRect(
      fileLayout.gridX,
      fileLayout.gridY,
      fileLayout.gridWidth,
      fileLayout.rowCount * fileLayout.rowHeight
    );

    for (const span of fileLayout.spans) {
      if (!span.visible) continue;
      const color =
        span.kind === "metadata"
          ? theme.metadata
          : colorForTensor(span.tensor!, dtypeColors, () => colorIndex++);
      context.fillStyle = color;
      context.globalAlpha = span.kind === "metadata" ? 0.92 : 0.88;
      for (const rect of span.rects) {
        if (!isVisible(rect, visibleTop, visibleBottom)) continue;
        context.fillRect(rect.x, rect.y, rect.width, rect.height);
      }
    }
    context.globalAlpha = 1;

    context.strokeStyle = theme.accent;
    context.globalAlpha = 0.28;
    context.lineWidth = 1 / zoom;
    for (let column = 0; column <= fileLayout.columns; column++) {
      const x =
        fileLayout.gridX +
        (column * fileLayout.gridWidth) / fileLayout.columns;
      context.beginPath();
      context.moveTo(x, fileLayout.gridY);
      context.lineTo(x, gridBottom);
      context.stroke();
    }
    for (let row = 0; row <= fileLayout.rowCount; row++) {
      const y = fileLayout.gridY + row * fileLayout.rowHeight;
      if (y >= visibleTop && y <= visibleBottom) {
        context.beginPath();
        context.moveTo(fileLayout.gridX, y);
        context.lineTo(fileLayout.gridX + fileLayout.gridWidth, y);
        context.stroke();
      }
    }
    context.globalAlpha = 1;

    for (const span of fileLayout.spans) {
      if (!span.tensor || !span.visible) continue;
      if (selected?.id === span.tensor.id) {
        context.strokeStyle = theme.selection;
        context.lineWidth = 2 / zoom;
        for (const rect of span.rects) {
          if (isVisible(rect, visibleTop, visibleBottom)) {
            context.strokeRect(rect.x, rect.y, rect.width, rect.height);
          }
        }
      }
      drawTensorLabel(
        context,
        span.tensor,
        span.rects,
        zoom,
        visibleTop,
        visibleBottom,
        theme
      );
    }
  }
}

function drawAddressRuler(
  context: CanvasRenderingContext2D,
  layout: AddressMapLayout,
  zoom: number,
  offset: { x: number; y: number },
  viewport: { width: number; height: number },
  theme: CanvasTheme
) {
  context.save();
  context.fillStyle = theme.surface;
  context.fillRect(0, 0, 104, viewport.height);
  context.strokeStyle = theme.border;
  context.beginPath();
  context.moveTo(103.5, 0);
  context.lineTo(103.5, viewport.height);
  context.stroke();
  context.textBaseline = "middle";

  for (const fileLayout of layout.files) {
    const headerY = offset.y + (fileLayout.gridY - 15) * zoom;
    if (headerY > -20 && headerY < viewport.height + 20) {
      context.fillStyle = theme.text;
      context.font = "600 11px ui-monospace, monospace";
      context.fillText(fileLayout.file.name, 8, headerY, viewport.width - 16);
      context.fillStyle = theme.muted;
      context.font = "9px ui-monospace, monospace";
      context.fillText(
        `${formatBytes(fileLayout.bytesPerCell)} / cell · ${formatBytes(fileLayout.bytesPerRow)} / row${
          formattedAlignment(fileLayout.file)
            ? ` · ${formattedAlignment(fileLayout.file)} alignment`
            : ""
        }`,
        Math.max(112, viewport.width - 300),
        headerY
      );
    }

    context.fillStyle = theme.muted;
    context.font = "9px ui-monospace, monospace";
    for (let row = 0; row < fileLayout.rowCount; row++) {
      const y =
        offset.y +
        (fileLayout.gridY + row * fileLayout.rowHeight) * zoom +
        (fileLayout.rowHeight * zoom) / 2;
      if (y < -10 || y > viewport.height + 10) continue;
      context.fillText(
        formatAddress(BigInt(row) * fileLayout.bytesPerRow),
        8,
        y,
        90
      );
    }
  }
  context.restore();
}

function drawTensorLabel(
  context: CanvasRenderingContext2D,
  tensor: TensorRecord,
  rects: AddressRect[],
  zoom: number,
  visibleTop: number,
  visibleBottom: number,
  theme: CanvasTheme
) {
  const rect = rects
    .filter((candidate) => isVisible(candidate, visibleTop, visibleBottom))
    .sort((a, b) => b.width - a.width)[0];
  if (!rect || rect.width * zoom < 24 || rect.height * zoom < 9) return;

  const pixelWidth = rect.width * zoom;
  const pixelHeight = rect.height * zoom;
  const fontSize = (pixelHeight >= 13 && pixelWidth >= 48 ? 10 : 8) / zoom;
  const padding = Math.min(4, pixelWidth * 0.1) / zoom;
  const availableWidth = rect.width - padding * 2;

  context.save();
  context.beginPath();
  context.rect(rect.x, rect.y, rect.width, rect.height);
  context.clip();
  context.fillStyle = theme.labelText;
  context.font = `600 ${fontSize}px ui-monospace, monospace`;
  context.textBaseline = "middle";
  context.fillText(
    fitCanvasText(context, tensor.name, availableWidth),
    rect.x + padding,
    rect.y + Math.min(pixelHeight / 2, 7) / zoom
  );
  if (pixelHeight >= 27 && pixelWidth >= 64) {
    context.font = `${9 / zoom}px ui-monospace, monospace`;
    context.textBaseline = "top";
    context.fillText(
      fitCanvasText(
        context,
        `${tensor.dtype} · ${formatBytes(tensor.byteLength)}`,
        availableWidth
      ),
      rect.x + padding,
      rect.y + 16 / zoom
    );
  }
  context.restore();
}

function fitCanvasText(
  context: CanvasRenderingContext2D,
  text: string,
  maxWidth: number
): string {
  if (context.measureText(text).width <= maxWidth) return text;
  const ellipsis = "…";
  if (context.measureText(ellipsis).width > maxWidth) return "";
  let low = 0;
  let high = text.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (context.measureText(`${text.slice(0, middle)}${ellipsis}`).width <= maxWidth) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return `${text.slice(0, low)}${ellipsis}`;
}

interface CanvasTheme {
  mapBackground: string;
  surface: string;
  metadata: string;
  text: string;
  muted: string;
  border: string;
  accent: string;
  selection: string;
  labelText: string;
}

function readCanvasTheme(canvas: HTMLCanvasElement): CanvasTheme {
  const styles = getComputedStyle(canvas);
  const read = (name: string, fallback: string) =>
    styles.getPropertyValue(name).trim() || fallback;
  return {
    mapBackground: read("--wv-map-bg", "#0b1923"),
    surface: read("--wv-surface", "#09141d"),
    metadata: read("--wv-surface-raised", "#324b5a"),
    text: read("--wv-text", "#e6f0f6"),
    muted: read("--wv-muted", "#9db2bf"),
    border: read("--wv-border", "#29404f"),
    accent: read("--wv-accent", "#6ee7ff"),
    selection: read("--wv-text", "#ffffff"),
    labelText: "#071019"
  };
}

function useThemeRevision(): number {
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setRevision((value) => value + 1);
    media.addEventListener("change", update);
    const observer = new MutationObserver(update);
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ["class", "style"]
    });
    return () => {
      media.removeEventListener("change", update);
      observer.disconnect();
    };
  }, []);
  return revision;
}

function colorForTensor(
  tensor: TensorRecord,
  colors: Map<string, string>,
  nextIndex: () => number
): string {
  let color = colors.get(tensor.dtype);
  if (!color) {
    color = PALETTE[nextIndex() % PALETTE.length] ?? "#6ee7ff";
    colors.set(tensor.dtype, color);
  }
  return color;
}

function isVisible(
  rect: AddressRect,
  visibleTop: number,
  visibleBottom: number
): boolean {
  return rect.y + rect.height >= visibleTop && rect.y <= visibleBottom;
}

function clampOffset(
  offset: { x: number; y: number },
  layout: AddressMapLayout,
  viewport: { width: number; height: number },
  zoom: number
) {
  const minX = Math.min(0, viewport.width - layout.width * zoom);
  const minY = Math.min(0, viewport.height - layout.contentHeight * zoom);
  return {
    x: Math.min(0, Math.max(minX, offset.x)),
    y: Math.min(0, Math.max(minY, offset.y))
  };
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><b>{value}</b></div>;
}

function FilePicker({
  children,
  large = false,
  onFilesSelected,
  onPickerResult
}: {
  children: string;
  large?: boolean;
  onFilesSelected: (files: File[]) => void;
  onPickerResult?: (blocked: boolean) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const stopWatch = useRef<((blocked?: boolean) => void) | undefined>(undefined);
  const report = useRef(onPickerResult);
  report.current = onPickerResult;

  // The chooser is a native dialog: it always moves focus away from the page.
  // Embedded webviews that ignore file inputs produce no dialog and no focus
  // change, which is the only observable difference from a working browser.
  const watchChooser = () => {
    stopWatch.current?.();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const stop = (blocked?: boolean) => {
      if (timeout !== undefined) clearTimeout(timeout);
      window.removeEventListener("blur", opened);
      document.removeEventListener("visibilitychange", opened);
      stopWatch.current = undefined;
      if (blocked !== undefined) report.current?.(blocked);
    };
    const opened = () => stop(false);
    timeout = setTimeout(() => stop(true), CHOOSER_GRACE_MS);
    window.addEventListener("blur", opened);
    document.addEventListener("visibilitychange", opened);
    stopWatch.current = stop;
  };

  const chooserWorked = () => {
    stopWatch.current?.();
    report.current?.(false);
  };

  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.addEventListener("cancel", chooserWorked);
    return () => {
      input.removeEventListener("cancel", chooserWorked);
      stopWatch.current?.();
    };
  }, []);

  return (
    <span className={`wv-file-picker wv-button primary${large ? " large" : ""}`}>
      <span aria-hidden="true">{children}</span>
      <input
        ref={inputRef}
        type="file"
        multiple
        aria-label={children}
        accept=".gguf,.safetensors,.onnx,.json,application/octet-stream"
        onClick={watchChooser}
        onChange={(event) => {
          chooserWorked();
          const files = Array.from(event.currentTarget.files ?? []);
          event.currentTarget.value = "";
          if (files.length) onFilesSelected(files);
        }}
      />
    </span>
  );
}

function MetadataInspector({
  file,
  files,
  query,
  onQueryChange,
  onFileChange
}: {
  file: ParsedFile;
  files: ParsedFile[];
  query: string;
  onQueryChange: (value: string) => void;
  onFileChange: (id: string) => void;
}) {
  const entries = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return Object.entries(file.metadata).filter(([key, value]) => {
      if (!needle) return true;
      return (
        key.toLowerCase().includes(needle) ||
        formatMetadataValue(value).toLowerCase().includes(needle)
      );
    });
  }, [file, query]);

  return (
    <div className="wv-metadata">
      <div className="wv-tensor-type">{file.format.toUpperCase()}</div>
      <h2>{file.name}</h2>
      {files.length > 1 && (
        <select
          className="wv-file-select"
          aria-label="Metadata file"
          value={file.id}
          onChange={(event) => onFileChange(event.target.value)}
        >
          {files.map((candidate) => (
            <option key={candidate.id} value={candidate.id}>
              {candidate.name}
            </option>
          ))}
        </select>
      )}
      <div className="wv-metadata-summary">
        <span>{Object.keys(file.metadata).length} entries</span>
        <span>{formatBytes(file.size)}</span>
      </div>
      <input
        className="wv-filter"
        aria-label="Filter metadata"
        placeholder="Filter metadata keys or values"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
      />
      <div className="wv-metadata-list">
        {entries.map(([key, value]) => (
          <div className="wv-metadata-entry" key={key}>
            <div>
              <code>{key}</code>
              <span>{metadataType(value)}</span>
            </div>
            <p>{formatMetadataValue(value)}</p>
          </div>
        ))}
        {!entries.length && (
          <div className="wv-metadata-empty">
            {Object.keys(file.metadata).length
              ? "No metadata matches this filter."
              : "This file does not declare model metadata."}
          </div>
        )}
      </div>
    </div>
  );
}

export function formatMetadataValue(value: unknown): string {
  if (typeof value === "string") {
    return value.length > 400 ? `${value.slice(0, 400)}…` : value;
  }
  if (
    typeof value === "number" ||
    typeof value === "boolean" ||
    typeof value === "bigint"
  ) {
    return String(value);
  }
  if (value === null || value === undefined) return String(value);
  if (Array.isArray(value)) {
    const shown = value.slice(0, 16).map(formatMetadataScalar).join(", ");
    return `[${shown}${value.length > 16 ? `, … (+${value.length - 16})` : ""}]`;
  }
  try {
    const serialized = JSON.stringify(
      value,
      (_key, item) => (typeof item === "bigint" ? item.toString() : item)
    );
    return serialized.length > 600 ? `${serialized.slice(0, 600)}…` : serialized;
  } catch {
    return String(value);
  }
}

function formatMetadataScalar(value: unknown): string {
  if (typeof value === "string") {
    const shortened = value.length > 48 ? `${value.slice(0, 48)}…` : value;
    return JSON.stringify(shortened);
  }
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "object" && value !== null) return "{…}";
  return String(value);
}

function metadataType(value: unknown): string {
  if (Array.isArray(value)) return `array · ${value.length}`;
  if (value === null) return "null";
  return typeof value;
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
