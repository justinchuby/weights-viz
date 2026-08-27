import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowLeftRight,
  ChevronDown,
  ChevronUp,
  CircleSlash2,
  Columns2,
  Download,
  FileUp,
  GitFork,
  Info,
  Link2,
  Minus,
  MousePointerClick,
  Plus,
  RotateCcw,
  Square,
  X
} from "lucide-react";
import type {
  ModelComparison,
  ParsedFile,
  ParsedModel,
  TensorComparison,
  TensorComparisonStatus,
  TensorRecord,
  TensorSample
} from "@weights-viz/core";
import {
  compareModels,
  tensorComparisonKey
} from "@weights-viz/core";
import {
  addressMapMaxScrollY,
  createAddressMapLayout,
  hitTestAddressMap,
  isClickGesture,
  type AddressHit,
  type AddressMapLayout,
  type AddressRect
} from "./address-map";
import {
  formatAddress,
  formatBytes,
  formatParameterCount,
  formatShape
} from "./format";
import { colorForDtype } from "./dtype-color";
import {
  classifyTensorRole,
  colorForTensorRole,
  tensorRoleLabel
} from "./tensor-role";

interface WeightsExplorerProps {
  models: ParsedModel[];
  busy?: boolean;
  error?: string;
  onChooseFiles?: () => void;
  onFilesSelected?: (files: File[]) => void;
  onOpenUrl?: (url: string) => void;
  onSample?: (tensor: TensorRecord) => Promise<TensorSample>;
  defaultUrl?: string;
  intro?: string;
  compact?: boolean;
  defaultCompare?: boolean;
  installAvailable?: boolean;
  onInstall?: () => void;
}

interface HoverInfo {
  hit: AddressHit;
  clientX: number;
  clientY: number;
}

interface TensorNavigationTarget {
  tensor: TensorRecord;
  sequence: number;
}

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
  defaultUrl = "",
  intro,
  compact = false,
  defaultCompare = false,
  installAvailable = false,
  onInstall
}: WeightsExplorerProps) {
  const [activeModelId, setActiveModelId] = useState<string>();
  const [selected, setSelected] = useState<TensorRecord>();
  const [sample, setSample] = useState<TensorSample>();
  const [sampleError, setSampleError] = useState<string>();
  const [url, setUrl] = useState(defaultUrl);
  const [urlOpen, setUrlOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [inspectorMode, setInspectorMode] = useState<"metadata" | "tensor">("metadata");
  const [metadataFileId, setMetadataFileId] = useState<string>();
  const [metadataQuery, setMetadataQuery] = useState("");
  const [pickerBlocked, setPickerBlocked] = useState(false);
  const [tensorNavigation, setTensorNavigation] =
    useState<TensorNavigationTarget>();
  const [comparisonMode, setComparisonMode] = useState(defaultCompare);
  const [rightModelId, setRightModelId] = useState<string>();
  const tensorNavigationSequence = useRef(0);
  const activeModel =
    models.find((model) => model.id === activeModelId) ?? models[0];
  const metadataFile =
    activeModel?.files.find((file) => file.id === metadataFileId) ??
    activeModel?.files[0];
  const comparisonRight =
    models.find((model) => model.id === rightModelId && model.id !== activeModel?.id) ??
    models.find((model) => model.id !== activeModel?.id);

  useEffect(() => {
    if (activeModel && activeModel.id !== activeModelId) {
      setActiveModelId(activeModel.id);
    }
  }, [activeModel, activeModelId]);

  useEffect(() => {
    if (!activeModel) return;
    setSelected(undefined);
    setSample(undefined);
    setSampleError(undefined);
    setMetadataFileId(activeModel.files[0]?.id);
    setInspectorMode("metadata");
    setMetadataQuery("");
    setTensorNavigation(undefined);
  }, [activeModel?.id]);

  useEffect(() => {
    if (defaultCompare) setComparisonMode(true);
  }, [defaultCompare]);

  useEffect(() => {
    if (comparisonRight && comparisonRight.id !== rightModelId) {
      setRightModelId(comparisonRight.id);
    }
    if (models.length === 1) setComparisonMode(false);
  }, [comparisonRight, models.length, rightModelId]);

  const matchingTensors = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!activeModel || !needle) return [];
    return activeModel.files.flatMap((file) =>
      file.tensors.filter(
        (tensor) =>
          tensor.name.toLowerCase().includes(needle) ||
          tensor.dtype.toLowerCase().includes(needle)
      )
    );
  }, [activeModel, query]);
  const selectedMatchIndex = selected
    ? matchingTensors.indexOf(selected)
    : -1;

  const selectTensor = (tensor: TensorRecord) => {
    setSelected(tensor);
    setInspectorMode("tensor");
    setSample(undefined);
    setSampleError(undefined);
  };

  const navigateTensorMatches = (direction: 1 | -1) => {
    if (matchingTensors.length === 0) return;
    const nextIndex =
      selectedMatchIndex < 0
        ? direction > 0
          ? 0
          : matchingTensors.length - 1
        : (selectedMatchIndex + direction + matchingTensors.length) %
          matchingTensors.length;
    const tensor = matchingTensors[nextIndex]!;
    selectTensor(tensor);
    setTensorNavigation({
      tensor,
      sequence: ++tensorNavigationSequence.current
    });
  };

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
          <h1>Weights <span>Viz</span></h1>
          <span className="wv-brand-tag">local-first model inspector</span>
        </div>
        <div className="wv-actions">
          <a
            className="wv-icon-button"
            href="https://github.com/justinchuby/weights-viz"
            target="_blank"
            rel="noreferrer"
            title="View source on GitHub"
            aria-label="View source on GitHub"
          >
            <GitFork className="wv-icon" aria-hidden="true" />
          </a>
          {onFilesSelected ? (
            <FilePicker
              label="Open files"
              onFilesSelected={onFilesSelected}
              onPickerResult={setPickerBlocked}
            >
              <FileUp className="wv-icon" aria-hidden="true" />
              <span>Open</span>
            </FilePicker>
          ) : onChooseFiles ? (
            <button className="wv-button primary wv-icon-label" onClick={onChooseFiles}>
              <FileUp className="wv-icon" aria-hidden="true" />
              <span>Open</span>
            </button>
          ) : null}
          {onOpenUrl && (
            <button
              className={`wv-icon-button${urlOpen ? " active" : ""}`}
              type="button"
              title="Load model URLs"
              aria-label="Load model URLs"
              aria-expanded={urlOpen}
              onClick={() => setUrlOpen((open) => !open)}
            >
              <Link2 className="wv-icon" aria-hidden="true" />
            </button>
          )}
          {installAvailable && onInstall && (
            <button
              className="wv-icon-button"
              type="button"
              title="Install app"
              aria-label="Install app"
              onClick={onInstall}
            >
              <Download className="wv-icon" aria-hidden="true" />
            </button>
          )}
          {onOpenUrl && urlOpen && (
            <form
              className="wv-url-popover"
              onSubmit={(event) => {
                event.preventDefault();
                if (!url.trim()) return;
                onOpenUrl(url.trim());
                setUrlOpen(false);
              }}
            >
              <label htmlFor="wv-model-urls">Model URLs</label>
              <textarea
                id="wv-model-urls"
                rows={3}
                placeholder="Paste one or more model URLs"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
              />
              <div>
                <small>Separate multiple URLs with spaces or new lines.</small>
                <button className="wv-button primary" type="submit">Load</button>
              </div>
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
            <X className="wv-icon" aria-hidden="true" />
          </button>
        </div>
      )}
      {busy && <div className="wv-progress"><span /></div>}

      {!activeModel ? (
        <section className="wv-empty">
          <div className="wv-empty-grid" />
          <div className="wv-empty-content">
            <div className="wv-file-mark">01</div>
            <h2>{busy ? "Loading model metadata…" : "See what your model is made of."}</h2>
            <p>
              {busy
                ? "Reading file headers and tensor indexes. Large sharded models may take a moment."
                : intro ??
                  "Drop GGUF, SafeTensors, or ONNX files here. Files stay on this device; remote models use byte-range requests."}
            </p>
            {!busy && onFilesSelected ? (
              <FilePicker large onFilesSelected={onFilesSelected} onPickerResult={setPickerBlocked}>
                Choose model files
              </FilePicker>
            ) : !busy && onChooseFiles ? (
              <button className="wv-button primary large" onClick={onChooseFiles}>
                Choose model files
              </button>
            ) : null}
          </div>
        </section>
      ) : comparisonMode && comparisonRight ? (
        <ComparisonWorkspace
          key={`${activeModel.id}:${comparisonRight.id}`}
          models={models}
          left={activeModel}
          right={comparisonRight}
          compact={compact}
          onLeftChange={setActiveModelId}
          onRightChange={setRightModelId}
          onExit={() => setComparisonMode(false)}
        />
      ) : (
        <section key={activeModel.id} className="wv-workspace">
          <div className="wv-map-panel">
            <div className="wv-panel-title">
              <div>
                <span>BYTE MAP</span>
                {models.length > 1 ? (
                  <ModelSelectControl
                    className="wv-model-select"
                    aria-label="Active model"
                    value={activeModel.id}
                    models={models}
                    onChange={setActiveModelId}
                  />
                ) : (
                  <h2>{activeModel.name}</h2>
                )}
              </div>
              <div className="wv-panel-actions">
                {models.length > 1 && (
                  <button
                    className="wv-button wv-icon-label wv-toolbar-control"
                    type="button"
                    title="Compare two models"
                    onClick={() => setComparisonMode(true)}
                  >
                    <Columns2 className="wv-icon" aria-hidden="true" />
                    <span>Compare</span>
                  </button>
                )}
                <p>Wheel to scroll · pinch or Ctrl/⌘ + wheel to zoom · drag to pan</p>
              </div>
            </div>
            <div className="wv-map-toolbar">
              <div className="wv-tensor-search">
                <input
                  id="tensor-filter"
                  className="wv-filter"
                  aria-label="Filter tensors"
                  placeholder="Filter tensors by name or dtype"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key !== "Enter") return;
                    event.preventDefault();
                    navigateTensorMatches(event.shiftKey ? -1 : 1);
                  }}
                />
                <span
                  className="wv-search-count"
                  aria-live="polite"
                  title={`${matchingTensors.length} matching tensors`}
                >
                  {query.trim()
                    ? `${selectedMatchIndex + 1}/${matchingTensors.length}`
                    : "—"}
                </span>
                <button
                  type="button"
                  aria-label="Previous tensor match"
                  title="Previous match (Shift+Enter)"
                  disabled={matchingTensors.length === 0}
                  onClick={() => navigateTensorMatches(-1)}
                >
                  <ChevronUp className="wv-icon" aria-hidden="true" />
                </button>
                <button
                  type="button"
                  aria-label="Next tensor match"
                  title="Next match (Enter)"
                  disabled={matchingTensors.length === 0}
                  onClick={() => navigateTensorMatches(1)}
                >
                  <ChevronDown className="wv-icon" aria-hidden="true" />
                </button>
              </div>
              <div className="wv-legend" aria-label="Data type legend">
                {[...new Set(activeModel.files.flatMap((file) => file.tensors.map((tensor) => tensor.dtype)))]
                  .slice(0, 8)
                  .map((dtype) => (
                    <span key={dtype}>
                      <i style={{ background: colorForDtype(dtype) }} />
                      {dtype}
                    </span>
                  ))}
                <small>shade = tensor role</small>
              </div>
            </div>
            <WeightMap
              model={activeModel}
              query={query}
              {...(selected ? { selected } : {})}
              {...(tensorNavigation ? { navigationTarget: tensorNavigation } : {})}
              onSelect={(tensor) => {
                setTensorNavigation(undefined);
                selectTensor(tensor);
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
                  <Detail
                    label="Parameters"
                    value={formatParameterCount(selected.shape)}
                  />
                  <Detail label="Size" value={formatBytes(selected.byteLength)} />
                  <Detail
                    label="Role"
                    value={tensorRoleLabel(classifyTensorRole(selected.name))}
                  />
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
                <MousePointerClick aria-hidden="true" />
                Select a tensor to see its exact range, shape, storage, and values.
              </div>
            )}
          </aside>
        </section>
      )}
    </main>
  );
}

type ComparisonFilter = "all" | "changed" | "left-only" | "right-only";

function ComparisonWorkspace({
  models,
  left,
  right,
  compact,
  onLeftChange,
  onRightChange,
  onExit
}: {
  models: ParsedModel[];
  left: ParsedModel;
  right: ParsedModel;
  compact: boolean;
  onLeftChange: (id: string) => void;
  onRightChange: (id: string) => void;
  onExit: () => void;
}) {
  const comparison = useMemo(() => compareModels(left, right), [left, right]);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ComparisonFilter>("all");
  const [resolutionStep, setResolutionStep] = useState(0);
  const [selectedPair, setSelectedPair] = useState<TensorComparison>();
  const [leftNavigation, setLeftNavigation] = useState<TensorNavigationTarget>();
  const [rightNavigation, setRightNavigation] = useState<TensorNavigationTarget>();
  const navigationSequence = useRef(0);

  const indexes = useMemo(() => comparisonIndexes(comparison), [comparison]);
  const matchingPairs = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return [];
    return comparison.tensors.filter(
      (pair) =>
        pair.name.toLowerCase().includes(needle) ||
        pair.left?.dtype.toLowerCase().includes(needle) ||
        pair.right?.dtype.toLowerCase().includes(needle)
    );
  }, [comparison, query]);
  const selectedMatchIndex = selectedPair
    ? matchingPairs.indexOf(selectedPair)
    : -1;
  const referenceFileSize = largestFileSize([left, right]);

  useEffect(() => {
    setSelectedPair(undefined);
    setLeftNavigation(undefined);
    setRightNavigation(undefined);
  }, [left.id, right.id]);

  const selectPair = (pair: TensorComparison) => {
    setSelectedPair(pair);
    const sequence = ++navigationSequence.current;
    setLeftNavigation(
      pair.left ? { tensor: pair.left, sequence } : undefined
    );
    setRightNavigation(
      pair.right ? { tensor: pair.right, sequence } : undefined
    );
  };

  const clearSelection = () => {
    setSelectedPair(undefined);
    setLeftNavigation(undefined);
    setRightNavigation(undefined);
  };

  const selectTensor = (side: "left" | "right", tensor: TensorRecord) => {
    const pair =
      side === "left"
        ? indexes.leftPairs.get(tensorComparisonKey(tensor))
        : indexes.rightPairs.get(tensorComparisonKey(tensor));
    if (pair) selectPair(pair);
  };

  const navigateMatches = (direction: 1 | -1) => {
    if (!matchingPairs.length) return;
    const index =
      selectedMatchIndex < 0
        ? direction > 0
          ? 0
          : matchingPairs.length - 1
        : (selectedMatchIndex + direction + matchingPairs.length) %
          matchingPairs.length;
    selectPair(matchingPairs[index]!);
  };

  return (
    <section className={`wv-comparison${compact ? " compact" : ""}`}>
      <div className="wv-comparison-toolbar">
        <button
          className="wv-button wv-icon-label wv-toolbar-control"
          type="button"
          title="Return to single-model view"
          onClick={onExit}
        >
          <Square className="wv-icon" aria-hidden="true" />
          <span>Single</span>
        </button>
        <label>
          <span>Left</span>
          <ModelSelectControl
            value={left.id}
            models={models}
            onChange={(value) => {
              if (value === right.id) onRightChange(left.id);
              onLeftChange(value);
            }}
          />
        </label>
        <button
          className="wv-swap wv-toolbar-control"
          type="button"
          title="Swap models"
          aria-label="Swap models"
          onClick={() => {
            onLeftChange(right.id);
            onRightChange(left.id);
          }}
        >
          <ArrowLeftRight className="wv-icon" aria-hidden="true" />
        </button>
        <label>
          <span>Right</span>
          <ModelSelectControl
            value={right.id}
            models={models}
            onChange={(value) => {
              if (value === left.id) onLeftChange(right.id);
              onRightChange(value);
            }}
          />
        </label>
        <div className="wv-tensor-search">
          <input
            className="wv-filter"
            aria-label="Search compared tensors"
            placeholder="Search exact tensor names or dtype"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              event.preventDefault();
              navigateMatches(event.shiftKey ? -1 : 1);
            }}
          />
          <span className="wv-search-count">
            {query.trim()
              ? `${selectedMatchIndex + 1}/${matchingPairs.length}`
              : "—"}
          </span>
          <button
            type="button"
            aria-label="Previous tensor match"
            disabled={!matchingPairs.length}
            onClick={() => navigateMatches(-1)}
          >
            <ChevronUp className="wv-icon" aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-label="Next tensor match"
            disabled={!matchingPairs.length}
            onClick={() => navigateMatches(1)}
          >
            <ChevronDown className="wv-icon" aria-hidden="true" />
          </button>
        </div>
        <div className="wv-diff-summary" aria-label="Tensor comparison summary">
          <DiffSummaryButton
            active={filter === "all"}
            label="All"
            value={comparison.tensors.length}
            onClick={() => setFilter("all")}
          />
          <DiffSummaryButton
            active={filter === "changed"}
            label="Changed"
            value={comparison.summary.changed + comparison.summary.ambiguous}
            onClick={() => setFilter("changed")}
          />
          <DiffSummaryButton
            active={filter === "left-only"}
            label="Removed"
            value={comparison.summary["left-only"]}
            onClick={() => setFilter("left-only")}
          />
          <DiffSummaryButton
            active={filter === "right-only"}
            label="Added"
            value={comparison.summary["right-only"]}
            onClick={() => setFilter("right-only")}
          />
          <span className="wv-unchanged-count">
            {comparison.summary.unchanged.toLocaleString()} unchanged
          </span>
          <span
            className="wv-structure-note"
            role="note"
            title="Tensor metadata and encoded layout are compared. Weight values and raw file bytes are not compared."
          >
            <Info className="wv-icon" aria-hidden="true" />
            <span>Metadata only</span>
          </span>
        </div>
      </div>
      <div className="wv-comparison-grid">
        <ComparisonPaneHeader side="LEFT" model={left} />
        <ComparisonPaneHeader side="RIGHT" model={right} />
        <WeightMap
          model={left}
          query={query}
          {...(selectedPair?.left ? { selected: selectedPair.left } : {})}
          {...(leftNavigation ? { navigationTarget: leftNavigation } : {})}
          includeTensor={(tensor) =>
            comparisonStatusVisible(
              indexes.leftStatuses.get(tensorComparisonKey(tensor)),
              filter
            )
          }
          diffStatuses={indexes.leftStatuses}
          referenceFileSize={referenceFileSize}
          resolutionStep={resolutionStep}
          onResolutionStepChange={setResolutionStep}
          {...(selectedPair && !selectedPair.left
            ? { emptyState: "No exact-name counterpart on the left" }
            : {})}
          onSelect={(tensor) => selectTensor("left", tensor)}
          onClearSelection={clearSelection}
        />
        <WeightMap
          model={right}
          query={query}
          {...(selectedPair?.right ? { selected: selectedPair.right } : {})}
          {...(rightNavigation ? { navigationTarget: rightNavigation } : {})}
          includeTensor={(tensor) =>
            comparisonStatusVisible(
              indexes.rightStatuses.get(tensorComparisonKey(tensor)),
              filter
            )
          }
          diffStatuses={indexes.rightStatuses}
          referenceFileSize={referenceFileSize}
          resolutionStep={resolutionStep}
          onResolutionStepChange={setResolutionStep}
          {...(selectedPair && !selectedPair.right
            ? { emptyState: "No exact-name counterpart on the right" }
            : {})}
          onSelect={(tensor) => selectTensor("right", tensor)}
          onClearSelection={clearSelection}
        />
        <ComparisonTensorDetail
          {...(selectedPair?.left ? { tensor: selectedPair.left } : {})}
          {...(selectedPair ? { pair: selectedPair } : {})}
          missingLabel="Not present on the left"
        />
        <ComparisonTensorDetail
          {...(selectedPair?.right ? { tensor: selectedPair.right } : {})}
          {...(selectedPair ? { pair: selectedPair } : {})}
          missingLabel="Not present on the right"
        />
      </div>
    </section>
  );
}

function comparisonIndexes(comparison: ModelComparison) {
  const leftStatuses = new Map<string, TensorComparisonStatus>();
  const rightStatuses = new Map<string, TensorComparisonStatus>();
  const leftPairs = new Map<string, TensorComparison>();
  const rightPairs = new Map<string, TensorComparison>();
  for (const pair of comparison.tensors) {
    if (pair.left) {
      const key = tensorComparisonKey(pair.left);
      leftStatuses.set(key, pair.status);
      leftPairs.set(key, pair);
    }
    if (pair.right) {
      const key = tensorComparisonKey(pair.right);
      rightStatuses.set(key, pair.status);
      rightPairs.set(key, pair);
    }
  }
  return { leftStatuses, rightStatuses, leftPairs, rightPairs };
}

function comparisonStatusVisible(
  status: TensorComparisonStatus | undefined,
  filter: ComparisonFilter
): boolean {
  if (filter === "all") return true;
  if (filter === "changed") return status === "changed" || status === "ambiguous";
  return status === filter;
}

function largestFileSize(models: ParsedModel[]): bigint {
  return models.reduce(
    (largest, model) =>
      model.files.reduce(
        (current, file) => (file.size > current ? file.size : current),
        largest
      ),
    0n
  );
}

function ModelSelectControl({
  value,
  models,
  onChange,
  className,
  "aria-label": ariaLabel
}: {
  value: string;
  models: ParsedModel[];
  onChange: (value: string) => void;
  className?: string;
  "aria-label"?: string;
}) {
  return (
    <SelectControl
      value={value}
      options={models.map((model) => ({ value: model.id, label: model.name }))}
      onChange={onChange}
      {...(className ? { className } : {})}
      {...(ariaLabel ? { "aria-label": ariaLabel } : {})}
    />
  );
}

function SelectControl({
  value,
  options,
  onChange,
  className,
  "aria-label": ariaLabel
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
  className?: string;
  "aria-label"?: string;
}) {
  return (
    <span className={`wv-select-control wv-toolbar-control${className ? ` ${className}` : ""}`}>
      <select
        {...(ariaLabel ? { "aria-label": ariaLabel } : {})}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
      <ChevronDown className="wv-icon" aria-hidden="true" />
    </span>
  );
}

function ComparisonPaneHeader({
  side,
  model
}: {
  side: string;
  model: ParsedModel;
}) {
  const tensorCount = model.files.reduce(
    (total, file) => total + file.tensors.length,
    0
  );
  const totalSize = model.files.reduce(
    (total, file) => total + file.size,
    0n
  );
  const formats = [...new Set(model.files.map((file) => file.format.toUpperCase()))];
  return (
    <header className="wv-comparison-pane-head">
      <span>{side}</span>
      <b>{model.name}</b>
      <small>
        {formats.join(" + ")} · {model.files.length} files ·{" "}
        {tensorCount.toLocaleString()} tensors · {formatBytes(totalSize)}
      </small>
    </header>
  );
}

function ComparisonTensorDetail({
  tensor,
  pair,
  missingLabel
}: {
  tensor?: TensorRecord;
  pair?: TensorComparison;
  missingLabel: string;
}) {
  if (!pair) {
    return (
      <footer className="wv-comparison-detail empty">
        Select a tensor to inspect its exact-name correlation.
      </footer>
    );
  }
  if (!tensor) {
    return <footer className="wv-comparison-detail missing">{missingLabel}</footer>;
  }
  const changes = Object.entries(pair.changes)
    .filter(([, changed]) => changed)
    .map(([name]) => name);
  return (
    <footer className={`wv-comparison-detail ${pair.status}`}>
      <div>
        <span>{pair.status.replace("-", " ")}</span>
        <b>{tensor.name}</b>
      </div>
      <small>
        {tensor.dtype} · {formatShape(tensor.shape)} ·{" "}
        {formatParameterCount(tensor.shape)} params · {formatBytes(tensor.byteLength)}
      </small>
      <code>
        {formatAddress(tensor.byteOffset)} → {formatAddress(tensorEnd(tensor))}
      </code>
      {changes.length > 0 && <em>Changed: {changes.join(", ")}</em>}
    </footer>
  );
}

function DiffSummaryButton({
  active,
  label,
  value,
  onClick
}: {
  active: boolean;
  label: string;
  value: number;
  onClick: () => void;
}) {
  return (
    <button
      className={active ? "active" : ""}
      type="button"
      onClick={onClick}
    >
      <b>{value.toLocaleString()}</b>
      <span>{label}</span>
    </button>
  );
}

function WeightMap({
  model,
  query,
  selected,
  navigationTarget,
  includeTensor,
  diffStatuses,
  referenceFileSize,
  resolutionStep: controlledResolutionStep,
  onResolutionStepChange,
  emptyState,
  onSelect,
  onClearSelection
}: {
  model: ParsedModel;
  query: string;
  selected?: TensorRecord;
  navigationTarget?: TensorNavigationTarget;
  includeTensor?: (tensor: TensorRecord) => boolean;
  diffStatuses?: Map<string, TensorComparisonStatus>;
  referenceFileSize?: bigint;
  resolutionStep?: number;
  onResolutionStepChange?: (step: number) => void;
  emptyState?: string;
  onSelect: (tensor: TensorRecord) => void;
  onClearSelection?: () => void;
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
  const [internalResolutionStep, setInternalResolutionStep] = useState(0);
  const resolutionStep = controlledResolutionStep ?? internalResolutionStep;
  const setResolutionStep = (step: number) => {
    if (controlledResolutionStep === undefined) setInternalResolutionStep(step);
    onResolutionStepChange?.(step);
  };
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
              (tensor.name.toLowerCase().includes(needle) ||
                tensor.dtype.toLowerCase().includes(needle)) &&
              (includeTensor?.(tensor) ?? true)
          : includeTensor,
        resolutionStep,
        referenceFileSize
      );
    },
    [includeTensor, model, query, referenceFileSize, resolutionStep, size.width]
  );
  const maxVerticalScroll = addressMapMaxScrollY(layout, size.height, zoom);

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
    setResolutionStep(0);
    setOffset({ x: 0, y: 0 });
  }, [model.id]);

  useEffect(() => {
    setOffset((current) => clampOffset(current, layout, size, zoom));
  }, [layout, size, zoom]);

  useEffect(() => {
    if (!navigationTarget) return;
    const fileLayout = layout.files.find(
      (candidate) => candidate.file.id === navigationTarget.tensor.fileId
    );
    const rect = fileLayout?.spans.find(
      (span) =>
        span.tensor === navigationTarget.tensor ||
        (span.tensor?.id === navigationTarget.tensor.id &&
          span.tensor.fileId === navigationTarget.tensor.fileId)
    )?.rects[0];
    if (!rect) return;
    setOffset(
      clampOffset(
        {
          x: size.width / 2 - (rect.x + rect.width / 2) * zoom,
          y: size.height / 2 - (rect.y + rect.height / 2) * zoom
        },
        layout,
        size,
        zoom
      )
    );
    setHover(undefined);
  }, [layout, navigationTarget, size, zoom]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = Math.max(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(size.width * ratio);
    canvas.height = Math.floor(size.height * ratio);
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    const context = canvas.getContext("2d");
    if (!context) return;
    const theme = readCanvasTheme(canvas);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, size.width, size.height);
    context.save();
    context.translate(offset.x, offset.y);
    context.scale(zoom, zoom);
    drawAddressMap(
      context,
      layout,
      selected,
      diffStatuses,
      zoom,
      offset,
      size,
      theme,
      ratio
    );
    context.restore();
    drawAddressRulerBackground(context, size, theme, ratio);
  }, [diffStatuses, layout, offset, selected, size, themeRevision, zoom]);

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
      if (event.ctrlKey || event.metaKey) {
        const rect = canvas.getBoundingClientRect();
        setZoomAt(
          zoom * (event.deltaY < 0 ? 1.18 : 0.85),
          event.clientX - rect.left,
          event.clientY - rect.top
        );
        return;
      }
      const deltaScale =
        event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? size.height
            : 1;
      setOffset((current) =>
        clampOffset(
          {
            x: current.x - event.deltaX * deltaScale,
            y: current.y - event.deltaY * deltaScale
          },
          layout,
          size,
          zoom
        )
      );
      setHover(undefined);
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
            else onClearSelection?.();
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
      {emptyState && (
        <div className="wv-map-empty-counterpart" role="status">
          <CircleSlash2 aria-hidden="true" />
          {emptyState}
        </div>
      )}
      <AddressRulerOverlay
        layout={layout}
        zoom={zoom}
        offset={offset}
        viewport={size}
      />
      <div className="wv-view-controls">
        <div className="wv-resolution">
          <button
            aria-label="Finer resolution"
            disabled={resolutionStep <= -2}
            onClick={() => setResolutionStep(Math.max(-2, resolutionStep - 1))}
          >
            <Minus className="wv-icon" aria-hidden="true" />
          </button>
          <span title="Shared resolution for every file">
            {formatBytes(layout.bytesPerCell)} / cell
          </span>
          <button
            aria-label="Coarser resolution"
            disabled={resolutionStep >= 8}
            onClick={() => setResolutionStep(Math.min(8, resolutionStep + 1))}
          >
            <Plus className="wv-icon" aria-hidden="true" />
          </button>
        </div>
        <div className="wv-zoom">
          <button
            aria-label="Zoom out"
            onClick={() => setZoomAt(zoom / 1.4, size.width / 2, size.height / 2)}
          >
            <Minus className="wv-icon" aria-hidden="true" />
          </button>
          <span>{Math.round(zoom * 100)}%</span>
          <button
            aria-label="Zoom in"
            onClick={() => setZoomAt(zoom * 1.4, size.width / 2, size.height / 2)}
          >
            <Plus className="wv-icon" aria-hidden="true" />
          </button>
          <button
            aria-label="Reset view"
            onClick={() => {
              setZoom(1);
              setResolutionStep(0);
              setOffset({ x: 0, y: 0 });
            }}
          >
            <RotateCcw className="wv-icon" aria-hidden="true" />
          </button>
        </div>
      </div>
      <VerticalMapScrollbar
        viewportHeight={size.height}
        max={maxVerticalScroll}
        value={-offset.y}
        onChange={(value) =>
          setOffset((current) =>
            clampOffset({ ...current, y: -value }, layout, size, zoom)
          )
        }
      />
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
              <span>
                {tensorRoleLabel(classifyTensorRole(hover.hit.tensor.name))} ·{" "}
                {hover.hit.tensor.dtype} · {formatShape(hover.hit.tensor.shape)}
              </span>
              <span>
                {formatParameterCount(hover.hit.tensor.shape)} params ·{" "}
                {formatBytes(hover.hit.tensor.byteLength)}
              </span>
              <code>Pointer: {formatAddress(hover.hit.address)}</code>
              <code>
                Tensor: {formatAddress(hover.hit.start)} → {formatAddress(hover.hit.end)}
              </code>
            </>
          ) : hover.hit.kind === "filtered" && hover.hit.tensor ? (
            <>
              <b>{hover.hit.tensor.name} · filtered</b>
              <span>
                {tensorRoleLabel(classifyTensorRole(hover.hit.tensor.name))} ·{" "}
                {hover.hit.tensor.dtype} · {formatShape(hover.hit.tensor.shape)}
              </span>
              <span>
                {formatParameterCount(hover.hit.tensor.shape)} params ·{" "}
                {formatBytes(hover.hit.tensor.byteLength)}
              </span>
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

function VerticalMapScrollbar({
  viewportHeight,
  max,
  value,
  onChange
}: {
  viewportHeight: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    startY: number;
    startValue: number;
    travel: number;
  } | undefined>(undefined);
  if (max <= 0) return null;

  const trackHeight = Math.max(1, viewportHeight - 68);
  const scrollableHeight = max + viewportHeight;
  const thumbHeight = Math.min(
    trackHeight,
    Math.max(32, trackHeight * (viewportHeight / scrollableHeight))
  );
  const travel = Math.max(1, trackHeight - thumbHeight);
  const current = Math.min(max, Math.max(0, value));
  const thumbTop = (current / max) * travel;
  const setClamped = (next: number) =>
    onChange(Math.min(max, Math.max(0, next)));

  return (
    <div
      ref={trackRef}
      className="wv-map-scrollbar"
      role="scrollbar"
      aria-label="Map vertical scroll"
      aria-orientation="vertical"
      aria-valuemin={0}
      aria-valuemax={Math.round(max)}
      aria-valuenow={Math.round(current)}
      tabIndex={0}
      onKeyDown={(event) => {
        const smallStep = Math.max(40, viewportHeight * 0.1);
        if (event.key === "ArrowUp") setClamped(current - smallStep);
        else if (event.key === "ArrowDown") setClamped(current + smallStep);
        else if (event.key === "PageUp") setClamped(current - viewportHeight * 0.9);
        else if (event.key === "PageDown") setClamped(current + viewportHeight * 0.9);
        else if (event.key === "Home") setClamped(0);
        else if (event.key === "End") setClamped(max);
        else return;
        event.preventDefault();
      }}
      onPointerDown={(event) => {
        if (event.target !== event.currentTarget) return;
        const rect = event.currentTarget.getBoundingClientRect();
        setClamped(
          ((event.clientY - rect.top - thumbHeight / 2) / travel) * max
        );
      }}
    >
      <div
        className="wv-map-scrollbar-thumb"
        style={{ height: thumbHeight, transform: `translateY(${thumbTop}px)` }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = {
            pointerId: event.pointerId,
            startY: event.clientY,
            startValue: current,
            travel
          };
          event.preventDefault();
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag || drag.pointerId !== event.pointerId) return;
          setClamped(
            drag.startValue +
              ((event.clientY - drag.startY) / drag.travel) * max
          );
        }}
        onPointerUp={(event) => {
          if (dragRef.current?.pointerId !== event.pointerId) return;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
          dragRef.current = undefined;
        }}
        onPointerCancel={() => {
          dragRef.current = undefined;
        }}
      />
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
  diffStatuses: Map<string, TensorComparisonStatus> | undefined,
  zoom: number,
  offset: { x: number; y: number },
  viewport: { width: number; height: number },
  theme: CanvasTheme,
  pixelRatio: number
) {
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
      const diffStatus = span.tensor
        ? diffStatuses?.get(tensorComparisonKey(span.tensor))
        : undefined;
      const color =
        span.kind === "metadata"
          ? theme.metadata
          : colorForTensor(span.tensor!);
      context.fillStyle = color;
      context.globalAlpha =
        span.kind === "metadata"
          ? 0.92
          : diffStatus === "unchanged"
            ? 0.38
            : 0.9;
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
      const x = snapStrokeCoordinate(
        fileLayout.gridX +
          (column * fileLayout.gridWidth) / fileLayout.columns,
        zoom,
        offset.x,
        pixelRatio
      );
      context.beginPath();
      context.moveTo(x, fileLayout.gridY);
      context.lineTo(x, gridBottom);
      context.stroke();
    }
    for (let row = 0; row <= fileLayout.rowCount; row++) {
      const y = snapStrokeCoordinate(
        fileLayout.gridY + row * fileLayout.rowHeight,
        zoom,
        offset.y,
        pixelRatio
      );
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
      const diffStatus = diffStatuses?.get(tensorComparisonKey(span.tensor));
      context.strokeStyle =
        diffStatus === "changed" || diffStatus === "ambiguous"
          ? theme.diffChanged
          : diffStatus === "left-only"
            ? theme.diffRemoved
            : diffStatus === "right-only"
              ? theme.diffAdded
            : theme.mapBackground;
      context.globalAlpha = diffStatus ? 1 : 0.92;
      context.lineWidth =
        diffStatus && diffStatus !== "unchanged" ? 2 / zoom : 1 / zoom;
      for (const rect of span.rects) {
        if (isVisible(rect, visibleTop, visibleBottom)) {
          context.strokeRect(rect.x, rect.y, rect.width, rect.height);
          if (diffStatus === "left-only" || diffStatus === "right-only") {
            drawDiffPattern(
              context,
              rect,
              zoom,
              diffStatus === "right-only" ? "added" : "removed"
            );
          }
        }
      }
      context.globalAlpha = 1;
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

    const selectedSpan = fileLayout.spans.find(
      (span) => span.visible && span.tensor?.id === selected?.id
    );
    if (selectedSpan) {
      context.strokeStyle = theme.selection;
      context.lineWidth = 2 / zoom;
      for (const rect of selectedSpan.rects) {
        if (isVisible(rect, visibleTop, visibleBottom)) {
          context.strokeRect(rect.x, rect.y, rect.width, rect.height);
        }
      }
    }
  }
}

function AddressRulerOverlay({
  layout,
  zoom,
  offset,
  viewport
}: {
  layout: AddressMapLayout;
  zoom: number;
  offset: { x: number; y: number };
  viewport: { width: number; height: number };
}) {
  return (
    <div className="wv-address-overlay" aria-hidden="true">
      {layout.files.map((fileLayout) => {
        const headerY = offset.y + (fileLayout.gridY - 15) * zoom;
        return (
          <div key={fileLayout.file.id}>
            {headerY > -20 && headerY < viewport.height + 20 && (
              <>
                <strong
                  className="wv-address-file-name"
                  style={{ top: headerY }}
                >
                  {fileLayout.file.name}
                </strong>
                <span
                  className="wv-address-scale"
                  style={{ top: headerY }}
                >
                  {formatBytes(fileLayout.bytesPerCell)} / cell ·{" "}
                  {formatBytes(fileLayout.bytesPerRow)} / row
                  {formattedAlignment(fileLayout.file)
                    ? ` · ${formattedAlignment(fileLayout.file)} alignment`
                    : ""}
                </span>
              </>
            )}
            {Array.from({ length: fileLayout.rowCount }, (_, row) => {
              const y =
                offset.y +
                (fileLayout.gridY + row * fileLayout.rowHeight) * zoom +
                (fileLayout.rowHeight * zoom) / 2;
              return y < -10 || y > viewport.height + 10 ? null : (
                <code
                  className="wv-address-row"
                  key={row}
                  style={{ top: y }}
                >
                  {formatAddress(BigInt(row) * fileLayout.bytesPerRow)}
                </code>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function drawAddressRulerBackground(
  context: CanvasRenderingContext2D,
  viewport: { width: number; height: number },
  theme: CanvasTheme,
  pixelRatio: number
) {
  context.fillStyle = theme.surface;
  context.fillRect(0, 0, 104, viewport.height);
  context.strokeStyle = theme.border;
  context.beginPath();
  const borderX = snapStrokeCoordinate(104, 1, 0, pixelRatio);
  context.moveTo(borderX, 0);
  context.lineTo(borderX, viewport.height);
  context.stroke();
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
  const visibleRects = rects.filter((candidate) =>
    isVisible(candidate, visibleTop, visibleBottom)
  );
  const firstRect = visibleRects[0];
  const widestRect = visibleRects.reduce<AddressRect | undefined>(
    (widest, candidate) =>
      !widest || candidate.width > widest.width ? candidate : widest,
    undefined
  );
  const rect =
    firstRect && widestRect && firstRect.width >= widestRect.width * 0.75
      ? firstRect
      : widestRect;
  if (!rect || rect.width * zoom < 24 || rect.height * zoom < 9) return;

  const pixelWidth = rect.width * zoom;
  const pixelHeight = rect.height * zoom;
  const fontSize = (pixelHeight >= 14 && pixelWidth >= 48 ? 11 : 9) / zoom;
  const padding = Math.min(4, pixelWidth * 0.1) / zoom;
  const availableWidth = rect.width - padding * 2;

  context.save();
  context.beginPath();
  context.rect(rect.x, rect.y, rect.width, rect.height);
  context.clip();
  context.fillStyle = theme.labelText;
  context.font = `600 ${fontSize}px "SF Mono", Menlo, Monaco, Consolas, monospace`;
  context.textBaseline = "middle";
  context.fillText(
    fitCanvasText(context, tensor.name, availableWidth),
    rect.x + padding,
    rect.y + Math.min(pixelHeight / 2, 7) / zoom
  );
  if (pixelHeight >= 27 && pixelWidth >= 64) {
    context.font = `${10 / zoom}px "SF Mono", Menlo, Monaco, Consolas, monospace`;
    context.textBaseline = "top";
    context.fillText(
      fitCanvasText(
        context,
        `${tensorRoleLabel(classifyTensorRole(tensor.name))} · ${tensor.dtype} · ${formatBytes(tensor.byteLength)}`,
        availableWidth
      ),
      rect.x + padding,
      rect.y + 16 / zoom
    );
  }
  context.restore();
}

function snapStrokeCoordinate(
  value: number,
  scale: number,
  translation: number,
  pixelRatio: number
): number {
  const physicalLineWidth = Math.max(1, Math.round(pixelRatio));
  const phase = physicalLineWidth % 2 === 0 ? 0 : 0.5;
  const screenCoordinate = translation + value * scale;
  const snappedPhysical =
    Math.round(screenCoordinate * pixelRatio - phase) + phase;
  return (snappedPhysical / pixelRatio - translation) / scale;
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
  diffChanged: string;
  diffAdded: string;
  diffRemoved: string;
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
    labelText: "#071019",
    diffChanged: read("--wv-diff-changed", "#ffb020"),
    diffAdded: read("--wv-diff-added", "#65d98a"),
    diffRemoved: read("--wv-diff-removed", "#ff5c8a")
  };
}

function drawDiffPattern(
  context: CanvasRenderingContext2D,
  rect: AddressRect,
  zoom: number,
  kind: "added" | "removed"
) {
  const step = 11 / zoom;
  const mark = 3 / zoom;
  context.save();
  context.beginPath();
  context.rect(rect.x, rect.y, rect.width, rect.height);
  context.clip();
  context.globalAlpha = 0.5;
  context.lineWidth = 1 / zoom;
  for (let y = rect.y + step / 2; y < rect.y + rect.height; y += step) {
    for (let x = rect.x + step / 2; x < rect.x + rect.width; x += step) {
      context.beginPath();
      context.moveTo(x - mark, y);
      context.lineTo(x + mark, y);
      if (kind === "added") {
        context.moveTo(x, y - mark);
        context.lineTo(x, y + mark);
      }
      context.stroke();
    }
  }
  context.restore();
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
  tensor: TensorRecord
): string {
  return colorForTensorRole(
    colorForDtype(tensor.dtype),
    classifyTensorRole(tensor.name)
  );
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
  const minY = -addressMapMaxScrollY(layout, viewport.height, zoom);
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
  label,
  large = false,
  onFilesSelected,
  onPickerResult
}: {
  children: ReactNode;
  label?: string;
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
      <span className="wv-icon-label" aria-hidden="true">{children}</span>
      <input
        ref={inputRef}
        type="file"
        multiple
        aria-label={label ?? (typeof children === "string" ? children : "Open files")}
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
        <SelectControl
          className="wv-file-select"
          aria-label="Metadata file"
          value={file.id}
          options={files.map((candidate) => ({
            value: candidate.id,
            label: candidate.name
          }))}
          onChange={onFileChange}
        />
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
