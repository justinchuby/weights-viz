import { useEffect, useId, useMemo, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import {
  Binary,
  BookOpen,
  Boxes,
  Cpu,
  Gauge,
  Scale,
  Settings2,
  X
} from "lucide-react";
import type { TensorRecord, WeightFormat } from "@weights-viz/core";
import {
  createDtypeEducation,
  formatDecimal,
  type BlockEncoding,
  type EncodingSegment,
  type PackingGroup
} from "./dtype-education";
import { formatBytes, formatParameterCount } from "./format";
import { Q40DecodeAnimation } from "./Q40DecodeAnimation";
import {
  DtypeEncodingAnimation,
  dtypeAnimationKind
} from "./DtypeEncodingAnimation";
import {
  isKQuantDtype,
  K_QUANT_LAYOUTS,
  KQuantAnimation
} from "./KQuantAnimation";
import { ggufStorageLayout } from "./gguf-storage-layouts";

interface DtypeExplorerProps {
  format: WeightFormat;
  tensor: TensorRecord;
  onClose: () => void;
  showTensorMetrics?: boolean;
}

export function DtypeExplorer({
  format,
  tensor,
  onClose,
  showTensorMetrics = true
}: DtypeExplorerProps) {
  const lesson = useMemo(
    () => createDtypeEducation(format, tensor),
    [format, tensor]
  );
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const titleId = useId();
  const summaryId = useId();
  const previousFocusRef = useRef<HTMLElement | null>(
    typeof document !== "undefined" && document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
  );
  const animationKind = dtypeAnimationKind(format, lesson.dtype);
  const hasGgufBlockContract = format === "gguf" && lesson.block !== undefined;

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const appShell = document.querySelector<HTMLElement>(".wv-shell");
    const previousInert = appShell?.inert ?? false;
    document.body.style.overflow = "hidden";
    if (appShell) appShell.inert = true;
    closeButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable?.length) return;
      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      if (appShell) appShell.inert = previousInert;
      window.removeEventListener("keydown", handleKeyDown);
      previousFocusRef.current?.focus();
    };
  }, [onClose]);

  return createPortal(
    <div
      className="wv-dtype-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        ref={dialogRef}
        className="wv-dtype-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={summaryId}
      >
        <header className="wv-dtype-head">
          <div className="wv-dtype-heading">
            <span>{lesson.formatLabel}</span>
            <div>
              <BookOpen aria-hidden="true" />
              <div>
                <h2 id={titleId}>{lesson.dtype}</h2>
                <p>{lesson.family}</p>
              </div>
            </div>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            aria-label="Close dtype explanation"
            onClick={onClose}
          >
            <X aria-hidden="true" />
          </button>
        </header>

        <div className="wv-dtype-body">
          <p className="wv-dtype-summary" id={summaryId}>{lesson.summary}</p>

          {showTensorMetrics && (
            <div className="wv-dtype-metrics">
              <Metric
                label="Tensor values"
                value={formatParameterCount(tensor.shape)}
              />
              <Metric label="Encoded size" value={formatBytes(tensor.byteLength)} />
              {lesson.tensorBitsPerValue !== undefined && (
                <Metric
                  label="Actual bits / value"
                  value={formatDecimal(lesson.tensorBitsPerValue)}
                />
              )}
              {lesson.f32CompressionRatio !== undefined && (
                <Metric
                  label="F32 / stored size"
                  value={`${formatDecimal(lesson.f32CompressionRatio)}×`}
                />
              )}
            </div>
          )}

          <div className="wv-dtype-grid">
            {lesson.segments.length > 0 && !hasGgufBlockContract && (
              <LessonCard
                icon={<Binary aria-hidden="true" />}
                title={`${lesson.bitsPerValue ?? "Variable"}-bit value`}
                subtitle="Bit layout"
              >
                <BitLayout segments={lesson.segments} />
              </LessonCard>
            )}

            {lesson.packing && !hasGgufBlockContract && (
              <LessonCard
                icon={<Boxes aria-hidden="true" />}
                title="Packing"
                subtitle={`${lesson.packing.values} value${
                  lesson.packing.values === 1 ? "" : "s"
                } in ${lesson.packing.bytes} byte${
                  lesson.packing.bytes === 1 ? "" : "s"
                }`}
              >
                <PackingDiagram packing={lesson.packing} />
              </LessonCard>
            )}

            {lesson.block && (
              <LessonCard
                icon={<Boxes aria-hidden="true" />}
                title="Storage contract"
                subtitle={`${lesson.block.elements} weights · ${
                  lesson.block.bytes
                } bytes · ${formatDecimal(
                  lesson.block.effectiveBitsPerValue
                )} bpw`}
                wide
              >
                <BlockDiagram block={lesson.block} dtype={lesson.dtype} />
              </LessonCard>
            )}

            {lesson.formula && (
              <LessonCard
                icon={<Scale aria-hidden="true" />}
                title="Decode one weight"
                subtitle="Conceptual formula"
                wide
              >
                <div className="wv-dtype-formula">
                  <code>{lesson.formula.expression}</code>
                  <p>{lesson.formula.explanation}</p>
                </div>
              </LessonCard>
            )}
          </div>

          <div className="wv-dtype-storage">
            <strong>How {lesson.formatLabel} stores it</strong>
            <p>{lesson.storageNote}</p>
          </div>

          <div className="wv-dtype-concepts">
            {lesson.concepts.map((concept) => (
              <article key={concept.term}>
                <strong>{concept.term}</strong>
                <p>{concept.explanation}</p>
              </article>
            ))}
          </div>

          {animationKind === "q4" ? (
            <Q40DecodeAnimation />
          ) : animationKind === "k-quant" && isKQuantDtype(lesson.dtype) ? (
            <KQuantAnimation dtype={lesson.dtype} />
          ) : (
            <DtypeEncodingAnimation format={format} lesson={lesson} />
          )}

          {lesson.quantization && (
            <section className="wv-quant-guide">
              <header>
                <span>GGUF QUANTIZATION IN PRACTICE</span>
                <h3>From source weights to a fast dot product</h3>
                <p>{lesson.quantization.purpose}</p>
              </header>
              <div className="wv-quant-columns">
                <ProcessCard
                  icon={<Cpu aria-hidden="true" />}
                  title="During inference"
                  steps={lesson.quantization.runtimeSteps}
                />
                <ProcessCard
                  icon={<Gauge aria-hidden="true" />}
                  title="Why kernels are fast"
                  steps={lesson.quantization.optimizations}
                />
                <ProcessCard
                  icon={<Settings2 aria-hidden="true" />}
                  title="When the model is quantized"
                  steps={lesson.quantization.creationSteps}
                />
              </div>
              <div className="wv-quant-parameters">
                <h4>How the parameters are chosen</h4>
                {lesson.quantization.parameters.map((parameter) => (
                  <div key={parameter.name}>
                    <strong>{parameter.name}</strong>
                    <p>{parameter.selection}</p>
                  </div>
                ))}
              </div>
              <div className="wv-quant-tradeoff">
                <Scale aria-hidden="true" />
                <p>{lesson.quantization.tradeoff}</p>
              </div>
            </section>
          )}
        </div>
      </section>
    </div>,
    document.body
  );
}

function ProcessCard({
  icon,
  title,
  steps
}: {
  icon: ReactNode;
  title: string;
  steps: string[];
}) {
  return (
    <article>
      <header>
        {icon}
        <strong>{title}</strong>
      </header>
      <ol>
        {steps.map((step) => <li key={step}>{step}</li>)}
      </ol>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function LessonCard({
  icon,
  title,
  subtitle,
  wide = false,
  children
}: {
  icon: ReactNode;
  title: string;
  subtitle: string;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <article className={`wv-dtype-card${wide ? " wide" : ""}`}>
      <header>
        {icon}
        <div>
          <strong>{title}</strong>
          <span>{subtitle}</span>
        </div>
      </header>
      {children}
    </article>
  );
}

function BitLayout({ segments }: { segments: EncodingSegment[] }) {
  return (
    <div className="wv-bit-layout">
      {segments.map((segment, index) => (
        <div
          className={`wv-bit-segment ${segment.tone}`}
          key={`${segment.label}:${index}`}
          style={{ flexGrow: segment.bits }}
        >
          <strong>{segment.label}</strong>
          <span>{segment.bits}b</span>
        </div>
      ))}
    </div>
  );
}

function PackingDiagram({ packing }: { packing: PackingGroup }) {
  return (
    <div className="wv-packing-diagram">
      <div className="wv-packed-values">
        {Array.from({ length: packing.values }, (_, index) => (
          <span key={index} style={{ flexGrow: packing.bitsPerValue }}>
            v{index}
            <small>{packing.bitsPerValue}b</small>
          </span>
        ))}
      </div>
      <div className="wv-packed-bytes">
        {packing.bytes <= 8 ? (
          Array.from({ length: packing.bytes }, (_, index) => (
            <span key={index}>byte {index}</span>
          ))
        ) : (
          <span>{packing.bytes} consecutive bytes</span>
        )}
      </div>
    </div>
  );
}

function BlockDiagram({
  block,
  dtype
}: {
  block: BlockEncoding;
  dtype: string;
}) {
  const fields = ggufStorageLayout(dtype);
  const kSections = isKQuantDtype(dtype)
    ? K_QUANT_LAYOUTS[dtype].sections
    : undefined;
  return (
    <div className="wv-block-diagram">
      <div>
        {block.sections.map((section) => (
          <span
            className={section.tone}
            key={section.label}
            style={{ flexGrow: section.bits }}
          >
            {section.label}
            <small>{section.bits} bits</small>
          </span>
        ))}
      </div>
      <p>
        The dtype ABI fixes one block at <strong>{block.elements} weights</strong>;
        a file cannot choose another block size. Metadata is shared by that group.
      </p>
      {(fields || kSections) && (
        <ul className="wv-block-fields">
          {fields
            ? fields.map((field) => (
                <li key={field.name}>
                  <code>{field.name}</code>
                  <span>{field.type} × {field.count} · {field.bytes} B</span>
                  <small>{field.role}</small>
                </li>
              ))
            : kSections?.map((section) => (
                <li key={section.label}>
                  <code>{section.label}</code>
                  <span>{section.bytes} B</span>
                  <small>
                    {section.tone === "global"
                      ? "shared by all 256 weights"
                      : section.tone === "local"
                        ? "compact parameters for each sub-block"
                        : "packed low-bit weight codes"}
                  </small>
                </li>
              ))}
        </ul>
      )}
    </div>
  );
}
