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
  type DtypeEducation,
  type EncodingSegment,
  type PackingGroup
} from "./dtype-education";
import { formatBytes, formatParameterCount } from "./format";
import { Q40DecodeAnimation } from "./Q40DecodeAnimation";
import {
  DtypeEncodingAnimation,
  dtypeAnimationKind
} from "./DtypeEncodingAnimation";
import { GgufQuantContractAnimation } from "./GgufQuantContractAnimation";
import {
  isKQuantDtype,
  K_QUANT_LAYOUTS,
  kQuantContractDetails,
  kQuantFieldMeaning,
  kQuantSubBlockStorage,
  KQuantAnimation
} from "./KQuantAnimation";
import {
  ggufQuantContract,
  type GgufQuantSymbol
} from "./gguf-quant-contracts";
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
                <BlockDiagram
                  block={lesson.block}
                  dtype={lesson.dtype}
                  formula={lesson.formula}
                />
              </LessonCard>
            )}

            {lesson.formula && !hasGgufBlockContract && (
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
          ) : animationKind === "gguf-contract" ? (
            <GgufQuantContractAnimation dtype={lesson.dtype} />
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
  dtype,
  formula
}: {
  block: BlockEncoding;
  dtype: string;
  formula?: DtypeEducation["formula"];
}) {
  const fields = ggufStorageLayout(dtype);
  const exactContract = ggufQuantContract(dtype);
  const kLayout = isKQuantDtype(dtype) ? K_QUANT_LAYOUTS[dtype] : undefined;
  const kContract = isKQuantDtype(dtype) ? kQuantContractDetails(dtype) : undefined;
  const fallback = dtype === "Q4_0" ? q40ContractDetails() : undefined;
  const metadata = exactContract?.metadata ?? kContract?.metadata ?? fallback?.metadata;
  const codes = exactContract?.codes ?? kContract?.codes ?? fallback?.codes;
  const packing = exactContract?.packing ?? kContract?.packing ?? fallback?.packing;
  const terms = exactContract?.symbols ?? kContract?.terms ?? fallback?.terms;
  const derivation = kContract?.derivation;
  let byteOffset = 0;
  const fieldOffsets = fields?.map((field) => {
    const start = byteOffset;
    byteOffset += field.bytes;
    return { field, start, end: byteOffset - 1 };
  });

  return (
    <div className="wv-block-diagram">
      {fieldOffsets && (
        <section className="wv-storage-vocabulary">
          <header>
            <strong>Field names</strong>
            <small>what each physical part contains and where it applies</small>
          </header>
          <dl className="wv-storage-field-definitions">
            {fieldOffsets.map(({ field, start, end }) => (
              <div key={field.name}>
                <dt>
                  <code>{field.name}</code>
                  <strong>{fieldNameMeaning(dtype, field.name)}</strong>
                </dt>
                <dd>
                  <span>{field.type}[{field.count}] · {field.bytes} B · record bytes {start}…{end}</span>
                  <span>Contents: {field.role}</span>
                  <span>Scope: {fieldScope(dtype, field.name)}</span>
                </dd>
              </div>
            ))}
          </dl>
        </section>
      )}
      {terms && terms.length > 0 && (
        <section className="wv-storage-vocabulary">
          <header>
            <strong>Terms used by the formula and animation</strong>
            <small>symbol → meaning → source</small>
          </header>
          <dl className="wv-storage-term-definitions">
            {terms.map((term, index) => (
              <div key={`${term.symbol}-${index}`}>
                <dt>
                  <code>{term.symbol}</code>
                  <strong>{term.meaning}</strong>
                </dt>
                <dd>{term.source}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
      {derivation && kLayout && (
        <section className="wv-storage-derivation">
          <header>
            <strong>
              {kLayout.minBits !== undefined
                ? "How encoder-only a[g] / b[g] become stored s[g] / m[g]"
                : kLayout.dtype === "Q6_K"
                  ? "How encoder-only a[g] becomes stored signed_s[g]"
                  : "How encoder-only a[g] becomes stored s[g]"}
            </strong>
            <small>follow one sub-block from source weights to physical scales[] bits</small>
          </header>
          <ol>
            {derivation.map((step) => (
              <li key={step.title}>
                <strong>{step.title}</strong>
                <code>{step.expression}</code>
                <p>{step.detail}</p>
              </li>
            ))}
          </ol>
        </section>
      )}
      {metadata && codes && !derivation && (
        <div className="wv-storage-rules wv-storage-rules-production">
          <StorageRuleList title="How metadata is produced" items={metadata} />
          <StorageRuleList title="How codes are produced" items={codes} />
        </div>
      )}
      <header className="wv-storage-layout-heading">
        <strong>{kLayout ? "Physical super-block record" : "Physical block record"}</strong>
        <small>
          {kLayout
            ? `1 record = 1 super-block = 256 weights = ${kLayout.bytes} bytes`
            : "left-to-right ABI order; width is proportional to stored bytes"}
        </small>
      </header>
      <div className="wv-storage-field-strip">
        {fieldOffsets
          ? fieldOffsets.map(({ field, start, end }) => (
              <span
                className={storageFieldTone(field.name)}
                key={field.name}
                style={{ flexGrow: field.bytes }}
              >
                {field.name}: {field.type}[{field.count}]
                <small>{field.bytes} B · bytes {start}…{end}</small>
              </span>
            ))
          : block.sections.map((section) => (
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
      {kLayout ? (
        <section className="wv-storage-subblocks">
          <header>
            <strong>Zoomed logical sub-blocks</strong>
            <small>not contiguous byte slices: each view gathers bits from the physical fields above</small>
          </header>
          <div>
            {Array.from({ length: kLayout.subBlocks }, (_, group) => {
              const start = group * kLayout.valuesPerSubBlock;
              const end = start + kLayout.valuesPerSubBlock - 1;
              const storage = kQuantSubBlockStorage(kLayout.dtype, group);
              return (
                <span key={group}>
                  <strong>sub-block {group}: weights {start}…{end}</strong>
                  <small>logical q[{start}…{end}] is reconstructed from:</small>
                  <b>local metadata</b>
                  {storage.metadata.map((location) => (
                    <code key={location}>{location}</code>
                  ))}
                  <b>packed q bits</b>
                  {storage.codes.map((location) => (
                    <code key={location}>{location}</code>
                  ))}
                </span>
              );
            })}
          </div>
          <p>
            Each <code>q[i]</code> is one decoded integer code. <code>qs</code>{" "}
            is the stored byte array containing packed bits for many q values;
            a sub-block view gathers its q bits and metadata from the listed
            locations rather than enlarging one contiguous record segment.
          </p>
        </section>
      ) : (
        <p>
          The dtype ABI fixes one block at <strong>{block.elements} weights</strong>;
          a file cannot choose another block size. Metadata is shared by that group.
        </p>
      )}
      {metadata && codes && packing && derivation && (
        <div className="wv-storage-rules">
          <StorageRuleList title="Metadata meanings" items={metadata} />
          <StorageRuleList title="Code meanings and ranges" items={codes} />
          <StorageRuleList title="Exact byte / bit layout" items={packing} />
        </div>
      )}
      {packing && !derivation && (
        <div className="wv-storage-rules wv-storage-rules-layout">
          <StorageRuleList title="Exact byte / bit layout" items={packing} />
        </div>
      )}
      {formula && (
        <section className="wv-storage-reconstruction">
          <header>
            <strong>Reconstruction using the terms above</strong>
          </header>
          <code>{formula.expression}</code>
          <p>{formula.explanation}</p>
        </section>
      )}
    </div>
  );
}

function StorageRuleList({
  title,
  items
}: {
  title: string;
  items: readonly string[];
}) {
  return (
    <section>
      <strong>{title}</strong>
      <ol>
        {items.map((item) => <li key={item}>{item}</li>)}
      </ol>
    </section>
  );
}

function fieldNameMeaning(dtype: string, name: string): string {
  if (isKQuantDtype(dtype)) return kQuantFieldMeaning(name);
  if (name === "qs") return "packed quantized symbols / codes";
  if (name === "qh") return "packed high bits of quantized codes";
  if (name === "ql") return "packed low bits of quantized codes";
  if (name === "hmask") return "high-bit mask for quantized codes";
  if (name === "d") return dtype === "NVFP4" ? "local scale codes" : "scale or scale step";
  if (name === "dmin") return "minimum-magnitude scale step";
  if (name === "m") return "stored affine minimum";
  if (name === "s") return "stored scaled code sum";
  if (name === "scales") return "packed local scale metadata";
  if (name === "scales_h") return "high bits of local scales";
  if (name === "scales_l") return "low bits of local scales";
  if (name === "signs") return "packed sign masks";
  if (name === "bsums") return "precomputed sums of code groups";
  if (name === "e") return "shared exponent / scale code";
  return name;
}

function fieldScope(dtype: string, name: string): string {
  if (["qs", "qh", "ql", "hmask", "signs"].includes(name)) {
    return "individual weight codes, packed across the fixed block";
  }
  if (name === "bsums") return "one auxiliary sum per 16 codes";
  if (name === "scales" || name === "scales_h" || name === "scales_l") {
    return dtype.includes("_K") || dtype === "IQ4_XS" || dtype.startsWith("IQ")
      ? "local groups inside the super-block"
      : "the fixed block";
  }
  if (dtype === "NVFP4" && name === "d") return "one scale per 16 weights";
  return "the complete fixed block / super-block";
}

function q40ContractDetails(): {
  metadata: readonly string[];
  codes: readonly string[];
  packing: readonly string[];
  terms: readonly GgufQuantSymbol[];
} {
  return {
    metadata: [
      "signedMaxAbs is the block element with the largest absolute magnitude, preserving its sign.",
      "d = signedMaxAbs / −8, rounded once into the FP16 d field.",
      "The same d applies to all 32 weights in the block."
    ],
    codes: [
      "q is a conceptual signed integer from −8 through +7.",
      "stored nibble = q + 8, so the physical four-bit code is 0…15.",
      "A code is a discrete level ID, not a truncated floating-point weight."
    ],
    packing: [
      "qs contains 16 bytes for 32 codes.",
      "qs[j] bits 0…3 store weight j’s code.",
      "qs[j] bits 4…7 store weight j+16’s code."
    ],
    terms: [
      { symbol: "i", meaning: "weight index inside this block", source: "0…31 in source order" },
      { symbol: "j", meaning: "packed-byte index", source: "0…15; qs[j] stores weights j and j+16" },
      { symbol: "w[i]", meaning: "original source float at index i", source: "encoder input; not stored in the Q4_0 record" },
      { symbol: "signedMaxAbs", meaning: "signed block extreme", source: "selected from the 32 source weights by greatest absolute magnitude" },
      { symbol: "d", meaning: "shared FP16 scale", source: "physical record field d" },
      { symbol: "nibble", meaning: "stored four-bit code 0…15", source: "low or high half of one qs byte" },
      { symbol: "q", meaning: "signed integer level −8…7", source: "q = nibble − 8" },
      { symbol: "w′", meaning: "reconstructed approximation", source: "w′ = d × q" }
    ]
  };
}

function storageFieldTone(name: string): "metadata" | "codes" {
  return ["qs", "qh", "ql", "hmask", "signs"].includes(name)
    ? "codes"
    : "metadata";
}
