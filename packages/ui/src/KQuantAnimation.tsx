import { useState } from "react";
import { ChevronRight, Layers3, PackageOpen } from "lucide-react";
import {
  AnimationControls,
  AnimationStepCopy,
  type AnimationStep,
  useAnimationPlayer
} from "./DtypeAnimationPlayer";
import { ggufStorageLayout } from "./gguf-storage-layouts";

export interface KQuantLayout {
  dtype: "Q2_K" | "Q3_K" | "Q4_K" | "Q5_K" | "Q6_K";
  bytes: number;
  codeBits: number;
  subBlocks: number;
  valuesPerSubBlock: number;
  scaleBits: number;
  minBits?: number;
  sections: Array<{ label: string; bytes: number; tone: "global" | "local" | "codes" }>;
}

export const K_QUANT_LAYOUTS: Record<KQuantLayout["dtype"], KQuantLayout> = {
  Q2_K: {
    dtype: "Q2_K",
    bytes: 84,
    codeBits: 2,
    subBlocks: 16,
    valuesPerSubBlock: 16,
    scaleBits: 4,
    minBits: 4,
    sections: kQuantSections("Q2_K", {
      scales: "local",
      qs: "codes",
      d: "global",
      dmin: "global"
    })
  },
  Q3_K: {
    dtype: "Q3_K",
    bytes: 110,
    codeBits: 3,
    subBlocks: 16,
    valuesPerSubBlock: 16,
    scaleBits: 6,
    sections: kQuantSections("Q3_K", {
      hmask: "codes",
      qs: "codes",
      scales: "local",
      d: "global"
    })
  },
  Q4_K: {
    dtype: "Q4_K",
    bytes: 144,
    codeBits: 4,
    subBlocks: 8,
    valuesPerSubBlock: 32,
    scaleBits: 6,
    minBits: 6,
    sections: kQuantSections("Q4_K", {
      d: "global",
      dmin: "global",
      scales: "local",
      qs: "codes"
    })
  },
  Q5_K: {
    dtype: "Q5_K",
    bytes: 176,
    codeBits: 5,
    subBlocks: 8,
    valuesPerSubBlock: 32,
    scaleBits: 6,
    minBits: 6,
    sections: kQuantSections("Q5_K", {
      d: "global",
      dmin: "global",
      scales: "local",
      qh: "codes",
      qs: "codes"
    })
  },
  Q6_K: {
    dtype: "Q6_K",
    bytes: 210,
    codeBits: 6,
    subBlocks: 16,
    valuesPerSubBlock: 16,
    scaleBits: 8,
    sections: kQuantSections("Q6_K", {
      ql: "codes",
      qh: "codes",
      scales: "local",
      d: "global"
    })
  }
};

export function isKQuantDtype(dtype: string): dtype is KQuantLayout["dtype"] {
  return dtype in K_QUANT_LAYOUTS;
}

const STEPS: readonly AnimationStep[] = [
  {
    label: "One 256-weight super-block",
    detail: "The outer container amortizes global metadata across 256 neighboring weights."
  },
  {
    label: "Fit each sub-block locally",
    detail: "Small groups get independent ideal ranges, so an outlier in one group does not flatten all 256 weights."
  },
  {
    label: "Quantize the quantization parameters",
    detail: "The local scales and minima are themselves compressed under one or two FP16 super-block factors."
  },
  {
    label: "Pack the physical record",
    detail: "Metadata and code bit-planes occupy fixed byte ranges defined by the GGML ABI."
  },
  {
    label: "Reconstruct inside the kernel",
    detail: "The kernel combines global metadata, local metadata, and one code while accumulating a dot product."
  }
];

export function KQuantAnimation({ dtype }: { dtype: KQuantLayout["dtype"] }) {
  const layout = K_QUANT_LAYOUTS[dtype];
  const player = useAnimationPlayer(STEPS.length);
  const [selectedSubBlock, setSelectedSubBlock] = useState(0);
  const { step } = player;
  const affine = layout.minBits !== undefined;

  return (
    <section className="wv-kquant-demo">
      <header>
        <div>
          <span>HIERARCHICAL QUANTIZATION</span>
          <h3>
            <Layers3 aria-hidden="true" />
            Open up a {dtype} super-block
          </h3>
          <p>
            A fixed ABI layout: 256 weights → {layout.subBlocks} sub-blocks ×{" "}
            {layout.valuesPerSubBlock} weights → {layout.bytes} stored bytes.
            Click a sub-block to follow it.
          </p>
        </div>
        <div className="wv-kquant-ratio">
          <small>effective storage</small>
          <strong>{((layout.bytes * 8) / 256).toFixed(4).replace(/0+$/, "").replace(/\.$/, "")} bpw</strong>
        </div>
      </header>

      <div className="wv-kquant-player">
        <AnimationStepCopy
          step={step}
          steps={STEPS}
          announce={!player.playing}
        />

        <div className={`wv-kquant-stack step-${step}`}>
          <div className="wv-kquant-superblock">
            <header>
              <span>SUPER-BLOCK</span>
              <strong>256 source weights</strong>
              <small>global scope</small>
            </header>
            <div
              className="wv-kquant-groups"
              style={{ gridTemplateColumns: `repeat(${layout.subBlocks}, minmax(34px, 1fr))` }}
            >
              {Array.from({ length: layout.subBlocks }, (_, group) => (
                <button
                  type="button"
                  className={selectedSubBlock === group ? "selected" : ""}
                  aria-label={`Inspect sub-block ${group}, weights ${group * layout.valuesPerSubBlock} through ${(group + 1) * layout.valuesPerSubBlock - 1}`}
                  aria-pressed={selectedSubBlock === group}
                  key={group}
                  onClick={() => {
                    setSelectedSubBlock(group);
                    player.selectStep(1);
                  }}
                >
                  <small>sub {group}</small>
                  <div>
                    {Array.from({ length: layout.valuesPerSubBlock }, (_, value) => (
                      <i key={value} />
                    ))}
                  </div>
                  <span>{group * layout.valuesPerSubBlock}…{(group + 1) * layout.valuesPerSubBlock - 1}</span>
                </button>
              ))}
            </div>
          </div>

          <ChevronRight className="wv-kquant-down" aria-hidden="true" />

          <div className="wv-kquant-local">
            <header>
              <span>SELECTED SUB-BLOCK {selectedSubBlock}</span>
              <strong>{layout.valuesPerSubBlock} weights choose a local range</strong>
            </header>
            <div className="wv-kquant-range">
              {[-0.84, -0.46, -0.12, 0.08, 0.31, 0.72].map((value) => (
                <i
                  key={value}
                  style={{ height: `${18 + Math.abs(value) * 36}px` }}
                  className={value < 0 ? "negative" : "positive"}
                />
              ))}
            </div>
            <div className="wv-kquant-local-values">
              <span>
                <small>ideal local scale</small>
                <strong>a[{selectedSubBlock}]</strong>
              </span>
              {affine && (
                <span>
                  <small>ideal local minimum magnitude</small>
                  <strong>b[{selectedSubBlock}]</strong>
                </span>
              )}
            </div>
          </div>

          <ChevronRight className="wv-kquant-down" aria-hidden="true" />

          <div className="wv-kquant-hierarchy">
            <div className="wv-kquant-global-meta">
              <span>FP16 super-block metadata</span>
              <strong>{affine ? "d + dmin" : "d"}</strong>
              <small>{globalDerivation(dtype)}</small>
            </div>
            <ChevronRight aria-hidden="true" />
            <div className="wv-kquant-local-meta">
              <span>tiny metadata for sub {selectedSubBlock}</span>
              <strong>
                {layout.scaleBits}-bit s[{selectedSubBlock}]
                {affine ? ` + ${layout.minBits}-bit m[${selectedSubBlock}]` : ""}
              </strong>
              <small>
                {localDerivation(dtype)}
              </small>
            </div>
          </div>

          <ChevronRight className="wv-kquant-down" aria-hidden="true" />

          <div className="wv-kquant-layout">
            <header>
              <PackageOpen aria-hidden="true" />
              <span>
                <strong>Physical {dtype} record</strong>
                <small>exact field order · {layout.bytes} bytes total</small>
              </span>
            </header>
            <div>
              {layout.sections.map((section) => (
                <span
                  className={section.tone}
                  key={section.label}
                  style={{ flexGrow: section.bytes }}
                >
                  <strong>{section.label}</strong>
                  <small>{section.bytes} B</small>
                </span>
              ))}
            </div>
          </div>

          <ChevronRight className="wv-kquant-down" aria-hidden="true" />

          <div className="wv-kquant-decode">
            <div>
              <span>one packed code</span>
              <strong>{sampleCode(dtype)}</strong>
              <small>{layout.codeBits} bits from sub {selectedSubBlock}</small>
            </div>
            <ChevronRight aria-hidden="true" />
            <code>{decodeFormula(dtype, selectedSubBlock)}</code>
            <ChevronRight aria-hidden="true" />
            <div>
              <span>register only</span>
              <strong>w′ × activation → Σ</strong>
              <small>no expanded weight tensor</small>
            </div>
          </div>
        </div>

        <div className="wv-kquant-key">
          <span><i className="global" /> one value for all 256 weights</span>
          <span><i className="local" /> one compact value per sub-block</span>
          <span><i className="codes" /> one low-bit code per weight</span>
        </div>
      </div>

      <AnimationControls
        steps={STEPS}
        step={step}
        playing={player.playing}
        onPrevious={player.previous}
        onNext={player.next}
        onSelect={player.selectStep}
        onToggle={player.togglePlaying}
        onRestart={player.restart}
      />
    </section>
  );
}

function signedScaleLabel(dtype: KQuantLayout["dtype"]): string {
  if (dtype === "Q3_K") return "(s − 32)";
  if (dtype === "Q6_K") return "signed s";
  return "s";
}

function globalDerivation(dtype: KQuantLayout["dtype"]): string {
  if (dtype === "Q2_K") {
    return "d ≈ max(local scales) / 15; dmin ≈ max(local minima) / 15";
  }
  if (dtype === "Q4_K" || dtype === "Q5_K") {
    return "d ≈ max(local scales) / 63; dmin ≈ max(local minima) / 63";
  }
  return `d is chosen so every signed local scale fits ${K_QUANT_LAYOUTS[dtype].scaleBits} bits`;
}

function localDerivation(dtype: KQuantLayout["dtype"]): string {
  if (dtype === "Q3_K") {
    return "stored s ≈ round(localScale / d) + 32 · localScale = d × (s − 32)";
  }
  if (dtype === "Q6_K") {
    return "stored signed_s ≈ round(localScale / d) · localScale = d × signed_s";
  }
  return `s ≈ round(localScale / d) · localScale = d × ${signedScaleLabel(dtype)} · m ≈ round(localMin / dmin)`;
}

function sampleCode(dtype: KQuantLayout["dtype"]): string {
  if (dtype === "Q2_K") return "q = 2";
  if (dtype === "Q3_K") return "q = −3";
  if (dtype === "Q4_K") return "q = 9";
  if (dtype === "Q5_K") return "q = 18";
  return "q = −11";
}

function decodeFormula(dtype: KQuantLayout["dtype"], group: number): string {
  if (dtype === "Q2_K" || dtype === "Q4_K" || dtype === "Q5_K") {
    return `w′ = (d × s[${group}]) × q − (dmin × m[${group}])`;
  }
  if (dtype === "Q3_K") {
    return `w′ = d × (s[${group}] − 32) × q`;
  }
  return `w′ = d × signed_s[${group}] × q`;
}

function kQuantSections(
  dtype: KQuantLayout["dtype"],
  tones: Record<string, "global" | "local" | "codes">
): KQuantLayout["sections"] {
  const fields = ggufStorageLayout(dtype);
  if (!fields) {
    throw new Error(`Missing GGUF storage layout for ${dtype}`);
  }
  return fields.map((field) => {
    const tone = tones[field.name];
    if (!tone) {
      throw new Error(`Missing ${dtype}.${field.name} animation tone`);
    }
    return {
      label: `${field.name}: ${field.type}[${field.count}]`,
      bytes: field.bytes,
      tone
    };
  });
}
