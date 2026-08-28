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

export interface KQuantMetadataByte {
  index: number;
  segments: Array<{
    label: string;
    from: number;
    to: number;
    tone: "scale" | "minimum";
  }>;
}

interface KQuantMetadataExample {
  ideal: readonly string[];
  global: readonly string[];
  stored: readonly string[];
  scaleCode: number;
  minimumCode?: number;
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
    label: "Turn local ranges into stored integers",
    detail: "Divide each ideal local parameter by its global FP16 step, round it, and place the resulting small integer inside scales[]."
  },
  {
    label: "Place every field in the record",
    detail: "The highlighted scales[] field is one byte range inside the full GGML record, next to global metadata and packed weight codes."
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

          <div className="wv-kquant-metadata-stage">
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
            <KQuantMetadataMap dtype={dtype} group={selectedSubBlock} />
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

function KQuantMetadataMap({
  dtype,
  group
}: {
  dtype: KQuantLayout["dtype"];
  group: number;
}) {
  const bytes = kQuantMetadataBytes(dtype, group);
  const sections = K_QUANT_LAYOUTS[dtype].sections;
  const scalesIndex = sections.findIndex(({ label }) =>
    label.startsWith("scales:")
  );
  if (scalesIndex < 0) {
    throw new Error(`Missing ${dtype} scales field`);
  }
  const fieldOffset = sections
    .slice(0, scalesIndex)
    .reduce((total, section) => total + section.bytes, 0);
  const scalesField = ggufStorageLayout(dtype)?.find(
    ({ name }) => name === "scales"
  );
  if (!scalesField) {
    throw new Error(`Missing ${dtype} scales field contract`);
  }
  const example = kQuantMetadataExample(dtype, group);
  const selectedByteIndexes = new Set(bytes.map(({ index }) => index));

  return (
    <div className="wv-kquant-bit-map">
      <header>
        <span>
          <strong>Where sub {group} metadata is stored</strong>
          <small>
            <code>scales</code> field starts at record byte {fieldOffset}
          </small>
        </span>
        <em>{metadataEncoding(dtype)}</em>
      </header>
      <div className="wv-kquant-record-locator">
        <header>
          <strong>Start with the complete {dtype} record</strong>
          <small>{K_QUANT_LAYOUTS[dtype].bytes} bytes total</small>
        </header>
        <div>
          {sections.map((section) => {
            const scales = section.label.startsWith("scales:");
            return (
              <span
                className={scales ? "selected" : ""}
                key={section.label}
                style={{ flexGrow: section.bytes }}
              >
                <b>{section.label.split(":")[0]}</b>
                <small>{section.bytes} B</small>
                {scales && <em>sub-block metadata lives here</em>}
              </span>
            );
          })}
        </div>
        <p>
          <span>record bytes {fieldOffset}…{fieldOffset + scalesField.bytes - 1}</span>
          <b>zoom into <code>scales[{scalesField.count}]</code></b>
          <span aria-hidden="true">↓</span>
        </p>
      </div>
      <div className="wv-kquant-worked-example">
        <header>
          <strong>Concrete example</strong>
          <small>illustrative values, exact encoding rule</small>
        </header>
        <ExampleStage label="ideal local parameters" values={example.ideal} />
        <ChevronRight aria-hidden="true" />
        <ExampleStage label="global FP step" values={example.global} />
        <ChevronRight aria-hidden="true" />
        <ExampleStage label="stored integers" values={example.stored} />
      </div>
      <div className="wv-kquant-scale-locator">
        <header>
          <strong>
            <code>scales[{scalesField.count}]</code> inside the physical record
          </strong>
          <small>
            array index + {fieldOffset} = record byte
          </small>
        </header>
        <div>
          {Array.from({ length: scalesField.count }, (_, index) => (
            <span
              className={selectedByteIndexes.has(index) ? "selected" : ""}
              key={index}
            >
              <b>{index}</b>
              <small>byte {fieldOffset + index}</small>
            </span>
          ))}
        </div>
      </div>
      <div className="wv-kquant-bit-details">
        {bytes.map((byte) => (
          <div className="wv-kquant-meta-byte" key={byte.index}>
            <header>
              <code>scales[{byte.index}]</code>
              <small>record byte {fieldOffset + byte.index}</small>
            </header>
            <div className="wv-kquant-meta-bits">
              {Array.from({ length: 8 }, (_, index) => 7 - index).map((bit) => {
                const segment = byte.segments.find(
                  ({ from, to }) => bit >= from && bit <= to
                );
                const storedBit = segment
                  ? metadataBit(example, segment, bit)
                  : undefined;
                return (
                  <span
                    aria-label={
                      segment
                        ? `bit ${bit}, ${segment.label}`
                        : `bit ${bit}, metadata for another sub-block`
                    }
                    className={segment?.tone ?? "other"}
                    key={bit}
                    title={
                      segment?.label ?? "Used by another sub-block in this packed byte"
                    }
                  >
                    <small>{bit}</small>
                    <b>{storedBit ?? "·"}</b>
                  </span>
                );
              })}
            </div>
            <ul>
              {byte.segments.map((segment) => (
                <li className={segment.tone} key={`${segment.label}:${segment.from}`}>
                  bits {segment.from === segment.to ? segment.from : `${segment.from}…${segment.to}`} → {segment.label}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <small className="wv-kquant-bit-note">
        Blue bits assemble <b>s[{group}] = {example.scaleCode}</b>
        {example.minimumCode !== undefined && (
          <> · gold bits assemble <b>m[{group}] = {example.minimumCode}</b></>
        )}
        {" "}· <b>·</b> belongs to another sub-block
      </small>
    </div>
  );
}

function ExampleStage({
  label,
  values
}: {
  label: string;
  values: readonly string[];
}) {
  return (
    <span>
      <small>{label}</small>
      {values.map((value) => (
        <code key={value}>{value}</code>
      ))}
    </span>
  );
}

export function kQuantMetadataBytes(
  dtype: KQuantLayout["dtype"],
  group: number
): KQuantMetadataByte[] {
  const layout = K_QUANT_LAYOUTS[dtype];
  if (!Number.isInteger(group) || group < 0 || group >= layout.subBlocks) {
    throw new RangeError(`Invalid ${dtype} sub-block ${group}`);
  }
  if (dtype === "Q2_K") {
    return [
      {
        index: group,
        segments: [
          { label: `s[${group}]`, from: 0, to: 3, tone: "scale" },
          { label: `m[${group}]`, from: 4, to: 7, tone: "minimum" }
        ]
      }
    ];
  }
  if (dtype === "Q3_K") {
    const quarter = Math.floor(group / 4);
    const lane = group % 4;
    const lowIndex = group < 8 ? group : group - 8;
    return [
      {
        index: lowIndex,
        segments: [
          {
            label: `s[${group}] low 4`,
            from: group < 8 ? 0 : 4,
            to: group < 8 ? 3 : 7,
            tone: "scale"
          }
        ]
      },
      {
        index: 8 + lane,
        segments: [
          {
            label: `s[${group}] high 2`,
            from: quarter * 2,
            to: quarter * 2 + 1,
            tone: "scale"
          }
        ]
      }
    ];
  }
  if (dtype === "Q4_K" || dtype === "Q5_K") {
    if (group < 4) {
      return [
        {
          index: group,
          segments: [
            { label: `s[${group}]`, from: 0, to: 5, tone: "scale" }
          ]
        },
        {
          index: group + 4,
          segments: [
            { label: `m[${group}]`, from: 0, to: 5, tone: "minimum" }
          ]
        }
      ];
    }
    return [
      {
        index: group + 4,
        segments: [
          { label: `s[${group}] low 4`, from: 0, to: 3, tone: "scale" },
          { label: `m[${group}] low 4`, from: 4, to: 7, tone: "minimum" }
        ]
      },
      {
        index: group - 4,
        segments: [
          { label: `s[${group}] high 2`, from: 6, to: 7, tone: "scale" }
        ]
      },
      {
        index: group,
        segments: [
          { label: `m[${group}] high 2`, from: 6, to: 7, tone: "minimum" }
        ]
      }
    ];
  }
  return [
    {
      index: group,
      segments: [
        { label: `signed s[${group}]`, from: 0, to: 7, tone: "scale" }
      ]
    }
  ];
}

function metadataEncoding(dtype: KQuantLayout["dtype"]): string {
  if (dtype === "Q2_K") return "one byte: mmmm ssss";
  if (dtype === "Q3_K") return "6 bits assembled from two bytes";
  if (dtype === "Q4_K" || dtype === "Q5_K") {
    return "6-bit s and m packed across scales[12]";
  }
  return "one signed int8 byte";
}

function kQuantMetadataExample(
  dtype: KQuantLayout["dtype"],
  group: number
): KQuantMetadataExample {
  if (dtype === "Q2_K") {
    return {
      ideal: [`a[${group}] = 0.40`, `b[${group}] = 0.12`],
      global: ["d = 0.04", "dmin = 0.02"],
      stored: [`s[${group}] = round(0.40/0.04) = 10`, `m[${group}] = round(0.12/0.02) = 6`],
      scaleCode: 10,
      minimumCode: 6
    };
  }
  if (dtype === "Q4_K" || dtype === "Q5_K") {
    return {
      ideal: [`a[${group}] = 0.42`, `b[${group}] = 0.17`],
      global: ["d = 0.01", "dmin = 0.005"],
      stored: [`s[${group}] = round(0.42/0.01) = 42`, `m[${group}] = round(0.17/0.005) = 34`],
      scaleCode: 42,
      minimumCode: 34
    };
  }
  if (dtype === "Q3_K") {
    return {
      ideal: [`a[${group}] = +0.10`],
      global: ["d = 0.01", "bias = 32"],
      stored: [`s[${group}] = round(0.10/0.01) + 32 = 42`],
      scaleCode: 42
    };
  }
  return {
    ideal: [`a[${group}] = +0.42`],
    global: ["d = 0.01"],
    stored: [`signed_s[${group}] = round(0.42/0.01) = 42`],
    scaleCode: 42
  };
}

function metadataBit(
  example: KQuantMetadataExample,
  segment: KQuantMetadataByte["segments"][number],
  physicalBit: number
): number {
  const value = segment.tone === "minimum"
    ? example.minimumCode
    : example.scaleCode;
  if (value === undefined) {
    throw new Error(`Missing value for ${segment.label}`);
  }
  const logicalBit = segment.label.includes("high 2")
    ? physicalBit - segment.from + 4
    : physicalBit - segment.from;
  return (value >> logicalBit) & 1;
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
