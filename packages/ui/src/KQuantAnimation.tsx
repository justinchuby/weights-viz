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
  sections: Array<{
    name: string;
    type: string;
    count: number;
    bytes: number;
    role: string;
    tone: "global" | "local" | "codes";
  }>;
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

export interface KQuantCodeExample {
  source: string;
  quantize: string;
  code: string;
  storage: readonly string[];
  unpack: string;
  decode: string;
}

export interface KQuantTermDefinition {
  symbol: string;
  meaning: string;
  source: string;
}

export interface KQuantContractDetails {
  metadata: readonly string[];
  codes: readonly string[];
  packing: readonly string[];
  terms: readonly KQuantTermDefinition[];
  derivation: readonly KQuantDerivationStep[];
}

export interface KQuantDerivationStep {
  title: string;
  expression: string;
  detail: string;
}

export interface KQuantSubBlockStorage {
  metadata: readonly string[];
  codes: readonly string[];
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
  const codeExample = kQuantCodeExample(dtype, selectedSubBlock);

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
                  key={section.name}
                  style={{ flexGrow: section.bytes }}
                >
                  <strong>{section.name}: {section.type}[{section.count}]</strong>
                  <small>{section.bytes} B</small>
                </span>
              ))}
            </div>
          </div>

          <ChevronRight className="wv-kquant-down" aria-hidden="true" />

          <div className="wv-kquant-decode">
            <header>
              <strong>
                One illustrative weight in sub {selectedSubBlock}, lane 0, end to end
              </strong>
              <small>
                illustrative numbers · exact {dtype} quantize, bit-packing, and decode rules
              </small>
            </header>
            <div className="wv-kquant-code-journey">
              <CodeStage label="1 · source float" values={[codeExample.source]} />
              <ChevronRight aria-hidden="true" />
              <CodeStage
                label={`2 · choose a ${layout.codeBits}-bit code`}
                values={[codeExample.quantize, codeExample.code]}
              />
              <ChevronRight aria-hidden="true" />
              <CodeStage label="3 · write record bits" values={codeExample.storage} />
              <ChevronRight aria-hidden="true" />
              <CodeStage
                label="4 · read and reconstruct"
                values={[codeExample.unpack, codeExample.decode]}
              />
            </div>
            <p>
              Using the terms defined in the Storage contract above, the kernel
              rebuilds <code>w′</code> and immediately computes{" "}
              <code>w′ × activation → Σ</code>.
            </p>
          </div>
        </div>

        <div className="wv-kquant-key">
          <span><i className="global" /> one value for all 256 weights</span>
          <span><i className="local" /> one compact value per sub-block</span>
          <span><i className="codes" /> packed q codes (defined in Storage contract)</span>
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
  const scalesIndex = sections.findIndex(({ name }) => name === "scales");
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
            const scales = section.name === "scales";
            return (
              <span
                className={scales ? "selected" : ""}
                key={section.name}
                style={{ flexGrow: section.bytes }}
              >
                <b>{section.name}</b>
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

function CodeStage({
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

export function kQuantSubBlockStorage(
  dtype: KQuantLayout["dtype"],
  group: number
): KQuantSubBlockStorage {
  const layout = K_QUANT_LAYOUTS[dtype];
  if (!Number.isInteger(group) || group < 0 || group >= layout.subBlocks) {
    throw new RangeError(`Invalid ${dtype} sub-block ${group}`);
  }

  const scalesOffset = kQuantFieldOffset(dtype, "scales");
  const metadata = kQuantMetadataBytes(dtype, group).flatMap(({ index, segments }) =>
    segments.map(
      ({ label, from, to }) =>
        `${label}: scales[${index}] bits ${from}…${to} (record byte ${scalesOffset + index})`
    )
  );

  if (dtype === "Q2_K") {
    const base = Math.floor(group / 8) * 32 + (group % 2) * 16;
    const bit = 2 * Math.floor((group % 8) / 2);
    const offset = kQuantFieldOffset(dtype, "qs");
    return {
      metadata,
      codes: [
        `q codes: qs[${base}…${base + 15}] bits ${bit}…${bit + 1} (record bytes ${offset + base}…${offset + base + 15})`
      ]
    };
  }

  if (dtype === "Q3_K") {
    const qsBase = Math.floor(group / 8) * 32 + (group % 2) * 16;
    const qsBit = 2 * Math.floor((group % 8) / 2);
    const hmaskBase = (group % 2) * 16;
    const hmaskBit = Math.floor(group / 2);
    const qsOffset = kQuantFieldOffset(dtype, "qs");
    const hmaskOffset = kQuantFieldOffset(dtype, "hmask");
    return {
      metadata,
      codes: [
        `q low 2: qs[${qsBase}…${qsBase + 15}] bits ${qsBit}…${qsBit + 1} (record bytes ${qsOffset + qsBase}…${qsOffset + qsBase + 15})`,
        `q high mask: hmask[${hmaskBase}…${hmaskBase + 15}] bit ${hmaskBit} (record bytes ${hmaskOffset + hmaskBase}…${hmaskOffset + hmaskBase + 15})`
      ]
    };
  }

  if (dtype === "Q4_K" || dtype === "Q5_K") {
    const qsBase = 32 * Math.floor(group / 2);
    const nibble = group % 2 === 0 ? "low bits 0…3" : "high bits 4…7";
    const qsOffset = kQuantFieldOffset(dtype, "qs");
    const codes = [
      `q low 4: qs[${qsBase}…${qsBase + 31}] ${nibble} (record bytes ${qsOffset + qsBase}…${qsOffset + qsBase + 31})`
    ];
    if (dtype === "Q5_K") {
      const qhOffset = kQuantFieldOffset(dtype, "qh");
      codes.push(
        `q bit 4: qh[0…31] bit ${group} (record bytes ${qhOffset}…${qhOffset + 31})`
      );
    }
    return { metadata, codes };
  }

  const half = Math.floor(group / 8);
  const withinHalf = group % 8;
  const pair = Math.floor(withinHalf / 2);
  const laneBase = (withinHalf % 2) * 16;
  const qlBase = half * 64 + (pair % 2) * 32 + laneBase;
  const qhBase = half * 32 + laneBase;
  const nibble = pair >= 2 ? "high bits 4…7" : "low bits 0…3";
  const qhBit = 2 * pair;
  const qlOffset = kQuantFieldOffset(dtype, "ql");
  const qhOffset = kQuantFieldOffset(dtype, "qh");
  return {
    metadata,
    codes: [
      `q low 4: ql[${qlBase}…${qlBase + 15}] ${nibble} (record bytes ${qlOffset + qlBase}…${qlOffset + qlBase + 15})`,
      `q high 2: qh[${qhBase}…${qhBase + 15}] bits ${qhBit}…${qhBit + 1} (record bytes ${qhOffset + qhBase}…${qhOffset + qhBase + 15})`
    ]
  };
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

export function kQuantCodeExample(
  dtype: KQuantLayout["dtype"],
  group: number
): KQuantCodeExample {
  const layout = K_QUANT_LAYOUTS[dtype];
  if (!Number.isInteger(group) || group < 0 || group >= layout.subBlocks) {
    throw new RangeError(`Invalid ${dtype} sub-block ${group}`);
  }

  if (dtype === "Q2_K") {
    const qsOffset = kQuantFieldOffset(dtype, "qs");
    const pair = Math.floor((group % 8) / 2);
    const index = Math.floor(group / 8) * 32 + (group % 2) * 16;
    const bit = 2 * pair;
    return {
      source: `w[${group * 16}] = 0.68`,
      quantize: "local scale = 0.04×10 = 0.40; local min = 0.02×6 = 0.12",
      code: "q = clamp(round((0.68 + 0.12) / 0.40), 0, 3) = 2",
      storage: [
        "q = 2 = 10₂",
        `qs[${index}] bits ${bit}…${bit + 1} ← 10₂`,
        `qs starts at record byte ${qsOffset}, so this slice is in byte ${qsOffset + index}`
      ],
      unpack: `q = (qs[${index}] >> ${bit}) & 0b11 = 2`,
      decode: "w′ = (0.04×10)×2 − (0.02×6) = 0.68"
    };
  }

  if (dtype === "Q3_K") {
    const hmaskOffset = kQuantFieldOffset(dtype, "hmask");
    const qsOffset = kQuantFieldOffset(dtype, "qs");
    const pair = Math.floor((group % 8) / 2);
    const qsIndex = Math.floor(group / 8) * 32 + (group % 2) * 16;
    const qsBit = 2 * pair;
    const hmaskIndex = (group % 2) * 16;
    const hmaskBit = Math.floor(group / 2);
    return {
      source: `w[${group * 16}] = −0.30`,
      quantize: "local scale = 0.01×(42−32) = 0.10",
      code: "q = clamp(round(−0.30 / 0.10), −4, 3) = −3; stored = q+4 = 1",
      storage: [
        `qs[${qsIndex}] bits ${qsBit}…${qsBit + 1} ← 01₂`,
        `hmask[${hmaskIndex}] bit ${hmaskBit} ← 0`,
        `record bytes: hmask[${hmaskIndex}] = ${hmaskOffset + hmaskIndex}, qs[${qsIndex}] = ${qsOffset + qsIndex}`
      ],
      unpack: `q = low2 − (high bit set ? 0 : 4); here high bit ${hmaskBit} = 0, so q = 1−4 = −3`,
      decode: "w′ = 0.01×(42−32)×(−3) = −0.30"
    };
  }

  if (dtype === "Q4_K") {
    const qsOffset = kQuantFieldOffset(dtype, "qs");
    const index = 32 * Math.floor(group / 2);
    const high = group % 2 === 1;
    return {
      source: `w[${group * 32}] = 3.61`,
      quantize: "local scale = 0.01×42 = 0.42; local min = 0.005×34 = 0.17",
      code: "q = clamp(round((3.61 + 0.17) / 0.42), 0, 15) = 9",
      storage: [
        "q = 9 = 1001₂ (one of 16 four-bit codes)",
        `qs[${index}] ${high ? "high bits 4…7" : "low bits 0…3"} ← 1001₂`,
        `qs starts at record byte ${qsOffset}, so this nibble is in byte ${qsOffset + index}`
      ],
      unpack: `q = ${high ? `(qs[${index}] >> 4)` : `qs[${index}] & 0x0f`} = 9`,
      decode: "w′ = (0.01×42)×9 − (0.005×34) = 3.61"
    };
  }

  if (dtype === "Q5_K") {
    const qhOffset = kQuantFieldOffset(dtype, "qh");
    const qsOffset = kQuantFieldOffset(dtype, "qs");
    const qsIndex = 32 * Math.floor(group / 2);
    const highNibble = group % 2 === 1;
    return {
      source: `w[${group * 32}] = 7.39`,
      quantize: "local scale = 0.01×42 = 0.42; local min = 0.005×34 = 0.17",
      code: "q = clamp(round((7.39 + 0.17) / 0.42), 0, 31) = 18",
      storage: [
        `low 4 bits 0010₂ → qs[${qsIndex}] ${highNibble ? "bits 4…7" : "bits 0…3"}`,
        `fifth bit 1 → qh[0] bit ${group}`,
        `record bytes: qh[0] = ${qhOffset}, qs[${qsIndex}] = ${qsOffset + qsIndex}`
      ],
      unpack: `q = ${highNibble ? `((qs[${qsIndex}] >> 4) & 15)` : `(qs[${qsIndex}] & 15)`} + ((qh[0] bit ${group}) << 4) = 2+16 = 18`,
      decode: "w′ = (0.01×42)×18 − (0.005×34) = 7.39"
    };
  }

  const qlOffset = kQuantFieldOffset(dtype, "ql");
  const qhOffset = kQuantFieldOffset(dtype, "qh");
  const half = Math.floor(group / 8);
  const withinHalf = group % 8;
  const pair = Math.floor(withinHalf / 2);
  const lane = (withinHalf % 2) * 16;
  const qlIndex = half * 64 + (pair % 2) * 32 + lane;
  const qhIndex = half * 32 + lane;
  const highNibble = pair >= 2;
  const qhBit = 2 * pair;
  return {
    source: `w[${group * 16}] = −4.62`,
    quantize: "local scale = 0.01×42 = 0.42",
    code: "q = clamp(round(−4.62 / 0.42), −32, 31) = −11; stored = q+32 = 21",
    storage: [
      `stored 21 = 010101₂; low 4 bits 0101₂ → ql[${qlIndex}] ${highNibble ? "bits 4…7" : "bits 0…3"}`,
      `high 2 bits 01₂ → qh[${qhIndex}] bits ${qhBit}…${qhBit + 1}`,
      `record bytes: ql[${qlIndex}] = ${qlOffset + qlIndex}, qh[${qhIndex}] = ${qhOffset + qhIndex}`
    ],
    unpack: `q = join(qh bits ${qhBit}…${qhBit + 1}, ql ${highNibble ? "high" : "low"} nibble) − 32 = 21−32 = −11`,
    decode: "w′ = 0.01×42×(−11) = −4.62"
  };
}

function kQuantFieldOffset(
  dtype: KQuantLayout["dtype"],
  fieldName: string
): number {
  const sections = K_QUANT_LAYOUTS[dtype].sections;
  const index = sections.findIndex(({ name }) => name === fieldName);
  if (index < 0) {
    throw new Error(`Missing ${dtype}.${fieldName} field`);
  }
  return sections
    .slice(0, index)
    .reduce((offset, section) => offset + section.bytes, 0);
}

function kQuantParameterDerivation(
  dtype: KQuantLayout["dtype"]
): KQuantDerivationStep[] {
  const layout = K_QUANT_LAYOUTS[dtype];
  const firstMetadata = kQuantSubBlockStorage(dtype, 0).metadata.join("; ");

  if (layout.minBits !== undefined) {
    const max = 2 ** layout.scaleBits - 1;
    const example = dtype === "Q4_K"
      ? "Example: a[0]=0.42 and d=0.01 give s[0]=42; b[0]=0.17 and dmin=0.005 give m[0]=34."
      : "The rounding error is why decoded localScale/localMin can differ slightly from the encoder-only a/b values.";
    return [
      {
        title: "1. Fit this sub-block",
        expression: "w[g,l] ≈ a[g] × q[g,l] − b[g]",
        detail: "The encoder searches a local real-valued scale a[g], positive offset magnitude b[g], and allowed integer q codes that minimize reconstruction error. a[g] and b[g] are temporary and are not file fields."
      },
      {
        title: "2. Choose shared FP16 steps",
        expression: "a[g] ≈ d × s[g] · b[g] ≈ dmin × m[g]",
        detail: "Across all sub-blocks, the encoder chooses global steps d and dmin. Only d and dmin are stored directly as FP16 fields."
      },
      {
        title: "3. Make stored integers",
        expression: `s[g] = clamp(round(a[g] / d), 0, ${max}) · m[g] = clamp(round(b[g] / dmin), 0, ${max})`,
        detail: `${example} s[g] and m[g] are integers; neither is another floating-point scale.`
      },
      {
        title: "4. Pack s[g] and m[g]",
        expression: "s[g], m[g] → scales[] bits",
        detail: `For g=0: ${firstMetadata}. The exact locations for every g appear in the sub-block map below.`
      },
      {
        title: "5. Recreate local parameters",
        expression: "localScale[g] = d × s[g] · localMin[g] = dmin × m[g]",
        detail: "The decoder unpacks s[g] and m[g], rebuilds the two local real values, then combines them with each unpacked q[g,l]."
      }
    ];
  }

  if (dtype === "Q3_K") {
    return [
      {
        title: "1. Fit this sub-block",
        expression: "w[g,l] ≈ a[g] × q[g,l]",
        detail: "The encoder searches a local real-valued scale a[g] and signed q codes that minimize reconstruction error. a[g] is temporary and is not stored."
      },
      {
        title: "2. Choose one FP16 step",
        expression: "a[g] ≈ d × (s[g] − 32)",
        detail: "One global FP16 d is shared by all 16 sub-blocks."
      },
      {
        title: "3. Make the biased integer",
        expression: "s[g] = clamp(round(a[g] / d) + 32, 0, 63)",
        detail: "s[g] is the stored unsigned six-bit integer. The fixed 32 bias converts it back to a signed local-scale integer."
      },
      {
        title: "4. Pack s[g]",
        expression: "six bits of s[g] → scales[]",
        detail: `For g=0: ${firstMetadata}.`
      },
      {
        title: "5. Recreate the local scale",
        expression: "localScale[g] = d × (s[g] − 32)",
        detail: "The decoder subtracts the format bias after unpacking s[g]; there is no stored b[g] or m[g] in Q3_K."
      }
    ];
  }

  return [
    {
      title: "1. Fit this sub-block",
      expression: "w[g,l] ≈ a[g] × q[g,l]",
      detail: "The encoder searches a local real-valued scale a[g] and signed q codes that minimize reconstruction error. a[g] is temporary and is not stored."
    },
    {
      title: "2. Choose one FP16 step",
      expression: "a[g] ≈ d × signed_s[g]",
      detail: "One global FP16 d is shared by all 16 sub-blocks."
    },
    {
      title: "3. Make the stored integer",
      expression: "signed_s[g] = clamp(round(a[g] / d), −128, 127)",
      detail: "Q6_K stores the sign directly in an int8; it does not use Q3_K's +32 bias."
    },
    {
      title: "4. Store signed_s[g]",
      expression: "signed_s[g] → scales[g]",
      detail: `For g=0: ${firstMetadata}.`
    },
    {
      title: "5. Recreate the local scale",
      expression: "localScale[g] = d × signed_s[g]",
      detail: "The decoder multiplies the shared FP16 step by the stored int8 before applying each q[g,l]."
    }
  ];
}

export function kQuantContractDetails(
  dtype: KQuantLayout["dtype"]
): KQuantContractDetails {
  const layout = K_QUANT_LAYOUTS[dtype];
  const affine = layout.minBits !== undefined;
  const qRange =
    dtype === "Q2_K"
      ? "0…3"
      : dtype === "Q3_K"
        ? "−4…3"
        : dtype === "Q4_K"
          ? "0…15"
          : dtype === "Q5_K"
            ? "0…31"
            : "−32…31";
  const metadata = affine
    ? [
        `d is the FP16 global step used to reconstruct every local scale in the 256-weight super-block.`,
        `dmin is the FP16 global step used to reconstruct every local minimum magnitude.`,
        `s[g] and m[g] are the ${layout.scaleBits}-bit local integers for sub-block g; localScale[g] = d × s[g], localMin[g] = dmin × m[g].`
      ]
    : dtype === "Q3_K"
      ? [
          "d is the FP16 global step shared by all 16 sub-blocks.",
          "s[g] is a packed unsigned six-bit value 0…63; subtracting the fixed bias 32 gives the signed local-scale integer.",
          "localScale[g] = d × (s[g] − 32); neither the decoded local scale nor the bias 32 occupies another field."
        ]
      : [
          "d is the FP16 global step shared by all 16 sub-blocks.",
          "signed_s[g] is one int8 value in scales[g], so its sign is physically stored rather than implied by a bias.",
          "localScale[g] = d × signed_s[g]."
        ];

  return {
    metadata,
    derivation: kQuantParameterDerivation(dtype),
    codes: [
      `q is one logical integer code for one weight. qs is a physical byte array that packs many q codes together; extract q's bits from qs (and any companion high-bit field) before decoding.`,
      `${layout.codeBits} bits per q allow the ${qRange} decode range used by ${dtype}.`,
      "The original floating-point weight is not stored. Its q code plus the shared/local metadata reconstruct an approximation w′.",
      codeStorageRule(dtype)
    ],
    packing: kQuantPackingRules(dtype),
    terms: [
      term("super-block", "the complete fixed record", `256 source weights encoded in ${layout.bytes} bytes`),
      term("i", "weight index inside the super-block", "0…255 in source order"),
      term("g / sub-block", "local group index", `${layout.subBlocks} groups numbered 0…${layout.subBlocks - 1}, each covering ${layout.valuesPerSubBlock} consecutive weights`),
      term("lane / l", "position inside sub-block g", `0…${layout.valuesPerSubBlock - 1}; source index i = g×${layout.valuesPerSubBlock} + l`),
      term("w[i]", "original source floating-point weight", "encoder input; it is not present in the stored record"),
      term("w′[i]", "reconstructed approximation of w[i]", "decoder output produced from metadata and q"),
      term("a[g]", "encoder-only local scale fitted to sub-block g", "chosen with q[g,l] to minimize reconstruction error; quantized into s[g], never written to the record"),
      ...(affine
        ? [
            term("b[g]", "encoder-only positive offset magnitude for sub-block g", "chosen with a[g] and q[g,l] to minimize reconstruction error; quantized into m[g], never written to the record"),
            term("dmin / globalMin", "global minimum step", "FP16 record field dmin"),
            term("m[g] / subMin", "stored local-minimum integer", `m[g]=clamp(round(b[g]/dmin)); its ${layout.minBits} bits are packed in scales[]; localMin[g]=dmin×m[g]`)
          ]
        : []),
      term("d / globalScale", "global scale step", "FP16 record field d"),
      term(
        dtype === "Q6_K" ? "signed_s[g] / subScale" : "s[g] / subScale",
        "stored local-scale integer",
        localScaleTermSource(dtype)
      ),
      term("q / code", `integer level ${qRange} for one weight`, codeTermSource(dtype)),
      term("activation", "the matching runtime input value", "the other operand of the fused dot product; not stored in this weight record"),
      term("Σ", "dot-product accumulator", "register sum of reconstructed weight × activation products")
    ]
  };
}

export function kQuantFieldMeaning(name: string): string {
  if (name === "qs") return "byte array packing many per-weight q codes";
  if (name === "ql") return "packed low bits of each q code";
  if (name === "qh") return "packed high bits of each q code";
  if (name === "hmask") return "high-bit mask completing each signed q code";
  if (name === "scales") return "packed local scale/minimum integers";
  if (name === "d") return "global scale step";
  if (name === "dmin") return "global minimum-magnitude step";
  return name;
}

function codeStorageRule(dtype: KQuantLayout["dtype"]): string {
  if (dtype === "Q2_K") {
    return "Each q uses one two-bit slice inside qs; four slices for separated weight groups share each qs byte.";
  }
  if (dtype === "Q3_K") {
    return "Each signed q is rebuilt from two low bits in qs and one high/subtract-control bit in hmask.";
  }
  if (dtype === "Q4_K") {
    return "Each q is one nibble in qs; two 32-weight sub-blocks use the low and high nibbles of the same 32-byte stripe.";
  }
  if (dtype === "Q5_K") {
    return "Each q uses four low bits in qs plus its fifth bit in qh.";
  }
  return "Each signed q uses four low bits in ql plus two high bits in qh, then subtracts the fixed storage bias 32.";
}

function kQuantPackingRules(
  dtype: KQuantLayout["dtype"]
): readonly string[] {
  if (dtype === "Q2_K") {
    return [
      "scales[g] bits 0…3 = s[g]; bits 4…7 = m[g].",
      "For sub-block g and lane l: qs[floor(g/8)×32 + (g mod 2)×16 + l].",
      "The q slice starts at bit 2×floor((g mod 8)/2); mask with 0b11."
    ];
  }
  if (dtype === "Q3_K") {
    return [
      "scales[12] packs sixteen six-bit s[g] values; decoding subtracts the implicit bias 32.",
      "qs uses the same interleaved two-bit location rule as Q2_K.",
      "For source index i=16g+l: hmask[i mod 32] bit floor(i/32) completes q; clear means subtract 4, set means subtract 0."
    ];
  }
  if (dtype === "Q4_K") {
    return [
      "scales[12] packs eight six-bit s[g] and eight six-bit m[g] values.",
      "For sub-block g and lane l: byte = qs[32×floor(g/2)+l].",
      "Even g uses that byte’s low nibble; odd g uses its high nibble."
    ];
  }
  if (dtype === "Q5_K") {
    return [
      "scales[12] uses exactly the same s[g]/m[g] layout as Q4_K.",
      "For sub-block g and lane l: the low four bits are the matching nibble in qs[32×floor(g/2)+l].",
      "The fifth bit is qh[l] bit g; joining it above the qs nibble yields q 0…31."
    ];
  }
  return [
    "scales[g] is the signed int8 local-scale integer for consecutive 16-weight sub-block g.",
    "Within each 128-weight half, ql low nibbles encode weights 0…63 and high nibbles encode 64…127.",
    "Within that half, qh bits 0…1, 2…3, 4…5, and 6…7 are the two high-bit planes for the four consecutive 32-weight quarters; join with ql, then subtract 32."
  ];
}

function localScaleTermSource(dtype: KQuantLayout["dtype"]): string {
  if (dtype === "Q3_K") {
    return "s[g]=clamp(round(a[g]/d)+32); its six bits are packed in scales[]; localScale=d×(s[g]−32)";
  }
  if (dtype === "Q6_K") {
    return "signed_s[g]=clamp(round(a[g]/d)); record field scales[g] stores that int8; localScale=d×signed_s[g]";
  }
  return `s[g]=clamp(round(a[g]/d)); its ${K_QUANT_LAYOUTS[dtype].scaleBits} bits are packed in scales[]; localScale=d×s[g]`;
}

function codeTermSource(dtype: KQuantLayout["dtype"]): string {
  if (dtype === "Q2_K" || dtype === "Q4_K") return "packed directly in qs";
  if (dtype === "Q3_K") return "two low bits from qs plus the matching hmask bit";
  if (dtype === "Q5_K") return "four low bits from qs plus the matching qh bit";
  return "four low bits from ql plus two high bits from qh, followed by −32";
}

function term(
  symbol: string,
  meaning: string,
  source: string
): KQuantTermDefinition {
  return { symbol, meaning, source };
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
      name: field.name,
      type: field.type,
      count: field.count,
      bytes: field.bytes,
      role: field.role,
      tone
    };
  });
}
