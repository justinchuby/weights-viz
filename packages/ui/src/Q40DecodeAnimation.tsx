import type { ReactNode } from "react";
import {
  ChevronRight,
  Sparkles
} from "lucide-react";
import {
  AnimationControls,
  AnimationStepCopy,
  useAnimationPlayer
} from "./DtypeAnimationPlayer";

const SCALE = 0.25;

export const Q40_BLOCK_LAYOUT = {
  values: 32,
  bytes: 18,
  scaleCount: 1,
  scaleType: "FP16",
  scaleBytes: 2,
  packedBytes: 16,
  bias: 8,
  qMin: -8,
  qMax: 7
} as const;

const SOURCE_PAIRS = [
  { lowIndex: 0, highIndex: 16, low: -2, high: 0.31 },
  { lowIndex: 1, highIndex: 17, low: -1.18, high: 0.68 },
  { lowIndex: 2, highIndex: 18, low: -0.57, high: 1.29 },
  { lowIndex: 3, highIndex: 19, low: 0.08, high: 1.71 }
] as const;

export interface Q40DemoPair {
  lowIndex: number;
  highIndex: number;
  low: number;
  high: number;
  lowCode: number;
  highCode: number;
  lowNibble: number;
  highNibble: number;
  packedByte: number;
  decodedLow: number;
  decodedHigh: number;
}

export const Q40_DEMO_PAIRS: Q40DemoPair[] = SOURCE_PAIRS.map((pair) => {
  const lowCode = quantizeQ40(pair.low, SCALE);
  const highCode = quantizeQ40(pair.high, SCALE);
  const lowNibble = lowCode + 8;
  const highNibble = highCode + 8;
  return {
    ...pair,
    lowCode,
    highCode,
    lowNibble,
    highNibble,
    packedByte: lowNibble | (highNibble << 4),
    decodedLow: lowCode * SCALE,
    decodedHigh: highCode * SCALE
  };
});

const STEPS = [
  {
    label: "Choose the one block scale",
    detail: "The reference quantizer finds the signed max-absolute weight and stores d = signedMaxAbs / −8 as FP16."
  },
  {
    label: "Create signed q codes",
    detail: "Every one of the 32 weights divides by the same d, rounds, and clamps to q ∈ [−8, 7]."
  },
  {
    label: "Apply the implicit bias and pack",
    detail: "Add 8, then pair q[i] with q[i+16] in the low and high nibbles."
  },
  {
    label: "Decode",
    detail: "Read a nibble n, recover q = n − 8, then reconstruct w′ = d × q. The gap is quantization error."
  },
  {
    label: "Fused dot product",
    detail: "A kernel unpacks, rescales, multiplies, and accumulates without materializing an F32 tensor."
  }
] as const;

export function quantizeQ40(value: number, scale: number): number {
  return Math.max(-8, Math.min(7, Math.round(value / scale)));
}

export function Q40DecodeAnimation() {
  const player = useAnimationPlayer(STEPS.length);
  const { step } = player;

  return (
    <section className="wv-q4-demo" aria-labelledby="wv-q4-demo-title">
      <header>
        <div>
          <span>INTERACTIVE DECODE</span>
          <h3 id="wv-q4-demo-title">
            <Sparkles aria-hidden="true" />
            Watch a Q4_0 block become math
          </h3>
          <p>
            A faithful miniature of GGML packing. A real block stores one{" "}
            {Q40_BLOCK_LAYOUT.scaleType} scale and {Q40_BLOCK_LAYOUT.values}{" "}
            four-bit codes in {Q40_BLOCK_LAYOUT.bytes} bytes.
          </p>
        </div>
        <div className="wv-q4-scale">
          <small>shared FP16 scale</small>
          <strong>d = {SCALE}</strong>
        </div>
      </header>

      <div className="wv-q4-player">
        <AnimationStepCopy
          step={step}
          steps={STEPS}
          announce={!player.playing}
        />

        <div
          className={`wv-q4-scale-rule${step === 0 ? " active" : ""}${
            step > 0 ? " complete" : ""
          }`}
        >
          <code>d = signedMaxAbs(block) / −8</code>
          <span>
            Here the largest magnitude is <strong>−2.00</strong>, so the stored
            FP16 scale is <strong>d = 0.25</strong>. One d serves all 32 weights.
          </span>
        </div>

        <div className="wv-q4-flow" data-step={step}>
          <FlowStage title="F32 source" subtitle="8 of 32 lanes shown" active={step === 0}>
            {Q40_DEMO_PAIRS.map((pair) => (
              <PairCell
                key={pair.lowIndex}
                leftLabel={`w${pair.lowIndex}`}
                leftValue={formatNumber(pair.low)}
                rightLabel={`w${pair.highIndex}`}
                rightValue={formatNumber(pair.high)}
              />
            ))}
          </FlowStage>

          <FlowArrow active={step >= 1} label="÷ d · round" />

          <FlowStage title="Signed codes" subtitle="q ∈ [−8, 7] · not stored yet" revealed={step >= 1} active={step === 1}>
            {Q40_DEMO_PAIRS.map((pair) => (
              <PairCell
                key={pair.lowIndex}
                leftLabel={`q${pair.lowIndex}`}
                leftValue={formatSigned(pair.lowCode)}
                rightLabel={`q${pair.highIndex}`}
                rightValue={formatSigned(pair.highCode)}
              />
            ))}
          </FlowStage>

          <FlowArrow active={step >= 2} label="n = q + 8" />

          <FlowStage title="Packed bytes" subtitle="high n[i+16] · low n[i]" revealed={step >= 2} active={step === 2}>
            {Q40_DEMO_PAIRS.map((pair) => (
              <div className="wv-q4-byte" key={pair.lowIndex}>
                <span>{pair.highNibble.toString(2).padStart(4, "0")}</span>
                <span>{pair.lowNibble.toString(2).padStart(4, "0")}</span>
                <small>0x{pair.packedByte.toString(16).padStart(2, "0").toUpperCase()}</small>
              </div>
            ))}
          </FlowStage>

          <FlowArrow active={step >= 3} label="q = n−8 · × d" />

          <FlowStage title="Approximate F32" subtitle="w′ = d × q · error" revealed={step >= 3} active={step === 3}>
            {Q40_DEMO_PAIRS.map((pair) => (
              <PairCell
                key={pair.lowIndex}
                leftLabel={`w′${pair.lowIndex}`}
                leftValue={`${formatNumber(pair.decodedLow)} ${formatError(pair.low, pair.decodedLow)}`}
                rightLabel={`w′${pair.highIndex}`}
                rightValue={`${formatNumber(pair.decodedHigh)} ${formatError(pair.high, pair.decodedHigh)}`}
              />
            ))}
          </FlowStage>
        </div>

        <div className={`wv-q4-dot${step === 4 ? " active" : ""}`}>
          <div className="wv-q4-dot-bytes">
            {Q40_DEMO_PAIRS.map((pair) => (
              <span key={pair.lowIndex}>
                {pair.packedByte.toString(16).padStart(2, "0").toUpperCase()}
              </span>
            ))}
          </div>
          <ChevronRight aria-hidden="true" />
          <div className="wv-q4-kernel">
            <span>UNPACK</span>
            <span>× d</span>
            <span>× activation</span>
          </div>
          <ChevronRight aria-hidden="true" />
          <strong>Σ dot</strong>
          <small>Registers carry the values through the whole fused kernel.</small>
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

function FlowStage({
  title,
  subtitle,
  active = false,
  revealed = true,
  children
}: {
  title: string;
  subtitle: string;
  active?: boolean;
  revealed?: boolean;
  children: ReactNode;
}) {
  return (
    <div className={`wv-q4-stage${active ? " active" : ""}${revealed ? " revealed" : ""}`}>
      <header>
        <strong>{title}</strong>
        <small>{subtitle}</small>
      </header>
      <div>{children}</div>
    </div>
  );
}

function FlowArrow({ active, label }: { active: boolean; label: string }) {
  return (
    <div className={`wv-q4-arrow${active ? " active" : ""}`}>
      <span>{label}</span>
      <ChevronRight aria-hidden="true" />
    </div>
  );
}

function PairCell({
  leftLabel,
  leftValue,
  rightLabel,
  rightValue
}: {
  leftLabel: string;
  leftValue: string;
  rightLabel: string;
  rightValue: string;
}) {
  return (
    <div className="wv-q4-pair">
      <span>
        <small>{leftLabel}</small>
        <strong>{leftValue}</strong>
      </span>
      <span>
        <small>{rightLabel}</small>
        <strong>{rightValue}</strong>
      </span>
    </div>
  );
}

function formatNumber(value: number): string {
  return value.toFixed(2).replace("-0.00", "0.00");
}

function formatSigned(value: number): string {
  return value > 0 ? `+${value}` : `${value}`;
}

function formatError(source: number, decoded: number): string {
  const error = decoded - source;
  return `(${error >= 0 ? "+" : ""}${error.toFixed(2)})`;
}
