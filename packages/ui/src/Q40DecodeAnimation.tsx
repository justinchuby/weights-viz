import { useEffect, useState, type ReactNode } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
  Sparkles
} from "lucide-react";

const SCALE = 0.25;

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
    label: "Source block",
    detail: "Start with 32 floating-point weights. Eight representative lanes are shown."
  },
  {
    label: "Quantize",
    detail: "Divide by the shared scale, round, and clamp each result to −8…7."
  },
  {
    label: "Pack nibbles",
    detail: "Add 8, then pair q[i] with q[i+16] in the low and high nibbles."
  },
  {
    label: "Decode",
    detail: "Subtract 8 and multiply by the shared scale. The small gap is quantization error."
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
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  const activeStep = STEPS[step] ?? STEPS[0];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(() => {
      setStep((current) => (current + 1) % STEPS.length);
    }, step === STEPS.length - 1 ? 3000 : 2200);
    return () => window.clearTimeout(timer);
  }, [playing, step]);

  const selectStep = (nextStep: number) => {
    setStep(nextStep);
    setPlaying(false);
  };

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
            A faithful miniature of GGML packing. A real block stores one FP16
            scale and 32 four-bit codes in 18 bytes.
          </p>
        </div>
        <div className="wv-q4-scale">
          <small>shared FP16 scale</small>
          <strong>d = {SCALE}</strong>
        </div>
      </header>

      <div className="wv-q4-player">
        <div className="wv-q4-step-copy" aria-live="polite">
          <span>{step + 1} / {STEPS.length}</span>
          <div>
            <strong>{activeStep.label}</strong>
            <p>{activeStep.detail}</p>
          </div>
        </div>

        <div className="wv-q4-flow" data-step={step}>
          <FlowStage title="F32 source" subtitle="sampled block lanes" active={step === 0}>
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

          <FlowStage title="Signed codes" subtitle="q ∈ [−8, 7]" revealed={step >= 1} active={step === 1}>
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

          <FlowArrow active={step >= 2} label="+ 8 · pack" />

          <FlowStage title="Packed bytes" subtitle="high 4b · low 4b" revealed={step >= 2} active={step === 2}>
            {Q40_DEMO_PAIRS.map((pair) => (
              <div className="wv-q4-byte" key={pair.lowIndex}>
                <span>{pair.highNibble.toString(2).padStart(4, "0")}</span>
                <span>{pair.lowNibble.toString(2).padStart(4, "0")}</span>
                <small>0x{pair.packedByte.toString(16).padStart(2, "0").toUpperCase()}</small>
              </div>
            ))}
          </FlowStage>

          <FlowArrow active={step >= 3} label="unpack · × d" />

          <FlowStage title="Approximate F32" subtitle="decoded value · error" revealed={step >= 3} active={step === 3}>
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

      <footer className="wv-q4-controls">
        <button
          type="button"
          aria-label="Previous animation step"
          onClick={() => selectStep((step - 1 + STEPS.length) % STEPS.length)}
        >
          <ChevronLeft aria-hidden="true" />
        </button>
        <button
          className="wv-q4-play"
          type="button"
          aria-label={playing ? "Pause animation" : "Play animation"}
          aria-pressed={playing}
          onClick={() => setPlaying((current) => !current)}
        >
          {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
          {playing ? "Pause" : "Play"}
        </button>
        <div className="wv-q4-timeline" aria-label="Animation steps">
          {STEPS.map((item, index) => (
            <button
              type="button"
              className={index === step ? "active" : ""}
              aria-label={`Step ${index + 1}: ${item.label}`}
              aria-current={index === step ? "step" : undefined}
              key={item.label}
              onClick={() => selectStep(index)}
            />
          ))}
        </div>
        <button
          type="button"
          aria-label="Next animation step"
          onClick={() => selectStep((step + 1) % STEPS.length)}
        >
          <ChevronRight aria-hidden="true" />
        </button>
        <button
          type="button"
          aria-label="Restart animation"
          onClick={() => {
            setStep(0);
            setPlaying(!prefersReducedMotion());
          }}
        >
          <RotateCcw aria-hidden="true" />
        </button>
      </footer>
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

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}
