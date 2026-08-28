import { useEffect, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw
} from "lucide-react";

export interface AnimationStep {
  label: string;
  detail: string;
}

export function useAnimationPlayer(stepCount: number) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  const lastStep = Math.max(0, stepCount - 1);

  useEffect(() => {
    if (!playing) return;
    if (step >= lastStep) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => {
      setStep((current) => Math.min(current + 1, lastStep));
    }, 2200);
    return () => window.clearTimeout(timer);
  }, [lastStep, playing, step]);

  const selectStep = (nextStep: number) => {
    setStep(nextStep);
    setPlaying(false);
  };

  return {
    step,
    playing,
    selectStep,
    togglePlaying: () => {
      if (playing) {
        setPlaying(false);
        return;
      }
      if (step >= lastStep) {
        setStep(0);
      }
      setPlaying(true);
    },
    previous: () => selectStep(Math.max(0, step - 1)),
    next: () => selectStep(Math.min(lastStep, step + 1)),
    restart: () => {
      setStep(0);
      setPlaying(!prefersReducedMotion());
    }
  };
}

export function AnimationStepCopy({
  step,
  steps,
  announce
}: {
  step: number;
  steps: readonly AnimationStep[];
  announce: boolean;
}) {
  const activeStep = steps[step] ?? {
    label: "Encoding",
    detail: "Follow the representation through each storage stage."
  };
  return (
    <div
      className="wv-animation-step-copy"
      aria-live={announce ? "polite" : "off"}
      aria-atomic="true"
    >
      <span>{step + 1} / {steps.length}</span>
      <div>
        <strong>{activeStep.label}</strong>
        <p>{activeStep.detail}</p>
      </div>
    </div>
  );
}

export function AnimationControls({
  steps,
  step,
  playing,
  onPrevious,
  onNext,
  onSelect,
  onToggle,
  onRestart
}: {
  steps: readonly AnimationStep[];
  step: number;
  playing: boolean;
  onPrevious: () => void;
  onNext: () => void;
  onSelect: (step: number) => void;
  onToggle: () => void;
  onRestart: () => void;
}) {
  return (
    <footer className="wv-animation-controls">
      <button type="button" aria-label="Previous animation step" onClick={onPrevious}>
        <ChevronLeft aria-hidden="true" />
      </button>
      <button
        className="wv-animation-play"
        type="button"
        aria-label={playing ? "Pause animation" : "Play animation"}
        aria-pressed={playing}
        onClick={onToggle}
      >
        {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
        {playing ? "Pause" : "Play"}
      </button>
      <div className="wv-animation-timeline" aria-label="Animation steps">
        {steps.map((item, index) => (
          <button
            type="button"
            className={index === step ? "active" : ""}
            aria-label={`Step ${index + 1}: ${item.label}`}
            aria-current={index === step ? "step" : undefined}
            key={item.label}
            onClick={() => onSelect(index)}
          />
        ))}
      </div>
      <button type="button" aria-label="Next animation step" onClick={onNext}>
        <ChevronRight aria-hidden="true" />
      </button>
      <button type="button" aria-label="Restart animation" onClick={onRestart}>
        <RotateCcw aria-hidden="true" />
      </button>
    </footer>
  );
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}
