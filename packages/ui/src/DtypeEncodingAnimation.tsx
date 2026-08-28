import { Binary, Braces, Boxes, Cpu, Sparkles } from "lucide-react";
import type { ReactNode } from "react";
import type { WeightFormat } from "@weights-viz/core";
import type { DtypeEducation } from "./dtype-education";
import {
  AnimationControls,
  AnimationStepCopy,
  type AnimationStep,
  useAnimationPlayer
} from "./DtypeAnimationPlayer";
import { ggufQuantContract } from "./gguf-quant-contracts";
import { ggufStorageLayout } from "./gguf-storage-layouts";

export type DtypeAnimationKind =
  | "q4"
  | "k-quant"
  | "gguf-contract"
  | "floating"
  | "integer"
  | "packed"
  | "block"
  | "codebook"
  | "ternary"
  | "microscale"
  | "companion"
  | "schema"
  | "legacy";

export interface AnimationStory {
  kind: DtypeAnimationKind;
  title: string;
  intro: string;
  stages: readonly AnimationStep[];
  source: string[];
  encoded: string[];
  storage: string[];
  decoded: string[];
}

export function dtypeAnimationKind(
  format: WeightFormat,
  dtypeInput: string
): DtypeAnimationKind {
  const dtype = dtypeInput.toUpperCase();
  if (dtype === "Q4_0") return "q4";
  if (/^Q[2-6]_K$/.test(dtype)) return "k-quant";
  if (format === "gguf" && ggufQuantContract(dtype)) return "gguf-contract";
  if (dtype === "STRING" || dtype === "UNDEFINED") return "schema";
  if (["Q4_2", "Q4_3"].includes(dtype) || /_(?:4_4|4_8|8_8)$/.test(dtype)) {
    return "legacy";
  }
  if (dtype.startsWith("IQ")) return "codebook";
  if (dtype.startsWith("TQ")) return "ternary";
  if (dtype === "MXFP4" || dtype === "NVFP4") return "microscale";
  if (dtype === "Q8_1" || dtype === "Q8_K") return "companion";
  if (format === "gguf" && dtype.startsWith("Q")) return "block";
  if (/^(?:U?INT)[24]$/.test(dtype)) return "packed";
  if (
    dtype.includes("FLOAT") ||
    dtype.startsWith("F") ||
    dtype.startsWith("BF") ||
    dtype.startsWith("C") ||
    dtype.startsWith("COMPLEX") ||
    dtype === "DOUBLE"
  ) {
    return "floating";
  }
  return "integer";
}

export function DtypeEncodingAnimation({
  format,
  lesson
}: {
  format: WeightFormat;
  lesson: DtypeEducation;
}) {
  const story = createDtypeAnimationStory(format, lesson);
  const player = useAnimationPlayer(story.stages.length);
  const { step } = player;

  return (
    <section className={`wv-encoding-demo ${story.kind}`}>
      <header>
        <div>
          <span>ANIMATED ENCODING</span>
          <h3>
            <Sparkles aria-hidden="true" />
            {story.title}
          </h3>
          <p>{story.intro}</p>
        </div>
        <div className="wv-encoding-badge">
          <small>lesson family</small>
          <strong>{story.kind.replace("-", " ")}</strong>
        </div>
      </header>

      <div className="wv-encoding-player">
        <AnimationStepCopy
          step={step}
          steps={story.stages}
          announce={!player.playing}
        />
        <div className="wv-encoding-flow">
          <StoryStage
            icon={<Braces aria-hidden="true" />}
            title="Logical values"
            values={story.source}
            state={stageState(step, 0)}
          />
          <FlowConnector active={step >= 1} />
          <StoryStage
            icon={<Binary aria-hidden="true" />}
            title={encodedTitle(story.kind)}
            values={story.encoded}
            state={stageState(step, 1)}
            {...(story.kind === "floating" ? { bits: lesson.segments } : {})}
          />
          <FlowConnector active={step >= 2} />
          <StoryStage
            icon={<Boxes aria-hidden="true" />}
            title="Physical storage"
            values={story.storage}
            state={stageState(step, 2)}
            storage
          />
          <FlowConnector active={step >= 3} />
          <StoryStage
            icon={<Cpu aria-hidden="true" />}
            title="Decode / consume"
            values={story.decoded}
            state={stageState(step, 3)}
          />
        </div>
      </div>

      <AnimationControls
        steps={story.stages}
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

function StoryStage({
  icon,
  title,
  values,
  state,
  bits,
  storage = false
}: {
  icon: ReactNode;
  title: string;
  values: string[];
  state: "waiting" | "active" | "complete";
  bits?: DtypeEducation["segments"];
  storage?: boolean;
}) {
  return (
    <div className={`wv-encoding-stage ${state}`}>
      <header>
        {icon}
        <strong>{title}</strong>
      </header>
      {bits?.length ? (
        <div className="wv-encoding-fields">
          {bits.map((segment) => (
            <span
              className={segment.tone}
              key={segment.label}
              style={{ flexGrow: segment.bits }}
            >
              <b>{segment.label}</b>
              <small>{segment.bits}b</small>
            </span>
          ))}
        </div>
      ) : (
        <div className={storage ? "wv-encoding-bytes" : "wv-encoding-values"}>
          {values.map((value, index) => (
            <span key={`${value}:${index}`}>{value}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function FlowConnector({ active }: { active: boolean }) {
  return (
    <div className={`wv-encoding-arrow${active ? " active" : ""}`}>
      <i />
      <span>›</span>
    </div>
  );
}

function stageState(
  current: number,
  stage: number
): "waiting" | "active" | "complete" {
  if (current === stage) return "active";
  return current > stage ? "complete" : "waiting";
}

function encodedTitle(kind: DtypeAnimationKind): string {
  if (kind === "codebook") return "Codebook lookup";
  if (kind === "ternary") return "Ternary symbols";
  if (kind === "microscale") return "Scale + FP4";
  if (kind === "companion") return "Codes + sums";
  if (kind === "schema") return "Container record";
  if (kind === "legacy") return "Historical type ID";
  if (kind === "gguf-contract") return "Exact block contract";
  if (kind === "block") return "Block codes";
  return "Bit representation";
}

export function createDtypeAnimationStory(
  format: WeightFormat,
  lesson: DtypeEducation
): AnimationStory {
  const kind = dtypeAnimationKind(format, lesson.dtype);
  const bits = lesson.bitsPerValue ?? "variable";
  const exactStorage = format === "gguf"
    ? ggufStorageLayout(lesson.dtype)?.map(
        (field) =>
          `${field.name}: ${field.type} ×${field.count} · ${field.bytes} B · ${field.role}`
      )
    : undefined;
  const commonStages: readonly AnimationStep[] = [
    {
      label: "Read the values",
      detail: "Begin with the logical values the model or graph wants to represent."
    },
    {
      label: "Choose a representation",
      detail: encodingDetail(kind, lesson.dtype)
    },
    {
      label: "Write the bytes",
      detail: storageDetail(format, bits)
    },
    {
      label: "Decode where it is used",
      detail: decodeDetail(kind, lesson)
    }
  ];

  if (kind === "codebook") {
    const scalarCodebook = ["IQ4_NL", "IQ4_XS"].includes(lesson.dtype);
    const signedGridCodebook = ["IQ1_S", "IQ1_M"].includes(lesson.dtype);
    return {
      kind,
      title: scalarCodebook
        ? `Watch ${lesson.dtype} choose nonlinear levels`
        : `Watch ${lesson.dtype} choose multi-weight grids`,
      intro: scalarCodebook
        ? "Each four-bit index selects one scalar nonlinear level; block or local scales restore its magnitude inside the inference kernel."
        : signedGridCodebook
          ? "One packed index selects a grid whose entries are already signed; delta or grid-shift bits and local scales adapt that vector to the source group without a separate sign mask."
        : "One packed index selects a small vector grid for several weights at once; separate sign patterns and local scales adapt that grid to the source group.",
      stages: commonStages,
      source: scalarCodebook
        ? ["−1.10", "−0.22", "+0.05", "+0.94"]
        : ["w0…w7 vector"],
      encoded: scalarCodebook
        ? ["idx 1", "idx 5", "idx 8", "idx 14"]
        : signedGridCodebook
          ? ["signed grid index", "delta / shift bit", "local scale bits"]
        : ["grid index", "sign index / bits", "local scale bits"],
      storage:
        exactStorage ??
        (scalarCodebook
          ? ["01", "05", "08", "0E", "scale"]
          : signedGridCodebook
            ? ["grid", "delta / shift", "scales"]
          : ["grid", "signs", "scales"]),
      decoded: scalarCodebook
        ? ["scalar table[idx]", "× scale", "× activation", "Σ"]
        : signedGridCodebook
          ? ["lookup signed grid", "apply delta / shift", "× local/global scale", "vector dot"]
        : ["lookup 8-value grid", "apply sign pattern", "× local/global scale", "vector dot"]
    };
  }
  if (kind === "ternary") {
    return {
      kind,
      title: `Watch ${lesson.dtype} collapse to three levels`,
      intro:
        "Every source weight becomes negative, zero, or positive; packing and sign selection make the runtime path exceptionally small.",
      stages: commonStages,
      source: ["−0.91", "−0.14", "+0.06", "+0.77", "0.00"],
      encoded: ["−1", "0", "0", "+1", "0"],
      storage:
        exactStorage ??
        (lesson.dtype === "TQ1_0"
          ? ["base-3", "5 trits", "1 byte", "scale"]
          : ["00", "01", "10", "11", "scale"]),
      decoded: ["−activation", "skip", "+activation", "× scale"]
    };
  }
  if (kind === "microscale") {
    const local = lesson.dtype === "NVFP4";
    return {
      kind,
      title: `Watch ${lesson.dtype} move the FP4 range`,
      intro: local
        ? "Four independent 16-value groups each choose a UE4M3 scale before their E2M1 values are packed."
        : "One E8M0 power-of-two scale moves the E2M1 FP4 range over a 32-value block.",
      stages: commonStages,
      source: ["−5.2", "−0.7", "+1.1", "+4.8"],
      encoded: local
        ? ["group max", "÷ 6", "UE4M3 scale", "E2M1 codes"]
        : ["block max", "E8M0 scale", "E2M1 codes"],
      storage: exactStorage ?? (local
        ? ["4 scale B", "32 FP4 B"]
        : ["1 exponent B", "16 FP4 B"]),
      decoded: ["FP4(code)", "× group scale", "fused GEMM"]
    };
  }
  if (kind === "companion") {
    return {
      kind,
      title: `Watch ${lesson.dtype} prepare a dot product`,
      intro:
        "These blocks keep ordinary int8 codes plus precomputed sums that remove repeated offset work in paired quantized kernels.",
      stages: commonStages,
      source: ["−0.8", "+0.2", "+1.0", "−0.4"],
      encoded: ["int8 q", "shared scale", "group Σq"],
      storage:
        exactStorage ??
        (lesson.dtype === "Q8_K"
          ? ["d F32", "256 q", "16 sums"]
          : ["d F16", "scaled sum", "32 q"]),
      decoded: ["integer dot", "offset correction", "× scales", "Σ"]
    };
  }
  if (lesson.dtype === "Q1_0") {
    return {
      kind,
      title: "Watch Q1_0 reduce each weight to its sign",
      intro:
        "One FP16 mean-absolute magnitude serves 128 weights; each weight contributes only a single positive-or-negative bit.",
      stages: commonStages,
      source: ["−0.9", "−0.2", "+0.1", "+0.7"],
      encoded: ["d = mean(|w|)", "w < 0 → 0", "w ≥ 0 → 1"],
      storage: exactStorage ?? ["d FP16", "128 sign bits"],
      decoded: ["bit 0 → −d", "bit 1 → +d", "sign-select activation", "Σ"]
    };
  }
  if (lesson.dtype === "Q2_0") {
    return {
      kind,
      title: "Watch Q2_0 create four reconstruction levels",
      intro:
        "One FP16 max-absolute scale serves 64 weights. Each weight becomes a biased two-bit code selecting −d, 0, +d, or +2d.",
      stages: commonStages,
      source: ["−0.8", "−0.1", "+0.6", "+1.4"],
      encoded: [
        "d = max(|w|)",
        "q = clamp(round(w/d), −1, 2)",
        "stored = q + 1"
      ],
      storage: exactStorage ?? ["d FP16", "64 two-bit codes"],
      decoded: ["q = stored−1", "w′ = d × q", "× activation", "Σ"]
    };
  }
  if (lesson.dtype === "Q5_0") {
    return {
      kind,
      title: "Watch Q5_0 split each five-bit code",
      intro:
        "A signed q from −16 through +15 receives an implicit +16 bias; its low nibble and fifth bit travel in separate arrays.",
      stages: commonStages,
      source: ["−1.7", "−0.2", "+0.5", "+1.4"],
      encoded: [
        "d = signedMaxAbs / −16",
        "q = clamp(round(w/d), −16, 15)",
        "stored = q + 16"
      ],
      storage: exactStorage ?? ["d FP16", "qh high bits", "qs low nibbles"],
      decoded: ["join qh + qs", "q = stored−16", "w′ = d × q"]
    };
  }
  if (lesson.dtype === "Q8_0") {
    return {
      kind,
      title: "Watch Q8_0 keep one signed byte per weight",
      intro:
        "One FP16 scale serves 32 weights, but q is already a signed int8—there is no bias or split bit-plane.",
      stages: commonStages,
      source: ["−1.2", "−0.1", "+0.4", "+1.0"],
      encoded: ["d = max(|w|) / 127", "q = round(w/d)", "signed int8"],
      storage: exactStorage ?? ["d FP16", "32 × int8 q"],
      decoded: ["w′ = d × signed q", "× activation", "Σ"]
    };
  }
  if (kind === "block") {
    return {
      kind,
      title: `Watch a ${lesson.dtype} block compress`,
      intro:
        "Neighboring weights share metadata, while each low-bit code records only its position within the reconstructed range.",
      stages: commonStages,
      source: ["−1.4", "−0.3", "+0.4", "+1.2"],
      encoded: lesson.dtype.endsWith("_1")
        ? ["find min", "find max", "q = round((w−min)/d)"]
        : ["find maxAbs", "d = range / levels", "q = round(w/d)"],
      storage:
        exactStorage ?? [
          ...(lesson.dtype.endsWith("_1") ? ["scale", "minimum"] : ["scale"]),
          `${bits}-bit codes`
        ],
      decoded: [
        lesson.formula?.expression ?? "metadata × code",
        "× activation",
        "Σ"
      ]
    };
  }
  if (kind === "schema") {
    const stringType = lesson.dtype === "STRING";
    return {
      kind,
      title: `Watch ${lesson.dtype} travel through TensorProto`,
      intro: stringType
        ? "Strings are length-delimited protobuf fields, not a fixed-width numerical array."
        : "UNDEFINED is schema state: it names no numerical encoding and must be resolved before execution.",
      stages: commonStages,
      source: stringType ? ["“cat”", "“模型”"] : ["type = ?"],
      encoded: stringType ? ["length 3", "UTF-8 bytes", "length 6"] : ["enum 0"],
      storage: stringType
        ? ["42", "03", "63", "61", "74", "42", "06", "…"]
        : ["TensorProto", "data_type: 0"],
      decoded: stringType ? ["protobuf parser", "text tensor"] : ["type inference", "concrete dtype required"]
    };
  }
  if (kind === "legacy") {
    return {
      kind,
      title: `See why ${lesson.dtype} needs a matching runtime`,
      intro:
        "The identifier remains recognizable for diagnostics, but its historical or pre-interleaved byte contract is not a recommended portable output format.",
      stages: commonStages,
      source: ["historical tensor"],
      encoded: ["GGML type ID", lesson.dtype],
      storage: ["legacy layout", "kernel-specific order"],
      decoded: ["compatibility check", "matching old kernel", "or reject"]
    };
  }
  if (kind === "floating") {
    const complex = ["C64", "COMPLEX64", "COMPLEX128"].includes(lesson.dtype);
    const exponentOnly = ["F8_E8M0", "FLOAT8E8M0"].includes(lesson.dtype);
    if (complex) {
      return {
        kind,
        title: `Watch ${lesson.dtype} preserve two components`,
        intro:
          "A complex value stores an independent real floating-point component followed by an imaginary floating-point component.",
        stages: commonStages,
        source: ["−1.25 + 0.75i"],
        encoded: ["real = −1.25", "imag = +0.75"],
        storage: byteLabels(bits),
        decoded: ["load real", "load imaginary", "re + im·i"]
      };
    }
    if (exponentOnly) {
      return {
        kind,
        title: `Watch ${lesson.dtype} encode a power of two`,
        intro:
          "E8M0 has no sign or fraction. Each element independently stores an unsigned exponent; a surrounding format may choose to use that value as a group scale.",
        stages: [
          {
            label: "Start with a positive scale value",
            detail: "E8M0 represents non-negative power-of-two values rather than general signed weights."
          },
          {
            label: "Choose the exponent",
            detail: "Round to a representable power of two and encode only the biased eight-bit exponent—there is no sign or significand."
          },
          {
            label: "Write one byte per element",
            detail: `${format === "onnx" ? "TensorProto" : "SafeTensors"} stores each E8M0 element independently; grouping is not part of this dtype.`
          },
          {
            label: "Recover the power-of-two value",
            detail: "Decode the exponent as a scale value; any group-sharing semantics come from the surrounding operator or block format."
          }
        ],
        source: ["scale ≈ 8.0"],
        encoded: ["E = 130", "no sign", "no fraction"],
        storage: ["E: uint8 ×1 · 1 B"],
        decoded: ["2^(E−127)", "= 8.0"]
      };
    }
    return {
      kind,
      title: `Watch ${lesson.dtype} split a number into fields`,
      intro:
        "The sign, exponent, and fraction move independently: sign chooses direction, exponent chooses scale, and fraction adds detail.",
      stages: commonStages,
      source: ["−3.25"],
      encoded: lesson.segments.map(({ label, bits: width }) => `${label} ${width}b`),
      storage: byteLabels(bits),
      decoded: [
        lesson.formula?.expression ?? "fields → value",
        "round to representable value"
      ]
    };
  }
  if (kind === "packed") {
    return {
      kind,
      title: `Watch ${lesson.dtype} values share bytes`,
      intro:
        "Sub-byte lanes are laid side by side; masks and shifts recover each independent integer before graph scaling.",
      stages: commonStages,
      source: ["0", "1", "2", "3"],
      encoded: bits === 2 ? ["00", "01", "10", "11"] : ["0011", "1100"],
      storage: ["one shared byte", bits === 2 ? "11·10·01·00" : "1100·0011"],
      decoded: ["shift", "mask", "(q−zeroPoint)×scale"]
    };
  }
  const boolean = lesson.dtype === "BOOL";
  const unsigned =
    lesson.dtype.startsWith("U") || lesson.dtype.startsWith("UINT");
  return {
    kind,
    title: `Watch ${lesson.dtype} become bytes`,
    intro: boolean
      ? "A logical false or true occupies one full byte even though only 0 and 1 are valid."
      : unsigned
        ? "A non-negative integer maps directly to an unsigned fixed-width bit pattern."
        : "A signed integer maps to a fixed-width two’s-complement pattern.",
    stages: commonStages,
    source: boolean
      ? ["false", "true"]
      : unsigned
        ? ["0", "42", "173"]
        : ["−37", "0", "42"],
    encoded: boolean
      ? ["00000000", "00000001"]
      : unsigned
        ? ["unsigned binary"]
        : ["two’s complement"],
    storage: byteLabels(bits),
    decoded: ["load bytes", "interpret dtype", "use value"]
  };
}

function byteLabels(bits: number | string): string[] {
  if (typeof bits !== "number") return ["variable bytes"];
  const bytes = Math.max(1, Math.ceil(bits / 8));
  if (bytes <= 8) {
    return Array.from({ length: bytes }, (_, index) => `B${index}`);
  }
  return [`${bytes} consecutive bytes`];
}

function encodingDetail(kind: DtypeAnimationKind, dtype: string): string {
  if (kind === "floating") return `Split ${dtype} into its sign, exponent, and significand fields.`;
  if (kind === "packed") return "Place several independent sub-byte codes into one byte.";
  if (kind === "codebook") {
    return ["IQ4_NL", "IQ4_XS"].includes(dtype)
      ? "Select the scalar nonlinear level that minimizes reconstruction error."
      : "Select a multi-weight grid, sign pattern, and scale that minimize group reconstruction error.";
  }
  if (kind === "ternary") return "Round normalized values into the alphabet {−1, 0, +1}.";
  if (kind === "microscale") return "Choose a group scale, then round normalized values to E2M1 FP4.";
  if (kind === "companion") return "Quantize int8 lanes and precompute sums for paired dot products.";
  if (kind === "block") return "Fit shared block metadata, then quantize each weight into a small code.";
  if (kind === "schema") return "Encode the protobuf field or preserve the unresolved schema marker.";
  if (kind === "legacy") return "Preserve the historical type identifier and byte ordering contract.";
  return "Convert each value to its fixed-width integer bit pattern.";
}

function storageDetail(format: WeightFormat, bits: number | string): string {
  const container =
    format === "gguf" ? "GGUF/GGML" : format === "onnx" ? "TensorProto" : "SafeTensors";
  return `${container} writes the representation contiguously; nominal width is ${bits} bits per value before container alignment.`;
}

function decodeDetail(kind: DtypeAnimationKind, lesson: DtypeEducation): string {
  if (kind === "schema") return "A parser returns text or requires type inference; no numerical dequantization is invented.";
  if (kind === "legacy") return "Only a runtime implementing the exact historical layout may consume it.";
  if (kind === "codebook") {
    return ["IQ4_NL", "IQ4_XS"].includes(lesson.dtype)
      ? "Look up each scalar nonlinear level, apply its scale, and immediately multiply it."
      : "Expand each indexed multi-weight grid, apply its sign pattern and scales, and feed the vector directly to the dot product.";
  }
  if (kind === "ternary") return "Turn symbols into add, skip, or subtract operations, then apply the scale.";
  return lesson.formula?.explanation ?? "Load the bits using the declared dtype and feed the resulting lanes to the operator.";
}
