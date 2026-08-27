import type { TensorRecord, WeightFormat } from "@weights-viz/core";

export type EncodingTone =
  | "sign"
  | "exponent"
  | "fraction"
  | "integer"
  | "metadata"
  | "codes";

export interface EncodingSegment {
  label: string;
  bits: number;
  tone: EncodingTone;
}

export interface PackingGroup {
  values: number;
  bytes: number;
  bitsPerValue: number;
}

export interface BlockSection {
  label: string;
  bits: number;
  tone: "metadata" | "codes";
}

export interface BlockEncoding {
  elements: number;
  bytes: number;
  effectiveBitsPerValue: number;
  sections: BlockSection[];
}

export interface EncodingConcept {
  term: string;
  explanation: string;
}

export interface DtypeEducation {
  format: WeightFormat;
  formatLabel: string;
  dtype: string;
  family: string;
  summary: string;
  storageNote: string;
  bitsPerValue?: number;
  segments: EncodingSegment[];
  packing?: PackingGroup;
  block?: BlockEncoding;
  formula?: {
    expression: string;
    explanation: string;
  };
  concepts: EncodingConcept[];
  elementCount: bigint;
  tensorBitsPerValue?: number;
  f32CompressionRatio?: number;
}

interface ScalarSpec {
  bits: number;
  family: string;
  segments: EncodingSegment[];
  summary: string;
}

const FORMAT_LABELS: Record<WeightFormat, string> = {
  safetensors: "SafeTensors",
  gguf: "GGUF",
  onnx: "ONNX"
};

const FIXED_WIDTH_BITS: Record<string, number> = {
  BOOL: 8,
  F4: 4,
  F6_E2M3: 6,
  F6_E3M2: 6,
  U8: 8,
  UINT8: 8,
  I8: 8,
  INT8: 8,
  F8_E5M2: 8,
  F8_E4M3: 8,
  F8_E8M0: 8,
  F8_E4M3FNUZ: 8,
  F8_E5M2FNUZ: 8,
  FLOAT8E4M3FN: 8,
  FLOAT8E4M3FNUZ: 8,
  FLOAT8E5M2: 8,
  FLOAT8E5M2FNUZ: 8,
  FLOAT8E8M0: 8,
  I16: 16,
  INT16: 16,
  U16: 16,
  UINT16: 16,
  F16: 16,
  FLOAT16: 16,
  BF16: 16,
  BFLOAT16: 16,
  I32: 32,
  INT32: 32,
  U32: 32,
  UINT32: 32,
  F32: 32,
  FLOAT: 32,
  C64: 64,
  COMPLEX64: 64,
  F64: 64,
  DOUBLE: 64,
  I64: 64,
  INT64: 64,
  U64: 64,
  UINT64: 64,
  COMPLEX128: 128,
  UINT4: 4,
  INT4: 4,
  FLOAT4E2M1: 4,
  UINT2: 2,
  INT2: 2
};

const FLOAT_LAYOUTS: Record<string, [number, number, number]> = {
  F4: [1, 2, 1],
  FLOAT4E2M1: [1, 2, 1],
  F6_E2M3: [1, 2, 3],
  F6_E3M2: [1, 3, 2],
  F8_E4M3: [1, 4, 3],
  F8_E4M3FNUZ: [1, 4, 3],
  FLOAT8E4M3FN: [1, 4, 3],
  FLOAT8E4M3FNUZ: [1, 4, 3],
  F8_E5M2: [1, 5, 2],
  F8_E5M2FNUZ: [1, 5, 2],
  FLOAT8E5M2: [1, 5, 2],
  FLOAT8E5M2FNUZ: [1, 5, 2],
  F16: [1, 5, 10],
  FLOAT16: [1, 5, 10],
  BF16: [1, 8, 7],
  BFLOAT16: [1, 8, 7],
  F32: [1, 8, 23],
  FLOAT: [1, 8, 23],
  F64: [1, 11, 52],
  DOUBLE: [1, 11, 52]
};

export function createDtypeEducation(
  format: WeightFormat,
  tensor: TensorRecord
): DtypeEducation {
  const dtype = tensor.dtype.toUpperCase();
  const elementCount = product(tensor.shape);
  const tensorBitsPerValue = ratio(tensor.byteLength * 8n, elementCount);
  const f32CompressionRatio =
    tensor.byteLength > 0n
      ? ratio(elementCount * 4n, tensor.byteLength)
      : undefined;

  if (format === "gguf") {
    const blockBytes = encodingNumber(tensor, "blockBytes");
    const blockElements = encodingNumber(tensor, "blockElements");
    if (blockBytes && blockElements) {
      const nominalBits = ggufNominalBits(dtype);
      const block = createBlockEncoding(
        dtype,
        blockBytes,
        blockElements,
        nominalBits
      );
      const packing = ggufPacking(dtype, nominalBits);
      const kind = ggufBlockKind(dtype);
      return {
        format,
        formatLabel: FORMAT_LABELS[format],
        dtype,
        family: kind.family,
        summary: kind.summary,
        storageNote:
          "GGUF stores this as row-aligned GGML blocks. Every block carries packed values plus the metadata needed to reconstruct them.",
        ...(nominalBits ? { bitsPerValue: nominalBits } : {}),
        segments: nominalBits
          ? [{ label: "quantized code", bits: nominalBits, tone: "codes" }]
          : [],
        ...(packing ? { packing } : {}),
        block,
        formula: kind.formula,
        concepts: ggufConcepts(dtype, block),
        elementCount,
        ...(tensorBitsPerValue !== undefined ? { tensorBitsPerValue } : {}),
        ...(f32CompressionRatio !== undefined ? { f32CompressionRatio } : {})
      };
    }
  }

  const scalar = scalarSpec(dtype);
  const bitsPerValue =
    scalar?.bits ??
    encodingNumber(tensor, "scalarBytes") ??
    FIXED_WIDTH_BITS[dtype];
  const resolvedBits =
    scalar?.bits ??
    (typeof bitsPerValue === "number"
      ? format === "gguf"
        ? bitsPerValue * 8
        : bitsPerValue
      : undefined);
  const family = scalar?.family ?? "Format-defined storage";
  const segments = scalar?.segments ?? [];
  const lowBitInteger = /^(?:U?INT)[24]$/.test(dtype);
  const lowPrecisionFloat =
    resolvedBits !== undefined &&
    resolvedBits < 16 &&
    (dtype.includes("FLOAT") || dtype.startsWith("F"));

  return {
    format,
    formatLabel: FORMAT_LABELS[format],
    dtype,
    family,
    summary:
      scalar?.summary ??
      `The file declares ${dtype}, but its exact bit layout is not described by this version of the visualizer.`,
    storageNote: formatStorageNote(format, dtype),
    ...(resolvedBits !== undefined ? { bitsPerValue: resolvedBits } : {}),
    segments,
    ...(resolvedBits !== undefined
      ? { packing: packingForBits(resolvedBits) }
      : {}),
    ...(lowBitInteger
      ? {
          formula: {
            expression: "real = (q − zeroPoint) × scale",
            explanation:
              "ONNX quantized graphs usually keep scale and zero-point in separate tensors or QuantizeLinear/DequantizeLinear nodes."
          }
        }
      : lowPrecisionFloat
        ? {
            formula: {
              expression: "value = (−1)ˢ × 2ᵉ × significand",
              explanation:
                "A tiny exponent and fraction trade precision and range for compact storage; no separate quantization scale is required."
            }
          }
        : scalar?.family === "Floating point"
          ? {
              formula: {
                expression: "value = (−1)ˢ × 2ᵉ × significand",
                explanation:
                  "The exponent controls range while the fraction controls precision."
              }
            }
          : {}),
    concepts: scalarConcepts(format, dtype, scalar),
    elementCount,
    ...(tensorBitsPerValue !== undefined ? { tensorBitsPerValue } : {}),
    ...(f32CompressionRatio !== undefined ? { f32CompressionRatio } : {})
  };
}

function scalarSpec(dtype: string): ScalarSpec | undefined {
  const floatLayout = FLOAT_LAYOUTS[dtype];
  if (floatLayout) {
    const [sign, exponent, fraction] = floatLayout;
    return {
      bits: sign + exponent + fraction,
      family: "Floating point",
      segments: [
        ...(sign
          ? [{ label: "sign", bits: sign, tone: "sign" as const }]
          : []),
        { label: "exponent", bits: exponent, tone: "exponent" },
        { label: "fraction", bits: fraction, tone: "fraction" }
      ],
      summary: `${dtype} divides each value into sign, exponent, and fraction fields. More exponent bits increase range; more fraction bits increase precision.`
    };
  }
  if (dtype === "F8_E8M0" || dtype === "FLOAT8E8M0") {
    return {
      bits: 8,
      family: "Exponent-only scale",
      segments: [{ label: "shared exponent", bits: 8, tone: "exponent" }],
      summary:
        "E8M0 is an unsigned exponent-only value, commonly used as a shared power-of-two scale rather than a standalone signed weight."
    };
  }
  if (dtype === "C64" || dtype === "COMPLEX64") {
    return complexSpec(dtype, 32);
  }
  if (dtype === "COMPLEX128") {
    return complexSpec(dtype, 64);
  }
  if (dtype === "BOOL") {
    return {
      bits: 8,
      family: "Boolean",
      segments: [{ label: "0 or 1", bits: 8, tone: "integer" }],
      summary:
        "A logical value occupies one full byte even though only the values 0 and 1 are valid."
    };
  }
  const bits = FIXED_WIDTH_BITS[dtype];
  if (!bits) return undefined;
  const signed = dtype.startsWith("I") || dtype.startsWith("INT");
  const unsigned = dtype.startsWith("U") || dtype.startsWith("UINT");
  if (!signed && !unsigned) return undefined;
  return {
    bits,
    family: signed ? "Signed integer" : "Unsigned integer",
    segments: signed
      ? [
          { label: "sign", bits: 1, tone: "sign" },
          { label: "two's-complement value", bits: bits - 1, tone: "integer" }
        ]
      : [{ label: "integer value", bits, tone: "integer" }],
    summary: signed
      ? `${dtype} stores a two's-complement integer in ${bits} bits.`
      : `${dtype} stores a non-negative integer in ${bits} bits.`
  };
}

function complexSpec(dtype: string, componentBits: number): ScalarSpec {
  return {
    bits: componentBits * 2,
    family: "Complex floating point",
    segments: [
      { label: "real", bits: componentBits, tone: "fraction" },
      { label: "imaginary", bits: componentBits, tone: "exponent" }
    ],
    summary: `${dtype} stores a real and an imaginary floating-point component back to back.`
  };
}

function createBlockEncoding(
  dtype: string,
  bytes: number,
  elements: number,
  nominalBits: number | undefined
): BlockEncoding {
  const totalBits = bytes * 8;
  if (!nominalBits) {
    return {
      elements,
      bytes,
      effectiveBitsPerValue: totalBits / elements,
      sections: [
        {
          label: "packed codes + block metadata",
          bits: totalBits,
          tone: "codes"
        }
      ]
    };
  }
  const codeBits = nominalBits
    ? Math.min(totalBits, Math.floor(elements * nominalBits))
    : totalBits;
  const metadataBits = totalBits - codeBits;
  return {
    elements,
    bytes,
    effectiveBitsPerValue: totalBits / elements,
    sections: [
      ...(metadataBits > 0
        ? [
            {
              label: "scale / lookup metadata",
              bits: metadataBits,
              tone: "metadata" as const
            }
          ]
        : []),
      { label: `${elements} packed codes`, bits: codeBits, tone: "codes" }
    ]
  };
}

function ggufNominalBits(dtype: string): number | undefined {
  if (dtype.startsWith("MXFP4") || dtype.startsWith("NVFP4")) return 4;
  if (dtype.startsWith("TQ1")) return undefined;
  const match = /^(?:I?Q|TQ)(\d)/.exec(dtype);
  return match?.[1] ? Number(match[1]) : undefined;
}

function ggufPacking(
  dtype: string,
  nominalBits: number | undefined
): PackingGroup | undefined {
  if (
    nominalBits === undefined ||
    !["Q4_0", "Q4_1", "Q8_0", "Q8_1"].includes(dtype)
  ) {
    return undefined;
  }
  return packingForBits(nominalBits);
}

function ggufBlockKind(dtype: string): {
  family: string;
  summary: string;
  formula: NonNullable<DtypeEducation["formula"]>;
} {
  if (dtype.startsWith("IQ")) {
    return {
      family: "Importance-aware codebook quantization",
      summary:
        "IQ blocks spend their small indices on a learned or predefined codebook and use importance-aware layouts to preserve the most valuable weights.",
      formula: {
        expression: "weight ≈ scale × codebook[index]",
        explanation:
          "The packed bits select a codebook entry; block scales restore the local magnitude."
      }
    };
  }
  if (/^Q\d_K/.test(dtype)) {
    return {
      family: "K-quant super-block",
      summary:
        "K-quants group 256 weights into a super-block with a global scale and compact per-group scales or minima.",
      formula: {
        expression: "weight ≈ globalScale × subScale × q + minimum",
        explanation:
          "Hierarchical scales adapt to local ranges while sharing metadata across a larger block."
      }
    };
  }
  if (dtype.startsWith("TQ")) {
    return {
      family: "Ternary block quantization",
      summary:
        "Ternary blocks encode weights with a tiny alphabet centered around zero, then restore magnitude with a block scale.",
      formula: {
        expression: "weight ≈ scale × {−1, 0, +1}",
        explanation:
          "Three code values need less than two effective bits before scale metadata."
      }
    };
  }
  if (dtype.startsWith("MXFP4")) {
    return {
      family: "Microscaled FP4",
      summary:
        "MXFP4 combines 4-bit E2M1 floating-point values with a shared E8M0 power-of-two scale for each block.",
      formula: {
        expression: "weight ≈ 2ᴱ × FP4(E2M1)",
        explanation:
          "The shared exponent moves the small FP4 range to match each local group."
      }
    };
  }
  if (dtype.startsWith("NVFP4")) {
    return {
      family: "Two-level scaled FP4",
      summary:
        "NVFP4 stores compact FP4 values with local block scales and a higher-level scale to cover a wider model range.",
      formula: {
        expression: "weight ≈ globalScale × blockScale × FP4",
        explanation:
          "Two scale levels preserve local detail while keeping each weight at four payload bits."
      }
    };
  }
  if (dtype === "Q4_1" || dtype === "Q5_1") {
    return {
      family: "Affine block quantization",
      summary:
        `${dtype} packs integer codes and stores both a scale and a minimum for each block.`,
      formula: {
        expression: "weight ≈ scale × q + minimum",
        explanation:
          "The minimum acts like an offset, so asymmetric ranges can use all available codes."
      }
    };
  }
  return {
    family: "Symmetric block quantization",
    summary:
      `${dtype} packs many low-bit codes into one block and shares scale metadata across them.`,
    formula: {
      expression: "weight ≈ scale × (q − bias)",
      explanation:
        "A shared scale restores magnitude; signed formats may use no explicit bias."
    }
  };
}

function ggufConcepts(
  dtype: string,
  block: BlockEncoding
): EncodingConcept[] {
  const kind = ggufBlockKind(dtype);
  return [
    {
      term: "Block",
      explanation: `${block.elements} neighboring weights share one ${block.bytes}-byte encoded unit.`
    },
    {
      term: "Effective bits",
      explanation: `${formatDecimal(block.effectiveBitsPerValue)} bits per weight includes codes and all block metadata.`
    },
    {
      term: dtype.includes("_1") ? "Minimum / offset" : "Scale",
      explanation: kind.formula?.explanation ?? "Block metadata reconstructs the local value range."
    },
    {
      term: "Row alignment",
      explanation:
        "GGML block types require the innermost tensor row to contain a whole number of blocks."
    }
  ];
}

function scalarConcepts(
  format: WeightFormat,
  dtype: string,
  scalar: ScalarSpec | undefined
): EncodingConcept[] {
  const concepts: EncodingConcept[] = [];
  if (scalar?.family.includes("Floating")) {
    concepts.push(
      {
        term: "Exponent",
        explanation:
          "Controls dynamic range: how tiny or huge a representable value can be."
      },
      {
        term: "Fraction",
        explanation:
          "Controls precision: how many nearby values can be distinguished."
      }
    );
  } else {
    concepts.push({
      term: "Packing",
      explanation:
        "Sub-byte values share storage bytes; byte-aligned values occupy one or more complete bytes."
    });
  }
  if (format === "onnx") {
    concepts.push({
      term: "Graph quantization",
      explanation:
        "Scale and zero-point are usually separate initializers consumed by Q/DQ or quantized operators."
    });
  } else if (format === "safetensors") {
    concepts.push({
      term: "Container vs encoding",
      explanation:
        "SafeTensors records dtype, shape, and byte range but does not define an extra block scale around ordinary tensors."
    });
  } else {
    concepts.push({
      term: "Scalar GGML type",
      explanation:
        `${dtype} is stored directly per element rather than sharing a quantization block.`
    });
  }
  concepts.push({
    term: "Endianness",
    explanation:
      "Multi-byte fields are serialized in a defined byte order; bits within each field retain the dtype layout."
  });
  return concepts;
}

function formatStorageNote(format: WeightFormat, dtype: string): string {
  if (format === "safetensors") {
    return `${dtype} values occupy one contiguous SafeTensors payload range. The JSON header supplies dtype, shape, and exact byte offsets.`;
  }
  if (format === "onnx") {
    return `${dtype} values live in TensorProto raw_data or an external data file. Quantization parameters are graph data, not hidden inside the ONNX container.`;
  }
  return `${dtype} is a scalar GGML storage type inside GGUF; each element is stored directly without shared block metadata.`;
}

function packingForBits(bits: number): PackingGroup {
  const divisor = greatestCommonDivisor(bits, 8);
  return {
    values: 8 / divisor,
    bytes: bits / divisor,
    bitsPerValue: bits
  };
}

function greatestCommonDivisor(left: number, right: number): number {
  let a = left;
  let b = right;
  while (b !== 0) {
    const remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}

function encodingNumber(
  tensor: TensorRecord,
  key: string
): number | undefined {
  const value = tensor.encoding?.[key];
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

function product(values: bigint[]): bigint {
  return values.reduce((result, value) => result * value, 1n);
}

function ratio(numerator: bigint, denominator: bigint): number | undefined {
  if (denominator <= 0n) return undefined;
  return Number((numerator * 100n) / denominator) / 100;
}

export function formatDecimal(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}
