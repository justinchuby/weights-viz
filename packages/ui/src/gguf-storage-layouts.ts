export interface GgufStorageField {
  name: string;
  type: string;
  count: number;
  bytes: number;
  role: string;
}

const LAYOUTS: Record<string, readonly GgufStorageField[]> = {
  Q4_0: [
    field("d", "FP16", 1, 2, "one scale for 32 weights"),
    field("qs", "uint8", 16, 16, "32 biased 4-bit codes")
  ],
  Q4_1: [
    field("d", "FP16", 1, 2, "one scale for 32 weights"),
    field("m", "FP16", 1, 2, "one minimum for 32 weights"),
    field("qs", "uint8", 16, 16, "32 unsigned 4-bit codes")
  ],
  Q5_0: [
    field("d", "FP16", 1, 2, "one scale for 32 weights"),
    field("qh", "uint8", 4, 4, "the fifth bit of 32 codes"),
    field("qs", "uint8", 16, 16, "low four bits of 32 codes")
  ],
  Q5_1: [
    field("d", "FP16", 1, 2, "one scale for 32 weights"),
    field("m", "FP16", 1, 2, "one minimum for 32 weights"),
    field("qh", "uint8", 4, 4, "the fifth bit of 32 codes"),
    field("qs", "uint8", 16, 16, "low four bits of 32 codes")
  ],
  Q8_0: [
    field("d", "FP16", 1, 2, "one scale for 32 weights"),
    field("qs", "int8", 32, 32, "one signed code per weight")
  ],
  Q8_1: [
    field("d", "FP16", 1, 2, "one scale for 32 weights"),
    field("s", "FP16", 1, 2, "independently rounded encoder scale × sum(q), for dot products"),
    field("qs", "int8", 32, 32, "one signed code per weight")
  ],
  Q2_K: [
    field("scales", "uint8", 16, 16, "one 4-bit scale and 4-bit minimum per 16 weights"),
    field("qs", "uint8", 64, 64, "256 interleaved two-bit codes"),
    field("d", "FP16", 1, 2, "global multiplier for local scales"),
    field("dmin", "FP16", 1, 2, "global multiplier for local minima")
  ],
  Q3_K: [
    field("hmask", "uint8", 32, 32, "third code bit for 256 weights"),
    field("qs", "uint8", 64, 64, "low two code bits for 256 weights"),
    field("scales", "uint8", 12, 12, "16 packed signed six-bit local scales"),
    field("d", "FP16", 1, 2, "global multiplier for local scales")
  ],
  Q4_K: [
    field("d", "FP16", 1, 2, "global multiplier for local scales"),
    field("dmin", "FP16", 1, 2, "global multiplier for local minima"),
    field("scales", "uint8", 12, 12, "eight packed six-bit scale/minimum pairs"),
    field("qs", "uint8", 128, 128, "256 four-bit codes")
  ],
  Q5_K: [
    field("d", "FP16", 1, 2, "global multiplier for local scales"),
    field("dmin", "FP16", 1, 2, "global multiplier for local minima"),
    field("scales", "uint8", 12, 12, "eight packed six-bit scale/minimum pairs"),
    field("qh", "uint8", 32, 32, "fifth code bit for 256 weights"),
    field("qs", "uint8", 128, 128, "low four code bits for 256 weights")
  ],
  Q6_K: [
    field("ql", "uint8", 128, 128, "low four code bits for 256 weights"),
    field("qh", "uint8", 64, 64, "high two code bits for 256 weights"),
    field("scales", "int8", 16, 16, "one signed local scale per 16 weights"),
    field("d", "FP16", 1, 2, "global multiplier for local scales")
  ],
  Q8_K: [
    field("d", "FP32", 1, 4, "one scale for 256 values"),
    field("qs", "int8", 256, 256, "one signed code per value"),
    field("bsums", "int16", 16, 32, "one sum per 16 codes")
  ],
  IQ2_XXS: [
    field("d", "FP16", 1, 2, "one scale for 256 weights"),
    field(
      "qs",
      "uint16",
      32,
      64,
      "per 32 weights: four grid indices, four sign indices, and one embedded scale nibble"
    )
  ],
  IQ2_XS: [
    field("d", "FP16", 1, 2, "one super-block scale"),
    field("qs", "uint16", 32, 64, "each word holds a 9-bit grid index and 7-bit sign index"),
    field("scales", "uint8", 8, 8, "two 4-bit local scales per byte")
  ],
  IQ3_XXS: [
    field("d", "FP16", 1, 2, "one scale for 256 weights"),
    field(
      "qs",
      "uint8",
      96,
      96,
      "64 grid-index bytes followed by 32 packed sign/scale bytes"
    )
  ],
  IQ1_S: [
    field("d", "FP16", 1, 2, "one scale for 256 weights"),
    field("qs", "uint8", 32, 32, "low grid-index bits"),
    field("qh", "uint16", 8, 16, "four index highs, one local scale, and delta sign per word")
  ],
  IQ3_S: [
    field("d", "FP16", 1, 2, "one super-block scale"),
    field("qs", "uint8", 64, 64, "low grid-index bits"),
    field("qh", "uint8", 8, 8, "high grid-index bits"),
    field("signs", "uint8", 32, 32, "packed sign bits"),
    field("scales", "uint8", 4, 4, "two 4-bit local scales per byte")
  ],
  IQ2_S: [
    field("d", "FP16", 1, 2, "one super-block scale"),
    field(
      "qs",
      "uint8",
      64,
      64,
      "first 32 bytes hold low grid-index bits; final 32 bytes hold sign masks"
    ),
    field("qh", "uint8", 8, 8, "high grid-index bits"),
    field("scales", "uint8", 8, 8, "packed local scales")
  ],
  IQ4_NL: [
    field("d", "FP16", 1, 2, "one scale for 32 weights"),
    field("qs", "uint8", 16, 16, "32 four-bit codebook indices")
  ],
  IQ4_XS: [
    field("d", "FP16", 1, 2, "one super-block scale"),
    field("scales_h", "uint16", 1, 2, "two high bits for each of eight local scales"),
    field("scales_l", "uint8", 4, 4, "packed low four bits for eight local scales"),
    field("qs", "uint8", 128, 128, "256 four-bit codebook indices")
  ],
  IQ1_M: [
    field("qs", "uint8", 32, 32, "low grid-index bits"),
    field("qh", "uint8", 16, 16, "high index and grid-shift bits"),
    field("scales", "uint8", 8, 8, "embedded global FP16 bits plus sixteen local 3-bit scales")
  ],
  TQ1_0: [
    field("qs", "uint8", 48, 48, "240 values, five base-3 symbols per byte"),
    field("qh", "uint8", 4, 4, "16 remaining symbols, four per byte"),
    field("d", "FP16", 1, 2, "one scale for 256 weights")
  ],
  TQ2_0: [
    field("qs", "uint8", 64, 64, "256 two-bit ternary symbols"),
    field("d", "FP16", 1, 2, "one scale for 256 weights")
  ],
  MXFP4: [
    field("e", "E8M0", 1, 1, "one power-of-two scale for 32 weights"),
    field("qs", "uint8", 16, 16, "32 packed E2M1 FP4 values")
  ],
  NVFP4: [
    field("d", "UE4M3", 4, 4, "one scale per 16-weight group"),
    field("qs", "uint8", 32, 32, "64 packed E2M1 FP4 values")
  ],
  Q1_0: [
    field("d", "FP16", 1, 2, "mean-absolute scale for 128 weights"),
    field("qs", "uint8", 16, 16, "128 sign bits")
  ],
  Q2_0: [
    field("d", "FP16", 1, 2, "max-absolute scale for 64 weights"),
    field("qs", "uint8", 16, 16, "64 two-bit codes")
  ]
};

export function ggufStorageLayout(
  dtype: string
): readonly GgufStorageField[] | undefined {
  return LAYOUTS[dtype.toUpperCase()];
}

function field(
  name: string,
  type: string,
  count: number,
  bytes: number,
  role: string
): GgufStorageField {
  return { name, type, count, bytes, role };
}
