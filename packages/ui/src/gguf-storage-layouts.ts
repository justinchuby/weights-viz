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
    field("s", "FP16", 1, 2, "d × sum(q), for dot products"),
    field("qs", "int8", 32, 32, "one signed code per weight")
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
      "packed grid/sign indices with embedded 4-bit local scales"
    )
  ],
  IQ2_XS: [
    field("d", "FP16", 1, 2, "one super-block scale"),
    field("qs", "uint16", 32, 64, "grid, sign, and index payload"),
    field("scales", "uint8", 8, 8, "one local scale byte per 32 weights")
  ],
  IQ3_XXS: [
    field("d", "FP16", 1, 2, "one scale for 256 weights"),
    field(
      "qs",
      "uint8",
      96,
      96,
      "packed grid/sign indices with embedded 4-bit local scales"
    )
  ],
  IQ1_S: [
    field("d", "FP16", 1, 2, "one scale for 256 weights"),
    field("qs", "uint8", 32, 32, "low grid-index bits"),
    field("qh", "uint16", 8, 16, "high index and delta bits")
  ],
  IQ3_S: [
    field("d", "FP16", 1, 2, "one super-block scale"),
    field("qs", "uint8", 64, 64, "low grid-index bits"),
    field("qh", "uint8", 8, 8, "high grid-index bits"),
    field("signs", "uint8", 32, 32, "packed sign bits"),
    field("scales", "uint8", 4, 4, "packed local scales")
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
    field("scales_h", "uint16", 1, 2, "high bits of local scales"),
    field("scales_l", "uint8", 4, 4, "low bits of local scales"),
    field("qs", "uint8", 128, 128, "256 four-bit codebook indices")
  ],
  IQ1_M: [
    field("qs", "uint8", 32, 32, "low grid-index bits"),
    field("qh", "uint8", 16, 16, "high index and grid-shift bits"),
    field("scales", "uint8", 8, 8, "packed FP16/global and local scale bits")
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
