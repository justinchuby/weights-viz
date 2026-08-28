export interface GgufQuantContract {
  dtype: string;
  family: string;
  values: number;
  bytes: number;
  groups: {
    count: number;
    values: number;
    label: string;
  };
  metadata: readonly string[];
  codes: readonly string[];
  packing: readonly string[];
  decode: string;
  runtime: string;
}

const CONTRACTS: Record<string, GgufQuantContract> = {
  Q4_1: contract(
    "Q4_1",
    "Affine 4-bit block",
    32,
    20,
    1,
    32,
    "one affine range",
    ["m = min(block)", "d = (max(block) − m) / 15", "d and m are rounded to FP16"],
    ["q ∈ [0, 15]", "q = clamp(round((w − m) / d), 0, 15)", "no implicit signed bias"],
    ["qs[j].low → q[j]", "qs[j].high → q[j + 16]", "16 split-half nibble bytes"],
    "w′ = d × q + m",
    "The minimum is added during the fused dot product; it is not a stored zero-point per weight."
  ),
  Q5_0: contract(
    "Q5_0",
    "Symmetric 5-bit block",
    32,
    22,
    1,
    32,
    "one shared signed range",
    ["a = signed max-absolute element", "d = a / −16", "one FP16 d serves all 32 weights"],
    ["conceptual q ∈ [−16, 15]", "stored = q + 16 ∈ [0, 31]", "the +16 bias consumes no bytes"],
    ["qs carries the low four bits", "qh bit i carries code i’s fifth bit", "qs pairs i and i + 16 as low/high nibbles"],
    "storedCode = join(qh, qs); w′ = d × (storedCode − 16)",
    "The kernel joins the split bit-plane in registers before multiplying by d."
  ),
  Q5_1: contract(
    "Q5_1",
    "Affine 5-bit block",
    32,
    24,
    1,
    32,
    "one affine range",
    ["m = min(block)", "d = (max(block) − m) / 31", "d and m are rounded to FP16"],
    ["q ∈ [0, 31]", "q = clamp(round((w − m) / d), 0, 31)", "no signed bias"],
    ["qs carries four low bits", "qh bit i carries code i’s fifth bit", "qs pairs i and i + 16 as low/high nibbles"],
    "w′ = d × join(qh, qs) + m",
    "The affine minimum correction can pair with the precomputed sum in Q8_1 dot-product blocks."
  ),
  Q8_0: contract(
    "Q8_0",
    "Symmetric int8 block",
    32,
    34,
    1,
    32,
    "one shared signed range",
    ["d = max(|w|) / 127", "d is stored as FP16", "one d serves all 32 weights"],
    ["q = round(w / d)", "q ∈ [−127, 127]", "−128 is deliberately unused"],
    ["qs[i] is one signed int8 code", "codes remain in logical order", "no bias and no split bit-plane"],
    "w′ = d × int8(q)",
    "Q8_0 is also a common runtime activation partner for older Q4/Q5 dot-product kernels."
  ),
  Q8_1: contract(
    "Q8_1",
    "Dot-product companion block",
    32,
    36,
    1,
    32,
    "one activation block",
    ["encoder d = max(|w|) / 127", "s = encoder_d × Σ q[i]", "d and s are rounded to FP16 independently"],
    ["q = round(w / d)", "q ∈ [−127, 127]", "s is auxiliary metadata, not a minimum"],
    ["d and s share the leading 4-byte FP16 pair", "qs stores 32 signed int8 codes", "codes remain in logical order"],
    "w′ = stored_d × q; stored_s ≈ encoder_d × Σq",
    "Paired affine kernels reuse the independently rounded s for offset correction instead of summing all 32 activation codes again."
  ),
  Q8_K: contract(
    "Q8_K",
    "Dot-product companion block",
    256,
    292,
    16,
    16,
    "16 sum groups",
    ["d = signed max-absolute element / −127", "d is FP32, not FP16", "bsums[g] = Σ q for the 16 codes in group g"],
    ["q = round(w / d)", "q is a signed int8 code", "16 int16 sums are auxiliary"],
    ["d first, then qs[256], then bsums[16]", "each bsums entry covers exactly 16 consecutive codes", "no code bias or bit-plane split"],
    "w′[i] = d × q[i]",
    "K-quant matrix kernels consume bsums for affine correction; ordinary dequantization does not need them."
  ),
  IQ2_XXS: contract(
    "IQ2_XXS",
    "2-bit importance grid",
    256,
    66,
    8,
    32,
    "8 local-scale groups",
    ["one FP16 global d", "each 32-weight group embeds one 4-bit scale s", "local factor = d × (0.5 + s) × 0.25"],
    ["four 8-weight grids per group", "8-bit index selects one of 256 grids", "each grid gets a 7-bit sign pattern"],
    ["four grid-index bytes occupy the first uint32", "four 7-bit sign indices occupy bits 0…27 of the second", "the group’s 4-bit scale occupies bits 28…31"],
    "w′ = d × (0.5 + s) × 0.25 × grid[index][lane] × sign",
    "The grid contains multi-weight magnitudes; it is not a scalar two-bit lookup."
  ),
  IQ2_XS: contract(
    "IQ2_XS",
    "2-bit importance grid",
    256,
    74,
    16,
    16,
    "16 local-scale groups",
    ["one FP16 global d", "each 4-bit scale s serves 16 weights", "local factor = d × (0.5 + s) × 0.25"],
    ["one 9-bit index selects an 8-weight grid", "the upper 7 bits select its sign pattern", "512 possible grids"],
    ["each qs uint16 stores 9 index bits + 7 sign bits", "each scales byte stores two 4-bit local scales", "four grid words reconstruct each 32-weight region"],
    "w′ = d × (0.5 + s) × 0.25 × grid[index][lane] × sign",
    "The runtime expands a whole 8-value grid per index and applies signs while accumulating."
  ),
  IQ3_XXS: contract(
    "IQ3_XXS",
    "3-bit importance grid",
    256,
    98,
    8,
    32,
    "8 local-scale groups",
    ["one FP16 global d", "each 32-weight group embeds one 4-bit scale s", "local factor = d × (0.5 + s) × 0.5"],
    ["two 8-bit indices reconstruct each 8 weights", "each index selects a four-value grid from 256 entries", "one 7-bit sign pattern expands to eight signs"],
    ["qs[0…63] stores grid indices", "qs[64…95] stores four sign indices plus one scale nibble per group", "scale occupies bits 28…31 of each 32-bit metadata word"],
    "w′ = d × (0.5 + s) × 0.5 × grid[index][lane] × sign",
    "Two four-value lookups and one expanded sign mask produce each eight-weight vector."
  ),
  IQ1_S: contract(
    "IQ1_S",
    "1-bit importance grid",
    256,
    50,
    8,
    32,
    "8 scale/delta groups",
    ["one FP16 global d", "qh bits 12…14 store a 3-bit scale s", "local factor = d × (2s + 1); bit 15 chooses δ = ±0.125"],
    ["four 11-bit indices per 32 weights", "each index selects an 8-value signed ternary grid", "there is no separate sign mask"],
    ["qs stores the low eight index bits", "qh packs four 3-bit index highs, scale, and delta sign", "one uint16 qh word describes 32 weights"],
    "w′ = d × (2s + 1) × (signedGrid[index][lane] + δ)",
    "The ±⅛ delta shifts the signed {-1,0,+1} grid so one-bit-class storage can express more useful levels."
  ),
  IQ3_S: contract(
    "IQ3_S",
    "3-bit importance grid",
    256,
    110,
    8,
    32,
    "8 local-scale groups",
    ["one FP16 global d", "each 4-bit s serves 32 weights", "local factor = d × (1 + 2s)"],
    ["two 9-bit indices reconstruct each 8 weights", "each index selects a four-value grid from 512 entries", "one explicit sign byte serves eight weights"],
    ["qs stores the low 8 bits of each 9-bit grid index", "qh supplies one high bit per grid index", "signs stores masks; scales packs two nibbles per byte"],
    "w′ = d × (1 + 2s) × grid[index][lane] × sign",
    "The odd-multiple local scale and sign byte are applied as the two four-value grids are expanded."
  ),
  IQ2_S: contract(
    "IQ2_S",
    "2-bit importance grid",
    256,
    82,
    16,
    16,
    "16 local-scale groups",
    ["one FP16 global d", "each 4-bit s serves 16 weights", "local factor = d × (0.5 + s) × 0.25"],
    ["one 10-bit index selects an 8-value grid", "1024 possible grids", "one explicit sign byte serves eight weights"],
    ["qs[0…31] stores low index bytes", "qs[32…63] stores sign masks", "qh supplies two high index bits; scales packs two nibbles per byte"],
    "w′ = d × (0.5 + s) × 0.25 × grid[index][lane] × sign",
    "The physical qs array deliberately contains two logical regions: indices first, then signs."
  ),
  IQ4_NL: contract(
    "IQ4_NL",
    "Nonlinear 4-bit codebook",
    32,
    18,
    1,
    32,
    "one shared nonlinear range",
    ["one FP16 d serves 32 weights", "conversion searches d and nearest fixed levels", "the 16-level table is part of the runtime ABI"],
    ["a nibble selects one signed nonlinear level", "levels = −127, −104, −83, −65, −49, −35, −22, −10, 1, 13, 25, 38, 53, 69, 89, 113", "no sign mask and no zero-point"],
    ["qs[j].low → index j", "qs[j].high → index j + 16", "16 split-half nibble bytes"],
    "w′ = d × nonlinearLevel[nibble]",
    "The nonuniform table spends more resolution near zero while retaining large-magnitude endpoints."
  ),
  IQ4_XS: contract(
    "IQ4_XS",
    "Hierarchical nonlinear 4-bit codebook",
    256,
    136,
    8,
    32,
    "8 signed-scale groups",
    ["one FP16 global d", "each group stores a signed 6-bit local scale ls − 32", "four low bits live in scales_l; two high bits live in scales_h"],
    ["each nibble selects the IQ4_NL signed level table", "local scale code ls ∈ [0, 63]", "there is no separate sign mask"],
    ["scales_h contains eight 2-bit scale highs", "scales_l packs eight 4-bit scale lows", "qs pairs lanes i and i + 16 within each 32-weight group"],
    "w′ = d × (ls − 32) × nonlinearLevel[nibble]",
    "The signed local scale can flip the fixed signed codebook while adapting each 32-weight group."
  ),
  IQ1_M: contract(
    "IQ1_M",
    "Embedded-scale 1-bit importance grid",
    256,
    56,
    16,
    16,
    "16 local-scale groups",
    ["no standalone d field exists", "the FP16 global d is reconstructed from four high nibbles inside scales[8]", "3-bit s gives local factor d × (2s + 1)"],
    ["11-bit indices select signed 8-value ternary grids", "qh also carries the ±0.125 delta sign", "there is no separate sign mask"],
    ["qs stores low eight index bits", "qh packs index highs and delta/shift bits", "scales packs the global FP16 as the top nibble of four uint16 views; each lower 12-bit region holds four local 3-bit scales"],
    "w′ = embedded_d × (2s + 1) × (signedGrid[index][lane] + δ)",
    "This format saves bytes by weaving the global FP16 scale through otherwise spare bits of the local-scale array."
  ),
  TQ1_0: contract(
    "TQ1_0",
    "Base-3 ternary block",
    256,
    54,
    1,
    256,
    "one ternary range",
    ["d = max(|w|)", "one FP16 d serves all 256 weights", "weights quantize to −1, 0, or +1"],
    ["trit t = q + 1 ∈ {0,1,2}", "five trits fit in one base-3 byte because 3⁵ = 243", "no fourth unused code per value"],
    ["qs[48] carries 240 weights at five trits per byte", "qh[4] carries the final 16 at four trits per byte", "d is the trailing FP16 field"],
    "w′ = d × (base3Digit − 1)",
    "The decoder extracts trits with integer multiply/shift operations rather than division."
  ),
  TQ2_0: contract(
    "TQ2_0",
    "Two-bit ternary block",
    256,
    66,
    1,
    256,
    "one ternary range",
    ["d = max(|w|)", "one FP16 d serves all 256 weights", "weights quantize to −1, 0, or +1"],
    ["stored code = q + 1 ∈ {0,1,2}", "binary 11 is unused", "four two-bit codes fit in each byte"],
    ["qs uses four interleaved 2-bit planes", "shifts 0, 2, 4, and 6 select each plane", "d is the trailing FP16 field"],
    "w′ = d × (twoBitCode − 1)",
    "TQ2_0 uses about 22% more storage than TQ1_0 (66 versus 54 bytes) in exchange for simpler masks and shifts."
  ),
  MXFP4: contract(
    "MXFP4",
    "OCP microscaled FP4 block",
    32,
    17,
    1,
    32,
    "one shared exponent",
    ["e is one unsigned E8M0 exponent", "conversion chooses e from the block maximum", "GGML’s E8M0-half helper includes a special denormal path for e = 0"],
    ["each nibble is one signed E2M1 FP4 code", "magnitudes represent 0, 0.5, 1, 1.5, 2, 3, 4, 6", "bit 3 is the sign"],
    ["e is the first byte", "qs[j].low → lane j", "qs[j].high → lane j + 16"],
    "w′ = E8M0_HALF(e) × doubledE2M1[nibble]",
    "GGML stores a doubled integer E2M1 table and halves the shared scale, yielding the standard FP4 values."
  ),
  NVFP4: contract(
    "NVFP4",
    "Locally scaled FP4 block",
    64,
    36,
    4,
    16,
    "4 local-scale groups",
    ["each 16-weight group gets one unsigned UE4M3 scale", "scale code ≈ UE4M3(max(|w|) / 6)", "there is no global block scale"],
    ["each nibble is one signed E2M1 FP4 code", "magnitudes represent 0, 0.5, 1, 1.5, 2, 3, 4, 6", "bit 3 is the sign"],
    ["d[0…3] stores four UE4M3 scale bytes", "each scale maps to one consecutive 16-weight group", "qs packs split-half nibbles inside each group"],
    "w′ = UE4M3(d[group]) × doubledE2M1[nibble]",
    "The UE4M3 helper returns the compensating half-scale used with GGML’s doubled E2M1 lookup table."
  ),
  Q1_0: contract(
    "Q1_0",
    "Binary sign block",
    128,
    18,
    1,
    128,
    "one magnitude",
    ["d = mean(|w|)", "one positive FP16 d serves 128 weights", "no minimum or zero-point"],
    ["bit i = 1 when w[i] ≥ 0", "bit i = 0 when w[i] < 0", "the code stores sign only"],
    ["qs contains 16 bytes", "bits are packed LSB-first within each byte", "eight consecutive weights share one byte"],
    "w′ = signBit ? +d : −d",
    "Choosing d = mean(|weight|) minimizes squared error for a fixed ±d reconstruction."
  ),
  Q2_0: contract(
    "Q2_0",
    "Biased two-bit block",
    64,
    18,
    1,
    64,
    "one four-level range",
    ["d = max(|w|)", "one FP16 d serves 64 weights", "no stored minimum"],
    ["q = clamp(round(w / d), −1, 2)", "stored = q + 1 ∈ [0,3]", "levels are −d, 0, +d, +2d"],
    ["four consecutive codes share one byte", "code i uses bits 2(i mod 4)…2(i mod 4)+1", "the +1 storage bias consumes no bytes"],
    "w′ = d × (twoBitCode − 1)",
    "The implicit +1 storage bias creates an asymmetric four-level alphabet even though d comes from max absolute magnitude."
  )
};

export function ggufQuantContract(
  dtype: string
): GgufQuantContract | undefined {
  return CONTRACTS[dtype.toUpperCase()];
}

export const GGUF_QUANT_CONTRACT_DTYPES = Object.freeze(
  Object.keys(CONTRACTS)
);

function contract(
  dtype: string,
  family: string,
  values: number,
  bytes: number,
  groupCount: number,
  groupValues: number,
  groupLabel: string,
  metadata: readonly string[],
  codes: readonly string[],
  packing: readonly string[],
  decode: string,
  runtime: string
): GgufQuantContract {
  return {
    dtype,
    family,
    values,
    bytes,
    groups: {
      count: groupCount,
      values: groupValues,
      label: groupLabel
    },
    metadata,
    codes,
    packing,
    decode,
    runtime
  };
}
