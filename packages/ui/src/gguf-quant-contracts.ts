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
  symbols: readonly GgufQuantSymbol[];
  runtime: string;
  worked: GgufQuantWorkedExample;
}

export interface GgufQuantSymbol {
  symbol: string;
  meaning: string;
  source: string;
}

export type GgufQuantWorkedStageKind =
  | "source"
  | "metadata"
  | "storage"
  | "reconstruction";

interface GgufQuantWorkedStageBase {
  kind: GgufQuantWorkedStageKind;
  title: string;
  detail: string;
  symbols: readonly string[];
}

export interface GgufQuantWorkedStorageAccess {
  field: string;
  index: string;
  bits: string;
  action: string;
}

export type GgufQuantWorkedStage =
  | (GgufQuantWorkedStageBase & {
      kind: "source" | "metadata" | "reconstruction";
    })
  | (GgufQuantWorkedStageBase & {
      kind: "storage";
      accesses: readonly [
        GgufQuantWorkedStorageAccess,
        ...GgufQuantWorkedStorageAccess[]
      ];
    });

export interface GgufQuantWorkedExample {
  selection: {
    group: number;
    position: number;
    weight: number;
  };
  stages: readonly [
    GgufQuantWorkedStage & { kind: "source" },
    GgufQuantWorkedStage & { kind: "metadata" },
    GgufQuantWorkedStage & { kind: "storage" },
    GgufQuantWorkedStage & { kind: "reconstruction" }
  ];
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
    ["group 0 is the entire 32-weight record; all positions 0…31 share its FP16 d", "conversion searches d and nearest fixed levels", "the 16-level table is part of the runtime ABI"],
    ["position = i ∈ [0, 31] is the weight’s offset inside group 0", "a nibble selects one scalar signed nonlinear level; IQ4_NL has no multi-value grid and therefore no grid lane", "levels = −127, −104, −83, −65, −49, −35, −22, −10, 1, 13, 25, 38, 53, 69, 89, 113", "no sign mask and no zero-point"],
    ["position 0…15 → low nibble of qs[position]", "position 16…31 → high nibble of qs[position − 16]", "16 split-half nibble bytes store all 32 positions"],
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
    ["each nibble selects one scalar from the IQ4_NL signed level table; there is no grid lane", "local scale code ls ∈ [0, 63]", "there is no separate sign mask"],
    ["scales_h contains eight 2-bit scale highs", "scales_l packs eight 4-bit scale lows", "qs pairs positions i and i + 16 within each 32-weight group"],
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
    ["e is the first byte", "qs[j].low → position j", "qs[j].high → position j + 16"],
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
    symbols: contractSymbols(dtype, values, groupCount, groupValues),
    runtime,
    worked: workedExample(dtype)
  };
}

function workedExample(dtype: string): GgufQuantWorkedExample {
  switch (dtype) {
    case "Q4_1":
      return worked(
        0,
        19,
        19,
        "Select source weight 19 in the only 32-weight affine group.",
        "Use the group’s FP16 d and m; quantization selects unsigned q for position 19.",
        [
          access("d", "0", "all 16 bits", "read the affine step"),
          access("m", "0", "all 16 bits", "read the shared minimum"),
          access("qs", "3", "bits 4…7", "extract position 19’s high nibble as q")
        ],
        "Reconstruct position 19 as w′ = d × q + m.",
        ["w′"],
        ["d", "m", "q"],
        ["d", "m", "q"],
        ["w′", "d", "q", "m"]
      );
    case "Q5_0":
      return worked(
        0,
        18,
        18,
        "Select source weight 18 in the only 32-weight signed group.",
        "Use FP16 d and select position 18’s biased five-bit storedCode.",
        [
          access("d", "0", "all 16 bits", "read the shared signed scale"),
          access("qs", "2", "bits 4…7", "take storedCode bits 0…3"),
          access("qh", "little-endian uint32 view of bytes 0…3", "bit 18", "take position 18’s storedCode bit 4")
        ],
        "Join the qh bit above the qs nibble, then compute w′ = d × (storedCode − 16).",
        ["w′"],
        ["d", "storedCode"],
        ["d", "storedCode", "join(qh, qs)", "qh", "qs"],
        ["w′", "d", "storedCode", "16"]
      );
    case "Q5_1":
      return worked(
        0,
        7,
        7,
        "Select source weight 7 in the only 32-weight affine group.",
        "Use FP16 d and m; quantization selects position 7’s unsigned five-bit code.",
        [
          access("d", "0", "all 16 bits", "read the affine step"),
          access("m", "0", "all 16 bits", "read the shared minimum"),
          access("qs", "7", "bits 0…3", "take code bits 0…3"),
          access("qh", "little-endian uint32 view of bytes 0…3", "bit 7", "take code bit 4")
        ],
        "Join the qh bit and qs nibble, then compute w′ = d × join(qh, qs) + m.",
        ["w′"],
        ["d", "m", "join(qh, qs)"],
        ["d", "m", "join(qh, qs)"],
        ["w′", "d", "join(qh, qs)", "m"]
      );
    case "Q8_0":
      return worked(
        0,
        11,
        11,
        "Select source weight 11 in the only 32-weight signed group.",
        "Use the group’s FP16 d and select signed int8 q for position 11.",
        [
          access("d", "0", "all 16 bits", "read the shared scale"),
          access("qs", "11", "all 8 bits", "read q using the declared int8 storage type")
        ],
        "Reconstruct position 11 as w′ = d × int8(q).",
        ["w′"],
        ["d", "q"],
        ["d", "q", "int8(…)"],
        ["w′", "d", "q", "int8(…)"]
      );
    case "Q8_1":
      return worked(
        0,
        13,
        13,
        "Select source weight 13 in the only 32-weight activation group.",
        "Select signed q for position 13; stored_d decodes it while stored_s carries the independently rounded encoder_d × Σq correction.",
        [
          access("d", "0", "all 16 bits", "read stored_d"),
          access("s", "0", "all 16 bits", "read stored_s for the fused correction"),
          access("qs", "13", "all 8 bits", "read position 13’s signed q")
        ],
        "Reconstruct position 13 as w′ = stored_d × q; a paired affine kernel consumes stored_s without recomputing Σq.",
        ["w′"],
        ["q", "stored_d", "stored_s", "encoder_d", "Σq"],
        ["q", "stored_d", "stored_s"],
        ["w′", "stored_d", "q", "stored_s", "Σq"]
      );
    case "Q8_K":
      return worked(
        6,
        7,
        103,
        "Select weight i = 103: position 7 of 16-weight sum group 6.",
        "Use FP32 d and signed q[i]; group 6 also selects the auxiliary sum used by affine dot products.",
        [
          access("d", "0", "all 32 bits", "read the block scale"),
          access("qs", "103", "all 8 bits", "read q[i] as signed int8"),
          access("bsums", "6", "all 16 bits", "read the sum for codes 96…111")
        ],
        "Reconstruct this position as w′[i] = d × q[i]; bsums[6] remains an auxiliary dot-product input.",
        ["w′[i]", "i"],
        ["d", "q[i]", "i"],
        ["d", "q[i]", "i"],
        ["w′[i]", "d", "q[i]", "i"]
      );
    case "IQ2_XXS":
      return worked(
        3,
        21,
        117,
        "Select weight 117: position 21 in group 3 gives subgrid = floor(21 / 8) = 2 and grid lane = 21 mod 8 = 5.",
        "Select group 3’s scale s, subgrid 2’s index and sign pattern, then lane 5 of the compiled grid.",
        [
          access("d", "0", "all 16 bits", "read the global scale"),
          access("qs", "byte 26 (uint16[13] low byte)", "bits 0…7", "read subgrid 2’s index"),
          access("qs", "group 3 metadata uint32", "bits 14…20", "read subgrid 2’s seven-bit sign index"),
          access("qs", "group 3 metadata uint32", "bits 28…31", "read s")
        ],
        "Expand grid[index], select lane 5 and its sign, then compute w′ = d × (0.5 + s) × 0.25 × grid[index][lane] × sign.",
        ["w′", "lane"],
        ["d", "s", "index", "grid", "lane", "sign"],
        ["d", "s", "index", "sign"],
        ["w′", "d", "s", "grid", "index", "lane", "sign", "0.5", "0.25"]
      );
    case "IQ2_XS":
      return worked(
        5,
        9,
        89,
        "Select weight 89: position 9 in group 5 gives its second grid word and grid lane = 9 mod 8 = 1.",
        "Select group 5’s scale s plus grid word 11’s index and sign pattern; the codebook entry remains symbolic.",
        [
          access("d", "0", "all 16 bits", "read the global scale"),
          access("scales", "2", "bits 4…7", "read group 5’s s"),
          access("qs", "11", "bits 0…8", "read the nine-bit index"),
          access("qs", "11", "bits 9…15", "read the seven-bit sign index")
        ],
        "Expand grid[index], select lane 1 and its sign, then compute w′ = d × (0.5 + s) × 0.25 × grid[index][lane] × sign.",
        ["w′", "lane"],
        ["d", "s", "index", "grid", "lane", "sign"],
        ["d", "s", "index", "sign"],
        ["w′", "d", "s", "grid", "index", "lane", "sign", "0.5", "0.25"]
      );
    case "IQ3_XXS":
      return worked(
        4,
        11,
        139,
        "Select weight 139: position 11 in group 4 gives eight-weight subvector 1 and four-value grid lane = 11 mod 4 = 3.",
        "Select group 4’s s and sign pattern, then index byte 34 for the first four-value half of that subvector.",
        [
          access("d", "0", "all 16 bits", "read the global scale"),
          access("qs", "34", "bits 0…7", "read the selected four-value grid index"),
          access("qs", "group 4 metadata uint32 at bytes 80…83", "bits 7…13", "read subvector 1’s sign index"),
          access("qs", "group 4 metadata uint32 at bytes 80…83", "bits 28…31", "read s")
        ],
        "Expand grid[index], take lane 3 and its sign, then compute w′ = d × (0.5 + s) × 0.5 × grid[index][lane] × sign.",
        ["w′", "lane"],
        ["d", "s", "index", "grid", "lane", "sign"],
        ["d", "s", "index", "sign"],
        ["w′", "d", "s", "grid", "index", "lane", "sign", "0.5"]
      );
    case "IQ1_S":
      return worked(
        5,
        22,
        182,
        "Select i = 182: position 22 in group 5 gives subgrid = floor(22 / 8) = 2 and grid lane = 22 mod 8 = 6.",
        "Select index, s, and δ for group 5; signedGrid[index] is the compiled eight-value entry.",
        [
          access("d", "0", "all 16 bits", "read the global scale"),
          access("qs", "22", "bits 0…7", "read the low eight index bits"),
          access("qh", "5", "bits 6…8", "read subgrid 2’s three index highs"),
          access("qh", "5", "bits 12…14", "read s"),
          access("qh", "5", "bit 15", "select the sign of δ")
        ],
        "Join index, take signedGrid[index][lane 6], add δ, and compute w′ = d × (2s + 1) × (signedGrid[index][lane] + δ).",
        ["w′", "i", "group", "subgrid", "lane"],
        ["index", "s", "δ", "signedGrid"],
        ["d", "index", "s", "δ"],
        ["w′", "d", "2s + 1", "signedGrid", "index", "lane", "δ"]
      );
    case "IQ3_S":
      return worked(
        2,
        30,
        94,
        "Select weight 94: position 30 in group 2 gives subvector 3, sign-mask position 6, and four-value grid lane = 30 mod 4 = 2.",
        "Select group 2’s s, grid index 23 for the second four-position half, and sign-mask bit 6 from signs byte 11.",
        [
          access("d", "0", "all 16 bits", "read the global scale"),
          access("scales", "1", "bits 0…3", "read group 2’s s"),
          access("qs", "23", "bits 0…7", "read the low eight index bits"),
          access("qh", "2", "bit 7", "read index 23’s high bit"),
          access("signs", "11", "bit 6", "read sign-mask position 6")
        ],
        "Join index, take grid[index][lane 2] and sign-mask bit 6, then compute w′ = d × (1 + 2s) × grid[index][lane] × sign.",
        ["w′", "lane"],
        ["d", "s", "index", "grid", "lane", "sign"],
        ["d", "s", "index", "sign"],
        ["w′", "d", "1", "2", "s", "grid", "index", "lane", "sign"]
      );
    case "IQ2_S":
      return worked(
        9,
        10,
        154,
        "Select weight 154: position 10 in group 9 gives its second grid word (global grid 19) and grid lane = 10 mod 8 = 2.",
        "Select group 9’s s, grid 19’s ten-bit index, and its explicit sign mask; the grid value stays symbolic.",
        [
          access("d", "0", "all 16 bits", "read the global scale"),
          access("scales", "4", "bits 4…7", "read group 9’s s"),
          access("qs", "19", "bits 0…7", "read the low index byte"),
          access("qh", "4", "bits 6…7", "read grid 19’s two index highs"),
          access("qs", "51", "bit 2", "read lane 2’s sign from the sign-mask region")
        ],
        "Join index, select grid[index][lane 2] and sign, then compute w′ = d × (0.5 + s) × 0.25 × grid[index][lane] × sign.",
        ["w′", "lane"],
        ["d", "s", "index", "grid", "lane", "sign"],
        ["d", "s", "index", "sign"],
        ["w′", "d", "s", "grid", "index", "lane", "sign", "0.5", "0.25"]
      );
    case "IQ4_NL":
      return worked(
        0,
        25,
        25,
        "There is one group: group 0 = weights 0…31. Select position 25, so i = group × 32 + position = 0 × 32 + 25 = weight 25.",
        "Use group 0’s shared FP16 d. Position 25 is in the second half, so its code is the high nibble of qs[25 − 16] = qs[9].",
        [
          access("d", "0", "all 16 bits", "read the shared scale"),
          access("qs", "9", "bits 4…7", "read position 25’s nibble")
        ],
        "Use nibble as the compiled nonlinearLevel index, then compute w′ = d × nonlinearLevel[nibble].",
        ["w′"],
        ["d", "nibble", "nonlinearLevel"],
        ["d", "nibble"],
        ["w′", "d", "nonlinearLevel", "nibble"]
      );
    case "IQ4_XS":
      return worked(
        6,
        27,
        219,
        "Select weight 219: position 27 in 32-weight signed-scale group 6.",
        "Select group 6’s six-bit ls and position 27’s nonlinear table nibble.",
        [
          access("d", "0", "all 16 bits", "read the global scale"),
          access("scales_l", "3", "bits 0…3", "read ls bits 0…3 for group 6"),
          access("scales_h", "0", "bits 12…13", "read ls bits 4…5 for group 6"),
          access("qs", "107", "bits 4…7", "read position 27’s nibble")
        ],
        "Join ls, subtract its fixed bias, look up nonlinearLevel[nibble], and compute w′ = d × (ls − 32) × nonlinearLevel[nibble].",
        ["w′"],
        ["d", "ls", "nibble", "nonlinearLevel"],
        ["d", "ls", "nibble"],
        ["w′", "d", "ls", "32", "nonlinearLevel", "nibble"]
      );
    case "IQ1_M":
      return worked(
        10,
        5,
        165,
        "Select weight 165: position 5 in group 10 gives global eight-value vector 20 and grid lane = 5 mod 8 = 5.",
        "Select vector 20’s index and δ plus group 10’s s; embedded_d is assembled once for the block.",
        [
          access("scales", "uint16 views 0…3", "bits 12…15 of each", "join the four nibbles into embedded_d"),
          access("scales", "uint16 view 2", "bits 6…8", "read group 10’s s"),
          access("qs", "20", "bits 0…7", "read the low index bits"),
          access("qh", "10", "bits 0…2", "read vector 20’s index highs"),
          access("qh", "10", "bit 3", "select δ")
        ],
        "Join index, select signedGrid[index][lane 5] and δ, then compute w′ = embedded_d × (2s + 1) × (signedGrid[index][lane] + δ).",
        ["w′", "lane"],
        ["embedded_d", "s", "index", "signedGrid", "δ"],
        ["embedded_d", "s", "index", "δ"],
        ["w′", "embedded_d", "2s + 1", "signedGrid", "index", "lane", "δ"]
      );
    case "TQ1_0":
      return worked(
        0,
        137,
        137,
        "Select source weight 137 in the single 256-weight ternary range.",
        "Quantization selects base3Digit 0, 1, or 2 for position 137.",
        [
          access("d", "0", "all 16 bits", "read the trailing shared magnitude"),
          access("qs", "9", "interleaved trit position 4", "compute q = uint8(qs[9] × 3⁴), then base3Digit = (uint16(q) × 3) >> 8")
        ],
        "Remove the ternary storage bias and reconstruct w′ = d × (base3Digit − 1).",
        ["w′"],
        ["base3Digit"],
        ["d", "base3Digit"],
        ["w′", "d", "base3Digit", "1"]
      );
    case "TQ2_0":
      return worked(
        0,
        137,
        137,
        "Select source weight 137 in the single 256-weight ternary range.",
        "Quantization selects twoBitCode 0, 1, or 2; binary 11 remains unused.",
        [
          access("d", "0", "all 16 bits", "read the trailing shared magnitude"),
          access("qs", "41", "bits 0…1", "extract position 137 from the second 32-byte interleaved plane")
        ],
        "Remove the ternary storage bias and reconstruct w′ = d × (twoBitCode − 1).",
        ["w′"],
        ["twoBitCode"],
        ["d", "twoBitCode"],
        ["w′", "d", "twoBitCode", "1"]
      );
    case "MXFP4":
      return worked(
        0,
        21,
        21,
        "Select source weight 21 in the only 32-weight microscale group.",
        "Select shared exponent e and position 21’s signed E2M1 nibble.",
        [
          access("e", "0", "all 8 bits", "read the E8M0 exponent code"),
          access("qs", "5", "bits 4…7", "read position 21’s nibble")
        ],
        "Decode E8M0_HALF(e), look up doubledE2M1[nibble], and compute w′ = E8M0_HALF(e) × doubledE2M1[nibble].",
        ["w′"],
        ["e", "nibble"],
        ["e", "nibble"],
        ["w′", "E8M0_HALF(e)", "e", "doubledE2M1", "nibble"]
      );
    case "NVFP4":
      return worked(
        2,
        11,
        43,
        "Select weight 43: position 11 in 16-weight local-scale group 2.",
        "Select d[group] and position 11’s signed E2M1 nibble.",
        [
          access("d", "2", "all 8 bits", "read group 2’s UE4M3 scale code"),
          access("qs", "19", "bits 4…7", "read group-local position 11’s nibble")
        ],
        "Decode UE4M3(d[group]), look up doubledE2M1[nibble], and compute w′ = UE4M3(d[group]) × doubledE2M1[nibble].",
        ["w′", "group"],
        ["d[group]", "group", "nibble"],
        ["d[group]", "group", "nibble"],
        ["w′", "UE4M3(…)", "d[group]", "group", "doubledE2M1", "nibble"]
      );
    case "Q1_0":
      return worked(
        0,
        77,
        77,
        "Select source weight 77 in the only 128-weight magnitude group.",
        "Use FP16 d and select signBit for position 77.",
        [
          access("d", "0", "all 16 bits", "read the shared positive magnitude"),
          access("qs", "9", "bit 5", "read position 77’s LSB-first signBit")
        ],
        "Select the +d / −d level directly from signBit to produce w′.",
        ["w′"],
        ["d", "signBit"],
        ["d", "signBit"],
        ["w′", "signBit", "+d / −d", "d"]
      );
    case "Q2_0":
      return worked(
        0,
        46,
        46,
        "Select source weight 46 in the only 64-weight four-level group.",
        "Use FP16 d and select position 46’s biased twoBitCode.",
        [
          access("d", "0", "all 16 bits", "read the shared magnitude"),
          access("qs", "11", "bits 4…5", "read position 46’s twoBitCode")
        ],
        "Remove the implicit bias and reconstruct w′ = d × (twoBitCode − 1).",
        ["w′"],
        ["d", "twoBitCode"],
        ["d", "twoBitCode"],
        ["w′", "d", "twoBitCode", "1"]
      );
    default:
      throw new Error(`Missing worked GGUF quantization example for ${dtype}`);
  }
}

function worked(
  group: number,
  position: number,
  weight: number,
  source: string,
  metadata: string,
  accesses: readonly [
    GgufQuantWorkedStorageAccess,
    ...GgufQuantWorkedStorageAccess[]
  ],
  reconstruction: string,
  sourceSymbols: readonly string[],
  metadataSymbols: readonly string[],
  storageSymbols: readonly string[],
  reconstructionSymbols: readonly string[]
): GgufQuantWorkedExample {
  return {
    selection: { group, position, weight },
    stages: [
      {
        kind: "source",
        title: "Select one source weight",
        detail: source,
        symbols: Array.from(
          new Set([
            "w[i]",
            "i",
            "group",
            "position",
            ...sourceSymbols.filter(
              (symbol) => symbol !== "w′" && symbol !== "w′[i]"
            )
          ])
        )
      },
      { kind: "metadata", title: "Select metadata and code", detail: metadata, symbols: metadataSymbols },
      {
        kind: "storage",
        title: "Extract exact record bits",
        detail: "Read only the named record fields and packed slices for this weight position.",
        symbols: storageSymbols,
        accesses
      },
      {
        kind: "reconstruction",
        title: "Reconstruct the selected value",
        detail: reconstruction,
        symbols: reconstructionSymbols
      }
    ]
  };
}

function contractSymbols(
  dtype: string,
  values: number,
  groupCount: number,
  groupValues: number
): readonly GgufQuantSymbol[] {
  const specific = symbolOrigins(dtype);
  const common = [
    symbol("w[i]", "original source weight at index i", "encoder input; it is not stored in the record"),
    symbol("i", "weight index inside the fixed record", `0…${values - 1} in source order`),
    symbol(
      "group",
      "metadata-sharing group index",
      groupCount === 1
        ? `only group 0; it is the whole ${values}-weight record`
        : `${groupCount} groups numbered 0…${groupCount - 1}; group = floor(i / ${groupValues})`
    ),
    symbol("position", "weight offset inside the selected group", `0…${groupValues - 1}; position = i mod ${groupValues}; i = group×${groupValues} + position`)
  ];
  const existing = new Set(specific.map(({ symbol: name }) => name));
  return [...common.filter(({ symbol: name }) => !existing.has(name)), ...specific];
}

function access(
  field: string,
  index: string,
  bits: string,
  action: string
): GgufQuantWorkedStorageAccess {
  return { field, index, bits, action };
}

function symbolOrigins(dtype: string): readonly GgufQuantSymbol[] {
  switch (dtype) {
    case "Q4_1":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("d", "affine step size", "FP16 record field d"),
        symbol("q", "unsigned four-bit code", "the selected position’s nibble in qs"),
        symbol("m", "shared minimum", "FP16 record field m")
      ];
    case "Q5_0":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("storedCode", "unsigned five-bit code", "four low bits from qs joined with bit i from qh"),
        symbol("join(qh, qs)", "five-bit assembly operation", "take the selected position’s qh bit as bit 4 and its qs nibble as bits 0…3"),
        symbol("qh", "high-code bit plane", "the four qh bytes interpreted as one little-endian uint32"),
        symbol("qs", "low four code bits", "the selected position’s nibble in record field qs"),
        symbol("d", "shared signed scale", "FP16 record field d"),
        symbol("16", "implicit code bias", "fixed Q5_0 format constant; it occupies no record bytes")
      ];
    case "Q5_1":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("join(qh, qs)", "unsigned five-bit code", "one high bit from qh plus the selected position’s four-bit qs nibble"),
        symbol("d", "affine step size", "FP16 record field d"),
        symbol("m", "shared minimum", "FP16 record field m")
      ];
    case "Q8_0":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("d", "shared scale", "FP16 record field d"),
        symbol("q", "signed eight-bit code", "the selected position’s int8 value in qs"),
        symbol("int8(…)", "signed interpretation", "the declared int8_t storage type of qs")
      ];
    case "Q8_1":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("stored_d", "rounded decode scale", "FP16 record field d"),
        symbol("q", "signed eight-bit code", "the selected position’s int8 value in qs"),
        symbol("stored_s", "stored scaled code sum", "FP16 record field s"),
        symbol("encoder_d", "pre-rounding encoder scale", "temporary quantizer value; not stored"),
        symbol("Σq", "sum of all 32 codes", "computed by the encoder before s is rounded")
      ];
    case "Q8_K":
      return [
        symbol("w′[i]", "reconstructed weight i", "decoder output at record index i, where i is 0…255"),
        symbol("d", "shared block scale", "FP32 record field d"),
        symbol("q[i]", "signed code for weight i", "record field qs[i]"),
        symbol("i", "record-local weight index", "position inside the fixed 256-weight block")
      ];
    case "IQ2_XXS":
      return gridSymbols(
        "iq2xxs_grid",
        8,
        "8-bit grid-index byte in qs",
        "7-bit sign index packed in the group’s second uint32",
        "4-bit s in bits 28…31 of that uint32",
        "d",
        "FP16 record field d",
        ["0.5", "fixed half-step offset", "0.25", "fixed IQ2_XXS normalization"]
      );
    case "IQ2_XS":
      return gridSymbols(
        "iq2xs_grid",
        8,
        "low nine bits of the group’s uint16 qs word",
        "upper seven bits of the same qs word, expanded through ksigns_iq2xs",
        "one nibble from record field scales",
        "d",
        "FP16 record field d",
        ["0.5", "fixed half-step offset", "0.25", "fixed IQ2_XS normalization"]
      );
    case "IQ3_XXS":
      return gridSymbols(
        "iq3xxs_grid",
        4,
        "one 8-bit index in qs[0…63]",
        "7-bit sign index packed in qs[64…95], expanded through ksigns_iq2xs",
        "4-bit s in bits 28…31 of the group metadata word",
        "d",
        "FP16 record field d",
        ["0.5", "fixed half-step offset", "0.5", "fixed IQ3_XXS normalization"]
      );
    case "IQ1_S":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected group, grid, and lane"),
        symbol("d", "global block scale", "FP16 record field d"),
        symbol("i", "weight index inside the fixed block", "0…255 in source order"),
        symbol("group", "32-weight group number", "floor(i / 32), in the range 0…7 for record index i"),
        symbol("subgrid", "eight-weight vector number inside the group", "floor((i mod 32) / 8), in the range 0…3"),
        symbol("s", "three-bit local scale code", "bits 12…14 of qh[group]"),
        symbol("index", "11-bit table index", "index = qs[4×group + subgrid] | (((qh[group] >> (3×subgrid)) & 7) << 8)"),
        symbol("signedGrid", "fixed 2048-entry table of signed eight-value vectors", "GGML’s compiled iq1s_grid codebook—not bytes in the file; each lane is −1, 0, or +1, and quantization selects the best table index"),
        symbol("lane", "position inside the selected eight-value grid", "lane = position mod 8, in the range 0…7; i = 32×group + 8×subgrid + lane"),
        symbol("δ", "small signed grid offset", "qh[group] bit 15 chooses −0.125 when set, +0.125 when clear"),
        symbol("2s + 1", "odd local multiplier", "fixed IQ1_S decode rule applied to the stored scale code s")
      ];
    case "IQ3_S":
      return gridSymbols(
        "iq3s_grid",
        4,
        "eight low bits from qs plus one matching high bit from qh",
        "the matching sign-mask position in record field signs",
        "one nibble from record field scales",
        "d",
        "FP16 record field d",
        ["1", "fixed odd-scale base", "2", "fixed multiplier applied to s"]
      );
    case "IQ2_S":
      return gridSymbols(
        "iq2s_grid",
        8,
        "eight low bits from qs[0…31] plus two matching high bits from qh",
        "the matching mask bit from qs[32…63]",
        "one nibble from record field scales",
        "d",
        "FP16 record field d",
        ["0.5", "fixed half-step offset", "0.25", "fixed IQ2_S normalization"]
      );
    case "IQ4_NL":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("i", "weight index inside this record", "i ∈ [0, 31]; IQ4_NL stores 32 weights per record"),
        symbol("group", "metadata-sharing group number", "IQ4_NL has exactly one group: group 0 is the whole 32-weight record and shares d"),
        symbol("position", "weight offset inside group 0", "position = i mod 32 = i, in the range 0…31; i = group×32 + position"),
        symbol("d", "shared block scale", "FP16 record field d"),
        symbol("nibble", "four-bit codebook index", "the selected position’s low or high nibble in qs"),
        symbol("nonlinearLevel", "fixed 16-entry signed level table", "GGML’s compiled kvalues_iq4nl table, not bytes in this record")
      ];
    case "IQ4_XS":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("d", "global block scale", "FP16 record field d"),
        symbol("ls", "unsigned six-bit local-scale code", "four low bits from scales_l plus two high bits from scales_h"),
        symbol("32", "local-scale zero bias", "fixed IQ4_XS format constant; it occupies no record bytes"),
        symbol("nibble", "four-bit codebook index", "the selected position’s nibble in qs"),
        symbol("nonlinearLevel", "fixed 16-entry signed level table", "GGML’s compiled kvalues_iq4nl table")
      ];
    case "IQ1_M":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected group, grid, and lane"),
        symbol("embedded_d", "global FP16 scale", "reassembled from the top four bits of four uint16 views of scales[8]"),
        symbol("s", "three-bit local scale code", "the matching three-bit slice in the lower 12 bits of scales"),
        symbol("index", "11-bit table index", "eight low bits from qs plus three high bits from the matching qh nibble"),
        symbol("signedGrid", "fixed 2048-entry table of signed eight-value vectors", "GGML’s compiled iq1s_grid table; each lane is −1, 0, or +1"),
        symbol("lane", "position inside the selected eight-value grid", "lane = position mod 8, in the range 0…7"),
        symbol("δ", "small signed grid offset", "bit 3 of the matching qh nibble chooses the IQ1_M ±0.125 constant"),
        symbol("2s + 1", "odd local multiplier", "fixed IQ1_M decode rule applied to s")
      ];
    case "TQ1_0":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("d", "shared magnitude", "trailing FP16 record field d"),
        symbol("base3Digit", "stored ternary digit 0, 1, or 2", "extracted from the selected position’s packed byte in qs or qh"),
        symbol("1", "ternary storage bias", "fixed format constant mapping digits 0/1/2 to −1/0/+1")
      ];
    case "TQ2_0":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("d", "shared magnitude", "trailing FP16 record field d"),
        symbol("twoBitCode", "stored two-bit code 0, 1, or 2", "the selected position’s two-bit slice in qs; binary 11 is unused"),
        symbol("1", "ternary storage bias", "fixed format constant mapping codes 0/1/2 to −1/0/+1")
      ];
    case "MXFP4":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("e", "shared E8M0 exponent code", "leading uint8 record field e"),
        symbol("E8M0_HALF(e)", "decoded half-scale", "GGML helper applied to e, including its e = 0 path"),
        symbol("nibble", "signed E2M1 FP4 code", "the selected position’s low or high nibble in qs"),
        symbol("doubledE2M1", "fixed doubled FP4 lookup table", "GGML runtime table; the half-scale compensates for the doubled entries")
      ];
    case "NVFP4":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("group", "local 16-weight group index", "floor(i / 16), selecting d[0…3]"),
        symbol("d[group]", "UE4M3 local-scale code", "record byte d[group]"),
        symbol("UE4M3(…)", "decoded half-scale", "GGML UE4M3 decode helper"),
        symbol("nibble", "signed E2M1 FP4 code", "the selected position’s low or high nibble in qs"),
        symbol("doubledE2M1", "fixed doubled FP4 lookup table", "GGML runtime table")
      ];
    case "Q1_0":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("signBit", "stored sign selector", "the selected position’s bit in record field qs"),
        symbol("d", "shared positive magnitude", "FP16 record field d"),
        symbol("+d / −d", "the two reconstruction levels", "selected directly by signBit; no zero level is stored")
      ];
    case "Q2_0":
      return [
        symbol("w′", "one reconstructed weight", "decoder output for the selected position"),
        symbol("d", "shared magnitude", "FP16 record field d"),
        symbol("twoBitCode", "unsigned two-bit code", "the selected position’s two-bit slice in qs"),
        symbol("1", "implicit storage bias", "fixed Q2_0 format constant; it occupies no record bytes")
      ];
    default:
      return [];
  }
}

function gridSymbols(
  table: string,
  laneWidth: 4 | 8,
  indexSource: string,
  signSource: string,
  scaleSource: string,
  scaleSymbol: string,
  globalScaleSource: string,
  constants: readonly [string, string, string, string]
): readonly GgufQuantSymbol[] {
  const items = [
    symbol("w′", "one reconstructed weight", "decoder output for the selected grid lane"),
    symbol(scaleSymbol, "global block scale", globalScaleSource),
    symbol("s", "local scale code", scaleSource),
    symbol("index", "grid-table index", indexSource),
    symbol("grid", "fixed multi-value magnitude table", `GGML’s compiled ${table} table; it is not stored in this record`),
    symbol("lane", "position inside the selected grid vector", `lane = position mod ${laneWidth}, in the range 0…${laneWidth - 1}`),
    symbol("sign", "per-lane +1 or −1 multiplier", signSource),
    symbol(constants[0], constants[1], "fixed format constant; it occupies no record bytes")
  ];
  if (constants[2] !== constants[0]) {
    items.push(
      symbol(constants[2], constants[3], "fixed format constant; it occupies no record bytes")
    );
  } else {
    items[items.length - 1] = symbol(
      constants[0],
      `${constants[1]}; ${constants[3]}`,
      "the same fixed literal is used twice and occupies no record bytes"
    );
  }
  return items;
}

function symbol(
  name: string,
  meaning: string,
  source: string
): GgufQuantSymbol {
  return { symbol: name, meaning, source };
}
