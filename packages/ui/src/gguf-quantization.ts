export interface QuantizationParameter {
  name: string;
  selection: string;
}

export interface GgufQuantizationGuide {
  purpose: string;
  runtimeSteps: string[];
  optimizations: string[];
  creationSteps: string[];
  parameters: QuantizationParameter[];
  tradeoff: string;
}

export function createGgufQuantizationGuide(
  dtype: string
): GgufQuantizationGuide | undefined {
  if (["Q4_2", "Q4_3"].includes(dtype)) return legacyGuide(dtype);
  if (/_(?:4_4|4_8|8_8)$/.test(dtype)) return interleavedGuide(dtype);
  if (dtype === "IQ4_NL") return nonlinearCodebookGuide();
  if (dtype.startsWith("IQ")) return importanceGuide(dtype);
  if (dtype === "Q8_1" || dtype === "Q8_K") return dotPartnerGuide(dtype);
  if (/^Q[2-8]_K$/.test(dtype)) return kQuantGuide(dtype);
  if (dtype === "Q4_1" || dtype === "Q5_1") return affineGuide(dtype);
  if (dtype.startsWith("TQ")) return ternaryGuide(dtype);
  if (dtype === "MXFP4") return microscaleGuide();
  if (dtype === "NVFP4") return nvfp4Guide();
  if (dtype === "Q1_0") return q1Guide();
  if (dtype === "Q2_0") return q2Guide();
  if (/^Q[1-8]_0$/.test(dtype)) return symmetricGuide(dtype);
  return undefined;
}

function symmetricGuide(dtype: string): GgufQuantizationGuide {
  const bits = dtype.match(/^Q(\d)/)?.[1] ?? "low-bit";
  return {
    purpose: `${dtype} is a simple blockwise ${bits}-bit weight format: cheap to create, predictable to decode, and widely supported by CPU kernels.`,
    runtimeSteps: [
      "Load one weight block and its shared scale while loading the matching activation values.",
      "Unpack integer codes in vector registers, apply the block scale, and accumulate dot products into wider integer or floating-point lanes.",
      "Reduce the vector accumulators and add the result to the output tile; the full weight matrix is never materialized as F32."
    ],
    optimizations: [
      "SIMD shifts, masks, and multiply-add instructions decode many packed codes together.",
      "Kernels fuse dequantization with GEMV/GEMM so decoded weights live only in registers.",
      "Blocks and rows are tiled for cache reuse; architecture-specific paths use AVX/AVX-512, NEON, Metal, CUDA, or similar vector units."
    ],
    creationSteps: [
      "Split each tensor row into the block size fixed by the GGML type.",
      "Measure the block range and choose a shared scale that maps the largest useful magnitude into the available code range.",
      "Round each scaled value to its nearest code, clamp it, pack the bits, and optionally search nearby scales to reduce reconstruction error."
    ],
    parameters: [
      {
        name: "Block size",
        selection:
          "Fixed by the dtype ABI, not learned; every runtime must agree on the same number of weights and bytes."
      },
      {
        name: "Scale",
        selection:
          "Usually initialized from the block maximum absolute value divided by the largest code magnitude, then optionally refined for lower error."
      },
      {
        name: "Codes",
        selection:
          "Nearest representable integers after division by the scale; ties, clipping, and scale refinement are quantizer-specific."
      }
    ],
    tradeoff:
      "Smaller codes save bandwidth and cache, but a single scale must represent the whole block. Outliers can therefore reduce precision for the remaining weights."
  };
}

function affineGuide(dtype: string): GgufQuantizationGuide {
  return {
    purpose: `${dtype} adds a block minimum to use the code range efficiently when values are not symmetric around zero.`,
    runtimeSteps: [
      "Unpack the unsigned low-bit code for each weight.",
      "Reconstruct values with scale × code + minimum while accumulating against activations.",
      "Use precomputed activation sums where available so the minimum term can be applied once per block instead of once per multiply."
    ],
    optimizations: [
      "Packed codes are expanded with SIMD bit operations and consumed immediately by dot-product instructions.",
      "The offset contribution can be algebraically folded into a block activation sum.",
      "Scale, minimum, and codes are streamed together, reducing memory traffic compared with expanded floating-point weights."
    ],
    creationSteps: [
      "Find a representative minimum and maximum for each block.",
      "Set the initial scale to (maximum − minimum) / (codeLevels − 1).",
      "Quantize (weight − minimum) / scale, then optionally adjust the endpoints or scale to minimize block error."
    ],
    parameters: [
      {
        name: "Minimum",
        selection:
          "Starts from the block minimum; optimized quantizers may move it slightly if clipping an outlier lowers total error."
      },
      {
        name: "Scale",
        selection:
          "Covers the selected block interval with all available integer levels."
      },
      {
        name: "Zero point",
        selection:
          "Implicit in the stored minimum rather than necessarily encoded as a separate integer zero-point field."
      }
    ],
    tradeoff:
      "Affine blocks model shifted distributions better than symmetric blocks, at the cost of extra metadata and offset arithmetic."
  };
}

function kQuantGuide(dtype: string): GgufQuantizationGuide {
  return {
    purpose: `${dtype} uses 256-weight super-blocks with compact local scales (and, for affine variants, minima) under a shared higher-level scale.`,
    runtimeSteps: [
      "Load the super-block metadata and decode the small per-group scales or minima.",
      "Unpack each sub-block's weight codes and form integer dot products against a matching activation tile.",
      "Apply local and global scales to partial sums, then accumulate them into the output tile."
    ],
    optimizations: [
      "Hierarchical metadata adapts to local ranges without paying for a full floating-point scale per tiny group.",
      "Kernels keep scale tables in registers and process code planes in SIMD-friendly batches.",
      "Quantized dot-product partner layouts avoid repeatedly converting activations and enable integer multiply-add pipelines."
    ],
    creationSteps: [
      "Divide a row into 256-value super-blocks and smaller local groups.",
      "Estimate an ideal scale or scale-plus-minimum for every local group, often using weighted reconstruction error.",
      "Choose global scale factors that compactly quantize those local parameters, then quantize the weight codes using the reconstructed local ranges."
    ],
    parameters: [
      {
        name: "Local scales",
        selection:
          "Fit each sub-group's range or minimize its weighted squared error."
      },
      {
        name: "Global scale",
        selection:
          "Chosen so the set of local scales can itself be represented by the small metadata fields."
      },
      {
        name: "Error weights",
        selection:
          "May be uniform or derived from weight magnitude and calibration importance, depending on the quantizer and options."
      }
    ],
    tradeoff:
      "K-quants usually improve quality at the same average bits per weight, but their hierarchical decode is more complex and kernel support matters."
  };
}

function importanceGuide(dtype: string): GgufQuantizationGuide {
  return {
    purpose: `${dtype} is an importance-aware codebook format. It spends representational accuracy where calibration data predicts errors will matter most.`,
    runtimeSteps: [
      "Read compact indices, signs, and scale metadata for a block.",
      "Look up small codebook/grid values and combine them into vector lanes.",
      "Fuse lookup, scaling, and dot-product accumulation so no expanded weight tensor is written to memory."
    ],
    optimizations: [
      "Small fixed codebooks fit in registers or fast constant storage.",
      "Table lookup and shuffle instructions decode several indices in parallel.",
      "Specialized kernels arrange code bits to match vector lanes and amortize scale handling across each block."
    ],
    creationSteps: [
      "Collect an importance matrix from representative calibration prompts, or use a fallback heuristic when no calibration data is available.",
      "For each block, search codebook entries and scales that minimize importance-weighted reconstruction error.",
      "Pack the selected indices, signs, auxiliary fields, and scales in the dtype's fixed block layout."
    ],
    parameters: [
      {
        name: "Importance matrix",
        selection:
          "Estimated from activation statistics on calibration data; frequently used tensor dimensions receive a larger error penalty."
      },
      {
        name: "Codebook index",
        selection:
          "Selected by nearest or searched weighted-error match after accounting for the block scale."
      },
      {
        name: "Scale",
        selection:
          "Optimized jointly or iteratively with codebook assignments rather than determined only by the largest absolute weight."
      }
    ],
    tradeoff:
      "IQ formats can preserve quality at extremely low bit rates, but conversion is slower, calibration quality matters, and runtimes need dedicated lookup kernels."
  };
}

function ternaryGuide(dtype: string): GgufQuantizationGuide {
  return {
    purpose: `${dtype} restricts weights to a tiny alphabet around zero and shares magnitude metadata across a block.`,
    runtimeSteps: [
      "Decode compact ternary symbols into negative, zero, or positive lanes.",
      "Accumulate additions and subtractions of activation values instead of general weight multiplies where the kernel permits.",
      "Apply the shared scale to each block's accumulated result."
    ],
    optimizations: [
      "A zero symbol skips useful work and the ±1 symbols can replace multiplication with sign selection.",
      "Bit extraction or small lookup tables decode many ternary symbols at once.",
      "The very small weight stream increases cache residency and reduces memory bandwidth."
    ],
    creationSteps: [
      "Find maxAbs for the block and store it as the shared scale.",
      "Normalize each weight by maxAbs and round to −1, 0, or +1.",
      "Pack the ternary symbols; nearest-integer rounding gives fixed decision boundaries at ±0.5 × scale."
    ],
    parameters: [
      {
        name: "Scale",
        selection:
          "The GGML reference quantizer sets the block scale directly to maxAbs."
      },
      {
        name: "Decision threshold",
        selection:
          "Fixed by nearest-integer rounding: magnitudes below half the scale map to zero; it is not searched by the reference quantizer."
      }
    ],
    tradeoff:
      "Ternary formats are exceptionally compact and can simplify arithmetic, but aggressive information loss makes model- and layer-level selection important."
  };
}

function microscaleGuide(): GgufQuantizationGuide {
  return {
    purpose:
      "MXFP4 combines E2M1 four-bit floating-point values with a shared E8M0 power-of-two scale for each 32-value block.",
    runtimeSteps: [
      "Load the shared exponent and packed FP4 values.",
      "Decode FP4 lanes, adjust their exponent by the shared E8M0 scale, and feed them directly into a dot-product tile.",
      "Accumulate in a wider floating-point type to prevent the low-precision products from dominating accumulation error."
    ],
    optimizations: [
      "Power-of-two scaling becomes exponent adjustment instead of a general multiply.",
      "Accelerators with microscaling support can consume the block format directly; other kernels vectorize nibble unpack and lookup.",
      "Fused conversion and matrix multiplication avoids an intermediate F16/F32 weight buffer."
    ],
    creationSteps: [
      "Partition values into 32-value blocks and find the block's useful maximum magnitude.",
      "Round a shared scale to an E8M0 power of two that puts values inside the FP4 E2M1 range.",
      "Round each scaled value to the nearest FP4 value, with overflow and subnormal behavior defined by the converter."
    ],
    parameters: [
      {
        name: "E8M0 block scale",
        selection:
          "Derived from the block maximum and rounded to a power of two; exact saturation and rounding rules follow the converter implementation."
      },
      {
        name: "FP4 values",
        selection:
          "Nearest E2M1 representable values after division by the shared scale."
      }
    ],
    tradeoff:
      "MXFP4 is hardware-friendly and regular, but a single power-of-two scale can be sensitive to block outliers."
  };
}

function nvfp4Guide(): GgufQuantizationGuide {
  return {
    purpose:
      "NVFP4 stores 64 E2M1 FP4 values as four independently scaled 16-value groups, using one UE4M3 scale byte per group.",
    runtimeSteps: [
      "Load packed FP4 values and the four local UE4M3 scales.",
      "Apply each scale to its 16 E2M1 lanes while converting them inside the matrix-multiply kernel.",
      "Multiply by the activation tile and accumulate in a wider type."
    ],
    optimizations: [
      "Tensor-core-capable hardware can fuse FP4 conversion, scaling, and matrix multiplication.",
      "Small local groups improve numerical utilization while packed nibbles minimize bandwidth.",
      "Scale values are reused across several products and retained in registers."
    ],
    creationSteps: [
      "Split the block into four groups of 16 values and measure each group's maximum absolute value.",
      "Set each local scale from maxAbs / 6, then encode that scale in unsigned E4M3.",
      "Divide by the decoded local scale, round to E2M1 FP4, and pack two values per byte."
    ],
    parameters: [
      {
        name: "Local scales",
        selection:
          "One per 16 values, initialized as the group's maximum absolute value divided by FP4 E2M1's maximum magnitude, 6, then rounded to UE4M3."
      },
      {
        name: "FP4 rounding",
        selection:
          "Uses the converter's nearest-value and saturation behavior after division by the decoded local scale."
      }
    ],
    tradeoff:
      "Fine local scaling improves FP4 fidelity, but the fastest path depends on recent GPU support and a matching kernel."
  };
}

function nonlinearCodebookGuide(): GgufQuantizationGuide {
  return {
    purpose:
      "IQ4_NL maps each weight to a fixed 16-entry nonlinear codebook and shares one F16 scale across 32 values; unlike importance-aware IQ variants, it does not require an importance matrix.",
    runtimeSteps: [
      "Load the block scale and unpack two four-bit codebook indices per byte.",
      "Look up the 16 fixed nonlinear values in vector lanes.",
      "Multiply lookup values by the shared scale while accumulating the dot product."
    ],
    optimizations: [
      "The tiny fixed codebook fits in registers or a fast shuffle table.",
      "Nibble unpack, lookup, scaling, and dot accumulation are fused.",
      "Nonlinear levels better match weight distributions without extra per-weight bits."
    ],
    creationSteps: [
      "Estimate an initial block scale from the source magnitudes.",
      "Assign each normalized weight to the nearest value in the fixed IQ4_NL codebook.",
      "Refine the shared scale against reconstruction error and pack the selected four-bit indices."
    ],
    parameters: [
      {
        name: "Codebook",
        selection:
          "Fixed by the IQ4_NL format; it is not trained from calibration prompts."
      },
      {
        name: "Scale",
        selection:
          "Selected and refined per block to minimize error for assignments to the fixed nonlinear levels."
      }
    ],
    tradeoff:
      "IQ4_NL gains accuracy from better-spaced levels while remaining simple, but dedicated lookup kernels are preferable to generic integer decode."
  };
}

function q1Guide(): GgufQuantizationGuide {
  return {
    purpose:
      "Q1_0 stores one sign bit per weight and one F16 scale for 128 values, yielding only two reconstructed levels: −scale and +scale.",
    runtimeSteps: [
      "Read the shared scale and unpack 128 sign bits.",
      "Select +activation or −activation from each sign bit and accumulate the block sum.",
      "Multiply the accumulated result by the shared scale."
    ],
    optimizations: [
      "One-bit weights minimize memory traffic and can replace multiplication with sign selection.",
      "Wide bit operations, XOR/sign masks, or lookup tables decode many weights together.",
      "The scale is applied once to a block partial sum instead of once per product."
    ],
    creationSteps: [
      "Compute the mean absolute value of the 128 source weights.",
      "Store that mean magnitude as the block scale.",
      "Encode each weight as the positive or negative sign level nearest to it."
    ],
    parameters: [
      {
        name: "Scale",
        selection:
          "The reference quantizer uses mean(abs(weight)) for the block, not maxAbs."
      },
      {
        name: "Sign bit",
        selection:
          "Chosen from the sign of each source weight; zero follows the converter's comparison rule."
      }
    ],
    tradeoff:
      "Q1_0 offers extreme compression and simple arithmetic, but two levels cannot preserve zeros or small magnitudes."
  };
}

function q2Guide(): GgufQuantizationGuide {
  return {
    purpose:
      "Q2_0 stores 64 two-bit codes and one F16 scale. The reference encoder normally uses the three levels −scale, 0, and +scale.",
    runtimeSteps: [
      "Load the shared scale and unpack four two-bit codes per byte.",
      "Convert each code with scale × (code − 1) while accumulating against activations.",
      "Apply vector shifts and masks so several code lanes feed each dot-product instruction."
    ],
    optimizations: [
      "Four weights per byte sharply reduce bandwidth.",
      "The small centered code alphabet is cheap to decode with integer vector operations.",
      "Fused dequantization keeps reconstructed values in registers."
    ],
    creationSteps: [
      "Find maxAbs for the 64-value block and store it as the scale.",
      "Normalize each weight by maxAbs, round to −1, 0, or +1, then add one to form the stored code.",
      "Pack four two-bit codes per byte; the fourth code point is not normally selected by the reference quantizer."
    ],
    parameters: [
      {
        name: "Scale",
        selection:
          "The reference quantizer sets scale directly to maxAbs rather than dividing by a multi-level integer maximum."
      },
      {
        name: "Code",
        selection:
          "round(weight / scale) + 1, producing the centered levels used by the decoder."
      }
    ],
    tradeoff:
      "Q2_0 preserves a zero level and signs at very low cost, but its coarse magnitude alphabet can introduce substantial error."
  };
}

function dotPartnerGuide(dtype: string): GgufQuantizationGuide {
  return {
    purpose: `${dtype} is designed for efficient quantized dot products and is commonly used as a companion or intermediate representation rather than the smallest archival format.`,
    runtimeSteps: [
      "Quantize or load the dot-product operand in the expected block layout.",
      "Form wide integer partial sums against the low-bit weight blocks.",
      "Combine block scales and any precomputed sums before accumulating into the output."
    ],
    optimizations: [
      "Byte-sized lanes map directly to common integer dot-product instructions.",
      "Stored block sums let affine or hierarchical kernels fold offset terms outside the inner multiply loop.",
      "The regular layout is chosen to feed vector units rather than minimize every last metadata bit."
    ],
    creationSteps: [
      "Use the fixed companion block size expected by the target weight kernel.",
      "Choose an absolute-range scale, round values to signed eight-bit lanes, and compute any block sum fields.",
      "For transient activations, repeat this conversion per input tile and immediately consume it in the dot product."
    ],
    parameters: [
      {
        name: "Scale",
        selection:
          "Maps the block's useful absolute range to the signed integer range."
      },
      {
        name: "Block sum",
        selection:
          "Computed exactly from quantized codes and stored when the paired kernel needs an offset correction."
      }
    ],
    tradeoff:
      "These formats use more bits than low-bit weights, but make the arithmetic path simple and fast."
  };
}

function interleavedGuide(dtype: string): GgufQuantizationGuide {
  return {
    purpose: `${dtype} is a removed GGUF type identifier for a historical interleaved, architecture-oriented layout. It remains listed so older files and IDs can be explained, not as a current conversion target.`,
    runtimeSteps: [
      "A historical matching kernel loaded codes in its preferred lane order.",
      "That kernel performed vector dot products with fewer shuffle and transpose operations.",
      "Current GGUF runtimes may reject the identifier or require conversion to a supported layout."
    ],
    optimizations: [
      "Offline interleaving moves lane-reordering work out of every inference call.",
      "Tile-aligned memory access improves vector loads and reduces permutation instructions.",
      "The benefit is architecture-specific; a runtime may convert or reject the layout when no matching kernel exists."
    ],
    creationSteps: [
      "Do not select this removed identifier for a new GGUF model.",
      "Choose a current portable dtype and let the runtime pack or reorder tiles internally for its target kernel.",
      "Convert historical files once rather than preserving an obsolete on-disk layout."
    ],
    parameters: [
      {
        name: "Numerical parameters",
        selection:
          "Inherited from the base Q4_0 or IQ4_NL quantizer."
      },
      {
        name: "Tile layout",
        selection:
          "Historically chosen from the target CPU kernel's row and register blocking; no longer a supported GGUF output choice."
      }
    ],
    tradeoff:
      "Offline interleaving could improve one CPU path, but tying the file to a removed kernel layout reduced portability without improving quantization accuracy."
  };
}

function legacyGuide(dtype: string): GgufQuantizationGuide {
  return {
    purpose: `${dtype} is a legacy GGML type identifier retained so old files can be recognized. Current converters should choose a supported modern format.`,
    runtimeSteps: [
      "A compatible legacy runtime interprets the historical block layout.",
      "Modern runtimes may reject the tensor or require conversion before matrix operations."
    ],
    optimizations: [
      "No new optimized kernel should be assumed for a deprecated identifier.",
      "Converting once to a current layout is safer than decoding the legacy form on every operation."
    ],
    creationSteps: [
      "Do not select this dtype for new models.",
      "Choose a current Q4, K-quant, IQ, or floating-point format based on quality and target hardware."
    ],
    parameters: [
      {
        name: "Converter choice",
        selection:
          "Use a current quantization preset with active runtime support rather than reproducing the legacy encoding."
      }
    ],
    tradeoff:
      "The identifier is useful for diagnostics and migration, not as a recommended deployment target."
  };
}
