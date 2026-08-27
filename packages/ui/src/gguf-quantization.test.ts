import { GGUF_DTYPE_CATALOG } from "@weights-viz/core";
import { describe, expect, it } from "vitest";
import { createGgufQuantizationGuide } from "./gguf-quantization";

describe("GGUF quantization guides", () => {
  it("covers every recognized quantized GGML type", () => {
    const quantized = GGUF_DTYPE_CATALOG.filter(
      (entry) =>
        entry.blockBytes !== undefined ||
        entry.dtype.startsWith("Q") ||
        entry.dtype.startsWith("IQ") ||
        entry.dtype.startsWith("TQ") ||
        entry.dtype.startsWith("MXFP") ||
        entry.dtype.startsWith("NVFP")
    );

    for (const entry of quantized) {
      const guide = createGgufQuantizationGuide(entry.dtype);
      expect(guide, entry.dtype).toBeDefined();
      expect(guide?.runtimeSteps.length, entry.dtype).toBeGreaterThan(0);
      expect(guide?.optimizations.length, entry.dtype).toBeGreaterThan(0);
      expect(guide?.creationSteps.length, entry.dtype).toBeGreaterThan(0);
      expect(guide?.parameters.length, entry.dtype).toBeGreaterThan(0);
    }
  });

  it.each([
    ["Q4_0", "block maximum"],
    ["Q4_1", "minimum"],
    ["Q4_K", "Local scales"],
    ["IQ2_XXS", "Importance matrix"],
    ["TQ1_0", "Decision threshold"],
    ["MXFP4", "E8M0"],
    ["NVFP4", "Local scales"],
    ["Q1_0", "mean(abs(weight))"],
    ["Q2_0", "maxAbs rather than"],
    ["Q4_0_4_8", "Tile layout"]
  ])("explains how %s parameters are selected", (dtype, expectedParameter) => {
    const guide = createGgufQuantizationGuide(dtype);
    expect(
      guide?.parameters.some(
        (parameter) =>
          parameter.name.includes(expectedParameter) ||
          parameter.selection.includes(expectedParameter)
      )
    ).toBe(true);
  });

  it("identifies Q8_K as a dot-product companion", () => {
    expect(createGgufQuantizationGuide("Q8_K")?.purpose).toContain(
      "companion"
    );
  });

  it("keeps fixed IQ4_NL separate from importance-matrix IQ formats", () => {
    const guide = createGgufQuantizationGuide("IQ4_NL");
    expect(guide?.purpose).toContain("fixed 16-entry nonlinear codebook");
    expect(guide?.purpose).toContain("does not require an importance matrix");
    expect(guide?.parameters.some(({ name }) => name === "Codebook")).toBe(true);
  });

  it("documents reference ternary scaling and removed interleaved layouts", () => {
    const ternary = createGgufQuantizationGuide("TQ1_0");
    const interleaved = createGgufQuantizationGuide("Q4_0_4_8");

    expect(ternary?.parameters[0]?.selection).toContain("maxAbs");
    expect(ternary?.parameters[1]?.selection).toContain("not searched");
    expect(interleaved?.purpose).toContain("removed GGUF type identifier");
    expect(interleaved?.creationSteps[0]).toContain("Do not select");
  });
});
