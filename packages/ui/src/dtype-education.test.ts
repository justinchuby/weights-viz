import type { TensorRecord, WeightFormat } from "@weights-viz/core";
import { describe, expect, it } from "vitest";
import { createDtypeEducation } from "./dtype-education";

describe("dtype education", () => {
  it("explains GGUF block quantization using parser metadata", () => {
    const lesson = createDtypeEducation(
      "gguf",
      makeTensor("Q4_0", 32n, 18n, {
        ggmlTypeId: 2,
        blockBytes: 18,
        blockElements: 32
      })
    );

    expect(lesson.family).toBe("Symmetric block quantization");
    expect(lesson.bitsPerValue).toBe(4);
    expect(lesson.block).toMatchObject({
      elements: 32,
      bytes: 18,
      effectiveBitsPerValue: 4.5
    });
    expect(lesson.formula?.expression).toContain("scale");
    expect(lesson.tensorBitsPerValue).toBe(4.5);
    expect(lesson.packing).toEqual({
      values: 2,
      bytes: 1,
      bitsPerValue: 4
    });
  });

  it("does not flatten split GGUF quantization bit planes", () => {
    const lesson = createDtypeEducation(
      "gguf",
      makeTensor("Q5_0", 32n, 22n, {
        ggmlTypeId: 6,
        blockBytes: 22,
        blockElements: 32
      })
    );

    expect(lesson.block?.effectiveBitsPerValue).toBe(5.5);
    expect(lesson.packing).toBeUndefined();
  });

  it("shows how six-bit SafeTensors values cross byte boundaries", () => {
    const lesson = createDtypeEducation(
      "safetensors",
      makeTensor("F6_E2M3", 4n, 3n)
    );

    expect(lesson.segments.map(({ bits }) => bits)).toEqual([1, 2, 3]);
    expect(lesson.packing).toEqual({
      values: 4,
      bytes: 3,
      bitsPerValue: 6
    });
    expect(lesson.storageNote).toContain("JSON header");
  });

  it.each([
    ["UINT4", 4, 2],
    ["INT2", 2, 4]
  ])(
    "explains packed ONNX %s values and external quantization parameters",
    (dtype, bits, values) => {
      const lesson = createDtypeEducation(
        "onnx",
        makeTensor(dtype, BigInt(values), 1n)
      );

      expect(lesson.bitsPerValue).toBe(bits);
      expect(lesson.packing?.values).toBe(values);
      expect(lesson.formula?.expression).toContain("zeroPoint");
      expect(lesson.concepts.some(({ term }) => term === "Graph quantization")).toBe(
        true
      );
    }
  );

  it("degrades gracefully for a future unknown dtype", () => {
    const lesson = createDtypeEducation(
      "safetensors",
      makeTensor("FUTURE_TYPE", 8n, 13n)
    );

    expect(lesson.family).toBe("Format-defined storage");
    expect(lesson.bitsPerValue).toBeUndefined();
    expect(lesson.summary).toContain("not described");
  });
});

function makeTensor(
  dtype: string,
  elements: bigint,
  byteLength: bigint,
  encoding?: Record<string, string | number | boolean>
): TensorRecord {
  return {
    id: dtype,
    name: `${dtype}.weight`,
    fileId: "model",
    dtype,
    shape: [elements],
    byteOffset: 0n,
    byteLength,
    ...(encoding ? { encoding } : {}),
    sampleSupport: "unsupported"
  };
}
