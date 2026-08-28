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
    expect(lesson.formula?.expression).toContain("d × q");
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
    expect(lesson.formula?.expression).toContain("storedCode − 16");
  });

  it("distinguishes the recent binary and two-bit block algorithms", () => {
    const q1 = createDtypeEducation(
      "gguf",
      makeTensor("Q1_0", 128n, 18n, {
        blockBytes: 18,
        blockElements: 128
      })
    );
    const q2 = createDtypeEducation(
      "gguf",
      makeTensor("Q2_0", 64n, 18n, {
        blockBytes: 18,
        blockElements: 64
      })
    );

    expect(q1.formula?.expression).toContain("signBit");
    expect(q1.formula?.explanation).toContain("mean(|weight|)");
    expect(q2.formula?.expression).toContain("twoBitCode − 1");
    expect(q2.formula?.explanation).toContain("+1 storage bias");
  });

  it("distinguishes affine and symmetric K-quant decode formulas", () => {
    const affine = createDtypeEducation(
      "gguf",
      makeTensor("Q4_K", 256n, 144n, {
        blockBytes: 144,
        blockElements: 256
      })
    );
    const symmetric = createDtypeEducation(
      "gguf",
      makeTensor("Q3_K", 256n, 110n, {
        blockBytes: 110,
        blockElements: 256
      })
    );
    const signedScale = createDtypeEducation(
      "gguf",
      makeTensor("Q6_K", 256n, 210n, {
        blockBytes: 210,
        blockElements: 256
      })
    );

    expect(affine.formula?.expression).toContain("globalMin × subMin");
    expect(symmetric.formula?.expression).not.toContain("subMin");
    expect(symmetric.formula?.expression).toContain("(subScale − 32)");
    expect(signedScale.formula?.expression).toBe(
      "weight ≈ globalScale × subScale × q"
    );
  });

  it("describes Q8 companion metadata without treating sums as offsets", () => {
    const q8_1 = createDtypeEducation(
      "gguf",
      makeTensor("Q8_1", 32n, 36n, {
        blockBytes: 36,
        blockElements: 32
      })
    );
    const q8_k = createDtypeEducation(
      "gguf",
      makeTensor("Q8_K", 256n, 292n, {
        blockBytes: 292,
        blockElements: 256
      })
    );

    expect(q8_1.family).toBe("Dot-product companion block");
    expect(q8_1.block?.sections[0]?.label).toContain("scaled sum");
    expect(q8_1.concepts.some(({ term }) => term === "Scaled sum")).toBe(true);
    expect(q8_k.formula?.expression).toContain("d × q");
    expect(q8_k.block?.sections[0]?.label).toContain("group sums");
  });

  it("uses exact complex-format contracts in the static lesson", () => {
    const iq1m = createDtypeEducation(
      "gguf",
      makeTensor("IQ1_M", 256n, 56n, {
        blockBytes: 56,
        blockElements: 256
      })
    );
    const mxfp4 = createDtypeEducation(
      "gguf",
      makeTensor("MXFP4", 32n, 17n, {
        blockBytes: 17,
        blockElements: 32
      })
    );

    expect(iq1m.formula?.expression).toContain("embedded_d");
    expect(iq1m.formula?.explanation).toContain("weaving");
    expect(mxfp4.formula?.expression).toContain("E8M0_HALF");
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
