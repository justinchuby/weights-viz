import {
  GGUF_DTYPE_CATALOG,
  ONNX_DTYPE_CATALOG,
  SAFETENSORS_DTYPE_CATALOG,
  type TensorRecord,
  type WeightFormat
} from "@weights-viz/core";
import { describe, expect, it } from "vitest";
import {
  createDtypeAnimationStory,
  dtypeAnimationKind
} from "./DtypeEncodingAnimation";
import { isKQuantDtype, K_QUANT_LAYOUTS } from "./KQuantAnimation";
import { ggufStorageLayout } from "./gguf-storage-layouts";
import { createDtypeEducation } from "./dtype-education";

describe("dtype animation coverage", () => {
  it("assigns an animation family to every catalog entry", () => {
    const entries = [
      ...SAFETENSORS_DTYPE_CATALOG,
      ...GGUF_DTYPE_CATALOG,
      ...ONNX_DTYPE_CATALOG
    ];

    expect(entries).toHaveLength(92);
    expect(
      entries.every((entry) =>
        Boolean(dtypeAnimationKind(entry.format, entry.dtype))
      )
    ).toBe(true);
  });

  it("routes representative encodings to distinct visual stories", () => {
    expect(dtypeAnimationKind("gguf", "Q4_0")).toBe("q4");
    expect(dtypeAnimationKind("gguf", "Q4_K")).toBe("k-quant");
    expect(dtypeAnimationKind("gguf", "IQ2_XS")).toBe("codebook");
    expect(dtypeAnimationKind("gguf", "TQ1_0")).toBe("ternary");
    expect(dtypeAnimationKind("gguf", "NVFP4")).toBe("microscale");
    expect(dtypeAnimationKind("onnx", "UINT4")).toBe("packed");
    expect(dtypeAnimationKind("safetensors", "F8_E4M3")).toBe("floating");
    expect(dtypeAnimationKind("onnx", "STRING")).toBe("schema");
    expect(story("gguf", "IQ2_XS", 256, 74).title).toContain(
      "multi-weight grids"
    );
    expect(story("gguf", "IQ4_NL", 32, 18).title).toContain(
      "nonlinear levels"
    );
  });

  it("uses encoding-specific stories for edge-case scalar and block types", () => {
    expect(story("gguf", "Q1_0", 128, 18).encoded).toContain(
      "d = mean(|w|)"
    );
    expect(story("gguf", "Q2_0", 64, 18).encoded).toContain(
      "stored = q + 1"
    );
    expect(story("gguf", "Q5_0", 32, 22).encoded).toContain(
      "stored = q + 16"
    );
    expect(story("gguf", "Q8_0", 32, 34).intro).toContain("no bias");
    expect(story("onnx", "UINT8", 1, 1).source).not.toContain("−37");
    expect(story("onnx", "BOOL", 1, 1).source).toEqual(["false", "true"]);
    expect(story("onnx", "COMPLEX64", 1, 8).title).toContain(
      "two components"
    );
    const e8m0 = story("onnx", "FLOAT8E8M0", 1, 1);
    expect(e8m0.title).toContain("power of two");
    expect(e8m0.stages[1]?.detail).not.toContain("significand fields");
    expect(e8m0.stages[2]?.detail).toContain("independently");
    expect(story("gguf", "IQ1_S", 256, 50).encoded).not.toContain(
      "sign index / bits"
    );
    expect(story("onnx", "STRING", 1, 1).storage).toEqual([
      "42",
      "03",
      "63",
      "61",
      "74",
      "42",
      "06",
      "…"
    ]);
  });
});

function story(
  format: WeightFormat,
  dtype: string,
  elements: number,
  bytes: number
) {
  const tensor: TensorRecord = {
    id: dtype,
    name: dtype,
    fileId: "test",
    dtype,
    shape: [BigInt(elements)],
    byteOffset: 0n,
    byteLength: BigInt(bytes),
    ...(format === "gguf"
      ? {
          encoding: {
            blockBytes: bytes,
            blockElements: elements
          }
        }
      : {}),
    sampleSupport: "unsupported"
  };
  return createDtypeAnimationStory(
    format,
    createDtypeEducation(format, tensor)
  );
}

describe("K-quant animation layouts", () => {
  it.each(Object.values(K_QUANT_LAYOUTS))(
    "$dtype accounts for all 256 values and physical bytes",
    (layout) => {
      expect(layout.subBlocks * layout.valuesPerSubBlock).toBe(256);
      expect(
        layout.sections.reduce((total, section) => total + section.bytes, 0)
      ).toBe(layout.bytes);
    }
  );

  it("uses the ABI-defined Q4_K hierarchy", () => {
    expect(K_QUANT_LAYOUTS.Q4_K).toMatchObject({
      bytes: 144,
      codeBits: 4,
      subBlocks: 8,
      valuesPerSubBlock: 32,
      scaleBits: 6,
      minBits: 6
    });
  });
});

describe("GGUF physical animation layouts", () => {
  it("accounts for every parser-recognized block format byte-for-byte", () => {
    const blockEntries = GGUF_DTYPE_CATALOG.filter(
      (entry) => entry.blockBytes !== undefined
    );

    for (const entry of blockEntries) {
      const kLayout = isKQuantDtype(entry.dtype)
        ? K_QUANT_LAYOUTS[entry.dtype]
        : undefined;
      const fields = ggufStorageLayout(entry.dtype);
      const bytes =
        kLayout?.sections.reduce((total, section) => total + section.bytes, 0) ??
        fields?.reduce((total, field) => total + field.bytes, 0);
      expect(bytes, `${entry.dtype} physical layout`).toBe(entry.blockBytes);
    }
  });

  it("describes IQ2_S index and sign payloads within qs", () => {
    expect(
      ggufStorageLayout("IQ2_S")?.find((field) => field.name === "qs")?.role
    ).toContain("final 32 bytes hold sign masks");
  });
});
