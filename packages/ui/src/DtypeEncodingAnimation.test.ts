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
import {
  GGUF_QUANT_CONTRACT_DTYPES,
  ggufQuantContract
} from "./gguf-quant-contracts";
import {
  isKQuantDtype,
  K_QUANT_LAYOUTS,
  kQuantCodeExample,
  kQuantMetadataBytes
} from "./KQuantAnimation";
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
    expect(dtypeAnimationKind("gguf", "IQ2_XS")).toBe("gguf-contract");
    expect(dtypeAnimationKind("gguf", "TQ1_0")).toBe("gguf-contract");
    expect(dtypeAnimationKind("gguf", "NVFP4")).toBe("gguf-contract");
    expect(dtypeAnimationKind("onnx", "UINT4")).toBe("packed");
    expect(dtypeAnimationKind("safetensors", "F8_E4M3")).toBe("floating");
    expect(dtypeAnimationKind("onnx", "STRING")).toBe("schema");
    expect(ggufQuantContract("IQ2_XS")?.codes.join(" ")).toContain(
      "9-bit index"
    );
    expect(ggufQuantContract("IQ4_NL")?.codes.join(" ")).toContain(
      "nonlinear level"
    );
  });

  it("uses encoding-specific stories for edge-case scalar and block types", () => {
    expect(ggufQuantContract("Q1_0")?.metadata).toContain("d = mean(|w|)");
    expect(ggufQuantContract("Q2_0")?.codes).toContain(
      "stored = q + 1 ∈ [0,3]"
    );
    expect(ggufQuantContract("Q5_0")?.codes).toContain(
      "stored = q + 16 ∈ [0, 31]"
    );
    expect(ggufQuantContract("Q8_0")?.packing.join(" ")).toContain("no bias");
    expect(story("onnx", "UINT8", 1, 1).source).not.toContain("−37");
    expect(story("onnx", "BOOL", 1, 1).source).toEqual(["false", "true"]);
    expect(story("onnx", "COMPLEX64", 1, 8).title).toContain(
      "two components"
    );
    const e8m0 = story("onnx", "FLOAT8E8M0", 1, 1);
    expect(e8m0.title).toContain("power of two");
    expect(e8m0.stages[1]?.detail).not.toContain("significand fields");
    expect(e8m0.stages[2]?.detail).toContain("independently");
    expect(ggufQuantContract("IQ1_S")?.codes.join(" ")).toContain(
      "no separate sign mask"
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
      const fields = ggufStorageLayout(layout.dtype);
      expect(layout.subBlocks * layout.valuesPerSubBlock).toBe(256);
      expect(
        layout.sections.reduce((total, section) => total + section.bytes, 0)
      ).toBe(layout.bytes);
      expect(
        layout.sections.map(({ name, type, count, bytes, role }) => ({
          name,
          type,
          count,
          bytes,
          role
        }))
      ).toEqual(
        fields
      );
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

  it("maps selected sub-block metadata to exact scales bytes and bits", () => {
    expect(kQuantMetadataBytes("Q2_K", 0)).toEqual([
      {
        index: 0,
        segments: [
          { label: "s[0]", from: 0, to: 3, tone: "scale" },
          { label: "m[0]", from: 4, to: 7, tone: "minimum" }
        ]
      }
    ]);
    expect(kQuantMetadataBytes("Q3_K", 12)).toEqual([
      {
        index: 4,
        segments: [
          { label: "s[12] low 4", from: 4, to: 7, tone: "scale" }
        ]
      },
      {
        index: 8,
        segments: [
          { label: "s[12] high 2", from: 6, to: 7, tone: "scale" }
        ]
      }
    ]);
    expect(kQuantMetadataBytes("Q4_K", 4)).toEqual([
      {
        index: 8,
        segments: [
          { label: "s[4] low 4", from: 0, to: 3, tone: "scale" },
          { label: "m[4] low 4", from: 4, to: 7, tone: "minimum" }
        ]
      },
      {
        index: 0,
        segments: [
          { label: "s[4] high 2", from: 6, to: 7, tone: "scale" }
        ]
      },
      {
        index: 4,
        segments: [
          { label: "m[4] high 2", from: 6, to: 7, tone: "minimum" }
        ]
      }
    ]);
    expect(kQuantMetadataBytes("Q6_K", 15)).toEqual([
      {
        index: 15,
        segments: [
          { label: "signed s[15]", from: 0, to: 7, tone: "scale" }
        ]
      }
    ]);
  });

  it("traces one code from source float through exact K-quant record bits", () => {
    expect(kQuantCodeExample("Q2_K", 15)).toMatchObject({
      source: "w[240] = 0.68",
      storage: expect.arrayContaining([
        "qs[48] bits 6…7 ← 10₂",
        "qs starts at record byte 16, so this slice is in byte 64"
      ])
    });
    expect(kQuantCodeExample("Q3_K", 15)).toMatchObject({
      source: "w[240] = −0.30",
      storage: expect.arrayContaining([
        "qs[48] bits 6…7 ← 01₂",
        "hmask[16] bit 7 ← 0"
      ])
    });
    expect(kQuantCodeExample("Q4_K", 7)).toMatchObject({
      source: "w[224] = 3.61",
      storage: expect.arrayContaining([
        "qs[96] high bits 4…7 ← 1001₂",
        "qs starts at record byte 16, so this nibble is in byte 112"
      ])
    });
    expect(kQuantCodeExample("Q5_K", 7)).toMatchObject({
      source: "w[224] = 7.39",
      storage: expect.arrayContaining([
        "low 4 bits 0010₂ → qs[96] bits 4…7",
        "fifth bit 1 → qh[0] bit 7"
      ])
    });
    expect(kQuantCodeExample("Q6_K", 15)).toMatchObject({
      source: "w[240] = −4.62",
      storage: expect.arrayContaining([
        "stored 21 = 010101₂; low 4 bits 0101₂ → ql[112] bits 4…7",
        "high 2 bits 01₂ → qh[48] bits 6…7"
      ])
    });
  });
});

describe("GGUF physical animation layouts", () => {
  it("gives every complex block an exact specialized lesson", () => {
    const blockEntries = GGUF_DTYPE_CATALOG.filter(
      (entry) => entry.blockBytes !== undefined
    );

    for (const entry of blockEntries) {
      const specialized =
        entry.dtype === "Q4_0" ||
        isKQuantDtype(entry.dtype) ||
        ggufQuantContract(entry.dtype) !== undefined;
      expect(specialized, `${entry.dtype} specialized contract`).toBe(true);
    }
    expect(GGUF_QUANT_CONTRACT_DTYPES).toHaveLength(21);
  });

  it("accounts for every parser-recognized block format byte-for-byte", () => {
    const blockEntries = GGUF_DTYPE_CATALOG.filter(
      (entry) => entry.blockBytes !== undefined
    );

    for (const entry of blockEntries) {
      const kLayout = isKQuantDtype(entry.dtype)
        ? K_QUANT_LAYOUTS[entry.dtype]
        : undefined;
      const contract = ggufQuantContract(entry.dtype);
      const fields = ggufStorageLayout(entry.dtype);
      const bytes = fields?.reduce((total, field) => total + field.bytes, 0);
      const values =
        contract?.values ??
        (kLayout
          ? kLayout.subBlocks * kLayout.valuesPerSubBlock
          : entry.dtype === "Q4_0"
            ? 32
            : undefined);
      expect(bytes, `${entry.dtype} physical layout`).toBe(entry.blockBytes);
      expect(values, `${entry.dtype} weight count`).toBe(entry.blockElements);
    }
  });

  it.each(GGUF_QUANT_CONTRACT_DTYPES)(
    "%s documents scope, parameters, codes, packing, and decode",
    (dtype) => {
      const contract = ggufQuantContract(dtype);
      expect(contract).toBeDefined();
      if (!contract) throw new Error(`Missing ${dtype} contract`);
      expect(contract.metadata.length).toBeGreaterThanOrEqual(3);
      expect(contract.codes.length).toBeGreaterThanOrEqual(3);
      expect(contract.packing.length).toBeGreaterThanOrEqual(3);
      expect(contract.decode).toBeTruthy();
      expect(contract.symbols.length).toBeGreaterThanOrEqual(4);
      for (const item of contract.symbols) {
        expect(item.symbol).toBeTruthy();
        expect(item.meaning).toBeTruthy();
        expect(item.source).toBeTruthy();
      }
      expect(contract.runtime).toBeTruthy();
      expect(contract.groups.count * contract.groups.values).toBe(
        contract.values
      );
    }
  );

  it("describes IQ2_S index and sign payloads within qs", () => {
    expect(
      ggufStorageLayout("IQ2_S")?.find((field) => field.name === "qs")?.role
    ).toContain("final 32 bytes hold sign masks");
  });

  it("traces every IQ1_S reconstruction symbol to storage or runtime data", () => {
    const symbols = ggufQuantContract("IQ1_S")?.symbols;
    expect(symbols?.map((item) => item.symbol)).toEqual([
      "w′",
      "d",
      "group",
      "subgrid",
      "s",
      "index",
      "signedGrid",
      "lane",
      "δ",
      "2s + 1"
    ]);
    expect(symbols?.find((item) => item.symbol === "index")?.source).toContain(
      "qs[4×group + subgrid]"
    );
    expect(
      symbols?.find((item) => item.symbol === "signedGrid")?.source
    ).toContain("iq1s_grid");
    expect(symbols?.find((item) => item.symbol === "lane")?.source).toContain(
      "32×group + 8×subgrid + lane"
    );
    expect(symbols?.find((item) => item.symbol === "δ")?.source).toContain(
      "bit 15"
    );
  });

  it("preserves special companion and microscale rounding contracts", () => {
    expect(ggufQuantContract("Q8_1")?.metadata.join(" ")).toContain(
      "rounded to FP16 independently"
    );
    expect(ggufQuantContract("MXFP4")?.metadata.join(" ")).toContain(
      "special denormal path for e = 0"
    );
    expect(ggufQuantContract("MXFP4")?.decode).toContain("E8M0_HALF");
  });

  it("explains Q5 bit-plane assembly as well as its operands", () => {
    expect(ggufQuantContract("Q5_0")?.symbols.map((item) => item.symbol)).toContain(
      "join(qh, qs)"
    );
  });
});
