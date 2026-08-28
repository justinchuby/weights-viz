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
  kQuantContractDetails,
  kQuantFieldMeaning,
  kQuantMetadataBytes,
  kQuantSubBlockStorage
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
    const iq4Nl = ggufQuantContract("IQ4_NL");
    expect(iq4Nl?.symbols.find(({ symbol }) => symbol === "group")?.source).toContain(
      "whole 32-weight record"
    );
    expect(iq4Nl?.symbols.find(({ symbol }) => symbol === "position")?.source).toContain(
      "position = i mod 32"
    );
    expect(iq4Nl?.packing).toContain(
      "position 16…31 → high nibble of qs[position − 16]"
    );
    expect(iq4Nl?.codes.join(" ")).toContain(
      "no multi-value grid and therefore no grid lane"
    );
    expect(iq4Nl?.worked.stages[0].detail).toContain(
      "0 × 32 + 25 = weight 25"
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

  it("separates metadata-group position from codebook grid lane", () => {
    const gridDtypes = new Set([
      "IQ2_XXS",
      "IQ2_XS",
      "IQ3_XXS",
      "IQ1_S",
      "IQ3_S",
      "IQ2_S",
      "IQ1_M"
    ]);

    for (const dtype of GGUF_QUANT_CONTRACT_DTYPES) {
      const contract = ggufQuantContract(dtype)!;
      expect(contract.symbols.some(({ symbol }) => symbol === "group")).toBe(true);
      expect(contract.symbols.some(({ symbol }) => symbol === "position")).toBe(true);
      const lane = contract.symbols.find(({ symbol }) => symbol === "lane");
      if (gridDtypes.has(dtype)) {
        expect(lane?.source).toContain("lane = position mod");
      } else {
        expect(lane).toBeUndefined();
      }
    }

    const iq3s = ggufQuantContract("IQ3_S")!;
    expect(iq3s.worked.stages[0].detail).toContain(
      "sign-mask position 6, and four-value grid lane = 30 mod 4 = 2"
    );
    expect(iq3s.worked.stages[2].accesses).toContainEqual({
      field: "signs",
      index: "11",
      bits: "bit 6",
      action: "read sign-mask position 6"
    });
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

  it("maps Q4_K logical sub-blocks back to shared physical fields", () => {
    expect(kQuantSubBlockStorage("Q4_K", 2)).toEqual({
      metadata: [
        "s[2]: scales[2] bits 0…5 (record byte 6)",
        "m[2]: scales[6] bits 0…5 (record byte 10)"
      ],
      codes: [
        "q low 4: qs[32…63] low bits 0…3 (record bytes 48…79)"
      ]
    });
    expect(kQuantSubBlockStorage("Q4_K", 3).codes).toEqual([
      "q low 4: qs[32…63] high bits 4…7 (record bytes 48…79)"
    ]);
  });

  it.each(Object.keys(K_QUANT_LAYOUTS) as Array<keyof typeof K_QUANT_LAYOUTS>)(
    "%s defines fields, terms, code ranges, metadata, and bit layout before animation",
    (dtype) => {
      const details = kQuantContractDetails(dtype);
      expect(details.metadata.length).toBeGreaterThanOrEqual(3);
      expect(details.codes.length).toBeGreaterThanOrEqual(3);
      expect(details.packing.length).toBeGreaterThanOrEqual(3);
      expect(details.derivation).toHaveLength(5);
      expect(details.terms.map((term) => term.symbol)).toEqual(
        expect.arrayContaining([
          "super-block",
          "g / sub-block",
          "lane / l",
          "w[i]",
          "w′[i]",
          "d / globalScale",
          "q / code",
          "activation",
          "Σ"
        ])
      );
      for (const field of K_QUANT_LAYOUTS[dtype].sections) {
        expect(kQuantFieldMeaning(field.name)).not.toBe(field.name);
      }
    }
  );

  it("traces Q4_K fitted parameters into stored scales bits", () => {
    const details = kQuantContractDetails("Q4_K");
    expect(details.derivation.map((step) => step.expression)).toEqual([
      "w[g,l] ≈ a[g] × q[g,l] − b[g]",
      "a[g] ≈ d × s[g] · b[g] ≈ dmin × m[g]",
      "s[g] = clamp(round(a[g] / d), 0, 63) · m[g] = clamp(round(b[g] / dmin), 0, 63)",
      "s[g], m[g] → scales[] bits",
      "localScale[g] = d × s[g] · localMin[g] = dmin × m[g]"
    ]);
    expect(details.derivation[2]?.detail).toContain(
      "a[0]=0.42 and d=0.01 give s[0]=42"
    );
    expect(details.derivation[3]?.detail).toContain(
      "s[0]: scales[0] bits 0…5 (record byte 4)"
    );
    expect(details.terms.find(({ symbol }) => symbol === "b[g]")?.source).toContain(
      "quantized into m[g]"
    );
    expect(
      details.terms.find(({ symbol }) => symbol === "s[g] / subScale")?.source
    ).toContain("s[g]=clamp(round(a[g]/d))");
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

  it.each(GGUF_QUANT_CONTRACT_DTYPES)(
    "%s traces one concrete position through source, metadata, storage, and reconstruction",
    (dtype) => {
      const contract = ggufQuantContract(dtype);
      const fields = ggufStorageLayout(dtype);
      expect(contract).toBeDefined();
      expect(fields).toBeDefined();
      if (!contract || !fields) throw new Error(`Missing ${dtype} contract`);

      const { selection, stages } = contract.worked;
      expect(stages.map(({ kind }) => kind)).toEqual([
        "source",
        "metadata",
        "storage",
        "reconstruction"
      ]);
      expect(selection.group).toBeGreaterThanOrEqual(0);
      expect(selection.group).toBeLessThan(contract.groups.count);
      expect(selection.position).toBeGreaterThanOrEqual(0);
      expect(selection.position).toBeLessThan(contract.groups.values);
      expect(selection.weight).toBe(
        selection.group * contract.groups.values + selection.position
      );

      const symbolNames = new Set(contract.symbols.map(({ symbol }) => symbol));
      for (const stage of stages) {
        expect(stage.title).toBeTruthy();
        expect(stage.detail).toBeTruthy();
        expect(stage.symbols.length).toBeGreaterThan(0);
        for (const symbol of stage.symbols) {
          expect(symbolNames.has(symbol), `${dtype} defines ${symbol}`).toBe(true);
        }
      }

      const storage = stages[2];
      expect(storage.kind).toBe("storage");
      if (storage.kind !== "storage") throw new Error(`Missing ${dtype} storage stage`);
      expect(storage.accesses.length).toBeGreaterThanOrEqual(2);
      const fieldNames = new Set(fields.map(({ name }) => name));
      for (const access of storage.accesses) {
        expect(fieldNames.has(access.field), `${dtype} field ${access.field}`).toBe(
          true
        );
        expect(access.index).toBeTruthy();
        expect(access.bits).toBeTruthy();
        expect(access.action).toBeTruthy();
      }
      expect(stages[3].symbols.some((symbol) => symbol.startsWith("w′"))).toBe(
        true
      );
    }
  );

  it("describes IQ2_S index and sign payloads within qs", () => {
    expect(
      ggufStorageLayout("IQ2_S")?.find((field) => field.name === "qs")?.role
    ).toContain("final 32 bytes hold sign masks");
  });

  it("reads Q5_0 position 18's fifth bit from the uint32 qh plane", () => {
    const storage = ggufQuantContract("Q5_0")?.worked.stages[2];
    expect(storage?.kind).toBe("storage");
    if (storage?.kind !== "storage") throw new Error("Missing Q5_0 storage stage");
    expect(storage.accesses).toContainEqual({
      field: "qh",
      index: "little-endian uint32 view of bytes 0…3",
      bits: "bit 18",
      action: "take position 18’s storedCode bit 4"
    });
  });

  it("traces every IQ1_S reconstruction symbol to storage or runtime data", () => {
    const symbols = ggufQuantContract("IQ1_S")?.symbols;
    expect(symbols?.map((item) => item.symbol)).toEqual([
      "w[i]",
      "position",
      "w′",
      "d",
      "i",
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
    expect(symbols?.find((item) => item.symbol === "position")?.source).toContain(
      "position = i mod 32"
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
