import { describe, expect, it } from "vitest";

import {
  GGUF_DTYPE_CATALOG,
  ONNX_DTYPE_CATALOG,
  SAFETENSORS_DTYPE_CATALOG
} from "../src";

describe("dtype catalogs", () => {
  it("exposes every parser catalog with stable unique identifiers", () => {
    expect(SAFETENSORS_DTYPE_CATALOG).toHaveLength(22);
    expect(ONNX_DTYPE_CATALOG).toHaveLength(27);
    expect(GGUF_DTYPE_CATALOG).toHaveLength(43);

    for (const catalog of [
      SAFETENSORS_DTYPE_CATALOG,
      ONNX_DTYPE_CATALOG,
      GGUF_DTYPE_CATALOG
    ]) {
      expect(new Set(catalog.map((entry) => entry.dtype)).size).toBe(
        catalog.length
      );
    }
  });

  it("includes recent packed, microscaled, and ternary formats", () => {
    expect(SAFETENSORS_DTYPE_CATALOG).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ dtype: "F4", bitsPerValue: 4 }),
        expect.objectContaining({ dtype: "F6_E2M3", bitsPerValue: 6 }),
        expect.objectContaining({ dtype: "F8_E8M0", bitsPerValue: 8 })
      ])
    );
    expect(ONNX_DTYPE_CATALOG).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ dtype: "FLOAT4E2M1", typeId: 23 }),
        expect.objectContaining({ dtype: "INT2", typeId: 26 })
      ])
    );
    expect(GGUF_DTYPE_CATALOG).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          dtype: "TQ1_0",
          blockBytes: 54,
          blockElements: 256
        }),
        expect.objectContaining({
          dtype: "MXFP4",
          blockBytes: 17,
          blockElements: 32
        }),
        expect.objectContaining({
          dtype: "NVFP4",
          blockBytes: 36,
          blockElements: 64
        })
      ])
    );
  });
});
