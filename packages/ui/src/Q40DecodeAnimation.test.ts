import { describe, expect, it } from "vitest";
import { Q40_DEMO_PAIRS, quantizeQ40 } from "./Q40DecodeAnimation";

describe("Q4_0 animation data", () => {
  it("rounds and clamps signed four-bit codes", () => {
    expect(quantizeQ40(-3, 0.25)).toBe(-8);
    expect(quantizeQ40(-0.57, 0.25)).toBe(-2);
    expect(quantizeQ40(1.29, 0.25)).toBe(5);
    expect(quantizeQ40(3, 0.25)).toBe(7);
  });

  it("packs q[i] low and q[i+16] high like GGML Q4_0", () => {
    expect(Q40_DEMO_PAIRS.map(({ packedByte }) => packedByte)).toEqual([
      0x90, 0xb3, 0xd6, 0xf8
    ]);
  });

  it("decodes each signed code with the shared block scale", () => {
    expect(Q40_DEMO_PAIRS[1]).toMatchObject({
      lowCode: -5,
      highCode: 3,
      decodedLow: -1.25,
      decodedHigh: 0.75
    });
  });
});
