import { describe, expect, it } from "vitest";
import { colorForDtype } from "./dtype-color";

describe("dtype colors", () => {
  it("assigns fixed colors to common floating-point dtypes", () => {
    expect(colorForDtype("BF16")).toBe("#6ee7ff");
    expect(colorForDtype("F32")).toBe("#69e6a6");
    expect(colorForDtype("FLOAT")).toBe("#69e6a6");
    expect(colorForDtype("F8_E4M3")).toBe("#9b8cff");
    expect(colorForDtype("FLOAT8E4M3FN")).toBe("#9b8cff");
  });

  it("keeps unknown dtype colors stable", () => {
    expect(colorForDtype("Q4_K")).toBe(colorForDtype("Q4_K"));
    expect(colorForDtype("Q4_K")).not.toBe(colorForDtype("Q5_K"));
  });
});
