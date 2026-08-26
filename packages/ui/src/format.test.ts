import { describe, expect, it } from "vitest";
import { formatParameterCount, parameterCount } from "./format";

describe("tensor parameter counts", () => {
  it("multiplies every shape dimension", () => {
    expect(parameterCount([1024n, 5120n])).toBe(5_242_880n);
    expect(formatParameterCount([1024n, 5120n])).toBe("5,242,880");
  });

  it("counts a scalar as one parameter", () => {
    expect(parameterCount([])).toBe(1n);
  });
});
