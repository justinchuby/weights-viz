import { describe, expect, it } from "vitest";
import { formatMetadataValue } from "./WeightsExplorer";

describe("formatMetadataValue", () => {
  it("formats bigint values without losing precision", () => {
    expect(formatMetadataValue(9_007_199_254_740_993n)).toBe(
      "9007199254740993"
    );
  });

  it("summarizes large GGUF metadata arrays", () => {
    const value = Array.from({ length: 20 }, (_, index) => `token-${index}`);
    const formatted = formatMetadataValue(value);

    expect(formatted).toContain('"token-0"');
    expect(formatted).toContain("… (+4)");
    expect(formatted).not.toContain("token-19");
  });

  it("serializes nested values containing bigint", () => {
    expect(formatMetadataValue({ count: 42n, enabled: true })).toBe(
      '{"count":"42","enabled":true}'
    );
  });
});
