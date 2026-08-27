import { describe, expect, it } from "vitest";

import {
  compareModels,
  type ParsedModel,
  type TensorRecord
} from "../src";

describe("compareModels", () => {
  it("correlates only tensors with exactly equal names", () => {
    const left = model([
      tensor("same", "F32", [2n], 8n),
      tensor("changed", "F32", [2n], 8n),
      tensor("left.name", "F32", [1n], 4n)
    ]);
    const right = model([
      tensor("same", "F32", [2n], 8n),
      tensor("changed", "F16", [2n], 4n),
      tensor("left_name", "F32", [1n], 4n)
    ]);

    const comparison = compareModels(left, right);

    expect(comparison.summary).toEqual({
      unchanged: 1,
      changed: 1,
      "left-only": 1,
      "right-only": 1,
      ambiguous: 0
    });
    expect(
      comparison.tensors.find((item) => item.name === "changed")
    ).toMatchObject({
      status: "changed",
      changes: { dtype: true, byteLength: true }
    });
    expect(
      comparison.tensors.find((item) => item.name === "left.name")?.status
    ).toBe("left-only");
    expect(
      comparison.tensors.find((item) => item.name === "left_name")?.status
    ).toBe("right-only");
  });

  it("marks duplicate exact names as ambiguous", () => {
    const comparison = compareModels(
      model([tensor("duplicate"), tensor("duplicate")]),
      model([tensor("duplicate")])
    );

    expect(comparison.summary.ambiguous).toBe(1);
    expect(comparison.tensors[0]?.status).toBe("ambiguous");
  });
});

function model(tensors: TensorRecord[]): ParsedModel {
  return {
    id: crypto.randomUUID(),
    name: "model",
    diagnostics: [],
    files: [
      {
        id: crypto.randomUUID(),
        name: "weights",
        format: "safetensors",
        size: 64n,
        metadata: {},
        diagnostics: [],
        tensors
      }
    ]
  };
}

function tensor(
  name: string,
  dtype = "F32",
  shape = [1n],
  byteLength = 4n
): TensorRecord {
  return {
    id: crypto.randomUUID(),
    name,
    fileId: "file",
    dtype,
    shape,
    byteOffset: 0n,
    byteLength,
    sampleSupport: "values"
  };
}
