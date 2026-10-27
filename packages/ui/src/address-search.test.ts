import { compareModels, type ParsedModel, type TensorRecord } from "@weights-viz/core";
import { describe, expect, it } from "vitest";
import {
  findAddressMatches,
  findComparisonMatches,
  searchesGaps,
  tensorMatchesQuery
} from "./address-search";

function model(id = "model"): ParsedModel {
  const tensors: TensorRecord[] = [
    { id: "bubble", name: "bubble_projection", fileId: "file", dtype: "F32",
      shape: [1n], byteOffset: 8n, byteLength: 4n, sampleSupport: "values" },
    { id: "gap", name: "gap_bias", fileId: "file", dtype: "F16",
      shape: [2n], byteOffset: 24n, byteLength: 4n, sampleSupport: "values" }
  ];
  return {
    id, name: id, diagnostics: [],
    files: [{ id: "file", name: `${id}.gguf`, format: "gguf", size: 32n,
      metadata: {}, diagnostics: [], tensors }]
  };
}

describe("address search", () => {
  it.each(["gap", "Gap", " gaps ", "bubble", "BUBBLE", "bubbles"])("recognizes gap alias %j", (query) => {
    expect(searchesGaps(query)).toBe(true);
    expect(findAddressMatches(model(), query).filter((match) => match.kind === "gap")).toEqual([
      { kind: "gap", gap: { fileId: "file", start: 12n, end: 24n } },
      { kind: "gap", gap: { fileId: "file", start: 28n, end: 32n } }
    ]);
  });

  it("keeps normal tensor name/dtype matching and ordering unchanged", () => {
    const source = model();
    expect(findAddressMatches(source, " projection ")).toEqual([
      { kind: "tensor", tensor: source.files[0]!.tensors[0] }
    ]);
    expect(findAddressMatches(source, "f16")).toEqual([
      { kind: "tensor", tensor: source.files[0]!.tensors[1] }
    ]);
    expect(findAddressMatches(source, "bub")).toEqual([
      { kind: "tensor", tensor: source.files[0]!.tensors[0] }
    ]);
    expect(findAddressMatches(source, "missing")).toEqual([]);
    expect(findAddressMatches(source, " ")).toEqual([]);
    expect(tensorMatchesQuery(source.files[0]!.tensors[0]!, "")).toBe(true);
  });

  it("includes matching tensor names rather than reserving alias queries for gaps", () => {
    const source = model();
    const matches = findAddressMatches(source, "gap");
    expect(matches[0]).toEqual({ kind: "tensor", tensor: source.files[0]!.tensors[1] });
    expect(matches.filter((match) => match.kind === "gap")).toHaveLength(2);
    expect(findAddressMatches(source, "gap_bias")).toHaveLength(1);
    expect(searchesGaps("gap_bias")).toBe(false);
  });

  it("preserves independent file identities and physical gap order", () => {
    const source = model();
    source.files.push({
      ...source.files[0]!, id: "other", name: "other.gguf",
      tensors: [{ ...source.files[0]!.tensors[0]!, fileId: "other" }]
    });
    expect(findAddressMatches(source, "gaps")).toEqual([
      { kind: "gap", gap: { fileId: "file", start: 12n, end: 24n } },
      { kind: "gap", gap: { fileId: "file", start: 28n, end: 32n } },
      { kind: "gap", gap: { fileId: "other", start: 12n, end: 32n } }
    ]);
  });

  it("returns no invented gaps for a fully covered file", () => {
    const source = model();
    source.files[0]!.tensors = [{ ...source.files[0]!.tensors[0]!, byteOffset: 0n, byteLength: 32n }];
    expect(findAddressMatches(source, "gaps")).toEqual([]);
  });

  it("finds the encoder's 17,247-byte external alignment gap without its data file", () => {
    const source = model();
    const file = source.files[0]!;
    file.name = "encoder.onnx.data";
    file.format = "onnx";
    file.size = 140705796n;
    file.tensors = [
      { ...file.tensors[0]!, name: "before", dtype: "UINT8", shape: [140688545n],
        byteOffset: 0n, byteLength: 140688545n, storage: "external" },
      { ...file.tensors[1]!, name: "val_201_Q4", shape: [1n],
        byteOffset: 140705792n, byteLength: 4n, storage: "external" }
    ];
    expect(findAddressMatches(source, "bubble")).toEqual([
      { kind: "gap", gap: { fileId: file.id, start: 140688545n, end: 140705792n } }
    ]);
  });
});

describe("comparison gap search", () => {
  it("keeps left/right gaps independent even when their file IDs and offsets match", () => {
    const left = model("left");
    const right = model("right");
    const comparison = compareModels(left, right);
    expect(findComparisonMatches(comparison, left, right, "bubbles")).toEqual([
      { kind: "gap", side: "left", gap: { fileId: "file", start: 12n, end: 24n } },
      { kind: "gap", side: "left", gap: { fileId: "file", start: 28n, end: 32n } },
      { kind: "gap", side: "right", gap: { fileId: "file", start: 12n, end: 24n } },
      { kind: "gap", side: "right", gap: { fileId: "file", start: 28n, end: 32n } }
    ]);
  });

  it("retains paired tensor matches alongside gap aliases", () => {
    const left = model("left");
    const right = model("right");
    const comparison = compareModels(left, right);
    const matches = findComparisonMatches(comparison, left, right, "gap");
    expect(matches[0]).toMatchObject({ kind: "tensor", pair: { name: "gap_bias" } });
    expect(matches).toHaveLength(5);
    expect(findComparisonMatches(comparison, left, right, "F32")).toEqual([
      { kind: "tensor", pair: comparison.tensors.find((pair) => pair.name === "bubble_projection") }
    ]);
    expect(findComparisonMatches(comparison, left, right, "")).toEqual([]);
  });
});
