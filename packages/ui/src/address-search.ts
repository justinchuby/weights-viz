import type { ModelComparison, ParsedModel, TensorComparison, TensorRecord } from "@weights-viz/core";
import { fileAddressGaps, type AddressGap } from "./address-map";

export type AddressSearchMatch =
  | { kind: "tensor"; tensor: TensorRecord }
  | { kind: "gap"; gap: AddressGap };

export type ComparisonSearchMatch =
  | { kind: "tensor"; pair: TensorComparison }
  | { kind: "gap"; gap: AddressGap; side: "left" | "right" };

export function searchesGaps(query: string): boolean {
  return /^(gaps?|bubbles?)$/.test(query.trim().toLowerCase());
}

export function tensorMatchesQuery(tensor: TensorRecord, query: string): boolean {
  const needle = query.trim().toLowerCase();
  return tensor.name.toLowerCase().includes(needle) || tensor.dtype.toLowerCase().includes(needle);
}

export function findAddressMatches(model: ParsedModel, query: string): AddressSearchMatch[] {
  if (!query.trim()) return [];
  return model.files.flatMap((file) => [
    ...file.tensors
      .filter((tensor) => tensorMatchesQuery(tensor, query))
      .map((tensor): AddressSearchMatch => ({ kind: "tensor", tensor })),
    ...(searchesGaps(query) ? fileAddressGaps(file).map((gap): AddressSearchMatch => ({
      kind: "gap", gap
    })) : [])
  ]);
}

export function findComparisonMatches(
  comparison: ModelComparison,
  left: ParsedModel,
  right: ParsedModel,
  query: string
): ComparisonSearchMatch[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];
  const matches: ComparisonSearchMatch[] = comparison.tensors
    .filter((pair) =>
      pair.name.toLowerCase().includes(needle) ||
      pair.left?.dtype.toLowerCase().includes(needle) ||
      pair.right?.dtype.toLowerCase().includes(needle)
    )
    .map((pair) => ({ kind: "tensor", pair }));
  if (searchesGaps(query)) {
    for (const [side, model] of [["left", left], ["right", right]] as const) {
      for (const file of model.files) {
        for (const gap of fileAddressGaps(file)) matches.push({ kind: "gap", gap, side });
      }
    }
  }
  return matches;
}
