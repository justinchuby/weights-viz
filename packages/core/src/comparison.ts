import type { ParsedModel, TensorRecord } from "./types";

export type TensorComparisonStatus =
  | "unchanged"
  | "changed"
  | "left-only"
  | "right-only"
  | "ambiguous";

export interface TensorComparison {
  name: string;
  status: TensorComparisonStatus;
  left?: TensorRecord;
  right?: TensorRecord;
  changes: {
    dtype: boolean;
    shape: boolean;
    parameters: boolean;
    byteLength: boolean;
    storage: boolean;
  };
}

export interface ModelComparison {
  tensors: TensorComparison[];
  summary: Record<TensorComparisonStatus, number>;
}

export function compareModels(
  left: ParsedModel,
  right: ParsedModel
): ModelComparison {
  const leftByName = indexTensors(left);
  const rightByName = indexTensors(right);
  const names = [
    ...leftByName.keys(),
    ...[...rightByName.keys()].filter((name) => !leftByName.has(name))
  ];
  const summary: ModelComparison["summary"] = {
    unchanged: 0,
    changed: 0,
    "left-only": 0,
    "right-only": 0,
    ambiguous: 0
  };
  const tensors = names.map<TensorComparison>((name) => {
    const leftMatches = leftByName.get(name) ?? [];
    const rightMatches = rightByName.get(name) ?? [];
    let comparison: TensorComparison;
    if (leftMatches.length > 1 || rightMatches.length > 1) {
      comparison = createComparison(
        name,
        "ambiguous",
        leftMatches[0],
        rightMatches[0]
      );
    } else if (!leftMatches.length) {
      comparison = createComparison(name, "right-only", undefined, rightMatches[0]);
    } else if (!rightMatches.length) {
      comparison = createComparison(name, "left-only", leftMatches[0], undefined);
    } else {
      const provisional = createComparison(
        name,
        "unchanged",
        leftMatches[0],
        rightMatches[0]
      );
      comparison = {
        ...provisional,
        status: Object.values(provisional.changes).some(Boolean)
          ? "changed"
          : "unchanged"
      };
    }
    summary[comparison.status] += 1;
    return comparison;
  });
  return { tensors, summary };
}

export function tensorComparisonKey(tensor: TensorRecord): string {
  return `${tensor.fileId}\0${tensor.id}`;
}

function indexTensors(model: ParsedModel): Map<string, TensorRecord[]> {
  const result = new Map<string, TensorRecord[]>();
  for (const tensor of model.files.flatMap((file) => file.tensors)) {
    const matches = result.get(tensor.name) ?? [];
    matches.push(tensor);
    result.set(tensor.name, matches);
  }
  return result;
}

function createComparison(
  name: string,
  status: TensorComparisonStatus,
  left?: TensorRecord,
  right?: TensorRecord
): TensorComparison {
  const leftParameters = left ? parameterCount(left) : undefined;
  const rightParameters = right ? parameterCount(right) : undefined;
  return {
    name,
    status,
    ...(left ? { left } : {}),
    ...(right ? { right } : {}),
    changes: {
      dtype: Boolean(left && right && left.dtype !== right.dtype),
      shape: Boolean(left && right && !sameShape(left.shape, right.shape)),
      parameters: Boolean(
        leftParameters !== undefined &&
          rightParameters !== undefined &&
          leftParameters !== rightParameters
      ),
      byteLength: Boolean(
        left && right && left.byteLength !== right.byteLength
      ),
      storage: Boolean(left && right && left.storage !== right.storage)
    }
  };
}

function sameShape(left: bigint[], right: bigint[]): boolean {
  return (
    left.length === right.length &&
    left.every((dimension, index) => dimension === right[index])
  );
}

function parameterCount(tensor: TensorRecord): bigint {
  return tensor.shape.reduce((product, dimension) => product * dimension, 1n);
}
