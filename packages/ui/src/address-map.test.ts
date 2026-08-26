import type { ParsedFile, ParsedModel, TensorRecord } from "@weights-viz/core";
import { describe, expect, it } from "vitest";
import {
  chooseBytesPerCell,
  createAddressMapLayout,
  hitTestAddressMap,
  isClickGesture
} from "./address-map";

describe("address map", () => {
  it.each([
    [100n * 1024n ** 2n, 64n * 1024n],
    [1024n ** 3n, 512n * 1024n],
    [10n * 1024n ** 3n, 8n * 1024n ** 2n],
    [100n * 1024n ** 3n, 64n * 1024n ** 2n],
    [1024n ** 4n, 512n * 1024n ** 2n],
    [2n * 1024n ** 4n, 1024n ** 3n]
  ])("chooses a power-of-two scale for %s bytes", (size, expected) => {
    expect(chooseBytesPerCell(size)).toBe(expected);
  });

  it("projects fractional tensor boundaries and preserves gaps", () => {
    const model = makeModel(8n * 1024n ** 2n, [
      makeTensor("first", 516n * 1024n, 1024n * 1024n),
      makeTensor("second", 2n * 1024n ** 2n, 512n * 1024n)
    ]);
    const layout = createAddressMapLayout(model, 1132);
    const file = layout.files[0]!;
    const first = file.spans.find((span) => span.tensor?.name === "first")!;
    const gapAddress = 7n * 256n * 1024n;
    const gapRow = gapAddress / file.bytesPerRow;
    const gapX =
      file.gridX +
      (Number(gapAddress % file.bytesPerRow) / Number(file.bytesPerRow)) *
        file.gridWidth;
    const gapY =
      file.gridY + (Number(gapRow) + 0.5) * file.rowHeight;

    expect(first.rects[0]?.x).toBeGreaterThan(file.gridX);
    expect(first.rects[0]?.width).toBeGreaterThan(0);
    const hit = hitTestAddressMap(layout, gapX, gapY);
    expect(hit).toBeDefined();
    if (!hit) throw new Error("Expected an address hit");
    expect(hit.kind).toBe("unmapped");
    expect(hit.bytesPerCell).toBe(file.bytesPerCell);
    expect(hit.cellEnd - hit.cellStart).toBe(file.bytesPerCell);
  });

  it("wraps tensor fills across address rows", () => {
    const model = makeModel(256n * 1024n ** 2n, [
      makeTensor("wrapped", 3n * 1024n ** 2n, 12n * 1024n ** 2n)
    ]);
    const layout = createAddressMapLayout(model, 1132);
    const wrapped = layout.files[0]!.spans.find(
      (span) => span.tensor?.name === "wrapped"
    );

    expect(wrapped?.rects.length).toBeGreaterThan(1);
    expect(wrapped?.rects[0]?.x).toBeGreaterThan(layout.files[0]!.gridX);
    expect(wrapped?.rects.at(-1)?.width).toBeGreaterThan(0);
  });

  it("maps external ONNX tensors in their resolved data-file address space", () => {
    const tensor = {
      ...makeTensor("external", 32n, 16n),
      storage: "external" as const,
      externalLocation: "model.onnx.data",
      externalOffset: 32n,
      externalLength: 16n
    };
    const layout = createAddressMapLayout(makeModel(128n, [tensor]), 1132);

    expect(
      layout.files[0]?.spans.find((span) => span.tensor?.name === "external")
    ).toMatchObject({
      start: 32n,
      end: 48n,
      tensor: { name: "external" }
    });
  });

  it("stacks files as independent address spaces", () => {
    const first = makeFile("first", 100n * 1024n ** 2n, []);
    const second = makeFile("second", 2n * 1024n ** 4n, []);
    const layout = createAddressMapLayout(
      {
        id: "model",
        name: "model",
        diagnostics: [],
        files: [first, second]
      },
      1132
    );

    expect(layout.files[1]!.gridY).toBeGreaterThan(
      layout.files[0]!.gridY +
        layout.files[0]!.rowCount * layout.files[0]!.rowHeight
    );
    expect(layout.files[0]!.bytesPerRow).toBe(4n * 1024n ** 2n);
    expect(layout.files[1]!.bytesPerRow).toBe(64n * 1024n ** 3n);
  });

  it("filters tensors without extending the metadata range", () => {
    const model = makeModel(8n * 1024n ** 2n, [
      makeTensor("hidden", 256n * 1024n, 256n * 1024n),
      makeTensor("shown", 2n * 1024n ** 2n, 256n * 1024n)
    ]);
    const layout = createAddressMapLayout(
      model,
      1132,
      (tensor) => tensor.name === "shown"
    );
    const metadata = layout.files[0]!.spans.find(
      (span) => span.kind === "metadata"
    );

    expect(metadata?.end).toBe(256n * 1024n);
    const hidden = layout.files[0]!.spans.find(
      (span) => span.tensor?.name === "hidden"
    );
    expect(hidden?.visible).toBe(false);

    const file = layout.files[0]!;
    const hiddenAddress = 384n * 1024n;
    const row = hiddenAddress / file.bytesPerRow;
    const x =
      file.gridX +
      (Number(hiddenAddress % file.bytesPerRow) / Number(file.bytesPerRow)) *
        file.gridWidth;
    const y = file.gridY + (Number(row) + 0.5) * file.rowHeight;
    expect(hitTestAddressMap(layout, x, y)?.kind).toBe("filtered");
  });

  it("distinguishes clicks from drags at four pixels", () => {
    expect(isClickGesture(0, 0, 2, 2)).toBe(true);
    expect(isClickGesture(0, 0, 4, 0)).toBe(false);
  });
});

function makeModel(size: bigint, tensors: TensorRecord[]): ParsedModel {
  return {
    id: "model",
    name: "model",
    diagnostics: [],
    files: [makeFile("file", size, tensors)]
  };
}

function makeFile(
  name: string,
  size: bigint,
  tensors: TensorRecord[]
): ParsedFile {
  return {
    id: name,
    name,
    format: "gguf",
    size,
    metadata: {},
    diagnostics: [],
    tensors
  };
}

function makeTensor(
  name: string,
  byteOffset: bigint,
  byteLength: bigint
): TensorRecord {
  return {
    id: name,
    name,
    fileId: "file",
    dtype: "F32",
    shape: [byteLength / 4n],
    byteOffset,
    byteLength,
    sampleSupport: "values"
  };
}
