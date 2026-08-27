import type { ParsedFile, ParsedModel, TensorRecord } from "@weights-viz/core";

export const ADDRESS_GRID_COLUMNS = 64;
export const NARROW_ADDRESS_GRID_COLUMNS = 32;
export const ADDRESS_GRID_TARGET_ROWS = 128;
export const DRAG_THRESHOLD_PX = 4;

const GUTTER_WIDTH = 104;
const RIGHT_PADDING = 12;
const TOP_PADDING = 12;
const FILE_HEADER_HEIGHT = 30;
const FILE_GAP = 24;
const RATIO_SCALE = 1_000_000_000n;

export interface AddressRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface AddressSpan {
  start: bigint;
  end: bigint;
  kind: "metadata" | "tensor";
  visible: boolean;
  tensor?: TensorRecord;
  rects: AddressRect[];
}

export interface AddressFileLayout {
  file: ParsedFile;
  columns: number;
  bytesPerCell: bigint;
  bytesPerRow: bigint;
  rowCount: number;
  gridX: number;
  gridY: number;
  gridWidth: number;
  rowHeight: number;
  spans: AddressSpan[];
}

export interface AddressMapLayout {
  files: AddressFileLayout[];
  bytesPerCell: bigint;
  width: number;
  contentHeight: number;
}

export interface AddressHit {
  file: ParsedFile;
  address: bigint;
  cellStart: bigint;
  cellEnd: bigint;
  bytesPerCell: bigint;
  start: bigint;
  end: bigint;
  kind: "metadata" | "tensor" | "filtered" | "unmapped";
  tensor?: TensorRecord;
}

export function createAddressMapLayout(
  model: ParsedModel,
  width: number,
  includeTensor: (tensor: TensorRecord) => boolean = () => true,
  resolutionStep = 0,
  referenceFileSize?: bigint
): AddressMapLayout {
  const columns =
    width < 700 ? NARROW_ADDRESS_GRID_COLUMNS : ADDRESS_GRID_COLUMNS;
  const gridWidth = Math.max(columns, width - GUTTER_WIDTH - RIGHT_PADDING);
  const rowHeight = gridWidth / columns;
  const largestFileSize =
    referenceFileSize ??
    model.files.reduce(
      (largest, file) => maxBigInt(largest, file.size),
      0n
    );
  const bytesPerCell = scaleBytesPerCell(
    chooseBytesPerCell(largestFileSize, columns),
    resolutionStep
  );
  const files: AddressFileLayout[] = [];
  let y = TOP_PADDING;

  for (const file of model.files) {
    const bytesPerRow = bytesPerCell * BigInt(columns);
    const rowCount = Math.max(1, Number(ceilDiv(file.size, bytesPerRow)));
    const gridY = y + FILE_HEADER_HEIGHT;
    const layout: AddressFileLayout = {
      file,
      columns,
      bytesPerCell,
      bytesPerRow,
      rowCount,
      gridX: GUTTER_WIDTH,
      gridY,
      gridWidth,
      rowHeight,
      spans: []
    };

    layout.spans = createFileSpans(layout, includeTensor);
    files.push(layout);
    y = gridY + rowCount * rowHeight + FILE_GAP;
  }

  return {
    files,
    bytesPerCell,
    width,
    contentHeight: Math.max(y - FILE_GAP + TOP_PADDING, 1)
  };
}

export function chooseBytesPerCell(
  fileSize: bigint,
  columns = ADDRESS_GRID_COLUMNS,
  targetRows = ADDRESS_GRID_TARGET_ROWS
): bigint {
  if (fileSize <= 0n) return 1n;
  const targetCells = BigInt(Math.max(1, columns * targetRows));
  return nextPowerOfTwo(ceilDiv(fileSize, targetCells));
}

export function addressMapMaxScrollY(
  layout: AddressMapLayout,
  viewportHeight: number,
  zoom: number
): number {
  if (layout.contentHeight * zoom <= viewportHeight) return 0;
  const lastFile = layout.files.at(-1);
  if (!lastFile) return 0;
  return (
    (lastFile.gridY +
      Math.max(0, lastFile.rowCount - 1) * lastFile.rowHeight) *
    zoom
  );
}

function scaleBytesPerCell(bytesPerCell: bigint, step: number): bigint {
  if (step >= 0) return bytesPerCell << BigInt(step);
  return maxBigInt(1n, bytesPerCell >> BigInt(-step));
}

export function hitTestAddressMap(
  layout: AddressMapLayout,
  x: number,
  y: number
): AddressHit | undefined {
  const fileLayout = layout.files.find((candidate) => {
    const height = candidate.rowCount * candidate.rowHeight;
    return (
      x >= candidate.gridX &&
      x <= candidate.gridX + candidate.gridWidth &&
      y >= candidate.gridY &&
      y < candidate.gridY + height
    );
  });
  if (!fileLayout) return undefined;

  const row = Math.min(
    fileLayout.rowCount - 1,
    Math.max(0, Math.floor((y - fileLayout.gridY) / fileLayout.rowHeight))
  );
  const fraction = Math.min(
    1,
    Math.max(0, (x - fileLayout.gridX) / fileLayout.gridWidth)
  );
  const address =
    BigInt(row) * fileLayout.bytesPerRow +
    scaledBigInt(fileLayout.bytesPerRow, fraction);
  if (address >= fileLayout.file.size) return undefined;
  const column = Math.min(
    fileLayout.columns - 1,
    Math.max(0, Math.floor(fraction * fileLayout.columns))
  );
  const cellStart =
    BigInt(row) * fileLayout.bytesPerRow +
    BigInt(column) * fileLayout.bytesPerCell;
  const cellEnd = minBigInt(
    fileLayout.file.size,
    cellStart + fileLayout.bytesPerCell
  );
  const cell = {
    cellStart,
    cellEnd,
    bytesPerCell: fileLayout.bytesPerCell
  };

  const spanIndex = findSpanIndexAtOrBefore(fileLayout.spans, address);
  const span = spanIndex >= 0 ? fileLayout.spans[spanIndex] : undefined;
  if (span?.tensor && address < span.end) {
    return {
      file: fileLayout.file,
      address,
      ...cell,
      start: span.start,
      end: span.end,
      kind: span.visible ? "tensor" : "filtered",
      tensor: span.tensor
    };
  }

  if (span?.kind === "metadata" && address < span.end) {
    return {
      file: fileLayout.file,
      address,
      ...cell,
      start: span.start,
      end: span.end,
      kind: "metadata"
    };
  }

  const previous = spanIndex >= 0 ? fileLayout.spans[spanIndex] : undefined;
  const next = fileLayout.spans[spanIndex + 1];
  return {
    file: fileLayout.file,
    address,
    ...cell,
    start: previous?.end ?? 0n,
    end: next?.start ?? fileLayout.file.size,
    kind: "unmapped"
  };
}

export function isClickGesture(
  startX: number,
  startY: number,
  endX: number,
  endY: number
): boolean {
  return Math.hypot(endX - startX, endY - startY) < DRAG_THRESHOLD_PX;
}

function createFileSpans(
  layout: AddressFileLayout,
  includeTensor: (tensor: TensorRecord) => boolean
): AddressSpan[] {
  const allTensors = layout.file.tensors.flatMap((tensor) =>
    tensorSegments(tensor)
      .map((segment) => ({
        start: maxBigInt(0n, segment.byteOffset),
        end: minBigInt(
          layout.file.size,
          segment.byteOffset + segment.byteLength
        )
      }))
      .filter((segment) => segment.end > segment.start)
      .map<AddressSpan>((segment) => ({
        ...segment,
        kind: "tensor",
        visible: includeTensor(tensor),
        tensor,
        rects: projectRange(layout, segment.start, segment.end)
      }))
  );
  allTensors.sort((a, b) => compareBigInt(a.start, b.start));

  const firstTensorStart = allTensors[0]?.start ?? 0n;
  const metadata: AddressSpan[] =
    firstTensorStart > 0n
      ? [
          {
            start: 0n,
            end: firstTensorStart,
            kind: "metadata",
            visible: true,
            rects: projectRange(layout, 0n, firstTensorStart)
          }
        ]
      : [];
  return [...metadata, ...allTensors];
}

function tensorSegments(
  tensor: TensorRecord
): Array<{ byteOffset: bigint; byteLength: bigint }> {
  return tensor.byteSegments?.length
    ? tensor.byteSegments
    : [{ byteOffset: tensor.byteOffset, byteLength: tensor.byteLength }];
}

function projectRange(
  layout: AddressFileLayout,
  start: bigint,
  end: bigint
): AddressRect[] {
  if (end <= start) return [];
  const firstRow = start / layout.bytesPerRow;
  const lastRow = (end - 1n) / layout.bytesPerRow;
  const rects: AddressRect[] = [];

  for (let row = firstRow; row <= lastRow; row++) {
    const rowStart = row * layout.bytesPerRow;
    const segmentStart = maxBigInt(start, rowStart);
    const segmentEnd = minBigInt(end, rowStart + layout.bytesPerRow);
    const x =
      layout.gridX +
      ratio(segmentStart - rowStart, layout.bytesPerRow) * layout.gridWidth;
    const width =
      ratio(segmentEnd - segmentStart, layout.bytesPerRow) * layout.gridWidth;
    rects.push({
      x,
      y: layout.gridY + Number(row) * layout.rowHeight,
      width,
      height: layout.rowHeight
    });
  }
  return rects;
}

function nextPowerOfTwo(value: bigint): bigint {
  let result = 1n;
  while (result < value) result <<= 1n;
  return result;
}

function ceilDiv(value: bigint, divisor: bigint): bigint {
  return (value + divisor - 1n) / divisor;
}

function ratio(part: bigint, whole: bigint): number {
  if (whole <= 0n) return 0;
  return Number((part * RATIO_SCALE) / whole) / Number(RATIO_SCALE);
}

function scaledBigInt(value: bigint, fraction: number): bigint {
  const scaledFraction = BigInt(
    Math.floor(Math.min(1, Math.max(0, fraction)) * Number(RATIO_SCALE))
  );
  return (value * scaledFraction) / RATIO_SCALE;
}

function compareBigInt(left: bigint, right: bigint): number {
  return left === right ? 0 : left < right ? -1 : 1;
}

function findSpanIndexAtOrBefore(
  spans: AddressSpan[],
  address: bigint
): number {
  let low = 0;
  let high = spans.length - 1;
  let result = -1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const span = spans[middle]!;
    if (span.start <= address) {
      result = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return result;
}

function minBigInt(left: bigint, right: bigint): bigint {
  return left < right ? left : right;
}

function maxBigInt(left: bigint, right: bigint): bigint {
  return left > right ? left : right;
}
