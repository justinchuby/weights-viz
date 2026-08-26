import { ParseError } from "./errors";

const decoder = new TextDecoder();

export class BinaryReader {
  private readonly view: DataView;
  offset = 0;

  constructor(
    readonly bytes: Uint8Array,
    readonly littleEndian = true
  ) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  get remaining(): number {
    return this.bytes.byteLength - this.offset;
  }

  seek(offset: number): void {
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > this.bytes.byteLength) {
      throw new ParseError(`Invalid buffer offset ${offset}`);
    }
    this.offset = offset;
  }

  skip(length: number): void {
    this.ensure(length);
    this.offset += length;
  }

  u8(): number {
    this.ensure(1);
    return this.view.getUint8(this.offset++);
  }

  i8(): number {
    this.ensure(1);
    return this.view.getInt8(this.offset++);
  }

  u16(): number {
    this.ensure(2);
    const value = this.view.getUint16(this.offset, this.littleEndian);
    this.offset += 2;
    return value;
  }

  i16(): number {
    this.ensure(2);
    const value = this.view.getInt16(this.offset, this.littleEndian);
    this.offset += 2;
    return value;
  }

  u32(): number {
    this.ensure(4);
    const value = this.view.getUint32(this.offset, this.littleEndian);
    this.offset += 4;
    return value;
  }

  i32(): number {
    this.ensure(4);
    const value = this.view.getInt32(this.offset, this.littleEndian);
    this.offset += 4;
    return value;
  }

  f32(): number {
    this.ensure(4);
    const value = this.view.getFloat32(this.offset, this.littleEndian);
    this.offset += 4;
    return value;
  }

  f64(): number {
    this.ensure(8);
    const value = this.view.getFloat64(this.offset, this.littleEndian);
    this.offset += 8;
    return value;
  }

  u64(): bigint {
    this.ensure(8);
    const value = this.view.getBigUint64(this.offset, this.littleEndian);
    this.offset += 8;
    return value;
  }

  i64(): bigint {
    this.ensure(8);
    const value = this.view.getBigInt64(this.offset, this.littleEndian);
    this.offset += 8;
    return value;
  }

  text(length: number): string {
    this.ensure(length);
    const result = decoder.decode(
      this.bytes.subarray(this.offset, this.offset + length)
    );
    this.offset += length;
    return result;
  }

  private ensure(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.remaining) {
      throw new ParseError(
        `Need ${length} bytes at buffer offset ${this.offset}, only ${this.remaining} remain`
      );
    }
  }
}

export function bigintToSafeNumber(value: bigint, label: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new ParseError(`${label} is too large to address in memory: ${value}`);
  }
  return Number(value);
}

export function product(values: bigint[]): bigint {
  return values.reduce((result, value) => result * value, 1n);
}

export function sampleStats(values: number[], totalElements: bigint) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
  }
  return {
    values,
    sampledElements: values.length,
    totalElements,
    min: values.length ? min : Number.NaN,
    max: values.length ? max : Number.NaN,
    mean: values.length ? sum / values.length : Number.NaN
  };
}
