export class ParseError extends Error {
  constructor(
    message: string,
    readonly offset?: bigint
  ) {
    super(message);
    this.name = "ParseError";
  }
}

export function assertRange(
  offset: bigint,
  length: bigint,
  size: bigint,
  label: string
): void {
  if (offset < 0n || length < 0n || offset + length > size) {
    throw new ParseError(
      `${label} range ${offset}..${offset + length} exceeds file size ${size}`,
      offset
    );
  }
}
