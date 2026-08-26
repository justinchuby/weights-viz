export function formatBytes(value: bigint): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let current = Number(value);
  let unit = 0;
  while (current >= 1024 && unit < units.length - 1) {
    current /= 1024;
    unit++;
  }
  return `${current < 10 && unit > 0 ? current.toFixed(2) : current.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

export function formatAddress(value: bigint): string {
  return `0x${value.toString(16).padStart(8, "0")}`;
}

export function formatShape(shape: bigint[]): string {
  return shape.length ? shape.map(String).join(" × ") : "scalar";
}

export function parameterCount(shape: bigint[]): bigint {
  return shape.reduce((count, dimension) => count * dimension, 1n);
}

export function formatParameterCount(shape: bigint[]): string {
  return parameterCount(shape).toLocaleString("en-US");
}
