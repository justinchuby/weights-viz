# Changelog

## Unreleased

- Add playable encoding stories for every catalog dtype family, including
  floating-point fields, sub-byte packing, codebooks, ternary weights,
  microscaling, companion blocks, and schema-only types.
- Expand Q4_0 with its exact 18-byte block contract, FP16 scale derivation,
  implicit bias, signed codes, nibble pairing, reconstruction error, and fused
  dot-product execution.
- Add byte-accurate super-block animations for Q2_K through Q6_K, including
  fixed sub-block geometry, compressed local scales/minima, code bit-planes,
  physical field order, parameter derivation, and reconstruction formulas.
- Give every complex GGML block format a dedicated ABI contract lesson covering
  fixed grouping, exact struct fields, metadata derivation, code semantics,
  bit-plane or codebook packing, and kernel reconstruction.
- Map selected K-quant sub-block scales and minima to their exact `scales[]`
  array indices, record byte offsets, and packed bit ranges.
- Stop dtype lesson autoplay on its final step instead of looping; explicit
  replay starts again from the beginning.
- Keep completed animation stages at full brightness so the accumulated lesson
  remains readable as later stages play.
- Trace every symbol in complex GGUF reconstruction formulas back to its exact
  record field, packed bits, fixed runtime table, lane index, or format constant.
- Define every K-quant physical field and show a concrete source float becoming
  a q code, entering exact `qs`/`ql`/`qh`/`hmask` bits, and decoding again.
- Move the complete field glossary, formula vocabulary, byte ranges, code
  ranges, and bit-layout rules into the top Storage contract; animations now
  reference that primer and focus on transformations.
- Draw each K-quant physical super-block record together with its logical
  sub-block ranges and exact shared-array bit locations, distinguish one
  logical `q` from the packed `qs` byte array, and enlarge lesson typography.
- Trace encoder-only local parameters `a[g]` and `b[g]` through rounding into
  stored `s[g]` and `m[g]`, including concrete `scales[]` bits and record bytes.
- Reorder every complex GGUF lesson from fields and terms through production,
  layout, and reconstruction, then animate one concrete lane instead of
  repeating the Storage contract definitions.

## 0.3.0

- Add a standalone Dtype Atlas covering all 92 recognized SafeTensors, GGUF,
  and ONNX format entries without requiring a model file.
- Add interactive dtype lessons for bit fields, sub-byte packing, quantization
  blocks, scales, offsets, effective bits per weight, and decode formulas.
- Explain how GGUF quantization parameters are selected and how specialized
  kernels fuse unpacking, dequantization, and dot products during inference.
- Allow the Canvas to zoom down to 25% while keeping address labels legible.
- Improve PWA icons for macOS masking and add a comparison walkthrough GIF.
- Upgrade the build toolchain to Vite 8, TypeScript 7, Vitest 4, and pnpm 11,
  along with the latest React, VS Code, and extension packaging dependencies.

## 0.2.0

- Add side-by-side model comparison with exact tensor-name correlation, shared
  resolution, synchronized search selection, and changed/only-side highlighting.
- Compare GGUF, SafeTensors, and ONNX models in any format combination.
- Group conventionally named sharded GGUF files and discover sibling shards when
  opening any shard in VS Code.
- Add Explorer and Command Palette workflows for selecting and comparing models.
- Make the web app installable as an offline-capable PWA and allow multiple
  remote model URLs to be loaded together.

## 0.1.2

- Allow the final address row to scroll to the top of the map.
- Add editor-style previous/next tensor search navigation.
- Increase the default map resolution to approximately 128 rows per file.
- Infer ONNX external-data layouts from the manifest without requiring the
  referenced `.data` files.
- Keep the Canvas and inspector within the VS Code webview when its width or
  height shrinks.
- Show the opened model filename in VS Code editor tab titles.

## 0.1.1

- Add **Weights Viz: Open Model Files** and an in-viewer **Open files** button
  that use the native VS Code file dialog, because webviews cannot open the
  browser file picker.
- Add the Explorer **Visualize Weights** action for supported model and
  TextProto files.
- Add shareable remote model URLs and faster loading for large sharded models.
- Add unified multi-file resolution, stable dtype colors, semantic tensor-role
  shading, parameter counts, and responsive Canvas sizing.
- Add wheel/pinch navigation and a synchronized vertical map scrollbar.

## 0.1.0

- Initial GGUF, SafeTensors, and ONNX byte-map visualization.
- SafeTensors shard discovery and remote HTTP Range loading.
- Tensor details and on-demand value sampling.
