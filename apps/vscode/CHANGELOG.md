# Changelog

## Unreleased

- Add a playable Q4_0 decode lesson showing block scaling, signed quantization,
  GGML nibble packing, reconstruction error, and fused dot-product execution.

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
