# Changelog

## Unreleased

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
- Add editor-style previous/next tensor search navigation.
- Increase the default map resolution to approximately 128 rows per file.
- Infer ONNX external-data layouts from the manifest without requiring the
  referenced `.data` files.

## 0.1.0

- Initial GGUF, SafeTensors, and ONNX byte-map visualization.
- SafeTensors shard discovery and remote HTTP Range loading.
- Tensor details and on-demand value sampling.
