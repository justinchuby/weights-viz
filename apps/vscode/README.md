# Weights Viz for VS Code

Inspect GGUF, SafeTensors, and ONNX model weight layouts without leaving VS
Code.

## Features

- Open `.gguf`, `.safetensors`, `.onnx`, and `.safetensors.index.json` files
  with the **Weights Visualization** custom editor by default.
- Right-click a supported model or `textproto` file in the Explorer and choose
  **Visualize Weights** to open the viewer explicitly.
- View a Canvas byte map ordered by exact file address, with visible alignment
  gaps and drag-to-pan navigation.
- Inspect tensor names, shapes, dtypes, exact offsets, lengths, and storage.
- Open the **Dtype Atlas** or click a dtype label to learn its bit fields,
  packing, quantization blocks, scale selection, inference path, and kernel
  optimizations without loading a separate reference model.
- Search file-level metadata, including GGUF model and tokenizer KV entries.
- Sample values from SafeTensors and supported GGUF types on demand.
- Open `*.safetensors.index.json` to combine model shards automatically.
- Open any standard `model-00001-of-000NN.gguf` shard to discover and combine
  its sibling shards automatically.
- Visualize ONNX external-data address spaces inferred from the manifest without
  requiring the referenced `.data` files.
- Compare any two GGUF, SafeTensors, or ONNX models side by side. Exact matching
  tensor names are correlated; changed, added, and removed tensors are
  highlighted directly in the maps. This compares tensor metadata and encoded
  layout, not tensor values or raw file bytes.
- Run **Weights Viz: Compare Model Weights...**, or select the first model from
  Explorer with **Weights Viz: Select Model for Compare** and compare from the
  second model's context menu.
- Run **Weights Viz: Open Model Files** to pick model files with the native VS
  Code dialog, or use the **Open files** button inside the viewer.
- Run **Weights Viz: Open Model URL** to inspect public remote models with HTTP
  Range requests.

VS Code webviews cannot open the browser file dialog, so the viewer asks the
extension host for the native picker instead of using an HTML file input.

## Privacy

- Local model files are parsed on your machine and are never uploaded.
- The extension reads only files you explicitly open, SafeTensors shards
  referenced by an index, and model URLs you explicitly request.
- Parsed metadata and requested value samples pass only between the extension
  host and its local VS Code webview.
- The extension contains no telemetry, analytics, advertising, authentication,
  or secret-storage integration.
- Network requests occur only when you explicitly run **Open Model URL**.
  Requests go to that URL (and referenced remote shards), use HTTP Range where
  supported, and always omit credentials.

See the [project README](https://github.com/justinchuby/weights-viz#readme) for
the full support matrix and remote-server requirements.
