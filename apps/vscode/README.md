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
- Search file-level metadata, including GGUF model and tokenizer KV entries.
- Sample values from SafeTensors and supported GGUF types on demand.
- Open `*.safetensors.index.json` to combine model shards automatically.
- Visualize ONNX initializers in the physical address spaces of their referenced
  external `.data` files.
- Run **Weights Viz: Open Model Files** to pick model files with the native VS
  Code dialog, or use the **Open files** button inside the viewer.
- Run **Weights Viz: Open Model URL** to inspect public remote models with HTTP
  Range requests.

VS Code webviews cannot open the browser file dialog, so the viewer asks the
extension host for the native picker instead of using an HTML file input.

## Privacy

- Local model files are parsed on your machine and are never uploaded.
- The extension reads only files you explicitly open, SafeTensors shards
  referenced by an index, and ONNX external-data files referenced by a manifest
  you open.
- Parsed metadata and requested value samples pass only between the extension
  host and its local VS Code webview.
- The extension contains no telemetry, analytics, advertising, authentication,
  or secret-storage integration.
- Network requests occur only when you explicitly run **Open Model URL**.
  Requests go to that URL (and referenced remote shards), use HTTP Range where
  supported, and always omit credentials.

See the [project README](https://github.com/justinchuby/weights-viz#readme) for
the full support matrix and remote-server requirements.
