# Weights Viz for VS Code

Inspect GGUF, SafeTensors, and ONNX model weight layouts without leaving VS
Code.

## Features

- Open model files with the **Weights Visualization** custom editor.
- View a proportional Canvas byte map organized by file, tensor, and byte block.
- Inspect tensor names, shapes, dtypes, exact offsets, lengths, and storage.
- Search file-level metadata, including GGUF model and tokenizer KV entries.
- Sample values from SafeTensors and supported GGUF types on demand.
- Open `*.safetensors.index.json` to combine model shards automatically.
- Inspect ONNX initializer and external-data location metadata.
- Run **Weights Viz: Open Model Files** to pick model files with the native VS
  Code dialog, or use the **Open files** button inside the viewer.
- Run **Weights Viz: Open Model URL** to inspect public remote models with HTTP
  Range requests.

VS Code webviews cannot open the browser file dialog, so the viewer asks the
extension host for the native picker instead of using an HTML file input.

Model files are parsed locally. Remote requests omit credentials.

See the [project README](https://github.com/justinchuby/weights-viz#readme) for
the full support matrix and remote-server requirements.
