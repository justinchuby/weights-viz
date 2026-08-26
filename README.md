# Weights Viz

A local-first byte map for model weights. Open a GGUF, SafeTensors, or ONNX
model and explore its files, tensors, data types, shapes, byte ranges, and
sampled values without loading the whole model into memory.

**Live app:** https://www.justinchuby.com/weights-viz/

The project ships the same React visualization in two hosts:

- A static Vite web app with drag-and-drop, multi-file selection, and URL input.
- A VS Code custom editor for model files and SafeTensors indexes.

## Supported formats

| Format | Layout and metadata | Values | Multi-file / external data |
| --- | --- | --- | --- |
| SafeTensors | Header metadata, dtype, shape, relative and absolute ranges | On-demand samples and statistics for scalar dtypes | `*.safetensors.index.json` automatically joins shards |
| GGUF v2/v3 | KV metadata, alignment, tensor directory, GGML dtype and ranges | On-demand samples for scalar and supported common quantized dtypes | Each GGUF is shown as one model |
| ONNX | Graph initializer name, dtype, shape, inline range, and external-data declarations | Metadata only | External location, offset, and length are displayed but not fetched |

Unknown GGUF quantization types remain fully visible in the byte map and are
marked as unavailable for value decoding.

## Opening local files

The web app uses a real `<input type="file">`, drag and drop, and clipboard
paste. Embedded webviews such as the VS Code built-in browser and in-app
browsers silently ignore file inputs: the click produces no dialog and no
error. The app detects that case, because a real file dialog always moves focus
away from the page, and then shows an inline notice pointing at drag and drop
and the URL field.

In VS Code the viewer never relies on an HTML file input. **Open files** and
**Weights Viz: Open Model Files** ask the extension host for the native
`showOpenDialog` picker, which also works over Remote SSH and in Codespaces.

## Remote files

Paste a public model URL into the web app or run **Weights Viz: Open Model URL**
in VS Code.

- GGUF and SafeTensors are read with progressive HTTP Range requests. Only the
  header, tensor directory, and explicitly requested sample ranges are fetched.
- SafeTensors index URLs resolve relative shard URLs automatically.
- Remote ONNX files are downloaded in full because protobuf does not provide a
  separately addressable metadata header. The client enforces a 50 MiB limit.
- The static web app has no proxy. The origin must allow CORS, expose the
  `Content-Range` response header, and return valid `206 Partial Content`
  responses. A clear error is shown when these requirements are not met.
- Remote requests omit credentials and do not support custom authorization
  headers in the first release.

Local files never leave the device.

## Development

Requirements: Node.js 20 or newer and pnpm 10.

```sh
pnpm install
pnpm dev
```

The workspace contains:

```text
apps/web       Static web app
apps/vscode    VS Code extension and webview
packages/core  Random-access sources, format parsers, model loading
packages/ui    Shared React Canvas visualization
```

Build and test everything:

```sh
pnpm typecheck
pnpm test
pnpm build
```

The static site is emitted to `apps/web/dist`. The extension bundle and its
webview are emitted to `apps/vscode/dist`.

Create an installable VSIX:

```sh
pnpm package:vscode
```

Pushes to `main` deploy the static app through GitHub Pages. Version tags such
as `v0.1.0` create a GitHub release containing the VSIX.

## Performance and safety

Offsets and lengths use `bigint`, so multi-gigabyte files retain exact
addresses. Parsers perform bounded reads, validate declared ranges, and cap
metadata, strings, arrays, and remote downloads. Parsing runs in a web worker in
the browser. The byte map is drawn on Canvas and only adds a fixed grid at
higher zoom levels rather than creating one DOM node per weight.

VS Code uses true ranged reads for local filesystem files. Virtual filesystem
providers do not expose a ranged-read API; files above 64 MiB must be opened
from a local filesystem.

## Current visualization model

The map uses three semantic levels:

1. Model files.
2. Tensors sized by their on-disk byte length and colored by dtype.
3. Fixed byte regions shown after zooming in.

Hovering reports the tensor or region byte range. Selecting a tensor opens its
shape, storage, exact addresses, and value-sampling controls in the inspector.
The Metadata tab exposes file-level model metadata, including searchable GGUF
KV entries such as architecture, model name, quantization details, and
tokenizer configuration.
