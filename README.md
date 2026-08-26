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
| SafeTensors | All official dtypes, including packed F4/F6, plus header metadata, shape, and exact ranges | On-demand samples and statistics for scalar dtypes | `*.safetensors.index.json` automatically joins shards |
| GGUF v2/v3 | All current GGML tensor type IDs and block sizes, KV metadata, alignment, and exact ranges | On-demand samples for scalar and supported common quantized dtypes | Each GGUF is shown as one model |
| ONNX | All current `TensorProto.DataType` values through `INT2`, initializer shape, and exact external-data ranges | Metadata only | Referenced `.data` files become the visualized address spaces |

Recognizing a dtype and calculating its encoded byte length are separate from
numeric decoding. Known GGML quantization types remain accurately sized and
named even when value sampling is unavailable. Future unknown type IDs remain
visible using their declared file ranges and produce a diagnostic.

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

- Share a web visualization by passing the model URL as `?url=...`; the app
  loads it automatically and keeps successful URL loads in the address bar.
- GGUF and SafeTensors are read with progressive HTTP Range requests. Only the
  header, tensor directory, and explicitly requested sample ranges are fetched.
- Hugging Face `.../blob/...` file-page links are converted automatically to
  CORS-compatible `.../resolve/...` byte-range URLs. Repository home pages do
  not identify a file; choose a model file under **Files and versions** first.
- Public Hugging Face files work directly. Gated/private repositories require
  authentication, so download those files locally and open them from disk.
- SafeTensors index URLs resolve relative shard URLs automatically.
- The ONNX protobuf manifest is downloaded in full with a 50 MiB limit.
  Referenced external-data files are resolved relative to the manifest URL and
  accessed with HTTP Range requests; the byte map shows those physical files
  rather than the protobuf container.
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
the browser. The byte map is drawn on Canvas with visible-row culling rather
than creating one DOM node per weight, including for files up to 2 TiB.

VS Code uses true ranged reads for local filesystem files. Virtual filesystem
providers do not expose a ranged-read API; files above 64 MiB must be opened
from a local filesystem.

## Current visualization model

Each file is an independent linear address space. Addresses run left to right
and then wrap onto the next row. Every file in a model uses the same adaptive
power-of-two bytes-per-cell resolution, which can be adjusted from the
bottom-right controls. The 64-column background grid is only an address scale:
tensor fills retain their exact fractional start and end positions, can cross
rows, and leave alignment gaps visible. Multiple files are stacked vertically
instead of being assigned synthetic contiguous addresses.
Tensor hue represents dtype, while subtle shade variations distinguish inferred
roles such as attention, MLP/expert, embedding/output, normalization/bias, and
convolution.

Hovering reports the exact pointer address and the surrounding tensor,
metadata, or unmapped range. Drag the map with the primary pointer button to
pan. A wheel or two-finger trackpad gesture scrolls the map; trackpad pinch or
Ctrl/Cmd + wheel zooms around the pointer. Selecting a tensor opens its shape,
storage, exact addresses, and value-sampling controls in the inspector.
The Metadata tab exposes file-level model metadata, including searchable GGUF
KV entries such as architecture, model name, quantization details, and
tokenizer configuration.
