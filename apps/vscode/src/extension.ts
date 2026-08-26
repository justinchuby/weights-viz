import * as path from "node:path";
import { open } from "node:fs/promises";
import * as vscode from "vscode";
import {
  detectFormat,
  GgufParser,
  loadModelUrl,
  loadSources,
  OnnxParser,
  SafeTensorsParser,
  type Parser,
  type RandomAccessSource,
  type TensorRecord,
  type WeightFormat
} from "@weights-viz/core";

class VscodeFileSource implements RandomAccessSource {
  readonly id: string;
  readonly name: string;
  readonly size: bigint;

  private constructor(readonly uri: vscode.Uri, size: number) {
    this.id = uri.toString();
    this.name = path.posix.basename(uri.path);
    this.size = BigInt(size);
  }

  static async create(uri: vscode.Uri): Promise<VscodeFileSource> {
    const stat = await vscode.workspace.fs.stat(uri);
    return new VscodeFileSource(uri, stat.size);
  }

  async read(offset: bigint, length: number): Promise<Uint8Array> {
    if (offset < 0n || offset + BigInt(length) > this.size) {
      throw new Error(`Read exceeds ${this.name}`);
    }
    if (this.uri.scheme === "file") {
      const handle = await open(this.uri.fsPath, "r");
      try {
        const bytes = new Uint8Array(length);
        const result = await handle.read(bytes, 0, length, offset);
        if (result.bytesRead !== length) throw new Error(`Short read from ${this.name}`);
        return bytes;
      } finally {
        await handle.close();
      }
    }
    if (this.size > 64n * 1024n * 1024n) {
      throw new Error(
        "This virtual workspace does not expose ranged reads. Open the file from a local filesystem."
      );
    }
    const bytes = await vscode.workspace.fs.readFile(this.uri);
    return bytes.slice(Number(offset), Number(offset) + length);
  }
}

class WeightsDocument implements vscode.CustomDocument {
  constructor(readonly uri: vscode.Uri) {}
  dispose(): void {}
}

class WeightsEditorProvider implements vscode.CustomReadonlyEditorProvider<WeightsDocument> {
  readonly parsers: Record<WeightFormat, Parser> = {
    safetensors: new SafeTensorsParser(),
    gguf: new GgufParser(),
    onnx: new OnnxParser()
  };

  constructor(private readonly context: vscode.ExtensionContext) {}

  openCustomDocument(uri: vscode.Uri): WeightsDocument {
    return new WeightsDocument(uri);
  }

  async resolveCustomEditor(
    document: WeightsDocument,
    panel: vscode.WebviewPanel
  ): Promise<void> {
    const sources = new Map<string, RandomAccessSource>();
    panel.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this.context.extensionUri, "dist", "webview")
      ]
    };
    const connection = connectPanel(panel, sources, this.parsers, async () => {
      const localSources = await discoverSources(document.uri);
      localSources.forEach((source) => sources.set(source.id, source));
      return loadSources(localSources);
    });
    panel.webview.html = webviewHtml(panel.webview, this.context.extensionUri);
    await connection;
  }
}

async function connectPanel(
  panel: vscode.WebviewPanel,
  sources: Map<string, RandomAccessSource>,
  parsers: Record<WeightFormat, Parser>,
  load: () => Promise<Awaited<ReturnType<typeof loadSources>>>
): Promise<void> {
  let ready = false;
  let outcome:
    | { type: "models"; models: Awaited<ReturnType<typeof loadSources>> }
    | { type: "error"; error: string }
    | undefined;
  const publish = async () => {
    if (ready && outcome) await panel.webview.postMessage(encode(outcome));
  };
  const subscription = panel.webview.onDidReceiveMessage(async (message: unknown) => {
    const decoded = revive(message) as {
      type: string;
      requestId?: string;
      tensor?: TensorRecord;
    };
    if (decoded.type === "ready") {
      ready = true;
      await publish();
      return;
    }
    if (decoded.type === "sample" && decoded.requestId && decoded.tensor) {
      try {
        const source = sources.get(decoded.tensor.fileId);
        if (!source) throw new Error("Source file is unavailable");
        const parser = parsers[await detectFormat(source)];
        if (!parser.sample) throw new Error("Sampling is unavailable");
        const sample = await parser.sample(source, decoded.tensor, 256);
        await panel.webview.postMessage(
          encode({ type: "sample", requestId: decoded.requestId, sample })
        );
      } catch (error) {
        await panel.webview.postMessage(
          encode({
            type: "error",
            requestId: decoded.requestId,
            error: errorMessage(error)
          })
        );
      }
    }
  });
  panel.onDidDispose(() => subscription.dispose());
  try {
    outcome = { type: "models", models: await load() };
  } catch (error) {
    outcome = { type: "error", error: errorMessage(error) };
  }
  await publish();
}

async function discoverSources(uri: vscode.Uri): Promise<RandomAccessSource[]> {
  const primary = await VscodeFileSource.create(uri);
  if (!primary.name.toLowerCase().endsWith(".safetensors.index.json")) {
    return [primary];
  }
  if (primary.size > 16n * 1024n * 1024n) {
    throw new Error("SafeTensors index exceeds 16 MiB");
  }
  const bytes = await primary.read(0n, Number(primary.size));
  const index = JSON.parse(new TextDecoder().decode(bytes)) as {
    weight_map?: Record<string, string>;
  };
  const names = [...new Set(Object.values(index.weight_map ?? {}))];
  const sources: RandomAccessSource[] = [primary];
  for (const name of names) {
    const shardUri = vscode.Uri.joinPath(uri, "..", name);
    sources.push(await VscodeFileSource.create(shardUri));
  }
  return sources;
}

function webviewHtml(webview: vscode.Webview, extensionUri: vscode.Uri): string {
  const script = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, "dist", "webview", "assets", "webview.js")
  );
  const style = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, "dist", "webview", "assets", "style.css")
  );
  const nonce = crypto.randomUUID().replaceAll("-", "");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
  <link rel="stylesheet" href="${style}">
  <title>Weights Viz</title>
</head>
<body>
  <div id="root"></div>
  <script nonce="${nonce}" type="module" src="${script}"></script>
</body>
</html>`;
}

function encode(value: unknown): unknown {
  if (typeof value === "bigint") return { $bigint: value.toString() };
  if (Array.isArray(value)) return value.map(encode);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [
        key,
        encode(item)
      ])
    );
  }
  return value;
}

function revive(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(revive);
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (typeof record.$bigint === "string") return BigInt(record.$bigint);
    return Object.fromEntries(
      Object.entries(record).map(([key, item]) => [key, revive(item)])
    );
  }
  return value;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function activate(context: vscode.ExtensionContext): void {
  const provider = new WeightsEditorProvider(context);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider("weightsViz.viewer", provider, {
      webviewOptions: { retainContextWhenHidden: true }
    }),
    vscode.commands.registerCommand("weightsViz.openUrl", async () => {
      const url = await vscode.window.showInputBox({
        title: "Open model URL",
        prompt: "Enter a GGUF, SafeTensors, SafeTensors index, or ONNX URL",
        validateInput: (value) => {
          try {
            new URL(value);
            return undefined;
          } catch {
            return "Enter a valid absolute URL";
          }
        }
      });
      if (!url) return;
      const panel = vscode.window.createWebviewPanel(
        "weightsViz.remote",
        "Weights Viz: Remote model",
        vscode.ViewColumn.Active,
        {
          enableScripts: true,
          localResourceRoots: [
            vscode.Uri.joinPath(context.extensionUri, "dist", "webview")
          ]
        }
      );
      const sources = new Map<string, RandomAccessSource>();
      const connection = connectPanel(panel, sources, provider.parsers, async () => {
        return loadModelUrl(url, undefined, (source) => {
          sources.set(source.id, source);
        });
      });
      panel.webview.html = webviewHtml(panel.webview, context.extensionUri);
      await connection;
    })
  );
}

export function deactivate(): void {}
