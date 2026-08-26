import * as path from "node:path";
import { open } from "node:fs/promises";
import * as vscode from "vscode";
import {
  detectFormat,
  GgufParser,
  loadModelUrl,
  loadSources,
  onnxExternalLocations,
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
  let initial: Awaited<ReturnType<typeof loadSources>> | undefined;
  const opened: Awaited<ReturnType<typeof loadSources>> = [];
  let failure: string | undefined;
  const publish = async () => {
    if (!ready) return;
    if (failure !== undefined && !opened.length) {
      await panel.webview.postMessage(encode({ type: "error", error: failure }));
    } else if (initial || opened.length) {
      await panel.webview.postMessage(
        encode({ type: "models", models: [...(initial ?? []), ...opened] })
      );
    }
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
    if (decoded.type === "open") {
      try {
        const picked = await pickModelFiles();
        const chosen = await collectSources(picked);
        chosen.forEach((source) => sources.set(source.id, source));
        if (chosen.length) opened.push(...(await loadSources(chosen)));
        await publish();
      } catch (error) {
        await panel.webview.postMessage(encode({ type: "error", error: errorMessage(error) }));
      }
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
    initial = await load();
  } catch (error) {
    failure = errorMessage(error);
  }
  await publish();
}

async function pickModelFiles(): Promise<vscode.Uri[]> {
  const picked = await vscode.window.showOpenDialog({
    canSelectMany: true,
    canSelectFolders: false,
    openLabel: "Open",
    title: "Open model files",
    filters: {
      "Model weights": ["gguf", "safetensors", "onnx", "json"],
      "All files": ["*"]
    }
  });
  return picked ?? [];
}

async function collectSources(uris: vscode.Uri[]): Promise<RandomAccessSource[]> {
  const collected = new Map<string, RandomAccessSource>();
  for (const uri of uris) {
    for (const source of await discoverSources(uri)) collected.set(source.id, source);
  }
  return [...collected.values()];
}

async function discoverSources(uri: vscode.Uri): Promise<RandomAccessSource[]> {
  const primary = await VscodeFileSource.create(uri);
  if (primary.name.toLowerCase().endsWith(".onnx")) {
    const manifest = await new OnnxParser().parse(primary);
    const sources: RandomAccessSource[] = [primary];
    for (const location of onnxExternalLocations(manifest)) {
      const externalUri = vscode.Uri.joinPath(uri, "..", location);
      sources.push(await VscodeFileSource.create(externalUri));
    }
    return sources;
  }
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
  const createPanel = (title: string) =>
    vscode.window.createWebviewPanel("weightsViz.remote", title, vscode.ViewColumn.Active, {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, "dist", "webview")]
    });
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider("weightsViz.viewer", provider, {
      webviewOptions: { retainContextWhenHidden: true }
    }),
    vscode.commands.registerCommand("weightsViz.openFiles", async () => {
      const picked = await pickModelFiles();
      if (!picked.length) return;
      const panel = createPanel("Weights Viz: Model files");
      const sources = new Map<string, RandomAccessSource>();
      const connection = connectPanel(panel, sources, provider.parsers, async () => {
        const opened = await collectSources(picked);
        opened.forEach((source) => sources.set(source.id, source));
        return loadSources(opened);
      });
      panel.webview.html = webviewHtml(panel.webview, context.extensionUri);
      await connection;
    }),
    vscode.commands.registerCommand(
      "weightsViz.visualize",
      async (uri: vscode.Uri | undefined) => {
        if (!uri) {
          throw new Error("Visualize Weights requires a model file.");
        }
        await vscode.commands.executeCommand(
          "vscode.openWith",
          uri,
          "weightsViz.viewer"
        );
      }
    ),
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
      const panel = createPanel("Weights Viz: Remote model");
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
