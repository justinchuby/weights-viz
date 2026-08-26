import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import type { ParsedModel, TensorRecord, TensorSample } from "@weights-viz/core";
import { WeightsExplorer } from "@weights-viz/ui";

declare function acquireVsCodeApi(): {
  postMessage(message: unknown): void;
};

const vscode = acquireVsCodeApi();

function revive(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(revive);
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (typeof record.$bigint === "string") return BigInt(record.$bigint);
    return Object.fromEntries(Object.entries(record).map(([key, item]) => [key, revive(item)]));
  }
  return value;
}

function App() {
  const [models, setModels] = useState<ParsedModel[]>([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string>();
  const [pendingSamples] = useState(
    () => new Map<string, { resolve: (sample: TensorSample) => void; reject: (error: Error) => void }>()
  );

  useEffect(() => {
    const receive = (event: MessageEvent) => {
      const message = revive(event.data) as {
        type: string;
        models?: ParsedModel[];
        error?: string;
        requestId?: string;
        sample?: TensorSample;
      };
      if (message.type === "models") {
        setModels(message.models ?? []);
        setBusy(false);
      } else if (message.type === "error") {
        if (message.requestId) {
          pendingSamples.get(message.requestId)?.reject(new Error(message.error));
          pendingSamples.delete(message.requestId);
        } else {
          setError(message.error);
          setBusy(false);
        }
      } else if (message.type === "sample" && message.requestId && message.sample) {
        pendingSamples.get(message.requestId)?.resolve(message.sample);
        pendingSamples.delete(message.requestId);
      }
    };
    window.addEventListener("message", receive);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", receive);
  }, [pendingSamples]);

  const chooseFiles = () => {
    setError(undefined);
    setBusy(true);
    vscode.postMessage({ type: "open" });
  };

  const sample = (tensor: TensorRecord) =>
    new Promise<TensorSample>((resolve, reject) => {
      const requestId = crypto.randomUUID();
      pendingSamples.set(requestId, { resolve, reject });
      vscode.postMessage({ type: "sample", requestId, tensor: encode(tensor) });
    });

  return (
    <WeightsExplorer
      models={models}
      busy={busy}
      {...(error ? { error } : {})}
      intro="Open a GGUF, SafeTensors, or ONNX file with the Weights Visualization editor, or choose files below. VS Code webviews cannot show the browser file dialog, so this button asks VS Code for its native picker."
      onChooseFiles={chooseFiles}
      onSample={sample}
    />
  );
}

function encode(value: unknown): unknown {
  if (typeof value === "bigint") return { $bigint: value.toString() };
  if (Array.isArray(value)) return value.map(encode);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, encode(item)])
    );
  }
  return value;
}

createRoot(document.getElementById("root")!).render(<App />);
