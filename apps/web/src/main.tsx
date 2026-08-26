import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { ParsedModel, TensorRecord, TensorSample } from "@weights-viz/core";
import { WeightsExplorer } from "@weights-viz/ui";

type WorkerRequest =
  | { type: "files"; files: File[] }
  | { type: "url"; url: string }
  | { type: "sample"; tensor: TensorRecord; maxValues: number };

function App() {
  const [models, setModels] = useState<ParsedModel[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const workerRef = useRef<Worker | null>(null);
  const nextId = useRef(1);
  const pending = useRef(
    new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>()
  );

  useEffect(() => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (
      event: MessageEvent<{ id: number; ok: boolean; result?: unknown; error?: string }>
    ) => {
      const request = pending.current.get(event.data.id);
      if (!request) return;
      pending.current.delete(event.data.id);
      if (event.data.ok) request.resolve(event.data.result);
      else request.reject(new Error(event.data.error ?? "Worker operation failed"));
    };
    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  const request = <T,>(payload: WorkerRequest): Promise<T> =>
    new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error("The parser worker is still starting"));
        return;
      }
      const id = nextId.current++;
      pending.current.set(id, {
        resolve: (value) => resolve(value as T),
        reject
      });
      workerRef.current.postMessage({ ...payload, id });
    });

  const loadFiles = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    setError(undefined);
    try {
      const loaded = await request<ParsedModel[]>({ type: "files", files });
      setModels((current) => [...current, ...loaded]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  const loadUrl = async (url: string) => {
    setBusy(true);
    setError(undefined);
    try {
      const loaded = await request<ParsedModel[]>({ type: "url", url });
      setModels((current) => [...current, ...loaded]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const prevent = (event: DragEvent) => event.preventDefault();
    const drop = (event: DragEvent) => {
      event.preventDefault();
      void loadFiles(Array.from(event.dataTransfer?.files ?? []));
    };
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", drop);
    };
  });

  return (
    <>
      <WeightsExplorer
        models={models}
        busy={busy}
        {...(error ? { error } : {})}
        onFilesSelected={(files) => void loadFiles(files)}
        onOpenUrl={(url) => void loadUrl(url)}
        onSample={(tensor) =>
          request<TensorSample>({ type: "sample", tensor, maxValues: 256 })
        }
      />
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
