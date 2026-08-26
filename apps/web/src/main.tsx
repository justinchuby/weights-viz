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
  const [workerReady, setWorkerReady] = useState(false);
  const [sharedUrl] = useState(modelUrlFromQuery);
  const workerRef = useRef<Worker | null>(null);
  const sharedUrlHandled = useRef(false);
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
    setWorkerReady(true);
    return () => {
      worker.terminate();
      if (workerRef.current === worker) workerRef.current = null;
    };
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
      const pageUrl = new URL(window.location.href);
      pageUrl.searchParams.set("url", url);
      window.history.replaceState(null, "", pageUrl);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!workerReady || !sharedUrl || sharedUrlHandled.current) return;
    sharedUrlHandled.current = true;
    void loadUrl(sharedUrl);
  }, [workerReady, sharedUrl]);

  useEffect(() => {
    const prevent = (event: DragEvent) => event.preventDefault();
    const drop = (event: DragEvent) => {
      event.preventDefault();
      void loadFiles(Array.from(event.dataTransfer?.files ?? []));
    };
    const paste = (event: ClipboardEvent) => {
      const files = Array.from(event.clipboardData?.files ?? []);
      if (!files.length) return;
      event.preventDefault();
      void loadFiles(files);
    };
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", drop);
    window.addEventListener("paste", paste);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", drop);
      window.removeEventListener("paste", paste);
    };
  });

  return (
    <>
      <WeightsExplorer
        models={models}
        busy={busy}
        {...(error ? { error } : {})}
        {...(sharedUrl ? { defaultUrl: sharedUrl } : {})}
        onFilesSelected={(files) => void loadFiles(files)}
        onOpenUrl={(url) => void loadUrl(url)}
        onSample={(tensor) =>
          request<TensorSample>({ type: "sample", tensor, maxValues: 256 })
        }
      />
    </>
  );
}

function modelUrlFromQuery(): string | undefined {
  const value = new URLSearchParams(window.location.search).get("url")?.trim();
  return value ? new URL(value, window.location.href).toString() : undefined;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
