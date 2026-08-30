import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  listHuggingFaceModelFiles,
  type ParsedModel,
  type RemoteLoadProgress,
  type TensorRecord,
  type TensorSample
} from "@weights-viz/core";
import { WeightsExplorer } from "@weights-viz/ui";

type WorkerRequest =
  | { type: "files"; files: File[] }
  | { type: "url"; url: string }
  | { type: "sample"; tensor: TensorRecord; maxValues: number };

interface InstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

function App() {
  const [models, setModels] = useState<ParsedModel[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [remoteProgress, setRemoteProgress] = useState<RemoteLoadProgress>();
  const [workerReady, setWorkerReady] = useState(false);
  const [sharedUrls] = useState(modelUrlsFromQuery);
  const [dtypeAtlas, setDtypeAtlas] = useState(dtypeAtlasFromQuery);
  const [installPrompt, setInstallPrompt] = useState<InstallPromptEvent>();
  const workerRef = useRef<Worker | null>(null);
  const sharedUrlHandled = useRef(false);
  const nextId = useRef(1);
  const pending = useRef(
    new Map<number, {
      resolve: (value: unknown) => void;
      reject: (error: Error) => void;
      onProgress?: (progress: RemoteLoadProgress) => void;
    }>()
  );

  useEffect(() => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (
      event: MessageEvent<{
        id: number;
        type?: "progress";
        ok?: boolean;
        result?: unknown;
        error?: string;
        progress?: RemoteLoadProgress;
      }>
    ) => {
      const request = pending.current.get(event.data.id);
      if (!request) return;
      if (event.data.type === "progress" && event.data.progress) {
        request.onProgress?.(event.data.progress);
        return;
      }
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

  useEffect(() => {
    const syncView = () => setDtypeAtlas(dtypeAtlasFromQuery());
    window.addEventListener("popstate", syncView);
    return () => window.removeEventListener("popstate", syncView);
  }, []);

  const request = <T,>(
    payload: WorkerRequest,
    onProgress?: (progress: RemoteLoadProgress) => void
  ): Promise<T> =>
    new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error("The parser worker is still starting"));
        return;
      }
      const id = nextId.current++;
      pending.current.set(id, {
        resolve: (value) => resolve(value as T),
        reject,
        ...(onProgress ? { onProgress } : {})
      });
      workerRef.current.postMessage({ ...payload, id });
    });

  const loadFiles = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    setError(undefined);
    setRemoteProgress(undefined);
    try {
      const loaded = await request<ParsedModel[]>({ type: "files", files });
      setModels((current) => [...current, ...loaded]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };

  const loadUrls = async (urls: string[]) => {
    if (!urls.length) return;
    setBusy(true);
    setError(undefined);
    setRemoteProgress(undefined);
    try {
      const loaded = await Promise.all(
        urls.map((url) =>
          request<ParsedModel[]>({ type: "url", url }, setRemoteProgress)
        )
      );
      setModels((current) => [...current, ...loaded.flat()]);
      const pageUrl = new URL(window.location.href);
      const existing = new Set(pageUrl.searchParams.getAll("url"));
      for (const url of urls) {
        if (!existing.has(url)) pageUrl.searchParams.append("url", url);
      }
      window.history.replaceState(null, "", pageUrl);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
      setRemoteProgress(undefined);
    }
  };

  useEffect(() => {
    if (!workerReady || !sharedUrls.length || sharedUrlHandled.current) return;
    sharedUrlHandled.current = true;
    setBusy(true);
    setError(undefined);
    setRemoteProgress(undefined);
    void Promise.all(
      sharedUrls.map((url) =>
        request<ParsedModel[]>({ type: "url", url }, setRemoteProgress)
      )
    )
      .then((loaded) => setModels(loaded.flat()))
      .catch((reason) =>
        setError(reason instanceof Error ? reason.message : String(reason))
      )
      .finally(() => {
        setBusy(false);
        setRemoteProgress(undefined);
      });
  }, [sharedUrls, workerReady]);

  useEffect(() => {
    const captureInstallPrompt = (event: Event) => {
      event.preventDefault();
      setInstallPrompt(event as InstallPromptEvent);
    };
    window.addEventListener("beforeinstallprompt", captureInstallPrompt);
    return () =>
      window.removeEventListener("beforeinstallprompt", captureInstallPrompt);
  }, []);

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
        {...(remoteProgress ? { progress: remoteProgress } : {})}
        {...(error ? { error } : {})}
        {...(sharedUrls.length ? { defaultUrl: sharedUrls.join("\n") } : {})}
        defaultCompare={sharedUrls.length > 1}
        defaultDtypeAtlas={dtypeAtlas}
        installAvailable={Boolean(installPrompt)}
        onInstall={() => {
          if (!installPrompt) return;
          void installPrompt.prompt().then(() => installPrompt.userChoice).then(() => {
            setInstallPrompt(undefined);
          });
        }}
        onFilesSelected={(files) => void loadFiles(files)}
        onCloseModel={(modelId) =>
          setModels((current) => current.filter((model) => model.id !== modelId))
        }
        onBrowseHuggingFace={(repository) =>
          listHuggingFaceModelFiles(repository)
        }
        onDtypeAtlasChange={(open) => {
          const pageUrl = new URL(window.location.href);
          if (open) pageUrl.searchParams.set("view", "dtypes");
          else pageUrl.searchParams.delete("view");
          window.history.pushState(null, "", pageUrl);
          setDtypeAtlas(open);
        }}
        onOpenUrl={(input) =>
          void loadUrls(
            input
              .split(/\s+/)
              .map((value) => value.trim())
              .filter(Boolean)
          )
        }
        onSample={(tensor) =>
          request<TensorSample>({ type: "sample", tensor, maxValues: 256 })
        }
      />
    </>
  );
}

function modelUrlsFromQuery(): string[] {
  const params = new URLSearchParams(window.location.search);
  const values = [
    ...params.getAll("url"),
    ...(params.get("compare") ? [params.get("compare")!] : [])
  ];
  return [
    ...new Set(
      values
        .map((value) => value.trim())
        .filter(Boolean)
        .map((value) => new URL(value, window.location.href).toString())
    )
  ];
}

function dtypeAtlasFromQuery(): boolean {
  return new URLSearchParams(window.location.search).get("view") === "dtypes";
}

if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register("./sw.js");
  });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
