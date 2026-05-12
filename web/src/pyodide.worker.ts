/// <reference lib="webworker" />
// Pyodide runs in a dedicated worker so the UI stays responsive while
// training the 200-line GPT. The worker streams stdout chunks back to
// the main thread; the main thread parses them into structured updates.

import { prepareTrainingScript } from "./patch";

const PYODIDE_VERSION = "0.29.4";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Logical-to-physical URL rewrites. When the user mounts a known canonical
// dataset URL, we transparently fetch a same-origin bundled copy instead —
// identical bytes, no network round-trip. The reported URL stays the
// canonical one so the UI matches what's documented in microgpt.py.
const URL_ALIASES: Record<string, string> = {
  "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt": "/input.txt",
};

export type RunPhase = "init" | "train" | "infer";

export type WorkerInbound =
  | { type: "init"; inputUrl: string; interruptBuffer?: Uint8Array }
  | {
      type: "train";
      code: string;
      options: { numSteps: number; seed: number };
    }
  | {
      type: "infer";
      options: { temperature: number; numSamples: number };
    }
  | { type: "ping" };

export type WorkerOutbound =
  | { type: "status"; status: string; phase?: RunPhase; detail?: string }
  | { type: "ready"; numDocs: number; inputUrl: string }
  | { type: "stdout"; text: string; end: "\n" | "\r"; phase: RunPhase }
  | { type: "stderr"; text: string; end: "\n" | "\r"; phase: RunPhase }
  | { type: "done"; phase: RunPhase; durationMs: number }
  | { type: "cancelled"; phase: RunPhase }
  | { type: "error"; phase?: RunPhase; message: string };

type PyodideInterface = {
  setStdout(opts: {
    isatty?: boolean;
    write?: (buffer: Uint8Array) => number;
  }): void;
  setStderr(opts: {
    isatty?: boolean;
    write?: (buffer: Uint8Array) => number;
  }): void;
  setInterruptBuffer(buffer: Uint8Array): void;
  FS: { writeFile(path: string, data: string | Uint8Array): void };
  runPythonAsync(code: string): Promise<unknown>;
};

let pyodide: PyodideInterface | null = null;
let pyodideLoading: Promise<PyodideInterface> | null = null;
let readyState: { inputUrl: string; promise: Promise<void> } | null = null;
let currentPhase: RunPhase = "init";
let interruptBuffer: Uint8Array | null = null;

function post(msg: WorkerOutbound) {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg);
}

async function loadPyodideOnce(): Promise<PyodideInterface> {
  if (pyodide) return pyodide;
  if (pyodideLoading) return pyodideLoading;

  pyodideLoading = (async () => {
    post({ type: "status", status: "loading-pyodide", detail: `Pyodide v${PYODIDE_VERSION}` });
    const mod = (await import(
      /* @vite-ignore */ `${PYODIDE_INDEX_URL}pyodide.mjs`
    )) as { loadPyodide: (cfg: { indexURL: string }) => Promise<PyodideInterface> };

    const py = await mod.loadPyodide({ indexURL: PYODIDE_INDEX_URL });

    // Stdio is wired up once. isatty:true keeps Pyodide's stream layer from
    // buffering, which is essential for `print(..., end='\r')` to flush live.
    py.setStdout({ write: createStreamWriter("stdout"), isatty: true });
    py.setStderr({ write: createStreamWriter("stderr"), isatty: true });

    // Cooperative cancellation: if the main thread allocated a
    // SharedArrayBuffer and shipped it over, hand it to Pyodide. Writing
    // `2` to byte 0 raises KeyboardInterrupt at the next Python check.
    if (interruptBuffer) py.setInterruptBuffer(interruptBuffer);

    // Force Python's `print` to flush on every call. Without this, `end='\r'`
    // updates can stall in CPython's own buffer.
    await py.runPythonAsync(`
import builtins as _b
_orig_print = _b.print
def _p(*a, **kw):
    kw.setdefault('flush', True)
    return _orig_print(*a, **kw)
_b.print = _p
`);

    pyodide = py;
    return py;
  })();
  return pyodideLoading;
}

// Fetch the dataset from the requested URL and seed it into Pyodide's FS as
// `input.txt`. Once written, the script's `os.path.exists('input.txt')`
// branch short-circuits the urllib fallback — the displayed URL literal
// is honest documentation but never actually executes.
function ensureReady(inputUrl: string): Promise<void> {
  if (readyState?.inputUrl === inputUrl) return readyState.promise;

  const promise = (async () => {
    const py = await loadPyodideOnce();
    const fetchUrl = URL_ALIASES[inputUrl] ?? inputUrl;
    post({
      type: "status",
      status: "fetching-data",
      detail: prettySource(inputUrl),
    });
    const resp = await fetch(fetchUrl);
    if (!resp.ok) throw new Error(`Failed to fetch dataset (${fetchUrl}): ${resp.status}`);
    const txt = await resp.text();
    py.FS.writeFile("input.txt", txt);
    const numDocs = txt.split("\n").filter((l) => l.trim().length > 0).length;
    post({ type: "ready", numDocs, inputUrl });
  })();

  readyState = { inputUrl, promise };
  return promise;
}

function prettySource(url: string): string {
  if (url.startsWith("/") || url.startsWith("./")) return "bundled input.txt";
  try {
    const u = new URL(url, self.location.href);
    return u.hostname || url;
  } catch {
    return url;
  }
}

// Stream-aware stdout/stderr writer: emit one message per \n or \r so the
// main thread receives complete line fragments with the terminator carried
// as metadata.
function createStreamWriter(kind: "stdout" | "stderr") {
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  return (chunk: Uint8Array): number => {
    buf += decoder.decode(chunk, { stream: true });
    let lastFlush = 0;
    for (let i = 0; i < buf.length; i++) {
      const ch = buf[i];
      if (ch === "\n" || ch === "\r") {
        const text = buf.slice(lastFlush, i);
        post({ type: kind, text, end: ch, phase: currentPhase } as WorkerOutbound);
        lastFlush = i + 1;
      }
    }
    buf = buf.slice(lastFlush);
    return chunk.length;
  };
}

function isInterrupt(err: unknown): boolean {
  return /KeyboardInterrupt/.test(err instanceof Error ? err.message : String(err));
}

function armInterrupt() {
  if (interruptBuffer) interruptBuffer[0] = 0;
}

self.onmessage = async (ev: MessageEvent<WorkerInbound>) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      if (msg.interruptBuffer) {
        interruptBuffer = msg.interruptBuffer;
        if (pyodide) pyodide.setInterruptBuffer(interruptBuffer);
      }
      await ensureReady(msg.inputUrl);
      return;
    }

    if (msg.type === "train") {
      if (!readyState) throw new Error("Worker received `train` before `init`");
      await readyState.promise;
      const py = await loadPyodideOnce();
      currentPhase = "train";
      armInterrupt();
      // Run the (seed-patched, demo-call-stripped) script: defines all
      // functions and re-initializes `state_dict` with the current seed.
      // Then explicitly call `train(num_steps=N)` so num_steps flows
      // through Python's own kwarg machinery — no source patching needed.
      const prepared = prepareTrainingScript(msg.code, msg.options.seed);
      const t0 = performance.now();
      post({ type: "status", status: "running", phase: "train" });
      try {
        await py.runPythonAsync(prepared);
        await py.runPythonAsync(`train(num_steps=${msg.options.numSteps})`);
      } catch (err) {
        if (isInterrupt(err)) {
          post({ type: "cancelled", phase: "train" });
          return;
        }
        throw err;
      }
      const t1 = performance.now();
      post({ type: "done", phase: "train", durationMs: t1 - t0 });
      return;
    }

    if (msg.type === "infer") {
      if (!readyState) throw new Error("Worker received `infer` before `init`");
      await readyState.promise;
      const py = await loadPyodideOnce();
      currentPhase = "infer";
      armInterrupt();
      const t0 = performance.now();
      post({ type: "status", status: "running", phase: "infer" });
      try {
        await py.runPythonAsync(
          `infer(temperature=${msg.options.temperature}, num_samples=${msg.options.numSamples})`,
        );
      } catch (err) {
        if (isInterrupt(err)) {
          post({ type: "cancelled", phase: "infer" });
          return;
        }
        throw err;
      }
      const t1 = performance.now();
      post({ type: "done", phase: "infer", durationMs: t1 - t0 });
      return;
    }
  } catch (err) {
    const message =
      err instanceof Error ? `${err.name}: ${err.message}\n${err.stack ?? ""}` : String(err);
    post({ type: "error", phase: currentPhase, message });
  }
};

// Start downloading Pyodide as soon as the worker boots so the WASM is warm
// by the time the main thread sends `init` with the input URL.
loadPyodideOnce().catch((err) => {
  post({
    type: "error",
    message: err instanceof Error ? err.message : String(err),
  });
});

export {};
