import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CodeViewer } from "./CodeViewer";
import { CodeDiff } from "./CodeDiff";
import {
  DEFAULT_INFER,
  DEFAULT_TRAIN,
  patchAltDisplayCode,
  patchDisplayCode,
  type InferOptions,
  type TrainOptions,
} from "./patch";
import type { RunPhase, WorkerInbound, WorkerOutbound } from "./pyodide.worker";

type RunStatus =
  | "booting"
  | "loading-pyodide"
  | "fetching-data"
  | "ready"
  | "training"
  | "inferring"
  | "done"
  | "error";

type LogLine = {
  id: number;
  kind: "stdout" | "stderr" | "info";
  text: string;
};

type Sample = { id: number; text: string };

type Stats = {
  numDocs?: number;
  vocabSize?: number;
  numParams?: number;
};

type Progress = { step: number; total: number; loss: number } | null;

// Built-in datasets bundled into `web/public/`. The Dataset card exposes
// these as a compact `‹ names ›` cycler in addition to the free-form URL
// input, so the common case (try the four canonical demos) is one click.
type DatasetPreset = { name: string; url: string };
const DATASET_PRESETS: readonly DatasetPreset[] = [
  { name: "names", url: "/names.txt" },
  { name: "dinosaurs", url: "/dinosaurs.txt" },
  { name: "pokemon", url: "/pokemon.txt" },
  { name: "shakespeare", url: "/shakespeare.txt" },
];
const DEFAULT_INPUT_URL = DATASET_PRESETS[0].url;

function readInputUrlFromLocation(): string {
  if (typeof window === "undefined") return DEFAULT_INPUT_URL;
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("input_url");
  return fromQuery && fromQuery.trim() ? fromQuery.trim() : DEFAULT_INPUT_URL;
}

function syncInputUrlToLocation(url: string) {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  if (url === DEFAULT_INPUT_URL) params.delete("input_url");
  else params.set("input_url", url);
  const qs = params.toString();
  const next = window.location.pathname + (qs ? `?${qs}` : "") + window.location.hash;
  window.history.replaceState(null, "", next);
}

function prettyInputSource(url: string): string {
  try {
    if (url.startsWith("/") || url.startsWith("./")) return url;
    return new URL(url, window.location.href).hostname || url;
  } catch {
    return url;
  }
}

// Implementation tabs. The vanilla Python file is the only one we actually
// run in Pyodide — selecting `pytorch` or `tinygrad` opens a read-only
// side-by-side diff against vanilla. The runtime controls (Train / Infer)
// continue to operate on the vanilla source regardless of this tab.
type Impl = "python" | "pytorch" | "tinygrad";
type CompareWith = Exclude<Impl, "python">;

const COMPARE_LABELS: Record<CompareWith, { tab: string; filename: string }> = {
  pytorch: { tab: "pytorch", filename: "microgpt_pytorch.py" },
  tinygrad: { tab: "tinygrad", filename: "microgpt_tinygrad.py" },
};

function isCompareWith(value: string | null): value is CompareWith {
  return value === "pytorch" || value === "tinygrad";
}

function readCompareFromLocation(): CompareWith | null {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  const raw = params.get("compare");
  return isCompareWith(raw) ? raw : null;
}

function syncCompareToLocation(value: CompareWith | null) {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  if (value) params.set("compare", value);
  else params.delete("compare");
  const qs = params.toString();
  const next = window.location.pathname + (qs ? `?${qs}` : "") + window.location.hash;
  window.history.replaceState(null, "", next);
}

// One config per status, instead of four parallel records keyed by it.
// New status? Add it once; exhaustiveness is enforced by the type.
const STATUS_CONFIG: Record<
  RunStatus,
  { label: string; pill: string; dot: string; animate: boolean }
> = {
  booting: {
    label: "Booting worker",
    pill: "bg-ink-700 text-ink-200 border-ink-600",
    dot: "bg-violet-300",
    animate: false,
  },
  "loading-pyodide": {
    label: "Loading Pyodide",
    pill: "bg-amber-500/10 text-amber-200 border-amber-400/30",
    dot: "bg-violet-300",
    animate: true,
  },
  "fetching-data": {
    label: "Loading dataset",
    pill: "bg-amber-500/10 text-amber-200 border-amber-400/30",
    dot: "bg-violet-300",
    animate: true,
  },
  ready: {
    label: "Ready",
    pill: "bg-emerald-500/10 text-emerald-200 border-emerald-400/30",
    dot: "bg-emerald-300",
    animate: false,
  },
  training: {
    label: "Training",
    pill: "bg-violet-500/15 text-violet-100 border-violet-400/40",
    dot: "bg-violet-300",
    animate: true,
  },
  inferring: {
    label: "Sampling",
    pill: "bg-fuchsia-500/15 text-fuchsia-100 border-fuchsia-400/40",
    dot: "bg-fuchsia-300",
    animate: true,
  },
  done: {
    label: "Done",
    pill: "bg-emerald-500/10 text-emerald-200 border-emerald-400/30",
    dot: "bg-emerald-300",
    animate: false,
  },
  error: {
    label: "Error",
    pill: "bg-rose-500/15 text-rose-200 border-rose-400/40",
    dot: "bg-rose-300",
    animate: false,
  },
};

// Allocate a single-byte SharedArrayBuffer for cooperative cancellation.
// Pyodide reads byte 0; writing 2 raises KeyboardInterrupt at the next
// Python check. Requires cross-origin isolation (COOP+COEP headers); if
// the page isn't isolated, this returns null and onStop falls back to
// terminate-and-reboot.
function allocateInterruptBuffer(): Uint8Array | null {
  if (typeof SharedArrayBuffer === "undefined") return null;
  if (typeof self !== "undefined" && !self.crossOriginIsolated) return null;
  try {
    return new Uint8Array(new SharedArrayBuffer(1));
  } catch {
    return null;
  }
}

export default function App() {
  const [code, setCode] = useState<string>("");
  const [codeError, setCodeError] = useState<string | null>(null);

  const [status, setStatus] = useState<RunStatus>("booting");
  const [statusDetail, setStatusDetail] = useState<string>("");
  const [phase, setPhase] = useState<RunPhase>("init");
  const [trained, setTrained] = useState<boolean>(false);

  const [logLines, setLogLines] = useState<LogLine[]>([]);
  const [transient, setTransient] = useState<string>("");
  const [stats, setStats] = useState<Stats>({});
  const [progress, setProgress] = useState<Progress>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [trainMs, setTrainMs] = useState<number | null>(null);
  const [inferMs, setInferMs] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [trainOpts, setTrainOpts] = useState<TrainOptions>(DEFAULT_TRAIN);
  const [inferOpts, setInferOpts] = useState<InferOptions>(DEFAULT_INFER);
  const [inputUrl, setInputUrl] = useState<string>(() => readInputUrlFromLocation());
  const [inputUrlDraft, setInputUrlDraft] = useState<string>(inputUrl);

  const [compareWith, setCompareWithState] = useState<CompareWith | null>(() =>
    readCompareFromLocation(),
  );
  // Lazy cache of fetched alt-impl sources, keyed by the CompareWith tag.
  // First selection of a tab triggers a fetch; subsequent toggles to the
  // same tab are free. Errors are remembered too so the UI doesn't refetch
  // a 404 on every re-open.
  const [altCodeCache, setAltCodeCache] = useState<
    Partial<Record<CompareWith, { code?: string; error?: string }>>
  >({});

  // Allocated once per app load. Shared across all worker generations so a
  // post-stop reboot still has cooperative cancellation wired up.
  const [interruptBuffer] = useState<Uint8Array | null>(allocateInterruptBuffer);

  const workerRef = useRef<Worker | null>(null);
  // Held outside React state so the message handler can read it without
  // re-creating `createWorker` whenever the URL changes (which would tear
  // down and recreate the worker on every dataset switch — see #2 in the
  // simplification notes).
  const inputUrlRef = useRef<string>(inputUrl);
  const transientLineRef = useRef<string>("");
  const logIdRef = useRef<number>(0);
  const sampleIdRef = useRef<number>(0);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    inputUrlRef.current = inputUrl;
  }, [inputUrl]);

  const pushLog = useCallback((kind: LogLine["kind"], text: string) => {
    setLogLines((prev) => {
      logIdRef.current += 1;
      return [...prev, { id: logIdRef.current, kind, text }];
    });
  }, []);

  const parseLine = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    let m: RegExpMatchArray | null;
    if ((m = trimmed.match(/^num docs:\s*(\d+)/i))) {
      setStats((s) => ({ ...s, numDocs: Number(m![1]) }));
      return;
    }
    if ((m = trimmed.match(/^vocab size:\s*(\d+)/i))) {
      setStats((s) => ({ ...s, vocabSize: Number(m![1]) }));
      return;
    }
    if ((m = trimmed.match(/^num params:\s*(\d+)/i))) {
      setStats((s) => ({ ...s, numParams: Number(m![1]) }));
      return;
    }
    if ((m = trimmed.match(/^step\s+(\d+)\s*\/\s*(\d+)\s*\|\s*loss\s+([\d.]+)/i))) {
      setProgress({ step: Number(m[1]), total: Number(m[2]), loss: Number(m[3]) });
      return;
    }
    if ((m = trimmed.match(/^sample\s+\d+:\s*(.+)$/i))) {
      const text = m[1].trim();
      sampleIdRef.current += 1;
      const id = sampleIdRef.current;
      setSamples((prev) => [...prev, { id, text }]);
      return;
    }
  }, []);

  const handleMessage = useCallback(
    (msg: WorkerOutbound) => {
      switch (msg.type) {
        case "status": {
          setStatusDetail(msg.detail ?? "");
          if (msg.status === "loading-pyodide") setStatus("loading-pyodide");
          else if (msg.status === "fetching-data") setStatus("fetching-data");
          else if (msg.status === "running") {
            setErrorMessage(null);
            if (msg.phase === "train") setStatus("training");
            else if (msg.phase === "infer") setStatus("inferring");
          }
          if (msg.phase) setPhase(msg.phase);
          break;
        }
        case "ready": {
          setStatus("ready");
          setStatusDetail("");
          if (typeof msg.numDocs === "number") {
            setStats((s) => ({ ...s, numDocs: msg.numDocs }));
          }
          if (msg.inputUrl) {
            setInputUrl((prev) => {
              if (prev !== msg.inputUrl) {
                // Dataset changed → trained model is invalid for the new vocab.
                setTrained(false);
                setProgress(null);
                setTrainMs(null);
                setSamples([]);
                syncInputUrlToLocation(msg.inputUrl);
              }
              return msg.inputUrl;
            });
            setInputUrlDraft(msg.inputUrl);
          }
          break;
        }
        case "stdout":
        case "stderr": {
          // The worker pre-tokenizes on \r and \n, so every message arrives
          // as a complete line fragment terminated by either. We only need
          // to track the most recent \r line as a "transient" overwrite so
          // a subsequent \n can promote it to a permanent log entry.
          if (msg.end === "\r") {
            transientLineRef.current = msg.text;
            setTransient(msg.text);
            parseLine(msg.text);
          } else {
            const line = msg.text || transientLineRef.current;
            if (line) {
              pushLog(msg.type, line);
              parseLine(line);
            }
            if (transientLineRef.current) {
              transientLineRef.current = "";
              setTransient("");
            }
          }
          break;
        }
        case "done": {
          setStatus("done");
          if (msg.phase === "train") {
            setTrainMs(msg.durationMs);
            setTrained(true);
          } else if (msg.phase === "infer") {
            setInferMs(msg.durationMs);
          }
          break;
        }
        case "cancelled": {
          setStatus("ready");
          setStatusDetail("");
          pushLog("info", `Stopped (${msg.phase}).`);
          if (transientLineRef.current) {
            transientLineRef.current = "";
            setTransient("");
          }
          break;
        }
        case "error": {
          setStatus("error");
          setErrorMessage(msg.message);
          break;
        }
      }
    },
    [parseLine, pushLog]
  );

  const createWorker = useCallback(() => {
    const w = new Worker(new URL("./pyodide.worker.ts", import.meta.url), {
      type: "module",
    });
    w.onmessage = (ev: MessageEvent<WorkerOutbound>) => handleMessage(ev.data);
    w.onerror = (ev) => {
      setStatus("error");
      setErrorMessage(ev.message || "Worker crashed");
    };
    const initMsg: WorkerInbound = {
      type: "init",
      inputUrl: inputUrlRef.current,
      ...(interruptBuffer ? { interruptBuffer } : {}),
    };
    w.postMessage(initMsg);
    workerRef.current = w;
    return w;
  }, [handleMessage, interruptBuffer]);

  useEffect(() => {
    fetch("/microgpt.py")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then(setCode)
      .catch((err) => setCodeError(err.message ?? String(err)));

    createWorker();
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, [createWorker]);

  // Lazy-fetch the alt-impl source the first time it's selected. We never
  // refetch a cached success or error (page reload re-arms the cache).
  useEffect(() => {
    if (!compareWith) return;
    if (altCodeCache[compareWith]) return;
    const filename = COMPARE_LABELS[compareWith].filename;
    let cancelled = false;
    fetch(`/${filename}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        if (cancelled) return;
        setAltCodeCache((prev) => ({ ...prev, [compareWith]: { code: text } }));
      })
      .catch((err) => {
        if (cancelled) return;
        setAltCodeCache((prev) => ({
          ...prev,
          [compareWith]: { error: err?.message ?? String(err) },
        }));
      });
    return () => {
      cancelled = true;
    };
  }, [compareWith, altCodeCache]);

  // Single setter that mirrors selection into the URL so a reload keeps
  // the active comparison.
  const setCompareWith = useCallback((next: CompareWith | null) => {
    setCompareWithState(next);
    syncCompareToLocation(next);
  }, []);

  useEffect(() => {
    const el = terminalRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [logLines, transient]);

  const resetTransientBuffers = () => {
    transientLineRef.current = "";
    setTransient("");
  };

  const onTrain = useCallback(() => {
    if (!workerRef.current || !code) return;
    setLogLines([]);
    setStats({});
    setProgress(null);
    setSamples([]);
    sampleIdRef.current = 0;
    setTrainMs(null);
    setInferMs(null);
    setErrorMessage(null);
    setTrained(false);
    resetTransientBuffers();

    const msg: WorkerInbound = { type: "train", code, options: trainOpts };
    workerRef.current.postMessage(msg);
    setStatus("training");
    setPhase("train");
  }, [code, trainOpts]);

  const onInfer = useCallback(() => {
    if (!workerRef.current) return;
    setSamples([]);
    sampleIdRef.current = 0;
    setInferMs(null);
    setErrorMessage(null);
    resetTransientBuffers();

    const msg: WorkerInbound = { type: "infer", options: inferOpts };
    workerRef.current.postMessage(msg);
    setStatus("inferring");
    setPhase("infer");
  }, [inferOpts]);

  const onStop = useCallback(() => {
    // Preferred path: cooperative cancellation via the shared interrupt
    // buffer. Pyodide raises KeyboardInterrupt at the next Python check;
    // the worker catches it and posts `cancelled`. No worker reboot, the
    // trained model stays in memory.
    if (interruptBuffer && workerRef.current) {
      interruptBuffer[0] = 2;
      return;
    }
    // Fallback: SharedArrayBuffer isn't available (no cross-origin
    // isolation). The only way to stop a running runPythonAsync is to
    // terminate the worker and start over, which costs us the trained
    // model and a Pyodide reboot.
    workerRef.current?.terminate();
    workerRef.current = null;
    setStatus("booting");
    setStatusDetail("");
    setTrained(false);
    pushLog("info", "Run stopped. Restarting Pyodide worker…");
    setTimeout(() => createWorker(), 50);
  }, [createWorker, interruptBuffer, pushLog]);

  const onLoadDataset = useCallback(
    (urlOverride?: string) => {
      if (!workerRef.current) return;
      const next = (urlOverride ?? inputUrlDraft).trim();
      if (!next) return;
      // Optimistically reflect the change so the status pill flips immediately.
      setStatus("fetching-data");
      setStatusDetail(prettyInputSource(next));
      setErrorMessage(null);
      const initMsg: WorkerInbound = { type: "init", inputUrl: next };
      workerRef.current.postMessage(initMsg);
    },
    [inputUrlDraft]
  );

  const progressPct = useMemo(() => {
    if (!progress) return 0;
    return Math.min(100, (progress.step / progress.total) * 100);
  }, [progress]);

  const displayCode = useMemo(
    () => patchDisplayCode(code, trainOpts, inferOpts, inputUrl),
    [code, trainOpts, inferOpts, inputUrl],
  );

  const altEntry = compareWith ? altCodeCache[compareWith] : undefined;
  const altCodeRaw = altEntry?.code ?? "";
  const altCodeError = altEntry?.error ?? null;
  // Apply the same UI-sync (slider + dataset URL) to the alt-impl viewer
  // and collapse its env-var indirection so the rendered diff lines up
  // with vanilla. Cosmetic only — `altCodeRaw` is never executed.
  const altCode = useMemo(
    () => patchAltDisplayCode(altCodeRaw, trainOpts, inferOpts, inputUrl),
    [altCodeRaw, trainOpts, inferOpts, inputUrl],
  );

  const isTraining = status === "training";
  const isInferring = status === "inferring";
  const isBusy =
    status === "booting" || status === "loading-pyodide" || status === "fetching-data";
  const workerReady = !isBusy;

  return (
    <div className="relative z-10 h-screen flex flex-col overflow-hidden">
      <Header status={status} statusDetail={statusDetail} />

      <main className="flex-1 min-h-0 px-4 md:px-6 lg:px-6 pb-4">
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px] xl:grid-cols-[minmax(0,1fr)_360px] gap-4 h-full">
          <CodePane
            code={displayCode}
            codeError={codeError}
            compareWith={compareWith}
            setCompareWith={setCompareWith}
            altCode={altCode}
            altCodeError={altCodeError}
          />
          <div className="flex flex-col gap-3 min-h-0 overflow-y-auto pr-1 -mr-1">
            <DatasetCard
              inputUrl={inputUrl}
              draft={inputUrlDraft}
              setDraft={setInputUrlDraft}
              onLoad={onLoadDataset}
              onSelectPreset={(url) => {
                setInputUrlDraft(url);
                onLoadDataset(url);
              }}
              numDocs={stats.numDocs}
              status={status}
              isTraining={isTraining}
              isInferring={isInferring}
            />
            <TrainCard
              workerReady={workerReady}
              isTraining={isTraining}
              isInferring={isInferring}
              hasCode={Boolean(code)}
              options={trainOpts}
              setOptions={setTrainOpts}
              onTrain={onTrain}
              onStop={onStop}
              progress={progress}
              progressPct={progressPct}
              trainMs={trainMs}
              status={status}
            />
            <InferCard
              workerReady={workerReady}
              trained={trained}
              isTraining={isTraining}
              isInferring={isInferring}
              options={inferOpts}
              setOptions={setInferOpts}
              onInfer={onInfer}
              onStop={onStop}
              inferMs={inferMs}
            />
            <StatsRow stats={stats} progress={progress} trainMs={trainMs} />
            <Terminal
              logLines={logLines}
              transient={transient}
              terminalRef={terminalRef}
              errorMessage={errorMessage}
            />
            <SamplesGrid samples={samples} phase={phase} isInferring={isInferring} />
          </div>
        </div>
      </main>
    </div>
  );
}

function Header({
  status,
  statusDetail,
}: {
  status: RunStatus;
  statusDetail: string;
}) {
  return (
    <header className="flex items-center justify-between gap-4 px-4 md:px-6 lg:px-6 pt-6 pb-4">
      <div className="flex items-center gap-3">
        <Logo />
        <div className="flex flex-col">
          <h1 className="flex items-center gap-1.5 text-[22px] leading-none font-semibold tracking-tight">
            <span className="bg-gradient-to-r from-violet-300 via-fuchsia-300 to-amber-200 bg-clip-text text-transparent">
              microgpt
            </span>
            <AboutPopover />
          </h1>
          <p className="text-[12px] text-ink-300 mt-1">
            200 lines of pure Python — train a GPT in your browser
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <StatusPill status={status} detail={statusDetail} />
      </div>
    </header>
  );
}

function Logo() {
  return (
    <div className="size-10 rounded-xl bg-gradient-to-br from-violet-500/30 to-fuchsia-500/20 border border-violet-400/30 flex items-center justify-center shadow-lg shadow-violet-500/10">
      <span className="text-[20px] leading-none font-bold text-violet-200 -translate-y-[1px]">µ</span>
    </div>
  );
}

function AboutPopover() {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span ref={containerRef} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="About microgpt"
        aria-expanded={open}
        className="inline-flex items-center justify-center size-4 rounded-full border border-ink-600 bg-ink-800/80 text-[10px] font-semibold text-ink-300 hover:text-ink-100 hover:border-ink-500 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-400/60"
      >
        i
      </button>
      {open ? (
        <div
          role="dialog"
          aria-label="About microgpt"
          className="absolute left-0 top-[calc(100%+8px)] z-50 w-[340px] rounded-xl border border-ink-700 bg-ink-900/95 backdrop-blur-md shadow-xl shadow-black/40 p-3.5"
        >
          <div className="space-y-2.5 text-[12px] leading-relaxed text-ink-200">
            <p>
              <span className="font-mono text-ink-100">microgpt</span> is Andrej
              Karpathy's 200-line, dependency-free Python implementation of a GPT —
              dataset, tokenizer, autograd, GPT-2 architecture, Adam, training and
              inference loops, all in a single file.
            </p>
            <a
              href="https://karpathy.github.io/2026/02/12/microgpt/"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 text-violet-200 hover:text-violet-100 underline decoration-violet-400/40 underline-offset-4 hover:decoration-violet-300 transition-colors"
            >
              karpathy.github.io/2026/02/12/microgpt
              <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden>
                <path
                  d="M6 3h7v7M13 3 4 12"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </a>
          </div>
        </div>
      ) : null}
    </span>
  );
}

function StatusPill({ status, detail }: { status: RunStatus; detail: string }) {
  const cfg = STATUS_CONFIG[status];
  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-[12px] font-medium border ${cfg.pill}`}
    >
      <span
        className={`size-1.5 rounded-full ${cfg.dot} ${cfg.animate ? "animate-pulse" : ""}`}
      />
      <span>{cfg.label}</span>
      {detail ? <span className="text-ink-300 font-normal">· {detail}</span> : null}
    </div>
  );
}

function CodePane({
  code,
  codeError,
  compareWith,
  setCompareWith,
  altCode,
  altCodeError,
}: {
  code: string;
  codeError: string | null;
  compareWith: CompareWith | null;
  setCompareWith: (next: CompareWith | null) => void;
  altCode: string;
  altCodeError: string | null;
}) {
  const activeTab: Impl = compareWith ?? "python";
  // Click an alt tab to open the comparison; click the active alt tab again
  // to close it and return to the single vanilla view.
  const onSelectTab = (tab: Impl) => {
    if (tab === "python") setCompareWith(null);
    else setCompareWith(activeTab === tab ? null : tab);
  };

  const altReady = compareWith != null && altCode.length > 0;
  const filenameLabel = compareWith
    ? `microgpt.py ↔ ${COMPARE_LABELS[compareWith].filename}`
    : "microgpt.py";

  return (
    <section className="relative rounded-2xl overflow-hidden border border-ink-700 bg-ink-900/60 backdrop-blur-sm flex flex-col min-h-0">
      <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-ink-700 bg-ink-900/80">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex gap-1.5 shrink-0">
            <span className="size-2.5 rounded-full bg-rose-400/70" />
            <span className="size-2.5 rounded-full bg-amber-400/70" />
            <span className="size-2.5 rounded-full bg-emerald-400/70" />
          </div>
          <ImplTabs active={activeTab} onSelect={onSelectTab} />
          <span className="text-[11px] text-ink-400 font-mono truncate hidden md:inline">
            {filenameLabel}
          </span>
        </div>
        <div className="text-[11px] text-ink-400 font-mono hidden sm:block tabular-nums shrink-0">
          {code ? `${code.split("\n").length} lines · ${(new Blob([code]).size / 1024).toFixed(1)} KB` : "—"}
        </div>
      </div>
      <div className="flex-1 min-h-0">
        {codeError ? (
          <div className="p-6 text-rose-300 font-mono text-sm">
            Failed to load microgpt.py: {codeError}
          </div>
        ) : !code ? (
          <div className="p-6 text-ink-400 font-mono text-sm">Loading source…</div>
        ) : compareWith ? (
          altCodeError ? (
            <div className="p-6 text-rose-300 font-mono text-sm">
              Failed to load {COMPARE_LABELS[compareWith].filename}: {altCodeError}
            </div>
          ) : !altReady ? (
            <div className="p-6 text-ink-400 font-mono text-sm">
              Loading {COMPARE_LABELS[compareWith].filename}…
            </div>
          ) : (
            <CodeDiff leftCode={code} rightCode={altCode} />
          )
        ) : (
          <CodeViewer code={code} />
        )}
      </div>
    </section>
  );
}

function ImplTabs({
  active,
  onSelect,
}: {
  active: Impl;
  onSelect: (tab: Impl) => void;
}) {
  const tabs: { id: Impl; label: string }[] = [
    { id: "python", label: "python" },
    { id: "pytorch", label: "pytorch" },
    { id: "tinygrad", label: "tinygrad" },
  ];
  return (
    <div
      role="tablist"
      aria-label="Implementation"
      className="inline-flex items-center gap-0.5 rounded-md border border-ink-700 bg-ink-800/60 p-0.5"
    >
      {tabs.map((t) => {
        const isActive = active === t.id;
        return (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onSelect(t.id)}
            className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
              isActive
                ? "bg-ink-700/80 text-ink-100"
                : "text-ink-300 hover:text-ink-100 hover:bg-ink-700/40"
            }`}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

function DatasetCard({
  inputUrl,
  draft,
  setDraft,
  onLoad,
  onSelectPreset,
  numDocs,
  status,
  isTraining,
  isInferring,
}: {
  inputUrl: string;
  draft: string;
  setDraft: (v: string) => void;
  onLoad: () => void;
  onSelectPreset: (url: string) => void;
  numDocs?: number;
  status: RunStatus;
  isTraining: boolean;
  isInferring: boolean;
}) {
  const isLoading = status === "fetching-data";
  const isBusy = isLoading || isTraining || isInferring;
  const trimmed = draft.trim();
  const dirty = trimmed !== inputUrl && trimmed.length > 0;

  // Locate the active preset (if any). When the input URL doesn't match
  // a preset (user pasted a custom URL), arrows still work — they start
  // from index 0 and we surface the "custom" label inline.
  const presetIdx = DATASET_PRESETS.findIndex((p) => p.url === inputUrl);
  const activePreset = presetIdx >= 0 ? DATASET_PRESETS[presetIdx] : null;
  const cyclePreset = (delta: 1 | -1) => {
    const base = presetIdx >= 0 ? presetIdx : 0;
    const next =
      (base + delta + DATASET_PRESETS.length) % DATASET_PRESETS.length;
    onSelectPreset(DATASET_PRESETS[next].url);
  };

  // The four bundled `/*.txt` paths are the only non-http URLs we allow.
  // Anything else must be a real http(s) URL so `fetch()` can reach it —
  // a bare filename or a stray "raw.github.com/…" without scheme would
  // otherwise silently 404 (or worse, resolve relative to the app origin).
  const isKnownLocal = DATASET_PRESETS.some((p) => p.url === trimmed);
  const isHttpUrl = /^https?:\/\//i.test(trimmed);
  const urlError =
    dirty && !isKnownLocal && !isHttpUrl
      ? "URL must start with http:// or https://"
      : null;
  const canSubmit = dirty && !isBusy && !urlError;

  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-900/60 backdrop-blur-sm p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <PhaseBadge variant="dataset" />
          <h2 className="text-[13px] font-semibold tracking-wide text-ink-200 uppercase">
            Dataset
          </h2>
          <div className="ml-1 inline-flex items-center gap-0.5 rounded-md border border-ink-700 bg-ink-800/60 pl-0.5 pr-0.5 py-0.5">
            <button
              type="button"
              onClick={() => cyclePreset(-1)}
              disabled={isBusy}
              aria-label="Previous dataset preset"
              title="Previous dataset"
              className="inline-flex items-center justify-center size-4 rounded text-ink-300 hover:text-ink-100 hover:bg-ink-700/70 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <ChevronLeftIcon />
            </button>
            <span className="min-w-[64px] text-center text-[11px] font-mono text-ink-200 tabular-nums">
              {activePreset ? activePreset.name : "custom"}
            </span>
            <button
              type="button"
              onClick={() => cyclePreset(1)}
              disabled={isBusy}
              aria-label="Next dataset preset"
              title="Next dataset"
              className="inline-flex items-center justify-center size-4 rounded text-ink-300 hover:text-ink-100 hover:bg-ink-700/70 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <ChevronRightIcon />
            </button>
          </div>
        </div>
        <span className="text-[11px] text-ink-400 font-mono tabular-nums">
          {isLoading
            ? "loading…"
            : numDocs != null
            ? `${numDocs.toLocaleString()} docs`
            : "—"}
        </span>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (canSubmit) onLoad();
        }}
        className="flex items-stretch gap-2"
      >
        <div className="relative flex-1 min-w-0">
          <input
            type="text"
            inputMode="url"
            spellCheck={false}
            autoComplete="off"
            value={draft}
            disabled={isBusy && !isLoading}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="https://… (raw text, one document per line)"
            aria-invalid={urlError ? true : undefined}
            aria-describedby={urlError ? "dataset-url-error" : undefined}
            className={`w-full bg-ink-800/80 border rounded-lg px-2.5 py-1.5 text-[12px] font-mono text-ink-100 placeholder:text-ink-400 focus:outline-none focus:ring-2 transition-colors disabled:opacity-60 disabled:cursor-not-allowed ${
              urlError
                ? "border-rose-400/60 focus:border-rose-400 focus:ring-rose-500/30"
                : "border-ink-600 focus:border-violet-400 focus:ring-violet-500/30"
            }`}
            aria-label="Dataset URL"
          />
        </div>
        <button
          type="submit"
          disabled={!canSubmit}
          className="inline-flex items-center justify-center gap-1.5 px-3 h-[34px] rounded-lg font-semibold text-[12px] text-ink-950 bg-gradient-to-r from-violet-300 to-fuchsia-300 hover:brightness-110 active:brightness-95 transition-[filter] disabled:opacity-40 disabled:cursor-not-allowed disabled:bg-none disabled:bg-ink-700 disabled:text-ink-300 shadow-sm shadow-violet-500/20"
        >
          <DownloadIcon />
          <span>{isLoading ? "Loading…" : "Load"}</span>
        </button>
      </form>
      {urlError ? (
        <p
          id="dataset-url-error"
          role="alert"
          className="mt-2 text-[11px] font-mono text-rose-300"
        >
          {urlError}
        </p>
      ) : null}
    </section>
  );
}

function TrainCard({
  workerReady,
  isTraining,
  isInferring,
  hasCode,
  options,
  setOptions,
  onTrain,
  onStop,
  progress,
  progressPct,
  trainMs,
  status,
}: {
  workerReady: boolean;
  isTraining: boolean;
  isInferring: boolean;
  hasCode: boolean;
  options: TrainOptions;
  setOptions: (updater: (prev: TrainOptions) => TrainOptions) => void;
  onTrain: () => void;
  onStop: () => void;
  progress: Progress;
  progressPct: number;
  trainMs: number | null;
  status: RunStatus;
}) {
  const update = <K extends keyof TrainOptions>(key: K) => (value: TrainOptions[K]) =>
    setOptions((prev) => ({ ...prev, [key]: value }));
  const disabled = !hasCode || !workerReady || isInferring;
  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-900/60 backdrop-blur-sm p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <PhaseBadge variant="train" />
          <h2 className="text-[13px] font-semibold tracking-wide text-ink-200 uppercase">
            Training
          </h2>
        </div>
        <span className="text-[11px] text-ink-400 font-mono tabular-nums">
          {trainMs != null ? `${(trainMs / 1000).toFixed(1)}s elapsed` : ""}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-3 mb-3">
        <Slider
          label="Steps"
          hint={`${options.numSteps}`}
          min={10}
          max={1000}
          step={10}
          value={options.numSteps}
          onChange={update("numSteps")}
          disabled={isTraining}
        />
        <NumberInput
          label="Seed"
          value={options.seed}
          onChange={update("seed")}
          disabled={isTraining}
        />
      </div>

      <div className="mb-3">
        <div className="flex items-center justify-between text-[11px] mb-1.5">
          <span className="text-ink-300 font-mono">
            {progress
              ? `step ${progress.step} / ${progress.total}`
              : isTraining
              ? "starting…"
              : "idle"}
          </span>
          <span className="text-ink-300 font-mono tabular-nums">
            {progress ? `loss ${progress.loss.toFixed(4)}` : ""}
          </span>
        </div>
        <div className="relative h-2 rounded-full bg-ink-800 overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-violet-400 via-fuchsia-400 to-amber-300 transition-[width] duration-150 ease-out"
            style={{ width: `${progressPct}%` }}
          />
          {status === "training" && progressPct < 1 ? (
            <div
              className="absolute inset-0 opacity-30"
              style={{
                background:
                  "linear-gradient(90deg, transparent, rgba(167,139,250,0.6), transparent)",
                backgroundSize: "200% 100%",
                animation: "shimmer 1.6s linear infinite",
              }}
            />
          ) : null}
        </div>
      </div>

      {!isTraining ? (
        <button
          type="button"
          onClick={onTrain}
          disabled={disabled}
          className="w-full inline-flex items-center justify-center gap-2 h-11 rounded-xl font-semibold text-[14px] text-ink-950 bg-gradient-to-r from-violet-300 via-fuchsia-300 to-amber-200 hover:brightness-110 active:brightness-95 transition-[filter] disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-violet-500/20"
        >
          <PlayIcon />
          <span>{trainMs != null ? "Re-train" : "Train microgpt"}</span>
        </button>
      ) : (
        <button
          type="button"
          onClick={onStop}
          className="w-full inline-flex items-center justify-center gap-2 h-11 rounded-xl font-semibold text-[14px] text-rose-100 bg-rose-500/20 hover:bg-rose-500/30 border border-rose-400/40 transition-colors"
        >
          <StopIcon />
          <span>Stop training</span>
        </button>
      )}

      <p className="mt-2.5 text-[11px] text-ink-400 leading-relaxed">
        200 steps takes ~30–90s in Pyodide. Full 1000 takes a few minutes.
      </p>
    </section>
  );
}

function InferCard({
  workerReady,
  trained,
  isTraining,
  isInferring,
  options,
  setOptions,
  onInfer,
  onStop,
  inferMs,
}: {
  workerReady: boolean;
  trained: boolean;
  isTraining: boolean;
  isInferring: boolean;
  options: InferOptions;
  setOptions: (updater: (prev: InferOptions) => InferOptions) => void;
  onInfer: () => void;
  onStop: () => void;
  inferMs: number | null;
}) {
  const update = <K extends keyof InferOptions>(key: K) => (value: InferOptions[K]) =>
    setOptions((prev) => ({ ...prev, [key]: value }));
  const disabled = !workerReady || !trained || isTraining;
  return (
    <section
      className={`rounded-2xl border bg-ink-900/60 backdrop-blur-sm p-4 transition-opacity ${
        trained
          ? "border-fuchsia-400/20"
          : "border-ink-700 opacity-90"
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <PhaseBadge variant="infer" />
          <h2 className="text-[13px] font-semibold tracking-wide text-ink-200 uppercase">
            Inference
          </h2>
        </div>
        <span className="text-[11px] text-ink-400 font-mono tabular-nums">
          {inferMs != null
            ? `${(inferMs / 1000).toFixed(2)}s`
            : !trained
            ? "train first"
            : ""}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-3 mb-3">
        <Slider
          label="Temperature"
          hint={options.temperature.toFixed(2)}
          min={0.05}
          max={1.5}
          step={0.05}
          value={options.temperature}
          onChange={(v) => update("temperature")(Number(v.toFixed(2)))}
          disabled={isInferring}
        />
        <Slider
          label="Samples"
          hint={`${options.numSamples}`}
          min={1}
          max={50}
          step={1}
          value={options.numSamples}
          onChange={update("numSamples")}
          disabled={isInferring}
        />
      </div>

      {!isInferring ? (
        <button
          type="button"
          onClick={onInfer}
          disabled={disabled}
          className="w-full inline-flex items-center justify-center gap-2 h-11 rounded-xl font-semibold text-[14px] text-ink-950 bg-gradient-to-r from-fuchsia-300 via-amber-200 to-emerald-200 hover:brightness-110 active:brightness-95 transition-[filter] disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-fuchsia-500/20"
        >
          <SparkIcon />
          <span>{inferMs != null ? "Generate again" : "Generate samples"}</span>
        </button>
      ) : (
        <button
          type="button"
          onClick={onStop}
          className="w-full inline-flex items-center justify-center gap-2 h-11 rounded-xl font-semibold text-[14px] text-rose-100 bg-rose-500/20 hover:bg-rose-500/30 border border-rose-400/40 transition-colors"
        >
          <StopIcon />
          <span>Stop sampling</span>
        </button>
      )}

      <p className="mt-2.5 text-[11px] text-ink-400 leading-relaxed">
        Calls{" "}
        <code className="text-ink-200 bg-ink-800/60 px-1 py-0.5 rounded">infer()</code> on
        the trained model. Tweak temperature and resample without retraining.
      </p>
    </section>
  );
}

function PhaseBadge({ variant }: { variant: "train" | "infer" | "dataset" }) {
  const styles =
    variant === "train"
      ? "from-violet-400/30 to-fuchsia-400/30 text-violet-100 border-violet-400/40"
      : variant === "infer"
      ? "from-fuchsia-400/30 to-amber-300/30 text-fuchsia-100 border-fuchsia-400/40"
      : "from-sky-400/30 to-emerald-300/30 text-sky-100 border-sky-400/40";
  const letter = variant === "train" ? "T" : variant === "infer" ? "I" : "D";
  return (
    <span
      className={`inline-flex items-center justify-center size-5 rounded-md bg-gradient-to-br ${styles} border text-[10px] font-bold`}
    >
      {letter}
    </span>
  );
}

function Slider({
  label,
  hint,
  min,
  max,
  step,
  value,
  onChange,
  disabled,
}: {
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <label className={`flex flex-col gap-1.5 min-w-0 ${disabled ? "opacity-50" : ""}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] font-medium text-ink-200">{label}</span>
        <span className="text-[11px] font-mono text-ink-300 tabular-nums">{hint}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="accent-violet-400 h-1.5"
      />
    </label>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <label className={`flex flex-col gap-1.5 min-w-0 ${disabled ? "opacity-50" : ""}`}>
      <span className="text-[12px] font-medium text-ink-200">{label}</span>
      <input
        type="number"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="bg-ink-800/80 border border-ink-600 rounded-lg px-2.5 py-1.5 text-[13px] font-mono text-ink-100 focus:border-violet-400 focus:outline-none focus:ring-2 focus:ring-violet-500/30 transition-colors disabled:cursor-not-allowed"
      />
    </label>
  );
}

function StatsRow({
  stats,
  progress,
  trainMs,
}: {
  stats: Stats;
  progress: Progress;
  trainMs: number | null;
}) {
  return (
    <section className="grid grid-cols-3 gap-2">
      <StatCard label="Vocab" value={stats.vocabSize} format="int" />
      <StatCard label="Params" value={stats.numParams} format="int" />
      <StatCard
        label={progress ? "Loss" : "Train time"}
        value={progress ? progress.loss : trainMs != null ? trainMs / 1000 : undefined}
        format={progress ? "float" : "duration"}
        accent={progress ? "violet" : "default"}
      />
    </section>
  );
}

function StatCard({
  label,
  value,
  format,
  accent = "default",
}: {
  label: string;
  value: number | undefined;
  format: "int" | "float" | "duration";
  accent?: "default" | "violet";
}) {
  const formatted = useMemo(() => {
    if (value == null) return "—";
    if (format === "int") return value.toLocaleString();
    if (format === "float") return value.toFixed(4);
    if (format === "duration") return `${value.toFixed(1)}s`;
    return String(value);
  }, [value, format]);

  return (
    <div
      className={`rounded-xl border px-3 py-2.5 ${
        accent === "violet"
          ? "border-violet-400/30 bg-violet-500/10"
          : "border-ink-700 bg-ink-900/60"
      }`}
    >
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-medium">{label}</div>
      <div
        className={`mt-1 font-mono tabular-nums text-[18px] leading-tight ${
          accent === "violet" ? "text-violet-100" : "text-ink-100"
        }`}
      >
        {formatted}
      </div>
    </div>
  );
}

function Terminal({
  logLines,
  transient,
  terminalRef,
  errorMessage,
}: {
  logLines: LogLine[];
  transient: string;
  terminalRef: React.RefObject<HTMLDivElement | null>;
  errorMessage: string | null;
}) {
  return (
    <section className="flex-1 min-h-[140px] rounded-2xl border border-ink-700 bg-black/40 overflow-hidden flex flex-col">
      <div className="flex items-center justify-between px-3.5 py-2 border-b border-ink-700 bg-ink-900/60">
        <span className="text-[12px] font-medium text-ink-200">Output</span>
        <span className="text-[11px] text-ink-400 font-mono">stdout · stderr</span>
      </div>
      <div
        ref={terminalRef}
        className="flex-1 overflow-auto font-mono text-[12px] leading-[1.55] px-3.5 py-2.5"
      >
        {logLines.length === 0 && !transient && !errorMessage ? (
          <div className="text-ink-400 italic">
            Output will stream here when you train or sample.
          </div>
        ) : null}
        {logLines.map((line) => (
          <div
            key={line.id}
            className={`whitespace-pre-wrap break-all ${
              line.kind === "stderr"
                ? "text-rose-300"
                : line.kind === "info"
                ? "text-ink-400 italic"
                : "text-ink-100"
            }`}
          >
            {line.text || "\u00A0"}
          </div>
        ))}
        {transient ? (
          <div className="whitespace-pre-wrap break-all text-violet-200">
            {transient}
            <span className="inline-block w-1.5 h-3.5 align-[-2px] bg-violet-300/70 ml-0.5 animate-pulse" />
          </div>
        ) : null}
        {errorMessage ? (
          <pre className="mt-2 whitespace-pre-wrap text-rose-300 text-[12px] leading-[1.55]">
            {errorMessage}
          </pre>
        ) : null}
      </div>
    </section>
  );
}

function SamplesGrid({
  samples,
  phase,
  isInferring,
}: {
  samples: Sample[];
  phase: RunPhase;
  isInferring: boolean;
}) {
  if (samples.length === 0 && !isInferring) return null;
  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-900/60 p-3.5">
      <div className="flex items-center justify-between mb-2.5">
        <h3 className="text-[12px] font-semibold tracking-wide text-ink-200 uppercase">
          Generated outputs
        </h3>
        <span className="text-[11px] text-ink-400 font-mono">
          {samples.length}
          {phase === "infer" && isInferring ? " · streaming" : ""}
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {samples.map((s) => (
          <span
            key={s.id}
            className="inline-flex items-center px-2.5 py-1 rounded-full text-[12px] font-mono bg-gradient-to-r from-fuchsia-500/15 to-amber-300/10 text-fuchsia-100 border border-fuchsia-400/20"
            style={{
              animation: `glow-pulse 0.6s ease-out`,
              animationIterationCount: 1,
            }}
          >
            {s.text || <span className="text-ink-400">·</span>}
          </span>
        ))}
        {isInferring && samples.length === 0 ? (
          <span className="text-[12px] text-ink-400 italic">sampling…</span>
        ) : null}
      </div>
    </section>
  );
}

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M4 2.5a.5.5 0 0 1 .77-.42l8 5.5a.5.5 0 0 1 0 .84l-8 5.5A.5.5 0 0 1 4 13.5v-11Z" />
    </svg>
  );
}

function SparkIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 0.5l1.6 4.4 4.4 1.6-4.4 1.6L8 12.5 6.4 8.1 2 6.5l4.4-1.6L8 0.5ZM3 12l.7 1.9 1.9.7-1.9.7L3 17l-.7-1.7L0.4 14.6l1.9-.7L3 12Z" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <rect x="3" y="3" width="10" height="10" rx="2" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 1.5a.5.5 0 0 1 .5.5v7.293l2.146-2.147a.5.5 0 1 1 .708.708l-3 3a.5.5 0 0 1-.708 0l-3-3a.5.5 0 1 1 .708-.708L7.5 9.293V2a.5.5 0 0 1 .5-.5Z" />
      <path d="M2 12.5a.5.5 0 0 1 .5-.5h11a.5.5 0 0 1 0 1h-11a.5.5 0 0 1-.5-.5Z" />
    </svg>
  );
}

function ChevronLeftIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M10 3.5 5.5 8 10 12.5"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChevronRightIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M6 3.5 10.5 8 6 12.5"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
