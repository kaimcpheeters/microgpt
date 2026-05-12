#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "tinygrad",
# ]
# ///
"""Benchmark training and inference time across all microgpt implementations.

  pure python  src/microgpt.py
  pytorch      src/microgpt_pytorch.py
  tinygrad     src/microgpt_tinygrad.py

Loads each script (minus its trailing `train(); infer()` finale) into a fresh
namespace, then drives a configurable number of train steps and infer samples
against the resulting module-level globals. Each script seeds RNGs at import
time, so a fresh exec gives a reproducible starting point per run.

Usage:
    uv run benchmarks/benchmark.py
    uv run benchmarks/benchmark.py --train-steps 1000 --infer-samples 20
    uv run benchmarks/benchmark.py --only pytorch,tinygrad --input-url <URL>
"""

import argparse
import io
import os
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

IMPLEMENTATIONS = [
    ("pure python", SRC_DIR / "microgpt.py"),
    ("pytorch",     SRC_DIR / "microgpt_pytorch.py"),
    ("tinygrad",    SRC_DIR / "microgpt_tinygrad.py"),
]


def load_module_without_main(path: Path) -> dict:
    """Exec the .py script into a fresh namespace, minus the trailing train()/infer()."""
    lines = path.read_text().splitlines()
    while lines and lines[-1].strip() in ("", "train()", "infer()"):
        lines.pop()
    code = "\n".join(lines) + "\n"
    namespace = {"__name__": f"__{path.stem}_bench__", "__file__": str(path)}
    with redirect_stdout(io.StringIO()):
        exec(compile(code, str(path), "exec"), namespace)
    return namespace


def _time(fn) -> float:
    start = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        fn()
    return time.perf_counter() - start


def benchmark(label: str, path: Path, train_steps: int, infer_samples: int) -> dict[str, float]:
    print(f"\n=== {label}: {path.relative_to(REPO_ROOT)} ===")

    t0 = time.perf_counter()
    ns = load_module_without_main(path)
    load_t = time.perf_counter() - t0
    print(f"  {'load + param init':<32} {load_t:8.3f} s")

    train_t = _time(lambda: ns["train"](num_steps=train_steps))
    print(f"  {'train(num_steps=' + str(train_steps) + ')':<32} {train_t:8.3f} s"
          f"   ({train_t / train_steps * 1000:8.2f} ms/step)")

    infer_t = _time(lambda: ns["infer"](num_samples=infer_samples))
    print(f"  {'infer(num_samples=' + str(infer_samples) + ')':<32} {infer_t:8.3f} s"
          f"   ({infer_t / infer_samples * 1000:8.2f} ms/sample)")

    return {"load": load_t, "train": train_t, "infer": infer_t}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=100,
                        help="number of training steps per implementation (default: 100)")
    parser.add_argument("--infer-samples", type=int, default=10,
                        help="number of samples to generate per implementation (default: 10)")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated subset of impls to run, e.g. 'pytorch,tinygrad' "
                             "(default: run all)")
    parser.add_argument("--input-url", type=str, default=None,
                        help="dataset URL; pre-fetches to the URL-derived filename (e.g. names.txt) and sets "
                             "MICROGPT_INPUT_URL so every impl's load_dataset() short-circuits on the cache "
                             "(default: each impl's built-in)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override MICROGPT_SEED for impls that read it (pytorch/tinygrad; pure python "
                             "uses its hardcoded seed by design)")
    args = parser.parse_args()

    # `--only pytorch,tinygrad` filters IMPLEMENTATIONS to the requested labels (order preserved from the
    # CLI so the first --only entry is the speedup baseline in the summary table).
    impls = IMPLEMENTATIONS
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        by_label = {label: (label, path) for label, path in IMPLEMENTATIONS}
        missing = [w for w in wanted if w not in by_label]
        if missing:
            parser.error(f"--only: unknown label(s) {missing}; valid: {list(by_label)}")
        impls = [by_label[w] for w in wanted]

    # --input-url: set env var for the env-aware ports' load_dataset(), AND pre-fetch the file to
    # the URL-derived basename. Every impl's load_dataset() short-circuits on `os.path.exists(fname)`
    # where `fname = input_url.rsplit('/', 1)[-1]`, so a single pre-fetch is shared across all three.
    # Vanilla microgpt.py has no env-var support (kept hardcoded so the GUI source-rewriter stays
    # trivial); it picks up the cache because it derives the same fname from its built-in URL.
    if args.input_url:
        os.environ["MICROGPT_INPUT_URL"] = args.input_url
        fname = args.input_url.rsplit("/", 1)[-1] or "input.txt"
        cached = REPO_ROOT / fname
        cached.unlink(missing_ok=True)
        import urllib.request
        print(f"[setup] fetching {args.input_url} -> {cached.name}")
        urllib.request.urlretrieve(args.input_url, cached)
    if args.seed is not None:
        os.environ["MICROGPT_SEED"] = str(args.seed)

    old_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    results: dict[str, dict[str, float]] = {}
    try:
        for label, path in impls:
            results[label] = benchmark(label, path, args.train_steps, args.infer_samples)
    finally:
        os.chdir(old_cwd)

    print("\n=== summary ===")
    labels = [label for label, _ in impls]
    header_cells = [f"{'phase':<8}"] + [f"{label:>22}" for label in labels]
    header = "  " + " ".join(header_cells)
    print(header)
    print("  " + "-" * (len(header) - 2))

    baseline = labels[0]
    for phase in ("train", "infer"):
        row = [f"{phase:<8}"]
        base_t = results[baseline][phase]
        for i, label in enumerate(labels):
            t = results[label][phase]
            if i == 0:
                cell = f"{t:>13.3f} s"
                row.append(f"{cell:>22}")
            else:
                speedup = base_t / t if t > 0 else float("inf")
                tag = f"{t:8.3f} s ({speedup:5.2f}x)"
                row.append(f"{tag:>22}")
        print("  " + " ".join(row))


if __name__ == "__main__":
    main()
