#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""End-to-end integration test for src/microgpt.py.

The source script calls `train(); infer()` at the bottom (full 1000-step run),
so we load it with those two trailing calls stripped, then drive a tiny
train + infer pass against the same module-level globals.

Run with:
    uv run tests/test_microgpt.py
"""

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "microgpt.py"


def load_module_without_main():
    """Exec src/microgpt.py into a fresh namespace, minus the trailing train()/infer()."""
    lines = SRC.read_text().splitlines()
    while lines and lines[-1].strip() in ("", "train()", "infer()"):
        lines.pop()
    code = "\n".join(lines) + "\n"
    namespace = {"__name__": "__microgpt_test__", "__file__": str(SRC)}
    with redirect_stdout(io.StringIO()):
        exec(compile(code, str(SRC), "exec"), namespace)
    return namespace


class TestMicroGPTIntegration(unittest.TestCase):
    def test_end_to_end(self):
        old_cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        try:
            ns = load_module_without_main()

            self.assertGreater(len(ns["docs"]), 0)
            self.assertGreater(ns["vocab_size"], 1)
            self.assertGreater(len(ns["params"]), 0)

            buf = io.StringIO()
            with redirect_stdout(buf):
                ns["train"](num_steps=5)
                ns["infer"](num_samples=3)
            output = buf.getvalue()
            self.assertIn("inference", output)
            self.assertIn("sample", output)
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
