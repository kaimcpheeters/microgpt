#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tinygrad",
# ]
# ///
"""End-to-end integration test for src/microgpt_tinygrad.py.

Parallel to test_microgpt.py: load the script with the trailing
`train(); infer()` stripped, then drive a tiny train + infer pass.

Run with:
    uv run tests/test_microgpt_tinygrad.py
"""

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "microgpt_tinygrad.py"


def load_module_without_main():
    lines = SRC.read_text().splitlines()
    while lines and lines[-1].strip() in ("", "train()", "infer()"):
        lines.pop()
    code = "\n".join(lines) + "\n"
    namespace = {"__name__": "__microgpt_tinygrad_test__", "__file__": str(SRC)}
    with redirect_stdout(io.StringIO()):
        exec(compile(code, str(SRC), "exec"), namespace)
    return namespace


class TestMicroGPTTinygradIntegration(unittest.TestCase):
    def test_end_to_end(self):
        from tinygrad import Tensor

        old_cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        try:
            ns = load_module_without_main()

            self.assertGreater(len(ns["docs"]), 0)
            self.assertGreater(ns["vocab_size"], 1)
            self.assertGreater(len(ns["params"]), 0)
            for p in ns["params"]:
                self.assertIsInstance(p, Tensor)
                self.assertTrue(p.requires_grad)
            self.assertIn("Adam", ns)
            self.assertIsInstance(ns["optimizer"], ns["Adam"])

            buf = io.StringIO()
            with redirect_stdout(buf):
                ns["train"](num_steps=3)
                ns["infer"](num_samples=2)
            output = buf.getvalue()
            self.assertIn("inference", output)
            self.assertIn("sample", output)
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
