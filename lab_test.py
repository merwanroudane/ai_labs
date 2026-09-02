"""
Execute every Code Lab in every chapter and report failures.

The smoke test only checks that pages *render*; this checks that the code the
learner is invited to run actually runs.

    python lab_test.py            # every lab
    python lab_test.py p04 p05    # only these pages
    python lab_test.py --list     # just list them
"""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import os
import sys
import time
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.CRITICAL)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")

VIEWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "views")

# Labs that legitimately depend on a previous lab's namespace inside the app.
# lab_test runs each lab in isolation, so these are expected to raise.
CHAINED = {"ch02_tune", "ch02_train"}

# Labs that are intentionally slow; still run, but flagged in the report.
SLOW_WARN_S = 20.0


def base_namespace() -> dict:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from core import palette as _pal
    from core import nav as _nav
    from core.palette import (C, CLASS_COLORS, PARULA, SEQ, alpha,
                              ramp, scale)

    class _FakeSt:
        """Minimal stand-in so labs that touch `st` do not explode."""
        def __getattr__(self, _name):
            return lambda *a, **k: None

    return {"np": np, "numpy": np, "pd": pd, "pandas": pd, "px": px, "go": go,
            "make_subplots": make_subplots, "C": C, "SEQ": SEQ,
            "CLASS_COLORS": CLASS_COLORS, "PARULA": PARULA, "palette": _pal,
            "alpha": alpha, "ramp": ramp, "scale": scale, "nav": _nav,
            "st": _FakeSt(), "__name__": "__mlplat_lab__"}


def extract_labs(path: str) -> list[tuple[str, str, int]]:
    """Return [(key, source, lineno), ...] for every code_lab(...) call."""
    tree = ast.parse(io.open(path, encoding="utf-8").read(), filename=path)
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "code_lab"):
            continue
        src = key = None
        # positional: code_lab(title, code, key, ...)
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            src = node.args[1].value
        if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
            key = node.args[2].value
        for kw in node.keywords:
            if kw.arg == "code" and isinstance(kw.value, ast.Constant):
                src = kw.value.value
            if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                key = kw.value.value
        if isinstance(src, str):
            out.append((key or f"line{node.lineno}", src, node.lineno))
    return out


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    list_only = "--list" in sys.argv

    files = sorted(f for f in os.listdir(VIEWS) if f.endswith(".py"))
    if args:
        files = [f for f in files if any(a.lower() in f.lower() for a in args)]

    total = failed = skipped = 0
    failures = []

    for fn in files:
        path = os.path.join(VIEWS, fn)
        labs = extract_labs(path)
        if not labs:
            continue
        print(f"\n{fn}  ({len(labs)} labs)")
        for key, src, lineno in labs:
            total += 1
            if list_only:
                print(f"   · {key:<24} line {lineno:>5}  "
                      f"{len(src.splitlines()):>3} lines")
                continue

            # 1. syntax must be valid no matter what
            try:
                compile(src, f"<lab:{key}>", "exec")
            except SyntaxError as exc:
                failed += 1
                failures.append((fn, key, f"SyntaxError line {exc.lineno}: {exc.msg}"))
                print(f"   ✗ {key:<24} SYNTAX ERROR line {exc.lineno}: {exc.msg}")
                continue

            if key in CHAINED:
                skipped += 1
                print(f"   ~ {key:<24} skipped (chains off a previous lab)")
                continue

            ns = base_namespace()
            buf = io.StringIO()
            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(src, f"<lab:{key}>", "exec"), ns, ns)
                dt = time.perf_counter() - t0
                flag = "  [SLOW]" if dt > SLOW_WARN_S else ""
                print(f"   ✓ {key:<24} {dt:>6.2f}s  "
                      f"{len(buf.getvalue().splitlines()):>3} output lines{flag}")
            except Exception:
                dt = time.perf_counter() - t0
                failed += 1
                tb = traceback.format_exc().strip().splitlines()
                msg = tb[-1] if tb else "?"
                loc = [l for l in tb if f"<lab:{key}>" in l]
                where = loc[-1].strip() if loc else ""
                failures.append((fn, key, f"{msg}   {where}"))
                print(f"   ✗ {key:<24} {dt:>6.2f}s  {msg}")

    if list_only:
        print(f"\n{total} labs found")
        return 0

    print(f"\n{'=' * 70}")
    print(f"labs run: {total - skipped}   passed: {total - skipped - failed}   "
          f"failed: {failed}   skipped: {skipped}")
    if failures:
        print("\nFAILURES")
        for fn, key, msg in failures:
            print(f"  {fn} :: {key}\n      {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
