"""
Headless smoke test for every page of the platform.

Runs each view with Streamlit's AppTest, walks *every* sub-section of every
chapter, and reports any exception. Usage:

    python smoke_test.py             # all pages
    python smoke_test.py p04 p05     # only pages whose filename contains these
"""

from __future__ import annotations

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
for _n in ("streamlit.runtime.scriptrunner_utils.script_run_context",
           "streamlit.runtime.state.session_state_proxy",
           "streamlit.runtime.caching.cache_data_api",
           "streamlit.elements.lib.policies"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

from streamlit.testing.v1 import AppTest  # noqa: E402

VIEWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "views")
TIMEOUT = 300


def run_one(path: str, subsection: str | None = None):
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
    if subsection is not None:
        key = None
        # discover the radio key by running once
        probe = AppTest.from_file(path, default_timeout=TIMEOUT).run()
        for r in probe.radio:
            if r.key and r.key.startswith("sec::"):
                key = r.key
                break
        if key is None:
            return at.run(), None
        at.session_state[key] = subsection
    return at.run(), None


def main() -> int:
    only = [a.lower() for a in sys.argv[1:]]
    files = sorted(f for f in os.listdir(VIEWS) if f.endswith(".py"))
    if only:
        files = [f for f in files if any(o in f.lower() for o in only)]

    failures = 0
    total_sections = 0

    for fn in files:
        path = os.path.join(VIEWS, fn)
        t0 = time.perf_counter()
        try:
            at = AppTest.from_file(path, default_timeout=TIMEOUT).run()
        except Exception as exc:                                   # noqa: BLE001
            print(f"[LOAD FAIL] {fn}: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        if at.exception:
            for e in at.exception:
                print(f"[FAIL] {fn} (first section)\n        {e.value}")
            failures += 1
            continue

        # find the sub-section radio and walk every option
        sec_key, options = None, []
        for r in at.radio:
            if r.key and r.key.startswith("sec::"):
                sec_key, options = r.key, list(r.options)
                break

        if not sec_key:
            print(f"[ok]   {fn}  (single page, {time.perf_counter()-t0:.1f}s)")
            total_sections += 1
            continue

        bad = []
        for opt in options:
            try:
                a2 = AppTest.from_file(path, default_timeout=TIMEOUT)
                a2.session_state[sec_key] = opt
                a2.run()
                if a2.exception:
                    bad.append((opt, a2.exception[0].value))
            except Exception as exc:                               # noqa: BLE001
                bad.append((opt, f"{type(exc).__name__}: {exc}"))
            total_sections += 1

        dt = time.perf_counter() - t0
        if bad:
            failures += 1
            print(f"[FAIL] {fn}  ({len(bad)}/{len(options)} sections, {dt:.1f}s)")
            for opt, err in bad:
                first = str(err).strip().splitlines()
                print(f"        - {opt}\n          {first[-1] if first else err}")
        else:
            print(f"[ok]   {fn}  ({len(options)} sections, {dt:.1f}s)")

    print(f"\n{'=' * 62}")
    print(f"pages checked: {len(files)}   sections run: {total_sections}   "
          f"failing pages: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
