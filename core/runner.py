"""
core.runner
===========
The in-app Python execution engine used by every "Code Lab".

A Code Lab is an editable code buffer + a Run button.  Running it:

  * executes the buffer in a persistent per-lab namespace,
  * captures ``stdout`` / ``stderr``,
  * auto-renders any Plotly figure left in ``fig`` (or ``fig1``, ``fig2``…),
  * auto-renders any matplotlib figure that was created,
  * auto-renders a trailing DataFrame / Series / ndarray,
  * shows a clean traceback when something blows up.

The namespace is pre-loaded with the usual scientific stack so the learner can
start typing immediately without boilerplate imports.
"""

from __future__ import annotations

import contextlib
import io
import time
import traceback
from typing import Any

import streamlit as st

# --------------------------------------------------------------------------


def _base_namespace() -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from . import palette as _pal
    from . import nav as _nav
    from .palette import C, CLASS_COLORS, PARULA, SEQ, alpha, ramp, scale

    ns: dict[str, Any] = {
        "np": np, "numpy": np,
        "pd": pd, "pandas": pd,
        "px": px, "go": go, "make_subplots": make_subplots,
        "C": C, "SEQ": SEQ, "CLASS_COLORS": CLASS_COLORS,
        "PARULA": PARULA, "palette": _pal,
        "alpha": alpha, "ramp": ramp, "scale": scale,
        "nav": _nav,
        "st": st,
        "__name__": "__mlplat_lab__",
    }
    return ns


def get_namespace(key: str) -> dict[str, Any]:
    """Persistent namespace for a given lab key (survives Streamlit reruns)."""
    store = st.session_state.setdefault("_lab_namespaces", {})
    if key not in store:
        store[key] = _base_namespace()
    return store[key]


def reset_namespace(key: str) -> None:
    store = st.session_state.setdefault("_lab_namespaces", {})
    store[key] = _base_namespace()


# --------------------------------------------------------------------------


def _render_value(val: Any) -> bool:
    """Render one auto-detected result object. Returns True if rendered."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from .theme import PLOTLY_CONFIG

    if isinstance(val, go.Figure):
        st.plotly_chart(val, width="stretch", config=PLOTLY_CONFIG)
        return True
    if isinstance(val, (pd.DataFrame, pd.Series)):
        st.dataframe(val, width="stretch")
        return True
    if isinstance(val, np.ndarray) and val.ndim <= 2:
        st.dataframe(pd.DataFrame(val), width="stretch")
        return True
    return False


def execute(code: str, key: str) -> dict[str, Any]:
    """
    Execute ``code`` in the lab namespace ``key``.

    Returns a dict with ``ok``, ``stdout``, ``error``, ``elapsed``.
    Rendering of figures/tables happens inside this call so ordering is
    natural (printed text first, then visuals).
    """
    ns = get_namespace(key)
    out = io.StringIO()
    err: str | None = None
    t0 = time.perf_counter()

    # snapshot so we can detect newly created objects
    before = set(ns.keys())

    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(code, f"<lab:{key}>", "exec"), ns, ns)
        ok = True
    except Exception:
        ok = False
        err = traceback.format_exc(limit=8)

    elapsed = time.perf_counter() - t0
    text = out.getvalue()

    if text.strip():
        st.markdown("**Output**")
        st.code(text.rstrip(), language="text")

    if ok:
        # 1) explicit figure names, in order
        rendered = False
        for name in ("fig", "fig1", "fig2", "fig3", "fig4", "figure"):
            if name in ns:
                rendered |= _render_value(ns[name])

        # 2) matplotlib figures
        try:
            import matplotlib.pyplot as plt
            if plt.get_fignums():
                for num in plt.get_fignums():
                    st.pyplot(plt.figure(num))
                plt.close("all")
                rendered = True
        except Exception:
            pass

        # 3) a trailing `result` / `out` / `df` object
        if not rendered:
            for name in ("result", "out", "df", "table", "summary"):
                if name in ns and name not in before:
                    if _render_value(ns[name]):
                        break

    if err:
        st.error("Execution failed")
        st.code(err, language="text")

    return {"ok": ok, "stdout": text, "error": err, "elapsed": elapsed}


# --------------------------------------------------------------------------


def code_lab(title: str,
             code: str,
             key: str,
             height: int | None = None,
             description: str | None = None,
             autorun: bool = False,
             show_editor: bool = True,
             language: str = "python") -> None:
    """
    Render a full Code Lab widget.

    Parameters
    ----------
    title        heading shown next to the RUN badge
    code         the starting source
    key          unique key (also the namespace id)
    height       editor height in px (auto from line count if None)
    description  one-line explanation above the editor
    autorun      run once automatically on first visit
    """
    st.markdown(
        f'<div class="mp-lab"><span class="mp-badge">Code Lab</span>'
        f'<span>{title}</span></div>',
        unsafe_allow_html=True,
    )
    if description:
        st.caption(description)

    edit_key = f"code::{key}"
    if edit_key not in st.session_state:
        st.session_state[edit_key] = code

    n_lines = st.session_state[edit_key].count("\n") + 1
    h = height or min(760, max(150, 21 * n_lines + 40))

    if show_editor:
        with st.expander("✏️  Edit the source  ·  then press **Run**", expanded=False):
            st.text_area("source", key=edit_key, height=h,
                         label_visibility="collapsed")
        st.code(st.session_state[edit_key], language=language)
    else:
        st.code(st.session_state[edit_key], language=language)

    c1, c2, c3, _ = st.columns([1.05, 1.15, 1.15, 4])
    run = c1.button("▶  Run", key=f"run::{key}", type="primary",
                    width="stretch")
    if c2.button("↺  Restore", key=f"rst::{key}", width="stretch"):
        st.session_state[edit_key] = code
        reset_namespace(key)
        st.rerun()
    if c3.button("🧹  Clear vars", key=f"clr::{key}", width="stretch"):
        reset_namespace(key)
        st.toast("Namespace cleared")

    first_key = f"ran::{key}"
    should_run = run or (autorun and not st.session_state.get(first_key, False))

    if should_run:
        st.session_state[first_key] = True
        with st.status("Running…", expanded=True) as status:
            res = execute(st.session_state[edit_key], key)
            if res["ok"]:
                status.update(label=f"Finished in {res['elapsed']*1000:.0f} ms",
                              state="complete", expanded=True)
            else:
                status.update(label="Error", state="error", expanded=True)
