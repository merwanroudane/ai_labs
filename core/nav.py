"""
core.nav
========
Sub-section navigation inside a chapter page, plus progress tracking.

A chapter page declares::

    SECTIONS = [
        ("1.1", "What Is Machine Learning?", render_1_1),
        ("1.2", "Why Use Machine Learning?", render_1_2),
        ...
    ]
    nav.render_chapter("ch01", SECTIONS)

which produces:
  * a sidebar radio listing every sub-section
  * "Previous / Next sub-section" buttons at the bottom
  * a completion tick stored in session_state
"""

from __future__ import annotations

from typing import Callable, Sequence

import streamlit as st

from .palette import C

Section = tuple[str, str, Callable[[], None]]


# --------------------------------------------------------------------------


def _progress_store() -> dict:
    return st.session_state.setdefault("_progress", {})


def mark_done(chapter: str, sec_num: str) -> None:
    _progress_store().setdefault(chapter, set()).add(sec_num)


def chapter_progress(chapter: str, total: int) -> float:
    done = len(_progress_store().get(chapter, set()))
    return 0.0 if total == 0 else done / total


def overall_progress(chapters: dict[str, int]) -> float:
    done = sum(len(_progress_store().get(ch, set())) for ch in chapters)
    total = sum(chapters.values()) or 1
    return done / total


# --------------------------------------------------------------------------


def render_chapter(chapter_id: str,
                   sections: Sequence[Section],
                   sidebar_title: str = "Sub-sections") -> None:
    """Render the sub-section picker and the currently selected sub-section."""
    labels = [f"{n} · {t}" for n, t, _ in sections]
    state_key = f"sec::{chapter_id}"

    if state_key not in st.session_state:
        st.session_state[state_key] = labels[0]
    # guard against a stale label after code edits
    if st.session_state[state_key] not in labels:
        st.session_state[state_key] = labels[0]

    with st.sidebar:
        st.markdown(f'<div class="mp-sbtitle">{sidebar_title}</div>',
                    unsafe_allow_html=True)
        st.radio("subsection", labels, key=state_key,
                 label_visibility="collapsed")

        done = _progress_store().get(chapter_id, set())
        pct = len(done) / len(sections)
        st.markdown(
            f'<div class="mp-sbtitle" style="margin-top:14px">Chapter progress</div>',
            unsafe_allow_html=True)
        st.progress(pct, text=f"{len(done)} / {len(sections)} read")

    idx = labels.index(st.session_state[state_key])
    num, title, fn = sections[idx]

    # ---- breadcrumb -------------------------------------------------------
    st.markdown(
        f'<div style="font-size:.78rem;color:{C["muted"]};letter-spacing:.08em;'
        f'text-transform:uppercase;font-weight:650;margin-bottom:2px">'
        f'Sub-section {idx + 1} of {len(sections)}</div>',
        unsafe_allow_html=True)

    fn()

    mark_done(chapter_id, num)

    # ---- prev / next ------------------------------------------------------
    st.markdown("<hr/>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2.2, 1.6, 2.2])
    if idx > 0:
        if c1.button(f"←  {sections[idx-1][0]} · {sections[idx-1][1]}",
                     width="stretch", key=f"prev::{chapter_id}"):
            st.session_state[state_key] = labels[idx - 1]
            st.rerun()
    c2.markdown(
        f'<div style="text-align:center;color:{C["muted"]};font-size:.85rem;'
        f'padding-top:8px">{idx + 1} / {len(sections)}</div>',
        unsafe_allow_html=True)
    if idx < len(sections) - 1:
        if c3.button(f"{sections[idx+1][0]} · {sections[idx+1][1]}  →",
                     width="stretch", key=f"next::{chapter_id}",
                     type="primary"):
            st.session_state[state_key] = labels[idx + 1]
            st.rerun()


def sidebar_tools(chapter_id: str) -> None:
    """Small per-chapter utility block in the sidebar."""
    with st.sidebar:
        st.markdown('<div class="mp-sbtitle">Display</div>',
                    unsafe_allow_html=True)
        from .palette import COLORSCALES, DEFAULT_COLORSCALE
        st.selectbox("Colourscale for surfaces & heatmaps",
                     list(COLORSCALES.keys()),
                     index=list(COLORSCALES).index(DEFAULT_COLORSCALE),
                     key="cscale")
        st.slider("Animation speed (ms per frame)", 40, 600,
                  st.session_state.get("anim_ms", 200), 20, key="anim_ms")


def anim_ms(default: int = 200) -> int:
    return int(st.session_state.get("anim_ms", default))


def cscale():
    from .palette import resolve_colorscale
    return resolve_colorscale(st.session_state.get("cscale", "Parula"))
