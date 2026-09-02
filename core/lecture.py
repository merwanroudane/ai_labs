"""
core.lecture
============
The lecture rendering DSL.

Every chapter page is written as a list of *subsections*; each subsection is a
plain Python function that calls these helpers.  Keeping the vocabulary small
and fixed is what makes 19 chapters feel like one course instead of 19
different websites.

Vocabulary
----------
hero(...)            chapter banner
section(n, title)    numbered section rule
sub(title)           lighter sub-heading
lead(text)           highlighted opening paragraph
md(text)             ordinary markdown prose
math(tex, caption)   centred display equation
where(dict)          "where ..." symbol table under an equation
derive(steps)        numbered derivation, each step = (prose, latex|None)
idea/tip/note/warn/pitfall/proof(title, body)   coloured callouts
keypoints(list)      dark summary card
table(headers, rows) styled HTML table
figure(fig, caption) plotly chart with the platform config
anim_header(title)   the ANIMATION badge line
quiz(...)            single-choice self-check
exercise(...)        end-of-chapter exercise with reveal
refs(list)           further-reading block
"""

from __future__ import annotations

import html
import re
from typing import Iterable, Sequence

import streamlit as st

from .palette import C
from .theme import PLOTLY_CONFIG

# ==========================================================================
# Inline maths inside raw HTML
# --------------------------------------------------------------------------
# Streamlit renders `$...$` with KaTeX only inside `st.markdown` *text*, never
# inside HTML we inject ourselves.  Since the styled tables, callouts and key
# cards are all raw HTML, every `$...$` inside them would show as literal
# LaTeX.  `tex()` converts a useful subset of LaTeX to styled Unicode HTML so
# those components display real mathematics.
# ==========================================================================

_SCRIPT = {"A": "𝒜", "B": "ℬ", "C": "𝒞", "D": "𝒟", "E": "ℰ", "F": "ℱ",
           "G": "𝒢", "H": "ℋ", "I": "ℐ", "J": "𝒥", "K": "𝒦", "L": "ℒ",
           "M": "ℳ", "N": "𝒩", "O": "𝒪", "P": "𝒫", "Q": "𝒬", "R": "ℛ",
           "S": "𝒮", "T": "𝒯", "U": "𝒰", "V": "𝒱", "W": "𝒲", "X": "𝒳",
           "Y": "𝒴", "Z": "𝒵"}

_BB = {"A": "𝔸", "C": "ℂ", "E": "𝔼", "N": "ℕ", "P": "ℙ", "Q": "ℚ",
       "R": "ℝ", "S": "𝕊", "Z": "ℤ", "1": "𝟙"}

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
}

_SYMBOLS = {
    r"\sum": "∑", r"\prod": "∏", r"\int": "∫", r"\infty": "∞",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\ll": "≪", r"\gg": "≫", r"\neq": "≠", r"\ne": "≠",
    r"\approx": "≈", r"\equiv": "≡", r"\sim": "∼", r"\propto": "∝",
    r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←",
    r"\Rightarrow": "⇒", r"\Leftarrow": "⇐", r"\Longrightarrow": "⟹",
    r"\Leftrightarrow": "⇔", r"\mapsto": "↦",
    r"\in": "∈", r"\notin": "∉", r"\subseteq": "⊆", r"\subset": "⊂",
    r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅", r"\forall": "∀",
    r"\exists": "∃", r"\times": "×", r"\cdot": "·", r"\cdots": "⋯",
    r"\dots": "…", r"\ldots": "…", r"\pm": "±", r"\mp": "∓",
    r"\nabla": "∇", r"\partial": "∂", r"\ell": "ℓ", r"\sqrt": "√",
    r"\lVert": "‖", r"\rVert": "‖", r"\|": "‖",
    r"\lvert": "|", r"\rvert": "|",
    r"\langle": "⟨", r"\rangle": "⟩",
    r"\left": "", r"\right": "", r"\!": "", r"\,": "\u2009",
    r"\;": "\u2009", r"\:": "\u2009", r"\ ": " ",
    r"\quad": "\u2003", r"\qquad": "\u2003\u2003",
    r"\{": "{", r"\}": "}", r"\%": "%", r"\&": "&", r"\#": "#",
    r"\log": "log", r"\exp": "exp", r"\min": "min", r"\max": "max",
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan", r"\det": "det",
    r"\tr": "tr", r"\operatorname": "", r"\text": "", r"\mathrm": "",
    r"\displaystyle": "", r"\overset": "", r"\underbrace": "",
    r"\top": "ᵀ", r"\circ": "∘", r"\star": "⋆", r"\odot": "⊙",
    r"\oplus": "⊕", r"\otimes": "⊗", r"\perp": "⊥", r"\angle": "∠",
}

_SUP = str.maketrans("0123456789+-=()aincdksxyTt",
                     "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃⁱⁿᶜᵈᵏˢˣʸᵀᵗ")
_SUB = str.maketrans("0123456789+-=()aeijkmnopstxy",
                     "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₖₘₙₒₚₛₜₓᵧ")

_COMBINING = {"hat": "\u0302", "tilde": "\u0303", "bar": "\u0304",
              "vec": "\u20d7", "dot": "\u0307"}


def _braced(s: str, i: int) -> tuple[str, int]:
    """Read a balanced {...} group starting at index i (which must be '{')."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _arg(s: str, i: int) -> tuple[str, int]:
    """Read one LaTeX argument at index i: either {group} or a single token."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return _braced(s, i)
    if s[i] == "\\":
        m = re.match(r"\\[A-Za-z]+", s[i:])
        if m:
            return m.group(0), i + m.end()
    return s[i], i + 1


def _tex_inner(s: str) -> str:
    """Convert one LaTeX fragment (no $ delimiters) to Unicode-ish HTML."""
    out, i = [], 0
    while i < len(s):
        ch = s[i]

        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+)\*?", s[i:])
            if not m:
                # escaped punctuation like \{ \, \;
                two = s[i:i + 2]
                out.append(_SYMBOLS.get(two, two[1:]))
                i += 2
                continue
            name, end = m.group(1), i + m.end()

            if name in ("mathcal", "mathscr"):
                a, end = _arg(s, end)
                out.append("".join(_SCRIPT.get(c, c) for c in a))
            elif name in ("mathbb", "mathbf1"):
                a, end = _arg(s, end)
                out.append("".join(_BB.get(c, c) for c in a))
            elif name in ("mathbf", "boldsymbol", "bm", "pmb"):
                a, end = _arg(s, end)
                out.append("<b>" + _tex_inner(a) + "</b>")
            elif name in ("mathrm", "text", "textrm", "mathsf", "operatorname"):
                a, end = _arg(s, end)
                out.append(f'<span style="font-style:normal">{a}</span>')
            elif name in _COMBINING:
                a, end = _arg(s, end)
                inner = _tex_inner(a)
                out.append(inner + _COMBINING[name] if len(a) == 1
                           else inner + _COMBINING[name])
            elif name == "frac":
                a, end = _arg(s, end)
                b, end = _arg(s, end)
                out.append(f"{_tex_inner(a)}⁄{_tex_inner(b)}")
            elif name in ("dfrac", "tfrac"):
                a, end = _arg(s, end)
                b, end = _arg(s, end)
                out.append(f"{_tex_inner(a)}⁄{_tex_inner(b)}")
            elif name in _GREEK:
                out.append(_GREEK[name])
            elif "\\" + name in _SYMBOLS:
                out.append(_SYMBOLS["\\" + name])
            elif name in ("underbrace", "overbrace"):
                a, end = _arg(s, end)
                out.append(_tex_inner(a))
            else:
                out.append(name)
            i = end
            continue

        if ch in "_^":
            a, j = _arg(s, i + 1)
            inner = _tex_inner(a)
            plain = re.sub(r"<[^>]+>", "", inner)
            tbl = _SUB if ch == "_" else _SUP
            if plain and all(c in tbl for c in map(ord, plain)):
                out.append(plain.translate(tbl))
            else:
                tag = "sub" if ch == "_" else "sup"
                out.append(f"<{tag}>{inner}</{tag}>")
            i = j
            continue

        if ch in "{}":
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def tex(s: str) -> str:
    """Replace every ``$...$`` span in ``s`` with styled Unicode HTML."""
    if "$" not in s:
        return s

    def repl(m: re.Match) -> str:
        body = _tex_inner(m.group(1))
        return (f'<span style="font-family:Cambria,\'Times New Roman\',serif;'
                f'font-style:italic;font-size:1.04em;'
                f'color:{C["primary_dark"]}">{body}</span>')

    return re.sub(r"\$([^$]+)\$", repl, s)

# --------------------------------------------------------------------------
# Headings
# --------------------------------------------------------------------------


def hero(kicker: str, title: str, blurb: str, chips: Sequence[str] = ()) -> None:
    chip_html = "".join(f'<span class="mp-chip">{html.escape(c)}</span>'
                        for c in chips)
    st.markdown(
        f"""
        <div class="mp-hero">
          <div class="mp-kicker">{html.escape(kicker)}</div>
          <h1>{html.escape(title)}</h1>
          <p>{tex(blurb)}</p>
          <div class="mp-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(num: str, title: str) -> None:
    st.markdown(
        f'<div class="mp-sec"><span class="mp-num">{html.escape(num)}</span>'
        f'<h2>{tex(title)}</h2></div>',
        unsafe_allow_html=True,
    )


def sub(title: str) -> None:
    st.markdown(f'<div class="mp-sub">{tex(title)}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Prose
# --------------------------------------------------------------------------


def lead(text: str) -> None:
    st.markdown(f'<div class="mp-lead">{tex(text)}</div>', unsafe_allow_html=True)


def md(text: str) -> None:
    st.markdown(text)


def spacer(px: int = 10) -> None:
    st.markdown(f'<div style="height:{px}px"></div>', unsafe_allow_html=True)


def rule() -> None:
    st.markdown("<hr/>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Mathematics
# --------------------------------------------------------------------------


def math(expr: str, caption: str | None = None) -> None:
    st.latex(expr)
    if caption:
        st.markdown(f'<div class="mp-eqcap">{caption}</div>',
                    unsafe_allow_html=True)


def where(symbols: dict[str, str], intro: str = "where") -> None:
    """Symbol glossary rendered directly beneath an equation."""
    rows = "".join(
        f"<tr><td style='width:20%;white-space:nowrap'>"
        f"<code style='background:{C['primary_soft']};color:{C['primary_dark']};"
        f"padding:2px 7px;border-radius:5px'>{_tex_inner(k)}</code></td>"
        f"<td>{tex(v)}</td></tr>"
        for k, v in symbols.items()
    )
    st.markdown(
        f"<div style='font-size:.9rem;color:{C['ink_soft']};margin:-4px 0 14px 0'>"
        f"<em>{intro}</em>"
        f"<table style='width:100%;border-collapse:collapse;margin-top:5px'>"
        f"{rows}</table></div>",
        unsafe_allow_html=True,
    )


def derive(steps: Sequence[tuple], title: str = "Derivation") -> None:
    """
    A numbered derivation.

    ``steps`` is a sequence of ``(prose, latex_or_None)`` tuples.
    """
    with st.expander(f"📐  {title}  —  click to unfold every algebraic step",
                     expanded=False):
        for i, step in enumerate(steps, start=1):
            prose, formula = (step + (None,))[:2] if len(step) == 1 else step
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:flex-start;"
                f"margin-top:{'2px' if i == 1 else '14px'}'>"
                f"<span style='background:{C['proof']};color:#fff;border-radius:6px;"
                f"min-width:22px;height:22px;display:inline-flex;align-items:center;"
                f"justify-content:center;font-size:.75rem;font-weight:700;'>{i}</span>"
                f"<span style='line-height:1.6'>{tex(prose)}</span></div>",
                unsafe_allow_html=True,
            )
            if formula:
                st.latex(formula)


# --------------------------------------------------------------------------
# Callouts
# --------------------------------------------------------------------------


def _call(kind: str, label: str, title: str, body: str) -> None:
    head = label + (f" · {html.escape(title)}" if title else "")
    st.markdown(
        f'<div class="mp-call mp-{kind}"><span class="mp-ct">{head}</span>{tex(body)}</div>',
        unsafe_allow_html=True,
    )


def _flex(kind: str, label: str, title: str, body: str | None) -> None:
    """Callouts accept either (title, body) or just (body)."""
    if body is None:
        title, body = "", title
    _call(kind, label, title, body)


def idea(title: str, body: str | None = None) -> None:
    _flex("idea", "💡 Key idea", title, body)


def tip(title: str, body: str | None = None) -> None:
    _flex("tip", "🧭 Practice tip", title, body)


def note(title: str, body: str | None = None) -> None:
    _flex("note", "📌 Note", title, body)


def warn(title: str, body: str | None = None) -> None:
    _flex("warn", "⚠️ Caution", title, body)


def pitfall(title: str, body: str | None = None) -> None:
    _flex("pitfall", "🚨 Common pitfall", title, body)


def proof(title: str, body: str | None = None) -> None:
    _flex("proof", "∎ Result", title, body)


def codenote(title: str, body: str | None = None) -> None:
    _flex("code", "⌨️ In code", title, body)


def keypoints(items: Iterable[str], title: str = "What to remember") -> None:
    lis = "".join(f"<li>{tex(i)}</li>" for i in items)
    st.markdown(
        f'<div class="mp-keys"><h4>{html.escape(title)}</h4><ul>{lis}</ul></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Tables & figures
# --------------------------------------------------------------------------


def table(headers: Sequence[str], rows: Sequence[Sequence[str]],
          caption: str | None = None) -> None:
    th = "".join(f"<th>{tex(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{tex(str(c))}</td>" for c in row) + "</tr>"
                 for row in rows)
    st.markdown(
        f'<div class="mp-tbl"><table><thead><tr>{th}</tr></thead>'
        f'<tbody>{tr}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f'<div class="mp-eqcap">{caption}</div>',
                    unsafe_allow_html=True)


def figure(fig, caption: str | None = None, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)
    if caption:
        st.markdown(f'<div class="mp-eqcap">{caption}</div>',
                    unsafe_allow_html=True)


def anim_header(title: str, hint: str = "press ▶ Play, or drag the slider") -> None:
    st.markdown(
        f'<div class="mp-animhdr"><span class="mp-badge">▶ Animation</span>'
        f'<span>{html.escape(title)}</span>'
        f'<span style="font-weight:400;color:{C["muted"]};font-size:.85rem">'
        f'— {html.escape(hint)}</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------


def quiz(question: str, options: Sequence[str], answer: int,
         explanation: str, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**🧠 Check yourself** — {question}")
        choice = st.radio("options", options, index=None, key=f"quiz::{key}",
                          label_visibility="collapsed")
        if choice is not None:
            if options.index(choice) == answer:
                st.success(f"Correct. {explanation}")
            else:
                st.error(f"Not quite. The answer is **{options[answer]}**. "
                         f"{explanation}")


def exercise(number: int, prompt: str, solution: str,
             code: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**Exercise {number}.** {prompt}")
        with st.expander("Show worked answer"):
            st.markdown(solution)
            if code:
                st.code(code, language="python")


def refs(items: Sequence[tuple[str, str]], title: str = "Further reading") -> None:
    """``items`` = sequence of (label, url-or-citation)."""
    body = "".join(
        f"<li style='margin-bottom:6px'>{tex(lab)} "
        + (f"— <a href='{url}' target='_blank' style='color:{C['info']}'>{url}</a>"
           if url.startswith("http") else f"— <em>{url}</em>")
        + "</li>"
        for lab, url in items
    )
    st.markdown(
        f"<div style='background:{C['surface_alt']};border-radius:13px;"
        f"padding:14px 20px;margin-top:22px;border:1px solid {C['line']}'>"
        f"<div style='font-weight:700;font-size:.8rem;letter-spacing:.1em;"
        f"text-transform:uppercase;color:{C['muted']};margin-bottom:8px'>"
        f"{html.escape(title)}</div><ul style='margin:0;padding-left:18px;"
        f"font-size:.92rem;color:{C['ink_soft']}'>{body}</ul></div>",
        unsafe_allow_html=True,
    )
