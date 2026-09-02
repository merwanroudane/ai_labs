"""
core.palette
============
Central colour system for the whole platform.

Everything visual in the platform pulls its colours from here so that plots,
callouts, badges, headers and code blocks stay perfectly consistent.

Contains:
  * A semantic palette (primary / accent / success / warning / danger / ...)
  * A categorical sequence for discrete series (colour-blind aware ordering)
  * Continuous colourscales:  Parula, Jet, Turbo, BlueRed, Sinha, Viridis
  * Helpers to build plotly colourscales and to interpolate colours
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 1. Semantic palette
# --------------------------------------------------------------------------

C = {
    # brand
    "primary":      "#6C4DF6",   # violet  – headers, main model curve
    "primary_dark": "#4B2FD1",
    "primary_soft": "#EDE8FF",
    "accent":       "#00C2A8",   # teal    – secondary series
    "accent_dark":  "#00907C",
    "accent_soft":  "#DBFAF4",

    # semantic
    "success":      "#06D6A0",
    "info":         "#118AB2",
    "warning":      "#FF9F1C",
    "danger":       "#EF476F",
    "proof":        "#8338EC",
    "pitfall":      "#E5383B",

    # data roles
    "train":        "#118AB2",
    "valid":        "#FF9F1C",
    "test":         "#EF476F",
    "truth":        "#073B4C",
    "pred":         "#6C4DF6",
    "residual":     "#EF476F",
    "gradient":     "#FF006E",

    # neutrals
    "ink":          "#0E1428",
    "ink_soft":     "#3A4160",
    "muted":        "#7A8199",
    "line":         "#D8DCE8",
    "surface":      "#FFFFFF",
    "surface_alt":  "#F6F7FB",
    "surface_deep": "#EDEFF7",
    "code_bg":      "#12172B",
}

# --------------------------------------------------------------------------
# 2. Categorical sequence for discrete series
# --------------------------------------------------------------------------

SEQ = [
    "#6C4DF6",  # violet
    "#00C2A8",  # teal
    "#FF9F1C",  # amber
    "#EF476F",  # rose
    "#118AB2",  # cerulean
    "#06D6A0",  # mint
    "#8338EC",  # purple
    "#FB5607",  # orange
    "#3A86FF",  # blue
    "#FFBE0B",  # yellow
    "#2EC4B6",  # turquoise
    "#D90429",  # red
]

# class colours used in every classification demo (keeps blue/orange/green
# meaning "class 0 / 1 / 2" across all 19 chapters)
CLASS_COLORS = ["#3A86FF", "#FB5607", "#06D6A0", "#8338EC", "#EF476F",
                "#FFBE0B", "#118AB2", "#FF006E", "#2EC4B6", "#7A8199"]


# --------------------------------------------------------------------------
# 3. Continuous colourscales
# --------------------------------------------------------------------------

# MATLAB R2014b "parula" – 9 anchor stops, linearly interpolated.
PARULA_STOPS = [
    "#352A87", "#0363E1", "#1485D4", "#06A7C6", "#38B99E",
    "#92BF73", "#D9BA56", "#FCCE2E", "#F9FB0E",
]

JET_STOPS = [
    "#00007F", "#0000FF", "#007FFF", "#00FFFF", "#7FFF7F",
    "#FFFF00", "#FF7F00", "#FF0000", "#7F0000",
]

TURBO_STOPS = [
    "#30123B", "#4145AB", "#4675ED", "#39A2FC", "#1BCFD4",
    "#24ECA6", "#61FC6C", "#A4FC3B", "#D1E834", "#F3C63A",
    "#FE9B2D", "#F36315", "#D93806", "#B11901", "#7A0403",
]

BLUERED_STOPS = ["#053061", "#4393C3", "#D1E5F0", "#F7F7F7",
                 "#FDDBC7", "#D6604D", "#67001F"]

# a warm sequential scale used for "energy / intensity" style surfaces
SINHA_STOPS = ["#0B0033", "#3C1053", "#7B2D6E", "#C0397A",
               "#EE6C4D", "#F9A03F", "#FFD166", "#FFF3B0"]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(round(r)), int(round(g)), int(round(b)))


def interpolate(c1: str, c2: str, t: float) -> str:
    """Linear interpolation between two hex colours (t in [0, 1])."""
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def ramp(stops: list[str], n: int) -> list[str]:
    """Expand a list of anchor stops into ``n`` evenly spaced hex colours."""
    if n <= 1:
        return [stops[0]]
    out = []
    segs = len(stops) - 1
    for i in range(n):
        x = i / (n - 1) * segs
        j = min(int(x), segs - 1)
        out.append(interpolate(stops[j], stops[j + 1], x - j))
    return out


def scale(stops: list[str], n: int = 32) -> list[list]:
    """Build a plotly colourscale ``[[value, hex], ...]`` from anchor stops."""
    cols = ramp(stops, n)
    return [[i / (n - 1), c] for i, c in enumerate(cols)]


# Ready-made plotly colourscales -------------------------------------------
PARULA = scale(PARULA_STOPS, 64)
JET = scale(JET_STOPS, 64)
TURBO = scale(TURBO_STOPS, 64)
BLUERED = scale(BLUERED_STOPS, 64)
SINHA = scale(SINHA_STOPS, 64)

COLORSCALES = {
    "Parula": PARULA,
    "Jet": JET,
    "Turbo": TURBO,
    "BlueRed": BLUERED,
    "Sinha": SINHA,
    "Viridis": "Viridis",
    "Plasma": "Plasma",
    "Cividis": "Cividis",
    "RdBu": "RdBu",
}

DEFAULT_COLORSCALE = "Parula"


def resolve_colorscale(name: str = DEFAULT_COLORSCALE):
    """Return a plotly-ready colourscale for a friendly name."""
    return COLORSCALES.get(name, PARULA)


def class_color(i: int) -> str:
    return CLASS_COLORS[i % len(CLASS_COLORS)]


def series_color(i: int) -> str:
    return SEQ[i % len(SEQ)]


def alpha(hex_color: str, a: float) -> str:
    """Return an ``rgba(...)`` string for a hex colour at opacity ``a``."""
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{a})"
