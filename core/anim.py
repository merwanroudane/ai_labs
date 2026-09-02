"""
core.anim
=========
The animation engine.

Every animated explanation in the platform is a Plotly figure carrying
``frames`` plus a ``▶ Play`` / ``⏸ Pause`` control bar and a scrub slider.

Public helpers
--------------
play_controls(...)      -> (updatemenus, sliders) ready to drop into a layout
animate(fig, frames...) -> attaches controls to an existing figure
frame_figure(...)       -> build a complete animated figure in one call

Design rules used everywhere:
  * frame durations are given in milliseconds
  * axis ranges are FIXED across frames (otherwise the animation "jumps")
  * every animation gets a slider so the learner can scrub manually
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go

from .palette import C

# --------------------------------------------------------------------------


def play_controls(
    frame_labels: Sequence[str],
    duration: int = 220,
    transition: int = 0,
    slider_prefix: str = "step: ",
    y_buttons: float = 1.10,
    x_buttons: float = 0.0,
    show_slider: bool = True,
    loop_label: str = "▶  Play",
):
    """Return ``(updatemenus, sliders)`` for a frame-based Plotly animation."""

    updatemenus = [dict(
        type="buttons",
        direction="left",
        showactive=False,
        x=x_buttons, xanchor="left",
        y=y_buttons, yanchor="top",
        pad=dict(t=0, r=6, b=0, l=0),
        bgcolor="#FFFFFF",
        bordercolor=C["line"],
        borderwidth=1,
        font=dict(size=12, color=C["primary_dark"], family="Inter, sans-serif"),
        buttons=[
            dict(label=loop_label,
                 method="animate",
                 args=[None, dict(frame=dict(duration=duration, redraw=True),
                                  transition=dict(duration=transition),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸  Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate")]),
            dict(label="⏮  Reset",
                 method="animate",
                 args=[[frame_labels[0]], dict(frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0),
                                               mode="immediate")]),
        ],
    )]

    sliders = []
    if show_slider:
        sliders = [dict(
            active=0,
            x=0.0, xanchor="left", y=-0.13, yanchor="top",
            len=1.0,
            pad=dict(t=32, b=8),
            currentvalue=dict(prefix=slider_prefix, visible=True,
                              xanchor="left", font=dict(size=12,
                                                        color=C["ink_soft"])),
            transition=dict(duration=transition),
            bgcolor="#E4E7F2",
            bordercolor=C["line"],
            borderwidth=1,
            tickcolor=C["muted"],
            font=dict(size=10, color=C["muted"]),
            steps=[dict(method="animate", label=lab,
                        args=[[lab], dict(frame=dict(duration=0, redraw=True),
                                          transition=dict(duration=0),
                                          mode="immediate")])
                   for lab in frame_labels],
        )]

    return updatemenus, sliders


def animate(fig: go.Figure,
            frames: list[go.Frame],
            duration: int = 220,
            transition: int = 0,
            slider_prefix: str = "step: ",
            show_slider: bool = True,
            y_buttons: float = 1.10) -> go.Figure:
    """Attach ``frames`` plus a play bar to ``fig`` and return it."""
    fig.frames = frames
    labels = [f.name for f in frames]
    um, sl = play_controls(labels, duration=duration, transition=transition,
                           slider_prefix=slider_prefix,
                           show_slider=show_slider, y_buttons=y_buttons)
    fig.update_layout(updatemenus=um, sliders=sl)
    # reserve a band at the top so the title never collides with the play bar,
    # and pin the title to the very top of the figure container
    fig.update_layout(margin=dict(t=104),
                      title=dict(yref="container", y=0.975, yanchor="top"))
    if show_slider:
        fig.update_layout(margin=dict(b=96))
    return fig


def frame_figure(base_traces: Iterable[go.BaseTraceType],
                 frames: list[go.Frame],
                 layout: dict[str, Any] | None = None,
                 duration: int = 220,
                 transition: int = 0,
                 slider_prefix: str = "step: ",
                 show_slider: bool = True,
                 height: int = 470) -> go.Figure:
    """One-shot constructor for an animated figure."""
    fig = go.Figure(data=list(base_traces))
    if layout:
        fig.update_layout(**layout)
    fig.update_layout(height=height)
    return animate(fig, frames, duration=duration, transition=transition,
                   slider_prefix=slider_prefix, show_slider=show_slider)


# --------------------------------------------------------------------------
# Convenience builders used repeatedly across chapters
# --------------------------------------------------------------------------


def trail_frames(xs: np.ndarray,
                 ys: np.ndarray,
                 name: str = "path",
                 color: str = C["gradient"],
                 marker_size: int = 11,
                 line_width: float = 2.6,
                 extra_per_frame=None) -> list[go.Frame]:
    """
    Frames that draw a growing polyline plus a leading marker.

    ``extra_per_frame(k)`` may return a list of *additional* traces appended
    after the trail traces for frame ``k``.
    """
    frames = []
    for k in range(1, len(xs) + 1):
        data = [
            go.Scatter(x=xs[:k], y=ys[:k], mode="lines",
                       line=dict(color=color, width=line_width),
                       name=name, showlegend=False),
            go.Scatter(x=[xs[k - 1]], y=[ys[k - 1]], mode="markers",
                       marker=dict(color=color, size=marker_size,
                                   line=dict(color="#FFFFFF", width=2)),
                       name="current", showlegend=False),
        ]
        if extra_per_frame is not None:
            data += list(extra_per_frame(k - 1))
        frames.append(go.Frame(data=data, name=str(k)))
    return frames


def reveal_frames(x: np.ndarray,
                  y: np.ndarray,
                  color: str = C["primary"],
                  width: float = 3.0,
                  name: str = "curve") -> list[go.Frame]:
    """Frames that progressively draw a curve from left to right."""
    n = len(x)
    steps = min(n, 60)
    idx = np.unique(np.linspace(2, n, steps).astype(int))
    return [go.Frame(data=[go.Scatter(x=x[:k], y=y[:k], mode="lines",
                                      line=dict(color=color, width=width),
                                      name=name)],
                     name=str(j + 1))
            for j, k in enumerate(idx)]


def fixed_axes(fig: go.Figure,
               x: Sequence[float],
               y: Sequence[float],
               pad: float = 0.06) -> go.Figure:
    """Freeze axis ranges so the animation does not jitter between frames."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = (x.max() - x.min()) or 1.0
    dy = (y.max() - y.min()) or 1.0
    fig.update_xaxes(range=[x.min() - pad * dx, x.max() + pad * dx])
    fig.update_yaxes(range=[y.min() - pad * dy, y.max() + pad * dy])
    return fig


def annotate_step(text: str,
                  x: float = 0.99, y: float = 0.02,
                  color: str = C["ink"],
                  bg: str = "rgba(255,255,255,.88)") -> dict:
    """A consistent little status box used inside frames."""
    return dict(x=x, y=y, xref="paper", yref="paper",
                xanchor="right", yanchor="bottom",
                text=text, showarrow=False,
                font=dict(size=12, color=color, family="JetBrains Mono, monospace"),
                bgcolor=bg, bordercolor=C["line"], borderwidth=1,
                borderpad=6)


def rotating_3d(fig: go.Figure,
                n_frames: int = 40,
                radius: float = 1.9,
                elevation: float = 0.9,
                duration: int = 90) -> go.Figure:
    """Add an orbit animation (play button) to a 3-D scene."""
    frames = []
    for k in range(n_frames):
        ang = 2 * np.pi * k / n_frames
        cam = dict(eye=dict(x=radius * np.cos(ang),
                            y=radius * np.sin(ang),
                            z=elevation))
        frames.append(go.Frame(layout=dict(scene_camera=cam), name=str(k + 1)))
    fig.frames = frames
    um, sl = play_controls([f.name for f in frames], duration=duration,
                           slider_prefix="angle ", y_buttons=1.06)
    fig.update_layout(updatemenus=um, sliders=sl)
    return fig
