"""Chapter 12 — Custom Models and Training with TensorFlow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import anim, datasets as ds, nav
from core.lecture import (anim_header, codenote, derive, exercise, figure, hero,
                          idea, keypoints, lead, math, md, note, pitfall, proof,
                          quiz, refs, rule, section, sub, table, tip, warn,
                          where)
from core.palette import C, SEQ, alpha
from core.runner import code_lab
from core.theme import inject

inject()
CH = "ch12"

hero(
    kicker="Part II · Chapter 12",
    title="Custom Models and Training with TensorFlow",
    blurb=(
        "Below the Keras API is a numerical computing library. This chapter opens "
        "it: tensors and variables, custom losses, metrics, layers and models, "
        "the <code>GradientTape</code> that makes autodiff explicit, hand-written "
        "training loops, and <code>tf.function</code> — the graph compiler whose "
        "rules you must know before you can trust it."
    ),
    chips=["The layer below Keras", "9 sub-sections", "8 animations",
           "10 code labs", "GradientTape & tf.function"],
)
nav.sidebar_tools(CH)


def _tf_ok() -> bool:
    # find_spec, not import: importing TensorFlow costs ~500 MB of RSS and we
    # only need to know whether the labs will be able to run.
    import importlib.util
    try:
        return importlib.util.find_spec("tensorflow") is not None
    except Exception:
        return False


if not _tf_ok():
    st.warning(
        "TensorFlow is not importable here, so the labs will report an "
        "ImportError when run. Every explanation and animation still works.",
        icon="⚠️")


# ==========================================================================
def s_12_1():
    section("12.1", "A Quick Tour of TensorFlow")

    lead(
        "TensorFlow is not a deep-learning library that happens to do numerics. "
        "It is a numerical computing library — like NumPy, but with GPU support, "
        "automatic differentiation and distributed execution — that happens to "
        "have deep learning built on top."
    )

    sub("What it actually provides")

    table(
        ["Capability", "How", "Where in this course"],
        [["<b>N-dimensional array maths</b>",
          "<code>tf.Tensor</code>, with a NumPy-compatible API",
          "§12.2"],
         ["<b>Automatic differentiation</b>", "<code>tf.GradientTape</code>",
          "§12.8, Appendix B"],
         ["<b>GPU / TPU acceleration</b>",
          "Operations dispatch to the fastest available device automatically",
          "§12.1, Ch. 19"],
         ["<b>Graph compilation</b>",
          "<code>tf.function</code> traces Python into an optimised graph",
          "§12.9"],
         ["<b>Distributed execution</b>",
          "<code>tf.distribute</code> strategies", "Ch. 19"],
         ["<b>Input pipelines</b>", "<code>tf.data</code>", "Ch. 13"],
         ["<b>Deployment</b>",
          "SavedModel → TF Serving, TF Lite, TensorFlow.js", "Ch. 19"],
         ["<b>High-level modelling</b>", "<code>tf.keras</code>",
          "Ch. 10, 11, 14–18"]],
    )

    sub("The API layers")

    md(
        """
Think of TensorFlow as a stack, and know which level you are working at:

| Level | API | Use when |
|---|---|---|
| **High** | `tf.keras` — `Sequential`, `Model`, `fit` | Almost always |
| **Mid** | Custom layers, losses, metrics, models | The architecture is unusual |
| **Low** | `GradientTape`, custom training loops | The *training procedure* is unusual (GANs, RL, meta-learning) |
| **Kernel** | C++ ops, `tf.raw_ops`, custom CUDA | Almost never |

Each level is written in terms of the one below, so you can drop down exactly as
far as you need and no further.
        """
    )

    tip(
        "Do not drop down without a reason",
        "A custom training loop is roughly 30 lines that <code>model.fit</code> "
        "gives you for free — plus, quietly, callbacks, distribution strategies, "
        "metric aggregation, progress bars, and correct handling of "
        "<code>training=True/False</code> for batch norm and dropout. Every one "
        "of those is a bug you now own. Write the custom loop when the training "
        "<i>procedure</i> genuinely differs, not to feel closer to the metal.",
    )

    sub("Eager execution vs graph mode")

    table(
        ["", "Eager (the default)", "Graph (<code>@tf.function</code>)"],
        [["When ops run", "Immediately, line by line", "After the whole function "
          "is traced into a graph"],
         ["Debugging", "<b>Easy</b> — <code>print</code>, breakpoints, "
          "<code>.numpy()</code> all work",
          "Hard — Python <code>print</code> runs only during tracing"],
         ["Speed", "Baseline", "<b>Much faster</b> — fused ops, dead code "
          "elimination, constant folding, parallel scheduling"],
         ["Portability", "Python only", "Serialisable; runs in TF Serving, "
          "TF Lite, TF.js with no Python"],
         ["Use", "Development and debugging",
          "Production and any hot loop"]],
    )

    idea(
        "Develop eagerly, deploy as a graph",
        "The intended workflow: write and debug in eager mode where everything "
        "behaves like NumPy, then add <code>@tf.function</code> once it works. "
        "Keras does this for you — <code>model.fit</code> compiles the training "
        "step automatically, which is why it is faster than a naive hand-written "
        "loop. When something breaks inside a "
        "<code>tf.function</code>, set <code>tf.config.run_functions_eagerly("
        "True)</code> to turn tracing off and debug normally.",
    )

    anim_header("The TensorFlow stack, from Python down to the hardware")

    layers_stack = [
        ("Your Python code", C["accent"],
         "model.fit(...) · custom training loops"),
        ("tf.keras", SEQ[0], "Model · Layer · Loss · Metric · Optimizer"),
        ("TensorFlow Python API", SEQ[1],
         "tf.Tensor · tf.Variable · tf.GradientTape · tf.function"),
        ("Execution engine", SEQ[2],
         "graph optimisation (XLA) · kernel dispatch · memory"),
        ("Kernels (C++)", SEQ[3], "one implementation per op, per device"),
        ("Hardware", C["ink"], "CPU · GPU (CUDA) · TPU"),
    ]
    frames = []
    for k in range(1, len(layers_stack) + 1):
        shapes, texts = [], []
        for i, (nm, col, detail) in enumerate(layers_stack):
            active = i < k
            y = len(layers_stack) - i
            shapes.append(go.Scatter(
                x=[0, 6, 6, 0, 0], y=[y - .42, y - .42, y + .42, y + .42, y - .42],
                fill="toself",
                fillcolor=alpha(col, .8) if active else alpha(C["line"], .35),
                line=dict(color="#fff", width=2), hoverinfo="skip",
                showlegend=False))
            texts.append((3, y, f"<b>{nm}</b>" if active else nm,
                          detail if active else ""))
        ann = [dict(x=t[0], y=t[1], text=t[2], showarrow=False,
                    font=dict(size=12, color="#fff")) for t in texts]
        ann += [dict(x=6.15, y=t[1], text=t[3], showarrow=False, xanchor="left",
                     font=dict(size=9.5, color=C["ink_soft"])) for t in texts]
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann,
                                                title=f"level {k}: "
                                                      f"{layers_stack[k-1][0]}")))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=460, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.3, 13]),
                    yaxis=dict(visible=False, range=[.2, 6.9]),
                    annotations=list(frames[0].layout.annotations),
                    title=f"level 1: {layers_stack[0][0]}")
    anim.animate(f, frames, duration=nav.anim_ms(1100), slider_prefix="level ")
    figure(f)

    code_lab(
        "What TensorFlow sees: devices, eager mode, and the graph speed-up",
        '''import numpy as np, time
import tensorflow as tf

print(f"TensorFlow version : {tf.__version__}")
print(f"Keras version      : {tf.keras.__version__}")
print(f"eager execution on : {tf.executing_eagerly()}")

# ============ 1. WHAT HARDWARE IS AVAILABLE ============================
print("\\n=== devices ===")
for dev in tf.config.list_physical_devices():
    print(f"  {dev.device_type:<6} {dev.name}")
gpus = tf.config.list_physical_devices("GPU")
print(f"\\nGPUs available: {len(gpus)}")
if not gpus:
    print("  (running on CPU -- everything below still works, just slower)")

# ============ 2. EAGER EXECUTION FEELS LIKE NUMPY ======================
print("\\n=== eager execution ===")
a = tf.constant([[1., 2., 3.], [4., 5., 6.]])
b = tf.constant([[10.], [20.]])
print(f"a          = \\n{a.numpy()}")
print(f"a + b      = \\n{(a + b).numpy()}        <- broadcasting, as in NumPy")
print(f"a @ a.T    = \\n{(a @ tf.transpose(a)).numpy()}")
print(f"tf.reduce_sum(a, axis=0) = {tf.reduce_sum(a, axis=0).numpy()}")
print(f"the result of every line is available IMMEDIATELY -- that is eager mode")

# ============ 3. GRAPH MODE IS MUCH FASTER =============================
print("\\n=== eager vs @tf.function ===")

def dense_block(x, W1, b1, W2, b2):
    h = tf.nn.relu(x @ W1 + b1)
    h = tf.nn.relu(h @ W2 + b2)
    return tf.reduce_mean(h)

graph_block = tf.function(dense_block)

rng = tf.random.Generator.from_seed(0)
x  = rng.normal((256, 200))
W1 = rng.normal((200, 300)); b1 = tf.zeros(300)
W2 = rng.normal((300, 200)); b2 = tf.zeros(200)

graph_block(x, W1, b1, W2, b2)          # warm-up: this call TRACES the function

N = 400
t0 = time.perf_counter()
for _ in range(N): dense_block(x, W1, b1, W2, b2)
t_eager = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N): graph_block(x, W1, b1, W2, b2)
t_graph = time.perf_counter() - t0

print(f"eager  : {t_eager*1000/N:.3f} ms per call")
print(f"graph  : {t_graph*1000/N:.3f} ms per call")
print(f"speedup: {t_eager/t_graph:.2f}x")
print("\\nThe gain grows with the number of small ops -- the graph fuses them,")
print("removes dead code, folds constants and schedules in parallel.")

# ============ 4. WHERE DO OPS RUN? =====================================
print("\\n=== device placement ===")
with tf.device("/CPU:0"):
    c = tf.constant([1., 2., 3.])
    print(f"explicitly on CPU : {c.device}")
d = tf.constant([1., 2., 3.]) * 2
print(f"automatic         : {d.device}")
print("TensorFlow places each op on the fastest available device by default.")

# ============ 5. DEBUGGING INSIDE A tf.function ========================
print("\\n=== debugging graph mode ===")
@tf.function
def tracked(x):
    print("  PYTHON print -- runs only while TRACING")     # traced once
    tf.print("  tf.print   -- runs on EVERY call")          # runs every time
    return x * 2

print("first call (traces):")
tracked(tf.constant(1.))
print("second call (reuses the graph):")
tracked(tf.constant(2.))
print("third call, different dtype (re-traces):")
tracked(tf.constant(3))

print("\\nturn tracing off entirely to debug:")
tf.config.run_functions_eagerly(True)
tracked(tf.constant(4.))
tf.config.run_functions_eagerly(False)      # remember to turn it back on
''',
        key="ch12_tour",
    )

    keypoints([
        "TensorFlow is a numerical computing library with autodiff, GPU support "
        "and graph compilation; Keras sits on top.",
        "Four API levels — use the highest one that does the job.",
        "<b>Eager</b> mode for development, <b>graph</b> mode "
        "(<code>@tf.function</code>) for speed and deployment.",
        "<code>model.fit</code> already compiles its training step — a naive "
        "custom loop is <i>slower</i> unless you add <code>@tf.function</code>.",
        "<code>tf.config.run_functions_eagerly(True)</code> is the debugging "
        "escape hatch.",
    ])


# ==========================================================================
def s_12_2():
    section("12.2", "Using TensorFlow Like NumPy")

    lead(
        "Tensors behave like NumPy arrays with three differences that will each "
        "bite you exactly once: they are immutable, they are strict about types, "
        "and some operation names differ."
    )

    sub("Tensors and operations")

    table(
        ["NumPy", "TensorFlow", "Note"],
        [["<code>np.array([1,2])</code>", "<code>tf.constant([1,2])</code>", ""],
         ["<code>a.shape</code>", "<code>a.shape</code>", "Same"],
         ["<code>a.dtype</code>", "<code>a.dtype</code>", "Same"],
         ["<code>a + b</code>, <code>a * b</code>, <code>a @ b</code>",
          "identical", "Operator overloading works"],
         ["<code>a.sum(axis=0)</code>", "<code>tf.reduce_sum(a, axis=0)</code>",
          "<b>reduce_</b> prefix"],
         ["<code>a.mean()</code>", "<code>tf.reduce_mean(a)</code>", "Likewise"],
         ["<code>a.max()</code>", "<code>tf.reduce_max(a)</code>", "Likewise"],
         ["<code>a.T</code>", "<code>tf.transpose(a)</code>",
          "<b>Not a view</b> — it copies"],
         ["<code>np.log</code>, <code>np.exp</code>",
          "<code>tf.math.log</code>, <code>tf.exp</code>", "Mostly under "
          "<code>tf.math</code>"],
         ["<code>a[0] = 5</code>", "<b>impossible</b>",
          "Tensors are <b>immutable</b>"]],
    )

    sub("The float64 trap")

    pitfall(
        "TensorFlow does not convert types automatically — ever",
        "<code>tf.constant(2.) + tf.constant(40)</code> raises "
        "<code>InvalidArgumentError</code>, and so does mixing "
        "<code>float32</code> with <code>float64</code>. This looks pedantic but "
        "it is deliberate: automatic conversion silently costs performance, and "
        "<code>float64</code> is <b>2× slower and 2× larger</b> than "
        "<code>float32</code> on a GPU (and up to 32× slower on hardware with "
        "limited FP64 units). NumPy defaults to <code>float64</code>, so "
        "<code>tf.constant(np.array([1., 2.]))</code> gives you a float64 tensor "
        "that will not mix with your float32 model. Use "
        "<code>dtype=tf.float32</code> or <code>.astype(np.float32)</code> at the "
        "boundary.",
    )

    md("Convert explicitly with `tf.cast`:")

    md(
        """
```python
t = tf.constant([1., 2.], dtype=tf.float64)
tf.constant(2.0) + tf.cast(t, tf.float32)      # works
```
        """
    )

    sub("Variables — the mutable ones")

    md(
        "`tf.Tensor` is immutable, which is fine for data but useless for weights. "
        "`tf.Variable` is the mutable container, and it is what every trainable "
        "parameter is:"
    )

    table(
        ["Operation", "Meaning"],
        [["<code>v.assign(x)</code>", "Replace the whole value"],
         ["<code>v.assign_add(x)</code> / <code>assign_sub</code>",
          "In-place $\\pm$"],
         ["<code>v[0].assign(5.)</code>", "Assign to a slice"],
         ["<code>v.scatter_nd_update(idx, vals)</code>",
          "Sparse update at specific indices"],
         ["<code>v.trainable</code>",
          "Whether the optimiser should update it (batch norm's moving "
          "statistics are <code>trainable=False</code>)"]],
    )

    warn(
        "<code>v = v + 1</code> destroys the variable",
        "It creates a new <b>Tensor</b> and rebinds the Python name to it — the "
        "<code>Variable</code> is gone, along with its identity, its "
        "<code>trainable</code> flag, and the optimiser's slot state for it. "
        "Always use <code>v.assign_add(1)</code>. This is the single most common "
        "bug in hand-written training loops.",
    )

    sub("The other data structures")

    table(
        ["Structure", "What it is", "Used for"],
        [["<code>tf.SparseTensor</code>",
          "Indices, values and a dense shape",
          "Mostly-zero data — one-hot encodings, embeddings lookups"],
         ["<code>tf.RaggedTensor</code>",
          "Rows of <b>different lengths</b>",
          "Variable-length sequences without padding (Ch. 15–16)"],
         ["<code>tf.TensorArray</code>",
          "A dynamically-sized list of tensors",
          "Accumulating results inside a loop in graph mode"],
         ["<code>tf.string</code>",
          "Byte strings as a first-class dtype",
          "Text preprocessing (Ch. 13, 16)"],
         ["<code>tf.sets</code>, <code>tf.queue</code>",
          "Set operations and thread-safe queues",
          "Specialised pipelines"]],
    )

    anim_header("Broadcasting: how shapes are aligned")
    md(
        "The rule, applied right to left: two dimensions are compatible if they "
        "are equal, or if one of them is 1. A missing leading dimension counts as "
        "1. Each frame shows a pair of shapes being aligned."
    )

    cases = [
        ((3, 4), (3, 4), "identical shapes"),
        ((3, 4), (4,), "the (4,) is treated as (1,4) and stretched down"),
        ((3, 4), (3, 1), "the column is stretched across"),
        ((3, 1), (1, 4), "both stretch — the outer product shape"),
        ((2, 3, 4), (4,), "stretched across both leading axes"),
        ((3, 4), (3,), "INCOMPATIBLE: 4 ≠ 3 and neither is 1"),
    ]
    frames = []
    for sa, sb, note_ in cases:
        ok = "INCOMPATIBLE" not in note_
        res = tuple(max(x, y) for x, y in
                    zip((1,) * (3 - len(sa)) + sa, (1,) * (3 - len(sb)) + sb))
        def grid(shape, x0, colr):
            h = shape[-2] if len(shape) >= 2 else 1
            w = shape[-1]
            xs, ys = [], []
            for i in range(h):
                for j in range(w):
                    xs.append(x0 + j * .5); ys.append(-i * .5)
            return go.Scatter(x=xs, y=ys, mode="markers",
                              marker=dict(size=17, color=colr, symbol="square",
                                          line=dict(color="#fff", width=1.5)),
                              showlegend=False, hoverinfo="skip")
        data = [grid(sa, 0, C["train"]), grid(sb, 4, C["warning"])]
        if ok:
            data.append(grid(res[-2:], 8, C["success"]))
        frames.append(go.Frame(name=f"{sa}·{sb}", data=data,
                               layout=go.Layout(
                                   title=f"{sa} ⊕ {sb} → "
                                         f"{res[-len(max(sa, sb, key=len)):] if ok else 'ERROR'}"
                                         f"   —   {note_}")))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=380, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.6, 11]),
                    yaxis=dict(visible=False, range=[-2, .7]),
                    title=frames[0].layout.title.text)
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="case ")
    figure(f, "Blue ⊕ orange → green. The last case fails because 4 and 3 are "
              "both ≠ 1 and unequal.")

    code_lab(
        "Tensors, dtypes, variables, and the structures beyond dense arrays",
        '''import numpy as np, time
import tensorflow as tf

# ============ 1. TENSORS BEHAVE LIKE ARRAYS ============================
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print("=== basics ===")
print(f"tensor      : shape {t.shape}  dtype {t.dtype}")
print(f"t[:, 1:]    = \\n{t[:, 1:].numpy()}")
print(f"t[..., 1]   = {t[..., 1].numpy()}          (Ellipsis works)")
print(f"t + 10      = \\n{(t + 10).numpy()}")
print(f"tf.square   = \\n{tf.square(t).numpy()}")
print(f"t @ t.T     = \\n{(t @ tf.transpose(t)).numpy()}")

print("\\n=== reductions carry a 'reduce_' prefix ===")
for np_name, tf_fn in [("sum",  tf.reduce_sum), ("mean", tf.reduce_mean),
                       ("max",  tf.reduce_max), ("min",  tf.reduce_min)]:
    print(f"  np.{np_name}(a)  ->  tf.reduce_{np_name}(a)  = "
          f"{tf_fn(t).numpy():.4f}")

# ============ 2. TENSORS ARE IMMUTABLE =================================
print("\\n=== immutability ===")
try:
    t[0, 0] = 99.
except TypeError as e:
    print(f"  t[0,0] = 99  ->  TypeError: {str(e)[:60]}...")
print("  every operation returns a NEW tensor")

# ============ 3. THE TYPE STRICTNESS ===================================
print("\\n=== TensorFlow never converts types for you ===")
for expr, fn in [("tf.constant(2.) + tf.constant(40)",
                  lambda: tf.constant(2.) + tf.constant(40)),
                 ("float32 + float64",
                  lambda: tf.constant(2.) + tf.constant(40., dtype=tf.float64)),
                 ("tf.constant(2.) + tf.cast(tf.constant(40), tf.float32)",
                  lambda: tf.constant(2.) + tf.cast(tf.constant(40), tf.float32))]:
    try:
        print(f"  {expr:<58} = {fn().numpy()}")
    except Exception as e:
        print(f"  {expr:<58} -> {type(e).__name__}")

print("\\n=== the NumPy float64 trap ===")
a = np.array([1., 2., 3.])
print(f"  np.array([1.,2.,3.]).dtype     = {a.dtype}")
print(f"  tf.constant(a).dtype           = {tf.constant(a).dtype}   <- float64!")
print(f"  tf.constant([1.,2.,3.]).dtype  = {tf.constant([1.,2.,3.]).dtype}")
print(f"  fix: tf.constant(a, dtype=tf.float32) or a.astype(np.float32)")

# --- and float64 really is slower ------------------------------------
print("\\n  float32 vs float64 matmul (2000x2000):")
for dt in (tf.float32, tf.float64):
    x = tf.random.normal((2000, 2000), dtype=dt)
    _ = x @ x
    t0 = time.perf_counter()
    for _ in range(5): _ = x @ x
    print(f"    {str(dt.name):<8} {(time.perf_counter()-t0)/5*1000:>8.1f} ms   "
          f"{x.numpy().nbytes/1e6:>6.1f} MB")

# ============ 4. VARIABLES =============================================
print("\\n=== tf.Variable is the mutable one ===")
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(f"  initial            : {v.numpy()[0]}")
v.assign(2 * v);                    print(f"  after assign(2*v)  : {v.numpy()[0]}")
v[0, 1].assign(42.);                print(f"  after v[0,1]=42    : {v.numpy()[0]}")
v[:, 2].assign([0., 1.]);           print(f"  after v[:,2]=[0,1] : {v.numpy()[0]}")
v.assign_add(tf.ones_like(v));      print(f"  after assign_add(1): {v.numpy()[0]}")
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])
print(f"  after scatter      : \\n{v.numpy()}")

print("\\n  THE CLASSIC BUG:")
w = tf.Variable([1., 2.])
print(f"    before  : type={type(w).__name__}, trainable={w.trainable}")
w = w + 1                                       # <- destroys the Variable
print(f"    w = w+1 : type={type(w).__name__}, has .trainable? "
      f"{hasattr(w, 'trainable')}")
w2 = tf.Variable([1., 2.]); w2.assign_add([1., 1.])
print(f"    assign_add: type={type(w2).__name__}, trainable={w2.trainable}  <- correct")

# ============ 5. THE OTHER STRUCTURES ==================================
print("\\n=== sparse tensors ===")
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]], values=[1., 2., 3.],
                    dense_shape=[3, 4])
print(f"  3 non-zeros in a 3x4 tensor")
print(tf.sparse.to_dense(s).numpy())
print(f"  dense storage : {3*4} floats;  sparse storage : "
      f"{3} values + {3*2} indices")

print("\\n=== ragged tensors: rows of different lengths ===")
r = tf.ragged.constant([[1, 2, 3], [4], [], [5, 6]])
print(f"  {r}")
print(f"  row lengths      : {r.row_lengths().numpy()}")
print(f"  to_tensor() pads : \\n{r.to_tensor().numpy()}")
print(f"  concatenated     : {tf.concat([r, tf.ragged.constant([[7],[8],[9],[10]])], axis=1)}")

print("\\n=== strings are a real dtype ===")
txt = tf.constant(["café", "coffee", "咖啡"])
print(f"  {txt.numpy()}")
print(f"  byte lengths     : {tf.strings.length(txt).numpy()}")
print(f"  unicode lengths  : {tf.strings.length(txt, unit='UTF8_CHAR').numpy()}")
print(f"  upper            : {tf.strings.upper(txt).numpy()}")
print(f"  split            : {tf.strings.unicode_split(txt[0], 'UTF-8')}")

print("\\n=== TensorArray: a growable list inside a graph ===")
# NOTE: these labs are executed from a string, so AutoGraph cannot read their
# source (rule 6 of section 12.9!). We therefore write the loop explicitly
# with tf.while_loop instead of letting AutoGraph convert `for i in tf.range`.
@tf.function
def cumulative(n):
    ta = tf.TensorArray(tf.float32, size=n)
    def body(i, total, ta):
        total = total + tf.cast(i, tf.float32)
        return i + 1, total, ta.write(i, total)
    _, _, ta = tf.while_loop(lambda i, t, a: i < n, body,
                             [tf.constant(0), tf.constant(0.), ta])
    return ta.stack()
print(f"  cumulative(6) = {cumulative(tf.constant(6)).numpy()}")

# ============ 6. BROADCASTING ==========================================
print("\\n=== broadcasting ===")
print(f"{'a.shape':>12}{'b.shape':>12}{'result':>14}")
for sa, sb in [((3,4),(3,4)), ((3,4),(4,)), ((3,4),(3,1)), ((3,1),(1,4)),
               ((2,3,4),(4,)), ((3,4),(3,))]:
    try:
        res = (tf.ones(sa) + tf.ones(sb)).shape
        print(f"{str(sa):>12}{str(sb):>12}{str(tuple(res)):>14}")
    except Exception:
        print(f"{str(sa):>12}{str(sb):>12}{'ERROR':>14}")
''',
        key="ch12_numpy",
    )

    keypoints([
        "Tensors are NumPy arrays with GPU support and autodiff — but "
        "<b>immutable</b>.",
        "Reductions use the <code>reduce_</code> prefix; most maths lives under "
        "<code>tf.math</code>.",
        "<b>No automatic type conversion.</b> Watch for NumPy's float64 default "
        "at the boundary.",
        "<code>tf.Variable</code> is the mutable container; use "
        "<code>assign*</code>, never <code>v = v + 1</code>.",
        "Sparse, ragged, string and TensorArray cover what dense float tensors "
        "cannot.",
    ])


# ==========================================================================
def s_12_3():
    section("12.3", "Custom Loss Functions and Saving Them")

    lead(
        "A loss is any function of $(y_{\\text{true}}, y_{\\text{pred}})$ that "
        "returns a tensor. The interesting part is what happens when you try to "
        "save the model."
    )

    sub("The simple form")

    md(
        "Huber loss (§10.3) is not a Keras built-in in every version, and it is "
        "the standard example:"
    )

    math(r"""
    L_\delta(a) =
    \begin{cases}
      \tfrac12 a^2 & |a| \le \delta\\
      \delta\bigl(|a| - \tfrac12\delta\bigr) & \text{otherwise}
    \end{cases}
    """)

    md(
        """
```python
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small = tf.abs(error) < 1
    squared = tf.square(error) / 2
    linear = tf.abs(error) - 0.5
    return tf.where(is_small, squared, linear)

model.compile(loss=huber_fn, optimizer="nadam")
```
        """
    )

    warn(
        "Use <code>tf.where</code>, not a Python <code>if</code>",
        "The loss receives a whole <b>batch</b> of errors, so the condition is a "
        "<i>tensor</i> of booleans, not a single boolean. "
        "<code>if tf.abs(error) &lt; 1:</code> raises "
        "<code>OperatorNotAllowedInGraphError</code> inside a "
        "<code>tf.function</code>. <code>tf.where(cond, a, b)</code> selects "
        "element-wise. This applies to every custom component in this chapter.",
    )

    sub("Saving and loading — the three cases")

    table(
        ["Your loss is…", "To reload", "Why"],
        [["A <b>plain function</b>",
          "<code>load_model(path, custom_objects={'huber_fn': huber_fn})</code>",
          "The name is saved, the code is not"],
         ["A <b>function returning a function</b> (a closure over a "
          "hyperparameter)",
          "Same, but <b>the hyperparameter is lost</b> — it reverts to the "
          "default",
          "A closure's captured values are not serialisable"],
         ["A <b><code>Loss</code> subclass</b> with "
          "<code>get_config()</code>",
          "<code>custom_objects={'HuberLoss': HuberLoss}</code> — "
          "<b>and the hyperparameter is restored</b>",
          "<code>get_config</code> writes it into the file"],
         ["Registered with "
          "<code>@keras.utils.register_keras_serializable()</code>",
          "Nothing — <code>load_model(path)</code> just works",
          "The class is in a global registry"]],
    )

    md(
        """
```python
@keras.utils.register_keras_serializable()
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small = tf.abs(error) < self.threshold
        squared = tf.square(error) / 2
        linear = self.threshold * (tf.abs(error) - self.threshold / 2)
        return tf.where(is_small, squared, linear)

    def get_config(self):
        base = super().get_config()
        return {**base, "threshold": self.threshold}
```
        """
    )

    idea(
        "get_config is the whole serialisation contract",
        "Keras saves <code>get_config()</code>'s dictionary into the file and "
        "later calls <code>YourClass(**config)</code> to rebuild the object. So "
        "the rule is simple: <b>every constructor argument that affects behaviour "
        "must appear in <code>get_config</code></b>, and its value must be JSON-"
        "serialisable. If you store a tensor or a lambda, serialisation will fail "
        "— store the number or the name instead.",
    )

    anim_header("Threshold δ reshaping the Huber loss and its gradient")

    a = np.linspace(-3.5, 3.5, 500)
    frames = []
    for dlt in np.linspace(.15, 2.6, 32):
        L = np.where(np.abs(a) <= dlt, .5 * a ** 2,
                     dlt * (np.abs(a) - .5 * dlt))
        G = np.where(np.abs(a) <= dlt, a, dlt * np.sign(a))
        frames.append(go.Frame(name=f"{dlt:.2f}", data=[
            go.Scatter(x=a, y=L, mode="lines",
                       line=dict(color=C["primary"], width=3.6)),
            go.Scatter(x=a, y=G, mode="lines",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=[-dlt, -dlt], y=[-3, 7], mode="lines",
                       line=dict(color=C["muted"], width=1.5, dash="dot")),
            go.Scatter(x=[dlt, dlt], y=[-3, 7], mode="lines",
                       line=dict(color=C["muted"], width=1.5, dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"δ = {dlt:.2f}   ·   gradient saturates at ±{dlt:.2f}   ·   "
            f"loss at |a|=3 is {dlt*(3-.5*dlt):.3f} instead of {4.5:.3f}")])))

    f = go.Figure(data=[
        go.Scatter(x=a, y=np.where(np.abs(a) <= .15, .5 * a ** 2,
                                   .15 * (np.abs(a) - .075)),
                   mode="lines", name="Huber loss",
                   line=dict(color=C["primary"], width=3.6)),
        go.Scatter(x=a, y=np.where(np.abs(a) <= .15, a, .15 * np.sign(a)),
                   mode="lines", name="its gradient",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=[-.15, -.15], y=[-3, 7], mode="lines", showlegend=False,
                   line=dict(color=C["muted"], width=1.5, dash="dot")),
        go.Scatter(x=[.15, .15], y=[-3, 7], mode="lines", showlegend=False,
                   line=dict(color=C["muted"], width=1.5, dash="dot")),
    ])
    f.update_layout(height=430, xaxis_title="error a = y − ŷ",
                    yaxis=dict(range=[-3, 7]),
                    title="Huber: the gradient is BOUNDED by δ",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(140), slider_prefix="δ = ")
    figure(f, "The red gradient curve flattens at ±δ — that bound is exactly why "
              "Huber is robust: no single outlier can produce an unbounded step.")

    code_lab(
        "Custom losses four ways, and what survives a save/load round trip",
        '''import numpy as np, os, tempfile, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)
rng = np.random.default_rng(0)
X = rng.normal(0, 1, (2000, 8)).astype("float32")
y = (X @ rng.normal(0, 1, 8) + rng.normal(0, .3, 2000)).astype("float32")
y[rng.choice(2000, 40, replace=False)] += rng.normal(0, 20, 40)     # outliers
Xtr, Xte, ytr, yte = X[:1500], X[1500:], y[:1500], y[1500:]
tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch12_"))

def build(loss):
    m = keras.Sequential([keras.layers.Input(shape=(8,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
    m.compile(loss=loss, optimizer=keras.optimizers.Adam(1e-2),
              metrics=["mae"])
    return m

# ============ 1. A PLAIN FUNCTION ======================================
def huber_fn(y_true, y_pred):
    error   = y_true - y_pred
    is_small = tf.abs(error) < 1.0
    squared = tf.square(error) / 2
    linear  = tf.abs(error) - 0.5
    return tf.where(is_small, squared, linear)    # NOT a Python if

m1 = build(huber_fn)
m1.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)
print(f"=== plain function ===")
print(f"test MAE = {m1.evaluate(Xte, yte, verbose=0)[1]:.4f}")
m1.save(tmp/"m1.keras")
r1 = keras.models.load_model(tmp/"m1.keras",
                             custom_objects={"huber_fn": huber_fn})
print(f"reloaded (with custom_objects): MAE "
      f"{r1.evaluate(Xte, yte, verbose=0)[1]:.4f}")
try:
    keras.models.load_model(tmp/"m1.keras")
except Exception as e:
    print(f"reloaded WITHOUT custom_objects -> {type(e).__name__}")

# ============ 2. A CLOSURE (the hyperparameter is LOST) ================
def create_huber(threshold=1.0):
    def huber(y_true, y_pred):
        error = y_true - y_pred
        is_small = tf.abs(error) < threshold
        return tf.where(is_small, tf.square(error)/2,
                        threshold*(tf.abs(error) - threshold/2))
    return huber

m2 = build(create_huber(2.0))
m2.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)
m2.save(tmp/"m2.keras")
r2 = keras.models.load_model(tmp/"m2.keras",
                             custom_objects={"huber": create_huber(1.0)})
print(f"\\n=== closure ===")
print(f"trained with threshold=2.0, reloaded with the DEFAULT 1.0")
print(f"the threshold was NOT saved -- a closure's captured values are lost")

# ============ 3. A Loss SUBCLASS with get_config =======================
@keras.utils.register_keras_serializable(package="MLPlatform")
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small = tf.abs(error) < self.threshold
        squared  = tf.square(error) / 2
        linear   = self.threshold * (tf.abs(error) - self.threshold/2)
        return tf.where(is_small, squared, linear)

    def get_config(self):                      # THE serialisation contract
        return {**super().get_config(), "threshold": self.threshold}

m3 = build(HuberLoss(threshold=2.0))
m3.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)
m3.save(tmp/"m3.keras")
r3 = keras.models.load_model(tmp/"m3.keras")     # registered -> no custom_objects
print(f"\\n=== Loss subclass, registered ===")
print(f"loaded with NO custom_objects argument")
print(f"threshold restored: {r3.loss.threshold}   <- get_config did its job")
print(f"config saved: {HuberLoss(2.0).get_config()}")
print(f"test MAE = {r3.evaluate(Xte, yte, verbose=0)[1]:.4f}")

# ============ 4. DOES IT ACTUALLY HELP? ================================
print(f"\\n=== robustness on data with 2 % outliers ===")
print(f"{'loss':<28}{'test MAE':>11}{'test MSE':>12}")
for nm, loss in [("mse",                 "mse"),
                 ("mae",                 "mae"),
                 ("HuberLoss(0.5)",      HuberLoss(0.5)),
                 ("HuberLoss(1.0)",      HuberLoss(1.0)),
                 ("HuberLoss(5.0)",      HuberLoss(5.0)),
                 ("keras.losses.Huber()", keras.losses.Huber())]:
    tf.random.set_seed(0)
    m = build(loss)
    m.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)
    p = m.predict(Xte, verbose=0).ravel()
    print(f"{nm:<28}{np.abs(p-yte).mean():>11.4f}{np.mean((p-yte)**2):>12.4f}")

# ============ 5. WHY tf.where AND NOT if ===============================
print("\\n=== the Python-if trap ===")
@tf.function
def bad_huber(y_true, y_pred):
    error = y_true - y_pred
    if tf.abs(error) < 1.0:                     # a TENSOR, not a bool
        return tf.square(error)/2
    return tf.abs(error) - .5
try:
    bad_huber(tf.constant([1., 2.]), tf.constant([1.5, 5.]))
except Exception as e:
    print(f"  Python if inside @tf.function -> {type(e).__name__}")
    print(f"  {str(e).splitlines()[0][:90]}")
print(f"  tf.where works element-wise: "
      f"{huber_fn(tf.constant([1., 2.]), tf.constant([1.5, 5.])).numpy().round(4)}")

shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch12_loss",
    )

    keypoints([
        "A loss is any function of $(y_{\\text{true}}, y_{\\text{pred}})$ "
        "returning a tensor.",
        "Use <b><code>tf.where</code></b>, never a Python <code>if</code> — the "
        "condition is a batch of booleans.",
        "Plain functions need <code>custom_objects</code> on load; closures "
        "<b>lose their hyperparameters</b>.",
        "Subclass <code>keras.losses.Loss</code> and implement "
        "<b><code>get_config</code></b> to save hyperparameters.",
        "<code>@keras.utils.register_keras_serializable()</code> removes the need "
        "for <code>custom_objects</code> entirely.",
    ])


# ==========================================================================
def s_12_4():
    section("12.4", "Custom Activations, Initializers, Regularizers, Constraints")

    lead(
        "All four follow the same pattern: a plain function for the simple case, "
        "a subclass with <code>get_config</code> when there is a hyperparameter "
        "to preserve."
    )

    table(
        ["Component", "Signature", "Base class", "Applied"],
        [["<b>Activation</b>", "<code>f(z) → tensor</code>",
          "<code>keras.layers.Layer</code>", "To the layer's output"],
         ["<b>Initializer</b>", "<code>f(shape, dtype) → tensor</code>",
          "<code>keras.initializers.Initializer</code>", "Once, at build time"],
         ["<b>Regularizer</b>", "<code>f(weights) → scalar</code>",
          "<code>keras.regularizers.Regularizer</code>",
          "Added to the loss every step"],
         ["<b>Constraint</b>", "<code>f(weights) → tensor</code>",
          "<code>keras.constraints.Constraint</code>",
          "<b>After</b> every optimiser step (a projection)"]],
    )

    md(
        """
```python
def my_softplus(z):                      # activation
    return tf.math.log(1.0 + tf.exp(z))

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):          # returns a SCALAR
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights):        # a projection
    return tf.where(weights < 0., tf.zeros_like(weights), weights)

layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)
```
        """
    )

    note(
        "A regulariser returns a <b>scalar</b>; a constraint returns a "
        "<b>tensor</b>",
        "The regulariser's output is <i>added to the loss</i>, so it must reduce "
        "to a single number. The constraint's output <i>replaces the weights</i>, "
        "so it must have the same shape. Getting these backwards produces a "
        "confusing shape error rather than an obvious one.",
    )

    sub("Numerical stability — a real example")

    pitfall(
        "The naive softplus overflows",
        "<code>tf.math.log(1.0 + tf.exp(z))</code> computes $e^{z}$ first. In "
        "float32, $e^{89}$ already overflows to <code>inf</code>, and "
        "$\\log(1 + \\infty) = \\infty$. The correct implementation uses the "
        "identity $\\log(1+e^z) = \\max(z, 0) + \\log(1 + e^{-|z|})$, which never "
        "exponentiates a positive number. TensorFlow's "
        "<code>tf.math.softplus</code> already does this — the lesson is that "
        "hand-rolled maths needs the stable form, and the "
        "<code>@tf.custom_gradient</code> decorator lets you supply a stable "
        "gradient too.",
    )

    math(r"""
    \mathrm{softplus}(z) = \log\bigl(1 + e^{z}\bigr)
    \;=\;
    \max(z, 0) + \log\bigl(1 + e^{-|z|}\bigr)
    """)

    derive(
        [("For $z \\ge 0$, factor $e^{z}$ out of the logarithm:",
          r"\log(1 + e^{z}) = \log\bigl(e^{z}(e^{-z} + 1)\bigr) "
          r"= z + \log(1 + e^{-z})"),
         ("For $z < 0$ the original form is already safe, since $e^{z} < 1$:",
          r"\log(1 + e^{z}) = 0 + \log(1 + e^{z}) = 0 + \log(1 + e^{-|z|})"),
         ("Combining both branches gives the single stable expression:",
          r"\log(1 + e^{z}) = \max(z, 0) + \log\bigl(1 + e^{-|z|}\bigr)"),
         ("Now the exponent $-|z|$ is always $\\le 0$, so $e^{-|z|} \\in (0, 1]$ "
          "and nothing can overflow. This is the same log-sum-exp trick as the "
          "softmax stabilisation in §4.7 and the <code>from_logits=True</code> "
          "argument in §10.4.", None)],
        title="Deriving the numerically stable softplus",
    )

    anim_header("Where the naive implementation breaks")

    z = np.linspace(-100, 100, 600)
    with np.errstate(over="ignore"):
        naive = np.log(1.0 + np.exp(np.float32(z)))
    stable = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    frames = []
    for hi in np.linspace(5, 100, 34):
        m = np.abs(z) <= hi
        n_inf = int(np.sum(~np.isfinite(naive[m])))
        frames.append(go.Frame(name=f"{hi:.0f}", data=[
            go.Scatter(x=z[m], y=np.where(np.isfinite(naive[m]), naive[m], np.nan),
                       mode="lines", line=dict(color=C["danger"], width=4)),
            go.Scatter(x=z[m], y=stable[m], mode="lines",
                       line=dict(color=C["success"], width=2.5, dash="dash")),
        ], layout=go.Layout(
            xaxis=dict(range=[-hi, hi]), yaxis=dict(range=[-2, max(hi, 6)]),
            annotations=[anim.annotate_step(
                f"range ±{hi:.0f}   ·   naive gives inf for {n_inf} of "
                f"{int(m.sum())} points"
                + ("   ← OVERFLOW" if n_inf else ""),
                color=C["danger"] if n_inf else C["success"])])))

    f = go.Figure(data=[
        go.Scatter(x=z[np.abs(z) <= 5],
                   y=naive[np.abs(z) <= 5], mode="lines",
                   name="log(1 + exp(z))  — naive",
                   line=dict(color=C["danger"], width=4)),
        go.Scatter(x=z[np.abs(z) <= 5], y=stable[np.abs(z) <= 5], mode="lines",
                   name="max(z,0) + log1p(exp(−|z|))  — stable",
                   line=dict(color=C["success"], width=2.5, dash="dash")),
    ])
    f.update_layout(height=430, xaxis_title="z", yaxis_title="softplus(z)",
                    title="The naive softplus overflows in float32 above z ≈ 89",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(160), slider_prefix="range ±")
    figure(f, "The red curve simply stops existing past z ≈ 89 — every value "
              "there is inf, and so is every gradient.")

    code_lab(
        "All four custom components, plus @tf.custom_gradient",
        '''import numpy as np, tempfile, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)
tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch12b_"))

# ============ 1. THE FOUR, AS PLAIN FUNCTIONS ==========================
def my_softplus(z):
    return tf.math.log(1.0 + tf.exp(z))

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))      # a SCALAR

def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)   # a TENSOR

layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)
print("=== the four hooks ===")
print(f"activation  : f(z) -> tensor,   my_softplus(2.0) = {my_softplus(2.).numpy():.5f}")
W = my_glorot_initializer((100, 30))
print(f"initializer : f(shape) -> tensor, std = {W.numpy().std():.5f} "
      f"(target {np.sqrt(2/130):.5f})")
print(f"regularizer : f(W) -> SCALAR,   value = {my_l1_regularizer(W).numpy():.4f}")
print(f"constraint  : f(W) -> TENSOR,   negatives before "
      f"{np.mean(W.numpy() < 0):.1%} -> after "
      f"{np.mean(my_positive_weights(W).numpy() < 0):.1%}")

# ============ 2. NUMERICAL STABILITY ===================================
print("\\n=== the naive softplus overflows ===")
print(f"{'z':>8}{'naive':>16}{'stable':>16}{'tf.math.softplus':>20}")
for zv in [0., 10., 50., 88., 90., 200.]:
    t = tf.constant(zv, dtype=tf.float32)
    naive = tf.math.log(1.0 + tf.exp(t)).numpy()
    stable = (tf.maximum(t, 0.) + tf.math.log1p(tf.exp(-tf.abs(t)))).numpy()
    print(f"{zv:>8.0f}{naive:>16.6f}{stable:>16.6f}"
          f"{tf.math.softplus(t).numpy():>20.6f}")
print("float32 exp() overflows above ~88.7 -- and so does every gradient.")

# --- and the GRADIENT overflows too ----------------------------------
def grad_of(fn, zv):
    t = tf.Variable(zv, dtype=tf.float32)
    with tf.GradientTape() as tape:
        y = fn(t)
    return tape.gradient(y, t).numpy()
print(f"\\ngradient at z=100: naive {grad_of(my_softplus, 100.):.6f}   "
      f"stable {grad_of(tf.math.softplus, 100.):.6f}   (should be ~1.0)")

# ============ 3. @tf.custom_gradient ===================================
print("\\n=== supplying your own gradient ===")
@tf.custom_gradient
def my_better_softplus(z):
    exp_z = tf.exp(-tf.abs(z))
    def my_gradient(grad):
        return grad / (1 + tf.exp(-z))            # sigmoid(z), computed stably
    return tf.maximum(z, 0.) + tf.math.log1p(exp_z), my_gradient

print(f"{'z':>8}{'value':>14}{'gradient':>14}{'true sigmoid(z)':>18}")
for zv in [-50., 0., 50., 200.]:
    t = tf.Variable(zv, dtype=tf.float32)
    with tf.GradientTape() as tape:
        y = my_better_softplus(t)
    g = tape.gradient(y, t)
    print(f"{zv:>8.0f}{y.numpy():>14.5f}{g.numpy():>14.6f}"
          f"{tf.sigmoid(t).numpy():>18.6f}")

# ============ 4. SUBCLASSES WITH get_config ============================
@keras.utils.register_keras_serializable(package="MLPlatform")
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor=0.01):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}

@keras.utils.register_keras_serializable(package="MLPlatform")
class MyPositiveWeights(keras.constraints.Constraint):
    def __call__(self, weights):
        return tf.where(weights < 0., tf.zeros_like(weights), weights)
    def get_config(self):
        return {}

@keras.utils.register_keras_serializable(package="MLPlatform")
class MyGlorotNormal(keras.initializers.Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain
    def __call__(self, shape, dtype=None):
        stddev = self.gain * tf.sqrt(2. / (shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)
    def get_config(self):
        return {"gain": self.gain}

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (1500, 8)).astype("float32")
y = (X @ rng.normal(0, 1, 8) + rng.normal(0, .3, 1500)).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(32, activation="relu",
                       kernel_initializer=MyGlorotNormal(gain=1.2),
                       kernel_regularizer=MyL1Regularizer(0.005),
                       kernel_constraint=MyPositiveWeights()),
    keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-2))
model.fit(X, y, epochs=15, batch_size=64, verbose=0)

W = model.layers[0].get_weights()[0]
print(f"\\n=== after training with the constraint ===")
print(f"negative weights in layer 0: {np.mean(W < 0):.2%}   <- projected to 0")
print(f"exactly-zero weights       : {np.mean(W == 0):.2%}")

model.save(tmp/"custom.keras")
reloaded = keras.models.load_model(tmp/"custom.keras")
print(f"\\nreloaded with no custom_objects (all registered)")
print(f"  regulariser factor restored: "
      f"{reloaded.layers[0].kernel_regularizer.factor}")
print(f"  initialiser gain restored  : "
      f"{reloaded.layers[0].kernel_initializer.gain}")
print(f"  predictions identical      : "
      f"{np.allclose(model.predict(X[:20], verbose=0), reloaded.predict(X[:20], verbose=0))}")

# ============ 5. REGULARISER vs CONSTRAINT =============================
print("\\n=== a regulariser is a PENALTY, a constraint is a PROJECTION ===")
m_reg = keras.Sequential([keras.layers.Input(shape=(8,)),
                          keras.layers.Dense(32, activation="relu",
                                             kernel_regularizer=MyL1Regularizer(.05)),
                          keras.layers.Dense(1)])
m_con = keras.Sequential([keras.layers.Input(shape=(8,)),
                          keras.layers.Dense(32, activation="relu",
                                             kernel_constraint=keras.constraints.max_norm(0.5)),
                          keras.layers.Dense(1)])
for nm, m in [("l1 regulariser (penalty)", m_reg), ("max_norm (projection)", m_con)]:
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-2))
    m.fit(X, y, epochs=20, batch_size=64, verbose=0)
    W = m.layers[0].get_weights()[0]
    norms = np.linalg.norm(W, axis=0)
    print(f"  {nm:<28} max column norm {norms.max():.4f}   "
          f"{np.mean(np.abs(W) < 1e-4):.1%} near-zero weights")
print("  max_norm GUARANTEES the bound; l1 only pushes toward zero.")

shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch12_components",
    )

    keypoints([
        "Activation: $f(z)$. Initializer: $f(\\text{shape})$. Regularizer: "
        "$f(W) \\to$ <b>scalar</b>. Constraint: $f(W) \\to$ <b>tensor</b>.",
        "A regulariser is a <b>penalty</b> added to the loss; a constraint is a "
        "<b>projection</b> applied after each step.",
        "Subclass + <code>get_config</code> whenever a hyperparameter must "
        "survive saving.",
        "Watch numerical stability: use the log-sum-exp form, and "
        "<code>@tf.custom_gradient</code> when the autodiff gradient is unstable.",
        "<code>@keras.utils.register_keras_serializable()</code> makes "
        "<code>load_model</code> work with no arguments.",
    ])


# ==========================================================================
def s_12_5():
    section("12.5", "Custom Metrics")

    lead(
        "Metrics look like losses but differ in one crucial respect: they must be "
        "<b>streamed</b>. Getting that wrong gives you a number that is subtly, "
        "silently incorrect."
    )

    sub("Losses vs metrics")

    table(
        ["", "Loss", "Metric"],
        [["Purpose", "Trained on — must be differentiable",
          "Reported — <b>need not</b> be differentiable"],
         ["Examples", "Cross-entropy, MSE, Huber",
          "Accuracy, precision, recall, F1, AUC"],
         ["Computed", "Per batch, averaged",
          "<b>Accumulated across batches</b> via a streaming state"],
         ["Base class", "<code>keras.losses.Loss</code>",
          "<code>keras.metrics.Metric</code>"],
         ["Required methods", "<code>call</code>",
          "<code>update_state</code>, <code>result</code>, "
          "<code>reset_state</code>"]],
    )

    sub("Why streaming matters")

    derive(
        [("Suppose you compute precision batch by batch and average the results. "
          "Take two batches:", None),
         ("Batch 1: 5 predicted positive, 4 correct → precision $4/5 = 0.8$.<br>"
          "Batch 2: 1 predicted positive, 0 correct → precision $0/1 = 0.0$.",
          None),
         ("The naive average of the batch precisions:",
          r"\frac{0.8 + 0.0}{2} = 0.40"),
         ("The <b>true</b> precision over both batches pools the counts:",
          r"\frac{TP_1 + TP_2}{(TP_1 + FP_1) + (TP_2 + FP_2)} = \frac{4 + 0}{5 + 1} "
          r"= \frac{4}{6} = 0.667"),
         ("The naive average is wrong by 27 percentage points, because it weights "
          "a 1-instance batch as heavily as a 5-instance one. A <b>streaming</b> "
          "metric keeps running totals of $TP$ and $FP$ and divides at the end — "
          "which is what <code>update_state</code> / <code>result</code> exist "
          "for.", None),
         ("Any metric that is <b>not</b> a simple mean over instances — precision, "
          "recall, $F_1$, AUC, IoU — must be streamed. Mean-based metrics (MAE, "
          "MSE, accuracy) can be averaged, and <code>keras.metrics.Mean</code> "
          "handles them.", None)],
        title="Why averaging batch metrics gives the wrong answer",
    )

    md(
        """
```python
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_metrics = create_huber(self.threshold)(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(sample_metrics))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}
```
        """
    )

    codenote(
        "reset_state is called at the start of every epoch",
        "The base <code>Metric</code> class provides a default that zeroes every "
        "weight created with <code>add_weight</code>, so you usually do not need "
        "to write it. Override it only if your state is more complex than a set "
        "of scalars.",
    )

    anim_header("Streaming vs naive averaging, batch by batch")
    md(
        "Precision computed both ways over 30 unbalanced batches. The green line "
        "streams the counts; the red line averages the per-batch precisions. Watch "
        "them diverge — and note that the red line is not even monotone in the "
        "data."
    )

    rng = np.random.default_rng(4)
    n_b = 30
    tps, fps = [], []
    for b in range(n_b):
        size = int(rng.integers(1, 40))
        pred_pos = int(rng.integers(0, size + 1))
        tp = int(rng.binomial(pred_pos, .72)) if pred_pos else 0
        tps.append(tp); fps.append(pred_pos - tp)

    stream, naive = [], []
    ct, cf, acc = 0, 0, []
    for b in range(n_b):
        ct += tps[b]; cf += fps[b]
        stream.append(ct / max(ct + cf, 1))
        pp = tps[b] + fps[b]
        acc.append(tps[b] / pp if pp else 0.0)
        naive.append(float(np.mean(acc)))

    frames = []
    for k in range(1, n_b + 1):
        frames.append(go.Frame(name=str(k), data=[
            go.Bar(x=list(range(1, k + 1)), y=tps[:k],
                   marker=dict(color=C["success"])),
            go.Bar(x=list(range(1, k + 1)), y=fps[:k],
                   marker=dict(color=C["danger"])),
            go.Scatter(x=list(range(1, k + 1)), y=stream[:k], mode="lines+markers",
                       line=dict(color=C["success"], width=3.4)),
            go.Scatter(x=list(range(1, k + 1)), y=naive[:k], mode="lines+markers",
                       line=dict(color=C["danger"], width=3, dash="dash")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"after {k} batches   ·   streaming = {stream[k-1]:.4f}   ·   "
            f"naive average = {naive[k-1]:.4f}   ·   "
            f"error = {abs(naive[k-1]-stream[k-1]):.4f}",
            color=C["danger"])])))

    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.38, .62],
                      subplot_titles=("TP (green) and FP (red) per batch",
                                      "precision: streamed vs naively averaged"))
    f.add_trace(go.Bar(x=[1], y=tps[:1], name="TP",
                       marker=dict(color=C["success"])), 1, 1)
    f.add_trace(go.Bar(x=[1], y=fps[:1], name="FP",
                       marker=dict(color=C["danger"])), 1, 1)
    f.add_trace(go.Scatter(x=[1], y=stream[:1], mode="lines+markers",
                           name="streaming (correct)",
                           line=dict(color=C["success"], width=3.4)), 2, 1)
    f.add_trace(go.Scatter(x=[1], y=naive[:1], mode="lines+markers",
                           name="mean of batch precisions (wrong)",
                           line=dict(color=C["danger"], width=3, dash="dash")), 2, 1)
    f.update_layout(height=520, barmode="stack",
                    title="A metric that is not a mean must be streamed")
    f.update_xaxes(title_text="batch", row=2, col=1)
    f.update_yaxes(range=[0, 1.05], title_text="precision", row=2, col=1)
    anim.animate(f, frames, duration=nav.anim_ms(220), slider_prefix="batch ")
    figure(f)

    code_lab(
        "Streaming metrics done right, and the error when done wrong",
        '''import numpy as np, tempfile, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)
tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch12c_"))

# ============ 1. THE NAIVE AVERAGE IS WRONG ============================
print("=== why batch-averaging a metric fails ===")
batches = [(4, 5), (0, 1), (18, 20), (1, 2), (0, 3)]      # (TP, predicted positive)
naive = np.mean([tp/pp if pp else 0. for tp, pp in batches])
true  = sum(tp for tp, _ in batches) / sum(pp for _, pp in batches)
print(f"{'batch':>7}{'TP':>5}{'pred pos':>10}{'batch precision':>18}")
for i, (tp, pp) in enumerate(batches, 1):
    print(f"{i:>7}{tp:>5}{pp:>10}{(tp/pp if pp else 0.):>18.4f}")
print(f"\\nmean of batch precisions : {naive:.4f}   <- WRONG")
print(f"pooled (true) precision  : {true:.4f}")
print(f"error                    : {abs(naive-true):.4f}")

# Keras's built-in Precision streams correctly
p = keras.metrics.Precision()
for tp, pp in batches:
    y_true = np.r_[np.ones(tp), np.zeros(pp-tp), np.ones(3)]
    y_pred = np.r_[np.ones(pp),                  np.zeros(3)]
    p.update_state(y_true, y_pred)
print(f"keras.metrics.Precision  : {p.result().numpy():.4f}   <- correct")

# ============ 2. A CUSTOM STREAMING METRIC =============================
@keras.utils.register_keras_serializable(package="MLPlatform")
class HuberMetric(keras.metrics.Metric):
    """Streaming mean Huber loss."""
    def __init__(self, threshold=1.0, name="huber_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, y_pred.dtype)
        error = y_true - y_pred
        is_small = tf.abs(error) < self.threshold
        per_sample = tf.where(is_small, tf.square(error)/2,
                              self.threshold*(tf.abs(error) - self.threshold/2))
        if sample_weight is not None:
            per_sample = per_sample * tf.cast(sample_weight, per_sample.dtype)
        self.total.assign_add(tf.reduce_sum(per_sample))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}

m = HuberMetric(2.0)
m.update_state(tf.constant([[2.]]), tf.constant([[10.]]))     # error 8
print(f"\\n=== custom streaming metric ===")
print(f"after 1 sample (error 8, threshold 2): {m.result().numpy():.4f}")
print(f"  check: 2*(8 - 1) = {2*(8-1)}")
m.update_state(tf.constant([[0.], [5.]]), tf.constant([[1.], [9.25]]))
print(f"after 3 samples total               : {m.result().numpy():.4f}")
print(f"internal state -> total={m.total.numpy():.4f} count={m.count.numpy():.0f}")
m.reset_state()
print(f"after reset_state()                 : total={m.total.numpy()} "
      f"count={m.count.numpy()}")

# ============ 3. A METRIC THAT IS NOT DIFFERENTIABLE ===================
@keras.utils.register_keras_serializable(package="MLPlatform")
class F1Score(keras.metrics.Metric):
    """F1 from streamed TP/FP/FN -- NOT differentiable, and that is fine."""
    def __init__(self, threshold=0.5, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_hat  = tf.cast(tf.reshape(y_pred, [-1]) > self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_hat))
        self.fp.assign_add(tf.reduce_sum((1-y_true) * y_hat))
        self.fn.assign_add(tf.reduce_sum(y_true * (1-y_hat)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + keras.backend.epsilon())
        recall    = self.tp / (self.tp + self.fn + keras.backend.epsilon())
        return 2*precision*recall / (precision + recall + keras.backend.epsilon())

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}

# ============ 4. USE THEM IN A REAL MODEL ==============================
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=3000, n_features=15, weights=[.85, .15],
                           random_state=42)
X = X.astype("float32"); y = y.astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y,
                                      random_state=42)

model = keras.Sequential([keras.layers.Input(shape=(15,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")])
model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3),
              metrics=["accuracy", keras.metrics.Precision(),
                       keras.metrics.Recall(), F1Score(), keras.metrics.AUC()])
model.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)

res = model.evaluate(Xte, yte, verbose=0, return_dict=True)
print()
print("=== evaluated on the test set ===")
for n, v in res.items():
    print(f"  {n:<24} {v:.4f}")

# verify F1 against sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
pred = (model.predict(Xte, verbose=0).ravel() > .5).astype(int)
print(f"\\n  sklearn f1_score        {f1_score(yte, pred):.4f}   <- matches")
print(f"  sklearn precision       {precision_score(yte, pred):.4f}")
print(f"  sklearn recall          {recall_score(yte, pred):.4f}")
print(f"\\n  the positive class is only {yte.mean():.1%} of the data, so accuracy")
print(f"  {res['accuracy']:.4f} is far less informative than F1 "
      f"{f1_score(yte, pred):.4f} (chapter 3)")

# ============ 5. SAVING ================================================
model.save(tmp/"metrics.keras")
r = keras.models.load_model(tmp/"metrics.keras")
r_res = r.evaluate(Xte, yte, verbose=0, return_dict=True)
print(f"reloaded with registered metrics; test F1 = "
      f"{[v for k, v in r_res.items() if 'f1' in k][0]:.4f}")
shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch12_metrics",
    )

    quiz(
        "Why must a custom precision metric implement <code>update_state</code> "
        "rather than just returning a value per batch?",
        ["Because precision is not differentiable",
         "Because averaging per-batch precisions gives the wrong pooled value",
         "Because Keras requires it for all metrics",
         "Because it is faster"],
        1,
        "Precision is a ratio of pooled counts, not a mean over instances. "
        "Averaging batch precisions weights a 1-instance batch as heavily as a "
        "1000-instance one. Streaming $TP$ and $FP$ and dividing once at the end "
        "is the only correct way.",
        key="ch12q1",
    )

    keypoints([
        "Metrics need <b>not</b> be differentiable; losses must be.",
        "Any metric that is not a plain mean over instances must be "
        "<b>streamed</b>.",
        "<code>update_state</code> accumulates, <code>result</code> computes, "
        "<code>reset_state</code> zeroes at each epoch.",
        "State goes in <code>add_weight(...)</code> so Keras manages and saves it.",
        "Averaging per-batch precision can be off by tens of percentage points.",
    ])


# ==========================================================================
def s_12_6():
    section("12.6", "Custom Layers")

    lead(
        "Three kinds: layers with no weights, layers with weights, and layers "
        "that behave differently during training. Each has a required method."
    )

    sub("Stateless layers")

    md(
        "If the layer has no weights, `keras.layers.Lambda` is enough — but a "
        "subclass is more readable and serialises properly:"
    )

    md(
        """
```python
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
```
        """
    )

    sub("Layers with weights — the three methods")

    table(
        ["Method", "When it runs", "What it does"],
        [["<code>__init__</code>", "When you construct the layer",
          "Stores hyperparameters. <b>The input shape is not known yet.</b>"],
         ["<code>build(input_shape)</code>", "Once, on the first call",
          "Creates the weights with <code>add_weight</code>, now that the input "
          "shape is known"],
         ["<code>call(X, training=None)</code>", "Every forward pass",
          "The actual computation"],
         ["<code>get_config</code>", "On save",
          "Returns the hyperparameters for reconstruction"],
         ["<code>compute_output_shape</code>", "Shape inference",
          "Usually inferred automatically in Keras 3"]],
    )

    idea(
        "Why build() exists at all",
        "You want to write <code>Dense(30)</code> without also stating the input "
        "size — the framework should infer it from whatever is connected. So the "
        "weight shape $(n_{\\text{in}}, 30)$ cannot be known at construction "
        "time. <code>build</code> is called <b>lazily</b>, on the first call, when "
        "the input shape finally is known. That is also why "
        "<code>model.summary()</code> fails on a subclassed model that has never "
        "been called (§10.6) — nothing has been built yet.",
    )

    md(
        """
```python
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[input_shape[-1], self.units],
            initializer="he_normal", trainable=True)
        self.bias = self.add_weight(
            name="bias", shape=[self.units],
            initializer="zeros", trainable=True)

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def get_config(self):
        return {**super().get_config(), "units": self.units,
                "activation": keras.activations.serialize(self.activation)}
```
        """
    )

    sub("Layers that behave differently during training")

    md(
        "Dropout and batch norm need to know whether they are training. The "
        "`training` argument of `call` carries that:"
    )

    md(
        """
```python
class MyGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        if training:                       # a PYTHON bool here — this is fine
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        return X
```
        """
    )

    note(
        "This <code>if</code> is allowed",
        "<code>training</code> is a genuine Python boolean (or <code>None</code>) "
        "passed by the framework, <b>not</b> a tensor — so branching on it is "
        "fine, and <code>tf.function</code> simply traces two separate graphs, "
        "one per branch. Contrast with §12.3, where the condition was a tensor of "
        "per-element booleans and required <code>tf.where</code>.",
    )

    sub("Multiple inputs and outputs")

    md(
        "`call` receives a list, and returns a list. `compute_output_shape` "
        "likewise:"
    )

    md(
        """
```python
class MyMultiLayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        return X1 + X2, X1 * X2, X1 / X2
```
        """
    )

    anim_header("The lifecycle of a custom layer")

    steps = [
        ("__init__(units=30)", "hyperparameters stored; NO weights yet",
         C["accent"], "self.units = 30"),
        ("first call: layer(X) with X.shape=(32, 64)",
         "the input shape is finally known", C["warning"],
         "input_shape = (None, 64)"),
        ("build((None, 64)) runs — ONCE",
         "weights created with add_weight", SEQ[0],
         "kernel (64, 30) · bias (30,)"),
        ("call(X) runs", "the forward computation", SEQ[1],
         "return activation(X @ kernel + bias)"),
        ("call(X) again, and again…", "build is NOT called again", SEQ[2],
         "weights are reused"),
        ("model.save()", "get_config() is written into the file", SEQ[3],
         "{'units': 30, 'activation': 'relu'}"),
        ("load_model()", "YourClass(**config), then build on first call",
         C["success"], "the layer is reconstructed"),
    ]
    frames = []
    for k in range(1, len(steps) + 1):
        shapes, ann = [], []
        for i, (nm, desc, col, detail) in enumerate(steps):
            active = i < k
            cur = i == k - 1
            y = len(steps) - i
            shapes.append(go.Scatter(
                x=[0, 5.6, 5.6, 0, 0],
                y=[y - .40, y - .40, y + .40, y + .40, y - .40],
                fill="toself",
                fillcolor=(alpha(col, .85) if cur else
                           alpha(col, .35) if active else alpha(C["line"], .3)),
                line=dict(color="#fff" if active else C["line"], width=2),
                hoverinfo="skip", showlegend=False))
            ann.append(dict(x=.2, y=y, text=f"<b>{nm}</b>" if cur else nm,
                            showarrow=False, xanchor="left",
                            font=dict(size=11,
                                      color="#fff" if active else C["muted"])))
            if active:
                ann.append(dict(x=5.8, y=y, text=detail, showarrow=False,
                                xanchor="left",
                                font=dict(size=9.5, color=C["ink_soft"],
                                          family="JetBrains Mono, monospace")))
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann,
                                                title=steps[k - 1][1])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=470, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.2, 13.5]),
                    yaxis=dict(visible=False, range=[.3, 7.8]),
                    annotations=list(frames[0].layout.annotations),
                    title=steps[0][1])
    anim.animate(f, frames, duration=nav.anim_ms(1200), slider_prefix="step ")
    figure(f)

    code_lab(
        "Five custom layers, from stateless to multi-input",
        '''import numpy as np, tempfile, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)
tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch12d_"))

# ============ 1. STATELESS ============================================
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
print("=== stateless (Lambda) ===")
print(f"  exp([0, 1, 2]) = {exponential_layer(tf.constant([0., 1., 2.])).numpy().round(4)}")

# ============ 2. WITH WEIGHTS =========================================
@keras.utils.register_keras_serializable(package="MLPlatform")
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        print(f"    build() called with input_shape={input_shape}")
        self.kernel = self.add_weight(name="kernel",
                                      shape=[input_shape[-1], self.units],
                                      initializer="he_normal", trainable=True)
        self.bias = self.add_weight(name="bias", shape=[self.units],
                                    initializer="zeros", trainable=True)

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def get_config(self):
        return {**super().get_config(), "units": self.units,
                "activation": keras.activations.serialize(self.activation)}

print("\\n=== the lifecycle ===")
layer = MyDense(30, activation="relu")
print(f"  after __init__: has kernel? {hasattr(layer, 'kernel')}  "
      f"built? {layer.built}")
out = layer(tf.random.normal((8, 64)))
print(f"  after first call: built? {layer.built}  "
      f"kernel {tuple(layer.kernel.shape)}  output {tuple(out.shape)}")
out2 = layer(tf.random.normal((16, 64)))
print(f"  second call: build NOT re-run, output {tuple(out2.shape)}")
print(f"  trainable weights: {[w.name for w in layer.trainable_weights]}")

# ============ 3. TRAINING-DEPENDENT BEHAVIOUR =========================
@keras.utils.register_keras_serializable(package="MLPlatform")
class MyGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    def call(self, X, training=None):
        if training:                       # a Python bool -- allowed
            return X + tf.random.normal(tf.shape(X), stddev=self.stddev)
        return X
    def get_config(self):
        return {**super().get_config(), "stddev": self.stddev}

noise = MyGaussianNoise(0.5)
x = tf.ones((4, 3))
print(f"\\n=== training-dependent ===")
print(f"  training=True  : {noise(x, training=True).numpy()[0].round(3)}")
print(f"  training=False : {noise(x, training=False).numpy()[0].round(3)}  <- identity")

# ============ 4. MULTIPLE INPUTS AND OUTPUTS ==========================
@keras.utils.register_keras_serializable(package="MLPlatform")
class MyMultiLayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        return X1 + X2, X1 * X2, tf.math.divide_no_nan(X1, X2)

ml = MyMultiLayer()
a, b, c = ml([tf.constant([[6., 8.]]), tf.constant([[2., 4.]])])
print(f"\\n=== multi in / multi out ===")
print(f"  sum {a.numpy()}  product {b.numpy()}  quotient {c.numpy()}")

# ============ 5. A LAYER WITH ITS OWN LOSS AND METRIC =================
@keras.utils.register_keras_serializable(package="MLPlatform")
class ActivityRegularizedDense(keras.layers.Layer):
    """Penalises large ACTIVATIONS (not weights) -- an internal loss."""
    def __init__(self, units, factor=0.01, **kwargs):
        super().__init__(**kwargs)
        self.units, self.factor = units, factor
        self.mean_act = keras.metrics.Mean(name="mean_activation")

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-1], self.units],
                                      initializer="he_normal", name="kernel")
        self.bias = self.add_weight(shape=[self.units], initializer="zeros",
                                    name="bias")

    def call(self, X, training=None):
        out = tf.nn.relu(X @ self.kernel + self.bias)
        if training:
            self.add_loss(self.factor * tf.reduce_mean(tf.square(out)))
            self.mean_act.update_state(tf.reduce_mean(out))
        return out

    def get_config(self):
        return {**super().get_config(), "units": self.units,
                "factor": self.factor}

# ============ 6. BUILD A REAL MODEL ===================================
rng = np.random.default_rng(0)
X = rng.normal(0, 1, (2000, 20)).astype("float32")
y = (X @ rng.normal(0, 1, 20) + rng.normal(0, .3, 2000)).astype("float32")

print("\\n=== a model made of custom layers ===")
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    MyGaussianNoise(0.05),
    MyDense(64, activation="relu"),
    ActivityRegularizedDense(32, factor=0.001),
    MyDense(1),
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-2))
model.fit(X, y, epochs=20, batch_size=64, verbose=0)
print()
print(f"  test MSE = {np.ravel(model.evaluate(X, y, verbose=0))[0]:.5f}")
model.summary()

# ============ 7. SAVE AND RELOAD ======================================
model.save(tmp/"custom_layers.keras")
r = keras.models.load_model(tmp/"custom_layers.keras")
print(f"\\nreloaded. predictions identical: "
      f"{np.allclose(model.predict(X[:20], verbose=0), r.predict(X[:20], verbose=0), atol=1e-5)}")
print(f"  MyDense units restored : {r.layers[1].units}")
print(f"  noise stddev restored  : {r.layers[0].stddev}")
print(f"  reg factor restored    : {r.layers[2].factor}")

# ============ 8. build() IS WHY summary() CAN FAIL ====================
print("\\n=== the unbuilt-model trap ===")
class Sub(keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = MyDense(16, activation="relu")
        self.d2 = MyDense(1)
    def call(self, X):
        return self.d2(self.d1(X))

s = Sub()
try:
    s.summary()
except Exception as e:
    print(f"  summary() before any call -> {type(e).__name__}")
s(tf.zeros((1, 20)))               # one call builds everything
print(f"  after one call, summary() works: {s.count_params()} parameters")

shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch12_layers",
    )

    keypoints([
        "<code>__init__</code> stores hyperparameters; <code>build</code> creates "
        "weights <b>lazily</b> once the input shape is known; <code>call</code> "
        "computes.",
        "<code>add_weight</code> registers the variable so Keras tracks, trains "
        "and saves it.",
        "<code>call(X, training=None)</code> — branching on "
        "<code>training</code> is fine because it is a Python bool.",
        "<code>get_config</code> is required for saving; register the class to "
        "avoid <code>custom_objects</code>.",
        "A layer can add its own loss (<code>add_loss</code>) and its own metrics.",
    ])


# ==========================================================================
def s_12_7():
    section("12.7", "Custom Models and Internal Losses")

    lead(
        "Subclass <code>keras.Model</code> when the architecture involves loops, "
        "conditionals or recursion. And use <code>add_loss</code> when the loss "
        "depends on something inside the network rather than on the labels."
    )

    sub("A residual block, as a custom layer")

    md(
        "The canonical example — a skip connection, which Chapter 14 shows is the "
        "reason 150-layer networks are trainable:"
    )

    math(r"""
    \mathbf{y} \;=\; \mathbf{x} \;+\; \mathcal{F}(\mathbf{x})
    """)

    proof(
        "Why a skip connection fixes vanishing gradients",
        "Differentiate: $\\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}} = "
        "\\mathbf{I} + \\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}}$. The "
        "identity term means the Jacobian can never be near-zero, so the product "
        "of §11.1 has a guaranteed floor of 1 per block. Even if "
        "$\\mathcal{F}$'s gradient vanishes entirely, the gradient still flows "
        "through the skip path unattenuated. This is the entire mechanism behind "
        "ResNet.",
    )

    sub("Losses and metrics based on model internals")

    md(
        "Sometimes the loss depends on **hidden activations**, not on "
        "$y_{\\text{true}}$ — a reconstruction loss, an activity penalty, a KL "
        "divergence in a VAE (Chapter 17). `add_loss` is how you attach one:"
    )

    md(
        """
```python
class ReconstructingRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation="relu") for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
        self.recon_mean = keras.metrics.Mean(name="reconstruction_error")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)   # back to input size

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)                  # <- the internal loss
        if training:
            self.recon_mean.update_state(recon_loss)
        return self.out(Z)
```
        """
    )

    idea(
        "The reconstruction loss is a regulariser",
        "Forcing the hidden representation to retain enough information to "
        "reconstruct the input stops the network from throwing away everything "
        "except what the current labels need. That makes the representation more "
        "general and often improves generalisation — it is exactly the "
        "<b>unsupervised auxiliary task</b> idea of §11.5, expressed as an "
        "internal loss. Chapter 17 builds a whole family of models on it.",
    )

    warn(
        "The internal loss weight is a hyperparameter you must tune",
        "<code>self.add_loss(0.05 * recon_loss)</code> — that 0.05 controls how "
        "much the model cares about reconstruction versus prediction. Too large "
        "and the model becomes an autoencoder that ignores the labels; too small "
        "and the regularisation does nothing. Cross-validate it.",
    )

    anim_header("A residual block: the gradient always has a path home")
    md(
        "Two networks of increasing depth. The plain one's gradient decays "
        "geometrically; the residual one's does not, because $\\mathbf{I} + "
        "\\partial\\mathcal{F}$ has a floor of 1."
    )

    depths = np.arange(1, 61)
    gamma = .72
    plain = gamma ** depths
    resid = np.ones_like(depths, dtype=float)
    rngr = np.random.default_rng(0)
    resid = np.cumprod(1 + rngr.normal(0, .04, len(depths)) * gamma)

    frames = []
    for k in range(2, 61):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=depths[:k], y=plain[:k], mode="lines",
                       line=dict(color=C["danger"], width=3.4)),
            go.Scatter(x=depths[:k], y=resid[:k], mode="lines",
                       line=dict(color=C["success"], width=3.4)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"depth {k}   ·   plain network gradient = {plain[k-1]:.3e}   ·   "
            f"residual network gradient = {resid[k-1]:.3f}   ·   "
            f"ratio {resid[k-1]/max(plain[k-1],1e-300):.2e}")])))

    f = go.Figure(data=[
        go.Scatter(x=depths[:2], y=plain[:2], mode="lines",
                   name="plain stack:  ∏ ∂F/∂x  →  0",
                   line=dict(color=C["danger"], width=3.4)),
        go.Scatter(x=depths[:2], y=resid[:2], mode="lines",
                   name="residual:  ∏ (I + ∂F/∂x)  ≈  1",
                   line=dict(color=C["success"], width=3.4)),
    ])
    f.update_layout(height=430, yaxis_type="log", xaxis_title="depth",
                    yaxis_title="gradient magnitude at the input",
                    title="Skip connections give the gradient a free path",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(60), slider_prefix="depth ")
    figure(f)

    code_lab(
        "Residual blocks, a custom Model, and internal losses",
        '''import numpy as np, tempfile, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)
tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch12e_"))

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (4000, 20)).astype("float32")
w = rng.normal(0, 1, 20)
y = (np.tanh(X @ w) * 3 + X[:, 0]*X[:, 1] + rng.normal(0, .3, 4000)).astype("float32")
Xtr, Xte, ytr, yte = X[:3000], X[3000:], y[:3000], y[3000:]

# ============ 1. A RESIDUAL BLOCK =====================================
@keras.utils.register_keras_serializable(package="MLPlatform")
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.n_layers, self.n_neurons = n_layers, n_neurons
        self.hidden = [keras.layers.Dense(n_neurons, activation="relu",
                                          kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z                       # THE SKIP CONNECTION

    def get_config(self):
        return {**super().get_config(), "n_layers": self.n_layers,
                "n_neurons": self.n_neurons}

# ============ 2. A CUSTOM MODEL with a Python loop ====================
@keras.utils.register_keras_serializable(package="MLPlatform")
class ResidualRegressor(keras.Model):
    def __init__(self, output_dim, n_blocks=3, width=30, **kwargs):
        super().__init__(**kwargs)
        self.output_dim, self.n_blocks, self.width = output_dim, n_blocks, width
        self.hidden1 = keras.layers.Dense(width, activation="relu",
                                          kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, width)
        self.block2 = ResidualBlock(2, width)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + self.n_blocks):      # a real Python loop
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

    def get_config(self):
        return {**super().get_config(), "output_dim": self.output_dim,
                "n_blocks": self.n_blocks, "width": self.width}

model = ResidualRegressor(1, n_blocks=3)
model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
model.fit(Xtr, ytr, epochs=30, batch_size=64, verbose=0)
print(f"=== residual regressor ===")
print(f"test MSE = {np.ravel(model.evaluate(Xte, yte, verbose=0))[0]:.5f}")
print(f"parameters = {model.count_params():,}  "
      f"(block1 is REUSED 4 times -- its weights are shared)")

# ============ 3. SKIP CONNECTIONS AT DEPTH ============================
print("\\n=== plain vs residual, as depth grows ===")
def plain_net(depth, width=30):
    layers = [keras.layers.Input(shape=(20,))]
    for _ in range(depth):
        layers.append(keras.layers.Dense(width, activation="relu",
                                         kernel_initializer="he_normal"))
    layers.append(keras.layers.Dense(1))
    return keras.Sequential(layers)

def residual_net(depth, width=30):
    inp = keras.layers.Input(shape=(20,))
    z = keras.layers.Dense(width, activation="relu",
                           kernel_initializer="he_normal")(inp)
    for _ in range(depth//2):
        z = ResidualBlock(2, width)(z)
    return keras.Model(inp, keras.layers.Dense(1)(z))

print(f"{'depth':>7}{'plain test MSE':>18}{'residual test MSE':>21}")
for depth in [4, 12, 30]:
    tf.random.set_seed(0)
    p = plain_net(depth); p.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
    p.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)
    tf.random.set_seed(0)
    r = residual_net(depth); r.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
    r.fit(Xtr, ytr, epochs=25, batch_size=64, verbose=0)
    print(f"{depth:>7}{np.ravel(p.evaluate(Xte, yte, verbose=0))[0]:>18.5f}"
          f"{np.ravel(r.evaluate(Xte, yte, verbose=0))[0]:>21.5f}")
print("The gap widens with depth -- that is exactly what skip connections buy.")

# ============ 4. AN INTERNAL LOSS =====================================
@keras.utils.register_keras_serializable(package="MLPlatform")
class ReconstructingRegressor(keras.Model):
    """Predicts y AND reconstructs x -- the reconstruction is a regulariser."""
    def __init__(self, output_dim, recon_weight=0.05, **kwargs):
        super().__init__(**kwargs)
        self.output_dim, self.recon_weight = output_dim, recon_weight
        self.hidden = [keras.layers.Dense(30, activation="relu",
                                          kernel_initializer="he_normal")
                       for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
        self.recon_mean = keras.metrics.Mean(name="recon_error")

    def build(self, batch_input_shape):
        self.reconstruct = keras.layers.Dense(batch_input_shape[-1])
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(self.recon_weight * recon_loss)     # <- INTERNAL LOSS
        if training:
            self.recon_mean.update_state(recon_loss)
        return self.out(Z)

    def get_config(self):
        return {**super().get_config(), "output_dim": self.output_dim,
                "recon_weight": self.recon_weight}

print("\\n=== the reconstruction loss as a regulariser ===")
print(f"{'recon_weight':>14}{'test MSE':>12}{'recon error':>15}")
for wgt in [0.0, 0.01, 0.05, 0.3, 2.0]:
    tf.random.set_seed(0)
    m = ReconstructingRegressor(1, recon_weight=wgt)
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
    h = m.fit(Xtr[:600], ytr[:600], epochs=40, batch_size=32, verbose=0)
    print(f"{wgt:>14}{np.ravel(m.evaluate(Xte, yte, verbose=0))[0]:>12.5f}"
          f"{float(m.recon_mean.result()):>15.5f}")
print("\\nToo large and the model becomes an autoencoder that ignores y.")
print("Too small and the regularisation does nothing. It is a hyperparameter.")

# ============ 5. add_loss vs a compiled loss ==========================
m = ReconstructingRegressor(1, recon_weight=.05)
m.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
m.fit(Xtr[:400], ytr[:400], epochs=5, batch_size=32, verbose=0)
print(f"\\n=== where add_loss shows up ===")
print(f"  model.losses holds {len(m.losses)} internal loss term(s);"
      f" they are added to the compiled loss at every step")
print(f"  these are ADDED to the compiled loss automatically at every step")

model.save(tmp/"resid.keras")
r = keras.models.load_model(tmp/"resid.keras")
print(f"\\nreloaded residual regressor: test MSE "
      f"{np.ravel(r.evaluate(Xte, yte, verbose=0))[0]:.5f}")
shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch12_models",
    )

    keypoints([
        "Subclass <code>keras.Model</code> for loops, conditionals and shared "
        "sub-blocks.",
        "A residual block computes $\\mathbf{x} + \\mathcal{F}(\\mathbf{x})$; its "
        "Jacobian $\\mathbf{I} + \\partial\\mathcal{F}$ never vanishes.",
        "<code>self.add_loss(...)</code> attaches a loss computed from "
        "<b>internal activations</b>, not from labels.",
        "The internal-loss weight is a hyperparameter — tune it.",
        "Reusing a layer object shares its weights; that is a feature.",
    ])


# ==========================================================================
def s_12_8():
    section("12.8", "Autodiff and Custom Training Loops")

    lead(
        "<code>GradientTape</code> records operations so it can replay them "
        "backwards. Once you can compute gradients explicitly, you can write any "
        "training procedure you like."
    )

    sub("GradientTape")

    md(
        """
```python
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```

The tape records every operation involving a **watched** variable. `tf.Variable`
objects are watched automatically; constants are not — call `tape.watch(x)` to
add one.
        """
    )

    table(
        ["Behaviour", "Rule"],
        [["Tape lifetime", "A tape is <b>consumed</b> by the first "
          "<code>gradient()</code> call. Use "
          "<code>persistent=True</code> for more, then <code>del tape</code>"],
         ["What is watched", "<code>tf.Variable</code> automatically; anything "
          "else needs <code>tape.watch()</code>"],
         ["Nesting", "Nest tapes to get second derivatives"],
         ["Stopping gradients", "<code>tf.stop_gradient(x)</code> makes the "
          "forward value pass through but blocks the backward flow"],
         ["Memory", "The tape stores every intermediate activation — this is why "
          "training memory scales with depth"]],
    )

    sub("The training loop, written out")

    md(
        """
```python
for epoch in range(1, n_epochs + 1):
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train, y_train)

        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)      # training=True!
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)  # + internal losses

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))

        for metric in metrics:
            metric.update_state(y_batch, y_pred)

    for metric in metrics:
        metric.reset_state()
```
        """
    )

    pitfall(
        "Five things model.fit does that your loop must not forget",
        "<b>(1) <code>training=True</code></b> — otherwise batch norm uses its "
        "moving statistics and dropout is disabled during training.<br>"
        "<b>(2) <code>model.losses</code></b> — the internal losses from "
        "<code>add_loss</code> (§12.7) and every "
        "<code>kernel_regularizer</code>. Forget these and your regularisation "
        "silently does nothing.<br>"
        "<b>(3) Constraints</b> — applied after the optimiser step, not by it.<br>"
        "<b>(4) <code>reset_state()</code></b> on every metric at the end of each "
        "epoch, or your metrics accumulate across the whole run.<br>"
        "<b>(5) <code>@tf.function</code></b> on the training step, or the loop "
        "runs eagerly and is several times slower than "
        "<code>fit</code>.",
    )

    sub("Higher-order derivatives and Jacobians")

    table(
        ["What you want", "How"],
        [["Gradient of a scalar", "<code>tape.gradient(y, x)</code>"],
         ["Second derivative / Hessian",
          "Nested tapes: <code>with tape2: with tape1: …</code>"],
         ["Jacobian of a vector output",
          "<code>tape.jacobian(y, x)</code> — much faster than looping"],
         ["Only some paths", "<code>tf.stop_gradient(...)</code>"],
         ["Gradient with respect to a constant",
          "<code>tape.watch(c)</code> first"]],
    )

    anim_header("The tape recording forward and replaying backward")

    ops = [("x", "input", C["accent"]),
           ("a = x²", "square", SEQ[0]),
           ("b = 3a", "scale", SEQ[1]),
           ("c = sin(b)", "sin", SEQ[2]),
           ("L = c²", "square", SEQ[3])]
    grads = ["∂L/∂L = 1",
             "∂L/∂c = 2c",
             "∂L/∂b = 2c·cos(b)",
             "∂L/∂a = 2c·cos(b)·3",
             "∂L/∂x = 2c·cos(b)·3·2x"]

    frames = []
    for k in range(1, 2 * len(ops) + 1):
        forward = min(k, len(ops))
        backward = max(0, k - len(ops))
        shapes, ann = [], []
        for i, (nm, kind, col) in enumerate(ops):
            fwd_on = i < forward
            bwd_on = backward > 0 and i >= len(ops) - backward
            shapes.append(go.Scatter(
                x=[i * 1.5, i * 1.5 + 1.15, i * 1.5 + 1.15, i * 1.5, i * 1.5],
                y=[-.35, -.35, .35, .35, -.35], fill="toself",
                fillcolor=(alpha(C["danger"], .85) if bwd_on else
                           alpha(col, .85) if fwd_on else alpha(C["line"], .3)),
                line=dict(color="#fff", width=2), hoverinfo="skip",
                showlegend=False))
            ann.append(dict(x=i * 1.5 + .575, y=0, text=nm, showarrow=False,
                            font=dict(size=11,
                                      color="#fff" if (fwd_on or bwd_on)
                                      else C["muted"])))
        if backward:
            ann.append(dict(x=3.5, y=-1.1,
                            text=f"<b>{grads[backward-1]}</b>", showarrow=False,
                            font=dict(size=13, color=C["danger"])))
        title = ("FORWARD pass — the tape records each op"
                 if k <= len(ops) else
                 "BACKWARD pass — the tape replays them in reverse")
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann, title=title)))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=340, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.4, 7.5]),
                    yaxis=dict(visible=False, range=[-1.6, .8]),
                    annotations=list(frames[0].layout.annotations),
                    title=frames[0].layout.title.text)
    anim.animate(f, frames, duration=nav.anim_ms(900), slider_prefix="step ")
    figure(f, "Reverse-mode autodiff: one forward pass records, one backward pass "
              "applies the chain rule. Appendix B derives why this is optimal for "
              "scalar outputs.")

    code_lab(
        "GradientTape from first principles, then a complete training loop",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE BASICS ============================================
def f(w1, w2):
    return 3*w1**2 + 2*w1*w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)
g = tape.gradient(z, [w1, w2])
print("=== GradientTape ===")
print(f"f(5,3) = {z.numpy()}")
print(f"gradients = {[float(x) for x in g]}")
print(f"analytic  = [6*w1 + 2*w2, 2*w1] = [{6*5+2*3}, {2*5}]")

# --- the tape is CONSUMED by the first gradient() call ----------------
with tf.GradientTape() as tape:
    z = f(w1, w2)
_ = tape.gradient(z, w1)
try:
    tape.gradient(z, w2)
except RuntimeError as e:
    print(f"\\nsecond gradient() on the same tape -> RuntimeError")
    print(f"  {str(e)[:80]}")

with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)
print(f"persistent=True: dz/dw1 = {tape.gradient(z, w1).numpy()}, "
      f"dz/dw2 = {tape.gradient(z, w2).numpy()}")
del tape                                    # free it explicitly

# ============ 2. WATCHING NON-VARIABLES ================================
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
    z = f(c1, c2)
print(f"\\ngradient w.r.t. constants: {tape.gradient(z, [c1, c2])}   <- None!")
with tf.GradientTape() as tape:
    tape.watch(c1); tape.watch(c2)
    z = f(c1, c2)
print(f"after tape.watch()       : "
      f"{[float(x) for x in tape.gradient(z, [c1, c2])]}")

# ============ 3. STOPPING GRADIENTS ====================================
def f_stop(w1, w2):
    return 3*w1**2 + tf.stop_gradient(2*w1*w2)
with tf.GradientTape() as tape:
    z = f_stop(w1, w2)
print(f"\\nwith tf.stop_gradient on the second term: "
      f"{[None if g is None else float(g) for g in tape.gradient(z, [w1, w2])]}")
print(f"  the forward VALUE is unchanged ({f_stop(w1,w2).numpy()} vs "
      f"{f(w1,w2).numpy()}) but no gradient flows through it")

# ============ 4. HIGHER-ORDER DERIVATIVES ==============================
print("\\n=== second derivatives ===")
x = tf.Variable(3.0)
with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        y = x**4
    dy = t1.gradient(y, x)
d2y = t2.gradient(dy, x)
print(f"y = x^4 at x=3 : y={y.numpy()}  dy/dx={dy.numpy()} (4x^3={4*27})  "
      f"d2y/dx2={d2y.numpy()} (12x^2={12*9})")

# --- a full Hessian ---------------------------------------------------
v = tf.Variable([1., 2., 3.])
with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        s = tf.reduce_sum(v**3) + v[0]*v[1]*v[2]
    grad = t1.gradient(s, v)
hess = t2.jacobian(grad, v)
print(f"\\nHessian of sum(v^3) + v0*v1*v2 at v=[1,2,3]:")
print(hess.numpy().round(3))

# ============ 5. JACOBIANS =============================================
print("\\n=== jacobian of a vector output ===")
u = tf.Variable([[1., 2.], [3., 4.]])
with tf.GradientTape() as tape:
    out = tf.stack([tf.reduce_sum(u**2), tf.reduce_prod(u)])
J = tape.jacobian(out, u)
print(f"output shape {tuple(out.shape)}, input shape {tuple(u.shape)} "
      f"-> jacobian shape {tuple(J.shape)}")
print(f"d(sum u^2)/du = \\n{J[0].numpy()}   (should be 2u)")

# ============ 6. A COMPLETE CUSTOM TRAINING LOOP =======================
print("\\n" + "="*62)
print("A CUSTOM TRAINING LOOP -- and everything fit() does for you")
print("="*62)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data.astype("float32")/16., d.target,
                                      test_size=.25, stratify=d.target,
                                      random_state=42)
ytr = ytr.astype("int32"); yte = yte.astype("int32")

def build():
    return keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        keras.layers.Dense(10, activation="softmax")])

def random_batch(X, y, batch_size=32, rng=np.random.default_rng(0)):
    idx = rng.integers(len(X), size=batch_size)
    return X[idx], y[idx]

def train_custom(model, n_epochs=20, batch_size=32, compiled=True):
    optimizer  = keras.optimizers.Adam(1e-3)
    loss_fn    = keras.losses.SparseCategoricalCrossentropy()
    mean_loss  = keras.metrics.Mean()
    accuracy   = keras.metrics.SparseCategoricalAccuracy()
    n_steps    = len(Xtr) // batch_size
    rng = np.random.default_rng(0)

    def step(Xb, yb):
        with tf.GradientTape() as tape:
            y_pred = model(Xb, training=True)              # (1) training=True
            main = tf.reduce_mean(loss_fn(yb, y_pred))
            loss = tf.add_n([main] + model.losses)         # (2) internal losses
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, y_pred

    step_fn = tf.function(step) if compiled else step      # (5) compile it

    history = []
    for epoch in range(n_epochs):
        for _ in range(n_steps):
            Xb, yb = random_batch(Xtr, ytr, batch_size, rng)
            loss, y_pred = step_fn(tf.constant(Xb), tf.constant(yb))
            for v in model.variables:                      # (3) constraints
                if getattr(v, "constraint", None) is not None:
                    v.assign(v.constraint(v))
            mean_loss.update_state(loss)
            accuracy.update_state(yb, y_pred)
        history.append((float(mean_loss.result()), float(accuracy.result())))
        for m in (mean_loss, accuracy):
            m.reset_state()                                # (4) reset each epoch
    return history

tf.random.set_seed(0)
m_custom = build()
t0 = time.perf_counter()
hist = train_custom(m_custom, compiled=True)
t_custom = time.perf_counter() - t0
acc_custom = keras.metrics.SparseCategoricalAccuracy()
acc_custom.update_state(yte, m_custom(Xte, training=False))

tf.random.set_seed(0)
m_fit = build()
m_fit.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
t0 = time.perf_counter()
m_fit.fit(Xtr, ytr, epochs=20, batch_size=32, verbose=0)
t_fit = time.perf_counter() - t0

print(f"\\n{'approach':<28}{'time':>9}{'test accuracy':>16}")
print(f"{'custom loop (@tf.function)':<28}{t_custom:>8.2f}s"
      f"{float(acc_custom.result()):>16.4f}")
print(f"{'model.fit':<28}{t_fit:>8.2f}s"
      f"{m_fit.evaluate(Xte, yte, verbose=0)[1]:>16.4f}")

# --- and WITHOUT @tf.function ----------------------------------------
tf.random.set_seed(0)
m_slow = build()
t0 = time.perf_counter()
train_custom(m_slow, n_epochs=5, compiled=False)
t_slow = (time.perf_counter() - t0) * 4          # scale to 20 epochs
print(f"{'custom loop, NO tf.function':<28}{t_slow:>8.2f}s{'(estimated)':>16}")
print(f"\\n@tf.function speeds the loop up by ~{t_slow/t_custom:.1f}x")
print("model.fit does this for you -- which is why a naive loop is slower.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=[h[0] for h in hist], mode="lines", name="loss",
                line=dict(color=C["danger"], width=2.5))
fig.add_scatter(y=[h[1] for h in hist], mode="lines", name="accuracy",
                line=dict(color=C["success"], width=2.5), yaxis="y2")
fig.update_layout(height=380, xaxis_title="epoch",
                  yaxis=dict(title="loss"),
                  yaxis2=dict(title="accuracy", overlaying="y", side="right"),
                  title="The hand-written training loop")
''',
        key="ch12_autodiff",
    )

    keypoints([
        "<code>GradientTape</code> records the forward pass and replays it "
        "backwards; variables are watched automatically.",
        "A tape is consumed by one <code>gradient()</code> call unless "
        "<code>persistent=True</code>.",
        "Nest tapes for second derivatives; <code>tape.jacobian</code> for vector "
        "outputs.",
        "A custom loop must handle <code>training=True</code>, "
        "<code>model.losses</code>, constraints, metric resets and "
        "<code>@tf.function</code>.",
        "Write a custom loop only when the training <i>procedure</i> genuinely "
        "differs.",
    ])


# ==========================================================================
def s_12_9():
    section("12.9", "TensorFlow Functions, Graphs & Exercises")

    lead(
        "<code>@tf.function</code> traces your Python into a graph. Understanding "
        "<i>when</i> it traces — and what Python code therefore runs only once — "
        "is the difference between a fast model and a mysterious bug."
    )

    sub("Tracing and retracing")

    md(
        """
The first call **traces** the function: TensorFlow runs the Python body in a
symbolic mode where every tensor is a placeholder, recording the ops into a graph.
Subsequent calls **reuse** that graph — the Python body does not run again.

A **new trace** happens whenever the input *signature* changes:
        """
    )

    table(
        ["Change", "Retraces?", "Why"],
        [["Same shapes and dtypes", "❌ reuses the graph", "Signature matches"],
         ["Different <b>shape</b>", "✅",
          "Shapes are part of the signature — one graph per shape"],
         ["Different <b>dtype</b>", "✅", "Likewise"],
         ["A <b>Python</b> argument with a new value", "<b>✅ every time</b>",
          "Python values are baked into the graph as constants"],
         ["A <b>tensor</b> argument with a new value", "❌",
          "Tensors are placeholders, not constants"]],
    )

    pitfall(
        "Passing Python numbers causes a retrace per value",
        "<code>@tf.function def f(x, n): …</code> called as "
        "<code>f(t, 1)</code>, <code>f(t, 2)</code>, <code>f(t, 3)</code> builds "
        "<b>three graphs</b> — each with <code>n</code> hard-coded. In a training "
        "loop passing the step number, you would build one graph per step and the "
        "program would grind to a halt while consuming unbounded memory. "
        "<b>Pass tensors</b>: <code>f(t, tf.constant(1))</code>. Use "
        "<code>input_signature=</code> to make retracing impossible by "
        "construction.",
    )

    sub("The tf.function rules")

    table(
        ["Rule", "Detail"],
        [["Use TF operations, not NumPy",
          "<code>np.sum(x)</code> is evaluated <b>once at trace time</b> and "
          "frozen as a constant; <code>tf.reduce_sum(x)</code> becomes a graph "
          "node"],
         ["Side effects run only during tracing",
          "<code>print()</code>, logging, appending to a Python list — all happen "
          "once. Use <code>tf.print</code> for runtime output"],
         ["<code>for</code> over a Python range is unrolled",
          "Iterating <code>range(1000)</code> creates 1000 copies of the body in "
          "the graph. Use <code>tf.range</code> to get a real loop"],
         ["Create variables only on the first call",
          "Otherwise every trace creates new ones. Create them in "
          "<code>build()</code> or <code>__init__</code>"],
         ["Python <code>if</code> on a tensor fails",
          "Use <code>tf.cond</code>, or <code>tf.where</code> for element-wise "
          "selection"],
         ["Source code must be available",
          "TensorFlow reads it to autograph the control flow — so no "
          "<code>exec</code>'d strings, no interactive-shell-only definitions"]],
    )

    idea(
        "AutoGraph rewrites your control flow",
        "TensorFlow inspects the function's source and rewrites Python "
        "<code>for</code>/<code>while</code>/<code>if</code> that involve tensors "
        "into <code>tf.while_loop</code> and <code>tf.cond</code> nodes. "
        "<code>tf.autograph.to_code(f.python_function)</code> shows you the "
        "generated source — worth reading once, because it explains exactly which "
        "of your constructs became graph ops and which were baked in as "
        "constants.",
    )

    anim_header("Tracing: how many graphs does this build?")

    calls = [
        ("f(tf.constant(1.0))", "float32 scalar", True, 1),
        ("f(tf.constant(2.0))", "float32 scalar — SAME signature", False, 1),
        ("f(tf.constant(3.0))", "float32 scalar — same again", False, 1),
        ("f(tf.constant([1., 2.]))", "float32 shape (2,) — NEW", True, 2),
        ("f(tf.constant(1))", "int32 scalar — NEW dtype", True, 3),
        ("f(1.0)", "PYTHON float — new constant", True, 4),
        ("f(2.0)", "PYTHON float — another constant", True, 5),
        ("f(3.0)", "PYTHON float — and another", True, 6),
    ]
    frames = []
    for k in range(1, len(calls) + 1):
        xs = list(range(1, k + 1))
        ys = [c[3] for c in calls[:k]]
        cols = [C["danger"] if c[2] else C["success"] for c in calls[:k]]
        frames.append(go.Frame(name=str(k), data=[
            go.Bar(x=xs, y=ys, marker=dict(color=cols),
                   text=[c[0] for c in calls[:k]], textposition="outside",
                   textfont=dict(size=9)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"call {k}: {calls[k-1][0]}   ·   {calls[k-1][1]}   ·   "
            + ("RETRACE" if calls[k - 1][2] else "reuses the graph")
            + f"   ·   {calls[k-1][3]} graph(s) so far",
            color=C["danger"] if calls[k - 1][2] else C["success"])])))

    f = go.Figure(data=[go.Bar(x=[1], y=[1], marker=dict(color=C["danger"]))])
    f.update_layout(height=430, xaxis_title="call number",
                    yaxis_title="cumulative number of concrete graphs",
                    yaxis=dict(range=[0, 7]),
                    title="Red = a new trace, green = the graph was reused")
    anim.animate(f, frames, duration=nav.anim_ms(900), slider_prefix="call ")
    figure(f, "The last three calls pass a Python float, so each value builds its "
              "own graph. In a loop that is unbounded memory growth.")

    code_lab(
        "Tracing, retracing, AutoGraph, and every rule broken on purpose",
        '''import numpy as np, time
import tensorflow as tf

# ============ 1. WHEN DOES IT RETRACE? =================================
trace_count = {"n": 0}

@tf.function
def f(x):
    trace_count["n"] += 1
    print(f"    [TRACING #{trace_count['n']}] x = {x}")   # only while tracing
    return x ** 2

print("=== tracing ===")
for call in ["tf.constant(1.0)", "tf.constant(2.0)", "tf.constant([1., 2.])",
             "tf.constant(1)", "1.0", "2.0"]:
    print(f"  calling f({call}):")
    f(eval(call))
print(f"\\ntotal graphs built: {trace_count['n']}")
print(f"concrete functions : {len(f._list_all_concrete_functions())}")
print("\\nPython arguments bake into the graph -> one graph per VALUE.")

# --- input_signature makes retracing impossible ----------------------
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def g(x):
    print("    [TRACING g]")
    return tf.reduce_sum(x)
print("\\n=== input_signature ===")
for arr in [[1.], [1., 2.], [1., 2., 3.]]:
    g(tf.constant(arr))
print("  traced ONCE for any 1-D float32 input")
try:
    g(tf.constant([[1., 2.]]))
except (ValueError, TypeError) as e:
    print(f"  wrong shape -> {type(e).__name__} (caught at the boundary)")

# ============ 2. NUMPY IS FROZEN AT TRACE TIME =========================
print("\\n=== NumPy vs TF operations inside a tf.function ===")
@tf.function
def uses_numpy(x):
    return x + np.random.rand()          # evaluated ONCE, then frozen

@tf.function
def uses_tf(x):
    return x + tf.random.uniform(())     # a real graph node

print(f"  numpy version: {[float(uses_numpy(tf.constant(0.))) for _ in range(4)]}")
print(f"                 ^ all identical -- the value was baked in")
print(f"  tf version   : {[round(float(uses_tf(tf.constant(0.))), 4) for _ in range(4)]}")
print(f"                 ^ different every call")

# ============ 3. print vs tf.print =====================================
print("\\n=== side effects ===")
@tf.function
def logger(x):
    print("    python print  -- trace time only")
    tf.print("    tf.print      -- every call")
    return x
print("  first call:");  logger(tf.constant(1.))
print("  second call:"); logger(tf.constant(2.))

# ============ 4. PYTHON LOOPS ARE UNROLLED =============================
print("\\n=== loops ===")
@tf.function
def python_loop(x):
    for i in range(20):                  # UNROLLED: 20 copies in the graph
        x = x + 1
    return x

@tf.function
def tf_loop(x, n):                       # a real tf.while_loop node
    return tf.while_loop(lambda i, x: i < n,
                         lambda i, x: (i + 1, x + 1),
                         [tf.constant(0), x])[1]

pl = python_loop.get_concrete_function(tf.constant(0.))
tl = tf_loop.get_concrete_function(tf.constant(0.), tf.constant(20))
print(f"  python range(20) -> {len(pl.graph.get_operations())} graph ops")
print(f"  tf.range(n)      -> {len(tl.graph.get_operations())} graph ops")
print(f"  and tf_loop works for ANY n without retracing:")
for n in [5, 50, 500]:
    print(f"    n={n:>3} -> {float(tf_loop(tf.constant(0.), tf.constant(n)))}")

# ============ 5. AUTOGRAPH ============================================
print("\\n=== what AutoGraph generates ===")
@tf.function
def conditional(x):                      # a TENSOR condition -> tf.cond
    return tf.cond(tf.reduce_sum(x) > 0, lambda: x * 2, lambda: x / 2)
print(f"  conditional([1., 2.])  = {conditional(tf.constant([1., 2.])).numpy()}")
print(f"  conditional([-1., -2.]) = {conditional(tf.constant([-1., -2.])).numpy()}")
print("  In a normal .py file you could write a plain Python `if` here and")
print("  AutoGraph would rewrite it into exactly this tf.cond. It cannot do so")
print("  in this lab because the source is a string -- rule 6 of section 12.9.")
print("  Inspect the rewrite yourself with:")
print("    tf.autograph.to_code(your_function.python_function)")

# ============ 6. THE VARIABLE-CREATION RULE ============================
print("\\n=== variables must be created only once ===")
@tf.function
def bad():
    v = tf.Variable(1.0)                 # a new variable on every trace
    return v + 1
try:
    bad(); bad()
except ValueError as e:
    print(f"  creating a Variable inside tf.function -> ValueError")
    print(f"    {str(e).splitlines()[0][:80]}")

counter = tf.Variable(0)
@tf.function
def good():
    counter.assign_add(1)                # created OUTSIDE, assigned inside
    return counter
good(); good(); good()
print(f"  variable created outside, assigned inside: counter = {counter.numpy()}")

# ============ 7. THE SPEED-UP ==========================================
print("\\n=== how much does it actually buy? ===")
def workload(x, W1, W2, W3):
    h = tf.nn.relu(x @ W1)
    h = tf.nn.relu(h @ W2)
    h = tf.nn.gelu(h @ W3)
    return tf.reduce_mean(tf.square(h)) + tf.reduce_sum(tf.abs(h)) * 1e-4

fast = tf.function(workload)
rng = tf.random.Generator.from_seed(0)
sizes = [(64, 64), (256, 256), (1024, 512)]
print(f"{'shape':>14}{'eager':>12}{'tf.function':>15}{'speedup':>10}")
for m, n in sizes:
    x  = rng.normal((m, n))
    W1 = rng.normal((n, n)); W2 = rng.normal((n, n)); W3 = rng.normal((n, n))
    fast(x, W1, W2, W3)                                       # warm-up
    N = 100
    t0 = time.perf_counter()
    for _ in range(N): workload(x, W1, W2, W3)
    te = (time.perf_counter()-t0)/N*1000
    t0 = time.perf_counter()
    for _ in range(N): fast(x, W1, W2, W3)
    tg = (time.perf_counter()-t0)/N*1000
    print(f"{f'{m}x{n}':>14}{te:>11.3f}ms{tg:>14.3f}ms{te/tg:>9.2f}x")
print("\\nThe gain is largest when there are many small ops to fuse.")

# ============ 8. DEBUGGING =============================================
print("\\n=== turning tracing off to debug ===")
tf.config.run_functions_eagerly(True)
logger(tf.constant(9.))
tf.config.run_functions_eagerly(False)
print("  (remember to turn it back off -- it is much slower)")
''',
        key="ch12_tffunction",
    )

    rule()

    sub("Exercises")

    exercise(
        1, "How would you describe TensorFlow in a short sentence? What are its "
        "main features? Can you name other popular deep learning libraries?",
        "**TensorFlow is an open-source library for numerical computation, "
        "particularly well suited and fine-tuned for large-scale machine "
        "learning.** Its core is similar to NumPy but adds GPU support, "
        "distributed computing, computation-graph analysis and optimisation "
        "(portable graphs that can be trained in Python and run elsewhere), "
        "automatic differentiation, and an excellent optimiser suite.\n\n"
        "Other features: image and signal processing ops (`tf.image`, "
        "`tf.signal`), input pipelines (`tf.data`), deployment tooling "
        "(TF Serving, TF Lite, TF.js).\n\n"
        "**Other popular libraries:** PyTorch, JAX, MXNet, Caffe2, Theano "
        "(discontinued), Microsoft Cognitive Toolkit (discontinued), and — at a "
        "higher level — Keras (which now runs on all three of TensorFlow, JAX and "
        "PyTorch).")

    exercise(
        2, "Is TensorFlow a drop-in replacement for NumPy? What are the main "
        "differences between the two?",
        "**No.** They are similar but differ in several important ways:\n\n"
        "* **Function names differ.** `tf.reduce_sum()` vs `np.sum()`, "
        "`tf.reduce_mean()` vs `np.mean()`, and so on. The `reduce_` prefix is a "
        "historical artefact of the original distributed implementation.\n"
        "* **Some functions behave differently.** `tf.transpose(t)` **creates a "
        "copy** with the axes permuted; NumPy's `a.T` is a transposed **view** "
        "with no copy at all.\n"
        "* **Tensors are immutable.** You cannot do `t[0] = 5`. Use "
        "`tf.Variable` for mutable state.\n"
        "* **No automatic type conversion.** NumPy happily mixes int and float, "
        "and float32 with float64; TensorFlow raises an error.\n"
        "* **NumPy defaults to float64**, TensorFlow to float32 — a common source "
        "of confusion at the boundary.\n"
        "* TensorFlow supports GPUs, automatic differentiation, and graph "
        "compilation; NumPy does none of these.")

    exercise(
        3, "Do you get the same result with `tf.range(10)` and "
        "`tf.constant(np.arange(10))`?",
        "**The values are the same but the dtypes differ.**\n\n"
        "`tf.range(10)` produces a tensor of dtype **`int32`** (TensorFlow's "
        "default integer type).\n\n"
        "`np.arange(10)` produces a NumPy array of dtype **`int64`** on most "
        "platforms, and `tf.constant` preserves that — so you get an **`int64`** "
        "tensor.\n\n"
        "This matters because TensorFlow will refuse to combine the two: "
        "`tf.range(10) + tf.constant(np.arange(10))` raises "
        "`InvalidArgumentError`. Fix it with `tf.cast` or by specifying the dtype "
        "explicitly.")

    exercise(
        4, "Can you name six other data structures available in TensorFlow, beyond "
        "regular tensors?",
        "**(1) `tf.SparseTensor`** — efficient representation of mostly-zero "
        "tensors (indices, values, dense shape). `tf.sparse` has the operations.\n\n"
        "**(2) `tf.RaggedTensor`** — lists of tensors of *different lengths*, "
        "without padding. `tf.ragged` has the operations.\n\n"
        "**(3) `tf.TensorArray`** — a dynamically-sized list of tensors, needed "
        "for accumulating results inside a graph-mode loop.\n\n"
        "**(4) String tensors** — `tf.string` is a first-class dtype holding byte "
        "strings; `tf.strings` has the operations, including Unicode-aware ones.\n\n"
        "**(5) Sets** — represented as regular or sparse tensors; `tf.sets` "
        "provides union, intersection, difference.\n\n"
        "**(6) Queues** — `tf.queue` provides FIFO, priority, shuffling and "
        "padding queues for multi-threaded pipelines.\n\n"
        "(Also: `tf.lookup` tables, and `tf.Variable` itself.)")

    exercise(
        5, "You can define a custom loss function by writing a function or by "
        "subclassing the `keras.losses.Loss` class. When would you use each "
        "option?",
        "**Write a function** when the loss has no hyperparameters, or when you "
        "do not need to save and reload the model with its hyperparameters "
        "intact. It is the simplest option and works fine for most cases.\n\n"
        "**Subclass `keras.losses.Loss`** when the loss has hyperparameters that "
        "must be **saved along with the model**. Implementing `get_config()` lets "
        "Keras write those hyperparameters into the file and restore them on "
        "load.\n\n"
        "The trap in between is a **closure** — a function returning a function "
        "that captures a hyperparameter. It works for training but the "
        "hyperparameter is **silently lost** on save/load, reverting to whatever "
        "default you pass to `custom_objects` (§12.3's lab demonstrates this).")

    exercise(
        6, "Similarly, you can define a custom metric in a function or as a "
        "subclass of `keras.metrics.Metric`. When would you use each option?",
        "**Write a function** when the metric is a **simple mean over "
        "instances** — like MAE or accuracy. Keras will call it on each batch and "
        "average the results, which is correct for means.\n\n"
        "**Subclass `keras.metrics.Metric`** when either:\n\n"
        "1. The metric **cannot be computed by averaging batch results** — "
        "precision, recall, $F_1$, AUC, IoU are all ratios of pooled counts, and "
        "averaging them gives the wrong answer (§12.5 shows an error of 27 "
        "percentage points).\n"
        "2. You need hyperparameters saved with the model, as with a custom loss.\n\n"
        "The subclass implements `update_state` (accumulate), `result` (compute) "
        "and inherits `reset_state` (zero at each epoch).")

    exercise(
        7, "When should you create a custom layer versus a custom model?",
        "The distinction is essentially about **internal versus top-level "
        "structure**:\n\n"
        "* Create a **custom layer** for any reusable building block — anything "
        "you would place inside a larger model. A residual block, an attention "
        "head, a specialised normalisation.\n"
        "* Create a **custom model** for the **object containing the layers** — "
        "the thing you call `fit`, `save`, `predict` and `evaluate` on.\n\n"
        "The practical rule: if it should be composable inside something else, "
        "make it a `Layer`. If it is the outermost object, make it a `Model`. "
        "(Note `keras.Model` subclasses `keras.Layer`, so a Model *is* a Layer "
        "with extra methods — which is why you can nest models.)")

    exercise(
        8, "What are some use cases that require writing your own custom training "
        "loop?",
        "In general: **only when `fit()` is not flexible enough**, because a "
        "custom loop is longer, more error-prone, and harder to maintain.\n\n"
        "Genuine cases:\n\n"
        "* **Multiple optimisers for different parts of the network** — the wide "
        "& deep paper uses one optimiser for the wide path and another for the "
        "deep path.\n"
        "* **GANs** (Chapter 17) — the generator and discriminator are trained "
        "alternately with different objectives, and the discriminator's loss "
        "depends on the generator's current output.\n"
        "* **Reinforcement learning** (Chapter 18) — the data is generated by the "
        "policy being trained, so there is no fixed dataset to iterate over.\n"
        "* **Meta-learning / MAML** — gradients through an inner optimisation "
        "loop.\n"
        "* **Complex gradient manipulation** — gradient reversal layers, "
        "adversarial training, custom clipping schemes.\n\n"
        "For debugging, `model.compile(..., run_eagerly=True)` is usually a "
        "better first step than rewriting the loop.")

    exercise(
        9, "Can custom Keras components contain arbitrary Python code, or must "
        "they be convertible to TF functions?",
        "**They should be convertible to TF functions**, which means using TF "
        "operations rather than arbitrary Python for anything that touches "
        "tensors, and following the rules of §12.9.\n\n"
        "If you absolutely need arbitrary Python, you have two escape hatches:\n\n"
        "1. Wrap it in a **`tf.py_function`**, which calls back into the Python "
        "interpreter. This works, but it degrades performance badly and it limits "
        "portability — the model can then only run where Python and your "
        "libraries are available, ruling out TF Lite and TF.js.\n"
        "2. Set **`dynamic=True`** when creating the custom layer, or "
        "**`run_eagerly=True`** when calling `model.compile()`. Then Keras will "
        "not attempt to convert it, at the cost of speed.")

    exercise(
        10, "What are the main rules to respect if you want a function to be "
        "convertible to a TF function?",
        "**(1)** Use **TensorFlow operations**, not NumPy or standard-library "
        "equivalents — `tf.reduce_sum` not `np.sum`. NumPy calls are evaluated "
        "once at trace time and frozen as constants.\n\n"
        "**(2)** Avoid **side effects** (printing, logging, appending to lists, "
        "writing files) — they run only during tracing. Use `tf.print` for "
        "runtime output.\n\n"
        "**(3)** **Wrap external calls** in `tf.py_function` if you must use "
        "them, accepting the performance and portability cost.\n\n"
        "**(4)** Call other functions freely — they will be traced too — but "
        "they must follow the same rules.\n\n"
        "**(5)** Create **`tf.Variable`s only on the first call**, ideally in "
        "`build()` or `__init__`, never in the traced body.\n\n"
        "**(6)** Keep the **source code available** — TensorFlow reads it to "
        "autograph control flow, so functions defined in an interactive shell or "
        "via `exec` may fail.\n\n"
        "**(7)** Prefer **vectorised operations to Python loops**; a Python `for` "
        "over `range()` is unrolled into the graph, whereas `tf.range` produces a "
        "real loop node.")

    exercise(
        11, "When would you need to create a dynamic Keras model? How do you do "
        "that? Why not make all your models dynamic?",
        "**When you need it:** for **debugging**, since a dynamic model does not "
        "compile any custom code, so Python `print` and breakpoints work, "
        "exceptions carry useful tracebacks, and you can step through with a "
        "debugger. Also when the model genuinely requires Python control flow "
        "that AutoGraph cannot convert, or arbitrary non-TF library calls.\n\n"
        "**How:** set `dynamic=True` when creating a custom layer or model. "
        "Alternatively, `model.compile(..., run_eagerly=True)` makes an entire "
        "model run eagerly without changing the layers.\n\n"
        "**Why not always:** it **slows training and inference significantly** — "
        "no graph fusion, no constant folding, no parallel scheduling — and it "
        "removes the ability to **export the model** to any environment without "
        "Python, so TF Serving, TF Lite and TensorFlow.js are all off the table.")

    exercise(
        12, "Implement a custom layer that performs layer normalization. "
        "(a) The `build()` method should define two trainable weights α and β, "
        "both of shape `input_shape[-1:]` and data type `tf.float32`. α should be "
        "initialized with 1s, and β with 0s. (b) The `call()` method should "
        "compute the mean μ and standard deviation σ of each instance's features. "
        "You can use `tf.nn.moments(inputs, axes=-1, keepdims=True)`, which "
        "returns the mean μ and the variance σ². Then the function should compute "
        "and return `α ⊗ (X − μ)/(σ + ε) + β`, where ⊗ is element-wise "
        "multiplication and ε is a smoothing term. (c) Ensure that your custom "
        "layer produces the same output as `keras.layers.LayerNormalization`.",
        "**Layer normalisation** (Ba et al., 2016) normalises across the "
        "**features of each instance** rather than across the **batch**. That one "
        "difference makes it independent of batch size, which is why it works "
        "where batch norm fails: recurrent networks (Chapter 15), Transformers "
        "(Chapter 16), and any setting with tiny batches (§11.3).\n\n"
        "$\\mathrm{LN}(\\mathbf{x}) = \\boldsymbol\\alpha \\otimes "
        "\\dfrac{\\mathbf{x} - \\mu}{\\sigma + \\varepsilon} + \\boldsymbol\\beta$, "
        "where $\\mu$ and $\\sigma$ are computed over the **last axis**.\n\n"
        "Two details worth noting: Keras divides by $\\sigma + \\varepsilon$ "
        "rather than $\\sqrt{\\sigma^2 + \\varepsilon}$ in some versions, so match "
        "whichever your comparison target uses; and the weights have shape "
        "`input_shape[-1:]` — one α and one β per feature, shared across "
        "instances.",
        code='''class LayerNormalization(keras.layers.Layer):
    def __init__(self, eps=0.001, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, batch_input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=batch_input_shape[-1:],
            initializer="ones")
        self.beta = self.add_weight(
            name="beta", shape=batch_input_shape[-1:],
            initializer="zeros")

    def call(self, X):
        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / (tf.sqrt(variance) + self.eps) + self.beta

    def get_config(self):
        return {**super().get_config(), "eps": self.eps}

# (c) verify against Keras
X = tf.random.normal((100, 20))
custom = LayerNormalization()
keras_ln = keras.layers.LayerNormalization()
print(tf.reduce_mean(keras.losses.mean_absolute_error(
    keras_ln(X), custom(X))))          # should be ~1e-8''')

    exercise(
        13, "Train a model using a custom training loop to tackle the Fashion "
        "MNIST dataset. (a) Display the epoch, iteration, mean training loss, and "
        "mean accuracy over each epoch (updated at each iteration), as well as the "
        "validation loss and accuracy at the end of each epoch. (b) Try using a "
        "different optimizer with a different learning rate for the upper layers "
        "and the lower layers.",
        "**(a)** The structure is the loop of §12.8. The one thing worth getting "
        "right is the display: use `keras.metrics.Mean` for the running loss and "
        "`keras.metrics.SparseCategoricalAccuracy` for the running accuracy, "
        "update them every iteration, print with `\\r` for an in-place status "
        "line, and **`reset_state()` at the end of every epoch**.\n\n"
        "**(b)** Two optimisers is exactly the case where a custom loop earns its "
        "keep. Split the model into a `lower_layers` model and an `upper_layers` "
        "model, keep one optimiser for each, and inside the tape compute the loss "
        "once but call `tape.gradient` twice — with `persistent=True`, or by "
        "passing a list of both variable sets and slicing the result.\n\n"
        "Typical setting: a **small** learning rate for the lower layers (they "
        "hold general features you do not want to disturb — the same argument as "
        "fine-tuning in §11.5) and a **larger** one for the upper layers. This is "
        "sometimes called *discriminative fine-tuning*.",
        code='''lower_layers = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="relu"),
])
upper_layers = keras.Sequential([
    keras.layers.Dense(10, activation="softmax"),
])
model = keras.Sequential([lower_layers, upper_layers])

lower_optimizer = keras.optimizers.SGD(learning_rate=1e-4)   # small
upper_optimizer = keras.optimizers.Nadam(learning_rate=1e-3) # larger

for epoch in range(1, n_epochs + 1):
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train, y_train)
        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)

        for layers, optimizer in ((lower_layers, lower_optimizer),
                                  (upper_layers, upper_optimizer)):
            gradients = tape.gradient(loss, layers.trainable_variables)
            optimizer.apply_gradients(zip(gradients, layers.trainable_variables))
        del tape                       # persistent tapes must be freed''')

    rule()

    keypoints([
        "<code>@tf.function</code> traces once per <b>input signature</b>; Python "
        "arguments cause a trace per value.",
        "NumPy calls and <code>print</code> inside a traced function run "
        "<b>once</b> and freeze.",
        "Python <code>for range()</code> is unrolled; <code>tf.range</code> gives "
        "a real loop.",
        "Create variables outside the traced body, or in "
        "<code>build()</code>/<code>__init__</code>.",
        "<code>run_functions_eagerly(True)</code> for debugging; turn it back off.",
    ], title="Chapter 12 in five lines")

    refs([
        ("Abadi et al. — *TensorFlow: A System for Large-Scale Machine Learning*",
         "OSDI 2016"),
        ("Baydin et al. — *Automatic Differentiation in Machine Learning: A "
         "Survey*", "https://www.jmlr.org/papers/v18/17-468.html"),
        ("He et al. — *Deep Residual Learning for Image Recognition* (ResNet)",
         "https://doi.org/10.1109/CVPR.2016.90"),
        ("Ba, Kiros & Hinton — *Layer Normalization*",
         "https://arxiv.org/abs/1607.06450"),
        ("TensorFlow — *Better performance with tf.function*",
         "https://www.tensorflow.org/guide/function"),
    ])


# ==========================================================================
SECTIONS = [
    ("12.1", "A Quick Tour of TensorFlow", s_12_1),
    ("12.2", "Using TensorFlow Like NumPy", s_12_2),
    ("12.3", "Custom Loss Functions", s_12_3),
    ("12.4", "Custom Activations & Constraints", s_12_4),
    ("12.5", "Custom Metrics", s_12_5),
    ("12.6", "Custom Layers", s_12_6),
    ("12.7", "Custom Models & Internal Losses", s_12_7),
    ("12.8", "Autodiff & Training Loops", s_12_8),
    ("12.9", "TF Functions, Graphs & Exercises", s_12_9),
]

nav.render_chapter(CH, SECTIONS)
