"""Chapter 13 — Loading and Preprocessing Data with TensorFlow."""

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
CH = "ch13"

hero(
    kicker="Part II · Chapter 13",
    title="Loading and Preprocessing Data with TensorFlow",
    blurb=(
        "A GPU that finishes a batch in 8 ms and then waits 40 ms for the next one "
        "is running at 17 % utilisation. This chapter is about the other 83 %: "
        "the <code>tf.data</code> pipeline, interleaving and prefetching, the "
        "TFRecord format, and the preprocessing layers that let you ship "
        "normalisation, vocabularies and augmentation <i>inside</i> the model."
    ),
    chips=["Input pipelines", "9 sub-sections", "8 animations",
           "9 code labs", "Feed the GPU"],
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
        "**TensorFlow is not installed in this environment**, so the code "
        "labs on this page cannot run — the lecture, the mathematics and "
        "the animation all work normally. This is expected on the hosted "
        "demo, where TensorFlow needs more memory than the free tier "
        "allows. To run the labs, clone "
        "[the repository](https://github.com/merwanroudane/ai_labs) and "
        "install with `pip install -r requirements-local.txt`.",
        icon="⚠️")


# ==========================================================================
def s_13_1():
    section("13.1", "The tf.data API — Chaining Transformations")

    lead(
        "A <code>tf.data.Dataset</code> is a lazy sequence of items. You build a "
        "pipeline by chaining transformations, each of which returns a new "
        "dataset without touching a byte of data until you iterate."
    )

    sub("Creating a dataset")

    table(
        ["Source", "Constructor"],
        [["An in-memory tensor",
          "<code>tf.data.Dataset.from_tensor_slices(X)</code> — slices along "
          "axis 0"],
         ["A tuple of tensors (features + labels)",
          "<code>from_tensor_slices((X, y))</code>"],
         ["A Python generator", "<code>Dataset.from_generator(gen, ...)</code>"],
         ["CSV files", "<code>tf.data.experimental.make_csv_dataset(...)</code>, "
          "or <code>TextLineDataset</code> + a parse function"],
         ["TFRecord files", "<code>tf.data.TFRecordDataset(filepaths)</code>"],
         ["A list of file paths",
          "<code>Dataset.list_files(pattern)</code> — shuffles by default"],
         ["A directory of images",
          "<code>keras.utils.image_dataset_from_directory(...)</code>"]],
    )

    sub("The core transformations")

    table(
        ["Method", "Effect", "Note"],
        [["<code>.map(fn, num_parallel_calls=N)</code>",
          "Apply <code>fn</code> to every item",
          "<b>Always set <code>num_parallel_calls=tf.data.AUTOTUNE</code></b>"],
         ["<code>.filter(pred)</code>", "Keep items where <code>pred</code> is "
          "true", "The predicate must be a TF function"],
         ["<code>.batch(n)</code>", "Group into batches of $n$",
          "<code>drop_remainder=True</code> gives a fixed shape"],
         ["<code>.unbatch()</code>", "The inverse", ""],
         ["<code>.repeat(n)</code>", "Repeat the dataset $n$ times",
          "<code>None</code> = forever"],
         ["<code>.shuffle(buffer_size)</code>", "Shuffle through a buffer",
          "§13.2 — the buffer size matters enormously"],
         ["<code>.take(n)</code> / <code>.skip(n)</code>",
          "First $n$ / all but the first $n$", "Useful for train/valid splits"],
         ["<code>.prefetch(n)</code>", "Prepare the next $n$ batches in the "
          "background", "<b>Always the last step</b>"],
         ["<code>.cache()</code>", "Memorise the dataset after the first epoch",
          "Place it <b>after</b> expensive maps, <b>before</b> shuffling"],
         ["<code>.interleave(fn, cycle_length=N)</code>",
          "Read from $N$ sources in round-robin", "§13.2"]],
    )

    sub("Order matters")

    idea(
        "The canonical pipeline order",
        "<code>list_files → interleave → map(parse) → cache → shuffle → repeat "
        "→ batch → map(batched preprocessing) → prefetch</code>.<br><br>"
        "The reasoning: <b>cache after the expensive per-item work</b> so it is "
        "done once, not once per epoch; <b>shuffle after cache</b> so each epoch "
        "gets a different order; <b>batch after shuffle</b> or you shuffle whole "
        "batches rather than items; and any <b>vectorisable</b> preprocessing "
        "goes <i>after</i> <code>batch</code>, where it runs on 32 items at once "
        "instead of one at a time. <code>prefetch</code> is always last.",
    )

    pitfall(
        "Three orderings that quietly cost you accuracy or speed",
        "<b><code>.batch().shuffle()</code></b> shuffles the <i>order of "
        "batches</i>, but the same 32 items stay together in every epoch. That is "
        "much weaker randomisation than shuffling items, and it measurably hurts "
        "SGD.<br>"
        "<b><code>.shuffle().cache()</code></b> caches <i>one particular "
        "shuffled order</i> and then replays it identically every epoch — you "
        "have silently disabled shuffling.<br>"
        "<b><code>.map(fn)</code> before <code>.batch()</code></b> for a "
        "vectorisable <code>fn</code> runs the op once per item instead of once "
        "per batch, typically 10–30× slower.",
    )

    anim_header("Building a pipeline, one transformation at a time")

    stages = [
        ("from_tensor_slices(X, y)", "12 individual (x, y) items", 12, 1),
        (".map(parse)", "each item transformed — still 12", 12, 1),
        (".cache()", "results memorised after epoch 1", 12, 1),
        (".shuffle(buffer_size=8)", "order randomised through a buffer", 12, 1),
        (".repeat(2)", "the sequence is repeated", 24, 1),
        (".batch(4)", "grouped into batches of 4", 6, 4),
        (".prefetch(AUTOTUNE)", "next batches prepared while the GPU works", 6, 4),
    ]
    rng = np.random.default_rng(0)

    frames = []
    for k, (nm, desc, count, bsz) in enumerate(stages):
        xs, ys, cols, sizes = [], [], [], []
        for i in range(count):
            row, col = divmod(i, 8)
            xs.append(col); ys.append(-row)
            cols.append(SEQ[(i // max(bsz, 1)) % len(SEQ)] if bsz > 1
                        else alpha(C["primary"], .85))
            sizes.append(30 if bsz == 1 else 34)
        frames.append(go.Frame(name=str(k + 1), data=[
            go.Scatter(x=xs, y=ys, mode="markers",
                       marker=dict(size=sizes, color=cols, symbol="square",
                                   line=dict(color="#fff", width=2)),
                       showlegend=False, hoverinfo="skip")],
            layout=go.Layout(title=f"{nm}   —   {desc}")))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=380, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.7, 8]),
                    yaxis=dict(visible=False, range=[-3.4, .8]),
                    title=f"{stages[0][0]}   —   {stages[0][1]}")
    anim.animate(f, frames, duration=nav.anim_ms(1200), slider_prefix="step ")
    figure(f, "Colour groups items into batches from step 6 onward.")

    code_lab(
        "Building and inspecting a tf.data pipeline",
        '''import numpy as np
import tensorflow as tf

# ============ 1. THE SIMPLEST DATASET ==================================
X = tf.range(10)
ds = tf.data.Dataset.from_tensor_slices(X)
print("=== a dataset is a lazy sequence ===")
print(f"  {ds}")
print(f"  items: {[int(x) for x in ds]}")

# ============ 2. CHAINING ==============================================
print("\\n=== chaining transformations ===")
ds = tf.data.Dataset.from_tensor_slices(tf.range(10))
ds = ds.repeat(3).batch(7)
for i, item in enumerate(ds):
    print(f"  batch {i}: {item.numpy()}")
print("  note the last batch is short -- use drop_remainder=True to avoid that")

ds2 = tf.data.Dataset.from_tensor_slices(tf.range(10)).repeat(3).batch(7,
        drop_remainder=True)
print(f"  with drop_remainder: {[list(b.numpy()) for b in ds2]}")

# ============ 3. map, filter, take =====================================
print("\\n=== map / filter / take ===")
ds = tf.data.Dataset.from_tensor_slices(tf.range(10))
ds = (ds.map(lambda x: x ** 2, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(lambda x: x % 2 == 0)
        .take(4))
print(f"  squares, evens only, first 4: {[int(x) for x in ds]}")

# ============ 4. FEATURES AND LABELS TOGETHER ==========================
print("\\n=== (X, y) pairs ===")
rng = np.random.default_rng(0)
Xa = rng.normal(0, 1, (100, 4)).astype("float32")
ya = (Xa[:, 0] > 0).astype("int32")
ds = tf.data.Dataset.from_tensor_slices((Xa, ya)).batch(8)
for xb, yb in ds.take(2):
    print(f"  X batch {tuple(xb.shape)}  y batch {tuple(yb.shape)}  "
          f"labels {yb.numpy()}")

# ============ 5. ORDER MATTERS: batch-then-shuffle is WRONG ============
print("\\n=== .shuffle().batch()  vs  .batch().shuffle() ===")
base = tf.data.Dataset.from_tensor_slices(tf.range(12))

print("  CORRECT -- shuffle items, then batch:")
for epoch in range(2):
    right = base.shuffle(12, seed=epoch, reshuffle_each_iteration=True).batch(4)
    print(f"    epoch {epoch}: {[list(b.numpy()) for b in right]}")

print("  WRONG -- batch first, then shuffle the batches:")
for epoch in range(2):
    wrong = base.batch(4).shuffle(3, seed=epoch)
    print(f"    epoch {epoch}: {[list(b.numpy()) for b in wrong]}")
print("  the same items stay glued together forever -- much weaker randomisation")

# ============ 6. ORDER MATTERS: shuffle before cache is WRONG =========
print("\\n=== .cache() freezes whatever came before it ===")
print("  shuffle().cache()  -- caches ONE order and replays it:")
bad = base.shuffle(12, seed=0).cache()
for epoch in range(3):
    print(f"    epoch {epoch}: {[int(x) for x in bad]}")
print("  cache().shuffle()  -- a fresh order every epoch:")
good = base.cache().shuffle(12, seed=0, reshuffle_each_iteration=True)
for epoch in range(3):
    print(f"    epoch {epoch}: {[int(x) for x in good]}")

# ============ 7. map BEFORE vs AFTER batch =============================
import time
print("\\n=== vectorise: map AFTER batch ===")
big = tf.data.Dataset.from_tensor_slices(
    tf.random.normal((20000, 32))).cache()

def per_item(x):    return (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-7)
def per_batch(xb):  return (xb - tf.reduce_mean(xb, axis=1, keepdims=True)) / \\
                           (tf.math.reduce_std(xb, axis=1, keepdims=True) + 1e-7)

for nm, pipe in [("map(per_item).batch(64)",
                  big.map(per_item, num_parallel_calls=tf.data.AUTOTUNE).batch(64)),
                 ("batch(64).map(per_batch)",
                  big.batch(64).map(per_batch, num_parallel_calls=tf.data.AUTOTUNE))]:
    for _ in pipe.take(5): pass                       # warm-up
    t0 = time.perf_counter()
    for _ in pipe: pass
    print(f"  {nm:<28} {time.perf_counter()-t0:.3f}s")
print("  the same maths, applied once per batch instead of once per item")

# ============ 8. USE IT WITH KERAS =====================================
print("\\n=== straight into model.fit ===")
from tensorflow import keras
train_ds = (tf.data.Dataset.from_tensor_slices((Xa, ya))
            .cache().shuffle(100, seed=42).batch(16)
            .prefetch(tf.data.AUTOTUNE))
model = keras.Sequential([keras.layers.Input(shape=(4,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds, epochs=10, verbose=0)
print(f"  trained directly on the Dataset -- accuracy "
      f"{model.evaluate(train_ds, verbose=0)[1]:.4f}")
print("  no NumPy arrays passed to fit() at all")

# ============ 9. INSPECTING A PIPELINE ================================
print("\\n=== element_spec tells you the shapes and dtypes ===")
print(f"  {train_ds.element_spec}")
print(f"  cardinality (number of batches): "
      f"{tf.data.experimental.cardinality(train_ds).numpy()}")
''',
        key="ch13_tfdata",
    )

    keypoints([
        "A <code>Dataset</code> is <b>lazy</b> — nothing happens until you "
        "iterate.",
        "Canonical order: <code>interleave → map → cache → shuffle → repeat → "
        "batch → map → prefetch</code>.",
        "<b><code>.batch().shuffle()</code> is not shuffling</b> — it permutes "
        "batches, not items.",
        "<b><code>.shuffle().cache()</code> freezes one order</b> and replays it "
        "every epoch.",
        "Put vectorisable preprocessing <b>after</b> <code>batch</code>; always "
        "pass <code>num_parallel_calls=AUTOTUNE</code>.",
    ])


# ==========================================================================
def s_13_2():
    section("13.2", "Shuffling, Interleaving, and Prefetching")

    lead(
        "Three transformations that decide whether your expensive accelerator is "
        "busy or idle. Each solves a different bottleneck."
    )

    sub("Shuffling with a buffer")

    md(
        "`shuffle(buffer_size)` fills a buffer with the first `buffer_size` items, "
        "then repeatedly emits a random item from the buffer and refills the gap "
        "from the source:"
    )

    derive(
        [("The quality of the shuffle is bounded by the buffer size. An item at "
          "position $i$ in the source can only be emitted at a position within "
          "roughly $\\pm$ buffer_size of $i$.", None),
         ("So with $m$ items and buffer $b$, the maximum displacement is:",
          r"\bigl|\text{output position} - \text{source position}\bigr| "
          r"\lesssim b"),
         ("<b>The failure case.</b> If the file is sorted by label — all the "
          "0s, then all the 1s — and $b \\ll m/K$, then <b>every batch contains "
          "one class</b>. The gradient of each step points at one class, training "
          "oscillates violently, and batch-norm statistics are meaningless.", None),
         ("<b>Two fixes.</b> Either $b$ large enough to span the class blocks "
          "(memory permitting), or — much better — <b>shuffle the source itself</b>: "
          "split the data into many files, shuffle the file list, and "
          "<code>interleave</code> reads from several files at once. Then even a "
          "small buffer draws from many different regions.", None),
         ("<code>reshuffle_each_iteration=True</code> (the default) gives a new "
          "permutation each epoch. Set it to <code>False</code> — or place "
          "<code>cache()</code> after the shuffle — and every epoch sees the "
          "identical order.", None)],
        title="Why buffer size decides shuffle quality",
    )

    sub("Interleaving from multiple files")

    md(
        "`interleave` opens `cycle_length` files at once and reads from them in "
        "round-robin. Two benefits: reads run in parallel (I/O bound work "
        "overlaps), and the resulting stream mixes items from different regions "
        "of the dataset, which repairs a badly-ordered source."
    )

    md(
        """
```python
dataset = (tf.data.Dataset.list_files(train_filepaths, seed=42)
           .interleave(lambda path: tf.data.TextLineDataset(path).skip(1),
                       cycle_length=5,
                       num_parallel_calls=tf.data.AUTOTUNE))
```
        """
    )

    sub("Prefetching")

    md(
        "Without prefetch, the CPU prepares batch $n+1$ **only after** the GPU "
        "has finished batch $n$. With prefetch, they overlap:"
    )

    math(r"""
    T_{\text{no prefetch}} = \sum_{n} \bigl(t_{\text{CPU}} + t_{\text{GPU}}\bigr)
    \qquad\qquad
    T_{\text{prefetch}} \approx \sum_{n} \max\bigl(t_{\text{CPU}},\, t_{\text{GPU}}\bigr)
    """)

    proof(
        "The maximum possible speed-up is 2×, and where it lands",
        "$\\frac{t_{\\text{CPU}} + t_{\\text{GPU}}}{\\max(t_{\\text{CPU}}, "
        "t_{\\text{GPU}})}$ is maximised at <b>2</b> when the two are equal, and "
        "tends to 1 when either dominates. So prefetching alone cannot fix a "
        "pipeline that is badly CPU-bound — for that you need "
        "<code>num_parallel_calls</code> on the maps and "
        "<code>num_parallel_calls</code> on the interleave, which raise the "
        "throughput of the CPU side itself. Prefetch removes the "
        "<i>serialisation</i>; parallelism removes the <i>bottleneck</i>.",
    )

    codenote(
        "tf.data.AUTOTUNE",
        "Passing <code>tf.data.AUTOTUNE</code> instead of a number lets "
        "TensorFlow tune the parallelism at runtime, based on the measured "
        "throughput of each stage and the available CPU. Use it for "
        "<code>num_parallel_calls</code>, <code>cycle_length</code> and the "
        "<code>prefetch</code> depth unless you have measured something better.",
    )

    anim_header("Prefetching: CPU and GPU overlapping")
    md(
        "Two timelines. Without prefetch the two devices alternate and each waits "
        "for the other. With prefetch the CPU works on batch $n+1$ while the GPU "
        "trains on batch $n$."
    )

    n_b = 7
    t_cpu, t_gpu = 3, 4
    seq_cpu, seq_gpu = [], []
    t = 0
    for i in range(n_b):
        seq_cpu.append((t, t + t_cpu)); t += t_cpu
        seq_gpu.append((t, t + t_gpu)); t += t_gpu
    total_seq = t

    pre_cpu, pre_gpu = [], []
    c_end = 0; g_end = 0
    for i in range(n_b):
        c_start = c_end
        c_end = c_start + t_cpu
        pre_cpu.append((c_start, c_end))
        g_start = max(c_end, g_end)
        g_end = g_start + t_gpu
        pre_gpu.append((g_start, g_end))
    total_pre = g_end

    frames = []
    for k in range(1, n_b + 1):
        bars = []
        for i in range(k):
            for row, (seq, col, lbl) in enumerate(
                    [(seq_cpu, C["warning"], "CPU"), (seq_gpu, C["primary"], "GPU")]):
                s0, s1 = seq[i]
                bars.append(go.Scatter(x=[s0, s1, s1, s0, s0],
                                       y=[3 - row - .35, 3 - row - .35,
                                          3 - row + .35, 3 - row + .35, 3 - row - .35],
                                       fill="toself", fillcolor=alpha(col, .85),
                                       line=dict(color="#fff", width=1.5),
                                       showlegend=False, hoverinfo="skip"))
            for row, (seq, col) in enumerate([(pre_cpu, C["warning"]),
                                              (pre_gpu, C["primary"])]):
                s0, s1 = seq[i]
                bars.append(go.Scatter(x=[s0, s1, s1, s0, s0],
                                       y=[1 - row - .35, 1 - row - .35,
                                          1 - row + .35, 1 - row + .35, 1 - row - .35],
                                       fill="toself", fillcolor=alpha(col, .85),
                                       line=dict(color="#fff", width=1.5),
                                       showlegend=False, hoverinfo="skip"))
        frames.append(go.Frame(name=str(k), data=bars,
                               layout=go.Layout(annotations=[
                                   dict(x=-1.5, y=3, text="<b>CPU</b>", showarrow=False,
                                        font=dict(size=11, color=C["warning"])),
                                   dict(x=-1.5, y=2, text="<b>GPU</b>", showarrow=False,
                                        font=dict(size=11, color=C["primary"])),
                                   dict(x=-1.5, y=1, text="<b>CPU</b>", showarrow=False,
                                        font=dict(size=11, color=C["warning"])),
                                   dict(x=-1.5, y=0, text="<b>GPU</b>", showarrow=False,
                                        font=dict(size=11, color=C["primary"])),
                                   dict(x=total_seq / 2, y=3.9,
                                        text="<b>WITHOUT prefetch</b> — strictly "
                                             "alternating",
                                        showarrow=False,
                                        font=dict(size=12, color=C["danger"])),
                                   dict(x=total_seq / 2, y=1.9,
                                        text="<b>WITH prefetch</b> — overlapped",
                                        showarrow=False,
                                        font=dict(size=12, color=C["success"])),
                                   anim.annotate_step(
                                       f"{k} batches   ·   sequential = "
                                       f"{seq_gpu[k-1][1]} time units   ·   "
                                       f"prefetched = {pre_gpu[k-1][1]} units   ·   "
                                       f"speed-up "
                                       f"{seq_gpu[k-1][1]/pre_gpu[k-1][1]:.2f}x")])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=400, plot_bgcolor="#FFFFFF",
                    xaxis=dict(title="time", range=[-2.5, total_seq + 1]),
                    yaxis=dict(visible=False, range=[-.7, 4.4]),
                    annotations=list(frames[0].layout.annotations),
                    title="CPU prepares, GPU trains")
    anim.animate(f, frames, duration=nav.anim_ms(600), slider_prefix="batch ")
    figure(f)

    code_lab(
        "Shuffle buffers, interleaving and the prefetch speed-up, measured",
        '''import numpy as np, os, time, tempfile, shutil
from pathlib import Path
import tensorflow as tf

tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch13_"))

# ============ 1. SHUFFLE BUFFER SIZE ===================================
print("=== the buffer bounds how far an item can move ===")
src = tf.data.Dataset.from_tensor_slices(tf.range(30))
for buf in [1, 3, 10, 30]:
    out = [int(x) for x in src.shuffle(buf, seed=42)]
    disp = max(abs(pos - val) for pos, val in enumerate(out))
    print(f"  buffer {buf:>3}: {out[:14]}...  max displacement {disp}")

# --- the failure case: a SORTED source --------------------------------
print("\\n=== a sorted source with a small buffer ===")
labels = tf.constant([0]*300 + [1]*300 + [2]*300)
sorted_ds = tf.data.Dataset.from_tensor_slices(labels)
print(f"  {'buffer':>8}{'distinct classes per batch of 32 (mean)':>42}")
for buf in [10, 100, 500, 900]:
    batches = list(sorted_ds.shuffle(buf, seed=0).batch(32))
    n_distinct = np.mean([len(np.unique(b.numpy())) for b in batches])
    print(f"  {buf:>8}{n_distinct:>42.2f}")
print("  buffer 10 -> almost every batch is a SINGLE class. Training will thrash.")

# ============ 2. SHARDING THE SOURCE IS THE REAL FIX ===================
print("\\n=== write the data as many shards, then interleave ===")
n_shards = 20
paths = []
for s in range(n_shards):
    p = tmp / f"shard_{s:02d}.csv"
    # each shard holds a contiguous slice of the SORTED data
    block = labels.numpy()[s*45:(s+1)*45]
    p.write_text("label\\n" + "\\n".join(str(int(v)) for v in block))
    paths.append(str(p))
print(f"  wrote {n_shards} shards to {tmp.name}/")

files = tf.data.Dataset.list_files(str(tmp/"shard_*.csv"), seed=42)
inter = files.interleave(lambda p: tf.data.TextLineDataset(p).skip(1),
                         cycle_length=n_shards,
                         num_parallel_calls=tf.data.AUTOTUNE)
inter = inter.map(lambda s: tf.strings.to_number(s, tf.int32))
batches = list(inter.shuffle(50, seed=0).batch(32))
print(f"  with interleave + a buffer of only 50: "
      f"{np.mean([len(np.unique(b.numpy())) for b in batches]):.2f} classes/batch")
print("  interleaving mixes distant regions, so a SMALL buffer now suffices")

# ============ 3. reshuffle_each_iteration ==============================
print("\\n=== reshuffle_each_iteration ===")
for flag in (True, False):
    d = tf.data.Dataset.range(8).shuffle(8, seed=1,
                                         reshuffle_each_iteration=flag)
    orders = [[int(x) for x in d] for _ in range(3)]
    print(f"  reshuffle={str(flag):<5}: {orders}")

# ============ 4. PREFETCH: MEASURE THE SPEED-UP ========================
print("\\n=== prefetching ===")
def slow_parse(x):
    """Pretend this is JPEG decoding + augmentation."""
    for _ in range(3):
        x = tf.sort(tf.random.normal((400,)) + tf.cast(x, tf.float32))
    return tf.reduce_mean(x)

base = tf.data.Dataset.range(300)
configs = {
    "no parallelism, no prefetch":
        base.map(slow_parse).batch(16),
    "parallel map, no prefetch":
        base.map(slow_parse, num_parallel_calls=tf.data.AUTOTUNE).batch(16),
    "parallel map + prefetch":
        base.map(slow_parse, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(16).prefetch(tf.data.AUTOTUNE),
    "parallel map + cache + prefetch":
        base.map(slow_parse, num_parallel_calls=tf.data.AUTOTUNE)
            .cache().batch(16).prefetch(tf.data.AUTOTUNE),
}
print(f"{'pipeline':<36}{'epoch 1':>11}{'epoch 2':>11}")
for nm, pipe in configs.items():
    t0 = time.perf_counter()
    for _ in pipe: pass
    e1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in pipe: pass
    e2 = time.perf_counter() - t0
    print(f"{nm:<36}{e1:>10.3f}s{e2:>10.3f}s")
print("\\ncache() makes epoch 2 nearly free -- the expensive map ran only once.")

# ============ 5. INTERLEAVE PARALLELISM ================================
print("\\n=== cycle_length ===")
files = tf.data.Dataset.list_files(str(tmp/"shard_*.csv"), seed=42)
print(f"{'cycle_length':>14}{'time':>10}")
for cl in [1, 4, 20]:
    d = files.interleave(lambda p: tf.data.TextLineDataset(p).skip(1),
                         cycle_length=cl, num_parallel_calls=tf.data.AUTOTUNE)
    t0 = time.perf_counter()
    for _ in d: pass
    print(f"{cl:>14}{time.perf_counter()-t0:>9.4f}s")

# ============ 6. THE THEORETICAL LIMIT =================================
print("\\n=== the prefetch speed-up is capped at 2x ===")
print(f"{'t_cpu':>8}{'t_gpu':>8}{'sequential':>13}{'overlapped':>13}{'speed-up':>11}")
for tc, tg in [(1, 9), (3, 7), (5, 5), (7, 3), (9, 1)]:
    seq, ovl = tc + tg, max(tc, tg)
    print(f"{tc:>8}{tg:>8}{seq:>13}{ovl:>13}{seq/ovl:>11.2f}x")
print("Maximum when the two are balanced. Prefetch removes the SERIALISATION;")
print("num_parallel_calls removes the CPU BOTTLENECK. You usually need both.")

shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch13_shuffle",
    )

    quiz(
        "Your training data is one file sorted by class, and you use "
        "<code>shuffle(1000)</code> on 1 000 000 rows. What happens?",
        ["Nothing — 1000 is a fine buffer",
         "Most batches contain a single class, so training oscillates badly",
         "The shuffle raises an error",
         "It works but uses too much memory"],
        1,
        "The buffer spans 1000 of 1 000 000 rows — 0.1 % of the file. Inside a "
        "class block, every item in the buffer has the same label. Shard the file "
        "and <code>interleave</code>, or shuffle the source once on disk.",
        key="ch13q1",
    )

    keypoints([
        "<code>shuffle(b)</code> can displace an item by at most about $b$ "
        "positions — a small $b$ on sorted data is no shuffle at all.",
        "The real fix is <b>sharding + interleave</b>, which mixes distant "
        "regions with a small buffer.",
        "<code>prefetch</code> overlaps CPU and GPU; the speed-up is capped at "
        "<b>2×</b> and is maximal when they are balanced.",
        "<code>num_parallel_calls=AUTOTUNE</code> raises CPU throughput; prefetch "
        "alone cannot fix a CPU-bound pipeline.",
        "<code>cache()</code> after the expensive maps makes every epoch after "
        "the first nearly free.",
    ])


# ==========================================================================
def s_13_3():
    section("13.3", "The TFRecord Format and Protocol Buffers")

    lead(
        "TensorFlow's preferred storage format: a flat sequence of length-prefixed "
        "binary records, each holding a serialised protobuf. Fast to read "
        "sequentially, compressible, and shardable."
    )

    sub("The file format")

    md(
        "A TFRecord file is simply a sequence of records, each stored as:"
    )

    table(
        ["Bytes", "Content"],
        [["8", "length of the data, as a little-endian <code>uint64</code>"],
         ["4", "CRC-32C checksum of the length"],
         ["<i>length</i>", "the data itself"],
         ["4", "CRC-32C checksum of the data"]],
        "That is the whole specification. There is no index and no random "
        "access — which is exactly why sequential reads are so fast.",
    )

    sub("Protocol buffers")

    md(
        "The data in each record is usually a serialised `Example` protobuf, "
        "which is a nested structure of typed lists:"
    )

    md(
        """
```protobuf
message BytesList  { repeated bytes value = 1; }
message FloatList  { repeated float value = 1 [packed = true]; }
message Int64List  { repeated int64 value = 1 [packed = true]; }

message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
}
message Features { map<string, Feature> feature = 1; }
message Example  { Features features = 1; }
```
        """
    )

    md("So an `Example` is a **dictionary from string keys to typed lists** — and "
       "the three types are the only ones available. Everything else "
       "(images, text, booleans) is encoded into one of them.")

    table(
        ["Your data", "Store as", "How"],
        [["A float feature", "<code>FloatList</code>", "Directly"],
         ["An integer or boolean", "<code>Int64List</code>", "Cast to int64"],
         ["A string", "<code>BytesList</code>", "<code>.encode('utf-8')</code>"],
         ["An image", "<code>BytesList</code>",
          "<b>Keep it JPEG/PNG-encoded</b> — decoding at read time is far cheaper "
          "than storing raw pixels"],
         ["An array of floats", "<code>FloatList</code>",
          "Flatten, and store the shape separately; or "
          "<code>tf.io.serialize_tensor</code> into a <code>BytesList</code>"],
         ["A variable-length sequence", "<code>SequenceExample</code>",
          "Context features plus <code>FeatureLists</code>"]],
    )

    idea(
        "Store images encoded, not decoded",
        "A 224×224×3 uint8 image is 150 KB raw and about 15 KB as JPEG — a 10× "
        "difference in file size and therefore in I/O. Decoding costs CPU, but "
        "CPU is what <code>num_parallel_calls</code> parallelises, whereas disk "
        "bandwidth is a hard wall. <b>Almost always store the encoded bytes</b> "
        "and call <code>tf.io.decode_jpeg</code> in the parse function.",
    )

    sub("Reading them back")

    md(
        "Parsing needs a **description** of the expected features, because the "
        "protobuf itself carries no schema:"
    )

    table(
        ["Descriptor", "For", "Parse with"],
        [["<code>tf.io.FixedLenFeature(shape, dtype, default)</code>",
          "Features with a known fixed size",
          "<code>tf.io.parse_single_example</code> / "
          "<code>parse_example</code>"],
         ["<code>tf.io.VarLenFeature(dtype)</code>",
          "Variable-length features",
          "Returns a <code>SparseTensor</code>; use "
          "<code>tf.sparse.to_dense</code>"],
         ["<code>tf.io.RaggedFeature(dtype)</code>",
          "Variable-length, as a ragged tensor", "Returns a RaggedTensor"]],
    )

    tip(
        "Parse whole batches, not single examples",
        "<code>tf.io.parse_example</code> takes a <b>batch</b> of serialised "
        "strings and is far faster than calling "
        "<code>parse_single_example</code> once per record. So the pipeline order "
        "is <code>TFRecordDataset → batch → map(parse_example)</code>, not the "
        "other way round — the same vectorisation argument as §13.1.",
    )

    anim_header("Anatomy of a TFRecord file")

    parts = [
        ("length (8 bytes, uint64)", C["accent"], 8),
        ("CRC of length (4 bytes)", C["muted"], 4),
        ("serialised Example protobuf", C["primary"], 30),
        ("CRC of data (4 bytes)", C["muted"], 4),
    ]
    frames = []
    for rec in range(1, 5):
        shapes, ann = [], []
        x = 0
        for r in range(rec):
            for nm, col, w in parts:
                shapes.append(go.Scatter(
                    x=[x, x + w, x + w, x, x], y=[-.4, -.4, .4, .4, -.4],
                    fill="toself", fillcolor=alpha(col, .85),
                    line=dict(color="#fff", width=1.5),
                    showlegend=False, hoverinfo="skip"))
                if r == rec - 1:
                    ann.append(dict(x=x + w / 2, y=0, text=nm.split("(")[0].strip(),
                                    showarrow=False,
                                    font=dict(size=8.5, color="#fff")))
                x += w
            ann.append(dict(x=x - 23, y=-.75, text=f"record {r+1}",
                            showarrow=False,
                            font=dict(size=10, color=C["ink_soft"])))
        frames.append(go.Frame(name=str(rec), data=shapes,
                               layout=go.Layout(annotations=ann,
                                                xaxis=dict(range=[-2, max(50, rec*46+4)]),
                                                title=f"{rec} record(s) — "
                                                      f"{rec*46} bytes, no index, "
                                                      f"sequential reads only")))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=290, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-2, 50], title="byte offset"),
                    yaxis=dict(visible=False, range=[-1.1, .8]),
                    annotations=list(frames[0].layout.annotations),
                    title=frames[0].layout.title.text)
    anim.animate(f, frames, duration=nav.anim_ms(1100), slider_prefix="records ")
    figure(f)

    code_lab(
        "Write, compress, read and parse TFRecords",
        '''import numpy as np, os, time, tempfile, shutil
from pathlib import Path
import tensorflow as tf

tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch13b_"))
rng = np.random.default_rng(0)

# ============ 1. THE THREE FEATURE TYPES ===============================
BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature   = tf.train.Feature
Features  = tf.train.Features
Example   = tf.train.Example

def _bytes(v):  return Feature(bytes_list=BytesList(value=[v]))
def _float(v):  return Feature(float_list=FloatList(value=v))
def _int64(v):  return Feature(int64_list=Int64List(value=v))

example = Example(features=Features(feature={
    "name":   _bytes(b"Alice"),
    "id":     _int64([123]),
    "emails": _bytes(b"a@b.com"),
    "scores": _float([1.5, 2.5, 3.5]),
}))
serialized = example.SerializeToString()
print("=== one Example protobuf ===")
print(f"  serialised to {len(serialized)} bytes")
print(f"  first 40 bytes: {serialized[:40]}")

# ============ 2. WRITE AND READ ========================================
path = tmp / "my_data.tfrecord"
with tf.io.TFRecordWriter(str(path)) as w:
    for i in range(5):
        ex = Example(features=Features(feature={
            "id":     _int64([i]),
            "name":   _bytes(f"person_{i}".encode()),
            "scores": _float(list(rng.normal(0, 1, 3).astype("float32"))),
        }))
        w.write(ex.SerializeToString())
print(f"\\n=== wrote {path.name}: {path.stat().st_size} bytes for 5 records ===")

feature_description = {
    "id":     tf.io.FixedLenFeature([],  tf.int64,  default_value=0),
    "name":   tf.io.FixedLenFeature([],  tf.string, default_value=""),
    "scores": tf.io.FixedLenFeature([3], tf.float32),
}
raw = tf.data.TFRecordDataset([str(path)])
parsed = raw.map(lambda s: tf.io.parse_single_example(s, feature_description))
for rec in parsed.take(3):
    print(f"  id={int(rec['id'])}  name={rec['name'].numpy().decode()}  "
          f"scores={rec['scores'].numpy().round(3)}")

# ============ 3. PARSE A WHOLE BATCH (much faster) =====================
print("\\n=== parse_single_example vs parse_example ===")
big = tmp / "big.tfrecord"
with tf.io.TFRecordWriter(str(big)) as w:
    for i in range(20000):
        ex = Example(features=Features(feature={
            "id": _int64([i]), "scores": _float([float(i), float(i*2), float(i*3)])}))
        w.write(ex.SerializeToString())
desc = {"id": tf.io.FixedLenFeature([], tf.int64),
        "scores": tf.io.FixedLenFeature([3], tf.float32)}

single = (tf.data.TFRecordDataset([str(big)])
          .map(lambda s: tf.io.parse_single_example(s, desc),
               num_parallel_calls=tf.data.AUTOTUNE).batch(128))
batched = (tf.data.TFRecordDataset([str(big)]).batch(128)
           .map(lambda s: tf.io.parse_example(s, desc),
                num_parallel_calls=tf.data.AUTOTUNE))
for nm, pipe in [("map(parse_single).batch(128)", single),
                 ("batch(128).map(parse_example)", batched)]:
    for _ in pipe.take(3): pass
    t0 = time.perf_counter()
    for _ in pipe: pass
    print(f"  {nm:<32} {time.perf_counter()-t0:.3f}s")

# ============ 4. COMPRESSION ===========================================
print("\\n=== compression ===")
opts = tf.io.TFRecordOptions(compression_type="GZIP")
gz = tmp / "big.tfrecord.gz"
with tf.io.TFRecordWriter(str(gz), opts) as w:
    for i in range(20000):
        ex = Example(features=Features(feature={
            "id": _int64([i]), "scores": _float([float(i), float(i*2), float(i*3)])}))
        w.write(ex.SerializeToString())
print(f"  uncompressed : {big.stat().st_size/1e6:.3f} MB")
print(f"  GZIP         : {gz.stat().st_size/1e6:.3f} MB "
      f"({gz.stat().st_size/big.stat().st_size:.1%})")
t0 = time.perf_counter()
for _ in tf.data.TFRecordDataset([str(big)]).batch(256): pass
t_raw = time.perf_counter()-t0
t0 = time.perf_counter()
for _ in tf.data.TFRecordDataset([str(gz)], compression_type="GZIP").batch(256): pass
t_gz = time.perf_counter()-t0
print(f"  read time    : {t_raw:.3f}s vs {t_gz:.3f}s")
print("  worth it when the network or disk is the bottleneck, not the CPU")

# ============ 5. STORE IMAGES ENCODED, NOT RAW ========================
print("\\n=== images: encoded vs raw ===")
img = tf.cast(rng.integers(0, 255, (224, 224, 3)), tf.uint8)
jpeg = tf.io.encode_jpeg(img, quality=90)
png  = tf.io.encode_png(img)
raw_bytes = tf.io.serialize_tensor(img)
print(f"  raw uint8 tensor : {len(raw_bytes.numpy())/1024:>8.1f} KB")
print(f"  PNG (lossless)   : {len(png.numpy())/1024:>8.1f} KB")
print(f"  JPEG q=90        : {len(jpeg.numpy())/1024:>8.1f} KB   "
      f"({len(jpeg.numpy())/len(raw_bytes.numpy()):.1%} of raw)")
print("  (this is random noise, the worst case for compression;")
print("   on real photographs JPEG is typically 5-10 % of raw)")

img_path = tmp / "images.tfrecord"
with tf.io.TFRecordWriter(str(img_path)) as w:
    for i in range(20):
        im = tf.cast(rng.integers(0, 255, (64, 64, 3)), tf.uint8)
        ex = Example(features=Features(feature={
            "image": _bytes(tf.io.encode_jpeg(im).numpy()),
            "label": _int64([i % 5]),
            "height": _int64([64]), "width": _int64([64])}))
        w.write(ex.SerializeToString())

img_desc = {"image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)}
def parse_image(serialized):
    ex = tf.io.parse_single_example(serialized, img_desc)
    image = tf.io.decode_jpeg(ex["image"], channels=3)
    image = tf.image.resize(image, [32, 32]) / 255.0
    return image, ex["label"]

img_ds = (tf.data.TFRecordDataset([str(img_path)])
          .map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(4).prefetch(tf.data.AUTOTUNE))
for xb, yb in img_ds.take(1):
    print(f"\\n  decoded batch: images {tuple(xb.shape)} dtype {xb.dtype}, "
          f"labels {yb.numpy()}")

# ============ 6. VARIABLE-LENGTH FEATURES ==============================
print("\\n=== VarLenFeature for ragged data ===")
var_path = tmp / "var.tfrecord"
with tf.io.TFRecordWriter(str(var_path)) as w:
    for tokens in [[1, 2, 3], [4], [], [5, 6, 7, 8, 9]]:
        ex = Example(features=Features(feature={"tokens": _int64(tokens)}))
        w.write(ex.SerializeToString())
var_desc = {"tokens": tf.io.VarLenFeature(tf.int64)}
for rec in tf.data.TFRecordDataset([str(var_path)]).map(
        lambda s: tf.io.parse_single_example(s, var_desc)):
    dense = tf.sparse.to_dense(rec["tokens"])
    print(f"  length {len(dense):>2}: {dense.numpy()}")

# ============ 7. SHARDING FOR PARALLEL READS ===========================
print("\\n=== sharding ===")
n_shards = 8
writers = [tf.io.TFRecordWriter(str(tmp/f"train_{s:02d}.tfrecord"))
           for s in range(n_shards)]
for i in range(4000):
    ex = Example(features=Features(feature={"id": _int64([i])}))
    writers[i % n_shards].write(ex.SerializeToString())   # ROUND-ROBIN
for w in writers: w.close()
print(f"  wrote {n_shards} shards, round-robin so each is already mixed")

files = tf.data.Dataset.list_files(str(tmp/"train_*.tfrecord"), seed=42)
pipe = (files.interleave(tf.data.TFRecordDataset,
                         cycle_length=n_shards,
                         num_parallel_calls=tf.data.AUTOTUNE)
        .batch(64)
        .map(lambda s: tf.io.parse_example(s, {"id": tf.io.FixedLenFeature([], tf.int64)}),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))
first = next(iter(pipe))["id"].numpy()
print(f"  first batch ids: {first[:16]}")
print(f"  spread across the dataset: min {first.min()} max {first.max()}")

shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch13_tfrecord",
    )

    keypoints([
        "A TFRecord is length-prefixed records with CRCs — no index, sequential "
        "reads only, which is what makes it fast.",
        "Each record is usually an <code>Example</code> protobuf: a dict of "
        "<b>bytes / float / int64</b> lists. Those three types are all you get.",
        "<b>Store images encoded</b> (JPEG/PNG) and decode in the parse function.",
        "<code>batch → parse_example</code> beats <code>parse_single_example → "
        "batch</code>.",
        "Shard round-robin so each file is already mixed; then "
        "<code>interleave</code>.",
    ])


# ==========================================================================
def s_13_4():
    section("13.4", "Keras Preprocessing Layers")

    lead(
        "Preprocessing as <b>layers</b> rather than as a separate pipeline. The "
        "point is not convenience — it is that the preprocessing then ships "
        "<i>inside</i> the saved model, which eliminates training/serving skew."
    )

    sub("Training/serving skew")

    pitfall(
        "The single most common production ML bug",
        "You standardise with scikit-learn during training, save the model, and "
        "the serving code re-implements the standardisation — in a different "
        "language, or with a slightly different mean, or forgetting one column. "
        "The model now receives inputs it was never trained on and its accuracy "
        "silently collapses. Nothing errors; the numbers are just quietly wrong. "
        "<b>Preprocessing layers make this impossible</b>, because the "
        "transformation is part of the model artefact.",
    )

    sub("The layers")

    table(
        ["Layer", "Learns (via <code>adapt</code>)", "Does"],
        [["<code>Normalization</code>", "mean and variance per feature",
          "$(x - \\mu)/\\sigma$"],
         ["<code>Discretization</code>", "bin boundaries (quantiles)",
          "Continuous → bin index"],
         ["<code>CategoryEncoding</code>", "—",
          "Integer → one-hot / multi-hot / count"],
         ["<code>StringLookup</code>", "the vocabulary",
          "String → integer index"],
         ["<code>IntegerLookup</code>", "the vocabulary", "Integer → dense index"],
         ["<code>Hashing</code>", "—",
          "String → integer via a hash; <b>no vocabulary needed</b>"],
         ["<code>TextVectorization</code>", "the vocabulary",
          "Text → integer sequence or TF-IDF"],
         ["<code>Rescaling</code>, <code>Resizing</code>, <code>CenterCrop</code>",
          "—", "Image geometry and range"],
         ["<code>RandomFlip</code>, <code>RandomRotation</code>, …", "—",
          "Augmentation — <b>active only during training</b>"]],
    )

    sub("adapt()")

    md(
        "Layers that learn something need `adapt()` on the training data — this "
        "is the layer's `fit`:"
    )

    md(
        """
```python
norm_layer = keras.layers.Normalization()
norm_layer.adapt(X_train)                  # computes mean and variance

model = keras.Sequential([norm_layer, keras.layers.Dense(1)])
```
        """
    )

    warn(
        "<code>adapt</code> on the <b>training set only</b>",
        "It is a fitted statistic, exactly like <code>StandardScaler.fit</code> "
        "in §2.4. Adapting on the full dataset leaks test information into "
        "training. Adapting on a large random <b>sample</b> of the training set "
        "is fine and much faster — the mean of a million rows is not measurably "
        "different from the mean of a hundred thousand.",
    )

    sub("Two placements, and the trade-off")

    table(
        ["Placement", "Speed", "Skew risk", "When"],
        [["<b>Inside the model</b> (a layer)",
          "Slower — runs on every epoch, and on the GPU where it may not "
          "vectorise well",
          "<b>None</b> — it ships with the model", "Deployment is the priority"],
         ["<b>In the <code>tf.data</code> pipeline</b> (a <code>map</code>)",
          "Faster — runs on the CPU in parallel with training, and is "
          "<code>cache</code>-able so it happens once",
          "<b>High</b> unless you are careful",
          "Training speed is the priority"],
         ["<b>Both</b> — pipeline for training, layer for serving",
          "Fast <i>and</i> safe",
          "None", "<b>The production answer</b>: preprocess in the pipeline for "
          "training, then wrap the trained model with the same layers for export"]],
    )

    idea(
        "The production pattern",
        "Preprocess in <code>tf.data</code> during training (fast, cached, "
        "parallel). At export time, build a <i>new</i> model that is "
        "<code>Sequential([preprocessing_layers, trained_model])</code> and save "
        "<b>that</b>. You get the speed during training and the "
        "skew-immunity in production. The lab below builds exactly this.",
    )

    anim_header("Where the same normalisation can live")

    layouts = [
        ("raw data → model", ["raw X", "Dense", "Dense", "ŷ"],
         [C["muted"], SEQ[0], SEQ[0], C["accent"]],
         "no preprocessing — the model sees unscaled features"),
        ("sklearn outside → model", ["raw X", "StandardScaler", "Dense", "ŷ"],
         [C["muted"], C["danger"], SEQ[0], C["accent"]],
         "FAST but the scaler must be reimplemented in production — SKEW RISK"),
        ("tf.data map → model", ["raw X", "map(normalise)", "Dense", "ŷ"],
         [C["muted"], C["warning"], SEQ[0], C["accent"]],
         "fast, cached, parallel — but still outside the saved model"),
        ("Normalization layer inside", ["raw X", "Normalization", "Dense", "ŷ"],
         [C["muted"], C["success"], SEQ[0], C["accent"]],
         "SAFE — ships inside the .keras file, no skew possible"),
        ("both: pipeline for training, layer for export",
         ["raw X", "Normalization", "trained model", "ŷ"],
         [C["muted"], C["success"], SEQ[0], C["accent"]],
         "THE PRODUCTION PATTERN — fast training, safe serving"),
    ]
    frames = []
    for title, boxes, cols, desc in layouts:
        shapes, ann = [], []
        for i, (nm, col) in enumerate(zip(boxes, cols)):
            x0 = i * 2.6
            shapes.append(go.Scatter(
                x=[x0, x0 + 2.1, x0 + 2.1, x0, x0],
                y=[-.42, -.42, .42, .42, -.42], fill="toself",
                fillcolor=alpha(col, .85), line=dict(color="#fff", width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=x0 + 1.05, y=0, text=nm, showarrow=False,
                            font=dict(size=10.5, color="#fff")))
            if i < len(boxes) - 1:
                ann.append(dict(x=x0 + 2.35, y=0, text="→", showarrow=False,
                                font=dict(size=16, color=C["ink_soft"])))
        ann.append(dict(x=5.2, y=-1.0, text=desc, showarrow=False,
                        font=dict(size=11, color=C["ink_soft"])))
        frames.append(go.Frame(name=title.split()[0], data=shapes,
                               layout=go.Layout(annotations=ann, title=title)))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=300, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.4, 11]),
                    yaxis=dict(visible=False, range=[-1.5, .9]),
                    annotations=list(frames[0].layout.annotations),
                    title=layouts[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(1700), slider_prefix="option ")
    figure(f)

    code_lab(
        "Every numeric preprocessing layer, and the production pattern",
        '''import numpy as np, tempfile, shutil, time
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)
tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch13c_"))
rng = np.random.default_rng(0)

# ============ 1. Normalization =========================================
print("=== Normalization ===")
X = rng.normal([10., 1000., -5.], [2., 300., 0.5], (5000, 3)).astype("float32")
norm = keras.layers.Normalization()
norm.adapt(X)                                    # this is the layer's "fit"
print(f"  learned mean     : {norm.mean.numpy().ravel().round(3)}")
print(f"  learned variance : {norm.variance.numpy().ravel().round(3)}")
out = norm(X)
print(f"  output mean/std  : {out.numpy().mean(0).round(5)} / "
      f"{out.numpy().std(0).round(5)}")

# per-axis control
norm_all = keras.layers.Normalization(axis=None)   # one scalar for everything
norm_all.adapt(X)
print(f"  axis=None -> a single mean {float(norm_all.mean):.3f} for all features")

# ============ 2. Discretization ========================================
print("\\n=== Discretization ===")
ages = rng.uniform(18, 90, 2000).astype("float32")
disc = keras.layers.Discretization(num_bins=5)
disc.adapt(ages)
print(f"  learned bin boundaries: {np.round(disc.bin_boundaries, 1)}")
sample = np.array([20., 35., 50., 65., 85.], dtype="float32")
print(f"  {sample} -> bins {disc(sample).numpy()}")

fixed = keras.layers.Discretization(bin_boundaries=[30., 45., 60.])
print(f"  with explicit boundaries [30,45,60]: {fixed(sample).numpy()}")

# ============ 3. CategoryEncoding ======================================
print("\\n=== CategoryEncoding ===")
cats = tf.constant([0, 2, 1, 4, 2])
for mode in ["one_hot", "multi_hot", "count"]:
    enc = keras.layers.CategoryEncoding(num_tokens=5, output_mode=mode)
    r = enc(cats if mode == "one_hot" else tf.constant([[0, 2, 2], [1, 1, 4]]))
    print(f"  {mode:<10}: shape {tuple(r.shape)}")
    print(f"{np.array2string(r.numpy(), prefix='              ')}")

# --- Discretization + CategoryEncoding = binned one-hot --------------
binned = keras.Sequential([disc, keras.layers.CategoryEncoding(num_tokens=5)])
print(f"\\n  ages {sample} one-hot by bin:")
print(f"{np.array2string(binned(sample).numpy(), prefix='    ')}")

# ============ 4. adapt ON THE TRAINING SET ONLY ========================
print("\\n=== adapt leaks if you use the full dataset ===")
Xtr, Xte = X[:4000], X[4000:]
n_leak = keras.layers.Normalization(); n_leak.adapt(X)          # WRONG
n_ok   = keras.layers.Normalization(); n_ok.adapt(Xtr)          # right
print(f"  adapted on all data  : mean {n_leak.mean.numpy().ravel()[1]:.4f}")
print(f"  adapted on train only: mean {n_ok.mean.numpy().ravel()[1]:.4f}")
print(f"  difference is small here, but it is still leakage (section 2.4)")

# --- adapting on a SAMPLE is fine and much faster --------------------
big = rng.normal(0, 1, (400_000, 20)).astype("float32")
t0 = time.perf_counter(); a = keras.layers.Normalization(); a.adapt(big)
t_full = time.perf_counter()-t0
t0 = time.perf_counter(); b = keras.layers.Normalization(); b.adapt(big[:20_000])
t_samp = time.perf_counter()-t0
print(f"\\n  adapt on 400,000 rows: {t_full:.3f}s")
print(f"  adapt on  20,000 rows: {t_samp:.3f}s   "
      f"mean differs by {np.abs(a.mean.numpy()-b.mean.numpy()).max():.5f}")

# ============ 5. THE PRODUCTION PATTERN ================================
print("\\n" + "="*62)
print("Preprocess in tf.data for SPEED, wrap in layers for SAFETY")
print("="*62)
y = (X[:, 0]*.5 + X[:, 1]*.001 + rng.normal(0, .2, 5000)).astype("float32")
ytr, yte = y[:4000], y[4000:]

norm = keras.layers.Normalization(); norm.adapt(Xtr)

# --- train FAST: normalise in the pipeline, cached -------------------
# Note we re-express the layer as plain TF ops here. That is exactly the
# duplication that creates training/serving skew -- and exactly why we wrap
# the trained model with the real layer for export, below.
_mean = tf.constant(norm.mean.numpy().ravel())
_std  = tf.constant(np.sqrt(norm.variance.numpy().ravel()) + 1e-7)

def normalise(x, t):
    return (x - _mean) / _std, t

train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
            .map(normalise, num_parallel_calls=tf.data.AUTOTUNE)
            .cache().shuffle(1000, seed=42).batch(32).prefetch(tf.data.AUTOTUNE))

core = keras.Sequential([keras.layers.Input(shape=(3,)),
                         keras.layers.Dense(32, activation="relu"),
                         keras.layers.Dense(1)])
core.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-2))
core.fit(train_ds, epochs=20, verbose=0)
print(f"\\n  core model expects NORMALISED input")
print(f"    on normalised test data: MSE "
      f"{np.ravel(core.evaluate(norm(Xte), yte, verbose=0))[0]:.5f}")
print(f"    on RAW test data       : MSE "
      f"{np.ravel(core.evaluate(Xte, yte, verbose=0))[0]:.5f}   <- garbage")

# --- export SAFE: wrap the trained model with the same layer ---------
serving_model = keras.Sequential([keras.layers.Input(shape=(3,)), norm, core])
serving_model.compile(loss="mse")          # needed before evaluate()
print(f"\\n  serving model takes RAW input")
print(f"    on RAW test data       : MSE "
      f"{np.ravel(serving_model.evaluate(Xte, yte, verbose=0))[0]:.5f}   <- correct")

serving_model.save(tmp/"serving.keras")
reloaded = keras.models.load_model(tmp/"serving.keras")
print(f"\\n  saved and reloaded; the normalisation statistics came with it:")
print(f"    reloaded mean = "
      f"{reloaded.layers[0].mean.numpy().ravel().round(3)}")
print(f"    raw-input predictions identical: "
      f"{np.allclose(serving_model.predict(Xte[:20], verbose=0), reloaded.predict(Xte[:20], verbose=0))}")
print("\\n  NO separate preprocessing code exists in production. No skew possible.")

shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch13_preproc",
    )

    keypoints([
        "Preprocessing layers ship <b>inside</b> the model, eliminating "
        "training/serving skew.",
        "<code>adapt()</code> is the layer's <code>fit</code> — call it on the "
        "<b>training set only</b> (a sample is fine).",
        "<code>Normalization</code>, <code>Discretization</code>, "
        "<code>CategoryEncoding</code> cover the numeric cases.",
        "Preprocessing in <code>tf.data</code> is faster (CPU, parallel, "
        "cacheable); in a layer it is safer.",
        "<b>Do both:</b> pipeline for training, then wrap the trained model with "
        "the layers for export.",
    ])


# ==========================================================================
def s_13_5():
    section("13.5", "Categorical Features — Lookup, Hashing, Embeddings")

    lead(
        "One-hot encoding a feature with 50 000 categories gives you a 50 "
        "000-column sparse matrix. Embeddings replace it with a dense vector of "
        "50, learned by the model."
    )

    sub("StringLookup and IntegerLookup")

    md(
        "Build a vocabulary and map each category to an index. Two special "
        "buckets matter:"
    )

    table(
        ["Concept", "Meaning", "Argument"],
        [["<b>OOV bucket</b>", "Where unknown categories go at inference time",
          "<code>num_oov_indices</code> (default 1, at index 0)"],
         ["<b>Mask token</b>", "A reserved index meaning 'padding'",
          "<code>mask_token=''</code> → index 0"],
         ["<code>output_mode</code>",
          "<code>'int'</code> (index), <code>'one_hot'</code>, "
          "<code>'multi_hot'</code>, <code>'tf_idf'</code>", ""],
         ["<code>max_tokens</code>",
          "Keep only the $N$ most frequent; the rest go to OOV", ""]],
    )

    warn(
        "Always keep at least one OOV bucket",
        "Production data <b>will</b> contain categories your training set never "
        "saw — a new city, a new product code, a typo. Without an OOV bucket the "
        "lookup raises an error and your service returns a 500. With "
        "<code>num_oov_indices &gt; 1</code>, unknown values are hashed across "
        "several buckets, which stops all unknowns from collapsing into one "
        "meaningless embedding.",
    )

    sub("Hashing — no vocabulary at all")

    math(r"""
    \mathrm{index}(c) \;=\; \mathrm{hash}(c) \bmod B
    """)

    md(
        "The **hashing trick** needs no `adapt` and no stored vocabulary — which "
        "means it handles unbounded, streaming or unseen categories with no code "
        "changes. The cost is **collisions**: two different categories can land "
        "in the same bucket and become indistinguishable."
    )

    derive(
        [("With $n$ distinct categories and $B$ buckets, treat the hash as "
          "uniform. The probability that a given category shares its bucket with "
          "no other:",
          r"\Pr[\text{no collision for one category}] = "
          r"\left(1 - \frac{1}{B}\right)^{n-1} \approx e^{-(n-1)/B}"),
         ("So the expected fraction of categories that collide is:",
          r"P_{\text{collision}} \approx 1 - e^{-n/B}"),
         ("Inverting, to keep the collision rate below $p$ you need:",
          r"B \;\ge\; \frac{-n}{\ln(1-p)}"),
         ("For $n = 10\\,000$ categories and a 1 % collision target: "
          "$B \\ge 10000/0.01005 \\approx 995\\,000$ buckets. For a 10 % target, "
          "$B \\approx 95\\,000$. <b>The rule of thumb is $B \\approx 10n$ to "
          "$100n$</b> — hashing saves you the vocabulary, not the dimensions.",
          None),
         ("<b>When hashing is nevertheless right:</b> when $n$ is unknown or "
          "unbounded, when categories arrive in a stream, when you cannot afford "
          "to store or synchronise a vocabulary across serving replicas, or when "
          "collisions are tolerable because the feature is one of many.", None)],
        title="How many hash buckets do you need?",
    )

    sub("Embeddings")

    md(
        "An embedding is a **trainable lookup table**: category $i$ maps to row "
        "$i$ of a matrix $\\mathbf{E} \\in \\mathbb{R}^{V \\times d}$, and those "
        "rows are learned by backpropagation like any other weights."
    )

    math(r"""
    \mathbf{e}_i \;=\; \mathbf{E}_{i,:}
    \qquad\Longleftrightarrow\qquad
    \mathbf{e} \;=\; \mathbf{E}^\top \mathbf{o}_i
    """)
    where({r"\mathbf{o}_i": "the one-hot vector for category $i$",
           r"V": "vocabulary size", r"d": "embedding dimension"})

    proof(
        "An embedding IS a Dense layer on a one-hot input",
        "$\\mathbf{E}^\\top\\mathbf{o}_i$ selects the $i$-th row, so an "
        "<code>Embedding</code> layer computes exactly what a bias-free "
        "<code>Dense(d)</code> layer would compute on a one-hot input. The "
        "difference is purely computational: the embedding does an <b>array "
        "index</b>, $\\mathcal{O}(d)$, while the Dense layer does a "
        "<b>matrix multiply</b> against a mostly-zero vector, "
        "$\\mathcal{O}(Vd)$. For $V = 50\\,000$ that is a 50 000× difference in "
        "work for an identical result — which is the entire reason embeddings "
        "exist as a separate layer.",
    )

    table(
        ["Rule of thumb for $d$", "Formula", "$V = 100$", "$V = 10^4$", "$V = 10^6$"],
        [["Fourth root", "$d \\approx V^{1/4}$", "4", "10", "32"],
         ["Half the vocabulary, capped", "$d = \\min(50, (V+1)/2)$", "50", "50",
          "50"],
         ["Google's rule", "$d \\approx V^{0.25}$ to $V^{0.4}$", "4–6", "10–40",
          "32–250"]],
        "In practice: start at 8–50 and treat it as a hyperparameter.",
    )

    anim_header("Collision rate as the number of buckets grows")

    n_cats = 10000
    Bs = np.logspace(2, 6.3, 60)
    p_coll = 1 - np.exp(-n_cats / Bs)

    frames = []
    for k in range(2, len(Bs) + 1):
        frames.append(go.Frame(name=f"{Bs[k-1]:.0f}", data=[
            go.Scatter(x=Bs[:k], y=p_coll[:k], mode="lines",
                       line=dict(color=C["primary"], width=3.6)),
            go.Scatter(x=[Bs[k - 1]], y=[p_coll[k - 1]], mode="markers",
                       marker=dict(color=C["danger"], size=14,
                                   line=dict(color="#fff", width=2))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{int(Bs[k-1]):,} buckets for {n_cats:,} categories   ·   "
            f"collision rate {p_coll[k-1]:.1%}   ·   "
            f"B/n = {Bs[k-1]/n_cats:.1f}",
            color=C["danger"] if p_coll[k - 1] > .1 else C["success"])])))

    f = go.Figure(data=[
        go.Scatter(x=Bs[:2], y=p_coll[:2], mode="lines",
                   name="1 − exp(−n/B)", line=dict(color=C["primary"], width=3.6)),
        go.Scatter(x=Bs[:1], y=p_coll[:1], mode="markers", showlegend=False,
                   marker=dict(color=C["danger"], size=14,
                               line=dict(color="#fff", width=2))),
    ])
    for lvl, lab in [(.5, "50 %"), (.1, "10 %"), (.01, "1 %")]:
        f.add_hline(y=lvl, line_dash="dot", line_color=C["muted"],
                    annotation_text=lab)
    f.add_vline(x=n_cats, line_dash="dash", line_color=C["warning"],
                annotation_text="B = n")
    f.update_layout(height=430, xaxis_type="log", xaxis_title="number of buckets B",
                    yaxis_title="fraction of categories that collide",
                    title=f"Hashing {n_cats:,} categories",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(90), slider_prefix="B = ")
    figure(f, "To get below 10 % collisions you need roughly 10× as many buckets "
              "as categories.")

    code_lab(
        "Lookup, hashing collisions, and learned embeddings",
        '''import numpy as np, tempfile, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)
tmp = Path(tempfile.mkdtemp(prefix="mlplat_ch13d_"))
rng = np.random.default_rng(0)

# ============ 1. StringLookup ==========================================
print("=== StringLookup ===")
cities = ["Paris", "Tokyo", "Cairo", "Lima", "Oslo"]
lookup = keras.layers.StringLookup()
lookup.adapt(cities)
print(f"  vocabulary: {lookup.get_vocabulary()}")
print(f"           (index 0 is the OOV bucket)")
test = tf.constant(["Tokyo", "Lima", "Atlantis", "Paris"])
print(f"  {test.numpy()} -> {lookup(test).numpy()}")
print(f"  'Atlantis' was never seen -> index 0 (OOV)")

# --- more OOV buckets spread the unknowns ---------------------------
multi = keras.layers.StringLookup(num_oov_indices=5)
multi.adapt(cities)
unknown = tf.constant(["Atlantis", "Gotham", "Narnia", "Utopia", "Xanadu",
                       "Camelot", "Shangri-La"])
print(f"\\n  with num_oov_indices=5, unknowns hash across buckets:")
print(f"    {list(unknown.numpy())} -> {multi(unknown).numpy()}")

# --- one-hot straight out of the layer -------------------------------
oh = keras.layers.StringLookup(output_mode="one_hot")
oh.adapt(cities)
print(f"\\n  output_mode='one_hot' shape: {tuple(oh(test).shape)}")

# --- inverse lookup ---------------------------------------------------
inv = keras.layers.StringLookup(vocabulary=lookup.get_vocabulary(), invert=True)
print(f"  inverse: {lookup(test).numpy()} -> {inv(lookup(test)).numpy()}")

# ============ 2. HASHING ===============================================
print("\\n=== Hashing: no vocabulary needed ===")
hasher = keras.layers.Hashing(num_bins=8)
print(f"  {list(test.numpy())} -> {hasher(test).numpy()}")
print(f"  same input, same bucket, no adapt() ever called")
print(f"  a brand-new category also works: "
      f"{hasher(tf.constant(['Reykjavik'])).numpy()}")

# --- measure the collision rate --------------------------------------
print("\\n=== collisions: theory vs measurement ===")
n_cats = 5000
words = tf.constant([f"category_{i:05d}" for i in range(n_cats)])
print(f"  {n_cats:,} distinct categories")
print(f"{'buckets':>10}{'B/n':>8}{'theory 1-e^(-n/B)':>21}{'measured':>12}")
for B in [500, 2500, 5000, 25000, 100000]:
    h = keras.layers.Hashing(num_bins=B)
    idx = h(words).numpy()
    _, counts = np.unique(idx, return_counts=True)
    collided = int(np.sum(counts[counts > 1]))
    theory = 1 - np.exp(-n_cats / B)
    print(f"{B:>10,}{B/n_cats:>8.1f}{theory:>21.1%}{collided/n_cats:>12.1%}")
print("  rule of thumb: B ~ 10n for ~10 % collisions, B ~ 100n for ~1 %")

# ============ 3. EMBEDDINGS ============================================
print("\\n=== an Embedding IS a Dense layer on a one-hot input ===")
V, d = 6, 4
emb = keras.layers.Embedding(input_dim=V, output_dim=d)
emb.build((None,))
E = emb.embeddings.numpy()
idx = tf.constant([0, 3, 5])
via_lookup = emb(idx).numpy()
one_hot = tf.one_hot(idx, V).numpy()
via_matmul = one_hot @ E
print(f"  embedding matrix E: {E.shape}")
print(f"  emb([0,3,5])        first row = {via_lookup[0].round(4)}")
print(f"  (one_hot @ E)       first row = {via_matmul[0].round(4)}")
print(f"  identical: {np.allclose(via_lookup, via_matmul)}")
print(f"\\n  cost: lookup is O(d)={d} ops; matmul is O(V*d)={V*d} ops")
print(f"  for V=50,000 that is a {50000*d/d:.0f}x difference for the SAME result")

# ============ 4. THE FULL CATEGORICAL PIPELINE =========================
print("\\n" + "="*62)
print("A real categorical pipeline: lookup -> embedding -> model")
print("="*62)
N_CITIES, N = 200, 8000
city_names = np.array([f"city_{i:03d}" for i in range(N_CITIES)])
city_effect = rng.normal(0, 2, N_CITIES)
city_idx = rng.integers(0, N_CITIES, N)
num_feat = rng.normal(0, 1, (N, 3)).astype("float32")
y = (num_feat[:, 0]*2 + city_effect[city_idx] + rng.normal(0, .4, N)).astype("float32")
cities_col = city_names[city_idx]

split = 6000
c_tr, c_te = cities_col[:split], cities_col[split:]
n_tr, n_te = num_feat[:split],   num_feat[split:]
y_tr, y_te = y[:split],          y[split:]

def build(mode, emb_dim=8, n_bins=64):
    city_in = keras.layers.Input(shape=(), dtype=tf.string, name="city")
    num_in  = keras.layers.Input(shape=(3,), name="numeric")
    if mode == "one_hot":
        lk = keras.layers.StringLookup(output_mode="one_hot")
        lk.adapt(tf.constant(c_tr))
        c = lk(city_in)
    elif mode == "embedding":
        lk = keras.layers.StringLookup()
        lk.adapt(tf.constant(c_tr))
        c = keras.layers.Embedding(lk.vocabulary_size(), emb_dim)(lk(city_in))
    elif mode == "hash_embedding":
        h = keras.layers.Hashing(num_bins=n_bins)
        c = keras.layers.Embedding(n_bins, emb_dim)(h(city_in))
    nrm = keras.layers.Normalization(); nrm.adapt(n_tr)
    z = keras.layers.Concatenate()([c, nrm(num_in)])
    z = keras.layers.Dense(32, activation="relu")(z)
    m = keras.Model([city_in, num_in], keras.layers.Dense(1)(z))
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-2))
    return m

print(f"{'encoding':<26}{'params':>10}{'test MSE':>12}")
models = {}
for nm, kw in [("one-hot (200 cols)", dict(mode="one_hot")),
               ("embedding dim 2",    dict(mode="embedding", emb_dim=2)),
               ("embedding dim 8",    dict(mode="embedding", emb_dim=8)),
               ("embedding dim 32",   dict(mode="embedding", emb_dim=32)),
               ("hash 64 + embedding", dict(mode="hash_embedding", n_bins=64)),
               ("hash 2000 + embedding", dict(mode="hash_embedding", n_bins=2000))]:
    tf.random.set_seed(0)
    m = build(**kw)
    m.fit((tf.constant(c_tr), n_tr), y_tr, epochs=25, batch_size=64,
          verbose=0)
    models[nm] = m
    print(f"{nm:<26}{m.count_params():>10,}"
          f"{np.ravel(m.evaluate((tf.constant(c_te), n_te), y_te, verbose=0))[0]:>12.4f}")
print("\\nhash 64 collides badly (200 cities into 64 buckets) and it shows.")

# ============ 5. WHAT THE EMBEDDING LEARNED ============================
print("\\n=== the embedding recovers the city effects ===")
m = models["embedding dim 8"]
emb_layer = [l for l in m.layers if isinstance(l, keras.layers.Embedding)][0]
lk_layer  = [l for l in m.layers if isinstance(l, keras.layers.StringLookup)][0]
E = emb_layer.embeddings.numpy()
vocab = lk_layer.get_vocabulary()
# correlate the first principal component of the embedding with the true effect
from sklearn.decomposition import PCA
pc1 = PCA(1).fit_transform(E[1:])[:, 0]          # skip the OOV row
true = np.array([city_effect[int(v.split("_")[1])] for v in vocab[1:]])
print(f"  |corr(PC1 of the embedding, true city effect)| = "
      f"{abs(np.corrcoef(pc1, true)[0,1]):.4f}")
print("  the model discovered the latent structure with no supervision on it")

# ============ 6. UNSEEN CATEGORIES AT SERVING TIME =====================
print("\\n=== a brand-new city arrives in production ===")
new = np.array(["city_999_BRAND_NEW"] * 3)
new_num = rng.normal(0, 1, (3, 3)).astype("float32")
print(f"  StringLookup + embedding: "
      f"{models['embedding dim 8'].predict((tf.constant(new), new_num), verbose=0).ravel().round(3)}"
      f"   <- OOV row, a sensible default")
print(f"  Hashing + embedding     : "
      f"{models['hash 2000 + embedding'].predict((tf.constant(new), new_num), verbose=0).ravel().round(3)}"
      f"   <- some bucket, no error")
print("  Neither crashes. Without an OOV bucket, StringLookup would.")

shutil.rmtree(tmp, ignore_errors=True)
''',
        key="ch13_embeddings",
    )

    quiz(
        "You have 10 000 product IDs and use <code>Hashing(num_bins=1000)</code>. "
        "Roughly what fraction of products collide with another?",
        ["About 10 %", "About 63 %", "About 100 %", "About 1 %"],
        1,
        "$1 - e^{-n/B} = 1 - e^{-10} \\approx 0.99995$ — essentially all of them. "
        "Even at $B = n$ the rate is $1 - e^{-1} \\approx 63\\%$. You need "
        "$B \\approx 10n$ for ~10 % and $B \\approx 100n$ for ~1 %.",
        key="ch13q2",
    )

    keypoints([
        "<code>StringLookup</code> builds a vocabulary; <b>always keep an OOV "
        "bucket</b> for production.",
        "<code>Hashing</code> needs no vocabulary but collides: "
        "$P \\approx 1 - e^{-n/B}$, so $B \\approx 10n$–$100n$.",
        "An <code>Embedding</code> is a trainable lookup table — mathematically a "
        "Dense layer on a one-hot input, computationally $V\\times$ cheaper.",
        "Embedding dimension: start at $V^{1/4}$ or 8–50, then tune it.",
        "Embeddings <b>learn latent structure</b> — similar categories end up "
        "with similar vectors.",
    ])


# ==========================================================================
def s_13_6():
    section("13.6", "Text Preprocessing")

    lead(
        "Turning strings into integer sequences. The layer does five things in "
        "order, and each one is a decision you can override."
    )

    sub("TextVectorization")

    table(
        ["Stage", "Default", "Argument"],
        [["1. <b>Standardise</b>", "Lowercase and strip punctuation",
          "<code>standardize=</code> — or a callable"],
         ["2. <b>Split</b>", "On whitespace",
          "<code>split=</code> — <code>'whitespace'</code>, "
          "<code>'character'</code>, or a callable"],
         ["3. <b>N-grams</b>", "None (unigrams)", "<code>ngrams=</code>"],
         ["4. <b>Index</b>", "By descending frequency",
          "<code>max_tokens=</code>, learned by <code>adapt</code>"],
         ["5. <b>Output</b>", "Integer sequences",
          "<code>output_mode=</code> — <code>'int'</code>, "
          "<code>'multi_hot'</code>, <code>'count'</code>, "
          "<code>'tf_idf'</code>"]],
    )

    md(
        "Index 0 is reserved for **padding**, index 1 for **OOV**. So a "
        "vocabulary of `max_tokens=1000` gives you 998 real words."
    )

    sub("output_mode — four different models")

    table(
        ["<code>output_mode</code>", "Output", "Suits"],
        [["<code>'int'</code>", "A padded sequence of indices",
          "RNNs, Transformers — anything order-sensitive (Ch. 15–16)"],
         ["<code>'multi_hot'</code>", "A $V$-length 0/1 bag of words",
          "Simple dense classifiers; ignores order and counts"],
         ["<code>'count'</code>", "A $V$-length vector of counts",
          "Bag of words with frequency"],
         ["<code>'tf_idf'</code>",
          "Counts weighted by inverse document frequency",
          "Classic text classification, often surprisingly strong"]],
    )

    math(r"""
    \mathrm{tfidf}(t, d) \;=\; \mathrm{tf}(t, d) \cdot
      \left(1 + \log\frac{1 + N}{1 + \mathrm{df}(t)}\right)
    """)
    where({r"\mathrm{tf}(t,d)": "how many times term $t$ appears in document $d$",
           r"\mathrm{df}(t)": "how many documents contain $t$",
           r"N": "total number of documents"})

    idea(
        "Why IDF works",
        "A term appearing in <b>every</b> document ($\\mathrm{df} = N$) has "
        "IDF $\\approx 1$ and carries almost no discriminative information — "
        "\"the\" tells you nothing about which document you are reading. A term "
        "in <b>one</b> document has a large IDF and is highly identifying. IDF is "
        "therefore a principled, corpus-derived alternative to a hand-written "
        "stop-word list.",
    )

    sub("Masking")

    md(
        "Padding creates fake zeros at the end of short sequences. Without "
        "masking, the model treats them as real tokens. `mask_zero=True` on the "
        "`Embedding` layer propagates a boolean mask forward so downstream RNN and "
        "attention layers skip those positions."
    )

    pitfall(
        "Padding without masking corrupts your model",
        "Consider a bag of 3-word reviews padded to length 100. Without masking, "
        "97 % of every input is the padding token, an RNN's final state is "
        "dominated by 97 steps of nothing, and mean-pooling divides by 100 instead "
        "of 3. Set <code>mask_zero=True</code> — and check that every downstream "
        "layer <b>supports masking</b>, because layers that do not will silently "
        "drop it.",
    )

    sub("Pretrained language model components")

    md(
        "For real NLP you rarely build the vocabulary yourself. TensorFlow Hub "
        "and Hugging Face provide pretrained embeddings and full models whose "
        "tokenisers come with them — Chapter 16 covers this properly. The point "
        "of `TextVectorization` is that you understand what those tokenisers are "
        "doing."
    )

    anim_header("Text through the five stages")

    doc = "The Movie was GREAT!! Truly great, and the ending was superb."
    stages = [
        ("raw text", doc),
        ("1. standardise (lowercase, strip punctuation)",
         "the movie was great truly great and the ending was superb"),
        ("2. split on whitespace",
         "['the','movie','was','great','truly','great','and','the','ending','was','superb']"),
        ("3. index by frequency rank (0=pad, 1=OOV)",
         "[2, 9, 3, 4, 11, 4, 8, 2, 10, 3, 12]"),
        ("4a. output_mode='int' (padded to 15)",
         "[2, 9, 3, 4, 11, 4, 8, 2, 10, 3, 12, 0, 0, 0, 0]"),
        ("4b. output_mode='multi_hot'",
         "[0,0,1,1,1,0,0,0,1,1,1,1,1]  — presence only, order lost"),
        ("4c. output_mode='count'",
         "[0,0,2,2,2,0,0,0,1,1,1,1,1]  — 'the' and 'was' and 'great' appear twice"),
        ("4d. output_mode='tf_idf'",
         "[0,0,0.4,0.4,1.8,0,0,0,1.1,1.1,1.6,1.6,1.6]  — common words down-weighted"),
    ]
    frames = []
    for i, (nm, val) in enumerate(stages):
        frames.append(go.Frame(name=str(i + 1), data=[
            go.Scatter(x=[0], y=[0], mode="markers",
                       marker=dict(size=1, color="rgba(0,0,0,0)"),
                       showlegend=False, hoverinfo="skip")],
            layout=go.Layout(
                title=nm,
                annotations=[dict(x=.5, y=.55, xref="paper", yref="paper",
                                  text=f"<b>{nm}</b>", showarrow=False,
                                  font=dict(size=13, color=C["primary_dark"])),
                             dict(x=.5, y=.30, xref="paper", yref="paper",
                                  text=val, showarrow=False,
                                  font=dict(size=11.5, color=C["ink"],
                                            family="JetBrains Mono, monospace"))])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=250, plot_bgcolor=C["surface_alt"],
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    annotations=list(frames[0].layout.annotations),
                    title=stages[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(1700), slider_prefix="stage ")
    figure(f)

    code_lab(
        "TextVectorization end to end, and why masking matters",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

corpus = tf.constant([
    "The movie was great!! Truly great, and the ending was superb.",
    "A dull, plodding film. I fell asleep.",
    "Great acting, great script, great everything.",
    "I have never been so bored.",
    "Superb photography and a script with real teeth.",
    "Two hours I will never get back.",
])
labels = np.array([1, 0, 1, 0, 1, 0], dtype="float32")

# ============ 1. THE FIVE STAGES =======================================
print("=== TextVectorization ===")
tv = keras.layers.TextVectorization(max_tokens=30, output_mode="int",
                                    output_sequence_length=12)
tv.adapt(corpus)
vocab = tv.get_vocabulary()
print(f"  vocabulary ({len(vocab)} tokens, ranked by frequency):")
print(f"    {vocab}")
print(f"    index 0 = '' (PAD), index 1 = '[UNK]' (OOV)")

out = tv(corpus[:2])
print(f"\\n  '{corpus[0].numpy().decode()[:50]}...'")
print(f"    -> {out[0].numpy()}")
print(f"  note: punctuation stripped, lowercased, padded with 0s to length 12")

# ============ 2. THE FOUR OUTPUT MODES =================================
print("\\n=== output_mode ===")
for mode in ["int", "multi_hot", "count", "tf_idf"]:
    kw = dict(max_tokens=15, output_mode=mode)
    if mode == "int":
        kw["output_sequence_length"] = 10
    layer = keras.layers.TextVectorization(**kw)
    layer.adapt(corpus)
    r = layer(corpus[:1]).numpy()[0]
    print(f"  {mode:<10} shape {r.shape}  {np.round(r, 2)}")
print("\\n  'int' keeps ORDER; the other three are bags of words.")

# ============ 3. CUSTOM STANDARDISATION AND SPLITTING ==================
print("\\n=== customising the stages ===")
def custom_standardize(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r"http\\S+", " URL ")
    text = tf.strings.regex_replace(text, r"\\d+", " NUMBER ")
    return tf.strings.regex_replace(text, r"[^a-z ]", " ")

messy = tf.constant(["Visit http://x.com for 50% off!!! Call 555-1234."])
print(f"  raw        : {messy.numpy()[0].decode()}")
print(f"  standardised: {custom_standardize(messy).numpy()[0].decode()}")

tv_custom = keras.layers.TextVectorization(standardize=custom_standardize,
                                           max_tokens=20)
tv_custom.adapt(tf.concat([corpus, messy], 0))
print(f"  vocabulary now contains URL/NUMBER tokens: "
      f"{[v for v in tv_custom.get_vocabulary() if v in ('url','number')]}")

# --- character-level and n-grams ------------------------------------
char = keras.layers.TextVectorization(split="character", max_tokens=30)
char.adapt(corpus)
print(f"\\n  split='character' vocabulary: {char.get_vocabulary()[:14]}...")
bigram = keras.layers.TextVectorization(ngrams=2, max_tokens=40,
                                        output_mode="multi_hot")
bigram.adapt(corpus)
print(f"  ngrams=2 vocabulary includes: "
      f"{[v for v in bigram.get_vocabulary() if ' ' in v][:5]}")

# ============ 4. TF-IDF, VERIFIED ======================================
print("\\n=== how IDF down-weights common words ===")
tfidf = keras.layers.TextVectorization(max_tokens=20, output_mode="tf_idf")
tfidf.adapt(corpus)
v = tfidf.get_vocabulary()
# Keras 3 no longer exposes .idf_weights, so compute the IDF ourselves from
# the same formula the layer uses:  1 + log((1+N) / (1+df))
docs = [c.numpy().decode().lower() for c in corpus]
N = len(docs)
df = np.array([sum(tok in doc.split() for doc in docs) for tok in v])
weights = 1 + np.log((1 + N) / (1 + df))
order = np.argsort(weights)
print(f"  {'token':<14}{'IDF weight':>12}{'appears in':>13}")
for i in np.r_[order[:4], order[-4:]]:
    print(f"  {v[i]:<14}{weights[i]:>12.4f}{int(df[i]):>10} docs")
print("  common words get a LOW weight; rare words get a HIGH one")

# ============ 5. MASKING ===============================================
print("\\n" + "="*62)
print("Why mask_zero=True matters")
print("="*62)
# build a dataset of VERY uneven lengths
short = ["good"] * 40 + ["bad"] * 40
long_ = [" ".join(["good"] * 25)] * 40 + [" ".join(["bad"] * 25)] * 40
texts = tf.constant(short + long_)
y = np.r_[np.ones(40), np.zeros(40), np.ones(40), np.zeros(40)].astype("float32")

vec = keras.layers.TextVectorization(max_tokens=10, output_sequence_length=30)
vec.adapt(texts)
X = vec(texts)
print(f"  sequence length 30; the short reviews are 1 real token + 29 pads")
print(f"  first short review : {X[0].numpy()}")

def build(mask):
    inp = keras.layers.Input(shape=(30,), dtype="int64")
    e = keras.layers.Embedding(10, 8, mask_zero=mask)(inp)
    z = keras.layers.GlobalAveragePooling1D()(e)
    m = keras.Model(inp, keras.layers.Dense(1, activation="sigmoid")(z))
    m.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-2),
              metrics=["accuracy"])
    return m

for mask in (False, True):
    tf.random.set_seed(0)
    m = build(mask)
    m.fit(X, y, epochs=40, batch_size=16, verbose=0)
    print(f"  mask_zero={str(mask):<6} accuracy "
          f"{m.evaluate(X, y, verbose=0)[1]:.4f}")
print("\\n  Without masking, GlobalAveragePooling divides by 30 for every review,")
print("  so a 1-token review's signal is diluted 30x by meaningless padding.")

# --- what the mask actually looks like -------------------------------
e_layer = keras.layers.Embedding(10, 4, mask_zero=True)
out = e_layer(X[:1])
print(f"\\n  the propagated mask for the first review:")
print(f"    {e_layer.compute_mask(X[:1]).numpy()[0]}")
print(f"    True = real token, False = padding to be ignored")

# ============ 6. A COMPLETE TEXT CLASSIFIER ============================
print("\\n=== end to end ===")
train_texts = tf.constant([
    "an absolute delight from start to finish", "a masterclass in tension",
    "gorgeous photography and a script with teeth", "warm funny and devastating",
    "i loved every minute of it", "beautifully acted and paced",
    "i have never been so bored", "the plot collapses and never recovers",
    "wooden acting and wincing dialogue", "two hours i will never get back",
    "dull plodding and utterly forgettable", "a complete waste of time",
])
train_y = np.array([1,1,1,1,1,1, 0,0,0,0,0,0], dtype="float32")

vec = keras.layers.TextVectorization(max_tokens=60, output_sequence_length=12)
vec.adapt(train_texts)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype=tf.string),
    vec,                                        # preprocessing INSIDE the model
    keras.layers.Embedding(vec.vocabulary_size(), 16, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-2),
              metrics=["accuracy"])
model.fit(tf.reshape(train_texts, (-1, 1)), train_y, epochs=60, verbose=0)

new = tf.constant([["a genuinely wonderful film"],
                   ["boring and completely forgettable"],
                   ["gorgeous and utterly devastating"]])
print(f"  the model takes RAW STRINGS -- no preprocessing code outside it")
for txt, p in zip(new.numpy().ravel(), model.predict(new, verbose=0).ravel()):
    print(f"    P(positive) = {p:.3f}   '{txt.decode()}'")
''',
        key="ch13_text",
    )

    keypoints([
        "<code>TextVectorization</code>: standardise → split → n-gram → index → "
        "output. Each stage is overridable.",
        "Index 0 = padding, index 1 = OOV, so <code>max_tokens=N</code> gives "
        "$N-2$ real words.",
        "<code>'int'</code> keeps order (RNNs, Transformers); the other modes are "
        "bags of words.",
        "TF-IDF down-weights terms that appear everywhere — a principled "
        "stop-word list.",
        "<b><code>mask_zero=True</code></b> or padding silently dominates short "
        "sequences.",
    ])


# ==========================================================================
def s_13_7():
    section("13.7", "Image Preprocessing and Augmentation")

    lead(
        "Two distinct jobs that look similar: <b>preprocessing</b> makes every "
        "image the right shape and range and applies at all times; "
        "<b>augmentation</b> injects randomness and applies <i>only during "
        "training</i>."
    )

    table(
        ["Layer", "Kind", "Active at inference?"],
        [["<code>Resizing(h, w)</code>", "preprocessing", "✅ always"],
         ["<code>Rescaling(1./255)</code>", "preprocessing", "✅ always"],
         ["<code>CenterCrop(h, w)</code>", "preprocessing", "✅ always"],
         ["<code>RandomFlip('horizontal')</code>", "augmentation",
          "❌ training only"],
         ["<code>RandomRotation(0.1)</code>", "augmentation", "❌"],
         ["<code>RandomZoom(0.2)</code>", "augmentation", "❌"],
         ["<code>RandomTranslation</code>, <code>RandomContrast</code>, "
          "<code>RandomBrightness</code>", "augmentation", "❌"]],
    )

    idea(
        "Augmentation is a way of injecting known invariances",
        "A horizontally flipped cat is still a cat. Rather than hoping the network "
        "discovers that from data, you <b>tell</b> it by showing flipped copies. "
        "This is the same mechanism as §3.9's shifted-MNIST exercise, and it is "
        "why augmentation is so effective on small datasets: you are adding "
        "<i>information</i> (an invariance) rather than just more samples.",
    )

    warn(
        "Only apply invariances that are actually true",
        "<code>RandomFlip('horizontal')</code> is correct for cats and wrong for "
        "text or digits — a horizontally flipped 2 is not a 2, and a flipped "
        "'b' is a 'd'. <code>RandomFlip('vertical')</code> is wrong for almost "
        "all natural photographs (the sky is at the top) but right for satellite "
        "or microscopy images. <b>Every augmentation encodes a claim about your "
        "domain; make sure the claim is true.</b>",
    )

    sub("Where to put augmentation")

    table(
        ["Placement", "Pro", "Con"],
        [["Inside the model, first layers",
          "Runs on the GPU; ships with the model; automatically disabled at "
          "inference",
          "Occupies GPU time that could be training; not cacheable"],
         ["In the <code>tf.data</code> pipeline",
          "Runs on the CPU in parallel with training; cheap",
          "You must remember to <b>not</b> apply it to the validation set"]],
    )

    tip(
        "Cache before augmenting, never after",
        "<code>cache()</code> memorises whatever came before it. If you cache "
        "<i>after</i> the augmentation, you freeze one particular set of random "
        "crops and flips and replay them every epoch — which is exactly the "
        "opposite of what augmentation is for. Order: "
        "<code>load → decode → resize → cache → shuffle → batch → augment → "
        "prefetch</code>.",
    )

    anim_header("The same image under increasing augmentation")

    @st.cache_data(show_spinner=False)
    def make_img():
        H = W = 48
        yy, xx = np.mgrid[0:H, 0:W] / H
        img = np.zeros((H, W, 3))
        img[..., 0] = .25 + .5 * xx
        img[..., 1] = .35 + .4 * yy
        img[..., 2] = .55
        disc = ((xx - .38) ** 2 + (yy - .40) ** 2) < .030
        img[disc] = [.95, .30, .20]
        bar = (np.abs(yy - .76) < .07) & (xx > .2) & (xx < .8)
        img[bar] = [.15, .80, .55]
        return np.clip(img, 0, 1)

    base_img = make_img()

    def transform(img, angle=0., dx=0., dy=0., flip=False, zoom=1., bright=0.):
        H, W = img.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        cy, cx = (H - 1) / 2, (W - 1) / 2
        ys, xs = (yy - cy) / zoom, (xx - cx) / zoom
        ca, sa = np.cos(angle), np.sin(angle)
        yr = ca * ys - sa * xs + cy - dy * H
        xr = sa * ys + ca * xs + cx - dx * W
        if flip:
            xr = W - 1 - xr
        yi = np.clip(np.round(yr), 0, H - 1).astype(int)
        xi = np.clip(np.round(xr), 0, W - 1).astype(int)
        return np.clip(img[yi, xi] + bright, 0, 1)

    rng = np.random.default_rng(1)
    variants = [("original (no augmentation)", dict())]
    for i in range(11):
        s = (i + 1) / 11
        variants.append((f"random augmentation, strength {s:.2f}",
                         dict(angle=rng.uniform(-.35, .35) * s,
                              dx=rng.uniform(-.15, .15) * s,
                              dy=rng.uniform(-.15, .15) * s,
                              flip=bool(rng.random() < .5 * s),
                              zoom=1 + rng.uniform(-.25, .25) * s,
                              bright=rng.uniform(-.2, .2) * s)))

    def u8(a):
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)

    frames = [go.Frame(name=str(i), data=[
        go.Image(z=u8(transform(base_img, **kw))),
        go.Image(z=u8(base_img)),
    ], layout=go.Layout(title=nm)) for i, (nm, kw) in enumerate(variants)]

    f = make_subplots(rows=1, cols=2, subplot_titles=("augmented", "original"))
    f.add_trace(go.Image(z=u8(base_img)), 1, 1)
    f.add_trace(go.Image(z=u8(base_img)), 1, 2)
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    f.update_layout(height=340, title=variants[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(700), slider_prefix="variant ")
    figure(f, "Rotation, translation, horizontal flip, zoom and brightness — each "
              "encodes a claim that the label is unchanged by that transformation.")

    code_lab(
        "Image pipelines, augmentation, and how much it actually buys",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)
rng = np.random.default_rng(0)

# ---- a small synthetic image dataset ---------------------------------
def make_dataset(n=1200, size=32):
    """Two classes: a bright disc (1) or a bright bar (0), at random positions."""
    X = np.zeros((n, size, size, 3), dtype="float32")
    y = np.zeros(n, dtype="float32")
    yy, xx = np.mgrid[0:size, 0:size] / size
    for i in range(n):
        img = np.stack([np.full((size, size), .2 + .3*rng.random()),
                        np.full((size, size), .3 + .2*rng.random()),
                        np.full((size, size), .5)], -1)
        cy, cx = rng.uniform(.25, .75, 2)
        if i % 2 == 0:
            mask = ((xx-cx)**2 + (yy-cy)**2) < .020        # disc
            y[i] = 1
        else:
            mask = (np.abs(yy-cy) < .07) & (np.abs(xx-cx) < .30)   # bar
            y[i] = 0
        img[mask] = [.95, .3, .2]
        X[i] = np.clip(img + rng.normal(0, .03, img.shape), 0, 1)
    return X, y

X, y = make_dataset(1200)
Xtr, ytr = X[:300], y[:300]                # deliberately SMALL training set
Xva, yva = X[300:600], y[300:600]
Xte, yte = X[600:], y[600:]
print(f"train {Xtr.shape}  valid {Xva.shape}  test {Xte.shape}")

# ============ 1. PREPROCESSING vs AUGMENTATION =========================
print("\\n=== preprocessing is always on; augmentation is training-only ===")
prep = keras.Sequential([keras.layers.Resizing(24, 24),
                         keras.layers.Rescaling(1./1.)], name="preprocess")
aug = keras.Sequential([keras.layers.RandomFlip("horizontal"),
                        keras.layers.RandomRotation(0.10),
                        keras.layers.RandomZoom(0.15),
                        keras.layers.RandomTranslation(0.1, 0.1),
                        keras.layers.RandomContrast(0.15)], name="augment")

sample = tf.constant(Xtr[:1])
p1, p2 = prep(sample, training=True), prep(sample, training=False)
print(f"  preprocessing: training vs inference identical? "
      f"{np.allclose(p1.numpy(), p2.numpy())}")
a1 = aug(sample, training=True).numpy()
a2 = aug(sample, training=True).numpy()
a3 = aug(sample, training=False).numpy()
print(f"  augmentation : two training passes identical? {np.allclose(a1, a2)}")
print(f"                 inference pass == original?    "
      f"{np.allclose(a3, sample.numpy())}")

# ============ 2. HOW MUCH DOES AUGMENTATION BUY? =======================
print("\\n=== 300 training images, with and without augmentation ===")
def build(use_aug):
    layers = [keras.layers.Input(shape=(32, 32, 3))]
    if use_aug:
        layers.append(aug)
    layers += [
        keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
    m = keras.Sequential(layers)
    m.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-3),
              metrics=["accuracy"])
    return m

print(f"{'setup':<26}{'train acc':>12}{'valid acc':>12}{'test acc':>11}{'gap':>9}")
for nm, use in [("no augmentation", False), ("with augmentation", True)]:
    tf.random.set_seed(0)
    m = build(use)
    m.fit(Xtr, ytr, epochs=40, batch_size=32, validation_data=(Xva, yva),
          verbose=0)
    tr = m.evaluate(Xtr, ytr, verbose=0)[1]
    va = m.evaluate(Xva, yva, verbose=0)[1]
    te = m.evaluate(Xte, yte, verbose=0)[1]
    print(f"{nm:<26}{tr:>12.4f}{va:>12.4f}{te:>11.4f}{tr-te:>9.4f}")
print("\\nAugmentation closes the train/test gap -- it is a regulariser.")

# ============ 3. AUGMENT IN THE PIPELINE (CPU) INSTEAD =================
print("\\n=== augmenting on the CPU, in tf.data ===")
train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
            .cache()                                   # BEFORE augmentation
            .shuffle(300, seed=42)
            .batch(32)
            .map(lambda x, t: (aug(x, training=True), t),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))
valid_ds = (tf.data.Dataset.from_tensor_slices((Xva, yva))
            .batch(32).cache().prefetch(tf.data.AUTOTUNE))   # NO augmentation

tf.random.set_seed(0)
m = build(False)                       # no augmentation layer inside the model
m.fit(train_ds, epochs=40, validation_data=valid_ds, verbose=0)
print(f"  pipeline augmentation test accuracy: "
      f"{m.evaluate(Xte, yte, verbose=0)[1]:.4f}")
print("  note valid_ds is NOT augmented -- that is your responsibility here")

# ============ 4. THE CACHE-ORDER TRAP ==================================
print("\\n=== cache AFTER augmentation freezes the randomness ===")
def first_batch_twice(pipeline):
    it = iter(pipeline)
    a = next(it)[0].numpy()
    it = iter(pipeline)
    b = next(it)[0].numpy()
    return np.allclose(a, b)

good = (tf.data.Dataset.from_tensor_slices((Xtr, ytr)).cache().batch(32)
        .map(lambda x, t: (aug(x, training=True), t)))
bad  = (tf.data.Dataset.from_tensor_slices((Xtr, ytr)).batch(32)
        .map(lambda x, t: (aug(x, training=True), t)).cache())
print(f"  cache BEFORE augment: epochs identical? {first_batch_twice(good)}"
      f"   <- correct, fresh randomness")
print(f"  cache AFTER  augment: epochs identical? {first_batch_twice(bad)}"
      f"   <- WRONG, one frozen set of augmentations")

# ============ 5. INVARIANCES THAT ARE NOT TRUE =========================
print("\\n=== an augmentation encodes a CLAIM about your domain ===")
claims = [
    ("RandomFlip('horizontal')", "cats, cars, faces", "digits, text, 'b' vs 'd'"),
    ("RandomFlip('vertical')",   "satellite, microscopy", "photographs (sky is up)"),
    ("RandomRotation(0.5)",      "microscopy, astronomy", "digits (6 vs 9), road signs"),
    ("RandomContrast",           "almost everything", "medical images with calibrated intensity"),
    ("RandomCrop",               "large natural images", "images where the border matters"),
]
print(f"  {'augmentation':<28}{'valid for':<26}{'INVALID for'}")
for a, ok, bad_ in claims:
    print(f"  {a:<28}{ok:<26}{bad_}")

# ============ 6. TIMING: GPU-side vs CPU-side ==========================
print("\\n=== where should augmentation run? ===")
big_X = np.repeat(Xtr, 8, axis=0)
big_y = np.repeat(ytr, 8, axis=0)
for nm, pipeline, model_builder in [
    ("augment inside the model",
     tf.data.Dataset.from_tensor_slices((big_X, big_y)).batch(32).prefetch(tf.data.AUTOTUNE),
     lambda: build(True)),
    ("augment in tf.data (CPU)",
     tf.data.Dataset.from_tensor_slices((big_X, big_y)).batch(32)
       .map(lambda x, t: (aug(x, training=True), t),
            num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE),
     lambda: build(False)),
]:
    tf.random.set_seed(0)
    m = model_builder()
    t0 = time.perf_counter()
    m.fit(pipeline, epochs=3, verbose=0)
    print(f"  {nm:<28} {time.perf_counter()-t0:.2f}s for 3 epochs")
print("  on a CPU-only machine the difference is small; on a GPU, moving")
print("  augmentation off the accelerator can matter a great deal")
''',
        key="ch13_images",
    )

    keypoints([
        "<b>Preprocessing</b> (resize, rescale, crop) always applies; "
        "<b>augmentation</b> is training-only.",
        "Augmentation injects a <b>known invariance</b> — it adds information, "
        "not just samples.",
        "Every augmentation is a claim about your domain; a flipped digit is not "
        "the same digit.",
        "<b>Cache before augmenting</b>, never after, or you freeze one set of "
        "random transforms.",
        "In-model augmentation is safe and GPU-side; pipeline augmentation is "
        "faster but you must exclude the validation set yourself.",
    ])


# ==========================================================================
def s_13_8():
    section("13.8", "The TensorFlow Datasets Project")

    lead(
        "TFDS gives you hundreds of standard datasets, already sharded into "
        "TFRecords, already split, with the download and checksum handling done."
    )

    md(
        """
```python
import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]

# or, more usefully:
train_set, valid_set, test_set = tfds.load(
    name="mnist",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True,          # yields (image, label) tuples
    shuffle_files=True,
)
```
        """
    )

    table(
        ["Argument", "Effect"],
        [["<code>split=</code>",
          "Slicing syntax: <code>'train[:90%]'</code>, "
          "<code>'train[:1000]'</code>, <code>'train+test'</code>"],
         ["<code>as_supervised=True</code>",
          "Yields <code>(features, label)</code> tuples instead of a dict — what "
          "<code>fit</code> expects"],
         ["<code>shuffle_files=True</code>",
          "Shuffles the shard order; combine with a shuffle buffer (§13.2)"],
         ["<code>with_info=True</code>",
          "Also returns metadata: class names, splits, citation, size"],
         ["<code>batch_size=</code>",
          "Batch during loading; usually better to batch yourself"],
         ["<code>data_dir=</code>", "Where to cache the downloaded files"]],
    )

    codenote(
        "The datasets are already sharded TFRecords",
        "Which means §13.2's advice applies directly: "
        "<code>shuffle_files=True</code> plus a modest shuffle buffer gives good "
        "randomisation, because each shard is a different region of the data and "
        "TFDS reads several at once.",
    )

    note(
        "TFDS is a separate package",
        "<code>pip install tensorflow-datasets</code>. It is not part of "
        "TensorFlow itself, and it downloads on first use — so a first "
        "<code>tfds.load</code> in a fresh environment can take minutes. Use "
        "<code>data_dir</code> to point it at a shared cache in a team setting.",
    )

    sub("The standard TFDS pipeline")

    md(
        """
```python
def preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    return tf.keras.applications.resnet50.preprocess_input(image), label

train_set = (train_set
             .repeat()
             .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
             .shuffle(10_000, seed=42)
             .batch(32)
             .prefetch(tf.data.AUTOTUNE))
```
        """
    )

    anim_header("The complete input pipeline, assembled")

    steps = [
        ("tfds.load(shuffle_files=True)", "shard order randomised", C["accent"]),
        (".interleave / parallel reads", "several shards read at once", SEQ[0]),
        (".map(decode + resize)", "AUTOTUNE parallel CPU work", SEQ[1]),
        (".cache()", "expensive work done once", SEQ[2]),
        (".shuffle(10_000)", "item-level randomisation", SEQ[3]),
        (".repeat()", "endless stream", SEQ[4]),
        (".batch(32)", "grouped for the GPU", SEQ[5]),
        (".map(augment)", "vectorised over the batch", SEQ[6]),
        (".prefetch(AUTOTUNE)", "CPU runs ahead of the GPU", C["success"]),
    ]
    frames = []
    for k in range(1, len(steps) + 1):
        shapes, ann = [], []
        for i, (nm, desc, col) in enumerate(steps):
            on = i < k
            cur = i == k - 1
            y = len(steps) - i
            shapes.append(go.Scatter(
                x=[0, 4.6, 4.6, 0, 0],
                y=[y - .38, y - .38, y + .38, y + .38, y - .38],
                fill="toself",
                fillcolor=(alpha(col, .9) if cur else alpha(col, .35) if on
                           else alpha(C["line"], .28)),
                line=dict(color="#fff" if on else C["line"], width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=.15, y=y, text=nm, showarrow=False, xanchor="left",
                            font=dict(size=10.5,
                                      color="#fff" if on else C["muted"],
                                      family="JetBrains Mono, monospace")))
            if on:
                ann.append(dict(x=4.85, y=y, text=desc, showarrow=False,
                                xanchor="left",
                                font=dict(size=9.5, color=C["ink_soft"])))
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann,
                                                title=f"{k}. {steps[k-1][0]}")))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=500, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.15, 12]),
                    yaxis=dict(visible=False, range=[.3, 10]),
                    annotations=list(frames[0].layout.annotations),
                    title=f"1. {steps[0][0]}")
    anim.animate(f, frames, duration=nav.anim_ms(950), slider_prefix="step ")
    figure(f)

    code_lab(
        "TFDS if available; otherwise the identical pipeline on local data",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. TRY TFDS ==============================================
print("=== tensorflow_datasets ===")
HAVE_TFDS = False
try:
    import tensorflow_datasets as tfds
    HAVE_TFDS = True
    print(f"  tensorflow_datasets {tfds.__version__} is installed")
    print(f"  {len(tfds.list_builders())} datasets available, e.g.:")
    for name in tfds.list_builders()[:12]:
        print(f"    {name}")
except ImportError:
    print("  tensorflow_datasets is NOT installed")
    print("    pip install tensorflow-datasets")
    print("  Below we build the identical pipeline on a local dataset instead.")

if HAVE_TFDS:
    try:
        (train_raw, valid_raw, test_raw), info = tfds.load(
            "mnist", split=["train[:90%]", "train[90%:]", "test"],
            as_supervised=True, shuffle_files=True, with_info=True)
        print(f"\\n  loaded {info.name}: {info.description[:70]}...")
        print(f"  classes  : {info.features['label'].names}")
        print(f"  splits   : "
              f"{ {k: v.num_examples for k, v in info.splits.items()} }")
        print(f"  size     : {info.dataset_size}")
    except Exception as e:
        print(f"  (download failed: {type(e).__name__}) -- using local data")
        HAVE_TFDS = False

# ============ 2. THE SAME PIPELINE, LOCAL DATA =========================
if not HAVE_TFDS:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    Xi = (d.images / 16.0).astype("float32")[..., None]     # (n, 8, 8, 1)
    Xtr, Xte, ytr, yte = train_test_split(Xi, d.target.astype("int64"),
                                          test_size=.25, stratify=d.target,
                                          random_state=42)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=.2, random_state=42)
    train_raw = tf.data.Dataset.from_tensor_slices((Xtr, ytr))
    valid_raw = tf.data.Dataset.from_tensor_slices((Xva, yva))
    test_raw  = tf.data.Dataset.from_tensor_slices((Xte, yte))
    IMG, CH, N_CLASS = 8, 1, 10
    print(f"\\n  using the local 8x8 digits set: {len(Xtr)} train, "
          f"{len(Xva)} valid, {len(Xte)} test")
else:
    IMG, CH, N_CLASS = 28, 1, 10

# ============ 3. THE CANONICAL PIPELINE ================================
print("\\n=== assembling the canonical pipeline ===")
augment = keras.Sequential([keras.layers.RandomTranslation(0.08, 0.08),
                            keras.layers.RandomZoom(0.08)])

SCALE = 255.0 if HAVE_TFDS else 1.0     # a PYTHON constant, decided outside

def preprocess(image, label):
    # No Python `if` on a tensor here -- the branch is resolved at trace time
    return tf.cast(image, tf.float32) / SCALE, label

def make_pipeline(raw, training, batch=64, shuffle_buffer=2000):
    ds = raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()                                  # AFTER the expensive map
    if training:
        ds = ds.shuffle(shuffle_buffer, seed=42).repeat()
    ds = ds.batch(batch)
    if training:                                     # augment AFTER batching
        def _aug(x, y):
            return augment(x, training=True), y
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)             # ALWAYS last

train_ds = make_pipeline(train_raw, training=True)
valid_ds = make_pipeline(valid_raw, training=False)
test_ds  = make_pipeline(test_raw,  training=False)
print(f"  train element_spec: {train_ds.element_spec}")

# ============ 4. TRAIN ON IT ===========================================
model = keras.Sequential([
    keras.layers.Input(shape=(IMG, IMG, CH)),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(N_CLASS, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
steps = 30
t0 = time.perf_counter()
model.fit(train_ds, epochs=12, steps_per_epoch=steps,
          validation_data=valid_ds, verbose=0)
print(f"\\n  trained in {time.perf_counter()-t0:.1f}s")
print(f"  test accuracy = {model.evaluate(test_ds, verbose=0)[1]:.4f}")

# ============ 5. WHY .repeat() NEEDS steps_per_epoch ===================
print("\\n=== .repeat() makes the dataset INFINITE ===")
print(f"  cardinality of the training pipeline: "
      f"{tf.data.experimental.cardinality(train_ds).numpy()}   (-1 = INFINITE)")
print(f"  so fit() cannot know when an epoch ends -> pass steps_per_epoch")
print(f"  the alternative is to drop .repeat() and let the dataset end naturally")

# ============ 6. PIPELINE THROUGHPUT ===================================
print("\\n=== throughput of each variant ===")
raw_n = sum(1 for _ in train_raw)
variants = {
    "naive: map, batch":
        train_raw.map(preprocess).batch(64),
    "+ num_parallel_calls":
        train_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(64),
    "+ cache":
        train_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                 .cache().batch(64),
    "+ prefetch (the full recipe)":
        train_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                 .cache().batch(64).prefetch(tf.data.AUTOTUNE),
}
print(f"{'pipeline':<32}{'epoch 1':>11}{'epoch 2':>11}{'items/s (ep2)':>16}")
for nm, pipe in variants.items():
    t0 = time.perf_counter()
    for _ in pipe: pass
    e1 = time.perf_counter()-t0
    t0 = time.perf_counter()
    for _ in pipe: pass
    e2 = time.perf_counter()-t0
    print(f"{nm:<32}{e1:>10.3f}s{e2:>10.3f}s{raw_n/max(e2,1e-9):>16,.0f}")
''',
        key="ch13_tfds",
    )

    keypoints([
        "<code>tfds.load</code> gives sharded TFRecords, standard splits and "
        "metadata for hundreds of datasets.",
        "<code>as_supervised=True</code> yields <code>(x, y)</code> tuples; "
        "<code>split=</code> takes slicing syntax.",
        "<code>shuffle_files=True</code> plus a modest buffer is good "
        "randomisation because each shard is a different region.",
        "The canonical order: load → map → cache → shuffle → repeat → batch → "
        "augment → prefetch.",
        "<code>.repeat()</code> makes the dataset infinite, so "
        "<code>fit</code> needs <code>steps_per_epoch</code>.",
    ])


# ==========================================================================
def s_13_9():
    section("13.9", "Exercises & Chapter Review")

    lead("Ten exercises. Number 10 is a full pipeline project.")

    exercise(
        1, "Why would you want to use the tf.data API?",
        "Ingesting a large dataset and preprocessing it efficiently is a "
        "non-trivial engineering problem: you need **multithreading**, "
        "**queueing**, **batching**, **prefetching** and so on. The tf.data API "
        "does all of this for you, in a few lines of composable code.\n\n"
        "It also handles cases that would otherwise need bespoke work: reading "
        "from **multiple files in parallel** (interleaving), reading data that "
        "**does not fit in memory** by streaming it, and **overlapping CPU "
        "preprocessing with GPU training** so the accelerator is never idle "
        "(§13.2's animation).\n\n"
        "And crucially: it reads from many sources — tensors, generators, text "
        "files, TFRecords, SQL — with the same downstream API.")

    exercise(
        2, "What are the benefits of splitting a large dataset into multiple "
        "files?",
        "**(1) Shuffling at a coarse grain.** With many files you can shuffle the "
        "*file list* and interleave reads, which mixes distant regions of the data "
        "before the shuffle buffer even sees them (§13.2). A single huge file "
        "would need a shuffle buffer as large as the file.\n\n"
        "**(2) Parallel I/O.** Multiple files can be read simultaneously by "
        "several threads or machines, multiplying read throughput.\n\n"
        "**(3) Distributed training.** Different machines can be assigned "
        "different shards without any coordination.\n\n"
        "**(4) Manageability.** Downloading, moving, or replacing a corrupted "
        "100 MB shard is easy; doing so for a 500 GB file is not.")

    exercise(
        3, "During training, how can you tell that your input pipeline is the "
        "bottleneck? What can you do to fix it?",
        "**Diagnosis:** use TensorBoard's **profiler** to visualise the timeline "
        "of the GPU and the input pipeline. If the GPU shows gaps waiting for "
        "data, or is at low utilisation (`nvidia-smi` showing e.g. 20 %), the "
        "input pipeline is the bottleneck. A quick check: train on a tiny "
        "`cache()`d dataset — if throughput jumps, the pipeline was the "
        "constraint.\n\n"
        "**Fixes, in order of usual impact:**\n"
        "* Add `prefetch(tf.data.AUTOTUNE)` at the end.\n"
        "* Add `num_parallel_calls=tf.data.AUTOTUNE` to every `map`.\n"
        "* Add `cache()` after the expensive per-item work.\n"
        "* **Vectorise** the preprocessing: move it after `batch()`.\n"
        "* Read from **multiple files in parallel** with `interleave`.\n"
        "* Preprocess **once, ahead of time**, and save the result as TFRecords.\n"
        "* Buy faster storage (SSD/NVMe), more RAM, or more CPU cores.")

    exercise(
        4, "Can you save any binary data to a TFRecord file, or only serialized "
        "protocol buffers?",
        "**Any binary data.** A TFRecord file is just a sequence of "
        "length-prefixed byte strings with checksums (§13.3) — it has no idea "
        "what is inside them. You can write arbitrary bytes with "
        "`TFRecordWriter.write()`.\n\n"
        "That said, `tf.train.Example` protobufs are what everyone uses in "
        "practice, because TensorFlow has fast, parallel, C++-implemented parsing "
        "ops for them (`tf.io.parse_example`). Writing your own binary format "
        "means writing your own parser, which will almost certainly be slower and "
        "will not be usable inside a `tf.function`.")

    exercise(
        5, "Why would you go through the hassle of converting all your data to "
        "the Example protobuf format? Why not use your own protobuf definition?",
        "Because `tf.train.Example` is **directly supported by TensorFlow's "
        "parsing ops** — `tf.io.parse_example` and `parse_single_example` are "
        "optimised C++ implementations that parse a whole batch in parallel and "
        "work inside a `tf.function`.\n\n"
        "You *can* define your own protobuf, compile it with `protoc`, and use it. "
        "But then you need to ship the descriptor with the model, and you must "
        "parse it with `tf.io.decode_proto` — which is more general but "
        "significantly slower — or write a `tf.py_function` wrapper, which breaks "
        "graph mode entirely.\n\n"
        "The `Example` format is expressive enough for almost everything: three "
        "list types (bytes, float, int64) covers images, text, numbers and "
        "arrays.")

    exercise(
        6, "When using TFRecords, when would you want to activate compression? "
        "Why not do it systematically?",
        "**Activate it** when the files must be **downloaded over a network** — "
        "compression reduces transfer time, which usually dominates. Also when "
        "storage cost matters and the data compresses well (text and sparse "
        "features compress far better than already-compressed JPEGs).\n\n"
        "**Not systematically**, because compression costs **CPU time on every "
        "read**, at every epoch. If your data is on a fast local SSD, you have "
        "just converted an I/O-cheap pipeline into a CPU-bound one, and the CPU "
        "is exactly the resource your preprocessing already needs. The lab in "
        "§13.3 measures both sides of this trade.")

    exercise(
        7, "Data can be preprocessed directly when writing the data files, or "
        "within the tf.data pipeline, or in preprocessing layers within your "
        "model. Can you list a few pros and cons of each option?",
        "**When writing the data files (ahead of time)**\n"
        "* ✅ Training runs faster — no preprocessing at all during training.\n"
        "* ✅ Can reduce file size, and often reveals data problems early.\n"
        "* ❌ Hard to experiment: changing a preprocessing decision means "
        "regenerating the whole dataset.\n"
        "* ❌ **Training/serving skew**: the serving code must reimplement the "
        "identical transformation.\n"
        "* ❌ You cannot do data augmentation this way (it must vary per epoch).\n\n"
        "**In the tf.data pipeline**\n"
        "* ✅ Easy to tweak and experiment with.\n"
        "* ✅ Runs on the CPU in parallel with GPU training; `cache()`able so it "
        "happens once.\n"
        "* ✅ Augmentation works naturally.\n"
        "* ❌ Slows training slightly compared to preprocessing ahead of time.\n"
        "* ❌ **Still training/serving skew** unless you replicate it exactly.\n\n"
        "**In preprocessing layers inside the model**\n"
        "* ✅ **No skew whatsoever** — the transformation is part of the saved "
        "artefact.\n"
        "* ✅ One source of truth; a single file to deploy.\n"
        "* ❌ Slows training: it runs every epoch, and on the GPU where some ops "
        "vectorise poorly.\n"
        "* ❌ Not cacheable.\n\n"
        "**The production answer** (§13.4): preprocess in the pipeline for "
        "training speed, then wrap the trained model with the same layers for "
        "export.")

    exercise(
        8, "Can you name a few common ways you can encode categorical integer "
        "features? What about text?",
        "**Categorical integer features:**\n"
        "* **One-hot encoding** — fine for low cardinality (< ~50).\n"
        "* **Multi-hot / count encoding** — when an instance can have several "
        "values of the same feature.\n"
        "* **Embeddings** — a trainable dense vector per category; the standard "
        "for high cardinality (§13.5).\n"
        "* **Hashing** — no vocabulary needed, at the cost of collisions.\n"
        "* **Target / mean encoding** — replace the category with the mean target "
        "for that category. Powerful but leaks badly unless done out-of-fold "
        "(§2.8's exercise 4).\n"
        "* **Ordinal encoding** — only when a genuine order exists (§2.4).\n\n"
        "**Text:**\n"
        "* **Bag of words** — multi-hot or count vectors; order discarded.\n"
        "* **TF-IDF** — counts weighted by inverse document frequency.\n"
        "* **N-grams** — bags of word or character pairs/triples, which recovers "
        "some local order.\n"
        "* **Word embeddings** — Word2Vec, GloVe, or learned from scratch.\n"
        "* **Subword tokenisation** — BPE, WordPiece, SentencePiece; handles "
        "unknown words by splitting them (Chapter 16).\n"
        "* **Contextual embeddings** — BERT, GPT and friends, where the same word "
        "gets a different vector depending on context (Chapter 16).")

    exercise(
        9, "Load the Fashion MNIST dataset; split it into a training set, a "
        "validation set, and a test set; shuffle the training set; and save each "
        "dataset to multiple TFRecord files. Each record should be a serialized "
        "`Example` protobuf with two features: the serialized image (use "
        "`tf.io.serialize_tensor()`) and the label. Then use tf.data to create an "
        "efficient dataset for each set. Finally, use a Keras model to train these "
        "datasets, including a preprocessing layer to standardize each input "
        "feature. Try to make the input pipeline as efficient as possible, using "
        "TensorBoard to visualize profiling data.",
        "The full recipe:\n\n"
        "1. **Write the shards.** Open $N$ `TFRecordWriter`s and write records "
        "**round-robin** (`writers[i % n_shards]`), so each shard is already "
        "mixed and a small shuffle buffer suffices later.\n\n"
        "2. **Serialise the image.** `tf.io.serialize_tensor(image)` gives you a "
        "byte string; store it in a `BytesList`. On read, "
        "`tf.io.parse_tensor(data, out_type=tf.uint8)` recovers it — and you must "
        "then **`set_shape`** explicitly, because `parse_tensor` returns an "
        "unknown shape and Keras needs a static one.\n\n"
        "3. **Read efficiently.** "
        "`list_files → interleave(cycle_length=n_shards) → map(parse, AUTOTUNE) → "
        "cache → shuffle → batch → prefetch`.\n\n"
        "4. **Standardise with a layer.** `Normalization` with "
        "`adapt` on a sample of the training set, placed first in the model — so "
        "the statistics ship with the artefact.\n\n"
        "5. **Profile.** `TensorBoard(profile_batch=(100, 200))`, then open the "
        "*Profile* tab and look at the *Trace Viewer* and the *Input Pipeline "
        "Analyzer*, which will tell you outright what fraction of step time was "
        "spent waiting for data.\n\n"
        "The most common bug in this exercise: forgetting `set_shape` after "
        "`parse_tensor`, which produces a cryptic error deep inside the first "
        "`Dense` layer.",
        code='''def create_example(image, label):
    image_data = tf.io.serialize_tensor(image)
    return Example(features=Features(feature={
        "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),
        "label": Feature(int64_list=Int64List(value=[label])),
    }))

def write_tfrecords(name, dataset, n_shards=10):
    paths = [f"{name}.tfrecord-{i:05d}-of-{n_shards:05d}"
             for i in range(n_shards)]
    writers = [tf.io.TFRecordWriter(p) for p in paths]
    for index, (image, label) in dataset.enumerate():
        shard = index % n_shards
        writers[shard].write(create_example(image, label).SerializeToString())
    for w in writers:
        w.close()
    return paths

feature_descriptions = {
    "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}

def preprocess(tfrecord):
    example = tf.io.parse_single_example(tfrecord, feature_descriptions)
    image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
    image = tf.reshape(image, shape=[28, 28])          # REQUIRED
    return image, example["label"]

def mnist_dataset(filepaths, n_read_threads=5, shuffle_buffer_size=None,
                  n_parse_threads=5, batch_size=32, cache=True):
    dataset = tf.data.TFRecordDataset(filepaths,
                                      num_parallel_reads=n_read_threads)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)''')

    exercise(
        10, "In this exercise you will download a dataset, split it, create a "
        "`tf.data.Dataset` to load it and preprocess it efficiently, then build "
        "and train a binary classification model containing an `Embedding` layer.",
        "The IMDb reviews dataset is the standard target. The full pipeline:\n\n"
        "1. **Load** with `tfds.load('imdb_reviews', as_supervised=True)`, or "
        "download the raw text files and use "
        "`keras.utils.text_dataset_from_directory`.\n\n"
        "2. **Split** — IMDb ships with train/test only, so carve a validation "
        "set out of train with `split=['train[:90%]', 'train[90%:]', 'test']`.\n\n"
        "3. **Vectorise.** A `TextVectorization` layer with `max_tokens=10_000` "
        "and `output_sequence_length=250`. **`adapt` on the training text "
        "only.**\n\n"
        "4. **Embed and pool.** `Embedding(10_000, 128, mask_zero=True)` followed "
        "by `GlobalAveragePooling1D()`. This simple architecture reaches about "
        "**87 %** — a bag of embeddings, with order discarded.\n\n"
        "5. **Improve.** A bidirectional LSTM (Chapter 15) reaches ~89 %; a "
        "fine-tuned pretrained Transformer (Chapter 16) reaches **95 %+**.\n\n"
        "Two things worth doing: put the `TextVectorization` layer **inside the "
        "model** so it takes raw strings at serving time; and use "
        "`mask_zero=True`, or the 250-token padding will swamp short reviews "
        "(§13.6).",
        code='''import tensorflow_datasets as tfds

(train_raw, valid_raw, test_raw), info = tfds.load(
    "imdb_reviews", split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True, with_info=True)

vectorize = keras.layers.TextVectorization(
    max_tokens=10_000, output_sequence_length=250)
vectorize.adapt(train_raw.map(lambda text, label: text))   # TRAIN ONLY

def make(ds, training, batch=32):
    ds = ds.cache()
    if training:
        ds = ds.shuffle(10_000, seed=42)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    keras.layers.Input(shape=(), dtype=tf.string),
    vectorize,
    keras.layers.Embedding(vectorize.vocabulary_size(), 128, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.fit(make(train_raw, True), epochs=10,
          validation_data=make(valid_raw, False))''')

    rule()

    sub("The pipeline checklist")

    table(
        ["Check", "Why"],
        [["<code>num_parallel_calls=AUTOTUNE</code> on every "
          "<code>map</code>", "Otherwise preprocessing is single-threaded"],
         ["<code>prefetch(AUTOTUNE)</code> last", "Overlaps CPU and GPU"],
         ["<code>cache()</code> after the expensive maps, before shuffling",
          "Expensive work happens once; shuffling stays fresh"],
         ["Shuffle <b>items</b>, then batch", "Batching first is not shuffling"],
         ["Vectorisable preprocessing after <code>batch</code>",
          "10–30× fewer op invocations"],
         ["Data sharded, <code>interleave</code>d, round-robin written",
          "Parallel reads and good randomisation with a small buffer"],
         ["<code>adapt()</code> on the training set only",
          "Otherwise you leak test statistics"],
         ["Preprocessing layers in the exported model",
          "Eliminates training/serving skew"],
         ["Augmentation after <code>cache()</code>",
          "Otherwise the randomness is frozen"],
         ["Validation set <b>not</b> augmented",
          "You are measuring the wrong distribution otherwise"]],
    )

    keypoints([
        "The pipeline exists to keep the accelerator busy — profile before you "
        "optimise the model.",
        "Order matters: cache after expensive maps, shuffle items not batches, "
        "vectorise after batching, prefetch last.",
        "TFRecord + <code>Example</code> protobufs + sharding is the format that "
        "makes all of this fast.",
        "Preprocessing <b>layers</b> eliminate training/serving skew; "
        "<code>adapt</code> on training data only.",
        "Embeddings replace huge one-hot vectors with learned dense ones; hashing "
        "trades collisions for a vocabulary.",
    ], title="Chapter 13 in five lines")

    refs([
        ("TensorFlow — *Better performance with the tf.data API*",
         "https://www.tensorflow.org/guide/data_performance"),
        ("TensorFlow — *TFRecord and tf.train.Example*",
         "https://www.tensorflow.org/tutorials/load_data/tfrecord"),
        ("Weinberger et al. — *Feature Hashing for Large Scale Multitask "
         "Learning*", "ICML 2009"),
        ("Mikolov et al. — *Efficient Estimation of Word Representations in "
         "Vector Space* (Word2Vec)", "https://arxiv.org/abs/1301.3781"),
        ("Shorten & Khoshgoftaar — *A Survey on Image Data Augmentation for Deep "
         "Learning*", "https://doi.org/10.1186/s40537-019-0197-0"),
        ("TensorFlow Datasets catalogue",
         "https://www.tensorflow.org/datasets/catalog/overview"),
    ])


# ==========================================================================
SECTIONS = [
    ("13.1", "The tf.data API", s_13_1),
    ("13.2", "Shuffling, Interleaving, Prefetching", s_13_2),
    ("13.3", "TFRecord & Protobufs", s_13_3),
    ("13.4", "Keras Preprocessing Layers", s_13_4),
    ("13.5", "Categorical Features & Embeddings", s_13_5),
    ("13.6", "Text Preprocessing", s_13_6),
    ("13.7", "Image Preprocessing & Augmentation", s_13_7),
    ("13.8", "TensorFlow Datasets", s_13_8),
    ("13.9", "Exercises & Review", s_13_9),
]

nav.render_chapter(CH, SECTIONS)
