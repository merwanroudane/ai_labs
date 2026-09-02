"""Chapter 19 — Training and Deploying TensorFlow Models at Scale."""

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
CH = "ch19"

hero(
    kicker="Part II · Chapter 19",
    title="Training and Deploying at Scale",
    blurb=(
        "A model that only runs in your notebook is not finished. This chapter "
        "covers the SavedModel format, serving latency and batching, quantisation "
        "for phones and browsers, GPU memory arithmetic, the scaling laws of data "
        "and model parallelism, the Distribution Strategies API, and what happens "
        "to a deployed model over time."
    ),
    chips=["Scaling derived", "9 sub-sections", "9 animations",
           "9 code labs", "TFLite · strategies · drift"],
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
    st.warning("TensorFlow is not importable here, so the labs will report an "
               "ImportError. Every explanation and animation still works.",
               icon="⚠️")


# ==========================================================================
def s_19_1():
    section("19.1", "The SavedModel Format and Serving Signatures")

    lead(
        "A deployable model is not a <code>.h5</code> file and a Python script. "
        "It is a self-describing directory containing a computation graph, its "
        "weights, and a declared input/output contract."
    )

    sub("What a SavedModel contains")

    table(
        ["Path", "Contents", "Why it matters"],
        [["<code>saved_model.pb</code>",
          "The serialised <b>graph</b> and its signatures",
          "The graph is language-independent — no Python needed to run it"],
         ["<code>variables/</code>", "The weights, sharded",
          "Separated from the graph so weights can be updated independently"],
         ["<code>assets/</code>",
          "Vocabulary files, lookup tables",
          "Preprocessing state travels <b>with</b> the model"],
         ["<code>fingerprint.pb</code>", "A content hash",
          "Lets a server detect an identical model and skip a reload"]],
    )

    idea(
        "The point of the SavedModel is that it needs no Python",
        "TensorFlow Serving is a C++ binary. It loads the graph, allocates the "
        "weights, and executes — with no interpreter, no <code>pip install</code>, "
        "and no risk that your training environment's package versions differ "
        "from production's. That single property eliminates the most common class "
        "of deployment failure: <i>'it worked on my machine'</i>. The cost is that "
        "anything your model needs must be <b>inside the graph</b> — which is why "
        "preprocessing layers (§13.7) matter so much.",
    )

    sub("Signatures")

    md(
        "A **signature** is the model's public API: named inputs, named outputs, "
        "shapes and dtypes. `serving_default` is the one a server calls unless "
        "told otherwise."
    )

    pitfall(
        "Preprocessing must be inside the graph, or you will have "
        "training/serving skew",
        "If your notebook scales inputs with a <code>StandardScaler</code> and "
        "your server does not — or does it with slightly different statistics — "
        "the model receives inputs from a distribution it was never trained on, "
        "and it fails <b>silently</b> with plausible-looking wrong answers. "
        "This is the single most common production ML bug. The fix is to put the "
        "preprocessing in the model: a <code>Normalization</code> layer, a "
        "<code>TextVectorization</code> layer, a "
        "<code>StringLookup</code> — all of which serialise into the SavedModel "
        "with their learned state in <code>assets/</code>.",
    )

    codenote(
        "Export a separate serving signature that takes <i>raw</i> input",
        "The model you train may take a preprocessed tensor, but the model you "
        "<b>serve</b> should take whatever the client actually has: a JPEG byte "
        "string, a raw sentence, a dictionary of untransformed features. Define "
        "that as a <code>tf.function</code> with an explicit "
        "<code>input_signature</code> and pass it in the <code>signatures=</code> "
        "argument when exporting. You can export several — "
        "<code>serving_default</code> for one at a time, another for batches, "
        "another that returns embeddings instead of predictions.",
    )

    sub("Versioning")

    md(
        "TF Serving watches a directory of **numbered subdirectories** and loads "
        "the highest number it finds. That numbering convention is the whole "
        "deployment mechanism."
    )

    table(
        ["Strategy", "How", "Rollback"],
        [["<b>Blue/green</b>", "Load the new version, switch all traffic at once",
          "Instant — point back at the old version"],
         ["<b>Canary</b>", "Route 1–5 % of traffic to the new version first",
          "Instant, and you have real metrics before committing"],
         ["<b>Shadow / dark launch</b>",
          "Send <b>all</b> traffic to both; serve only the old one's answers",
          "Zero risk — the new model's output is logged, never used"],
         ["<b>A/B test</b>", "Split traffic and measure a business metric",
          "The only way to know if 'better offline' means 'better in "
          "production'"]],
    )

    warn(
        "Offline metrics and production metrics disagree more often than you "
        "expect",
        "A model with better held-out accuracy can perform worse in production: "
        "the offline test set is stale, the loss function is a proxy for the "
        "business objective rather than the objective itself, and the deployed "
        "model changes user behaviour (a recommender that shows different items "
        "gets different data back). <b>Shadow deployment then a canary</b> is the "
        "only reliable sequence, and it should be the default for anything that "
        "matters.",
    )

    anim_header("Canary release: shifting traffic while watching the metrics")

    rng = np.random.default_rng(3)
    stages = [0, 1, 5, 10, 25, 50, 75, 100]
    frames = []
    for i, pct in enumerate(stages):
        old_lat = 42 + rng.normal(0, 1.5)
        new_lat = 38 + rng.normal(0, 1.5)
        err_old, err_new = 0.021, 0.018 + (0.004 if i == 3 else 0.0)
        blended_err = (1 - pct/100)*err_old + (pct/100)*err_new
        frames.append(go.Frame(name=f"{pct}%", data=[
            go.Bar(x=["v1 (current)", "v2 (canary)"], y=[100-pct, pct],
                   marker=dict(color=[C["muted"], C["primary"]]),
                   text=[f"{100-pct}%", f"{pct}%"], textposition="inside"),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"stage {i+1}/{len(stages)}   ·   v2 traffic {pct}%   ·   "
            f"v1 p50 {old_lat:.1f} ms, v2 p50 {new_lat:.1f} ms   ·   "
            f"blended error rate {blended_err:.4f}"
            + ("   ·   ⚠ v2 error rate spiked — ROLL BACK" if i == 3 else ""),
            color=C["danger"] if i == 3 else C["success"])])))

    f = go.Figure(data=[go.Bar(x=["v1 (current)", "v2 (canary)"], y=[100, 0],
                              marker=dict(color=[C["muted"], C["primary"]]),
                              text=["100%", "0%"], textposition="inside")])
    f.update_layout(height=420, yaxis_title="share of traffic (%)",
                    yaxis=dict(range=[0, 105]),
                    title="A canary release, one stage at a time")
    anim.animate(f, frames, duration=nav.anim_ms(1100), slider_prefix="stage ")
    figure(f, "The point of a canary is that the bad stage costs you 10 % of "
              "requests for a few minutes, not 100 % for however long it takes "
              "to notice.")

    code_lab(
        "Export a SavedModel with preprocessing inside it, and inspect it",
        '''import numpy as np, os, shutil, json, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

df = _ds.housing()
num_cols = [c for c in df.columns
            if df[c].dtype.kind in "if" and c != "median_house_value"]
X = df[num_cols].fillna(df[num_cols].median()).to_numpy().astype("float32")
y = (df["median_house_value"].to_numpy()/100000.0).astype("float32")
n_tr = int(.8*len(X))
print(f"=== {X.shape[0]} rows, {X.shape[1]} numeric features ===")

# ============ 1. THE WRONG WAY: PREPROCESSING OUTSIDE THE MODEL ========
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X[:n_tr])
tf.random.set_seed(0)
bad = keras.Sequential([keras.layers.Input(shape=(X.shape[1],)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
bad.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-2))
bad.fit(scaler.transform(X[:n_tr]), y[:n_tr], epochs=12, batch_size=128,
        verbose=0)
print()
print("=== what happens if the server forgets to scale ===")
print(f"  correctly scaled input : MSE "
      f"{bad.evaluate(scaler.transform(X[n_tr:]), y[n_tr:], verbose=0):.5f}")
print(f"  RAW input (the bug)    : MSE "
      f"{bad.evaluate(X[n_tr:], y[n_tr:], verbose=0):.5f}")
print("  no exception, no warning. Just silently wrong answers.")
print("  this is TRAINING/SERVING SKEW and it is the most common production")
print("  ML bug there is.")

# ============ 2. THE RIGHT WAY: PREPROCESSING IN THE GRAPH =============
norm = keras.layers.Normalization()
norm.adapt(X[:n_tr])                        # learns mean and variance
tf.random.set_seed(0)
good = keras.Sequential([keras.layers.Input(shape=(X.shape[1],), name="features"),
                         norm,                            # INSIDE the model
                         keras.layers.Dense(32, activation="relu"),
                         keras.layers.Dense(1, name="price")])
good.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-2))
good.fit(X[:n_tr], y[:n_tr], epochs=12, batch_size=128, verbose=0)
print()
print("=== with the Normalization layer inside ===")
print(f"  raw input straight from the client: MSE "
      f"{good.evaluate(X[n_tr:], y[n_tr:], verbose=0):.5f}")
print(f"  the layer's learned mean : {np.round(np.ravel(norm.mean)[:4], 3)}")
print(f"  the learned variance     : {np.round(np.ravel(norm.variance)[:4], 3)}")
print("  those numbers SERIALISE WITH THE MODEL. They cannot drift apart.")

# ============ 3. EXPORTING ============================================
base = os.path.join(os.environ.get("TEMP", "/tmp"), "mlplat_serving")
shutil.rmtree(base, ignore_errors=True)
v1 = os.path.join(base, "0001")
good.export(v1)                                   # the SavedModel format
print()
print(f"=== exported to {os.path.basename(v1)} ===")
total = 0
for root, dirs, files in os.walk(v1):
    for fn in sorted(files):
        fp = os.path.join(root, fn)
        sz = os.path.getsize(fp)
        total += sz
        rel = os.path.relpath(fp, v1)
        print(f"  {rel:<42}{sz/1024:>10.1f} KB")
print(f"  {'TOTAL':<42}{total/1024:>10.1f} KB")

# ============ 4. INSPECTING THE SIGNATURE ==============================
loaded = tf.saved_model.load(v1)
print()
print("=== the serving signature ===")
print(f"  available signatures: {list(loaded.signatures.keys())}")
sig = loaded.signatures["serving_default"]
print(f"  inputs:")
for k, v in sig.structured_input_signature[1].items():
    print(f"    {k:<20} shape {tuple(v.shape)}  dtype {v.dtype.name}")
print(f"  outputs:")
for k, v in sig.structured_outputs.items():
    print(f"    {k:<20} shape {tuple(v.shape)}  dtype {v.dtype.name}")

out = sig(tf.constant(X[n_tr:n_tr+3]))
key = list(out.keys())[0]
print()
print(f"  calling it: {np.round(np.ravel(out[key]), 4)}")
print(f"  keras says: {np.round(np.ravel(good.predict(X[n_tr:n_tr+3], verbose=0)), 4)}")

# ============ 5. A CUSTOM SIGNATURE THAT TAKES RAW CLIENT INPUT ========
print()
print("=== a second signature, taking whatever the CLIENT actually has ===")

class ServingModule(tf.Module):
    def __init__(self, model, feature_names):
        super().__init__()
        self.model = model
        self.feature_names = feature_names

    @tf.function(input_signature=[tf.TensorSpec([None, len(num_cols)],
                                                tf.float32, name="features")])
    def predict_price(self, features):
        p = self.model(features)
        return {"price_usd": p[:, 0] * 100000.0,      # back to real units
                "price_scaled": p[:, 0]}

    @tf.function(input_signature=[tf.TensorSpec([None, len(num_cols)],
                                                tf.float32, name="features")])
    def predict_bucket(self, features):
        p = self.model(features)[:, 0]
        return {"bucket": tf.cast(tf.minimum(tf.maximum(p, 0.0), 5.0), tf.int32)}

mod = ServingModule(good, num_cols)
v2 = os.path.join(base, "0002")
tf.saved_model.save(mod, v2, signatures={
    "serving_default": mod.predict_price,
    "bucket": mod.predict_bucket})

l2 = tf.saved_model.load(v2)
print(f"  signatures now: {sorted(l2.signatures.keys())}")
r1 = l2.signatures["serving_default"](tf.constant(X[n_tr:n_tr+3]))
r2 = l2.signatures["bucket"](tf.constant(X[n_tr:n_tr+3]))
print(f"  serving_default -> "
      f"{ {k: np.round(np.ravel(v)[:3], 1).tolist() for k, v in r1.items()} }")
print(f"  bucket          -> "
      f"{ {k: np.ravel(v)[:3].tolist() for k, v in r2.items()} }")
print("  one model, several public APIs. The client never sees a scaler.")

# ============ 6. VERSIONED DIRECTORIES =================================
print()
print("=== the versioning convention ===")
for d in sorted(os.listdir(base)):
    n_files = sum(len(f) for _, _, f in os.walk(os.path.join(base, d)))
    print(f"  {base}{os.sep}{d}{os.sep}   ({n_files} files)")
print("  TF Serving watches this directory and loads the HIGHEST number.")
print("  to deploy: write a new numbered directory. To roll back: delete it")
print("  (or point --model_version_policy at the older one).")
print()
print("  the server command would be:")
print("    docker run -p 8501:8501 \\\\")
print(f"      -v {base}:/models/housing \\\\")
print("      -e MODEL_NAME=housing tensorflow/serving")

# ============ 7. THE MODEL IS NOW PYTHON-FREE ==========================
print()
print("=== what is actually in the graph ===")
cf = sig
print(f"  the loaded object is a {type(loaded).__name__}, not a Keras model")
print(f"  it has NO fit(), NO predict(), NO Python layer objects:")
print(f"    hasattr(loaded, 'fit')     = {hasattr(loaded, 'fit')}")
print(f"    hasattr(loaded, 'predict') = {hasattr(loaded, 'predict')}")
print(f"  it has {len(loaded.variables)} variables totalling "
      f"{sum(int(np.prod(v.shape)) for v in loaded.variables):,} values")
print("  that is the point: TF Serving is a C++ binary. No interpreter,")
print("  no pip install, no version skew between training and production.")

import plotly.graph_objects as go
sizes, names = [], []
for root, dirs, files in os.walk(v1):
    for fn in files:
        names.append(os.path.relpath(os.path.join(root, fn), v1))
        sizes.append(os.path.getsize(os.path.join(root, fn))/1024)
fig = go.Figure(go.Bar(x=names, y=sizes, marker=dict(color=SEQ[:len(names)])))
fig.update_layout(height=380, yaxis_title="KB",
                  title="What is inside a SavedModel")
''',
        key="ch19_savedmodel",
    )

    keypoints([
        "A SavedModel is a <b>self-describing directory</b>: graph, weights, "
        "assets — and needs no Python to run.",
        "<b>Put preprocessing inside the model</b>, or training/serving skew will "
        "fail silently.",
        "A <b>signature</b> is the public API; export several, taking whatever "
        "the client actually has.",
        "Deploy by writing a <b>new numbered directory</b>; roll back by pointing "
        "at the old one.",
        "<b>Shadow, then canary</b> — offline metrics and production metrics "
        "disagree more often than expected.",
    ])


# ==========================================================================
def s_19_2():
    section("19.2", "Serving Latency, Batching and Throughput")

    lead(
        "Serving is a queueing problem wearing a machine-learning costume. The "
        "numbers that matter are p99 latency and requests per second, and they "
        "trade against each other."
    )

    sub("REST vs gRPC")

    table(
        ["", "REST + JSON", "gRPC + protobuf"],
        [["Payload for 1 000 floats", "~12 KB of text", "<b>~4 KB</b> binary"],
         ["Serialisation cost", "Parse text, allocate objects",
          "<b>Near-zero</b> — memory layout matches the wire format"],
         ["Debuggability", "<b>curl works</b>", "Needs a client stub"],
         ["Streaming", "No", "<b>Yes</b>"],
         ["When", "Low volume, ad-hoc, browser clients",
          "<b>Anything high-throughput</b>"]],
    )

    codenote(
        "Base64 your image bytes; never send a JSON array of pixels",
        "A 224×224×3 image as a JSON array of floats is roughly <b>2 MB</b> of "
        "text that must be parsed into 150 000 Python floats. The same image as a "
        "base64-encoded JPEG is about <b>30 KB</b>, and the decode happens inside "
        "the graph with <code>tf.io.decode_jpeg</code>. That is a 60× reduction "
        "in bandwidth and a much larger one in CPU. Accept "
        "<code>tf.string</code> and decode in the signature.",
    )

    sub("Why batching helps, and when it stops helping")

    derive(
        [("<b>Per-request cost has a fixed part and a variable part.</b> Let "
          "$c_0$ be the fixed overhead of a forward pass (kernel launches, "
          "memory allocation, framework dispatch) and $c_1$ the marginal cost per "
          "item. Serving a batch of $B$ costs:",
          r"T(B) \;=\; c_0 + c_1 B"),
         ("Throughput is items per second:",
          r"\text{throughput}(B) = \frac{B}{c_0 + c_1 B}"
          r" \;\xrightarrow{B\to\infty}\; \frac{1}{c_1}"),
         ("So throughput rises steeply at first and then <b>saturates</b> at "
          "$1/c_1$. Doubling the batch beyond the knee buys almost nothing.",
          None),
         ("<b>But latency for an individual request gets worse.</b> With a "
          "batching window $w$, a request waits on average $w/2$ before its batch "
          "closes, then $T(B)$ to be computed:",
          r"\mathbb{E}[\text{latency}] \approx \frac{w}{2} + c_0 + c_1 B"),
         ("<b>The tail is what actually breaks.</b> By Little's law, if the "
          "arrival rate $\\lambda$ approaches the service rate $\\mu$, the queue "
          "length — and therefore latency — grows without bound:",
          r"\mathbb{E}[L] = \frac{\rho}{1-\rho},\qquad \rho = \frac{\lambda}{\mu}"),
         ("At 90 % utilisation the expected queue is 9 requests; at 99 % it is "
          "99. <b>This is why you provision for well under 100 % utilisation</b>, "
          "and why p99 latency explodes long before average latency looks "
          "worrying.", None)],
        title="The batching trade-off, and why the tail explodes",
    )

    warn(
        "Report p99, not the mean",
        "If a page makes 10 backend calls, the probability that <i>all</i> of "
        "them are fast is $0.99^{10} \\approx 0.90$ — so 10 % of page loads hit "
        "at least one p99 request. Users experience the tail, not the mean. A "
        "service whose mean latency is 20 ms and whose p99 is 2 s is a bad "
        "service, and the mean will never tell you.",
    )

    sub("What to actually tune")

    table(
        ["Knob", "Effect", "Typical"],
        [["<b>max_batch_size</b>", "Throughput ↑, tail latency ↑", "16–128"],
         ["<b>batch_timeout_micros</b>",
          "The window; larger fills batches better but adds a floor",
          "1 000–10 000 µs"],
         ["<b>num_batch_threads</b>", "Parallel batches in flight",
          "= number of cores"],
         ["<b>Replicas</b>", "The only thing that raises the ceiling",
          "Provision for peak, not mean"],
         ["<b>Model size</b>",
          "Quantisation and distillation cut $c_1$ directly — §19.3",
          "Often the biggest win"]],
    )

    anim_header("Throughput saturates while tail latency keeps climbing")

    c0, c1 = 0.0045, 0.00022
    batches = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    frames = []
    for k in range(1, len(batches) + 1):
        B = batches[:k]
        thr = B / (c0 + c1*B)
        lat = (c0 + c1*B) * 1000
        p99 = lat * (1 + 0.9*np.log1p(B/8))
        frames.append(go.Frame(name=str(batches[k-1]), data=[
            go.Scatter(x=B, y=thr, mode="lines+markers",
                       line=dict(color=C["success"], width=3)),
            go.Scatter(x=B, y=lat, mode="lines+markers", yaxis="y2",
                       line=dict(color=C["warning"], width=3)),
            go.Scatter(x=B, y=p99, mode="lines+markers", yaxis="y2",
                       line=dict(color=C["danger"], width=3, dash="dot")),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"batch size {batches[k-1]}   ·   throughput {thr[-1]:.0f} req/s "
            f"({thr[-1]/(1/c1):.0%} of the ceiling)   ·   mean latency "
            f"{lat[-1]:.1f} ms   ·   p99 {p99[-1]:.1f} ms")])))

    f = go.Figure(data=[
        go.Scatter(x=batches[:1], y=[1/(c0+c1)], mode="lines+markers",
                   name="throughput (req/s)",
                   line=dict(color=C["success"], width=3)),
        go.Scatter(x=batches[:1], y=[(c0+c1)*1000], mode="lines+markers",
                   name="mean latency (ms)", yaxis="y2",
                   line=dict(color=C["warning"], width=3)),
        go.Scatter(x=batches[:1], y=[(c0+c1)*1000], mode="lines+markers",
                   name="p99 latency (ms)", yaxis="y2",
                   line=dict(color=C["danger"], width=3, dash="dot")),
    ])
    f.add_hline(y=1/c1, line_dash="dot", line_color=C["muted"],
                annotation_text="throughput ceiling = 1/c₁")
    f.update_layout(height=460, xaxis_title="batch size", xaxis_type="log",
                    yaxis=dict(title="throughput (req/s)"),
                    yaxis2=dict(title="latency (ms)", overlaying="y",
                                side="right", type="log"),
                    title="T(B) = c₀ + c₁B",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(600), slider_prefix="B = ")
    figure(f, "The knee is where throughput stops improving but latency does "
              "not stop growing. Batch to the knee, not beyond it.")

    code_lab(
        "Measure c₀ and c₁, find the batching knee, and simulate a queue",
        '''import numpy as np, time, json, base64
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. A REPRESENTATIVE MODEL ================================
model = keras.Sequential([keras.layers.Input(shape=(64,)),
                          keras.layers.Dense(256, activation="relu"),
                          keras.layers.Dense(256, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])
print(f"=== a {model.count_params():,}-parameter model ===")

# a tf.function with a fixed signature: what a server actually runs
@tf.function(input_signature=[tf.TensorSpec([None, 64], tf.float32)])
def serve(x):
    return model(x, training=False)

rng = np.random.default_rng(0)
_ = serve(tf.constant(rng.normal(0, 1, (1, 64)).astype("float32")))  # warm up

# ============ 2. MEASURE c0 AND c1 =====================================
def timed(B, n_rep=30):
    x = tf.constant(rng.normal(0, 1, (B, 64)).astype("float32"))
    serve(x)                                        # warm the trace
    t0 = time.perf_counter()
    for _ in range(n_rep):
        serve(x)
    return (time.perf_counter()-t0)/n_rep

sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
times = [timed(B) for B in sizes]
A = np.column_stack([np.ones(len(sizes)), sizes])
c0, c1 = np.linalg.lstsq(A, np.array(times), rcond=None)[0]
print()
print("=== T(B) = c0 + c1*B, fitted ===")
print(f"  c0 (fixed overhead per call) = {c0*1000:.4f} ms")
print(f"  c1 (marginal cost per item)  = {c1*1e6:.3f} us")
print(f"  throughput ceiling 1/c1      = {1/c1:,.0f} items/s")
print()
print(f"{'batch':>7}{'measured ms':>14}{'fitted ms':>12}"
      f"{'items/s':>12}{'% of ceiling':>15}")
for B, t in zip(sizes, times):
    thr = B/t
    print(f"{B:>7}{t*1000:>14.4f}{(c0+c1*B)*1000:>12.4f}{thr:>12,.0f}"
          f"{thr/(1/c1):>15.1%}")

# --- where is the knee? ----------------------------------------------
thr = np.array([B/t for B, t in zip(sizes, times)])
gain = thr[1:]/thr[:-1]
knee = next((sizes[i+1] for i, g in enumerate(gain) if g < 1.15), sizes[-1])
print()
print(f"  the marginal gain drops below 15% at batch size {knee}")
print(f"  beyond that you are paying latency for almost no throughput.")

# ============ 3. PAYLOAD SIZE: JSON vs BINARY ==========================
print()
print("=== payload size ===")
vec = rng.normal(0, 1, (1, 64)).astype("float32")
as_json = json.dumps({"instances": vec.tolist()})
as_bytes = vec.tobytes()
print(f"  64 floats as JSON text : {len(as_json):>8,} bytes")
print(f"  64 floats as raw binary: {len(as_bytes):>8,} bytes  "
      f"({len(as_json)/len(as_bytes):.1f}x smaller)")

img = (rng.random((224, 224, 3))*255).astype("uint8")
img_json = json.dumps({"instances": [img.tolist()]})
img_jpeg = tf.io.encode_jpeg(tf.constant(img)).numpy()
img_b64 = base64.b64encode(img_jpeg)
print()
print(f"  a 224x224x3 image:")
print(f"    as a JSON array of pixels : {len(img_json):>10,} bytes")
print(f"    as raw uint8              : {img.nbytes:>10,} bytes")
print(f"    as JPEG                   : {len(img_jpeg):>10,} bytes")
print(f"    as base64 JPEG (JSON-safe): {len(img_b64):>10,} bytes  "
      f"({len(img_json)/len(img_b64):.0f}x smaller than the pixel array)")
print("  ALWAYS accept tf.string and decode inside the graph.")

# --- a signature that does exactly that ------------------------------
@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def serve_jpeg(encoded):
    def one(b):
        im = tf.io.decode_jpeg(b, channels=3)
        im = tf.image.resize(im, [32, 32])
        return tf.reshape(tf.cast(im, tf.float32)/255.0, [-1])[:64]
    x = tf.map_fn(one, encoded, fn_output_signature=tf.float32)
    return model(x, training=False)

out = serve_jpeg(tf.constant([img_jpeg, img_jpeg]))
print(f"\\n  decoding inside the graph: input 2 JPEG strings -> "
      f"output {tuple(out.shape)}")

# ============ 4. QUEUEING: WHY p99 EXPLODES ============================
print()
print("="*66)
print("A queue simulation: why you never run at 95% utilisation")
print("="*66)
def simulate(arrival_rate, service_time, n=40000, seed=0):
    """M/D/1: Poisson arrivals, deterministic service."""
    r = np.random.default_rng(seed)
    inter = r.exponential(1/arrival_rate, n)
    arrive = np.cumsum(inter)
    start = np.zeros(n); finish = np.zeros(n)
    for i in range(n):
        start[i] = max(arrive[i], finish[i-1] if i else 0.0)
        finish[i] = start[i] + service_time
    lat = (finish - arrive)*1000
    return lat

svc = c0 + c1*32                       # serving batches of 32
mu = 1/svc
print(f"  service time {svc*1000:.3f} ms -> capacity {mu:,.0f} batches/s")
print()
print(f"{'utilisation':>13}{'mean ms':>11}{'p50':>9}{'p95':>10}{'p99':>11}"
      f"{'p99.9':>11}")
for rho in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    lat = simulate(rho*mu, svc)
    print(f"{rho:>13.0%}{lat.mean():>11.3f}{np.percentile(lat,50):>9.3f}"
          f"{np.percentile(lat,95):>10.3f}{np.percentile(lat,99):>11.3f}"
          f"{np.percentile(lat,99.9):>11.3f}")
print()
print("  the MEAN barely moves from 50% to 90% utilisation.")
print("  the p99 grows by an order of magnitude. Users feel the TAIL.")
print(f"  theory: E[queue] = rho/(1-rho) -> "
      f"{0.9/(1-0.9):.0f} at 90%, {0.99/(1-0.99):.0f} at 99%")

# ============ 5. THE BATCHING WINDOW ===================================
print()
print("=== the batching window is a latency floor ===")
print(f"{'window (ms)':>13}{'mean wait':>12}{'batch fill at 2000 req/s':>27}"
      f"{'total mean latency':>21}")
arrival = 2000.0
for w in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    fill = min(64, max(1, arrival*w/1000))
    wait = w/2
    compute = (c0 + c1*fill)*1000
    print(f"{w:>13.1f}{wait:>12.3f}{fill:>27.1f}{wait+compute:>21.3f}")
print("  with no window every request is a batch of 1: maximum overhead.")
print("  with a 10 ms window you fill batches nicely and pay 5 ms of")
print("  pure waiting. The optimum depends on your arrival rate --")
print("  which is why it must be TUNED, not copied.")

# ============ 6. WHAT ACTUALLY RAISES THE CEILING ======================
print()
print("=== three ways to serve 100 000 req/s ===")
target = 100000
print(f"{'approach':<38}{'req/s each':>13}{'replicas needed':>18}")
for nm, per in [("as measured, batch 32", 32/(c0+c1*32)),
                ("half the model size (c1/2)", 32/(c0+c1/2*32)),
                ("batch 128 instead of 32", 128/(c0+c1*128))]:
    print(f"{nm:<38}{per:>13,.0f}{int(np.ceil(target/per)):>18}")
print("  making the model smaller (section 19.3) attacks c1 directly and")
print("  is usually cheaper than adding machines.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=sizes, y=[B/t for B, t in zip(sizes, times)],
                mode="lines+markers", name="measured throughput",
                line=dict(color=C["success"], width=3))
fig.add_scatter(x=sizes, y=[B/(c0+c1*B) for B in sizes], mode="lines",
                name="fitted B/(c0+c1B)",
                line=dict(color=C["muted"], width=2, dash="dash"))
fig.add_hline(y=1/c1, line_dash="dot", line_color=C["danger"],
              annotation_text="ceiling 1/c1")
fig.update_layout(height=400, xaxis_type="log", xaxis_title="batch size",
                  yaxis_title="items / second",
                  title="Throughput saturates")
''',
        key="ch19_serving",
    )

    keypoints([
        "$T(B) = c_0 + c_1 B$: throughput saturates at $1/c_1$, so batch to the "
        "knee and no further.",
        "<b>gRPC + protobuf</b> for volume; accept <code>tf.string</code> and "
        "decode images <b>inside the graph</b>.",
        "By Little's law, queue length is $\\rho/(1-\\rho)$ — <b>p99 explodes "
        "long before the mean moves</b>.",
        "The batching window is a <b>latency floor</b> of $w/2$; tune it against "
        "your arrival rate.",
        "Shrinking the model attacks $c_1$ directly and is usually cheaper than "
        "adding replicas.",
    ])


# ==========================================================================
def s_19_3():
    section("19.3", "Mobile and Embedded — TFLite and Quantisation")

    lead(
        "A phone has no GPU cluster, a strict battery budget, and an app-size "
        "limit. Quantisation buys a 4× size reduction and a 2–3× speed-up for a "
        "usually negligible accuracy cost."
    )

    sub("What the converter does")

    table(
        ["Optimisation", "Effect", "Cost"],
        [["<b>Freeze and prune the graph</b>",
          "Drops training-only ops (dropout, optimiser state)", "None"],
         ["<b>Operator fusion</b>",
          "Conv + BN + ReLU become one kernel", "None"],
         ["<b>Constant folding</b>",
          "Evaluates anything not depending on the input", "None"],
         ["<b>Float16 quantisation</b>", "<b>2× smaller</b>",
          "Essentially none"],
         ["<b>Dynamic-range int8</b>",
          "<b>4× smaller</b>, weights int8, activations float",
          "Usually < 1 % accuracy"],
         ["<b>Full-integer int8</b>",
          "4× smaller <b>and</b> int-only hardware (Edge TPU, DSP)",
          "Needs a representative dataset; 1–2 % typical"],
         ["<b>Quantisation-aware training</b>",
          "Recovers most of the int8 loss",
          "A fine-tuning run"]],
    )

    sub("How int8 quantisation works")

    math(r"""
    r \;\approx\; S\,(q - Z),
    \qquad
    q = \mathrm{round}\!\left(\frac{r}{S}\right) + Z
    """)

    where({
        r"r": "the real (float32) value",
        r"q \in [-128, 127]": "the stored 8-bit integer",
        r"S": "the <b>scale</b> — how much one integer step is worth",
        r"Z": "the <b>zero-point</b> — which integer represents exactly 0.0",
    })

    derive(
        [("<b>Choosing the scale.</b> For a tensor with range "
          "$[r_{\\min}, r_{\\max}]$ mapped onto $[q_{\\min}, q_{\\max}]$:",
          r"S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}},"
          r"\qquad Z = q_{\min} - \frac{r_{\min}}{S}"),
         ("<b>The quantisation error is bounded by half a step:</b>",
          r"\bigl|r - S(q - Z)\bigr| \;\le\; \frac{S}{2}"),
         ("So the relative error depends entirely on the ratio of the step to "
          "the typical magnitude. With 256 levels spread over a range, the "
          "signal-to-quantisation-noise ratio is:",
          r"\mathrm{SQNR} \approx 6.02\,b + 1.76\ \text{dB}"),
         ("giving about <b>50 dB at 8 bits</b> — far more precision than a "
          "neural network's weights actually need, which is why this works at "
          "all.", None),
         ("<b>The catch is outliers.</b> $S$ is set by the extremes, so a single "
          "weight of 100 in a tensor otherwise spanning $[-1, 1]$ makes every "
          "step 50× coarser and destroys the precision of everything else. This "
          "is why <b>per-channel</b> quantisation (a separate $S$ per output "
          "channel) is standard for convolution weights, and why activation "
          "outliers are the central difficulty in quantising large language "
          "models.", None),
         ("<b>Why activations need a calibration set.</b> Weights are known at "
          "conversion time, so their range is exact. Activation ranges depend on "
          "the input, so full-integer quantisation must <b>run the model on "
          "representative data</b> to observe them. Give it data from the real "
          "distribution — calibrating on random noise produces ranges that are "
          "wrong in a way that is very hard to debug.", None)],
        title="The arithmetic of int8 quantisation",
    )

    pitfall(
        "The representative dataset must be representative",
        "100–500 samples <b>from the real input distribution</b> is enough. "
        "Random noise, or images from a different preprocessing pipeline, gives "
        "activation ranges that are simply wrong — and the resulting model is "
        "quietly much worse, with no error at conversion time. If accuracy drops "
        "badly after full-integer quantisation, suspect the calibration set "
        "before anything else.",
    )

    sub("Other ways to make a model small")

    table(
        ["Technique", "Typical reduction", "Note"],
        [["<b>Quantisation</b>", "4×", "Almost free; do this first"],
         ["<b>Weight pruning</b>", "2–10× with sparse formats",
          "Needs hardware/runtime support to translate into speed"],
         ["<b>Distillation</b>", "<b>10–100×</b>",
          "Train a small model on the big one's <i>soft</i> outputs"],
         ["<b>A smaller architecture</b> (MobileNet, EfficientNet-Lite)",
          "10–50×", "<b>Usually the right first move</b>"],
         ["<b>Weight clustering</b>", "2–4× with compression",
          "Share weights across a codebook"]],
    )

    idea(
        "Why distillation works better than it has any right to",
        "A teacher's <b>soft</b> output — 'this is 0.7 cat, 0.2 lynx, 0.1 dog' — "
        "carries far more information per example than the hard label 'cat'. "
        "Hinton called this <i>dark knowledge</i>: the relative probabilities of "
        "the wrong classes encode the teacher's learned similarity structure. "
        "Training the student on those soft targets (with a temperature applied "
        "to both, §16.1) transfers that structure, and a student can reach "
        "accuracy it could never achieve training on the hard labels alone.",
    )

    anim_header("Quantising a weight tensor, and what outliers do to it")

    rng = np.random.default_rng(8)
    w_clean = rng.normal(0, 0.35, 4000)
    frames = []
    for outlier in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        w = w_clean.copy()
        if outlier > 0:
            w[:3] = outlier
        r_min, r_max = w.min(), w.max()
        S = (r_max - r_min) / 255.0
        Z = -128 - r_min / S
        q = np.clip(np.round(w / S) + Z, -128, 127)
        deq = S * (q - Z)
        err = np.abs(w - deq)
        frames.append(go.Frame(name=f"{outlier:g}", data=[
            go.Histogram(x=w, nbinsx=90, marker=dict(color=alpha(C["primary"], .7))),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"outlier magnitude {outlier:g}   ·   range "
            f"[{r_min:.2f}, {r_max:.2f}]   ·   step S = {S:.5f}   ·   "
            f"mean |error| = {err[3:].mean():.5f} on the ORDINARY weights"
            + ("   ·   3 outliers have wrecked the precision of 3 997 weights"
               if outlier >= 4 else ""),
            color=C["danger"] if outlier >= 4 else C["success"])])))

    f = go.Figure(data=[go.Histogram(x=w_clean, nbinsx=90,
                                     marker=dict(color=alpha(C["primary"], .7)))])
    f.update_layout(height=420, xaxis_title="weight value", yaxis_title="count",
                    yaxis_type="log",
                    title="One tensor, 4 000 weights, 256 quantisation levels")
    anim.animate(f, frames, duration=nav.anim_ms(1400), slider_prefix="outlier ")
    figure(f, "The scale is set by the extremes. Three outliers cost every other "
              "weight an order of magnitude of precision — which is why "
              "per-channel scales exist.")

    code_lab(
        "Convert to TFLite, quantise four ways, and measure the real trade-off",
        '''import numpy as np, os, time
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

Xtr, ytr, Xte, yte, labels, real = _ds.fashion_mnist(n_train=10000, n_test=2000)
Xtr = Xtr.astype("float32")[..., None]
Xte = Xte.astype("float32")[..., None]
print(f"=== Fashion-MNIST: train {Xtr.shape}, test {Xte.shape} ===")

# ============ 1. A MODEL WORTH DEPLOYING ===============================
tf.random.set_seed(0)
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
model.fit(Xtr, ytr, epochs=8, batch_size=128, verbose=0)
base_acc = model.evaluate(Xte, yte, verbose=0, return_dict=True)["accuracy"]
print(f"  {model.count_params():,} parameters, "
      f"float32 test accuracy {base_acc:.4f}")

TMP = os.environ.get("TEMP", "/tmp")
keras_path = os.path.join(TMP, "mlplat_model.keras")
model.save(keras_path)
print(f"  .keras file: {os.path.getsize(keras_path)/1024:.1f} KB")

# ============ 2. FOUR CONVERSIONS ======================================
def representative_dataset():
    """100 samples FROM THE REAL DISTRIBUTION -- not random noise."""
    for i in range(100):
        yield [Xtr[i:i+1]]

def convert(mode):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "none":
        pass
    elif mode == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()

def run_tflite(blob, X, n=800):
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = np.zeros(n, dtype=int)
    t0 = time.perf_counter()
    for i in range(n):
        x = X[i:i+1]
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            x = np.clip(np.round(x/s) + z, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
        interp.invoke()
        preds[i] = int(np.argmax(interp.get_tensor(out["index"])[0]))
    dt = (time.perf_counter()-t0)/n
    return preds, dt

print()
print("=== four conversions ===")
print(f"{'mode':<26}{'size KB':>11}{'vs float32':>12}{'accuracy':>11}"
      f"{'drop':>9}{'ms/inference':>15}")
blobs = {}
for mode, nm in [("none", "float32 (no optimisation)"),
                 ("float16", "float16"),
                 ("dynamic", "dynamic-range int8"),
                 ("int8", "FULL integer int8")]:
    try:
        blob = convert(mode)
    except Exception as e:
        print(f"{nm:<26}  conversion failed: {type(e).__name__}")
        continue
    blobs[nm] = blob
    p, dt = run_tflite(blob, Xte)
    acc = float((p == yte[:len(p)]).mean())
    sz = len(blob)/1024
    ref = len(blobs.get("float32 (no optimisation)", blob))/1024
    print(f"{nm:<26}{sz:>11.1f}{ref/sz:>11.2f}x{acc:>11.4f}"
          f"{base_acc-acc:>+9.4f}{dt*1000:>15.4f}")
print()
print("  int8 is 4x smaller because a float32 weight becomes one byte.")
print("  the accuracy cost is usually well under 1% -- a network's weights")
print("  simply do not need 24 bits of mantissa.")

# ============ 3. THE ARITHMETIC, BY HAND ===============================
print()
print("=== r = S*(q - Z), verified against TFLite's own tensors ===")
interp = tf.lite.Interpreter(model_content=blobs["FULL integer int8"])
interp.allocate_tensors()
qtensors = [d for d in interp.get_tensor_details()
            if d["dtype"] == np.int8 and len(d["shape"]) > 1]
print(f"  {len(qtensors)} int8 tensors in the model")
for d in qtensors[:3]:
    scales = d["quantization_parameters"]["scales"]
    zeros = d["quantization_parameters"]["zero_points"]
    print(f"    {d['name'][:44]:<46} shape {tuple(d['shape'])}")
    print(f"      {len(scales)} scale(s): "
          f"{np.round(scales[:4], 6)}{' ...' if len(scales) > 4 else ''}")
    print(f"      zero point(s): {zeros[:4]}"
          f"{' ...' if len(zeros) > 4 else ''}")
print("  MORE THAN ONE SCALE means PER-CHANNEL quantisation: each output")
print("  filter gets its own S, so one large filter cannot ruin the rest.")

# ============ 4. WHAT OUTLIERS COST ====================================
print()
print("=== per-tensor vs per-channel, on a weight matrix with an outlier ===")
rng = np.random.default_rng(0)
W = rng.normal(0, 0.3, (64, 8)).astype("float32")
W[0, 3] = 12.0                                   # ONE outlier, in channel 3

def quantise(x, axis=None):
    if axis is None:
        lo, hi = x.min(), x.max()
        S = (hi-lo)/255.0; Z = -128 - lo/S
        q = np.clip(np.round(x/S)+Z, -128, 127)
        return S*(q-Z), np.array([S])
    lo = x.min(axis=0, keepdims=True); hi = x.max(axis=0, keepdims=True)
    S = (hi-lo)/255.0; Z = -128 - lo/S
    q = np.clip(np.round(x/S)+Z, -128, 127)
    return S*(q-Z), S.ravel()

for nm, ax in [("per-TENSOR (one scale)", None),
               ("per-CHANNEL (one scale per column)", 0)]:
    deq, S = quantise(W, ax)
    err = np.abs(W - deq)
    clean = err[:, [c for c in range(8) if c != 3]]
    print(f"  {nm:<38}")
    print(f"    scale(s): {np.round(S[:4], 5)}"
          f"{' ...' if len(S) > 4 else ''}")
    print(f"    mean |error| on the 7 CLEAN channels: {clean.mean():.6f}")
print("  per-channel confines the damage to the channel that contains the")
print("  outlier. That is why it is the default for convolution weights.")

# ============ 5. THE CALIBRATION SET MATTERS ===========================
print()
print("=== calibrating on the wrong data ===")
def convert_with(gen):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = gen
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()

def noise_gen():
    r = np.random.default_rng(0)
    for _ in range(100):
        yield [r.random((1, 28, 28, 1)).astype("float32")]

def real_gen():
    for i in range(100):
        yield [Xtr[i:i+1]]

print(f"{'calibration data':<34}{'accuracy':>11}{'drop from float32':>20}")
for nm, gen in [("100 real training images", real_gen),
                ("100 uniform-random images", noise_gen)]:
    b = convert_with(gen)
    p, _ = run_tflite(b, Xte, n=600)
    acc = float((p == yte[:len(p)]).mean())
    print(f"{nm:<34}{acc:>11.4f}{base_acc-acc:>+20.4f}")
print("  calibrating on noise gives activation ranges that are simply wrong.")
print("  no error is raised at conversion time. Suspect this FIRST when")
print("  full-integer quantisation destroys your accuracy.")

# ============ 6. DISTILLATION ==========================================
print()
print("="*66)
print("Distillation: a 100x smaller model, trained on soft targets")
print("="*66)
teacher = model
student_arch = lambda: keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(10)])

T = 4.0
soft = tf.nn.softmax(
    tf.math.log(teacher.predict(Xtr, verbose=0) + 1e-9) / T).numpy()
print(f"  teacher {teacher.count_params():,} params -> "
      f"student {student_arch().count_params():,} params "
      f"({teacher.count_params()/student_arch().count_params():.0f}x smaller)")
print(f"  soft targets at T={T}: e.g. {np.round(soft[0], 3)}")
print(f"  hard label was {ytr[0]} -- the soft target also says which OTHER")
print(f"  classes the teacher considers similar. That is the 'dark knowledge'.")

print()
print(f"{'student trained on':<34}{'test accuracy':>16}")
for nm, use_soft in [("hard labels only", False),
                     ("SOFT teacher outputs", True)]:
    tf.random.set_seed(0)
    st = student_arch()
    if use_soft:
        st.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                   optimizer=keras.optimizers.Adam(2e-3))
        st.fit(Xtr, soft, epochs=14, batch_size=128, verbose=0)
    else:
        st.compile(loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True), optimizer=keras.optimizers.Adam(2e-3))
        st.fit(Xtr, ytr, epochs=14, batch_size=128, verbose=0)
    pred = st.predict(Xte, verbose=0).argmax(1)
    print(f"{nm:<34}{float((pred == yte).mean()):>16.4f}")
print(f"  teacher, for reference:            {base_acc:.4f}")
print("  the same student, the same data, the same epochs -- only the")
print("  TARGETS differ.")

import plotly.graph_objects as go
names = list(blobs.keys())
sizes_kb = [len(blobs[n])/1024 for n in names]
fig = go.Figure(go.Bar(x=names, y=sizes_kb,
                       marker=dict(color=SEQ[:len(names)]),
                       text=[f"{s:.0f} KB" for s in sizes_kb],
                       textposition="outside"))
fig.update_layout(height=400, yaxis_title="model size (KB)",
                  title="TFLite conversion modes")
''',
        key="ch19_tflite",
    )

    quiz(
        "Full-integer quantisation drops your accuracy from 92 % to 61 %. What "
        "should you check first?",
        ["The model is too small to quantise",
         "The <b>representative dataset</b> — calibrating on data from the wrong "
         "distribution gives wrong activation ranges, with no error raised",
         "int8 is simply not accurate enough for this task",
         "The learning rate during training"],
        1,
        "Weight ranges are exact (the weights are known), but activation ranges "
        "are <i>observed</i> by running the calibration set. If that set is not "
        "drawn from the real input distribution, every activation scale is wrong "
        "— and the converter reports nothing. Fix the calibration set before "
        "suspecting anything else.",
        key="ch19q1",
    )

    keypoints([
        "$r = S(q - Z)$; the error is bounded by $S/2$, and 8 bits gives "
        "~50 dB — more than a network needs.",
        "<b>Dynamic-range int8</b> is 4× smaller for usually < 1 % accuracy; do "
        "it first.",
        "<b>Per-channel scales</b> stop one outlier destroying the precision of "
        "a whole tensor.",
        "Full-integer quantisation needs a <b>representative</b> calibration set "
        "— noise silently ruins it.",
        "<b>Distillation</b> on soft targets transfers the teacher's similarity "
        "structure, not just its answers.",
    ])

# ==========================================================================
def s_19_4():
    section("19.4", "GPUs, Memory Arithmetic and Mixed Precision")

    lead(
        "Almost every 'out of memory' error is predictable from four numbers. "
        "Learn the arithmetic and you stop guessing at batch sizes."
    )

    sub("Where the memory goes")

    derive(
        [("Training memory has four components. For a model with $N$ parameters "
          "trained in float32 with Adam:", None),
         ("<b>1. Parameters</b>: 4 bytes each.",
          r"M_{\text{params}} = 4N"),
         ("<b>2. Gradients</b>: one per parameter, same dtype.",
          r"M_{\text{grads}} = 4N"),
         ("<b>3. Optimiser state</b>: Adam keeps a first and second moment.",
          r"M_{\text{opt}} = 8N \quad (\text{SGD: } 0,\;\text{momentum: } 4N)"),
         ("<b>4. Activations</b>: every intermediate tensor needed for the "
          "backward pass, which scales with the <b>batch size</b>:",
          r"M_{\text{act}} = 4 \cdot B \cdot \sum_{\ell} \bigl|a_\ell\bigr|"),
         ("So the total is:",
          r"M \;\approx\; 16N \;+\; 4B\sum_\ell |a_\ell|"),
         ("<b>The parameter part is fixed; only the activation part responds to "
          "the batch size.</b> A 1-billion-parameter model needs 16 GB before a "
          "single example is loaded — which is why large-model training is about "
          "sharding parameters (ZeRO, FSDP), not about batch size.", None),
         ("<b>At inference</b> there are no gradients, no optimiser state, and "
          "activations can be freed as soon as they are consumed:",
          r"M_{\text{inference}} \approx 4N + 4B\max_\ell |a_\ell|"),
         ("which is roughly <b>a quarter</b> of the training requirement, and "
          "why a model that needs 4 GPUs to train serves happily on one.", None)],
        title="Predicting GPU memory before you run out of it",
    )

    sub("Mixed precision")

    table(
        ["", "float32", "float16", "bfloat16"],
        [["Exponent bits", "8", "<b>5</b>", "<b>8</b> — same as float32"],
         ["Mantissa bits", "23", "10", "7"],
         ["Max value", "$3.4\\times10^{38}$",
          "<b>65 504</b>", "$3.4\\times10^{38}$"],
         ["Smallest normal", "$1.2\\times10^{-38}$",
          "<b>$6\\times10^{-5}$</b>", "$1.2\\times10^{-38}$"],
         ["Needs loss scaling", "—", "<b>Yes</b>", "No"],
         ["Hardware", "Everything",
          "NVIDIA Tensor Cores (compute ≥ 7.0)", "TPU, A100+, MI200+"]],
    )

    proof(
        "Loss scaling exists because float16 has only five exponent bits",
        "Gradients in a deep network are routinely around $10^{-7}$ — comfortably "
        "representable in float32, and <b>exactly zero</b> in float16, whose "
        "smallest normal value is $6\\times10^{-5}$. Multiply the loss by a "
        "large constant $S$ before the backward pass and every gradient is "
        "scaled by $S$ too, moving them into float16's representable range; "
        "divide by $S$ before applying them and the update is unchanged. "
        "<b>Dynamic</b> loss scaling raises $S$ while things are fine and halves "
        "it on any overflow, so it needs no tuning. "
        "<code>bfloat16</code> avoids the whole problem by keeping float32's "
        "exponent and sacrificing mantissa bits instead — which is the right "
        "trade for neural networks, and why it won on TPUs.",
    )

    codenote(
        "Mixed precision in three lines, and the one thing you must not forget",
        "<code>keras.mixed_precision.set_global_policy('mixed_float16')</code> "
        "makes every layer compute in float16 while keeping a float32 master "
        "copy of the weights. <b>The final layer must be float32</b> — a softmax "
        "over float16 logits loses precision exactly where you need it, and "
        "overflows on large logits. Write "
        "<code>Dense(10, dtype='float32')</code> on the output layer, or "
        "separate the logits from the activation. Keras's built-in optimisers "
        "handle loss scaling automatically; a custom training loop needs "
        "<code>LossScaleOptimizer</code> explicitly.",
    )

    sub("Placement and growth")

    pitfall(
        "TensorFlow grabs all GPU memory at startup by default",
        "On first use TF allocates <b>essentially the entire GPU</b>, so a second "
        "process — a notebook you forgot, an evaluation job, a colleague — fails "
        "with OOM even though the GPU is nearly idle. Two fixes: "
        "<code>tf.config.experimental.set_memory_growth(gpu, True)</code> to "
        "allocate on demand, or "
        "<code>set_logical_device_configuration</code> with an explicit memory "
        "limit to partition it. Set this <b>before any tensor is created</b> — "
        "it cannot be changed afterwards.",
    )

    anim_header("Memory as batch size grows, at three precisions")

    N_params = 25_000_000
    act_per_item = 8_000_000     # bytes of activations per example at fp32
    batches = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    gpu_gb = 16.0
    configs = [
        ("float32 + Adam", 16, 4),
        ("mixed float16 + Adam", 18, 2),
        ("float32 + SGD", 8, 4),
    ]
    frames = []
    for k in range(1, len(batches) + 1):
        B = batches[:k]
        data = []
        for i, (nm, per_p, act_bytes) in enumerate(configs):
            fixed = per_p * N_params / 1e9
            total = fixed + B * (act_per_item/4*act_bytes) / 1e9
            data.append(go.Scatter(x=B, y=total, mode="lines+markers",
                                   line=dict(color=SEQ[i], width=3)))
        cur = [per_p*N_params/1e9 + batches[k-1]*(act_per_item/4*ab)/1e9
               for _, per_p, ab in configs]
        fits = [c <= gpu_gb for c in cur]
        frames.append(go.Frame(name=str(batches[k-1]), data=data,
                               layout=go.Layout(annotations=[anim.annotate_step(
                                   f"batch {batches[k-1]}   ·   "
                                   + "   ·   ".join(
                                       f"{nm.split('+')[0].strip()}: "
                                       f"{c:.1f} GB {'✓' if f else '✗ OOM'}"
                                       for (nm, _, _), c, f
                                       in zip(configs, cur, fits)),
                                   color=C["success"] if fits[1]
                                   else C["danger"])])))

    f = go.Figure(data=[
        go.Scatter(x=batches[:1],
                   y=[per_p*N_params/1e9 + 1*(act_per_item/4*ab)/1e9],
                   mode="lines+markers", name=nm,
                   line=dict(color=SEQ[i], width=3))
        for i, (nm, per_p, ab) in enumerate(configs)])
    f.add_hline(y=gpu_gb, line_dash="dash", line_color=C["danger"],
                annotation_text="16 GB GPU")
    f.update_layout(height=440, xaxis_type="log", xaxis_title="batch size",
                    yaxis_title="memory (GB)", yaxis_type="log",
                    title=f"A {N_params/1e6:.0f}M-parameter model",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(550), slider_prefix="B = ")
    figure(f, "The flat left-hand part is the parameters, gradients and "
              "optimiser state — fixed. Only the sloped part responds to batch "
              "size.")

    code_lab(
        "Memory arithmetic, device placement, and mixed precision",
        '''import numpy as np, time, os
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. WHAT HARDWARE IS ACTUALLY HERE? =======================
print("=== devices ===")
gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")
print(f"  CPUs: {len(cpus)}   GPUs: {len(gpus)}")
for d in gpus:
    print(f"    {d.name}  {tf.config.experimental.get_device_details(d)}")
if not gpus:
    print("  no GPU here -- every number below is computed rather than")
    print("  measured, but the arithmetic is exactly what you would use")
    print("  to size a job before submitting it.")
print(f"  TF built with CUDA: {tf.test.is_built_with_cuda()}")

# ============ 2. THE MEMORY FORMULA ====================================
def memory_report(n_params, batch, act_bytes_per_item, optimizer="adam",
                  dtype_bytes=4):
    per_param = {"sgd": 2, "momentum": 3, "adam": 4}[optimizer] * dtype_bytes
    params = n_params * dtype_bytes
    grads = n_params * dtype_bytes
    opt = n_params * (per_param - 2*dtype_bytes)
    acts = batch * act_bytes_per_item
    return dict(params=params, grads=grads, optimizer=opt, activations=acts,
                total=params+grads+opt+acts)

print()
print("=== predicting memory before you run out of it ===")
for n_p, nm in [(5_000_000, "ResNet-ish, 5M"),
                (110_000_000, "BERT-base, 110M"),
                (7_000_000_000, "a 7B language model")]:
    r = memory_report(n_p, batch=32, act_bytes_per_item=8_000_000)
    print(f"  {nm:<28} params {r['params']/1e9:>6.2f} GB   "
          f"+grads {r['grads']/1e9:>6.2f}   +Adam {r['optimizer']/1e9:>6.2f}   "
          f"= {(r['total']-r['activations'])/1e9:>7.2f} GB before any data")
print()
print("  a 7B model needs 112 GB of FIXED state to train with Adam.")
print("  that is why large-model training shards the PARAMETERS")
print("  (ZeRO / FSDP), not just the batch.")

print()
print("=== optimiser choice, 110M parameters ===")
print(f"{'optimizer':<14}{'state per param':>18}{'fixed GB':>12}")
for opt in ["sgd", "momentum", "adam"]:
    r = memory_report(110_000_000, 0, 0, optimizer=opt)
    print(f"{opt:<14}{(r['optimizer']/110e6):>15.0f} B"
          f"{(r['total'])/1e9:>12.2f}")
print("  plain SGD needs a THIRD of Adam's memory. On a model that barely")
print("  fits, that is a real option.")

# ============ 3. TRAINING vs INFERENCE =================================
print()
print("=== inference needs about a quarter of training ===")
n_p = 110_000_000
train = memory_report(n_p, 32, 8_000_000)
infer = n_p*4 + 32*2_000_000
print(f"  training  : {train['total']/1e9:.2f} GB")
print(f"  inference : {infer/1e9:.2f} GB   "
      f"({train['total']/infer:.1f}x less)")
print("  no gradients, no optimiser state, and activations are freed as")
print("  soon as they are consumed.")

# ============ 4. MEASURE ACTIVATIONS FOR A REAL MODEL ==================
print()
print("=== measured activation footprint ===")
def act_bytes(model, batch=1):
    """Sum the size of every layer output -- what the backward pass keeps."""
    total = 0
    x = tf.zeros((batch,) + model.input_shape[1:])
    for layer in model.layers:
        x = layer(x)
        total += int(np.prod(x.shape[1:])) * 4 * batch
    return total

m = keras.Sequential([keras.layers.Input(shape=(32, 32, 3)),
                      keras.layers.Conv2D(64, 3, padding="same",
                                          activation="relu"),
                      keras.layers.Conv2D(64, 3, padding="same",
                                          activation="relu"),
                      keras.layers.MaxPool2D(),
                      keras.layers.Conv2D(128, 3, padding="same",
                                          activation="relu"),
                      keras.layers.MaxPool2D(),
                      keras.layers.Flatten(),
                      keras.layers.Dense(256, activation="relu"),
                      keras.layers.Dense(10)])
a1 = act_bytes(m, 1)
print(f"  model: {m.count_params():,} parameters "
      f"({m.count_params()*4/1e6:.1f} MB of weights)")
print(f"  activations for ONE example: {a1/1e6:.2f} MB")
print(f"{'batch':>8}{'activations MB':>17}{'total training MB':>21}")
for B in [1, 8, 32, 128, 512]:
    r = memory_report(m.count_params(), B, a1)
    print(f"{B:>8}{B*a1/1e6:>17.1f}{r['total']/1e6:>21.1f}")
print("  activations DOMINATE for a convolutional model -- which is why")
print("  gradient checkpointing (recompute instead of store) buys so much.")

# ============ 5. FLOAT16 vs BFLOAT16 vs FLOAT32 ========================
print()
print("=== what each format can represent ===")
print(f"{'format':<12}{'exp bits':>10}{'mantissa':>10}{'max':>14}"
      f"{'min normal':>14}")
for nm, e, mant, mx, mn in [("float32", 8, 23, 3.4e38, 1.2e-38),
                            ("float16", 5, 10, 65504.0, 6.1e-5),
                            ("bfloat16", 8, 7, 3.4e38, 1.2e-38)]:
    print(f"{nm:<12}{e:>10}{mant:>10}{mx:>14.2e}{mn:>14.2e}")

print()
print("=== a typical gradient in each format ===")
for g in [1e-2, 1e-4, 1e-6, 1e-8]:
    f16 = np.float16(g)
    bf16 = tf.cast(tf.constant(g, tf.float32), tf.bfloat16).numpy()
    print(f"  {g:>9.0e}  float16 -> {float(f16):>10.3e}"
          f"{'  UNDERFLOW TO ZERO' if float(f16) == 0 else ''}"
          f"   bfloat16 -> {float(bf16):>10.3e}")
print("  gradients of 1e-6 are completely ordinary in a deep network,")
print("  and they are EXACTLY ZERO in float16. Hence loss scaling.")

# ============ 6. LOSS SCALING, DEMONSTRATED ============================
print()
print("=== loss scaling ===")
grads = np.array([1e-3, 1e-5, 1e-7, 1e-9], dtype="float32")
print(f"{'scale S':>10}{'gradients in float16 after scaling':>44}"
      f"{'survivors':>12}")
for S in [1, 128, 1024, 32768]:
    scaled = (grads*S).astype("float16").astype("float32")/S
    alive = int((scaled != 0).sum())
    print(f"{S:>10}{str(np.array([f'{v:.1e}' for v in scaled])):>44}"
          f"{alive:>10}/4")
print("  multiply the loss by S -> every gradient is scaled by S ->")
print("  divide before applying. The UPDATE is identical; the only")
print("  difference is that it no longer underflows.")
print("  dynamic loss scaling raises S while it works and halves it on")
print("  any overflow, so there is nothing to tune.")

# ============ 7. THE POLICY, AND THE OUTPUT LAYER =====================
print()
print("=== keras.mixed_precision ===")
print(f"  current global policy: {keras.mixed_precision.global_policy().name}")
try:
    keras.mixed_precision.set_global_policy("mixed_float16")
    mp = keras.Sequential([keras.layers.Input(shape=(16,)),
                           keras.layers.Dense(32, activation="relu"),
                           keras.layers.Dense(10, dtype="float32")])  # FLOAT32!
    print(f"  under 'mixed_float16':")
    for l in mp.layers:
        print(f"    {l.__class__.__name__:<10} compute {l.dtype_policy.compute_dtype:<9}"
              f" variables {l.dtype_policy.variable_dtype}")
    print("  the WEIGHTS stay float32 (a master copy); only the COMPUTE is")
    print("  float16. And the output layer is forced back to float32 --")
    print("  a softmax over float16 logits loses precision exactly where")
    print("  you need it, and overflows on large logits.")
finally:
    keras.mixed_precision.set_global_policy("float32")
    print(f"  restored policy: {keras.mixed_precision.global_policy().name}")

# ============ 8. DEVICE PLACEMENT ======================================
print()
print("=== placement ===")
with tf.device("/CPU:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(f"  explicitly placed tensor: {a.device}")
b = tf.matmul(a, a)
print(f"  result of matmul        : {b.device}")
print("  TF places ops on the fastest device that has a kernel for them,")
print("  and copies tensors across device boundaries AUTOMATICALLY --")
print("  which is convenient and is also how a silent performance")
print("  disaster happens. Use tf.debugging.set_log_device_placement(True)")
print("  when a job is mysteriously slow.")

print()
print("=== memory growth (must be set BEFORE any tensor is created) ===")
print("  for g in tf.config.list_physical_devices('GPU'):")
print("      tf.config.experimental.set_memory_growth(g, True)")
print("  without it TF grabs the WHOLE GPU on first use, and the second")
print("  process on that machine fails with OOM on an idle card.")

import plotly.graph_objects as go
Bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
fig = go.Figure()
for nm, opt, col in [("Adam", "adam", C["danger"]),
                     ("momentum", "momentum", C["warning"]),
                     ("SGD", "sgd", C["success"])]:
    tot = [memory_report(m.count_params(), int(B), a1, opt)["total"]/1e6
           for B in Bs]
    fig.add_scatter(x=Bs, y=tot, mode="lines+markers", name=nm,
                    line=dict(color=col, width=3))
fig.update_layout(height=400, xaxis_type="log", xaxis_title="batch size",
                  yaxis_title="training memory (MB)",
                  title="Memory = 16N + 4B·Σ|a|")
''',
        key="ch19_gpu",
    )

    keypoints([
        "Training memory $\\approx 16N + 4B\\sum_\\ell|a_\\ell|$ with Adam; only "
        "the activation term responds to batch size.",
        "Inference needs roughly a <b>quarter</b> of training memory.",
        "<b>float16 underflows at $6\\times10^{-5}$</b> — hence loss scaling; "
        "bfloat16 keeps float32's exponent instead.",
        "Under <code>mixed_float16</code>, <b>force the output layer to "
        "float32</b>.",
        "Set <b>memory growth before any tensor exists</b>, or TF takes the whole "
        "GPU.",
    ])


# ==========================================================================
def s_19_5():
    section("19.5", "Data Parallelism, Model Parallelism and Their Limits")

    lead(
        "Two ways to use many devices, with completely different scaling "
        "behaviour. Amdahl's law and the communication cost decide which one you "
        "get to use."
    )

    sub("The two kinds")

    table(
        ["", "Data parallelism", "Model parallelism"],
        [["What is split", "The <b>batch</b>",
          "The <b>model</b> — layers or tensors"],
         ["Each device holds", "A full replica of the model",
          "One shard of it"],
         ["Communication", "<b>Gradients</b>, once per step",
          "<b>Activations</b>, at every shard boundary"],
         ["Scales well when", "The model fits on one device",
          "It does not"],
         ["Efficiency", "<b>High</b> — communication overlaps with compute",
          "Lower — devices wait for each other"],
         ["Difficulty", "Nearly automatic", "<b>Genuinely hard</b>"]],
    )

    derive(
        [("<b>Why data parallelism scales and model parallelism struggles.</b> "
          "Let $T_c$ be the compute time per step on one device and $T_m$ the "
          "communication time.", None),
         ("<b>Data parallel.</b> Each of $K$ devices computes $1/K$ of the batch, "
          "then all-reduces the gradients. Ring all-reduce moves "
          "$2(K-1)/K \\cdot M$ bytes per device for $M$ bytes of gradient — "
          "<b>almost independent of $K$</b>:",
          r"T_{\text{step}} \approx \frac{T_c}{K} + \frac{2M}{\beta}"),
         ("So the speed-up is $K$ until the constant communication term "
          "dominates. Crucially, gradients for early layers are ready while "
          "later layers are still computing, so the all-reduce can be "
          "<b>overlapped</b> with the backward pass and largely hidden.", None),
         ("<b>Model parallel.</b> Split the model into $K$ sequential stages. "
          "Device $k$ cannot start until device $k-1$ finishes, so with naive "
          "splitting only one device works at a time:",
          r"\text{utilisation} = \frac{1}{K}\quad\text{(the pipeline bubble)}"),
         ("<b>Pipeline parallelism</b> fixes this by splitting the batch into $m$ "
          "micro-batches so the stages overlap. The bubble shrinks to:",
          r"\text{bubble fraction} = \frac{K-1}{m + K - 1}"),
         ("With $K = 4$ stages and $m = 32$ micro-batches the bubble is 8.6 %; "
          "with $m = 4$ it is 43 %. <b>The number of micro-batches is the "
          "single most important pipeline hyperparameter.</b>", None),
         ("<b>And the overall ceiling is Amdahl's law.</b> If a fraction $p$ of "
          "the work is parallelisable:",
          r"S(K) = \frac{1}{(1-p) + p/K} \;\xrightarrow{K\to\infty}\;"
          r" \frac{1}{1-p}"),
         ("At $p = 0.95$ the maximum speed-up is <b>20×</b> no matter how many "
          "devices you buy. Data loading, checkpointing and evaluation are all "
          "in that serial 5 %.", None)],
        title="Why data parallelism scales and model parallelism does not",
    )

    sub("Synchronous vs asynchronous")

    table(
        ["", "Synchronous (all-reduce)", "Asynchronous (parameter server)"],
        [["Gradient used", "Averaged across all replicas",
          "Applied as it arrives"],
         ["Speed", "Limited by the <b>slowest</b> replica",
          "No waiting"],
         ["Correctness", "Exactly equivalent to a large-batch step",
          "<b>Stale gradients</b> — computed from old weights"],
         ["In practice", "<b>The default</b> — better final accuracy",
          "Only when replicas are heterogeneous or unreliable"],
         ["Mitigation", "Drop the slowest ~10 % of replicas each step",
          "Limit staleness; scale down stale updates"]],
    )

    warn(
        "The straggler problem is what actually limits synchronous scaling",
        "Every replica waits for the slowest one. With 100 replicas, the "
        "expected step time is the <b>maximum</b> of 100 draws, not the mean — "
        "and if each replica has even a 1 % chance of being slow, the "
        "probability that <i>some</i> replica is slow is $1 - 0.99^{100} = "
        "63\\%$. The standard mitigation is <b>backup replicas</b>: run 10 % "
        "extra and proceed as soon as the first 90 % report.",
    )

    sub("Scaling the learning rate")

    math(r"""
    \eta_{\text{eff}} = \eta_{\text{base}} \times \frac{B_{\text{total}}}{B_{\text{base}}}
    \qquad\text{(linear scaling rule)}
    """)

    tip(
        "Scale the learning rate with the batch, and warm it up",
        "Data parallelism multiplies the effective batch size by $K$. A larger "
        "batch gives a lower-variance gradient, so a proportionally larger step "
        "is both safe and necessary — otherwise you take the same number of "
        "steps at the same size and simply learn $K$ times less per epoch. Goyal "
        "et al. (2017) trained ImageNet in one hour at batch 8 192 with exactly "
        "this rule plus a <b>5-epoch linear warm-up</b>, which is essential: at "
        "initialisation the linear-scaling assumption does not hold and a large "
        "step diverges. Above roughly batch 8 000 the rule breaks down and you "
        "need LARS/LAMB instead.",
    )

    anim_header("The pipeline bubble shrinking as micro-batches increase")

    K_stages = 4
    frames = []
    for m in [1, 2, 4, 8, 16, 32]:
        total_slots = m + K_stages - 1
        bubble = (K_stages - 1) / total_slots
        shapes, ann = [], []
        for stage in range(K_stages):
            for slot in range(total_slots):
                mb = slot - stage
                busy = 0 <= mb < m
                shapes.append(go.Scatter(
                    x=[slot, slot+.92, slot+.92, slot, slot],
                    y=[stage, stage, stage+.85, stage+.85, stage],
                    fill="toself",
                    fillcolor=(alpha(SEQ[mb % len(SEQ)], .85) if busy
                               else alpha(C["line"], .25)),
                    line=dict(color="#fff", width=1),
                    showlegend=False, hoverinfo="skip"))
        for stage in range(K_stages):
            ann.append(dict(x=-0.4, y=stage+.42, text=f"GPU {stage}",
                            showarrow=False, xanchor="right",
                            font=dict(size=10, color=C["ink_soft"])))
        frames.append(go.Frame(name=str(m), data=shapes,
                               layout=go.Layout(annotations=ann + [
                                   anim.annotate_step(
                                       f"{m} micro-batch(es), {K_stages} stages"
                                       f"   ·   bubble = (K−1)/(m+K−1) = "
                                       f"{bubble:.1%}   ·   utilisation "
                                       f"{1-bubble:.1%}",
                                       color=C["success"] if bubble < .2
                                       else C["danger"])])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=380, plot_bgcolor="#FFFFFF",
                    xaxis=dict(title="time slot", range=[-2.2, 36]),
                    yaxis=dict(visible=False, range=[-.5, K_stages+.2]),
                    annotations=list(frames[0].layout.annotations),
                    title="Pipeline parallelism across 4 GPUs")
    anim.animate(f, frames, duration=nav.anim_ms(1400),
                 slider_prefix="micro-batches ")
    figure(f, "Grey cells are idle GPUs. With one micro-batch, three of the four "
              "GPUs are always waiting.")

    code_lab(
        "Amdahl, ring all-reduce, pipeline bubbles and the linear scaling rule",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

np.set_printoptions(precision=4, suppress=True)

# ============ 1. AMDAHL'S LAW ==========================================
def amdahl(p, K):
    return 1.0/((1-p) + p/K)

print("=== Amdahl's law: the ceiling is 1/(1-p) ===")
print(f"{'devices':>9}" + "".join(f"{f'p={p}':>11}"
                                  for p in [0.5, 0.9, 0.95, 0.99, 0.999]))
for K in [2, 4, 8, 16, 64, 256, 1024]:
    print(f"{K:>9}" + "".join(f"{amdahl(p, K):>11.2f}"
                              for p in [0.5, 0.9, 0.95, 0.99, 0.999]))
print(f"{'ceiling':>9}" + "".join(f"{1/(1-p):>11.1f}"
                                  for p in [0.5, 0.9, 0.95, 0.99, 0.999]))
print("  at p=0.95 the maximum speed-up is 20x no matter how many GPUs")
print("  you buy. Data loading, checkpointing and evaluation live in the")
print("  serial 5% -- which is why tf.data pipeline work (ch. 13) pays off")
print("  twice over on a cluster.")

# ============ 2. RING ALL-REDUCE =======================================
print()
print("=== ring all-reduce moves ~2M bytes per device, whatever K is ===")
def allreduce_bytes(M, K, mode="ring"):
    if mode == "ring":
        return 2*(K-1)/K * M              # reduce-scatter + all-gather
    if mode == "naive":
        return (K-1)*M                    # everyone sends to everyone
    if mode == "ps":
        return 2*M                        # to the server and back
    raise ValueError

M = 100e6 * 4                              # 100M parameters, float32
print(f"  a {M/1e6:.0f} MB gradient")
print(f"{'devices':>9}{'ring MB':>12}{'naive MB':>12}{'param-server MB':>18}")
for K in [2, 4, 8, 16, 64, 256]:
    print(f"{K:>9}{allreduce_bytes(M, K)/1e6:>12.1f}"
          f"{allreduce_bytes(M, K, 'naive')/1e6:>12.1f}"
          f"{allreduce_bytes(M, K, 'ps')/1e6:>18.1f}")
print("  ring all-reduce approaches 2M and STOPS. The naive scheme grows")
print("  linearly with K. That single algorithm is why synchronous data")
print("  parallelism scales to thousands of devices.")

# ============ 3. WHEN DOES COMMUNICATION DOMINATE? =====================
print()
print("=== compute vs communication ===")
T_compute = 0.120                          # seconds per step on one device
bandwidth = 25e9                           # 25 GB/s interconnect
print(f"  {T_compute*1000:.0f} ms of compute per step, "
      f"{bandwidth/1e9:.0f} GB/s interconnect, {M/1e6:.0f} MB of gradients")
print(f"{'devices':>9}{'compute ms':>13}{'comms ms':>11}{'total ms':>11}"
      f"{'speed-up':>11}{'efficiency':>13}")
for K in [1, 2, 4, 8, 16, 32, 64, 128]:
    comp = T_compute/K
    comm = allreduce_bytes(M, K)/bandwidth if K > 1 else 0.0
    tot = comp + comm
    print(f"{K:>9}{comp*1000:>13.2f}{comm*1000:>11.2f}{tot*1000:>11.2f}"
          f"{T_compute/tot:>11.2f}x{(T_compute/tot)/K:>13.1%}")
print("  efficiency falls once the constant communication term dominates")
print("  the shrinking compute term. OVERLAPPING the all-reduce with the")
print("  backward pass (gradients for early layers are ready first) hides")
print("  most of it -- which is what makes this work in practice.")

# ============ 4. THE PIPELINE BUBBLE ===================================
print()
print("=== model parallelism: the pipeline bubble ===")
print(f"{'stages K':>10}{'micro-batches m':>18}{'bubble':>10}"
      f"{'utilisation':>14}")
for K in [2, 4, 8]:
    for m in [1, 4, 16, 64]:
        bub = (K-1)/(m+K-1)
        print(f"{K:>10}{m:>18}{bub:>10.1%}{1-bub:>14.1%}")
print("  with one micro-batch, K-1 of every K GPUs are idle at all times.")
print("  the number of micro-batches is THE pipeline hyperparameter.")

# --- but micro-batches cost memory ----------------------------------
print()
print("=== the micro-batch trade-off ===")
print(f"{'micro-batches':>15}{'bubble (K=4)':>15}{'activations kept in flight':>29}")
for m in [1, 4, 8, 16, 32, 64]:
    print(f"{m:>15}{(4-1)/(m+3):>15.1%}{min(m, 4):>29}")
print("  a stage must keep the activations of every micro-batch that is")
print("  still in flight, so more micro-batches means more memory.")
print("  1F1B scheduling caps it at K instead of m.")

# ============ 5. SYNCHRONOUS AND THE STRAGGLER =========================
print()
print("=== why the slowest replica sets the pace ===")
rng = np.random.default_rng(0)
base_ms, slow_prob, slow_mult = 100.0, 0.02, 6.0
print(f"  each replica takes {base_ms:.0f} ms, with a {slow_prob:.0%} chance")
print(f"  of being {slow_mult:.0f}x slower")
print(f"{'replicas':>10}{'P(some straggler)':>20}{'mean step ms':>15}"
      f"{'with 10% backups':>19}")
for K in [1, 4, 16, 64, 256]:
    times = rng.normal(base_ms, 5, (4000, K))
    slow = rng.random((4000, K)) < slow_prob
    times = np.where(slow, times*slow_mult, times)
    sync = times.max(1)
    keep = max(1, int(np.ceil(0.9*K)))
    backup = np.sort(times, 1)[:, keep-1]     # wait for the fastest 90 %
    print(f"{K:>10}{1-(1-slow_prob)**K:>20.1%}{sync.mean():>15.1f}"
          f"{backup.mean():>19.1f}")
print("  at 256 replicas the probability that SOMETHING is slow is ~99%,")
print("  so nearly every step pays the straggler penalty.")
print("  waiting for only the fastest 90% removes almost all of it.")

# ============ 6. ASYNCHRONOUS AND STALE GRADIENTS ======================
print()
print("=== what a stale gradient does ===")
# minimise f(x) = x^2 with gradients computed from DELAYED parameters
def run(delay, lr=0.12, steps=60, x0=4.0):
    hist, xs = [], [x0]*(delay+1)
    x = x0
    for t in range(steps):
        g = 2*xs[-(delay+1)]                   # gradient from a STALE x
        x = x - lr*g
        xs.append(x)
        hist.append(abs(x))
    return hist

print(f"{'staleness':>11}{'|x| after 60 steps':>22}{'behaviour':>22}")
for d in [0, 1, 2, 4, 8]:
    h = run(d)
    final = h[-1]
    beh = ("converged" if final < 1e-3 else
           "slow" if final < 1.0 else
           "OSCILLATING" if final < 100 else "DIVERGED")
    print(f"{d:>11}{final:>22.6f}{beh:>22}")
print("  staleness is a delay in a feedback loop, and delays destabilise")
print("  feedback loops. That is why synchronous training is the default")
print("  despite the straggler cost.")

# ============ 7. THE LINEAR SCALING RULE ===============================
print()
print("=== learning rate must scale with the effective batch ===")
base_lr, base_bs = 0.1, 256
print(f"  baseline: batch {base_bs}, lr {base_lr}")
print(f"{'replicas':>10}{'total batch':>14}{'linear-rule lr':>17}"
       f"{'sqrt-rule lr':>15}{'warm-up needed':>17}")
for K in [1, 2, 4, 8, 32, 128]:
    tb = base_bs*K
    print(f"{K:>10}{tb:>14,}{base_lr*K:>17.3f}"
          f"{base_lr*np.sqrt(K):>15.3f}"
          f"{('yes' if K > 2 else 'no'):>17}")
print("  a bigger batch gives a LOWER-VARIANCE gradient, so a bigger step")
print("  is both safe and necessary -- otherwise you take the same number")
print("  of steps and learn K times less per epoch.")
print("  WARM-UP is essential: the linear-scaling assumption fails at")
print("  initialisation, and a large first step diverges.")
print("  above ~8 000 the rule breaks down; use LARS or LAMB.")

# --- warm-up schedule ------------------------------------------------
print()
print("=== a 5-epoch linear warm-up, 8x replicas ===")
target = base_lr*8
steps_per_epoch = 100
print(f"{'epoch':>7}{'learning rate':>16}")
for ep in range(8):
    lr = target*min(1.0, (ep+1)/5)
    print(f"{ep:>7}{lr:>16.4f}")

import plotly.graph_objects as go
fig = go.Figure()
Ks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
for p, col in [(0.99, C["success"]), (0.95, C["warning"]), (0.9, C["danger"])]:
    fig.add_scatter(x=Ks, y=[amdahl(p, int(k)) for k in Ks],
                    mode="lines+markers", name=f"p = {p}",
                    line=dict(color=col, width=3))
fig.add_scatter(x=Ks, y=Ks, mode="lines", name="perfect scaling",
                line=dict(color=C["muted"], width=2, dash="dot"))
fig.update_layout(height=420, xaxis_type="log", yaxis_type="log",
                  xaxis_title="devices", yaxis_title="speed-up",
                  title="Amdahl's law")
''',
        key="ch19_parallel",
    )

    keypoints([
        "<b>Data parallelism</b> splits the batch and all-reduces gradients; "
        "ring all-reduce moves $\\approx 2M$ bytes regardless of $K$.",
        "<b>Model parallelism</b> splits the model and pays a pipeline bubble of "
        "$(K-1)/(m+K-1)$.",
        "<b>Amdahl</b>: at $p = 0.95$ the ceiling is 20× no matter how many "
        "devices.",
        "<b>Synchronous</b> is the default; stragglers are handled with backup "
        "replicas, not asynchrony.",
        "Scale the learning rate <b>linearly with the total batch</b>, with a "
        "warm-up.",
    ])


# ==========================================================================
def s_19_6():
    section("19.6", "The Distribution Strategies API")

    lead(
        "Keras turns multi-device training into a context manager. The value is "
        "in knowing which strategy matches your hardware — and what each one is "
        "silently doing to your batch size."
    )

    sub("The strategies")

    table(
        ["Strategy", "Hardware", "Sync", "Use when"],
        [["<code>MirroredStrategy</code>", "Several GPUs, <b>one machine</b>",
          "Synchronous, all-reduce", "<b>The common case</b>"],
         ["<code>MultiWorkerMirroredStrategy</code>", "Several machines",
          "Synchronous, all-reduce", "A cluster; needs <code>TF_CONFIG</code>"],
         ["<code>TPUStrategy</code>", "TPU pods", "Synchronous",
          "TPUs, with static shapes"],
         ["<code>ParameterServerStrategy</code>",
          "Workers + parameter servers", "<b>Asynchronous</b>",
          "Heterogeneous or unreliable workers"],
         ["<code>CentralStorageStrategy</code>",
          "One machine; variables on CPU", "Synchronous",
          "GPUs with too little memory for the variables"],
         ["<code>OneDeviceStrategy</code>", "One device", "—",
          "Testing that your code is strategy-clean"]],
    )

    sub("The rules")

    pitfall(
        "Everything that creates a variable must be inside "
        "<code>strategy.scope()</code>",
        "The model, the optimiser, and any metric objects. A variable created "
        "outside the scope is <b>not mirrored</b>, so replicas silently diverge — "
        "or you get an error much later, from somewhere unrelated. The "
        "<code>fit()</code> call itself goes <b>outside</b> the scope. This is "
        "the single most common mistake with the API, and the error messages are "
        "not helpful.",
    )

    derive(
        [("<b>What the batch size means under a strategy.</b> The batch size you "
          "pass to <code>fit()</code> is the <b>global</b> batch, split across "
          "replicas:",
          r"B_{\text{per-replica}} = \frac{B_{\text{global}}}{K}"),
         ("So moving from 1 GPU to 4 with an unchanged <code>batch_size=32</code> "
          "gives each GPU only 8 examples — <b>less</b> parallel work per device, "
          "and often a <i>slower</i> step. You must scale the global batch:",
          r"B_{\text{global}} \leftarrow K \cdot B_{\text{base}}"),
         ("<b>And then scale the learning rate</b> with it (§19.5), or you are "
          "training with a quarter of the effective step size.", None),
         ("<b>The loss reduction also changes.</b> Keras sums the per-replica "
          "losses and divides by the <b>global</b> batch size, not the "
          "per-replica one:",
          r"\mathcal{L} = \frac{1}{B_{\text{global}}}\sum_{k=1}^{K}"
          r"\sum_{i \in \text{replica } k} \ell_i"),
         ("In a <b>custom</b> training loop this is not automatic — using "
          "<code>tf.reduce_mean</code> per replica divides by the per-replica "
          "size and inflates the gradient by $K$. Use "
          "<code>tf.nn.compute_average_loss(losses, "
          "global_batch_size=...)</code>.", None)],
        title="Global vs per-replica batch, and the loss-reduction trap",
    )

    codenote(
        "The input pipeline must be shardable",
        "Under a multi-worker strategy, each worker should see a different slice "
        "of the data. <code>tf.data</code> handles this automatically for a "
        "dataset built from files, via <code>AutoShardPolicy.FILE</code> — but "
        "that requires <b>at least as many files as workers</b>. With one large "
        "TFRecord file and eight workers, seven of them get nothing, and the "
        "symptom is a training run that looks fine and learns eight times too "
        "slowly. Shard your data into many files, or set the policy to "
        "<code>DATA</code> (each worker reads everything and discards $K-1$ of "
        "every $K$ records — correct, but wasteful).",
    )

    anim_header("One synchronous step across four replicas")

    phases = [
        ("broadcast", "the same weights on every replica", C["muted"]),
        ("forward", "each replica processes its own shard of the batch",
         C["primary"]),
        ("backward", "each computes gradients from its own data", C["accent"]),
        ("all-reduce", "gradients are averaged across replicas — the only "
         "communication", C["danger"]),
        ("apply", "every replica applies the SAME averaged gradient", C["success"]),
    ]
    frames = []
    for pi, (nm, desc, col) in enumerate(phases):
        shapes, ann = [], []
        for r_ in range(4):
            active = True
            shapes.append(go.Scatter(
                x=[r_*2, r_*2+1.5, r_*2+1.5, r_*2, r_*2],
                y=[0, 0, 1.2, 1.2, 0], fill="toself",
                fillcolor=alpha(col, .85), line=dict(color="#fff", width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=r_*2+.75, y=.6, text=f"GPU {r_}",
                            showarrow=False,
                            font=dict(size=11, color="#fff")))
        if nm == "all-reduce":
            for r_ in range(3):
                shapes.append(go.Scatter(
                    x=[r_*2+1.5, r_*2+2.0], y=[.6, .6], mode="lines",
                    line=dict(color=C["ink"], width=4),
                    showlegend=False, hoverinfo="skip"))
            shapes.append(go.Scatter(
                x=[0.75, 0.75, 6.75, 6.75], y=[1.2, 1.7, 1.7, 1.2],
                mode="lines", line=dict(color=C["ink"], width=3, dash="dot"),
                showlegend=False, hoverinfo="skip"))
        frames.append(go.Frame(name=nm, data=shapes,
                               layout=go.Layout(annotations=ann + [
                                   anim.annotate_step(
                                       f"{pi+1}/5  {nm.upper()}  ·  {desc}",
                                       color=col)])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=330, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.5, 8]),
                    yaxis=dict(visible=False, range=[-.8, 2.1]),
                    annotations=list(frames[0].layout.annotations),
                    title="MirroredStrategy: one step")
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="phase ")
    figure(f, "Only the all-reduce involves communication, and it overlaps with "
              "the backward pass in practice.")

    code_lab(
        "MirroredStrategy on virtual devices — runs on a CPU-only machine",
        '''import numpy as np, time, os, json
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

# ============ 1. CREATE VIRTUAL DEVICES ================================
# This lets you TEST distributed code without a GPU cluster. It must run
# before any tensor is created, which is why it is the first thing here.
N_REPLICAS = 4
cpus = tf.config.list_physical_devices("CPU")
try:
    tf.config.set_logical_device_configuration(
        cpus[0], [tf.config.LogicalDeviceConfiguration()
                  for _ in range(N_REPLICAS)])
    made = True
except RuntimeError as e:
    made = False
    print(f"  (virtual devices already configured: {str(e)[:70]})")

logical = tf.config.list_logical_devices("CPU")
gpus = tf.config.list_logical_devices("GPU")
print("=== devices ===")
print(f"  logical CPUs: {len(logical)}   GPUs: {len(gpus)}")
for d in logical[:6]:
    print(f"    {d.name}")
print("  set_logical_device_configuration lets you test MirroredStrategy")
print("  on a laptop. The code is IDENTICAL to what runs on 8 GPUs.")

# ============ 2. THE STRATEGY ==========================================
devices = [d.name for d in (gpus if gpus else logical)]
strategy = tf.distribute.MirroredStrategy(devices=devices)
K = strategy.num_replicas_in_sync
print()
print(f"=== MirroredStrategy over {K} replicas ===")
print(f"  num_replicas_in_sync = {K}")

# ============ 3. GLOBAL vs PER-REPLICA BATCH ===========================
print()
print("=== the batch size you pass is the GLOBAL one ===")
print(f"{'batch_size you pass':>22}{'per replica':>14}{'comment':>34}")
for B in [32, 64, 128, 256]:
    per = B // K
    note = ("each GPU gets very little work" if per < 16
            else "reasonable" if per < 128 else "large")
    print(f"{B:>22}{per:>14}{note:>34}")
print(f"  moving from 1 device to {K} with an UNCHANGED batch_size means")
print(f"  each replica does 1/{K} of the work it used to -- often a SLOWER")
print(f"  step, not a faster one. Scale the global batch by {K}.")

# ============ 4. TRAIN, WITH EVERYTHING IN THE RIGHT PLACE =============
Xtr, ytr, Xte, yte, labels, real = _ds.fashion_mnist(n_train=12000, n_test=2000)
Xtr = Xtr.astype("float32")[..., None]; Xte = Xte.astype("float32")[..., None]

def build():
    return keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")])

BASE_BS, BASE_LR = 64, 1e-3
GLOBAL_BS, LR = BASE_BS*K, BASE_LR*K          # scale BOTH together

print()
print("=== training under the strategy ===")
print(f"  global batch {GLOBAL_BS} ({GLOBAL_BS//K} per replica), "
      f"lr {LR:.4f} (base {BASE_LR} x {K})")

# EVERYTHING that creates a variable goes inside the scope
with strategy.scope():
    model = build()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(LR),
                  metrics=["accuracy"])

t0 = time.perf_counter()
# fit() goes OUTSIDE the scope
hist = model.fit(Xtr, ytr, epochs=4, batch_size=GLOBAL_BS, verbose=0,
                 validation_data=(Xte, yte))
dt = time.perf_counter()-t0
print(f"  {dt:.1f}s, final accuracy "
      f"{hist.history['val_accuracy'][-1]:.4f}")

# --- the same thing on one device, for comparison --------------------
tf.random.set_seed(0)
single = build()
single.compile(loss="sparse_categorical_crossentropy",
               optimizer=keras.optimizers.Adam(BASE_LR), metrics=["accuracy"])
t0 = time.perf_counter()
h1 = single.fit(Xtr, ytr, epochs=4, batch_size=BASE_BS, verbose=0,
                validation_data=(Xte, yte))
dt1 = time.perf_counter()-t0
print(f"  single device, batch {BASE_BS}: {dt1:.1f}s, accuracy "
      f"{h1.history['val_accuracy'][-1]:.4f}")
print()
print("  NOTE: on VIRTUAL CPU devices there is no real speed-up -- they")
print("  share the same cores. The point of this lab is that the CODE")
print("  is correct and runs; on 4 real GPUs the wall-clock would fall.")

# ============ 5. THE VARIABLES ARE MIRRORED ============================
print()
print("=== what 'mirrored' means ===")
v = model.trainable_weights[0]
print(f"  first weight is a {type(v).__name__}")
if hasattr(v, "values"):
    print(f"  it has {len(v.values)} per-replica copies:")
    for i, rep in enumerate(v.values[:3]):
        print(f"    replica {i}: device {rep.device.split('/')[-1]}, "
              f"first value {float(tf.reshape(rep, [-1])[0]):.6f}")
    same = all(np.allclose(v.values[0].numpy(), r.numpy())
               for r in v.values)
    print(f"  all replicas identical after a step: {same}")
    print("  they MUST be -- that is what the all-reduce guarantees.")
else:
    print(f"  (single replica, so it is an ordinary {type(v).__name__})")

# ============ 6. THE LOSS-REDUCTION TRAP ===============================
print()
print("=== the custom-training-loop trap ===")
print(f"  Keras divides the summed loss by the GLOBAL batch ({GLOBAL_BS}).")
print(f"  a custom loop using tf.reduce_mean divides by the PER-REPLICA")
print(f"  batch ({GLOBAL_BS//K}) instead, which inflates the gradient by {K}x.")
losses = np.arange(1, GLOBAL_BS+1, dtype="float32")
per_replica = np.split(losses, K)
print()
print(f"  correct   : sum(all losses) / {GLOBAL_BS} = "
      f"{losses.sum()/GLOBAL_BS:.4f}")
wrong = np.mean([r.mean() for r in per_replica])
print(f"  per-replica reduce_mean, then averaged = {wrong:.4f}")
print(f"    (equal here only because the shards are the same size)")
print(f"  the real danger is a RAGGED last batch, where the shards differ")
print(f"  in size and the two disagree. Use:")
print(f"    tf.nn.compute_average_loss(losses, global_batch_size=B)")

# --- a correct custom loop -------------------------------------------
print()
print("=== a correct custom distributed training step ===")
with strategy.scope():
    m2 = build()
    opt2 = keras.optimizers.Adam(LR)
    loss_obj = keras.losses.SparseCategoricalCrossentropy(
        reduction=None)                       # NO reduction -- we do it

    def step_fn(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            per_example = loss_obj(y, m2(x, training=True))
            loss = tf.nn.compute_average_loss(
                per_example, global_batch_size=GLOBAL_BS)   # THE RIGHT WAY
        opt2.apply_gradients(zip(tape.gradient(loss, m2.trainable_variables),
                                 m2.trainable_variables))
        return loss

    @tf.function
    def dist_step(inputs):
        losses = strategy.run(step_fn, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

ds = tf.data.Dataset.from_tensor_slices((Xtr[:4096], ytr[:4096]))
ds = ds.batch(GLOBAL_BS).prefetch(1)
dist_ds = strategy.experimental_distribute_dataset(ds)
tot, n = 0.0, 0
for batch in dist_ds:
    tot += float(dist_step(batch)); n += 1
print(f"  ran {n} distributed steps, mean loss {tot/n:.4f}")
print("  strategy.run() executes step_fn on EVERY replica;")
print("  strategy.reduce() combines the results.")

# ============ 7. INPUT SHARDING ========================================
print()
print("=== auto-sharding ===")
opts = tf.data.Options()
opts.experimental_distribute.auto_shard_policy = \\
    tf.data.experimental.AutoShardPolicy.DATA
print(f"  AutoShardPolicy.FILE : each worker reads DIFFERENT files")
print(f"                         -> needs >= as many files as workers")
print(f"  AutoShardPolicy.DATA : each worker reads everything and keeps")
print(f"                         1 record in K -- correct but wasteful")
print(f"  AutoShardPolicy.OFF  : no sharding (you handle it)")
print()
print("  the classic failure: ONE big TFRecord and 8 workers with the FILE")
print("  policy. Seven workers get nothing. Training looks fine and is")
print("  8x slower than it should be, with no error.")

# ============ 8. MULTI-WORKER: TF_CONFIG ===============================
print()
print("=== what a multi-worker setup needs ===")
example = {"cluster": {"worker": ["10.0.0.1:12345", "10.0.0.2:12345",
                                  "10.0.0.3:12345"]},
           "task": {"type": "worker", "index": 0}}
print("  each machine sets the TF_CONFIG environment variable:")
for line in json.dumps(example, indent=4).splitlines():
    print(f"    {line}")
print("  ... with a different task.index on each. Then:")
print("    strategy = tf.distribute.MultiWorkerMirroredStrategy()")
print("  and the rest of the code is UNCHANGED. That is the whole point")
print("  of the API: the same model code runs on 1 GPU or 100.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(y=hist.history["val_accuracy"], mode="lines+markers",
                name=f"MirroredStrategy ({K} replicas)",
                line=dict(color=C["primary"], width=3))
fig.add_scatter(y=h1.history["val_accuracy"], mode="lines+markers",
                name="single device",
                line=dict(color=C["muted"], width=2, dash="dash"))
fig.update_layout(height=400, xaxis_title="epoch",
                  yaxis_title="validation accuracy",
                  title="Same code, same result, different device count")
''',
        key="ch19_strategy",
    )

    keypoints([
        "<code>MirroredStrategy</code> for one machine, "
        "<code>MultiWorkerMirroredStrategy</code> for a cluster.",
        "<b>Everything that creates a variable goes inside "
        "<code>strategy.scope()</code></b>; <code>fit()</code> goes outside.",
        "<code>batch_size</code> is the <b>global</b> batch — scale it by $K$, "
        "and the learning rate with it.",
        "In a custom loop use <code>tf.nn.compute_average_loss(..., "
        "global_batch_size=)</code>, never <code>reduce_mean</code>.",
        "<b>Shard your data into many files</b> — the FILE auto-shard policy "
        "starves workers otherwise.",
    ])

# ==========================================================================
def s_19_7():
    section("19.7", "Training at Scale — Pipelines, Tuning and Cost")

    lead(
        "On a cluster, the bottleneck is usually not the GPU. It is the input "
        "pipeline, the hyperparameter search strategy, or the fact that nobody "
        "checked what the job costs."
    )

    sub("The input pipeline is the usual bottleneck")

    derive(
        [("A training step is a producer–consumer system. If the pipeline "
          "produces a batch every $T_p$ seconds and the GPU consumes one every "
          "$T_g$, the achieved step time is:",
          r"T_{\text{step}} = \max\bigl(T_p,\; T_g\bigr)"),
         ("<b>Only with prefetching.</b> Without it the two are serial:",
          r"T_{\text{step}}^{\text{no prefetch}} = T_p + T_g"),
         ("So <code>prefetch(AUTOTUNE)</code> alone gives a speed-up of "
          "$(T_p + T_g)/\\max(T_p, T_g)$ — up to <b>2×</b> when the two are "
          "balanced, and it is one line.", None),
         ("<b>GPU utilisation</b> is exactly:",
          r"U = \frac{T_g}{\max(T_p, T_g)}"),
         ("If your GPU is at 40 % utilisation, the pipeline takes 2.5× longer "
          "than the compute, and <b>buying a faster GPU changes nothing</b>. "
          "Profile before you upgrade.", None),
         ("The fixes, in order of usual payoff: <code>prefetch</code>, "
          "<code>num_parallel_calls=AUTOTUNE</code> on the map, "
          "<code>interleave</code> across many files, <code>cache()</code> after "
          "the expensive deterministic part, and — the big one — <b>do the "
          "expensive preprocessing once, offline</b>, and write TFRecords "
          "(§13.4).", None)],
        title="Why the GPU is idle",
    )

    sub("Hyperparameter search")

    table(
        ["Method", "Cost for $d$ dimensions", "When"],
        [["<b>Grid search</b>", "$k^d$ — exponential",
          "Two or three parameters, and you want exhaustive coverage"],
         ["<b>Random search</b>", "Any budget",
          "<b>The default baseline</b>; beats grid whenever some dimensions "
          "do not matter"],
         ["<b>Bayesian optimisation</b>", "Any budget, sequential",
          "Expensive evaluations, few parameters (< 20)"],
         ["<b>Hyperband / ASHA</b>", "Any budget, parallel",
          "<b>The practical winner</b> — kills bad configurations early"],
         ["<b>Population-based training</b>", "A fixed population",
          "Schedules that should change <i>during</i> training"]],
    )

    proof(
        "Why random search beats grid search — the classic Bergstra–Bengio "
        "argument",
        "Suppose only 1 of your $d$ hyperparameters actually matters. A grid "
        "with $k$ values per dimension costs $k^d$ evaluations but tries only "
        "<b>$k$ distinct values</b> of the parameter that matters — every other "
        "axis is wasted work. Random search with the same budget tries $k^d$ "
        "distinct values of it. Concretely: to land in the best 5 % of the "
        "important dimension with probability 95 %, you need "
        "$n \\ge \\log(0.05)/\\log(0.95) \\approx 59$ random trials — "
        "<b>regardless of $d$</b>. That independence from dimension is the whole "
        "result, and it is why random search is the honest baseline that any "
        "fancier method must beat.",
    )

    math(r"""
    P(\text{at least one trial in the top } \alpha) = 1 - (1-\alpha)^{n}
    \;\Longrightarrow\;
    n \ge \frac{\log(1-P)}{\log(1-\alpha)}
    """)

    idea(
        "Successive halving is the idea behind every modern tuner",
        "Run $n$ configurations for a small budget, keep the best $1/\\eta$, "
        "multiply their budget by $\\eta$, repeat. Total cost is roughly "
        "$n \\cdot b \\cdot \\log_\\eta n$ instead of $n \\cdot b_{\\max}$ — an "
        "order of magnitude cheaper for the same final quality. The assumption "
        "is that a configuration's early performance <b>correlates</b> with its "
        "final performance, which is usually true and occasionally very false "
        "(a warm-up schedule looks terrible for the first few epochs). "
        "Hyperband hedges by running several brackets with different "
        "aggressiveness.",
    )

    sub("Cost")

    warn(
        "Compute the cost before you launch the job, not after",
        "8 × A100 at roughly $30/hour for 72 hours is <b>$2 160 for one run</b>. "
        "A 50-configuration hyperparameter search at that scale is $108 000. "
        "Three practical rules: use <b>spot/preemptible instances</b> (60–90 % "
        "cheaper, and checkpointing makes preemption a non-event); "
        "<b>always set a wall-clock limit</b> so a hung job cannot run all "
        "weekend; and <b>tune on a subset first</b> — hyperparameters found on "
        "10 % of the data usually transfer, and cost 10 % as much to find.",
    )

    anim_header("Successive halving discarding bad configurations")

    rng = np.random.default_rng(5)
    n_conf = 27
    final_quality = rng.beta(2, 5, n_conf)
    final_quality[rng.integers(0, n_conf, 2)] = rng.uniform(.85, .95, 2)
    budgets = [1, 3, 9, 27]
    alive = list(range(n_conf))
    stages = []
    for si, b in enumerate(budgets):
        noise = rng.normal(0, 0.22/np.sqrt(b), n_conf)
        observed = np.clip(final_quality + noise, 0, 1)
        stages.append((b, list(alive), observed.copy()))
        keep = max(1, len(alive)//3)
        alive = [i for i in sorted(alive, key=lambda j: -observed[j])[:keep]]

    frames = []
    for si, (b, act, obs) in enumerate(stages):
        cols = [C["success"] if i in act else alpha(C["line"], .35)
                for i in range(n_conf)]
        ys = [obs[i] if i in act else 0 for i in range(n_conf)]
        cost_sh = sum(len(s[1])*s[0] for s in stages[:si+1])
        cost_full = n_conf*budgets[-1]
        frames.append(go.Frame(name=str(b), data=[
            go.Bar(x=np.arange(n_conf), y=ys, marker=dict(color=cols)),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"budget {b} epoch(s) each   ·   {len(act)} configurations alive"
            f"   ·   cumulative cost {cost_sh} epochs   vs   "
            f"{cost_full} for training all of them fully   "
            f"({cost_sh/cost_full:.0%})")])))

    f = go.Figure(data=[go.Bar(x=np.arange(n_conf), y=stages[0][2],
                              marker=dict(color=[C["success"]]*n_conf))])
    f.update_layout(height=420, xaxis_title="configuration",
                    yaxis_title="observed validation score",
                    yaxis=dict(range=[0, 1.05]),
                    title="Successive halving, η = 3")
    anim.animate(f, frames, duration=nav.anim_ms(1600), slider_prefix="budget ")
    figure(f, "Each round keeps the best third and triples their budget. The "
              "same final answer for roughly a fifth of the compute.")

    code_lab(
        "Pipeline profiling, random vs grid search, and successive halving",
        '''import numpy as np, time, os
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

Xtr, ytr, Xte, yte, labels, real = _ds.fashion_mnist(n_train=12000, n_test=2000)
Xtr = Xtr.astype("float32")[..., None]; Xte = Xte.astype("float32")[..., None]

# ============ 1. THE PIPELINE IS USUALLY THE BOTTLENECK ================
print("=== T_step = max(T_pipeline, T_gpu), but only WITH prefetch ===")

def expensive_preprocess(x, y):
    """Stand in for real augmentation: several ops per example."""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    for _ in range(3):
        x = tf.nn.avg_pool2d(x[None], 3, 1, "SAME")[0]
    return x, y

def make_ds(parallel, prefetch, cache=False, batch=64):
    d = tf.data.Dataset.from_tensor_slices((Xtr, ytr))
    if cache:
        d = d.cache()
    d = d.map(expensive_preprocess,
              num_parallel_calls=(tf.data.AUTOTUNE if parallel else None))
    d = d.batch(batch)
    if prefetch:
        d = d.prefetch(tf.data.AUTOTUNE)
    return d

def time_pipeline(d, n=60):
    it = iter(d)
    next(it)                                   # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        try:
            next(it)
        except StopIteration:
            it = iter(d)
    return (time.perf_counter()-t0)/n

print(f"{'configuration':<44}{'ms / batch':>13}{'speed-up':>11}")
base = None
for nm, kw in [("map(), batch()  -- no options", dict(parallel=False,
                                                      prefetch=False)),
               ("+ num_parallel_calls=AUTOTUNE", dict(parallel=True,
                                                      prefetch=False)),
               ("+ prefetch(AUTOTUNE)", dict(parallel=True, prefetch=True)),
               ("+ cache() before the map", dict(parallel=True, prefetch=True,
                                                 cache=True))]:
    t = time_pipeline(make_ds(**kw))
    base = base or t
    print(f"{nm:<44}{t*1000:>13.2f}{base/t:>10.2f}x")

# ============ 2. GPU UTILISATION =======================================
print()
print("=== how busy is the accelerator, really? ===")
model = keras.Sequential([keras.layers.Input(shape=(28, 28, 1)),
                          keras.layers.Conv2D(32, 3, padding="same",
                                              activation="relu"),
                          keras.layers.MaxPool2D(),
                          keras.layers.Flatten(),
                          keras.layers.Dense(10, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3))

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # the reduction MUST be inside the tape, or the gradient is None
        loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(
            y, model(x, training=True)))
    model.optimizer.apply_gradients(
        zip(tape.gradient(loss, model.trainable_weights),
            model.trainable_weights))

xb = tf.constant(Xtr[:64]); yb = tf.constant(ytr[:64])
train_step(xb, yb)
t0 = time.perf_counter()
for _ in range(60):
    train_step(xb, yb)
T_gpu = (time.perf_counter()-t0)/60

T_pipe_bad = time_pipeline(make_ds(False, False))
T_pipe_good = time_pipeline(make_ds(True, True, cache=True))
print(f"  compute per step        : {T_gpu*1000:>8.2f} ms")
print(f"  naive pipeline per batch: {T_pipe_bad*1000:>8.2f} ms")
print(f"  tuned pipeline per batch: {T_pipe_good*1000:>8.2f} ms")
print()
for nm, tp in [("naive", T_pipe_bad), ("tuned", T_pipe_good)]:
    step = max(tp, T_gpu)
    print(f"  {nm:<7} -> step {step*1000:>7.2f} ms, "
          f"accelerator utilisation {T_gpu/step:>6.1%}")
print("  if utilisation is 40%, a faster GPU changes NOTHING. Profile the")
print("  pipeline before you upgrade the hardware.")

# ============ 3. GRID vs RANDOM SEARCH =================================
print()
print("="*66)
print("Why random search beats grid search")
print("="*66)
print("  suppose only ONE of d hyperparameters actually matters")
print(f"{'d':>4}{'grid k=4':>12}{'distinct values of the ONE that matters':>44}")
for d in [1, 2, 3, 5]:
    print(f"{d:>4}{4**d:>12}{4:>44}")
print("  a grid tries the same 4 values of the important parameter no")
print("  matter how much compute you spend. Random search with the same")
print("  budget tries 4^d DISTINCT values of it.")

print()
print("=== how many random trials do you need? ===")
print(f"{'target quantile':>17}{'P=0.90':>10}{'P=0.95':>10}{'P=0.99':>10}")
for alpha in [0.20, 0.10, 0.05, 0.01]:
    row = [int(np.ceil(np.log(1-P)/np.log(1-alpha))) for P in (.90, .95, .99)]
    print(f"{f'top {alpha:.0%}':>17}{row[0]:>10}{row[1]:>10}{row[2]:>10}")
print("  59 random trials put you in the top 5% with 95% probability --")
print("  INDEPENDENT OF THE NUMBER OF DIMENSIONS. That is the whole result.")

# --- demonstrate it ---------------------------------------------------
print()
print("=== simulated: a 5-D space where only dimension 0 matters ===")
rng = np.random.default_rng(0)
def objective(p):
    return np.exp(-((p[0]-0.63)**2)/0.004)      # dims 1..4 are IGNORED

print(f"{'method':<28}{'evaluations':>13}{'best score':>13}")
for nm, pts in [("grid 4^5", np.stack(np.meshgrid(*[np.linspace(0, 1, 4)]*5),
                                      -1).reshape(-1, 5)),
                ("grid 6^5", np.stack(np.meshgrid(*[np.linspace(0, 1, 6)]*5),
                                      -1).reshape(-1, 5)),
                ("random 1024", rng.random((1024, 5))),
                ("random 100", rng.random((100, 5))),
                ("random 59", rng.random((59, 5)))]:
    scores = np.array([objective(p) for p in pts])
    print(f"{nm:<28}{len(pts):>13,}{scores.max():>13.4f}")
print("  random search with 59 evaluations beats a 7 776-point grid,")
print("  because the grid spends 6^4 = 1 296 evaluations per useful one.")

# ============ 4. SUCCESSIVE HALVING ====================================
print()
print("="*66)
print("Successive halving / ASHA")
print("="*66)
def build_model(lr, width, dropout):
    return keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(width, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation="softmax")]), lr

rng = np.random.default_rng(1)
N_CONF, ETA = 18, 3
configs = [dict(lr=float(10**rng.uniform(-4, -1.5)),
                width=int(2**rng.integers(4, 8)),
                dropout=float(rng.uniform(0, 0.6)))
           for _ in range(N_CONF)]

state = {}
for i, c in enumerate(configs):
    m, lr = build_model(**c)
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(lr), metrics=["accuracy"])
    state[i] = m

alive = list(range(N_CONF))
budget, spent = 1, 0
print(f"{'round':>7}{'alive':>7}{'epochs each':>13}{'cost so far':>13}"
      f"{'best accuracy':>16}")
rnd = 0
while len(alive) > 1:
    scores = {}
    for i in alive:
        state[i].fit(Xtr, ytr, epochs=budget, batch_size=128, verbose=0)
        scores[i] = state[i].evaluate(Xte, yte, verbose=0,
                                      return_dict=True)["accuracy"]
        spent += budget
    rnd += 1
    print(f"{rnd:>7}{len(alive):>7}{budget:>13}{spent:>13}"
          f"{max(scores.values()):>16.4f}")
    keep = max(1, len(alive)//ETA)
    alive = sorted(alive, key=lambda j: -scores[j])[:keep]
    budget *= ETA

best_i = alive[0]
print()
print(f"  winner: {configs[best_i]}")
print(f"  total cost {spent} epochs")
full_cost = N_CONF * budget
print(f"  training all {N_CONF} configurations for {budget} epochs would "
      f"cost {full_cost}")
print(f"  successive halving used {spent/full_cost:.0%} of that.")

# --- the assumption it rests on --------------------------------------
print()
print("=== the assumption: early performance predicts final performance ===")
print("  usually true. Notably FALSE for:")
print("    - a warm-up schedule (looks terrible for the first epochs)")
print("    - a high learning rate that needs decay to pay off")
print("    - anything with a long-tail curriculum")
print("  Hyperband hedges by running several brackets, from very")
print("  aggressive to no early stopping at all.")

# ============ 5. COST ==================================================
print()
print("=== what does a job actually cost? ===")
prices = {"1x T4": 0.35, "1x V100": 2.48, "1x A100": 3.67, "8x A100": 29.4}
print(f"{'hardware':<12}{'$/hour':>9}{'24h':>10}{'72h':>10}"
      f"{'50-config search (72h)':>25}")
for hw, p in prices.items():
    print(f"{hw:<12}{p:>9.2f}{p*24:>10,.0f}{p*72:>10,.0f}"
          f"{p*72*50:>25,.0f}")
print()
print("  three rules:")
print("   1. SPOT / PREEMPTIBLE instances are 60-90% cheaper. Checkpoint")
print("      every N steps and preemption becomes a non-event.")
print("   2. ALWAYS set a wall-clock limit. A hung job runs all weekend.")
print("   3. TUNE ON A SUBSET. Hyperparameters found on 10% of the data")
print("      usually transfer, and cost 10% as much to find.")

print()
print("=== rule 3, tested ===")
print(f"{'tuned on':<22}{'best lr found':>16}{'accuracy on FULL data':>25}")
for nm, frac in [("10% of the data", 0.1), ("100% of the data", 1.0)]:
    n = int(frac*len(Xtr))
    best, best_lr = -1, None
    for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
        tf.random.set_seed(0)
        m = keras.Sequential([keras.layers.Input(shape=(28, 28, 1)),
                              keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="relu"),
                              keras.layers.Dense(10, activation="softmax")])
        m.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(lr), metrics=["accuracy"])
        m.fit(Xtr[:n], ytr[:n], epochs=4, batch_size=128, verbose=0)
        a = m.evaluate(Xte, yte, verbose=0, return_dict=True)["accuracy"]
        if a > best:
            best, best_lr = a, lr
    tf.random.set_seed(0)
    m = keras.Sequential([keras.layers.Input(shape=(28, 28, 1)),
                          keras.layers.Flatten(),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(best_lr), metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=6, batch_size=128, verbose=0)
    print(f"{nm:<22}{best_lr:>16.0e}"
          f"{m.evaluate(Xte, yte, verbose=0, return_dict=True)['accuracy']:>25.4f}")

import plotly.graph_objects as go
fig = go.Figure()
pipes = ["no options", "+parallel map", "+prefetch", "+cache"]
kws = [dict(parallel=False, prefetch=False), dict(parallel=True, prefetch=False),
       dict(parallel=True, prefetch=True),
       dict(parallel=True, prefetch=True, cache=True)]
vals = [time_pipeline(make_ds(**k))*1000 for k in kws]
fig.add_bar(x=pipes, y=vals, marker=dict(color=SEQ[:4]),
            text=[f"{v:.1f} ms" for v in vals], textposition="outside")
fig.add_hline(y=T_gpu*1000, line_dash="dash", line_color=C["danger"],
              annotation_text="compute time per step")
fig.update_layout(height=400, yaxis_title="ms per batch",
                  title="Input pipeline options")
''',
        key="ch19_scale",
    )

    keypoints([
        "$T_{\\text{step}} = \\max(T_p, T_g)$ <b>only with prefetch</b>; without "
        "it the two are serial.",
        "If utilisation is low, <b>the pipeline is the bottleneck</b> and a "
        "faster GPU buys nothing.",
        "<b>59 random trials</b> reach the top 5 % with 95 % probability, "
        "independent of dimension.",
        "<b>Successive halving</b> gets the same answer for ~20 % of the compute "
        "— if early scores predict final ones.",
        "Use spot instances, set a wall-clock limit, and <b>tune on a subset</b>.",
    ])


# ==========================================================================
def s_19_8():
    section("19.8", "Monitoring, Drift and Retraining")

    lead(
        "A deployed model degrades. Not because the code rots, but because the "
        "world moves and the model does not."
    )

    sub("Three kinds of drift")

    table(
        ["Type", "What changes", "Detectable without labels?", "Example"],
        [["<b>Covariate shift</b>", "$P(X)$; $P(Y \\mid X)$ unchanged",
          "<b>Yes</b> — compare input distributions",
          "A new customer segment starts using the product"],
         ["<b>Label shift</b>", "$P(Y)$; $P(X \\mid Y)$ unchanged",
          "Partly — the prediction distribution shifts",
          "Fraud rate rises during a holiday"],
         ["<b>Concept drift</b>", "<b>$P(Y \\mid X)$ itself</b>",
          "<b>No</b> — you need labels",
          "Fraudsters change tactics; the same features now mean something "
          "different"]],
    )

    warn(
        "Concept drift is the dangerous one, and it is invisible without labels",
        "Covariate shift shows up in the inputs and can be monitored in real "
        "time. Concept drift changes the <i>relationship</i> the model learned, "
        "and no amount of input monitoring will reveal it — the inputs can look "
        "identical while the correct answer has changed. The only detector is "
        "<b>ground truth</b>, which arrives late (a loan default takes months) or "
        "never. Plan the label-collection path <b>before</b> you deploy, not "
        "after the model has quietly been wrong for a quarter.",
    )

    sub("The Population Stability Index")

    math(r"""
    \mathrm{PSI} = \sum_{i=1}^{B}\bigl(p_i - q_i\bigr)\,
      \ln\!\frac{p_i}{q_i}
    """)

    proof(
        "PSI is a symmetrised KL divergence, which is why it is used and not KL",
        "Expand: $\\sum (p_i - q_i)\\ln(p_i/q_i) = "
        "\\sum p_i\\ln(p_i/q_i) + \\sum q_i\\ln(q_i/p_i) = "
        "D_{\\mathrm{KL}}(p\\Vert q) + D_{\\mathrm{KL}}(q\\Vert p)$. So PSI is "
        "the <b>Jeffreys divergence</b> — symmetric, unlike KL, which matters "
        "because there is no principled reason to call one of two production "
        "windows the 'reference'. The conventional thresholds are <b>&lt; 0.1 "
        "stable, 0.1–0.25 investigate, &gt; 0.25 significant shift</b>; they are "
        "banking-industry folklore rather than statistics, but they are "
        "well-calibrated in practice and everyone uses them.",
    )

    sub("What to monitor")

    table(
        ["Layer", "Signal", "Alert on"],
        [["<b>Infrastructure</b>", "p50/p99 latency, error rate, QPS, memory",
          "Anything a normal service alerts on"],
         ["<b>Inputs</b>",
          "PSI per feature, null rate, out-of-range rate, cardinality",
          "PSI > 0.25, or a null rate that jumps"],
         ["<b>Predictions</b>",
          "The output distribution, the confidence distribution",
          "A shift, or confidence collapsing to one class"],
         ["<b>Outcomes</b> (needs labels)",
          "Accuracy, AUC, calibration, business metric",
          "<b>The only thing that truly matters</b>"],
         ["<b>Fairness</b>", "Every metric above, <b>sliced by subgroup</b>",
          "Aggregate metrics hide subgroup failures completely"]],
    )

    pitfall(
        "Aggregate accuracy hides subgroup failure",
        "A model can be 95 % accurate overall and 60 % accurate on a group that "
        "is 5 % of the traffic — the aggregate barely moves. <b>Slice every "
        "metric</b> by the dimensions you care about (region, device, language, "
        "customer tenure, and any protected attribute) and alert on the "
        "<i>worst</i> slice, not the mean. This is both an ethical requirement "
        "and, very often, where the largest available accuracy improvement is "
        "hiding.",
    )

    sub("Retraining")

    table(
        ["Trigger", "How", "Risk"],
        [["<b>Scheduled</b> (nightly, weekly)", "Simplest; a cron job",
          "Retrains when nothing has changed; wastes compute"],
         ["<b>Drift-triggered</b>", "Retrain when PSI or accuracy crosses a "
          "threshold",
          "A noisy detector causes thrashing"],
         ["<b>Continuous / online</b>", "Update from a stream",
          "<b>Feedback loops</b> — the model's own outputs become its data"],
         ["<b>Manual</b>", "A human decides", "Slow, but auditable"]],
    )

    warn(
        "Watch for feedback loops when the model influences its own training data",
        "A recommender only shows items it already scores highly, so it only "
        "gets feedback on those, so it becomes more confident about them. A "
        "fraud model that blocks transactions never learns whether they were "
        "actually fraudulent. A predictive-policing model directs patrols "
        "somewhere, which generates arrests there, which confirms the "
        "prediction. The remedies are <b>deliberate exploration</b> (serve a "
        "random slice), <b>logging propensities</b> so you can reweight, and "
        "<b>holdout populations</b> the model never touches — all of which cost "
        "something and must be designed in from the start.",
    )

    anim_header("A model degrading, and PSI catching it before the labels do")

    weeks = np.arange(0, 40)
    rng = np.random.default_rng(6)
    drift_start = 14
    shift = np.clip((weeks - drift_start)/14.0, 0, 1.6)
    psi = 0.02 + 0.30*shift + rng.normal(0, .012, len(weeks))
    acc = 0.912 - 0.16*shift**1.3 + rng.normal(0, .006, len(weeks))
    label_lag = 6

    frames = []
    for k in range(2, len(weeks)+1):
        obs_acc = np.full(k, np.nan)
        n_known = max(0, k - label_lag)
        obs_acc[:n_known] = acc[:n_known]
        alert = psi[k-1] > 0.25
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=weeks[:k], y=psi[:k], mode="lines+markers",
                       line=dict(color=C["danger"], width=3)),
            go.Scatter(x=weeks[:k], y=obs_acc, mode="lines+markers", yaxis="y2",
                       line=dict(color=C["primary"], width=3),
                       connectgaps=False),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"week {k-1}   ·   PSI {psi[k-1]:.3f}"
            f" ({'STABLE' if psi[k-1] < .1 else 'investigate' if psi[k-1] < .25 else 'SIGNIFICANT SHIFT'})"
            f"   ·   labelled accuracy known only up to week "
            f"{max(0, k-label_lag-1)} ({label_lag}-week lag)",
            color=C["danger"] if alert else C["success"])])))

    f = go.Figure(data=[
        go.Scatter(x=weeks[:2], y=psi[:2], mode="lines+markers",
                   name="PSI (available immediately)",
                   line=dict(color=C["danger"], width=3)),
        go.Scatter(x=weeks[:2], y=[np.nan, np.nan], mode="lines+markers",
                   name=f"accuracy ({label_lag}-week label lag)", yaxis="y2",
                   line=dict(color=C["primary"], width=3)),
    ])
    f.add_hline(y=0.25, line_dash="dash", line_color=C["danger"],
                annotation_text="PSI = 0.25 (significant)")
    f.add_hline(y=0.10, line_dash="dot", line_color=C["warning"],
                annotation_text="PSI = 0.10 (investigate)")
    f.update_layout(height=440, xaxis_title="week since deployment",
                    yaxis=dict(title="PSI", range=[0, .6]),
                    yaxis2=dict(title="accuracy", overlaying="y", side="right",
                                range=[.7, .95]),
                    title="Input drift is visible weeks before the labels arrive",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    anim.animate(f, frames, duration=nav.anim_ms(180), slider_prefix="week ")
    figure(f, "PSI crosses 0.25 around week 22; the accuracy drop is not "
              "measurable until week 28. That six-week gap is why input "
              "monitoring exists.")

    code_lab(
        "PSI, drift detection, sliced metrics and a retraining trigger",
        '''import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

df = _ds.housing()
num = [c for c in df.columns
       if df[c].dtype.kind in "if" and c != "median_house_value"]
X = df[num].fillna(df[num].median()).to_numpy().astype("float32")
y = (df["median_house_value"].to_numpy() > 200000).astype("int32")
n_tr = int(.6*len(X))
print(f"=== {len(X)} rows, {X.shape[1]} features, "
      f"{y.mean():.1%} positive ===")

# ============ 1. PSI FROM SCRATCH ======================================
def psi(reference, current, bins=10, eps=1e-6):
    """Population Stability Index = KL(p||q) + KL(q||p), the Jeffreys
    divergence. Bin edges come from the REFERENCE, always."""
    edges = np.quantile(reference, np.linspace(0, 1, bins+1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0].astype(float)
    q = np.histogram(current, edges)[0].astype(float)
    p = np.maximum(p/p.sum(), eps); q = np.maximum(q/q.sum(), eps)
    return float(np.sum((p - q)*np.log(p/q)))

print()
print("=== PSI is a symmetrised KL divergence ===")
rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 20000)
print(f"{'current distribution':<34}{'PSI':>10}{'verdict':>18}")
for nm, cur in [("identical", rng.normal(0, 1, 20000)),
                ("mean shifted by 0.1", rng.normal(0.1, 1, 20000)),
                ("mean shifted by 0.3", rng.normal(0.3, 1, 20000)),
                ("mean shifted by 0.6", rng.normal(0.6, 1, 20000)),
                ("variance doubled", rng.normal(0, 1.41, 20000)),
                ("a different shape (t3)", rng.standard_t(3, 20000))]:
    v = psi(ref, cur)
    verdict = ("stable" if v < .1 else "investigate" if v < .25
               else "SIGNIFICANT")
    print(f"{nm:<34}{v:>10.4f}{verdict:>18}")
print("  thresholds 0.1 / 0.25 are banking folklore, not statistics --")
print("  but they are well calibrated in practice and universally used.")

# --- verify the identity ---------------------------------------------
a = rng.normal(0, 1, 40000); b = rng.normal(0.4, 1.2, 40000)
edges = np.quantile(a, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
p = np.histogram(a, edges)[0]/40000; q = np.histogram(b, edges)[0]/40000
p, q = np.maximum(p, 1e-6), np.maximum(q, 1e-6)
kl_pq = float((p*np.log(p/q)).sum()); kl_qp = float((q*np.log(q/p)).sum())
print()
print(f"  KL(p||q) = {kl_pq:.6f}")
print(f"  KL(q||p) = {kl_qp:.6f}")
print(f"  sum      = {kl_pq+kl_qp:.6f}")
print(f"  PSI      = {psi(a, b):.6f}   identical: "
      f"{np.isclose(kl_pq+kl_qp, psi(a, b))}")
print(f"  PSI(a,b) == PSI(b,a): {np.isclose(psi(a, b), psi(b, a), atol=1e-3)}")
print("  the SYMMETRY is the point -- there is no principled reason to call")
print("  one of two production windows the 'reference'.")

# ============ 2. A MODEL, AND A WORLD THAT MOVES =======================
norm = keras.layers.Normalization(); norm.adapt(X[:n_tr])
tf.random.set_seed(0)
model = keras.Sequential([keras.layers.Input(shape=(X.shape[1],)), norm,
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")])
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(3e-3),
              metrics=["accuracy"])
model.fit(X[:n_tr], y[:n_tr], epochs=12, batch_size=256, verbose=0)
base = model.evaluate(X[n_tr:], y[n_tr:], verbose=0,
                      return_dict=True)["accuracy"]
print()
print(f"=== deployed model: baseline accuracy {base:.4f} ===")

# ============ 3. THE THREE DRIFTS ======================================
print()
print("=== simulating each kind of drift ===")
Xp, yp = X[n_tr:].copy(), y[n_tr:].copy()
r = np.random.default_rng(1)

def covariate_shift(Xa, ya, mag):
    Xb = Xa.copy()
    Xb[:, 0] = Xb[:, 0] + mag*Xa[:, 0].std()      # shift ONE feature
    return Xb, ya

def label_shift(Xa, ya, mag):
    keep = np.where(ya == 1)[0]
    drop = r.choice(keep, int(mag*len(keep)*0.5), replace=False)
    mask = np.ones(len(ya), bool); mask[drop] = False
    return Xa[mask], ya[mask]

def concept_drift(Xa, ya, mag):
    flip = r.random(len(ya)) < mag*0.25            # P(Y|X) itself changes
    yb = ya.copy(); yb[flip] = 1 - yb[flip]
    return Xa, yb

print(f"{'drift':<20}{'magnitude':>11}{'mean PSI':>11}{'max PSI':>10}"
      f"{'accuracy':>11}{'caught by PSI?':>17}")
for nm, fn in [("covariate", covariate_shift), ("label", label_shift),
               ("CONCEPT", concept_drift)]:
    for mag in [0.0, 0.5, 1.0]:
        Xd, yd = fn(Xp, yp, mag)
        psis = [psi(X[:n_tr][:, j], Xd[:, j]) for j in range(X.shape[1])]
        acc = model.evaluate(Xd, yd, verbose=0, return_dict=True)["accuracy"]
        caught = "YES" if max(psis) > 0.25 else "no"
        print(f"{nm:<20}{mag:>11.1f}{np.mean(psis):>11.4f}{max(psis):>10.4f}"
              f"{acc:>11.4f}{caught:>17}")
print()
print("  COVARIATE shift is caught by PSI immediately.")
print("  CONCEPT drift destroys accuracy while every input PSI stays at ~0.")
print("  the inputs look IDENTICAL. Only labels reveal it. That is why it")
print("  is the dangerous one.")

# ============ 4. SLICED METRICS ========================================
print()
print("=== aggregate accuracy hides subgroup failure ===")
# a synthetic subgroup: the smallest 8 % by the first feature
grp = X[n_tr:][:, 0] < np.quantile(X[n_tr:][:, 0], 0.08)
Xs, ys = X[n_tr:].copy(), y[n_tr:].copy()
ys[grp] = 1 - ys[grp]                             # the model is wrong for them
overall = model.evaluate(Xs, ys, verbose=0, return_dict=True)["accuracy"]
in_grp = model.evaluate(Xs[grp], ys[grp], verbose=0,
                        return_dict=True)["accuracy"]
out_grp = model.evaluate(Xs[~grp], ys[~grp], verbose=0,
                         return_dict=True)["accuracy"]
print(f"  subgroup size                : {grp.mean():.1%} of traffic")
print(f"  OVERALL accuracy             : {overall:.4f}")
print(f"  accuracy OUTSIDE the subgroup: {out_grp:.4f}")
print(f"  accuracy INSIDE the subgroup : {in_grp:.4f}   <-- catastrophic")
print(f"  the aggregate moved by only {abs(out_grp-overall):.4f}.")
print("  ALERT ON THE WORST SLICE, not on the mean.")

# ============ 5. A DRIFT MONITOR =======================================
print()
print("="*66)
print("A production monitor: 26 weeks, with a 6-week label lag")
print("="*66)
WEEKS, LAG = 26, 6
reference = X[:n_tr]
log = []
for w in range(WEEKS):
    mag = max(0.0, (w - 9)/10.0)
    idx = r.choice(len(Xp), 900, replace=False)
    Xw, yw = Xp[idx].copy(), yp[idx].copy()
    Xw[:, 0] += mag*Xp[:, 0].std()                     # covariate shift
    flip = r.random(len(yw)) < 0.05*mag                # some concept drift too
    yw[flip] = 1 - yw[flip]
    feat_psi = [psi(reference[:, j], Xw[:, j]) for j in range(X.shape[1])]
    pred = model.predict(Xw, verbose=0).ravel()
    pred_psi = psi(model.predict(X[:n_tr][:2000], verbose=0).ravel(), pred)
    acc = float(((pred > .5).astype(int) == yw).mean())
    log.append(dict(week=w, max_feature_psi=max(feat_psi),
                    prediction_psi=pred_psi, mean_confidence=float(
                        np.abs(pred-0.5).mean()*2), accuracy=acc))

L = pd.DataFrame(log)
print(f"{'week':>6}{'max feature PSI':>18}{'prediction PSI':>17}"
      f"{'confidence':>13}{'accuracy (lagged)':>20}{'action':>22}")
for _, row in L.iterrows():
    w = int(row.week)
    lagged = f"{L.accuracy[w-LAG]:.4f}" if w >= LAG else "not yet known"
    if row.max_feature_psi > 0.25:
        act = "RETRAIN"
    elif row.max_feature_psi > 0.10:
        act = "investigate"
    else:
        act = "-"
    if w % 2 == 0:
        print(f"{w:>6}{row.max_feature_psi:>18.4f}{row.prediction_psi:>17.4f}"
              f"{row.mean_confidence:>13.4f}{lagged:>20}{act:>22}")

first_alert = int(L[L.max_feature_psi > 0.25].week.min())
first_acc_drop = int(L[L.accuracy < base - 0.03].week.min())
print()
print(f"  PSI first exceeded 0.25 in week {first_alert}")
print(f"  accuracy first dropped 3 points in week {first_acc_drop}")
print(f"  ... but that accuracy is only OBSERVABLE in week "
      f"{first_acc_drop + LAG} because of the label lag")
print(f"  input monitoring bought "
      f"{first_acc_drop + LAG - first_alert} weeks of warning.")

# ============ 6. RETRAINING ============================================
print()
print("=== does retraining actually fix it? ===")
w = WEEKS-1
mag = (w-9)/10.0
idx = r.choice(len(Xp), 4000, replace=False)
Xnow, ynow = Xp[idx].copy(), yp[idx].copy()
Xnow[:, 0] += mag*Xp[:, 0].std()
flip = r.random(len(ynow)) < 0.05*mag
ynow[flip] = 1 - ynow[flip]
half = len(Xnow)//2

print(f"  stale model on current data: "
      f"{model.evaluate(Xnow[half:], ynow[half:], verbose=0, return_dict=True)['accuracy']:.4f}")
for nm, Xf, yf in [("retrained on recent data only", Xnow[:half], ynow[:half]),
                   ("retrained on ALL data",
                    np.vstack([X[:n_tr], Xnow[:half]]),
                    np.concatenate([y[:n_tr], ynow[:half]]))]:
    n2 = keras.layers.Normalization(); n2.adapt(Xf)
    tf.random.set_seed(0)
    m2 = keras.Sequential([keras.layers.Input(shape=(X.shape[1],)), n2,
                           keras.layers.Dense(32, activation="relu"),
                           keras.layers.Dense(1, activation="sigmoid")])
    m2.compile(loss="binary_crossentropy",
               optimizer=keras.optimizers.Adam(3e-3), metrics=["accuracy"])
    m2.fit(Xf, yf, epochs=12, batch_size=256, verbose=0)
    print(f"  {nm:<32}"
          f"{m2.evaluate(Xnow[half:], ynow[half:], verbose=0, return_dict=True)['accuracy']:.4f}")
print("  under CONCEPT drift, old data is actively WRONG -- it teaches the")
print("  relationship that no longer holds. Weight recent data more, or")
print("  use a sliding window. Under COVARIATE shift, keeping the old data")
print("  usually helps.")

import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter(x=L.week, y=L.max_feature_psi, mode="lines+markers",
                name="max feature PSI", line=dict(color=C["danger"], width=3))
fig.add_scatter(x=L.week + LAG, y=L.accuracy, mode="lines+markers",
                name=f"accuracy (observed {LAG} weeks late)", yaxis="y2",
                line=dict(color=C["primary"], width=3))
fig.add_hline(y=0.25, line_dash="dash", line_color=C["danger"])
fig.update_layout(height=430, xaxis_title="week",
                  yaxis=dict(title="PSI"),
                  yaxis2=dict(title="accuracy", overlaying="y", side="right"),
                  title="Drift monitoring with a label lag")
''',
        key="ch19_drift",
    )

    keypoints([
        "<b>Covariate shift</b> changes $P(X)$ and is detectable without labels; "
        "<b>concept drift</b> changes $P(Y|X)$ and is not.",
        "<b>PSI = $D_{\\mathrm{KL}}(p\\Vert q) + D_{\\mathrm{KL}}(q\\Vert p)$</b> "
        "— symmetric; > 0.25 is a significant shift.",
        "<b>Slice every metric</b> — aggregate accuracy hides catastrophic "
        "subgroup failure.",
        "Plan the <b>label-collection path before deploying</b>; ground truth "
        "arrives late or never.",
        "Under concept drift, old data is <b>actively wrong</b> — weight recent "
        "data or use a sliding window.",
    ])


# ==========================================================================
def s_19_9():
    section("19.9", "The Deployment Checklist and Exercises")

    lead(
        "Everything in this chapter, as the list you actually run through before "
        "a model goes live."
    )

    sub("Before deployment")

    table(
        ["#", "Check", "Why"],
        [["1", "Preprocessing is <b>inside</b> the SavedModel",
          "Training/serving skew fails silently (§19.1)"],
         ["2", "The signature accepts what the client actually has",
          "No scaler, no tokenizer, no numpy on the client"],
         ["3", "A <b>reproducible</b> build: pinned versions, a data snapshot, a "
          "seed",
          "You will need to rebuild this model in six months"],
         ["4", "Latency measured at <b>p99</b>, at realistic batch sizes",
          "Users experience the tail (§19.2)"],
         ["5", "Load-tested <b>above</b> expected peak",
          "Queue length is $\\rho/(1-\\rho)$"],
         ["6", "Quantised if it goes to a device (§19.3)",
          "4× smaller, usually < 1 % accuracy"],
         ["7", "Metrics sliced by subgroup",
          "The aggregate hides the failure (§19.8)"],
         ["8", "A <b>rollback</b> that has actually been tested",
          "An untested rollback is not a rollback"],
         ["9", "Drift monitoring live <b>before</b> traffic arrives",
          "You cannot detect a shift with no baseline"],
         ["10", "The label-collection path exists",
          "Otherwise concept drift is invisible forever"]],
    )

    sub("A sensible order of operations")

    table(
        ["Stage", "Traffic", "Duration", "Abort if"],
        [["<b>Shadow</b>", "0 % served, 100 % scored", "1–7 days",
          "Predictions differ wildly from the current model"],
         ["<b>Canary</b>", "1 %", "Hours", "Error rate or latency regresses"],
         ["<b>Ramp</b>", "5 → 25 → 50 %", "Hours to days",
          "Any monitored metric regresses"],
         ["<b>Full</b>", "100 %", "—", "Keep the old version loaded for a week"]],
    )

    idea(
        "The most valuable artefact is not the model — it is the pipeline that "
        "rebuilds it",
        "Models are disposable; you will retrain many times. What must be "
        "durable is the <b>path from raw data to a deployed model</b>: versioned "
        "data, a reproducible feature pipeline, a training job that runs from a "
        "commit hash, an evaluation suite including sliced metrics, and an "
        "automated deployment with a tested rollback. Teams that build the model "
        "first and the pipeline later spend most of their time on manual "
        "retraining. It is the same lesson as Chapter 2's: <b>the pipeline is "
        "the product</b>.",
    )

    anim_header("A deployment, stage by stage")

    stages_dep = [
        ("build", "train from a commit hash, on a snapshot of the data",
         C["muted"], 0),
        ("evaluate", "held-out metrics, sliced by subgroup", C["primary"], 0),
        ("export", "SavedModel with preprocessing inside", C["accent"], 0),
        ("shadow", "100 % scored, 0 % served — compare against production",
         C["warning"], 0),
        ("canary", "1 % of real traffic, watching p99 and error rate",
         C["warning"], 1),
        ("ramp", "5 % → 25 % → 50 %, aborting on any regression",
         C["success"], 25),
        ("full", "100 %, old version stays loaded for a week", C["success"], 100),
        ("monitor", "PSI, sliced metrics, and the label lag", C["primary"], 100),
    ]
    frames = []
    for i, (nm, desc, col, traffic) in enumerate(stages_dep):
        shapes, ann = [], []
        for j, (n2, _, c2, _) in enumerate(stages_dep):
            done = j < i
            cur = j == i
            shapes.append(go.Scatter(
                x=[j*1.3, j*1.3+1.05, j*1.3+1.05, j*1.3, j*1.3],
                y=[0, 0, .8, .8, 0], fill="toself",
                fillcolor=(alpha(c2, .9) if cur else
                           alpha(C["success"], .35) if done else
                           alpha(C["line"], .25)),
                line=dict(color="#fff", width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=j*1.3+.52, y=.4, text=n2, showarrow=False,
                            font=dict(size=9,
                                      color="#fff" if (cur or done)
                                      else C["ink_soft"])))
        shapes.append(go.Scatter(
            x=[0, 0 + traffic/100*len(stages_dep)*1.3], y=[-.7, -.7],
            mode="lines", line=dict(color=C["danger"], width=12),
            showlegend=False, hoverinfo="skip"))
        ann.append(dict(x=len(stages_dep)*1.3/2, y=-1.15,
                        text=f"{traffic}% of production traffic",
                        showarrow=False,
                        font=dict(size=11, color=C["ink_soft"])))
        frames.append(go.Frame(name=nm, data=shapes,
                               layout=go.Layout(annotations=ann + [
                                   anim.annotate_step(
                                       f"{i+1}/{len(stages_dep)}  "
                                       f"{nm.upper()}  ·  {desc}", color=col)])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=340, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.5, len(stages_dep)*1.3]),
                    yaxis=dict(visible=False, range=[-1.6, 1.3]),
                    annotations=list(frames[0].layout.annotations),
                    title="From a commit to 100 % of traffic")
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="stage ")
    figure(f)

    code_lab(
        "A complete deployment: export, validate, benchmark, canary, rollback",
        '''import numpy as np, os, shutil, time, json, hashlib
import tensorflow as tf
from tensorflow import keras
from core import datasets as _ds

tf.random.set_seed(42); np.random.seed(42)

BASE = os.path.join(os.environ.get("TEMP", "/tmp"), "mlplat_deploy")
shutil.rmtree(BASE, ignore_errors=True)
os.makedirs(BASE, exist_ok=True)

Xtr, ytr, Xte, yte, labels, real = _ds.fashion_mnist(n_train=10000, n_test=2000)
Xtr = Xtr.astype("float32")[..., None]; Xte = Xte.astype("float32")[..., None]

# ============ 1. BUILD, REPRODUCIBLY ===================================
def build_and_train(seed, epochs=6):
    tf.random.set_seed(seed); np.random.seed(seed)
    m = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1), name="image"),
        keras.layers.Rescaling(1.0),               # preprocessing INSIDE
        keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax", name="probs")])
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=epochs, batch_size=128, verbose=0)
    return m

print("=== 1. BUILD (reproducibly) ===")
v1 = build_and_train(seed=0)
acc1 = v1.evaluate(Xte, yte, verbose=0, return_dict=True)["accuracy"]
print(f"  v1 accuracy {acc1:.4f}")
again = build_and_train(seed=0)
same = np.allclose(v1.predict(Xte[:100], verbose=0),
                   again.predict(Xte[:100], verbose=0), atol=1e-5)
print(f"  same seed reproduces identical predictions: {same}")
print("  if this is False, you cannot debug a production incident.")

# ============ 2. EVALUATE, SLICED ======================================
print()
print("=== 2. EVALUATE, sliced by class ===")
pred = v1.predict(Xte, verbose=0).argmax(1)
print(f"  overall accuracy {float((pred == yte).mean()):.4f}")
print(f"{'class':<16}{'n':>7}{'accuracy':>11}")
worst = (1.0, None)
for c in range(10):
    mask = yte == c
    a = float((pred[mask] == c).mean())
    if a < worst[0]:
        worst = (a, labels[c] if c < len(labels) else str(c))
    print(f"{(labels[c] if c < len(labels) else str(c)):<16}"
          f"{int(mask.sum()):>7}{a:>11.4f}")
print(f"  WORST SLICE: {worst[1]} at {worst[0]:.4f}")
print("  gate the release on the worst slice, not the mean.")

# ============ 3. EXPORT ================================================
print()
print("=== 3. EXPORT ===")
p1 = os.path.join(BASE, "0001")
v1.export(p1)
size = sum(os.path.getsize(os.path.join(r, f))
           for r, _, fs in os.walk(p1) for f in fs)
print(f"  {p1.split(os.sep)[-1]}: {size/1024:.0f} KB")

meta = {"version": 1, "accuracy": round(float(acc1), 5),
        "worst_slice": {"class": worst[1], "accuracy": round(worst[0], 5)},
        "train_rows": int(len(Xtr)), "seed": 0,
        "tf_version": tf.__version__,
        "data_fingerprint": hashlib.sha256(Xtr[:100].tobytes()).hexdigest()[:16]}
with open(os.path.join(p1, "model_card.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"  model card: {json.dumps(meta)[:110]}...")
print("  the data fingerprint is what lets you prove, later, WHICH data")
print("  this model was trained on.")

# ============ 4. VALIDATE THE EXPORT ===================================
print()
print("=== 4. VALIDATE (the exported graph, not the Python object) ===")
loaded = tf.saved_model.load(p1)
sig = loaded.signatures["serving_default"]
key = list(sig.structured_outputs.keys())[0]
a = v1.predict(Xte[:200], verbose=0)
b = np.asarray(sig(tf.constant(Xte[:200]))[key])
print(f"  max |keras - SavedModel| = {np.abs(a-b).max():.2e}")
print(f"  argmax agreement          = {float((a.argmax(1)==b.argmax(1)).mean()):.4f}")
assert np.abs(a-b).max() < 1e-4, "exported graph disagrees with the model!"
print("  ALWAYS assert this. An export can silently differ (a dropout layer")
print("  left in training mode, a preprocessing layer not adapted).")

# ============ 5. BENCHMARK AT p99 ======================================
print()
print("=== 5. BENCHMARK ===")
def bench(fn, B, n=60):
    x = tf.constant(Xte[:B])
    fn(x)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(x); ts.append(time.perf_counter()-t0)
    return np.array(ts)*1000

print(f"{'batch':>7}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}"
      f"{'items/s':>12}")
for B in [1, 8, 32, 128]:
    t = bench(lambda x: sig(x), B)
    print(f"{B:>7}{np.percentile(t,50):>10.3f}{np.percentile(t,95):>10.3f}"
          f"{np.percentile(t,99):>10.3f}{B/(np.percentile(t,50)/1000):>12,.0f}")
print("  report p99. If a page makes 10 calls, 10% of page loads hit it.")

# ============ 6. A CANDIDATE, AND A SHADOW COMPARISON ==================
print()
print("=== 6. SHADOW: score everything, serve nothing ===")
v2 = build_and_train(seed=1, epochs=9)
acc2 = v2.evaluate(Xte, yte, verbose=0, return_dict=True)["accuracy"]
p2 = os.path.join(BASE, "0002"); v2.export(p2)
print(f"  v2 accuracy {acc2:.4f} (v1 was {acc1:.4f})")

pred1 = v1.predict(Xte, verbose=0).argmax(1)
pred2 = v2.predict(Xte, verbose=0).argmax(1)
agree = float((pred1 == pred2).mean())
both_wrong = float(((pred1 != yte) & (pred2 != yte)).mean())
v2_fixes = float(((pred1 != yte) & (pred2 == yte)).mean())
v2_breaks = float(((pred1 == yte) & (pred2 != yte)).mean())
print(f"  the two models agree on {agree:.1%} of inputs")
print(f"  v2 FIXES  {v2_fixes:.2%} of cases v1 got wrong")
print(f"  v2 BREAKS {v2_breaks:.2%} of cases v1 got right")
print(f"  net change {v2_fixes - v2_breaks:+.2%}")
print("  a shadow run tells you WHICH cases change, not just the average.")
print("  a model with better accuracy that breaks a critical segment is")
print("  not an improvement.")

# --- and sliced ------------------------------------------------------
print()
print(f"  {'class':<16}{'v1':>8}{'v2':>8}{'delta':>9}")
regress = []
for c in range(10):
    m_ = yte == c
    a1 = float((pred1[m_] == c).mean()); a2 = float((pred2[m_] == c).mean())
    if a2 < a1 - 0.02:
        regress.append(labels[c] if c < len(labels) else str(c))
    print(f"  {(labels[c] if c < len(labels) else str(c)):<16}"
          f"{a1:>8.3f}{a2:>8.3f}{a2-a1:>+9.3f}")
print(f"  slices that regressed by >2 points: {regress or 'none'}")

# ============ 7. CANARY, WITH AN ABORT RULE ============================
print()
print("=== 7. CANARY ===")
rng = np.random.default_rng(0)
def serve(frac_v2, n=2000):
    idx = rng.choice(len(Xte), n, replace=False)
    to_v2 = rng.random(n) < frac_v2
    out = np.where(to_v2, pred2[idx], pred1[idx])
    err = float((out != yte[idx]).mean())
    err_v2 = float((pred2[idx][to_v2] != yte[idx][to_v2]).mean()) \\
        if to_v2.sum() else np.nan
    return err, err_v2, int(to_v2.sum())

baseline_err = float((pred1 != yte).mean())
ABORT = baseline_err*1.10
print(f"  baseline error {baseline_err:.4f}, abort if v2 error > "
      f"{ABORT:.4f} (+10%)")
print(f"{'stage':>8}{'v2 traffic':>13}{'requests to v2':>17}"
      f"{'blended error':>16}{'v2 error':>11}{'decision':>12}")
for frac in [0.01, 0.05, 0.25, 0.50, 1.00]:
    e, e2, n2 = serve(frac)
    dec = "ABORT" if (n2 > 30 and e2 > ABORT) else "proceed"
    print(f"{f'{frac:.0%}':>8}{frac:>13.2%}{n2:>17}{e:>16.4f}"
          f"{e2:>11.4f}{dec:>12}")

# ============ 8. ROLLBACK, TESTED ======================================
print()
print("=== 8. ROLLBACK (test it, or you do not have one) ===")
print(f"  versions present: {sorted(os.listdir(BASE))}")
print(f"  TF Serving would be serving {max(os.listdir(BASE))}")
t0 = time.perf_counter()
shutil.rmtree(os.path.join(BASE, "0002"))
rolled = tf.saved_model.load(os.path.join(BASE, "0001"))
dt = time.perf_counter()-t0
print(f"  rolled back by removing the directory: {dt*1000:.0f} ms")
print(f"  now serving: {sorted(os.listdir(BASE))}")
r_out = rolled.signatures["serving_default"](tf.constant(Xte[:200]))
r_key = list(r_out.keys())[0]
print(f"  rolled-back model matches v1: "
      f"{np.allclose(np.asarray(r_out[r_key]), a, atol=1e-4)}")
print("  an UNTESTED rollback is not a rollback. Run this in a drill.")

# ============ 9. THE CHECKLIST =========================================
print()
print("="*66)
checks = [
    ("preprocessing inside the SavedModel", True),
    ("exported graph matches the Python model", np.abs(a-b).max() < 1e-4),
    ("reproducible from a seed", same),
    ("p99 latency measured", True),
    ("metrics sliced by subgroup", True),
    ("no slice regressed by >2 points", len(regress) == 0),
    ("model card with a data fingerprint", True),
    ("rollback tested", True),
]
for nm, ok in checks:
    print(f"  [{'x' if ok else ' '}] {nm}")
print(f"  {sum(o for _, o in checks)}/{len(checks)} checks pass")
print("="*66)

import plotly.graph_objects as go
fig = go.Figure()
cls = [labels[c] if c < len(labels) else str(c) for c in range(10)]
a1s = [float((pred1[yte == c] == c).mean()) for c in range(10)]
a2s = [float((pred2[yte == c] == c).mean()) for c in range(10)]
fig.add_bar(x=cls, y=a1s, name="v1", marker=dict(color=C["muted"]))
fig.add_bar(x=cls, y=a2s, name="v2 (candidate)",
            marker=dict(color=C["primary"]))
fig.update_layout(height=420, barmode="group", yaxis_title="accuracy",
                  yaxis=dict(range=[0, 1.02]),
                  title="Shadow comparison, sliced by class")
''',
        key="ch19_deploy",
    )

    rule()

    sub("Exercises")

    exercise(
        1, "What does a SavedModel contain? How do you inspect its content?",
        "A SavedModel contains a TensorFlow model, including its architecture "
        "(a computation graph) and its weights. It is stored as a directory "
        "containing a **saved_model.pb** file, which defines the computation "
        "graph (represented as a serialised protocol buffer), and a "
        "**variables** subdirectory containing the variable values. For models "
        "containing a large number of weights, these variable values may be "
        "split across multiple files. A SavedModel also includes an **assets** "
        "subdirectory that may contain additional data, such as vocabulary files, "
        "class names, or some example instances for this model.\n\n"
        "To be more accurate, a SavedModel can contain one or more "
        "*metagraphs*. A metagraph is a computation graph plus some function "
        "signature definitions (including their input and output names, types, "
        "and shapes). Each metagraph is identified by a set of tags.\n\n"
        "**To inspect** a SavedModel, use the command-line tool "
        "`saved_model_cli` (or just load it with `tf.saved_model.load()` and "
        "inspect it in Python):\n\n"
        "```bash\n"
        "saved_model_cli show --dir my_model --all\n"
        "```")

    exercise(
        2, "When should you use TF Serving? What are its main features? What are "
        "some tools you can use to deploy it?",
        "TF Serving allows you to deploy multiple TensorFlow models (or multiple "
        "versions of the same model) and make them accessible to all your "
        "applications easily via a REST API or a gRPC API. Using your models "
        "directly in your applications would make it harder to deploy a new "
        "version of a model across all applications. Implementing your own "
        "microservice to wrap a TF model would require extra work, and it would "
        "be hard to match TF Serving's features.\n\n"
        "**Its main features:**\n\n"
        "* It can **monitor a directory** and automatically deploy the models "
        "that are placed there, without needing any code change or restart — the "
        "numbered-subdirectory convention of §19.1.\n"
        "* It is **battle-tested** and highly scalable.\n"
        "* It supports **A/B testing** of experimental models and deploying a new "
        "model version to just a subset of your users.\n"
        "* It **automatically batches** individual requests (§19.2), which "
        "significantly improves throughput.\n"
        "* It is written in C++, so there is **no Python interpreter** in "
        "production — no version skew.\n\n"
        "**To deploy it:** you can install TF Serving from source, but it is much "
        "simpler to install it using a **Docker image**. To deploy a cluster of "
        "TF Serving Docker images, you can use an orchestration tool such as "
        "**Kubernetes**, or use a fully hosted solution such as Google Vertex AI, "
        "Amazon SageMaker, or Azure ML.")

    exercise(
        3, "How do you deploy a model across multiple TF Serving instances?",
        "To deploy a model across multiple TF Serving instances, all you need to "
        "do is configure these TF Serving instances to monitor the **same "
        "`models` directory**, and then export your new model as a SavedModel "
        "into a subdirectory.\n\n"
        "The mechanism is deliberately simple: each instance watches the "
        "directory, notices the new highest-numbered subdirectory, loads it, and "
        "switches over. There is no coordination protocol to get wrong.\n\n"
        "In practice the directory is shared storage (GCS, S3, NFS), and the "
        "instances sit behind a load balancer. Note that they will **not** switch "
        "over simultaneously — for a brief window some instances serve v1 and "
        "some serve v2. If that matters (for example, if a client caches "
        "predictions and would notice the inconsistency), you need an explicit "
        "blue/green switch at the load balancer instead.")

    exercise(
        4, "When should you use the gRPC API rather than the REST API to query a "
        "model served by TF Serving?",
        "The **gRPC API** is more efficient than the REST API. However, its "
        "client libraries are not as widely available, and if you activate "
        "compression when using the REST API, you can get almost the same "
        "performance. So, the gRPC API is most useful when you need the "
        "**highest possible performance** and the clients are not web clients.\n\n"
        "The concrete difference (§19.2): gRPC uses protocol buffers, whose wire "
        "format matches the in-memory layout, so serialisation is nearly free. "
        "JSON must be parsed into objects — 1 000 floats are ~12 KB of text "
        "versus ~4 KB of binary, and the parsing cost is much larger than the "
        "size difference suggests.\n\n"
        "Whichever you use: for images, send **base64-encoded JPEG bytes** and "
        "decode inside the graph, never a JSON array of pixels. That is a 60× "
        "payload reduction and dwarfs the REST-versus-gRPC difference.")

    exercise(
        5, "What are the different ways TFLite reduces a model's size to make it "
        "run on a mobile or embedded device?",
        "To reduce a model's size so it can run on a mobile or embedded device, "
        "TFLite uses several techniques:\n\n"
        "* It has a **converter** which can optimise a SavedModel: it shrinks the "
        "model and reduces its latency. To do this, it **prunes all the "
        "operations that are not needed** to make predictions (such as training "
        "operations), and it **optimises and fuses** computations whenever "
        "possible — for example, three adjacent layers `Conv2D`, "
        "`BatchNormalization`, `ReLU` become a single fused kernel.\n"
        "* The converter can also perform **post-training quantisation**: this "
        "technique dramatically reduces the model's size, so it is much faster to "
        "download and store. Quantising a model's weights down to fixed-point, "
        "8-bit integers gives roughly a **4×** size reduction (§19.3), because "
        "each float32 weight becomes one byte.\n"
        "* It saves the optimised model using the **FlatBuffer** format, which "
        "can be loaded to RAM directly, without parsing. This reduces the loading "
        "time and memory footprint.")

    exercise(
        6, "What is quantisation-aware training, and why would you need it?",
        "**Quantisation-aware training** consists of adding fake quantisation "
        "operations to the model during training. This allows the model to learn "
        "to ignore the quantisation noise; the final weights will be more robust "
        "to quantisation.\n\n"
        "Why you need it: post-training quantisation maps float32 weights onto "
        "256 integer levels using $r = S(q - Z)$ (§19.3). The rounding error is "
        "bounded by $S/2$, and for most models that is harmless — accuracy drops "
        "well under 1 %. But for **small models, models with unusual weight "
        "distributions, or heavily quantised ones (4-bit)**, the accumulated "
        "error can be substantial.\n\n"
        "QAT solves this by simulating the quantisation *in the forward pass* "
        "during training, while keeping float gradients (via a straight-through "
        "estimator, since round() has zero derivative almost everywhere). The "
        "network then learns weights that survive being rounded — typically "
        "recovering most of the gap between post-training int8 and full float32.")

    exercise(
        7, "What are model parallelism and data parallelism? Why is the latter "
        "generally recommended?",
        "**Model parallelism** means chopping your model into multiple parts and "
        "running them in parallel across multiple devices, hopefully speeding up "
        "the model during training or inference. **Data parallelism** means "
        "creating multiple exact replicas of your model and deploying them across "
        "multiple devices; at each iteration of training, each replica is given a "
        "different batch of data, and it computes the gradients of the loss with "
        "regard to the model parameters. Then these gradients are averaged, and "
        "the resulting gradient is used to update the model parameters on all the "
        "replicas.\n\n"
        "**Data parallelism is generally recommended** for two structural "
        "reasons (§19.5):\n\n"
        "1. It is **much simpler to implement**, and it works the same way for "
        "any model. Model parallelism requires analysing the model to determine "
        "the best way to chop it into pieces, and that analysis differs for every "
        "architecture.\n"
        "2. It **scales better**. Ring all-reduce moves roughly $2M$ bytes per "
        "device *regardless of the device count*, and the communication overlaps "
        "with the backward pass. Model parallelism pays a pipeline bubble of "
        "$(K-1)/(m+K-1)$ and the devices spend much of their time waiting for "
        "each other.\n\n"
        "Model parallelism becomes necessary only when the model does **not fit "
        "on one device** — which is exactly the situation for very large "
        "language models, where both are used together.")

    exercise(
        8, "When training a model across multiple servers, what distribution "
        "strategies can you use? How do you choose which one to use?",
        "When training a model across multiple servers, you can use the following "
        "distribution strategies:\n\n"
        "* **`MultiWorkerMirroredStrategy`** performs mirrored data parallelism. "
        "The model is replicated across all available servers and devices, and "
        "each replica gets a different batch of data at each training iteration "
        "and computes its own gradients. The mean of the gradients is computed "
        "and shared across all replicas using a distributed AllReduce "
        "implementation (NCCL by default), and all replicas perform the same "
        "parameter update. It uses **synchronous** updates.\n"
        "* **`ParameterServerStrategy`** performs asynchronous data parallelism. "
        "The model is replicated across all devices on all workers, and the "
        "parameters are sharded across all parameter servers. Each worker has its "
        "own training loop, running asynchronously; at each training iteration, "
        "it gets the latest parameter values from the parameter servers, computes "
        "the gradients of the loss, and sends them to the parameter servers, "
        "which apply them.\n\n"
        "**How to choose:** `MultiWorkerMirroredStrategy` is the default and "
        "should be your first choice — synchronous training gives better final "
        "accuracy, because there are no stale gradients (§19.5's lab shows "
        "staleness destabilising even the simplest optimisation). "
        "`ParameterServerStrategy` is worth considering when the workers are "
        "**heterogeneous or unreliable**, since it removes the straggler "
        "problem — though in practice backup replicas solve that within the "
        "synchronous approach.")

    exercise(
        9, "Train a model (any model you like) and deploy it to TF Serving or "
        "Google Vertex AI. Write the client code to query it using the REST API "
        "or the gRPC API. Update the model and deploy the new version. Your "
        "client code will now query the new version. Roll back to the first "
        "version.",
        "§19.9's lab walks the full sequence — build, evaluate sliced, export, "
        "validate the exported graph, benchmark at p99, shadow-compare, canary "
        "with an abort rule, and roll back — using local SavedModel directories "
        "so it runs offline.\n\n"
        "To run it against a real server:\n\n"
        "```bash\n"
        "docker run -p 8501:8501 --mount type=bind,\\\n"
        "  source=/path/to/my_model,target=/models/my_model \\\n"
        "  -e MODEL_NAME=my_model -t tensorflow/serving\n"
        "```\n\n"
        "Then `POST` to `http://localhost:8501/v1/models/my_model:predict` with "
        "`{\"instances\": [...]}`. To deploy a new version, write "
        "`my_model/0002/`; TF Serving picks it up automatically. To **roll back**, "
        "either delete `0002/` or pin the version explicitly with a "
        "`--model_config_file` specifying `model_version_policy`.\n\n"
        "The thing worth actually practising is the rollback. An untested "
        "rollback is not a rollback.")

    exercise(
        10, "Train any model across multiple GPUs on the same machine using the "
        "MirroredStrategy (if you do not have access to GPUs, you can use "
        "Colaboratory with a GPU runtime and create two logical GPUs). Train the "
        "model again using the CentralStorageStrategy and compare the training "
        "time.",
        "§19.6's lab does exactly this with "
        "`tf.config.set_logical_device_configuration`, which creates virtual "
        "devices so `MirroredStrategy` runs on a CPU-only machine. The code is "
        "identical to what runs on real GPUs.\n\n"
        "What to watch for:\n\n"
        "* **The batch size you pass is the global one.** Moving to 4 replicas "
        "with an unchanged `batch_size=32` gives each replica 8 examples and is "
        "often *slower*. Scale it by $K$, and scale the learning rate with it.\n"
        "* **Everything that creates a variable goes inside "
        "`strategy.scope()`** — the model, the optimiser, the metrics. `fit()` "
        "goes outside.\n"
        "* On **virtual** devices there is no real speed-up (they share the same "
        "cores). The lab demonstrates *correctness*, not throughput.\n"
        "* `CentralStorageStrategy` keeps the variables on the CPU and the "
        "computation on the GPUs. It is slower when the variables are small "
        "(every step pays a CPU↔GPU round trip) and useful when they are too "
        "large to replicate.")

    exercise(
        11, "Train a small model on Vertex AI, using TensorFlow Cloud Tuner for "
        "hyperparameter tuning.",
        "The concepts that transfer, whatever the platform (§19.7):\n\n"
        "* **Random search is the baseline any tuner must beat.** 59 random "
        "trials reach the top 5 % with 95 % probability, *independent of the "
        "number of dimensions* — which is why grid search is almost never right "
        "above two parameters.\n"
        "* **Use early stopping (Hyperband/ASHA).** Successive halving reaches "
        "the same answer for roughly a fifth of the compute, by killing bad "
        "configurations after a small budget.\n"
        "* **Tune on a subset first.** Hyperparameters found on 10 % of the data "
        "usually transfer and cost 10 % as much to find.\n"
        "* **Set a wall-clock budget and use preemptible instances.** A "
        "50-configuration search on 8×A100 for 72 hours is roughly $108 000 at "
        "list price; spot instances cut that by 60–90 %, and checkpointing makes "
        "preemption a non-event.\n"
        "* **Log everything to one place** — the configuration, the git commit, "
        "the data fingerprint, and every metric sliced by subgroup. The search is "
        "only useful if you can later explain *why* the winner won.")

    rule()

    keypoints([
        "<b>Preprocessing inside the SavedModel</b>; validate that the exported "
        "graph matches the Python model.",
        "Measure <b>p99</b> at realistic batch sizes and load-test above peak.",
        "<b>Shadow → canary → ramp → full</b>, with an abort rule defined in "
        "advance.",
        "<b>Slice every metric</b>, and gate the release on the worst slice.",
        "Monitor input drift from day one — and <b>test the rollback</b>.",
    ], title="Chapter 19 in five lines")

    refs([
        ("TensorFlow Serving — architecture and configuration",
         "https://www.tensorflow.org/tfx/guide/serving"),
        ("TensorFlow Lite — post-training quantisation",
         "https://www.tensorflow.org/lite/performance/post_training_quantization"),
        ("Jacob et al. — *Quantization and Training of Neural Networks for "
         "Efficient Integer-Arithmetic-Only Inference*",
         "https://arxiv.org/abs/1712.05877"),
        ("Hinton, Vinyals & Dean — *Distilling the Knowledge in a Neural "
         "Network*", "https://arxiv.org/abs/1503.02531"),
        ("Goyal et al. — *Accurate, Large Minibatch SGD: Training ImageNet in "
         "1 Hour*", "https://arxiv.org/abs/1706.02677"),
        ("Micikevicius et al. — *Mixed Precision Training*",
         "https://arxiv.org/abs/1710.03740"),
        ("Rajbhandari et al. — *ZeRO: Memory Optimizations Toward Training "
         "Trillion Parameter Models*", "https://arxiv.org/abs/1910.02054"),
        ("Huang et al. — *GPipe: Efficient Training of Giant Neural Networks "
         "using Pipeline Parallelism*", "https://arxiv.org/abs/1811.06965"),
        ("Bergstra & Bengio — *Random Search for Hyper-Parameter Optimization*",
         "https://www.jmlr.org/papers/v13/bergstra12a.html"),
        ("Li et al. — *Hyperband: A Novel Bandit-Based Approach to "
         "Hyperparameter Optimization*", "https://arxiv.org/abs/1603.06560"),
        ("Sculley et al. — *Hidden Technical Debt in Machine Learning Systems*",
         "https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html"),
        ("Breck et al. — *The ML Test Score: A Rubric for ML Production "
         "Readiness*",
         "https://research.google/pubs/pub46555/"),
    ])


# ==========================================================================
SECTIONS = [
    ("19.1", "SavedModel & Serving Signatures", s_19_1),
    ("19.2", "Latency, Batching & Throughput", s_19_2),
    ("19.3", "Mobile, Embedded & Quantisation", s_19_3),
    ("19.4", "GPUs & Mixed Precision", s_19_4),
    ("19.5", "Data vs Model Parallelism", s_19_5),
    ("19.6", "Distribution Strategies", s_19_6),
    ("19.7", "Pipelines, Tuning & Cost", s_19_7),
    ("19.8", "Monitoring, Drift & Retraining", s_19_8),
    ("19.9", "The Checklist & Exercises", s_19_9),
]

nav.render_chapter(CH, SECTIONS)
