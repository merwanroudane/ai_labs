"""Chapter 14 — Deep Computer Vision Using Convolutional Neural Networks."""

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
CH = "ch14"

hero(
    kicker="Part II · Chapter 14",
    title="Deep Computer Vision Using CNNs",
    blurb=(
        "A dense layer on a 224×224 image needs 150 528 weights per neuron and "
        "learns nothing about the fact that adjacent pixels are related. "
        "Convolution replaces it with a small shared kernel — and that one change "
        "buys translation equivariance, a 10 000× parameter reduction, and the "
        "entire modern history of computer vision."
    ),
    chips=["Convolution arithmetic", "9 sub-sections", "9 animations",
           "9 code labs", "LeNet → ResNet → ViT"],
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
def s_14_1():
    section("14.1", "From the Visual Cortex to Convolutional Layers")

    lead(
        "Hubel and Wiesel's experiments on cat visual cortex in 1958–59 found "
        "neurons that respond only to a small region of the visual field, and only "
        "to a specific orientation within it. Convolution is that observation "
        "turned into an architecture."
    )

    sub("The three biological facts that became design decisions")

    table(
        ["Observation", "Architectural consequence"],
        [["Neurons have a small <b>local receptive field</b>",
          "A neuron connects to a small patch, not the whole image"],
         ["Different neurons respond to different <b>orientations</b>",
          "Several <b>filters</b> per layer, each detecting a different pattern"],
         ["Higher-level neurons combine lower-level ones into larger patterns",
          "<b>Stacked</b> layers, each with a larger effective receptive field"],
         ["The same detector appears across the visual field",
          "<b>Weight sharing</b> — one kernel is slid over the whole image"]],
    )

    sub("Why not just use a dense layer?")

    derive(
        [("Take a modest colour image, $224 \\times 224 \\times 3 = 150\\,528$ "
          "inputs, and a first hidden layer of 1 000 neurons.", None),
         ("A <b>fully connected</b> layer needs:",
          r"150\,528 \times 1\,000 + 1\,000 \;=\; 150\,529\,000 \text{ parameters}"),
         ("A <b>convolutional</b> layer with 64 filters of size $3\\times3\\times3$ "
          "needs:",
          r"3 \times 3 \times 3 \times 64 + 64 \;=\; 1\,792 \text{ parameters}"),
         ("That is a factor of <b>84 000</b>, and the convolutional layer produces "
          "a far richer output ($224\\times224\\times64$ activations rather than "
          "1 000).", None),
         ("<b>But parameter count is not the main point.</b> The dense layer has "
          "no notion that pixel $(10, 10)$ is adjacent to pixel $(10, 11)$ — "
          "shuffle every pixel with a fixed permutation and it learns exactly as "
          "well. The convolutional layer <i>cannot</i> learn a shuffled image, "
          "because its structure encodes locality.", None),
         ("<b>Translation equivariance.</b> If $T_\\delta$ shifts an image by "
          "$\\delta$ pixels and $*$ is convolution, then",
          r"\bigl(T_\delta \mathbf{x}\bigr) * \mathbf{w} \;=\; "
          r"T_\delta\bigl(\mathbf{x} * \mathbf{w}\bigr)"),
         ("Shift the input, and the feature map shifts identically. A cat detector "
          "learned in the top-left corner works in the bottom-right for free. The "
          "dense layer would have to learn it separately at every position.", None)],
        title="Dense vs convolutional: 150 million parameters versus 1 792",
    )

    sub("The convolution operation")

    math(r"""
    z_{i,j,k} \;=\; b_k \;+\;
    \sum_{u=0}^{f_h - 1}\;\sum_{v=0}^{f_w - 1}\;\sum_{k'=0}^{f_{n'} - 1}
      x_{i',\,j',\,k'} \cdot w_{u,\,v,\,k',\,k}
    \qquad\text{with}\quad
    \begin{cases} i' = i \times s_h + u \\ j' = j \times s_w + v \end{cases}
    """)

    where({
        r"z_{i,j,k}": "the output at row $i$, column $j$ of feature map $k$",
        r"f_h, f_w": "the kernel height and width",
        r"f_{n'}": "the number of feature maps in the <b>previous</b> layer",
        r"s_h, s_w": "the vertical and horizontal <b>strides</b>",
        r"w_{u,v,k',k}": "the weight connecting position $(u,v)$ of map $k'$ to "
                         "map $k$ — <b>shared across all $(i,j)$</b>",
        r"b_k": "one bias per feature map, not per neuron",
    })

    note(
        "It is technically cross-correlation, not convolution",
        "True mathematical convolution flips the kernel: "
        "$\\sum_u \\sum_v x_{i-u,j-v} w_{u,v}$. Every deep learning framework "
        "omits the flip and computes <b>cross-correlation</b>. It makes no "
        "practical difference — the kernel is learned, so it simply learns the "
        "flipped version — but it is worth knowing when comparing with signal "
        "processing literature.",
    )

    sub("Output size, padding and stride")

    math(r"""
    \text{'valid' padding:}\qquad
    o \;=\; \left\lceil \frac{n - f + 1}{s} \right\rceil
    \qquad\qquad
    \text{'same' padding:}\qquad
    o \;=\; \left\lceil \frac{n}{s} \right\rceil
    """)
    where({r"n": "input size along that axis",
           r"f": "kernel size", r"s": "stride", r"o": "output size"})

    table(
        ["<code>padding=</code>", "Meaning", "Output size", "Edge pixels"],
        [["<code>'valid'</code>", "No padding; the kernel stays inside the input",
          "Shrinks by $f-1$ each layer",
          "Under-sampled — corner pixels are used once"],
         ["<code>'same'</code>", "Zero-pad so the output matches the input "
          "(at stride 1)", "$\\lceil n/s \\rceil$",
          "Equally sampled, but the padding introduces artificial zeros"]],
    )

    idea(
        "Why 'same' padding is the default choice",
        "With <code>'valid'</code> padding a $3\\times3$ kernel shrinks the image "
        "by 2 pixels per layer. After 20 layers a $32\\times32$ image is gone. "
        "More subtly, corner pixels participate in exactly one output while "
        "central pixels participate in nine — so the network systematically "
        "under-weights the edges. <code>'same'</code> fixes both, at the cost of a "
        "border of artificial zeros that the network learns to ignore.",
    )

    anim_header("Convolution: the kernel sliding across the input")
    md(
        "A $3\\times3$ edge-detection kernel over a $9\\times9$ input. The red "
        "square is the current receptive field; the growing grid on the right is "
        "the feature map. Every output position uses the **same nine weights** — "
        "that is weight sharing."
    )

    rng = np.random.default_rng(0)
    N = 9
    img = np.zeros((N, N))
    img[:, 4:] = 1.0                      # a vertical edge
    img += rng.normal(0, .04, (N, N))
    kernel = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])  # Sobel-x
    F = 3
    O = N - F + 1
    out = np.zeros((O, O))
    positions = [(i, j) for i in range(O) for j in range(O)]
    for i, j in positions:
        out[i, j] = np.sum(img[i:i + F, j:j + F] * kernel)

    frames = []
    for step, (i, j) in enumerate(positions):
        partial = np.full((O, O), np.nan)
        for k in range(step + 1):
            pi, pj = positions[k]
            partial[pi, pj] = out[pi, pj]
        frames.append(go.Frame(name=str(step + 1), data=[
            go.Heatmap(z=img[::-1], colorscale=nav.cscale(), showscale=False,
                       xgap=1, ygap=1, zmin=-.3, zmax=1.3),
            go.Scatter(x=[j - .5, j + F - .5, j + F - .5, j - .5, j - .5],
                       y=[N - 1 - i + .5, N - 1 - i + .5, N - 1 - i - F + .5,
                          N - 1 - i - F + .5, N - 1 - i + .5],
                       mode="lines", line=dict(color=C["danger"], width=4)),
            go.Heatmap(z=partial[::-1], colorscale=nav.cscale(), showscale=False,
                       xgap=1.5, ygap=1.5, zmin=out.min(), zmax=out.max()),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"position ({i},{j})   ·   output = {out[i,j]:+.3f}   ·   "
            f"{step+1}/{len(positions)} positions   ·   9 shared weights")])))

    f = make_subplots(rows=1, cols=2, column_widths=[.55, .45],
                      subplot_titles=("input 9×9, kernel window in red",
                                      "feature map 7×7"))
    f.add_trace(go.Heatmap(z=img[::-1], colorscale=nav.cscale(), showscale=False,
                           xgap=1, ygap=1, zmin=-.3, zmax=1.3), 1, 1)
    f.add_trace(go.Scatter(x=[-.5, 2.5, 2.5, -.5, -.5],
                           y=[8.5, 8.5, 5.5, 5.5, 8.5], mode="lines",
                           showlegend=False,
                           line=dict(color=C["danger"], width=4)), 1, 1)
    f.add_trace(go.Heatmap(z=np.full((O, O), np.nan), colorscale=nav.cscale(),
                           showscale=False, xgap=1.5, ygap=1.5), 1, 2)
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    f.update_layout(height=430, title="Convolution with a vertical-edge kernel")
    anim.animate(f, frames, duration=nav.anim_ms(90), slider_prefix="step ")
    figure(f, "The kernel detects the vertical edge: the feature map is bright "
              "exactly where the input transitions from dark to light.")

    code_lab(
        "Convolution from scratch, the parameter count, and equivariance",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============ 1. CONVOLUTION FROM SCRATCH ==============================
def conv2d_naive(x, kernel, stride=1, padding="valid"):
    """x: (H, W), kernel: (f, f). Returns the feature map."""
    f = kernel.shape[0]
    if padding == "same":
        pad = f // 2
        x = np.pad(x, pad, mode="constant")
    H, W = x.shape
    oh = (H - f) // stride + 1
    ow = (W - f) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = x[i*stride:i*stride+f, j*stride:j*stride+f]
            out[i, j] = np.sum(patch * kernel)          # element-wise, then sum
    return out

rng = np.random.default_rng(0)
img = np.zeros((12, 12)); img[:, 6:] = 1.0
sobel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
mine = conv2d_naive(img, sobel_x)

tf_out = tf.nn.conv2d(img[None, ..., None].astype("float32"),
                      sobel_x[..., None, None].astype("float32"),
                      strides=1, padding="VALID").numpy()[0, ..., 0]
print("=== convolution from scratch vs tf.nn.conv2d ===")
print(f"  my output shape {mine.shape}, TF output shape {tf_out.shape}")
print(f"  max |difference| = {np.abs(mine - tf_out).max():.2e}")
print(f"  the vertical edge shows up as a bright column:")
print(f"  {np.round(mine[5], 2)}")

# ============ 2. OUTPUT-SIZE ARITHMETIC ================================
print("\\n=== output size: valid vs same ===")
def out_size(n, f, s, padding):
    return int(np.ceil((n - f + 1) / s)) if padding == "valid" else int(np.ceil(n / s))
print(f"{'input':>7}{'kernel':>8}{'stride':>8}{'valid':>8}{'same':>7}{'TF valid':>10}{'TF same':>9}")
for n, f, s in [(28,3,1), (28,3,2), (28,5,1), (28,5,2), (32,7,2), (224,7,2)]:
    x = tf.zeros((1, n, n, 1))
    tv = keras.layers.Conv2D(1, f, strides=s, padding="valid")(x).shape[1]
    ts = keras.layers.Conv2D(1, f, strides=s, padding="same")(x).shape[1]
    print(f"{n:>7}{f:>8}{s:>8}{out_size(n,f,s,'valid'):>8}"
          f"{out_size(n,f,s,'same'):>7}{tv:>10}{ts:>9}")

# ============ 3. THE PARAMETER COUNT ===================================
print("\\n=== dense vs convolutional, 224x224x3 input ===")
H = W = 224; CH_IN = 3
dense_params = H*W*CH_IN * 1000 + 1000
conv_params  = 3*3*CH_IN * 64 + 64
print(f"  Dense(1000)          : {dense_params:>13,} parameters")
print(f"  Conv2D(64, 3x3)      : {conv_params:>13,} parameters")
print(f"  ratio                : {dense_params/conv_params:>13,.0f}x")
print(f"  and the conv layer OUTPUTS {H*W*64:,} activations vs the dense "
      f"layer's {1000:,}")

d = keras.Sequential([keras.layers.Input(shape=(224,224,3)),
                      keras.layers.Flatten(), keras.layers.Dense(1000)])
c = keras.Sequential([keras.layers.Input(shape=(224,224,3)),
                      keras.layers.Conv2D(64, 3, padding="same")])
print(f"\\n  Keras confirms: dense {d.count_params():,}  conv {c.count_params():,}")

# ============ 4. TRANSLATION EQUIVARIANCE ==============================
print("\\n=== translation equivariance ===")
x = np.zeros((1, 16, 16, 1), dtype="float32")
x[0, 4:8, 4:8, 0] = 1.0                              # a square
k = np.ones((3, 3, 1, 1), dtype="float32") / 9

shifted = np.roll(x, shift=3, axis=2)                # shift 3 px right
conv_then_shift = np.roll(tf.nn.conv2d(x, k, 1, "SAME").numpy(), 3, axis=2)
shift_then_conv = tf.nn.conv2d(shifted, k, 1, "SAME").numpy()
print(f"  conv(shift(x)) == shift(conv(x))? "
      f"{np.allclose(conv_then_shift, shift_then_conv)}")
print(f"  max |difference| = {np.abs(conv_then_shift - shift_then_conv).max():.2e}")
print("  a feature learned in one place works everywhere -- for free")

# --- a DENSE layer has no such property ------------------------------
dense = keras.layers.Dense(16)
flat_x   = tf.reshape(x, (1, -1))
flat_shx = tf.reshape(shifted, (1, -1))
print(f"\\n  dense(x) vs dense(shifted x): max |difference| = "
      f"{np.abs(dense(flat_x).numpy() - dense(flat_shx).numpy()).max():.4f}")
print("  completely different -- the dense layer has no idea they are related")

# ============ 5. A DENSE NET CANNOT TELL A SHUFFLED IMAGE ==============
print("\\n=== shuffle every pixel with a FIXED permutation ===")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
dg = load_digits()
X = (dg.images / 16.0).astype("float32")[..., None]
Xtr, Xte, ytr, yte = train_test_split(X, dg.target, test_size=.25,
                                      stratify=dg.target, random_state=42)
perm = np.random.default_rng(0).permutation(64)
def shuffle_px(a):
    return a.reshape(len(a), 64)[:, perm].reshape(len(a), 8, 8, 1)

def dense_model():
    return keras.Sequential([keras.layers.Input(shape=(8,8,1)),
                             keras.layers.Flatten(),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(10, activation="softmax")])
def conv_model():
    return keras.Sequential([keras.layers.Input(shape=(8,8,1)),
                             keras.layers.Conv2D(32, 3, activation="relu",
                                                 padding="same"),
                             keras.layers.MaxPooling2D(),
                             keras.layers.Conv2D(64, 3, activation="relu",
                                                 padding="same"),
                             keras.layers.GlobalAveragePooling2D(),
                             keras.layers.Dense(10, activation="softmax")])

print(f"{'model':<12}{'normal images':>16}{'shuffled pixels':>18}{'drop':>9}")
for nm, builder in [("dense", dense_model), ("conv", conv_model)]:
    accs = []
    for Xa, Xb in [(Xtr, Xte), (shuffle_px(Xtr), shuffle_px(Xte))]:
        tf.random.set_seed(0)
        m = builder()
        m.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
        m.fit(Xa, ytr, epochs=25, batch_size=64, verbose=0)
        accs.append(m.evaluate(Xb, yte, verbose=0)[1])
    print(f"{nm:<12}{accs[0]:>16.4f}{accs[1]:>18.4f}{accs[0]-accs[1]:>9.4f}")
print("\\nThe dense model is UNAFFECTED -- it never used spatial structure.")
print("The conv model degrades sharply -- its whole prior was destroyed.")
print("That prior is exactly what makes it better on real images.")
''',
        key="ch14_conv",
    )

    keypoints([
        "A conv layer slides a <b>small shared kernel</b> over the input: local "
        "receptive fields plus weight sharing.",
        "$3\\times3\\times3\\times64 + 64 = 1\\,792$ parameters versus 150 million "
        "for a comparable dense layer.",
        "<b>Translation equivariance</b>: $(T_\\delta \\mathbf{x}) * \\mathbf{w} "
        "= T_\\delta(\\mathbf{x} * \\mathbf{w})$ — a feature learned anywhere works "
        "everywhere.",
        "Output size: $\\lceil (n-f+1)/s \\rceil$ for <code>'valid'</code>, "
        "$\\lceil n/s \\rceil$ for <code>'same'</code>.",
        "The parameter saving is secondary; the <b>spatial prior</b> is the real "
        "gain — a dense net cannot tell a shuffled image from a real one.",
    ])


# ==========================================================================
def s_14_2():
    section("14.2", "Filters, Feature Maps, and Memory")

    lead(
        "A convolutional layer holds many filters at once, each producing its own "
        "feature map. Understanding the shapes — and the memory they consume — is "
        "what stops you from writing a model that will not fit."
    )

    sub("Stacking feature maps")

    md(
        "A layer with $f_n$ filters, applied to an input with $f_{n'}$ channels, "
        "holds a **4-D weight tensor**:"
    )

    math(r"""
    \mathbf{W} \in \mathbb{R}^{\,f_h \times f_w \times f_{n'} \times f_n}
    \qquad\qquad
    \mathbf{b} \in \mathbb{R}^{\,f_n}
    """)

    md(
        "Each of the $f_n$ filters spans **all** the input channels — a "
        "$3\\times3$ filter on an RGB image is really $3\\times3\\times3 = 27$ "
        "weights. The output is $f_n$ stacked 2-D feature maps."
    )

    table(
        ["Quantity", "Formula", "Example: $3\\times3$, 64 filters, RGB input"],
        [["Parameters", "$f_h f_w f_{n'} f_n + f_n$", "$3\\cdot3\\cdot3\\cdot64 + 64 = 1\\,792$"],
         ["Output shape", "$(o_h, o_w, f_n)$", "$(224, 224, 64)$ with 'same'"],
         ["Multiply-adds per image",
          "$o_h o_w f_n \\cdot f_h f_w f_{n'}$",
          "$224 \\cdot 224 \\cdot 64 \\cdot 27 \\approx 8.7 \\times 10^{7}$"]],
    )

    sub("The receptive field grows with depth")

    derive(
        [("A single $3\\times3$ layer sees a $3\\times3$ patch. What does a stack "
          "of them see?", None),
          ("For a stack of layers with kernel sizes $f_l$ and strides $s_l$, the "
           "receptive field of layer $L$ measured in input pixels is:",
           r"R_L \;=\; R_{L-1} + \bigl(f_L - 1\bigr)\prod_{l=1}^{L-1} s_l,"
           r"\qquad R_0 = 1"),
         ("With all strides equal to 1 and all kernels $3\\times3$, this collapses "
          "to a linear growth:",
          r"R_L = 1 + 2L"),
         ("So ten $3\\times3$ layers see $21\\times21$ pixels. Slow. But add "
          "<b>stride-2</b> layers (or pooling) and the products compound:",
          r"R_L = R_{L-1} + (f_L - 1)\,2^{\,\#\text{stride-2 layers before } L}"),
         ("<b>Two $3\\times3$ layers have the same receptive field as one "
          "$5\\times5$ layer</b>, with fewer parameters "
          "($2 \\times 9 = 18$ versus $25$ per channel pair) and an extra "
          "non-linearity in between. Three $3\\times3$ layers match one "
          "$7\\times7$ ($27$ versus $49$). This is the central insight of VGGNet "
          "(§14.4) and it is why $3\\times3$ became the universal default.", None)],
        title="Receptive field growth, and why 3×3 won",
    )

    sub("Memory — the thing that actually stops you")

    md(
        "Training memory is dominated not by the weights but by the "
        "**activations**, because backpropagation must keep every intermediate "
        "value for the backward pass (§12.8):"
    )

    math(r"""
    M_{\text{train}} \;\approx\;
    \underbrace{4 P}_{\text{weights}}
    \;+\; \underbrace{4 P}_{\text{gradients}}
    \;+\; \underbrace{8 P}_{\text{optimiser state (Adam)}}
    \;+\; \underbrace{4 B \sum_{l} A_l}_{\text{activations}}
    """)
    where({r"P": "number of parameters", r"B": "batch size",
           r"A_l": "number of activations produced by layer $l$",
           r"4": "bytes per float32 value"})

    pitfall(
        "The first layers are the memory hogs, not the last",
        "A $224\\times224\\times64$ feature map is 3.2 million values — 12.8 MB "
        "per image in float32, 410 MB for a batch of 32, and that is <b>one "
        "layer</b>. By the time you reach $7\\times7\\times512$ it is 25 088 "
        "values, 1 600× smaller. So an out-of-memory error almost always comes "
        "from the early, high-resolution layers.<br><br>"
        "<b>Fixes, in order:</b> reduce the batch size; use stride-2 in the first "
        "layer (halves each dimension, quartering the memory); use mixed "
        "precision (float16 activations, halving the total); use gradient "
        "checkpointing (recompute activations in the backward pass, trading "
        "~30 % time for a large memory saving).",
    )

    anim_header("Memory and shape through a real CNN")

    layers_spec = [
        ("input", 224, 3, 0),
        ("Conv 7×7/2, 64", 112, 64, 7 * 7 * 3 * 64 + 64),
        ("MaxPool 3×3/2", 56, 64, 0),
        ("Conv 3×3, 128", 56, 128, 3 * 3 * 64 * 128 + 128),
        ("MaxPool /2", 28, 128, 0),
        ("Conv 3×3, 256", 28, 256, 3 * 3 * 128 * 256 + 256),
        ("MaxPool /2", 14, 256, 0),
        ("Conv 3×3, 512", 14, 512, 3 * 3 * 256 * 512 + 512),
        ("MaxPool /2", 7, 512, 0),
        ("GlobalAvgPool", 1, 512, 0),
        ("Dense 1000", 1, 1000, 512 * 1000 + 1000),
    ]
    acts = [s * s * c for _, s, c, _ in layers_spec]
    prms = [p for _, _, _, p in layers_spec]
    BATCH = 32

    frames = []
    for k in range(1, len(layers_spec) + 1):
        names = [l[0] for l in layers_spec[:k]]
        a_mb = [a * 4 * BATCH / 1e6 for a in acts[:k]]
        p_mb = [p * 4 / 1e6 for p in prms[:k]]
        frames.append(go.Frame(name=str(k), data=[
            go.Bar(x=names, y=a_mb, marker=dict(color=C["danger"])),
            go.Bar(x=names, y=p_mb, marker=dict(color=C["primary"])),
        ], layout=go.Layout(annotations=[anim.annotate_step(
            f"{layers_spec[k-1][0]}   ·   output "
            f"{layers_spec[k-1][1]}×{layers_spec[k-1][1]}×{layers_spec[k-1][2]}"
            f"   ·   activations {a_mb[-1]:.1f} MB (batch {BATCH})"
            f"   ·   weights {p_mb[-1]:.2f} MB")])))

    f = go.Figure(data=[
        go.Bar(x=[layers_spec[0][0]], y=[acts[0] * 4 * BATCH / 1e6],
               name=f"activations (batch {BATCH})",
               marker=dict(color=C["danger"])),
        go.Bar(x=[layers_spec[0][0]], y=[prms[0] * 4 / 1e6], name="weights",
               marker=dict(color=C["primary"])),
    ])
    f.update_layout(height=460, barmode="group", yaxis_type="log",
                    yaxis_title="megabytes (log scale)",
                    title="Activations dominate memory — and they peak early",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    f.update_xaxes(tickangle=-35)
    anim.animate(f, frames, duration=nav.anim_ms(750), slider_prefix="layer ")
    figure(f, "The first conv layer alone holds more activation memory than every "
              "weight in the network.")

    code_lab(
        "Shapes, receptive fields, memory, and the 3×3 argument",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============ 1. THE 4-D WEIGHT TENSOR =================================
print("=== a Conv2D layer's weights ===")
layer = keras.layers.Conv2D(64, kernel_size=3, padding="same")
out = layer(tf.zeros((1, 32, 32, 3)))
W, b = layer.get_weights()
print(f"  input (1, 32, 32, 3) -> output {tuple(out.shape)}")
print(f"  W shape {W.shape} = (f_h, f_w, f_n', f_n)")
print(f"  b shape {b.shape} = one bias per FEATURE MAP, not per neuron")
print(f"  parameters = 3*3*3*64 + 64 = {3*3*3*64+64:,}  "
      f"(Keras says {layer.count_params():,})")

# ============ 2. RECEPTIVE FIELD =======================================
print("\\n=== receptive field growth ===")
def receptive_field(specs):
    """specs: list of (kernel, stride). Returns the RF after each layer."""
    R, jump, out = 1, 1, []
    for f, s in specs:
        R = R + (f - 1) * jump
        jump = jump * s
        out.append((R, jump))
    return out

print(f"{'architecture':<40}{'final RF':>10}{'params/channel-pair':>22}")
for nm, specs, pp in [
        ("one 5x5",                    [(5,1)],                 25),
        ("two 3x3",                    [(3,1),(3,1)],           18),
        ("one 7x7",                    [(7,1)],                 49),
        ("three 3x3",                  [(3,1),(3,1),(3,1)],     27),
        ("ten 3x3",                    [(3,1)]*10,              90),
        ("3x3 /2, then four 3x3",      [(3,2)]+[(3,1)]*4,       45)]:
    rf = receptive_field(specs)[-1][0]
    print(f"{nm:<40}{rf:>10}{pp:>22}")
print("\\nTwo 3x3 == one 5x5 receptive field, 28 % fewer parameters,")
print("and TWO non-linearities instead of one. That is the VGG argument.")

print(f"\\n{'depth (3x3, stride 1)':>24}{'receptive field':>18}")
for d in [1, 2, 5, 10, 20]:
    print(f"{d:>24}{receptive_field([(3,1)]*d)[-1][0]:>18}")
print(f"{'with a /2 every 2 layers':>24}")
for d in [2, 4, 8, 16]:
    specs = []
    for i in range(d):
        specs.append((3, 2 if i % 2 == 1 else 1))
    print(f"{d:>24}{receptive_field(specs)[-1][0]:>18}")
print("Striding compounds -- that is how a CNN sees the whole image quickly.")

# ============ 3. MEMORY ================================================
print("\\n=== memory accounting for a batch of 32 ===")
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu"),
    keras.layers.MaxPooling2D(3, strides=2, padding="same"),
    keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1000, activation="softmax"),
])
BATCH = 32
print(f"{'layer':<26}{'output shape':>20}{'activations':>14}{'MB/batch':>11}"
      f"{'params':>12}")
total_act = 0
for l in model.layers:
    shp = l.output.shape
    n_act = int(np.prod([d for d in shp[1:] if d is not None]))
    total_act += n_act
    print(f"{l.name:<26}{str(tuple(shp[1:])):>20}{n_act:>14,}"
          f"{n_act*4*BATCH/1e6:>11.1f}{l.count_params():>12,}")

P = model.count_params()
act_mb = total_act * 4 * BATCH / 1e6
print(f"\\n  parameters        : {P:>12,}  = {P*4/1e6:>8.1f} MB")
print(f"  gradients         : {P:>12,}  = {P*4/1e6:>8.1f} MB")
print(f"  Adam state (2x)   : {2*P:>12,}  = {2*P*4/1e6:>8.1f} MB")
print(f"  activations (b=32): {total_act*BATCH:>12,}  = {act_mb:>8.1f} MB")
print(f"  TOTAL             : {'':>12}    {(4*P*4 + act_mb*1e6)/1e6:>8.1f} MB")
print(f"\\n  activations are {act_mb/(P*4/1e6):.1f}x the weight memory")

print(f"\\n=== batch size vs memory ===")
print(f"{'batch':>8}{'activation MB':>16}{'total MB':>12}")
for B in [1, 8, 32, 128, 512]:
    a = total_act * 4 * B / 1e6
    print(f"{B:>8}{a:>16.1f}{a + 4*P*4/1e6:>12.1f}")
print("  batch size is your primary memory dial")

# ============ 4. WHERE THE MEMORY IS ===================================
print("\\n=== the first layers hold most of the activations ===")
acts = [(l.name, int(np.prod([d for d in l.output.shape[1:] if d is not None])))
        for l in model.layers]
tot = sum(a for _, a in acts)
run = 0
for nm, a in acts:
    run += a
    print(f"  {nm:<26}{a/tot:>7.1%}  cumulative {run/tot:>6.1%}")

# ============ 5. TWO 3x3 vs ONE 5x5, MEASURED ==========================
print("\\n=== does the VGG argument hold in practice? ===")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d = load_digits()
X = (d.images/16.).astype("float32")[..., None]
Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=.25,
                                      stratify=d.target, random_state=42)
print(f"{'design':<26}{'params':>10}{'test accuracy':>16}")
for nm, kernels in [("one 5x5", [5]), ("two 3x3", [3, 3]),
                    ("one 7x7", [7]), ("three 3x3", [3, 3, 3])]:
    tf.random.set_seed(0)
    layers = [keras.layers.Input(shape=(8,8,1))]
    for k in kernels:
        layers.append(keras.layers.Conv2D(32, k, padding="same",
                                          activation="relu"))
    layers += [keras.layers.GlobalAveragePooling2D(),
               keras.layers.Dense(10, activation="softmax")]
    m = keras.Sequential(layers)
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=30, batch_size=64, verbose=0)
    print(f"{nm:<26}{m.count_params():>10,}"
          f"{m.evaluate(Xte, yte, verbose=0)[1]:>16.4f}")
''',
        key="ch14_filters",
    )

    keypoints([
        "A conv layer's weights are 4-D: $f_h \\times f_w \\times f_{n'} \\times "
        "f_n$, plus one bias per feature map.",
        "Receptive field grows as $R_L = R_{L-1} + (f_L-1)\\prod s_l$ — striding "
        "compounds, so it grows fast.",
        "<b>Two $3\\times3$ layers = one $5\\times5$ receptive field</b>, fewer "
        "parameters, one more non-linearity.",
        "Training memory is dominated by <b>activations</b>, not weights, and they "
        "peak in the <b>early</b> layers.",
        "Memory dials in order: batch size, early striding, mixed precision, "
        "gradient checkpointing.",
    ])


# ==========================================================================
def s_14_3():
    section("14.3", "Pooling Layers")

    lead(
        "Pooling shrinks the feature maps. It has no parameters at all — it is "
        "pure downsampling, and its job is to buy invariance and cut computation."
    )

    table(
        ["Layer", "Operation", "Use"],
        [["<code>MaxPooling2D(2)</code>", "Maximum of each $2\\times2$ window",
          "The classic; keeps the strongest activation"],
         ["<code>AveragePooling2D(2)</code>", "Mean of each window",
          "Rare now — max preserves the strongest features better"],
         ["<code>GlobalAveragePooling2D()</code>",
          "Mean of each <b>entire</b> feature map: $(h,w,c) \\to (c,)$",
          "<b>Standard</b> before the output layer since GoogLeNet"],
         ["<code>GlobalMaxPooling2D()</code>", "Max over each feature map",
          "Occasionally used for detection"],
         ["Strided convolution", "A conv with stride 2 instead of pooling",
          "Modern alternative — <b>learned</b> downsampling"]],
    )

    sub("What pooling buys — and what it costs")

    derive(
        [("<b>Invariance.</b> A $2\\times2$ max-pool outputs the same value if the "
          "strongest activation moves anywhere within its window. Stack several "
          "and a feature can move considerably without changing the deep "
          "representation.", None),
         ("After $k$ pooling layers of size 2, an input feature can move by "
          "roughly $2^k$ pixels before the deepest representation changes:",
          r"\text{invariance radius} \;\approx\; 2^{k}"),
         ("<b>Computation.</b> Each $2\\times2$ pool divides the spatial extent by "
          "4, so all subsequent layers cost a quarter as much:",
          r"\text{cost after } k \text{ pools} \;\propto\; \frac{1}{4^{k}}"),
         ("<b>The cost.</b> You destroy spatial precision. That is fine for "
          "classification ('is there a cat?') and <b>disastrous</b> for "
          "segmentation ('which pixels are cat?'). It is why segmentation "
          "architectures (§14.8) must undo the pooling with upsampling and skip "
          "connections.", None),
         ("<b>Invariance is not always wanted.</b> If the position of a feature "
          "<i>is</i> the answer — object detection, pose estimation, medical "
          "localisation — too much pooling actively harms you. Modern "
          "architectures often replace pooling with stride-2 convolutions, which "
          "downsample while <i>learning</i> what to keep.", None)],
        title="The invariance/precision trade-off",
    )

    sub("Global average pooling")

    idea(
        "GAP replaced the giant Dense layer, and it was a huge win",
        "AlexNet flattened a $6\\times6\\times256$ feature map into 9 216 values "
        "and fed it to a 4 096-unit Dense layer — <b>37.7 million parameters</b> "
        "in one layer, more than half the network. GoogLeNet replaced it with "
        "<code>GlobalAveragePooling2D</code>: a $7\\times7\\times1024$ map becomes "
        "1 024 values with <b>zero parameters</b>. Less overfitting, far fewer "
        "weights, and the network accepts <b>any input size</b> because the "
        "output dimension no longer depends on the spatial extent.",
    )

    anim_header("Max, average and global pooling on the same feature map")

    rng = np.random.default_rng(3)
    fmap = np.abs(rng.normal(0, 1, (8, 8))) * 2
    fmap[2:4, 5:7] += 4.0                  # a strong local response

    def pool(a, size, mode):
        h, w = a.shape
        oh, ow = h // size, w // size
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                win = a[i * size:(i + 1) * size, j * size:(j + 1) * size]
                out[i, j] = win.max() if mode == "max" else win.mean()
        return out

    views = [
        ("original 8×8 feature map", fmap, "64 values"),
        ("MaxPooling2D(2) → 4×4", pool(fmap, 2, "max"),
         "16 values, keeps the strongest"),
        ("MaxPooling2D(2) again → 2×2", pool(pool(fmap, 2, "max"), 2, "max"),
         "4 values, invariance radius ≈ 4 px"),
        ("AveragePooling2D(2) → 4×4", pool(fmap, 2, "avg"),
         "16 values, smoother, dilutes the peak"),
        ("GlobalAveragePooling2D → scalar",
         np.full((1, 1), fmap.mean()), "1 value, ZERO parameters"),
        ("GlobalMaxPooling2D → scalar",
         np.full((1, 1), fmap.max()), "1 value, the strongest response"),
    ]
    frames = [go.Frame(name=str(i + 1), data=[
        go.Heatmap(z=z[::-1], colorscale=nav.cscale(), showscale=False,
                   xgap=2, ygap=2, zmin=0, zmax=fmap.max(),
                   text=np.round(z, 1)[::-1], texttemplate="%{text}",
                   textfont=dict(size=9))],
        layout=go.Layout(title=f"{nm}   —   {note_}"))
        for i, (nm, z, note_) in enumerate(views)]

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=420, title=f"{views[0][0]}   —   {views[0][2]}")
    f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    anim.animate(f, frames, duration=nav.anim_ms(1500), slider_prefix="stage ")
    figure(f)

    code_lab(
        "Pooling: invariance, cost, and pooling versus strided convolution",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============ 1. POOLING HAS NO PARAMETERS =============================
print("=== pooling layers ===")
x = tf.random.normal((1, 16, 16, 8), seed=0)
for nm, layer in [("MaxPooling2D(2)",         keras.layers.MaxPooling2D(2)),
                  ("AveragePooling2D(2)",     keras.layers.AveragePooling2D(2)),
                  ("MaxPooling2D(2, str=1)",  keras.layers.MaxPooling2D(2, strides=1)),
                  ("GlobalAveragePooling2D",  keras.layers.GlobalAveragePooling2D()),
                  ("GlobalMaxPooling2D",      keras.layers.GlobalMaxPooling2D())]:
    out = layer(x)
    print(f"  {nm:<26} (1,16,16,8) -> {str(tuple(out.shape)):<16} "
          f"params {layer.count_params()}")

# ============ 2. INVARIANCE TO SMALL SHIFTS ============================
print("\\n=== how much can a feature move? ===")
img = np.zeros((1, 32, 32, 1), dtype="float32")
img[0, 14:18, 14:18, 0] = 1.0

def represent(x, n_pools):
    z = tf.constant(x)
    for _ in range(n_pools):
        z = keras.layers.MaxPooling2D(2)(z)
    return z.numpy()

print(f"{'pools':>7}{'output':>12}" +
      "".join(f"{f'shift {s}px':>12}" for s in [1, 2, 4, 8]))
for n in range(4):
    base = represent(img, n)
    row = f"{n:>7}{str(base.shape[1:3]):>12}"
    for s in [1, 2, 4, 8]:
        shifted = represent(np.roll(img, s, axis=2), n)
        same = np.mean(np.isclose(base, shifted))
        row += f"{same:>11.1%}"
    print(row)
print("  more pooling -> more positions produce an IDENTICAL representation")
print("  (the numbers are the fraction of output values left unchanged)")

# ============ 3. THE COMPUTATIONAL SAVING ==============================
print("\\n=== pooling cuts the cost of everything downstream ===")
print(f"{'after k pools':>15}{'spatial':>12}{'values':>12}{'relative cost':>16}")
size, ch = 224, 64
for k in range(5):
    s = size // (2**k)
    print(f"{k:>15}{f'{s}x{s}':>12}{s*s*ch:>12,}{1/(4**k):>16.4f}")

# ============ 4. POOLING vs STRIDED CONVOLUTION ========================
print("\\n=== pooling vs learned downsampling ===")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d = load_digits()
X = (d.images/16.).astype("float32")[..., None]
Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=.25,
                                      stratify=d.target, random_state=42)

def build(mode):
    layers = [keras.layers.Input(shape=(8,8,1))]
    for filt in (32, 64):
        if mode == "pool":
            layers += [keras.layers.Conv2D(filt, 3, padding="same",
                                           activation="relu"),
                       keras.layers.MaxPooling2D(2, padding="same")]
        elif mode == "avgpool":
            layers += [keras.layers.Conv2D(filt, 3, padding="same",
                                           activation="relu"),
                       keras.layers.AveragePooling2D(2, padding="same")]
        else:
            layers += [keras.layers.Conv2D(filt, 3, strides=2, padding="same",
                                           activation="relu")]
    layers += [keras.layers.GlobalAveragePooling2D(),
               keras.layers.Dense(10, activation="softmax")]
    m = keras.Sequential(layers)
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return m

print(f"{'downsampling':<24}{'params':>10}{'test accuracy':>16}")
for nm, mode in [("MaxPooling2D", "pool"), ("AveragePooling2D", "avgpool"),
                 ("stride-2 Conv (learned)", "stride")]:
    tf.random.set_seed(0)
    m = build(mode)
    m.fit(Xtr, ytr, epochs=30, batch_size=64, verbose=0)
    print(f"{nm:<24}{m.count_params():>10,}"
          f"{m.evaluate(Xte, yte, verbose=0)[1]:>16.4f}")
print("  strided conv LEARNS what to keep, at the cost of extra parameters")

# ============ 5. GAP vs FLATTEN+DENSE ==================================
print("\\n=== GlobalAveragePooling replaced the giant Dense layer ===")
def head(kind):
    base = [keras.layers.Input(shape=(8,8,1)),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu")]
    if kind == "flatten":
        base += [keras.layers.Flatten(),
                 keras.layers.Dense(256, activation="relu")]
    else:
        base += [keras.layers.GlobalAveragePooling2D()]
    base.append(keras.layers.Dense(10, activation="softmax"))
    m = keras.Sequential(base)
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return m

print(f"{'head':<28}{'params':>10}{'train acc':>12}{'test acc':>11}{'gap':>9}")
for nm, kind in [("Flatten + Dense(256)", "flatten"),
                 ("GlobalAveragePooling2D", "gap")]:
    tf.random.set_seed(0)
    m = head(kind)
    m.fit(Xtr, ytr, epochs=30, batch_size=64, verbose=0)
    tr = m.evaluate(Xtr, ytr, verbose=0)[1]
    te = m.evaluate(Xte, yte, verbose=0)[1]
    print(f"{nm:<28}{m.count_params():>10,}{tr:>12.4f}{te:>11.4f}{tr-te:>9.4f}")

# --- the AlexNet arithmetic -----------------------------------------
print(f"\\n  AlexNet's first Dense layer: 6*6*256 = {6*6*256:,} inputs "
      f"-> 4096 units")
print(f"    = {6*6*256*4096 + 4096:,} parameters IN ONE LAYER")
print(f"  GAP on 7x7x1024 -> 1024 values, {0} parameters")

# ============ 6. GAP MAKES THE INPUT SIZE FREE =========================
print("\\n=== GAP accepts any input size ===")
gap_model = keras.Sequential([
    keras.layers.Input(shape=(None, None, 1)),          # UNKNOWN spatial size
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax")])
for size in [8, 16, 32, 64]:
    out = gap_model(tf.zeros((1, size, size, 1)))
    print(f"  input {size:>3}x{size:<3} -> output {tuple(out.shape)}")
print("  a Flatten+Dense head would REQUIRE one fixed input size")
''',
        key="ch14_pooling",
    )

    keypoints([
        "Pooling has <b>zero parameters</b>; it buys invariance and divides "
        "downstream cost by $4$ per $2\\times2$ pool.",
        "Invariance radius grows as $2^k$ after $k$ pooling layers — good for "
        "classification, bad for localisation.",
        "<b>GlobalAveragePooling2D</b> replaced the giant Dense head: zero "
        "parameters, less overfitting, any input size.",
        "Modern nets often use <b>stride-2 convolutions</b> instead — learned "
        "downsampling.",
        "Segmentation must undo pooling with upsampling and skip connections "
        "(§14.8).",
    ])

# ==========================================================================
def s_14_4():
    section("14.4", "CNN Architectures — LeNet to VGGNet")

    lead(
        "The architectures are a history of one question: how do you go deeper "
        "without the network becoming untrainable? Each generation answers it "
        "differently, and each answer is still in use."
    )

    sub("The general pattern")

    md(
        """
Every classical CNN follows the same skeleton:

* A stack of **convolution → ReLU → pooling** blocks.
* Image gets **smaller** (pooling / striding) and **deeper** (more filters) at
  each stage — the filter count typically **doubles** whenever the spatial size
  **halves**, keeping the per-layer cost roughly constant.
* A classification **head** at the end: originally Flatten + Dense, now almost
  always GlobalAveragePooling + Dense.
        """
    )

    codenote(
        "A common beginner mistake",
        "Using large kernels. A single $5\\times5$ or $7\\times7$ layer costs more "
        "parameters than two or three stacked $3\\times3$ layers with the same "
        "receptive field, and buys one non-linearity instead of two or three "
        "(§14.2). <b>The one exception is the very first layer</b>, where a large "
        "kernel with a large stride is used deliberately to cut the spatial size "
        "immediately, while the input has only 3 channels so the cost is low.",
    )

    sub("The lineage")

    table(
        ["Year", "Network", "Top-5 error", "Layers", "Params", "The idea"],
        [["1998", "<b>LeNet-5</b>", "— (MNIST)", "7", "60 K",
          "The first working CNN: conv → pool → conv → pool → dense"],
         ["2012", "<b>AlexNet</b>", "17.0 %", "8", "60 M",
          "Deeper, ReLU, dropout, GPU training, data augmentation"],
         ["2014", "<b>GoogLeNet</b> (Inception)", "6.7 %", "22", "6 M",
          "<b>Inception modules</b> — parallel kernel sizes; $1\\times1$ "
          "bottlenecks; GAP head"],
         ["2014", "<b>VGGNet</b>", "7.3 %", "16–19", "138 M",
          "Only $3\\times3$ convs, stacked deep. Simple and enormous"],
         ["2015", "<b>ResNet</b>", "3.6 %", "34–152", "25–60 M",
          "<b>Skip connections</b> — the breakthrough that unlocked real depth"],
         ["2016", "<b>Xception</b>", "~3.4 %", "36 blocks", "23 M",
          "<b>Depthwise separable</b> convolutions"],
         ["2017", "<b>SENet</b>", "2.25 %", "varies", "varies",
          "<b>Squeeze-and-excitation</b> — learned per-channel attention"],
         ["2019", "<b>EfficientNet</b>", "~2.9 %", "varies", "5–66 M",
          "<b>Compound scaling</b> of depth, width and resolution together"],
         ["2020", "<b>Vision Transformer</b>", "~2.6 %", "12–32", "86–632 M",
          "No convolution at all — patches + self-attention (Ch. 16)"]],
        "Top-5 error on ImageNet. Human performance is roughly 5 %.",
    )

    sub("The 1×1 convolution")

    idea(
        "A 1×1 convolution is not a no-op — it is a channel mixer",
        "It cannot see any spatial neighbourhood, but it spans <b>all</b> the "
        "input channels. So a $1\\times1$ conv with $f_n$ filters applies a "
        "learned $f_{n'} \\to f_n$ linear map <i>at every spatial position "
        "independently</i> — equivalently, a Dense layer applied pixel-wise. "
        "Two uses: <b>dimensionality reduction</b> (a cheap bottleneck before an "
        "expensive $3\\times3$ or $5\\times5$) and <b>adding non-linearity</b> "
        "without touching the spatial extent.",
    )

    derive(
        [("Consider a $28\\times28\\times256$ input and a $5\\times5$ convolution "
          "producing 64 maps. Direct cost:",
          r"28 \cdot 28 \cdot 64 \cdot (5 \cdot 5 \cdot 256) "
          r"\;=\; 3.21 \times 10^{8} \text{ multiply-adds}"),
         ("Now insert a $1\\times1$ bottleneck reducing 256 channels to 32 first:",
          r"\underbrace{28 \cdot 28 \cdot 32 \cdot (1 \cdot 1 \cdot 256)}_{\text{the }1\times1}"
          r" + \underbrace{28 \cdot 28 \cdot 64 \cdot (5 \cdot 5 \cdot 32)}_{\text{the }5\times5}"),
         ("which evaluates to:",
          r"6.4 \times 10^{6} + 4.0 \times 10^{7} \;=\; 4.66 \times 10^{7}"),
         ("A <b>6.9× reduction</b> in compute, for the same output shape and one "
          "extra non-linearity. This is the bottleneck trick, and it appears in "
          "Inception, ResNet-50+ and almost every efficient architecture since.",
          None)],
        title="Why the 1×1 bottleneck saves 7× the compute",
    )

    sub("Inception modules")

    md(
        "GoogLeNet's insight: rather than choosing between a $1\\times1$, "
        "$3\\times3$ or $5\\times5$ kernel, **run all of them in parallel and "
        "concatenate the results**. The network learns which scale matters, per "
        "layer, from data. $1\\times1$ bottlenecks keep it affordable."
    )

    anim_header("An Inception module assembling")

    branches = [
        ("1×1 conv, 64", C["accent"], 1),
        ("1×1 (96) → 3×3 (128)", SEQ[0], 2),
        ("1×1 (16) → 5×5 (32)", SEQ[1], 2),
        ("3×3 maxpool → 1×1 (32)", SEQ[2], 2),
        ("depth-concat → 256 maps", C["success"], 1),
    ]
    frames = []
    for k in range(1, len(branches) + 1):
        shapes, ann = [], []
        shapes.append(go.Scatter(x=[3.2, 6.0, 6.0, 3.2, 3.2],
                                 y=[3.6, 3.6, 4.3, 4.3, 3.6], fill="toself",
                                 fillcolor=alpha(C["muted"], .8),
                                 line=dict(color="#fff", width=2),
                                 showlegend=False, hoverinfo="skip"))
        ann.append(dict(x=4.6, y=3.95, text="input feature maps",
                        showarrow=False, font=dict(size=10, color="#fff")))
        for i, (nm, col, depth) in enumerate(branches[:min(k, 4)]):
            y = 2.6 - i * .85
            for d in range(depth):
                x0 = 1.0 + d * 2.6
                shapes.append(go.Scatter(
                    x=[x0, x0 + 2.2, x0 + 2.2, x0, x0],
                    y=[y - .3, y - .3, y + .3, y + .3, y - .3], fill="toself",
                    fillcolor=alpha(col, .85), line=dict(color="#fff", width=2),
                    showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=1.0 + (depth - 1) * 2.6 + 2.4, y=y, text=nm,
                            showarrow=False, xanchor="left",
                            font=dict(size=10, color=C["ink"])))
        if k >= 5:
            shapes.append(go.Scatter(x=[3.2, 6.0, 6.0, 3.2, 3.2],
                                     y=[-1.6, -1.6, -.9, -.9, -1.6],
                                     fill="toself",
                                     fillcolor=alpha(C["success"], .9),
                                     line=dict(color="#fff", width=2),
                                     showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=4.6, y=-1.25, text="depth concatenation → 256 maps",
                            showarrow=False, font=dict(size=10, color="#fff")))
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann,
                                                title=branches[k - 1][0])))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=430, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[.5, 12]),
                    yaxis=dict(visible=False, range=[-2.1, 4.7]),
                    annotations=list(frames[0].layout.annotations),
                    title=branches[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(1200), slider_prefix="branch ")
    figure(f, "Four parallel paths, concatenated along the channel axis. The "
              "1×1 bottlenecks are what make the 3×3 and 5×5 branches affordable.")

    code_lab(
        "Build LeNet-5, an Inception module, and a VGG block",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras
from functools import partial

tf.random.set_seed(42)

# ============ 1. LeNet-5 (1998) ========================================
print("=== LeNet-5 ===")
lenet = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 1)),
    keras.layers.Conv2D(6, 5, activation="tanh", padding="valid"),   # C1
    keras.layers.AveragePooling2D(2),                                # S2
    keras.layers.Conv2D(16, 5, activation="tanh", padding="valid"),  # C3
    keras.layers.AveragePooling2D(2),                                # S4
    keras.layers.Conv2D(120, 5, activation="tanh", padding="valid"), # C5
    keras.layers.Flatten(),
    keras.layers.Dense(84, activation="tanh"),                       # F6
    keras.layers.Dense(10, activation="softmax"),                    # output
])
print(f"  {len(lenet.layers)} layers, {lenet.count_params():,} parameters")
print(f"  note: tanh and AVERAGE pooling -- ReLU and max pooling came later")

# ============ 2. THE 1x1 BOTTLENECK ARITHMETIC =========================
print()
print("=== why 1x1 bottlenecks save ~7x the compute ===")
H = W = 28; C_IN = 256; C_OUT = 64
direct = H*W*C_OUT * (5*5*C_IN)
for reduce_to in [16, 32, 64, 128]:
    bottleneck = H*W*reduce_to*(1*1*C_IN) + H*W*C_OUT*(5*5*reduce_to)
    print(f"  reduce {C_IN} -> {reduce_to:>3} first: {bottleneck:>12,} MACs "
          f"({direct/bottleneck:>5.2f}x cheaper than {direct:,})")

x = tf.zeros((1, 28, 28, 256))
one_by_one = keras.layers.Conv2D(32, 1)
print()
print(f"  Conv2D(32, 1) on (28,28,256) -> {tuple(one_by_one(x).shape)}")
print(f"  parameters: 1*1*256*32 + 32 = {1*1*256*32+32:,}")
print(f"  it is a Dense(32) applied INDEPENDENTLY at each of the 784 positions")

# ============ 3. AN INCEPTION MODULE ===================================
print()
print("=== Inception module ===")
def inception_module(x, f1, f3_in, f3, f5_in, f5, fpool):
    b1 = keras.layers.Conv2D(f1, 1, activation="relu", padding="same")(x)

    b2 = keras.layers.Conv2D(f3_in, 1, activation="relu", padding="same")(x)
    b2 = keras.layers.Conv2D(f3, 3, activation="relu", padding="same")(b2)

    b3 = keras.layers.Conv2D(f5_in, 1, activation="relu", padding="same")(x)
    b3 = keras.layers.Conv2D(f5, 5, activation="relu", padding="same")(b3)

    b4 = keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
    b4 = keras.layers.Conv2D(fpool, 1, activation="relu", padding="same")(b4)

    return keras.layers.Concatenate(axis=-1)([b1, b2, b3, b4])

inp = keras.layers.Input(shape=(28, 28, 192))
out = inception_module(inp, 64, 96, 128, 16, 32, 32)     # GoogLeNet's 3a
incep = keras.Model(inp, out)
print(f"  input (28,28,192) -> output {tuple(out.shape)}")
print(f"  64 + 128 + 32 + 32 = {64+128+32+32} output maps")
print(f"  parameters: {incep.count_params():,}")

plain = keras.Sequential([keras.layers.Input(shape=(28,28,192)),
                          keras.layers.Conv2D(256, 5, padding="same")])
print(f"  a single 5x5 conv to 256 maps: {plain.count_params():,} parameters")
print(f"  Inception is {plain.count_params()/incep.count_params():.1f}x smaller "
      f"AND sees four scales at once")

# ============ 4. A VGG BLOCK ===========================================
print()
print("=== VGG: only 3x3, stacked ===")
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")

def vgg_block(x, filters, n_convs):
    for _ in range(n_convs):
        x = DefaultConv2D(filters)(x)
    return keras.layers.MaxPooling2D(2)(x)

inp = keras.layers.Input(shape=(224, 224, 3))
z = inp
for filters, n in [(64,2), (128,2), (256,3), (512,3), (512,3)]:
    z = vgg_block(z, filters, n)
z = keras.layers.Flatten()(z)
z = keras.layers.Dense(4096, activation="relu")(z)
z = keras.layers.Dense(4096, activation="relu")(z)
vgg16 = keras.Model(inp, keras.layers.Dense(1000, activation="softmax")(z))
print(f"  VGG-16: {vgg16.count_params():,} parameters")
conv_p = sum(l.count_params() for l in vgg16.layers
             if isinstance(l, keras.layers.Conv2D))
dense_p = sum(l.count_params() for l in vgg16.layers
              if isinstance(l, keras.layers.Dense))
print(f"    convolutional layers: {conv_p:>12,}  ({conv_p/vgg16.count_params():.1%})")
print(f"    dense layers        : {dense_p:>12,}  ({dense_p/vgg16.count_params():.1%})")
print("  89 % of VGG's weights are in three Dense layers doing very little work.")
print("  GoogLeNet's GlobalAveragePooling head removed all of them.")

inp2 = keras.layers.Input(shape=(224,224,3))
z = inp2
for filters, n in [(64,2), (128,2), (256,3), (512,3), (512,3)]:
    z = vgg_block(z, filters, n)
z = keras.layers.GlobalAveragePooling2D()(z)
vgg_gap = keras.Model(inp2, keras.layers.Dense(1000, activation="softmax")(z))
print()
print(f"  same conv stack + GAP head: {vgg_gap.count_params():,} parameters")
print(f"  a {vgg16.count_params()/vgg_gap.count_params():.1f}x reduction")

# ============ 5. THE DOUBLING RULE =====================================
print()
print("=== filters double when the spatial size halves ===")
print(f"{'stage':>7}{'spatial':>12}{'filters':>10}{'activations':>14}"
      f"{'relative cost':>16}")
size, filt = 224, 64
base = None
for stage in range(6):
    acts = size*size*filt
    cost = size*size*filt*9*(filt if stage else 3)
    if base is None: base = cost
    print(f"{stage:>7}{f'{size}x{size}':>12}{filt:>10}{acts:>14,}{cost/base:>16.3f}")
    size //= 2; filt = min(filt*2, 512)
print("  keeping the cost roughly flat is exactly why the rule exists")
''',
        key="ch14_architectures",
    )

    keypoints([
        "The skeleton is always conv → ReLU → pool, getting <b>smaller and "
        "deeper</b>; filters double when the size halves.",
        "A <b>$1\\times1$ convolution</b> is a per-pixel Dense layer over channels "
        "— used as a cheap bottleneck.",
        "The bottleneck trick cuts a $5\\times5$ layer's compute by ~7× for the "
        "same output shape.",
        "<b>Inception</b> runs several kernel sizes in parallel and concatenates; "
        "the network learns the scale.",
        "<b>VGG</b> proved that stacked $3\\times3$ is enough — but 89 % of its "
        "weights sat in a Dense head that GAP made unnecessary.",
    ])


# ==========================================================================
def s_14_5():
    section("14.5", "ResNet, Xception, SENet")

    lead(
        "Three ideas that each removed a different ceiling: skip connections "
        "removed the depth ceiling, depthwise separable convolutions removed the "
        "cost ceiling, and squeeze-and-excitation added channel attention almost "
        "for free."
    )

    sub("ResNet — the degradation problem")

    pitfall(
        "Deeper networks were performing WORSE — and not from overfitting",
        "He et al. observed that a 56-layer plain network had <b>higher training "
        "error</b> than a 20-layer one. That cannot be overfitting, which would "
        "show as low training error and high test error. It is an "
        "<b>optimisation</b> failure: the deeper network is strictly more "
        "expressive (it could set the extra layers to the identity and match the "
        "shallow one exactly) yet gradient descent cannot find that solution.",
    )

    md("The fix is to make the identity **the default** rather than something to "
       "be learned:")

    math(r"""
    \mathbf{y} \;=\; \mathcal{F}(\mathbf{x}) \;+\; \mathbf{x}
    \qquad\Longleftrightarrow\qquad
    \mathcal{F}(\mathbf{x}) \;=\; \mathbf{y} - \mathbf{x}
    """)

    derive(
        [("The block now learns the <b>residual</b> $\\mathcal{F} = "
          "\\mathbf{y} - \\mathbf{x}$ rather than the full mapping "
          "$\\mathbf{y}$.", None),
         ("<b>Optimisation.</b> To make the block an identity, the network only "
          "needs $\\mathcal{F} \\to \\mathbf{0}$ — drive the weights toward zero, "
          "which is exactly what weight decay and small initialisation already "
          "do. In a plain block it would have to <i>learn</i> the identity "
          "function, which is far harder.", None),
         ("<b>Gradient flow.</b> Differentiating the block:",
          r"\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \;=\; "
          r"\mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}"),
         ("Through $L$ stacked blocks the Jacobian is a product of such terms, "
          "and expanding it gives a sum that <b>always contains the term "
          "$\\mathbf{I}$</b>:",
          r"\prod_{l=1}^{L}\left(\mathbf{I} + \frac{\partial \mathcal{F}_l}{\partial \mathbf{x}}\right) "
          r"= \mathbf{I} + \sum_l \frac{\partial \mathcal{F}_l}{\partial \mathbf{x}} + \dots"),
         ("So even if every $\\partial\\mathcal{F}_l/\\partial\\mathbf{x}$ "
          "vanishes, the gradient still reaches the first layer undiminished. "
          "The $\\gamma^L$ decay of §11.1 has a floor of 1.", None),
         ("<b>The result.</b> ResNet trained 152 layers where plain networks "
          "failed past about 20, and won ILSVRC 2015 with 3.6 % top-5 error. "
          "Skip connections are now in essentially every deep architecture, "
          "including Transformers (Ch. 16).", None)],
        title="Why the residual formulation works",
    )

    sub("Bottleneck residual blocks")

    md(
        "ResNet-50 and deeper use a three-layer bottleneck: $1\\times1$ reduce → "
        "$3\\times3$ → $1\\times1$ expand. The $3\\times3$ operates on a quarter "
        "of the channels, so the block is far cheaper than two plain $3\\times3$ "
        "layers at full width."
    )

    sub("Xception — depthwise separable convolutions")

    md(
        "A normal convolution mixes **space and channels simultaneously**. A "
        "separable convolution splits it into two steps:"
    )

    table(
        ["Step", "Operation", "Cost"],
        [["<b>Depthwise</b>", "One $f\\times f$ spatial filter <i>per input "
          "channel</i>, no mixing", "$f_h f_w f_{n'} \\cdot o_h o_w$"],
         ["<b>Pointwise</b>", "A $1\\times1$ convolution mixing the channels",
          "$f_{n'} f_n \\cdot o_h o_w$"],
         ["<b>Total</b>", "", "$o_h o_w f_{n'}(f_h f_w + f_n)$"],
         ["Normal conv", "Both at once",
          "$o_h o_w f_{n'} f_n f_h f_w$"]],
    )

    math(r"""
    \frac{\text{separable}}{\text{normal}} \;=\;
    \frac{f_h f_w + f_n}{f_h f_w \, f_n}
    \;=\; \frac{1}{f_n} + \frac{1}{f_h f_w}
    """)

    proof(
        "The separable convolution costs about 1/9 as much",
        "For a $3\\times3$ kernel and $f_n = 256$ output maps, the ratio is "
        "$\\frac{1}{256} + \\frac19 = 0.0039 + 0.1111 = 0.115$ — an <b>8.7× "
        "reduction</b>. The second term dominates, so the saving converges to "
        "$1/(f_h f_w) = 1/9$ for $3\\times3$ kernels regardless of channel count. "
        "This is the entire basis of MobileNet and Xception, and why they run on "
        "phones.",
    )

    warn(
        "Do not use separable convolutions in the first layer",
        "Depthwise convolution assumes spatial correlations and cross-channel "
        "correlations can be modelled separately. That is a good assumption in "
        "deep layers, where channels are abstract features. It is a <b>bad</b> "
        "assumption on an RGB input, where the three channels are strongly "
        "correlated and there are only three of them — the saving is negligible "
        "and you lose real modelling capacity. Xception itself uses ordinary "
        "convolutions in its entry flow.",
    )

    sub("SENet — squeeze and excitation")

    md(
        "An SE block adds **per-channel attention** to any existing block, at "
        "about 0.5 % extra parameters:"
    )

    md(
        """
1. **Squeeze** — `GlobalAveragePooling2D` reduces $(h, w, c)$ to $(c,)$: one
   number summarising each feature map.
2. **Excite** — a tiny two-layer bottleneck MLP, $c \\to c/r \\to c$ (usually
   $r = 16$), with a **sigmoid** output.
3. **Scale** — multiply each feature map by its gate value in $(0, 1)$.
        """
    )

    idea(
        "Why the bottleneck in the SE block matters",
        "The $c \\to c/r \\to c$ shape forces the block to learn a "
        "<i>low-dimensional</i> summary of which channels co-occur. Without the "
        "bottleneck it could learn an arbitrary per-channel gate; with it, the "
        "block must generalise — 'if I see mouth-like features, boost the "
        "nose and eye channels too'. SENet won ILSVRC 2017 with 2.25 % top-5 "
        "error by adding this to an existing architecture.",
    )

    anim_header("Residual, separable and SE blocks side by side")

    designs = [
        ("plain block: 3×3 → 3×3",
         ["x", "3×3 conv", "BN+ReLU", "3×3 conv", "BN", "ReLU", "y"], False,
         "gradient must pass through every layer"),
        ("residual block: y = F(x) + x",
         ["x", "3×3 conv", "BN+ReLU", "3×3 conv", "BN", "⊕ x", "ReLU", "y"], True,
         "∂y/∂x = I + ∂F/∂x — the gradient always has a path"),
        ("bottleneck residual (ResNet-50+)",
         ["x", "1×1 reduce", "3×3 conv", "1×1 expand", "⊕ x", "ReLU", "y"], True,
         "the 3×3 runs on 1/4 of the channels"),
        ("depthwise separable (Xception)",
         ["x", "depthwise 3×3", "pointwise 1×1", "BN+ReLU", "y"], False,
         "≈ 1/9 the cost of a normal 3×3"),
        ("squeeze-and-excitation",
         ["x", "any block", "GAP → (c,)", "Dense c/16", "Dense c, sigmoid",
          "⊗ scale", "y"], True,
         "learned per-channel gate, ~0.5 % extra parameters"),
    ]
    frames = []
    for title, boxes, has_skip, desc in designs:
        shapes, ann = [], []
        n = len(boxes)
        for i, nm in enumerate(boxes):
            x0 = i * 1.75
            col = (C["accent"] if i in (0, n - 1) else
                   C["success"] if ("⊕" in nm or "⊗" in nm) else SEQ[i % 4])
            shapes.append(go.Scatter(
                x=[x0, x0 + 1.5, x0 + 1.5, x0, x0],
                y=[-.32, -.32, .32, .32, -.32], fill="toself",
                fillcolor=alpha(col, .88), line=dict(color="#fff", width=2),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=x0 + .75, y=0, text=nm, showarrow=False,
                            font=dict(size=9, color="#fff")))
        if has_skip:
            shapes.append(go.Scatter(
                x=[.75, .75, (n - 3) * 1.75 + .75, (n - 3) * 1.75 + .75],
                y=[.32, 1.15, 1.15, .32], mode="lines",
                line=dict(color=C["danger"], width=3),
                showlegend=False, hoverinfo="skip"))
            ann.append(dict(x=(n - 3) * 1.75 / 2 + .75, y=1.4,
                            text="skip / gate path", showarrow=False,
                            font=dict(size=10, color=C["danger"])))
        ann.append(dict(x=n * 1.75 / 2, y=-1.0, text=desc, showarrow=False,
                        font=dict(size=11, color=C["ink_soft"])))
        frames.append(go.Frame(name=title.split(":")[0], data=shapes,
                               layout=go.Layout(annotations=ann, title=title)))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=340, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.4, 13]),
                    yaxis=dict(visible=False, range=[-1.6, 1.8]),
                    annotations=list(frames[0].layout.annotations),
                    title=designs[0][0])
    anim.animate(f, frames, duration=nav.anim_ms(1900), slider_prefix="block ")
    figure(f)

    code_lab(
        "ResNet-34, separable convs, and SE blocks — built and measured",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras
from functools import partial

tf.random.set_seed(42)
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)

# ============ 1. A RESIDUAL UNIT =======================================
@keras.utils.register_keras_serializable(package="MLPlatform")
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.filters, self.strides = filters, strides
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [DefaultConv2D(filters, kernel_size=1,
                                              strides=strides),
                                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip = inputs
        for layer in self.skip_layers:
            skip = layer(skip)
        return self.activation(Z + skip)            # THE SKIP CONNECTION

    def get_config(self):
        return {**super().get_config(), "filters": self.filters,
                "strides": self.strides}

# ============ 2. ResNet-34 =============================================
print("=== ResNet-34 ===")
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    DefaultConv2D(64, kernel_size=7, strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same"),
])
prev_filters = 64
for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Dense(1000, activation="softmax"))
n_res = sum(isinstance(l, ResidualUnit) for l in model.layers)
print(f"  {n_res} residual units = {2*n_res + 2} weighted layers")
print(f"  {model.count_params():,} parameters")
print(f"  (VGG-16 has 138M for WORSE accuracy)")

# ============ 3. DEPTHWISE SEPARABLE COST ==============================
print()
print("=== separable vs normal convolution ===")
def costs(h, w, c_in, c_out, f=3):
    normal = h*w*c_out*(f*f*c_in)
    depth  = h*w*c_in*(f*f)
    point  = h*w*c_out*c_in
    return normal, depth+point

print(f"{'shape':>18}{'normal MACs':>15}{'separable MACs':>17}{'ratio':>9}"
      f"{'theory':>9}")
for h, c_in, c_out in [(56,64,128), (28,128,256), (14,256,512), (7,512,1024)]:
    n, s = costs(h, h, c_in, c_out)
    theory = 1/c_out + 1/9
    print(f"{f'{h}x{h}x{c_in}->{c_out}':>18}{n:>15,}{s:>17,}{s/n:>9.3f}"
          f"{theory:>9.3f}")
print("  the ratio converges to 1/9 = 0.111 for 3x3 kernels")

x = tf.zeros((1, 28, 28, 128))
normal = keras.layers.Conv2D(256, 3, padding="same")
sep    = keras.layers.SeparableConv2D(256, 3, padding="same")
normal(x); sep(x)
print()
print(f"  Conv2D(256,3)          : {normal.count_params():>9,} parameters")
print(f"  SeparableConv2D(256,3) : {sep.count_params():>9,} parameters "
      f"({sep.count_params()/normal.count_params():.3f}x)")

# ============ 4. A SQUEEZE-AND-EXCITATION BLOCK ========================
@keras.utils.register_keras_serializable(package="MLPlatform")
class SEBlock(keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    def build(self, input_shape):
        c = input_shape[-1]
        self.gap  = keras.layers.GlobalAveragePooling2D()
        self.fc1  = keras.layers.Dense(max(c // self.ratio, 1),
                                       activation="relu")
        self.fc2  = keras.layers.Dense(c, activation="sigmoid")
        self.resh = keras.layers.Reshape((1, 1, c))
    def call(self, inputs):
        z = self.gap(inputs)          # SQUEEZE: (h,w,c) -> (c,)
        z = self.fc2(self.fc1(z))     # EXCITE : c -> c/r -> c, sigmoid
        return inputs * self.resh(z)  # SCALE
    def get_config(self):
        return {**super().get_config(), "ratio": self.ratio}

print()
print("=== squeeze-and-excitation ===")
se = SEBlock(16)
out = se(tf.random.normal((2, 14, 14, 256)))
print(f"  input (2,14,14,256) -> output {tuple(out.shape)}  (shape unchanged)")
print(f"  SE parameters: 256*16 + 16 + 16*256 + 256 = {256*16+16+16*256+256:,}")
print(f"  a ResNet block at this width has ~{3*3*256*256*2:,} -> SE adds "
      f"{(256*16+16+16*256+256)/(3*3*256*256*2):.2%}")

gates = se.fc2(se.fc1(se.gap(tf.random.normal((1, 14, 14, 256))))).numpy()[0]
print(f"  gate values: min {gates.min():.3f}  mean {gates.mean():.3f}  "
      f"max {gates.max():.3f}   (all in (0,1))")

# ============ 5. DOES ANY OF IT HELP ON A REAL TASK? ===================
print()
print("=== plain vs residual vs separable vs SE, measured ===")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
d = load_digits()
X = (d.images/16.).astype("float32")[..., None]
Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=.25,
                                      stratify=d.target, random_state=42)

def build(kind, depth=8, filters=32):
    inp = keras.layers.Input(shape=(8, 8, 1))
    z = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inp)
    for _ in range(depth):
        if kind == "plain":
            z = keras.layers.Conv2D(filters, 3, padding="same")(z)
            z = keras.layers.BatchNormalization()(z)
            z = keras.layers.Activation("relu")(z)
        elif kind == "residual":
            z = ResidualUnit(filters)(z)
        elif kind == "separable":
            z = keras.layers.SeparableConv2D(filters, 3, padding="same")(z)
            z = keras.layers.BatchNormalization()(z)
            z = keras.layers.Activation("relu")(z)
        elif kind == "residual+se":
            z = SEBlock(8)(ResidualUnit(filters)(z))
    z = keras.layers.GlobalAveragePooling2D()(z)
    m = keras.Model(inp, keras.layers.Dense(10, activation="softmax")(z))
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return m

print(f"{'block type':<18}{'depth':>7}{'params':>10}{'fit time':>11}{'test acc':>11}")
for kind in ["plain", "residual", "separable", "residual+se"]:
    for depth in [4, 10]:
        tf.random.set_seed(0)
        m = build(kind, depth=depth)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr, epochs=10, batch_size=64, verbose=0)
        dt = time.perf_counter()-t0
        print(f"{kind:<18}{depth:>7}{m.count_params():>10,}{dt:>10.1f}s"
              f"{m.evaluate(Xte, yte, verbose=0)[1]:>11.4f}")
print()
print("The gap between plain and residual WIDENS with depth -- that is the")
print("degradation problem, and the skip connection is the fix.")
''',
        key="ch14_resnet",
    )

    quiz(
        "A 56-layer plain CNN has higher <i>training</i> error than a 20-layer "
        "one. What is going on?",
        ["Overfitting", "The optimiser cannot find the identity mapping — a "
         "degradation, not a generalisation, problem",
         "Not enough training data", "The learning rate is too small"],
        1,
        "Overfitting would show as <i>low</i> training error. The deeper network "
        "is strictly more expressive but gradient descent cannot reach the "
        "solution. Residual connections make the identity the default, which "
        "solves it.",
        key="ch14q1",
    )

    keypoints([
        "<b>ResNet</b>: $\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}$, so "
        "$\\partial\\mathbf{y}/\\partial\\mathbf{x} = \\mathbf{I} + "
        "\\partial\\mathcal{F}/\\partial\\mathbf{x}$ — the gradient always has a "
        "path.",
        "The degradation problem is an <b>optimisation</b> failure, not "
        "overfitting.",
        "<b>Depthwise separable</b> convolution costs $\\frac{1}{f_n} + "
        "\\frac{1}{f_hf_w} \\approx 1/9$ of a normal $3\\times3$.",
        "Do not use separable convs on the RGB input layer — the assumption fails "
        "there.",
        "<b>SE blocks</b> add learned per-channel attention for ~0.5 % extra "
        "parameters.",
    ])

# ==========================================================================
def s_14_6():
    section("14.6", "Pretrained Models and Transfer Learning")

    lead(
        "You will almost never train a vision model from scratch. Keras ships "
        "ImageNet-pretrained weights for a dozen architectures, one line away — "
        "and §11.5's freeze-then-fine-tune protocol is what turns them into your "
        "model."
    )

    sub("Using a pretrained model directly")

    md(
        """
```python
model = keras.applications.ResNet50(weights="imagenet")

images_resized = tf.image.resize(images, [224, 224])
inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)
Y_proba = model.predict(inputs)
top_k = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
```
        """
    )

    pitfall(
        "Every architecture has its OWN preprocess_input — and they differ",
        "<code>resnet50.preprocess_input</code> subtracts the ImageNet mean and "
        "converts RGB→BGR (a Caffe legacy). "
        "<code>xception.preprocess_input</code> maps to $[-1, 1]$. "
        "<code>efficientnet.preprocess_input</code> is a no-op because "
        "normalisation is baked into the model. Use the wrong one and accuracy "
        "collapses <b>silently</b> — the model still returns confident "
        "predictions, they are just wrong. Always import the "
        "<code>preprocess_input</code> that ships with your chosen architecture.",
    )

    sub("Transfer learning, the two-phase protocol")

    md(
        """
Repeating §11.5 in the vision setting, where it matters most:

1. `include_top=False` drops the 1 000-class ImageNet head.
2. `base_model.trainable = False`, add your own head, **compile**, train a few
   epochs at a normal learning rate.
3. `base_model.trainable = True` (or unfreeze the top $N$ layers),
   **recompile at a learning rate 10–100× lower**, train again.
        """
    )

    warn(
        "Batch normalisation layers need special care when fine-tuning",
        "A frozen <code>BatchNormalization</code> layer in Keras still updates "
        "its <b>moving statistics</b> unless the whole layer is in inference "
        "mode. When you unfreeze a pretrained backbone, those statistics start "
        "moving toward your (small) dataset and can destroy the pretrained "
        "features. The standard fix is to keep BN layers frozen throughout: "
        "<code>for layer in base.layers: if isinstance(layer, "
        "keras.layers.BatchNormalization): layer.trainable = False</code>.",
    )

    sub("Which architecture should you start from?")

    table(
        ["Constraint", "Choice", "Why"],
        [["General default", "<b>EfficientNetV2</b> or <b>ConvNeXt</b>",
          "Best accuracy per FLOP among convolutional models"],
         ["Mobile / edge", "<b>MobileNetV3</b>", "Depthwise separable, tiny"],
         ["Maximum accuracy, compute available", "<b>ViT</b> or <b>ConvNeXt-L</b>",
          "But ViT needs a lot of data or heavy pretraining (Ch. 16)"],
         ["A well-understood baseline", "<b>ResNet50</b>",
          "Ubiquitous, well-studied, plenty of published numbers to compare with"],
         ["Very small dataset (< 1 000 images)",
          "Any backbone, <b>fully frozen</b>, train only the head",
          "Fine-tuning would overfit immediately"]],
    )

    anim_header("What a pretrained backbone has already learned")
    md(
        "Layer by layer, the features a trained CNN detects become more abstract. "
        "The first layers are edges and colours — universal, and reusable for "
        "*any* image task. That is why transfer learning works."
    )

    rng = np.random.default_rng(4)
    N = 24
    yy, xx = np.mgrid[0:N, 0:N] / N

    def edge(theta):
        return np.sin(12 * (xx * np.cos(theta) + yy * np.sin(theta)))

    def blob(cx, cy, r):
        return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))

    def texture(k):
        return np.sin(k * 18 * xx) * np.cos(k * 14 * yy)

    def part(cx, cy):
        return (blob(cx, cy, .12) - .6 * blob(cx + .16, cy, .07)
                - .6 * blob(cx - .16, cy, .07))

    layer_feats = [
        ("layer 1 — oriented edges and colour blobs",
         [edge(t) for t in np.linspace(0, np.pi, 8)],
         "universal · reusable for ANY image task"),
        ("layer 2 — corners, curves, simple textures",
         [texture(k) for k in np.linspace(.4, 1.6, 4)]
         + [edge(t) * edge(t + 1.1) for t in np.linspace(0, 2, 4)],
         "still very generic"),
        ("layer 3 — repeating patterns and motifs",
         [texture(k) * blob(.5, .5, .35) for k in np.linspace(.5, 2.2, 8)],
         "beginning to be domain-specific"),
        ("layer 4 — object parts",
         [part(.5 + .1 * np.cos(a), .5 + .1 * np.sin(a))
          for a in np.linspace(0, 6, 8)],
         "specific to the training distribution"),
        ("layer 5 — whole objects / classes",
         [part(.5, .45) + blob(.5, .7, .18) + .5 * edge(a)
          for a in np.linspace(0, 3, 8)],
         "TASK-SPECIFIC · this is what you replace"),
    ]

    frames = []
    for i, (title, feats, note_) in enumerate(layer_feats):
        grid = np.zeros((2 * N, 4 * N))
        for j, fm in enumerate(feats[:8]):
            r, c = divmod(j, 4)
            grid[r * N:(r + 1) * N, c * N:(c + 1) * N] = fm
        frames.append(go.Frame(name=str(i + 1), data=[
            go.Heatmap(z=grid[::-1], colorscale=nav.cscale(), showscale=False)],
            layout=go.Layout(title=f"{title}   —   {note_}")))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=380, title=f"{layer_feats[0][0]}   —   "
                                      f"{layer_feats[0][2]}")
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False, scaleanchor="x")
    anim.animate(f, frames, duration=nav.anim_ms(1600), slider_prefix="layer ")
    figure(f, "Stylised, but the progression is real: Zeiler & Fergus (2014) "
              "visualised exactly this in a trained AlexNet.")

    code_lab(
        "Load a pretrained model, then fine-tune it in two phases",
        '''import numpy as np, time
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. WHAT KERAS SHIPS ======================================
print("=== keras.applications ===")
apps = [("MobileNetV3Small", 224), ("EfficientNetV2B0", 224),
        ("ResNet50", 224), ("Xception", 299), ("ConvNeXtTiny", 224)]
print(f"{'model':<22}{'input':>8}{'parameters':>14}{'top-1 (ImageNet)':>19}")
top1 = {"MobileNetV3Small": "68.1 %", "EfficientNetV2B0": "78.7 %",
        "ResNet50": "74.9 %", "Xception": "79.0 %", "ConvNeXtTiny": "81.3 %"}
for name, size in apps:
    try:
        cls = getattr(keras.applications, name)
        m = cls(weights=None, include_top=True)     # weights=None: no download
        print(f"{name:<22}{size:>8}{m.count_params():>14,}{top1[name]:>19}")
        del m
    except Exception as e:
        print(f"{name:<22}{'--':>8}{'unavailable':>14}")

# ============ 2. EACH ONE HAS ITS OWN preprocess_input =================
print()
print("=== preprocess_input differs by architecture ===")
img = tf.constant(np.full((1, 224, 224, 3), 128.0, dtype="float32"))
print(f"  a uniform mid-grey image (all pixels 128.0) becomes:")
for mod_name in ["resnet50", "xception", "mobilenet_v3", "efficientnet_v2"]:
    try:
        mod = getattr(keras.applications, mod_name)
        out = mod.preprocess_input(tf.identity(img)).numpy()
        print(f"    {mod_name:<18} range [{out.min():>8.3f}, {out.max():>8.3f}]  "
              f"mean {out.mean():>8.3f}")
    except Exception:
        pass
print("  use the WRONG one and accuracy collapses -- silently")

# ============ 3. TRANSFER LEARNING, TWO PHASES =========================
print()
print("="*62)
print("Transfer learning on a small dataset")
print("="*62)

# a small synthetic image task: 3 shape classes, 32x32 RGB, 240 train images
def make_shapes(n, size=32, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size, 3), dtype="float32")
    y = np.zeros(n, dtype="int32")
    yy, xx = np.mgrid[0:size, 0:size] / size
    for i in range(n):
        cls = i % 3
        y[i] = cls
        bg = rng.uniform(.1, .4, 3)
        img = np.ones((size, size, 3)) * bg
        cy, cx = rng.uniform(.3, .7, 2)
        fg = rng.uniform(.6, 1.0, 3)
        if cls == 0:      m = ((xx-cx)**2 + (yy-cy)**2) < .022        # disc
        elif cls == 1:    m = (np.abs(xx-cx) < .16) & (np.abs(yy-cy) < .16)  # square
        else:             m = np.abs(xx-cx) + np.abs(yy-cy) < .22     # diamond
        img[m] = fg
        X[i] = np.clip(img + rng.normal(0, .04, img.shape), 0, 1)
    return X, y

Xtr, ytr = make_shapes(240, seed=0)
Xva, yva = make_shapes(150, seed=1)
Xte, yte = make_shapes(300, seed=2)
print(f"  {len(Xtr)} training images, 3 classes, 32x32x3")

# --- BASELINE: from scratch -------------------------------------------
def scratch_model():
    return keras.Sequential([
        keras.layers.Input(shape=(32,32,3)),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(3, activation="softmax")])

tf.random.set_seed(0)
m = scratch_model()
m.compile(loss="sparse_categorical_crossentropy",
          optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
m.fit(Xtr, ytr, epochs=40, batch_size=32, verbose=0)
acc_scratch = m.evaluate(Xte, yte, verbose=0)[1]
print(f"  from scratch                 : {acc_scratch:.4f}")

# --- PRETRAIN on a RELATED task, then transfer ------------------------
# (a real project would use keras.applications with weights='imagenet';
#  we pretrain locally so this lab runs offline)
Xpre, ypre = make_shapes(3000, seed=99)
tf.random.set_seed(0)
base = keras.Sequential([
    keras.layers.Input(shape=(32,32,3)),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="c1"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="c2"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="c3"),
], name="backbone")
pre = keras.Sequential([base, keras.layers.GlobalAveragePooling2D(),
                        keras.layers.Dense(3, activation="softmax")])
pre.compile(loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
pre.fit(Xpre, ypre, epochs=15, batch_size=64, verbose=0)
print(f"  (pretrained a backbone on {len(Xpre)} related images)")

# --- PHASE 1: freeze the backbone, train the head ---------------------
base.trainable = False                                  # FREEZE
tl = keras.Sequential([base,
                       keras.layers.GlobalAveragePooling2D(),
                       keras.layers.Dense(64, activation="relu"),
                       keras.layers.Dense(3, activation="softmax")])
tl.compile(loss="sparse_categorical_crossentropy",      # MUST RECOMPILE
           optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
n_train = sum(int(np.prod(v.shape)) for v in tl.trainable_variables)
print(f"  phase 1: {n_train:,} of {tl.count_params():,} parameters trainable")
tl.fit(Xtr, ytr, epochs=25, batch_size=32, validation_data=(Xva, yva), verbose=0)
acc_frozen = tl.evaluate(Xte, yte, verbose=0)[1]
print(f"  transfer, frozen backbone    : {acc_frozen:.4f}")

# --- PHASE 2: unfreeze, LOWER the learning rate 100x ------------------
base.trainable = True                                   # UNFREEZE
tl.compile(loss="sparse_categorical_crossentropy",      # RECOMPILE AGAIN
           optimizer=keras.optimizers.Adam(1e-5),       # lr / 100
           metrics=["accuracy"])
n_train = sum(int(np.prod(v.shape)) for v in tl.trainable_variables)
print(f"  phase 2: {n_train:,} parameters trainable at lr=1e-5")
tl.fit(Xtr, ytr, epochs=25, batch_size=32, validation_data=(Xva, yva), verbose=0)
acc_ft = tl.evaluate(Xte, yte, verbose=0)[1]
print(f"  transfer, fine-tuned         : {acc_ft:.4f}")

# --- THE COUNTERFACTUAL: unfreeze immediately at a high lr -----------
base2 = keras.models.clone_model(base)
base2.set_weights(pre.layers[0].get_weights())
base2.trainable = True
bad = keras.Sequential([base2, keras.layers.GlobalAveragePooling2D(),
                        keras.layers.Dense(64, activation="relu"),
                        keras.layers.Dense(3, activation="softmax")])
bad.compile(loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
bad.fit(Xtr, ytr, epochs=50, batch_size=32, verbose=0)
print(f"  unfrozen immediately at 1e-3 : "
      f"{bad.evaluate(Xte, yte, verbose=0)[1]:.4f}   <- random head wrecks it")

print()
print(f"  {'approach':<32}{'test accuracy':>15}")
print(f"  {'from scratch':<32}{acc_scratch:>15.4f}")
print(f"  {'transfer, frozen':<32}{acc_frozen:>15.4f}")
print(f"  {'transfer, fine-tuned':<32}{acc_ft:>15.4f}")

# ============ 4. THE BATCHNORM TRAP ====================================
print()
print("=== frozen BatchNorm layers still update their statistics ===")
bn_model = keras.Sequential([keras.layers.Input(shape=(8,)),
                             keras.layers.Dense(8),
                             keras.layers.BatchNormalization()])
bn = bn_model.layers[1]
before = bn.moving_mean.numpy().copy()
bn_model.trainable = False
bn_model.compile(loss="mse", optimizer="adam")
bn_model.fit(np.random.rand(200, 8).astype("float32") * 10,
             np.random.rand(200, 8).astype("float32"), epochs=3, verbose=0)
after = bn.moving_mean.numpy()
print(f"  moving_mean changed by {np.abs(after-before).max():.6f} "
      f"while the model was 'frozen'")
print(f"  (Keras 3 puts frozen layers in inference mode, so this is usually 0;")
print(f"   in older versions and in some configurations it is not -- always check)")

# ============ 5. FEATURE EXTRACTION ONLY ===============================
print()
print("=== the cheapest option: extract features once, fit a linear model ===")
base.trainable = False
extractor = keras.Sequential([base, keras.layers.GlobalAveragePooling2D()])
Ftr = extractor.predict(Xtr, verbose=0)
Fte = extractor.predict(Xte, verbose=0)
print(f"  features: {Ftr.shape}  (computed ONCE, no backprop through the CNN)")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=2000).fit(Ftr, ytr)
print(f"  logistic regression on frozen features: {clf.score(Fte, yte):.4f}")
print("  for a very small dataset this is often the best AND fastest option")
''',
        key="ch14_transfer",
    )

    keypoints([
        "<code>keras.applications</code> gives ImageNet weights in one line — "
        "always use that architecture's own <code>preprocess_input</code>.",
        "Two phases: freeze → train the head → unfreeze → <b>recompile at "
        "lr/10–100</b> → fine-tune.",
        "Unfreezing before the head is trained lets random-head gradients destroy "
        "the pretrained features.",
        "Watch frozen <b>BatchNormalization</b> layers when fine-tuning a "
        "backbone.",
        "For very small datasets, extract features once and fit a linear model — "
        "cheapest and often best.",
    ])


# ==========================================================================
def s_14_7():
    section("14.7", "Localization and Object Detection")

    lead(
        "Classification says <i>what</i>. Localisation adds <i>where</i>, as a "
        "bounding box. Detection does both for an unknown number of objects at "
        "once — and that last step is what makes it hard."
    )

    sub("Localisation is just regression")

    md(
        "Predict four numbers per image alongside the class. Add a second output "
        "head and a second loss:"
    )

    math(r"""
    \mathcal{L} \;=\;
    \mathcal{L}_{\text{cls}}\bigl(\mathbf{y}, \hat{\mathbf{y}}\bigr)
    \;+\; \lambda \, \mathcal{L}_{\text{box}}\bigl(\mathbf{b}, \hat{\mathbf{b}}\bigr),
    \qquad
    \mathbf{b} = \bigl(x_c, y_c, w, h\bigr)
    """)

    tip(
        "Predict normalised coordinates, and use a sigmoid",
        "Express the box as fractions of the image ($x_c, y_c, w, h \\in [0,1]$) "
        "rather than pixels — then the same head works at any input resolution, "
        "and a sigmoid output enforces the valid range for free. Predicting "
        "$\\sqrt{w}$ and $\\sqrt{h}$ (as YOLO does) makes the loss weight small "
        "and large boxes more evenly.",
    )

    sub("Intersection over Union")

    math(r"""
    \mathrm{IoU}\bigl(A, B\bigr) \;=\;
    \frac{\bigl|A \cap B\bigr|}{\bigl|A \cup B\bigr|}
    \;=\; \frac{\text{area of overlap}}{\text{area of union}}
    """)

    table(
        ["IoU", "Interpretation"],
        [["1.0", "Perfect overlap"],
         ["≥ 0.5", "The standard threshold for 'correct detection'"],
         ["≥ 0.75", "A strict threshold; COCO averages IoU from 0.5 to 0.95"],
         ["0.0", "No overlap at all"]],
    )

    note(
        "IoU is a metric, not a good loss",
        "Its gradient is zero whenever the boxes do not overlap at all — which is "
        "exactly the situation early in training, when the model needs guidance "
        "most. That is why detectors optimise a smooth-L1 or MSE box loss and "
        "<i>report</i> IoU. Modern alternatives (GIoU, DIoU, CIoU) add terms that "
        "stay informative for non-overlapping boxes.",
    )

    sub("Detection — the two families")

    table(
        ["", "Two-stage (R-CNN family)", "One-stage (YOLO, SSD, RetinaNet)"],
        [["How", "Propose regions, then classify each one",
          "One forward pass predicts boxes and classes on a grid"],
         ["Accuracy", "Higher, historically", "Now competitive"],
         ["Speed", "Slower", "<b>Real-time</b> (30–150 FPS)"],
         ["Examples", "R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN",
          "YOLOv1…v8, SSD, RetinaNet, EfficientDet"],
         ["Key difficulty", "The proposal stage is expensive",
          "Extreme class imbalance — most grid cells are background"]],
    )

    sub("YOLO: you only look once")

    md(
        """
YOLO divides the image into an $S \\times S$ grid. Each cell predicts $B$
bounding boxes plus class probabilities, all in **one forward pass**:

* Box coordinates are predicted **relative to the cell**, so they lie in
  $[0, 1]$ and a sigmoid enforces it.
* Each box carries an **objectness** score — the model's confidence that a box
  is there at all, which handles the "how many objects?" problem.
* **Anchor boxes** (priors of different aspect ratios) let one cell predict a
  tall pedestrian and a wide car simultaneously.
        """
    )

    md("The output tensor has shape:")

    math(r"""
    S \times S \times \bigl[\, B \times (4 + 1) + K \,\bigr]
    """)
    where({r"S": "grid size (7 in YOLOv1, 13/26/52 in later versions)",
           r"B": "boxes per cell (2 in v1, 3 anchors per scale in v3+)",
           r"4": "$x, y, w, h$", r"1": "objectness score",
           r"K": "number of classes"})

    sub("Non-max suppression")

    md(
        "A single object typically fires several nearby cells. NMS keeps the "
        "best and deletes its duplicates:"
    )

    md(
        """
1. Drop every box with objectness below a threshold (say 0.5).
2. Take the highest-scoring remaining box; output it.
3. Delete every remaining box whose IoU with it exceeds a threshold (say 0.45).
4. Repeat from 2 until no boxes remain.
        """
    )

    warn(
        "The NMS IoU threshold is a real trade-off",
        "Set it <b>too low</b> and two genuinely distinct overlapping objects — "
        "a crowd of people, a stack of books — get merged into one detection. Set "
        "it <b>too high</b> and every object produces several duplicate boxes. "
        "0.45–0.5 is typical; <b>Soft-NMS</b> decays neighbours' scores instead "
        "of deleting them, which handles crowds much better.",
    )

    anim_header("Non-max suppression removing duplicates")

    rng = np.random.default_rng(7)
    truth = [(0.22, 0.30, 0.20, 0.28), (0.62, 0.55, 0.26, 0.34)]
    boxes, scores = [], []
    for (cx, cy, w, h) in truth:
        for _ in range(6):
            boxes.append((cx + rng.normal(0, .035), cy + rng.normal(0, .035),
                          w * rng.uniform(.82, 1.18), h * rng.uniform(.82, 1.18)))
            scores.append(rng.uniform(.45, .98))
    boxes = np.array(boxes); scores = np.array(scores)

    def to_corners(b):
        cx, cy, w, h = b
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def iou(a, b):
        ax1, ay1, ax2, ay2 = to_corners(a)
        bx1, by1, bx2, by2 = to_corners(b)
        ix = max(0., min(ax2, bx2) - max(ax1, bx1))
        iy = max(0., min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.

    order = list(np.argsort(-scores))
    keep, suppressed, steps = [], [], []
    remaining = [i for i in order if scores[i] >= .5]
    steps.append(("threshold: drop score < 0.5", list(keep), list(suppressed),
                  list(remaining)))
    while remaining:
        best = remaining.pop(0)
        keep.append(best)
        killed = [j for j in remaining if iou(boxes[best], boxes[j]) > .45]
        remaining = [j for j in remaining if j not in killed]
        suppressed.extend(killed)
        steps.append((f"keep box {best} (score {scores[best]:.2f}), "
                      f"suppress {len(killed)} overlapping",
                      list(keep), list(suppressed), list(remaining)))

    def box_trace(idx_list, colr, width, dash=None):
        xs, ys = [], []
        for i in idx_list:
            x1, y1, x2, y2 = to_corners(boxes[i])
            xs += [x1, x2, x2, x1, x1, None]
            ys += [y1, y1, y2, y2, y1, None]
        return go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                          line=dict(color=colr, width=width, dash=dash))

    frames = []
    for title, kp, sup, rem in steps:
        frames.append(go.Frame(name=title[:14], data=[
            box_trace(rem, C["muted"], 1.6, "dot"),
            box_trace(sup, alpha(C["danger"], .45), 1.4, "dot"),
            box_trace(kp, C["success"], 4),
        ], layout=go.Layout(title=title)))

    f = go.Figure(data=[
        box_trace([i for i in range(len(boxes))], C["muted"], 1.6, "dot"),
        box_trace([], C["danger"], 1.4, "dot"),
        box_trace([], C["success"], 4),
    ])
    f.update_layout(height=470, xaxis=dict(range=[0, 1], title="x",
                                           scaleanchor="y"),
                    yaxis=dict(range=[0, 1], title="y"),
                    plot_bgcolor=C["surface_alt"],
                    title=f"{len(boxes)} raw detections for 2 real objects")
    anim.animate(f, frames, duration=nav.anim_ms(1300), slider_prefix="step ")
    figure(f, "Grey dotted = still competing, faded red = suppressed, "
              "green = kept. Twelve raw boxes collapse to two detections.")

    sub("Fully convolutional networks")

    idea(
        "Replacing the Dense head with a 1×1 convolution frees the input size",
        "A Dense layer applied to a flattened $7\\times7\\times512$ map is "
        "<i>mathematically identical</i> to a $7\\times7$ convolution with as many "
        "filters as the Dense layer had units. But the convolution can be applied "
        "to a <b>larger</b> input, producing a grid of outputs instead of one — "
        "which is exactly the sliding-window detector, computed in a single "
        "efficient pass with all the intermediate work shared. This is the FCN "
        "trick, and it underlies both detection and segmentation.",
    )

    code_lab(
        "IoU, a localisation head, and non-max suppression from scratch",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. IoU FROM SCRATCH ======================================
def to_corners(box):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)"""
    cx, cy, w, h = box
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

def iou(a, b):
    ax1, ay1, ax2, ay2 = to_corners(a)
    bx1, by1, bx2, by2 = to_corners(b)
    ix = max(0., min(ax2, bx2) - max(ax1, bx1))
    iy = max(0., min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.

print("=== IoU ===")
gt = (0.5, 0.5, 0.4, 0.4)
print(f"  ground truth box (cx,cy,w,h) = {gt}")
print(f"{'prediction':>28}{'IoU':>8}{'verdict':>22}")
for nm, pred in [("identical",            (0.50, 0.50, 0.40, 0.40)),
                 ("shifted 0.05",         (0.55, 0.50, 0.40, 0.40)),
                 ("shifted 0.10",         (0.60, 0.50, 0.40, 0.40)),
                 ("half the size",        (0.50, 0.50, 0.20, 0.20)),
                 ("twice the size",       (0.50, 0.50, 0.80, 0.80)),
                 ("no overlap",           (0.95, 0.95, 0.10, 0.10))]:
    v = iou(gt, pred)
    verdict = ("correct (>=0.5)" if v >= .5 else
               "strict-correct" if v >= .75 else "MISS")
    print(f"{nm:>28}{v:>8.4f}{verdict:>22}")

# --- why IoU is a bad LOSS -------------------------------------------
print()
print("=== IoU has zero gradient when the boxes do not overlap ===")
print(f"{'x offset':>10}{'IoU':>9}{'d(IoU)/dx (numeric)':>24}")
for dx in [0.0, 0.2, 0.4, 0.41, 0.6, 1.0]:
    v  = iou(gt, (0.5+dx, 0.5, 0.4, 0.4))
    v2 = iou(gt, (0.5+dx+1e-4, 0.5, 0.4, 0.4))
    print(f"{dx:>10.2f}{v:>9.4f}{(v2-v)/1e-4:>24.4f}")
print("  past 0.4 the boxes are disjoint and the gradient is EXACTLY zero")
print("  -> train with smooth-L1 on the coordinates, REPORT IoU")

# ============ 2. A CLASSIFICATION + LOCALISATION MODEL =================
print()
print("=== two heads: class and box ===")
def make_data(n, size=32, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size, 3), dtype="float32")
    y = np.zeros(n, dtype="int32")
    B = np.zeros((n, 4), dtype="float32")
    yy, xx = np.mgrid[0:size, 0:size] / size
    for i in range(n):
        cls = i % 2; y[i] = cls
        img = np.ones((size, size, 3)) * rng.uniform(.1, .3, 3)
        cx, cy = rng.uniform(.25, .75, 2)
        r = rng.uniform(.10, .20)
        if cls == 0:  m = ((xx-cx)**2 + (yy-cy)**2) < r**2
        else:         m = (np.abs(xx-cx) < r) & (np.abs(yy-cy) < r)
        img[m] = rng.uniform(.7, 1.0, 3)
        X[i] = np.clip(img + rng.normal(0, .03, img.shape), 0, 1)
        B[i] = [cx, cy, 2*r, 2*r]                  # NORMALISED coordinates
    return X, y, B

Xtr, ytr, Btr = make_data(2000, seed=0)
Xte, yte, Bte = make_data(500, seed=1)

inp = keras.layers.Input(shape=(32, 32, 3))
z = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
z = keras.layers.MaxPooling2D()(z)
z = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(z)
z = keras.layers.MaxPooling2D()(z)
z = keras.layers.Conv2D(128, 3, activation="relu", padding="same")(z)
z = keras.layers.GlobalAveragePooling2D()(z)
z = keras.layers.Dense(64, activation="relu")(z)
cls_head = keras.layers.Dense(2, activation="softmax", name="class")(z)
box_head = keras.layers.Dense(4, activation="sigmoid", name="box")(z)  # in [0,1]

model = keras.Model(inp, [cls_head, box_head])
model.compile(
    loss=["sparse_categorical_crossentropy", keras.losses.Huber()],
    loss_weights=[1.0, 5.0],                       # lambda
    optimizer=keras.optimizers.Adam(1e-3),
    metrics=[["accuracy"], ["mae"]])
model.fit(Xtr, [ytr, Btr], epochs=30, batch_size=64, verbose=0)

res = model.evaluate(Xte, [yte, Bte], verbose=0, return_dict=True)
acc = [v for k, v in res.items() if "accuracy" in k][0]
mae = [v for k, v in res.items() if "mae" in k][0]
print(f"  class accuracy : {acc:.4f}")
print(f"  box MAE        : {mae:.4f}  (normalised units)")

pred_cls, pred_box = model.predict(Xte, verbose=0)
ious = np.array([iou(Bte[i], pred_box[i]) for i in range(len(Bte))])
print(f"  mean IoU       : {ious.mean():.4f}")
for thr in [0.5, 0.75, 0.9]:
    print(f"    IoU >= {thr}: {np.mean(ious >= thr):.1%} of predictions")

# ============ 3. NON-MAX SUPPRESSION FROM SCRATCH ======================
print()
print("=== non-max suppression ===")
def nms(boxes, scores, score_thr=0.5, iou_thr=0.45):
    idx = [i for i in np.argsort(-scores) if scores[i] >= score_thr]
    keep = []
    while idx:
        best = idx.pop(0)
        keep.append(best)
        idx = [j for j in idx if iou(boxes[best], boxes[j]) <= iou_thr]
    return keep

rng = np.random.default_rng(7)
truth = [(0.22, 0.30, 0.20, 0.28), (0.62, 0.55, 0.26, 0.34)]
raw_boxes, raw_scores = [], []
for (cx, cy, w, h) in truth:
    for _ in range(8):
        raw_boxes.append((cx + rng.normal(0, .03), cy + rng.normal(0, .03),
                          w*rng.uniform(.85,1.15), h*rng.uniform(.85,1.15)))
        raw_scores.append(rng.uniform(.4, .98))
raw_boxes = np.array(raw_boxes); raw_scores = np.array(raw_scores)

print(f"  {len(raw_boxes)} raw detections for {len(truth)} real objects")
print(f"{'iou_thr':>9}{'kept':>7}{'comment':>44}")
for thr in [0.1, 0.3, 0.45, 0.7, 0.95]:
    k = nms(raw_boxes, raw_scores, iou_thr=thr)
    note = ("merges distinct objects" if len(k) < len(truth) else
            "correct" if len(k) == len(truth) else "duplicates remain")
    print(f"{thr:>9.2f}{len(k):>7}{note:>44}")

# --- and TensorFlow's built-in ---------------------------------------
tf_boxes = tf.constant([to_corners(b)[[1,0,3,2]] for b in raw_boxes],
                       dtype=tf.float32)      # TF wants (y1,x1,y2,x2)
sel = tf.image.non_max_suppression(tf_boxes, tf.constant(raw_scores,
                                   dtype=tf.float32),
                                   max_output_size=10,
                                   iou_threshold=0.45, score_threshold=0.5)
print(f"\\n  tf.image.non_max_suppression kept {len(sel)} boxes: {sel.numpy()}")
print(f"  mine kept                        {len(nms(raw_boxes, raw_scores))} "
      f"boxes: {nms(raw_boxes, raw_scores)}")

# ============ 4. THE YOLO OUTPUT TENSOR ================================
print()
print("=== YOLO output shapes ===")
print(f"{'version':<12}{'grid':>10}{'boxes/cell':>13}{'classes':>10}"
      f"{'output tensor':>22}{'values':>12}")
for nm, S, B, K in [("YOLOv1", 7, 2, 20), ("YOLOv2", 13, 5, 20),
                    ("YOLOv3 (P3)", 52, 3, 80), ("YOLOv3 (P5)", 13, 3, 80)]:
    per_cell = B*(4+1) + K if nm == "YOLOv1" else B*(4+1+K)
    print(f"{nm:<12}{f'{S}x{S}':>10}{B:>13}{K:>10}"
          f"{f'{S}x{S}x{per_cell}':>22}{S*S*per_cell:>12,}")

# ============ 5. THE FULLY-CONVOLUTIONAL TRICK =========================
print()
print("=== Dense head vs 1x1 conv head ===")
backbone = keras.Sequential([
    keras.layers.Input(shape=(None, None, 3)),        # ANY size
    keras.layers.Conv2D(32, 3, strides=2, activation="relu", padding="same"),
    keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")])
fcn_head = keras.layers.Conv2D(10, 1, activation="softmax")   # 1x1 conv head
fcn = keras.Sequential([backbone, fcn_head])
print(f"  a 1x1-conv head applied to different input sizes:")
for size in [32, 64, 128]:
    out = fcn(tf.zeros((1, size, size, 3)))
    print(f"    input {size:>3}x{size:<3} -> output {tuple(out.shape)}   "
          f"= a {out.shape[1]}x{out.shape[2]} grid of 10-class predictions")
print("  ONE forward pass gives a whole grid of predictions --")
print("  that is a sliding-window detector with all the work shared.")
''',
        key="ch14_detection",
    )

    keypoints([
        "Localisation is <b>regression on four numbers</b>; use normalised "
        "coordinates and a sigmoid.",
        "<b>IoU</b> = intersection / union; ≥ 0.5 is the standard 'correct' "
        "threshold.",
        "IoU has zero gradient for disjoint boxes — train with smooth-L1, "
        "<i>report</i> IoU.",
        "<b>YOLO</b> predicts $S\\times S\\times[B(4{+}1){+}K]$ in one pass; "
        "objectness handles 'how many objects'.",
        "<b>NMS</b> removes duplicates; its IoU threshold trades duplicates "
        "against merged neighbours.",
    ])


# ==========================================================================
def s_14_8():
    section("14.8", "Semantic Segmentation")

    lead(
        "Classify <b>every pixel</b>. The difficulty is precisely the one pooling "
        "created: a CNN throws away spatial resolution on purpose, and "
        "segmentation needs it back."
    )

    table(
        ["Task", "Output", "Distinguishes instances?"],
        [["<b>Semantic segmentation</b>", "One class label per pixel",
          "❌ two adjacent cars are one 'car' blob"],
         ["<b>Instance segmentation</b>", "A mask per detected object",
          "✅ Mask R-CNN"],
         ["<b>Panoptic segmentation</b>",
          "Both: instances for countable things, semantic for stuff (sky, road)",
          "✅ for things, ❌ for stuff"]],
    )

    sub("The resolution problem")

    derive(
        [("A typical backbone downsamples by 32× — five stride-2 stages. A "
          "$224\\times224$ input becomes a $7\\times7$ feature map.", None),
         ("Upsampling that back naively gives a segmentation mask where each "
          "predicted 'pixel' covers a $32\\times32$ block of the original:",
          r"\text{effective resolution} \;=\; \frac{224}{32} = 7 \text{ across the "
          r"whole image}"),
         ("Every boundary is therefore accurate to $\\pm 16$ pixels at best. For "
          "a road-scene mask or a tumour outline, that is useless.", None),
         ("<b>Three fixes, used together in practice:</b>", None),
         ("<b>(1) Transposed convolution</b> — a learned upsampling layer. It "
          "inserts zeros between input pixels and then convolves, so the output "
          "is larger:",
          r"o \;=\; (n - 1)\,s \;+\; f \;-\; 2p"),
         ("<b>(2) Skip connections</b> — add or concatenate the "
          "<i>high-resolution</i> feature maps from early layers into the decoder. "
          "Those maps still know exactly where the edges are; the deep maps know "
          "what the object is. Combining them gives both. This is the U-Net "
          "architecture, and it is the single most important idea here.", None),
         ("<b>(3) Dilated (atrous) convolution</b> — insert gaps of size $r$ "
          "between kernel taps, enlarging the receptive field <i>without</i> "
          "downsampling:",
          r"R_{\text{dilated}} \;=\; r\,(f - 1) + 1")],
        title="Why segmentation needs a decoder",
    )

    sub("Transposed convolution")

    warn(
        "Transposed convolution causes checkerboard artefacts",
        "When the stride does not divide the kernel size evenly, some output "
        "pixels receive contributions from more input pixels than others, "
        "producing a visible checkerboard. Two fixes: make the kernel size a "
        "multiple of the stride (e.g. $4\\times4$ with stride 2), or replace it "
        "entirely with <code>UpSampling2D</code> (nearest-neighbour or bilinear) "
        "followed by an ordinary convolution — which is what most modern "
        "architectures now do.",
    )

    sub("U-Net")

    md(
        """
The dominant segmentation architecture, and it is beautifully simple:

* An **encoder** (contracting path) — a normal CNN, halving resolution and
  doubling channels at each stage.
* A **decoder** (expanding path) — upsampling, halving channels.
* **Skip connections** concatenating each encoder stage into the matching
  decoder stage.

The result is a symmetric U shape. It was designed for biomedical images in 2015
and remains the default for segmentation, depth estimation, image-to-image
translation and the denoising backbone of diffusion models (Chapter 17).
        """
    )

    sub("Metrics")

    math(r"""
    \mathrm{IoU}_c \;=\; \frac{TP_c}{TP_c + FP_c + FN_c}
    \qquad\qquad
    \mathrm{mIoU} \;=\; \frac{1}{K}\sum_{c=1}^{K}\mathrm{IoU}_c
    """)

    math(r"""
    \mathrm{Dice}_c \;=\; \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c}
    \;=\; \frac{2\,\mathrm{IoU}_c}{1 + \mathrm{IoU}_c}
    """)

    pitfall(
        "Pixel accuracy is meaningless for segmentation",
        "If 92 % of the pixels in a driving scene are road and sky, a model that "
        "labels <i>everything</i> road-or-sky scores 92 % pixel accuracy while "
        "detecting no pedestrians at all. This is §3.3's skewed-class problem at "
        "the pixel level. <b>Report mean IoU or Dice, per class</b>, and look at "
        "the worst class rather than the average.",
    )

    anim_header("The U-Net: contracting, then expanding with skips")

    stages = [
        ("encoder 1", 64, 64, "64ch"), ("encoder 2", 32, 128, "128ch"),
        ("encoder 3", 16, 256, "256ch"), ("bottleneck", 8, 512, "512ch"),
        ("decoder 3", 16, 256, "256ch + skip"), ("decoder 2", 32, 128, "128ch + skip"),
        ("decoder 1", 64, 64, "64ch + skip"), ("output", 64, 3, "K classes"),
    ]
    xs = [0, 1, 2, 3, 4, 5, 6, 7]
    ys = [0, -1, -2, -3, -2, -1, 0, 0]

    frames = []
    for k in range(1, len(stages) + 1):
        shapes, ann = [], []
        for i, (nm, size, ch, lbl) in enumerate(stages):
            on = i < k
            cur = i == k - 1
            h = .16 + .55 * (size / 64)
            col = (C["accent"] if i == 3 else
                   SEQ[0] if i < 3 else
                   C["success"] if i == len(stages) - 1 else SEQ[1])
            shapes.append(go.Scatter(
                x=[xs[i] - .32, xs[i] + .32, xs[i] + .32, xs[i] - .32, xs[i] - .32],
                y=[ys[i] - h/2, ys[i] - h/2, ys[i] + h/2, ys[i] + h/2, ys[i] - h/2],
                fill="toself",
                fillcolor=(alpha(col, .92) if cur else alpha(col, .5) if on
                           else alpha(C["line"], .28)),
                line=dict(color="#fff" if on else C["line"], width=2),
                showlegend=False, hoverinfo="skip"))
            if on:
                ann.append(dict(x=xs[i], y=ys[i] - h/2 - .32,
                                text=f"{size}×{size}<br>{lbl}", showarrow=False,
                                font=dict(size=8.5, color=C["ink_soft"])))
        # skip connections appear as the decoder is built
        for enc, dec in [(0, 6), (1, 5), (2, 4)]:
            if k > dec:
                shapes.append(go.Scatter(
                    x=[xs[enc] + .32, xs[dec] - .32],
                    y=[ys[enc] + .25, ys[dec] + .25], mode="lines",
                    line=dict(color=C["danger"], width=3, dash="dot"),
                    showlegend=False, hoverinfo="skip"))
        frames.append(go.Frame(name=str(k), data=shapes,
                               layout=go.Layout(annotations=ann,
                                                title=f"{stages[k-1][0]} — "
                                                      f"{stages[k-1][1]}×"
                                                      f"{stages[k-1][1]}, "
                                                      f"{stages[k-1][3]}")))

    f = go.Figure(data=frames[0].data)
    f.update_layout(height=420, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.7, 7.7]),
                    yaxis=dict(visible=False, range=[-4.2, 1.3]),
                    annotations=list(frames[0].layout.annotations),
                    title=f"{stages[0][0]} — {stages[0][1]}×{stages[0][1]}")
    anim.animate(f, frames, duration=nav.anim_ms(1000), slider_prefix="stage ")
    figure(f, "Red dotted lines are the skip connections: high-resolution detail "
              "from the encoder, injected into the decoder.")

    code_lab(
        "Transposed convolution, dilation, a U-Net, and mIoU",
        '''import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42); np.random.seed(42)

# ============ 1. THE RESOLUTION PROBLEM ================================
print("=== a 32x downsampling backbone ===")
print(f"{'input':>8}{'after /32':>12}{'pixels per output cell':>26}")
for size in [224, 512, 1024]:
    print(f"{size:>8}{size//32:>12}{f'{32}x{32}':>26}")
print("  every boundary is accurate to +/-16 pixels at best")

# ============ 2. TRANSPOSED CONVOLUTION ================================
print()
print("=== Conv2DTranspose: learned upsampling ===")
x = tf.zeros((1, 8, 8, 64))
print(f"{'layer':<44}{'output':>18}{'params':>10}")
for nm, layer in [
        ("Conv2DTranspose(32, 3, strides=2, 'same')",
         keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same")),
        ("Conv2DTranspose(32, 4, strides=2, 'same')",
         keras.layers.Conv2DTranspose(32, 4, strides=2, padding="same")),
        ("Conv2DTranspose(32, 2, strides=2, 'valid')",
         keras.layers.Conv2DTranspose(32, 2, strides=2, padding="valid")),
        ("UpSampling2D(2) + Conv2D(32, 3)",
         keras.Sequential([keras.layers.UpSampling2D(2),
                           keras.layers.Conv2D(32, 3, padding="same")]))]:
    out = layer(x)
    print(f"{nm:<44}{str(tuple(out.shape)):>18}{layer.count_params():>10}")

# --- the checkerboard artefact ---------------------------------------
print()
print("=== checkerboard artefacts ===")
ones = tf.ones((1, 6, 6, 1))
for f, s in [(3, 2), (4, 2), (5, 2), (6, 3)]:
    L = keras.layers.Conv2DTranspose(1, f, strides=s, padding="same",
                                     kernel_initializer="ones", use_bias=False)
    out = L(ones).numpy()[0, :, :, 0]
    uneven = out.std() / max(out.mean(), 1e-9)
    flag = "CHECKERBOARD" if f % s else "clean (f divisible by s)"
    print(f"  kernel {f}, stride {s}: output variation {uneven:>6.3f}   {flag}")
print("  use a kernel size divisible by the stride, or UpSampling2D + Conv2D")

# ============ 3. DILATED CONVOLUTION ===================================
print()
print("=== dilated (atrous) convolution: receptive field without downsampling ===")
print(f"{'dilation':>10}{'kernel':>9}{'receptive field':>18}{'params':>9}"
      f"{'output size':>14}")
x = tf.zeros((1, 32, 32, 16))
for r in [1, 2, 4, 8]:
    L = keras.layers.Conv2D(16, 3, dilation_rate=r, padding="same")
    out = L(x)
    print(f"{r:>10}{'3x3':>9}{r*(3-1)+1:>18}{L.count_params():>9}"
          f"{str(tuple(out.shape[1:3])):>14}")
print("  the receptive field grows while the OUTPUT SIZE stays the same")
print("  and the parameter count never changes")

# ============ 4. A U-NET ===============================================
print()
print("=== U-Net ===")
def unet(input_shape=(64, 64, 3), n_classes=3, base=16):
    inp = keras.layers.Input(shape=input_shape)

    def conv_block(x, filters):
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return keras.layers.BatchNormalization()(x)

    # --- ENCODER, keeping each stage for the skips --------------------
    skips = []
    z = inp
    for i in range(3):
        z = conv_block(z, base * 2**i)
        skips.append(z)                          # SAVE for the decoder
        z = keras.layers.MaxPooling2D(2)(z)

    # --- BOTTLENECK ---------------------------------------------------
    z = conv_block(z, base * 8)

    # --- DECODER, concatenating the skips -----------------------------
    for i in reversed(range(3)):
        z = keras.layers.Conv2DTranspose(base * 2**i, 2, strides=2,
                                         padding="same")(z)
        z = keras.layers.Concatenate()([z, skips[i]])     # THE SKIP
        z = conv_block(z, base * 2**i)

    out = keras.layers.Conv2D(n_classes, 1, activation="softmax")(z)
    return keras.Model(inp, out)

model = unet()
print(f"  input (64,64,3) -> output {tuple(model.output.shape[1:])}")
print(f"  {model.count_params():,} parameters, {len(model.layers)} layers")
n_concat = sum(isinstance(l, keras.layers.Concatenate) for l in model.layers)
print(f"  {n_concat} skip connections")

# --- what happens WITHOUT the skips ----------------------------------
def no_skip_unet(input_shape=(64, 64, 3), n_classes=3, base=16):
    inp = keras.layers.Input(shape=input_shape)
    z = inp
    for i in range(3):
        z = keras.layers.Conv2D(base*2**i, 3, padding="same", activation="relu")(z)
        z = keras.layers.MaxPooling2D(2)(z)
    z = keras.layers.Conv2D(base*8, 3, padding="same", activation="relu")(z)
    for i in reversed(range(3)):
        z = keras.layers.Conv2DTranspose(base*2**i, 2, strides=2,
                                         padding="same", activation="relu")(z)
    return keras.Model(inp, keras.layers.Conv2D(n_classes, 1,
                                                activation="softmax")(z))

# ============ 5. TRAIN AND MEASURE mIoU ================================
print()
print("=== does the skip connection matter? ===")
def make_seg(n, size=64, seed=0):
    """3 classes: background(0), disc(1), bar(2). Per-pixel labels."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size, 3), dtype="float32")
    Y = np.zeros((n, size, size), dtype="int32")
    yy, xx = np.mgrid[0:size, 0:size] / size
    for i in range(n):
        img = np.ones((size, size, 3)) * rng.uniform(.1, .3, 3)
        lab = np.zeros((size, size), dtype="int32")
        cx, cy = rng.uniform(.25, .6, 2)
        disc = ((xx-cx)**2 + (yy-cy)**2) < .012
        img[disc] = rng.uniform(.7, 1., 3); lab[disc] = 1
        by, bx = rng.uniform(.55, .85), rng.uniform(.2, .8)
        bar = (np.abs(yy-by) < .045) & (np.abs(xx-bx) < .25)
        img[bar] = rng.uniform(.6, .9, 3); lab[bar] = 2
        X[i] = np.clip(img + rng.normal(0, .03, img.shape), 0, 1)
        Y[i] = lab
    return X, Y

Xtr, Ytr = make_seg(400, seed=0)
Xte, Yte = make_seg(150, seed=1)
print(f"  {len(Xtr)} training images, per-pixel labels, 3 classes")
print(f"  class balance: " +
      "  ".join(f"{c}={np.mean(Ytr==c):.1%}" for c in range(3)))

def mean_iou(y_true, y_pred, K=3):
    ious = []
    for c in range(K):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        ious.append(tp / max(tp + fp + fn, 1))
    return np.array(ious)

print()
print(f"{'architecture':<22}{'params':>10}{'pixel acc':>12}"
      f"{'IoU bg':>9}{'IoU disc':>10}{'IoU bar':>9}{'mIoU':>8}")
for nm, builder in [("U-Net (with skips)", unet),
                    ("no skip connections", no_skip_unet)]:
    tf.random.set_seed(0)
    m = builder()
    m.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    m.fit(Xtr, Ytr, epochs=8, batch_size=16, verbose=0)
    pred = m.predict(Xte, verbose=0).argmax(-1)
    ious = mean_iou(Yte, pred)
    acc = np.mean(pred == Yte)
    print(f"{nm:<22}{m.count_params():>10,}{acc:>12.4f}"
          f"{ious[0]:>9.4f}{ious[1]:>10.4f}{ious[2]:>9.4f}{ious.mean():>8.4f}")

# ============ 6. WHY PIXEL ACCURACY LIES ===============================
print()
print("=== a model that predicts ONLY background ===")
lazy = np.zeros_like(Yte)
ious = mean_iou(Yte, lazy)
print(f"  pixel accuracy : {np.mean(lazy == Yte):.4f}   <- looks great")
print(f"  per-class IoU  : bg {ious[0]:.4f}  disc {ious[1]:.4f}  "
      f"bar {ious[2]:.4f}")
print(f"  mean IoU       : {ious.mean():.4f}   <- tells the truth")
print()
print("  Dice = 2*IoU/(1+IoU):  " +
      "  ".join(f"{2*v/(1+v):.4f}" for v in ious))
print("  ALWAYS report per-class IoU or Dice, never pixel accuracy alone.")

# ============ 7. VISUALISE ============================================
import plotly.graph_objects as go
from plotly.subplots import make_subplots
tf.random.set_seed(0)
best = unet()
best.compile(loss="sparse_categorical_crossentropy",
             optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
best.fit(Xtr, Ytr, epochs=8, batch_size=16, verbose=0)
pred = best.predict(Xte[:5], verbose=0).argmax(-1)
fig = make_subplots(rows=3, cols=5,
                    row_titles=["image", "ground truth", "prediction"])
for j in range(5):
    fig.add_trace(go.Image(z=(Xte[j]*255).astype(np.uint8)), 1, j+1)
    fig.add_trace(go.Heatmap(z=Yte[j][::-1], zmin=0, zmax=2, colorscale=PARULA,
                             showscale=False), 2, j+1)
    fig.add_trace(go.Heatmap(z=pred[j][::-1], zmin=0, zmax=2, colorscale=PARULA,
                             showscale=False), 3, j+1)
fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
fig.update_layout(height=460, title="U-Net segmentation")
''',
        key="ch14_segmentation",
    )

    keypoints([
        "Segmentation must recover the resolution that pooling destroyed.",
        "<b>Transposed convolution</b> upsamples with learned weights: "
        "$o = (n-1)s + f - 2p$. Beware checkerboard artefacts.",
        "<b>Skip connections</b> inject high-resolution encoder features into the "
        "decoder — the U-Net idea, and the one that matters most.",
        "<b>Dilated convolution</b> grows the receptive field without "
        "downsampling and without extra parameters.",
        "Report <b>per-class IoU or Dice</b>; pixel accuracy is meaningless on "
        "imbalanced scenes.",
    ])


# ==========================================================================
def s_14_9():
    section("14.9", "Exercises & Chapter Review")

    lead("Eleven exercises. Numbers 9–11 are substantial projects.")

    exercise(
        1, "What are the advantages of a CNN over a fully connected DNN for image "
        "classification?",
        "**Far fewer parameters.** Consecutive layers are only partially "
        "connected, and the network reuses the same weights at every position — a "
        "$3\\times3\\times3\\times64$ layer has 1 792 parameters where a "
        "comparable dense layer has 150 million (§14.1). This speeds training, "
        "reduces overfitting risk, and requires far less training data.\n\n"
        "**Reusable features.** When a CNN learns a kernel that detects a feature "
        "in one location, it can detect it **anywhere**. A dense DNN must learn "
        "the feature separately at every position — and can only recognise it in "
        "the positions it happened to see during training.\n\n"
        "**A useful spatial prior.** CNNs have prior knowledge built in about how "
        "pixels are organised: adjacent pixels are related. The lab in §14.1 "
        "shows this concretely — shuffle the pixels with a fixed permutation and "
        "the dense model is unaffected while the CNN collapses.\n\n"
        "**Any input size.** With a GAP or convolutional head, the same network "
        "handles any resolution; a dense head requires one fixed size.")

    exercise(
        2, "Consider a CNN composed of three convolutional layers, each with "
        "3 × 3 kernels, a stride of 2, and 'same' padding. The lowest layer "
        "outputs 100 feature maps, the middle one outputs 200, and the top one "
        "outputs 400. The input images are RGB images of 200 × 300 pixels. What "
        "is the total number of parameters in the CNN? If we are using 32-bit "
        "floats, at least how much RAM will this network require when making a "
        "prediction for a single instance? What about when training on a "
        "mini-batch of 50 images?",
        "**Parameters.**\n\n"
        "* Layer 1: $3 \\times 3 \\times 3 \\times 100 + 100 = 2\\,800$\n"
        "* Layer 2: $3 \\times 3 \\times 100 \\times 200 + 200 = 180\\,200$\n"
        "* Layer 3: $3 \\times 3 \\times 200 \\times 400 + 400 = 720\\,400$\n"
        "* **Total: 903 400 parameters** ≈ 3.6 MB in float32.\n\n"
        "**Feature-map sizes** (stride 2, 'same' padding, so each dimension is "
        "halved and rounded up):\n\n"
        "* Layer 1: $100 \\times 150 \\times 100 = 1\\,500\\,000$ values = 6 MB\n"
        "* Layer 2: $50 \\times 75 \\times 200 = 750\\,000$ values = 3 MB\n"
        "* Layer 3: $25 \\times 38 \\times 400 = 380\\,000$ values = 1.52 MB\n\n"
        "**One prediction.** Only two consecutive layers need to be in memory at "
        "once, so you need the larger pair — layers 1 and 2 = 9 MB — plus the "
        "parameters (3.6 MB) and the input image "
        "($200 \\times 300 \\times 3 \\times 4$ = 0.7 MB). **About 13.3 MB**, "
        "and less if the framework frees each layer as soon as the next is "
        "computed.\n\n"
        "**Training on 50 images.** Backpropagation needs **every** layer's "
        "activations kept for the backward pass, so:\n\n"
        "$50 \\times (6 + 3 + 1.52)\\text{ MB} = 526\\text{ MB}$ of activations, "
        "plus $50 \\times 0.7 = 35$ MB of inputs, plus 3.6 MB of parameters, plus "
        "3.6 MB of gradients. **About 568 MB** at minimum — and considerably more "
        "in practice, since frameworks keep extra buffers and optimiser state.")

    exercise(
        3, "If your GPU runs out of memory while training a CNN, what are five "
        "things you could try to solve the problem?",
        "**(1) Reduce the mini-batch size.** Activation memory is linear in the "
        "batch size (§14.2), so this is the most direct lever.\n\n"
        "**(2) Reduce dimensionality faster**, using a larger stride in one or "
        "more layers, or removing a few layers. A stride-2 layer quarters the "
        "activation memory of everything downstream.\n\n"
        "**(3) Use 16-bit floats** instead of 32-bit — mixed-precision training "
        "halves the activation memory and is usually *faster* on modern GPUs.\n\n"
        "**(4) Distribute the CNN across multiple devices** (Chapter 19).\n\n"
        "**(5) Use gradient checkpointing** — discard most activations during the "
        "forward pass and recompute them during the backward pass, trading about "
        "30 % extra time for a large memory saving.\n\n"
        "A sixth: reduce the input image resolution, if the task tolerates it.")

    exercise(
        4, "Why would you want to add a max pooling layer rather than a "
        "convolutional layer with the same stride?",
        "A max pooling layer has **no parameters at all**, whereas a "
        "convolutional layer has quite a few. Both downsample, but max pooling "
        "does it for free.\n\n"
        "Max pooling also provides a degree of **invariance to small "
        "translations** — the output is unchanged if the strongest activation "
        "moves anywhere within its window (§14.3) — and it introduces **no "
        "additional risk of overfitting** because it has nothing to learn.\n\n"
        "The counter-argument, and the reason modern architectures often prefer "
        "strided convolutions: a strided conv **learns** what to keep, whereas "
        "max pooling discards information according to a fixed rule. When the "
        "task needs precise localisation, that fixed rule is a liability.")

    exercise(
        5, "When would you want to add a local response normalization layer?",
        "An LRN layer makes the neurons that most strongly activate **inhibit "
        "neurons at the same location in neighbouring feature maps**, which "
        "encourages different feature maps to **specialise** and pushes them "
        "apart, forcing them to explore a wider range of features.\n\n"
        "It was used in AlexNet and is typically placed in the lower layers, after "
        "the ReLU. Its practical role has been almost entirely taken over by "
        "**batch normalisation**, which is more effective and easier to tune — LRN "
        "is essentially of historical interest now.")

    exercise(
        6, "Can you name the main innovations in AlexNet, as compared to LeNet-5? "
        "What about the main innovations in GoogLeNet, ResNet, SENet, Xception, "
        "and EfficientNet?",
        "**AlexNet vs LeNet-5:** much larger and deeper; it stacked convolutional "
        "layers **directly on top of each other** instead of stacking a pooling "
        "layer on top of each convolutional layer. It also introduced **ReLU** "
        "activations, **dropout**, **data augmentation**, local response "
        "normalisation, and training on GPUs.\n\n"
        "**GoogLeNet:** the **inception module**, which makes it possible to have "
        "a much deeper net than previous architectures with fewer parameters. It "
        "also introduced the $1\\times1$ bottleneck and replaced the giant dense "
        "head with **global average pooling**.\n\n"
        "**ResNet:** **skip connections**, which make it possible to go well "
        "beyond 100 layers. They are arguably the single most important "
        "architectural idea of the decade, and they now appear in Transformers as "
        "well.\n\n"
        "**SENet:** the **SE block** — a small two-layer network added after every "
        "inception module or residual unit, which recalibrates the relative "
        "importance of the feature maps. Roughly 0.5 % extra parameters for a "
        "consistent accuracy gain.\n\n"
        "**Xception:** **depthwise separable convolutions**, which look at spatial "
        "patterns and depthwise patterns separately, costing about 1/9 as much as "
        "a normal $3\\times3$ convolution.\n\n"
        "**EfficientNet:** the **compound scaling method** — scaling depth, width "
        "and resolution together by a principled ratio rather than one at a time — "
        "which finds a far better accuracy/compute trade-off.")

    exercise(
        7, "What is a fully convolutional network? How can you convert a dense "
        "layer into a convolutional layer?",
        "**A fully convolutional network** contains **only convolutional and "
        "pooling layers** — no dense layers. It can therefore process images of "
        "**any size** (as long as they are at least as large as the receptive "
        "field), and it outputs a **spatial grid** of predictions rather than a "
        "single one.\n\n"
        "FCNs are used for object detection (each grid cell predicts objects near "
        "it) and semantic segmentation (each cell predicts a class).\n\n"
        "**Converting a dense layer:** replace the lowest dense layer with a "
        "convolutional layer having a **kernel size equal to the input feature-map "
        "size**, `padding='valid'`, and **as many filters as the dense layer had "
        "units**. Higher dense layers become $1\\times1$ convolutions with the "
        "same filter count.\n\n"
        "You can then **copy the dense layers' weights directly** into the "
        "convolutional layers, after reshaping. The result computes exactly the "
        "same function on a fixed-size input — but now also works on larger ones, "
        "producing a grid.")

    exercise(
        8, "What is the main technical difficulty of semantic segmentation?",
        "A CNN **loses spatial resolution** as the signal flows through it, "
        "because of the layers with strides greater than 1. By the output layer, a "
        "typical backbone has downsampled by 32×, so a $224\\times224$ input has "
        "become a $7\\times7$ grid.\n\n"
        "So the network knows roughly *where* an object is, but not *precisely*. "
        "Some architectures recover the resolution with **transposed convolution** "
        "layers, and — critically — **skip connections** that reinject the "
        "high-resolution feature maps from the early layers, which still know "
        "exactly where the edges are.\n\n"
        "Modern approaches also use **dilated convolutions**, which enlarge the "
        "receptive field without downsampling at all.")

    exercise(
        9, "Build your own CNN from scratch and try to achieve the highest "
        "possible accuracy on MNIST.",
        "A CNN that reaches **99.4 %+** on MNIST:\n\n"
        "* Two `Conv2D(32, 3)` layers, then max pool, then two `Conv2D(64, 3)`, "
        "then max pool.\n"
        "* Batch normalisation after each convolution.\n"
        "* Dropout 0.25 after each pooling stage, 0.5 before the output.\n"
        "* `GlobalAveragePooling2D` or a small Dense head.\n"
        "* Adam with a `ReduceLROnPlateau` schedule, and `EarlyStopping`.\n"
        "* Light augmentation: `RandomRotation(0.05)`, "
        "`RandomZoom(0.1)`, `RandomTranslation(0.1, 0.1)`. **No horizontal flip** "
        "— a mirrored digit is not the same digit (§13.7).\n\n"
        "Above about 99.5 % you are fighting label noise: MNIST contains a "
        "handful of genuinely ambiguous or mislabelled images. Ensembling several "
        "models with different seeds is the standard way to squeeze out the last "
        "0.1 %.",
        code='''model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomTranslation(0.1, 0.1),

    keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
    keras.layers.BatchNormalization(), keras.layers.Activation("relu"),
    keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
    keras.layers.BatchNormalization(), keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(), keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
    keras.layers.BatchNormalization(), keras.layers.Activation("relu"),
    keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
    keras.layers.BatchNormalization(), keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(), keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
          callbacks=[keras.callbacks.EarlyStopping(patience=15,
                                                   restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)])''')

    exercise(
        10, "Use transfer learning for large image classification, going through "
        "these steps: (a) create a training set containing at least 100 images per "
        "class — for example, classify your own pictures based on the location, or "
        "use an existing dataset such as `tf_flowers`; (b) split it into a "
        "training set, a validation set, and a test set; (c) build the input "
        "pipeline, apply the appropriate preprocessing operations, and optionally "
        "add data augmentation; (d) fine-tune a pretrained model on this dataset.",
        "The `tf_flowers` dataset (3 670 images, 5 classes) is the standard "
        "target and this is the canonical recipe:\n\n"
        "1. `tfds.load('tf_flowers', as_supervised=True, "
        "split=['train[:75%]', 'train[75%:90%]', 'train[90%:]'])`.\n\n"
        "2. **Resize to 224×224** and apply the backbone's own "
        "`preprocess_input` — this is the step most people get wrong (§14.6).\n\n"
        "3. Augment with `RandomFlip('horizontal')` (valid for flowers), "
        "`RandomRotation(0.2)`, `RandomZoom(0.2)`. Augment **only** the training "
        "split.\n\n"
        "4. `Xception(weights='imagenet', include_top=False)` + "
        "`GlobalAveragePooling2D` + `Dense(5, softmax)`.\n\n"
        "5. **Phase 1**: freeze the base, train the head for ~3 epochs at "
        "`lr=0.2` with SGD (or `1e-3` with Adam). Reaches roughly **88 %**.\n\n"
        "6. **Phase 2**: unfreeze, **recompile at `lr=0.01`** (SGD) or `1e-5` "
        "(Adam), train ~10 more epochs. Reaches **95 %+**.\n\n"
        "Two things worth measuring: how much worse it gets if you unfreeze "
        "immediately, and how much worse it gets with the wrong "
        "`preprocess_input`. Both failures are silent.",
        code='''import tensorflow_datasets as tfds

(train_set_raw, valid_set_raw, test_set_raw), info = tfds.load(
    "tf_flowers", split=["train[:75%]", "train[75%:90%]", "train[90%:]"],
    as_supervised=True, with_info=True)
n_classes = info.features["label"].num_classes

preprocess = keras.Sequential([
    keras.layers.Resizing(224, 224, crop_to_aspect_ratio=True),
    keras.layers.Lambda(keras.applications.xception.preprocess_input),
])
augment = keras.Sequential([
    keras.layers.RandomFlip(mode="horizontal", seed=42),
    keras.layers.RandomRotation(factor=0.05, seed=42),
    keras.layers.RandomContrast(factor=0.2, seed=42),
])

def prepare(ds, training, batch=32):
    ds = ds.map(lambda x, y: (preprocess(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()                                   # BEFORE augmentation
    if training:
        ds = ds.shuffle(1000, seed=42).repeat()
    ds = ds.batch(batch)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

base = keras.applications.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base.input, outputs=output)

# PHASE 1
for layer in base.layers:
    layer.trainable = False
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
              metrics=["accuracy"])
model.fit(prepare(train_set_raw, True), epochs=3, steps_per_epoch=86,
          validation_data=prepare(valid_set_raw, False))

# PHASE 2 -- unfreeze and DROP the learning rate
for layer in base.layers:
    layer.trainable = True
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              metrics=["accuracy"])           # RECOMPILE
model.fit(prepare(train_set_raw, True), epochs=10, steps_per_epoch=86,
          validation_data=prepare(valid_set_raw, False))''')

    exercise(
        11, "Go through TensorFlow's Style Transfer tutorial. It is a fun way to "
        "generate art using deep learning.",
        "Neural style transfer (Gatys et al., 2015) is a beautiful demonstration "
        "of what a pretrained CNN's intermediate layers actually encode, and it "
        "requires **no training at all** — you optimise the *image*, not the "
        "weights.\n\n"
        "The mechanism:\n\n"
        "* **Content loss** — the MSE between the *deep* feature maps of the "
        "generated image and the content image. Deep layers encode *what* is "
        "there, so matching them preserves the objects and layout.\n\n"
        "* **Style loss** — the MSE between the **Gram matrices** "
        "$G_{ij}^{[l]} = \\sum_k F_{ik}^{[l]} F_{jk}^{[l]}$ of the *shallow* "
        "feature maps. The Gram matrix records which features **co-occur**, "
        "discarding *where* they occur — which is exactly what 'style' means: "
        "brushstroke textures and colour combinations, independent of layout.\n\n"
        "* **Total loss** $= \\alpha \\mathcal{L}_{\\text{content}} + "
        "\\beta \\mathcal{L}_{\\text{style}}$, minimised by gradient descent on "
        "the **pixels** of the generated image.\n\n"
        "The insight worth taking away: the same pretrained network encodes both "
        "content and style, at different depths, and the Gram matrix is the "
        "operation that separates them. That the ratio $\\beta/\\alpha$ smoothly "
        "trades one for the other is a striking demonstration of how "
        "interpretable those intermediate representations are.",
        code='''# the Gram matrix -- the whole trick
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    shape = tf.shape(input_tensor)
    num_locations = tf.cast(shape[1] * shape[2], tf.float32)
    return result / num_locations          # normalise by the spatial extent

content_layers = ['block5_conv2']          # DEEP  -> what is in the image
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                'block4_conv1', 'block5_conv1']   # SHALLOW -> how it looks

# optimise the IMAGE, not the weights
image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))''')

    rule()

    sub("The chapter as a decision table")

    table(
        ["Your task", "Architecture", "Head", "Key metric"],
        [["Image classification", "Pretrained backbone, fine-tuned",
          "GAP + Dense(K, softmax)", "Top-1 / top-5 accuracy"],
         ["Classification + one box", "Backbone + two heads",
          "Dense(K) and Dense(4, sigmoid)", "Accuracy and IoU"],
         ["Object detection", "YOLO / RetinaNet / Faster R-CNN",
          "Grid of boxes + objectness + classes", "mAP@[.5:.95]"],
         ["Semantic segmentation", "U-Net or DeepLab",
          "Conv2D(K, 1, softmax) at full resolution", "mean IoU / Dice"],
         ["Instance segmentation", "Mask R-CNN", "Box head + mask head",
          "mask mAP"],
         ["Tiny dataset (< 1 000)", "Frozen backbone",
          "Logistic regression on extracted features", "Accuracy"],
         ["Mobile / edge", "MobileNetV3", "GAP + Dense", "Accuracy per ms"]],
    )

    keypoints([
        "Convolution buys a <b>spatial prior</b> and translation equivariance, not "
        "just fewer parameters.",
        "Stacked $3\\times3$ layers beat one large kernel; $1\\times1$ bottlenecks "
        "make everything affordable.",
        "<b>Skip connections</b> are the single most important idea — they unlock "
        "depth, and they unlock segmentation.",
        "<b>Always start from a pretrained backbone</b>, and always use its own "
        "<code>preprocess_input</code>.",
        "Detection needs NMS; segmentation needs a decoder with skips and per-class "
        "IoU.",
    ], title="Chapter 14 in five lines")

    refs([
        ("LeCun et al. — *Gradient-Based Learning Applied to Document "
         "Recognition* (LeNet-5)", "https://doi.org/10.1109/5.726791"),
        ("Krizhevsky, Sutskever & Hinton — *ImageNet Classification with Deep "
         "CNNs* (AlexNet)", "NeurIPS 2012"),
        ("Simonyan & Zisserman — *Very Deep Convolutional Networks* (VGG)",
         "https://arxiv.org/abs/1409.1556"),
        ("Szegedy et al. — *Going Deeper with Convolutions* (GoogLeNet)",
         "https://doi.org/10.1109/CVPR.2015.7298594"),
        ("He et al. — *Deep Residual Learning for Image Recognition* (ResNet)",
         "https://doi.org/10.1109/CVPR.2016.90"),
        ("Chollet, F. — *Xception: Deep Learning with Depthwise Separable "
         "Convolutions*", "https://arxiv.org/abs/1610.02357"),
        ("Hu, Shen & Sun — *Squeeze-and-Excitation Networks*",
         "https://arxiv.org/abs/1709.01507"),
        ("Redmon et al. — *You Only Look Once* (YOLO)",
         "https://arxiv.org/abs/1506.02640"),
        ("Ronneberger, Fischer & Brox — *U-Net: Convolutional Networks for "
         "Biomedical Image Segmentation*", "https://arxiv.org/abs/1505.04597"),
        ("Tan & Le — *EfficientNet: Rethinking Model Scaling*",
         "https://arxiv.org/abs/1905.11946"),
    ])


# ==========================================================================
SECTIONS = [
    ("14.1", "Convolutional Layers", s_14_1),
    ("14.2", "Filters, Feature Maps, Memory", s_14_2),
    ("14.3", "Pooling Layers", s_14_3),
    ("14.4", "CNN Architectures", s_14_4),
    ("14.5", "ResNet, Xception, SENet", s_14_5),
    ("14.6", "Pretrained Models & Transfer", s_14_6),
    ("14.7", "Localization & Detection", s_14_7),
    ("14.8", "Semantic Segmentation", s_14_8),
    ("14.9", "Exercises & Review", s_14_9),
]

nav.render_chapter(CH, SECTIONS)
