"""Environment and setup."""

from __future__ import annotations

import importlib
import os
import platform
import sys
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import nav
from core.lecture import (codenote, hero, idea, keypoints, lead, md, note,
                          pitfall, refs, rule, section, sub, table, tip, warn)
from core.palette import C, SEQ, alpha
from core.theme import PLOTLY_CONFIG, inject

inject()
CH = "setup"

hero(
    kicker="Start here",
    title="Environment & setup",
    blurb=(
        "What this machine actually has, what the labs need, and how to install "
        "the pieces that are missing. Every check below runs live against the "
        "interpreter serving this page."
    ),
    chips=["live probe", "install commands", "GPU check",
           "reproducibility notes"],
)


REQUIRED = [
    ("streamlit", "streamlit", "The platform itself", "core"),
    ("numpy", "numpy", "Arrays, everywhere", "core"),
    ("pandas", "pandas", "Tabular data", "core"),
    ("plotly", "plotly", "Every figure in the platform", "core"),
    ("scikit-learn", "sklearn", "Chapters 1–9", "core"),
    ("scipy", "scipy", "Statistics, sparse matrices", "core"),
    ("matplotlib", "matplotlib", "A few chapter-14 visualisations", "core"),
    ("tensorflow", "tensorflow", "Chapters 10–19", "deep"),
    ("keras", "keras", "The modelling API (ships with TF)", "deep"),
    ("statsmodels", "statsmodels", "SARIMA in §15.3 (optional)", "optional"),
    ("sympy", "sympy", "Symbolic differentiation demo in §B.1 (optional)",
     "optional"),
    ("transformers", "transformers", "Hugging Face models in §16.9 (optional)",
     "optional"),
    ("tensorboard", "tensorboard", "The §10.7 callback (optional)", "optional"),
    ("keras-tuner", "keras_tuner", "Hyperparameter search in §10.8 (optional)",
     "optional"),
    ("tensorflow-datasets", "tensorflow_datasets",
     "A few dataset loaders (optional)", "optional"),
    ("gymnasium", "gymnasium",
     "Not needed — §18 ships its own environments", "optional"),
]


def _probe(mod: str):
    try:
        m = importlib.import_module(mod)
        return True, getattr(m, "__version__", "?")
    except Exception as e:
        return False, type(e).__name__


def render_status():
    section("1", "What this machine has")

    lead(
        "Probed live, right now, against the interpreter serving this page. "
        "Anything marked <b>core</b> is required; the platform degrades "
        "gracefully without the optional ones."
    )

    rows = []
    for name, mod, why, tier in REQUIRED:
        ok, ver = _probe(mod)
        rows.append((name, mod, why, tier, ok, ver))

    core_ok = all(r[4] for r in rows if r[3] == "core")
    deep_ok = all(r[4] for r in rows if r[3] == "deep")
    n_opt = sum(1 for r in rows if r[3] == "optional" and r[4])
    n_opt_total = sum(1 for r in rows if r[3] == "optional")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Core packages", "OK" if core_ok else "MISSING",
              "chapters 1–9" if core_ok else "the platform will not run")
    m2.metric("Deep-learning stack", "OK" if deep_ok else "missing",
              "chapters 10–19" if deep_ok else "those labs will not run")
    m3.metric("Optional", f"{n_opt} / {n_opt_total}",
              "each one degrades gracefully")
    m4.metric("Python", ".".join(map(str, sys.version_info[:3])),
              platform.system())

    if not core_ok:
        st.error("A core package is missing — the platform cannot work "
                 "properly. Install it with the command below.", icon="🚫")
    elif not deep_ok:
        st.warning("TensorFlow is unavailable, so every lab in chapters 10–19 "
                   "will report an ImportError. All the lectures, animations "
                   "and Part I labs still work.", icon="⚠️")
    else:
        st.success("Everything required is present. Every lab in the platform "
                   "will run.", icon="✅")

    table(
        ["Package", "Import", "Needed for", "Tier", "Status"],
        [[f"<code>{r[0]}</code>", f"<code>{r[1]}</code>", r[2],
          {"core": "<b>core</b>", "deep": "<b>deep</b>",
           "optional": "optional"}[r[3]],
          (f"✅ <b>{r[5]}</b>" if r[4] else f"❌ not installed")]
         for r in rows],
    )

    missing = [r[0] for r in rows if not r[4]]
    if missing:
        sub("Install what is missing")
        st.code("pip install " + " ".join(missing), language="bash")

    rule()

    sub("The interpreter")

    table(
        ["", "Value"],
        [["Python", sys.version.split()[0]],
         ["Executable", f"<code>{sys.executable}</code>"],
         ["Platform", f"{platform.system()} {platform.release()}"],
         ["Machine", platform.machine()],
         ["Processor", platform.processor() or "unknown"],
         ["CPU count", str(os.cpu_count())],
         ["Working directory", f"<code>{os.getcwd()}</code>"]],
    )


def render_hardware():
    section("2", "Hardware and TensorFlow")

    lead(
        "What TensorFlow can see, and what that means for how long the labs "
        "take."
    )

    ok, ver = _probe("tensorflow")
    if not ok:
        st.error(f"TensorFlow is not importable ({ver}). Install it with "
                 "the command below, then reload this page.", icon="🚫")
        st.code("pip install tensorflow", language="bash")
        return

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TensorFlow", tf.__version__)
    try:
        import keras
        m2.metric("Keras", keras.__version__)
    except Exception:
        m2.metric("Keras", "bundled")
    m3.metric("GPUs visible", len(gpus))
    m4.metric("Built with CUDA", "yes" if tf.test.is_built_with_cuda()
              else "no")

    if gpus:
        st.success(f"{len(gpus)} GPU(s) available — the deep-learning labs "
                   "will be several times faster than the timings quoted in "
                   "the text.", icon="🚀")
        table(["Device", "Details"],
              [[g.name, str(tf.config.experimental.get_device_details(g))]
               for g in gpus])
        codenote(
            "Set memory growth before creating any tensor",
            "By default TensorFlow allocates essentially the whole GPU on first "
            "use, so a second process on this machine will fail with OOM even "
            "on an idle card. See §19.4.",
        )
        st.code(
            "import tensorflow as tf\n"
            "for gpu in tf.config.list_physical_devices('GPU'):\n"
            "    tf.config.experimental.set_memory_growth(gpu, True)",
            language="python")
    else:
        st.info(
            "**No GPU — that is fine.** Every lab in this platform was written "
            "and timed on CPU. The heaviest ones (the DCGAN in §17.8, the "
            "diffusion model in §17.9, the DQN variants in §18.7) take a few "
            "minutes; everything else is well under a minute.", icon="💻")

    rule()

    sub("A quick benchmark")

    if st.button("▶  Run a 10-second benchmark", type="primary",
                 key="bench_run"):
        with st.spinner("Timing matrix multiplication and a training step…"):
            results = []
            for n in [256, 512, 1024, 2048]:
                a = tf.random.normal((n, n))
                tf.matmul(a, a)
                t0 = time.perf_counter()
                for _ in range(5):
                    tf.matmul(a, a)
                dt = (time.perf_counter()-t0)/5
                gflops = 2*n**3/dt/1e9
                results.append((f"matmul {n}×{n}", dt*1000, gflops))

            from tensorflow import keras
            model = keras.Sequential([
                keras.layers.Input(shape=(28, 28, 1)),
                keras.layers.Conv2D(32, 3, activation="relu"),
                keras.layers.MaxPool2D(),
                keras.layers.Flatten(),
                keras.layers.Dense(10, activation="softmax")])
            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer="adam")
            X = np.random.default_rng(0).random((256, 28, 28, 1)).astype("float32")
            y = np.random.default_rng(0).integers(0, 10, 256)
            model.fit(X, y, epochs=1, batch_size=32, verbose=0)
            t0 = time.perf_counter()
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            dt = (time.perf_counter()-t0)/3
            results.append(("one CNN epoch, 256 images", dt*1000, np.nan))

        table(["Operation", "Time", "GFLOP/s"],
              [[r[0], f"{r[1]:.2f} ms",
                (f"{r[2]:.1f}" if np.isfinite(r[2]) else "—")]
               for r in results])

        peak = max(r[2] for r in results if np.isfinite(r[2]))
        if peak > 500:
            st.success(f"Peak {peak:.0f} GFLOP/s — that looks like a GPU. Every "
                       "lab will be fast.", icon="🚀")
        elif peak > 50:
            st.info(f"Peak {peak:.0f} GFLOP/s — a reasonable multi-core CPU. "
                    "The heaviest labs take a few minutes.", icon="💻")
        else:
            st.warning(f"Peak {peak:.0f} GFLOP/s — the deep-learning labs will "
                       "be slow. Consider reducing the epoch counts in the "
                       "editable code before running them.", icon="🐌")
    else:
        st.caption("Multiplies a few matrices and trains a small CNN for three "
                   "epochs, so you know what to expect from the heavier labs.")


def render_install():
    section("3", "Installing from scratch")

    lead(
        "If you are setting this up on another machine, this is the whole "
        "procedure."
    )

    sub("1 · A clean environment")

    md("Always use a virtual environment. Mixing this into a system Python is "
       "how you end up with a broken system Python.")

    st.code(
        "# with venv (built in)\n"
        "python -m venv .venv\n"
        "source .venv/bin/activate        # Windows: .venv\\Scripts\\activate\n"
        "python -m pip install --upgrade pip\n\n"
        "# or with conda\n"
        "conda create -n mlplat python=3.11 -y\n"
        "conda activate mlplat",
        language="bash")

    sub("2 · The packages")

    st.code(
        "pip install streamlit numpy pandas plotly scikit-learn scipy \\\n"
        "            matplotlib tensorflow\n\n"
        "# optional, each one only affects a single lab\n"
        "pip install statsmodels sympy transformers tensorboard \\\n"
        "            keras-tuner tensorflow-datasets",
        language="bash")

    st.download_button(
        "Download requirements.txt",
        "\n".join([
            "# ML Platform — core requirements",
            "streamlit>=1.40",
            "numpy>=1.26",
            "pandas>=2.0",
            "plotly>=5.20",
            "scikit-learn>=1.4",
            "scipy>=1.11",
            "matplotlib>=3.8",
            "",
            "# chapters 10-19",
            "tensorflow>=2.16",
            "",
            "# optional — each affects one lab only",
            "statsmodels>=0.14      # SARIMA, section 15.3",
            "sympy>=1.12            # symbolic demo, section B.1",
            "transformers>=4.40     # Hugging Face, section 16.9",
            "tensorboard>=2.16      # callback, section 10.7",
            "keras-tuner>=1.4       # search, section 10.8",
            "tensorflow-datasets    # a few loaders",
            "",
        ]), file_name="requirements.txt", mime="text/plain")

    sub("3 · Run it")

    st.code("streamlit run app.py", language="bash")

    md("It opens on `http://localhost:8501`. The sidebar carries the whole "
       "syllabus; each chapter page has its own sub-section picker.")

    rule()

    sub("GPU support")

    table(
        ["Platform", "Command", "Note"],
        [["Linux, NVIDIA", "<code>pip install tensorflow[and-cuda]</code>",
          "Pulls the matching CUDA and cuDNN wheels — much easier than "
          "installing them yourself"],
         ["Windows, NVIDIA", "Use <b>WSL2</b>, then the Linux command",
          "Native Windows GPU support was dropped after TF 2.10"],
         ["macOS, Apple silicon",
          "<code>pip install tensorflow tensorflow-metal</code>",
          "Metal gives a real speed-up on M-series chips"],
         ["Anything else", "<code>pip install tensorflow</code>",
          "CPU only — which is fine here"]],
    )

    warn(
        "Do not fight the CUDA install",
        "Version mismatches between TensorFlow, CUDA and cuDNN are the single "
        "most common setup failure, and diagnosing them is a genuine waste of a "
        "day. Two escapes: <code>pip install tensorflow[and-cuda]</code>, which "
        "pins everything for you, or Google Colab, which is free and already "
        "configured. <b>Nothing in this platform requires a GPU.</b>",
    )

    sub("Running it in Colab")

    st.code(
        "!pip install streamlit -q\n"
        "!npm install -g localtunnel\n"
        "!streamlit run app.py &>/dev/null &\n"
        "!npx localtunnel --port 8501",
        language="bash")

    st.caption("Clunky but workable. The labs themselves paste straight into "
               "Colab cells, which is usually the better route.")


def render_notes():
    section("4", "Reproducibility and troubleshooting")

    lead(
        "Why your numbers will differ slightly from the text's, and what to do "
        "when something breaks."
    )

    sub("Why results vary")

    table(
        ["Source", "Effect", "Fix"],
        [["<b>Random seeds</b>", "Different splits, different initialisation",
          "<code>np.random.default_rng(42)</code>, "
          "<code>tf.random.set_seed(42)</code>"],
         ["<b>Thread scheduling</b>",
          "Floating-point addition is not associative, so a parallel reduction "
          "gives a slightly different sum each run",
          "<code>tf.config.threading.set_inter_op_parallelism_threads(1)</code> "
          "— slow, and rarely worth it"],
         ["<b>GPU non-determinism</b>",
          "Atomic adds complete in a different order",
          "<code>tf.config.experimental.enable_op_determinism()</code>"],
         ["<b>Library versions</b>",
          "Defaults change; algorithms are improved",
          "Pin them in <code>requirements.txt</code>"],
         ["<b>Hardware</b>", "Different vectorisation, different rounding",
          "Nothing — accept it"]],
    )

    pitfall(
        "Exact reproducibility is achievable, and usually not worth its cost",
        "Full determinism means single-threaded reductions and disabled "
        "autotuning, which can cost <b>2–10× throughput</b>. What you actually "
        "need is <b>statistical</b> reproducibility: run 5 seeds and report the "
        "median with a spread. If your conclusion changes between seeds, "
        "exact reproducibility would not have saved it — the conclusion was "
        "never there (§18.9).",
    )

    sub("Common failures")

    table(
        ["Symptom", "Cause", "Fix"],
        [["<code>ModuleNotFoundError</code> in a lab",
          "An optional package is missing",
          "The lab prints the install command; every optional import is "
          "guarded"],
         ["A lab is very slow",
          "No GPU, and the lab trains a real model",
          "Edit the source in the lab's own editor — reduce epochs or dataset "
          "size, then Run"],
         ["<code>ResourceExhaustedError</code>",
          "GPU out of memory",
          "Smaller batch, or enable memory growth (§19.4)"],
         ["A gradient is <code>None</code>",
          "A numpy call broke the tape",
          "Keep everything in <code>tf.*</code> (§B.5)"],
         ["<code>NaN</code> loss",
          "Learning rate too high, or $\\log(0)$",
          "Lower the rate; use <code>from_logits=True</code>"],
         ["A figure does not appear",
          "The lab assigns to something other than <code>fig</code>",
          "Name it <code>fig</code>, <code>fig1</code>… or call "
          "<code>st.plotly_chart</code>"],
         ["Session state looks stale",
          "Streamlit caches datasets and lab namespaces",
          "<b>Clear vars</b> in the lab, or press R to rerun"]],
    )

    sub("Using the code labs")

    table(
        ["Control", "Does"],
        [["<b>▶ Run</b>", "Executes the buffer in this lab's namespace"],
         ["<b>✏️ Edit the source</b>",
          "Opens an editor — change anything and press Run"],
         ["<b>↺ Restore</b>", "Puts the original code back and clears the "
                              "namespace"],
         ["<b>🧹 Clear vars</b>",
          "Empties the namespace but keeps your edits"]],
    )

    idea(
        "The labs are meant to be edited",
        "Every one is a starting point, not a demonstration. The fastest way to "
        "understand a result is to break it: change the regularisation to zero, "
        "set the learning rate to 10, remove the target network, delete the "
        "<code>+=</code> in the autodiff engine. The namespace persists between "
        "runs, so you can poke at a fitted model after the fact — "
        "<code>model.get_weights()</code>, "
        "<code>np.corrcoef(...)</code>, anything.",
    )

    rule()

    sub("What is not installed here")

    missing = [(n, w) for n, m, w, t in REQUIRED if not _probe(m)[0]]
    if missing:
        table(["Package", "Which lab needs it"],
              [[f"<code>{n}</code>", w] for n, w in missing])
        st.caption("Every one of these is optional. The lab that uses it "
                   "detects its absence, prints the install command, and "
                   "continues with a pure-Python fallback where one exists.")
    else:
        st.success("Nothing is missing — every optional package is present "
                   "too.", icon="✅")

    refs([
        ("TensorFlow install guide", "https://www.tensorflow.org/install"),
        ("Keras 3 documentation", "https://keras.io/"),
        ("scikit-learn install guide",
         "https://scikit-learn.org/stable/install.html"),
        ("Streamlit documentation", "https://docs.streamlit.io/"),
        ("Plotly Python reference", "https://plotly.com/python/"),
        ("Google Colab", "https://colab.research.google.com/"),
    ])


SECTIONS = [
    ("1", "What this machine has", render_status),
    ("2", "Hardware & TensorFlow", render_hardware),
    ("3", "Installing from scratch", render_install),
    ("4", "Reproducibility & troubleshooting", render_notes),
]

nav.render_chapter(CH, SECTIONS, sidebar_title="Setup")
