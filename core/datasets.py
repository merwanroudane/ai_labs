"""
core.datasets
=============
Cached dataset factories shared by every chapter and by the AI Lab.

Rules followed here:
  * everything is `@st.cache_data` so a dataset is generated / downloaded once
  * every remote dataset has a *synthetic fallback* so the platform still works
    with no internet connection
  * shapes and column names are stable so code labs can rely on them
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

RNG_SEED = 42


# --------------------------------------------------------------------------
# 1-D / 2-D synthetic teaching sets
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def linear_1d(n: int = 100, slope: float = 3.0, intercept: float = 4.0,
              noise: float = 1.0, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    X = 2 * rng.random((n, 1))
    y = intercept + slope * X[:, 0] + rng.normal(0, noise, n)
    return X, y


@st.cache_data(show_spinner=False)
def quadratic_1d(n: int = 120, seed: int = RNG_SEED, noise: float = 1.0):
    rng = np.random.default_rng(seed)
    X = 6 * rng.random((n, 1)) - 3
    y = 0.5 * X[:, 0] ** 2 + X[:, 0] + 2 + rng.normal(0, noise, n)
    return X, y


@st.cache_data(show_spinner=False)
def sine_1d(n: int = 150, seed: int = RNG_SEED, noise: float = 0.18):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3, 3, n)).reshape(-1, 1)
    y = np.sin(1.6 * X[:, 0]) + 0.35 * X[:, 0] + rng.normal(0, noise, n)
    return X, y


@st.cache_data(show_spinner=False)
def moons(n: int = 400, noise: float = 0.22, seed: int = RNG_SEED):
    from sklearn.datasets import make_moons
    return make_moons(n_samples=n, noise=noise, random_state=seed)


@st.cache_data(show_spinner=False)
def circles(n: int = 400, noise: float = 0.12, factor: float = 0.45,
            seed: int = RNG_SEED):
    from sklearn.datasets import make_circles
    return make_circles(n_samples=n, noise=noise, factor=factor,
                        random_state=seed)


@st.cache_data(show_spinner=False)
def blobs(n: int = 500, centers: int = 4, std: float = 0.9,
          seed: int = RNG_SEED):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, centers=centers, cluster_std=std,
                      random_state=seed, n_features=2)
    return X, y


@st.cache_data(show_spinner=False)
def anisotropic_blobs(n: int = 500, seed: int = RNG_SEED):
    X, y = blobs(n=n, centers=3, std=0.8, seed=seed)
    T = np.array([[0.62, -0.63], [-0.41, 0.85]])
    return X @ T, y


@st.cache_data(show_spinner=False)
def swiss_roll(n: int = 1200, noise: float = 0.15, seed: int = RNG_SEED):
    from sklearn.datasets import make_swiss_roll
    X, t = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    return X, t


@st.cache_data(show_spinner=False)
def s_curve(n: int = 1200, noise: float = 0.06, seed: int = RNG_SEED):
    from sklearn.datasets import make_s_curve
    X, t = make_s_curve(n_samples=n, noise=noise, random_state=seed)
    return X, t


# --------------------------------------------------------------------------
# Classic tabular sets
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def iris():
    from sklearn.datasets import load_iris
    d = load_iris(as_frame=True)
    return d.data, d.target, list(d.target_names), list(d.feature_names)


@st.cache_data(show_spinner=False)
def digits():
    """8x8 handwritten digits — a fast, always-offline stand-in for MNIST."""
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data, d.target, d.images


@st.cache_data(show_spinner=False)
def breast_cancer():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    return d.data, d.target, list(d.target_names)


@st.cache_data(show_spinner=False)
def wine():
    from sklearn.datasets import load_wine
    d = load_wine(as_frame=True)
    return d.data, d.target, list(d.target_names)


@st.cache_data(show_spinner=True)
def housing() -> pd.DataFrame:
    """
    California housing.  Tries scikit-learn's fetcher; if there is no network
    connection, falls back to a statistically similar synthetic frame so the
    end-to-end project chapter still runs.
    """
    try:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        df = d.frame.copy()
        df = df.rename(columns={"MedHouseVal": "median_house_value"})
        df["median_house_value"] *= 100_000
        df["ocean_proximity"] = pd.cut(
            df["Longitude"], bins=[-125, -122.5, -120, -118, -114],
            labels=["NEAR OCEAN", "<1H OCEAN", "INLAND", "NEAR BAY"],
        ).astype(str)
        df["_synthetic"] = False
        return df
    except Exception:
        rng = np.random.default_rng(RNG_SEED)
        n = 4000
        inc = np.clip(rng.lognormal(1.1, 0.45, n), 0.5, 15)
        age = rng.integers(1, 52, n).astype(float)
        rooms = 3 + 0.55 * inc + rng.normal(0, 0.6, n)
        lat = rng.uniform(32.5, 41.9, n)
        lon = rng.uniform(-124.3, -114.3, n)
        val = (35_000 + 42_000 * inc + 900 * (52 - age)
               + 8_000 * rooms - 2_500 * (lat - 34) ** 2
               + rng.normal(0, 35_000, n))
        df = pd.DataFrame({
            "MedInc": inc, "HouseAge": age, "AveRooms": rooms,
            "AveBedrms": rooms * rng.uniform(0.16, 0.24, n),
            "Population": rng.lognormal(6.9, 0.6, n),
            "AveOccup": rng.lognormal(1.0, 0.3, n),
            "Latitude": lat, "Longitude": lon,
            "median_house_value": np.clip(val, 25_000, 500_001),
        })
        df["ocean_proximity"] = pd.cut(
            df["Longitude"], bins=[-125, -122.5, -120, -118, -114],
            labels=["NEAR OCEAN", "<1H OCEAN", "INLAND", "NEAR BAY"],
        ).astype(str)
        df["_synthetic"] = True
        return df


# --------------------------------------------------------------------------
# Image sets for the deep-learning half
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=True)
def fashion_mnist(n_train: int = 6000, n_test: int = 1000):
    """
    Fashion-MNIST via Keras.  Falls back to the 8x8 digits set (upsampled)
    when TensorFlow or the download is unavailable.
    """
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    try:
        import tensorflow as tf
        (Xtr, ytr), (Xte, yte) = tf.keras.datasets.fashion_mnist.load_data()
        return (Xtr[:n_train] / 255.0, ytr[:n_train],
                Xte[:n_test] / 255.0, yte[:n_test], labels, True)
    except Exception:
        X, y, images = digits()
        k = min(n_train, len(images))
        return (images[:k] / 16.0, y[:k], images[k:k + n_test] / 16.0,
                y[k:k + n_test], [str(i) for i in range(10)], False)


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def ridership(n_days: int = 730, seed: int = RNG_SEED) -> pd.DataFrame:
    """Synthetic daily public-transport ridership with weekly seasonality,
    an annual cycle, a slow trend and holiday dips — used in the RNN chapter."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    weekly = 1.0 + 0.42 * np.sin(2 * np.pi * (t % 7) / 7 - 1.1)
    weekend = np.where((t % 7) >= 5, 0.55, 1.0)
    annual = 1.0 + 0.13 * np.sin(2 * np.pi * t / 365.25 - 0.6)
    trend = 1.0 + 0.00035 * t
    base = 1_050_000
    noise = rng.normal(0, 0.035, n_days)
    y = base * weekly * weekend * annual * trend * (1 + noise)
    holidays = rng.choice(n_days, size=n_days // 60, replace=False)
    y[holidays] *= 0.62
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "rail": y.round(0)}).set_index("date")


@st.cache_data(show_spinner=False)
def ar_series(n: int = 400, phi: float = 0.85, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

TINY_CORPUS = (
    "to be or not to be that is the question\n"
    "whether tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune\n"
    "or to take arms against a sea of troubles\n"
    "and by opposing end them to die to sleep\n"
    "no more and by a sleep to say we end\n"
    "the heartache and the thousand natural shocks\n"
    "that flesh is heir to tis a consummation\n"
    "devoutly to be wished to die to sleep\n"
    "to sleep perchance to dream ay theres the rub\n"
)

SENTIMENT_SAMPLES = [
    ("this movie was an absolute delight from start to finish", 1),
    ("a masterclass in tension and pacing i loved every minute", 1),
    ("gorgeous photography and a script with real teeth", 1),
    ("i have never been so bored in a cinema", 0),
    ("the plot collapses in the second act and never recovers", 0),
    ("wooden acting and dialogue that made me wince", 0),
    ("warm funny and quietly devastating", 1),
    ("two hours i will never get back", 0),
]


@st.cache_data(show_spinner=False)
def char_corpus(n_chars: int = 60_000, seed: int = RNG_SEED) -> str:
    """A synthetic English-like corpus with real character-level structure.

    Built from a small vocabulary and a hand-written grammar so a char-RNN has
    something genuinely learnable (spelling, spacing, punctuation, line breaks)
    without shipping any copyrighted text.
    """
    rng = np.random.default_rng(seed)
    subj = ["the king", "the queen", "my lord", "the soldier", "a stranger",
            "the messenger", "our captain", "the young prince", "she", "he",
            "the old man", "the servant", "no one", "every man"]
    verb = ["speaks", "waits", "returns", "answers", "listens", "remembers",
            "forgets", "departs", "arrives", "kneels", "watches", "trembles"]
    obj = ["in the garden", "before the gate", "at the hour of dawn",
           "without a word", "with a heavy heart", "beneath the tower",
           "upon the road", "in silence", "among the trees", "by the river"]
    conj = ["and", "but", "yet", "for", "though", "while", "because"]
    tail = [".", ".", ".", ",", "!", "?", ";"]

    out, total = [], 0
    while total < n_chars:
        n_clause = int(rng.integers(1, 4))
        parts = []
        for k in range(n_clause):
            s = f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)}"
            parts.append(s if k == 0 else f"{rng.choice(conj)} {s}")
        line = " ".join(parts) + str(rng.choice(tail))
        out.append(line)
        total += len(line) + 1
        if rng.random() < 0.18:
            out.append("")
            total += 1
    text = "\n".join(out)
    return (TINY_CORPUS + "\n" + text)[:n_chars]


@st.cache_data(show_spinner=False)
def sentiment_corpus(n: int = 4000, seed: int = RNG_SEED):
    """Synthetic labelled reviews: (texts, labels).

    Sentiment is carried by adjective/verb choice, and every review also
    contains neutral filler, so a bag-of-words model does well but word ORDER
    (negation, contrast) is what separates a good model from a great one.
    """
    rng = np.random.default_rng(seed)
    pos_adj = ["brilliant", "gorgeous", "tender", "hilarious", "gripping",
               "luminous", "assured", "generous", "thrilling", "moving"]
    neg_adj = ["dull", "clumsy", "tedious", "shrill", "lifeless", "muddled",
               "smug", "witless", "leaden", "interminable"]
    noun = ["script", "cast", "photography", "score", "pacing", "ending",
            "premise", "direction", "dialogue", "performance"]
    filler = ["i saw it on friday", "the cinema was half empty",
              "it runs about two hours", "there is an interval",
              "the sequel is announced", "the book came first",
              "my friend came along", "we sat near the front"]
    frames_pos = ["the {n} is {a}", "a truly {a} {n}", "i found the {n} {a}",
                  "what a {a} {n}", "the {n} was {a} throughout"]
    frames_neg = frames_pos
    contrast = ["but the {n} is {a}", "though the {n} is {a}",
                "even so the {n} is {a}"]

    texts, labels = [], []
    for _ in range(n):
        y = int(rng.integers(0, 2))
        main_adj = rng.choice(pos_adj if y else neg_adj)
        other_adj = rng.choice(neg_adj if y else pos_adj)
        parts = [rng.choice(frames_pos if y else frames_neg).format(
            n=rng.choice(noun), a=main_adj)]
        if rng.random() < 0.45:                       # a contrast clause
            parts.append(rng.choice(contrast).format(
                n=rng.choice(noun), a=other_adj))
        if rng.random() < 0.6:
            parts.insert(int(rng.integers(0, len(parts) + 1)),
                         rng.choice(filler))
        if rng.random() < 0.25:                       # negation flips it
            parts = ["it is not true that"] + parts
            y = 1 - y
        texts.append(" ".join(parts))
        labels.append(y)
    return texts, np.array(labels, dtype="int32")


_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


@st.cache_data(show_spinner=False)
def date_pairs(n: int = 12_000, seed: int = RNG_SEED):
    """(human-readable date, ISO date) pairs — the classic seq2seq benchmark.

    'March 3, 2019' -> '2019-03-03'.  The alignment between input and output
    positions is non-monotonic, which makes it perfect for visualising what an
    attention mechanism learns.
    """
    rng = np.random.default_rng(seed)
    src, tgt = [], []
    for _ in range(n):
        y = int(rng.integers(1950, 2051))
        m = int(rng.integers(1, 13))
        d = int(rng.integers(1, 29))
        style = int(rng.integers(0, 4))
        if style == 0:
            s = f"{_MONTHS[m-1]} {d}, {y}"
        elif style == 1:
            s = f"{d} {_MONTHS[m-1]} {y}"
        elif style == 2:
            s = f"{_MONTHS[m-1][:3]} {d} {y}"
        else:
            s = f"{d:02d}/{m:02d}/{y}"
        src.append(s)
        tgt.append(f"{y:04d}-{m:02d}-{d:02d}")
    return src, tgt
