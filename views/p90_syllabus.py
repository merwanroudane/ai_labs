"""Full syllabus, searchable across every chapter and sub-section."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import nav
from core.lecture import (figure, hero, idea, keypoints, lead, md, note, rule,
                          section, sub, table, tip, warn)
from core.palette import C, SEQ, alpha
from core.theme import PLOTLY_CONFIG, inject

inject()
CH = "syllabus"

hero(
    kicker="Start here · developed by Dr Merwan Roudane",
    title="Full syllabus & search",
    blurb=(
        "Every chapter, every sub-section, every code lab — 190 sub-sections "
        "across 19 chapters and 5 appendices, each with its animation and its "
        "runnable lab. Search it, filter it, or read the routes."
    ),
    chips=["19 chapters", "190 sub-sections", "~180 code labs",
           "searchable"],
)


# --------------------------------------------------------------------------
# (chapter number, title, url, icon, part, [(sub-number, sub-title, keywords)])
# --------------------------------------------------------------------------

SYLLABUS = [
    ("1", "The Machine Learning Landscape", "ch01", "🌍", "I", [
        ("1.1", "What Is Machine Learning?",
         "definition arthur samuel tom mitchell experience task performance"),
        ("1.2", "Types of ML Systems",
         "supervised unsupervised semi-supervised self-supervised "
         "reinforcement batch online instance-based model-based"),
        ("1.3", "Supervised vs Unsupervised",
         "classification regression clustering labels"),
        ("1.4", "Batch and Online Learning",
         "out-of-core learning rate incremental"),
        ("1.5", "Instance vs Model-Based Learning",
         "generalisation knn similarity"),
        ("1.6", "Main Challenges — Data",
         "insufficient quantity nonrepresentative sampling bias poor quality "
         "irrelevant features unreasonable effectiveness"),
        ("1.7", "Main Challenges — Algorithms",
         "overfitting underfitting regularisation"),
        ("1.8", "Testing and Validating",
         "test set validation set holdout no free lunch"),
        ("1.9", "Exercises", "exercises answers"),
    ]),
    ("2", "End-to-End Machine Learning Project", "ch02", "🏗️", "I", [
        ("2.1", "Frame the Problem",
         "business objective pipeline current solution"),
        ("2.2", "Get the Data", "download structure legal storage"),
        ("2.3", "Select a Performance Measure",
         "rmse mae norms l1 l2 minkowski"),
        ("2.4", "Explore and Visualise",
         "histograms correlation scatter matrix anscombe"),
        ("2.5", "Prepare the Data",
         "imputation scaling one-hot pipeline column transformer leakage"),
        ("2.6", "Select and Train a Model",
         "cross-validation shortlist underfitting"),
        ("2.7", "Fine-Tune the Model",
         "grid search random search ensemble test set"),
        ("2.8", "Present and Launch", "documentation monitoring"),
        ("2.9", "Exercises", "exercises answers"),
    ]),
    ("3", "Classification", "ch03", "🎯", "I", [
        ("3.1", "MNIST and the Binary Task", "digits binary classifier sgd"),
        ("3.2", "Measuring Accuracy Is Not Enough",
         "imbalanced dummy classifier base rate"),
        ("3.3", "Confusion Matrix", "true positive false negative"),
        ("3.4", "Precision and Recall", "f1 harmonic mean tradeoff threshold"),
        ("3.5", "The ROC Curve", "auc mann-whitney tpr fpr pr curve"),
        ("3.6", "Multiclass Classification", "ovr ovo softmax"),
        ("3.7", "Error Analysis", "confusion matrix normalisation"),
        ("3.8", "Multilabel and Multioutput", "knn chain classifier"),
        ("3.9", "Exercises", "exercises answers"),
    ]),
    ("4", "Training Models", "ch04", "📉", "I", [
        ("4.1", "Linear Regression and the Loss", "mse likelihood"),
        ("4.2", "The Normal Equation", "closed form complexity"),
        ("4.3", "SVD and the Pseudoinverse", "moore-penrose singular"),
        ("4.4", "Bias, Variance and Learning Curves",
         "decomposition overfitting underfitting"),
        ("4.5", "Gradient Descent", "convergence condition number step size"),
        ("4.6", "Stochastic and Mini-Batch GD", "schedule noise escape"),
        ("4.7", "Polynomial Regression", "features degree overfitting"),
        ("4.8", "Regularised Models — Ridge",
         "l2 tikhonov shrinkage gaussian prior"),
        ("4.9", "Lasso, Elastic Net and Early Stopping",
         "l1 sparsity subgradient laplace prior"),
    ]),
    ("5", "Support Vector Machines", "ch05", "🛡️", "I", [
        ("5.1", "Linear SVM Classification", "margin large margin"),
        ("5.2", "Soft Margin and C", "slack hinge loss"),
        ("5.3", "Nonlinear SVM", "polynomial features similarity"),
        ("5.4", "The Dual Problem", "lagrangian kkt complementary slackness"),
        ("5.5", "Kernels", "rbf polynomial mercer gram matrix"),
        ("5.6", "SVM Regression", "epsilon insensitive tube"),
        ("5.7", "Under the Hood — SMO", "sequential minimal optimisation"),
        ("5.8", "Practical Guidance", "scaling c gamma tuning"),
        ("5.9", "Exercises", "exercises answers"),
    ]),
    ("6", "Decision Trees", "ch06", "🌳", "I", [
        ("6.1", "Training and Visualising", "cart greedy"),
        ("6.2", "Making Predictions", "gini samples value white box"),
        ("6.3", "Gini vs Entropy", "impurity information gain"),
        ("6.4", "Regularisation", "max depth min samples pruning"),
        ("6.5", "Regression Trees", "piecewise constant mse"),
        ("6.6", "Instability and Limitations",
         "rotation sensitivity variance extrapolation"),
        ("6.7", "Cost-Complexity Pruning", "ccp alpha"),
        ("6.8", "Feature Importance", "impurity permutation bias"),
        ("6.9", "Exercises", "exercises answers"),
    ]),
    ("7", "Ensemble Learning and Random Forests", "ch07", "🧩", "I", [
        ("7.1", "Voting Classifiers", "hard soft wisdom of crowds"),
        ("7.2", "Why Ensembles Work", "correlation diversity variance formula"),
        ("7.3", "Bagging and Pasting", "bootstrap out-of-bag"),
        ("7.4", "Random Forests", "feature subsampling extra trees"),
        ("7.5", "AdaBoost", "exponential loss forward stagewise reweighting"),
        ("7.6", "Gradient Boosting",
         "residual functional gradient shrinkage subsample"),
        ("7.7", "Histogram Boosting and Stacking",
         "lightgbm xgboost blending meta learner"),
        ("7.8", "Choosing an Ensemble", "practical guidance"),
        ("7.9", "Exercises", "exercises answers"),
    ]),
    ("8", "Dimensionality Reduction", "ch08", "🗜️", "I", [
        ("8.1", "The Curse of Dimensionality", "distance concentration volume"),
        ("8.2", "Projection and Manifolds", "swiss roll hypothesis"),
        ("8.3", "PCA", "variance svd eckart-young components"),
        ("8.4", "Choosing the Number of Components",
         "explained variance elbow compression"),
        ("8.5", "Randomised and Incremental PCA", "partial fit memmap"),
        ("8.6", "Random Projection", "johnson-lindenstrauss"),
        ("8.7", "LLE and Manifold Learning", "isomap mds t-sne"),
        ("8.8", "Kernel PCA", "rbf preimage"),
        ("8.9", "Exercises", "exercises answers"),
    ]),
    ("9", "Unsupervised Learning", "ch09", "🔮", "I", [
        ("9.1", "Clustering — the Task", "applications segmentation"),
        ("9.2", "k-Means", "lloyd centroid convergence"),
        ("9.3", "Initialisation and k-Means++", "inertia local minima"),
        ("9.4", "Choosing k", "elbow silhouette"),
        ("9.5", "Limits of k-Means", "spherical assumption scaling"),
        ("9.6", "DBSCAN", "density eps min samples noise"),
        ("9.7", "Other Clustering Algorithms",
         "agglomerative spectral birch mean shift"),
        ("9.8", "Gaussian Mixtures and EM", "jensen responsibilities bic aic"),
        ("9.9", "Anomaly Detection and Exercises",
         "novelty isolation forest one-class"),
    ]),
    ("10", "Introduction to Neural Networks with Keras", "ch10", "🧠", "II", [
        ("10.1", "From Biology to the Perceptron", "mcculloch pitts xor"),
        ("10.2", "The Multilayer Perceptron", "hidden layers universal"),
        ("10.3", "Backpropagation", "chain rule reverse mode"),
        ("10.4", "Building a Classifier with Keras", "sequential api mnist"),
        ("10.5", "Regression MLPs", "wide and deep functional"),
        ("10.6", "The Functional API", "multi-input multi-output auxiliary"),
        ("10.7", "Subclassing, Saving and Callbacks",
         "checkpoint early stopping tensorboard"),
        ("10.8", "Hyperparameter Tuning", "keras tuner width depth"),
        ("10.9", "Exercises", "exercises answers"),
    ]),
    ("11", "Training Deep Neural Networks", "ch11", "🏔️", "II", [
        ("11.1", "Vanishing and Exploding Gradients",
         "glorot he initialisation variance"),
        ("11.2", "Activation Functions", "relu leaky elu selu gelu swish"),
        ("11.3", "Batch Normalisation", "internal covariate shift moving average"),
        ("11.4", "Gradient Clipping", "clipnorm clipvalue"),
        ("11.5", "Reusing Pretrained Layers", "transfer learning freezing"),
        ("11.6", "Unsupervised and Auxiliary Pretraining",
         "self-supervised greedy layerwise"),
        ("11.7", "Faster Optimisers",
         "momentum nesterov adagrad rmsprop adam adamw nadam"),
        ("11.8", "Learning Rate Schedules",
         "power exponential piecewise 1cycle warm restart"),
        ("11.9", "Regularisation — Dropout and Max-Norm",
         "monte carlo dropout l1 l2"),
    ]),
    ("12", "Custom Models and Training with TensorFlow", "ch12", "🔧", "II", [
        ("12.1", "TensorFlow Tensors and Operations", "eager numpy interop"),
        ("12.2", "Variables and Other Data Structures",
         "ragged sparse string sets queues"),
        ("12.3", "Custom Loss Functions", "huber serialisation"),
        ("12.4", "Custom Activations, Initialisers, Regularisers",
         "constraints get_config"),
        ("12.5", "Custom Metrics", "streaming stateful reset"),
        ("12.6", "Custom Layers", "build call compute_output_shape"),
        ("12.7", "Custom Models", "subclassing residual block"),
        ("12.8", "Custom Training Loops", "gradienttape apply_gradients"),
        ("12.9", "tf.function and AutoGraph", "tracing rules python side effects"),
    ]),
    ("13", "Loading and Preprocessing Data", "ch13", "🚰", "II", [
        ("13.1", "The tf.data API", "dataset from_tensor_slices chaining"),
        ("13.2", "Chaining Transformations",
         "map filter batch repeat interleave"),
        ("13.3", "Shuffling and Pipeline Order",
         "buffer size prefetch autotune"),
        ("13.4", "The TFRecord Format", "protobuf example compression"),
        ("13.5", "Protocol Buffers and Parsing", "parse_single_example features"),
        ("13.6", "Preprocessing Input Features",
         "normalisation discretisation one-hot"),
        ("13.7", "Keras Preprocessing Layers",
         "adapt training serving skew textvectorization"),
        ("13.8", "Embeddings and Hashing", "collisions cardinality"),
        ("13.9", "TensorFlow Datasets and Exercises", "tfds pipelines"),
    ]),
    ("14", "Deep Computer Vision with CNNs", "ch14", "👁️", "II", [
        ("14.1", "The Visual Cortex and Convolution",
         "receptive field local connectivity"),
        ("14.2", "Convolutional Layers",
         "filters stride padding arithmetic memory"),
        ("14.3", "Pooling Layers", "max average global invariance"),
        ("14.4", "CNN Architectures", "lenet alexnet googlenet vgg inception"),
        ("14.5", "ResNet and Skip Connections", "degradation identity path"),
        ("14.6", "Xception, SENet and Efficient Models",
         "depthwise separable squeeze excitation mobilenet"),
        ("14.7", "Using Pretrained Models and Transfer Learning",
         "keras applications fine-tuning"),
        ("14.8", "Object Detection", "yolo iou nms anchor boxes"),
        ("14.9", "Segmentation and Exercises", "u-net fcn transposed conv"),
    ]),
    ("15", "Processing Sequences Using RNNs and CNNs", "ch15", "🔁", "II", [
        ("15.1", "Recurrent Neurons and Layers",
         "memory cell return_sequences seq2seq seq2vec"),
        ("15.2", "Training RNNs — BPTT",
         "unrolling truncated stateful gradient product"),
        ("15.3", "Baselines and ARMA",
         "naive seasonal differencing stationarity sarima acf"),
        ("15.4", "Preparing the Data",
         "windowing chronological split leakage"),
        ("15.5", "Deep, Multivariate, Multi-Step",
         "recursive direct multi-output exogenous"),
        ("15.6", "Sequence-to-Sequence", "timedistributed encoder decoder"),
        ("15.7", "Handling Long Sequences",
         "layer normalisation recurrent dropout tanh"),
        ("15.8", "LSTM and GRU",
         "forget gate cell state constant error carousel peephole"),
        ("15.9", "1-D Convolutions & Exercises",
         "causal dilated wavenet receptive field"),
    ]),
    ("16", "NLP with RNNs and Attention", "ch16", "💬", "II", [
        ("16.1", "Character RNNs & Text Generation",
         "perplexity temperature top-k nucleus sampling"),
        ("16.2", "Stateful RNNs", "reset_states batch interleaving"),
        ("16.3", "Sentiment, Tokenisation & Masking",
         "bpe subword padding mask_zero embeddings"),
        ("16.4", "Encoder–Decoder for Translation",
         "teacher forcing exposure bias eos sos"),
        ("16.5", "Bidirectional RNNs & Beam Search",
         "length normalisation log probability"),
        ("16.6", "Attention Mechanisms",
         "bahdanau luong scaled dot product query key value alignment"),
        ("16.7", "The Transformer",
         "self-attention multi-head positional encoding pre-norm"),
        ("16.8", "The Transformer Zoo & ViT",
         "bert gpt t5 masked language model chinchilla scaling patches"),
        ("16.9", "Hugging Face & Exercises",
         "pipeline tokenizer fine-tuning lora"),
    ]),
    ("17", "Autoencoders, GANs and Diffusion", "ch17", "🎨", "II", [
        ("17.1", "Autoencoders and PCA",
         "undercomplete bottleneck eckart-young linear"),
        ("17.2", "Stacked AEs & Pretraining",
         "tied weights unsupervised linear probe"),
        ("17.3", "Convolutional & Recurrent AEs",
         "transposed convolution checkerboard repeatvector anomaly"),
        ("17.4", "Denoising Autoencoders",
         "gaussian dropout noise tweedie score inpainting"),
        ("17.5", "Sparse Autoencoders",
         "kl divergence l1 dead units dictionary interpretability"),
        ("17.6", "Variational Autoencoders",
         "elbo jensen reparameterisation posterior collapse beta"),
        ("17.7", "Generative Adversarial Networks",
         "minimax jensen-shannon mode collapse non-saturating wasserstein"),
        ("17.8", "DCGAN & StyleGAN",
         "progressive growing adain mapping network slerp"),
        ("17.9", "Diffusion Models & Exercises",
         "ddpm ddim score matching classifier-free guidance"),
    ]),
    ("18", "Reinforcement Learning", "ch18", "🕹️", "II", [
        ("18.1", "Rewards, Policies and Why It's Hard",
         "credit assignment exploration discount factor shaping"),
        ("18.2", "Neural Policies & Policy Search",
         "cross-entropy method evolution strategies stochastic policy"),
        ("18.3", "Policy Gradients & REINFORCE",
         "log-derivative trick baseline advantage reward-to-go"),
        ("18.4", "MDPs and the Bellman Equations",
         "markov value iteration policy iteration contraction"),
        ("18.5", "TD Learning and Q-Learning",
         "sarsa off-policy epsilon-greedy robbins-monro bootstrapping"),
        ("18.6", "Deep Q-Learning",
         "replay buffer target network deadly triad huber"),
        ("18.7", "Double, Dueling & Prioritised",
         "maximisation bias importance sampling n-step rainbow"),
        ("18.8", "Actor–Critic, PPO & the Landscape",
         "a2c gae clipping trust region sac ddpg td3"),
        ("18.9", "Practice and Exercises",
         "observation normalisation seeds debugging"),
    ]),
    ("19", "Training and Deploying at Scale", "ch19", "🚀", "II", [
        ("19.1", "SavedModel & Serving Signatures",
         "tf serving versioning canary training serving skew"),
        ("19.2", "Latency, Batching & Throughput",
         "grpc rest p99 little's law queueing"),
        ("19.3", "Mobile, Embedded & Quantisation",
         "tflite int8 per-channel distillation calibration"),
        ("19.4", "GPUs & Mixed Precision",
         "memory arithmetic float16 bfloat16 loss scaling growth"),
        ("19.5", "Data vs Model Parallelism",
         "all-reduce pipeline bubble amdahl stragglers linear scaling"),
        ("19.6", "Distribution Strategies",
         "mirrored multiworker scope global batch sharding"),
        ("19.7", "Pipelines, Tuning & Cost",
         "prefetch utilisation random search hyperband spot instances"),
        ("19.8", "Monitoring, Drift & Retraining",
         "psi covariate concept drift sliced metrics feedback loop"),
        ("19.9", "The Checklist & Exercises",
         "shadow deployment rollback model card"),
    ]),
]

APPENDICES = [
    ("F", "Foundations — what learning actually is", "foundations", "🧭", [
        ("F.1", "The Learning Problem",
         "erm empirical risk minimisation hypothesis space inductive bias "
         "no free lunch approximation estimation optimisation error"),
        ("F.2", "Why Generalisation Works",
         "iid hoeffding union bound uniform convergence pac sample complexity "
         "distribution shift dependence effective sample size"),
        ("F.3", "Capacity & VC Dimension",
         "shattering growth function sauer lemma rademacher complexity "
         "vacuous bounds zhang random labels"),
        ("F.4", "Loss Functions",
         "bayes optimal conditional mean median quantile surrogate hinge "
         "logistic exponential calibration proper scoring rule brier"),
        ("F.5", "Bias, Variance & Double Descent",
         "decomposition interpolation threshold benign overfitting "
         "implicit regularisation minimum norm"),
        ("F.6", "Optimisation Foundations",
         "smoothness convexity strong convexity condition number nesterov "
         "lower bound robbins monro noise ball saddle points"),
        ("F.7", "Evaluation & Uncertainty",
         "standard error mcnemar paired test cross validation variance "
         "winner curse multiple comparisons"),
        ("F.8", "Data, Features & Geometry",
         "measurement scales cyclic encoding one-hot scaling distance "
         "concentration intrinsic dimension manifold hypothesis"),
        ("F.9", "Vocabulary & Misconceptions",
         "glossary mental models common errors checklist"),
    ]),
    ("A", "ML Project Checklist", "checklist", "✅", [
        ("A.1", "Eight phases, 70+ checks",
         "frame data explore prepare shortlist tune present deploy"),
    ]),
    ("B", "Autodifferentiation", "autodiff", "➗", [
        ("B.1", "Four Ways to Get a Derivative",
         "manual numerical symbolic finite differences expression swell"),
        ("B.2", "Forward Mode — Dual Numbers", "epsilon squared jacobian vector"),
        ("B.3", "Reverse Mode",
         "tape adjoint topological baur-strassen checkpointing"),
        ("B.4", "Building an Engine", "micrograd closure value node"),
        ("B.5", "GradientTape in Practice",
         "persistent watch stop_gradient custom gradient hessian"),
        ("B.6", "Reference & Further Reading", "derivative rules jax pytorch"),
    ]),
    ("M", "Math Appendix", "math", "📐", [
        ("M.1", "Linear Algebra",
         "norms svd eigendecomposition condition number pseudoinverse"),
        ("M.2", "Matrix Calculus",
         "denominator layout softmax cross-entropy outer product"),
        ("M.3", "Probability",
         "bayes jensen ensemble variance central limit distributions"),
        ("M.4", "Statistics & Estimation",
         "maximum likelihood bootstrap confidence interval multiple comparisons"),
        ("M.5", "Information Theory",
         "entropy kl divergence gini mutual information psi"),
        ("M.6", "Convex Optimisation",
         "convexity saddle points duality kkt"),
    ]),
    ("L", "AI Lab", "ai-lab", "🧪", [
        ("L.1", "Six live workbenches",
         "supervised sweep arena clustering neural playground scratchpad"),
    ]),
    ("G", "Glossary & Symbols", "glossary", "📖", [
        ("G.1", "200+ terms, symbol table, dependency map",
         "glossary notation prerequisites reading routes"),
    ]),
]

PART_NAME = {"I": "Part I · The Fundamentals of Machine Learning",
             "II": "Part II · Neural Networks and Deep Learning"}


# --------------------------------------------------------------------------


def _all_rows():
    rows = []
    for num, title, url, icon, part, subs in SYLLABUS:
        for sn, st_, kw in subs:
            rows.append(dict(chapter=num, chapter_title=title, url=url,
                             icon=icon, part=part, sub=sn, sub_title=st_,
                             keywords=kw))
    for num, title, url, icon, subs in APPENDICES:
        for sn, st_, kw in subs:
            rows.append(dict(chapter=num, chapter_title=title, url=url,
                             icon=icon, part="A", sub=sn, sub_title=st_,
                             keywords=kw))
    return rows


def render_search():
    section("1", "Search the whole course")

    lead(
        "Every sub-section is tagged with the concepts it covers. Search "
        "matches titles and tags — try <i>vanishing</i>, <i>attention</i>, "
        "<i>drift</i>, <i>bootstrap</i>, <i>kernel</i>."
    )

    rows = _all_rows()

    c1, c2 = st.columns([2.4, 1.0])
    q = c1.text_input("Search", placeholder="attention, overfitting, quantisation…",
                      key="syl_q").strip().lower()
    scope = c2.selectbox("Scope", ["Everything", "Part I only", "Part II only",
                                   "Appendices only"], key="syl_scope")

    hits = rows
    if scope == "Part I only":
        hits = [r for r in hits if r["part"] == "I"]
    elif scope == "Part II only":
        hits = [r for r in hits if r["part"] == "II"]
    elif scope == "Appendices only":
        hits = [r for r in hits if r["part"] == "A"]
    if q:
        terms = q.split()
        hits = [r for r in hits
                if all(t in (r["sub_title"] + " " + r["keywords"] + " "
                             + r["chapter_title"]).lower() for t in terms)]

    st.caption(f"**{len(hits)}** of {len(rows)} sub-sections")

    if not hits:
        st.info("Nothing matches. Try a single word — the tags are concept "
                "names, not sentences.", icon="🔍")
        return

    if q:
        table(["Chapter", "Sub-section", "Covers"],
              [[f'{r["icon"]} <b>{r["chapter"]}</b> · {r["chapter_title"]}',
                f'<b>{r["sub"]}</b> {r["sub_title"]}',
                r["keywords"]] for r in hits[:60]])
        if len(hits) > 60:
            st.caption(f"…and {len(hits)-60} more. Narrow the search.")
    else:
        st.caption("Type something above, or browse the full listing in the "
                   "next section.")


def render_full():
    section("2", "The full listing")

    lead(
        "All 19 chapters and 4 appendices. Each sub-section has a lecture, an "
        "animation with a play button, and a runnable code lab."
    )

    for part in ("I", "II"):
        st.markdown(
            f'<div class="mp-sbtitle" style="margin-top:20px;font-size:.95rem">'
            f'{PART_NAME[part]}</div>', unsafe_allow_html=True)
        for num, title, url, icon, p, subs in SYLLABUS:
            if p != part:
                continue
            with st.expander(f"{icon}  **{num} · {title}**  "
                             f"— {len(subs)} sub-sections"):
                table(["#", "Sub-section", "Covers"],
                      [[f"<b>{s[0]}</b>", s[1], s[2]] for s in subs])
                _link(url, f"Open chapter {num}", icon)

    st.markdown(
        '<div class="mp-sbtitle" style="margin-top:20px;font-size:.95rem">'
        'Labs &amp; Reference</div>', unsafe_allow_html=True)
    for num, title, url, icon, subs in APPENDICES:
        with st.expander(f"{icon}  **{num} · {title}**"):
            table(["#", "Section", "Covers"],
                  [[f"<b>{s[0]}</b>", s[1], s[2]] for s in subs])
            _link(url, f"Open {title}", icon)


_FILES = {
    "ch01": "p01_landscape.py", "ch02": "p02_endtoend.py",
    "ch03": "p03_classification.py", "ch04": "p04_training.py",
    "ch05": "p05_svm.py", "ch06": "p06_trees.py", "ch07": "p07_ensembles.py",
    "ch08": "p08_dimred.py", "ch09": "p09_unsupervised.py",
    "ch10": "p10_ann.py", "ch11": "p11_deep.py", "ch12": "p12_custom_tf.py",
    "ch13": "p13_data_tf.py", "ch14": "p14_cnn.py", "ch15": "p15_rnn.py",
    "ch16": "p16_nlp.py", "ch17": "p17_generative.py", "ch18": "p18_rl.py",
    "ch19": "p19_scale.py", "ai-lab": "p20_ai_lab.py", "math": "p21_math.py",
    "checklist": "p22_checklist.py", "autodiff": "p23_autodiff.py",
    "foundations": "p25_foundations.py",
    "glossary": "p24_glossary.py",
}


def _file_for(url: str) -> str:
    return _FILES.get(url, "p00_home.py")


def _link(url: str, label: str, icon: str) -> None:
    """A page link, degrading to a plain caption outside the nav context."""
    try:
        st.page_link(f"views/{_file_for(url)}", label=label, icon=icon)
    except Exception:
        st.caption(f"{icon}  {label}  ·  `/{url}`")


def render_shape():
    section("3", "The shape of the course")

    rows = _all_rows()
    df = pd.DataFrame(rows)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chapters", len(SYLLABUS))
    m2.metric("Sub-sections", len(rows))
    m3.metric("Appendices & labs", len(APPENDICES))
    m4.metric("Animations", len(rows), "one per sub-section")

    counts = df.groupby(["part", "chapter"], sort=False).size().reset_index(
        name="n")
    order = [c[0] for c in SYLLABUS] + [a[0] for a in APPENDICES]
    counts["order"] = counts["chapter"].apply(
        lambda c: order.index(c) if c in order else 99)
    counts = counts.sort_values("order")

    f = go.Figure()
    for part, col, nm in [("I", C["primary"], "Part I"),
                          ("II", C["accent"], "Part II"),
                          ("A", C["muted"], "Appendices")]:
        sub_ = counts[counts.part == part]
        f.add_bar(x=sub_["chapter"], y=sub_["n"], name=nm,
                  marker=dict(color=col))
    f.update_layout(height=400, barmode="stack",
                    xaxis_title="chapter", yaxis_title="sub-sections",
                    title="Sub-sections per chapter")
    st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)

    rule()

    sub("What every sub-section contains")

    table(
        ["Element", "What it is"],
        [["<b>Lecture</b>",
          "Original prose, with the derivations written out rather than cited"],
         ["<b>Derivation</b>", "Step-by-step, each step stating what changed "
                               "and why"],
         ["<b>Animation</b>",
          "A Plotly figure with a ▶ play button and a scrub slider"],
         ["<b>Code lab</b>",
          "Runnable, editable Python with a persistent namespace"],
         ["<b>Callouts</b>",
          "Pitfalls, warnings, ideas, proofs, and code notes"],
         ["<b>Key points</b>", "Five lines at the end of every sub-section"]],
    )

    note(
        "On sources",
        "The chapter and sub-section <b>ordering</b> follows <i>Hands-On "
        "Machine Learning with Scikit-Learn, Keras &amp; TensorFlow</i> "
        "(A. Géron, 3rd edition) as a syllabus, because it is a well-organised "
        "curriculum. <b>All lecture text, derivations, animations and code on "
        "this platform are original</b> — no text or figures are reproduced "
        "from the book. Papers are cited where a result comes from one.",
    )

    rule()

    sub("Suggested routes")

    table(
        ["Goal", "Route", "Roughly"],
        [["<b>A complete pass</b>", "F → 1 → 19, plus M and B alongside",
          "The whole thing"],
         ["<b>Tabular data, fast</b>", "1 → 2 → 3 → 4 → 6 → 7 → 19",
          "7 chapters"],
         ["<b>Deep learning foundations</b>", "4 → 10 → 11 → B → 12 → 13",
          "6 chapters"],
         ["<b>Computer vision</b>", "10 → 11 → 13 → 14 → 17", "5 chapters"],
         ["<b>NLP</b>", "10 → 11 → 13 → 15 → 16", "5 chapters"],
         ["<b>Generative models</b>", "10 → 11 → 14 → 17", "4 chapters"],
         ["<b>Reinforcement learning</b>", "4 → 10 → 11 → 18", "4 chapters"],
         ["<b>Shipping a model</b>", "2 → 13 → 19 → A", "3 chapters + the "
                                                       "checklist"],
         ["<b>The theory, on its own</b>", "F → M → B", "3 appendices"],
         ["<b>Just the mathematics</b>", "M → B", "2 appendices"]],
    )

    idea(
        "Read the key points first",
        "Each sub-section ends with five lines. Reading just those, for a whole "
        "chapter, takes about two minutes and tells you which sub-sections you "
        "already know and which you do not. Then read the ones you do not, and "
        "run their labs. That is a much better use of an afternoon than reading "
        "linearly from the top.",
    )


SECTIONS = [
    ("1", "Search", render_search),
    ("2", "Full listing", render_full),
    ("3", "Shape & routes", render_shape),
]

nav.render_chapter(CH, SECTIONS, sidebar_title="Syllabus")
