"""Glossary and symbol table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import nav
from core.lecture import (figure, hero, idea, lead, math, md, note, rule,
                          section, sub, table, tip)
from core.palette import C, SEQ, alpha
from core.theme import PLOTLY_CONFIG, inject

inject()
CH = "glossary"

hero(
    kicker="Reference",
    title="Glossary & symbol table",
    blurb=(
        "Every term the platform uses, defined in one sentence, with the "
        "section that develops it. Searchable, filterable by chapter, and "
        "written to be read cold."
    ),
    chips=["200+ terms", "symbol table", "searchable", "cross-referenced"],
)


# --------------------------------------------------------------------------
# term, definition, chapter tag
# --------------------------------------------------------------------------

TERMS: list[tuple[str, str, str]] = [
    # ---- fundamentals -------------------------------------------------
    ("Machine learning",
     "Programming computers so they can learn from data, rather than being "
     "explicitly told the rules.", "1"),
    ("Training set",
     "The examples the model learns from. Its performance here is not "
     "evidence of anything on its own.", "2"),
    ("Validation set",
     "Held out from training, used to choose hyperparameters. Once you have "
     "chosen against it many times, it is optimistically biased.", "2"),
    ("Test set",
     "Held out and looked at ONCE, at the end. Tuning against it turns it "
     "into a validation set and leaves you with no test set.", "2"),
    ("Generalisation error",
     "Error on data the model has never seen. The only number that matters.",
     "2"),
    ("Overfitting",
     "The model has learned noise specific to the training set. Symptom: a "
     "large train–validation gap.", "4"),
    ("Underfitting",
     "The model is too simple to capture the pattern. Symptom: poor "
     "performance on both train and validation.", "4"),
    ("Bias (statistical)",
     "The error from wrong assumptions — a model too simple for the data. "
     "Not to be confused with a bias term or with social bias.", "4"),
    ("Variance (statistical)",
     "The error from sensitivity to the particular training sample.", "4"),
    ("Irreducible error",
     "Noise inherent in the data. No model can go below it.", "4"),
    ("Regularisation",
     "Any constraint that reduces a model's freedom to fit the training data. "
     "Formally, a prior on the parameters.", "4"),
    ("Hyperparameter",
     "A setting of the learning algorithm, not learned from data. Chosen on a "
     "validation set.", "2"),
    ("Cross-validation",
     "Rotating which fold is held out, so every example is used for both "
     "training and validation. Note the folds are correlated.", "2"),
    ("Data leakage",
     "Information from outside the training set contaminating it. The usual "
     "cause of a suspiciously good result.", "2"),
    ("Feature engineering",
     "Constructing inputs that make the pattern easier to learn. Often worth "
     "more than the model choice.", "2"),
    ("Pipeline",
     "A sequence of transforms plus an estimator, fitted as one object so it "
     "cannot leak and can be deployed.", "2"),
    ("Stratified sampling",
     "Sampling that preserves the class proportions of the population.", "2"),
    ("No free lunch theorem",
     "Averaged over all possible problems, every algorithm performs equally. "
     "You must make assumptions; the question is which.", "1"),

    # ---- classification -----------------------------------------------
    ("Precision",
     "Of the instances predicted positive, the fraction that really are. "
     "TP/(TP+FP).", "3"),
    ("Recall",
     "Of the instances that really are positive, the fraction found. "
     "TP/(TP+FN). Also called sensitivity.", "3"),
    ("F1 score", "The harmonic mean of precision and recall.", "3"),
    ("Confusion matrix",
     "A table of predicted class against actual class.", "3"),
    ("ROC curve",
     "True positive rate against false positive rate as the threshold moves.",
     "3"),
    ("AUC",
     "Area under the ROC curve. Equals the probability that a random positive "
     "scores above a random negative.", "3"),
    ("PR curve",
     "Precision against recall. Preferred over ROC when the positive class is "
     "rare.", "3"),
    ("Calibration",
     "Whether a predicted probability of 0.7 corresponds to being right 70 % "
     "of the time.", "3"),
    ("One-vs-rest / one-vs-one",
     "Two ways to turn a binary classifier into a multiclass one.", "3"),

    # ---- linear models ------------------------------------------------
    ("Normal equation",
     "The closed-form least-squares solution "
     "$\\hat\\theta = (\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top"
     "\\mathbf{y}$.", "4"),
    ("Gradient descent",
     "Iteratively step against the gradient. Converges for a convex loss with "
     "a small enough step.", "4"),
    ("Stochastic gradient descent",
     "Gradient descent on one example (or a mini-batch) at a time. The noise "
     "is a feature.", "4"),
    ("Learning rate",
     "The step size. Too large diverges, too small crawls.", "4"),
    ("Ridge regression",
     "Least squares with an $\\ell_2$ penalty. Equivalent to a Gaussian prior "
     "on the weights.", "4"),
    ("Lasso",
     "Least squares with an $\\ell_1$ penalty. The corners of the $\\ell_1$ "
     "ball are why it produces exactly-zero coefficients.", "4"),
    ("Elastic net", "A weighted combination of the ridge and lasso penalties.",
     "4"),
    ("Early stopping",
     "Stop training when validation error starts rising. A regulariser.", "4"),
    ("Logistic regression",
     "A linear model whose output is squashed by a sigmoid and trained with "
     "cross-entropy.", "4"),
    ("Softmax regression",
     "The multiclass generalisation of logistic regression.", "4"),

    # ---- SVM -----------------------------------------------------------
    ("Support vector machine",
     "A classifier that maximises the margin between classes.", "5"),
    ("Margin", "The distance from the decision boundary to the nearest point.",
     "5"),
    ("Support vector",
     "A training point that touches the margin. Only these affect the model.",
     "5"),
    ("Hard / soft margin",
     "Whether violations are forbidden or merely penalised (by $C$).", "5"),
    ("Kernel trick",
     "Replacing an inner product with a kernel function, working in a "
     "high-dimensional space without ever constructing it.", "5"),
    ("Mercer's condition",
     "The requirement that a kernel's Gram matrix be positive semi-definite.",
     "5"),
    ("Hinge loss", "$\\max(0, 1 - y\\hat y)$ — the SVM's loss.", "5"),

    # ---- trees and ensembles -------------------------------------------
    ("Decision tree",
     "A model that recursively splits the feature space on one feature at a "
     "time.", "6"),
    ("CART",
     "Classification and Regression Trees: the greedy binary-splitting "
     "algorithm scikit-learn uses.", "6"),
    ("Gini impurity",
     "$1 - \\sum p_i^2$. The second-order approximation of entropy.", "6"),
    ("Information gain",
     "The drop in entropy achieved by a split.", "6"),
    ("Pruning",
     "Removing branches that do not justify their complexity. "
     "Cost-complexity pruning uses a parameter $\\alpha$.", "6"),
    ("Ensemble", "A model built from many models.", "7"),
    ("Bagging",
     "Bootstrap aggregating: train each model on a bootstrap sample and "
     "average. Reduces variance by decorrelating.", "7"),
    ("Pasting", "Bagging without replacement.", "7"),
    ("Out-of-bag evaluation",
     "Evaluating each model on the ~37 % of examples its bootstrap sample "
     "missed — free validation.", "7"),
    ("Random forest",
     "Bagged trees that also subsample features at each split, lowering the "
     "correlation between trees further.", "7"),
    ("Extra trees",
     "A random forest that also randomises the split thresholds. More bias, "
     "less variance, much faster.", "7"),
    ("Boosting",
     "Training models sequentially, each correcting its predecessors' errors.",
     "7"),
    ("AdaBoost",
     "Boosting by reweighting misclassified examples. Equivalent to forward "
     "stagewise fitting of an exponential loss.", "7"),
    ("Gradient boosting",
     "Boosting by fitting each new model to the residual — functional "
     "gradient descent.", "7"),
    ("Stacking",
     "Learning a model that combines other models' predictions.", "7"),

    # ---- dimensionality ------------------------------------------------
    ("Curse of dimensionality",
     "In high dimensions, everything is far from everything else and volume "
     "concentrates near the boundary.", "8"),
    ("PCA",
     "Projection onto the directions of greatest variance — the top singular "
     "vectors of the centred data.", "8"),
    ("Explained variance ratio",
     "The fraction of total variance a component accounts for.", "8"),
    ("Incremental / randomised PCA",
     "PCA variants for data that does not fit in memory, or for speed.", "8"),
    ("Kernel PCA", "PCA performed in a kernel-induced feature space.", "8"),
    ("Manifold learning",
     "Methods assuming the data lies on a low-dimensional curved surface.",
     "8"),
    ("LLE",
     "Locally Linear Embedding: preserve each point's reconstruction from its "
     "neighbours.", "8"),
    ("t-SNE",
     "A visualisation method preserving local neighbourhoods. Distances "
     "between clusters are meaningless.", "8"),
    ("Johnson–Lindenstrauss lemma",
     "A random projection to $\\mathcal{O}(\\log n / \\epsilon^2)$ dimensions "
     "preserves pairwise distances — independent of the original dimension.",
     "8"),

    # ---- unsupervised --------------------------------------------------
    ("k-means",
     "Partition into $k$ clusters by alternating assignment and centroid "
     "update. Converges to a local optimum.", "9"),
    ("k-means++",
     "An initialisation that spreads the initial centroids, greatly reducing "
     "bad local optima.", "9"),
    ("Inertia",
     "Within-cluster sum of squares. Falls monotonically with $k$, so the "
     "'elbow' is a judgement call.", "9"),
    ("Silhouette score",
     "How much closer a point is to its own cluster than to the next nearest. "
     "Has an actual optimum in $k$.", "9"),
    ("DBSCAN",
     "Density-based clustering. Finds arbitrary shapes and labels outliers as "
     "noise; needs no $k$.", "9"),
    ("Gaussian mixture model",
     "A generative model: data drawn from a weighted mixture of Gaussians, "
     "fitted with EM.", "9"),
    ("Expectation-maximisation",
     "Alternate between inferring the latent assignments and maximising the "
     "parameters. Derived from Jensen's inequality.", "9"),
    ("BIC / AIC",
     "Information criteria that penalise model complexity when choosing the "
     "number of components.", "9"),

    # ---- neural networks -----------------------------------------------
    ("Perceptron",
     "A single linear threshold unit. Cannot represent XOR.", "10"),
    ("Multilayer perceptron",
     "Stacked dense layers with non-linear activations.", "10"),
    ("Backpropagation",
     "Reverse-mode autodiff applied to a neural network.", "10"),
    ("Activation function",
     "The non-linearity between layers. Without one, a deep net collapses to "
     "a linear model.", "10"),
    ("Epoch", "One full pass through the training set.", "10"),
    ("Batch size",
     "How many examples per gradient step. Larger means less noise and more "
     "parallelism.", "10"),
    ("Sequential / functional / subclassing API",
     "Keras's three ways to define a model, in increasing order of "
     "flexibility.", "10"),
    ("Callback",
     "A hook Keras calls during training — early stopping, checkpointing, "
     "learning-rate schedules.", "10"),
    ("Vanishing gradient",
     "Gradients shrinking exponentially with depth, so early layers barely "
     "learn.", "11"),
    ("Exploding gradient", "The same product growing instead of shrinking.",
     "11"),
    ("Glorot / He initialisation",
     "Variance-preserving weight initialisations, for tanh-like and ReLU-like "
     "activations respectively.", "11"),
    ("ReLU / leaky ReLU / ELU / SELU / GELU",
     "The family of activations that made deep networks trainable.", "11"),
    ("Dying ReLU",
     "A unit whose input is always negative outputs 0 forever, so no gradient "
     "reaches it.", "11"),
    ("Batch normalisation",
     "Normalising activations using batch statistics. Speeds training and "
     "regularises; does not work inside an RNN.", "11"),
    ("Layer normalisation",
     "Normalising across features of one instance. Batch- and "
     "step-independent, so it works in RNNs and Transformers.", "11"),
    ("Gradient clipping",
     "Capping the gradient's norm or value. Essential for RNNs.", "11"),
    ("Transfer learning",
     "Reusing a model trained on one task as the starting point for another.",
     "11"),
    ("Momentum",
     "Accumulating a velocity across steps. Terminal velocity is "
     "$\\eta g/(1-\\beta)$.", "11"),
    ("Adam / AdamW",
     "Adaptive optimisers combining momentum and per-parameter scaling. AdamW "
     "decouples weight decay.", "11"),
    ("1cycle schedule",
     "Ramp the learning rate up then down within one training run.", "11"),
    ("Dropout",
     "Randomly zeroing units during training. Approximately an ensemble of "
     "$2^n$ networks.", "11"),
    ("Monte Carlo dropout",
     "Keeping dropout on at inference and averaging, to estimate uncertainty.",
     "11"),
    ("Max-norm regularisation",
     "Constraining each unit's incoming weight vector's norm.", "11"),

    # ---- TensorFlow ----------------------------------------------------
    ("Tensor", "A multidimensional array, the basic TF data type.", "12"),
    ("Variable", "A mutable tensor — what a model's weights are.", "12"),
    ("GradientTape",
     "TensorFlow's context manager that records operations for reverse-mode "
     "autodiff.", "12"),
    ("tf.function",
     "Traces a Python function once into a graph. Python side effects happen "
     "only during tracing.", "12"),
    ("AutoGraph",
     "The source-to-source transform that converts Python control flow into "
     "graph operations.", "12"),
    ("tf.data",
     "The input-pipeline API. Order matters: shuffle before batch, prefetch "
     "last.", "13"),
    ("TFRecord",
     "TensorFlow's binary record format, holding serialised protocol buffers.",
     "13"),
    ("Preprocessing layer",
     "A Keras layer that learns its state via `adapt()` and serialises with "
     "the model — the fix for training/serving skew.", "13"),
    ("Embedding",
     "A learned dense vector representation of a categorical value.", "13"),

    # ---- vision --------------------------------------------------------
    ("Convolution",
     "A sliding weighted sum. Encodes locality and translation equivariance.",
     "14"),
    ("Receptive field",
     "The region of the input that influences one output unit.", "14"),
    ("Stride / padding / dilation",
     "The three knobs of convolution arithmetic.", "14"),
    ("Pooling", "Downsampling by taking a max or a mean over a window.", "14"),
    ("1×1 convolution",
     "A per-position linear map across channels. Used as a bottleneck.", "14"),
    ("Residual connection",
     "$y = x + F(x)$. Gives the gradient an identity path, making very deep "
     "networks trainable.", "14"),
    ("Depthwise separable convolution",
     "A spatial convolution per channel, then a 1×1 mix. Far fewer parameters.",
     "14"),
    ("Squeeze-and-excitation",
     "A learned per-channel gate computed from global context.", "14"),
    ("IoU", "Intersection over union — the object-detection overlap measure.",
     "14"),
    ("Non-max suppression",
     "Removing overlapping duplicate detections.", "14"),
    ("Semantic segmentation",
     "Classifying every pixel. U-Net's skip connections restore the spatial "
     "detail pooling removed.", "14"),

    # ---- sequences -----------------------------------------------------
    ("Recurrent neural network",
     "A network with a loop: the output at each step feeds back as input.",
     "15"),
    ("BPTT",
     "Backpropagation through time — unrolling the RNN and applying reverse "
     "mode.", "15"),
    ("Truncated BPTT",
     "Backpropagating only $k$ steps, trading memory for the longest learnable "
     "dependency.", "15"),
    ("Stateful RNN",
     "One that carries its state across batches. Four strict requirements, all "
     "of which fail silently.", "15"),
    ("LSTM",
     "A cell with a separately-gated long-term state updated additively, so "
     "its gradient does not vanish.", "15"),
    ("GRU",
     "A simplified LSTM with one state and tied input/forget gates.", "15"),
    ("Forget gate",
     "The LSTM gate that decides what to keep. Its bias is initialised to 1.",
     "15"),
    ("Causal convolution",
     "A convolution padded only on the left, so output $t$ cannot see the "
     "future.", "15"),
    ("Dilated convolution",
     "A convolution with gaps between taps. Exponential receptive field at "
     "constant parameter cost.", "15"),
    ("WaveNet",
     "A stack of dilated causal convolutions. Fully parallel, short gradient "
     "path.", "15"),
    ("Seasonal naive",
     "The forecasting baseline that repeats the value from one season ago. "
     "Often very hard to beat.", "15"),

    # ---- NLP -----------------------------------------------------------
    ("Perplexity",
     "$e^{\\mathcal{L}}$ — the effective number of choices per token.", "16"),
    ("Temperature",
     "$p_i \\propto p_i^{1/T}$. One knob between greedy and uniform sampling.",
     "16"),
    ("Top-k / nucleus sampling",
     "Truncating the distribution before sampling, to cut the unreliable "
     "tail.", "16"),
    ("Byte-pair encoding",
     "A subword tokenisation that merges frequent adjacent pairs. Eliminates "
     "out-of-vocabulary tokens by construction.", "16"),
    ("Masking",
     "Telling a layer to ignore padded positions. Unmasked padding fails "
     "silently.", "16"),
    ("Teacher forcing",
     "Feeding the decoder the true previous token during training. Makes "
     "training parallel; causes exposure bias.", "16"),
    ("Exposure bias",
     "The mismatch between training on true prefixes and inferring on the "
     "model's own.", "16"),
    ("Beam search",
     "Keeping the $k$ best partial sequences. Needs length normalisation.",
     "16"),
    ("Attention",
     "A weighted average of values, with weights from a softmax over "
     "query–key similarities. A differentiable dictionary lookup.", "16"),
    ("Self-attention",
     "Attention where queries, keys and values all come from the same "
     "sequence.", "16"),
    ("Multi-head attention",
     "Several attentions in parallel on lower-dimensional projections, at "
     "essentially no extra cost.", "16"),
    ("Positional encoding",
     "Information about position added to the embeddings, because "
     "self-attention is permutation-equivariant.", "16"),
    ("Transformer",
     "An architecture built entirely from attention and feed-forward layers. "
     "$\\mathcal{O}(1)$ sequential steps.", "16"),
    ("BERT / GPT",
     "Encoder-only with masked pretraining, versus decoder-only with causal "
     "pretraining.", "16"),
    ("Vision Transformer",
     "A Transformer over image patches as tokens.", "16"),
    ("LoRA",
     "Fine-tuning a frozen model through a low-rank update. Under 1 % of the "
     "parameters.", "16"),

    # ---- generative ----------------------------------------------------
    ("Autoencoder",
     "A network trained to reconstruct its input through a constraint.", "17"),
    ("Undercomplete", "An autoencoder whose code is smaller than its input.",
     "17"),
    ("Denoising autoencoder",
     "One trained to reconstruct a clean input from a corrupted one. Its "
     "residual estimates the score.", "17"),
    ("Sparse autoencoder",
     "One with a wide code and a penalty on activation. Used for "
     "interpretability.", "17"),
    ("Variational autoencoder",
     "An autoencoder that encodes to a distribution and is trained on the "
     "ELBO.", "17"),
    ("ELBO",
     "Evidence lower bound: reconstruction minus KL to the prior. From "
     "Jensen's inequality.", "17"),
    ("Reparameterisation trick",
     "$\\mathbf{z} = \\boldsymbol\\mu + \\boldsymbol\\sigma \\otimes "
     "\\boldsymbol\\varepsilon$, which moves the randomness off the gradient "
     "path.", "17"),
    ("Posterior collapse",
     "The KL term goes to zero and the decoder ignores the latent.", "17"),
    ("GAN",
     "A generator and a discriminator in a minimax game. The loss is learned, "
     "not written down.", "17"),
    ("Mode collapse",
     "The generator emits only a few outputs. Nothing in the loss rewards "
     "coverage.", "17"),
    ("Non-saturating loss",
     "Training the generator with fakes labelled real, so the gradient is "
     "largest when it is worst.", "17"),
    ("Wasserstein GAN",
     "Replacing Jensen–Shannon with the earth-mover distance, which has a "
     "gradient even for disjoint supports.", "17"),
    ("Diffusion model",
     "A denoiser trained across a schedule of noise levels, run backwards from "
     "pure noise.", "17"),
    ("Classifier-free guidance",
     "Extrapolating away from the unconditional prediction. Trades diversity "
     "for prompt adherence.", "17"),

    # ---- RL ------------------------------------------------------------
    ("Policy", "A mapping from state to a distribution over actions.", "18"),
    ("Return", "The discounted sum of future rewards.", "18"),
    ("Discount factor",
     "$\\gamma$. Sets the effective horizon $\\approx 1/(1-\\gamma)$ and is "
     "part of the objective, not a tuning knob.", "18"),
    ("Credit assignment",
     "Working out which past action caused a delayed reward.", "18"),
    ("Policy gradient",
     "$\\mathbb{E}[G_t \\nabla \\log \\pi_\\theta]$, from the log-derivative "
     "trick. The environment dynamics cancel.", "18"),
    ("REINFORCE", "The basic policy-gradient algorithm.", "18"),
    ("Baseline",
     "Any state-dependent function subtracted from the return. Free, because "
     "its expectation is zero.", "18"),
    ("Advantage", "$Q(s,a) - V(s)$: better or worse than average.", "18"),
    ("Markov decision process",
     "A model where the next state depends only on the current state and "
     "action.", "18"),
    ("Bellman equation",
     "The recursive relation defining the value function.", "18"),
    ("Value iteration",
     "Repeatedly applying the Bellman operator. Converges because the operator "
     "is a $\\gamma$-contraction.", "18"),
    ("Temporal difference",
     "Updating toward $r + \\gamma V(s')$ — a sampled Bellman target.", "18"),
    ("Q-learning",
     "Off-policy TD control using $\\max_{a'} Q(s',a')$ as the target.", "18"),
    ("SARSA", "On-policy TD control, using the action actually taken.", "18"),
    ("Off-policy",
     "Learning about one policy while behaving according to another. What "
     "makes replay buffers legal.", "18"),
    ("Replay buffer",
     "A store of past transitions, sampled uniformly to break correlation.",
     "18"),
    ("Target network",
     "A frozen copy of the Q network used for the regression target.", "18"),
    ("Deadly triad",
     "Function approximation + bootstrapping + off-policy learning, which "
     "together can diverge.", "18"),
    ("Double DQN",
     "Selecting with the online network and evaluating with the target, to "
     "remove maximisation bias.", "18"),
    ("Dueling DQN",
     "Splitting $Q$ into $V$ and $A$, with the mean advantage subtracted to "
     "make it identifiable.", "18"),
    ("Prioritised replay",
     "Sampling in proportion to TD error, with importance-sampling weights to "
     "stay unbiased.", "18"),
    ("Actor–critic",
     "An actor that acts and a critic that supplies the baseline.", "18"),
    ("GAE",
     "Generalised advantage estimation: an exponentially weighted average of "
     "$n$-step advantages.", "18"),
    ("PPO",
     "Clipping the probability ratio to a trust region so a batch can be "
     "reused.", "18"),
    ("SAC",
     "Maximum-entropy off-policy actor–critic. The default for continuous "
     "control.", "18"),

    # ---- deployment ----------------------------------------------------
    ("SavedModel",
     "A self-describing directory of graph, weights and assets. Runs without "
     "Python.", "19"),
    ("Signature",
     "A model's declared input/output contract.", "19"),
    ("Training/serving skew",
     "The preprocessing at serving time differing from training. The most "
     "common production ML bug.", "19"),
    ("Batching",
     "Grouping requests. $T(B) = c_0 + c_1 B$, so throughput saturates at "
     "$1/c_1$.", "19"),
    ("p99 latency",
     "The 99th percentile. Users experience the tail, not the mean.", "19"),
    ("Quantisation",
     "Storing weights as int8: $r = S(q-Z)$. About 4× smaller for usually "
     "under 1 % accuracy.", "19"),
    ("Distillation",
     "Training a small model on a large one's soft outputs, which carry the "
     "teacher's similarity structure.", "19"),
    ("Mixed precision",
     "Computing in float16 while keeping float32 weights. Needs loss scaling "
     "because float16 underflows at $6\\times10^{-5}$.", "19"),
    ("Data parallelism",
     "Replicating the model and splitting the batch. Ring all-reduce moves "
     "$\\approx 2M$ bytes regardless of device count.", "19"),
    ("Model parallelism",
     "Splitting the model across devices. Pays a pipeline bubble.", "19"),
    ("Amdahl's law",
     "Speed-up is capped at $1/(1-p)$ however many devices you add.", "19"),
    ("Distribution strategy",
     "Keras's API for multi-device training. Everything creating a variable "
     "goes inside `scope()`.", "19"),
    ("Covariate shift", "$P(X)$ changes; $P(Y|X)$ does not. Detectable.", "19"),
    ("Concept drift",
     "$P(Y|X)$ itself changes. Invisible without labels — the dangerous one.",
     "19"),
    ("PSI",
     "Population stability index — the Jeffreys divergence between two "
     "binned distributions. Over 0.25 is significant.", "19"),
    ("Canary release",
     "Routing a small share of traffic to a new version first, with a defined "
     "abort rule.", "19"),

    # ---- math ----------------------------------------------------------
    ("Eckart–Young theorem",
     "The truncated SVD is the best low-rank approximation. The basis of PCA "
     "and of linear autoencoders.", "M"),
    ("Condition number",
     "$\\sigma_{\\max}/\\sigma_{\\min}$. Gradient descent needs "
     "$\\mathcal{O}(\\kappa)$ steps — the reason to scale features.", "M"),
    ("Jensen's inequality",
     "$\\varphi(\\mathbb{E}X) \\le \\mathbb{E}[\\varphi(X)]$ for convex "
     "$\\varphi$. Gives the ELBO, EM, and maximisation bias.", "M"),
    ("Maximum likelihood",
     "Choosing parameters that make the data most probable. Every loss is a "
     "negative log-likelihood.", "M"),
    ("KL divergence",
     "$\\sum p\\log(p/q)$. Asymmetric: $D(p\\Vert q)$ is mode-covering, "
     "$D(q\\Vert p)$ is mode-seeking.", "M"),
    ("Cross-entropy",
     "$H(p,q) = H(p) + D_{\\mathrm{KL}}(p\\Vert q)$, so minimising it is "
     "minimising KL.", "M"),
    ("Convexity",
     "Every chord lies above the function. Guarantees a global optimum.", "M"),
    ("Lagrangian duality",
     "Turning a constrained primal into a dual. What makes the kernel trick "
     "possible.", "M"),
    ("KKT conditions",
     "The optimality conditions for a constrained problem. Complementary "
     "slackness is why only support vectors matter.", "M"),
    ("Saddle point",
     "A critical point with both positive and negative curvature. In high "
     "dimensions, almost every critical point is one.", "M"),

    # ---- autodiff ------------------------------------------------------
    ("Automatic differentiation",
     "Applying the chain rule to a computation graph. Exact, unlike finite "
     "differences.", "B"),
    ("Forward mode",
     "Propagating derivatives alongside values with dual numbers. Costs $n$ "
     "passes; no tape.", "B"),
    ("Reverse mode",
     "A forward pass recording a tape, then a backward pass of adjoints. One "
     "pass for the whole gradient.", "B"),
    ("Dual number",
     "$a + b\\varepsilon$ with $\\varepsilon^2 = 0$, which makes the Taylor "
     "series terminate exactly.", "B"),
    ("Adjoint", "$\\partial y/\\partial v$ for an intermediate $v$.", "B"),
    ("Baur–Strassen theorem",
     "The gradient costs at most a constant multiple of the function, "
     "independent of the input count.", "B"),
    ("Gradient checkpointing",
     "Storing only some activations and recomputing the rest. "
     "$\\mathcal{O}(\\sqrt{L})$ memory for ~30 % more compute.", "B"),
    ("Straight-through estimator",
     "Using a surrogate derivative for a non-differentiable operation. What "
     "makes quantisation-aware training possible.", "B"),
]


SYMBOLS: list[tuple[str, str, str]] = [
    ("$m$", "number of training instances", "2"),
    ("$n$", "number of features", "2"),
    ("$\\mathbf{x}^{(i)}$", "the $i$-th instance's feature vector", "2"),
    ("$y^{(i)}$", "the $i$-th instance's label", "2"),
    ("$\\mathbf{X}$", "the design matrix, $m \\times n$", "4"),
    ("$\\hat y$", "a prediction", "2"),
    ("$h_\\theta$", "the hypothesis, parameterised by $\\theta$", "2"),
    ("$\\boldsymbol\\theta$, $\\mathbf{w}$, $b$", "parameters, weights, bias",
     "4"),
    ("$J(\\theta)$, $\\mathcal{L}$", "the cost / loss function", "4"),
    ("$\\eta$", "the learning rate", "4"),
    ("$\\alpha$, $\\lambda$", "regularisation strength", "4"),
    ("$C$", "the SVM's inverse regularisation", "5"),
    ("$\\gamma$", "RBF kernel width (§5) or the discount factor (§18)", "5"),
    ("$K(\\cdot,\\cdot)$", "a kernel function", "5"),
    ("$\\sigma(\\cdot)$", "the logistic sigmoid", "4"),
    ("$\\odot$, $\\otimes$", "element-wise (Hadamard) product", "15"),
    ("$\\nabla$", "the gradient operator", "4"),
    ("$\\mathbf{H}$", "the Hessian", "M"),
    ("$\\mathbb{E}[\\cdot]$", "expectation", "M"),
    ("$\\mathrm{Var}(\\cdot)$", "variance", "M"),
    ("$D_{\\mathrm{KL}}$", "Kullback–Leibler divergence", "M"),
    ("$H(p)$", "entropy", "M"),
    ("$\\mathbf{h}_{(t)}$", "an RNN's hidden state at step $t$", "15"),
    ("$\\mathbf{c}_{(t)}$", "an LSTM's cell (long-term) state", "15"),
    ("$\\mathbf{Q},\\mathbf{K},\\mathbf{V}$", "attention queries, keys, values",
     "16"),
    ("$d_k$", "attention key dimension", "16"),
    ("$\\mathbf{z}$", "a latent code", "17"),
    ("$q_\\phi$, $p_\\theta$", "encoder and decoder distributions", "17"),
    ("$\\bar\\alpha_t$", "cumulative diffusion signal retention", "17"),
    ("$\\pi_\\theta(a\\mid s)$", "a policy", "18"),
    ("$G_t$", "the return from step $t$", "18"),
    ("$V^\\pi$, $Q^\\pi$", "state and action value functions", "18"),
    ("$A^\\pi$", "the advantage function", "18"),
    ("$\\delta_t$", "the TD error", "18"),
    ("$\\varepsilon$", "the dual-number infinitesimal ($\\varepsilon^2=0$)",
     "B"),
    ("$\\bar v$", "the adjoint $\\partial y/\\partial v$", "B"),
    ("$\\kappa$", "the condition number", "M"),
    ("$\\rho$", "correlation (§M.3) or utilisation (§19.2)", "M"),
]

CHAPTER_NAMES = {
    "1": "1 · ML landscape", "2": "2 · End-to-end project",
    "3": "3 · Classification", "4": "4 · Training models",
    "5": "5 · SVMs", "6": "6 · Decision trees",
    "7": "7 · Ensembles", "8": "8 · Dimensionality reduction",
    "9": "9 · Unsupervised", "10": "10 · Intro to ANNs",
    "11": "11 · Training deep nets", "12": "12 · Custom TF",
    "13": "13 · Loading data", "14": "14 · CNNs", "15": "15 · RNNs",
    "16": "16 · NLP & attention", "17": "17 · Generative",
    "18": "18 · Reinforcement learning", "19": "19 · At scale",
    "M": "Math appendix", "B": "Appendix B · Autodiff",
}


# --------------------------------------------------------------------------


def render_glossary():
    section("G", "Glossary")

    lead(
        f"{len(TERMS)} terms, each defined in one sentence and tagged with the "
        "chapter that develops it. Search matches the term and the definition."
    )

    c1, c2, c3 = st.columns([2.0, 1.4, 1.0])
    q = c1.text_input("Search", placeholder="attention, gradient, drift…",
                      key="gl_q").strip().lower()
    chapters = ["All"] + [CHAPTER_NAMES[k] for k in
                          sorted(set(t[2] for t in TERMS),
                                 key=lambda v: (v in "MB", v.zfill(2)))]
    chap = c2.selectbox("Chapter", chapters, key="gl_ch")
    sort_by = c3.selectbox("Order", ["A–Z", "By chapter"], key="gl_sort")

    rows = TERMS
    if q:
        rows = [t for t in rows if q in t[0].lower() or q in t[1].lower()]
    if chap != "All":
        key = [k for k, v in CHAPTER_NAMES.items() if v == chap][0]
        rows = [t for t in rows if t[2] == key]
    if sort_by == "A–Z":
        rows = sorted(rows, key=lambda t: t[0].lower())
    else:
        rows = sorted(rows, key=lambda t: ((t[2] in "MB"), t[2].zfill(2),
                                           t[0].lower()))

    st.caption(f"**{len(rows)}** of {len(TERMS)} terms")
    if not rows:
        st.info("Nothing matches. Try a shorter search.", icon="🔍")
        return

    table(["Term", "Definition", "Chapter"],
          [[f"<b>{t[0]}</b>", t[1], CHAPTER_NAMES.get(t[2], t[2])]
           for t in rows])


def render_symbols():
    section("S", "Symbol table")

    lead(
        "The notation used throughout, with the first chapter in which it "
        "appears. Where a symbol is reused with a different meaning, both are "
        "listed."
    )

    q = st.text_input("Search", placeholder="gamma, theta, gradient…",
                      key="sym_q").strip().lower()
    rows = SYMBOLS
    if q:
        rows = [s for s in rows if q in s[0].lower() or q in s[1].lower()]

    st.caption(f"**{len(rows)}** of {len(SYMBOLS)} symbols")
    table(["Symbol", "Meaning", "First used"],
          [[s[0], s[1], CHAPTER_NAMES.get(s[2], s[2])] for s in rows])

    rule()

    sub("Notation conventions")

    table(
        ["Convention", "Example", "Means"],
        [["Lowercase italic", "$x$, $y$, $\\eta$", "a scalar"],
         ["Lowercase bold", "$\\mathbf{x}$, $\\mathbf{w}$",
          "a <b>column</b> vector"],
         ["Uppercase bold", "$\\mathbf{X}$, $\\mathbf{W}$", "a matrix"],
         ["Superscript in parentheses", "$\\mathbf{x}^{(i)}$",
          "the $i$-th <b>instance</b>"],
         ["Subscript in parentheses", "$\\mathbf{h}_{(t)}$",
          "the value at <b>time</b> $t$"],
         ["A hat", "$\\hat y$, $\\hat\\theta$",
          "an <b>estimate</b> or prediction"],
         ["A bar", "$\\bar x$, $\\bar v$",
          "a mean, or (in §B) an <b>adjoint</b>"],
         ["Blackboard bold", "$\\mathbb{E}$, $\\mathbb{R}$",
          "expectation; the reals"],
         ["Script", "$\\mathcal{L}$, $\\mathcal{O}$",
          "a loss; asymptotic order"]],
    )

    note(
        "Vectors are columns here",
        "So $\\mathbf{w}^\\top\\mathbf{x}$ is a scalar and "
        "$\\mathbf{x}\\mathbf{w}^\\top$ is a matrix. Gradients use "
        "<b>denominator layout</b>: $\\partial y/\\partial \\mathbf{x}$ has the "
        "same shape as $\\mathbf{x}$, which is what lets you write "
        "$\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\eta\\nabla$ "
        "directly. Note that NumPy and Keras store data with instances as "
        "<b>rows</b>, so the code writes $\\mathbf{X}\\mathbf{w}$ where the "
        "maths writes $\\mathbf{w}^\\top\\mathbf{x}$ — that transpose is the "
        "single most common source of confusion moving between the two.",
    )


def render_map():
    section("D", "Dependency map")

    lead(
        "Which chapters you need before which. The arrows are hard "
        "prerequisites, not suggestions."
    )

    nodes = {
        "1": (0, 3.0), "2": (1.3, 3.0), "3": (2.6, 3.6), "4": (2.6, 2.4),
        "5": (3.9, 3.6), "6": (3.9, 2.4), "7": (5.2, 3.0),
        "8": (3.9, 1.2), "9": (5.2, 1.2),
        "10": (6.5, 3.0), "11": (7.8, 3.0), "12": (9.1, 3.8),
        "13": (9.1, 2.2), "14": (10.4, 3.6), "15": (10.4, 2.2),
        "16": (11.7, 2.2), "17": (11.7, 3.8), "18": (11.7, 0.8),
        "19": (13.0, 3.0), "M": (0, 1.2), "B": (6.5, 1.2),
    }
    edges = [("1", "2"), ("2", "3"), ("2", "4"), ("4", "5"), ("4", "6"),
             ("5", "7"), ("6", "7"), ("4", "8"), ("8", "9"),
             ("7", "10"), ("4", "10"), ("10", "11"), ("11", "12"),
             ("11", "13"), ("12", "14"), ("13", "14"), ("13", "15"),
             ("14", "15"), ("15", "16"), ("14", "17"), ("16", "17"),
             ("11", "18"), ("17", "19"), ("18", "19"), ("16", "19"),
             ("M", "4"), ("M", "8"), ("B", "11")]

    colours = {"M": C["muted"], "B": C["muted"]}
    part = {**{str(i): C["primary"] for i in range(1, 10)},
            **{str(i): C["accent"] for i in range(10, 20)}, **colours}

    f = go.Figure()
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        f.add_scatter(x=[xa, xb], y=[ya, yb], mode="lines",
                      line=dict(color=alpha(C["line"], .85), width=2),
                      showlegend=False, hoverinfo="skip")
    for n, (x, y) in nodes.items():
        f.add_scatter(x=[x], y=[y], mode="markers+text", text=[n],
                      textposition="middle center",
                      textfont=dict(size=12, color="#fff"),
                      marker=dict(size=40, color=part[n],
                                  line=dict(color="#fff", width=2)),
                      hovertext=CHAPTER_NAMES.get(n, n),
                      hoverinfo="text", showlegend=False)
    f.update_layout(height=470, plot_bgcolor="#FFFFFF",
                    xaxis=dict(visible=False, range=[-.8, 14]),
                    yaxis=dict(visible=False, range=[0.2, 4.4]),
                    title="Blue = Part I, teal = Part II, grey = appendices")
    st.plotly_chart(f, width="stretch", config=PLOTLY_CONFIG)

    rule()

    sub("Reading routes")

    table(
        ["If you want to…", "Read", "Skip for now"],
        [["Get productive with tabular data fast",
          "1 → 2 → 3 → 4 → 6 → 7", "5, 8, 9, and all of Part II"],
         ["Understand deep learning properly",
          "4 → 10 → 11 → B → 12", "Most of Part I's model zoo"],
         ["Work on images", "10 → 11 → 13 → 14", "15–18"],
         ["Work on text", "10 → 11 → 13 → 15 → 16", "14, 17, 18"],
         ["Build generative models", "10 → 11 → 14 → 17", "15, 16, 18"],
         ["Do reinforcement learning", "10 → 11 → 18",
          "Most of Part I beyond chapter 4"],
         ["Deploy something", "2 → 13 → 19",
          "Everything else, until it is deployed"],
         ["Fill in the mathematics", "M → B",
          "Nothing — these are read alongside"]],
    )

    idea(
        "Chapter 4 and chapter 11 are the two load-bearing chapters",
        "Chapter 4 introduces the gradient-descent machinery every later model "
        "uses, and the bias–variance framing every later evaluation uses. "
        "Chapter 11 introduces the initialisation, normalisation, optimiser and "
        "regularisation toolkit that every architecture in Part II assumes. "
        "If you are going to read two chapters carefully, read those two — "
        "everything else is an application of them.",
    )


SECTIONS = [
    ("G", "Glossary", render_glossary),
    ("S", "Symbol table", render_symbols),
    ("D", "Dependency map", render_map),
]

nav.render_chapter(CH, SECTIONS, sidebar_title="Reference")
