# ML Platform

An interactive, executable machine-learning course: 19 chapters and 5
appendices, 212 sub-sections, each with a full lecture, the mathematics written
out, a Plotly animation you drive with a play button, and a runnable code lab.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens on <http://localhost:8501>.

---

## What is in it

| | |
|---|---|
| **Chapters** | 19, in two parts |
| **Sub-sections** | 212, each self-contained |
| **Animations** | one per sub-section, with ▶ Play and a scrub slider |
| **Code labs** | 171, editable and runnable in the page |
| **Appendices** | foundations, mathematics, project checklist, autodifferentiation, glossary |
| **AI Lab** | six live workbenches — supervised, sweep, arena, clustering, neural playground, scratchpad |

Every sub-section follows the same shape:

- a **lecture** in original prose,
- **derivations** written step by step, each step saying what changed and why,
- an **animation** built with Plotly frames,
- a **code lab** — editable Python with a persistent namespace, so you can poke
  at a fitted model after the fact,
- **callouts** for pitfalls, warnings, proofs and code notes,
- **five key points** at the end.

---

## Foundations — the theory underneath

Read alongside chapters 1–4. Nine sub-sections on what a learning problem *is*
and why any of it is justified: ERM and inductive bias, why generalisation is
possible (Hoeffding → union bound → PAC), capacity and VC dimension, what a
loss function is really asking for, bias–variance *and* double descent,
convergence rates and the condition number, the statistics of evaluation, and
the geometry of high dimensions. Every result is verified numerically in its
lab rather than asserted.

## Part I — The Fundamentals of Machine Learning

| # | Chapter | Covers |
|---|---|---|
| 1 | The ML Landscape | Types of systems, the main challenges, no free lunch |
| 2 | End-to-End Project | Framing, exploration, pipelines, the test set |
| 3 | Classification | Precision/recall, ROC, AUC, error analysis |
| 4 | Training Models | Normal equation, SVD, gradient descent, ridge/lasso |
| 5 | Support Vector Machines | Margins, the dual, kernels, SMO |
| 6 | Decision Trees | CART, Gini vs entropy, pruning, instability |
| 7 | Ensembles & Random Forests | Bagging, boosting, stacking, the variance formula |
| 8 | Dimensionality Reduction | Curse of dimensionality, PCA, LLE, t-SNE |
| 9 | Unsupervised Learning | k-means, DBSCAN, Gaussian mixtures, EM |

## Part II — Neural Networks and Deep Learning

| # | Chapter | Covers |
|---|---|---|
| 10 | Intro to ANNs with Keras | Perceptron, MLP, backprop, the three Keras APIs |
| 11 | Training Deep Nets | Vanishing gradients, batch norm, optimisers, dropout |
| 12 | Custom Models & Training | Tensors, custom layers, `GradientTape`, `tf.function` |
| 13 | Loading & Preprocessing Data | `tf.data`, TFRecord, preprocessing layers, embeddings |
| 14 | Deep Computer Vision | Convolution arithmetic, ResNet, detection, segmentation |
| 15 | Sequences with RNNs & CNNs | BPTT, forecasting, LSTM/GRU, WaveNet |
| 16 | NLP, Attention & Transformers | Char-RNNs, beam search, attention, the Transformer, ViT |
| 17 | Autoencoders, GANs & Diffusion | ELBO, GAN dynamics, DDPM, classifier-free guidance |
| 18 | Reinforcement Learning | Policy gradients, Bellman, Q-learning, DQN, PPO |
| 19 | Training & Deploying at Scale | SavedModel, quantisation, parallelism, drift |

## Labs & Reference

| | |
|---|---|
| **AI Lab** | Six workbenches with live model fitting |
| **Foundations** | ERM, PAC, VC dimension, loss theory, double descent, convergence rates, evaluation statistics, high-dimensional geometry |
| **Math appendix** | Linear algebra, matrix calculus, probability, statistics, information theory, convex optimisation |
| **Appendix A** | The ML project checklist — 8 phases, 70+ checks, progress tracked |
| **Appendix B** | Autodifferentiation, including an engine built from scratch |
| **Glossary** | 200+ terms, symbol table, chapter dependency map |

---

## Running the code labs

Each lab has four controls:

| Control | Does |
|---|---|
| **▶ Run** | Executes the buffer in that lab's namespace |
| **✏️ Edit the source** | Opens an editor — change anything, then Run |
| **↺ Restore** | Puts the original code back, clears the namespace |
| **🧹 Clear vars** | Empties the namespace, keeps your edits |

The namespace persists between runs, so you can inspect a fitted model
afterwards. The labs are meant to be broken on purpose — set the regularisation
to zero, remove the target network, delete a `+=` in the autodiff engine. The
`Environment & setup` page probes this machine and tells you what is missing.

Preloaded in every lab: `np`, `pd`, `px`, `go`, `make_subplots`, `st`, the
platform palette (`C`, `SEQ`, `CLASS_COLORS`, `PARULA`, `alpha`), and
`from core import datasets as ds`. Anything assigned to `fig` is rendered.

---

## Project layout

```
ml_platform/
├── app.py               entry point; the st.navigation tree
├── requirements.txt
├── core/
│   ├── palette.py       colours, MATLAB Parula, colourscales
│   ├── theme.py         the Plotly template and the CSS
│   ├── lecture.py       the lecture DSL (section, derive, math, callouts…)
│   ├── anim.py          play/pause/reset controls and frame helpers
│   ├── runner.py        the in-page Python execution engine
│   ├── nav.py           sub-section navigation and progress
│   ├── datasets.py      every dataset, cached, with offline fallbacks
│   └── rl.py            self-contained CartPole, grid world and MDPs
├── views/               one file per page
├── smoke_test.py        renders every sub-section headlessly
├── lab_test.py          executes every code lab
└── runlabs.py           executes named labs from one page
```

### Tests

```bash
python smoke_test.py          # render every page and sub-section
python smoke_test.py p04 p05  # only these
python lab_test.py            # execute every code lab
python lab_test.py --list     # list the labs without running them

python runlabs.py views/p15_rnn.py ch15_lstm    # one lab, by name
```

`smoke_test.py` catches rendering errors; `lab_test.py` actually executes the
labs, which is what catches library API drift. Both are worth running after any
edit.

---

## Design notes

**Colours.** `core/palette.py` reproduces MATLAB's R2014b *Parula* colormap from
its 64 RGB stops, alongside Jet, Turbo, Blue–Red and a custom Sinha scale.
Parula is the default for every surface, heatmap and contour. The colourscale
and the animation speed are switchable from the sidebar on any chapter page.

**Mathematics.** Rendered with `st.latex` where it stands alone, and with a
LaTeX→Unicode converter (`core.lecture.tex`) inside tables and callouts, since
KaTeX does not run in injected HTML.

**Offline.** Every dataset has a synthetic fallback, and every optional import
is guarded, so the platform works with no network connection.

**Claims are checked, not asserted.** Where the lecture states a theoretical
result, the lab measures it: Hoeffding's coverage, the bias–variance identity
summing to the observed MSE, the VC bound holding while being 30–300× loose,
the winner's curse tracking √(2 ln K)·SE, distance concentration following
1/√(2d).

---

## Sources

The chapter and sub-section **ordering** follows *Hands-On Machine Learning with
Scikit-Learn, Keras & TensorFlow* (A. Géron, 3rd edition) as a syllabus, because
it is a well-organised curriculum. **All lecture text, derivations, animations
and code here are original** — no text or figures are reproduced from the book.
Papers are cited where a result comes from one; each chapter ends with its
references.

---

Dr Merwan Roudane · <merwanroudane920@gmail.com> · <https://github.com/merwanroudane>
