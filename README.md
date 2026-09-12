# Testing the compositionality continuum

A computational test suite for **Riveland, Pouget & Driscoll (2026), *The compositionality continuum as a principle for studying the neural basis of intelligence*, Nature Neuroscience 29, 2067–2080.**

The Perspective argues that compositionality is not a binary trait but a continuum with two axes — **module expressivity** and **syntactic complexity** — and that compositional *behaviour* can come apart from compositional *implementation*. This repository operationalizes both axes, validates the measurements against synthetic ground truth, and runs 13 experiments across four model classes to test 13 specific claims.

Everything runs in a single self-contained Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/REPO/blob/main/Compositionality_Continuum_Experiments.ipynb)

---

## The problem this repository had to solve first

The authors deliberately leave both axes informal, writing that *"future theoretical work might provide precise mathematical definitions for both of these concepts."* **There is therefore no way to test the theory without first inventing the measurements**, and a negative result could always mean the operationalization was wrong rather than the theory.

The notebook handles this by defining every metric in code with its assumptions written down, and **validating each one against synthetic data with known ground truth before any network is touched** (Section 2). That discipline paid off in an uncomfortable way — see *Known issues* below.

---

## Headline results

| Claim (paper section) | Verdict | Key evidence |
|---|---|---|
| Nonnegativity + energy efficiency → single-variable modules | **supported** | selectivity 0.204 → 0.462 with nonnegativity; flat 0.124–0.184 without |
| Multitask RNNs modularize computation | **supported** | lesioning one cluster costs 0.78–0.94 on all Anti tasks, 0.11–0.48 on Pro |
| Subspaces reused across tasks | **supported** (but see below) | mean off-diagonal alignment 0.67–0.71 |
| Task variables coded abstractly | **supported** | CCGP(rule) = 1.000; parallelism 0.86–0.90, asymmetric |
| Structure requires a compositional task set | **supported** | identical accuracy (0.996); CCGP 1.000 vs 0.556 |
| Held-out recombinations learned faster | *inconclusive* | advantage 0.516 (comp) vs 0.406 (non-comp) — largely generic |
| MoE: expressive modules, no syntax → fails OOD | **supported**, with a caveat | MoE OOD R² = −0.611, gate entropy 0.786 (not collapsed) |
| Modularity and syntax are not independent | **supported** | under a lookup task family, `factored` swings +0.870 → −1.305 |
| Recombination needs combination diversity | **supported** | threshold between 12 and 16 of 36 combinations |
| Scale buys behaviour, not composition | **supported** | 323× params: in-dist 0.282 → 0.968, OOD −0.668 → 0.005 |
| Standard transformers fail systematicity | **supported** | 1.00 in-distribution, **0.00** on 65 held-out constructions |
| Meta-learning restores systematicity | *inconclusive* | in-context learning reaches 1.00, systematicity stays 0.02 |
| Roles represented abstractly, apart from contents | **NOT supported** | within-day role decoding 1.00, cross-day CCGP 0.257 (chance) |

### Three results worth reading the notebook for

**1. The RNN discovered a Pro module and an Anti module.** Silencing one unit cluster costs 0.78–0.94 accuracy on all four Anti tasks but only 0.11–0.48 on Pro tasks; a second cluster shows the mirror-image pattern. The reusable component is the *rule*, demonstrated causally.

**2. Behavioural binding without an abstract role code.** A network reaches 0.932 accuracy on unseen location sets and its role is decodable within-day at 1.00 — yet cross-day role CCGP is at chance (0.257) while cross-day *position* CCGP is 0.750. Training on unlimited days improves behaviour by 0.31 and leaves role abstraction **completely unchanged**. This contradicts the paper's binding claim under our operationalization.

**3. The task family dominates the architecture.** Swapping a compositional teacher for a lookup teacher — same input/output statistics, same difficulty, same number of tasks — drives *every* architecture to negative OOD, including one whose architecture literally is function composition. In-distribution fits barely move (0.999 → 0.827), so this is not a capacity problem.

---

## What is in the notebook

| § | Experiment | Systems |
|---|---|---|
| 1–2 | Metric definitions and **validation against ground truth** | 5 synthetic regimes |
| 3 | Biological constraints → modularity | autoencoder, 2 × 5 sweep |
| 4–6 | Multitask RNN: motifs, causal lesions, geometry | 128-unit CTRNN, 8 tasks |
| 7 | Non-compositional task-set control | matched CTRNN |
| 8 | Motif transfer to a held-out recombination | pretrained vs scratch |
| 9 | Four architectures on one task family | monolithic / factored / MoE / hypernet |
| 10 | Compositional vs lookup task family | 4 architectures × 2 families |
| 11 | Combination-coverage sweep | 8 coverage levels |
| 12 | Scale sweep (behaviour vs implementation) | 872 → 281,608 params |
| 13 | Curriculum comparison | interleaved / progressive / blocked |
| 14 | Transformer systematicity + meta-learning | mini-SCAN, add-primitive split |
| 15 | Binding to abstract roles | ABCD grid navigation |
| 16–18 | Continuum map, scorecard, limitations | all of the above |

### The measurement layer

**Geometry of abstraction** (Bernardi et al.): CCGP, parallelism score, shattering dimensionality.

**Axis 1 — module expressivity**: single-variable selectivity (1 − normalized entropy of each unit's variance decomposition across factors); participation ratio of a module's own response.

**Axis 2 — syntactic complexity**: a four-tier ladder of recombination rules — constant → additive → low-rank bilinear → case-by-case lookup — each asked to predict **factor combinations never seen in training** under a split that holds out arrangements while keeping every component visible. Scored as `systematicity × (0.35 + 0.65 × order)`, so a lookup table scores zero and falls off the continuum, as the paper's definition requires.

---

## Known issues

These are documented rather than hidden, because two of them change how the results should be read.

### 1. The higher-order syntax tier never fires ⚠️

In the recorded run the low-rank bilinear tier **never outperformed the additive tier on any dataset**, including the `bilinear` positive control constructed specifically for it (syntax score 0.00, where it should have been highest). The `order` column is 0.0 everywhere.

**Consequence:** the syntax metric as implemented collapses to *"how well does an additive rule transfer to unseen combinations?"* It correctly separates systematic-additive from case-by-case — which is most of what the downstream experiments need — but **cannot detect systematic higher-order structure**, the upper half of the paper's axis 2. For this reason the continuum map in Section 16 uses a **CCGP-derived** syntax coordinate, not the ladder.

**Likely fixes:** SVD-based rather than random initialization of the ALS factors; more factor levels relative to response dimensionality; an explicit identifiability check against the `bilinear` control before the metric is used anywhere.

### 2. `head_reuse` returns `nan`

The attention-head reuse measurement (Section 14) produced `nan` for both models. `head_reuse` calls `nn.MultiheadAttention` directly on a layer extracted from its `TransformerEncoder`; the returned attention weights are not in the shape the function assumes, so the similarity list ends up empty and `np.mean([])` is `nan`.

The code cell is **deliberately left unpatched** so that every code cell in the notebook is exactly the code that produced the outputs shown. To fix it, assert on the returned shape before use:

```python
_, attn = layer(h, h, h, attn_mask=mask, need_weights=True,
                average_attn_weights=False)
assert attn is not None and attn.dim() == 4, attn.shape if attn is not None else None
A = attn.mean(0).cpu().numpy()          # (heads, T, T)
```

Treat module expressivity inside the attention mechanism as **unmeasured**.

### 3. Subspace alignment is not diagnostic

Alignment was essentially unchanged between compositional and non-compositional task sets (0.715 vs 0.676 for the stimulus epoch; 0.672 vs 0.695 for the response epoch, the latter slightly *higher* for the non-compositional set). High subspace overlap appears to be a generic property of trained multitask RNNs. **Do not infer motif reuse from alignment without a non-compositional control.**

### 4. The transfer experiment has a confound

The non-compositional pretrained network reaches criterion at **step 0** — zero-shot, before any fine-tuning — which should be impossible with arbitrary per-task angular offsets. The likely cause is that an unseen one-hot rule cue produces a default response that happens to fall within the 0.35π tolerance of that task's random offset. Fixing it requires averaging over many random offset draws, a tighter tolerance, and several held-out tasks.

### 5. MoE underfits

MoE reaches only 0.727 in-distribution, so its OOD failure (−0.611) is partly confounded with underfitting. A clean test of "expressivity without syntax" needs an MoE that fits as well as `hypernet` and *still* fails.

### 6. Single seeds

Every number is one run. `QUICK = True` in particular runs reduced budgets. Multiple seeds with confidence intervals is the highest-value robustness fix.

---

## Running it

Open in Colab and run all cells. No installation is needed beyond the Colab defaults (`torch`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`).

```python
QUICK = True     # ~8–12 min on a Colab CPU
QUICK = False    # ~35–50 min; this is the setting used for the recorded results
SEED  = 0
```

The recorded outputs were produced with `QUICK = False` on a GPU runtime (`device=cuda, threads=2`). Results are saved to `compositionality_results.pkl`.

**Runtime breakdown (full mode):** multitask RNN 92 s · transfer 200 s · meta-learning 1534 s · binding 622 s · everything else under 60 s each.

---

## Citing the source paper

```bibtex
@article{riveland2026compositionality,
  title   = {The compositionality continuum as a principle for studying
             the neural basis of intelligence},
  author  = {Riveland, Reidar and Pouget, Alexandre and Driscoll, Laura},
  journal = {Nature Neuroscience},
  volume  = {29},
  pages   = {2067--2080},
  year    = {2026},
  doi     = {10.1038/s41593-026-02382-1}
}
```

## Scope and honesty

This is a small-scale, single-seed, synthetic-task study using metrics of its own construction, one of which demonstrably does not work as intended. It tests whether the paper's claims survive **one explicit formalization at small scale**. That is the kind of work the Perspective explicitly asks for, and exactly why it should not be mistaken for a verdict on the theory.

## License

MIT for the code. The source paper is © its publisher and is not redistributed here.
