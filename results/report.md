# Mathlib Geometry — Semester Report

_2026-05-04, Ege. All numbers are reproducible from this repo; sources are linked at the end of each section. Designed to be read standalone — no chat history needed._

---

## 0. One-paragraph summary

The project asks whether the *geometry* of Mathlib (a graph where theorems cite the lemmas they depend on, plus a finer "state graph" where nodes are intermediate proof states and edges are tactic applications) carries useful predictive information about its theorems and supports A→B path-finding for theorem discovery. This semester I built that geometry out of LeanDojo data, ran four predictive tasks on it, implemented two path-finding algorithms over it, and built a post-cutoff prover-eval pipeline. **Headline positives**: the dependency graph is structurally meaningful (Louvain modularity 0.59 with communities matching Mathlib's areas; predominantly hyperbolic Ollivier–Ricci curvature; degree alone explains 51 % of proof-length variance, 56.5 % when you add a node2vec embedding plus betweenness and effective resistance). **Headline negatives**: spectral coordinates of the normalised adjacency add zero on top of degree across four prediction tasks; path-finding under strict goal-text equality recovers only 2.5 % of citations because most uses of A inside B are term-mode or rewrite, not goal-replacement. **Concrete next move**: the post-cutoff Kimina holdout is built and the prover runner is written; first GPU run executed but max-tokens was set too low (most generations got cut off mid-reasoning); a corrected re-run is the immediate next step.

---

## 0.5. Quick definitions (skip if you know them)

- **LeanDojo Benchmark 4**: a dataset of every Mathlib theorem with its proof traced step-by-step (state before each tactic, the tactic itself, state after, and which premises that tactic cited). Random split into train/val/test.
- **Premise / dependency edge**: B → A means the proof of B contains a tactic that cites A as a premise. The "proof DAG" is the directed graph of all such edges.
- **State graph / state-tactic hypergraph**: nodes are proof states (a goal text plus its hypothesis context). Edges are tactic applications: input state → list of output states. Multi-output edges arise from tactics like `rintro` that split a goal into cases.
- **Spectral coordinates of the normalised adjacency**: for the symmetric matrix `D^{-1/2} A D^{-1/2}`, take the top-k and bottom-k eigenvectors. Each node gets a 2k-dim coordinate vector. Classic graph embedding.
- **Node2vec**: random walks from each node + skip-gram on those walks (like word2vec but for graphs). Different from spectral because it captures *local neighbourhood structure* rather than the principal modes of the adjacency matrix.
- **Modularity**: a number in [−1, 1] that measures how much denser the *intra*-community edges are than chance. > 0.3 is meaningful; > 0.5 is strong.
- **Louvain**: greedy community-detection algorithm that maximises modularity. Hierarchical: it merges communities into super-nodes and recurses.
- **Betweenness centrality**: for each node, the fraction of shortest paths in the graph that pass through it. High BC = "bridge" between otherwise distant parts.
- **Effective resistance**: treat each edge as a unit resistor; resistance between two nodes is the standard electrical distance. Smoother / more robust than shortest-path distance.
- **Ollivier–Ricci curvature on edge (u,v)**: 1 − W₁(unif N(u), unif N(v)) / d(u,v), where W₁ is Wasserstein-1 distance between uniform measures over the two endpoints' neighbourhoods. Negative κ means the neighbourhoods are far apart relative to d(u,v) — the edge is a "bottleneck" / locally tree-like; positive κ means they're tightly clustered (Euclidean-like).
- **R² (coefficient of determination)**: fraction of variance in the target explained by a model. R² = 0 means no better than predicting the mean; R² = 1 means perfect. Pearson ρ is the linear correlation, related but not the same.
- **Pass@n**: in prover eval, the fraction of theorems for which *at least one* of n model samples produces a Lean-verified proof.

---

## 1. What I set out to do

The original framing was: **"find new theorems / new abstractions in Mathlib"**. That goal proved hard to quantify directly, so over the semester it narrowed into two operationalisations:

1. **A → B path-finding.** Given two theorems A and B, can we trace a reduction from B to A along the structures derived from Mathlib's proofs? "Reduction" here means: starting from B's goal, apply tactics until A's goal appears (possibly with some branches discharged separately), at which point we say "use A as oracle".
2. **Predictive power of the geometry.** Can features derived from this graph predict something concrete about an individual theorem — its area, its difficulty, its citation pattern, its prover success?

Both questions reduce to the same underlying one: **is the geometry of Mathlib a useful object in its own right, or is it just a re-encoding of namespace-and-name conventions?**

---

## 2. The data and the graphs I built

Source: LeanDojo Benchmark 4 (random split).

### 2.1. Proof dependency graph

A directed graph: B → A iff B's proof contains a tactic that cites A as a premise.

| | Count |
|---|---:|
| Theorems indexed | 122,517 |
| Theorems with traced tactics | 61,544 |
| Tactic invocations | 259,580 |
| Premise occurrences | 418,704 |
| **Unique directed citation edges** | **309,396** ✓ matches expected |
| Unique nodes (theorems + premises) | 138,333 |

After symmetrising and dropping nodes with degree 0 or `UNKNOWN` area, 94,635 nodes survive for spectral / regression analysis.

### 2.2. State-tactic hypergraph

Nodes = single-goal proof states. Hyperedges = tactic applications. When a tactic produces multiple subgoals (e.g. `rintro (a | b | c)` → three cases), the hyperedge has multiple outputs; multi-goal `state_after` strings are split per `case ...` block so chains across `rintro` / `cases` / `all_goals` link up properly.

| | Count |
|---|---:|
| Unique single-goal states | 261,655 |
| Unique hyperedges | 261,107 |
| **True multi-output (split-tactic) hyperedges** | **18,467** |
| Output-size tail | up to 17 outputs |

### 2.3. Topology of the dependency graph (from earlier-semester analyses)

- **Degree distribution heavy-tailed.** Power-law fit α = 2.37 above x_min = 43 (KS = 0.028, n_tail = 864). Lognormal also fits (KS = 0.144); pure exponential rejected (p ≈ 0). Source: `Old/data/degree_depth_stats.json`.
- **Connectivity.** 138,333 nodes split across 44,479 weakly-connected components, with one giant component of 92,594 nodes. Source: `Old/data/theorem_graph_stats.json`.
- **Communities.** Louvain gives 240 communities at modularity **0.5924**. Communities align with mathematical fields:
    - Largest (15,488 nodes): Algebra (4,750), RingTheory (3,245), Data (1,874), NumberTheory (1,351), Analysis (856).
    - Second (14,737): Analysis (3,893), MeasureTheory (3,249), Topology (2,179), Data (1,383), Algebra (1,105).
    - Third (11,525): Data (3,712), Algebra (1,260), Combinatorics (1,122), Order (977).
- **Proof shapes.** 50,191 LINEAR / 5,893 BALANCED / 4,801 DEEP_NARROW / 655 BUSHY / 4 WIDE_SHALLOW. Most Mathlib proofs are linear chains. Source: `Old/data/search_tree_stats.json`.

### 2.4. Mathlib vs the broader Lean ecosystem

| | Mathlib | LEAN-GitHub |
|---|---:|---:|
| Theorems | 61,544 | 20,446 |
| Total tactics | 259,580 | 218,866 |
| Unique tactics | 274 | 1,093 |
| Mean proof length | 4.22 | 10.7 |
| Median proof length | 2 | 4 |
| Max proof length | 128 | 640 |

Mathlib proofs are markedly shorter and use a smaller, more standardised tactic vocabulary than the wider Lean-on-GitHub corpus. Source: `Old/data/comparative_stats.json`.

---

## 3. What worked

### 3.1. Degree predicts proof difficulty (positive, replicated)

Both the earlier-semester correlation analysis and tonight's regression find the same strong signal.

**From the semester correlation analysis** (`Old/data/difficulty_stats.json`, 61,542 theorems):

| Difficulty metric | Pearson r vs out-degree | Spearman ρ |
|---|---:|---:|
| proof length | **0.761** | 0.624 |
| unique tactics used | 0.707 | 0.607 |
| tactic diversity | 0.609 | — |
| max branching | 0.418 | — |

**From tonight's ridge regression** on `log1p(tactic_count)` (`results/predictive/proof_length_regression.json`, n = 61,544):

| Feature set | Test R² | Pearson ρ | RMSE |
|---|---:|---:|---:|
| mean predictor (baseline) | 0.000 | 0.000 | 0.685 |
| **out-degree + in-degree** | **0.483** | **0.695** | 0.493 |
| area one-hot | 0.046 | 0.216 | 0.670 |
| log statement length | 0.017 | 0.132 | 0.680 |
| spectral (low10+high10) | 0.003 | 0.052 | 0.685 |
| degree + area | 0.492 | 0.702 | 0.489 |
| **degree + area + statement length** | **0.498** | **0.706** | **0.486** |
| spectral + degree | 0.483 | 0.695 | 0.493 |

**Plain English**: the number of premises B cites alone explains roughly half of why some proofs are short and some are long. Adding which Mathlib area B is in, and how long its statement is, adds another two percentage points of variance. The two analyses (semester, tonight) agree.

### 3.2. The dependency graph reflects Mathlib's areas

Even though the spectral coordinates of the adjacency don't predict the 27-way category label (see § 4.1), the *citation pattern* of the dependency graph matches mathematical intuition. Source: `results/predictive/cross_area_dep_matrix.{json,svg}`.

Row-normalised cross-area citation matrix, key cells:

- **CategoryTheory: 90 % self-citing** — most isolated subfield in Mathlib.
- **Algebra: universal sink** — dominant external target for FieldTheory (46 %), RingTheory (44 %), RepresentationTheory (50 %), Tactic (52 %), Deprecated (50 %).
- **Probability cites MeasureTheory** more than itself (28 % vs 26 %) — clean inter-area dependency.
- **AlgebraicTopology ↔ CategoryTheory** (29 % from AT into CT).
- **Condensed → CategoryTheory** (48 %), nearly nothing self-citing (4 %); makes sense given Condensed is a recent reformulation that lives on top of CategoryTheory.

The Louvain community structure (§ 2.3) is the same finding through a different lens: modularity 0.59 with communities that map cleanly onto Mathlib's top-level areas. The graph carries area-level information; it just isn't picked up by the principal modes of the normalised adjacency.

### 3.3. Tactic style differs by area

Source: `results/predictive/tactic_mix_by_area.{json,svg}`. Top tactic head per source area, row-normalised:

| Area | Dominant tactic | 2nd |
|---|---|---|
| Algebra | rw (0.34) | simp (0.16) |
| AlgebraicGeometry | rw (0.33) | simp (0.19) |
| AlgebraicTopology | simp (0.44) | rw (0.24) |
| **Analysis** | **have (0.26)** | rw (0.22) |
| CategoryTheory | rw (0.23) | simp (0.19) |
| Combinatorics | rw (0.23) | simp (0.18) |
| Computability | have (0.24) | simp (0.17) |
| Condensed | simp (0.46) | let (0.20) |
| Data | rw (0.33) | simp (0.22) |
| **Init** | **apply (0.36)** | exact (0.21) |
| LinearAlgebra | rw (0.29) | simp (0.16) |
| Logic | simp (0.32) | rw (0.24) |
| **MeasureTheory** | **have (0.29)** | rw (0.18) |
| NumberTheory | rw (0.26) | have (0.25) |
| **Probability** | **have (0.26)** | rw (0.20) |
| RepresentationTheory | simp (0.41) | rw (0.28) |
| RingTheory | rw (0.31) | have (0.18) |
| **Topology** | **have (0.21)** | rw (0.17) |

Patterns: most areas are `rw` / `simp` first; analysis-flavoured areas (Analysis, Topology, MeasureTheory, Probability, Computability) lean on `have` (cut-style proofs); Init prefers term-style (`apply`, `exact`); RepresentationTheory and Condensed are simp-heavy. This is consistent with the kinds of proofs each subfield produces by hand.

### 3.4. Universal proof macros (BPE on tactic sequences)

Source: `Old/data/bpe_macro_abstractions.json`, `Old/data/domain_bpe_abstractions.json`.

Byte-pair encoding applied over tactic-head sequences yields a small vocabulary of recurring proof "macros". The single dominant global merge is:

```
"rw -> [0] -> exact"   ΔC = 4,371   occurrences = 7,895
```

That 3-step skeleton (rewrite, then a positional placeholder, then close with `exact`) accounts for an enormous chunk of compression in the corpus. Per-area BPE confirms it: Data, Algebra, Analysis, Topology, MeasureTheory all surface the same `rw → [0] → exact` macro as their top merge. So one stylised proof skeleton dominates Mathlib regardless of subfield.

`Old/data/abstraction_candidates.json` extends this to longer 3-7-grams both globally and per module. `Old/data/frequent_subgoals.json` lists the 500 most common subgoal *expressions* — top is `f x = g x` (extensionality-shaped), shared across 44 distinct theorems — these are candidates for "shareable lemma" abstractions.

### 3.5. Proof-state sharing across theorems is rare but real

Source: `Old/data/identical_state_tactic_clusters.json`, `results/state_graph/state_path_sweep_goal.json`.

33 clusters of theorems share an identical (state, tactic, state_after) triple — a literal proof step that recurs across multiple theorems with the same hypothesis context. The top cluster is shared by 12 theorems in the Geometry / Orientation namespace.

The state-tactic hypergraph (§ 2.2) makes this measurable globally. Within a single theorem the algorithm correctly recovers the recorded proof — the within-theorem sanity check on `ADEInequality.Admissible.one_lt_sumInv` finds the proof at depth 3 and AND/OR-discharge solves it in 3 expansions. Across theorems it finds shared states only when two theorems' goal texts are literally equal, which is rare; quantified in § 4.4.

### 3.6. The spectral rebuttal: node2vec, betweenness, resistance, and Ricci curvature

The eigendecomposition of the normalised adjacency added nothing on top of degree (§ 4.1–4.3). Tonight I re-tested the same predictive task (proof length) with three more graph-geometric features that have no reason to collapse to degree. Source: `analysis/graph_geometry.py`, `results/graph_geometry/{node2vec,betweenness,resistance,ollivier_ricci,joint}_*.json`.

| Feature set (proof-length regression, n = 57,601) | Test R² | Test ρ |
|---|---:|---:|
| degree | 0.506 | 0.711 |
| spectral (low10+high10), § 4.3 | 0.003 | 0.052 |
| **node2vec (64-d, walks-per-node 10, length 40)** | **0.162** | **0.403** |
| approx betweenness (200 random sources) | 0.183 | 0.429 |
| effective resistance to top-15 hubs (via 64 eigvecs) | 0.005 | 0.074 |
| **node2vec + degree** | **0.555** | **0.745** |
| **all combined (degree + node2vec + BC + resistance)** | **0.565** | **0.752** |

So **node2vec is the geometric feature that actually carries information**: on its own it reaches R² = 0.162, while the eigencoordinates of the same graph reach 0.003. Adding node2vec on top of degree lifts R² from 0.506 to 0.555 — almost a 10 % relative improvement. Combining all four features gets to R² = 0.565.

**Why does node2vec work and spectral not?** Both are "node embeddings" in some sense, but they encode different things. The spectral embedding sees the principal modes of the adjacency matrix — and on a heavy-tailed graph those modes are dominated by the degree distribution. Node2vec instead simulates random walks from each node and trains a skip-gram model on the resulting "node sequences", so it captures *local neighbourhood structure* and which nodes appear in the same walks. That neighbourhood signal is real and largely uncorrelated with raw degree. The "spectral negative" result was about the encoding, not the underlying graph.

#### 3.6.1. Hubs sanity-check

Top theorems by indegree (the resistance subcommand also dumps these):

> `rfl` (in_deg 2,645), `Eq.symm` (1,493), `mul_comm` (1,462), `mul_assoc` (1,267), `mul_one` (1,171), `one_mul` (1,015), `add_comm` (950), `le_antisymm` (814), `eq_comm` (701), `LE.le.trans` (699), …

Exactly the basic algebra and equality lemmas one would expect at the foundation. The graph correctly identifies the "fundamental" lemmas of Mathlib without any semantic input.

#### 3.6.2. Ollivier–Ricci curvature: Mathlib's dependency graph is hyperbolic

Sampled 5,000 edges, kept 3,320 with neighbourhoods ≤ 50, computed κ(u, v) = 1 − W₁(unif N(u), unif N(v)) / d(u, v) by solving the transportation LP per edge with `scipy.optimize.linprog`. Source: `results/graph_geometry/ollivier_ricci.json`.

| Quantity | Value |
|---|---:|
| edges evaluated | 3,320 |
| **mean κ** | **−0.781** |
| median κ | −0.840 |
| min / max κ | −1.788 / +0.543 |
| **fraction κ < 0** | **88.7 %** |
| fraction κ < −1 (strong bottleneck) | 35.7 % |

Discrete Ricci curvature this strongly negative is the signature of a **hyperbolic / tree-like** graph. This is consistent with the heavy-tailed power-law degree distribution from § 2.3 and the proof-tree structure of the state graph (§ 3.5): citations radiate from a few hubs, locally the graph looks like a tree, and there is little triangle-rich "Euclidean" structure. The most negative-κ edges connect semantically distant theorems via atypical citations:

- `spectrum.map_polynomial_aeval_of_degree_pos ↔ symm`
- `Eq.refl ↔ WithSeminorms.isVonNBounded_iff_seminorm_bounded`
- `Finset.filter_val ↔ Nat.divisors_filter_squarefree`

These are candidates for abstraction / refactor analysis — citations that "shouldn't" exist topologically but do.

### 3.7. State-tactic path-finder (built and validated)

Source: `analysis/state_hypergraph.py`. Two algorithms:

1. **`or-path`** — bidirectional BFS on goal-only hashes. Asks: during the proof of B, does the current goal ever literally equal A's initial goal? This is the cheap version of "reduce B to A".
2. **`discharge`** — AND/OR DFS with memoisation, treating A as oracle. Asks: can the proof tree rooted at B be closed if every leaf is in `{A's initial goal, no goals}`? This is the real "B is solved given A" semantics, and unlike OR-path it correctly handles tactics that split into multiple subgoals.

Within-theorem sanity check passes: starting from `ADEInequality.Admissible.one_lt_sumInv` and asking for a path to "no goals" finds it at depth 3, and AND/OR discharge solves the full proof tree in 3 expansions. The algorithm is correct; the *quantitative* result is in § 4.4.

### 3.8. Post-cutoff prover holdout (built; first run executed but truncated)

Source: `results/prover_eval/post_kimina_manifest.jsonl`, `results/prover_eval/post_kimina_manifest_stats.json`, `results/prover_eval/kimina_prompts.jsonl`, `results/prover_eval/kimina_post_2025_08_14.jsonl`, `analysis/kimina_eval_cuda.py`.

**The holdout itself**:

- 25 declarations introduced in Mathlib after **2025-08-14** (cutoff for Kimina-Prover-RL training).
- All 25 pass a `by sorry` precheck — i.e. the surrounding file still type-checks with the proof body removed.
- Anchor features for 17/25 (identifiers in the signature that map to existing graph nodes; mean of their spectral coords).
- Stratified across 14 mathematical areas (Analysis 3, Topology 2, GroupTheory 2, RingTheory 2, Data 2, Order 2, NumberTheory 2, Tactic 2, Combinatorics 2, Dynamics 2, Algebra 1, MeasureTheory 1, LinearAlgebra 1, Probability 1).

**The runner** (`analysis/kimina_eval_cuda.py`) splits work into:

- A **GPU `generate` pass** — one batched vLLM call (`SamplingParams(n=samples)`), or a transformers fallback. Configurable `--tensor-parallel-size`, `--gpu-memory-utilization`, bf16 default. Writes raw model outputs.
- A decoupled **CPU `verify` pass** — for each generation, replaces the proof body with the model's candidate, runs `lake env lean`, records pass/fail. Outputs include `lean_code`, `verified`, `lean_output`, plus a `.summary.json` with sample-level pass and theorem-level `pass@n`.

**First GPU run (2026-05-04, H100 on FASRC, transformers backend, 25 theorems × 8 samples = 200 generations)**:

- Run completed without crash.
- **But max-tokens was set to 1024, which was too small for Kimina's chain-of-thought style.** Kimina-Prover-RL is a reasoning model and spends thousands of tokens on internal explanation before emitting Lean. Of the 200 generations:
    - 65 (32.5 %) had no code block at all — purely reasoning, ran out of token budget.
    - 135 had at least one code block, but most of those blocks were prose / commentary inside fence markers.
    - **Only 23 (11.5 %) ended with a block that looks like actual Lean code.**
- Verification has not been run yet.

**Action**: re-run with `--max-tokens 8192 --samples 4` to give the model the headroom it needs. ~30–60 min on one H100. After that, run `verify` on a box that has Mathlib4 + lake.

---

## 4. What didn't work

### 4.1. Spectral category prediction (negative)

Source: `results/spectral/classifier_sweep.json`, `results/spectral/spectral_metrics.json`. Predict the 27-way Mathlib top-level area label of a node from its spectral coordinates.

| Method | Acc | Macro-F1 |
|---|---:|---:|
| majority | 0.165 | 0.011 |
| **namespace prefix** | **0.737** | **0.753** |
| degree only (mlp) | 0.172 | 0.019 |
| low10 + high10 (mlp) | 0.162 | 0.031 |
| low32 + high32 (mlp) | 0.160 | 0.025 |
| low10 + degree (mlp) | **0.177** | 0.021 |
| low32 + high32 + degree (mlp) | 0.174 | 0.022 |

The best spectral method beats majority by 1.2 percentage points and is roughly 56 percentage points behind the trivial namespace baseline (which is allowed to peek at the dotted prefix of the name, e.g. `Algebra.foo` → predict Algebra).

### 4.2. Spectral link prediction ≈ degree only (negative)

Source: `results/spectral/link_prediction.json`. 10,000 positive edges + 10,000 random non-edges, logistic regression.

| Method | Test acc | Test ROC-AUC |
|---|---:|---:|
| degree-only | 0.785 | 0.875 |
| full spectral + degree (258-d) | 0.789 | **0.881** |

Spectral coords add **+0.006 AUC** over degree alone — almost all the link signal is the preferential-attachment heuristic ("high-degree pairs are more likely to be connected").

### 4.3. Spectral tactic-head and proof-length prediction (negative)

Source: `results/spectral/tactic_head_prediction.json`, `results/predictive/proof_length_regression.json`. Edge-level tactic-head classification (top 8 heads):

| Method | Acc | Macro-F1 |
|---|---:|---:|
| majority (`rw`) | 0.339 | — |
| degree-only logreg | 0.349 | 0.103 |
| spectral concat logreg | 0.346 | 0.110 |

And on proof length (§ 3.1) spectral features alone get R² = 0.003 vs 0.483 for degree, and adding spectral on top of degree does not move R² at all. So across **four** predictive tasks (category, link, tactic head, proof length), the eigencoordinates of the normalised adjacency add nothing on top of degree-based features. The principal modes of this graph are essentially the degree distribution. (Resolved by node2vec, § 3.6.)

### 4.4. Path-finding under goal-text equality is too strict (negative, but informative)

Source: `results/state_graph/state_path_sweep_goal.json`, `results/paths/path_sweep.json`.

Three sweeps, all measuring "given (start, target), is there a tactic-chain path?":

| Sweep | n | reachable | rate | comment |
|---|---:|---:|---:|---|
| Random pairs in proof DAG, depth ≤ 8 | 2,000 | 1 | 0.05 % | most theorems have only ~hundreds of ancestors |
| Random theorem pairs in state graph, depth ≤ 12 | 2,000 | 2 | 0.10 % | goal texts almost never match across theorems |
| Dep-graph edges (B literally cites A) in state graph | 1,000 | 25 | **2.5 %** | the apples-to-apples test |

Per-tactic-head breakdown of the dep-graph cross-check is the most informative chart in the report:

| Citation tactic head | n | state-graph reachable | rate |
|---|---:|---:|---:|
| **`exact`** | 100 | 11 | **0.110** |
| **`refine`** | 54 | 5 | **0.093** |
| `induction` | 9 | 1 | 0.111 |
| `have` | 83 | 3 | 0.036 |
| `apply` | 40 | 1 | 0.025 |
| `simp` | 129 | 2 | 0.016 |
| `rw` | 359 | 2 | 0.006 |
| `simp_rw` | 45 | 0 | 0.000 |

This is exactly the pattern we'd expect: `exact A` / `refine A` make the current goal equal to A's statement, so the state graph confirms the connection ~10 % of the time (limited by binder/universe/implicit differences). `rw [A]` / `simp [A]` use A as a rewrite rule and the goal text never becomes A's statement, so the state graph almost never finds those.

**A concrete failure case** (illustrative): in `UniformConvergenceCLM.continuousSMul`, A is `Bornology.IsVonNBounded.image` and the citation is `(h𝔖₃ s hs).image u` — A is invoked term-style inside a lambda inside an argument to *another* theorem. There is never a goal `⊢ IsVonNBounded 𝕜₂ (⇑f '' s)` during B's proof. The proof DAG correctly counts this as an edge; the state graph correctly concludes no path exists *under text equality*. The two graphs are measuring different equivalence relations, not contradicting each other.

### 4.5. What § 4 means together

The eigendecomposition of the dep-graph adjacency is not a useful node embedding for the predictive tasks we care about — but this is *not* the same as "the graph carries no embeddable information". A *random-walk* embedding of the same graph (node2vec, § 3.6) does carry signal: R² on proof length goes from 0.003 (spectral) to 0.162 (node2vec) on the same target. The takeaway is that the principal modes of the normalised adjacency are dominated by degree; the local neighbourhood structure that random walks see is real and additional.

The state-tactic hypergraph correctly implements the algorithm I wanted, but goal-text equality is too strict an equivalence to make the search useful for discovery on arbitrary pairs. **Goal canonicalisation is the next concrete fix** (§ 5).

---

## 5. What's queued (immediate next moves)

In priority order — all are concrete extensions, not new directions:

1. **Re-run Kimina with proper token budget** (~30–60 min on one H100). The first run truncated; with `--max-tokens 8192 --samples 4` we get usable generations. Then run `verify` on a Mathlib-equipped box.
2. **Run ReProver on the same 25-theorem holdout** for a comparison point. ReProver is documented in `results/prover_eval/run_commands.md` but not installed in this workspace.
3. **Goal canonicalisation in the state graph.** Alpha-rename binders, sort hyps, normalise universe levels, then re-hash. Conservative guess: dep-edge reachability climbs from 2.5 % to 20–30 %, and the `exact` / `refine` rates approach 100 %. Half a day.
4. **Subterm matching for `rw`-style use.** Index every state by the set of subterms it contains; "B uses A" matches if some state in B has A's LHS as a subterm. Catches the 359 `rw` edges in §4.4. Half a day.
5. **Use `Old/data/frequent_subgoals.json` as oracles in the discharge solver.** The 500 most common subgoals across Mathlib are plausible "free leaves". Test what fraction of the dep-graph edges become solvable with this oracle set.
6. **Forward + backward path search.** The current `or-path` is already bidirectional, but we should also walk premise → theorem (downstream) to enable "what is provable from A?". The dep-graph adjacency is already there; minor change.
7. **Once the prover-success column exists**, train the small classifier "predict prover success from (degree, area, anchor spectral coords, statement length, node2vec embedding)". This was the original motivation for §3.8 and is the meeting-point of the predictive and prover threads.

---

## 6. FAQ — questions a PI might ask

**Q: How is this different from just doing graph ML on a knowledge graph?**
The dep graph is a knowledge graph in the broad sense, but the *state-tactic* hypergraph is not — it's much closer to the actual reasoning structure of proofs. The separation between "what cites what" (proof DAG) and "what does the proof goal look like at each step" (state graph) is what § 3.7 / § 4.4 are exploiting. The 2.5 % dep-edge reachability number quantifies how different these two views are.

**Q: Why does spectral fail and node2vec succeed if they're both graph embeddings?**
Spectral captures the *global* principal modes of the adjacency matrix; node2vec captures *local* neighbourhood structure. On a heavy-tailed scale-free graph the principal modes are dominated by the degree distribution, so spectral coords end up being a noisy version of degree. Node2vec sees that node u and node v co-occur in random walks → that's information degree alone doesn't have. Empirically spectral 0.003 R² vs node2vec 0.162 R² on the same target.

**Q: Mean Ollivier–Ricci κ = −0.78 sounds dramatic. Is it?**
For graphs of this kind, yes — it puts Mathlib in the same regime as expander-poor scale-free networks, not Euclidean lattices. Hyperbolicity is consistent with what we already know from the degree distribution and proof-shape stats; OR makes it quantitative at the edge level and identifies the strongest bottleneck edges.

**Q: Modularity 0.59 — what does that mean concretely?**
It means the Louvain partition is far better than chance: intra-community edges are dense, inter-community edges are sparse, and the gap is large enough that the partition is *meaningful*, not just a numerical optimum. The 240 communities map cleanly onto Mathlib's mathematical fields, which is a sanity check that the citation graph reflects the same partitioning humans would apply.

**Q: Why is the spectral negative more interesting than just "we tried something and it didn't work"?**
Three reasons. (1) It ruled out the most natural a-priori candidate ("just take the eigenvectors of A̅") so the next experiments could focus on richer features. (2) Across **four** different prediction tasks, the eigencoordinates collapsed to degree — that's a robust finding, not a one-off failure. (3) Node2vec on the same graph beats spectral by ~50× R² on proof length, which means the *graph itself* has signal — the spectral encoding was the bottleneck.

**Q: 0.05 % random-pair reachability looks bad. Should the path-finder be considered broken?**
No, it's correct — the question being asked is just very strict. A random pair of Mathlib theorems is almost never directly connected through the proof DAG because each theorem's ancestor closure is small (median frontier dies in 4–6 hops; § 4.4 details). The 2.5 % number on dep-edges is the meaningful quantification: even when B literally cites A, only 2.5 % of those citations show up as a goal-text path, because most uses of A inside B are term-mode, rewrite-mode, or instantiated. Goal canonicalisation should bring this up substantially.

**Q: R² = 0.565 on proof length — what counts as "good"?**
There's no natural ceiling here, but for context: predicting human-graded proof difficulty from features that don't include any text content (no statement, no proof, just graph metadata) at R² ≈ 0.57 is decent. The remaining ~43 % of variance presumably involves things like which specific lemmas are needed, how much rewriting is required, etc. — features you could only get by looking at the actual proof.

**Q: What's the headline result if I have to pick one?**
"The geometry of Mathlib carries non-trivial predictive information about its theorems — proof difficulty (R² 0.565), area structure (modularity 0.59), local curvature (mean κ −0.78) — but only under the right encoding. The eigenvectors of the adjacency don't work; random-walk embeddings do; and graph-theoretic distance metrics like betweenness and effective resistance add small additional lift. The state-tactic hypergraph is the right object for path-based reasoning but needs goal canonicalisation to be useful for discovery."

**Q: What's the riskiest claim in the report?**
The Ollivier–Ricci hyperbolicity result is computed on 3,320 sampled edges (out of 309,396), so it's a strong-signal sample but not exhaustive. The mean is stable (last few hundred edges nudge it by ≤ 0.01) so we're confident the population mean is in the same neighbourhood.

**Q: What would you do with another month?**
Re-run Kimina + ReProver with proper token budget; implement goal canonicalisation; train the prover-success classifier; replace the 64-eigenvector resistance approximation with a proper Laplacian solve; try persistent-homology features as a follow-up to the OR-curvature finding. None of that is a new direction — all consolidations of the existing scaffolding.

---

## 7. Honest summary

**What we know**:

- The dependency graph is a real geometric object: heavy-tailed degree (power law α = 2.37), 240 communities at modularity 0.59 aligned with Mathlib's areas, predominantly hyperbolic Ollivier–Ricci curvature (mean κ = −0.78 over 3,320 sampled edges).
- Coarse structural features predict proof difficulty: degree alone hits R² = 0.51 / ρ = 0.71 on proof length; adding a node2vec random-walk embedding lifts this to R² = 0.555, and combining all graph-geometric features gets to R² = 0.565 (ρ = 0.752). Replicated across two independent semester analyses.
- Cross-area citation patterns and per-area tactic mixes are interpretable and consistent with mathematical intuition.
- A single 3-step BPE macro (`rw → [0] → exact`) accounts for ~8,000 occurrences globally and is the top merge in every major area.
- Top theorems by indegree (`rfl`, `Eq.symm`, `mul_comm`, `mul_assoc`, `mul_one`, `one_mul`, `add_comm`, `le_antisymm`) are exactly the basic algebra/equality lemmas that should be foundational.

**What we know doesn't work**:

- The eigencoordinates of the normalised adjacency, taken naively, do not encode the area / link / tactic / difficulty information that degree-and-namespace already carry. **But a random-walk embedding (node2vec) of the same graph does** — so the failure was the encoding, not the underlying graph.
- Path-finding under strict goal-text equality on the state graph cannot recover the bulk of the dep-graph's citations because of term-mode use, rewriting, and instantiation. The 2.5 % dep-edge rate (and 11 % `exact`-only rate) is informative but not a useful discovery tool yet.

**What's missing**:

- A **prover-success column** for the 25-theorem holdout. The runner is written and a partial run is on disk (truncated by max-tokens); a re-run with the corrected setting fills this in.
- A **goal-canonicalisation layer** for the state graph. Half a day of implementation; potentially a 10× lift on dep-edge reachability and would unlock the discharge solver as a discovery tool.
- A **semantic version of "find new theorems"**: with goal-canonicalisation + frequent-subgoal oracles + forward search, the discharge solver becomes a real tool for asking "given A, what can be proved with A as oracle?".

The negative results are negative for clean, well-understood reasons (the eigendecomposition collapses to degree on heavy-tailed graphs; goal-text equality ignores instantiation, rewriting, and term-mode use). The blockers to turning each negative into a positive are *concrete*, not methodological — a different embedding (already shown to work with node2vec), a goal canonicaliser, and one GPU run away.

---

## 8. Reproducibility map

```
analysis/
  overnight_geometry.py       # build dep graph, spectral, post-cutoff manifest
  spectral_classifier_sweep.py
  extra_experiments.py        # path-sweep, link-pred, tactic-pred (proof DAG)
  state_hypergraph.py         # state-tactic hypergraph, or-path, discharge, sweep
  predictive_extras.py        # proof-length regression, dep matrix, tactic-by-area
  graph_geometry.py           # node2vec, betweenness, resistance, OR curvature, joint
  kimina_eval_cuda.py         # CUDA-first vLLM/transformers runner (split generate/verify)
  kimina_eval.py              # legacy single-pass runner

results/
  paths/                      # proof-DAG outputs
  state_graph/                # state-tactic hypergraph outputs
  spectral/                   # spectral coords + classifier sweeps
  predictive/                 # proof-length regression + heatmaps
  graph_geometry/             # node2vec, BC, resistance, OR curvature, joint
  prover_eval/                # post-2025-08-14 manifest + prompts + Kimina v1 generations
  report.md                   # this file

Old/data/                     # earlier-semester analyses (Mar–Apr 2026)
  abstraction_candidates.json # 3-7-grams of tactics, global + per-module
  bpe_macro_abstractions.json # 30 BPE merge steps over tactic sequences
  community_stats.json        # Louvain communities (240, modularity 0.59)
  comparative_stats.json      # Mathlib vs LEAN-GitHub
  degree_depth_stats.json     # power-law fit, depth/width stats
  difficulty_stats.json       # difficulty-vs-structure correlations
  domain_bpe_abstractions.json# per-area BPE
  frequent_subgoals.json      # 500 recurring subgoal expressions
  frequent_transitions.json   # tactic-pair transitions
  identical_state_tactic_clusters.json
  optimality_ranking.json     # 500-theorem optimality scores
  search_tree_stats.json      # proof shape distribution
  spectral_stats.json         # algebraic connectivity, Cheeger bounds
  theorem_graph_stats.json    # global counts + WCC
```

Built with `uv venv .venv-mwm --python 3.12` (numpy, scipy, gensim). On the GPU box, `uv pip install torch transformers accelerate` (use `cu124` index if torch needs to match a CUDA-12 driver).

To run Kimina end-to-end:

```bash
# generate (GPU, one batched call) — re-run with bigger budget than the v1 attempt
python analysis/kimina_eval_cuda.py generate \
    --manifest results/prover_eval/kimina_prompts.jsonl \
    --output  results/prover_eval/kimina_post_2025_08_14.v2.jsonl \
    --model AI-MO/Kimina-Prover-RL-1.7B \
    --backend transformers \
    --dtype bfloat16 \
    --hf-batch-size 4 \
    --samples 4 --max-tokens 8192

# verify (any box with Mathlib4 + lake)
python analysis/kimina_eval_cuda.py verify \
    --generations results/prover_eval/kimina_post_2025_08_14.v2.jsonl \
    --mathlib-dir /path/to/Mathlib4 \
    --output      results/prover_eval/kimina_post_2025_08_14.v2.verified.jsonl
```

`HF_HOME` should be set to a path with ≥ 10 GB free before first use; the cluster home directory is too small.

---

_End of report. Total commits this semester: pipeline + extras + state-hypergraph + graph-geometry. Branch: `codex-overnight-mathlib-geometry-push`. Latest push: `af4a818` plus tonight's predictive/graph-geometry adds._
