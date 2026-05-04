# Mathlib Geometry — Semester Report

_2026-05-04, Ege. All numbers are reproducible from this repo; sources are
linked at the end of each section._

## TL;DR

I treated Mathlib as a graph (nodes = theorems and proof states, edges =
premise citations and tactic transitions) and asked two questions: can we
predict things about a theorem from where it lives in this geometry, and
can we trace a path from one theorem to another along it.

- **Graph topology is clean and meaningful**: dependency degree is heavy-
  tailed (power law α ≈ 2.37); community detection gives modularity 0.59
  with communities aligning to Mathlib top-level areas; proof shapes are
  dominated by linear chains (50,191 of 61,544 proofs).
- **Coarse structure predicts well**: out-degree alone explains
  R² = 0.48 of the variance in proof length (Pearson 0.70). Adding a
  64-d node2vec embedding lifts this to **R² = 0.555**, and combining
  degree + node2vec + approximate betweenness + effective resistance
  gets to **R² = 0.565 (Pearson 0.752)**. Cross-area citation matrices
  and per-area tactic mixes both recover the intuitive structure of
  Mathlib (CategoryTheory is 90 % self-citing, Algebra is the universal
  sink, Analysis/Topology/MeasureTheory are `have`-heavy, etc.).
- **Eigencoordinates of the normalised adjacency don't carry information,
  but other graph-geometric features do**: spectral features add 0.000 R²
  on proof length, while node2vec (random-walk) embeddings add 0.049 R²
  on the same task. So "the geometry helps" — it's just the
  random-walk encoding, not the principal modes of the adjacency.
- **The dep graph is hyperbolic**: 89 % of sampled edges have negative
  Ollivier–Ricci curvature (mean κ = −0.78, median −0.84). The graph
  is locally tree-like; this matches the power-law degree distribution
  and the disjoint-proof-tree topology in § 3.5.
- **A state-tactic hypergraph (261 k nodes, 18,467 true multi-output
  edges) is built and the path-finder works**, but goal-text equality is
  too strict to give a useful "find A → B" tool: even when B literally
  cites A, only 11 % of `exact` citations and 0.6 % of `rw` citations
  produce a state-graph path. The state graph and the proof DAG measure
  different equivalence relations; a useful discovery tool needs goal
  canonicalisation or unification-modulo matching.
- **Prover holdout is ready, GPU run is the missing piece**: 25
  post-2025-08-14 declarations are extracted, `sorry`-prechecked, and
  prompted; a vLLM-first CUDA runner with split generate/verify is
  written. One GPU-day of Kimina + ReProver fills in the prover-success
  column.

## 1. Goal

The original framing was: "find new theorems / new abstractions in
Mathlib." That goal proved hard to quantify directly, so over the
semester it narrowed into two operationalisations:

1. **A → B path-finding**: given two theorems A and B, can we trace a
   reduction from B to A along the structures derived from Mathlib's
   proofs?
2. **Predictive power of the geometry**: can features derived from this
   graph predict something about an individual theorem (its area, its
   difficulty, its citation pattern, its prover success)?

Both reduce to the same underlying question: *is the geometry of
Mathlib a useful object in its own right, or is it just a re-encoding of
namespace-and-name conventions?*

## 2. Dataset

LeanDojo Benchmark 4 (random split). Two derived graphs:

**Proof dependency graph** (`results/paths/dependency_edges.jsonl`,
`results/paths/dependency_stats.json`):

| | Count |
|---|---|
| Theorems indexed | 122,517 |
| Theorems with traced tactics | 61,544 |
| Tactic invocations | 259,580 |
| Premise occurrences | 418,704 |
| **Unique directed citation edges** | **309,396** ✓ matches expected |
| Unique nodes | 138,333 |

**State-tactic hypergraph** (`results/state_graph/state_graph_stats.json`):

| | Count |
|---|---|
| Unique single-goal states | 261,655 |
| Unique hyperedges | 261,107 |
| **True multi-output (split-tactic) hyperedges** | **18,467** |
| Output-size tail | up to 17 outputs |

Multi-goal `state_after` strings are split per `case ...` block, so
chains across `rintro` / `cases` / `all_goals` link up correctly.

**Topology** (`Old/data/degree_depth_stats.json`,
`Old/data/theorem_graph_stats.json`,
`Old/data/community_stats.json`,
`Old/data/search_tree_stats.json`):

- Degree distribution heavy-tailed: power-law α = 2.37 above x_min = 43
  (KS = 0.028, n_tail = 864). Lognormal also fits (KS = 0.144);
  exponential rejected (p ≈ 0).
- 138,333 nodes split across 44,479 weakly-connected components, with
  one giant component of 92,594 nodes.
- **Louvain community detection**: 240 communities, modularity
  **0.5924**. Communities align with mathematical fields:
    - Largest community (15,488 nodes): dominated by Algebra (4,750),
      RingTheory (3,245), Data (1,874), NumberTheory (1,351),
      Analysis (856).
    - Second (14,737): Analysis (3,893), MeasureTheory (3,249),
      Topology (2,179), Data (1,383), Algebra (1,105).
    - Third (11,525): Data (3,712), Algebra (1,260), Combinatorics
      (1,122), Order (977).
- Proof shape distribution: 50,191 LINEAR / 5,893 BALANCED /
  4,801 DEEP_NARROW / 655 BUSHY / 4 WIDE_SHALLOW. Most Mathlib proofs
  are linear chains.

**Cross-ecosystem comparison** (`Old/data/comparative_stats.json`):

|  | Mathlib | LEAN-GitHub |
|---|---|---|
| Theorems | 61,544 | 20,446 |
| Total tactics | 259,580 | 218,866 |
| Unique tactics | 274 | 1,093 |
| Mean proof length | 4.22 | 10.7 |
| Median proof length | 2 | 4 |
| Max proof length | 128 | 640 |

Mathlib proofs are markedly shorter and use a smaller tactic vocabulary
than the broader Lean-GitHub corpus.

## 3. What worked

### 3.1 Degree predicts proof difficulty (positive, replicated)

Both an earlier-semester analysis and tonight's regression find the same
strong correlation:

- `Old/data/difficulty_stats.json` (61,542 theorems): Pearson
  correlation between out-degree and proof difficulty:
    - proof length: **r = 0.761** (Spearman ρ = 0.624)
    - unique tactics used: r = 0.707 (ρ = 0.607)
    - tactic diversity: r = 0.609
    - max branching: r = 0.418
- Tonight's ridge regression on `log1p(tactic_count)`
  (`results/predictive/proof_length_regression.json`):

| Feature set | Test R² | Test ρ | Test RMSE |
|---|---|---|---|
| mean predictor (baseline) | 0.000 | 0.000 | 0.685 |
| **out-degree + in-degree** | **0.483** | **0.695** | 0.493 |
| area one-hot | 0.046 | 0.216 | 0.670 |
| log statement length | 0.017 | 0.132 | 0.680 |
| spectral (low10+high10) | **0.003** | 0.052 | 0.685 |
| degree + area | 0.492 | 0.702 | 0.489 |
| **degree + area + statement length** | **0.498** | **0.706** | **0.486** |
| spectral + degree | 0.483 | 0.695 | 0.493 |
| all features | 0.498 | 0.706 | 0.486 |

So **degree alone explains ~half the variance in proof length**, and
adding area + statement length adds a couple of points. Spectral
features add zero. The two analyses (semester, tonight) agree.

### 3.2 The dependency graph reflects Mathlib's areas

Even though spectral coords can't predict the 27-way category label
(see § 4.1), the *citation pattern* of the dependency graph matches
intuitive structure (`results/predictive/cross_area_dep_matrix.svg`):

- **CategoryTheory: 90 % self-citing** — most isolated subfield.
- **Algebra: universal sink** — dominant external target for FieldTheory
  (46 %), RingTheory (44 %), RepresentationTheory (50 %), Tactic (52 %),
  Deprecated (50 %).
- **Probability cites MeasureTheory** more than itself (28 % vs 26 %) —
  clean inter-area dependency.
- **AlgebraicTopology ↔ CategoryTheory** (29 % from AT into CT).
- **Condensed → CategoryTheory** (48 %), nearly nothing self-citing
  (4 %); makes sense given Condensed is a recent reformulation.

Louvain community detection (from §2) corroborates this: the partition
reaches modularity 0.59, with communities lining up with Mathlib's
top-level areas. So the graph carries area-level structural information
even though the eigencoordinates of its normalised adjacency, taken
naively, do not encode it discriminatively (see § 4).

### 3.3 Tactic style differs by area

Source: `results/predictive/tactic_mix_by_area.json`,
`results/predictive/tactic_mix_by_area.svg`. Top tactic head per source
area, row-normalised:

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

Patterns: most areas are `rw` / `simp` first; analysis-flavoured areas
(Analysis, Topology, MeasureTheory, Probability, Computability) lean on
`have` (cut-style proofs); Init prefers term-style (`apply`, `exact`);
RepresentationTheory and Condensed are simp-heavy.

### 3.4 Universal proof macros (BPE on tactic sequences)

Source: `Old/data/bpe_macro_abstractions.json`,
`Old/data/domain_bpe_abstractions.json`.

Byte-pair encoding applied over tactic-head sequences yields a small
vocabulary of recurring macros. Top global merge:

```
"rw -> [0] -> exact"   ΔC = 4,371   occurrences = 7,895
```

That single 3-step macro (rewrite, then a positional placeholder, then
close with `exact`) accounts for an enormous compression in the corpus.
Per-area BPE confirms it: Data, Algebra, Analysis, Topology, and
MeasureTheory all surface the same `rw → [0] → exact` macro as their
top merge. So one stylised proof skeleton dominates Mathlib regardless
of area.

`Old/data/abstraction_candidates.json` extends this to longer 3-7-grams
both globally and per module; `Old/data/frequent_subgoals.json` lists
the 500 most common subgoal *expressions* (top: `f x = g x`,
shared across 44 distinct theorems) — these are candidates for
"shareable lemma" abstractions.

### 3.5 Proof-state sharing across theorems is rare but real

Source: `Old/data/identical_state_tactic_clusters.json`,
`results/state_graph/state_path_sweep_goal.json`.

33 clusters of theorems share an identical (state, tactic, state_after)
triple — a literal proof step that recurs across multiple theorems with
the same hypothesis context. The top cluster is shared by 12 theorems
in the Geometry / Orientation namespace.

The state-tactic hypergraph (§ 2) makes this measurable globally. Within
a single theorem the algorithm correctly recovers the recorded proof
(within-theorem sanity check passes for `ADEInequality.Admissible.one_lt_sumInv`,
depth 3, AND/OR discharge in 3 expansions). Across theorems it sees
shared states only when two theorems' goal texts are literally equal,
which is rare; see § 4.4.

### 3.6 Richer graph-geometric features (the spectral rebuttal)

The eigendecomposition of the normalised adjacency added nothing on top of
degree (§ 4.1–4.3). Tonight we re-tested the same predictive task with
three more graph-geometric features that have no reason to collapse to
degree:

| Feature set (proof-length regression) | n | Test R² | Test ρ |
|---|---|---|---|
| degree | 57,601 | 0.506 | 0.711 |
| spectral (low10+high10), § 4.3 | 57,601 | 0.003 | 0.052 |
| **node2vec (64-d, walks-per-node 10, length 40)** | 57,601 | **0.162** | **0.403** |
| approximate betweenness (200 random sources) | 57,601 | 0.183 | 0.429 |
| effective resistance to top-15 hubs (via 64 eigvecs) | 57,601 | 0.005 | 0.074 |
| **node2vec + degree** | 57,601 | **0.555** | **0.745** |
| **all combined (degree + node2vec + BC + resistance)** | 57,601 | **0.565** | **0.752** |

So **node2vec is the geometric feature that *does* carry information**:
on its own it reaches R² = 0.162 / ρ = 0.403 — the eigencoordinates of
the same graph reach 0.003. Adding node2vec on top of degree lifts
R² from 0.506 to 0.555 (+0.049 absolute, +9.7 % relative). Combining
all four features gets R² = 0.565.

Source: `analysis/graph_geometry.py`,
`results/graph_geometry/node2vec_proof_length.json`,
`results/graph_geometry/joint_proof_length.json`.

The two extra side products are also worth keeping:

**Top-hubs-by-indegree** (`results/graph_geometry/resistance_summary.json`)
— sanity check that the dep graph is meaningful at the hub level:

```
rfl (in_deg 2,645), Eq.symm (1,493), mul_comm (1,462),
mul_assoc (1,267), mul_one (1,171), one_mul (1,015),
add_comm (950), le_antisymm (814), eq_comm (701),
LE.le.trans (699), …
```

Exactly the basic algebra and equality lemmas one would expect at the
foundation.

**Ollivier–Ricci curvature** (`results/graph_geometry/ollivier_ricci.json`).
Sampled 5,000 edges, kept 3,320 with neighbourhoods ≤ 50, computed
κ(u, v) = 1 − W₁(unif N(u), unif N(v)) / d(u, v) by solving the
transportation LP per edge with `scipy.optimize.linprog`:

| Quantity | Value |
|---|---|
| edges evaluated | 3,320 |
| mean κ | **−0.781** |
| median κ | −0.840 |
| min / max κ | −1.788 / +0.543 |
| fraction κ < 0 | **88.7 %** |
| fraction κ < −1 (strong bottleneck) | 35.7 % |

Discrete Ricci curvature this strongly negative is the signature of a
**hyperbolic / tree-like** graph. This is consistent with the heavy-
tailed power-law degree distribution from § 2 and the proof-tree
structure of the state graph (§ 3.5): citations radiate from a few
hubs, locally the graph looks like a tree, and there is little
triangle-rich "Euclidean" structure. The most negative-κ edges connect
semantically distant theorems via atypical citations
(e.g. `spectrum.map_polynomial_aeval_of_degree_pos ↔ symm`,
`Finset.filter_val ↔ Nat.divisors_filter_squarefree`) — candidates for
abstraction / refactor analysis.

### 3.7 Post-cutoff prover holdout (built, not yet run)

Source: `results/prover_eval/post_kimina_manifest.jsonl`,
`results/prover_eval/post_kimina_manifest_stats.json`,
`results/prover_eval/kimina_prompts.jsonl`,
`analysis/kimina_eval_cuda.py`.

- 25 declarations introduced after 2025-08-14 (cutoff for
  Kimina-Prover-RL training).
- All 25 pass the `by sorry` precheck (the surrounding file still
  type-checks with the proof body removed).
- Anchor features for 17/25 (identifiers in the signature that map to
  existing graph nodes; mean of their spectral coords).
- Stratified across 14 mathematical areas (Analysis 3, Topology 2,
  GroupTheory 2, RingTheory 2, Data 2, Order 2, NumberTheory 2,
  Tactic 2, Combinatorics 2, Dynamics 2, Algebra 1, MeasureTheory 1,
  LinearAlgebra 1, Probability 1).

`analysis/kimina_eval_cuda.py` is a CUDA-first runner: a
GPU `generate` pass (one batched vLLM call,
`SamplingParams(n=samples)`, configurable `--tensor-parallel-size`,
`--gpu-memory-utilization`, bf16 default) and a decoupled CPU
`verify` pass (per-sample `lake env lean`). Output schema includes
`lean_code`, `verified`, `lean_output`, plus a `.summary.json` with
sample-level pass and theorem-level `pass@n`.

## 4. What didn't work

### 4.1 Spectral category prediction

Source: `results/spectral/classifier_sweep.json`,
`results/spectral/spectral_metrics.json`.

Predict the 27-way Mathlib top-level area label of a node from its
spectral coordinates.

| Method | Acc | Macro-F1 |
|---|---|---|
| majority | 0.165 | 0.011 |
| **namespace prefix** | **0.737** | **0.753** |
| degree only (mlp) | 0.172 | 0.019 |
| low10 + high10 (mlp) | 0.162 | 0.031 |
| low32 + high32 (mlp) | 0.160 | 0.025 |
| low10 + degree (mlp) | **0.177** | 0.021 |
| low32 + high32 + degree (mlp) | 0.174 | 0.022 |

Best spectral method beats majority by 1.2 percentage points and is
roughly 56 percentage points behind the trivial namespace baseline.

### 4.2 Spectral link prediction (~ degree only)

Source: `results/spectral/link_prediction.json`. 10,000 positive edges
+ 10,000 random non-edges, logreg.

| Method | Test acc | Test ROC-AUC |
|---|---|---|
| degree-only | 0.785 | 0.875 |
| full spectral + degree (258-d) | 0.789 | **0.881** |

Spectral coords add **+0.006 AUC** over degree alone — this is the
preferential-attachment heuristic doing almost all the work.

### 4.3 Spectral tactic-head and proof-length prediction

Source: `results/spectral/tactic_head_prediction.json`,
`results/predictive/proof_length_regression.json`.

Edge-level tactic-head classification (`rw / simp / have / exact /
refine / simp_rw / simpa / apply`):

| Method | Acc | Macro-F1 |
|---|---|---|
| majority (`rw`) | 0.339 | — |
| degree-only logreg | 0.349 | 0.103 |
| spectral concat logreg | 0.346 | 0.110 |

And on proof length (§ 3.1) spectral features alone get R² = 0.003 vs
0.483 for degree, and adding spectral on top of degree does not move R².

So across **four** predictive tasks (category, link, tactic head, proof
length), the eigencoordinates of the normalised adjacency add nothing
on top of degree-based features. The principal modes of this graph are
essentially the degree distribution.

### 4.4 Path-finding by goal-text equality is too strict

Source: `results/state_graph/state_path_sweep_goal.json`,
`results/paths/path_sweep.json`.

Three sweeps, all measuring "given (start, target), is there a path?":

| Sweep | n | reachable | rate | comment |
|---|---|---|---|---|
| Random pairs in proof DAG (depth ≤ 8) | 2,000 | 1 | 0.05 % | most theorems have only ~hundreds of ancestors |
| Random theorem pairs in state graph (depth ≤ 12) | 2,000 | 2 | 0.10 % | goal texts almost never match across theorems |
| Dep-graph edges (B literally cites A) in state graph | 1,000 | 25 | **2.5 %** | the apples-to-apples test |

Per-tactic-head breakdown of the dep-graph cross-check is the most
informative chart in the report:

| Citation tactic head | n | state-graph reachable | rate |
|---|---|---|---|
| **`exact`** | 100 | 11 | **0.110** |
| **`refine`** | 54 | 5 | **0.093** |
| `induction` | 9 | 1 | 0.111 |
| `have` | 83 | 3 | 0.036 |
| `apply` | 40 | 1 | 0.025 |
| `simp` | 129 | 2 | 0.016 |
| `rw` | 359 | 2 | 0.006 |
| `simp_rw` | 45 | 0 | 0.000 |

This is exactly the pattern we'd expect: `exact A` / `refine A` make
the current goal equal to A's statement, so the state graph confirms
the connection ~10 % of the time (limited by binder/universe/implicit
differences). `rw [A]` / `simp [A]` use A as a rewrite rule and the
goal text never becomes A's statement, so the state graph almost never
finds those.

A concrete failure case: in
`UniformConvergenceCLM.continuousSMul`, A is `Bornology.IsVonNBounded.image`
and the citation is `(h𝔖₃ s hs).image u` — A is invoked term-style
inside a lambda inside an argument to *another* theorem. There is never
a goal `⊢ IsVonNBounded 𝕜₂ (⇑f '' s)` during B's proof. The proof DAG
correctly counts this as an edge; the state graph correctly concludes
no path exists *under text equality*.

### 4.5 What to take from § 4

The eigendecomposition of the dep-graph adjacency is not a useful node
embedding for the predictive tasks we care about — but this is not the
same as "the graph carries no embeddable information". A *random-walk*
embedding of the same graph (node2vec, § 3.6) does carry signal: R² on
proof length goes from 0.003 (spectral) to 0.162 (node2vec) on the
same target. The takeaway is that the principal modes of the
normalised adjacency are dominated by degree, but the local
neighbourhood structure that random walks see is real.

The state-tactic hypergraph correctly implements the algorithm I
wanted, but goal-text equality is too strict an equivalence to make
the search useful for discovery on arbitrary pairs. Goal canonicalisation
is the next concrete fix.

## 5. What's queued

In priority order — all are concrete extensions, not new directions:

1. **Run Kimina + ReProver on the 25-theorem holdout** (~1 GPU-day).
   Fills the prover-success column. The CUDA runner is written
   (`analysis/kimina_eval_cuda.py`); the manifest and prompts are
   ready. Once we have a verified pass/fail per theorem, we can train
   the small classifier "predict prover success from
   (degree, area, anchor spectral coords, statement length)" that
   was the original motivation for §3.6.
2. **Goal canonicalisation in the state graph.** Alpha-rename binders,
   sort hyps, normalise universe levels, then re-hash. Conservative
   guess: dep-edge reachability climbs from 2.5 % to 20–30 %, and the
   `exact` / `refine` rates approach 100 %. Cheap to add.
3. **Subterm matching for `rw`-style use.** Index every state by the
   set of subterms it contains; "B uses A" matches if some state in B
   has A's LHS as a subterm. Catches the 359 `rw` edges in §4.4.
4. **Use `Old/data/frequent_subgoals.json` as oracles in the
   discharge solver.** The 500 most common subgoals across Mathlib are
   plausible "free leaves". Test what fraction of the dep-graph edges
   become solvable with this oracle set.
5. **Forward + backward path search**: the current `or-path` is
   already bidirectional, but we should also walk premise → theorem
   (downstream) to enable "what is provable from A?". The dep-graph
   adjacency is already there; minor change.

## 6. Honest summary

What we know:

- The dep graph is a real object: heavy-tailed degree (power law
  α = 2.37), 240 communities at modularity 0.59, communities aligned
  with Mathlib's areas, **predominantly hyperbolic Ollivier–Ricci
  curvature** (mean κ = −0.78 over 3,320 sampled edges).
- Coarse structural features predict proof difficulty: degree alone
  hits R² = 0.51 / ρ = 0.71 on proof length; **adding a node2vec
  random-walk embedding lifts this to R² = 0.555, and combining all
  graph-geometric features gets to R² = 0.565 (ρ = 0.752)**.
- Cross-area citation patterns and per-area tactic mixes are
  interpretable and consistent with mathematical intuition.
- A single 3-step BPE macro (`rw → [0] → exact`) accounts for ~8 k
  occurrences globally and is the top merge in every major area.
- Top theorems by indegree: `rfl`, `Eq.symm`, `mul_comm`, `mul_assoc`,
  `mul_one`, `one_mul`, `add_comm`, `le_antisymm` — the graph correctly
  identifies the basic algebra/equality lemmas as foundational.

What we know doesn't work:

- The eigencoordinates of the normalised adjacency, taken naively, do
  not encode the area / link / tactic / difficulty information that
  degree-and-namespace already carry. **But a random-walk embedding
  (node2vec) of the same graph does** — so the failure was the
  encoding, not the underlying graph.
- Path-finding under strict goal-text equality on the state graph
  cannot recover the bulk of the dep-graph's citations because of
  term-mode use, rewriting, and instantiation.

What's missing:

- A prover-success column for the 25-theorem holdout. The runner is
  written; one GPU-day fills it.
- A goal-canonicalisation layer for the state graph. Half a day of
  implementation; potentially a 10× lift on dep-edge reachability.
- A semantic version of "find new theorems": with
  goal-canonicalisation + frequent-subgoal oracles + forward search,
  the discharge solver becomes a real tool for asking "given A, what
  can be proved with A as oracle?".

Negative results are negative for clean reasons (the eigendecomposition
collapses to degree; goal-text equality is too strict). The blockers to
turning each negative into a positive are concrete, not methodological.

## 7. Reproducibility map

```
analysis/
  overnight_geometry.py       # build dep graph, spectral, post-cutoff manifest
  spectral_classifier_sweep.py
  extra_experiments.py        # path-sweep, link-pred, tactic-pred (proof DAG)
  state_hypergraph.py         # state-tactic hypergraph, or-path, discharge, sweep
  predictive_extras.py        # proof-length regression, dep matrix, tactic-by-area
  graph_geometry.py           # node2vec, betweenness, resistance, Ollivier-Ricci, joint
  kimina_eval_cuda.py         # CUDA-first vLLM runner (split generate/verify)
  kimina_eval.py              # legacy single-pass runner

results/
  paths/                       # proof-DAG outputs
  state_graph/                 # state-tactic hypergraph outputs
  spectral/                    # spectral coords + classifier sweeps
  predictive/                  # proof-length regression + heatmaps
  graph_geometry/              # node2vec, BC, resistance, OR curvature, joint
  prover_eval/                 # post-2025-08-14 manifest + prompts

Old/data/                      # earlier-semester analyses (Mar–Apr 2026)
  abstraction_candidates.json  # 3-7-grams of tactics, global + per-module
  bpe_macro_abstractions.json  # 30 BPE merge steps over tactic sequences
  community_stats.json         # Louvain communities (240, modularity 0.59)
  comparative_stats.json       # Mathlib vs LEAN-GitHub
  degree_depth_stats.json      # power-law fit, depth/width stats
  difficulty_stats.json        # difficulty-vs-structure correlations
  domain_bpe_abstractions.json # per-area BPE
  frequent_subgoals.json       # 500 recurring subgoal expressions
  frequent_transitions.json    # tactic-pair transitions
  identical_state_tactic_clusters.json
  optimality_ranking.json      # 500-theorem optimality scores
  search_tree_stats.json       # proof shape distribution
  spectral_stats.json          # algebraic connectivity, Cheeger bounds
  theorem_graph_stats.json     # global counts + WCC
```

Built with `uv venv .venv-mwm --python 3.12`; numpy 2.4 only. Add
`torch / vllm / transformers` on the GPU box for the prover step.
