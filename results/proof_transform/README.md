# Proof-Transformation Graph + Frequent-Subgraph Search

This directory contains outputs from `analysis/proof_transform_graph.py`,
the implementation of the report's *second extension* (§7 Future Work):
unrolling tactic-level transitions into the underlying proof
transformations Lean actually performed, then mining for repeated
labelled sub-DAGs as candidate new lemmas.

## Pipeline

```bash
# 1. Replay LeanDojo traces in a fresh Dojo session per theorem.
#    Unrolls `rw [a, b, c]` and explicit-args `simp only [a, b]` into
#    one edge per lemma, with real Lean intermediate states. Bare
#    `simp`/`aesop` calls have their underlying lemma set captured via
#    the `?` suggestion variant and emitted as a single rich-label edge
#    (single-pass Dojo can't materialise intermediate states for a bare
#    automation call; the trace is forward-only). Decision procedures
#    (`omega`, `decide`, `ring`, `linarith`, ...) stay as single
#    opaque labelled edges.
python3 analysis/proof_transform_graph.py enrich \
    --mode sample --sample-size 5000 --workers 8 \
    --worker-python /Users/egecakar/Documents/Research/Fall_2025/leandojo/.venv/bin/python \
    --dojo-timeout 1800 --resume

# 2. Build per-theorem labelled DAGs from the enriched traces.
python3 analysis/proof_transform_graph.py build --match state

# 3. Mine frequent connected sub-DAGs (linear paths length 2..6 and
#    small subtrees rooted at branching tactics, size up to 5).
python3 analysis/proof_transform_graph.py mine \
    --min-edges 2 --max-edges 6 --max-tree-edges 5 \
    --min-support 3 --min-distinct-theorems 2 --min-gain 6

# Or run all three end-to-end:
python3 analysis/proof_transform_graph.py all --mode sample --sample-size 5000
```

## Outputs

| File | Description |
|---|---|
| `enriched_traces.jsonl` | One line per traced theorem; flat list of enriched edges. |
| `enrichment_summary.json` | Total runtime, fallback rate, edges by kind. |
| `theorem_dags.jsonl` | One line per theorem; per-theorem labelled DAG (nodes + edges). |
| `build_summary.json` | DAG-build stats. |
| `subgraph_motifs.jsonl` | One line per surviving motif candidate. |
| `subgraph_motifs_top.json` | Top *N* motifs by `complexity_gain` for human review. |
| `subgraph_summary.json` | Mining stats and config. |

## Edge kinds

| Kind | Source | Notes |
|---|---|---|
| `rewrite_step` | one lemma in a `rw [a, b, c]` chain | Real Lean intermediate state. |
| `simp_step` | one lemma in an explicit-args `simp only [a, b]` / `simp_rw` | Real intermediate state; stepwise replay must match the trace's `state_after`. |
| `simp_bundle` | fallback: a `simp` / `simp_all` / `simpa` / `dsimp` / `field_simp` / `aesop` call. | Emitted when (a) the original tactic had no explicit args ("bare automation": Dojo can't materialise intermediate states in a forward-only session because `simp?` already advances past them), or (b) stepwise replay of an explicit-args chain drifted from the trace's `state_after`. Label carries the *sorted lemma multiset*. `is_fallback=true`; `fallback_reason` records `bare_automation_intermediates_unavailable`, `drift`, etc. |
| `decision_opaque` | `omega`, `decide`, `native_decide`, `ring`, `ring_nf`, `linarith`, `nlinarith`, `polyrith`, `norm_num`, `norm_cast`, `push_cast`, `exact_mod_cast`, `positivity`, `gcongr`, `tauto`, `rfl`, `trivial`, `assumption`, `abel`, `abel_nf`, `group`, `noncomm_ring`. | These tactics do not apply a named lemma sequence, so there is nothing to unroll. Edge label = the head; matching uses just (state, head). |
| `term` | `exact`, `refine`, `apply`, `convert` | Single edge per call; premises = LeanDojo's annotated_premises (already a closed lemma list). |
| `structural` | everything else (`intro`, `have`, `obtain`, `cases`, `constructor`, `ext`, `rintro`, ...) | Single edge per call. |

## Subgraph mining

Two enumeration modes are mined together:

- **Linear paths** of edge-length 2..k. Canonical signature is the tuple
  `( node_label[0], edge_label[0], ..., node_label[k] )`, where each
  `node_label` is the full state hash (or goal hash with `--match goal`)
  and each `edge_label` is `(kind, head, lemma_tuple, direction_tuple)`.
- **Small subtrees** rooted at branching tactics (`constructor`, `cases`,
  `induction`, `refine ⟨_,_⟩`, `rcases`). Edge-set sizes 2..5. Canonical
  signature is an AHU-style post-order hash over the labelled DAG.

A motif candidate is recorded only if `support >= min_support`,
`distinct_theorems >= min_distinct_theorems`, and
`complexity_gain >= min_gain`, where

```
complexity_gain = (edge_count - 1) * support - edge_count
```

(saved tactics across all occurrences minus the one-time lemma
definition cost). Motifs whose every edge is `decision_opaque` are
discarded — they wouldn't be useful as a candidate lemma.

## Limitations and known approximations

1. **Bare automation has no intermediate states.** Once `simp?` runs, the
   Lean state has advanced past the would-be intermediates and cannot be
   rewound in a single Dojo session. We record the lemma multiset on a
   single edge and mark it `is_fallback=true`. A two-pass version (run
   each theorem twice — first to harvest lemma lists, second to replay
   with `simp only` chains from scratch) would recover real intermediate
   states at roughly 2× cost.

2. **`calc`/term-mode/`match` proofs are skipped.** LeanDojo extracts
   the inner `by ...` blocks of a `calc` proof as flat steps, but Dojo's
   initial state is the *outer* theorem statement. We detect the
   mismatch by checking `state_pp(state) == trace_before` at the start
   of each step and abort with a `state drift` error.

3. **Per-lemma `simp only [lemma]` is not a perfect surrogate for simp's
   internal algorithm.** simp interleaves rewriting with congruence and
   side-condition solving in a heuristic order; applying lemmas one at a
   time can produce a different intermediate (or fail to apply). The
   pipeline detects divergence by comparing the final replayed state to
   the trace's `state_after`; on mismatch it emits a `simp_bundle`
   fallback edge.

4. **LeanDojo Dojo open is slow.** ~5–60 s per theorem for a Nat-Defs
   sized file; ~2 min for a Matrix-Basic sized file (most of the time
   is `lake env lean` warming up the cached Mathlib oleans). The
   enrichment pass is the corpus-wide bottleneck; use multiple workers
   and `--resume`.

## Comparison to the previous shortcut implementation

The previous version of `proof_transform_graph.py` mined
`(operation_kind, premise_set)` signatures — i.e. repeated `simp` /
`rw` argument bags — rather than labelled sub-DAGs over proof states.
It did not unroll any automation: every `simp` was a single `simp_search`
edge whose label was the (set of) selected simp lemmas. Its outputs
(`primitive_ops.jsonl`, `motif_candidates.jsonl`, `motif_summary.json`)
are now superseded; the new pipeline writes to `enriched_traces.jsonl`,
`theorem_dags.jsonl`, `subgraph_motifs.jsonl`, etc.
