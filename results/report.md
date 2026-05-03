# Overnight Mathlib Geometry Results

## Certified dependency graph
- Unique LeanDojo premise edges: `309396`.
- Matches expected 309396 edge count: `True`.
- Nodes with theorem/premise metadata: `138333`.

## Spectral category prediction
- `majority`: accuracy `0.165`, macro-F1 `0.011`.
- `degree_only`: accuracy `0.011`, macro-F1 `0.006`.
- `namespace_prior`: accuracy `0.737`, macro-F1 `0.753`.
- `low_5`: accuracy `0.108`, macro-F1 `0.042`.
- `low_10`: accuracy `0.113`, macro-F1 `0.041`.
- `high_5`: accuracy `0.128`, macro-F1 `0.048`.
- `high_10`: accuracy `0.136`, macro-F1 `0.048`.
- `low5_high5`: accuracy `0.137`, macro-F1 `0.045`.
- `low10_high10`: accuracy `0.129`, macro-F1 `0.052`.
- Train/test nodes: `72753` / `18207`.

## Stronger spectral classifiers
- Baselines: majority accuracy `0.165`, namespace-prior accuracy `0.737`.
- Best accuracy: `0.177` using `mlp` on `low10+degree` (macro-F1 `0.021`).
- Best macro-F1: `0.032` using `mlp` on `low32` (accuracy `0.156`).
- Interpretation: the neural classifier finds at most weak signal; top-level category remains mostly not captured by these eigencoordinates.

## Post-Kimina prover eval
- Manifest rows ready for Kimina: `25`.
- `by sorry` precheck run locally: `True`.
- Kimina prompts are written to `results/prover_eval/kimina_prompts.jsonl`.
- Post-Kimina spectral anchor features are written to `results/spectral/post_kimina_anchor_features.csv` (`17/25` with anchors).
- Kimina model run not completed in this local workspace.

## ReProver status
- ReProver is not installed in this workspace. The local outputs therefore contain the manifest and commands/data needed for the parallel run, not ReProver proof results.

## Interpretation
- Dependency paths are certified only in the sense that each edge appears in LeanDojo premise annotations from existing Mathlib proofs.
- State-transition shortcuts and model generations are not counted unless Lean verifies them.
- Post-Kimina, after 2025-08-14, is the clean holdout for Kimina-style models; older post-LeanDojo results are contamination-caveated for Kimina.
