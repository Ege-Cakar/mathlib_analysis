# Prover Eval Commands

Kimina whole-proof eval, assuming model dependencies are installed and the Mathlib checkout has cache artifacts:

```bash
python3 analysis/kimina_eval.py \
  --manifest results/prover_eval/kimina_prompts.jsonl \
  --output results/prover_eval/kimina_post_2025_08_14.jsonl \
  --mathlib-dir /private/tmp/mwm_mathlib4 \
  --backend vllm \
  --model AI-MO/Kimina-Prover-RL-1.7B \
  --samples 8 \
  --limit 25
```

If `vllm` is unavailable, use `--backend transformers`.

ReProver is not installed in this workspace. Run its documented LeanDojo Benchmark 4 evaluation in a ReProver checkout first, then use `results/prover_eval/post_kimina_manifest.jsonl` as the target theorem manifest for any post-Kimina extension.
