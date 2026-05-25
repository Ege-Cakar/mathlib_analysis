# Mathlib as a Geometry of Mathematics (MWM)

Empirical study of formal proof graphs extracted from LeanDojo traces of
[Mathlib](https://github.com/leanprover-community/mathlib4). The most
recent direction (and the focus of the "ready for cluster" instructions
below) is the **proof-transformation graph + frequent-subgraph search**
that implements §7's second extension from the report — unrolling
automation tactics into the underlying primitive operations Lean
actually performed, then mining repeated labelled sub-DAGs as
candidate new lemmas.

See [results/proof_transform/README.md](results/proof_transform/README.md)
for the pipeline-specific docs, and [results/report.md](results/report.md)
for the full writeup.

## Quick start (Linux cluster, after `git clone`)

```bash
# 1. One-shot bootstrap (installs elan + uv, creates both venvs,
#    fetches the LeanDojo benchmark JSON, optionally primes Mathlib).
./scripts/setup.sh                     # standard
./scripts/setup.sh --prime-cache       # also pre-download Mathlib trace cache (~30 GB)

# 2. Smoke-test the pipeline (~minutes; 50 theorems, single-file scope).
./.venv/bin/python analysis/proof_transform_graph.py all \
    --mode sample --sample-size 50 --workers 4 \
    --min-support 2 --min-distinct-theorems 2 --min-gain 0

# 3. Full-corpus run (many hours; use a job scheduler).
./.venv/bin/python analysis/proof_transform_graph.py enrich \
    --mode full --workers $(nproc) --resume
./.venv/bin/python analysis/proof_transform_graph.py build
./.venv/bin/python analysis/proof_transform_graph.py mine \
    --min-support 3 --min-distinct-theorems 2 --min-gain 6
```

The setup script is idempotent — re-running it is safe and only redoes
work for missing artifacts. Use `./scripts/setup.sh --help` to see the
flags.

### What the setup script provisions

| Component | Path | Purpose |
|---|---|---|
| `elan` | `~/.elan` | Lean toolchain manager. Bootstrapped from `elan.lean-lang.org` if missing. |
| `uv` | `~/.local/bin/uv` | Python venv + package tool from Astral. Bootstrapped from `astral.sh/uv` if missing. |
| `.venv/` | repo root | Main analysis venv (Python 3.10–3.12). numpy / matplotlib / scipy / gensim / networkx / scikit-learn from [pyproject.toml](pyproject.toml). |
| `.venv-leandojo/` | repo root | Worker venv (Python 3.13). lean-dojo only, from [leandojo_worker/pyproject.toml](leandojo_worker/pyproject.toml). Kept separate because lean-dojo pins Python ≥ 3.13. |
| `data/leandojo/leandojo_benchmark_4/` | repo root | LeanDojo Benchmark 4 split JSON. Either fetched from `$LEANDOJO_BENCHMARK_URL`, extracted from a pre-staged tarball at `data/leandojo/leandojo_benchmark_4.tar.gz`, or re-used from the legacy `Old/data/...` layout. |
| `~/.cache/lean_dojo/` | `$HOME` | Mathlib trace cache. Populated lazily on the first `enrich` run, or eagerly with `setup.sh --prime-cache`. ~30 GB at the pinned Mathlib commit. |

### Cluster-relevant env vars

| Env var | Purpose | Default |
|---|---|---|
| `MWM_LEANDOJO_PYTHON` | Path to the worker Python interpreter. Use this when the worker venv lives on a different volume (e.g. `/scratch`). | `./.venv-leandojo/bin/python` |
| `MWM_LEANDOJO_DATA` | Path to the LeanDojo `random/` split directory. Use this when the benchmark JSON lives on a shared scratch path. | `./data/leandojo/leandojo_benchmark_4/random` (fallback: `./Old/data/...`) |
| `CACHE_DIR` | lean-dojo's trace cache directory. Set to `$SCRATCH/lean_dojo` on shared-storage clusters to avoid filling `$HOME`. | `$HOME/.cache/lean_dojo` |
| `PYTHON_MAIN_VERSION` | Python version `setup.sh` uses for `.venv`. | `3.12` |
| `PYTHON_LEANDOJO_VERSION` | Python version `setup.sh` uses for `.venv-leandojo`. | `3.13` |
| `LEAN_TOOLCHAIN` | Lean toolchain `setup.sh` installs via elan. | `leanprover/lean4:v4.10.0-rc2` |
| `MATHLIB_COMMIT` | Mathlib commit pinned for tracing. Must match the commit baked into the analysis modules and the LeanDojo benchmark. | `29dcec074de168ac2bf835a77ef68bbe069194c5` |

### Performance notes

- **Throughput**: smoke testing on `Mathlib/Data/Nat/Defs.lean` (50 theorems, single worker, macOS) ran at ≈ 2.9 s/theorem after the first warm-up. On a Linux cluster with cold OS file cache the first Dojo open in each worker takes ~30 s to 2 min (lake compiles the modified per-theorem file); subsequent theorems in the same file are much faster. Plan for ~3–10 s/theorem amortized on warm nodes.
- **Parallelism**: each worker holds its own Dojo subprocess, so workers parallelize cleanly. The dispatcher splits the corpus round-robin and writes to a single JSONL with `fcntl` LOCK_EX. Set `--workers $(nproc)` on dedicated nodes; cap lower if you share the node.
- **Disk**: the lean-dojo Mathlib cache is ~30 GB. Put it on scratch with `CACHE_DIR=$SCRATCH/lean_dojo`.
- **Resume**: `enrich --resume` re-reads `enriched_traces.jsonl`, collects already-processed theorem names, and only dispatches the rest. Use this for long full-corpus runs that get interrupted.

## Project layout

```
analysis/                          Python analysis scripts.
  proof_transform_graph.py         Pipeline driver (enrich/build/mine/all).
  _proof_transform_worker.py       LeanDojo Dojo-replay worker (Python 3.13).
  overnight_geometry.py            Dependency-graph and prediction-task pipeline.
  state_hypergraph.py              State-tactic hypergraph + path-finding.
  graph_geometry.py                node2vec, Ollivier-Ricci, centrality.
  spectral_classifier_sweep.py     Spectral baselines.
  predictive_extras.py             Extra predictive sweeps.
  kimina_eval.py / kimina_eval_cuda.py  Theorem-prover evaluation (optional, GPU).
  make_report_figures.py           Regenerate the report figures.
  audit_report_numbers.py          Sanity-check the numbers quoted in the report.

leandojo_worker/                   Pyproject for the .venv-leandojo venv.
scripts/                           Bootstrap and operational scripts.
  setup.sh                         One-shot cluster bootstrap.

data/leandojo/                     LeanDojo Benchmark 4 (downloaded by setup.sh).
results/                           Pipeline outputs (jsonl artifacts gitignored).
  proof_transform/                 Enriched traces, theorem DAGs, motif candidates.
  report.md, report.tex, report.pdf
```

See [agents.md](agents.md) for project-history notes and
[project_memories.md](project_memories.md) for accumulated context.
