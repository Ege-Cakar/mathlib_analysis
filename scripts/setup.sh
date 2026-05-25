#!/usr/bin/env bash
# scripts/setup.sh — one-shot bootstrap for the MWM proof-transformation
# pipeline. Designed to be run once after `git clone` on a fresh Linux
# cluster node (also works on macOS with brew preinstalled).
#
# What it does, in order:
#   1. Install elan (the Lean toolchain manager) if missing.
#   2. Install uv (Astral's Python package/venv tool) if missing.
#   3. Create .venv/ at the repo root (Python 3.10–3.12) and install
#      the main analysis dependencies from pyproject.toml.
#   4. Create .venv-leandojo/ at the repo root (Python 3.13) and
#      install lean-dojo from leandojo_worker/pyproject.toml.
#   5. Fetch the LeanDojo Benchmark 4 split JSON into ./data/leandojo/
#      if not already present (provide your own URL via
#      LEANDOJO_BENCHMARK_URL or place the tarball at
#      data/leandojo/leandojo_benchmark_4.tar.gz before running).
#   6. (Optional, gated by --prime-cache) Trigger lean-dojo's trace()
#      against Mathlib at the pinned commit. This downloads the
#      pre-traced repo from LeanDojo's remote cache (~30 GB, fast on
#      the cluster's network) or builds it locally (hours-days).
#
# Re-runnable: every step is idempotent and skips when the artifact is
# already present. Use --force to redo a specific step:
#   ./scripts/setup.sh --force-venvs   # delete and recreate both venvs
#   ./scripts/setup.sh --force-data    # re-download the benchmark
#   ./scripts/setup.sh --prime-cache   # also prime the Mathlib cache
#
# Customizable via env vars:
#   PYTHON_MAIN_VERSION       (default: 3.12)   # for .venv
#   PYTHON_LEANDOJO_VERSION   (default: 3.13)   # for .venv-leandojo
#   LEAN_TOOLCHAIN            (default: leanprover/lean4:v4.10.0-rc2)
#   LEANDOJO_BENCHMARK_URL    (URL or local path to the tarball)
#   MATHLIB_COMMIT            (default: the commit baked into the
#                              analysis modules)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_MAIN_VERSION="${PYTHON_MAIN_VERSION:-3.12}"
PYTHON_LEANDOJO_VERSION="${PYTHON_LEANDOJO_VERSION:-3.13}"
LEAN_TOOLCHAIN="${LEAN_TOOLCHAIN:-leanprover/lean4:v4.10.0-rc2}"
MATHLIB_COMMIT="${MATHLIB_COMMIT:-29dcec074de168ac2bf835a77ef68bbe069194c5}"
MATHLIB_URL="https://github.com/leanprover-community/mathlib4"
LEANDOJO_BENCHMARK_URL="${LEANDOJO_BENCHMARK_URL:-}"

FORCE_VENVS=0
FORCE_DATA=0
PRIME_CACHE=0
for arg in "$@"; do
    case "$arg" in
        --force-venvs)  FORCE_VENVS=1 ;;
        --force-data)   FORCE_DATA=1 ;;
        --prime-cache)  PRIME_CACHE=1 ;;
        -h|--help)
            sed -n '2,40p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*" >&2; }

# ---- 1. elan ---------------------------------------------------------
if ! command -v elan >/dev/null 2>&1; then
    if [[ -x "$HOME/.elan/bin/elan" ]]; then
        export PATH="$HOME/.elan/bin:$PATH"
    else
        log "Installing elan (Lean toolchain manager)..."
        curl -sSf https://elan.lean-lang.org/elan-init.sh \
            | bash -s -- -y --default-toolchain none
        export PATH="$HOME/.elan/bin:$PATH"
    fi
fi
log "elan: $(elan --version)"

# Make sure the toolchain LeanDojo will need is installed.
elan toolchain install "$LEAN_TOOLCHAIN" >/dev/null
log "lean toolchain: $LEAN_TOOLCHAIN ready"

# ---- 2. uv -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        log "Installing uv (Python package manager)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
log "uv: $(uv --version)"

# ---- 3. Main analysis venv ------------------------------------------
if [[ $FORCE_VENVS -eq 1 ]]; then
    rm -rf .venv .venv-leandojo
fi
if [[ ! -d .venv ]]; then
    log "Creating .venv (Python $PYTHON_MAIN_VERSION) for MWM analysis..."
    uv venv .venv --python "$PYTHON_MAIN_VERSION" --seed
    uv pip install --python ./.venv/bin/python -e .
else
    log ".venv/ already exists — skipping (pass --force-venvs to redo)."
fi

# ---- 4. LeanDojo worker venv ----------------------------------------
if [[ ! -d .venv-leandojo ]]; then
    log "Creating .venv-leandojo (Python $PYTHON_LEANDOJO_VERSION) for the lean-dojo worker..."
    uv venv .venv-leandojo --python "$PYTHON_LEANDOJO_VERSION" --seed
    uv pip install --python ./.venv-leandojo/bin/python -e ./leandojo_worker
else
    log ".venv-leandojo/ already exists — skipping (pass --force-venvs to redo)."
fi

# Verify the worker can import lean-dojo (warming up the bytecode cache
# pays off: the first cold import of PyGithub can take 10+ minutes on
# macOS due to Spotlight scanning).
log "Warming worker venv (importing lean_dojo once)..."
./.venv-leandojo/bin/python -c "import lean_dojo; print('lean_dojo', lean_dojo.__version__, 'ok')"

# ---- 5. LeanDojo benchmark data --------------------------------------
DATA_DIR="$REPO_ROOT/data/leandojo"
SPLIT_DIR="$DATA_DIR/leandojo_benchmark_4/random"
TARBALL="$DATA_DIR/leandojo_benchmark_4.tar.gz"

if [[ $FORCE_DATA -eq 1 ]]; then
    rm -rf "$DATA_DIR"
fi
mkdir -p "$DATA_DIR"

if [[ -d "$SPLIT_DIR" ]]; then
    log "Benchmark split already at $SPLIT_DIR — skipping download."
elif [[ -d "$REPO_ROOT/Old/data/leandojo/leandojo_benchmark_4/random" ]]; then
    log "Using legacy Old/data/leandojo/leandojo_benchmark_4 layout."
elif [[ -f "$TARBALL" ]]; then
    log "Extracting existing $TARBALL..."
    tar -xzf "$TARBALL" -C "$DATA_DIR"
elif [[ -n "$LEANDOJO_BENCHMARK_URL" ]]; then
    log "Downloading LeanDojo Benchmark 4 from $LEANDOJO_BENCHMARK_URL..."
    if [[ "$LEANDOJO_BENCHMARK_URL" == /* ]]; then
        cp "$LEANDOJO_BENCHMARK_URL" "$TARBALL"
    else
        curl -L --fail -o "$TARBALL" "$LEANDOJO_BENCHMARK_URL"
    fi
    tar -xzf "$TARBALL" -C "$DATA_DIR"
else
    warn "No LeanDojo Benchmark 4 split found."
    warn "Either:"
    warn "  - Place the tarball at $TARBALL and re-run, OR"
    warn "  - Set LEANDOJO_BENCHMARK_URL=<url-or-local-path> and re-run, OR"
    warn "  - Symlink an existing copy:"
    warn "      ln -s /path/to/your/leandojo_benchmark_4 $DATA_DIR/"
    warn "  - Or expose it via MWM_LEANDOJO_DATA=<path>/random when invoking the pipeline."
fi

# ---- 6. (Optional) prime the Mathlib trace cache ---------------------
if [[ $PRIME_CACHE -eq 1 ]]; then
    log "Priming the lean-dojo Mathlib cache at commit ${MATHLIB_COMMIT:0:8}..."
    log "(downloads from dl.fbaipublicfiles.com/lean-dojo if available; otherwise traces locally — hours)"
    ./.venv-leandojo/bin/python - <<PYEOF
from lean_dojo import LeanGitRepo, trace
repo = LeanGitRepo("$MATHLIB_URL", "$MATHLIB_COMMIT")
trace(repo)
print("Mathlib trace cache ready.")
PYEOF
else
    log "Skipping --prime-cache (LeanDojo will trace Mathlib lazily on first enrich)."
fi

# ---- Done ------------------------------------------------------------
log ""
log "Setup complete."
log ""
log "Smoke-test the pipeline:"
log "  ./.venv/bin/python analysis/proof_transform_graph.py all \\"
log "      --mode sample --sample-size 50 --workers 4 \\"
log "      --min-support 2 --min-distinct-theorems 2 --min-gain 0"
log ""
log "Full corpus run (many hours; use a job scheduler):"
log "  ./.venv/bin/python analysis/proof_transform_graph.py enrich \\"
log "      --mode full --workers \$(nproc) --resume"
log "  ./.venv/bin/python analysis/proof_transform_graph.py build"
log "  ./.venv/bin/python analysis/proof_transform_graph.py mine \\"
log "      --min-support 3 --min-distinct-theorems 2 --min-gain 6"
