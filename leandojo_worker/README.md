# `leandojo_worker/` — Python-3.13 venv for Dojo replay

This subdirectory exists only to host a separate venv for the
LeanDojo-replay worker. The actual worker code lives at
[`../analysis/_proof_transform_worker.py`](../analysis/_proof_transform_worker.py);
it is invoked as a subprocess by the cluster-side dispatcher in
[`../analysis/proof_transform_graph.py`](../analysis/proof_transform_graph.py).

## Why a separate venv

`lean-dojo>=4.20.0` requires Python ≥ 3.13 (it imports `ray` and
`PyGithub` whose 3.13-compatible wheels are the only ones lean-dojo
ships with). The rest of the MWM analysis (numpy, matplotlib, scipy,
gensim) is happiest on Python 3.10. Two venvs sidestep the conflict.

The dispatcher passes the worker venv's Python interpreter to spawn
each subprocess; cross-venv communication happens entirely through
JSONL files (no in-process imports).

## Bootstrap

`../scripts/setup.sh` will create `./.venv-leandojo/` at the repo root
and pip-install this `pyproject.toml` into it. You normally don't have
to do anything inside this directory.
