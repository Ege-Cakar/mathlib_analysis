#!/usr/bin/env python3
"""Proof-transformation graph + frequent-subgraph search (second extension).

Pipeline:

  enrich  →  Re-run automation tactics (simp / simp_all / simp_rw / dsimp /
             field_simp / simpa / aesop, plus rw-style chains) through
             LeanDojo's Dojo REPL. For bare automation calls, query Lean
             with the `?` suggestion variant to extract the explicit lemma
             list, then replay each lemma one at a time to capture real
             intermediate proof states. Decision procedures (omega, ring,
             decide, ...) stay as opaque single-edge entries. Output:
             results/proof_transform/enriched_traces.jsonl.

  build   →  Build per-theorem labelled DAGs from enriched traces. Nodes
             are state hashes; edges carry the (kind, head, lemma-tuple,
             direction-tuple) label.  Output: theorem_dags.jsonl.

  mine    →  Frequent labelled-subgraph mining: linear paths length 2..k
             and small subtrees rooted at branching tactics. Match by
             full state hash (or goal hash with --match goal). Score
             motifs by complexity_gain = (edges-1)*support - edges.
             Output: subgraph_motifs.jsonl + subgraph_motifs_top.json +
             subgraph_summary.json.

  all     →  enrich + build + mine.

The enrich step is implemented as a worker-subprocess pool because
`lean_dojo` requires Python ≥ 3.13 while the rest of the MWM analysis
runs on Python 3.10. Workers use the venv at `./.venv-leandojo/` by
default (created by `scripts/setup.sh`); override with
$MWM_LEANDOJO_PYTHON or --worker-python.

Other env vars honored:
  $MWM_LEANDOJO_DATA — directory holding the LeanDojo Benchmark 4
                       `random/{train,val,test}.json` split.  Defaults
                       to ./data/leandojo/leandojo_benchmark_4/random
                       (set up by scripts/setup.sh) or the legacy
                       ./Old/data/... layout if present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_dir() -> Path:
    """Return the LeanDojo Benchmark 4 `random/` split directory.

    Resolution order:
      1. $MWM_LEANDOJO_DATA — explicit override (e.g. a cluster scratch
         path).
      2. $ROOT/data/leandojo/leandojo_benchmark_4/random — the layout
         `scripts/setup.sh` extracts the tarball into.
      3. $ROOT/Old/data/leandojo/leandojo_benchmark_4/random — legacy
         layout from the original local checkout.
    """
    env = os.environ.get("MWM_LEANDOJO_DATA")
    if env:
        return Path(env)
    new = ROOT / "data/leandojo/leandojo_benchmark_4/random"
    if new.exists():
        return new
    return ROOT / "Old/data/leandojo/leandojo_benchmark_4/random"


def _resolve_worker_python() -> Path:
    """Return the Python interpreter the LeanDojo worker should run under.

    Resolution order:
      1. $MWM_LEANDOJO_PYTHON.
      2. $ROOT/.venv-leandojo/bin/python (created by scripts/setup.sh).
      3. $ROOT/.venv/bin/python (fallback when caller maintains a single
         venv that already has lean-dojo).
    """
    env = os.environ.get("MWM_LEANDOJO_PYTHON")
    if env:
        return Path(env)
    p = ROOT / ".venv-leandojo/bin/python"
    if p.exists():
        return p
    return ROOT / ".venv/bin/python"


LD_RANDOM = _resolve_data_dir()
OUT = ROOT / "results/proof_transform"

ENRICHED_PATH = OUT / "enriched_traces.jsonl"
DAGS_PATH = OUT / "theorem_dags.jsonl"
MOTIFS_PATH = OUT / "subgraph_motifs.jsonl"
TOP_PATH = OUT / "subgraph_motifs_top.json"
ENRICH_SUMMARY = OUT / "enrichment_summary.json"
MINE_SUMMARY = OUT / "subgraph_summary.json"

WORKER_PYTHON_DEFAULT = _resolve_worker_python()
WORKER_SCRIPT = ROOT / "analysis/_proof_transform_worker.py"

MATHLIB_URL = "https://github.com/leanprover-community/mathlib4"
MATHLIB_COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"

NO_GOALS = "no goals"
GOAL_RE = re.compile(r"⊢\s*(.+?)(?=\n\ncase\s|\Z)", re.DOTALL)
ANCHOR_RE = re.compile(r"<a>(.*?)</a>")

# ---- Tactic classification ------------------------------------------------
# Heads where the underlying operation is a search over a global lemma
# database (simp_set / aesop hint database).  We unroll these via the Dojo
# REPL by querying the `?` variant and replaying each lemma one at a time.
UNROLLABLE_AUTOMATION = {
    "simp", "simpa", "simp_all", "simp_rw", "dsimp", "field_simp", "aesop",
}

# Heads that decide a proposition by running an algorithm rather than
# applying a named lemma list. We keep these as a single opaque labelled
# edge whose label = the head; matching uses just (state, decision_kind).
DECISION_PROCEDURES = {
    "omega", "decide", "native_decide",
    "ring", "ring_nf", "noncomm_ring", "abel", "abel_nf", "group",
    "linarith", "nlinarith", "polyrith",
    "norm_num", "norm_cast", "push_cast", "exact_mod_cast",
    "positivity", "gcongr",
    "tauto", "rfl", "trivial", "assumption",
}

# Specified-rewrite heads. Their tactic text already contains the explicit
# lemma list (after LeanDojo's annotation); we unroll one rewrite at a time
# with `rw [lemma]` to obtain real intermediate states.
REWRITE_HEADS = {"rw", "rwa", "erw", "nth_rw", "rw_mod_cast"}

# Term-elaboration tactics. One edge per call; premises come from the
# LeanDojo annotation already (we do not re-run them lemma-by-lemma).
TERM_HEADS = {"exact", "refine", "apply", "convert"}


# ---- Generic helpers ------------------------------------------------------

def sha(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:n]


def category(path: str) -> str:
    parts = (path or "").split("/")
    if len(parts) > 2 and parts[0] == "Mathlib":
        return parts[1]
    return "UNKNOWN"


def tactic_head(tactic: str) -> str:
    t = (tactic or "").strip()
    return t.split(None, 1)[0] if t else ""


def split_lean_state(state: str) -> list[str]:
    """Split a (possibly multi-goal) Lean state into single-goal substrings.

    Mirrors state_hypergraph.split_lean_state so the two graphs share state
    hashes when run on identical Lean state texts.
    """
    s = (state or "").strip()
    if not s or s == NO_GOALS:
        return [NO_GOALS]
    m = re.match(r"^\d+\s+goals?\n(.*)", s, re.DOTALL)
    if m:
        s = m.group(1)
    if s.startswith("case ") or "\n\ncase " in s:
        parts = re.split(r"\n\n(?=case\s)", s)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            return parts
    return [s]


def state_hash(state_text: str) -> str:
    """Strict full-state hash matching state_hypergraph.state_id."""
    t = (state_text or "").strip()
    if t == NO_GOALS:
        return "NOGOALS__________"
    return sha(t)


def extract_goal_text(state: str) -> str | None:
    """Return the joined RHS of all `⊢ ...` blocks; None if no goals."""
    s = (state or "").strip()
    if not s or s == NO_GOALS:
        return None
    found = [m.strip() for m in GOAL_RE.findall(s)]
    if not found:
        return s
    return "\n||\n".join(found)


def goal_hash(state_text: str) -> str | None:
    g = extract_goal_text(state_text)
    if g is None:
        return None
    return sha(g)


def state_excerpt(state_text: str, n: int = 220) -> str:
    t = " ".join((state_text or "").split())
    return t[:n]


# ---- Annotation parsing (LeanDojo) ---------------------------------------

def annotated_text(tt: dict) -> str:
    ann = tt.get("annotated_tactic")
    if isinstance(ann, list) and ann:
        return ann[0]
    return tt.get("tactic", "")


def annotated_premises(tt: dict) -> list[dict]:
    """Return premises with name, def_path, and rewrite direction (fwd/rev)."""
    ann = tt.get("annotated_tactic")
    if not (isinstance(ann, list) and len(ann) > 1 and isinstance(ann[1], list)):
        return []
    text = annotated_text(tt)
    tags = list(ANCHOR_RE.finditer(text))
    out, seen = [], set()
    for i, p in enumerate(ann[1]):
        name = p.get("full_name")
        if not name or name in seen:
            continue
        seen.add(name)
        direction = "?"
        if i < len(tags):
            before = text[max(0, tags[i].start() - 8): tags[i].start()]
            direction = "rev" if ("←" in before or "<-" in before) else "fwd"
        out.append({"name": name, "path": p.get("def_path", ""), "direction": direction})
    return out


# ---- Iteration over LeanDojo benchmark JSON ------------------------------

def iter_theorems(data_path: Path) -> Iterable[tuple[str, dict]]:
    for split in ("train", "val", "test"):
        p = data_path / f"{split}.json"
        if not p.exists():
            continue
        with p.open() as f:
            for thm in json.load(f):
                yield split, thm


def stratified_sample(data_path: Path, n: int, seed: int) -> list[tuple[str, dict]]:
    """Sample n theorems with traced_tactics, proportionally by top-level Mathlib area."""
    import random
    rng = random.Random(seed)
    by_cat: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for split, thm in iter_theorems(data_path):
        if not thm.get("traced_tactics"):
            continue
        by_cat[category(thm.get("file_path") or "")].append((split, thm))
    if not by_cat:
        return []
    total = sum(len(v) for v in by_cat.values())
    out: list[tuple[str, dict]] = []
    for cat, lst in by_cat.items():
        share = max(1, round(n * len(lst) / total))
        rng.shuffle(lst)
        out.extend(lst[:share])
    rng.shuffle(out)
    return out[:n]


# ---- Enrichment: dispatch to LeanDojo worker subprocess pool -------------

def already_enriched(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            name = rec.get("name")
            if name:
                done.add(name)
    return done


def write_batch_payload(thms: list[tuple[str, dict]], path: Path) -> None:
    with path.open("w") as f:
        for split, thm in thms:
            f.write(
                json.dumps(
                    {
                        "split": split,
                        "full_name": thm.get("full_name") or "",
                        "file_path": thm.get("file_path") or "",
                        "traced_tactics": thm.get("traced_tactics") or [],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_enrichment(
    targets: list[tuple[str, dict]],
    out_path: Path,
    worker_python: Path,
    workers: int,
    dojo_timeout: int,
    fallback_on_drift: bool,
) -> dict:
    """Spawn N worker subprocesses each consuming a slice of targets.

    Each worker reads a JSONL payload file given on argv and appends one
    enriched-trace JSONL line per processed theorem to `out_path` (workers
    use an O_APPEND file lock around their writes).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not WORKER_SCRIPT.exists():
        raise SystemExit(f"missing worker script: {WORKER_SCRIPT}")
    if not worker_python.exists():
        raise SystemExit(f"missing worker python: {worker_python}")

    # Split targets across workers
    targets = list(targets)
    if not targets:
        return {"theorems": 0, "workers": 0}
    workers = max(1, min(workers, len(targets)))
    chunks: list[list[tuple[str, dict]]] = [[] for _ in range(workers)]
    for i, t in enumerate(targets):
        chunks[i % workers].append(t)

    payload_dir = out_path.parent / ".enrich_payload"
    payload_dir.mkdir(exist_ok=True)
    procs = []
    payloads = []
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        payload = payload_dir / f"payload_{i}.jsonl"
        write_batch_payload(chunk, payload)
        payloads.append(payload)
        cmd = [
            str(worker_python),
            str(WORKER_SCRIPT),
            "--payload",
            str(payload),
            "--out",
            str(out_path),
            "--mathlib-url",
            MATHLIB_URL,
            "--mathlib-commit",
            MATHLIB_COMMIT,
            "--dojo-timeout",
            str(dojo_timeout),
        ]
        if fallback_on_drift:
            cmd.append("--fallback-on-drift")
        log_path = out_path.parent / f".enrich_worker_{i}.log"
        log_fh = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append((proc, log_fh, log_path, len(chunk)))
        print(f"  worker {i}: pid={proc.pid} chunk={len(chunk)} log={log_path}")

    rc_summary = []
    for i, (proc, log_fh, log_path, n) in enumerate(procs):
        rc = proc.wait()
        log_fh.close()
        rc_summary.append({"worker": i, "rc": rc, "chunk": n, "log": str(log_path)})
        print(f"  worker {i}: rc={rc}")

    dt = time.time() - t0
    return {
        "theorems": len(targets),
        "workers": workers,
        "elapsed_s": dt,
        "per_worker": rc_summary,
        "payload_dir": str(payload_dir),
    }


def cmd_enrich(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)

    # Build target list
    if args.mode == "sample":
        targets = stratified_sample(data_path, args.sample_size, args.seed)
    else:  # full
        targets = [(s, t) for s, t in iter_theorems(data_path) if t.get("traced_tactics")]

    # Resume support
    if args.resume:
        done = already_enriched(ENRICHED_PATH)
        targets = [(s, t) for s, t in targets if (t.get("full_name") or "") not in done]
        print(f"resume: {len(done)} already enriched; {len(targets)} remaining")
    else:
        if ENRICHED_PATH.exists():
            ENRICHED_PATH.unlink()

    summary = run_enrichment(
        targets,
        ENRICHED_PATH,
        worker_python=Path(args.worker_python),
        workers=args.workers,
        dojo_timeout=args.dojo_timeout,
        fallback_on_drift=not args.no_fallback_on_drift,
    )

    summary.update(
        mode=args.mode,
        sample_size=args.sample_size if args.mode == "sample" else None,
        out=str(ENRICHED_PATH),
    )

    # Aggregate per-edge stats from the enriched file
    by_kind = Counter()
    fallback_count = 0
    n_records = 0
    n_edges = 0
    if ENRICHED_PATH.exists():
        with ENRICHED_PATH.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                n_records += 1
                for e in rec.get("edges", []):
                    n_edges += 1
                    by_kind[e.get("kind", "?")] += 1
                    if e.get("is_fallback"):
                        fallback_count += 1
    summary.update(
        records=n_records,
        edges=n_edges,
        edges_by_kind=dict(by_kind.most_common()),
        fallback_count=fallback_count,
    )
    ENRICH_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ---- Build per-theorem DAGs ---------------------------------------------

def _key_for_state(state_text: str, match_mode: str) -> str:
    if match_mode == "goal":
        gh = goal_hash(state_text)
        if gh is not None:
            return gh
    return state_hash(state_text)


def _edge_label(edge: dict) -> tuple:
    """Canonical edge label tuple used both for DAG storage and matching.

    The label encodes the *kind* of operation plus its identifying data:
    - simp_step / rewrite_step: kind + (lemma,) + (direction,)
    - simp_bundle (fallback): kind + sorted-lemma-tuple + ('?'-tuple)
    - decision_opaque: kind + (head,)
    - term: kind + (head,) + sorted premise names
    - structural / other: kind + (head,)
    """
    k = edge["kind"]
    head = edge.get("head", "")
    if k in ("simp_step", "rewrite_step"):
        lemmas = tuple(edge.get("lemmas", []))
        dirs = tuple(edge.get("directions", []))
        return (k, head, lemmas, dirs)
    if k == "simp_bundle":
        lemmas = tuple(sorted(edge.get("lemmas", [])))
        return (k, head, lemmas, ())
    if k == "decision_opaque":
        return (k, head, (), ())
    if k == "term":
        prems = tuple(sorted(p["name"] for p in edge.get("premises", [])))
        dirs = tuple(p.get("direction", "?") for p in edge.get("premises", []))
        return (k, head, prems, dirs)
    # structural / other
    prems = tuple(sorted(p["name"] for p in edge.get("premises", [])))
    return (k, head, prems, ())


def cmd_build(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.enriched)
    if not in_path.exists():
        raise SystemExit(f"missing {in_path}; run `enrich` first")

    match_mode = args.match
    n_thm = 0
    total_nodes = 0
    total_edges = 0
    edges_by_kind = Counter()

    with in_path.open() as fin, DAGS_PATH.open("w") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            name = rec.get("name") or ""
            file_path = rec.get("file_path") or ""
            edges_in = rec.get("edges") or []
            if not edges_in:
                continue

            nodes: dict[str, dict] = {}
            edges_out: list[dict] = []
            for e in edges_in:
                in_text = e.get("in_state", "")
                out_texts = e.get("out_states", []) or []
                in_key = _key_for_state(in_text, match_mode)
                out_keys = [_key_for_state(s, match_mode) for s in out_texts]
                if in_key not in nodes:
                    nodes[in_key] = {
                        "id": in_key,
                        "excerpt": state_excerpt(in_text),
                        "goal_hash": goal_hash(in_text),
                    }
                for s, k in zip(out_texts, out_keys):
                    if k not in nodes:
                        nodes[k] = {
                            "id": k,
                            "excerpt": state_excerpt(s),
                            "goal_hash": goal_hash(s),
                        }
                lbl = _edge_label(e)
                edges_out.append(
                    {
                        "in_node": in_key,
                        "out_nodes": out_keys,
                        "kind": lbl[0],
                        "head": lbl[1],
                        "lemma_tuple": list(lbl[2]),
                        "direction_tuple": list(lbl[3]),
                        "tactic_text": e.get("tactic_text") or e.get("tactic", ""),
                        "source_tactic_index": e.get("source_tactic_index"),
                        "sub_index": e.get("sub_index", 0),
                        "is_fallback": bool(e.get("is_fallback", False)),
                    }
                )
                edges_by_kind[lbl[0]] += 1

            fout.write(
                json.dumps(
                    {
                        "name": name,
                        "file_path": file_path,
                        "category": category(file_path),
                        "nodes": list(nodes.values()),
                        "edges": edges_out,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_thm += 1
            total_nodes += len(nodes)
            total_edges += len(edges_out)

    summary = {
        "theorems": n_thm,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "edges_by_kind": dict(edges_by_kind.most_common()),
        "match_mode": match_mode,
        "out": str(DAGS_PATH),
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---- Mine: frequent labelled subgraph search -----------------------------

def _path_signature(node_keys: list[str], edge_labels: list[tuple]) -> str:
    """Canonical signature for a linear path of length k edges.

    node_keys has length k+1; edge_labels has length k. The path is
    directional so the tuple is taken as-is (no node-set canonicalisation).
    """
    parts: list[str] = []
    for i, ek in enumerate(node_keys):
        parts.append(ek)
        if i < len(edge_labels):
            el = edge_labels[i]
            parts.append("|".join(repr(x) for x in el))
    return sha("␞".join(parts), 24)


def _subtree_signature(root: str, children: list[tuple], node_keys: dict[str, str]) -> str:
    """AHU-style canonical hash of a rooted, labelled, ordered-by-canon subtree.

    `children` is a list of (edge_label_tuple, child_canonical_hex) tuples.
    The node key (state hash) of the root is included.
    """
    children_sorted = sorted(
        ("|".join(repr(x) for x in el) + ":" + ch) for el, ch in children
    )
    s = root + "␟" + "␟".join(children_sorted)
    return sha(s, 24)


def _enumerate_paths(
    nodes: dict[str, dict],
    fwd: dict[str, list[int]],
    edges: list[dict],
    min_edges: int,
    max_edges: int,
):
    """Yield (node_keys, edge_indices) for every linear forward path with
    edge_length in [min_edges, max_edges].
    """
    for start_edge_idx in range(len(edges)):
        # Start each path at this edge; extend forward by always taking the
        # first output (we follow each output as a separate branch later).
        e0 = edges[start_edge_idx]
        # We seed with each out_node of the starting edge as the "current
        # position", allowing all forward extensions.
        for out_idx, on in enumerate(e0["out_nodes"]):
            stack = [(
                [e0["in_node"], on],
                [start_edge_idx],
                on,
            )]
            while stack:
                node_keys, eidx, cur = stack.pop()
                k = len(eidx)
                if min_edges <= k <= max_edges:
                    yield node_keys, eidx
                if k >= max_edges:
                    continue
                # Extend forward
                for nxt_idx in fwd.get(cur, ()):
                    if nxt_idx in eidx:
                        continue  # avoid revisiting an edge
                    nxt = edges[nxt_idx]
                    for nxt_on in nxt["out_nodes"]:
                        stack.append((
                            node_keys + [nxt_on],
                            eidx + [nxt_idx],
                            nxt_on,
                        ))


def _enumerate_subtrees(
    nodes: dict[str, dict],
    fwd: dict[str, list[int]],
    edges: list[dict],
    max_edges: int,
):
    """Yield (root_key, edge_indices_set) for connected sub-DAGs of edge
    count in [2, max_edges] whose first edge is a *branching* edge (i.e.
    |out_nodes| > 1).  We anchor on branches to focus mining on the
    'split-then-prove-each-branch' patterns the user is interested in.
    """
    for start_edge_idx, e0 in enumerate(edges):
        if len(e0["out_nodes"]) <= 1:
            continue
        # Frontier search: starting from {start_edge_idx}, repeatedly add
        # any forward-adjacent edge until size hits max_edges.
        root = e0["in_node"]
        seen_edge_sets: set[frozenset[int]] = set()
        stack: list[frozenset[int]] = [frozenset({start_edge_idx})]
        while stack:
            es = stack.pop()
            if es in seen_edge_sets:
                continue
            seen_edge_sets.add(es)
            if 2 <= len(es) <= max_edges:
                yield root, es
            if len(es) >= max_edges:
                continue
            # Candidate extensions: edges incident to any out_node of any
            # edge already in `es`.
            frontier_nodes = set()
            for ei in es:
                frontier_nodes.update(edges[ei]["out_nodes"])
            for fn in frontier_nodes:
                for nxt_idx in fwd.get(fn, ()):
                    if nxt_idx not in es:
                        stack.append(es | {nxt_idx})


def _subtree_canon(
    root: str,
    edge_set: set[int],
    edges: list[dict],
    fwd: dict[str, list[int]],
) -> str:
    """Compute an AHU-style canonical hash of the labelled sub-DAG rooted
    at `root` containing exactly the edges in `edge_set`.

    For DAGs that are technically not trees (a node has two in-edges in
    `edge_set`), the algorithm still computes a deterministic hash by
    treating each child appearance independently — different topologies
    therefore canonicalise to different hashes.
    """
    # Build local successors restricted to edge_set
    local_fwd: dict[str, list[int]] = defaultdict(list)
    for ei in edge_set:
        e = edges[ei]
        local_fwd[e["in_node"]].append(ei)

    cache: dict[tuple, str] = {}

    def canon(node: str, visited: frozenset[int]) -> str:
        children = []
        for ei in local_fwd.get(node, []):
            if ei in visited:
                # cycle guard (shouldn't happen in proof DAGs)
                continue
            e = edges[ei]
            el = (e["kind"], e["head"], tuple(e["lemma_tuple"]), tuple(e["direction_tuple"]))
            child_hash_parts = []
            for on in e["out_nodes"]:
                ch = canon(on, visited | {ei})
                child_hash_parts.append(ch)
            child_part = "␞".join(child_hash_parts)
            children.append((el, child_part))
        return _subtree_signature(node, children, {})

    return canon(root, frozenset())


def _gain(edge_count: int, support: int) -> int:
    """Tactics saved if every occurrence is replaced by a single call to a
    new lemma whose own proof is the motif body (cost ≈ edge_count).
    """
    return (edge_count - 1) * support - edge_count


def cmd_mine(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.dags)
    if not in_path.exists():
        raise SystemExit(f"missing {in_path}; run `build` first")

    # Two indexes: signature -> occurrence list.  Each occurrence carries
    # enough info to reconstruct one example.
    paths_occ: dict[str, list[dict]] = defaultdict(list)
    trees_occ: dict[str, list[dict]] = defaultdict(list)

    # Side maps for example reconstruction
    sig_node_keys: dict[str, list[str]] = {}
    sig_edge_labels: dict[str, list[tuple]] = {}

    n_thm = 0
    n_path_occurrences = 0
    n_tree_occurrences = 0

    with in_path.open() as fin:
        for line in fin:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_thm += 1
            name = rec["name"]
            file_path = rec.get("file_path", "")
            cat = rec.get("category", "UNKNOWN")
            edges = rec.get("edges", [])
            nodes_map = {n["id"]: n for n in rec.get("nodes", [])}
            if not edges:
                continue

            # Build forward-adjacency from in_node → list of edge indices
            fwd: dict[str, list[int]] = defaultdict(list)
            for i, e in enumerate(edges):
                fwd[e["in_node"]].append(i)

            # Path enumeration
            if args.include_paths:
                for node_keys, eidx in _enumerate_paths(
                    nodes_map, fwd, edges, args.min_edges, args.max_edges
                ):
                    edge_labels = [
                        (
                            edges[i]["kind"],
                            edges[i]["head"],
                            tuple(edges[i]["lemma_tuple"]),
                            tuple(edges[i]["direction_tuple"]),
                        )
                        for i in eidx
                    ]
                    sig = _path_signature(node_keys, edge_labels)
                    paths_occ[sig].append(
                        {
                            "theorem": name,
                            "file_path": file_path,
                            "category": cat,
                            "edge_indices": eidx,
                            "node_keys": node_keys,
                            "tactic_chain": [edges[i]["tactic_text"] for i in eidx],
                            "tactic_range": [
                                min(edges[i].get("source_tactic_index", -1) for i in eidx),
                                max(edges[i].get("source_tactic_index", -1) for i in eidx),
                            ],
                            "state_excerpts": [nodes_map.get(nk, {}).get("excerpt", "") for nk in node_keys],
                        }
                    )
                    n_path_occurrences += 1
                    if sig not in sig_edge_labels:
                        sig_node_keys[sig] = node_keys
                        sig_edge_labels[sig] = edge_labels

            # Subtree enumeration
            if args.include_trees:
                for root, eset in _enumerate_subtrees(
                    nodes_map, fwd, edges, args.max_tree_edges
                ):
                    sig = _subtree_canon(root, set(eset), edges, fwd)
                    eidx = sorted(eset)
                    edge_labels = [
                        (
                            edges[i]["kind"],
                            edges[i]["head"],
                            tuple(edges[i]["lemma_tuple"]),
                            tuple(edges[i]["direction_tuple"]),
                        )
                        for i in eidx
                    ]
                    trees_occ[sig].append(
                        {
                            "theorem": name,
                            "file_path": file_path,
                            "category": cat,
                            "edge_indices": eidx,
                            "root_node_key": root,
                            "tactic_chain": [edges[i]["tactic_text"] for i in eidx],
                            "tactic_range": [
                                min(edges[i].get("source_tactic_index", -1) for i in eidx),
                                max(edges[i].get("source_tactic_index", -1) for i in eidx),
                            ],
                            "state_excerpts": [
                                nodes_map.get(edges[i]["in_node"], {}).get("excerpt", "")
                                for i in eidx
                            ],
                        }
                    )
                    n_tree_occurrences += 1
                    if sig not in sig_edge_labels:
                        # node keys recorded for trees = just the root + the unique
                        # endpoints (best-effort; trees aren't linear)
                        nks = [root]
                        for i in eidx:
                            nks.extend(edges[i]["out_nodes"])
                        sig_node_keys[sig] = nks
                        sig_edge_labels[sig] = edge_labels

    # Filter and emit
    motifs: list[dict] = []
    for kind, sig_map in (("path", paths_occ), ("subtree", trees_occ)):
        for sig, occs in sig_map.items():
            support = len(occs)
            distinct_thms = len({o["theorem"] for o in occs})
            distinct_files = len({o["file_path"] for o in occs})
            cats = Counter(o["category"] for o in occs)

            edge_labels = sig_edge_labels.get(sig, [])
            node_keys = sig_node_keys.get(sig, [])
            edge_count = len(edge_labels)
            node_count = len(set(node_keys))

            if edge_count < args.min_edges:
                continue
            if support < args.min_support:
                continue
            if distinct_thms < args.min_distinct_theorems:
                continue
            # At least one non-opaque-decision edge to count as a candidate
            if all(el[0] == "decision_opaque" for el in edge_labels):
                continue
            gain = _gain(edge_count, support)
            if gain < args.min_gain:
                continue

            cid = "submot:" + sig[:18]
            motifs.append(
                {
                    "candidate_id": cid,
                    "motif_kind": kind,
                    "edge_count": edge_count,
                    "node_count": node_count,
                    "support": support,
                    "distinct_theorems": distinct_thms,
                    "distinct_files": distinct_files,
                    "categories": dict(cats.most_common()),
                    "complexity_gain": gain,
                    "signature": sig,
                    "edge_labels": [
                        {
                            "kind": el[0],
                            "head": el[1],
                            "lemma_tuple": list(el[2]),
                            "direction_tuple": list(el[3]),
                        }
                        for el in edge_labels
                    ],
                    "examples": occs[: args.max_examples],
                    "verification_status": "candidate_from_existing_verified_proofs_not_new_lean_declaration",
                }
            )

    # Sort: complexity_gain desc, then support desc, then distinct_thms desc, then sig
    motifs.sort(
        key=lambda m: (-m["complexity_gain"], -m["support"], -m["distinct_theorems"], m["signature"])
    )

    with MOTIFS_PATH.open("w") as f:
        for m in motifs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    TOP_PATH.write_text(json.dumps(motifs[: args.top], indent=2, ensure_ascii=False))

    summary = {
        "theorems": n_thm,
        "path_occurrences_total": n_path_occurrences,
        "subtree_occurrences_total": n_tree_occurrences,
        "unique_path_signatures": len(paths_occ),
        "unique_subtree_signatures": len(trees_occ),
        "candidates_after_filter": len(motifs),
        "config": {
            "min_edges": args.min_edges,
            "max_edges": args.max_edges,
            "max_tree_edges": args.max_tree_edges,
            "min_support": args.min_support,
            "min_distinct_theorems": args.min_distinct_theorems,
            "min_gain": args.min_gain,
            "include_paths": args.include_paths,
            "include_trees": args.include_trees,
        },
        "outputs": {
            "all_motifs": str(MOTIFS_PATH),
            "top_motifs": str(TOP_PATH),
        },
        "top_excerpt": [
            {
                "candidate_id": m["candidate_id"],
                "motif_kind": m["motif_kind"],
                "edge_count": m["edge_count"],
                "support": m["support"],
                "distinct_theorems": m["distinct_theorems"],
                "complexity_gain": m["complexity_gain"],
            }
            for m in motifs[:20]
        ],
    }
    MINE_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ---- CLI -----------------------------------------------------------------

def _add_enrich_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-path", default=str(LD_RANDOM))
    p.add_argument("--mode", choices=("sample", "full"), default="sample")
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--worker-python", default=str(WORKER_PYTHON_DEFAULT))
    p.add_argument("--dojo-timeout", type=int, default=600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-fallback-on-drift", action="store_true",
                   help="If stepwise replay drifts, raise an error instead of emitting a simp_bundle fallback edge.")


def _add_build_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--enriched", default=str(ENRICHED_PATH))
    p.add_argument("--match", choices=("state", "goal"), default="state")


def _add_mine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dags", default=str(DAGS_PATH))
    p.add_argument("--min-edges", type=int, default=2)
    p.add_argument("--max-edges", type=int, default=6)
    p.add_argument("--max-tree-edges", type=int, default=5)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--min-distinct-theorems", type=int, default=2)
    p.add_argument("--min-gain", type=int, default=6)
    p.add_argument("--max-examples", type=int, default=5)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--include-paths", dest="include_paths", action="store_true", default=True)
    p.add_argument("--no-paths", dest="include_paths", action="store_false")
    p.add_argument("--include-trees", dest="include_trees", action="store_true", default=True)
    p.add_argument("--no-trees", dest="include_trees", action="store_false")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_enrich(args)
    cmd_build(args)
    cmd_mine(args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enrich", help="LeanDojo replay to unroll automation tactics")
    _add_enrich_args(e)
    e.set_defaults(func=cmd_enrich)

    b = sub.add_parser("build", help="Build per-theorem labelled DAGs from enriched traces")
    _add_build_args(b)
    b.set_defaults(func=cmd_build)

    m = sub.add_parser("mine", help="Frequent labelled-subgraph mining")
    _add_mine_args(m)
    m.set_defaults(func=cmd_mine)

    a = sub.add_parser("all", help="enrich + build + mine end-to-end")
    _add_enrich_args(a)
    _add_build_args(a)
    _add_mine_args(a)
    a.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
