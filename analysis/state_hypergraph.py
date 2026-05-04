#!/usr/bin/env python3
"""State-tactic hypergraph for Mathlib path-finding.

Nodes are proof states (sha256 of normalised text). A hyperedge is a single
tactic application: input_state -> [output_state_1, ..., output_state_k].
Multi-output edges arise from tactics that split into subgoals
(`constructor`, `induction`, `cases`, ...).

Path-finding modes
------------------
or-path     Bidirectional BFS treating each hyperedge as 1 input -> any-of-k
            outputs. Answers "does some tactic chain starting at B produce a
            state equal to A?" — the cheap version of "reduce B to A".
discharge   AND/OR DFS with memoisation. Answers "can the proof of B be
            closed if I am allowed to take A as already proved?" — the real
            oracle semantics. A search node is a conjunction of open states;
            it is solved when every open state is in {target, "no goals"}.

Both algorithms support exact full-state matching and a looser goal-only
matching (hash of the text after the last "⊢ ...").

CLI
---
    build         Build the hypergraph from LeanDojo data.
    lookup        Theorem name → initial state hash + statement.
    or-path       Bidirectional BFS for one (start, target) pair.
    discharge     AND/OR discharge for one (start, target) pair.
    sweep         Random pair sweep, compares to proof-DAG sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
LD_RANDOM = ROOT / "Old/data/leandojo/leandojo_benchmark_4/random"
RESULTS = ROOT / "results"
SG_DIR = RESULTS / "state_graph"

NO_GOALS = "no goals"
NO_GOALS_ID = "NOGOALS__________"  # stable sentinel

GOAL_RE = re.compile(r"⊢\s*(.+?)(?=\n\ncase\s|\Z)", re.DOTALL)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def state_id(s: str) -> str:
    if s.strip() == NO_GOALS:
        return NO_GOALS_ID
    return hashlib.sha256(s.strip().encode()).hexdigest()[:16]


def extract_goals(state: str) -> list[str]:
    """Return the list of goal RHS strings (one per `⊢` clause)."""
    s = state.strip()
    if s == NO_GOALS or not s:
        return []
    return [m.strip() for m in GOAL_RE.findall(s)]


def goal_id(state: str) -> str | None:
    goals = extract_goals(state)
    if not goals:
        return None
    canonical = "\n||\n".join(g for g in goals)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def split_lean_state(state: str) -> list[str]:
    """Return a list of single-goal substrings, one per open goal.

    Handles three forms:
      - "no goals" (or empty) → [NO_GOALS].
      - "N goals\\n<body>" header form → split body on blank-line + "case ".
      - Body that starts with "case " or contains "\\n\\ncase " (LeanDojo
        often omits the "N goals" header) → split similarly.
      - Single-goal form with no header → return [stripped state].
    """
    s = state.strip()
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


# Backwards-compat alias used elsewhere in the file.
parse_lean_multigoal = split_lean_state


def iter_leandojo(split_dir: Path) -> Iterable[tuple[str, dict]]:
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.json"
        if not path.exists():
            continue
        with path.open() as f:
            for thm in json.load(f):
                yield split, thm


def tactic_head(tac: str) -> str:
    tac = tac.strip()
    return tac.split(None, 1)[0] if tac else ""


# ----------------------------- build ---------------------------------------

def build(args: argparse.Namespace) -> None:
    ensure_dir(SG_DIR)
    nodes_path = SG_DIR / "state_nodes.jsonl"
    edges_path = SG_DIR / "state_hyperedges.jsonl"
    init_path = SG_DIR / "theorem_initial_states.jsonl"
    stats_path = SG_DIR / "state_graph_stats.json"

    nodes: dict[str, dict] = {NO_GOALS_ID: {"id": NO_GOALS_ID, "state": NO_GOALS, "goal_hash": None, "occurrences": 0}}
    # edge key includes input_id, tactic string, sorted output_ids
    edges: dict[tuple, dict] = {}
    theorem_initial: list[dict] = []
    n_thms = n_traced = n_tactics = 0
    multi_output = 0

    for split, thm in iter_leandojo(Path(args.data_path)):
        n_thms += 1
        name = thm.get("full_name") or ""
        file_path = thm.get("file_path") or ""
        traced = thm.get("traced_tactics") or []
        if traced:
            n_traced += 1
        if traced:
            init_state = traced[0].get("state_before", "")
            init_goals = split_lean_state(init_state)
            primary = init_goals[0] if init_goals else init_state
            iid = state_id(primary)
            theorem_initial.append({
                "name": name,
                "file_path": file_path,
                "initial_state_id": iid,
                "initial_goal_hash": goal_id(primary),
            })

        for tt in traced:
            n_tactics += 1
            tac = tt.get("tactic", "")
            sb_raw = tt.get("state_before", "")
            sa_raw = tt.get("state_after", "")
            in_goals = split_lean_state(sb_raw)
            out_goals = split_lean_state(sa_raw)
            in_ids = [state_id(g) for g in in_goals]
            out_ids = [state_id(g) for g in out_goals]

            for g, gid in zip(in_goals, in_ids):
                if gid == NO_GOALS_ID:
                    continue
                if gid not in nodes:
                    nodes[gid] = {
                        "id": gid,
                        "state": g.strip(),
                        "goal_hash": goal_id(g),
                        "occurrences": 0,
                    }
                nodes[gid]["occurrences"] += 1
            for g, gid in zip(out_goals, out_ids):
                if gid == NO_GOALS_ID:
                    continue
                if gid not in nodes:
                    nodes[gid] = {
                        "id": gid,
                        "state": g.strip(),
                        "goal_hash": goal_id(g),
                        "occurrences": 0,
                    }
                nodes[gid]["occurrences"] += 1

            # One hyperedge per input goal: that goal -> all output goals.
            # Multi-input cases (e.g. all_goals on N cases) become N edges.
            for ig_id in in_ids:
                if ig_id == NO_GOALS_ID:
                    continue
                key = (ig_id, tac, tuple(sorted(out_ids)))
                if key not in edges:
                    edges[key] = {
                        "id": hashlib.sha256(repr(key).encode()).hexdigest()[:20],
                        "input": ig_id,
                        "outputs": list(out_ids),
                        "tactic": tac,
                        "tactic_head": tactic_head(tac),
                        "theorems": [],
                        "occurrences": 0,
                    }
                    if len(out_ids) > 1 and not (len(out_ids) == 1 and out_ids[0] == NO_GOALS_ID):
                        multi_output += 1
                if name and name not in edges[key]["theorems"]:
                    edges[key]["theorems"].append(name)
                edges[key]["occurrences"] += 1

    print(f"writing {len(nodes):,} nodes, {len(edges):,} hyperedges")
    with nodes_path.open("w") as f:
        for rec in nodes.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with edges_path.open("w") as f:
        for rec in edges.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with init_path.open("w") as f:
        for rec in theorem_initial:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_size_hist = Counter(len(e["outputs"]) for e in edges.values())
    head_counts = Counter(e["tactic_head"] for e in edges.values()).most_common(20)
    stats = {
        "theorems_indexed": n_thms,
        "theorems_with_traces": n_traced,
        "tactic_invocations": n_tactics,
        "unique_states": len(nodes),
        "unique_hyperedges": len(edges),
        "multi_output_hyperedges": multi_output,
        "output_size_histogram": dict(sorted(out_size_hist.items())),
        "top_tactic_heads_by_distinct_edges": head_counts,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("wrote", stats_path)


# ----------------------------- load ---------------------------------------

def load_graph(match: str = "state"):
    """Returns (forward, backward, nodes, theorem_initial, key_of, name_to_key).

    match="state": keys = full state_id (exact state match).
    match="goal":  keys = goal_id (states with same goal-RHS merge). When goal_id
                   is missing for a state, fall back to state_id.
    """
    nodes: dict[str, dict] = {}
    with (SG_DIR / "state_nodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            nodes[rec["id"]] = rec

    def key_of(state_id_value: str) -> str:
        if match == "state":
            return state_id_value
        if state_id_value == NO_GOALS_ID:
            return NO_GOALS_ID
        n = nodes.get(state_id_value)
        if n is None:
            return state_id_value
        return n.get("goal_hash") or state_id_value

    forward: dict[str, list[dict]] = defaultdict(list)
    backward: dict[str, list[dict]] = defaultdict(list)
    with (SG_DIR / "state_hyperedges.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            in_key = key_of(e["input"])
            out_keys = [key_of(o) for o in e["outputs"]]
            e["input_key"] = in_key
            e["output_keys"] = out_keys
            forward[in_key].append(e)
            for ok in out_keys:
                backward[ok].append(e)

    theorem_initial: dict[str, dict] = {}
    init_path = SG_DIR / "theorem_initial_states.jsonl"
    if init_path.exists():
        with init_path.open() as f:
            for line in f:
                rec = json.loads(line)
                rec["initial_key"] = key_of(rec["initial_state_id"])
                theorem_initial[rec["name"]] = rec
    return forward, backward, nodes, theorem_initial, key_of


def resolve_state(name_or_id: str, theorem_initial: dict, nodes: dict, key_of) -> str | None:
    if name_or_id in nodes:
        return key_of(name_or_id)
    if name_or_id in theorem_initial:
        return theorem_initial[name_or_id]["initial_key"]
    return None


# ----------------------- bidirectional OR BFS -----------------------------

def or_path(args: argparse.Namespace) -> None:
    forward, backward, nodes, theorem_initial, key_of = load_graph(match=args.match)
    start = resolve_state(args.start, theorem_initial, nodes, key_of)
    target = resolve_state(args.target, theorem_initial, nodes, key_of)
    if not start or not target:
        print(json.dumps({"found": False, "error": "could not resolve start or target"}))
        return
    result = bidirectional_or(start, target, forward, backward, args.max_depth, nodes)
    if args.output:
        out = Path(args.output)
        ensure_dir(out.parent)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def bidirectional_or(start: str, target: str, forward, backward, max_depth: int, nodes: dict) -> dict:
    if start == target:
        return {"found": True, "edges": [], "tactics": [], "depth": 0}

    fwd_parent: dict[str, tuple[str, dict] | None] = {start: None}
    bwd_parent: dict[str, tuple[str, dict] | None] = {target: None}
    fwd_frontier = {start}
    bwd_frontier = {target}
    meet: str | None = None
    depth = 0

    while fwd_frontier and bwd_frontier and depth < max_depth:
        depth += 1
        if len(fwd_frontier) <= len(bwd_frontier):
            new_frontier = set()
            for cur in fwd_frontier:
                for e in forward.get(cur, []):
                    for nxt in e["output_keys"]:
                        if nxt in fwd_parent:
                            continue
                        fwd_parent[nxt] = (cur, e)
                        new_frontier.add(nxt)
                        if nxt in bwd_parent:
                            meet = nxt
                            break
                    if meet:
                        break
                if meet:
                    break
            fwd_frontier = new_frontier
        else:
            new_frontier = set()
            for cur in bwd_frontier:
                for e in backward.get(cur, []):
                    inp = e["input_key"]
                    if inp in bwd_parent:
                        continue
                    bwd_parent[inp] = (cur, e)
                    new_frontier.add(inp)
                    if inp in fwd_parent:
                        meet = inp
                        break
                if meet:
                    break
            bwd_frontier = new_frontier
        if meet:
            break

    if not meet:
        return {"found": False, "depth_searched": depth, "fwd_visited": len(fwd_parent), "bwd_visited": len(bwd_parent)}

    fwd_chain: list[dict] = []
    cur = meet
    while fwd_parent[cur] is not None:
        prev, edge = fwd_parent[cur]
        fwd_chain.append(edge)
        cur = prev
    fwd_chain.reverse()

    bwd_chain: list[dict] = []
    cur = meet
    while bwd_parent[cur] is not None:
        nxt, edge = bwd_parent[cur]
        bwd_chain.append(edge)
        cur = nxt

    edges = fwd_chain + bwd_chain
    return {
        "found": True,
        "depth": len(edges),
        "meet_key": meet,
        "tactics": [e["tactic"] for e in edges],
        "edges": [
            {
                "input": e["input"],
                "outputs": e["outputs"],
                "tactic": e["tactic"],
                "tactic_head": e["tactic_head"],
                "theorems": e["theorems"][:5],
            }
            for e in edges
        ],
    }


# ----------------------- AND/OR discharge ---------------------------------

class DischargeSolver:
    def __init__(self, forward, oracle: str, max_depth: int, max_expansions: int = 50000):
        self.forward = forward
        self.oracle = oracle
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        # memo per (state, depth_remaining_floor)
        self.solved: dict[str, tuple[str, list]] = {}  # state -> (tactic, child_proofs) or marker
        self.failed: set[tuple[str, int]] = set()
        self.in_stack: set[str] = set()
        self.expansions = 0

    def solve(self, state: str, depth_left: int) -> tuple[bool, dict | None]:
        if state == self.oracle:
            return True, {"kind": "oracle", "state": state}
        if state == NO_GOALS_ID:
            return True, {"kind": "done"}
        if state in self.solved:
            tac, children = self.solved[state]
            return True, {"kind": "tactic", "state": state, "tactic": tac, "children": children}
        if depth_left <= 0:
            return False, None
        if state in self.in_stack:
            return False, None
        if (state, depth_left) in self.failed:
            return False, None
        if self.expansions >= self.max_expansions:
            return False, None

        self.in_stack.add(state)
        self.expansions += 1
        edges = self.forward.get(state, [])
        edges = sorted(edges, key=lambda e: (len(e["output_keys"]), -e.get("occurrences", 0)))

        for e in edges:
            child_proofs = []
            ok = True
            for child_key in e["output_keys"]:
                child_ok, child_proof = self.solve(child_key, depth_left - 1)
                if not child_ok:
                    ok = False
                    break
                child_proofs.append(child_proof)
            if ok:
                self.in_stack.discard(state)
                self.solved[state] = (e["tactic"], child_proofs)
                return True, {
                    "kind": "tactic",
                    "state": state,
                    "tactic": e["tactic"],
                    "tactic_head": e["tactic_head"],
                    "outputs": e["output_keys"],
                    "children": child_proofs,
                }

        self.in_stack.discard(state)
        self.failed.add((state, depth_left))
        return False, None


def discharge_cmd(args: argparse.Namespace) -> None:
    forward, backward, nodes, theorem_initial, key_of = load_graph(match=args.match)
    start = resolve_state(args.start, theorem_initial, nodes, key_of)
    target = resolve_state(args.target, theorem_initial, nodes, key_of)
    if not start or not target:
        print(json.dumps({"solved": False, "error": "could not resolve start or target"}))
        return
    solver = DischargeSolver(forward, oracle=target, max_depth=args.max_depth, max_expansions=args.max_expansions)
    t0 = time.perf_counter()
    ok, proof = solver.solve(start, args.max_depth)
    dt = time.perf_counter() - t0

    result = {
        "solved": ok,
        "elapsed_s": dt,
        "expansions": solver.expansions,
        "max_depth": args.max_depth,
        "max_expansions": args.max_expansions,
        "start_state_id": start,
        "oracle_state_id": target,
        "start_state_excerpt": nodes.get(start, {}).get("state", "")[:300],
        "oracle_state_excerpt": nodes.get(target, {}).get("state", "")[:300],
        "proof": proof,
    }
    if args.output:
        out = Path(args.output)
        ensure_dir(out.parent)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "proof"}, indent=2, ensure_ascii=False))
    if proof and args.show_proof:
        print("--- proof tree ---")
        print(json.dumps(proof, indent=2, ensure_ascii=False))


# ----------------------- lookup -------------------------------------------

def lookup_cmd(args: argparse.Namespace) -> None:
    _, _, nodes, theorem_initial, _ = load_graph(match="state")
    if args.name in theorem_initial:
        rec = theorem_initial[args.name]
        node = nodes.get(rec["initial_state_id"], {})
        print(json.dumps({
            "name": args.name,
            "initial_state_id": rec["initial_state_id"],
            "initial_goal_hash": rec.get("initial_goal_hash"),
            "file_path": rec.get("file_path"),
            "state_excerpt": node.get("state", "")[:600],
        }, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"error": f"theorem {args.name!r} not in initial-state index"}, indent=2))


# ----------------------- sweep --------------------------------------------

def sweep_cmd(args: argparse.Namespace) -> None:
    forward, backward, nodes, theorem_initial, key_of = load_graph(match=args.match)
    rng = random.Random(args.seed)
    names = list(theorem_initial)
    print(f"sweep over {len(names)} theorems with traced tactics, match={args.match}")
    pairs = []
    for _ in range(args.pairs):
        a = rng.choice(names)
        b = rng.choice(names)
        if a == b:
            continue
        pairs.append((a, b))

    or_results = []
    discharge_results = []
    for a, b in pairs:
        sa = theorem_initial[a]["initial_key"]
        sb = theorem_initial[b]["initial_key"]
        r = bidirectional_or(sb, sa, forward, backward, args.max_depth, nodes)
        or_results.append({"start_thm": b, "target_thm": a, "found": r["found"], "depth": r.get("depth")})

        if args.run_discharge:
            solver = DischargeSolver(forward, oracle=sa, max_depth=args.discharge_depth, max_expansions=args.discharge_expansions)
            ok, _ = solver.solve(sb, args.discharge_depth)
            discharge_results.append({
                "start_thm": b,
                "target_thm": a,
                "solved": ok,
                "expansions": solver.expansions,
            })

    or_found = sum(1 for r in or_results if r["found"])
    or_depths = [r["depth"] for r in or_results if r["found"]]
    out = {
        "config": {
            "pairs": args.pairs,
            "max_depth": args.max_depth,
            "discharge": args.run_discharge,
            "discharge_depth": args.discharge_depth,
            "discharge_expansions": args.discharge_expansions,
            "seed": args.seed,
        },
        "or_reachability": {
            "pairs_tried": len(or_results),
            "found": or_found,
            "found_fraction": or_found / max(1, len(or_results)),
            "depth_histogram": dict(sorted(Counter(or_depths).items())),
        },
    }
    if args.run_discharge:
        solved = sum(1 for r in discharge_results if r["solved"])
        out["discharge"] = {
            "pairs_tried": len(discharge_results),
            "solved": solved,
            "solved_fraction": solved / max(1, len(discharge_results)),
            "mean_expansions": sum(r["expansions"] for r in discharge_results) / max(1, len(discharge_results)),
        }
    if args.compare_to_dep_graph:
        cmp = compare_to_dep_graph(forward, backward, nodes, theorem_initial, args)
        out["dep_graph_cross_validation"] = cmp
    out_path = SG_DIR / f"state_path_sweep_{args.match}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


def compare_to_dep_graph(forward, backward, nodes, theorem_initial, args):
    """For each (B, A) edge in the proof DAG where both endpoints have traces,
    test state-graph OR-reachability."""
    edges_path = RESULTS / "paths/dependency_edges.jsonl"
    if not edges_path.exists():
        return {"error": "dependency_edges.jsonl missing"}
    rng = random.Random(args.seed)
    cands = []
    with edges_path.open() as f:
        for line in f:
            e = json.loads(line)
            if e["source"] in theorem_initial and e["target"] in theorem_initial:
                cands.append((e["source"], e["target"], e.get("tactic_head", "")))
    rng.shuffle(cands)
    cands = cands[: args.dep_pairs]
    found = 0
    by_head = {}
    depths = []
    for b, a, head in cands:
        sb = theorem_initial[b]["initial_key"]
        sa = theorem_initial[a]["initial_key"]
        r = bidirectional_or(sb, sa, forward, backward, args.max_depth, nodes)
        ok = r["found"]
        if ok:
            found += 1
            depths.append(r.get("depth", -1))
        by_head.setdefault(head, [0, 0])
        by_head[head][0] += 1
        by_head[head][1] += int(ok)
    by_head_summary = sorted(
        ({"tactic_head": h, "n": n, "found": k, "rate": k / n} for h, (n, k) in by_head.items() if n >= 5),
        key=lambda r: -r["rate"],
    )
    return {
        "pairs_tried": len(cands),
        "found": found,
        "found_fraction": found / max(1, len(cands)),
        "depth_histogram": dict(sorted(Counter(depths).items())),
        "rate_by_dep_tactic_head": by_head_summary,
    }


# ----------------------- CLI ----------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--data-path", default=str(LD_RANDOM))
    b.set_defaults(func=build)

    look = sub.add_parser("lookup")
    look.add_argument("--name", required=True)
    look.set_defaults(func=lookup_cmd)

    o = sub.add_parser("or-path")
    o.add_argument("--start", required=True)
    o.add_argument("--target", required=True)
    o.add_argument("--max-depth", type=int, default=10)
    o.add_argument("--match", choices=["state", "goal"], default="goal")
    o.add_argument("--output")
    o.set_defaults(func=or_path)

    d = sub.add_parser("discharge")
    d.add_argument("--start", required=True)
    d.add_argument("--target", required=True)
    d.add_argument("--max-depth", type=int, default=8)
    d.add_argument("--max-expansions", type=int, default=20000)
    d.add_argument("--match", choices=["state", "goal"], default="goal")
    d.add_argument("--output")
    d.add_argument("--show-proof", action="store_true")
    d.set_defaults(func=discharge_cmd)

    s = sub.add_parser("sweep")
    s.add_argument("--pairs", type=int, default=2000)
    s.add_argument("--max-depth", type=int, default=10)
    s.add_argument("--match", choices=["state", "goal"], default="goal")
    s.add_argument("--run-discharge", action="store_true")
    s.add_argument("--discharge-depth", type=int, default=6)
    s.add_argument("--discharge-expansions", type=int, default=2000)
    s.add_argument("--compare-to-dep-graph", action="store_true",
                   help="Also test state-graph reachability on dep-graph edges (B cites A)")
    s.add_argument("--dep-pairs", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=sweep_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
