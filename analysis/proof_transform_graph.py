#!/usr/bin/env python3
"""Build a trace-derived proof-transformation graph and search motifs.

This is the first implementation of the "unrolled proof graph" direction.  It
uses LeanDojo traces, so it can expose tactic inputs, outputs, and annotated
premises.  It cannot recover information LeanDojo did not record, such as the
exact matched subterm chosen by `simp`; those fields are represented by the
before/after goal hashes and excerpts until a Lean-side tracer is added.

Outputs:
  results/proof_transform/primitive_ops.jsonl
  results/proof_transform/graph_nodes.jsonl
  results/proof_transform/graph_edges.jsonl
  results/proof_transform/motif_candidates.jsonl
  results/proof_transform/motif_candidates_top.json
  results/proof_transform/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
LD_RANDOM = ROOT / "Old/data/leandojo/leandojo_benchmark_4/random"
OUT = ROOT / "results/proof_transform"

NO_GOALS = "no goals"
GOAL_RE = re.compile(r"⊢\s*(.+?)(?=\n\ncase\s|\Z)", re.DOTALL)
ANCHOR_RE = re.compile(r"<a>(.*?)</a>")

SIMP_HEADS = {"simp", "simpa", "simp_all", "dsimp", "field_simp"}
RW_HEADS = {"rw", "rwa", "erw", "nth_rw", "rw_mod_cast"}
SIMP_RW_HEADS = {"simp_rw"}
TERM_HEADS = {"exact", "refine", "apply"}
SEARCHABLE_KINDS = {"simp_search", "simp_rewrite", "specified_rewrite"}


def h(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def node_id(kind: str, value: str) -> str:
    return f"{kind}:{h(value)}"


def category(path: str) -> str:
    parts = path.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "Mathlib" else "UNKNOWN"


def tactic_head(tactic: str) -> str:
    return tactic.strip().split(None, 1)[0] if tactic.strip() else ""


def goals(state: str) -> list[str]:
    s = (state or "").strip()
    if not s or s == NO_GOALS:
        return []
    found = [m.strip() for m in GOAL_RE.findall(s)]
    return found or [s]


def goal_sig(state: str) -> tuple[str, str]:
    gs = goals(state)
    text = "\n||\n".join(gs) if gs else NO_GOALS
    return h(text), " ".join(text.split())[:260]


def annotated_text(tt: dict) -> str:
    ann = tt.get("annotated_tactic")
    return ann[0] if isinstance(ann, list) and ann else tt.get("tactic", "")


def annotated_premises(tt: dict) -> list[dict]:
    ann = tt.get("annotated_tactic")
    if not (isinstance(ann, list) and len(ann) > 1 and isinstance(ann[1], list)):
        return []
    out, seen = [], set()
    for p in ann[1]:
        name = p.get("full_name")
        if name and name not in seen:
            seen.add(name)
            out.append({
                "name": name,
                "path": p.get("def_path", ""),
                "pos": p.get("def_pos"),
            })
    return out


def premise_directions(tt: dict, premises: list[dict]) -> list[str]:
    text = annotated_text(tt)
    tags = list(ANCHOR_RE.finditer(text))
    dirs = []
    for i, _ in enumerate(premises):
        if i >= len(tags):
            dirs.append("?")
            continue
        before = text[max(0, tags[i].start() - 8):tags[i].start()]
        dirs.append("rev" if "←" in before or "<-" in before else "fwd")
    return dirs


def op_kind(head: str) -> str:
    if head in SIMP_HEADS:
        return "simp_search"
    if head in SIMP_RW_HEADS:
        return "simp_rewrite"
    if head in RW_HEADS:
        return "specified_rewrite"
    if head in TERM_HEADS:
        return "term_elaboration"
    return "premise_tactic" if head else "unknown"


def iter_theorems(data_dir: Path, max_theorems: int | None = None) -> Iterable[tuple[str, dict]]:
    n = 0
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.json"
        if not path.exists():
            continue
        for thm in json.load(path.open()):
            yield split, thm
            n += 1
            if max_theorems and n >= max_theorems:
                return


def add_node(nodes: dict[str, dict], rec: dict) -> str:
    nodes.setdefault(rec["id"], rec)
    return rec["id"]


def build(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nodes: dict[str, dict] = {}
    edge_count = 0
    op_count = 0
    stats = {
        "theorems": 0,
        "traced_theorems": 0,
        "primitive_ops": 0,
        "ops_by_kind": Counter(),
        "ops_by_head": Counter(),
        "premise_edges_by_kind": Counter(),
    }
    ops_path = OUT / "primitive_ops.jsonl"
    edges_path = OUT / "graph_edges.jsonl"
    nodes_path = OUT / "graph_nodes.jsonl"

    with ops_path.open("w") as ops_f, edges_path.open("w") as edge_f:
        for split, thm in iter_theorems(Path(args.data_path), args.max_theorems):
            stats["theorems"] += 1
            traced = thm.get("traced_tactics") or []
            if traced:
                stats["traced_theorems"] += 1
            thm_name = thm.get("full_name") or ""
            file_path = thm.get("file_path") or ""
            thm_id = add_node(nodes, {
                "id": node_id("theorem", thm_name),
                "kind": "theorem",
                "name": thm_name,
                "file_path": file_path,
                "category": category(file_path),
            })
            for i, tt in enumerate(traced):
                tactic = tt.get("tactic", "")
                head = tactic_head(tactic)
                kind = op_kind(head)
                before_hash, before_excerpt = goal_sig(tt.get("state_before", ""))
                after_hash, after_excerpt = goal_sig(tt.get("state_after", ""))
                before_id = add_node(nodes, {"id": node_id("goal", before_hash), "kind": "goal", "goal_hash": before_hash, "excerpt": before_excerpt})
                after_id = add_node(nodes, {"id": node_id("goal", after_hash), "kind": "goal", "goal_hash": after_hash, "excerpt": after_excerpt})
                op_id = add_node(nodes, {
                    "id": f"op:{h(f'{thm_name}:{i}:{tactic}:{before_hash}:{after_hash}', 20)}",
                    "kind": "operation",
                    "operation_kind": kind,
                    "tactic_head": head,
                    "tactic": tactic,
                })
                premises = annotated_premises(tt)
                dirs = premise_directions(tt, premises)
                prem_recs = []

                def write_edge(src: str, dst: str, role: str, **extra) -> None:
                    nonlocal edge_count
                    rec = {"src": src, "dst": dst, "role": role}
                    rec.update(extra)
                    edge_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    edge_count += 1

                write_edge(thm_id, op_id, "contains_step", tactic_index=i)
                write_edge(before_id, op_id, "input_to")
                write_edge(op_id, after_id, "produces")

                for j, prem in enumerate(premises):
                    prem_name = prem["name"]
                    decl_id = add_node(nodes, {
                        "id": node_id("decl", prem_name),
                        "kind": "declaration",
                        "name": prem_name,
                        "file_path": prem.get("path", ""),
                    })
                    prim_role = {
                        "simp_search": "selected_simp_rule",
                        "simp_rewrite": "selected_rewrite_rule",
                        "specified_rewrite": "specified_rewrite_rule",
                        "term_elaboration": "term_reference",
                    }.get(kind, "premise_reference")
                    prim_id = add_node(nodes, {
                        "id": f"prim:{h(f'{op_id}:{j}:{prem_name}', 20)}",
                        "kind": "primitive_operation",
                        "operation_kind": prim_role,
                        "direction": dirs[j] if j < len(dirs) else "?",
                    })
                    write_edge(op_id, prim_id, "unpacks_to", order=j)
                    write_edge(prim_id, decl_id, prim_role, order=j, direction=dirs[j] if j < len(dirs) else "?")
                    stats["premise_edges_by_kind"][prim_role] += 1
                    prem_recs.append({"name": prem_name, "direction": dirs[j] if j < len(dirs) else "?", "path": prem.get("path", "")})

                op = {
                    "id": op_id,
                    "theorem": thm_name,
                    "file_path": file_path,
                    "category": category(file_path),
                    "split": split,
                    "tactic_index": i,
                    "tactic": tactic,
                    "tactic_head": head,
                    "operation_kind": kind,
                    "before_goal_hash": before_hash,
                    "after_goal_hash": after_hash,
                    "before_goal_excerpt": before_excerpt,
                    "after_goal_excerpt": after_excerpt,
                    "premises": prem_recs,
                }
                ops_f.write(json.dumps(op, ensure_ascii=False) + "\n")
                op_count += 1
                stats["primitive_ops"] += 1
                stats["ops_by_kind"][kind] += 1
                stats["ops_by_head"][head] += 1

    with nodes_path.open("w") as f:
        for rec in nodes.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        **{k: v for k, v in stats.items() if not isinstance(v, Counter)},
        "graph_nodes": len(nodes),
        "graph_edges": edge_count,
        "ops_by_kind": dict(stats["ops_by_kind"].most_common()),
        "ops_by_head": dict(stats["ops_by_head"].most_common(30)),
        "premise_edges_by_kind": dict(stats["premise_edges_by_kind"].most_common()),
        "outputs": {
            "primitive_ops": str(ops_path),
            "graph_nodes": str(nodes_path),
            "graph_edges": str(edges_path),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def prem_key(op: dict) -> tuple[str, ...]:
    vals = [f'{p["direction"]}:{p["name"]}' for p in op["premises"]]
    if op["operation_kind"] == "simp_search":
        vals = sorted(vals)
    return tuple(vals)


def op_signature(op: dict, max_premises: int) -> tuple | None:
    ps = prem_key(op)
    if op["operation_kind"] not in SEARCHABLE_KINDS or not ps or len(ps) > max_premises:
        return None
    return ("single", op["operation_kind"], ps)


def example(op: dict) -> dict:
    return {
        "theorem": op["theorem"],
        "file_path": op["file_path"],
        "category": op["category"],
        "tactic_index": op["tactic_index"],
        "tactic": op["tactic"],
        "before": op["before_goal_excerpt"],
        "after": op["after_goal_excerpt"],
    }


def candidate_record(key: tuple, occs: list[dict], max_examples: int) -> dict:
    kind = "single_step" if key[0] == "single" else "two_step_chain"
    theorems = {o["theorem"] for o in occs}
    files = {o["file_path"] for o in occs}
    cats = Counter(o["category"] for o in occs)
    support = len(occs)
    score = support * math.log2(1 + len(theorems)) * math.log2(2 + len(cats))
    rec = {
        "candidate_id": f"cand:{h(repr(key), 20)}",
        "candidate_kind": kind,
        "support": support,
        "distinct_theorems": len(theorems),
        "distinct_files": len(files),
        "premise_count": key_premise_count(key),
        "categories": dict(cats.most_common()),
        "score": score,
        "verification_status": "candidate_from_existing_verified_proofs_not_new_lean_declaration",
        "examples": [example(o) for o in occs[:max_examples]],
    }
    if kind == "single_step":
        _, op_kind_value, premises = key
        rec.update({
            "operation_kind": op_kind_value,
            "premises": list(premises),
            "interpretation": "Repeated lower-level rewrite/simplification subgraph; candidate should be verified by synthesizing a lemma and replacing the repeated subgraph.",
        })
    else:
        _, first, second = key
        rec.update({
            "chain": [
                {"operation_kind": first[0], "premises": list(first[1])},
                {"operation_kind": second[0], "premises": list(second[1])},
            ],
            "interpretation": "Repeated two-operation proof-transform subgraph; candidate may summarize a common rewrite/simplify chain.",
        })
    return rec


def key_premise_count(key: tuple) -> int:
    if key[0] == "single":
        return len(key[2])
    return len(key[1][1]) + len(key[2][1])


def search(args: argparse.Namespace) -> None:
    ops_path = Path(args.ops)
    if not ops_path.exists():
        raise SystemExit(f"missing {ops_path}; run build first")
    singles: dict[tuple, list[dict]] = defaultdict(list)
    chains: dict[tuple, list[dict]] = defaultdict(list)
    prev_by_theorem: dict[str, dict] = {}

    with ops_path.open() as f:
        for line in f:
            op = json.loads(line)
            sig = op_signature(op, args.max_premises)
            if sig:
                singles[sig].append(op)
            prev = prev_by_theorem.get(op["theorem"])
            if prev:
                s1 = op_signature(prev, args.max_premises)
                s2 = sig
                if s1 and s2:
                    occ = dict(prev)
                    occ["tactic_index"] = f'{prev["tactic_index"]}-{op["tactic_index"]}'
                    occ["tactic"] = f'{prev["tactic"]} ;; {op["tactic"]}'
                    occ["after_goal_hash"] = op["after_goal_hash"]
                    occ["after_goal_excerpt"] = op["after_goal_excerpt"]
                    chains[("chain2", (s1[1], s1[2]), (s2[1], s2[2]))].append(occ)
            prev_by_theorem[op["theorem"]] = op

    records = []
    for bucket in (singles, chains):
        for key, occs in bucket.items():
            if (
                key_premise_count(key) >= args.min_premises
                and len(occs) >= args.min_support
                and len({o["theorem"] for o in occs}) >= args.min_theorems
            ):
                records.append(candidate_record(key, occs, args.max_examples))
    records.sort(key=lambda r: (-r["score"], -r["support"], r["candidate_id"]))

    out_path = OUT / "motif_candidates.jsonl"
    top_path = OUT / "motif_candidates_top.json"
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    top_path.write_text(json.dumps(records[:args.top], indent=2, ensure_ascii=False))

    summary_path = OUT / "motif_summary.json"
    summary = {
        "min_support": args.min_support,
        "min_theorems": args.min_theorems,
        "min_premises": args.min_premises,
        "max_premises": args.max_premises,
        "candidates": len(records),
        "top_candidates": str(top_path),
        "all_candidates": str(out_path),
        "top": records[: min(10, len(records))],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--data-path", default=str(LD_RANDOM))
    b.add_argument("--max-theorems", type=int)
    b.set_defaults(func=build)

    s = sub.add_parser("search")
    s.add_argument("--ops", default=str(OUT / "primitive_ops.jsonl"))
    s.add_argument("--min-support", type=int, default=12)
    s.add_argument("--min-theorems", type=int, default=5)
    s.add_argument("--min-premises", type=int, default=2)
    s.add_argument("--max-premises", type=int, default=6)
    s.add_argument("--max-examples", type=int, default=5)
    s.add_argument("--top", type=int, default=50)
    s.set_defaults(func=search)

    a = sub.add_parser("all")
    a.add_argument("--data-path", default=str(LD_RANDOM))
    a.add_argument("--max-theorems", type=int)
    a.add_argument("--min-support", type=int, default=12)
    a.add_argument("--min-theorems", type=int, default=5)
    a.add_argument("--min-premises", type=int, default=2)
    a.add_argument("--max-premises", type=int, default=6)
    a.add_argument("--max-examples", type=int, default=5)
    a.add_argument("--top", type=int, default=50)
    a.set_defaults(func=lambda args: (build(args), search(argparse.Namespace(
        ops=str(OUT / "primitive_ops.jsonl"),
        min_support=args.min_support,
        min_theorems=args.min_theorems,
        min_premises=args.min_premises,
        max_premises=args.max_premises,
        max_examples=args.max_examples,
        top=args.top,
    ))))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
