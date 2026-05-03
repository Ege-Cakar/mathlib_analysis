#!/usr/bin/env python3
"""Overnight Mathlib geometry pipeline.

This script only trusts LeanDojo premise annotations for certified dependency
edges. Experimental prover/model work is kept in separate outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LD_ROOT = ROOT / "Old/data/leandojo/leandojo_benchmark_4"
LD_RANDOM = LD_ROOT / "random"
RESULTS = ROOT / "results"
CUTOFF = "2025-08-14"


def category(path: str | None) -> str:
    if not path:
        return "UNKNOWN"
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "Mathlib":
        return parts[1]
    return "UNKNOWN"


def tactic_head(tactic: str) -> str:
    tactic = tactic.strip()
    return tactic.split(None, 1)[0] if tactic else ""


def stable_bucket(s: str, mod: int = 10) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_leandojo(split_dir: Path = LD_RANDOM):
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.json"
        if not path.exists():
            continue
        with path.open() as f:
            for theorem in json.load(f):
                yield split, theorem


def build_deps(args: argparse.Namespace) -> None:
    out_dir = RESULTS / "paths"
    ensure_dir(out_dir)
    edges_path = out_dir / "dependency_edges.jsonl"
    nodes_path = out_dir / "nodes.jsonl"
    stats_path = out_dir / "dependency_stats.json"

    edges: dict[tuple[str, str], dict] = {}
    nodes: dict[str, dict] = {}
    theorem_count = traced_count = tactic_count = 0
    premise_occurrences = 0

    for split, thm in iter_leandojo(Path(args.data_path)):
        theorem_count += 1
        name = thm.get("full_name")
        if not name:
            continue
        file_path = thm.get("file_path", "")
        nodes.setdefault(
            name,
            {
                "name": name,
                "file_path": file_path,
                "category": category(file_path),
                "kind": "theorem",
            },
        )
        tactics = thm.get("traced_tactics", [])
        traced_count += bool(tactics)
        tactic_count += len(tactics)

        for i, tt in enumerate(tactics):
            ann = tt.get("annotated_tactic") or []
            provenances = ann[1] if isinstance(ann, list) and len(ann) > 1 else []
            for prov in provenances:
                target = prov.get("full_name")
                if not target:
                    continue
                premise_occurrences += 1
                target_file = prov.get("def_path", "")
                nodes.setdefault(
                    target,
                    {
                        "name": target,
                        "file_path": target_file,
                        "category": category(target_file),
                        "kind": "premise",
                    },
                )
                key = (name, target)
                if key not in edges:
                    edges[key] = {
                        "source": name,
                        "target": target,
                        "source_file": file_path,
                        "target_file": target_file,
                        "source_category": category(file_path),
                        "target_category": category(target_file),
                        "split": split,
                        "tactic_index": i,
                        "tactic": tt.get("tactic", ""),
                        "tactic_head": tactic_head(tt.get("tactic", "")),
                        "target_pos": prov.get("def_pos"),
                        "target_end_pos": prov.get("def_end_pos"),
                        "occurrences": 0,
                    }
                edges[key]["occurrences"] += 1

    with edges_path.open("w") as f:
        for rec in sorted(edges.values(), key=lambda r: (r["source"], r["target"])):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with nodes_path.open("w") as f:
        for rec in sorted(nodes.values(), key=lambda r: str(r["name"])):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "theorems": theorem_count,
        "theorems_with_traces": traced_count,
        "tactics": tactic_count,
        "unique_dependency_edges": len(edges),
        "premise_occurrences": premise_occurrences,
        "nodes": len(nodes),
        "expected_unique_dependency_edges": 309396,
        "matches_expected_unique_dependency_edges": len(edges) == 309396,
        "top_source_categories": Counter(e["source_category"] for e in edges.values()).most_common(20),
        "top_tactic_heads": Counter(e["tactic_head"] for e in edges.values()).most_common(20),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote {edges_path}")
    print(f"Wrote {nodes_path}")
    print(f"Wrote {stats_path}")


def load_edges(edges_path: Path):
    edges = []
    with edges_path.open() as f:
        for line in f:
            edges.append(json.loads(line))
    return edges


def find_path(args: argparse.Namespace) -> None:
    edges_path = Path(args.edges)
    if not edges_path.exists():
        build_deps(argparse.Namespace(data_path=str(LD_RANDOM)))
    adj: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for edge in load_edges(edges_path):
        adj[edge["source"]].append((edge["target"], edge))

    q = deque([args.goal])
    parent: dict[str, tuple[str | None, dict | None]] = {args.goal: (None, None)}
    found = args.goal == args.oracle
    while q and not found:
        cur = q.popleft()
        if len(reconstruct(parent, cur)) >= args.max_depth:
            continue
        for nxt, edge in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = (cur, edge)
            if nxt == args.oracle:
                found = True
                break
            q.append(nxt)

    result = {
        "oracle": args.oracle,
        "goal": args.goal,
        "found": found,
        "edge_count": None,
        "path": [],
    }
    if found:
        path_edges = []
        cur = args.oracle
        while parent[cur][0] is not None:
            prev, edge = parent[cur]
            path_edges.append(edge)
            cur = prev
        result["path"] = list(reversed(path_edges))
        result["edge_count"] = len(result["path"])

    if args.output:
        out = Path(args.output)
        ensure_dir(out.parent)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def reconstruct(parent: dict[str, tuple[str | None, dict | None]], node: str) -> list[str]:
    path = [node]
    while parent[node][0] is not None:
        node = parent[node][0]  # type: ignore[assignment]
        path.append(node)
    return list(reversed(path))


def example_paths(args: argparse.Namespace) -> None:
    out_dir = RESULTS / "paths"
    ensure_dir(out_dir)
    examples = [
        ("WithLp.prod_norm_eq_add", "WithLp.prod_nnnorm_eq_add"),
        ("CategoryTheory.isIso_of_mono_of_epi", "CategoryTheory.ShortComplex.isIso₂_of_shortExact_of_isIso₁₃"),
        ("Algebra.trace_localization", "Algebra.intTrace_eq_trace"),
    ]
    written = []
    for oracle, goal in examples:
        out = out_dir / f"path_{safe_name(goal)}__to__{safe_name(oracle)}.json"
        ns = argparse.Namespace(
            edges=str(out_dir / "dependency_edges.jsonl"),
            oracle=oracle,
            goal=goal,
            max_depth=8,
            output=str(out),
        )
        try:
            find_path(ns)
            written.append(str(out))
        except Exception as exc:
            written.append(f"{goal} -> {oracle}: {exc}")
    (out_dir / "example_paths_manifest.json").write_text(json.dumps(written, indent=2))


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:140]


def read_nodes(nodes_path: Path) -> dict[str, dict]:
    nodes = {}
    with nodes_path.open() as f:
        for line in f:
            rec = json.loads(line)
            nodes[rec["name"]] = rec
    return nodes


def spectral(args: argparse.Namespace) -> None:
    out_dir = RESULTS / "spectral"
    ensure_dir(out_dir)
    edges_path = RESULTS / "paths/dependency_edges.jsonl"
    nodes_path = RESULTS / "paths/nodes.jsonl"
    if not edges_path.exists() or not nodes_path.exists():
        build_deps(argparse.Namespace(data_path=str(LD_RANDOM)))

    nodes_meta = read_nodes(nodes_path)
    edges = load_edges(edges_path)
    names = sorted({e["source"] for e in edges} | {e["target"] for e in edges})
    idx = {n: i for i, n in enumerate(names)}
    pairs = sorted({tuple(sorted((idx[e["source"]], idx[e["target"]]))) for e in edges if e["source"] != e["target"]})
    u = np.array([p[0] for p in pairs], dtype=np.int64)
    v = np.array([p[1] for p in pairs], dtype=np.int64)
    n = len(names)
    deg = np.zeros(n, dtype=np.float32)
    np.add.at(deg, u, 1)
    np.add.at(deg, v, 1)
    inv = np.zeros_like(deg)
    nz = deg > 0
    inv[nz] = 1.0 / np.sqrt(deg[nz])

    k = args.k + 1
    low_vals, low_vecs = subspace(normalized_matvec, (u, v, inv), n, k, args.iters, seed=0, sign=1.0)
    high_vals, high_vecs = subspace(normalized_matvec, (u, v, inv), n, args.k, args.iters, seed=1, sign=-1.0)
    low_vecs = low_vecs[:, 1 : args.k + 1] if low_vecs.shape[1] > args.k else low_vecs[:, : args.k]
    low_vals = low_vals[1 : args.k + 1] if len(low_vals) > args.k else low_vals[: args.k]

    feature_path = out_dir / "spectral_features.csv"
    fields = ["name", "file_path", "category", "degree"]
    fields += [f"low_{i+1}" for i in range(low_vecs.shape[1])]
    fields += [f"high_{i+1}" for i in range(high_vecs.shape[1])]
    with feature_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, name in enumerate(names):
            meta = nodes_meta.get(name, {})
            row = {
                "name": name,
                "file_path": meta.get("file_path", ""),
                "category": meta.get("category", "UNKNOWN"),
                "degree": int(deg[i]),
            }
            for j in range(low_vecs.shape[1]):
                row[f"low_{j+1}"] = float(low_vecs[i, j])
            for j in range(high_vecs.shape[1]):
                row[f"high_{j+1}"] = float(high_vecs[i, j])
            w.writerow(row)

    labels = np.array([nodes_meta.get(name, {}).get("category", "UNKNOWN") for name in names])
    valid = np.array([(lab != "UNKNOWN") and deg[i] > 0 for i, lab in enumerate(labels)])
    metrics = {
        "graph": {
            "nodes": n,
            "undirected_edges": len(pairs),
            "low_eigenvalues_normalized_adjacency": [float(x) for x in low_vals],
            "high_laplacian_eigenvalues_approx": [float(1 - x) for x in high_vals],
        },
        "sweeps": run_sweeps(names, labels, deg, low_vecs, high_vecs, valid),
    }
    (out_dir / "spectral_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_spectral_plots(out_dir, metrics)
    print(f"Wrote {feature_path}")
    print(f"Wrote {out_dir / 'spectral_metrics.json'}")


def normalized_matvec(x: np.ndarray, u: np.ndarray, v: np.ndarray, inv: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    scaled = x * inv[:, None]
    np.add.at(y, u, scaled[v])
    np.add.at(y, v, scaled[u])
    y *= inv[:, None]
    return y


def subspace(matvec, mat_args, n: int, k: int, iters: int, seed: int, sign: float):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k)).astype(np.float32)
    x, _ = np.linalg.qr(x)
    for _ in range(iters):
        y = sign * matvec(x, *mat_args)
        x, _ = np.linalg.qr(y)
    bx = matvec(x, *mat_args)
    small = x.T @ bx
    vals, vecs = np.linalg.eigh(small)
    order = np.argsort(vals)[::-1] if sign > 0 else np.argsort(vals)
    vals = vals[order]
    vecs = x @ vecs[:, order]
    return vals, vecs.astype(np.float32)


def run_sweeps(names, labels, deg, low, high, valid):
    y_all = labels
    train = np.array([stable_bucket(n) < 8 for n in names]) & valid
    test = (~train) & valid
    keep_cats = {c for c, n in Counter(y_all[train]).items() if n >= 25}
    train &= np.array([c in keep_cats for c in y_all])
    test &= np.array([c in keep_cats for c in y_all])

    out = {
        "majority": evaluate_predictions(y_all[test], np.repeat(Counter(y_all[train]).most_common(1)[0][0], test.sum())),
        "degree_only": centroid_predict(np.log1p(deg[:, None]), y_all, train, test),
        "namespace_prior": namespace_prior(names, y_all, train, test),
    }
    sweeps = {
        "low_5": low[:, :5],
        "low_10": low[:, :10],
        "high_5": high[:, :5],
        "high_10": high[:, :10],
        "low5_high5": np.hstack([low[:, :5], high[:, :5]]),
        "low10_high10": np.hstack([low[:, :10], high[:, :10]]),
    }
    for name, x in sweeps.items():
        out[name] = centroid_predict(x, y_all, train, test)
    out["n_train"] = int(train.sum())
    out["n_test"] = int(test.sum())
    out["categories"] = sorted(keep_cats)
    return out


def centroid_predict(x, y, train, test):
    cats = sorted(set(y[train]))
    centroids = np.vstack([x[train & (y == c)].mean(axis=0) for c in cats])
    xt = x[test]
    d2 = ((xt[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred = np.array([cats[i] for i in d2.argmin(axis=1)])
    top3 = [[cats[i] for i in row.argsort()[:3]] for row in d2]
    metrics = evaluate_predictions(y[test], pred)
    metrics["top3_accuracy"] = float(np.mean([yt in ps for yt, ps in zip(y[test], top3)]))
    return metrics


def namespace_prior(names, y, train, test):
    majority = Counter(y[train]).most_common(1)[0][0]
    table = {}
    for name, label, ok in zip(names, y, train):
        if ok:
            table.setdefault(name.split(".", 1)[0], Counter())[label] += 1
    pred = []
    for name in np.array(names)[test]:
        c = table.get(name.split(".", 1)[0])
        pred.append(c.most_common(1)[0][0] if c else majority)
    return evaluate_predictions(y[test], np.array(pred))


def evaluate_predictions(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    f1s = []
    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab))
        fp = np.sum((y_true != lab) & (y_pred == lab))
        fn = np.sum((y_true == lab) & (y_pred != lab))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s))}


def write_spectral_plots(out_dir: Path, metrics: dict) -> None:
    sweeps = {k: v for k, v in metrics["sweeps"].items() if isinstance(v, dict) and "accuracy" in v}
    names = list(sweeps)
    width, height = 980, 420
    left, top, bottom = 70, 40, 90
    plot_h = height - top - bottom
    gap = 18
    group_w = (width - left - 30 - gap * (len(names) - 1)) / max(1, len(names))
    bar_w = group_w * 0.38
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="25" font-family="Arial" font-size="18">Category Prediction From Spectral Features</text>',
        '<text x="760" y="25" font-family="Arial" font-size="13" fill="#2563eb">accuracy</text>',
        '<text x="835" y="25" font-family="Arial" font-size="13" fill="#dc2626">macro-F1</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - 20}" y2="{top + plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
    ]
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        parts.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{width-20}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="25" y="{y+4:.1f}" font-family="Arial" font-size="11">{tick:.1f}</text>')
    for i, name in enumerate(names):
        x = left + i * (group_w + gap)
        for j, (key, color) in enumerate((("accuracy", "#2563eb"), ("macro_f1", "#dc2626"))):
            val = float(sweeps[name][key])
            h = val * plot_h
            bx = x + j * bar_w
            by = top + plot_h - h
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" fill="{color}"/>')
        label = name.replace("_", " ")
        parts.append(
            f'<text x="{x + group_w/2:.1f}" y="{height-45}" font-family="Arial" '
            f'font-size="11" text-anchor="end" transform="rotate(-35 {x + group_w/2:.1f},{height-45})">{label}</text>'
        )
    parts.append("</svg>")
    (out_dir / "spectral_category_sweeps.svg").write_text("\n".join(parts))


DECL_RE = re.compile(r"^\s*(?:private\s+|protected\s+)?(theorem|lemma)\s+([A-Za-z0-9_'.«»]+)")
NS_RE = re.compile(r"^\s*namespace\s+([A-Za-z0-9_'.]+)")
END_RE = re.compile(r"^\s*end(?:\s+([A-Za-z0-9_'.]+))?\s*$")


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], text=True, capture_output=True, check=check)


def post_kimina_manifest(args: argparse.Namespace) -> None:
    out_dir = RESULTS / "prover_eval"
    ensure_dir(out_dir)
    out_path = out_dir / "post_kimina_manifest.jsonl"
    mathlib = Path(args.mathlib_dir) if args.mathlib_dir else None
    if not mathlib or not (mathlib / ".git").exists():
        write_blocked_manifest(out_dir, "No local Mathlib git checkout was supplied.")
        print(f"Blocked: pass --mathlib-dir pointing at a Mathlib checkout. Wrote {out_dir / 'post_kimina_manifest_status.json'}")
        return

    base = git(mathlib, "rev-list", "-n", "1", f"--before={args.cutoff} 23:59:59", "HEAD").stdout.strip()
    changed = git(mathlib, "diff", "--name-only", base, "HEAD", "--", "Mathlib").stdout.splitlines()
    lean_files = [p for p in changed if p.endswith(".lean")]
    random.Random(0).shuffle(lean_files)

    rows = []
    for rel in lean_files[: args.max_files]:
        cur_path = mathlib / rel
        if not cur_path.exists():
            continue
        current = cur_path.read_text(errors="ignore")
        old = git(mathlib, "show", f"{base}:{rel}", check=False)
        old_decls = {d["full_name"] for d in extract_decls(old.stdout if old.returncode == 0 else "", rel)}
        for decl in extract_decls(current, rel):
            if decl["full_name"] not in old_decls:
                rows.append(decl)
    rows.sort(key=lambda r: (r["category"], r["file_path"], r["line"], r["full_name"]))
    if args.limit:
        rows = stratified(rows, args.limit)

    if args.verify_sorry:
        for row in rows:
            ok, err = verify_sorry(mathlib, row)
            row["sorry_compiles"] = ok
            row["sorry_error"] = err[-2000:] if err else ""
        rows = [r for r in rows if r.get("sorry_compiles")]

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "post_kimina_manifest_stats.json").write_text(
        json.dumps(
            {
                "cutoff": args.cutoff,
                "base_commit": base,
                "changed_lean_files_scanned": len(lean_files[: args.max_files]),
                "manifest_rows": len(rows),
                "verify_sorry": bool(args.verify_sorry),
                "category_counts": Counter(r["category"] for r in rows).most_common(),
            },
            indent=2,
        )
    )
    print(f"Wrote {out_path}")


def write_blocked_manifest(out_dir: Path, reason: str) -> None:
    (out_dir / "post_kimina_manifest_status.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "reason": reason,
                "needed": "A local Mathlib git checkout after 2025-08-14.",
                "cutoff": CUTOFF,
            },
            indent=2,
        )
    )


def extract_decls(text: str, rel: str) -> list[dict]:
    out = []
    namespace: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if m := NS_RE.match(line):
            namespace.append(m.group(1))
            continue
        if END_RE.match(line) and namespace:
            namespace.pop()
            continue
        m = DECL_RE.match(line)
        if not m:
            continue
        raw_name = m.group(2)
        full_name = raw_name
        if "." not in raw_name and namespace:
            full_name = ".".join(namespace + [raw_name])
        header = declaration_header(lines, i)
        out.append(
            {
                "kind": m.group(1),
                "name": raw_name,
                "full_name": full_name.removeprefix("_root_."),
                "file_path": rel,
                "line": i + 1,
                "category": category(rel),
                "declaration_header": header,
            }
        )
    return out


def declaration_header(lines: list[str], start: int) -> str:
    chunk = []
    for line in lines[start : min(len(lines), start + 80)]:
        chunk.append(line)
        text = "\n".join(chunk)
        stripped = line.strip()
        if ":= by" in text:
            return text.split(":= by", 1)[0].rstrip() + " := by"
        if ":=" in line and not stripped.startswith(("let ", "letI ", "have ")):
            return text.rsplit(":=", 1)[0].rstrip() + " := by"
        if re.search(r"\sby\s*$", text):
            return re.sub(r"\sby\s*$", " := by", text).rstrip()
    return "\n".join(chunk[:12]).rstrip()


def stratified(rows: list[dict], limit: int) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    cats = sorted(by_cat, key=lambda c: len(by_cat[c]), reverse=True)
    out = []
    while len(out) < limit and cats:
        next_cats = []
        for c in cats:
            if by_cat[c] and len(out) < limit:
                out.append(by_cat[c].pop(0))
            if by_cat[c]:
                next_cats.append(c)
        cats = next_cats
    return out


def verify_sorry(mathlib: Path, row: dict) -> tuple[bool, str]:
    src = mathlib / row["file_path"]
    if src.exists():
        lines = src.read_text(errors="ignore").splitlines(keepends=True)
        start = max(0, int(row["line"]) - 1)
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if DECL_RE.match(lines[j]):
                end = j
                break
        replacement = row["declaration_header"].rstrip() + "\n  sorry\n\n"
        code = "".join(lines[:start]) + replacement + "".join(lines[end:])
    else:
        code = "import Mathlib\nimport Aesop\nset_option maxHeartbeats 0\n\n" + row["declaration_header"] + "\n  sorry\n"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / Path(row.get("file_path", "Check.lean")).name
        path.write_text(code)
        proc = subprocess.run(
            ["lake", "env", "lean", str(path)],
            cwd=mathlib,
            text=True,
            capture_output=True,
            timeout=120,
        )
    return proc.returncode == 0, proc.stdout + proc.stderr


def kimina_prompts(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    out = Path(args.output)
    ensure_dir(out.parent)
    with manifest.open() as f, out.open("w") as g:
        for line in f:
            row = json.loads(line)
            formal = row["declaration_header"] + "\n"
            prompt = (
                "Think about and solve the following problem step by step in Lean 4.\n"
                "# Formal statement:\n"
                "```lean4\n"
                "import Mathlib\nimport Aesop\nset_option maxHeartbeats 0\n"
                f"{formal}\n"
                "```\n"
                "Return exactly one Lean 4 code block containing a complete proof."
            )
            row["prompt"] = prompt
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")


def post_kimina_anchors(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    features = Path(args.features)
    out = Path(args.output)
    ensure_dir(out.parent)

    old: dict[str, dict] = {}
    with features.open() as f:
        reader = csv.DictReader(f)
        coord_fields = [c for c in reader.fieldnames or [] if c.startswith(("low_", "high_"))]
        for row in reader:
            old[row["name"]] = row
    suffix: dict[str, str | None] = {}
    for name in old:
        key = name.rsplit(".", 1)[-1]
        suffix[key] = name if key not in suffix else None

    fields = ["full_name", "file_path", "category", "line", "anchor_count", "anchors"] + [
        f"anchor_{c}" for c in coord_fields
    ]
    ident_re = re.compile(r"[^\W\d][\w'.]*", re.UNICODE)
    with manifest.open() as f, out.open("w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=fields)
        w.writeheader()
        for line in f:
            row = json.loads(line)
            hits = []
            seen = set()
            for tok in ident_re.findall(row.get("declaration_header", "")):
                if tok.startswith("_"):
                    continue
                local = tok.rsplit(".", 1)[-1]
                exact_ok = "." in tok or (local[:1].isupper())
                suffix_ok = len(local) > 1 and local[:1].isupper()
                hit = tok if exact_ok and tok in old else suffix.get(tok) if suffix_ok else None
                if hit and hit not in seen:
                    hits.append(hit)
                    seen.add(hit)
            out_row = {
                "full_name": row["full_name"],
                "file_path": row["file_path"],
                "category": row["category"],
                "line": row["line"],
                "anchor_count": len(hits),
                "anchors": ";".join(hits[:50]),
            }
            for c in coord_fields:
                vals = [float(old[h][c]) for h in hits]
                out_row[f"anchor_{c}"] = float(np.mean(vals)) if vals else ""
            w.writerow(out_row)
    print(f"Wrote {out}")


def report(args: argparse.Namespace) -> None:
    out = RESULTS / "report.md"
    ensure_dir(out.parent)
    deps = read_json(RESULTS / "paths/dependency_stats.json")
    spec = read_json(RESULTS / "spectral/spectral_metrics.json")
    clf = read_json(RESULTS / "spectral/classifier_sweep.json")
    manifest_status = read_json(RESULTS / "prover_eval/post_kimina_manifest_status.json")
    manifest_stats = read_json(RESULTS / "prover_eval/post_kimina_manifest_stats.json")
    kimina = RESULTS / "prover_eval/kimina_post_2025_08_14.jsonl"
    anchors = RESULTS / "spectral/post_kimina_anchor_features.csv"

    lines = [
        "# Overnight Mathlib Geometry Results",
        "",
        "## Certified dependency graph",
        f"- Unique LeanDojo premise edges: `{deps.get('unique_dependency_edges', 'not run')}`.",
        f"- Matches expected 309396 edge count: `{deps.get('matches_expected_unique_dependency_edges', 'not run')}`.",
        f"- Nodes with theorem/premise metadata: `{deps.get('nodes', 'not run')}`.",
        "",
        "## Spectral category prediction",
    ]
    if spec:
        sweeps = spec["sweeps"]
        for key in ["majority", "degree_only", "namespace_prior", "low_5", "low_10", "high_5", "high_10", "low5_high5", "low10_high10"]:
            if key in sweeps:
                m = sweeps[key]
                lines.append(f"- `{key}`: accuracy `{m['accuracy']:.3f}`, macro-F1 `{m['macro_f1']:.3f}`.")
        lines.append(f"- Train/test nodes: `{sweeps.get('n_train')}` / `{sweeps.get('n_test')}`.")
    else:
        lines.append("- Not run yet.")

    if clf:
        rows = []
        for spec_name, models in clf.get("models", {}).items():
            if "error" in models:
                continue
            for model_name, met in models.items():
                rows.append((met["accuracy"], met["macro_f1"], met["top3_accuracy"], spec_name, model_name))
        best_acc = max(rows) if rows else None
        best_f1 = max(rows, key=lambda r: r[1]) if rows else None
        lines += ["", "## Stronger spectral classifiers"]
        lines.append(
            "- Baselines: majority accuracy "
            f"`{clf['baselines']['majority']['accuracy']:.3f}`, namespace-prior accuracy "
            f"`{clf['baselines']['namespace_prior']['accuracy']:.3f}`."
        )
        if best_acc:
            lines.append(
                "- Best accuracy: "
                f"`{best_acc[0]:.3f}` using `{best_acc[4]}` on `{best_acc[3]}` "
                f"(macro-F1 `{best_acc[1]:.3f}`)."
            )
        if best_f1:
            lines.append(
                "- Best macro-F1: "
                f"`{best_f1[1]:.3f}` using `{best_f1[4]}` on `{best_f1[3]}` "
                f"(accuracy `{best_f1[0]:.3f}`)."
            )
        lines.append("- Interpretation: the neural classifier finds at most weak signal; top-level category remains mostly not captured by these eigencoordinates.")

    lines += ["", "## Post-Kimina prover eval"]
    if kimina.exists():
        n = passed = 0
        with kimina.open() as f:
            for line in f:
                n += 1
                passed += bool(json.loads(line).get("passed"))
        lines.append(f"- Kimina verified proofs: `{passed}/{n}`.")
    elif manifest_stats:
        lines.append(f"- Manifest rows ready for Kimina: `{manifest_stats.get('manifest_rows')}`.")
        lines.append(f"- `by sorry` precheck run locally: `{manifest_stats.get('verify_sorry')}`.")
        if (RESULTS / "prover_eval/kimina_prompts.jsonl").exists():
            lines.append("- Kimina prompts are written to `results/prover_eval/kimina_prompts.jsonl`.")
        if anchors.exists():
            with anchors.open() as f:
                anchor_rows = list(csv.DictReader(f))
            with_anchor = sum(int(r["anchor_count"]) > 0 for r in anchor_rows)
            lines.append(
                "- Post-Kimina spectral anchor features are written to "
                f"`results/spectral/post_kimina_anchor_features.csv` (`{with_anchor}/{len(anchor_rows)}` with anchors)."
            )
        lines.append("- Kimina model run not completed in this local workspace.")
    elif manifest_status:
        lines.append(f"- Manifest blocked: {manifest_status.get('reason')}")
    else:
        lines.append("- Not run yet.")

    lines += [
        "",
        "## ReProver status",
        "- ReProver is not installed in this workspace. The local outputs therefore contain the manifest and commands/data needed for the parallel run, not ReProver proof results.",
        "",
        "## Interpretation",
        "- Dependency paths are certified only in the sense that each edge appears in LeanDojo premise annotations from existing Mathlib proofs.",
        "- State-transition shortcuts and model generations are not counted unless Lean verifies them.",
        "- Post-Kimina, after 2025-08-14, is the clean holdout for Kimina-style models; older post-LeanDojo results are contamination-caveated for Kimina.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def all_pipeline(args: argparse.Namespace) -> None:
    build_deps(argparse.Namespace(data_path=args.data_path))
    example_paths(args)
    spectral(argparse.Namespace(k=args.k, iters=args.iters))
    post_kimina_manifest(
        argparse.Namespace(
            mathlib_dir=args.mathlib_dir,
            cutoff=args.cutoff,
            max_files=args.max_files,
            limit=args.manifest_limit,
            verify_sorry=False,
        )
    )
    report(args)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-deps")
    b.add_argument("--data-path", default=str(LD_RANDOM))
    b.set_defaults(func=build_deps)

    q = sub.add_parser("path")
    q.add_argument("--edges", default=str(RESULTS / "paths/dependency_edges.jsonl"))
    q.add_argument("--oracle", required=True)
    q.add_argument("--goal", required=True)
    q.add_argument("--max-depth", type=int, default=8)
    q.add_argument("--output")
    q.set_defaults(func=find_path)

    e = sub.add_parser("example-paths")
    e.set_defaults(func=example_paths)

    s = sub.add_parser("spectral")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--iters", type=int, default=30)
    s.set_defaults(func=spectral)

    m = sub.add_parser("post-kimina-manifest")
    m.add_argument("--mathlib-dir")
    m.add_argument("--cutoff", default=CUTOFF)
    m.add_argument("--max-files", type=int, default=400)
    m.add_argument("--limit", type=int, default=200)
    m.add_argument("--verify-sorry", action="store_true")
    m.set_defaults(func=post_kimina_manifest)

    kp = sub.add_parser("kimina-prompts")
    kp.add_argument("--manifest", default=str(RESULTS / "prover_eval/post_kimina_manifest.jsonl"))
    kp.add_argument("--output", default=str(RESULTS / "prover_eval/kimina_prompts.jsonl"))
    kp.set_defaults(func=kimina_prompts)

    ka = sub.add_parser("post-kimina-anchors")
    ka.add_argument("--manifest", default=str(RESULTS / "prover_eval/post_kimina_manifest.jsonl"))
    ka.add_argument("--features", default=str(RESULTS / "spectral/spectral_features.csv"))
    ka.add_argument("--output", default=str(RESULTS / "spectral/post_kimina_anchor_features.csv"))
    ka.set_defaults(func=post_kimina_anchors)

    r = sub.add_parser("report")
    r.set_defaults(func=report)

    a = sub.add_parser("all")
    a.add_argument("--data-path", default=str(LD_RANDOM))
    a.add_argument("--k", type=int, default=10)
    a.add_argument("--iters", type=int, default=30)
    a.add_argument("--mathlib-dir")
    a.add_argument("--cutoff", default=CUTOFF)
    a.add_argument("--max-files", type=int, default=400)
    a.add_argument("--manifest-limit", type=int, default=200)
    a.set_defaults(func=all_pipeline)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
