#!/usr/bin/env python3
"""Extra experiments for the MWM PI writeup.

Subcommands
-----------
path-sweep   : reachability and path-length histogram over random (B, A) pairs
               in the proof DAG, plus per-source BFS frontier sizes.
link-pred    : edge-vs-non-edge prediction from spectral features at the
               endpoints (logreg, with Hadamard product).
tactic-pred  : predict tactic_head (rw/simp/exact/...) of an edge from
               concatenated endpoint spectral features. Honest test of
               whether eigencoords carry tactic-level information.

Outputs land under results/paths/ and results/spectral/ alongside what the
overnight pipeline already produced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EDGES_PATH = RESULTS / "paths/dependency_edges.jsonl"
NODES_PATH = RESULTS / "paths/nodes.jsonl"
FEATURES_PATH = RESULTS / "spectral/spectral_features.csv"


def stable_bucket(s: str, mod: int = 10) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_adjacency():
    """Return (adj, sources, targets, edge_set, edge_records)."""
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    sources: set[str] = set()
    targets: set[str] = set()
    edge_set: set[tuple[str, str]] = set()
    edge_records: list[tuple[str, str, str, str]] = []  # source, target, tactic_head, source_cat
    with EDGES_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            s, t = rec["source"], rec["target"]
            adj[s].append((t, rec.get("tactic_head", "")))
            sources.add(s)
            targets.add(t)
            edge_set.add((s, t))
            edge_records.append((s, t, rec.get("tactic_head", ""), rec.get("source_category", "UNKNOWN")))
    return adj, sources, targets, edge_set, edge_records


def load_nodes():
    nodes = {}
    with NODES_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            nodes[rec["name"]] = rec
    return nodes


def bfs_path(adj, start, oracle, max_depth):
    if start == oracle:
        return 0, [start]
    parent = {start: None}
    depth = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        if depth[cur] >= max_depth:
            continue
        for nxt, _tac in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = cur
            depth[nxt] = depth[cur] + 1
            if nxt == oracle:
                # reconstruct
                path = [nxt]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return depth[nxt], list(reversed(path))
            q.append(nxt)
    return -1, []


def bfs_frontier(adj, start, max_depth):
    """Return list of frontier sizes at depths 1..max_depth."""
    seen = {start}
    frontier = [start]
    sizes = []
    for _ in range(max_depth):
        nxt = []
        for cur in frontier:
            for c, _tac in adj.get(cur, []):
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        sizes.append(len(nxt))
        if not nxt:
            break
        frontier = nxt
    return sizes


# ------------------------- path sweep ----------------------------

def path_sweep(args):
    print("Loading edges...")
    adj, sources, targets, edge_set, _ = load_adjacency()
    print(f"  {len(sources)} sources, {len(targets)} targets, {len(edge_set)} edges")
    all_nodes = list(sources | targets)
    source_nodes = [s for s in sources if len(adj[s]) > 0]
    rng = random.Random(args.seed)

    # 1) reachability between random pairs (B, A)
    pairs = []
    for _ in range(args.pairs):
        b = rng.choice(source_nodes)
        a = rng.choice(all_nodes)
        if a == b:
            continue
        pairs.append((b, a))

    found_lengths = []
    not_found = 0
    for b, a in pairs:
        d, _ = bfs_path(adj, b, a, args.max_depth)
        if d >= 0:
            found_lengths.append(d)
        else:
            not_found += 1
    found_count = len(found_lengths)
    pair_total = len(pairs)
    found_hist = Counter(found_lengths)

    # 2) ancestor-conditioned pairs: pick B, walk to depth d, take A from frontier
    cond = {d: [] for d in range(1, args.max_depth + 1)}
    sample_b = rng.sample(source_nodes, min(len(source_nodes), args.frontier_starts))
    frontier_sizes = []
    for b in sample_b:
        sizes = bfs_frontier(adj, b, args.max_depth)
        frontier_sizes.append(sizes)
        seen = {b}
        cur = [b]
        for d in range(1, args.max_depth + 1):
            nxt = []
            for x in cur:
                for c, _ in adj.get(x, []):
                    if c not in seen:
                        seen.add(c)
                        nxt.append(c)
            if not nxt:
                break
            cond[d].append(len(nxt))
            cur = nxt

    # collect frontier statistics
    by_depth = []
    for d in range(args.max_depth):
        sizes_d = [s[d] for s in frontier_sizes if len(s) > d]
        if not sizes_d:
            continue
        by_depth.append({
            "depth": d + 1,
            "n_sources": len(sizes_d),
            "mean": float(np.mean(sizes_d)),
            "median": float(np.median(sizes_d)),
            "p90": float(np.percentile(sizes_d, 90)),
            "p99": float(np.percentile(sizes_d, 99)),
            "max": int(np.max(sizes_d)),
        })

    out = {
        "config": {
            "pairs": args.pairs,
            "max_depth": args.max_depth,
            "frontier_starts": args.frontier_starts,
            "seed": args.seed,
        },
        "pair_reachability": {
            "pair_total": pair_total,
            "found": found_count,
            "found_fraction": found_count / max(1, pair_total),
            "length_histogram": dict(sorted(found_hist.items())),
            "length_mean_given_found": float(np.mean(found_lengths)) if found_lengths else None,
            "length_median_given_found": float(np.median(found_lengths)) if found_lengths else None,
        },
        "ancestor_frontier_per_source": by_depth,
    }
    out_path = RESULTS / "paths/path_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    # SVG of path-length histogram
    svg_path = RESULTS / "paths/path_length_histogram.svg"
    write_path_histogram(svg_path, found_hist, args.max_depth, found_count, pair_total)
    print(f"Wrote {svg_path}")


def write_path_histogram(out: Path, hist: Counter, max_depth: int, found: int, total: int) -> None:
    lengths = list(range(1, max_depth + 1))
    counts = [hist.get(d, 0) for d in lengths]
    height = 360
    width = 720
    left, top, bottom = 70, 50, 80
    plot_h = height - top - bottom
    plot_w = width - left - 30
    bar_w = plot_w / len(lengths) - 6
    max_count = max(counts) or 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="22" font-family="Arial" font-size="16">Path-length distribution (BFS over proof DAG)</text>',
        f'<text x="{left}" y="40" font-family="Arial" font-size="12" fill="#666">found {found} of {total} random theorem pairs reachable within depth {max_depth}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - 20}" y2="{top + plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
    ]
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        parts.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{width-20}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="20" y="{y+4:.1f}" font-family="Arial" font-size="11">{int(tick*max_count)}</text>')
    for i, (d, c) in enumerate(zip(lengths, counts)):
        x = left + i * (bar_w + 6) + 6
        h = c / max_count * plot_h
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2563eb"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 16}" font-family="Arial" font-size="11" text-anchor="middle">{d}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 4:.1f}" font-family="Arial" font-size="10" text-anchor="middle">{c}</text>')
    parts.append(f'<text x="{(left + width - 20) / 2}" y="{height - 30}" font-family="Arial" font-size="12" text-anchor="middle">path length (edges)</text>')
    parts.append('</svg>')
    out.write_text("\n".join(parts))


# ------------------------- spectral helpers ----------------------------

def load_spectral_features():
    print("Loading spectral features...")
    name_to_idx = {}
    rows = []
    with FEATURES_PATH.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        low_cols = [c for c in cols if c.startswith("low_")]
        high_cols = [c for c in cols if c.startswith("high_")]
        coords = []
        degrees = []
        cats = []
        for i, r in enumerate(reader):
            name_to_idx[r["name"]] = i
            cats.append(r["category"])
            degrees.append(float(r["degree"]))
            vec = [float(r[c]) for c in low_cols] + [float(r[c]) for c in high_cols]
            coords.append(vec)
        coords = np.array(coords, dtype=np.float32)
        degrees = np.array(degrees, dtype=np.float32)
    print(f"  {len(name_to_idx)} nodes, low={len(low_cols)}, high={len(high_cols)}")
    return name_to_idx, coords, degrees, np.array(cats), low_cols, high_cols


def standardize(train, *others):
    mu = train.mean(axis=0, keepdims=True)
    sig = train.std(axis=0, keepdims=True)
    sig[sig < 1e-8] = 1.0
    return [(arr - mu) / sig for arr in (train, *others)]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def train_logreg_binary(x_train, y_train, x_test, y_test, *, lr=0.15, epochs=40, batch=4096, weight_decay=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n, d = x_train.shape
    w = rng.normal(0, 0.01, size=(d,)).astype(np.float32)
    b = np.float32(0.0)
    best = None
    for ep in range(epochs):
        cur_lr = lr * (0.5 ** (ep // max(1, epochs // 3)))
        order = rng.permutation(n)
        for i in range(0, n, batch):
            ix = order[i : i + batch]
            z = x_train[ix] @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = (p - y_train[ix]) / len(ix)
            w -= cur_lr * (x_train[ix].T @ g + weight_decay * w)
            b -= cur_lr * g.sum()
        z_test = x_test @ w + b
        p_test = 1.0 / (1.0 + np.exp(-z_test))
        pred = (p_test > 0.5).astype(np.int32)
        acc = float(np.mean(pred == y_test))
        auc = roc_auc(y_test, p_test)
        if best is None or auc > best["auc"]:
            best = {"acc": acc, "auc": float(auc), "epoch": ep}
    return best


def roc_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if len(pos) * len(neg) > 2_000_000:
        sample = 2_000_000 // len(pos)
        neg = rng.choice(neg, size=min(len(neg), sample), replace=False)
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def train_logreg_multi(x_train, y_train, x_test, y_test, n_classes, *, lr=0.15, epochs=40, batch=4096, weight_decay=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n, d = x_train.shape
    w = rng.normal(0, 0.01, size=(d, n_classes)).astype(np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    yh = np.zeros((n, n_classes), dtype=np.float32)
    yh[np.arange(n), y_train] = 1.0
    best = None
    for ep in range(epochs):
        cur_lr = lr * (0.5 ** (ep // max(1, epochs // 3)))
        order = rng.permutation(n)
        for i in range(0, n, batch):
            ix = order[i : i + batch]
            p = softmax(x_train[ix] @ w + b)
            g = (p - yh[ix]) / len(ix)
            w -= cur_lr * (x_train[ix].T @ g + weight_decay * w)
            b -= cur_lr * g.sum(axis=0)
        probs = softmax(x_test @ w + b)
        pred = probs.argmax(axis=1)
        acc = float(np.mean(pred == y_test))
        f1s = []
        for c in range(n_classes):
            tp = np.sum((y_test == c) & (pred == c))
            fp = np.sum((y_test != c) & (pred == c))
            fn = np.sum((y_test == c) & (pred != c))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        macro_f1 = float(np.mean(f1s))
        if best is None or macro_f1 > best["macro_f1"]:
            best = {"accuracy": acc, "macro_f1": macro_f1, "epoch": ep}
    return best


# ------------------------- link prediction ----------------------------

def link_pred(args):
    name_to_idx, coords, degrees, cats, low_cols, high_cols = load_spectral_features()
    print("Loading edges...")
    _, sources, targets, edge_set, _ = load_adjacency()
    edges = list(edge_set)
    rng = random.Random(args.seed)
    rng.shuffle(edges)

    # filter to nodes present in features
    valid_edges = [(s, t) for s, t in edges if s in name_to_idx and t in name_to_idx]
    print(f"  {len(valid_edges)} edges with both endpoints in features")
    sample = valid_edges[: args.n_pairs]

    # negatives
    in_features = list(name_to_idx)
    neg = []
    while len(neg) < len(sample):
        u = rng.choice(in_features)
        v = rng.choice(in_features)
        if u == v:
            continue
        if (u, v) in edge_set or (v, u) in edge_set:
            continue
        neg.append((u, v))

    def feats(pairs):
        out = np.zeros((len(pairs), 4 * coords.shape[1] + 2), dtype=np.float32)
        for i, (u, v) in enumerate(pairs):
            iu, iv = name_to_idx[u], name_to_idx[v]
            cu, cv = coords[iu], coords[iv]
            out[i, : coords.shape[1]] = cu
            out[i, coords.shape[1] : 2 * coords.shape[1]] = cv
            out[i, 2 * coords.shape[1] : 3 * coords.shape[1]] = cu * cv
            out[i, 3 * coords.shape[1] : 4 * coords.shape[1]] = np.abs(cu - cv)
            out[i, -2] = math.log1p(degrees[iu])
            out[i, -1] = math.log1p(degrees[iv])
        return out

    def deg_only(pairs):
        out = np.zeros((len(pairs), 2), dtype=np.float32)
        for i, (u, v) in enumerate(pairs):
            out[i, 0] = math.log1p(degrees[name_to_idx[u]])
            out[i, 1] = math.log1p(degrees[name_to_idx[v]])
        return out

    pos_feats = feats(sample)
    neg_feats = feats(neg)
    pos_deg = deg_only(sample)
    neg_deg = deg_only(neg)

    x = np.vstack([pos_feats, neg_feats])
    y = np.concatenate([np.ones(len(pos_feats)), np.zeros(len(neg_feats))]).astype(np.float32)
    xd = np.vstack([pos_deg, neg_deg])

    # bucket-based split (deterministic)
    pair_keys = [f"{u}::{v}" for u, v in sample] + [f"{u}::{v}" for u, v in neg]
    train_mask = np.array([stable_bucket(k) < 8 for k in pair_keys])
    test_mask = ~train_mask

    x_train, x_test = standardize(x[train_mask], x[test_mask])
    xd_train, xd_test = standardize(xd[train_mask], xd[test_mask])
    y_train, y_test = y[train_mask], y[test_mask]

    full = train_logreg_binary(x_train, y_train, x_test, y_test)
    deg_only_metrics = train_logreg_binary(xd_train, y_train, xd_test, y_test)

    out = {
        "config": {
            "n_pairs": args.n_pairs,
            "n_features": int(x.shape[1]),
            "feature_layout": "low(u)|high(u)|low(v)|high(v) -> + cu|cv|cu*cv|abs(cu-cv) + log1p(deg(u))|log1p(deg(v))",
        },
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "spectral_full": full,
        "degree_only": deg_only_metrics,
    }
    out_path = RESULTS / "spectral/link_prediction.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


# ------------------------- tactic-head prediction ----------------------------

TOP_TACTICS = ["rw", "simp", "have", "exact", "refine", "simp_rw", "simpa", "apply"]


def tactic_pred(args):
    name_to_idx, coords, degrees, cats, low_cols, high_cols = load_spectral_features()
    print("Loading edges...")
    _, _, _, _, edge_records = load_adjacency()
    rng = random.Random(args.seed)
    rng.shuffle(edge_records)
    keep = []
    for s, t, head, _src_cat in edge_records:
        if head in TOP_TACTICS and s in name_to_idx and t in name_to_idx:
            keep.append((s, t, head))
        if len(keep) >= args.n_edges:
            break
    print(f"  {len(keep)} edges with top-{len(TOP_TACTICS)} tactic heads")

    label_idx = {h: i for i, h in enumerate(TOP_TACTICS)}
    labels = np.array([label_idx[h] for _, _, h in keep])
    feats = np.zeros((len(keep), 2 * coords.shape[1] + 2), dtype=np.float32)
    for i, (u, v, _) in enumerate(keep):
        iu, iv = name_to_idx[u], name_to_idx[v]
        feats[i, : coords.shape[1]] = coords[iu]
        feats[i, coords.shape[1] : 2 * coords.shape[1]] = coords[iv]
        feats[i, -2] = math.log1p(degrees[iu])
        feats[i, -1] = math.log1p(degrees[iv])

    keys = [f"{u}::{v}" for u, v, _ in keep]
    train_mask = np.array([stable_bucket(k) < 8 for k in keys])
    test_mask = ~train_mask

    x_train, x_test = standardize(feats[train_mask], feats[test_mask])
    y_train, y_test = labels[train_mask], labels[test_mask]

    majority = int(np.bincount(y_train).argmax())
    maj_acc = float(np.mean(y_test == majority))

    spectral_metrics = train_logreg_multi(x_train, y_train, x_test, y_test, len(TOP_TACTICS))

    # degree-only baseline
    xd = feats[:, -2:]
    xd_train, xd_test = standardize(xd[train_mask], xd[test_mask])
    deg_metrics = train_logreg_multi(xd_train, y_train, xd_test, y_test, len(TOP_TACTICS))

    out = {
        "config": {
            "n_edges": len(keep),
            "tactic_heads": TOP_TACTICS,
        },
        "class_distribution_train": {h: int((y_train == i).sum()) for i, h in enumerate(TOP_TACTICS)},
        "majority_baseline": {"accuracy": maj_acc, "predicted_class": TOP_TACTICS[majority]},
        "degree_only": deg_metrics,
        "spectral_concat": spectral_metrics,
    }
    out_path = RESULTS / "spectral/tactic_head_prediction.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


# ------------------------- CLI ----------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("path-sweep")
    a.add_argument("--pairs", type=int, default=2000)
    a.add_argument("--max-depth", type=int, default=8)
    a.add_argument("--frontier-starts", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=path_sweep)

    b = sub.add_parser("link-pred")
    b.add_argument("--n-pairs", type=int, default=10000)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=link_pred)

    c = sub.add_parser("tactic-pred")
    c.add_argument("--n-edges", type=int, default=80000)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=tactic_pred)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
