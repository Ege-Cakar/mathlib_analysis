#!/usr/bin/env python3
"""Richer graph-geometric features for the proof-DAG and downstream tasks.

Subcommands
-----------
node2vec      Random walks + skip-gram via gensim. 64-d embedding per node.
              Then run a proof-length ridge regression with these features
              alongside degree.

betweenness   Approximate betweenness centrality by random-source Brandes.
              Saves per-node BC, top-30 hubs, regression with BC as a
              feature.

resistance    Effective resistance approximated from the 64 eigenvectors
              already on disk: R(u,v) ≈ Σᵢ (vᵢ(u)-vᵢ(v))²/λᵢ. Compute the
              resistance from each theorem to a fixed set of top-by-indegree
              hubs as a low-dim feature.

curvature     Ollivier–Ricci curvature on a sample of edges via scipy LP
              on the transportation problem (uniform measures over 1-hop
              neighbourhoods, BFS distance). Histogram + bottleneck list.

joint         Combine all features in a single proof-length regression for
              the headline number.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LD_DIR = ROOT / "Old/data/leandojo/leandojo_benchmark_4/random"
RESULTS = ROOT / "results"
EDGES_PATH = RESULTS / "paths/dependency_edges.jsonl"
NODES_PATH = RESULTS / "paths/nodes.jsonl"
SPECTRAL_CSV = RESULTS / "spectral/spectral_features.csv"
SPECTRAL_METRICS = RESULTS / "spectral/spectral_metrics.json"
OUT_DIR = RESULTS / "graph_geometry"


def stable_bucket(s: str, mod: int = 10) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def category(path: str | None) -> str:
    if not path:
        return "UNKNOWN"
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "Mathlib":
        return parts[1]
    return "UNKNOWN"


# ----------------------- shared loaders ---------------------------------

def load_undirected_adj():
    """Returns (names_list, name_to_idx, adj as list[set[int]], in_deg, out_deg)."""
    edges = []
    in_deg: Counter = Counter()
    out_deg: Counter = Counter()
    nodes = set()
    with EDGES_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            s = e["source"]; t = e["target"]
            edges.append((s, t))
            in_deg[t] += 1
            out_deg[s] += 1
            nodes.add(s); nodes.add(t)
    names = sorted(nodes)
    name_to_idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    adj = [set() for _ in range(n)]
    for s, t in edges:
        si = name_to_idx[s]; ti = name_to_idx[t]
        if si == ti:
            continue
        adj[si].add(ti)
        adj[ti].add(si)
    return names, name_to_idx, adj, in_deg, out_deg


def load_theorems():
    for split in ("train", "val", "test"):
        path = LD_DIR / f"{split}.json"
        if not path.exists():
            continue
        with path.open() as f:
            for thm in json.load(f):
                name = thm.get("full_name") or ""
                fp = thm.get("file_path") or ""
                traced = thm.get("traced_tactics") or []
                if not name or not traced:
                    continue
                yield {
                    "name": name,
                    "file_path": fp,
                    "category": category(fp),
                    "tactic_count": len(traced),
                    "statement_len": len(traced[0].get("state_before", "") or ""),
                }


def load_spectral_eigenvalues():
    """Returns (low_vals, high_vals_normalized_laplacian)."""
    metrics = json.loads(SPECTRAL_METRICS.read_text())
    g = metrics["graph"]
    low_vals = np.array(g["low_eigenvalues_normalized_adjacency"], dtype=np.float64)
    high_lap = np.array(g["high_laplacian_eigenvalues_approx"], dtype=np.float64)
    return low_vals, high_lap


def load_spectral_features_dict():
    """Returns {name: np.array(64)}; eigenvectors are columns low_1..32, high_1..32."""
    coords = {}
    with SPECTRAL_CSV.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        low_cols = [c for c in cols if c.startswith("low_")]
        high_cols = [c for c in cols if c.startswith("high_")]
        for row in reader:
            try:
                vec = [float(row[c]) for c in low_cols] + [float(row[c]) for c in high_cols]
                coords[row["name"]] = np.array(vec, dtype=np.float32)
            except KeyError:
                pass
    return coords, low_cols, high_cols


# ----------------------- regression scaffold ----------------------------

def standardize(train, *others):
    mu = train.mean(axis=0, keepdims=True)
    sig = train.std(axis=0, keepdims=True)
    sig[sig < 1e-8] = 1.0
    return [(arr - mu) / sig for arr in (train, *others)]


def fit_ridge(x, y, lam=1.0):
    n, d = x.shape
    x_aug = np.hstack([x, np.ones((n, 1), dtype=np.float64)])
    A = x_aug.T @ x_aug
    A[:-1, :-1] += lam * np.eye(d, dtype=np.float64)
    b = x_aug.T @ y
    sol = np.linalg.solve(A, b)
    return sol[:-1], float(sol[-1])


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def pearson(y_true, y_pred):
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def regress(name, x_train, y_train, x_test, y_test, lam=1.0):
    if x_train.shape[1] == 0:
        m = float(y_train.mean())
        return {"feature_set": name, "n_features": 0,
                "test": {"r2": 0.0, "pearson": 0.0, "rmse": float(np.sqrt(((y_test - m) ** 2).mean()))}}
    w, b = fit_ridge(x_train, y_train, lam=lam)
    yp = x_test @ w + b
    return {
        "feature_set": name,
        "n_features": int(x_train.shape[1]),
        "test": {
            "r2": r2(y_test, yp),
            "pearson": pearson(y_test, yp),
            "rmse": float(np.sqrt(((y_test - yp) ** 2).mean())),
        },
    }


# ----------------------- node2vec --------------------------------------

def random_walks(adj, walks_per_node: int, length: int, seed: int = 0):
    rng = random.Random(seed)
    n = len(adj)
    walks = []
    nbrs = [list(s) for s in adj]
    for _ in range(walks_per_node):
        order = list(range(n))
        rng.shuffle(order)
        for v in order:
            if not nbrs[v]:
                continue
            walk = [v]
            cur = v
            for _ in range(length - 1):
                if not nbrs[cur]:
                    break
                cur = rng.choice(nbrs[cur])
                walk.append(cur)
            walks.append(walk)
    return walks


def node2vec_cmd(args: argparse.Namespace) -> None:
    ensure_dir(OUT_DIR)
    print("loading graph...")
    names, n2i, adj, in_deg, out_deg = load_undirected_adj()
    print(f"  n={len(names)}, edges (undirected) ~= {sum(len(s) for s in adj) // 2}")

    print(f"random walks ({args.walks_per_node} per node, length {args.walk_length})...")
    t0 = time.perf_counter()
    walks = random_walks(adj, args.walks_per_node, args.walk_length, seed=args.seed)
    print(f"  {len(walks)} walks in {time.perf_counter() - t0:.1f}s")
    walks_str = [[str(v) for v in w] for w in walks]

    from gensim.models import Word2Vec
    print(f"training Word2Vec dim={args.dim}, window={args.window}, epochs={args.epochs}, workers={args.workers}")
    t0 = time.perf_counter()
    model = Word2Vec(
        sentences=walks_str,
        vector_size=args.dim,
        window=args.window,
        min_count=0,
        sg=1,
        negative=args.negative,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(f"  trained in {time.perf_counter() - t0:.1f}s")

    print("writing embeddings...")
    emb_path = OUT_DIR / "node2vec_embeddings.npz"
    keys = list(model.wv.key_to_index)
    idxs = np.array([int(k) for k in keys], dtype=np.int64)
    vecs = np.zeros((len(names), args.dim), dtype=np.float32)
    for k, i in zip(keys, range(len(keys))):
        vecs[int(k)] = model.wv.get_vector(k)
    np.savez_compressed(emb_path, names=np.array(names), vectors=vecs)
    print(f"  wrote {emb_path}")

    # downstream regression
    theorems = list(load_theorems())
    print(f"proof-length regression on {len(theorems)} theorems")
    rows = []
    for thm in theorems:
        if thm["name"] not in n2i:
            continue
        rows.append((thm, n2i[thm["name"]]))
    n = len(rows)
    print(f"  {n} theorems with embeddings")
    rng = np.random.default_rng(0)
    y = np.array([math.log1p(thm["tactic_count"]) for thm, _ in rows], dtype=np.float64)
    deg_feat = np.zeros((n, 2), dtype=np.float64)
    n2v_feat = np.zeros((n, args.dim), dtype=np.float64)
    for i, (thm, idx) in enumerate(rows):
        deg_feat[i, 0] = math.log1p(out_deg.get(thm["name"], 0))
        deg_feat[i, 1] = math.log1p(in_deg.get(thm["name"], 0))
        n2v_feat[i] = vecs[idx]
    train = np.array([stable_bucket(thm["name"]) < 8 for thm, _ in rows])
    test = ~train
    feature_sets = {
        "degree":            deg_feat,
        "node2vec":          n2v_feat,
        "node2vec+degree":   np.hstack([n2v_feat, deg_feat]),
    }
    reports = []
    for name, x in feature_sets.items():
        x_train, x_test = standardize(x[train], x[test])
        rep = regress(name, x_train, y[train], x_test, y[test])
        reports.append(rep)
        print(f"  {name:>20}  R²={rep['test']['r2']:.3f}  ρ={rep['test']['pearson']:.3f}")
    out = {
        "config": {
            "walks_per_node": args.walks_per_node,
            "walk_length": args.walk_length,
            "dim": args.dim,
            "window": args.window,
            "epochs": args.epochs,
            "negative": args.negative,
        },
        "n_theorems": n,
        "reports": reports,
    }
    (OUT_DIR / "node2vec_proof_length.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_DIR / 'node2vec_proof_length.json'}")


# ----------------------- betweenness ------------------------------------

def brandes_partial(adj, src: int, accum: np.ndarray) -> None:
    """Single-source Brandes, accumulates dependency contributions in `accum`."""
    n = len(adj)
    s = []
    P: list[list[int]] = [[] for _ in range(n)]
    sigma = np.zeros(n, dtype=np.float64)
    sigma[src] = 1.0
    d = np.full(n, -1, dtype=np.int32)
    d[src] = 0
    Q = deque([src])
    while Q:
        v = Q.popleft()
        s.append(v)
        for w in adj[v]:
            if d[w] < 0:
                d[w] = d[v] + 1
                Q.append(w)
            if d[w] == d[v] + 1:
                sigma[w] += sigma[v]
                P[w].append(v)
    delta = np.zeros(n, dtype=np.float64)
    while s:
        w = s.pop()
        for v in P[w]:
            delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
        if w != src:
            accum[w] += delta[w]


def betweenness_cmd(args: argparse.Namespace) -> None:
    ensure_dir(OUT_DIR)
    print("loading graph...")
    names, n2i, adj, in_deg, out_deg = load_undirected_adj()
    n = len(names)
    rng = random.Random(args.seed)
    sources = rng.sample(range(n), min(args.sources, n))
    print(f"approx betweenness from {len(sources)} random sources over n={n}")
    bc = np.zeros(n, dtype=np.float64)
    t0 = time.perf_counter()
    for k, src in enumerate(sources):
        brandes_partial(adj, src, bc)
        if (k + 1) % 25 == 0:
            print(f"  source {k+1}/{len(sources)}  elapsed {time.perf_counter()-t0:.1f}s")
    bc *= n / max(1, len(sources))  # rescale to estimate full BC

    # save
    np.savez_compressed(OUT_DIR / "betweenness.npz",
                        names=np.array(names),
                        bc=bc.astype(np.float32))
    order = bc.argsort()[::-1][:30]
    top = [{"rank": i + 1, "name": names[j], "bc": float(bc[j]),
            "out_deg": int(out_deg.get(names[j], 0)),
            "in_deg": int(in_deg.get(names[j], 0))}
           for i, j in enumerate(order)]

    # regression
    theorems = list(load_theorems())
    rows = [(thm, n2i.get(thm["name"])) for thm in theorems if thm["name"] in n2i]
    yy = np.array([math.log1p(thm["tactic_count"]) for thm, _ in rows], dtype=np.float64)
    deg_feat = np.zeros((len(rows), 2), dtype=np.float64)
    bc_feat = np.zeros((len(rows), 1), dtype=np.float64)
    for i, (thm, idx) in enumerate(rows):
        deg_feat[i, 0] = math.log1p(out_deg.get(thm["name"], 0))
        deg_feat[i, 1] = math.log1p(in_deg.get(thm["name"], 0))
        bc_feat[i, 0] = math.log1p(max(0.0, bc[idx]))
    train = np.array([stable_bucket(thm["name"]) < 8 for thm, _ in rows])
    test = ~train
    reports = []
    for name, x in [("bc", bc_feat), ("degree", deg_feat),
                    ("bc+degree", np.hstack([bc_feat, deg_feat]))]:
        xt, xs = standardize(x[train], x[test])
        rep = regress(name, xt, yy[train], xs, yy[test])
        reports.append(rep); print(f"  {name:>10}  R²={rep['test']['r2']:.3f}  ρ={rep['test']['pearson']:.3f}")

    (OUT_DIR / "betweenness_summary.json").write_text(json.dumps({
        "n_nodes": n,
        "sources_sampled": len(sources),
        "top_30": top,
        "regression": reports,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'betweenness_summary.json'}")


# ----------------------- effective resistance ---------------------------

def resistance_cmd(args: argparse.Namespace) -> None:
    ensure_dir(OUT_DIR)
    print("loading spectral features and eigenvalues...")
    coords, low_cols, high_cols = load_spectral_features_dict()
    low_vals, high_lap = load_spectral_eigenvalues()
    low_vals = np.asarray(low_vals[: len(low_cols)], dtype=np.float64)
    high_lap = np.asarray(high_lap[: len(high_cols)], dtype=np.float64)
    # Laplacian eigenvalues for the corresponding eigenvectors:
    # for low (largest adjacency) eigvec, λ_L = 1 - μ;
    # for high (most negative adjacency) we already have approx λ_L from JSON.
    lap_eigs = np.concatenate([1.0 - low_vals, high_lap]).astype(np.float64)
    eps = 1e-4
    lap_eigs = np.where(lap_eigs > eps, lap_eigs, np.inf)  # ignore null / near-null
    inv_lap = 1.0 / lap_eigs
    print(f"  using {np.isfinite(inv_lap).sum()} eigenvectors with non-null Laplacian eigenvalue")

    # Pick reference hubs: top-K by indegree among nodes that have features
    print("loading dep graph for indegree...")
    names_all, n2i_all, adj, in_deg, out_deg = load_undirected_adj()
    items = [(n, in_deg[n]) for n in coords]
    items.sort(key=lambda r: -r[1])
    hubs = [n for n, _ in items[: args.hubs]]
    print(f"hubs (top {len(hubs)} by indegree among feature nodes):")
    for h in hubs[:10]:
        print(f"  {h}  in_deg={in_deg[h]}")

    hub_vecs = np.stack([coords[h].astype(np.float64) for h in hubs])  # (H, 64)

    # resistance feature per theorem: vector of distances to each hub
    theorems = list(load_theorems())
    rows = [thm for thm in theorems if thm["name"] in coords]
    print(f"computing R(theorem, hubs) for {len(rows)} theorems...")
    n = len(rows)
    R = np.zeros((n, len(hubs)), dtype=np.float64)
    yy = np.zeros(n, dtype=np.float64)
    for i, thm in enumerate(rows):
        cu = coords[thm["name"]].astype(np.float64)
        diff = hub_vecs - cu[None, :]  # (H, 64)
        R[i] = (diff * diff * inv_lap[None, :]).sum(axis=1)
        yy[i] = math.log1p(thm["tactic_count"])

    # log1p so heavy tail behaves
    R = np.log1p(np.clip(R, 0.0, None))
    train = np.array([stable_bucket(thm["name"]) < 8 for thm in rows])
    test = ~train

    deg_feat = np.zeros((n, 2), dtype=np.float64)
    for i, thm in enumerate(rows):
        deg_feat[i, 0] = math.log1p(out_deg.get(thm["name"], 0))
        deg_feat[i, 1] = math.log1p(in_deg.get(thm["name"], 0))

    reports = []
    for name, x in [("resistance_to_hubs", R),
                    ("degree", deg_feat),
                    ("resistance+degree", np.hstack([R, deg_feat]))]:
        xt, xs = standardize(x[train], x[test])
        rep = regress(name, xt, yy[train], xs, yy[test])
        reports.append(rep); print(f"  {name:>22}  R²={rep['test']['r2']:.3f}  ρ={rep['test']['pearson']:.3f}")

    (OUT_DIR / "resistance_summary.json").write_text(json.dumps({
        "hubs": hubs,
        "n_theorems": n,
        "regression": reports,
        "n_eigenvectors_used": int(np.isfinite(inv_lap).sum()),
    }, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'resistance_summary.json'}")


# ----------------------- Ollivier–Ricci ---------------------------------

def or_curvature(adj, u: int, v: int, distances) -> float:
    """OR curvature κ(u,v) = 1 - W₁(unif N(u), unif N(v)) / d(u,v).
    `distances` is a callable (a, B) -> array of distances from a to each b in B.
    """
    Nu = list(adj[u]) if adj[u] else [u]
    Nv = list(adj[v]) if adj[v] else [v]
    nu = len(Nu); nv = len(Nv)
    # cost matrix: d(x, y) for x in Nu, y in Nv
    C = np.zeros((nu, nv), dtype=np.float64)
    for i, x in enumerate(Nu):
        C[i] = distances(x, Nv)
    # transportation LP: minimize <C, T> s.t. T1 = 1/nu, T^T 1 = 1/nv, T >= 0.
    from scipy.optimize import linprog
    n_vars = nu * nv
    A_eq = np.zeros((nu + nv, n_vars), dtype=np.float64)
    for i in range(nu):
        A_eq[i, i * nv : (i + 1) * nv] = 1.0
    for j in range(nv):
        A_eq[nu + j, j::nv] = 1.0
    b_eq = np.concatenate([np.full(nu, 1.0 / nu), np.full(nv, 1.0 / nv)])
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return float("nan")
    W1 = float(res.fun)
    return 1.0 - W1  # d(u,v) = 1 by construction (edge)


def bfs_distances(adj, src: int, targets: list[int], horizon: int = 4) -> np.ndarray:
    n = len(adj)
    d = np.full(n, -1, dtype=np.int32)
    d[src] = 0
    Q = deque([src])
    target_set = set(targets)
    found = 0
    while Q and found < len(targets):
        v = Q.popleft()
        if d[v] >= horizon:
            continue
        for w in adj[v]:
            if d[w] < 0:
                d[w] = d[v] + 1
                if w in target_set:
                    found += 1
                Q.append(w)
    out = np.zeros(len(targets), dtype=np.float64)
    for i, t in enumerate(targets):
        if d[t] < 0:
            out[i] = horizon + 1  # treat unreached as horizon+1
        else:
            out[i] = float(d[t])
    return out


def curvature_cmd(args: argparse.Namespace) -> None:
    ensure_dir(OUT_DIR)
    print("loading graph...")
    names, n2i, adj, in_deg, out_deg = load_undirected_adj()
    rng = random.Random(args.seed)
    edges_list = []
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v:
                edges_list.append((u, v))
    print(f"  {len(edges_list)} undirected edges; sampling {args.sample}")
    sample = rng.sample(edges_list, min(args.sample, len(edges_list)))

    curvs = []
    bottlenecks = []
    t0 = time.perf_counter()
    for k, (u, v) in enumerate(sample):
        if max(len(adj[u]), len(adj[v])) > args.max_neighborhood:
            continue  # skip giant neighborhoods (LP becomes huge)
        try:
            kappa = or_curvature(adj, u, v, lambda a, B, _adj=adj: bfs_distances(_adj, a, B, horizon=args.horizon))
        except Exception:
            continue
        if not math.isfinite(kappa):
            continue
        curvs.append(kappa)
        if kappa < args.bottleneck_threshold:
            bottlenecks.append({
                "u": names[u],
                "v": names[v],
                "kappa": float(kappa),
                "deg_u": len(adj[u]),
                "deg_v": len(adj[v]),
            })
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(sample)}  elapsed {time.perf_counter()-t0:.1f}s  "
                  f"mean κ={np.mean(curvs):.3f}  min κ={np.min(curvs):.3f}")

    curvs_arr = np.array(curvs)
    out = {
        "config": {
            "sample": args.sample,
            "max_neighborhood": args.max_neighborhood,
            "horizon": args.horizon,
            "bottleneck_threshold": args.bottleneck_threshold,
        },
        "n_evaluated": int(curvs_arr.size),
        "mean_kappa": float(curvs_arr.mean()) if curvs_arr.size else None,
        "median_kappa": float(np.median(curvs_arr)) if curvs_arr.size else None,
        "min_kappa": float(curvs_arr.min()) if curvs_arr.size else None,
        "max_kappa": float(curvs_arr.max()) if curvs_arr.size else None,
        "fraction_negative": float((curvs_arr < 0).mean()) if curvs_arr.size else None,
        "fraction_below_threshold": float((curvs_arr < args.bottleneck_threshold).mean()) if curvs_arr.size else None,
        "bottlenecks_top": sorted(bottlenecks, key=lambda r: r["kappa"])[:30],
        "histogram": np.histogram(curvs_arr, bins=20)[0].tolist() if curvs_arr.size else [],
        "histogram_edges": np.histogram(curvs_arr, bins=20)[1].tolist() if curvs_arr.size else [],
    }
    (OUT_DIR / "ollivier_ricci.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'ollivier_ricci.json'}")


# ----------------------- joint regression -------------------------------

def joint_cmd(args: argparse.Namespace) -> None:
    """Combine degree, node2vec, betweenness, and resistance features."""
    ensure_dir(OUT_DIR)
    n2v_path = OUT_DIR / "node2vec_embeddings.npz"
    bc_path = OUT_DIR / "betweenness.npz"
    res_path = OUT_DIR / "resistance_summary.json"
    if not n2v_path.exists():
        print("missing node2vec — run `node2vec` first"); return
    if not bc_path.exists():
        print("missing betweenness — run `betweenness` first"); return

    n2v = np.load(n2v_path, allow_pickle=True)
    n2v_names = list(n2v["names"])
    n2v_idx = {n: i for i, n in enumerate(n2v_names)}
    n2v_vecs = n2v["vectors"]
    bc = np.load(bc_path, allow_pickle=True)
    bc_names = list(bc["names"])
    bc_idx = {n: i for i, n in enumerate(bc_names)}
    bc_vec = bc["bc"]

    coords, _, _ = load_spectral_features_dict()
    low_vals, high_lap = load_spectral_eigenvalues()
    lap_eigs = np.concatenate([1.0 - np.asarray(low_vals[:32], dtype=np.float64),
                               np.asarray(high_lap[:32], dtype=np.float64)])
    inv_lap = np.where(lap_eigs > 1e-4, 1.0 / lap_eigs, 0.0)
    # use a small random hub set to keep features compact
    items = sorted(coords.keys(), key=lambda n: -bc_idx.get(n, -1))[: args.hubs]
    hubs = items[: args.hubs]
    hub_vecs = np.stack([coords[h].astype(np.float64) for h in hubs])

    names_all, n2i_all, adj, in_deg, out_deg = load_undirected_adj()
    theorems = list(load_theorems())
    rows = [t for t in theorems if t["name"] in n2v_idx and t["name"] in coords]
    print(f"joint regression on {len(rows)} theorems")
    n = len(rows)
    yy = np.array([math.log1p(t["tactic_count"]) for t in rows], dtype=np.float64)
    deg_f = np.zeros((n, 2), dtype=np.float64)
    n2v_f = np.zeros((n, n2v_vecs.shape[1]), dtype=np.float64)
    bc_f = np.zeros((n, 1), dtype=np.float64)
    res_f = np.zeros((n, len(hubs)), dtype=np.float64)
    for i, t in enumerate(rows):
        deg_f[i, 0] = math.log1p(out_deg.get(t["name"], 0))
        deg_f[i, 1] = math.log1p(in_deg.get(t["name"], 0))
        n2v_f[i] = n2v_vecs[n2v_idx[t["name"]]]
        bc_f[i, 0] = math.log1p(max(0.0, bc_vec[bc_idx.get(t["name"], 0)]))
        cu = coords[t["name"]].astype(np.float64)
        diff = hub_vecs - cu[None, :]
        res_f[i] = np.log1p((diff * diff * inv_lap[None, :]).sum(axis=1))

    train = np.array([stable_bucket(t["name"]) < 8 for t in rows])
    test = ~train
    sets = {
        "degree":             deg_f,
        "node2vec":           n2v_f,
        "betweenness":        bc_f,
        "resistance":         res_f,
        "node2vec+degree":    np.hstack([n2v_f, deg_f]),
        "all_combined":       np.hstack([deg_f, n2v_f, bc_f, res_f]),
    }
    reports = []
    for name, x in sets.items():
        xt, xs = standardize(x[train], x[test])
        rep = regress(name, xt, yy[train], xs, yy[test], lam=2.0)
        reports.append(rep); print(f"  {name:>22}  R²={rep['test']['r2']:.3f}  ρ={rep['test']['pearson']:.3f}")
    (OUT_DIR / "joint_proof_length.json").write_text(json.dumps({"reports": reports, "n_theorems": n}, indent=2))
    print(f"wrote {OUT_DIR / 'joint_proof_length.json'}")


# ----------------------- CLI -------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    nv = sub.add_parser("node2vec")
    nv.add_argument("--walks-per-node", type=int, default=10)
    nv.add_argument("--walk-length", type=int, default=40)
    nv.add_argument("--dim", type=int, default=64)
    nv.add_argument("--window", type=int, default=5)
    nv.add_argument("--epochs", type=int, default=5)
    nv.add_argument("--negative", type=int, default=5)
    nv.add_argument("--workers", type=int, default=4)
    nv.add_argument("--seed", type=int, default=0)
    nv.set_defaults(func=node2vec_cmd)

    bc = sub.add_parser("betweenness")
    bc.add_argument("--sources", type=int, default=200)
    bc.add_argument("--seed", type=int, default=0)
    bc.set_defaults(func=betweenness_cmd)

    res = sub.add_parser("resistance")
    res.add_argument("--hubs", type=int, default=15)
    res.set_defaults(func=resistance_cmd)

    cu = sub.add_parser("curvature")
    cu.add_argument("--sample", type=int, default=5000)
    cu.add_argument("--max-neighborhood", type=int, default=80)
    cu.add_argument("--horizon", type=int, default=4)
    cu.add_argument("--bottleneck-threshold", type=float, default=-0.5)
    cu.add_argument("--seed", type=int, default=0)
    cu.set_defaults(func=curvature_cmd)

    j = sub.add_parser("joint")
    j.add_argument("--hubs", type=int, default=15)
    j.set_defaults(func=joint_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
