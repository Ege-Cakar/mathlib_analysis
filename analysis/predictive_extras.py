#!/usr/bin/env python3
"""Four predictive-power experiments on the Mathlib graph + states.

1. proof-length     Predict #tactics in a theorem's proof from
                    (degree, area, statement_length, spectral coords).
                    Baseline = mean predictor and degree-only.
2. dep-matrix       Cross-area citation matrix (source_cat → target_cat),
                    row-normalised, written as JSON + SVG heatmap.
3. tactic-by-area   tactic_head distribution per source area, written as
                    JSON + SVG heatmap.
4. spectral-only    Re-run experiment 1 with spectral features only, vs
                    degree only, vs the joint set. Final answer to "do
                    eigencoords help on a continuous target".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LD_DIR = ROOT / "Old/data/leandojo/leandojo_benchmark_4/random"
RESULTS = ROOT / "results"
EDGES_PATH = RESULTS / "paths/dependency_edges.jsonl"
NODES_PATH = RESULTS / "paths/nodes.jsonl"
SPECTRAL_PATH = RESULTS / "spectral/spectral_features.csv"
OUT_DIR = RESULTS / "predictive"
HEATMAP_DIR = OUT_DIR


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


# ----------------------- shared loading ---------------------------------

def load_theorems():
    """Yield dicts {name, file_path, category, tactic_count, statement_len, tactic_heads}."""
    for split in ("train", "val", "test"):
        path = LD_DIR / f"{split}.json"
        if not path.exists():
            continue
        with path.open() as f:
            for thm in json.load(f):
                name = thm.get("full_name") or ""
                if not name:
                    continue
                fp = thm.get("file_path") or ""
                traced = thm.get("traced_tactics") or []
                if not traced:
                    continue
                first_state = traced[0].get("state_before", "")
                tactic_heads = []
                for tt in traced:
                    head = (tt.get("tactic", "") or "").strip().split(None, 1)
                    tactic_heads.append(head[0] if head else "")
                yield {
                    "name": name,
                    "file_path": fp,
                    "category": category(fp),
                    "tactic_count": len(traced),
                    "statement_len": len(first_state),
                    "tactic_heads": tactic_heads,
                    "split": split,
                }


def load_edges_summary():
    """Returns:
      out_deg[name] = unique premises cited
      in_deg[name]  = unique theorems citing this name
      area_pair_counts[(src_cat, tgt_cat)] = total occurrences
      area_tactic_counts[(src_cat, tactic_head)] = total occurrences
    Streams the 165 MB edges JSONL.
    """
    out_deg: Counter = Counter()
    in_deg: Counter = Counter()
    area_pairs: Counter = Counter()
    area_tactic: Counter = Counter()
    with EDGES_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            s = e["source"]; t = e["target"]
            out_deg[s] += 1
            in_deg[t] += 1
            area_pairs[(e.get("source_category", "UNKNOWN"), e.get("target_category", "UNKNOWN"))] += e.get("occurrences", 1)
            area_tactic[(e.get("source_category", "UNKNOWN"), e.get("tactic_head", ""))] += e.get("occurrences", 1)
    return out_deg, in_deg, area_pairs, area_tactic


def load_spectral():
    coords = {}
    with SPECTRAL_PATH.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        low_cols = [c for c in cols if c.startswith("low_")][:10]
        high_cols = [c for c in cols if c.startswith("high_")][:10]
        for row in reader:
            try:
                vec = [float(row[c]) for c in low_cols] + [float(row[c]) for c in high_cols]
                coords[row["name"]] = np.array(vec, dtype=np.float32)
            except KeyError:
                pass
    return coords, low_cols, high_cols


# ----------------------- regression helpers -----------------------------

def standardize(train, *others):
    mu = train.mean(axis=0, keepdims=True)
    sig = train.std(axis=0, keepdims=True)
    sig[sig < 1e-8] = 1.0
    return [(arr - mu) / sig for arr in (train, *others)]


def fit_ridge(x, y, lam=1.0):
    """x: (n, d), y: (n,). Returns (w, b)."""
    n, d = x.shape
    x_aug = np.hstack([x, np.ones((n, 1), dtype=np.float32)])
    A = x_aug.T @ x_aug
    A[:-1, :-1] += lam * np.eye(d, dtype=np.float32)
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


def regression_report(name, w, b, x_train, y_train, x_test, y_test):
    y_pred_train = x_train @ w + b
    y_pred_test = x_test @ w + b
    return {
        "feature_set": name,
        "n_features": int(x_train.shape[1]),
        "train": {
            "r2": r2(y_train, y_pred_train),
            "pearson": pearson(y_train, y_pred_train),
            "rmse": float(np.sqrt(((y_train - y_pred_train) ** 2).mean())),
        },
        "test": {
            "r2": r2(y_test, y_pred_test),
            "pearson": pearson(y_test, y_pred_test),
            "rmse": float(np.sqrt(((y_test - y_pred_test) ** 2).mean())),
        },
    }


# ----------------------- experiment 1+5: proof length -------------------

CATS_TO_KEEP = [
    "Algebra", "AlgebraicGeometry", "AlgebraicTopology", "Analysis", "CategoryTheory",
    "Combinatorics", "Computability", "Data", "Dynamics", "FieldTheory", "Geometry",
    "GroupTheory", "LinearAlgebra", "Logic", "MeasureTheory", "ModelTheory",
    "NumberTheory", "Order", "Probability", "RepresentationTheory", "RingTheory",
    "SetTheory", "Tactic", "Topology",
]


def proof_length_regression(args, theorems, out_deg, in_deg, spectral_coords):
    rows = []
    for thm in theorems:
        if thm["tactic_count"] < 1:
            continue
        rows.append(thm)
    print(f"[proof-length] {len(rows)} theorems with traced tactics")

    rng = np.random.default_rng(0)
    cat_idx = {c: i for i, c in enumerate(CATS_TO_KEEP)}
    n = len(rows)

    # build feature matrices
    deg_feat = np.zeros((n, 2), dtype=np.float32)
    cat_feat = np.zeros((n, len(CATS_TO_KEEP)), dtype=np.float32)
    stmt_feat = np.zeros((n, 1), dtype=np.float32)
    spec_feat = np.zeros((n, 20), dtype=np.float32)
    has_spec = np.zeros(n, dtype=bool)
    for i, thm in enumerate(rows):
        deg_feat[i, 0] = math.log1p(out_deg.get(thm["name"], 0))
        deg_feat[i, 1] = math.log1p(in_deg.get(thm["name"], 0))
        ci = cat_idx.get(thm["category"], -1)
        if ci >= 0:
            cat_feat[i, ci] = 1.0
        stmt_feat[i, 0] = math.log1p(thm["statement_len"])
        if thm["name"] in spectral_coords:
            spec_feat[i] = spectral_coords[thm["name"]]
            has_spec[i] = True

    # target: log1p(#tactics) — long-tailed
    y = np.array([math.log1p(thm["tactic_count"]) for thm in rows], dtype=np.float32)

    # 80/20 hash split
    train = np.array([stable_bucket(thm["name"]) < 8 for thm in rows])
    test = ~train
    print(f"[proof-length] train={train.sum()}, test={test.sum()}")

    feature_sets = {
        "mean_only":              np.zeros((n, 0), dtype=np.float32),
        "degree":                 deg_feat,
        "area":                   cat_feat,
        "statement_len":          stmt_feat,
        "spectral":               spec_feat,
        "degree+area":            np.hstack([deg_feat, cat_feat]),
        "degree+area+statement":  np.hstack([deg_feat, cat_feat, stmt_feat]),
        "spectral+degree":        np.hstack([spec_feat, deg_feat]),
        "all":                    np.hstack([deg_feat, cat_feat, stmt_feat, spec_feat]),
    }
    reports = []
    for name, x in feature_sets.items():
        if x.shape[1] == 0:
            mean_train = float(y[train].mean())
            yp_test = np.full(test.sum(), mean_train, dtype=np.float32)
            yp_train = np.full(train.sum(), mean_train, dtype=np.float32)
            reports.append({
                "feature_set": name,
                "n_features": 0,
                "train": {"r2": 0.0, "pearson": 0.0, "rmse": float(np.sqrt(((y[train] - yp_train) ** 2).mean()))},
                "test": {"r2": 0.0, "pearson": 0.0, "rmse": float(np.sqrt(((y[test] - yp_test) ** 2).mean()))},
            })
            continue
        x_train, x_test = standardize(x[train], x[test])
        w, b = fit_ridge(x_train, y[train], lam=1.0)
        rep = regression_report(name, w, b, x_train, y[train], x_test, y[test])
        reports.append(rep)
        print(f"  {name:>30}  test R²={rep['test']['r2']:.3f}  ρ={rep['test']['pearson']:.3f}  rmse={rep['test']['rmse']:.3f}")

    # also: spectral-only on the subset of theorems that have spectral coords
    if has_spec.any():
        sub = has_spec
        x_sub = spec_feat[sub]
        y_sub = y[sub]
        train_sub = train[sub]
        test_sub = test[sub]
        x_train, x_test = standardize(x_sub[train_sub], x_sub[test_sub])
        w, b = fit_ridge(x_train, y_sub[train_sub], lam=1.0)
        rep = regression_report("spectral_only_on_spectral_subset", w, b,
                                x_train, y_sub[train_sub], x_test, y_sub[test_sub])
        rep["n_total"] = int(sub.sum())
        reports.append(rep)
        print(f"  spectral-only on subset: test R²={rep['test']['r2']:.3f}  ρ={rep['test']['pearson']:.3f}")

    out = {
        "n_theorems": int(n),
        "n_with_spectral": int(has_spec.sum()),
        "y_unit": "log1p(tactic_count)",
        "reports": reports,
    }
    out_path = OUT_DIR / "proof_length_regression.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return out


# ----------------------- experiment 2: dep matrix -----------------------

def dep_matrix(area_pairs):
    cats = sorted({c for c, _ in area_pairs} | {c for _, c in area_pairs})
    cats = [c for c in cats if c != "UNKNOWN"]
    cat_idx = {c: i for i, c in enumerate(cats)}
    n = len(cats)
    M = np.zeros((n, n), dtype=np.float64)
    for (s, t), c in area_pairs.items():
        if s in cat_idx and t in cat_idx:
            M[cat_idx[s], cat_idx[t]] += c
    row_sum = M.sum(axis=1, keepdims=True)
    R = np.divide(M, row_sum, out=np.zeros_like(M), where=row_sum > 0)
    out = {
        "categories": cats,
        "raw_counts": M.astype(int).tolist(),
        "row_normalized": R.tolist(),
        "row_totals": row_sum.flatten().astype(int).tolist(),
    }
    out_path = OUT_DIR / "cross_area_dep_matrix.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    write_heatmap(OUT_DIR / "cross_area_dep_matrix.svg",
                  R, cats, cats,
                  title="Cross-area dependency matrix (row-normalised: P(target_area | source_area))")
    return out


# ----------------------- experiment 3: tactic-by-area -------------------

def tactic_by_area(area_tactic):
    cats = sorted({c for c, _ in area_tactic if c != "UNKNOWN"})
    head_counts = Counter()
    for (_, h), c in area_tactic.items():
        if h:
            head_counts[h] += c
    top_heads = [h for h, _ in head_counts.most_common(15)]
    cat_idx = {c: i for i, c in enumerate(cats)}
    head_idx = {h: i for i, h in enumerate(top_heads)}
    M = np.zeros((len(cats), len(top_heads)), dtype=np.float64)
    for (c, h), v in area_tactic.items():
        if c in cat_idx and h in head_idx:
            M[cat_idx[c], head_idx[h]] += v
    row_sum = M.sum(axis=1, keepdims=True)
    R = np.divide(M, row_sum, out=np.zeros_like(M), where=row_sum > 0)
    out = {
        "categories": cats,
        "tactic_heads": top_heads,
        "raw_counts": M.astype(int).tolist(),
        "row_normalized": R.tolist(),
    }
    out_path = OUT_DIR / "tactic_mix_by_area.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    write_heatmap(OUT_DIR / "tactic_mix_by_area.svg",
                  R, cats, top_heads,
                  title="Tactic mix by source area (row-normalised)")
    return out


# ----------------------- shared SVG heatmap -----------------------------

def write_heatmap(path: Path, M, row_labels, col_labels, title="", cmap_max=None):
    rows, cols = M.shape
    cell = 28
    left = max(160, max(len(r) for r in row_labels) * 7)
    top = 60
    bottom = max(80, max(len(c) for c in col_labels) * 7)
    width = left + cols * cell + 30
    height = top + rows * cell + bottom
    cmax = cmap_max or float(M.max() or 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="22" font-family="Arial" font-size="15">{title}</text>',
    ]
    for j, c in enumerate(col_labels):
        x = left + j * cell + cell / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top - 6}" font-family="Arial" font-size="10" '
            f'text-anchor="end" transform="rotate(-55 {x:.1f},{top - 6})">{c}</text>'
        )
    for i, r in enumerate(row_labels):
        y = top + i * cell + cell / 2 + 4
        parts.append(f'<text x="{left - 6}" y="{y:.1f}" font-family="Arial" font-size="11" text-anchor="end">{r}</text>')
        for j in range(cols):
            v = float(M[i, j])
            t = max(0.0, min(1.0, v / cmax)) if cmax > 0 else 0.0
            # blue→white→red-ish; use simple white→navy ramp
            r_ = int(255 - 255 * t)
            g_ = int(255 - 200 * t)
            b_ = int(255 - 80 * t)
            x = left + j * cell
            yy = top + i * cell
            parts.append(f'<rect x="{x}" y="{yy}" width="{cell}" height="{cell}" fill="rgb({r_},{g_},{b_})" stroke="#eee"/>')
            if t > 0.05 and cell >= 22:
                parts.append(f'<text x="{x + cell/2:.1f}" y="{yy + cell/2 + 4:.1f}" font-family="Arial" font-size="9" text-anchor="middle" fill="{"white" if t > 0.5 else "black"}">{v:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    print(f"wrote {path}")


# ----------------------- main -------------------------------------------

def main() -> None:
    ensure_dir(OUT_DIR)
    print("loading edges + spectral + theorems...")
    out_deg, in_deg, area_pairs, area_tactic = load_edges_summary()
    spectral_coords, _, _ = load_spectral()
    theorems = list(load_theorems())
    print(f"  {len(theorems)} theorems with traces, "
          f"{len(out_deg)} sources, "
          f"{len(spectral_coords)} spectral nodes")

    proof_length_regression(None, theorems, out_deg, in_deg, spectral_coords)
    dep_matrix(area_pairs)
    tactic_by_area(area_tactic)


if __name__ == "__main__":
    main()
