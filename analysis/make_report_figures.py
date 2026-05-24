#!/usr/bin/env python3
"""Generate publication-style figures for the Mathlib geometry report.

Run after ``analysis/audit_report_numbers.py``.  Figures are written to
``results/report_figures`` as both PDF (for LaTeX) and SVG (for Markdown).
The numeric source for every panel is also recorded in ``figure_manifest.json``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUDIT = RESULTS / "audit/report_numbers.json"
OUT = RESULTS / "report_figures"
LD_DIR = ROOT / "Old/data/leandojo/leandojo_benchmark_4/random"
EDGES_PATH = RESULTS / "paths/dependency_edges.jsonl"


def load() -> dict:
    with AUDIT.open() as f:
        return json.load(f)


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.axisbelow": True,
        }
    )


def save(fig, name: str, manifest: dict, data: dict) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    manifest[name] = data


def split_goals(state_after: str) -> list[str]:
    if state_after in ("no goals", ""):
        return ["no goals"]
    if state_after.count("⊢") <= 1:
        return [state_after]
    parts = re.split(r"\n\n(?=case )", state_after)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]
    parts = re.split(r"\n\n(?=\S)", state_after)
    return [p.strip() for p in parts if p.strip()] if len(parts) > 1 else [state_after]


def proof_tree_metric(tactics: list[dict]) -> dict | None:
    if not tactics:
        return None
    active = [0]
    layer_counts = Counter({0: 1})
    leaf_depths = []
    branch_counts = []
    steps = 0
    for tt in tactics:
        if not active:
            break
        depth = active.pop(0)
        goals = split_goals(tt.get("state_after", ""))
        k = 0 if goals == ["no goals"] else len(goals)
        out_degree = max(k, 1)
        branch_counts.append(out_degree)
        child_depth = depth + 1
        steps += out_degree
        if k == 0:
            layer_counts[child_depth] += 1
            leaf_depths.append(child_depth)
        else:
            layer_counts[child_depth] += k
            active = [child_depth] * k + active
    if not branch_counts:
        return None
    return {
        "depth": max(layer_counts),
        "width": max(layer_counts.values()),
        "n_tactic_steps": steps,
        "n_leaves": max(len(leaf_depths), 1),
        "max_branching": max(branch_counts),
    }


def load_tree_metrics() -> list[dict]:
    metrics = []
    for split in ("train", "val", "test"):
        path = LD_DIR / f"{split}.json"
        if not path.exists():
            continue
        for thm in json.load(path.open()):
            m = proof_tree_metric(thm.get("traced_tactics") or [])
            if m:
                metrics.append(m)
    return metrics


def degree_tail_figure(audit: dict, manifest: dict) -> None:
    deg = Counter()
    with EDGES_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            deg[e["source"]] += 1
            deg[e["target"]] += 1
    vals = np.array([v for v in deg.values() if v > 0], dtype=int)
    xs = np.unique(vals)
    counts = np.array([(vals >= x).mean() for x in xs])
    fit = audit["old_results"]["power_law_fit"]
    xmin = float(fit["xmin"])
    alpha = float(fit["alpha"])
    y0 = counts[xs >= xmin][0]
    fit_x = xs[xs >= xmin]
    fit_y = y0 * (fit_x / xmin) ** (-(alpha - 1.0))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.loglog(xs, counts, color="#4c72b0", lw=1.8, label="empirical CCDF")
    ax.loglog(fit_x, fit_y, color="#c44e52", lw=1.4, ls="--", label=fr"tail slope $\alpha={alpha:.2f}$")
    ax.axvline(xmin, color="#555555", lw=1.0, ls=":", label=fr"$x_{{min}}={xmin:.0f}$")
    ax.set_xlabel("total dependency degree")
    ax.set_ylabel(r"$P(\mathrm{degree}\geq x)$")
    ax.set_title("Heavy-tailed dependency-degree distribution")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "degree_tail_ccdf", manifest, {
        "n_positive_degree_nodes": int(len(vals)),
        "max_degree": int(vals.max()),
        "median_degree": float(np.median(vals)),
        "p90_degree": float(np.quantile(vals, 0.90)),
        "p99_degree": float(np.quantile(vals, 0.99)),
        "alpha": alpha,
        "xmin": xmin,
    })


def proof_tree_spread_figure(audit: dict, manifest: dict) -> None:
    metrics = load_tree_metrics()
    depths = np.array([m["depth"] for m in metrics], dtype=int)
    widths = np.array([m["width"] for m in metrics], dtype=int)
    max_depth = int(depths.max())
    bins = np.arange(0, max_depth + 2) - 0.5
    xs = np.arange(1, max_depth + 1)
    ccdf = np.array([(depths >= x).mean() for x in xs])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    axes[0].hist(depths, bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.35)
    axes[0].axvline(np.median(depths), color="#c44e52", ls="--", lw=1.2, label=f"median={np.median(depths):.0f}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("proof-tree depth")
    axes[0].set_ylabel("proofs, log scale")
    axes[0].set_title("Depth distribution")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(xs, ccdf, color="#55a868", lw=1.8)
    axes[1].axvline(depths.mean(), color="#dd8452", ls=":", lw=1.2, label=f"mean={depths.mean():.2f}")
    axes[1].axvline(max_depth, color="#555555", ls="--", lw=1.0, label=f"max={max_depth}")
    axes[1].set_xlabel("depth threshold")
    axes[1].set_ylabel(r"$P(\mathrm{depth}\geq x)$")
    axes[1].set_title("Depth tail")
    axes[1].legend(frameon=False, fontsize=8)

    save(fig, "proof_tree_depth_spread", manifest, {
        "n_proofs": int(len(depths)),
        "depth_mean": float(depths.mean()),
        "depth_median": float(np.median(depths)),
        "depth_max": int(max_depth),
        "depth_p90": float(np.quantile(depths, 0.90)),
        "depth_p99": float(np.quantile(depths, 0.99)),
        "width_mean": float(widths.mean()),
        "width_median": float(np.median(widths)),
        "width_max": int(widths.max()),
    })


def bar(ax, labels, values, colors, ylabel=None, ylim=None):
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.72)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for i, v in enumerate(values):
        ax.text(i, v + (ylim[1] - ylim[0]) * 0.02 if ylim else v * 1.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)


def proof_length_figure(audit: dict, manifest: dict) -> None:
    p = audit["proof_length_prediction"]
    labels = ["spectral", "node2vec", "degree", "node2vec+degree", "all graph geometry"]
    values = [
        p["spectral"]["r2"],
        p["node2vec"]["r2"],
        p["degree"]["r2"],
        p["node2vec_degree"]["r2"],
        p["all_graph_geometry"]["r2"],
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bar(ax, labels, values, ["#c44e52", "#55a868", "#4c72b0", "#4c72b0", "#8172b3"], "test $R^2$", (0, 0.62))
    ax.set_title("Predicting proof length from graph geometry")
    save(fig, "proof_length_r2", manifest, dict(zip(labels, values)))


def category_figure(audit: dict, manifest: dict) -> None:
    c = audit["category_prediction"]
    labels = ["majority", "degree MLP", "best spectral+degree", "namespace prior"]
    values = [
        c["majority"]["accuracy"],
        0.17180205415499533,
        c["best_accuracy"]["accuracy"],
        c["namespace_prior"]["accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bar(ax, labels, values, ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"], "accuracy", (0, 0.82))
    ax.set_title("Top-level Mathlib area prediction")
    save(fig, "category_prediction_accuracy", manifest, dict(zip(labels, values)))


def path_figure(audit: dict, manifest: dict) -> None:
    p = audit["pathfinding"]
    labels = ["random proof-DAG pairs", "cited dep. edges\nstate graph"]
    values = [
        p["proof_dag_random_pairs"]["found_fraction"],
        p["state_graph_dep_edge_cross_validation"]["found_fraction"],
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    bar(ax, labels, values, ["#8c8c8c", "#c44e52"], "reachable fraction", (0, 0.03))
    ax.set_title("Strict goal-equality pathfinding is sparse")
    save(fig, "pathfinding_reachability", manifest, dict(zip(labels, values)))


def proof_shape_figure(audit: dict, manifest: dict) -> None:
    shapes = audit["old_results"]["search_tree"]["shape_distribution"]
    labels = list(shapes)
    values = [shapes[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(np.arange(len(labels)), values, color="#4c72b0", width=0.72)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("proofs, log scale")
    ax.set_title("Most traced Mathlib proofs are linear")
    for i, v in enumerate(values):
        ax.text(i, v * 1.08, f"{v:,}", ha="center", va="bottom", fontsize=8)
    save(fig, "proof_shape_distribution", manifest, shapes)


def ricci_figure(audit: dict, manifest: dict) -> None:
    raw = json.load((RESULTS / "graph_geometry/ollivier_ricci.json").open())
    hist = np.array(raw["histogram"], dtype=float)
    edges = np.array(raw["histogram_edges"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(centers, hist, width=widths * 0.92, color="#8172b3")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("sampled Ollivier-Ricci curvature $\\kappa$")
    ax.set_ylabel("edge count")
    ax.set_title("Sampled dependency edges are mostly negatively curved")
    save(fig, "ollivier_ricci_histogram", manifest, {"histogram": raw["histogram"], "histogram_edges": raw["histogram_edges"]})


def old_comparison_figure(audit: dict, manifest: dict) -> None:
    comp = audit["old_results"]["comparative"]
    keys = ["Mathlib", "LEAN-GitHub"]
    labels = ["Mathlib", "Lean-GitHub"]
    mean_lengths = [comp[k]["proof_length_mean"] for k in keys]
    unique_tactics = [comp[k]["n_unique_tactics"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    axes[0].bar(labels, mean_lengths, color=["#4c72b0", "#dd8452"])
    axes[0].set_title("Mean proof length")
    axes[0].set_ylabel("tactics")
    axes[1].bar(labels, unique_tactics, color=["#4c72b0", "#dd8452"])
    axes[1].set_title("Unique tactic heads")
    axes[1].set_ylabel("count")
    for ax in axes:
        for i, patch in enumerate(ax.patches):
            v = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2, v * 1.03, f"{v:.1f}" if v < 100 else f"{int(v):,}", ha="center", fontsize=8)
    save(fig, "mathlib_vs_lean_github", manifest, {"mean_proof_length": dict(zip(labels, mean_lengths)), "unique_tactics": dict(zip(labels, unique_tactics))})


def main() -> None:
    setup()
    audit = load()
    manifest = {}
    degree_tail_figure(audit, manifest)
    proof_tree_spread_figure(audit, manifest)
    proof_length_figure(audit, manifest)
    category_figure(audit, manifest)
    path_figure(audit, manifest)
    proof_shape_figure(audit, manifest)
    ricci_figure(audit, manifest)
    old_comparison_figure(audit, manifest)
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
