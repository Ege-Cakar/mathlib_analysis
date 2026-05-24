#!/usr/bin/env python3
"""Audit the numeric claims used in the Mathlib geometry report.

The script reads only compact JSON/JSONL artifacts, writes a machine-readable
audit to ``results/audit/report_numbers.json``, and writes a short Markdown
summary to ``results/audit/report_numbers.md``.  Large or APFS-dataless legacy
files are listed as unavailable rather than used silently.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OLD = ROOT / "Old"
OUT = RESULTS / "audit"


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def best_model(models: dict, metric: str) -> dict:
    rows = []
    for feature_set, by_model in models.items():
        for model, vals in by_model.items():
            rows.append((vals[metric], feature_set, model, vals))
    value, feature_set, model, vals = max(rows, key=lambda x: x[0])
    return {"feature_set": feature_set, "model": model, **vals}


def by_feature(reports: list[dict]) -> dict:
    return {r["feature_set"]: r for r in reports}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    dep = read_json(RESULTS / "paths/dependency_stats.json")
    state = read_json(RESULTS / "state_graph/state_graph_stats.json")
    clf = read_json(RESULTS / "spectral/classifier_sweep.json")
    link = read_json(RESULTS / "spectral/link_prediction.json")
    tactic = read_json(RESULTS / "spectral/tactic_head_prediction.json")
    proof = read_json(RESULTS / "predictive/proof_length_regression.json")
    joint = read_json(RESULTS / "graph_geometry/joint_proof_length.json")
    ricci = read_json(RESULTS / "graph_geometry/ollivier_ricci.json")
    paths = read_json(RESULTS / "paths/path_sweep.json")
    sg_goal = read_json(RESULTS / "state_graph/state_path_sweep_goal.json")
    kimina_manifest = read_json(RESULTS / "prover_eval/post_kimina_manifest_stats.json")
    kimina_rows = read_jsonl(RESULTS / "prover_eval/kimina_post_2025_08_14.jsonl")
    reprover_rows = read_jsonl(RESULTS / "prover_eval/reprover_status.jsonl")

    old = {
        "theorem_graph": read_json(OLD / "data/theorem_graph_stats.json"),
        "search_tree": read_json(OLD / "data/search_tree_stats.json"),
        "degree_depth": read_json(OLD / "data/degree_depth_stats.json"),
        "difficulty": read_json(OLD / "data/difficulty_stats.json"),
        "comparative": read_json(OLD / "data/comparative_stats.json"),
        "community": read_json(OLD / "data/community_stats.json"),
        "spectral": read_json(OLD / "data/spectral_stats.json"),
        "hypergraph": read_json(OLD / "data/hypergraph_stats.json"),
        "bpe": read_json(OLD / "data/bpe_macro_abstractions.json"),
        "shortcuts": read_json(OLD / "graph_search_output/mathlib_shortcuts.json"),
    }

    proof_reports = by_feature(proof["reports"])
    joint_reports = by_feature(joint["reports"])
    sg_cv = sg_goal["dep_graph_cross_validation"]
    kimina_verified = Counter(str(r.get("verified")) for r in kimina_rows)

    audit = {
        "current_dependency_graph": dep,
        "current_state_graph": state,
        "category_prediction": {
            "n_train": clf["n_train"],
            "n_test": clf["n_test"],
            "majority": clf["baselines"]["majority"],
            "namespace_prior": clf["baselines"]["namespace_prior"],
            "best_accuracy": best_model(clf["models"], "accuracy"),
            "best_macro_f1": best_model(clf["models"], "macro_f1"),
        },
        "link_prediction": {
            "degree_only": link["degree_only"],
            "spectral_full": link["spectral_full"],
        },
        "tactic_head_prediction": {
            "majority": tactic["majority_baseline"],
            "degree_only": tactic["degree_only"],
            "spectral_concat": tactic["spectral_concat"],
        },
        "proof_length_prediction": {
            "n_theorems": proof["n_theorems"],
            "n_with_spectral": proof["n_with_spectral"],
            "degree": proof_reports["degree"]["test"],
            "spectral": proof_reports["spectral"]["test"],
            "degree_area_statement": proof_reports["degree+area+statement"]["test"],
            "node2vec": joint_reports["node2vec"]["test"],
            "node2vec_degree": joint_reports["node2vec+degree"]["test"],
            "all_graph_geometry": joint_reports["all_combined"]["test"],
        },
        "curvature": {
            k: ricci[k]
            for k in (
                "n_evaluated",
                "mean_kappa",
                "median_kappa",
                "min_kappa",
                "max_kappa",
                "fraction_negative",
                "fraction_below_threshold",
            )
        },
        "pathfinding": {
            "proof_dag_random_pairs": paths["pair_reachability"],
            "state_graph_dep_edge_cross_validation": {
                k: sg_cv[k]
                for k in ("pairs_tried", "found", "found_fraction", "depth_histogram")
            },
            "state_graph_by_tactic_head": sg_cv["rate_by_dep_tactic_head"],
        },
        "prover_eval": {
            "post_kimina_manifest": kimina_manifest,
            "kimina_generation_rows": len(kimina_rows),
            "kimina_unique_theorems": len({r.get("full_name") for r in kimina_rows}),
            "kimina_samples_per_theorem": sorted({r.get("sample_id") for r in kimina_rows}),
            "kimina_verified_counts": dict(kimina_verified),
            "reprover_status": reprover_rows,
        },
        "old_results": {
            "theorem_graph": old["theorem_graph"],
            "search_tree": old["search_tree"],
            "power_law_fit": old["degree_depth"]["power_law_fit"],
            "difficulty": {
                "n_matched": old["difficulty"]["n_matched"],
                "top_pearson": old["difficulty"]["top_pearson"][:6],
                "top_spearman": old["difficulty"]["top_spearman"][:6],
                "linear_r2": old["difficulty"]["linear_r2"],
                "logistic_accuracy": old["difficulty"]["logistic_accuracy"],
                "median_proof_length": old["difficulty"]["median_proof_length"],
            },
            "comparative": old["comparative"],
            "community": {
                "n_communities": old["community"]["n_communities"],
                "modularity": old["community"]["modularity"],
                "top_communities": old["community"]["top_communities"][:6],
            },
            "spectral": old["spectral"],
            "legacy_hypergraph": old["hypergraph"],
            "top_bpe_macros": old["bpe"][:10],
            "shortcut_summary": {
                "n": len(old["shortcuts"]),
                "max_steps_saved": max(r["steps_saved"] for r in old["shortcuts"]),
                "top_examples": old["shortcuts"][:5],
            },
        },
        "old_files_not_used_in_headline_claims": [
            "Old/data/optimality_ranking.json",
            "Old/data/abstraction_candidates.json",
            "Old/data/frequent_subgoals.json",
            "Old/data/frequent_transitions.json",
            "Old/data/identical_state_tactic_clusters.json",
        ],
    }

    (OUT / "report_numbers.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    md = [
        "# Report Number Audit",
        "",
        f"- Dependency graph: {dep['nodes']:,} nodes, {dep['unique_dependency_edges']:,} unique edges; expected-edge check = {dep['matches_expected_unique_dependency_edges']}.",
        f"- State graph: {state['unique_states']:,} states, {state['unique_hyperedges']:,} hyperedges, {state['multi_output_hyperedges']:,} multi-output edges.",
        f"- Best category accuracy: {audit['category_prediction']['best_accuracy']['accuracy']:.3f}; namespace prior: {clf['baselines']['namespace_prior']['accuracy']:.3f}.",
        f"- Proof-length R2: spectral {proof_reports['spectral']['test']['r2']:.3f}, degree {proof_reports['degree']['test']['r2']:.3f}, node2vec+degree {joint_reports['node2vec+degree']['test']['r2']:.3f}, all graph geometry {joint_reports['all_combined']['test']['r2']:.3f}.",
        f"- Ollivier-Ricci: n={ricci['n_evaluated']:,}, mean kappa={ricci['mean_kappa']:.3f}, negative fraction={ricci['fraction_negative']:.3f}.",
        f"- Kimina generations: {len(kimina_rows)} rows over {audit['prover_eval']['kimina_unique_theorems']} theorems; verification counts {dict(kimina_verified)}.",
        f"- Old community result: {old['community']['n_communities']} communities, modularity {old['community']['modularity']:.3f}.",
        f"- Old proof-shape counts: {old['search_tree']['shape_distribution']}.",
        "",
        "The files listed under `old_files_not_used_in_headline_claims` were not used for headline claims because local reads are APFS-dataless/timeout-prone in this checkout.",
    ]
    (OUT / "report_numbers.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
