# Mathlib Workspaces Manager Agents Document (agents.md)

## Current Workspace Snapshot
The current analysis codebase at `/Users/egecakar/Documents/Research/MWM/analysis/` operates on LeanDojo and LeanTree benchmark datasets to extract structural properties of mathematical proofs found in Lean 4/Mathlib. 

### Key Modules:
- `build_hypergraph.py`: Parses JSON(L) proof data from LeanDojo and LeanTree to build a "proof hypergraph" where nodes are proof states and hyperedges are tactics that transition between them, enabling an understanding of multi-goal resolutions.
- `proof_search_trees.py`: Analyzes the topological shapes of "tactic proof trees", extracting metrics like depth, width, size, branching factor, leaf depth, balancedness, and clustering proofs into shapes (LINEAR, BUSHY, WIDE_SHALLOW, DEEP_NARROW, BALANCED). 
- `leandojo_deep_analysis.py`: Performs fine-grained analysis of tactic-mode proofs such as tactic bigrams/trigrams, co-occurrence matrices, and module-level heatmap specialties.

## Objectives
The core objectives requested are:
1. **Finding New Abstractions**: Analyze unrolled proofs to identify potential abstractions. Detect patterns where sequences of tactics (subgraphs/paths) can be condensed into broader abstractions.
2. **Proof Optimality Measurement**: Formulate an "optimality measure" based on proof search criteria (e.g. ideal branchings, minimal depth, optimal top-k tactic search paths). We must check the optimality of current Mathlib against this metric, identifying proofs that are too wide/deep and attempting to "optimize" them (or outline the theoretical path to doing so).

## Action Plan
1. Translate "finding abstractions" into graph-theoretic motifs. Which subgraphs occur frequently but are completely unrolled in Mathlib? We can use community detection or frequent subgraph mining on the hypergraph.
2. Formulate "Proof Optimality". A proof search algorithm (like Best-First Search or MCTS) prefers paths with minimal branching (low perplexity) or minimal depth. Optimality of a Mathlib theorem can be defined as the ratio of its actual depth/branching compared to the shortest theoretically bound paths in the hypergraph, or evaluated using heuristic transition probabilities.
3. Use the above formulations to script out new analyses that rank sub-optimal theorems and propose refactoring / abstractions.

This file serves as my baseline context and will be updated as we evolve the methods.
