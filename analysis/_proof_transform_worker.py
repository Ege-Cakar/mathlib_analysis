#!/usr/bin/env python3
"""Enrichment worker: replays LeanDojo traces, unrolling automation steps.

Runs under the Fall_2025/leandojo venv (Python 3.13, `lean_dojo` available).
Reads a payload JSONL file (`--payload`), where each line is

    {"split", "full_name", "file_path", "traced_tactics": [...]}

and appends one enriched-trace JSONL record per theorem to `--out`,
guarded by an fcntl LOCK_EX so multiple workers can share the file.

Edge kinds emitted:
    rewrite_step    — one edge per single lemma in a `rw`/`rwa`/`erw`/
                       `nth_rw`/`rw_mod_cast` chain.  Real intermediate
                       Lean states between rewrites.
    simp_step       — one edge per single lemma applied via
                       `simp only [lemma]` during stepwise replay of an
                       explicit-args simp call.  Real intermediate states.
    simp_bundle     — fallback: a single edge labelled with the
                       (sorted) lemma multiset.  Emitted when stepwise
                       replay diverges from the trace's `state_after`, or
                       when the original tactic was bare (no explicit
                       args).  Bare automation cannot be unrolled in a
                       single Dojo session because `simp?` already
                       advances past the would-be intermediate states.
    decision_opaque — `omega`, `decide`, `ring`, `linarith`, ...  One
                       edge per call; lemma list is empty.  These tactics
                       do not apply a named lemma sequence, so there is
                       nothing to unroll.
    term            — `exact` / `refine` / `apply` / `convert`.  One edge
                       per call; premises come from LeanDojo's
                       `annotated_premises`.
    structural      — `intro` / `have` / `obtain` / `cases` /
                       `constructor` / `ext` / etc.  One edge per call.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# `lean_dojo` import is slow on a cold OS file cache (~30 s + warm-up of
# PyGithub's many submodules). The worker pays this cost once at startup,
# then serves many theorems.
from lean_dojo import (
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    ProofFinished,
    LeanError,
    DojoCrashError,
    DojoInitError,
    DojoTacticTimeoutError,
)

ANCHOR_RE = re.compile(r"<a>(.*?)</a>")
TRY_THIS_RE = re.compile(r"Try this:\s*([^\n]+(?:\n\s+[^\n]+)*)", re.DOTALL)
NO_GOALS = "no goals"

UNROLLABLE_AUTOMATION = {
    "simp", "simpa", "simp_all", "simp_rw", "dsimp", "field_simp", "aesop",
}
DECISION_PROCEDURES = {
    "omega", "decide", "native_decide",
    "ring", "ring_nf", "noncomm_ring", "abel", "abel_nf", "group",
    "linarith", "nlinarith", "polyrith",
    "norm_num", "norm_cast", "push_cast", "exact_mod_cast",
    "positivity", "gcongr", "tauto", "rfl", "trivial", "assumption",
}
REWRITE_HEADS = {"rw", "rwa", "erw", "nth_rw", "rw_mod_cast"}
TERM_HEADS = {"exact", "refine", "apply", "convert"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tactic_head(tactic: str) -> str:
    t = (tactic or "").strip()
    return t.split(None, 1)[0] if t else ""


def has_explicit_brackets(tactic: str) -> bool:
    """Heuristic: `simp [a, b]`, `simp only [a, b]`, `rw [a]` all contain
    a `[`; bare `simp` / `omega` / `aesop` do not.
    """
    return "[" in (tactic or "")


def annotated_premises_ordered(tt: dict) -> list[dict]:
    """Return premises ordered by anchor position in the annotated tactic
    text. For `rw [a, b]` and `simp only [a, b]` this is the order Lean
    applies them. Each item has `name` (resolved full_name, used in the
    enriched record), `literal` (the exact text the user wrote, used for
    replay because it survives namespace shadowing), `direction`
    (`fwd`/`rev`/`?`), and `path` (definition file).
    """
    ann = tt.get("annotated_tactic")
    if not (isinstance(ann, list) and len(ann) > 1 and isinstance(ann[1], list)):
        return []
    text = ann[0] or ""
    tags = list(ANCHOR_RE.finditer(text))
    out: list[dict] = []
    seen: set[str] = set()
    for i, p in enumerate(ann[1]):
        name = p.get("full_name")
        if not name or name in seen:
            continue
        seen.add(name)
        direction = "?"
        literal = name
        if i < len(tags):
            literal = tags[i].group(1) or name
            before = text[max(0, tags[i].start() - 8) : tags[i].start()]
            direction = "rev" if ("←" in before or "<-" in before) else "fwd"
        out.append({
            "name": name,
            "literal": literal,
            "direction": direction,
            "path": p.get("def_path", ""),
        })
    return out


def parse_try_this_lemmas(message: Optional[str]) -> Optional[list[str]]:
    """Parse `Try this: simp only [a, b, c]` (or `aesop?`'s suggestion).
    Returns the lemma-name list in suggested order, or None if no
    bracketed list was found.
    """
    if not message:
        return None
    m = TRY_THIS_RE.search(message)
    if not m:
        return None
    suggestion = m.group(1).strip()
    depth = 0
    start = -1
    for idx, ch in enumerate(suggestion):
        if ch == "[":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                return _split_top_commas(suggestion[start + 1 : idx])
    return None


def _split_top_commas(s: str) -> list[str]:
    out, cur, depth = [], "", 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            t = cur.strip()
            if t:
                out.append(t)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def state_pp(state) -> str:
    """Lean printed proof state, or sentinel `no goals`."""
    if isinstance(state, ProofFinished):
        return NO_GOALS
    if isinstance(state, TacticState):
        return state.pp
    return ""


def states_match(a: str, b: str) -> bool:
    a = (a or "").strip()
    b = (b or "").strip()
    if a == b:
        return True
    if a == NO_GOALS and b == NO_GOALS:
        return True
    return " ".join(a.split()) == " ".join(b.split())


def build_simp_only_replay(orig_tactic: str, lemma: str, direction: str) -> str:
    """`simp only [<arrow><lemma>] <suffix>` preserving any trailing
    `at h ⊢` clause of the original tactic.
    """
    suffix = ""
    m = re.search(r"\sat(\s|$)", orig_tactic or "")
    if m:
        suffix = orig_tactic[m.start():]
    arrow = "← " if direction == "rev" else ""
    return f"simp only [{arrow}{lemma}]{suffix}"


def build_rw_replay(orig_tactic: str, head: str, lemma: str, direction: str) -> str:
    suffix = ""
    m = re.search(r"\sat(\s|$)", orig_tactic or "")
    if m:
        suffix = orig_tactic[m.start():]
    arrow = "← " if direction == "rev" else ""
    return f"{head} [{arrow}{lemma}]{suffix}"


def run_tac_safely(dojo, state, tactic: str):
    """Wrapper around Dojo.run_tac that catches structural failures
    (timeout, crash, unexpected exception). On structural failure it
    returns a marker string starting with `__`; otherwise it returns the
    `TacticResult` lean_dojo produced (`TacticState`, `ProofFinished`,
    `LeanError`, or `ProofGivenUp`).
    """
    try:
        return dojo.run_tac(state, tactic)
    except DojoTacticTimeoutError as e:
        return f"__TIMEOUT__: {e}"
    except DojoCrashError as e:
        return f"__CRASH__: {e}"
    except Exception as e:  # noqa
        return f"__EXC__: {type(e).__name__}: {e}"


def stepwise_replay(
    dojo,
    state,
    lemma_list: list[dict],
    orig_tactic: str,
    head: str,
    expected_after: str,
    kind_for_step: str,
):
    """Apply each lemma in `lemma_list` one at a time, starting from
    `state` (a Dojo TacticState).  Returns (chain, ok, why, last_state):

      chain       — list of {in_state, out_states, tactic_text, lemma,
                     direction, kind} records, one per replay step.
      ok          — True iff the pp of the final replayed state matches
                     `expected_after` (up to whitespace normalisation).
      why         — short reason string when ok is False.
      last_state  — the real Dojo TacticState after the last successful
                     replay step (or `state` if the list was empty).

    When ok is False the caller should ignore `chain` and emit a single
    fallback edge instead, then re-run the original tactic to advance
    the Dojo state to the trace's `state_after`.
    """
    chain = []
    cur = state
    last_state = state
    for lemma_rec in lemma_list:
        # Use the original source text for replay (`literal`) rather than
        # the resolved full_name; this matches what the user wrote and
        # avoids namespace-shadowing surprises.
        lemma = lemma_rec.get("literal") or lemma_rec["name"]
        direction = lemma_rec.get("direction", "?")
        if head in REWRITE_HEADS:
            replay = build_rw_replay(orig_tactic, head, lemma, direction)
        else:
            replay = build_simp_only_replay(orig_tactic, lemma, direction)
        nxt = run_tac_safely(dojo, cur, replay)
        if isinstance(nxt, str):
            return chain, False, f"replay_error:{nxt[:120]}", last_state
        if isinstance(nxt, LeanError):
            return chain, False, f"lean_error:{nxt.error[:160]}", last_state
        chain.append({
            "in_state": state_pp(cur),
            "out_states": [state_pp(nxt)],
            "tactic_text": replay,
            "lemma": lemma,
            "direction": direction,
            "kind": kind_for_step,
        })
        cur = nxt
        last_state = cur
        if isinstance(cur, ProofFinished):
            break

    ok = states_match(state_pp(cur), expected_after)
    return chain, ok, ("" if ok else "drift"), last_state


def split_goals(state: str) -> list[str]:
    """Split a multi-goal printed Lean state into individual single-goal
    substrings. Mirrors state_hypergraph.split_lean_state.
    """
    s = (state or "").strip()
    if not s or s == NO_GOALS:
        return [NO_GOALS]
    m = re.match(r"^\d+\s+goals?\n(.*)", s, re.DOTALL)
    if m:
        s = m.group(1)
    if s.startswith("case ") or "\n\ncase " in s:
        parts = re.split(r"\n\n(?=case\s)", s)
        return [p.strip() for p in parts if p.strip()]
    return [s]


def process_theorem(repo: LeanGitRepo, rec: dict, dojo_timeout: int, fallback_on_drift: bool) -> dict:
    name = rec["full_name"]
    file_path = rec["file_path"]
    split = rec.get("split", "")
    traced = rec.get("traced_tactics", [])

    result = {
        "name": name,
        "file_path": file_path,
        "split": split,
        "n_original_tactics": len(traced),
        "n_unrolled_edges": 0,
        "n_fallback": 0,
        "edges": [],
        "error": None,
        "dojo_open_s": None,
        "dojo_total_s": None,
    }
    if not traced:
        return result

    edges = result["edges"]

    try:
        thm = Theorem(repo, file_path, name)
        t_open0 = time.time()
        with Dojo(thm, timeout=dojo_timeout) as (dojo, init_state):
            result["dojo_open_s"] = time.time() - t_open0
            state = init_state

            for i, tt in enumerate(traced):
                tactic = tt.get("tactic", "") or ""
                head = tactic_head(tactic)
                trace_before = tt.get("state_before", "") or ""
                trace_after = tt.get("state_after", "") or ""
                prems = annotated_premises_ordered(tt)

                if isinstance(state, ProofFinished):
                    break
                if isinstance(state, LeanError):
                    result["error"] = f"step {i}: state was LeanError ({state.error[:160]})"
                    break

                in_pp = state_pp(state)

                # State-drift guard: the Dojo replay state should match the
                # trace's `state_before` for this step. They can disagree when
                # the original theorem uses `calc`, `match`, `where`, term-mode
                # proof body, or any other construct that LeanDojo extracts
                # only the inner `by ...` blocks of. We can't faithfully
                # replay those, so we abort with a clear error.
                if not states_match(in_pp, trace_before):
                    result["error"] = (
                        f"step {i}: state drift between Dojo replay and trace; "
                        "likely a calc / term-mode proof that LeanDojo split "
                        "into inner by-blocks (not a flat tactic sequence)"
                    )
                    break

                # ---- 1. Decision procedures: opaque edge -------------------
                if head in DECISION_PROCEDURES:
                    edges.append({
                        "kind": "decision_opaque",
                        "head": head,
                        "premises": prems,
                        "lemmas": [],
                        "directions": [],
                        "tactic_text": tactic,
                        "in_state": in_pp,
                        "out_states": [trace_after],
                        "source_tactic_index": i,
                        "sub_index": 0,
                        "is_fallback": False,
                        "is_branching": False,
                    })
                    nxt = run_tac_safely(dojo, state, tactic)
                    if isinstance(nxt, str):
                        result["error"] = f"step {i}: decision {head} failed: {nxt}"
                        break
                    state = nxt
                    continue

                # ---- 2. Unrollable automation (simp family + aesop) --------
                if head in UNROLLABLE_AUTOMATION:
                    if has_explicit_brackets(tactic) and prems:
                        chain, ok, why, last_state = stepwise_replay(
                            dojo, state, prems, tactic, head,
                            expected_after=trace_after, kind_for_step="simp_step",
                        )
                        if ok:
                            for j, c in enumerate(chain):
                                edges.append({
                                    "kind": "simp_step",
                                    "head": head,
                                    "premises": [{"name": c["lemma"], "direction": c["direction"]}],
                                    "lemmas": [c["lemma"]],
                                    "directions": [c["direction"]],
                                    "tactic_text": c["tactic_text"],
                                    "in_state": c["in_state"],
                                    "out_states": c["out_states"],
                                    "source_tactic_index": i,
                                    "sub_index": j,
                                    "is_fallback": False,
                                    "is_branching": False,
                                })
                            state = last_state
                            result["n_unrolled_edges"] += len(chain)
                            continue
                        elif fallback_on_drift:
                            edges.append({
                                "kind": "simp_bundle",
                                "head": head,
                                "premises": prems,
                                "lemmas": sorted(p["name"] for p in prems),
                                "directions": [p["direction"] for p in prems],
                                "tactic_text": tactic,
                                "in_state": in_pp,
                                "out_states": [trace_after],
                                "source_tactic_index": i,
                                "sub_index": 0,
                                "is_fallback": True,
                                "is_branching": False,
                                "fallback_reason": why,
                            })
                            result["n_fallback"] += 1
                            nxt = run_tac_safely(dojo, state, tactic)
                            if isinstance(nxt, str):
                                result["error"] = f"step {i}: simp fallback advance failed: {nxt}"
                                break
                            state = nxt
                            continue
                        else:
                            result["error"] = f"step {i}: stepwise replay drifted ({why})"
                            break
                    else:
                        # Bare automation: capture the lemma list via `?`
                        # suggestion, emit one rich-label simp_bundle edge.
                        sug_tactic = head + "?" + tactic[len(head):]
                        sug = run_tac_safely(dojo, state, sug_tactic)
                        lemmas: list[str] = []
                        used_orig = False
                        if isinstance(sug, (TacticState, ProofFinished)):
                            msg = getattr(sug, "message", None)
                            ls = parse_try_this_lemmas(msg)
                            if ls is not None:
                                lemmas = ls
                        else:
                            # `?` variant failed; fall back to the bare tactic
                            sug = run_tac_safely(dojo, state, tactic)
                            used_orig = True
                            if isinstance(sug, str):
                                result["error"] = f"step {i}: bare {head} failed: {sug}"
                                break
                            if isinstance(sug, LeanError):
                                result["error"] = f"step {i}: bare {head} LeanError: {sug.error[:180]}"
                                break
                        edges.append({
                            "kind": "simp_bundle",
                            "head": head,
                            "premises": prems,
                            "lemmas": sorted(lemmas),
                            "directions": [],
                            "tactic_text": tactic,
                            "in_state": in_pp,
                            "out_states": [state_pp(sug)],
                            "source_tactic_index": i,
                            "sub_index": 0,
                            "is_fallback": True,
                            "is_branching": False,
                            "fallback_reason": ("suggest_variant_failed_used_orig"
                                                if used_orig else
                                                "bare_automation_intermediates_unavailable"),
                        })
                        result["n_fallback"] += 1
                        state = sug
                        continue

                # ---- 3. Specified rewrite chains --------------------------
                if head in REWRITE_HEADS and prems:
                    chain, ok, why, last_state = stepwise_replay(
                        dojo, state, prems, tactic, head,
                        expected_after=trace_after, kind_for_step="rewrite_step",
                    )
                    if ok:
                        for j, c in enumerate(chain):
                            edges.append({
                                "kind": "rewrite_step",
                                "head": head,
                                "premises": [{"name": c["lemma"], "direction": c["direction"]}],
                                "lemmas": [c["lemma"]],
                                "directions": [c["direction"]],
                                "tactic_text": c["tactic_text"],
                                "in_state": c["in_state"],
                                "out_states": c["out_states"],
                                "source_tactic_index": i,
                                "sub_index": j,
                                "is_fallback": False,
                                "is_branching": False,
                            })
                        state = last_state
                        result["n_unrolled_edges"] += len(chain)
                        continue
                    elif fallback_on_drift:
                        edges.append({
                            "kind": "simp_bundle",
                            "head": head,
                            "premises": prems,
                            "lemmas": sorted(p["name"] for p in prems),
                            "directions": [p["direction"] for p in prems],
                            "tactic_text": tactic,
                            "in_state": in_pp,
                            "out_states": [trace_after],
                            "source_tactic_index": i,
                            "sub_index": 0,
                            "is_fallback": True,
                            "is_branching": False,
                            "fallback_reason": f"rw_drift:{why}",
                        })
                        result["n_fallback"] += 1
                        nxt = run_tac_safely(dojo, state, tactic)
                        if isinstance(nxt, str):
                            result["error"] = f"step {i}: rw fallback advance failed: {nxt}"
                            break
                        state = nxt
                        continue
                    else:
                        result["error"] = f"step {i}: rw stepwise drifted ({why})"
                        break

                # ---- 4. Term / structural / other -------------------------
                kind = "term" if head in TERM_HEADS else "structural"
                out_goals = split_goals(trace_after)
                edges.append({
                    "kind": kind,
                    "head": head,
                    "premises": prems,
                    "lemmas": [],
                    "directions": [],
                    "tactic_text": tactic,
                    "in_state": in_pp,
                    "out_states": out_goals if len(out_goals) > 1 else [trace_after],
                    "source_tactic_index": i,
                    "sub_index": 0,
                    "is_fallback": False,
                    "is_branching": len(out_goals) > 1,
                })
                nxt = run_tac_safely(dojo, state, tactic)
                if isinstance(nxt, str):
                    result["error"] = f"step {i}: {head} failed: {nxt}"
                    break
                state = nxt

        result["dojo_total_s"] = time.time() - t_open0
    except DojoInitError as e:
        result["error"] = f"DojoInitError: {e}"
    except DojoCrashError as e:
        result["error"] = f"DojoCrashError: {e}"
    except Exception as e:  # noqa
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=sys.stderr)

    return result


def append_locked(path: Path, record: dict) -> None:
    s = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(s)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mathlib-url", required=True)
    ap.add_argument("--mathlib-commit", required=True)
    ap.add_argument("--dojo-timeout", type=int, default=1800)
    ap.add_argument("--fallback-on-drift", action="store_true")
    args = ap.parse_args()

    payload_path = Path(args.payload)
    out_path = Path(args.out)
    if not payload_path.exists():
        log(f"payload not found: {payload_path}")
        sys.exit(2)

    log(f"loading payload {payload_path}")
    recs: list[dict] = []
    with payload_path.open() as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    log(f"loaded {len(recs)} theorems")

    repo = LeanGitRepo(args.mathlib_url, args.mathlib_commit)
    log(f"opened LeanGitRepo {args.mathlib_url}@{args.mathlib_commit[:8]}")

    n_ok = 0
    n_err = 0
    t0 = time.time()
    for i, rec in enumerate(recs):
        thm_t0 = time.time()
        try:
            out_rec = process_theorem(repo, rec, args.dojo_timeout, args.fallback_on_drift)
        except Exception as e:  # noqa
            out_rec = {
                "name": rec.get("full_name", ""),
                "file_path": rec.get("file_path", ""),
                "split": rec.get("split", ""),
                "edges": [],
                "n_original_tactics": len(rec.get("traced_tactics", [])),
                "n_unrolled_edges": 0,
                "n_fallback": 0,
                "error": f"toplevel:{type(e).__name__}: {e}",
            }
            traceback.print_exc(file=sys.stderr)

        append_locked(out_path, out_rec)
        thm_dt = time.time() - thm_t0
        if out_rec.get("error"):
            n_err += 1
            log(f"[{i+1}/{len(recs)}] {out_rec['name']}: ERR ({thm_dt:.1f}s) {out_rec['error']}")
        else:
            n_ok += 1
            log(
                f"[{i+1}/{len(recs)}] {out_rec['name']}: ok "
                f"({thm_dt:.1f}s, {len(out_rec['edges'])} edges, "
                f"unrolled={out_rec.get('n_unrolled_edges', 0)}, "
                f"fallback={out_rec.get('n_fallback', 0)})"
            )

    dt = time.time() - t0
    log(f"done: ok={n_ok} err={n_err} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
