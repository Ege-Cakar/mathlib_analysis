#!/usr/bin/env python3
"""CUDA / vLLM whole-proof runner for Kimina-Prover-RL on the post-cutoff
Mathlib holdout.

Design
------
- Generation step is GPU-only and batched. Use vLLM as the primary backend
  (single batched call, n samples per prompt). Fall back to a transformers
  path that does per-prompt batched generation in bf16 with KV cache only
  if vLLM is unavailable.
- Lean verification is decoupled. The default invocation writes raw model
  outputs to <output>.jsonl. A second `--verify-only` pass reads that file,
  runs `lake env lean` on each candidate, and writes verified results to
  <output>.verified.jsonl. Verification can run on a CPU-only machine with
  a Mathlib checkout, while generation needs the GPU.
- Per-prompt sampling is set up with `SamplingParams(n=samples)` so that all
  samples run inside one vLLM forward pass per prompt. We additionally batch
  *across* prompts so a single `llm.generate(...)` call covers the whole
  manifest.
- bf16 by default on Ampere+; fp16 fallback if user explicitly requests it.

Usage
-----
    # generation only, vLLM, 8 samples per prompt, on a single H100
    python3 analysis/kimina_eval_cuda.py generate \\
        --manifest results/prover_eval/kimina_prompts.jsonl \\
        --output  results/prover_eval/kimina_post_2025_08_14.jsonl \\
        --model AI-MO/Kimina-Prover-RL-1.7B \\
        --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \\
        --samples 8 --max-tokens 2048

    # verification pass on a CPU box that has Mathlib
    python3 analysis/kimina_eval_cuda.py verify \\
        --generations results/prover_eval/kimina_post_2025_08_14.jsonl \\
        --mathlib-dir /path/to/Mathlib4 \\
        --output      results/prover_eval/kimina_post_2025_08_14.verified.jsonl

    # one-shot generate-then-verify (needs GPU + Mathlib on the same box)
    python3 analysis/kimina_eval_cuda.py all \\
        --manifest results/prover_eval/kimina_prompts.jsonl \\
        --mathlib-dir /path/to/Mathlib4 \\
        --output    results/prover_eval/kimina_post_2025_08_14.jsonl \\
        --samples 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

DEFAULT_MODEL = "AI-MO/Kimina-Prover-RL-1.7B"
DECL_RE = re.compile(r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+")
CODE_BLOCK_RE = re.compile(r"```(?:lean4|lean)?\n(.*?)```", re.S)


# ----------------------------- helpers -----------------------------------

def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_lean_block(text: str) -> str:
    blocks = CODE_BLOCK_RE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    return (text or "").strip()


def candidate_block(row: dict, code: str) -> str:
    """Build the replacement block for `verify`: prefer the model's own
    declaration if it parses, otherwise prepend the manifest header."""
    lines = [ln for ln in code.strip().splitlines() if not ln.strip().startswith(("import ", "set_option "))]
    for i, line in enumerate(lines):
        if DECL_RE.match(line):
            return "\n".join(lines[i:]).rstrip() + "\n"
    proof = "\n".join(lines).strip()
    if proof.startswith("by"):
        return row["declaration_header"].removesuffix(" := by") + " := " + proof + "\n"
    return row["declaration_header"].rstrip() + "\n" + proof + "\n"


# ----------------------------- generation --------------------------------

def cuda_preflight(require: bool) -> dict:
    info = {"cuda_available": False, "torch_present": False}
    try:
        import torch
        info["torch_present"] = True
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["bf16_supported"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception as exc:
        info["error"] = repr(exc)
    if require and not info["cuda_available"]:
        sys.stderr.write("CUDA not available; aborting. Pass --allow-cpu to override.\n")
        sys.stderr.write(json.dumps(info, indent=2) + "\n")
        sys.exit(2)
    return info


def vllm_generate_batch(model_name: str, prompts: list[str], *, samples: int, temperature: float, top_p: float,
                        max_tokens: int, tp: int, gpu_mem: float, dtype: str, max_model_len: int | None,
                        seed: int) -> list[list[str]]:
    """Returns a list of length len(prompts), each a list of `samples` raw
    completion strings. One vLLM forward batch covers all prompts."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        dtype=dtype,
        max_model_len=max_model_len,
        seed=seed,
    )
    params = SamplingParams(n=samples, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    print(f"[vllm] generating {len(prompts)} prompts × {samples} samples = {len(prompts) * samples} completions")
    t0 = time.perf_counter()
    outs = llm.generate(prompts, params)
    dt = time.perf_counter() - t0
    print(f"[vllm] generation done in {dt:.1f}s "
          f"({len(prompts) * samples / max(dt, 1e-6):.1f} completions/s)")
    # vLLM returns a list of RequestOutput, each with .outputs (n samples).
    aligned = []
    by_pid = {o.request_id: o for o in outs} if hasattr(outs[0], "request_id") else None
    if by_pid is not None and len(by_pid) == len(prompts):
        # Preserve input order.
        for i in range(len(prompts)):
            ro = by_pid.get(str(i)) or outs[i]
            aligned.append([s.text for s in ro.outputs])
    else:
        for ro in outs:
            aligned.append([s.text for s in ro.outputs])
    return aligned


def hf_generate_batch(model_name: str, prompts: list[str], *, samples: int, temperature: float, top_p: float,
                      max_tokens: int, dtype: str, batch_size: int, seed: int) -> list[list[str]]:
    """Transformers backend. One batched generation per prompt (batch axis =
    samples). Slower than vLLM but works without it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"[transformers] loading {model_name} dtype={dtype} cuda={torch.cuda.is_available()}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    aligned: list[list[str]] = []
    for prompt in prompts:
        outs: list[str] = []
        remaining = samples
        while remaining > 0:
            cur = min(remaining, batch_size)
            inputs = tok([prompt] * cur, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    pad_token_id=tok.pad_token_id,
                )
            decoded = tok.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.extend(decoded)
            remaining -= cur
        aligned.append(outs)
    return aligned


def generate_cmd(args: argparse.Namespace) -> None:
    info = cuda_preflight(require=not args.allow_cpu)
    print("[cuda]", json.dumps(info, indent=2))

    rows = list(iter_jsonl(Path(args.manifest)))
    if args.limit:
        rows = rows[: args.limit]
    prompts = [row["prompt"] for row in rows]
    print(f"[manifest] {len(rows)} rows from {args.manifest}")

    if args.backend == "vllm":
        samples_per_prompt = vllm_generate_batch(
            args.model,
            prompts,
            samples=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            tp=args.tensor_parallel_size,
            gpu_mem=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            seed=args.seed,
        )
    elif args.backend == "transformers":
        samples_per_prompt = hf_generate_batch(
            args.model,
            prompts,
            samples=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            dtype=args.dtype,
            batch_size=args.hf_batch_size,
            seed=args.seed,
        )
    else:
        raise ValueError(f"unknown backend: {args.backend}")

    out_path = Path(args.output)
    records = []
    for row, gens in zip(rows, samples_per_prompt):
        for sid, raw in enumerate(gens):
            code = extract_lean_block(raw)
            records.append({
                "full_name": row["full_name"],
                "file_path": row["file_path"],
                "category": row.get("category"),
                "line": row.get("line"),
                "declaration_header": row["declaration_header"],
                "model": args.model,
                "sample_id": sid,
                "prompt": row["prompt"],
                "raw_generation": raw,
                "lean_code": code,
                "verified": None,
                "lean_output": None,
            })
    write_jsonl(out_path, records)
    print(f"wrote {len(records)} generations to {out_path}")


# ----------------------------- verification ------------------------------

def verify_one(mathlib: Path, row: dict, code: str, timeout: int) -> tuple[bool, str]:
    src = mathlib / row["file_path"]
    if not src.exists():
        return False, f"missing source file: {src}"
    if re.search(r"\b(sorry|admit)\b", code):
        return False, "rejected before Lean: contains sorry/admit"
    lines = src.read_text(errors="ignore").splitlines(keepends=True)
    start = max(0, int(row["line"]) - 1)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if DECL_RE.match(lines[j]):
            end = j
            break
    file_text = "".join(lines[:start]) + candidate_block(row, code) + "\n" + "".join(lines[end:])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / Path(row["file_path"]).name
        path.write_text(file_text)
        try:
            proc = subprocess.run(
                ["lake", "env", "lean", str(path)],
                cwd=mathlib,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return proc.returncode == 0, proc.stdout + proc.stderr


def verify_cmd(args: argparse.Namespace) -> None:
    rows = list(iter_jsonl(Path(args.generations)))
    print(f"[verify] {len(rows)} generations from {args.generations}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pass = 0
    with out_path.open("w") as out:
        for i, row in enumerate(rows):
            code = row.get("lean_code") or extract_lean_block(row.get("raw_generation", ""))
            ok, log = verify_one(Path(args.mathlib_dir), row, code, args.lean_timeout)
            row["lean_code"] = code
            row["verified"] = ok
            row["lean_output"] = log[-args.log_clip:]
            if ok:
                n_pass += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(rows)}]  passed so far: {n_pass}")
    by_thm = {}
    for r in iter_jsonl(out_path):
        by_thm.setdefault(r["full_name"], []).append(bool(r["verified"]))
    pass_at_n = sum(1 for v in by_thm.values() if any(v))
    print(f"[verify] sample-level pass: {n_pass}/{len(rows)} = {n_pass / max(1, len(rows)):.3f}")
    print(f"[verify] theorem-level pass@n: {pass_at_n}/{len(by_thm)} = {pass_at_n / max(1, len(by_thm)):.3f}")
    summary = out_path.with_suffix(".summary.json")
    summary.write_text(json.dumps({
        "generations": len(rows),
        "samples_passed": n_pass,
        "theorems_total": len(by_thm),
        "theorems_passed_at_n": pass_at_n,
        "pass_at_n_fraction": pass_at_n / max(1, len(by_thm)),
    }, indent=2))
    print(f"wrote summary to {summary}")


# ----------------------------- all --------------------------------------

def all_cmd(args: argparse.Namespace) -> None:
    generate_cmd(args)
    if args.mathlib_dir:
        verify_args = argparse.Namespace(
            generations=args.output,
            output=str(Path(args.output).with_suffix(".verified.jsonl")),
            mathlib_dir=args.mathlib_dir,
            lean_timeout=args.lean_timeout,
            log_clip=args.log_clip,
        )
        verify_cmd(verify_args)


# ----------------------------- CLI --------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="GPU/CUDA generation only")
    g.add_argument("--manifest", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    g.add_argument("--samples", type=int, default=8)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--tensor-parallel-size", type=int, default=1)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    g.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--hf-batch-size", type=int, default=4)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--allow-cpu", action="store_true")
    g.set_defaults(func=generate_cmd)

    v = sub.add_parser("verify", help="Lean verification of pre-generated outputs")
    v.add_argument("--generations", required=True)
    v.add_argument("--output", required=True)
    v.add_argument("--mathlib-dir", required=True)
    v.add_argument("--lean-timeout", type=int, default=120)
    v.add_argument("--log-clip", type=int, default=4000)
    v.set_defaults(func=verify_cmd)

    a = sub.add_parser("all", help="generate, then verify")
    for arg in ("--manifest",):
        a.add_argument(arg, required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--model", default=DEFAULT_MODEL)
    a.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    a.add_argument("--samples", type=int, default=8)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--max-tokens", type=int, default=2048)
    a.add_argument("--temperature", type=float, default=0.7)
    a.add_argument("--top-p", type=float, default=0.95)
    a.add_argument("--tensor-parallel-size", type=int, default=1)
    a.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    a.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    a.add_argument("--max-model-len", type=int, default=None)
    a.add_argument("--hf-batch-size", type=int, default=4)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--allow-cpu", action="store_true")
    a.add_argument("--mathlib-dir")
    a.add_argument("--lean-timeout", type=int, default=120)
    a.add_argument("--log-clip", type=int, default=4000)
    a.set_defaults(func=all_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
