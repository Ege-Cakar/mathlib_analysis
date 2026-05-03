#!/usr/bin/env python3
"""Kimina whole-proof runner for post-cutoff Mathlib declarations.

The output schema is fixed even when model dependencies are unavailable:
one JSON object per theorem/sample with the prompt, raw generation, extracted
Lean code, Lean verifier status, and stderr/stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


MODEL = "AI-MO/Kimina-Prover-RL-1.7B"
DECL_RE = re.compile(r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+")


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:lean4|lean)?\n(.*?)```", text, flags=re.S)
    return blocks[-1].strip() if blocks else text.strip()


def candidate_block(row: dict, code: str) -> str:
    lines = [ln for ln in code.strip().splitlines() if not ln.strip().startswith(("import ", "set_option "))]
    for i, line in enumerate(lines):
        if DECL_RE.match(line):
            return "\n".join(lines[i:]).rstrip() + "\n"
    proof = "\n".join(lines).strip()
    if proof.startswith("by"):
        return row["declaration_header"].removesuffix(" := by") + " := " + proof + "\n"
    return row["declaration_header"].rstrip() + "\n" + proof + "\n"


def verify(mathlib: Path, row: dict, code: str, timeout: int) -> tuple[bool, str]:
    src = mathlib / row["file_path"]
    if not src.exists():
        return False, f"missing source file: {src}"
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
        proc = subprocess.run(
            ["lake", "env", "lean", str(path)],
            cwd=mathlib,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    return proc.returncode == 0, proc.stdout + proc.stderr


def transformers_generate(model_name: str, prompts: list[str], max_new_tokens: int, temperature: float) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    outs = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
        outs.append(tok.decode(gen[0][inp["input_ids"].shape[1] :], skip_special_tokens=True))
    return outs


def vllm_generate(model_name: str, prompts: list[str], max_new_tokens: int, temperature: float) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, trust_remote_code=True)
    params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
    return [o.outputs[0].text for o in llm.generate(prompts, params)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="results/prover_eval/kimina_prompts.jsonl")
    p.add_argument("--output", default="results/prover_eval/kimina_post_2025_08_14.jsonl")
    p.add_argument("--mathlib-dir", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--backend", choices=["vllm", "transformers", "prompts-only"], default="vllm")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--lean-timeout", type=int, default=120)
    args = p.parse_args()

    rows = []
    with Path(args.manifest).open() as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            rows.append(json.loads(line))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as out:
        for row in rows:
            prompts = [row["prompt"]] * args.samples
            if args.backend == "prompts-only":
                generations = [""] * args.samples
            elif args.backend == "vllm":
                generations = vllm_generate(args.model, prompts, args.max_new_tokens, args.temperature)
            else:
                generations = transformers_generate(args.model, prompts, args.max_new_tokens, args.temperature)

            for sample_id, raw in enumerate(generations):
                code = extract_code(raw)
                passed = False
                lean_output = "not run"
                if code:
                    try:
                        if re.search(r"\b(sorry|admit)\b", code):
                            lean_output = "rejected before Lean: generated proof contains sorry/admit"
                        else:
                            passed, lean_output = verify(Path(args.mathlib_dir), row, code, args.lean_timeout)
                    except Exception as exc:
                        lean_output = repr(exc)
                rec = {
                    "full_name": row["full_name"],
                    "file_path": row["file_path"],
                    "category": row["category"],
                    "line": row["line"],
                    "sample_id": sample_id,
                    "model": args.model,
                    "prompt": row["prompt"],
                    "raw_generation": raw,
                    "lean_code": code,
                    "passed": passed,
                    "lean_output": lean_output[-8000:],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
