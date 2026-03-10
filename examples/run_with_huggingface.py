#!/usr/bin/env python3
"""
Run SpatialBench evaluations using Hugging Face Inference API.

Uses a model like Qwen2.5-Coder to solve tasks. Requires HF_TOKEN environment variable.

Usage:
    export HF_TOKEN=your_huggingface_token
    spatialbench run evals_canonical/qc/my_eval.json --agent local --script-path examples/run_with_huggingface.py

Or with a specific model:
    HF_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct spatialbench run evals/... --agent local --script-path examples/run_with_huggingface.py
"""
import json
import os
import re
from pathlib import Path

try:
    from huggingface_hub import InferenceClient
except ImportError:
    raise SystemExit(
        "Install huggingface_hub: pip install huggingface_hub"
    )

MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")


def main():
    work_dir = Path(os.environ.get("WORK_DIR", "."))
    task_file = work_dir / "task_prompt.txt"
    task_prompt = task_file.read_text()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN for Hugging Face Inference API"
        )

    client = InferenceClient(token=token)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a spatial biology analysis assistant. Reply with valid JSON only for the answer fields requested. Do not include any text outside the JSON object.",
            },
            {"role": "user", "content": task_prompt},
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    raw = response.choices[0].message.content

    answer = parse_answer(raw)
    out = work_dir / "eval_answer.json"
    out.write_text(json.dumps(answer, indent=2))
    print("Wrote", out)


def parse_answer(text: str) -> dict:
    """Extract JSON from model output (handles markdown code blocks and stray text)."""
    text = text.strip()

    # Try <EVAL_ANSWER>...</EVAL_ANSWER> block first (used in some tasks)
    m = re.search(r"<EVAL_ANSWER>\s*([\s\S]*?)\s*</EVAL_ANSWER>", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    # Try ```json ... ``` block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()

    # Find first { ... } object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Malformed JSON in model output")


if __name__ == "__main__":
    main()
