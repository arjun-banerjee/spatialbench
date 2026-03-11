#!/usr/bin/env python3
"""
Run SpatialBench evaluations with a local Hugging Face model via transformers.

Usage:
    pip install "transformers>=4.40" torch
    HF_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
      spatialbench run evals_canonical/qc/my_eval.json \
      --agent local --script-path examples/run_with_huggingface.py

Optional env vars:
    HF_MAX_NEW_TOKENS=4096
    HF_TEMPERATURE=0.0
    HF_TOP_P=0.95
    HF_TRUST_REMOTE_CODE=1
    DEBUG_TRACE=1
"""
import json
import os
import re
from pathlib import Path
from datetime import datetime

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise SystemExit(
        "Install local inference deps: pip install transformers torch"
    )

MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")


def _is_truthy_env(var_name: str) -> bool:
    value = os.environ.get(var_name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _write_debug_trace(
    work_dir: Path, task_prompt: str, model: str, raw: str, answer: dict, metadata: dict
) -> None:
    trace_dir = work_dir / "debug_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    (trace_dir / "task_prompt_snapshot.txt").write_text(task_prompt)
    (trace_dir / "model_raw_response.txt").write_text(raw)
    (trace_dir / "parsed_answer.json").write_text(json.dumps(answer, indent=2))

    trace_metadata = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model": model,
    }
    trace_metadata.update(metadata)

    (trace_dir / "request_metadata.json").write_text(json.dumps(trace_metadata, indent=2))
    print("Wrote debug trace to", trace_dir)


def main():
    work_dir = Path(os.environ.get("WORK_DIR", "."))
    task_file = work_dir / "task_prompt.txt"
    task_prompt = task_file.read_text()

    trust_remote_code = _is_truthy_env("HF_TRUST_REMOTE_CODE")
    max_new_tokens = int(os.environ.get("HF_MAX_NEW_TOKENS", "4096"))
    temperature = float(os.environ.get("HF_TEMPERATURE", "0.0"))
    top_p = float(os.environ.get("HF_TOP_P", "0.95"))
    do_sample = temperature > 0

    device = _pick_device()
    print(f"Loading local model {MODEL} on device={device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=trust_remote_code, torch_dtype="auto")
    model.to(device)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a spatial biology analysis assistant. Reply with valid JSON only "
                "for the answer fields requested. Do not include any text outside the JSON object."
            ),
        },
        {"role": "user", "content": task_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        # Fallback formatting for base models without chat templates.
        prompt = (
            f"System: {messages[0]['content']}\n\n"
            f"User: {messages[1]['content']}\n\n"
            "Assistant:"
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(input_ids, **generate_kwargs)

    completion_ids = output_ids[0, input_ids.shape[-1] :]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True)

    answer = parse_answer(raw)
    out = work_dir / "eval_answer.json"
    out.write_text(json.dumps(answer, indent=2))
    print("Wrote", out)

    if _is_truthy_env("DEBUG_TRACE"):
        usage = {
            "prompt_tokens": int(input_ids.shape[-1]),
            "completion_tokens": int(completion_ids.shape[-1]),
            "total_tokens": int(output_ids.shape[-1]),
        }
        _write_debug_trace(
            work_dir=work_dir,
            task_prompt=task_prompt,
            model=MODEL,
            raw=raw,
            answer=answer,
            metadata={
                "backend": "local_transformers",
                "device": device,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "usage": usage,
            },
        )


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
