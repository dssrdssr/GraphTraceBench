from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

ANSWER_RE = re.compile(r'-?\d+')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot / few-shot eval for OpenAI, Anthropic, or Gemini APIs with retry and resume support."
    )
    p.add_argument("--provider", type=str, required=True, choices=["openai", "anthropic", "gemini"])
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--few_shot_k", type=int, default=0)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--anthropic-base-url", default=None)
    p.add_argument("--gemini-base-url", default=None)
    p.add_argument("--max_retries", type=int, default=8, help="Maximum retries per sample")
    p.add_argument("--retry_initial_delay", type=float, default=3.0, help="Initial retry delay in seconds")
    p.add_argument("--retry_max_delay", type=float, default=60.0, help="Maximum retry delay in seconds")
    p.add_argument("--request_timeout", type=float, default=120.0, help="Request timeout in seconds where supported")
    p.add_argument("--resume", action="store_true", help="Resume from existing progress file if present")
    p.add_argument("--save_every", type=int, default=1, help="Flush progress every N completed samples")
    p.add_argument("--fail_fast", action="store_true", help="Stop immediately on non-retryable errors")
    return p.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_answer(text: str) -> Optional[int]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            return int(obj["answer"])
    except Exception:
        pass
    m = ANSWER_RE.search(text)
    return int(m.group()) if m else None


def get_user_prompt(sample: Dict[str, Any]) -> str:
    prompt = sample.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    msgs = sample.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError(f"Cannot find user prompt in sample with keys={list(sample.keys())}")


def get_assistant_answer(sample: Dict[str, Any]) -> str:
    target = sample.get("target")
    if target is not None:
        return json.dumps({"answer": int(target)}, ensure_ascii=False)
    completion = sample.get("completion")
    if isinstance(completion, str) and completion.strip():
        return completion
    msgs = sample.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError(f"Cannot find assistant answer/target in sample with keys={list(sample.keys())}")


def build_messages(example: Dict[str, Any], shots: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for s in shots:
        messages.append({"role": "user", "content": get_user_prompt(s)})
        messages.append({"role": "assistant", "content": get_assistant_answer(s)})
    messages.append({"role": "user", "content": get_user_prompt(example)})

    for i, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Invalid message at index {i}: {m}")
    return messages


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [r for r in results if r.get("pred") is not None and r.get("target") is not None]
    if not preds:
        return {"count": len(results), "valid_predictions": 0}
    abs_err = [abs(r["pred"] - r["target"]) for r in preds]
    sq_err = [(r["pred"] - r["target"]) ** 2 for r in preds]
    exact = sum(int(r["pred"] == r["target"]) for r in preds)
    return {
        "count": len(results),
        "valid_predictions": len(preds),
        "exact_match": exact / len(preds),
        "mae": sum(abs_err) / len(abs_err),
        "rmse": math.sqrt(sum(sq_err) / len(sq_err)),
    }


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        parts.append(f"[{m['role'].upper()}]\n{m['content']}")
    return "\n\n".join(parts)


def call_openai(model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, base_url: Optional[str], timeout: float) -> str:
    from openai import OpenAI

    actual_base = base_url or os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=actual_base, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, base_url: Optional[str], timeout: float) -> str:
    import anthropic

    actual_base = base_url or os.environ.get("ANTHROPIC_BASE_URL")
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=actual_base,
        timeout=timeout,
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": messages_to_prompt(messages)}],
    )
    chunks = []
    for blk in resp.content:
        txt = getattr(blk, "text", None)
        if txt:
            chunks.append(txt)
    return "\n".join(chunks)


def call_gemini(model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, base_url: Optional[str], timeout: float) -> str:
    from google import genai
    from google.genai import types

    actual_base = base_url or os.environ.get("GEMINI_BASE_URL")
    client_kwargs: Dict[str, Any] = {"api_key": os.environ["GEMINI_API_KEY"]}
    if actual_base:
        client_kwargs["http_options"] = {"base_url": actual_base}
    client = genai.Client(**client_kwargs)
    prompt = messages_to_prompt(messages)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return getattr(resp, "text", None) or str(resp)


def should_retry(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_signals = [
        "rate limit",
        "timeout",
        "temporarily",
        "temporarily unavailable",
        "overloaded",
        "internalservererror",
        "server error",
        "service unavailable",
        "connection",
        "apierror",
        "没有可用token",
        "token",
        "quota",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]
    non_retry_signals = [
        "invalid api key",
        "authentication",
        "permission",
        "not found",
        "does not exist",
        "invalid_request_error",
        "context length",
        "maximum context",
    ]
    if any(sig in text for sig in non_retry_signals):
        return False
    return any(sig in text for sig in retry_signals)


def call_with_retry(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    base_url: Optional[str],
    timeout: float,
    max_retries: int,
    initial_delay: float,
    max_delay: float,
) -> Tuple[str, int]:
    delay = initial_delay
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openai":
                text = call_openai(model, messages, temperature, max_tokens, base_url, timeout)
            elif provider == "anthropic":
                text = call_anthropic(model, messages, temperature, max_tokens, base_url, timeout)
            else:
                text = call_gemini(model, messages, temperature, max_tokens, base_url, timeout)
            return text, attempt
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not should_retry(exc):
                raise
            print(f"[retry] attempt {attempt}/{max_retries} failed: {exc}")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
    raise last_exc if last_exc is not None else RuntimeError("Unknown retry failure")


def progress_paths(output_file: str) -> Tuple[Path, Path]:
    out = Path(output_file)
    progress = out.with_suffix(out.suffix + ".progress.jsonl")
    return out, progress


def load_progress(progress_file: Path) -> List[Dict[str, Any]]:
    if not progress_file.exists():
        return []
    rows = []
    with progress_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_progress(progress_file: Path, row: Dict[str, Any]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_final(output_file: Path, args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    summary = summarize(rows)
    payload = {
        "summary": summary,
        "config": {
            "provider": args.provider,
            "model": args.model,
            "few_shot_k": args.few_shot_k,
            "max_samples": args.max_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "results": rows,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_sample_key(ex: Dict[str, Any], idx: int) -> str:
    if ex.get("id") is not None:
        return str(ex["id"])
    return f"idx:{idx}"


def main() -> None:
    args = parse_args()
    eval_rows = load_jsonl(args.eval_file)
    if args.max_samples is not None:
        eval_rows = eval_rows[: args.max_samples]
    train_rows = load_jsonl(args.train_file) if args.train_file else []
    rng = random.Random(args.seed)

    output_file, progress_file = progress_paths(args.output_file)
    completed_rows: List[Dict[str, Any]] = load_progress(progress_file) if args.resume else []
    completed_keys = {str(r.get("sample_key")) for r in completed_rows}

    if completed_rows:
        print(f"[resume] loaded {len(completed_rows)} completed samples from {progress_file}")

    pending: List[Tuple[int, Dict[str, Any]]] = []
    for idx, ex in enumerate(eval_rows):
        key = make_sample_key(ex, idx)
        if key not in completed_keys:
            pending.append((idx, ex))

    out_rows = list(completed_rows)
    bar = tqdm(pending, total=len(pending))
    processed_since_save = 0

    for idx, ex in bar:
        sample_key = make_sample_key(ex, idx)
        shots = rng.sample(train_rows, k=min(args.few_shot_k, len(train_rows))) if args.few_shot_k > 0 else []
        try:
            messages = build_messages(ex, shots)
            if args.provider == "openai":
                base_url = args.openai_base_url
            elif args.provider == "anthropic":
                base_url = args.anthropic_base_url
            else:
                base_url = args.gemini_base_url

            text, attempts = call_with_retry(
                provider=args.provider,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                base_url=base_url,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                initial_delay=args.retry_initial_delay,
                max_delay=args.retry_max_delay,
            )
            pred = parse_answer(text)
            row = {
                "sample_key": sample_key,
                "id": ex.get("id"),
                "task_type": ex.get("task_type"),
                "target": ex.get("target"),
                "raw_response": text,
                "pred": pred,
                "attempts": attempts,
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "sample_key": sample_key,
                "id": ex.get("id"),
                "task_type": ex.get("task_type"),
                "target": ex.get("target"),
                "raw_response": None,
                "pred": None,
                "attempts": args.max_retries,
                "status": "error",
                "error": str(exc),
            }
            if args.fail_fast:
                append_progress(progress_file, row)
                out_rows.append(row)
                write_final(output_file, args, out_rows)
                raise

        append_progress(progress_file, row)
        out_rows.append(row)
        processed_since_save += 1
        if processed_since_save >= args.save_every:
            write_final(output_file, args, out_rows)
            processed_since_save = 0

    write_final(output_file, args, out_rows)
    print(f"[done] progress file: {progress_file}")


if __name__ == "__main__":
    main()
