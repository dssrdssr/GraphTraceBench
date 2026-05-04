from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

ANSWER_RE = re.compile(r'-?\d+')


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a local HF model or local adapter on graph reasoning JSONL.')
    p.add_argument('--model_path', required=True, help='Base model path or merged model path')
    p.add_argument('--eval_file', required=True)
    p.add_argument('--train_file', default=None)
    p.add_argument('--few_shot_k', type=int, default=0)
    p.add_argument('--output_file', required=True)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--max_new_tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--is_vl', action='store_true', help='Use AutoProcessor + AutoModelForImageTextToText for Qwen2-VL/Qwen3-VL')
    p.add_argument('--adapter_path', default=None, help='Optional PEFT adapter path to load on top of base model')
    return p.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_answer(text: str) -> Optional[int]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'answer' in obj:
            return int(obj['answer'])
    except Exception:
        pass
    m = ANSWER_RE.search(text)
    return int(m.group()) if m else None


def build_messages(example: Dict[str, Any], shots: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages = [
        {'role': 'system', 'content': 'You are a careful graph reasoning assistant. Output only the requested answer format.'}
    ]
    for s in shots:
        user_text = s.get('prompt') or s['messages'][0]['content']
        target = s.get('target')
        assistant_text = json.dumps({'answer': int(target)})
        messages.append({'role': 'user', 'content': user_text})
        messages.append({'role': 'assistant', 'content': assistant_text})
    messages.append({'role': 'user', 'content': example.get('prompt') or example['messages'][0]['content']})
    return messages


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [r for r in results if r.get('pred') is not None and r.get('target') is not None]
    if not preds:
        return {'count': len(results), 'valid_predictions': 0}
    abs_err = [abs(r['pred'] - r['target']) for r in preds]
    sq_err = [(r['pred'] - r['target']) ** 2 for r in preds]
    exact = sum(int(r['pred'] == r['target']) for r in preds)
    return {
        'count': len(results),
        'valid_predictions': len(preds),
        'exact_match': exact / len(preds),
        'mae': sum(abs_err) / len(abs_err),
        'rmse': math.sqrt(sum(sq_err) / len(sq_err)),
    }


def main():
    args = parse_args()
    eval_rows = load_jsonl(args.eval_file)
    if args.max_samples is not None:
        eval_rows = eval_rows[: args.max_samples]
    train_rows = load_jsonl(args.train_file) if args.train_file else []
    rng = random.Random(42)

    if args.is_vl:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype='auto',
            trust_remote_code=True,
            device_map='auto',
        )
        if args.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path)
        apply_chat = processor.apply_chat_template
        eos_id = getattr(processor.tokenizer, 'eos_token_id', None)
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype='auto',
            trust_remote_code=True,
            device_map='auto',
        )
        if args.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        apply_chat = tokenizer.apply_chat_template
        eos_id = getattr(tokenizer, 'eos_token_id', None)

    results = []
    for ex in tqdm(eval_rows):
        shots = rng.sample(train_rows, k=min(args.few_shot_k, len(train_rows))) if args.few_shot_k > 0 else []
        messages = build_messages(ex, shots)
        prompt_text = apply_chat(messages, tokenize=False, add_generation_prompt=True)
        if args.is_vl:
            inputs = processor(text=[prompt_text], images=None, videos=None, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = tokenizer([prompt_text], return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            eos_token_id=eos_id,
        )
        new_tokens = gen[0][inputs['input_ids'].shape[1]:]
        if args.is_vl:
            text = processor.batch_decode([new_tokens], skip_special_tokens=True)[0]
        else:
            text = tokenizer.batch_decode([new_tokens], skip_special_tokens=True)[0]
        pred = parse_answer(text)
        results.append({'id': ex.get('id'), 'task_type': ex.get('task_type'), 'target': ex.get('target'), 'pred': pred, 'raw_response': text})

    summary = summarize(results)
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'summary': summary, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
