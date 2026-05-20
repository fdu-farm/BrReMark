"""
Minimal LLM-as-Judge evaluation script for BrReMark.

Usage:
    python eval/run_judge.py --task task3 --input results.json --output judge_results.json

Environment variables:
    JUDGE_API_KEY: OpenAI API key
    JUDGE_MODEL: Model name
    JUDGE_BASE_URL: API base URL
"""

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from prompts import TASK2_JUDGE_PROMPT, TASK3_JUDGE_PROMPT


def call_judge(client, model: str, prompt: str, max_retries: int = 3):
    import time
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(content[start : end + 1])
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Judge call failed: {e}")
                return None
            time.sleep(2**attempt)
    return None


def evaluate_task2(client, model, samples):
    results = []
    for s in tqdm(samples, desc="Task2 Judge"):
        gt = s.get("gt_caption", "")
        pred = s.get("predicted_caption", "")
        prompt = TASK2_JUDGE_PROMPT.format(
            gt_caption=gt.replace('"', '\\"'),
            pred_caption=pred.replace('"', '\\"'),
        )
        result = call_judge(client, model, prompt)
        results.append({"id": s.get("filename", ""), **(result or {"error": True})})
    return results


def evaluate_task3(client, model, samples):
    results = []
    for s in tqdm(samples, desc="Task3 Judge"):
        gt_disease = s.get("gt_disease", [])
        gt_all = ", ".join(gt_disease) if gt_disease else ""
        gt_impression = s.get("gt_impression", "")
        raw_response = s.get("raw_response", "")

        prompt = TASK3_JUDGE_PROMPT.format(
            gt_diagnosis=gt_all.replace('"', '\\"'),
            gt_impression=(gt_impression or "Not available").replace('"', '\\"'),
            model_response=raw_response[:4000],
        )
        result = call_judge(client, model, prompt)
        results.append({"id": s.get("filename", ""), "gt_disease": gt_all, **(result or {"error": True})})
    return results


def main():
    parser = argparse.ArgumentParser(description="BrReMark LLM-as-Judge Evaluation")
    parser.add_argument("--task", choices=["task2", "task3"], required=True)
    parser.add_argument("--input", required=True, help="Input results JSON file")
    parser.add_argument("--output", required=True, help="Output judge results JSON file")
    parser.add_argument("--model", default=None, help="Judge model (default: env JUDGE_MODEL)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()

    from openai import OpenAI

    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.getenv("JUDGE_MODEL", "gpt-4o-2024-11-20")

    if not api_key:
        raise ValueError("Set JUDGE_API_KEY or OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    with open(args.input) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples, using judge model: {model}")

    if args.task == "task2":
        results = evaluate_task2(client, model, samples)
    else:
        results = evaluate_task3(client, model, samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} judge results to {args.output}")

    # Print summary
    valid = [r for r in results if "error" not in r]
    print(f"\nValid results: {len(valid)}/{len(results)}")

    if args.task == "task3" and valid:
        avg_diag = sum(r.get("DIAGNOSTIC_ACCURACY", 0) for r in valid) / len(valid)
        avg_reason = sum(r.get("REASONING_QUALITY", 0) for r in valid) / len(valid)
        avg_safety = sum(r.get("SAFETY", 0) for r in valid) / len(valid)
        top1 = sum(1 for r in valid if r.get("Top_1", "").lower() == "correct") / len(valid)
        print(f"  DIAGNOSTIC_ACCURACY: {avg_diag:.2f}/10 ({avg_diag/10*100:.1f}%)")
        print(f"  REASONING_QUALITY:   {avg_reason:.2f}/10 ({avg_reason/10*100:.1f}%)")
        print(f"  SAFETY:              {avg_safety:.2f}/10 ({avg_safety/10*100:.1f}%)")
        print(f"  Top-1 Accuracy:      {top1*100:.1f}%")


if __name__ == "__main__":
    main()
