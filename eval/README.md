# Evaluation

BrReMark evaluates across three clinical tasks:

1. **Task 1: Anomaly Localization** — mAP@IoU thresholds (code-based, no LLM judge)
2. **Task 2: Image Description** — Clinical/Modality keyword F1 via LLM-as-judge
3. **Task 3: Differential Diagnosis** — Multi-dimensional scoring via LLM-as-judge

## Quick Start

```bash
pip install openai tqdm

# Set your OpenAI API key
export JUDGE_API_KEY="your-openai-api-key"
export JUDGE_MODEL="gpt-4o-2024-11-20"

# Run evaluation
python eval/run_judge.py --task task3 --input results.json --output judge_results.json
```

## LLM-as-Judge Details

See `prompts.py` for the full evaluation prompts used in our benchmark.
