# Inference

Two-turn "Mark-and-Rethink" inference using vLLM.

## Requirements

```bash
pip install openai pillow
```

## Usage

**Step 1: Start vLLM server**

```bash
bash inference/start_server.sh <model_path> [port] [gpu_id]
```

**Step 2: Run inference**

```bash
python inference/inference.py \
    --image <path_to_brain_mri.png> \
    --task diagnosis \
    --base-url http://localhost:8001/v1
```

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Path to brain MRI image |
| `--task` | `diagnosis` | Task type: `localization`, `description`, `diagnosis` |
| `--base-url` | `http://localhost:8001/v1` | vLLM server URL |
| `--model` | auto-detect | Model name (auto-detected from vLLM if not set) |
| `--max-tokens` | 512 | Max generation tokens per turn |
| `--no-draw` | false | Skip drawing bbox on image in Turn 2 |
| `--save-marked` | - | Save marked image to path |
| `--output` | - | Save results to JSON file |

## Pipeline

```
Turn 1: Image + Task Prompt
    → <think> initial hypothesis
    → <tool_call> mark_bbox([x1, y1, x2, y2])

Turn 2: Marked Image + Feedback
    → <rethink> verification reasoning
    → <answer> final diagnosis
```
