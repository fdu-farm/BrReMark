"""
BrReMark Two-Turn Inference(vLLM OpenAI API)

Pipeline:
  Turn 1: System + [Image, Task Prompt] → <think> + <tool_call> mark_bbox
  Turn 2: [Marked Image, Feedback] → <rethink> + <answer>

Requirements:
  pip install openai pillow

Usage:
  # 1. Start vLLM server
  bash inference/start_server.sh path/to/checkpoint 8001 0

  # 2. Run inference
  python inference/inference.py --image path/to/brain_mri.png --task task_name
"""

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from openai import OpenAI
from PIL import Image, ImageDraw


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a medical image analysis assistant specialized in brain MRI abnormality detection and diagnostic reasoning. Your goal is to analyze brain MRI images to detect abnormalities and provide diagnostic assessments. The main tasks include lesion localization, imaging description, and diagnosis. You can rely on your medical knowledge and use the marking tool to assist in analysis.

Your output should follow a two-phase reasoning format. First, observe the image, identify findings, and form initial hypothesis in a think section, then call the marking tool. After receiving the marked image, combine it with your medical knowledge to verify, analyze, and reach conclusion in a rethink section, followed by your final answer.

Format your response as:
<think>your observations and initial hypothesis</think>
<tool_call>{"name": "mark_bbox", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "finding", "score": 0.0}}</tool_call>

Then after seeing the marked image:
<rethink>evaluate your initial hypothesis, then combine the marked image with medical knowledge for diagnostic reasoning</rethink>
<answer>your final answer</answer>

For normal images, use bbox_2d: null with label "normal".
The score field is optional and should be in [0, 1] if provided."""


# =============================================================================
# Task-Specific Prompts
# =============================================================================

ROUND1_PROMPTS = {
    "localization": "Is there any lesion in this brain MRI? Mark its location if present. If possible, include a confidence score in [0,1] in the tool_call.",
    "description": "Describe the imaging characteristics of any lesion in this brain MRI.",
    "diagnosis": "What is the most likely diagnosis for this brain MRI? Provide your diagnostic reasoning based on the imaging features observed.",
}

ROUND2_PROMPTS = {
    "abnormal": "Region of interest has been marked at {bbox}. Please rethink and provide your final answer combining the marked image with your medical knowledge.",
    "abnormal_no_draw": "You identified a region of interest at {bbox}. Please rethink and provide your final answer based on your observation and medical knowledge.",
    "normal": "No abnormality marked. Confirm your assessment by reviewing the image.",
    "uncertain": "Review the image and provide your assessment: Is there any abnormality present?",
}


# =============================================================================
# Utilities
# =============================================================================

def encode_image(image_path: str) -> str:
    """Encode image file to base64 data URL."""
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


def encode_pil_image(image: Image.Image) -> str:
    """Encode PIL image to base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def create_marked_image(image: Image.Image, bbox: list) -> Image.Image:
    """Draw red bounding box on image."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    return marked


def extract_tool_call(text: str) -> Optional[dict]:
    """Extract and parse <tool_call> JSON from model output."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def extract_bbox(tool_call: Optional[dict]) -> Optional[list]:
    """Extract bbox_2d from parsed tool_call."""
    if tool_call:
        bbox = tool_call.get("arguments", {}).get("bbox_2d")
        if bbox is None:
            return None
        return bbox
    return None


def extract_score(tool_call: Optional[dict]) -> Optional[float]:
    """Extract optional confidence score from tool_call."""
    if not tool_call:
        return None
    args = tool_call.get("arguments", {})
    for key in ("score", "confidence"):
        if key in args:
            try:
                return float(args[key])
            except (TypeError, ValueError):
                continue
    return None


def extract_label(tool_call: Optional[dict]) -> Optional[str]:
    """Extract label from tool_call."""
    if tool_call:
        return tool_call.get("arguments", {}).get("label")
    return None


def extract_section(text: str, tag: str) -> str:
    """Extract content between XML-style tags."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def is_normal_from_tool_call(tool_call: Optional[dict]) -> bool:
    """Check if tool_call explicitly indicates normal (no lesion)."""
    if not tool_call:
        return False
    args = tool_call.get("arguments", {})
    label = (args.get("label") or "").lower()
    bbox = args.get("bbox_2d")
    has_valid_bbox = bbox is not None and isinstance(bbox, list) and len(bbox) == 4

    if label == "normal" and not has_valid_bbox:
        return True
    if bbox is None and label == "":
        return True
    return False


# =============================================================================
# Two-Turn Inference (matches local_vllm_two_turn_v2.py)
# =============================================================================

def run_two_turn_inference(
    client: OpenAI,
    model: str,
    image_path: str,
    task: str = "diagnosis",
    max_tokens: int = 512,
    draw_bbox: bool = True,
) -> Dict[str, Any]:
    """
    Two-turn inference matching SFT V2 training format.

    Turn 1: System + [Image, Task Prompt] → <think> + <tool_call>
    Turn 2: [Marked Image, Feedback] → <rethink> + <answer>

    Returns dict with: think, bbox, label, score, rethink, answer, is_normal, raw responses
    """
    image = Image.open(image_path).convert("RGB")
    image_url = encode_image(image_path)
    round1_prompt = ROUND1_PROMPTS[task]

    # =========================================================================
    # Turn 1: Hypothesis + Localization
    # =========================================================================
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": round1_prompt},
        ]},
    ]

    r0 = client.chat.completions.create(
        model=model, messages=messages, temperature=0.1, max_tokens=max_tokens
    ).choices[0].message.content

    think = extract_section(r0, "think")
    tool_call = extract_tool_call(r0)
    bbox = extract_bbox(tool_call)
    label = extract_label(tool_call)
    score = extract_score(tool_call)
    is_normal_r1 = is_normal_from_tool_call(tool_call)

    # =========================================================================
    # Turn 2: Verification + Diagnosis
    # =========================================================================
    if is_normal_r1:
        # Normal case: no bbox, use original image
        feedback = ROUND2_PROMPTS["normal"]
        marked_url = image_url
    else:
        if bbox and len(bbox) == 4:
            if draw_bbox:
                marked_image = create_marked_image(image, bbox)
                marked_url = encode_pil_image(marked_image)
                feedback = ROUND2_PROMPTS["abnormal"].format(bbox=bbox)
            else:
                marked_url = image_url
                feedback = ROUND2_PROMPTS["abnormal_no_draw"].format(bbox=bbox)
        else:
            # Invalid bbox but not marked as normal
            marked_url = image_url
            feedback = ROUND2_PROMPTS["uncertain"]

    # NOTE: Image comes BEFORE text to match training format
    messages.extend([
        {"role": "assistant", "content": r0},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": marked_url}},
            {"type": "text", "text": feedback},
        ]},
    ])

    r1 = client.chat.completions.create(
        model=model, messages=messages, temperature=0.1, max_tokens=max_tokens
    ).choices[0].message.content

    rethink = extract_section(r1, "rethink") or extract_section(r1, "think")
    answer = extract_section(r1, "answer") or r1

    return {
        "think": think,
        "tool_call": tool_call,
        "bbox": bbox,
        "label": label,
        "score": score,
        "is_normal": is_normal_r1,
        "rethink": rethink,
        "answer": answer,
        "raw_r1": r0,
        "raw_r2": r1,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BrReMark Two-Turn Inference")
    parser.add_argument("--image", required=True, help="Path to brain MRI image")
    parser.add_argument("--task", choices=["localization", "description", "diagnosis"], default="diagnosis")
    parser.add_argument("--model", default=None, help="Model path (auto-detect from vLLM if not set)")
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--no-draw", action="store_true", help="Don't draw bbox on image in Turn 2")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--save-marked", help="Save marked image to path")
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=300.0)

    # Auto-detect model from vLLM server
    if args.model is None:
        models = client.models.list()
        args.model = models.data[0].id
        print(f"Auto-detected model: {args.model}\n")

    # Run two-turn inference
    print("=" * 60)
    print(f"  BrReMark Two-Turn Inference")
    print(f"  Task: {args.task}")
    print(f"  Image: {args.image}")
    print("=" * 60)

    result = run_two_turn_inference(
        client=client,
        model=args.model,
        image_path=args.image,
        task=args.task,
        max_tokens=args.max_tokens,
        draw_bbox=not args.no_draw,
    )

    # Print Turn 1
    print(f"\n{'─' * 60}")
    print("TURN 1: Hypothesis & Localization")
    print(f"{'─' * 60}")
    print(f"\n[Think]\n{result['think']}\n")
    print(f"[Tool Call] {json.dumps(result['tool_call'], indent=2) if result['tool_call'] else 'None'}")
    print(f"[BBox] {result['bbox']}")
    print(f"[Label] {result['label']}")
    print(f"[Score] {result['score']}")
    print(f"[Normal] {result['is_normal']}")

    # Print Turn 2
    print(f"\n{'─' * 60}")
    print("TURN 2: Verification & Diagnosis")
    print(f"{'─' * 60}")
    print(f"\n[Rethink]\n{result['rethink']}\n")
    print(f"[Answer]\n{result['answer']}\n")

    # Save marked image
    if args.save_marked and result["bbox"]:
        image = Image.open(args.image).convert("RGB")
        marked = create_marked_image(image, result["bbox"])
        marked.save(args.save_marked)
        print(f"Marked image saved to {args.save_marked}")

    # Save results
    if args.output:
        out = {k: v for k, v in result.items() if not k.startswith("raw_")}
        out["raw_r1"] = result["raw_r1"]
        out["raw_r2"] = result["raw_r2"]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
