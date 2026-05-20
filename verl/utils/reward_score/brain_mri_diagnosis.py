"""
Brain MRI Open-Ended Reward Function (统一开放式)

支持三种开放式问题类型:
- detection: 开放式异常检测
- caption/description: 开放式影像描述
- open_diagnosis: 开放式诊断

Reward 组成 (总分 1.9):
- format_reward (0.4): 严格轮次检查
  - Round 1: 只能有 <think>，不能有 <rethink>/<answer>
  - Round 2: 只能有 <rethink> + <answer>，不能有 <think>
- localization_reward (0.5): 连续 IoU 评分
- llm_judge (1.0): GPT-4o 分支评分（按 question_type）

负样本处理:
- has_anomaly=False 时，明确 "no abnormality" 可得分
- 编造病灶会被严重扣分

总分 Cap:
- IoU < 0.1 时，总分上限 0.4（更强约束错误定位）
"""

import re
import json
import os
import math
import threading
from pathlib import Path

# Load .env file
_env_path = Path(__file__).parent.parent.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

# GPT client singleton
_gpt_client = None
_gpt_client_error_count = 0

# Local judge singleton (fallback when LLM_JUDGE_MODEL is unset/None)
_local_judge_model = None
_local_judge_tokenizer = None
_local_judge_lock = threading.Lock()
_local_judge_init_lock = threading.Lock()
_prompt_logged = set()


def _is_env_none(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in {"", "none", "null"}


def use_local_judge_backend() -> bool:
    """
    约定：
    - 如果 LLM_JUDGE_MODEL 未设置/为 None/null，则使用本地 Lingshu judge
    - 否则走 OpenAI 兼容 API（保持原逻辑）
    """
    model_name = os.environ.get("LLM_JUDGE_MODEL")
    if _is_env_none(model_name):
        return True
    normalized = model_name.strip().lower()
    # 兼容显式命名本地 judge
    return normalized in {"lingshu-judge", "lingshu_judge"}


def get_local_judge_model():
    """Load local judge model once."""
    global _local_judge_model, _local_judge_tokenizer
    if _local_judge_model is not None and _local_judge_tokenizer is not None:
        return _local_judge_model, _local_judge_tokenizer

    with _local_judge_init_lock:
        # Double-check after lock to avoid duplicate loads under high concurrency.
        if _local_judge_model is not None and _local_judge_tokenizer is not None:
            return _local_judge_model, _local_judge_tokenizer

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = os.environ.get("LOCAL_LLM_JUDGE_PATH", "./model/Lingshu-32B")
            dtype_name = os.environ.get("LOCAL_LLM_JUDGE_DTYPE", "bfloat16").lower()
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)

            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": os.environ.get("LOCAL_LLM_JUDGE_DEVICE_MAP", "auto"),
                "trust_remote_code": True,
            }
            # 低显存兜底，可手动开启
            if os.environ.get("LOCAL_LLM_JUDGE_LOAD_IN_8BIT", "0") == "1":
                model_kwargs["load_in_8bit"] = True
            if os.environ.get("LOCAL_LLM_JUDGE_LOAD_IN_4BIT", "0") == "1":
                model_kwargs["load_in_4bit"] = True

            print(f"[INFO] Loading local LLM judge from: {model_path}")
            _local_judge_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            _local_judge_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            _local_judge_model.eval()
            return _local_judge_model, _local_judge_tokenizer
        except Exception as e:
            print(f"[WARNING] Failed to init local LLM judge: {e}")
            return None, None


def get_gpt_client(force_recreate=False):
    """获取 GPT 客户端单例，支持错误后重建"""
    global _gpt_client, _gpt_client_error_count
    if use_local_judge_backend():
        return None

    if _gpt_client is None or force_recreate:
        try:
            from openai import OpenAI
            import httpx

            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")

            if not api_key:
                print("[WARNING] OPENAI_API_KEY not set, LLM Judge disabled")
                return None

            # Try without proxy first (more reliable)
            _gpt_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=120.0
            )
            _gpt_client_error_count = 0
            if force_recreate:
                print("[INFO] GPT client recreated successfully")
        except Exception as e:
            print(f"[WARNING] Failed to init GPT client: {e}")
    return _gpt_client


def reset_gpt_client_on_error():
    """在连续错误后重置客户端"""
    if use_local_judge_backend():
        return
    global _gpt_client, _gpt_client_error_count
    _gpt_client_error_count += 1
    if _gpt_client_error_count >= 5:
        print(f"[WARNING] {_gpt_client_error_count} consecutive errors, recreating GPT client...")
        _gpt_client = None
        get_gpt_client(force_recreate=True)


def _clean_and_parse_score_json(result: str):
    """Extract first flat JSON object from model output."""
    if not isinstance(result, str):
        return None
    result = result.strip()
    if not result:
        return None
    result = re.sub(r'```(?:json)?\s*', '', result)
    result = result.replace('```', '')
    match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _query_judge_model(prompt: str, max_tokens: int = 500):
    """
    Query remote API or local model depending on LLM_JUDGE_MODEL.
    Returns raw text (may be empty).
    """
    if use_local_judge_backend():
        model, tokenizer = get_local_judge_model()
        if model is None or tokenizer is None:
            return ""

        import torch

        max_new_tokens = int(os.environ.get("LOCAL_LLM_JUDGE_MAX_NEW_TOKENS", "256"))
        max_new_tokens = max(32, min(max_new_tokens, max_tokens))

        with _local_judge_lock:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt

            model_inputs = tokenizer(text, return_tensors="pt")
            # device_map=auto 时将输入放到首个设备
            target_device = getattr(model, "device", None)
            if target_device is not None:
                model_inputs = {k: v.to(target_device) for k, v in model_inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **model_inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
            input_len = model_inputs["input_ids"].shape[1]
            gen_ids = output[0][input_len:]
            return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    client = get_gpt_client()
    if client is None:
        return ""

    model_name = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-2024-11-20")
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def _split_by_env_feedback(solution_str: str) -> tuple:
    """
    Split solution string into round 1 and round 2 by environment feedback.

    Environment feedback is identified by patterns like:
    - <|im_end|>\n<|im_start|>user (Qwen chat format)
    - Region marked at (feedback text)

    Returns:
        (round1_str, round2_str) - round2_str may be empty if no feedback found
    """
    env_patterns = [
        r'<\|im_end\|>\s*\n?\s*<\|im_start\|>user',  # Qwen chat format
        r'Region of interest has been marked at',  # New feedback text
        r'Region marked at',  # Legacy feedback text (backward compatibility)
        r'No abnormality marked',  # Alternative feedback
    ]

    split_pos = -1
    for pattern in env_patterns:
        match = re.search(pattern, solution_str)
        if match:
            split_pos = match.start()
            break

    if split_pos == -1:
        return solution_str, ""

    return solution_str[:split_pos], solution_str[split_pos:]


def _find_tag_spans(text: str, tag: str) -> list:
    """Find all complete <tag>...</tag> spans."""
    pattern = rf'<{tag}>.*?</{tag}>'
    return [m.span() for m in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)]


def _first_tag_span(text: str, tag: str):
    spans = _find_tag_spans(text, tag)
    return spans[0] if spans else None


def _is_valid_bbox_2d(bbox) -> bool:
    """Validate bbox payload in tool_call JSON."""
    if bbox is None:
        return True
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in bbox)


def _has_valid_mark_bbox_tool_call(round1_str: str) -> bool:
    """
    Check whether round1 has exactly one parseable and valid mark_bbox tool_call JSON.
    Accepts either:
    - {"name": "mark_bbox", "arguments": {"bbox_2d": [...], "label": "..."}}
    - {"bbox_2d": [...], "label": "..."} (direct args)
    """
    matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', round1_str, re.DOTALL | re.IGNORECASE)
    if len(matches) != 1:
        return False

    try:
        action_json = json.loads(matches[0].strip())
    except json.JSONDecodeError:
        return False

    if not isinstance(action_json, dict):
        return False

    tool_name = action_json.get("name")
    if tool_name is not None and tool_name != "mark_bbox":
        return False

    args = action_json.get("arguments", action_json)
    if not isinstance(args, dict):
        return False

    has_bbox_key = ("bbox_2d" in args) or ("bbox" in args)
    if not has_bbox_key:
        return False

    bbox = args.get("bbox_2d", args.get("bbox"))
    if not _is_valid_bbox_2d(bbox):
        return False

    label = args.get("label", action_json.get("label"))
    if label is not None and not isinstance(label, str):
        return False

    return True


def _normalize_expected_modality(modality: str | None) -> str | None:
    """Normalize sequence names to a compact canonical label."""
    if not modality:
        return None

    raw = str(modality).strip().lower()
    if not raw or raw in {"na", "n/a", "none", "null", "unknown"}:
        return None

    compact = re.sub(r'[\s_\-]', '', raw)

    alias_map = {
        # Treat common weighted/native variants as the same core modality.
        "t1": {"t1", "t1n", "t1native", "t1w", "t1wi", "t1weighted"},
        "t2": {"t2", "t2w", "t2wi", "t2weighted"},
        "flair": {"flair", "t2f", "t2flair", "t2wflair", "t2wiflair"},
        "dwi": {"dwi", "diffusion", "diffusionweighted"},
        "adc": {"adc", "adcmap", "apparentdiffusioncoefficient"},
        "t1ce": {
            "t1ce", "t1c", "t1gd", "t1postcontrast", "postcontrast", "postgadolinium",
            "gadolinium", "t1plusc", "cet1", "contrastenhancedt1", "t1contrast",
        },
    }

    for canon, aliases in alias_map.items():
        if compact in aliases:
            return canon

    if "flair" in compact:
        return "flair"
    if "dwi" in compact or "diffusion" in compact:
        return "dwi"
    if "adc" in compact:
        return "adc"
    if "t1" in compact and ("gd" in compact or "ce" in compact or "contrast" in compact):
        return "t1ce"
    if compact in {"t1native", "nativet1"}:
        return "t1"
    if "t1" in compact and ("wi" in compact or "weighted" in compact):
        return "t1"
    if "t2" in compact and ("wi" in compact or "weighted" in compact):
        return "t2"
    if "t1" in compact:
        return "t1"
    if "t2" in compact:
        return "t2"

    return compact


def _extract_expected_modality(gt: dict, reference: dict, extra_info: dict) -> str | None:
    """Resolve expected modality from most reliable metadata fields."""
    extra_info = extra_info or {}
    gt = gt or {}
    reference = reference or {}

    candidates = [
        extra_info.get("modality"),
        extra_info.get("modality_type"),
        extra_info.get("mri_modality"),
        extra_info.get("sequence"),
        extra_info.get("sequence_type"),
        extra_info.get("mri_sequence"),
        gt.get("modality"),
        gt.get("modality_type"),
        gt.get("mri_modality"),
        gt.get("sequence"),
        gt.get("sequence_type"),
        gt.get("mri_sequence"),
        reference.get("modality"),
        reference.get("modality_type"),
        reference.get("mri_modality"),
        reference.get("sequence"),
        reference.get("sequence_type"),
        reference.get("mri_sequence"),
    ]

    # BraTS source labels are mapped to standardized clinical sequence names first.
    brats_to_core = {
        "t1n": "t1",
        "t1c": "t1ce",
        "t2w": "t2",
        "t2f": "flair",
    }

    for value in candidates:
        mapped_value = value
        if isinstance(value, str):
            compact_value = re.sub(r'[\s_\-]', '', value.strip().lower())
            mapped_value = brats_to_core.get(compact_value, value)

        normalized = _normalize_expected_modality(mapped_value)
        if normalized:
            return normalized
    return None


def _extract_explicit_modality_claims(text: str) -> list:
    """
    Extract explicit modality claims from model response.

    Covers patterns like:
    - "this image is T2" / "appears to be FLAIR"
    - "T2WI...", "The axial T2 MRI...", "FLAIR sequence shows..."
    - "T2-weighted image", "T1 contrast-enhanced"
    - Sentence-initial modality mentions

    IMPORTANT: This function is designed to avoid "false positive" claims:
    - Descriptive phrases like "T2 hyperintensity" are NOT sequence claims
    - If T1 is mentioned but the sentence also contains contrast-related words,
      it should be recognized as T1CE, not T1
    """
    if not text:
        return []

    # Token patterns: order matters - more specific patterns first
    token_patterns = [
        # T1CE variants (must come before T1)
        (r'\bt1\s*(?:ce|c|gd)\b', 't1ce'),
        (r'\bt1\s*\+\s*c\b', 't1ce'),
        (r'\bce[\s\-]?t1\b', 't1ce'),
        (r'\bcontrast[\s\-]?enhanced\s*t1\b', 't1ce'),
        (r'\bt1[\s\-]?contrast[\s\-]?enhanced\b', 't1ce'),
        (r'\b(?:post[\s\-]?contrast|post[\s\-]?gadolinium|gadolinium[\s\-]?enhanced)\b', 't1ce'),
        (r'\bt1[\s\-]?(?:with|w/)?\s*(?:gad|gadolinium|contrast)\b', 't1ce'),
        # FLAIR (must come before T2)
        (r'\bt2[\s\-]?flair\b', 'flair'),
        (r'\bflair\b', 'flair'),
        # DWI
        (r'\bdwi\b', 'dwi'),
        (r'\bdiffusion[\s\-]?weighted\b', 'dwi'),
        (r'\bdiffusion\s+(?:weighted\s+)?(?:image|imaging|mri|sequence)\b', 'dwi'),
        (r'\badc\s*(?:map)?\b', 'adc'),
        # T2 variants
        (r'\bt2f\b', 'flair'),
        (r'\bt2\s*\*', 't2'),  # T2*
        (r'\bt2[\s\-]?(?:w|wi|weighted)\b', 't2'),
        (r'\bt2[\s\-]?(?:weighted[\s\-]?)?(?:image|imaging|mri|sequence|scan)\b', 't2'),
        # T1 variants (after T1CE)
        (r'\bt1n\b', 't1'),
        (r'\bt1\s+native\b', 't1'),
        (r'\bt1[\s\-]?(?:w|wi|weighted)\b', 't1'),
        (r'\bt1[\s\-]?(?:weighted[\s\-]?)?(?:image|imaging|mri|sequence|scan)\b', 't1'),
    ]

    # Patterns that indicate contrast-enhanced T1 (used to upgrade T1 -> T1CE)
    t1ce_context_patterns = [
        r'\bpost[\s\-]?contrast\b',
        r'\bcontrast[\s\-]?enhanced\b',
        r'\bgadolinium\b',
        r'\bgad\b',
        r'\benhanced\b',
        r'\bwith\s+contrast\b',
        r'\bw/\s*contrast\b',
        r'\bce\b',
        r'\bt1ce\b',
        r'\bt1c\b',
        r'\bt1\s*\+\s*c\b',
    ]

    # Strong explicit T1CE mentions used for cross-sentence upgrade.
    # Example: "The sagittal T1 MRI shows ... On this T1 post-contrast sequence ..."
    explicit_t1ce_mention_patterns = [
        r'\bt1(?:wi|w)?\s*(?:post[\s\-]?contrast|contrast[\s\-]?enhanced|with\s+contrast|w/\s*contrast|ce|gd|c)\b',
        r'\bpost[\s\-]?contrast\s+t1(?:wi|w)?\b',
        r'\bt1ce\b',
        r'\bt1gd\b',
        r'\bt1\s*\+\s*c\b',
        r'\bt1c\b',
    ]

    # Patterns that indicate descriptive usage (NOT a sequence claim)
    # These describe lesion characteristics, not the current image sequence
    descriptive_exclusion_patterns = [
        # Signal intensity descriptions
        r'\b(?:t2|t1|flair)[\s/]*(?:hyper|hypo)?(?:intense|intensity|intensities)\b',
        r'\b(?:t2|t1|flair)[\s/]*(?:bright|dark|signal)\b',
        r'\b(?:t2|t1|flair)[\s/]*(?:high|low|intermediate)\s*(?:signal)?\b',
        r'\b(?:hyper|hypo)(?:intense|intensity)\s+(?:on\s+)?(?:t2|t1|flair)\b',
        # Lesion characteristic descriptions
        r'\b(?:surrounding|perilesional|adjacent|associated)\s+(?:t2|t1|flair)\b',
        r'\b(?:t2|t1|flair)[\s/]+(?:changes?|abnormalit(?:y|ies)|findings?)\b',
        # Combined modality descriptions (e.g., "T2/FLAIR hyperintensity")
        r'\bt2\s*/\s*flair\b',
        r'\bflair\s*/\s*t2\b',
    ]

    def _get_sentence_at_match(full_text: str, match_start: int, match_end: int) -> str:
        """Extract the full sentence containing the match."""
        # Find sentence start (look for . ! ? or beginning)
        sent_start = match_start
        for i in range(match_start - 1, max(0, match_start - 200), -1):
            if full_text[i] in '.!?\n':
                sent_start = i + 1
                break
        else:
            sent_start = max(0, match_start - 200)

        # Find sentence end (look for . ! ? or end)
        sent_end = match_end
        for i in range(match_end, min(len(full_text), match_end + 200)):
            if full_text[i] in '.!?\n':
                sent_end = i + 1
                break
        else:
            sent_end = min(len(full_text), match_end + 200)

        return full_text[sent_start:sent_end]

    def _is_descriptive_usage(sentence: str) -> bool:
        """Check if the modality mention is descriptive (not a sequence claim)."""
        sentence_lower = sentence.lower()
        for pattern in descriptive_exclusion_patterns:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                return True
        return False

    def _should_upgrade_t1_to_t1ce(sentence: str) -> bool:
        """Check if T1 should be upgraded to T1CE based on sentence context."""
        sentence_lower = sentence.lower()
        for pattern in t1ce_context_patterns:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                return True
        return False

    def _has_explicit_t1ce_mention(full_text: str) -> bool:
        """
        Check whether response contains an explicit T1CE mention anywhere.
        Used to resolve cross-sentence wording where the first sentence says T1
        and a later sentence clarifies it is post-contrast.
        """
        if not full_text:
            return False
        text_lower = full_text.lower()
        for pattern in explicit_t1ce_mention_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _resolve_modality_from_phrase(phrase: str, sentence_context: str = "", full_text: str = "") -> str | None:
        """
        Resolve modality from a phrase, considering sentence context.
        Returns normalized modality or None.
        """
        phrase_lower = phrase.lower().strip()

        # First, check if this is a descriptive usage
        if sentence_context and _is_descriptive_usage(sentence_context):
            return None

        for pattern, canonical in token_patterns:
            if re.search(pattern, phrase_lower, re.IGNORECASE):
                # If we matched T1 (not T1CE), check if sentence context suggests T1CE
                if canonical == 't1':
                    if sentence_context and _should_upgrade_t1_to_t1ce(sentence_context):
                        return _normalize_expected_modality('t1ce')
                    if full_text and _has_explicit_t1ce_mention(full_text):
                        return _normalize_expected_modality('t1ce')
                return _normalize_expected_modality(canonical)
        return None

    # === Strategy 1: Explicit claim patterns (original) ===
    claim_patterns = [
        re.compile(
            r'(?:this|the|current|given)\s+(?:image|scan|sequence|mri|study)[^.\n:;]{0,80}'
            r'(?:is|appears to be|looks like|seems to be|likely|consistent with)\s+'
            r'(?:an?\s+)?'
            r'([a-z0-9\-\s/+*]+)',
            re.IGNORECASE,
        ),
        # Chinese-style explicit claims
        re.compile(
            r'(?:该|此|这个|本)\s*(?:图像|影像|序列|MRI)[^。\n:；]{0,40}(?:为|是)\s*([a-z0-9\-\s/+*]+)',
            re.IGNORECASE,
        ),
    ]

    claims = []
    for claim_pattern in claim_patterns:
        for match in claim_pattern.finditer(text):
            phrase = match.group(1).lower().strip()
            sentence = _get_sentence_at_match(text, match.start(), match.end())
            resolved = _resolve_modality_from_phrase(phrase, sentence, text)
            if resolved:
                claims.append(resolved)

    # === Strategy 2: Sentence-initial modality mentions ===
    # Patterns like "T2WI shows...", "The axial T2 MRI demonstrates...", "FLAIR sequence reveals..."
    # These are strong indicators that the model is claiming the image is of that modality
    sentence_initial_patterns = [
        # "T2WI shows...", "FLAIR demonstrates...", "DWI reveals..."
        re.compile(
            r'(?:^|[.!?]\s+)'
            r'(?:the\s+)?(?:axial\s+|coronal\s+|sagittal\s+)?'
            r'(t1ce|t1c|t1n|t1\s*\+\s*c|t1[\s\-]?gd|flair|t2f|dwi|t2\s*\*|t2wi|t1wi|t2[\s\-]?weighted|t1[\s\-]?weighted|t2|t1|diffusion[\s\-]?weighted)'
            r'(?:[\s\-]?(?:image|imaging|mri|sequence|scan|weighted))?'
            r'\s+(?:image\s+|scan\s+|sequence\s+)?'
            r'(?:shows?|demonstrates?|reveals?|indicates?|displays?|depicts?|suggests?)',
            re.IGNORECASE | re.MULTILINE,
        ),
        # "The T2 MRI shows...", "This T2-weighted image demonstrates..."
        re.compile(
            r'(?:^|[.!?]\s+)'
            r'(?:this|the|a|an)\s+(?:axial\s+|coronal\s+|sagittal\s+)?'
            r'(t1ce|t1c|t1n|t1\s*\+\s*c|t1[\s\-]?gd|flair|t2f|dwi|t2\s*\*|t2wi|t1wi|t2[\s\-]?weighted|t1[\s\-]?weighted|t2|t1|diffusion[\s\-]?weighted)'
            r'(?:[\s\-]?(?:image|imaging|mri|sequence|scan|weighted))?',
            re.IGNORECASE | re.MULTILINE,
        ),
    ]

    for pattern in sentence_initial_patterns:
        for match in pattern.finditer(text):
            phrase = match.group(1).lower().strip()
            # Get the full sentence to check for T1CE context
            sentence = _get_sentence_at_match(text, match.start(), match.end())

            # Skip if this is a descriptive usage
            if _is_descriptive_usage(sentence):
                continue

            resolved = _resolve_modality_from_phrase(phrase, sentence, text)
            if resolved:
                claims.append(resolved)

    # === Strategy 3: Direct modality token scan in first 500 chars ===
    # If model starts with "T2WI..." or mentions modality early, it's likely a claim
    # Only trigger if we haven't found claims yet and there's a clear modality mention
    if not claims:
        # Check for standalone modality mentions at very beginning (first 100 chars)
        first_sentence_match = re.match(r'^[^.!?]{0,100}', text, re.IGNORECASE)
        if first_sentence_match:
            first_sentence = first_sentence_match.group(0)

            # Skip if this looks like a descriptive usage
            if not _is_descriptive_usage(first_sentence):
                resolved = _resolve_modality_from_phrase(first_sentence, first_sentence, text)
                if resolved:
                    claims.append(resolved)

    # Deduplicate while preserving order.
    return list(dict.fromkeys(claims))


def _get_modality_signal_gate(response: str, expected_modality: str | None) -> dict:
    """
    If model explicitly claims a wrong modality, signal-related score should be zero.
    """
    expected = _normalize_expected_modality(expected_modality)
    if not expected:
        return {"gate": 1.0, "conflict": False, "claims": []}

    claims = _extract_explicit_modality_claims(response)
    if not claims:
        return {"gate": 1.0, "conflict": False, "claims": []}

    conflict = any(claim != expected for claim in claims)
    return {
        "gate": 0.0 if conflict else 1.0,
        "conflict": conflict,
        "claims": claims,
    }


def _maybe_log_judge_prompt(task_name: str, prompt: str):
    """
    Optional prompt dump for debugging.
    Set LLM_JUDGE_DUMP_PROMPT=1 to print prompts.
    Set LLM_JUDGE_DUMP_PROMPT_ONCE=0 to print every sample.
    """
    if os.environ.get("LLM_JUDGE_DUMP_PROMPT", "0") != "1":
        return

    once_only = os.environ.get("LLM_JUDGE_DUMP_PROMPT_ONCE", "1") != "0"
    if once_only and task_name in _prompt_logged:
        return

    _prompt_logged.add(task_name)
    print(f"[LLM_JUDGE_PROMPT::{task_name}]\n{prompt}\n[LLM_JUDGE_PROMPT_END::{task_name}]")


def compute_format_reward(solution_str: str) -> float:
    """
    计算严格的轮次格式奖励 (最大 0.4)

    规则:
    - Round 1 (环境反馈之前): 只能有 <think>，不能有 <rethink>/<answer>
    - Round 2 (环境反馈之后): 只能有 <rethink> + <answer>，不能有 <think>

    分数分配:
    - <think> 在 Round 1 且 Round 1 无 <rethink>/<answer>: +0.10
    - <tool_call> 在 Round 1 且 tool_call JSON 可解析且参数有效: +0.10
    - <rethink> 在 Round 2 且 Round 2 无 <think>: +0.10
    - <answer> 在 Round 2 (在 <rethink> 之后): +0.10
    """
    format_reward = 0.0
    round1_str, round2_str = _split_by_env_feedback(solution_str)

    # === Round 1 检查 ===
    r1_think_spans = _find_tag_spans(round1_str, 'think')
    r1_tool_spans = _find_tag_spans(round1_str, 'tool_call')
    r1_has_think = len(r1_think_spans) == 1
    r1_has_tool_call = len(r1_tool_spans) == 1
    r1_has_rethink = '<rethink>' in round1_str
    r1_has_answer = '<answer>' in round1_str

    # <think> 得分: 必须且只能有一个 <think>，Round 1 无 <rethink>/<answer>
    if r1_has_think and not r1_has_rethink and not r1_has_answer:
        format_reward += 0.10

    # <tool_call> 得分: 必须且只能有一个 <tool_call>，在 </think> 之后，
    # 且 JSON 可解析并满足 mark_bbox 参数约束
    think_span = _first_tag_span(round1_str, 'think')
    tool_span = _first_tag_span(round1_str, 'tool_call')
    has_valid_tool_call = _has_valid_mark_bbox_tool_call(round1_str)
    if r1_has_tool_call and think_span and tool_span and tool_span[0] >= think_span[1] and has_valid_tool_call:
        format_reward += 0.10

    # === Round 2 检查 ===
    if round2_str:
        r2_rethink_spans = _find_tag_spans(round2_str, 'rethink')
        r2_answer_spans = _find_tag_spans(round2_str, 'answer')
        r2_has_rethink = len(r2_rethink_spans) == 1
        r2_has_answer = len(r2_answer_spans) == 1
        r2_has_think = bool(_find_tag_spans(round2_str, 'think'))
        r2_has_tool_call = bool(_find_tag_spans(round2_str, 'tool_call'))

        # <rethink> 得分: 必须且只能有一个 <rethink>，Round 2 不能有 <think>/<tool_call>
        if r2_has_rethink and not r2_has_think and not r2_has_tool_call:
            format_reward += 0.10

        # <answer> 得分: 必须且只能有一个 <answer>，并且在 </rethink> 之后
        if r2_has_answer and r2_has_rethink and not r2_has_think and not r2_has_tool_call:
            rethink_end = r2_rethink_spans[0][1]
            answer_start = r2_answer_spans[0][0]
            if answer_start >= rethink_end:
                format_reward += 0.10

    return format_reward


def compute_iou(box1, box2):
    """计算两个 bbox 的 IoU"""
    if not box1 or not box2:
        return 0.0
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _bbox_area(box):
    """计算 bbox 面积，非法框返回 0"""
    if not box or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _bbox_center(box):
    """计算 bbox 中心点，非法框返回 None"""
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _compute_oversize_penalty(pred_bbox, gt_bbox):
    """
    过大框惩罚（仅惩罚 pred_area / gt_area > 2 的情况）
    返回: (penalty, area_ratio)
    """
    pred_area = _bbox_area(pred_bbox)
    gt_area = _bbox_area(gt_bbox)
    if pred_area <= 0.0 or gt_area <= 0.0:
        return 0.0, None

    area_ratio = pred_area / gt_area
    if area_ratio <= 2.0:
        return 0.0, area_ratio

    # 比例每增加 1.0，额外扣 0.04，最大扣 0.2
    penalty = min(0.2, (area_ratio - 2.0) * 0.04)
    return penalty, area_ratio


def _compute_center_penalty(pred_bbox, gt_bbox):
    """
    中心偏移惩罚：用 GT 对角线归一化后惩罚偏移
    返回: (penalty, normalized_center_distance)
    """
    pred_center = _bbox_center(pred_bbox)
    gt_center = _bbox_center(gt_bbox)
    if pred_center is None or gt_center is None:
        return 0.0, None

    gt_w = max(0.0, float(gt_bbox[2] - gt_bbox[0]))
    gt_h = max(0.0, float(gt_bbox[3] - gt_bbox[1]))
    gt_diag = math.hypot(gt_w, gt_h)
    if gt_diag <= 1e-6:
        return 0.0, None

    center_dist = math.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1])
    normalized = center_dist / gt_diag
    if normalized <= 0.35:
        return 0.0, normalized

    # 偏移超过 0.35*gt_diag 后开始惩罚，最大扣 0.2
    penalty = min(0.2, (normalized - 0.35) * 0.25)
    return penalty, normalized


def extract_bbox(text):
    """从模型输出中提取 bbox"""
    pattern = r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(pattern, text)
    if match:
        return [int(match.group(i)) for i in range(1, 5)]

    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(pattern, text)
    if matches:
        return [int(x) for x in matches[-1]]
    return None


def keyword_match_score(response: str, reference: dict) -> float:
    """关键词匹配评分 (仅用于 open_diagnosis)，最大 0.3 — 不考虑良恶性"""
    score = 0.0
    response_lower = response.lower()

    # 诊断匹配 (0.3) — 不再检查良恶性
    diagnosis = reference.get('diagnosis', '').lower()
    if diagnosis:
        diagnosis_keywords = [kw for kw in diagnosis.split() if len(kw) > 3]
        if diagnosis_keywords:
            matched = sum(1 for kw in diagnosis_keywords if kw in response_lower)
            if matched >= len(diagnosis_keywords) * 0.5:
                score += 0.3

    return score


def check_negative_sample_response(response: str, has_anomaly: bool) -> dict:
    """
    检查负样本（无异常）的响应质量

    改进：
    1. 添加否定语规则，避免误伤 "no lesion" 这类正确表达
    2. 只在 <answer> 部分做负样本惩罚，允许在 <think> 阶段探索

    Returns:
        dict: {
            'correctly_identified': bool,  # 是否正确识别为无异常
            'fabricated_lesion': bool,     # 是否在 answer 中编造了病灶
            'penalty': float               # 惩罚系数 (0.0-1.0)
        }
    """
    response_lower = response.lower()

    # 提取 <answer> 部分用于负样本判断（允许 think/rethink 中探索）
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    answer_text = answer_match.group(1).lower() if answer_match else response_lower

    # 否定语前缀（用于过滤误伤）
    negation_prefixes = [
        'no ', 'no\n', 'without ', 'absent ', 'negative for ', 'lack of ',
        'not ', 'nor ', 'neither ', 'free of ', 'ruled out '
    ]

    # 病灶关键词（只在 answer 中检查，且排除否定语）
    lesion_keywords = [
        'tumor', 'mass', 'lesion', 'glioma', 'glioblastoma', 'meningioma',
        'metastasis', 'infarct', 'stroke', 'hemorrhage',
        'malignant', 'cancer', 'neoplasm'
    ]

    # 声称有异常的关键词（用于检测模型是否错误地声称有异常）
    claims_abnormality_keywords = [
        'yes', 'abnormal', 'abnormality', 'detected', 'found', 'identified',
        'present', 'observed', 'visible', 'evident', 'shows', 'reveals',
        'demonstrates', 'indicates', 'suggests', 'consistent with',
        'suspicious', 'concerning', 'pathological', 'pathology'
    ]

    # 检查是否在 answer 中有 bbox（声称检测到异常的最强信号）
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    has_bbox_in_answer = bool(re.search(bbox_pattern, answer_text))

    # 检查 answer 是否以 "Yes" 开头（明确声称有异常）
    answer_starts_with_yes = answer_text.strip().lower().startswith('yes')

    # 如果 answer 中包含 bbox 或明确以 Yes 开头，直接判定为"声称有异常"
    # 这种情况下不应该被认为是 "correctly_identified"
    definite_claims_abnormality = has_bbox_in_answer or answer_starts_with_yes

    # 检查是否正确识别无异常（只在 answer 中检查，且使用更严格的关键词）
    # 移除 "no significant" 这类过宽的关键词
    strict_no_abnormality_keywords = [
        'no abnormality', 'no abnormal', 'normal brain', 'unremarkable',
        'no focal', 'no lesion', 'no mass', 'no tumor', 'no evidence of',
        'within normal limits', 'negative for',
        'appears normal', 'no pathology', 'no findings'
    ]
    # 只在 answer 中检查是否正确识别无异常
    correctly_identified = any(kw in answer_text for kw in strict_no_abnormality_keywords)

    # 如果 answer 中明确声称有异常（有 bbox 或以 Yes 开头），则不能算作 correctly_identified
    if definite_claims_abnormality:
        correctly_identified = False

    # 检查是否在 answer 中编造病灶（排除否定语）
    fabricated_lesion = False
    for kw in lesion_keywords:
        if kw in answer_text:
            # 检查是否被否定语修饰
            is_negated = False
            kw_pos = answer_text.find(kw)
            if kw_pos > 0:
                # 检查关键词前面是否有否定语
                prefix_text = answer_text[max(0, kw_pos - 20):kw_pos]
                for neg in negation_prefixes:
                    if neg in prefix_text:
                        is_negated = True
                        break
            if not is_negated:
                fabricated_lesion = True
                break

    # 检查是否在 answer 中声称有异常（即使没有使用病灶关键词）
    claims_abnormality = definite_claims_abnormality  # 已经检测过 bbox 和 Yes
    if not claims_abnormality:
        # 检查是否使用了声称有异常的关键词
        for kw in claims_abnormality_keywords:
            if kw in answer_text:
                # 检查是否被否定语修饰
                is_negated = False
                kw_pos = answer_text.find(kw)
                if kw_pos > 0:
                    prefix_text = answer_text[max(0, kw_pos - 20):kw_pos]
                    for neg in negation_prefixes:
                        if neg in prefix_text:
                            is_negated = True
                            break
                if not is_negated:
                    claims_abnormality = True
                    break

    # 计算惩罚（只对负样本）
    penalty = 0.0
    if not has_anomaly:  # 负样本
        # 最高优先级：如果 answer 中有 bbox 或以 Yes 开头，直接完全惩罚
        if definite_claims_abnormality:
            penalty = 1.0  # 完全惩罚：在负样本上给出 bbox 或声称 Yes
        elif fabricated_lesion and not correctly_identified:
            penalty = 1.0  # 完全惩罚：在 answer 中编造病灶且没有说正常
        elif fabricated_lesion and correctly_identified:
            penalty = 0.3  # 轻微惩罚：虽然说了正常但 answer 中也提到了病灶（可能是鉴别诊断）
        elif claims_abnormality and not correctly_identified:
            penalty = 1.0  # 完全惩罚：声称有异常（如给出 bbox 或说 "yes"）但没有说正常

    return {
        'correctly_identified': correctly_identified,
        'fabricated_lesion': fabricated_lesion,
        'penalty': penalty
    }


def extract_model_reasoning(response: str) -> str:
    """
    提取模型的推理和答案部分，移除工具调用和工具返回结果

    保留:
    - <think>...</think> (第一轮思考)
    - <rethink>...</rethink> (第二轮重新思考)
    - <answer>...</answer>

    移除:
    - <tool_call>...</tool_call>
    - 工具返回的结果
    """
    # 提取所有 <think> 标签内容 (第一轮)
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, response, re.DOTALL)

    # 提取所有 <rethink> 标签内容 (第二轮)
    rethink_pattern = r'<rethink>(.*?)</rethink>'
    rethink_matches = re.findall(rethink_pattern, response, re.DOTALL)

    # 提取 <answer> 标签内容
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, response, re.DOTALL)

    # 重新组合
    filtered_content = []
    for think in think_matches:
        filtered_content.append(f"<think>\n{think.strip()}\n</think>")
    for rethink in rethink_matches:
        filtered_content.append(f"<rethink>\n{rethink.strip()}\n</rethink>")

    if answer_match:
        filtered_content.append(f"<answer>\n{answer_match.group(1).strip()}\n</answer>")

    return "\n\n".join(filtered_content)


def llm_judge_diagnosis(
    response: str,
    reference: dict,
    has_anomaly: bool,
    expected_modality: str | None = None,
) -> float:
    """
    使用 GPT-4o 评估开放式诊断质量 (open_diagnosis)

    评分维度 (每项 0-1):
    - diagnosis (50%): 诊断名称是否正确（不考虑良恶性）
    - reasoning (20%): 推理逻辑
    - signal (15%): 信号特征描述
    - anatomy (15%): 解剖定位

    如果 API 不可用，使用基于规则的备用方案

    负样本处理:
    - 正确识别无异常 → 中高分
    - 无依据编造病灶 → 分数归零或降分
    """
    if not use_local_judge_backend() and get_gpt_client() is None:
        return _rule_based_diagnosis_judge(response, reference, has_anomaly)

    # 提取模型推理部分，移除工具调用和工具返回结果
    filtered_response = extract_model_reasoning(response)

    # 负样本检查
    neg_check = check_negative_sample_response(response, has_anomaly)
    if not has_anomaly and neg_check['penalty'] >= 1.0:
        # 负样本编造病灶，直接返回 0
        return 0.0

    prompt = f"""You are evaluating a medical imaging AI's diagnosis against a reference.

**Note**: The model response below contains only the model's reasoning (<think>/<rethink> tags) and final answer (<answer> tag). Tool calls and tool responses have been filtered out.
**Expected MRI Sequence**: {expected_modality or 'N/A'}

**Reference Diagnosis:**
- Diagnosis: {reference.get('diagnosis', 'N/A')}
- Anatomy: {reference.get('anatomy', 'N/A')}
- Signal: {reference.get('signal', 'N/A')}
- Reasoning: {reference.get('reasoning', 'N/A')}

**Model Response:**
{filtered_response[:4000]}

**Scoring Instructions:**
Score each dimension from 0.0 to 1.0:
1. diagnosis: Is the final diagnosis correct? Focus on whether the diagnosis name matches the reference. Do NOT consider malignant/benign nature.
2. reasoning: Is the diagnostic reasoning logical and evidence-based?
3. signal: Are MRI signal characteristics (T1/T2/FLAIR/DWI) accurately described and consistent with the expected sequence/reference?
   - If the model explicitly claims a sequence that conflicts with **Expected MRI Sequence**, set signal=0.0.
4. anatomy: Does the model correctly identify the lesion location?

Output ONLY a JSON object with scores:
{{"diagnosis": 0.0-1.0, "reasoning": 0.0-1.0, "signal": 0.0-1.0, "anatomy": 0.0-1.0}}
"""
    _maybe_log_judge_prompt("diagnosis", prompt)

    try:
        result = _query_judge_model(prompt, max_tokens=500)
        scores = _clean_and_parse_score_json(result)
        if scores:
            modality_gate = _get_modality_signal_gate(filtered_response, expected_modality)
            signal_score = scores.get('signal', 0.0) * modality_gate['gate']

            # 加权计算：诊断 50%，推理 20%，信号 15%，解剖 15%
            weighted_score = (
                scores.get('diagnosis', 0.0) * 0.50 +
                scores.get('reasoning', 0.0) * 0.20 +
                signal_score * 0.15 +
                scores.get('anatomy', 0.0) * 0.15
            )
            # 应用负样本惩罚
            weighted_score *= (1.0 - neg_check['penalty'])
            return min(1.0, weighted_score)
        # 解析失败时回退，避免空content导致固定0分
        return _rule_based_diagnosis_judge(response, reference, has_anomaly)
    except Exception as e:
        print(f"[WARNING] LLM judge error: {e}")
        reset_gpt_client_on_error()
        return _rule_based_diagnosis_judge(response, reference, has_anomaly)


def llm_judge_detection(
    response: str,
    reference: dict,
    has_anomaly: bool,
    expected_modality: str | None = None,
) -> float:
    """
    使用 GPT-4o 评估开放式检测质量 (detection)

    评分维度 (每项 0-1):
    - abnormality (35%): 是否正确判断有/无异常
    - localization (25%): 解剖位置是否匹配 (仅正样本)
    - signal (20%): 信号特征与序列是否一致
    - reasoning (20%): 是否提供影像证据

    负样本特殊处理:
    - 明确 "no abnormality" → abnormality=1
    - 编造病灶 → abnormality=0, reasoning=0
    """
    if not use_local_judge_backend() and get_gpt_client() is None:
        return _rule_based_detection_judge(response, reference, has_anomaly, expected_modality)

    filtered_response = extract_model_reasoning(response)

    # 负样本检查
    neg_check = check_negative_sample_response(response, has_anomaly)
    if not has_anomaly and neg_check['penalty'] >= 1.0:
        # 负样本编造病灶，直接返回 0
        return 0.0

    prompt = f"""You are evaluating a medical imaging AI's abnormality detection.

**Task**: The model should detect whether there is any abnormality in the brain MRI.

**Ground Truth:**
- Has Abnormality: {has_anomaly}
- Expected MRI Sequence: {expected_modality or 'N/A'}
- Reference Anatomy: {reference.get('anatomy', 'N/A')}
- Reference Signal: {reference.get('signal', 'N/A')}
- Reference Diagnosis: {reference.get('diagnosis', 'N/A')}

**Model Response:**
{filtered_response[:4000]}

**Scoring Instructions:**
Score each dimension from 0.0 to 1.0:

1. abnormality: Did the model correctly identify whether there is an abnormality?
   - If ground truth has_anomaly=True: Score 1.0 if model says "yes/abnormal", 0.0 if "no/normal"
   - If ground truth has_anomaly=False: Score 1.0 if model says "no abnormality/normal/unremarkable", 0.0 if model describes a lesion

2. localization: Does the model correctly describe the anatomical location?
   - If has_anomaly=False, score 0.0 (no localization needed)
   - If has_anomaly=True, compare with reference anatomy

3. signal: Are the signal/sequence statements accurate and modality-consistent?
   - Compare signal characteristics with reference signal and expected sequence.
   - If the model explicitly claims a wrong sequence (e.g., says T2 while expected is T1CE), score 0.0.

4. reasoning: Does the model provide imaging evidence (signal characteristics, morphology, mass effect)?
   - If has_anomaly=False and model correctly says normal, score based on justification quality
   - If model fabricates lesions for a normal case, score 0.0

Output ONLY a JSON object:
{{"abnormality": 0.0-1.0, "localization": 0.0-1.0, "signal": 0.0-1.0, "reasoning": 0.0-1.0}}
"""
    _maybe_log_judge_prompt("detection", prompt)

    try:
        result = _query_judge_model(prompt, max_tokens=500)
        scores = _clean_and_parse_score_json(result)
        if scores:
            modality_gate = _get_modality_signal_gate(filtered_response, expected_modality)
            signal_score = scores.get('signal', 0.0) * modality_gate['gate']

            # 加权计算：异常判断 35%，定位 25%，信号 20%，推理 20%
            weighted_score = (
                scores.get('abnormality', 0.0) * 0.35 +
                scores.get('localization', 0.0) * 0.25 +
                signal_score * 0.20 +
                scores.get('reasoning', 0.0) * 0.20
            )
            # 应用负样本惩罚
            weighted_score *= (1.0 - neg_check['penalty'])
            return min(1.0, weighted_score)
        return _rule_based_detection_judge(response, reference, has_anomaly, expected_modality)
    except Exception as e:
        print(f"[WARNING] LLM judge detection error: {e}")
        reset_gpt_client_on_error()
        return _rule_based_detection_judge(response, reference, has_anomaly, expected_modality)


def llm_judge_caption(
    response: str,
    reference: dict,
    has_anomaly: bool,
    expected_modality: str | None = None,
) -> float:
    """
    使用 GPT-4o 评估开放式描述质量 (caption/description)

    评分维度 (每项 0-1):
    - anatomy (25%): 位置描述
    - signal (25%): 信号特征
    - morphology (25%): 形态/占位效应
    - coherence (25%): 前后自洽

    负样本处理:
    - 合理描述正常结构 → 中高分
    - 无依据编造病灶 → 降分
    """
    if not use_local_judge_backend() and get_gpt_client() is None:
        return _rule_based_caption_judge(response, reference, has_anomaly, expected_modality)

    filtered_response = extract_model_reasoning(response)

    # 负样本检查
    neg_check = check_negative_sample_response(response, has_anomaly)
    if not has_anomaly and neg_check['penalty'] >= 1.0:
        return 0.0

    prompt = f"""You are evaluating a medical imaging AI's description of a brain MRI.

**Task**: The model should provide a structured description of the imaging characteristics.

**Ground Truth:**
- Has Abnormality: {has_anomaly}
- Expected MRI Sequence: {expected_modality or 'N/A'}
- Reference Anatomy: {reference.get('anatomy', 'N/A')}
- Reference Signal: {reference.get('signal', 'N/A')}
- Reference Diagnosis: {reference.get('diagnosis', 'N/A')}

**Model Response:**
{filtered_response[:4000]}

**Scoring Instructions:**
Score each dimension from 0.0 to 1.0:

1. anatomy: Does the model accurately describe the anatomical location?
   - For normal cases: describing normal brain structures is acceptable
   - For abnormal cases: should match reference anatomy

2. signal: Are MRI signal characteristics accurately described?
   - T1/T2/FLAIR signal intensity
   - Enhancement patterns (if applicable)
   - If the model explicitly claims a sequence conflicting with **Expected MRI Sequence**, set signal=0.0.

3. morphology: Are morphological features and mass effect described?
   - Size, shape, borders
   - Edema, midline shift, compression

4. coherence: Is the description internally consistent and logical?
   - No contradictions
   - Findings support conclusions

Output ONLY a JSON object:
{{"anatomy": 0.0-1.0, "signal": 0.0-1.0, "morphology": 0.0-1.0, "coherence": 0.0-1.0}}
"""
    _maybe_log_judge_prompt("caption", prompt)

    try:
        result = _query_judge_model(prompt, max_tokens=500)
        scores = _clean_and_parse_score_json(result)
        if scores:
            modality_gate = _get_modality_signal_gate(filtered_response, expected_modality)
            signal_score = scores.get('signal', 0.0) * modality_gate['gate']

            # 等权重计算
            weighted_score = (
                scores.get('anatomy', 0.0) * 0.25 +
                signal_score * 0.25 +
                scores.get('morphology', 0.0) * 0.25 +
                scores.get('coherence', 0.0) * 0.25
            )
            # 应用负样本惩罚
            weighted_score *= (1.0 - neg_check['penalty'])
            return min(1.0, weighted_score)
        return _rule_based_caption_judge(response, reference, has_anomaly, expected_modality)
    except Exception as e:
        print(f"[WARNING] LLM judge caption error: {e}")
        reset_gpt_client_on_error()
        return _rule_based_caption_judge(response, reference, has_anomaly, expected_modality)


def _rule_based_diagnosis_judge(response: str, reference: dict, has_anomaly: bool) -> float:
    """
    基于规则的备用诊断评分方案（当 LLM Judge 不可用时）

    负样本处理:
    - 正确识别无异常 → 高分
    - 编造病灶 → 分数归零或降分
    """
    score = 0.0
    response_lower = response.lower()

    # 负样本检查
    neg_check = check_negative_sample_response(response, has_anomaly)

    if has_anomaly:
        # 正样本：诊断匹配 (1.0) — 不考虑良恶性
        diagnosis = reference.get('diagnosis', '').lower()
        if diagnosis:
            diagnosis_terms = [t for t in diagnosis.split() if len(t) > 3]
            if diagnosis_terms:
                matched = sum(1 for t in diagnosis_terms if t in response_lower)
                match_ratio = matched / len(diagnosis_terms)
                if match_ratio >= 0.5:
                    score += 1.0
                elif match_ratio >= 0.3:
                    score += 0.6
    else:
        # 负样本
        if neg_check['correctly_identified']:
            score += 0.8  # 正确识别无异常
        if not neg_check['fabricated_lesion']:
            score += 0.2  # 没有编造病灶

    # 应用惩罚
    score *= (1.0 - neg_check['penalty'])
    return min(1.0, score)


def _rule_based_detection_judge(
    response: str,
    reference: dict,
    has_anomaly: bool,
    expected_modality: str | None = None,
) -> float:
    """
    基于规则的备用检测评分方案
    """
    score = 0.0
    response_lower = response.lower()

    # 负样本检查
    neg_check = check_negative_sample_response(response, has_anomaly)
    modality_gate = _get_modality_signal_gate(response, expected_modality)

    if has_anomaly:
        # 正样本：检查是否识别到异常
        abnormal_keywords = ['abnormal', 'lesion', 'mass', 'tumor', 'finding']
        if any(kw in response_lower for kw in abnormal_keywords):
            score += 0.35  # abnormality

        # 检查解剖位置
        anatomy = reference.get('anatomy', '').lower()
        if anatomy:
            anatomy_terms = [t for t in anatomy.split() if len(t) > 3]
            if anatomy_terms:
                matched = sum(1 for t in anatomy_terms if t in response_lower)
                if matched >= len(anatomy_terms) * 0.3:
                    score += 0.25  # localization

        # 检查信号与推理证据
        evidence_keywords = ['signal', 'intensity', 'enhancement', 'edema', 'hyperintense', 'hypointense']
        has_signal_evidence = any(kw in response_lower for kw in evidence_keywords)
        if has_signal_evidence:
            score += 0.20 * modality_gate['gate']  # signal
            score += 0.20  # reasoning
    else:
        # 负样本
        if neg_check['correctly_identified']:
            score += 0.8  # 正确识别无异常
        if not neg_check['fabricated_lesion']:
            score += 0.2  # 没有编造病灶

    # 应用惩罚
    score *= (1.0 - neg_check['penalty'])
    return min(1.0, score)


def _rule_based_caption_judge(
    response: str,
    reference: dict,
    has_anomaly: bool,
    expected_modality: str | None = None,
) -> float:
    """
    基于规则的备用描述评分方案
    """
    score = 0.0
    response_lower = response.lower()

    # 负样本检查
    neg_check = check_negative_sample_response(response, has_anomaly)
    modality_gate = _get_modality_signal_gate(response, expected_modality)

    # 检查解剖描述
    anatomy_keywords = ['frontal', 'temporal', 'parietal', 'occipital', 'lobe', 'hemisphere', 'cortex', 'white matter']
    if any(kw in response_lower for kw in anatomy_keywords):
        score += 0.25

    # 检查信号描述
    signal_keywords = ['t1', 't2', 'flair', 'dwi', 'hyperintense', 'hypointense', 'isointense', 'signal']
    if any(kw in response_lower for kw in signal_keywords):
        score += 0.25 * modality_gate['gate']

    # 检查形态描述
    morphology_keywords = ['size', 'shape', 'border', 'margin', 'irregular', 'well-defined', 'mass effect', 'edema']
    if any(kw in response_lower for kw in morphology_keywords):
        score += 0.25

    # 基础连贯性分数
    if len(response) > 100:  # 有一定长度的描述
        score += 0.25

    # 应用惩罚
    score *= (1.0 - neg_check['penalty'])
    return min(1.0, score)


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    统一开放式 Reward 函数

    支持三种 question_type:
    - detection: 开放式异常检测
    - caption/description: 开放式影像描述
    - open_diagnosis: 开放式诊断

    Args:
        solution_str: 模型完整输出
        ground_truth: dict 或 JSON string
        extra_info: dict，额外信息

    Returns:
        float or dict: 0.0 - 1.9, or dict with breakdown if extra_info contains 'return_dict'
    """
    localization_reward = 0.0
    keyword_reward = 0.0
    llm_judge_reward = 0.0
    format_reward = 0.0

    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            gt = {}
    else:
        gt = ground_truth or {}

    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = {}
    elif not isinstance(extra_info, dict):
        extra_info = {}

    # 获取问题类型
    question_type = gt.get('question_type', 'open_diagnosis')
    has_anomaly = gt.get('has_anomaly', True)
    gt_bbox = gt.get('bbox')

    # Track IoU and bbox for debugging
    iou = 0.0
    pred_bbox = None
    area_ratio = None
    center_distance_norm = None
    oversize_penalty = 0.0
    center_penalty = 0.0

    # 负样本检查
    neg_check = check_negative_sample_response(solution_str, has_anomaly)

    # ==================== Ablation Control ====================
    # REWARD_ABLATION 环境变量控制消融实验:
    # - "full" 或未设置: 完整 reward (format + iou + llm_judge)
    # - "format_iou": 只有 format + iou，禁用 llm_judge
    # - "format_judge": 只有 format + llm_judge，禁用 iou
    # - "iou_judge": 只有 iou + llm_judge，禁用 format
    ablation_mode = os.environ.get("REWARD_ABLATION", "full").lower().strip()
    disable_llm_judge = ablation_mode == "format_iou"
    disable_iou = ablation_mode == "format_judge"
    disable_format = ablation_mode == "iou_judge"

    # ==================== 1. Localization reward (0.5) ====================
    if disable_iou:
        # 消融实验：禁用 IoU reward
        localization_reward = 0.0
        iou = -2.0  # Mark as ablation disabled
    elif has_anomaly and gt_bbox:
        # 正样本：计算 IoU
        pred_bbox = extract_bbox(solution_str)
        iou = compute_iou(pred_bbox, gt_bbox)

        # Continuous IoU reward with linear interpolation
        if iou >= 0.5:
            localization_reward = 0.5
        elif iou >= 0.3:
            localization_reward = 0.35 + (iou - 0.3) / 0.2 * 0.15
        elif iou >= 0.1:
            localization_reward = 0.2 + (iou - 0.1) / 0.2 * 0.15
        else:
            localization_reward = 0.0

        # 额外几何惩罚：抑制“大框覆盖”和中心偏移
        oversize_penalty, area_ratio = _compute_oversize_penalty(pred_bbox, gt_bbox)
        center_penalty, center_distance_norm = _compute_center_penalty(pred_bbox, gt_bbox)
        localization_reward = max(0.0, localization_reward - oversize_penalty - center_penalty)

        if localization_reward == 0.0 and extra_info and extra_info.get('return_dict', False):
            case_id = extra_info.get('case_id', 'unknown')
            dataset = extra_info.get('dataset', 'unknown')
            modality = extra_info.get('modality', 'unknown')
            view = extra_info.get('view', 'unknown')
            print(
                f"[localization=0] case_id={case_id}, dataset={dataset}, modality={modality}, view={view}, "
                f"q_type={question_type}, iou={iou:.4f}, pred_bbox={pred_bbox}, gt_bbox={gt_bbox}, "
                f"area_ratio={area_ratio}, center_dist_norm={center_distance_norm}, "
                f"oversize_penalty={oversize_penalty:.3f}, center_penalty={center_penalty:.3f}"
            )
    elif not has_anomaly:
        # 负样本：不计算 IoU
        iou = -1.0  # Mark as no anomaly case
        if neg_check['correctly_identified'] and not neg_check['fabricated_lesion']:
            # 正确识别为无异常且没有编造病灶
            localization_reward = 0.5
        elif neg_check['correctly_identified']:
            # 说了无异常但也提到了病灶
            localization_reward = 0.25

    # ==================== 2. LLM Judge reward (1.0) ====================
    reference = gt.get('reference_diagnosis', {})
    if isinstance(reference, str):
        try:
            reference = json.loads(reference)
        except json.JSONDecodeError:
            reference = {}

    expected_modality = _extract_expected_modality(gt, reference, extra_info)

    if disable_llm_judge:
        # 消融实验：禁用 LLM Judge reward
        llm_judge_reward = 0.0
    elif reference:
        # 根据 question_type 选择不同的评分函数
        if question_type == 'detection':
            llm_judge_reward = llm_judge_detection(solution_str, reference, has_anomaly, expected_modality)
        elif question_type in ['caption', 'description']:
            llm_judge_reward = llm_judge_caption(solution_str, reference, has_anomaly, expected_modality)
        else:  # open_diagnosis 或其他
            llm_judge_reward = llm_judge_diagnosis(solution_str, reference, has_anomaly, expected_modality)

    # 负样本惩罚：如果编造病灶，LLM Judge 分数归零
    if not has_anomaly and neg_check['penalty'] >= 1.0:
        llm_judge_reward = 0.0

    # ==================== 3. Format reward (0.4) ====================
    # 使用严格的轮次格式检查
    if disable_format:
        # 消融实验：禁用 format reward
        format_reward = 0.0
    else:
        format_reward = compute_format_reward(solution_str)

    # Keyword reward 已禁用，避免对 open_diagnosis 施加额外词面偏置
    keyword_reward = 0.0

    content_reward = localization_reward + llm_judge_reward

    # ==================== 4. 总分计算 + Cap ====================
    total_reward = content_reward + format_reward

    # 总分 Cap：IoU < 0.1 时限制总分上限（更强约束错误定位）
    # 注意：此 cap 只在完整模式 (full) 下生效，消融实验不应用
    # - format_iou 模式：没有 llm_judge，cap 无意义
    # - format_judge 模式：没有 iou，cap 不适用
    if ablation_mode == "full" and has_anomaly and gt_bbox and iou < 0.1:
        total_reward = min(total_reward, 0.4)

    # Return dict if requested
    if extra_info and extra_info.get('return_dict', False):
        return {
            'score': total_reward,
            'localization': localization_reward,
            'iou': iou,
            'keyword': keyword_reward,
            'llm_judge': llm_judge_reward,
            'format': format_reward,
            'question_type': question_type,
            'has_anomaly': has_anomaly,
            'expected_modality': expected_modality or "unknown",
            'ablation_mode': ablation_mode,
        }

    return total_reward
