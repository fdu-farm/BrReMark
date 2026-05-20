"""
LLM-as-Judge evaluation prompts for BrReMark benchmark.

Task 2 (Image Description): Extracts and standardizes medical keywords from
    GT and predicted reports, then computes Clinical F1, Modality F1, and
    Binary (normal/abnormal) classification.

Task 3 (Differential Diagnosis): Multi-dimensional scoring across
    Diagnostic Accuracy, Reasoning Quality, and Safety.
"""

# =============================================================================
# Task 2: Image Description — Keyword Extraction & Standardization
# =============================================================================

TASK2_JUDGE_PROMPT = """You are given two radiology reports: Ground Truth (GT) and Predicted (Pred). Your task is to extract and standardize medically important keywords from both reports.

Task: Extract keywords related to the following categories:
- Anatomical structures: e.g., brain regions, body parts.
- Imaging characteristics: e.g., hyperintensity, low density, enhancement, mass-like, signal changes.
- Disease or pathological findings: e.g., leukoencephalopathy, infarct, tumor.
- Negated findings: any finding explicitly stated as absent or negative, such as "no hemorrhage", "no mass" — keep the negation in the keyword.
- Imaging sequence and plane: e.g., T1, T2, FLAIR, DWI, sagittal, axial, coronal.

Standardization Rules:
- Normalize synonymous or semantically similar expressions into a single canonical form.
- Normalize anatomical mentions related to disease into their broader anatomical structures when appropriate.
- Ensure that after normalization, all terms that refer to the same concept are exactly string-equal, to support direct set-based comparison (e.g., for intersection/union using string matching).
- Prefer higher-level or broader terms when multiple expressions refer to variations of the same anatomical area.
- The goal is to eliminate variation in expression and granularity, so that conceptually equivalent phrases normalize to the same string.

Consistency:
- GT and Pred are labeled as "normal" or "abnormal" based on their findings.
- Is_Consistent is true if both GT and Pred are either "normal" or both "abnormal".
- Is_Consistent is false if one is "normal" and the other is "abnormal".

Input:
GT = "{gt_caption}"
Pred = "{pred_caption}"

Output Format (JSON):
{{
  "Raw_Keywords": {{
    "GT": ["keyword1", "keyword2", "..."],
    "Pred": ["keyword1", "keyword2", "..."]
  }},
  "Standardized_Keywords": {{
    "GT": ["standardized_keyword1", "standardized_keyword2", "..."],
    "Pred": ["standardized_keyword1", "standardized_keyword2", "..."]
  }},
  "Consistency": {{
    "GT": "normal" | "abnormal",
    "Pred": "normal" | "abnormal",
    "Is_Consistent": true | false
  }}
}}

Only return valid JSON with no extra text."""


# =============================================================================
# Task 3: Differential Diagnosis — Multi-Dimensional Scoring (0-10 scale)
# =============================================================================

TASK3_JUDGE_PROMPT = """You are an expert neuroradiologist evaluating a brain MRI diagnostic report. Score the prediction across multiple dimensions by comparing it against the ground truth.

## Input
Ground Truth Diagnosis: "{gt_diagnosis}"
Ground Truth Imaging Description: "{gt_impression}"

Model Response:
\"\"\"
{model_response}
\"\"\"

## Scoring Rubric (each dimension 0-10)

### 1. DIAGNOSTIC_ACCURACY (0-10)
Evaluate the correctness of the final diagnosis compared to ground truth.
- Focus on: semantic match with GT (synonyms/variants count, e.g. GBM = Glioblastoma), disease category alignment, whether GT appears in differentials
- Higher scores for exact or near-exact matches, lower scores for unrelated diagnoses

### 2. REASONING_QUALITY (0-10)
Evaluate the logical coherence and clinical grounding of the diagnostic reasoning.
- Focus on: logical chain from imaging features to diagnosis, specific MRI signal characteristics, alignment with GT imaging description, appropriate use of radiological terminology
- Higher scores for clear, well-supported reasoning; lower scores for vague, generic, or contradictory statements

### 3. SAFETY (0-10)
Evaluate clinical safety and potential for patient harm.
- Focus on: dangerous misclassifications (calling malignant as benign/normal), ruling out life-threatening conditions that are present, stating "normal" when serious pathology exists, appropriate hedging when uncertain
- Higher scores for safe and cautious diagnoses; lower scores for confidently wrong statements that could cause harm

## Output Format (JSON only)
{{
  "DIAGNOSTIC_ACCURACY": <0-10>,
  "REASONING_QUALITY": <0-10>,
  "SAFETY": <0-10>,
  "TOTAL_SCORE": <sum of all scores, max 30>,
  "Top_1": "Correct" | "Wrong",
  "BRIEF_JUSTIFICATION": "<1-2 sentence explanation>"
}}

Only return valid JSON with no extra text."""
