# Data Preparation

BrReMark uses brain MRI datasets for training. Due to licensing restrictions, we do not distribute the raw data. Below is how to prepare the training data.

## Data Sources

We use 3,696 cases from 6 abnormal datasets and 1 normal dataset across 5 MRI modalities (T1, T1-Gd, T2, FLAIR, DWI).

| Dataset | Modalities | Cases | Proportion |
|---------|-----------|-------|------------|
| BraTS-GLI | T1, T1-Gd, T2, FLAIR | 898 | 24.3% |
| BraTS-MEN | T1, T1-Gd, T2, FLAIR | 723 | 19.6% |
| BraTS-PED | T1, T1-Gd, T2, FLAIR | 99 | 2.7% |
| ATLAS | T1 | 655 | 17.7% |
| IXI (normal) | T1, T2 | 577 | 15.6% |
| UPENN-GBM | T1, T1-Gd, T2, FLAIR | 497 | 13.4% |
| ISLES | DWI | 247 | 6.7% |

## Image Preprocessing

- All images are 2D axial/coronal/sagittal slices extracted from 3D volumes
- Resolution: 480×480 pixels
- Modalities: T1, T1-Gd (T1CE), T2, FLAIR, DWI
- Views: axial, coronal, sagittal

## Parquet Schema

### SFT Data

Each row is a complete two-turn reasoning trajectory:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | System prompt + user question with `<image>` placeholder |
| `response` | str | Full trajectory: `<think>...<tool_call>...</tool_call>...<rethink>...<answer>...</answer>` |
| `image` | bytes | PIL Image (base64 encoded) |
| `extra_info` | str (JSON) | Metadata including `question_type`, `has_anomaly`, `bbox`, `data_source` |

### RL Data

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | System prompt + user question |
| `image` | bytes | PIL Image serialized as base64 |
| `extra_info` | str (JSON) | Metadata including `question_type`, `has_anomaly`, `bbox`, `reference_diagnosis`, `data_source` |
| `reward_model` | str (JSON) | Ground truth for reward computation |

The `data_source` field in `extra_info` maps to the reward function in `verl/utils/reward_score/`:
- `"brain_mri_diagnosis"` → `brain_mri_diagnosis.compute_score()`
- `"synthetic_brain_mri"` → `synthetic_brain_mri.compute_score()`

## Synthetic Data

Synthetic abnormal MRI volumes are generated using [SynthSeg](https://github.com/BBillot/SynthSeg) with EDT-weighted pathology injection. The pipeline injects real lesion masks into healthy brain label maps, then renders domain-randomized MRI volumes with pathology-specific intensity distributions. Synthetic data is used only in the RL stage (`scripts/train_rl.sh --phase 2`).
