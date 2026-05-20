"""
Synthetic Brain MRI Reward Function (纯定位训练)

专为 ADSP 合成数据设计：
- 只计算 IoU 定位奖励
- 跳过 LLM-judge 语义评分（合成图像的诊断是人为设定的）
- 保留格式奖励（保持输出格式一致性）

Reward 组成 (总分 0.9):
- format_reward (0.4): 轮次格式检查（复用主任务格式约束）
- localization_reward (0.5): 连续 IoU 评分
- 无 LLM-judge (0.0): 跳过语义评分
- 无 keyword_reward (0.0): 跳过关键词匹配

设计理由：
1. 合成图像的视觉外观与真实 MRI 差异大，LLM-judge 评分不可靠
2. 合成数据的目标是增强定位能力，而非语义理解
3. 通过 prompt 中的先验知识引导模型定位，不需要语义评估
"""

import re
import json

# 复用 brain_mri_diagnosis 中的基础函数
from .brain_mri_diagnosis import (
    compute_format_reward,
    compute_iou,
    extract_bbox,
    _compute_oversize_penalty,
    _compute_center_penalty,
    _bbox_area,
)


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    合成数据专用 Reward 函数 - 只计算定位奖励

    与 brain_mri_diagnosis.compute_score 的区别：
    1. 跳过 LLM-judge 语义评分
    2. 跳过 keyword_reward
    3. 总分上限 0.9（format 0.4 + localization 0.5）

    Args:
        solution_str: 模型完整输出
        ground_truth: dict 或 JSON string，包含:
            - has_anomaly: bool
            - bbox: [x1, y1, x2, y2]
            - question_type: 'synthetic_detection'
            - is_synthetic: True
            - pathology_type: str
            - expected_region: str
        extra_info: dict，额外信息

    Returns:
        float: 0.0 - 0.9
        或 dict（如果 extra_info['return_dict'] = True）
    """
    localization_reward = 0.0
    format_reward = 0.0

    # 解析 ground_truth
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            gt = {}
    else:
        gt = ground_truth or {}

    # 解析 extra_info
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = {}
    elif not isinstance(extra_info, dict):
        extra_info = {}

    # 获取 ground truth 信息
    gt_bbox = gt.get('bbox')
    has_anomaly = gt.get('has_anomaly', True)
    pathology_type = gt.get('pathology_type', 'unknown')
    expected_region = gt.get('expected_region', 'unknown')

    # Track metrics
    iou = 0.0
    pred_bbox = None
    area_ratio = None
    center_distance_norm = None
    oversize_penalty = 0.0
    center_penalty = 0.0

    # ==================== 1. Format reward (0.4) ====================
    # 保持与真实数据一致的格式要求
    format_reward = compute_format_reward(solution_str)

    # ==================== 2. Localization reward (0.5) ====================
    if has_anomaly and gt_bbox:
        # 提取预测 bbox
        pred_bbox = extract_bbox(solution_str)

        if pred_bbox:
            iou = compute_iou(pred_bbox, gt_bbox)

            # Continuous IoU reward with linear interpolation
            # 与 brain_mri_diagnosis 保持一致的 IoU 奖励曲线
            if iou >= 0.5:
                localization_reward = 0.5
            elif iou >= 0.3:
                # 线性插值: 0.3 -> 0.35, 0.5 -> 0.5
                localization_reward = 0.35 + (iou - 0.3) / 0.2 * 0.15
            elif iou >= 0.1:
                # 线性插值: 0.1 -> 0.2, 0.3 -> 0.35
                localization_reward = 0.2 + (iou - 0.1) / 0.2 * 0.15
            else:
                localization_reward = 0.0

            # 几何惩罚：抑制"大框覆盖"和中心偏移
            oversize_penalty, area_ratio = _compute_oversize_penalty(pred_bbox, gt_bbox)
            center_penalty, center_distance_norm = _compute_center_penalty(pred_bbox, gt_bbox)
            localization_reward = max(0.0, localization_reward - oversize_penalty - center_penalty)
        else:
            # 没有预测 bbox
            iou = 0.0
            localization_reward = 0.0

    # ==================== 3. 总分计算 ====================
    total_reward = format_reward + localization_reward

    # IoU < 0.1 时限制总分（与 brain_mri_diagnosis 保持一致）
    if has_anomaly and gt_bbox and iou < 0.1:
        total_reward = min(total_reward, 0.4)

    # Debug logging for zero localization cases
    if localization_reward == 0.0 and has_anomaly and gt_bbox:
        case_id = extra_info.get('case_id', 'unknown')
        print(
            f"[synthetic_localization=0] case_id={case_id}, "
            f"pathology={pathology_type}, region={expected_region}, "
            f"iou={iou:.4f}, pred_bbox={pred_bbox}, gt_bbox={gt_bbox}, "
            f"area_ratio={area_ratio}, center_dist_norm={center_distance_norm}"
        )

    # Return dict if requested
    if extra_info and extra_info.get('return_dict', False):
        # 将 bbox 转为字符串，避免 numpy 数组形状不一致问题
        return {
            'score': total_reward,
            'localization': localization_reward,
            'iou': iou,
            'format': format_reward,
            'llm_judge': 0.0,  # 跳过
            'keyword': 0.0,    # 跳过
            'question_type': 'synthetic_detection',
            'has_anomaly': has_anomaly,
            'is_synthetic': True,
            'pathology_type': pathology_type,
            'expected_region': expected_region,
            'pred_bbox': str(pred_bbox) if pred_bbox else 'None',
            'gt_bbox': str(gt_bbox) if gt_bbox else 'None',
            'oversize_penalty': oversize_penalty,
            'center_penalty': center_penalty,
        }

    return total_reward


def compute_score_normalized(solution_str, ground_truth, extra_info=None):
    """
    归一化版本的 reward 函数

    将 [0, 0.9] 范围的分数归一化到 [0, 1]，
    使其与真实数据的 reward 范围可比。

    用于混合训练时的 reward 对齐。
    """
    raw_score = compute_score(solution_str, ground_truth, extra_info)

    if isinstance(raw_score, dict):
        raw_score['score_normalized'] = raw_score['score'] / 0.9
        return raw_score

    return raw_score / 0.9
