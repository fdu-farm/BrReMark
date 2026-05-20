"""
Mark Bbox Tool Environment for Brain MRI Two-Turn RL Training.

This tool receives a bbox from the model's first response, draws it on the image,
and returns the marked image with feedback for the second turn.
"""

import json
from io import BytesIO
from PIL import Image, ImageDraw

from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents


class MarkBboxEnv(ToolBase):
    """Tool environment for marking bounding boxes on brain MRI images."""

    name = "mark_bbox"

    action_start = '<tool_call>'
    action_end = '</tool_call>'
    answer_start = '<answer>'
    answer_end = '</answer>'

    round2_prompts = {
        "default": "Region of interest has been marked at {bbox}. Please rethink and provide your final answer combining the marked image with your medical knowledge.",
        "normal": "No abnormality marked. Confirm your assessment by reviewing the image.",
    }

    chat_template = """<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""

    def __init__(self, _name, _desc, _params, **kwargs):
        self.multi_modal_data = None
        self.question_type = None
        self.is_normal = False
        self.tool_call_count = 0
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        """
        Execute the mark_bbox tool.

        Args:
            action_string: Model's response containing <tool_call> or <answer>

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Extract tool call FIRST (prioritize tool_call over answer)
        action_list = extract_tool_call_contents(self.action_start, self.action_end, action_string)

        # Only check for answer if NO tool_call found
        if not action_list:
            answers = extract_tool_call_contents(self.answer_start, self.answer_end, action_string)
            if answers:
                return '', 0.0, True, {}
            # No valid action, end episode
            return '', 0.0, True, {}

        # This env is designed for one tool round only.
        if self.tool_call_count >= 1:
            return '', 0.0, True, {"error": "tool_call_limit_exceeded"}

        # Parse the tool call JSON
        action_str = action_list[0].strip()
        try:
            action_json = json.loads(action_str)
        except json.JSONDecodeError:
            # Invalid JSON, end episode
            return '', 0.0, True, {}

        # Only accept dict format, reject other formats to enforce correct output
        if not isinstance(action_json, dict):
            return '', 0.0, True, {}

        # Extract args - handle {"arguments": {...}} or direct {...}
        args = action_json.get("arguments", action_json)
        if not isinstance(args, dict):
            return '', 0.0, True, {}

        bbox = args.get("bbox_2d", args.get("bbox", None))
        label = args.get("label", "region of interest")

        # Get the original image
        if self.multi_modal_data is None or 'image' not in self.multi_modal_data:
            return '', 0.0, True, {}

        pil_img = self.multi_modal_data['image'][0]

        # Handle null or invalid bbox: return original image for re-examination
        # Check if bbox is valid: must be a list of 4 numbers (not None)
        bbox_is_valid = (
            bbox is not None and
            isinstance(bbox, list) and
            len(bbox) == 4 and
            all(isinstance(x, (int, float)) and x is not None for x in bbox)
        )

        if not bbox_is_valid:
            # No valid bbox, return original image and ask for re-examination
            feedback = self._format_round2_prompt(bbox=None, question_type="normal")
            user_msg = "<image>\n" + feedback
            all_user_msg = self.chat_template.format(user_msg)
            obs_dict = {
                "prompt": all_user_msg,
                "multi_modal_data": {"image": [pil_img]}  # Return original image
            }
            self.tool_call_count += 1
            return obs_dict, 0.0, False, {"bbox": None, "label": label, "question_type": self.question_type}

        # Draw bbox on image
        marked_img = self.draw_bbox(pil_img, bbox)

        # Create feedback message
        feedback = self._format_round2_prompt(bbox=bbox, question_type=self.question_type)
        user_msg = "<image>\n" + feedback
        all_user_msg = self.chat_template.format(user_msg)

        # Return observation with marked image
        obs_dict = {
            "prompt": all_user_msg,
            "multi_modal_data": {"image": [marked_img]}
        }
        self.tool_call_count += 1
        return obs_dict, 0.0, False, {"bbox": bbox, "label": label, "question_type": self.question_type}

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, extra_info=None, reward_model=None, **kwargs):
        """Reset the environment with new data."""
        self.multi_modal_data = origin_multi_modal_data
        self.question_type, self.is_normal = self._extract_question_type(extra_info, reward_model)
        self.tool_call_count = 0
        if self.multi_modal_data:
            assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
            assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'

    def _extract_question_type(self, extra_info, reward_model):
        def _parse(val):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return None
            if isinstance(val, dict):
                return val
            return None

        rm = _parse(reward_model) or {}
        ei = _parse(extra_info) or {}

        # Parse ground_truth once for both rm and ei (reuse _parse for consistency)
        rm_gt = _parse(rm.get("ground_truth")) if isinstance(rm, dict) else None
        ei_gt = _parse(ei.get("ground_truth")) if isinstance(ei, dict) else None

        # Extract question_type from rm, ei, or their ground_truth fields
        question_type = None
        for source, gt in [(rm, rm_gt), (ei, ei_gt)]:
            if not isinstance(source, dict):
                continue
            if "question_type" in source:
                question_type = source.get("question_type")
                break
            if isinstance(gt, dict) and "question_type" in gt:
                question_type = gt.get("question_type")
                break

        # Determine is_normal from multiple sources (including parsed ground_truth)
        is_normal = False
        if isinstance(rm, dict):
            if rm.get("has_anomaly") is False:
                is_normal = True
            if isinstance(rm_gt, dict) and rm_gt.get("has_anomaly") is False:
                is_normal = True
        if isinstance(ei, dict):
            if ei.get("is_normal") is True:
                is_normal = True
            if isinstance(ei_gt, dict) and ei_gt.get("has_anomaly") is False:
                is_normal = True

        if isinstance(question_type, str):
            qt = question_type.strip().lower()
            if qt in {"open_diagnosis", "diagnosis"}:
                question_type = "diagnosis"
            elif qt in {"description", "desc"}:
                question_type = "description"
            elif qt in {"detection", "det"}:
                question_type = "detection"
            elif qt in {"normal", "no_abnormality"}:
                question_type = "normal"
            else:
                question_type = qt
        else:
            question_type = None

        if is_normal:
            question_type = "normal"

        return question_type, is_normal

    def _format_round2_prompt(self, bbox, question_type):
        # Use "normal" template for normal cases or invalid bbox, otherwise use "default"
        if question_type == "normal" or bbox is None:
            return self.round2_prompts["normal"]
        template = self.round2_prompts["default"]
        return template.format(bbox=bbox)

    def draw_bbox(self, pil_img, bbox, color="red", width=3):
        """Draw bounding box on image."""
        # Handle base64 string
        if isinstance(pil_img, str):
            import base64
            img_data = base64.b64decode(pil_img)
            pil_img = Image.open(BytesIO(img_data))
        img_copy = pil_img.copy()
        draw = ImageDraw.Draw(img_copy)
        x1, y1, x2, y2 = bbox
        # Ensure valid coordinates (x1 <= x2, y1 <= y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        return img_copy


if __name__ == '__main__':
    # Test the tool
    tool = MarkBboxEnv(_name=None, _desc=None, _params=None)

    action_text = '''<think>Initial observation shows abnormality in right frontal region.</think>
<tool_call>{"name": "mark_bbox", "arguments": {"bbox_2d": [100, 100, 200, 200], "label": "suspected tumor"}}</tool_call>'''

    # Create a test image
    test_img = Image.new('RGB', (300, 300), color='gray')
    tool.multi_modal_data = {'image': [test_img]}

    observation, reward, done, info = tool.execute(action_string=action_text)
    print(f"Done: {done}")
    print(f"Info: {info}")
    if isinstance(observation, dict):
        print(f"Prompt: {observation['prompt'][:100]}...")
