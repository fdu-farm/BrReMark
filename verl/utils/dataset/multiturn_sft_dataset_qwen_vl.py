# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset for Qwen2.5-VL Vision-Language Models.

Supports multi-turn conversations with images, where:
- Turn 1: User provides image + question, Assistant responds with <think>...<tool_call>...
- Turn 2: User provides tool result + marked image, Assistant responds with <rethink>...<answer>...

Loss is computed only on assistant responses.
"""

from typing import List, Union, Dict, Any, Optional
import base64
import re
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class MultiTurnSFTDatasetQwenVL(Dataset):
    """
    Multi-turn Vision-Language SFT Dataset for Qwen2.5-VL models.

    Handles multi-turn conversations with images where each turn may include:
    - Text content
    - Image content (base64 encoded)

    Data format in parquet:
    - messages: List of message dicts with 'role' and 'content'
      - content can be string (text only) or list of dicts for multimodal
      - images are stored as base64 strings in content dicts with type='image'

    Example messages format:
    [
        {"role": "user", "content": [{"type": "image", "image": "base64..."}, {"type": "text", "text": "Analyze..."}]},
        {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},
        {"role": "user", "content": [{"type": "text", "text": "Region marked..."}, {"type": "image", "image": "base64..."}]},
        {"role": "assistant", "content": "<rethink>...</rethink>\n<answer>...</answer>"}
    ]
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config: Dict):
        # Extract config values
        self.max_length = config.get("max_length", 2048)
        self.truncation = config.get("truncation", "error")
        self.model_path = config.get("model_path", None)

        # Multi-turn config
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files

        # Store tokenizer
        self.tokenizer = tokenizer

        # Load processor for image handling
        if self.model_path:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        else:
            self.processor = None
            print("Warning: model_path not provided, image processing may not work correctly")

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls.iloc[0] if isinstance(ls, pandas.core.series.Series) else ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        # Messages may be stored as JSON strings (to avoid pyarrow serialization issues)
        def parse_messages(msg):
            msg = series_to_item(msg)
            if isinstance(msg, str):
                import json
                return json.loads(msg)
            return msg

        self.messages_list = self.dataframe[self.messages_key].apply(parse_messages).tolist()

    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        try:
            # Debug: print type and preview
            if not isinstance(base64_str, str):
                print(f"ERROR: base64_str is not string! Type: {type(base64_str)}, Value: {base64_str}")
                return Image.new('RGB', (224, 224), color='black')
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
            print(f"  base64_str type: {type(base64_str)}")
            print(f"  base64_str value preview: {str(base64_str)[:100]}")
            return Image.new('RGB', (224, 224), color='black')

    def _process_message_content(self, content) -> tuple:
        """
        Process message content to extract text and images.

        This method handles both:
        1. Multimodal content (list format with image/text items)
        2. Text content with literal <image> markers (converted to multimodal format)

        The literal <image> marker handling ensures consistency with RL dataset
        processing (rl_dataset.py), which uses re.split to convert <image> markers
        to multimodal tokens.

        Returns:
            text: The text content (with <image> markers for placeholder positions)
            images: List of PIL Images (may be empty)
        """
        if isinstance(content, str):
            # Check if content contains literal <image> markers
            # If so, we need to preserve them as placeholders for multimodal processing
            # This matches the RL dataset behavior in _build_messages
            return content, []

        if isinstance(content, list):
            text_parts = []
            images = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        # Process text content - preserve <image> markers as-is
                        # They will be handled by the processor's apply_chat_template
                        text_parts.append(text_content)
                    elif item.get("type") == "image":
                        img_data = item.get("image")
                        if img_data:
                            # Check if already a PIL Image (from processor_messages)
                            if isinstance(img_data, Image.Image):
                                images.append(img_data)
                            else:
                                # Decode base64 string
                                images.append(self._decode_base64_image(img_data))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts), images

        return str(content), []

    def _build_processor_messages(self, messages: List[Dict]) -> tuple:
        """
        Build messages in Qwen2.5-VL processor format and collect all images.

        This method ensures consistency with RL dataset processing by:
        1. Converting literal <image> markers in text to multimodal content format
        2. Properly ordering image placeholders and actual image data

        The processing matches rl_dataset.py's _build_messages behavior where
        re.split("(<image>|<video>)", content) is used to convert markers.

        Returns:
            processor_messages: Messages formatted for apply_chat_template
            all_images: List of all PIL Images in order of appearance
        """
        processor_messages = []
        all_images = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            text, images = self._process_message_content(content)

            # Check if text contains literal <image> markers
            # This ensures consistency with RL dataset processing
            has_image_markers = "<image>" in text if text else False

            if images or has_image_markers:
                # Build multimodal content
                content_list = []

                if has_image_markers:
                    # Split text by <image> markers and interleave with image placeholders
                    # This matches RL dataset's re.split("(<image>|<video>)", content) behavior
                    image_idx = 0
                    for segment in re.split(r"(<image>)", text):
                        if segment == "<image>":
                            if image_idx < len(images):
                                content_list.append({"type": "image", "image": images[image_idx]})
                                all_images.append(images[image_idx])
                                image_idx += 1
                            else:
                                # Image marker without corresponding image data
                                # Just add the placeholder type (matches RL behavior)
                                content_list.append({"type": "image"})
                        elif segment:  # Non-empty text segment
                            content_list.append({"type": "text", "text": segment})
                else:
                    # No <image> markers in text, add images first then text
                    for img in images:
                        content_list.append({"type": "image", "image": img})
                        all_images.append(img)
                    content_list.append({"type": "text", "text": text})

                processor_messages.append({"role": role, "content": content_list})
            else:
                # Text-only content
                processor_messages.append({"role": role, "content": text})

        return processor_messages, all_images

    def __len__(self):
        return len(self.messages_list)

    def __getitem__(self, item) -> Dict[str, Any]:
        messages = self.messages_list[item]

        # Build processor-compatible messages and collect images
        processor_messages, all_images = self._build_processor_messages(messages)

        # Apply chat template to get full text
        if self.processor is not None:
            text = self.processor.apply_chat_template(
                processor_messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Process with processor
            if all_images:
                inputs = self.processor(
                    text=[text],
                    images=all_images,
                    return_tensors="pt",
                    padding=False,
                )
            else:
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=False,
                )

            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)

            # Get pixel values if present
            pixel_values = inputs.get("pixel_values", None)
            image_grid_thw = inputs.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw
        else:
            # Fallback to tokenizer only
            text = self.tokenizer.apply_chat_template(
                processor_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            inputs = self.tokenizer(
                text=text,
                return_tensors="pt",
                padding=False,
            )
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            pixel_values = None
            image_grid_thw = None

        # Create loss mask - only train on assistant responses
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        # Process each message to find assistant response positions
        for i, msg in enumerate(processor_messages):
            if msg["role"] != "assistant":
                continue

            # Get tokens for messages up to and including this one
            prefix_messages = processor_messages[:i + 1]
            if self.processor is not None:
                prefix_text = self.processor.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                if all_images:
                    # Count images up to this point
                    images_so_far = []
                    for m in prefix_messages:
                        _, imgs = self._process_message_content(m.get("content", ""))
                        images_so_far.extend(imgs)
                    prefix_inputs = self.processor(
                        text=[prefix_text],
                        images=images_so_far if images_so_far else None,
                        return_tensors="pt",
                        padding=False,
                    )
                else:
                    prefix_inputs = self.processor(
                        text=[prefix_text],
                        return_tensors="pt",
                        padding=False,
                    )
                end_pos = prefix_inputs["input_ids"].squeeze(0).shape[0]
            else:
                prefix_text = self.tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                prefix_inputs = self.tokenizer(
                    text=prefix_text,
                    return_tensors="pt",
                    padding=False,
                )
                end_pos = prefix_inputs["input_ids"].squeeze(0).shape[0]

            # Get tokens for messages before this one
            if i > 0:
                prev_messages = processor_messages[:i]
                if self.processor is not None:
                    prev_text = self.processor.apply_chat_template(
                        prev_messages,
                        tokenize=False,
                        add_generation_prompt=True  # Add generation prompt to find where assistant starts
                    )
                    if all_images:
                        images_before = []
                        for m in prev_messages:
                            _, imgs = self._process_message_content(m.get("content", ""))
                            images_before.extend(imgs)
                        prev_inputs = self.processor(
                            text=[prev_text],
                            images=images_before if images_before else None,
                            return_tensors="pt",
                            padding=False,
                        )
                    else:
                        prev_inputs = self.processor(
                            text=[prev_text],
                            return_tensors="pt",
                            padding=False,
                        )
                    start_pos = prev_inputs["input_ids"].squeeze(0).shape[0]
                else:
                    prev_text = self.tokenizer.apply_chat_template(
                        prev_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prev_inputs = self.tokenizer(
                        text=prev_text,
                        return_tensors="pt",
                        padding=False,
                    )
                    start_pos = prev_inputs["input_ids"].squeeze(0).shape[0]
            else:
                start_pos = 0

            # Set loss mask for assistant response
            loss_mask[start_pos:end_pos] = 1

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_length = self.max_length - sequence_length
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            padded_input_ids = torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros(pad_length, dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(pad_length, dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))

        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"Sequence length {sequence_length} exceeds max_length {self.max_length}")

        # Compute position ids
        position_ids = compute_position_id_with_mask(attention_mask)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        # Add image data if present
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = image_grid_thw

        return result


def collate_fn_multiturn_qwen_vl(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for multi-turn Qwen2.5-VL that handles variable-length pixel_values.
    """
    # Standard tensor fields - stack normally
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
    }

    # Handle pixel_values - concatenate along first dim
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        pixel_values_list = [item["pixel_values"] for item in batch if item.get("pixel_values") is not None]
        if len(pixel_values_list) > 0:
            result["pixel_values"] = torch.cat(pixel_values_list, dim=0)

    # Handle image_grid_thw - concatenate like pixel_values
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        grid_thw_list = [item["image_grid_thw"] for item in batch if item.get("image_grid_thw") is not None]
        if len(grid_thw_list) > 0:
            # Concatenate along first dim to get (total_images, 3)
            result["image_grid_thw"] = torch.cat(grid_thw_list, dim=0)

    return result
