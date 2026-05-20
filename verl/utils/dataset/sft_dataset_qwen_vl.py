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
SFT dataset for Qwen2.5-VL Vision-Language Models
Handles images alongside text using the official Qwen2.5-VL Processor.
"""

from typing import List, Union, Dict, Any
import base64
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDatasetQwenVL(Dataset):
    """
    Vision-Language SFT Dataset for Qwen2.5-VL models.

    Uses the official Qwen2_5_VLProcessor to handle both text and images.
    Returns tensors that can be directly collated by DataLoader.

    Data format in parquet:
    - extra_info: dict with 'question' and 'answer' keys
    - image: base64-encoded PNG image string

    Arguments:
        parquet_files: Path(s) to parquet file(s)
        tokenizer: Tokenizer or path to tokenizer (will load processor from same path)
        config: Dataset configuration dict
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        # Extract config values
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        self.image_key = config.get("image_key", "image")
        self.model_path = config.get("model_path", None)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.max_length = max_length

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
            # Try to infer from tokenizer path
            self.processor = None

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

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

        # Extract prompts (questions)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        self.prompts = self.prompts.tolist()

        # Extract responses (answers)
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        self.responses = self.responses.tolist()

        # Extract images (keep as base64 strings, decode on-the-fly)
        if self.image_key in self.dataframe.columns:
            self.images = self.dataframe[self.image_key].tolist()
        else:
            self.images = [None] * len(self.prompts)

    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='black')

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item) -> Dict[str, Any]:
        prompt = self.prompts[item]
        response = self.responses[item]
        image_base64 = self.images[item]

        # Decode image
        image = None
        if image_base64 is not None:
            image = self._decode_base64_image(image_base64)

        # Build message for chat template
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Apply chat template to get text
        if self.processor is not None:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process with processor (handles both text and image)
            if image is not None:
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=False,
                )
            else:
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=False,
                )

            # Remove batch dimension
            prompt_ids = inputs["input_ids"].squeeze(0)
            prompt_attention_mask = inputs["attention_mask"].squeeze(0)

            # Get pixel values if present
            pixel_values = None
            image_grid_thw = None
            if "pixel_values" in inputs:
                pixel_values = inputs["pixel_values"]  # Shape: (num_patches, hidden_dim)
            if "image_grid_thw" in inputs:
                image_grid_thw = inputs["image_grid_thw"].squeeze(0)  # Shape: (3,)
        else:
            # Fallback to tokenizer only (no image support)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_inputs = self.tokenizer(
                text=text,
                return_tensors="pt",
                padding=False,
            )
            prompt_ids = prompt_inputs["input_ids"].squeeze(0)
            prompt_attention_mask = prompt_inputs["attention_mask"].squeeze(0)
            pixel_values = None
            image_grid_thw = None

        # Tokenize response
        response_text = response + self.tokenizer.eos_token
        response_inputs = self.tokenizer(
            text=response_text,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False
        )
        response_ids = response_inputs["input_ids"].squeeze(0)
        response_attention_mask = response_inputs["attention_mask"].squeeze(0)

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        # Concatenate prompt and response
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad to max_length
            pad_length = self.max_length - sequence_length
            padded_input_ids = torch.full(
                (pad_length,),
                self.tokenizer.pad_token_id,
                dtype=input_ids.dtype
            )
            padded_attention_mask = torch.zeros(pad_length, dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(
                    f"Sequence length {sequence_length} exceeds max_length {self.max_length}"
                )

        # Compute position ids
        position_ids = compute_position_id_with_mask(attention_mask)

        # Create loss mask (only compute loss on response tokens)
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # Mask out the last token
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

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


def collate_fn_qwen_vl(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for Qwen2.5-VL that handles variable-length pixel_values.

    pixel_values shape varies per image based on resolution, so we concatenate them
    along the first dimension and track which patches belong to which batch item.
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
            # Concatenate all pixel values
            result["pixel_values"] = torch.cat(pixel_values_list, dim=0)

    # Handle image_grid_thw - stack into batch
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        grid_thw_list = [item["image_grid_thw"] for item in batch if item.get("image_grid_thw") is not None]
        if len(grid_thw_list) > 0:
            result["image_grid_thw"] = torch.stack(grid_thw_list)

    return result
