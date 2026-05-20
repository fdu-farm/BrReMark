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
SFT dataset for Vision-Language Models (VLMs)
Extends the base SFTDataset to handle images alongside text.
"""

from typing import List, Union
import base64
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from qwen_vl_utils import process_vision_info

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDatasetVL(Dataset):
    """
    Vision-Language SFT Dataset for models like Qwen2-VL

    This dataset expects parquet files with the following structure:
    - extra_info: dict with 'question' and 'answer' keys
    - image: base64-encoded PNG image string
    - metadata: additional information
    - ground_truth: ground truth annotations

    Arguments:
        parquet_files: Path(s) to parquet file(s)
        tokenizer: Tokenizer or path to tokenizer
        config: Dataset configuration
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        self.image_key = config.get("image_key", "image")

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

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

        # Extract images
        if self.image_key in self.dataframe.columns:
            self.images = self.dataframe[self.image_key].tolist()
        else:
            # No images - fallback to text-only
            self.images = [None] * len(self.prompts)

    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='black')

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        prompt = self.prompts[item]
        response = self.responses[item]
        image_base64 = self.images[item]

        # Decode image from base64
        if image_base64 is not None:
            image = self._decode_base64_image(image_base64)

            # Create multimodal message for Qwen2-VL
            # Format: [{'role': 'user', 'content': [{'type': 'image', 'image': PIL_Image}, {'type': 'text', 'text': prompt}]}]
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
            # Text-only fallback
            messages = [{"role": "user", "content": prompt}]

        # Apply chat template with image processing
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision info (for Qwen2-VL)
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize prompt
        prompt_inputs = tokenizer(
            text=text,
            return_tensors="pt",
            padding=False,
        )

        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_attention_mask = prompt_inputs["attention_mask"][0]

        # Add image pixel values if present
        if image_inputs is not None and len(image_inputs) > 0:
            prompt_inputs["pixel_values"] = image_inputs
        if video_inputs is not None and len(video_inputs) > 0:
            prompt_inputs["video_pixel_values"] = video_inputs

        # Tokenize response
        response_text = response + tokenizer.eos_token
        response_inputs = tokenizer(
            text=response_text,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False
        )
        response_ids = response_inputs["input_ids"][0]
        response_attention_mask = response_inputs["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        # Concatenate prompt and response
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # Padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        # Create loss mask (only compute loss on response tokens)
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # Mask out prompt tokens
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # Mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        # Add image pixel values if present
        if "pixel_values" in prompt_inputs:
            result["pixel_values"] = prompt_inputs["pixel_values"]
        if "video_pixel_values" in prompt_inputs:
            result["video_pixel_values"] = prompt_inputs["video_pixel_values"]

        return result
