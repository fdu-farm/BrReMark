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
Model Merger for FSDP Checkpoints

Merges sharded FSDP checkpoints into a single HuggingFace model.

Usage:
    # For VL models (Qwen2.5-VL, etc.), specify --original_model_path to ensure
    # config.json is correct for vLLM loading:
    python model_merger.py \\
        --local_dir checkpoint/rl_xxx/global_step_xxx/ \\
        --original_model_path Qwen/Qwen2.5-VL-7B-Instruct

Note:
    save_pretrained() may modify config.json fields (e.g., rope_scaling.type
    from "mrope" to "default"), which can cause vLLM loading to fail.
    Using --original_model_path ensures the correct config is preserved.
"""

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    PretrainedConfig,
    PreTrainedModel,
)


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(local_path: str, remote_path: str):
    # Push to hugging face
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    parser.add_argument("--original_model_path", type=str, default=None,
                        help="Path to original model for copying config.json (e.g., Qwen2.5-VL-7B-Instruct)")
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)

    state_dict: Dict[str, List[torch.Tensor]] = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print(f"Cannot find key {key} in rank {rank}.")

            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at ddp dimension can be discarded
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            # merge shards
            placements: Tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")
    config_path = os.path.join(hf_path, "config.json")

    # Backup original config.json before save_pretrained overwrites it
    original_config = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            original_config = json.load(f)

    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    architectures: List[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    with torch.device("meta"):
        model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

    assert isinstance(model, PreTrainedModel)
    model.to_empty(device="cpu")

    print(f"Saving model to {hf_path}...")
    model.save_pretrained(hf_path, state_dict=state_dict)

    # Restore config.json from original model
    # save_pretrained may modify rope_scaling.type, vision_config order, etc.
    # which can cause vLLM loading to fail
    if args.original_model_path:
        original_config_path = os.path.join(args.original_model_path, "config.json")
        if os.path.exists(original_config_path):
            print(f"Copying config.json from original model: {args.original_model_path}")
            with open(original_config_path, "r") as f:
                final_config = json.load(f)
            # Preserve pad_token_id from checkpoint config if it exists
            if original_config is not None and "pad_token_id" in original_config:
                final_config["pad_token_id"] = original_config["pad_token_id"]
            with open(config_path, "w") as f:
                json.dump(final_config, f, indent=2)
        else:
            print(f"Warning: config.json not found at {original_config_path}")
    else:
        print("Warning: --original_model_path not specified.")
        print("  For VL models (e.g., Qwen2.5-VL), you should specify the original model path")
        print("  to ensure config.json is correct for vLLM loading.")
        print("  Example: --original_model_path /path/to/Qwen2.5-VL-7B-Instruct")
        # Fallback: restore the backup config (may still have issues)
        if original_config is not None:
            print("Restoring config.json from backup (may have issues with vLLM)...")
            with open(config_path, "w") as f:
                json.dump(original_config, f, indent=2)

    del state_dict, model

    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
