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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import json

# Global thread pool for parallel LLM Judge calls
_reward_executor = None

def get_reward_executor(max_workers=16):
    """Get global thread pool for reward computation"""
    global _reward_executor
    if _reward_executor is None:
        _reward_executor = ThreadPoolExecutor(max_workers=max_workers)
    return _reward_executor

class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine if num_examine > 0 else 2  # Force at least 2 samples to print
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self.step_cnt = 0

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        if 'env_reward' in data.batch.keys():
            reward_tensor += data.batch['env_reward']
            print(f' [DEBUG reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}')

        already_print_data_sources = {}

        # Prepare all items for parallel processing
        items_to_process = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            reward_model_data = data_item.non_tensor_batch["reward_model"]
            if isinstance(reward_model_data, str):
                try:
                    reward_model_data = json.loads(reward_model_data)
                except json.JSONDecodeError:
                    reward_model_data = {}
            # Use reward_model_data directly as ground_truth (or nested ground_truth if exists)
            if isinstance(reward_model_data, dict):
                ground_truth = reward_model_data.get("ground_truth", reward_model_data)
            else:
                ground_truth = {}

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except json.JSONDecodeError:
                    extra_info = None

            # Enable detailed reward breakdown for wandb logging
            if extra_info is None:
                extra_info = {}
            extra_info['return_dict'] = True

            items_to_process.append({
                'idx': i,
                'prompt_str': prompt_str,
                'response_str': response_str,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'extra_info': extra_info,
                'valid_response_length': int(valid_response_length),
            })

        # Parallel reward computation using ThreadPoolExecutor
        # 默认：本地 judge（LLM_JUDGE_MODEL 为空/None）时使用更高并发
        llm_judge_model = os.environ.get("LLM_JUDGE_MODEL", "")
        is_local_judge = llm_judge_model.strip().lower() in {"", "none", "null", "lingshu-judge", "lingshu_judge"}
        default_workers = 64 if is_local_judge else 16
        max_workers = int(os.environ.get("REWARD_MAX_WORKERS", str(default_workers)))
        executor = get_reward_executor(max_workers=max_workers)

        def compute_single_reward(item):
            score = self.compute_score(
                data_source=item['data_source'],
                solution_str=item['response_str'],
                ground_truth=item['ground_truth'],
                extra_info=item['extra_info'],
            )
            return item['idx'], score, item

        futures = [executor.submit(compute_single_reward, item) for item in items_to_process]

        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"[WARNING] Reward computation error: {e}")
                results.append((None, 0.0, None))

        # Sort by index and apply rewards
        results.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))

        for idx, score, item in results:
            if idx is None:
                continue

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[idx, item['valid_response_length'] - 1] += reward

            data_source = item['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("=" * 80)
                print(f"[data_source] {data_source}")
                print(f"[has_tool_call] {'<tool_call>' in item['response_str']}")
                print(f"[has_answer] {'<answer>' in item['response_str']}")

                # Extract and print user prompt for debugging
                prompt_str = item.get('prompt_str', '')
                if '<|im_start|>user' in prompt_str:
                    # Find the last user message (the actual question)
                    user_parts = prompt_str.split('<|im_start|>user')
                    if len(user_parts) > 1:
                        last_user_msg = user_parts[-1]
                        # Remove the trailing tokens
                        if '<|im_end|>' in last_user_msg:
                            last_user_msg = last_user_msg.split('<|im_end|>')[0]
                        # Remove all image tokens for cleaner output
                        import re
                        last_user_msg = re.sub(r'<\|vision_start\|>.*?<\|vision_end\|>', '[IMAGE]', last_user_msg, flags=re.DOTALL)
                        last_user_msg = re.sub(r'<\|image_pad\|>', '', last_user_msg)
                        last_user_msg = re.sub(r'<\|vision_start\|>', '', last_user_msg)
                        last_user_msg = re.sub(r'<\|vision_end\|>', '', last_user_msg)
                        # Clean up extra whitespace
                        last_user_msg = ' '.join(last_user_msg.split())
                        print(f"[user_prompt] {last_user_msg.strip()[:500]}")

                # Split response by turns for better readability
                response_str = item['response_str']
                if '<|im_start|>user' in response_str:
                    parts = response_str.split('<|im_start|>user')
                    print("[Round 1 response]", parts[0][:800] if len(parts[0]) > 800 else parts[0])
                    if len(parts) > 1:
                        # Extract and print Round 2 user prompt (tool response)
                        round2_user = parts[1]
                        if '<|im_end|>' in round2_user:
                            round2_user_msg = round2_user.split('<|im_end|>')[0]
                            # Clean up image tokens
                            import re
                            round2_user_msg = re.sub(r'<\|vision_start\|>.*?<\|vision_end\|>', '[IMAGE]', round2_user_msg, flags=re.DOTALL)
                            round2_user_msg = re.sub(r'<\|image_pad\|>', '', round2_user_msg)
                            round2_user_msg = ' '.join(round2_user_msg.split())
                            print("[Round 2 user_prompt]", round2_user_msg[:500] if len(round2_user_msg) > 500 else round2_user_msg)

                        # Find the assistant response in round 2
                        round2 = '<|im_start|>user' + parts[1]
                        if '<|im_start|>assistant' in round2:
                            assistant_start = round2.find('<|im_start|>assistant')
                            round2_response = round2[assistant_start:]
                            print("[Round 2 response]", round2_response[:800] if len(round2_response) > 800 else round2_response)
                        else:
                            print("[Round 2] No assistant response found")
                else:
                    print("[response]", response_str[:1500])

                print("[ground_truth]", item['ground_truth'])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                print("=" * 80)

            self.step_cnt += 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
