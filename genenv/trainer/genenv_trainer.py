# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 GenEnv Authors
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
#
# This file is adapted from verl (https://github.com/volcengine/verl)
# with modifications for GenEnv co-training framework.

"""
GenEnv Trainer: Co-training framework for Agent and Environment LLMs.

This trainer implements the GenEnv algorithm which alternates between:
1. Agent Training: Train the agent policy using GRPO on the current dataset
2. Environment Generation: Generate new challenging tasks using the Env LLM
3. Dataset Augmentation: Merge new tasks with the original dataset

The key innovation is the adaptive curriculum where the Environment LLM
learns to generate tasks at the boundary of the Agent's capability.
"""

import os
import gc
import time
import tempfile
import numpy as np
import pandas as pd
import ray
import torch
from typing import Dict, List, Any, Optional
from omegaconf import OmegaConf
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from genenv.utils.reward_functions import RewardManager


@ray.remote
class EnvGeneratorWorker:
    """
    Environment Generator Worker using vLLM for efficient inference.

    This worker is responsible for generating new training tasks based on
    existing tasks and their solutions. Users should customize the generation
    prompt template in `_build_generation_prompt` method.

    Note (verl 0.4.1 adaptation): the number of GPUs reserved for this Ray actor
    must match ``tensor_parallel_size``. The original decorator hard-coded
    ``num_gpus=8`` while the default ``tensor_parallel_size`` was 4, which is a
    mismatch. We now leave the GPU count off the decorator and set it via
    ``.options(num_gpus=tensor_parallel_size)`` at the call site, and default to
    single-GPU TP since Qwen3-8B fits comfortably on one H800 (80GB).
    """

    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        """
        Initialize the Environment Generator.
        
        Args:
            model_path: Path to the Environment LLM (e.g., Qwen2.5-7B-Instruct)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.75,
            max_model_len=8192
        )
        
    def generate(self, prompts: List[str], n: int = 1, max_tokens: int = 8192) -> List[str]:
        """
        Generate new tasks from the given prompts.
        
        Args:
            prompts: List of formatted prompts for task generation
            n: Number of generations per prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated task descriptions
        """
        sampling_params = SamplingParams(n=n, temperature=1.0, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


def run_genenv_training(config):
    """
    Main entry point for GenEnv co-training.
    
    This function orchestrates the alternating training between Agent and Environment.
    
    Args:
        config: Hydra configuration object containing all training parameters
        
    Note:
        Users need to customize:
        - reward_fn: Your domain-specific reward function
        - env_prompt_template: How to prompt the Env LLM for task generation
        - task_parsing: How to extract new tasks from Env LLM outputs
    """
    from transformers import AutoTokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    
    print("[GenEnv] Starting co-training loop...")
    
    # Initialize tokenizers
    agent_model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    agent_tokenizer = AutoTokenizer.from_pretrained(agent_model_path, trust_remote_code=True)
    
    env_model_path = copy_local_path_from_hdfs(config.env_model_path)
    env_tokenizer = AutoTokenizer.from_pretrained(env_model_path, trust_remote_code=True)
    
    current_train_file = config.data.train_files
    
    for epoch in range(config.trainer.total_epochs):
        print(f"\n{'='*60}")
        print(f"[GenEnv] Epoch {epoch+1}/{config.trainer.total_epochs}")
        print(f"{'='*60}")
        
        # =====================================================================
        # Phase 1: Agent Training
        # =====================================================================
        print("\n[GenEnv] Phase 1: Agent Training")
        
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
        }
        resource_pool_spec = {'global_pool': [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {
            Role.ActorRollout: 'global_pool',
            Role.RefPolicy: 'global_pool',
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        # >>> USER CUSTOMIZATION: Replace with your reward function <<<
        reward_fn = RewardManager(tokenizer=agent_tokenizer, num_examine=0)
        val_reward_fn = RewardManager(tokenizer=agent_tokenizer, num_examine=1)
        
        # Update config with current train file
        agent_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        agent_config.data.train_files = current_train_file
        # verl 0.4.1 fit() loops over trainer.total_epochs internally; the GenEnv outer
        # loop already iterates epochs, so cap the inner trainer to a single epoch.
        agent_config.trainer.total_epochs = 1
        # Only validate before the very first agent-training pass to save time.
        agent_config.trainer.val_before_train = (epoch == 0)
        
        # Load from checkpoint if not first epoch
        if epoch > 0:
            actor_dir = os.path.join(config.trainer.default_local_dir, 'actor')
            if os.path.exists(actor_dir):
                steps = []
                for d in os.listdir(actor_dir):
                    if d.startswith('global_step_'):
                        try:
                            steps.append(int(d.split('_')[-1]))
                        except:
                            pass
                if steps:
                    latest_step = max(steps)
                    latest_ckpt = os.path.join(actor_dir, f'global_step_{latest_step}')
                    print(f"[GenEnv] Loading checkpoint from: {latest_ckpt}")
                    agent_config.actor_rollout_ref.model.path = latest_ckpt
        
        # Create and run trainer
        agent_trainer = RayPPOTrainer(
            config=agent_config,
            tokenizer=agent_tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
        agent_trainer.init_workers()
        # verl 0.4.1: fit() takes no args; it loops over config.trainer.total_epochs.
        # We drive the GenEnv outer epoch loop ourselves, so run one agent-training pass
        # per outer epoch by giving the inner trainer a single-epoch config.
        agent_trainer.fit()
        
        # =====================================================================
        # Phase 2: Evaluate Agent Performance on Current Dataset
        # =====================================================================
        print("\n[GenEnv] Phase 2: Evaluating Agent Performance")
        
        prompt_accuracies = _evaluate_agent_performance(
            agent_trainer=agent_trainer,
            agent_tokenizer=agent_tokenizer,
            config=config,
            current_train_file=current_train_file,
            reward_fn=reward_fn
        )
        
        # Save checkpoint
        agent_trainer._save_checkpoint()
        
        # Cleanup agent resources
        _cleanup_trainer(agent_trainer)
        
        # =====================================================================
        # Phase 3: Environment Generation
        # =====================================================================
        print("\n[GenEnv] Phase 3: Environment Generation")
        
        # Filter prompts based on agent performance
        filtered_prompts = _filter_prompts(
            prompt_accuracies=prompt_accuracies,
            filtering_k=config.genenv.filtering_k
        )
        
        # Generate new tasks
        new_dataset = _generate_new_tasks(
            filtered_prompts=filtered_prompts,
            env_model_path=env_model_path,
            env_tokenizer=env_tokenizer,
            config=config,
            epoch=epoch
        )
        
        # =====================================================================
        # Phase 4: Dataset Augmentation
        # =====================================================================
        print("\n[GenEnv] Phase 4: Dataset Augmentation")
        
        if new_dataset:
            current_train_file = _augment_dataset(
                new_dataset=new_dataset,
                original_train_file=config.data.train_files,
                config=config,
                epoch=epoch
            )
        else:
            print("[GenEnv] Warning: No new data generated. Reusing original dataset.")
            current_train_file = config.data.train_files
    
    print("\n[GenEnv] Training completed!")


def _evaluate_agent_performance(
    agent_trainer: RayPPOTrainer,
    agent_tokenizer,
    config,
    current_train_file: str,
    reward_fn
) -> Dict[str, Dict]:
    """
    Evaluate agent performance on the current training dataset.

    Adapted to verl 0.4.1: we reuse the trainer's own ``train_dataloader`` (already
    built in ``RayPPOTrainer.__init__`` from ``current_train_file``) and follow the
    exact generation pattern used in ``RayPPOTrainer._validate`` so that the batch
    carries ``raw_prompt_ids`` and the meta_info that the vLLM rollout worker needs.

    Returns:
        Dictionary mapping prompt -> {'scores': [...], 'gt': ground_truth}
    """
    prompt_accuracies: Dict[str, Dict] = {}
    n_gen = config.genenv.num_generations_per_prompt

    for batch_dict in agent_trainer.train_dataloader:
        test_batch: DataProto = DataProto.from_single_dict(batch_dict)
        batch_size = len(test_batch)

        # Decode the (left-padded) prompts to use as stable grouping keys, and grab
        # the ground truths, before any keys get popped/repeated.
        input_ids = test_batch.batch["input_ids"]
        attention_mask = test_batch.batch["attention_mask"]
        base_prompts = []
        for k in range(batch_size):
            valid_ids = input_ids[k][attention_mask[k] == 1]
            base_prompts.append(agent_tokenizer.decode(valid_ids, skip_special_tokens=True))
        base_gts = [item["ground_truth"] for item in test_batch.non_tensor_batch["reward_model"]]

        # Repeat each sample n_gen times (interleaved) so reward_model/raw_prompt_ids
        # stay aligned with the generations.
        test_batch = test_batch.repeat(repeat_times=n_gen, interleave=True)

        # Pop the keys needed for generation (mirrors _validate / fit).
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        for opt_key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "interaction_kwargs"):
            if opt_key in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(opt_key)
        gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch.meta_info = {
            "eos_token_id": agent_tokenizer.eos_token_id,
            "pad_token_id": agent_tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,  # sample so we can estimate per-prompt pass rate
            "validate": True,
        }

        # pad to be divisible by dp size, generate, then unpad
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            gen_batch, agent_trainer.actor_rollout_wg.world_size
        )
        output_padded = agent_trainer.actor_rollout_wg.generate_sequences(gen_batch_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        test_batch = test_batch.union(output)

        # Score with the reward function (verl 0.4.1 return_dict convention).
        reward_result = reward_fn(test_batch, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        rewards = reward_tensor.sum(-1).cpu().tolist()

        # rewards are interleaved: [p0_g0, p0_g1, ..., p1_g0, ...]
        for i in range(batch_size):
            prompt_text = base_prompts[i]
            if prompt_text not in prompt_accuracies:
                prompt_accuracies[prompt_text] = {"scores": [], "gt": base_gts[i]}
            prompt_rewards = rewards[i * n_gen: (i + 1) * n_gen]
            prompt_accuracies[prompt_text]["scores"].extend(prompt_rewards)

    return prompt_accuracies


def _filter_prompts(prompt_accuracies: Dict, filtering_k: float) -> Dict:
    """
    Filter out prompts that are too easy (always solved) or too hard (never solved).
    
    The GenEnv algorithm focuses on prompts at the boundary of the agent's capability.
    """
    num_prompts = len(prompt_accuracies)
    num_to_remove = int(num_prompts * filtering_k)
    
    prompt_avg_scores = {p: np.mean(v['scores']) for p, v in prompt_accuracies.items()}
    sorted_prompts = sorted(prompt_avg_scores.items(), key=lambda item: item[1])
    
    # Remove easiest and hardest prompts
    prompts_to_remove = set([p for p, s in sorted_prompts[:num_to_remove]])
    prompts_to_remove.update([p for p, s in sorted_prompts[-num_to_remove:]])
    
    filtered_prompts = {p: v for p, v in prompt_accuracies.items() if p not in prompts_to_remove}
    print(f"[GenEnv] Filtered {len(prompts_to_remove)} prompts. Remaining: {len(filtered_prompts)}")
    
    return filtered_prompts


def _generate_new_tasks(
    filtered_prompts: Dict,
    env_model_path: str,
    env_tokenizer,
    config,
    epoch: int
) -> List[Dict]:
    """
    Generate new training tasks using the Environment LLM.
    
    >>> USER CUSTOMIZATION REQUIRED <<<
    Modify the prompt template and parsing logic for your specific domain.
    """
    print(f"[GenEnv] Generating new tasks using Env LLM...")

    # verl 0.4.1: reserve exactly tensor_parallel_size GPUs for the actor so the
    # Ray GPU reservation matches vLLM's tensor parallel world size.
    env_tp = int(config.genenv.get("env_tensor_parallel_size", 1))
    env_worker = EnvGeneratorWorker.options(num_gpus=env_tp).remote(
        env_model_path, tensor_parallel_size=env_tp
    )
    
    prompts_list = []
    original_infos = []
    
    for prompt, data in filtered_prompts.items():
        # >>> USER CUSTOMIZATION: Modify this prompt template <<<
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Based on the original task and its correct answer, create a new, similar task and provide its correct answer. Format your response as:\nNew Task: <task>\nNew Answer: <answer>"},
            {"role": "user", "content": f"Original Task: {prompt}\nOriginal Answer: {data['gt']}"}
        ]
        env_prompt = env_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_list.append(env_prompt)
        original_infos.append(data)
    
    print(f"[GenEnv] Sending {len(prompts_list)} prompts to Env LLM...")
    generated_texts = ray.get(env_worker.generate.remote(prompts_list, n=1, max_tokens=8192))
    
    new_dataset = []
    import re
    
    for i, text in enumerate(generated_texts):
        try:
            # >>> USER CUSTOMIZATION: Modify parsing logic <<<
            match = re.search(r"New Task:(.*?)New Answer:(.*)", text, re.DOTALL | re.IGNORECASE)
            
            if match:
                new_task = match.group(1).strip()
                new_answer = match.group(2).strip()
                
                new_dataset.append({
                    config.data.prompt_key: [{"role": "user", "content": new_task}],
                    "reward_model": {"ground_truth": new_answer}
                })
        except Exception as e:
            print(f"[GenEnv] Warning: Error parsing output: {e}")
            continue
    
    ray.kill(env_worker)
    print(f"[GenEnv] Generated {len(new_dataset)} new tasks")
    
    return new_dataset


def _augment_dataset(
    new_dataset: List[Dict],
    original_train_file: str,
    config,
    epoch: int
) -> str:
    """
    Merge new generated tasks with the original training dataset.
    """
    new_df = pd.DataFrame(new_dataset)
    original_df = pd.read_parquet(original_train_file)
    
    # Ensure all required columns exist
    for col in original_df.columns:
        if col not in new_df.columns:
            if col == 'extra_info':
                new_df[col] = [{'id': f'genenv_{epoch}_{i}', 'index': i, 'split': 'env_generated'}
                               for i in range(len(new_df))]
            elif col == 'data_source':
                new_df[col] = 'genenv_generated'
            else:
                new_df[col] = None
    
    combined_df = pd.concat([original_df, new_df], ignore_index=True)
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".parquet") as tmp:
        combined_df.to_parquet(tmp.name)
        new_train_file = tmp.name
    
    print(f"[GenEnv] Dataset augmented: {len(original_df)} + {len(new_df)} = {len(combined_df)} samples")
    
    return new_train_file


def _cleanup_trainer(trainer):
    """Release trainer resources.

    verl 0.4.1 adaptation: there is no ``trainer.wg_dicts``. Worker groups are stored
    as direct attributes (``actor_rollout_wg``, ``ref_policy_wg``, ``critic_wg``,
    ``rm_wg``), each a ``RayWorkerGroup`` exposing ``.workers``. We kill the workers
    of whichever groups exist, then tear down the placement groups.
    """
    from ray.util.placement_group import remove_placement_group

    print("[GenEnv] Releasing trainer resources...")

    for wg_attr in ("actor_rollout_wg", "ref_policy_wg", "critic_wg", "rm_wg"):
        wg = getattr(trainer, wg_attr, None)
        if wg is None:
            continue
        for worker in getattr(wg, "workers", []):
            try:
                ray.kill(worker)
            except Exception:
                pass

    if hasattr(trainer, "resource_pool_manager"):
        for pool_name, pool in trainer.resource_pool_manager.resource_pool_dict.items():
            if hasattr(pool, "pgs") and pool.pgs:
                for pg in pool.pgs:
                    try:
                        remove_placement_group(pg)
                    except Exception:
                        pass

    del trainer
    gc.collect()
    time.sleep(5)

