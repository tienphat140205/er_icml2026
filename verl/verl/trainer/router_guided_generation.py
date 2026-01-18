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
Generate responses with format distribution sampling.

For each batch:
1. Router predicts probability distribution once
2. Sample n_samples formats from distribution
3. Each generation uses a different sampled format tag
"""
import ray
import numpy as np
import hydra
import os
import torch

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.router_utils import FORMAT_NAMES, apply_format_to_chat


def load_router(router_path: str, device: str = "cuda"):
    """Load the router model and tokenizer."""
    print(f"Loading router from: {router_path}")
    tokenizer = AutoTokenizer.from_pretrained(router_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        router_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_format_distribution(router_model, router_tokenizer, questions: list, device: str = "cuda"):
    """
    Get format probability distribution from router for a batch of questions.
    
    Returns:
        probs: (batch_size, num_formats) probability distribution
    """
    inputs = router_tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = router_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    return probs.cpu()


def sample_formats(probs: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Sample n_samples formats for each prompt from probability distribution.
    
    Args:
        probs: (batch_size, num_formats) probability distribution
        n_samples: number of formats to sample per prompt
        
    Returns:
        format_indices: (batch_size, n_samples) sampled format indices
    """
    return torch.multinomial(probs, n_samples, replacement=True)


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load router model on CPU to avoid OOM conflict with vLLM
    router_path = config.model.get('router_path', None)
    if router_path is None:
        raise ValueError("config.model.router_path must be specified for format sampling")
    
    router_device = "cpu"  # Router on CPU to avoid GPU memory conflict
    router_model, router_tokenizer = load_router(router_path, router_device)
    print(f"Router loaded from {router_path} on {router_device}")

    # Initialize generation workers
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='actor_rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    print(f"Start to generate responses for {len(config.data.path)} datasets.")
    for dataset_path in config.data.path:

        dataset = pd.read_parquet(dataset_path)
        chat_lst = dataset[config.data.prompt_key].tolist()
        chat_lst = [chat.tolist() for chat in chat_lst]

        # Extract questions for router (user content from first user message)
        questions = []
        for chat in chat_lst:
            for msg in chat:
                if msg['role'] == 'user':
                    questions.append(msg['content'])
                    break
            else:
                questions.append(str(chat))  # fallback

        total_samples = len(dataset)
        print(f"processing dataset: {dataset_path}")
        print(f"total_samples: {total_samples}")
        config_batch_size = config.data.batch_size

        dp_size = wg.world_size
        num_batch = -(-total_samples // config_batch_size)

        output_lst = [[] for _ in range(config.data.n_samples)]
        sampled_formats_all = [[] for _ in range(config.data.n_samples)]
        
        print(f'world_size: {wg.world_size}, dp_size: {dp_size}')
        
        for batch_idx in range(num_batch):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            start_idx = batch_idx * config_batch_size
            end_idx = min((batch_idx + 1) * config_batch_size, total_samples)
            batch_chat_lst = chat_lst[start_idx:end_idx]
            batch_questions = questions[start_idx:end_idx]
            real_batch_size = len(batch_chat_lst)

            # Step 1: Get format distribution from router (once per batch)
            probs = get_format_distribution(router_model, router_tokenizer, batch_questions, router_device)
            
            # Step 2: Sample n_samples formats for each prompt
            sampled_format_indices = sample_formats(probs, config.data.n_samples)  # (batch_size, n_samples)
            
            # Print format distribution for this batch
            print(f"[{batch_idx+1}/{num_batch}] Format distribution sampled:")
            for i, name in enumerate(FORMAT_NAMES):
                count = (sampled_format_indices == i).sum().item()
                pct = count / sampled_format_indices.numel() * 100
                print(f"  {name}: {count} ({pct:.1f}%)")

            # Step 3: Generate n_samples times, each with different sampled format
            for sample_idx in range(config.data.n_samples):
                # Apply format tag for this sample iteration
                format_indices_this_sample = sampled_format_indices[:, sample_idx]  # (batch_size,)
                
                tagged_chats = []
                for j, chat in enumerate(batch_chat_lst):
                    fmt_idx = format_indices_this_sample[j].item()
                    format_name = FORMAT_NAMES[fmt_idx]
                    tagged_chat = apply_format_to_chat(chat, format_name)
                    tagged_chats.append(tagged_chat)
                    sampled_formats_all[sample_idx].append(format_name)
                
                # Tokenize tagged prompts
                inputs = tokenizer.apply_chat_template(
                    tagged_chats,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    padding=True,
                    truncation=True,
                    max_length=config.rollout.prompt_length,
                    return_tensors='pt',
                    return_dict=True,
                    tokenize=True,
                )
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                position_ids = compute_position_id_with_mask(attention_mask)

                batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
                data = DataProto.from_dict(batch_dict)
                
                # Handle DP alignment
                if real_batch_size % dp_size != 0:
                    dummy_data_size = dp_size - real_batch_size % dp_size
                    if dummy_data_size <= real_batch_size:
                        dummy_data = data[:dummy_data_size]
                    else:
                        dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
                    data = DataProto.concat([data, dummy_data])

                print(f'[{batch_idx+1}/{num_batch}] Generating sample {sample_idx+1}/{config.data.n_samples}')
                output = wg.generate_sequences(data)
                
                # Remove dummy data
                output = output[:real_batch_size]
                output_text = tokenizer.batch_decode(
                    output.batch['input_ids'][:, -config.rollout.response_length:],
                    skip_special_tokens=False
                )

                # Remove padding
                pad_token = tokenizer.pad_token
                output_text_unpad = [text.replace(pad_token, '') for text in output_text]
                output_lst[sample_idx].extend(output_text_unpad)

        output_lst = np.array(output_lst, dtype=object)
        output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

        # Also transpose sampled formats
        sampled_formats_all = np.array(sampled_formats_all, dtype=object)
        sampled_formats_all = np.transpose(sampled_formats_all, axes=(1, 0)).tolist()

        # Add to dataframe
        dataset['responses'] = output_lst
        dataset['sampled_formats'] = sampled_formats_all

        # Write to parquet
        output_path = os.path.join(
            config.data.output_path, 
            f"{config.rollout.temperature}_{config.data.n_samples}_{os.path.basename(dataset_path)}"
        )
        output_dir = os.path.dirname(output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(output_path)
        print(f"Saved to: {output_path}")

    return None


if __name__ == '__main__':
    main()
