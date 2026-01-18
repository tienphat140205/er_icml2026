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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy


import numpy as np
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from collections import Counter, defaultdict
from verl.utils.evaluation import extract_solution, select_reward_fn, find_most_common

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Router = 7


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, current_training_step, total_training_steps, gamma=1.0, lam=1.0, num_repeat=1, rho=0.2, format_reward_agg='mean', soft_best_of_tau=1.0):
    """
    Compute advantage based on adv_estimator.
    
    Args:
        format_reward_agg: Aggregation method for format rewards ('mean', 'best_of', 'soft_best_of')
        soft_best_of_tau: Temperature for soft best-of (default 1.0)
    """
    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        # NOTE: values now only holds response_length dim
        values = values[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'ada_grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        sequences_strs = data.non_tensor_batch['sequences_strs']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        advantages, returns = core_algos.compute_ada_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            sequences_strs=sequences_strs,
            num_repeat=num_repeat,
            current_training_step=current_training_step,
            total_training_steps=total_training_steps,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'router_grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.batch['prompt_indices']
        format_indices = data.batch['format_indices']
        tag_lens = data.batch.get('tag_lens', None)  # May not exist for older batches
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        
        advantages, returns, format_rewards = core_algos.compute_router_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            format_indices=format_indices,
            rho=rho,
            format_reward_agg=format_reward_agg,
            soft_best_of_tau=soft_best_of_tau)
        # Keep advantages/returns at extended length (tag + response) for Actor update
        # Actor now receives extended_responses (tag + generated response)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        # Store format_rewards for router update later
        data.meta_info['format_rewards'] = format_rewards
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
               # values
               'critic/values/mean': torch.mean(valid_values).detach().item(),
               'critic/values/max': torch.max(valid_values).detach().item(),
               'critic/values/min': torch.min(valid_values).detach().item(),
               # vf explained var
               'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
           } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in
            set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_router = Role.Router in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'ada_grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'router_grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        # Create test dataloader for final evaluation
        test_files = self.config.data.get('test_files', None)
        if test_files is None:
            # Fallback to val_files if test_files not specified
            test_files = self.config.data.val_files
        
        self.test_dataset = RLHFDataset(parquet_files=test_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        
        # Test batch size - use val_batch_size config (or fallback to entire dataset)
        test_batch_size = self.config.data.get('val_batch_size', len(self.test_dataset))
        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))],
                                 [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, _ = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        #
        samples = list(zip(sample_inputs, sample_outputs, sample_scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text
        import json
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        val_output_path = os.path.join(local_global_step_folder, 'val_output')
        os.makedirs(val_output_path, exist_ok=True)
        with open(os.path.join(val_output_path, 'val_output.json'), 'w') as f:
            json.dump(samples, f, indent=4)
        #

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # create router
        if self.use_router:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Router)
            router_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Router], config=self.config.router)
            self.resource_pool_to_cls[resource_pool]['router'] = router_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()
        if self.use_router:
            self.router_wg = all_wg['router']
            self.router_wg.init_model()
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)
        if self.use_router:
            router_local_path = os.path.join(local_global_step_folder, 'router')
            router_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'router')
            self.router_wg.save_checkpoint(router_local_path,
                                          router_remote_path,
                                          self.global_steps,
                                          remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)
        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
        
        # Push raw checkpoint to HuggingFace Hub if configured
        push_to_hub = self.config.trainer.get('push_to_hub', None)
        
        if push_to_hub:
            print(f"=== Pushing raw checkpoint step {self.global_steps} to HuggingFace Hub ===")
            from huggingface_hub import HfApi
            import gc
            api = HfApi()
            
            checkpoint_repo = f"{push_to_hub}-step{self.global_steps}"
            print(f"Uploading checkpoint to {checkpoint_repo}")
            try:
                api.create_repo(repo_id=checkpoint_repo, exist_ok=True)
                
                # Upload subfolders one by one to avoid RAM OOM
                subfolders = ['actor', 'router', 'val_output']
                for subfolder in subfolders:
                    subfolder_path = os.path.join(local_global_step_folder, subfolder)
                    if os.path.exists(subfolder_path):
                        print(f"  Uploading {subfolder}...")
                        api.upload_folder(
                            folder_path=subfolder_path,
                            path_in_repo=subfolder,
                            repo_id=checkpoint_repo,
                            repo_type="model",
                        )
                        gc.collect()  # Force garbage collection after each upload
                
                # Upload remaining files (data.pt, etc.)
                for filename in os.listdir(local_global_step_folder):
                    filepath = os.path.join(local_global_step_folder, filename)
                    if os.path.isfile(filepath):
                        print(f"  Uploading {filename}...")
                        api.upload_file(
                            path_or_fileobj=filepath,
                            path_in_repo=filename,
                            repo_id=checkpoint_repo,
                            repo_type="model",
                        )
                        gc.collect()
                
                print(f"Successfully uploaded checkpoint to {checkpoint_repo}")
            except Exception as e:
                print(f"Failed to push checkpoint: {e}")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        router_path = os.path.join(global_step_folder, 'router')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load router
        if self.use_router:
            self.router_wg.load_checkpoint(router_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)  
        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # If total_epochs == 0, skip training and run final test directly
        if self.config.trainer.total_epochs == 0:
            pprint('total_epochs=0, skipping training and running final test directly')
            if self.config.trainer.get('run_final_test', True) and self.val_reward_fn is not None:
                test_metrics = self._final_test(logger)
                pprint(f'Final test metrics: {test_metrics}')
                logger.log(data=test_metrics, step=self.global_steps)
            return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # ======================== Router-Guided Rollout Setup ========================
                # For router_grpo: allocate K samples per prompt according to router probabilities
                # Each sample gets a format tag (DIRECT, COT, CODE, LONG_COT)
                
                if self.config.algorithm.adv_estimator == 'router_grpo':
                    from verl.utils.router_utils import alloc_format, prepare_router_rollout_batch, FORMAT_NAMES
                    
                    raw_prompts = batch.non_tensor_batch.get('raw_prompt')
                    
                    # Step 1: Forward router to get format probabilities
                    with _timer('forward_router', timing_raw):
                        router_output = self.router_wg.forward(batch)
                        router_probs = router_output.batch['router_probs']
                        random_indices = torch.randperm(router_probs.shape[0])[:5]
                        print(f"DEBUG: Router Probabilities (Random 5 Samples): {router_probs[random_indices].tolist()}")

                    
                    # Log mean probabilities per format for monitoring collapse
                    probs_mean = router_probs.mean(dim=0)
                    for f_idx, format_name in enumerate(FORMAT_NAMES):
                         metrics[f"router/rollout_prob/{format_name}"] = probs_mean[f_idx].detach().item()
                    
                    # Step 2: Allocate K samples across formats based on probabilities
                    K = self.config.actor_rollout_ref.rollout.n
                    format_alloc = alloc_format(K=K, probs=router_probs)

                    # Verify allocation
                    assert format_alloc.sum() == batch.batch['input_ids'].shape[0] * K, f"Format allocation sum {format_alloc.sum()} != B*K {batch.batch['input_ids'].shape[0] * K}"

                    alloc_total = format_alloc.sum(dim=0)
                    for f_idx, format_name in enumerate(FORMAT_NAMES):
                        metrics[f"router/format_alloc/count/{format_name}"] = alloc_total[f_idx].detach().item()
                    
                    
                    # Step 3: Expand prompts and tokenize (using helper function)
                    gen_batch = prepare_router_rollout_batch(
                        batch=batch,
                        format_alloc=format_alloc,
                        tokenizer=self.tokenizer,
                        max_prompt_length=self.config.data.max_prompt_length
                    )
                    
                else:
                    # Standard PPO: use the original batch structure
                    gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                    gen_batch.meta_info['do_sample'] = True
                
                with _timer('step', timing_raw):
                    
                    # 4.generate a batch
                    
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        
                        # --- Verification Logging ---
                        if self.config.algorithm.adv_estimator == 'router_grpo' and self.global_steps % 10 == 0:
                            print(f"\n[Verification Step {self.global_steps}] Checking Router GRPO Formats:")
                            from verl.utils.router_utils import FORMAT_NAMES
                            
                            def get_reasoning_format(sequence_str):
                                if "<COT>" in sequence_str:
                                    return "COT"
                                elif "<CODE>" in sequence_str:
                                    return "CODE"
                                elif "<ANSWER>" in sequence_str:
                                    return "DIRECT"
                                elif "<LONG_COT>" in sequence_str:
                                    return "LONG_COT"
                                else:
                                    return "UNKNOWN"

                            # Get format indices
                            fmt_indices = gen_batch.batch['format_indices'].cpu().numpy()
                            # Get prompts and responses
                            input_ids = gen_batch.batch['input_ids']
                            responses = gen_batch_output.batch['responses']
                            attention_mask = gen_batch.batch['attention_mask'] # To find real prompt end
                            
                            # Check 2 samples fully
                            num_check = min(2, len(fmt_indices))
                            indices = np.random.choice(len(fmt_indices), num_check, replace=False)
                            
                            for idx in indices:
                                assigned_fmt_idx = fmt_indices[idx]
                                assigned_fmt_name = FORMAT_NAMES[assigned_fmt_idx]
                                
                                prompt_ids = input_ids[idx]
                                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                                
                                response_text = self.tokenizer.decode(responses[idx], skip_special_tokens=True)
                                
                                full_text = prompt_text + response_text
                                detected_fmt = get_reasoning_format(full_text)
                                
                                match_status = "MATCH" if detected_fmt == assigned_fmt_name else "MISMATCH"
                                
                                print(f"  Sample {idx}: Assigned={assigned_fmt_name} | Detected={detected_fmt} | Status={match_status}")
                                print(f"    FULL PROMPT:\n{prompt_text}")
                                print(f"    FULL RESPONSE:\n{response_text}")
                                print("-" * 50)
                        # ----------------------------

                        # Remove prompt_indices/format_indices from non_tensor_batch to avoid union issues
                        # (since gen_batch_output will have its own structure)
                        # Actually gen_batch_output contains everything we need in batch
                    
                    if self.config.algorithm.adv_estimator == 'remax':
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if self.config.algorithm.adv_estimator == 'router_grpo':
                        # Pop input_ids, attention_mask, position_ids - they will come from gen_batch_output with correct B*K size
                        batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                        
                        # Repeat batch from B to B*K to match gen_batch_output size
                        K = self.config.actor_rollout_ref.rollout.n
                        batch = batch.repeat(repeat_times=K, interleave=True)
                        batch = batch.union(gen_batch_output)
                        
                        # Add uid, format_indices, tag_lens for advantage computation (B*K items)
                        # prompt_indices, format_indices, tag_lens are now in gen_batch.batch (tensors)
                        batch.batch['prompt_indices'] = gen_batch.batch['prompt_indices']
                        batch.batch['format_indices'] = gen_batch.batch['format_indices']
                        batch.batch['tag_lens'] = gen_batch.batch['tag_lens']
                        
                        # --- Tag Handling Refactor ---
                        # (Removed as per user request: Model should not learn tags)

                    else:
                        # Standard PPO: use random uid and repeat to align with repeated responses in rollout
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                 dtype=object)
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    
                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, sequences_strs = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        if sequences_strs:
                            batch.non_tensor_batch['sequences_strs'] = np.array(sequences_strs, dtype=object)
                        else:
                            batch.non_tensor_batch['sequences_strs'] = np.array([], dtype=object)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        #   D.compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  current_training_step=self.global_steps,
                                                  total_training_steps=self.total_training_steps,
                                                  rho=self.config.algorithm.get('router_rho', 0.2),
                                                  format_reward_agg=self.config.algorithm.get('format_reward_agg', 'mean'),
                                                  soft_best_of_tau=self.config.algorithm.get('soft_best_of_tau', 1.0),
                                                  )

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # ========== E+G) Aggregate format rewards and compute target π* ==========
                    if self.config.algorithm.adv_estimator == 'router_grpo':
                        format_rewards_dict = batch.meta_info.get('format_rewards', {})
                        
                        if len(format_rewards_dict) > 0:
                            # format_rewards_dict: {prompt_id: tensor(4,)} from compute_advantage
                            # Sort for guaranteed consistent ordering
                            prompt_ids = sorted(format_rewards_dict.keys())
                            
                            # Stack: (num_prompts, 4) where 4 = [DIRECT, COT, CODE, LONG_COT]
                            format_rewards = torch.stack([
                                format_rewards_dict[pid] for pid in prompt_ids
                            ])
                            
                            num_prompts = format_rewards.shape[0]
                            
                            # Use uniform reference distribution
                            ref_probs = torch.ones(num_prompts, 4, device=format_rewards.device) * 0.25
                            
                            # Compute target log probs: log π*(f|q)
                            beta_max = self.config.algorithm.get('router_beta_max', None)
                            beta_min = self.config.algorithm.get('router_beta_min', None)
                            
                            target_log_probs, effective_beta = core_algos.compute_router_target(
                                ref_probs=ref_probs,
                                format_rewards=format_rewards,
                                current_training_step=self.global_steps,
                                total_training_steps=self.total_training_steps,
                                beta_max=beta_max,
                                beta_min=beta_min
                            )
                            
                            # Log effective beta
                            metrics['router/effective_beta'] = effective_beta
                            
                            # Debug: Log format rewards and target distribution
                            format_rewards_mean = format_rewards.mean(dim=0)
                            format_rewards_std = format_rewards.std(dim=0)
                            target_probs_mean = torch.exp(target_log_probs).mean(dim=0)
                            
                            from verl.utils.router_utils import FORMAT_NAMES
                            for f_idx, fmt_name in enumerate(FORMAT_NAMES):
                                metrics[f'router/format_rewards_mean/{fmt_name}'] = format_rewards_mean[f_idx].item()
                                metrics[f'router/format_rewards_std/{fmt_name}'] = format_rewards_std[f_idx].item()
                                metrics[f'router/target_probs_mean/{fmt_name}'] = target_probs_mean[f_idx].item()
                            
                            # Store in router_batch (B items, matches raw_prompts order)
                            # Create router_batch here properly
                            router_batch = DataProto.from_dict(
                                tensors={'router_target_logprobs': target_log_probs},
                                non_tensors={'raw_prompt': raw_prompts},
                                meta_info={}
                            )
                        else:
                            router_batch = None
                    else:
                        router_batch = None
                    
                    # ========== H) Update router ==========
                    if self.config.algorithm.adv_estimator == 'router_grpo':
                            with _timer('update_router', timing_raw):
                                router_output = self.router_wg.update_router(router_batch)  # Use router_batch (B items)
                                router_output_metrics = reduce_metrics(router_output.meta_info['metrics'])
                                metrics.update(router_output_metrics)
                    
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    
                    # Final test evaluation with both adaptive and router-guided
                    if self.config.trainer.get('run_final_test', True) and self.val_reward_fn is not None:
                        test_metrics = self._final_test(logger)
                        pprint(f'Final test metrics: {test_metrics}')
                        logger.log(data=test_metrics, step=self.global_steps)
                    
                    return

    def _final_test(self, logger=None):
        """
        Run comprehensive final evaluation after training.
        Includes adaptive (Temperature=0.7, N=8) and router-guided evaluation (if enabled).
        Computes accuracy using self-consistency (majority voting).
        """
        from verl.utils.router_utils import prepare_router_rollout_batch, FORMAT_NAMES
        
        test_metrics = defaultdict(list)
        
        # Iterate over test dataloader
        for i, test_data in enumerate(self.test_dataloader):
            test_batch = DataProto.from_single_dict(test_data)
            batch_size = test_batch.batch['input_ids'].shape[0]
            data_sources = test_batch.non_tensor_batch.get('data_source')
            
            # Use config-based reward model key (like main_eval_list.py)
            reward_model_key = self.config.data.get('reward_model_key', 'reward_model')
            ground_truths = test_batch.non_tensor_batch.get(reward_model_key)
            
            # Skip batch if no ground truth data available
            if ground_truths is None:
                pprint(f'Warning: Batch {i} has no {reward_model_key}, skipping evaluation')
                continue

            # ----------------------------------------------------------------
            # 1. Greedy Evaluation (Pass@1)
            # ----------------------------------------------------------------
            greedy_gen_batch = test_batch.select(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            greedy_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False, # Greedy
                'validate': True,
                'temperature': 0.0,
                'n': 1,
            }
            
            # Generate Greedy
            greedy_gen_batch_padded, pad_size = pad_dataproto_to_divisor(greedy_gen_batch, self.actor_rollout_wg.world_size)
            greedy_output_padded = self.actor_rollout_wg.generate_sequences(greedy_gen_batch_padded)
            greedy_output = unpad_dataproto(greedy_output_padded, pad_size=pad_size)
            
            # Decode Greedy
            response_length = self.config.data.get('max_response_length', 4096)
            greedy_texts = self.tokenizer.batch_decode(greedy_output.batch['input_ids'][:, -response_length:], skip_special_tokens=False)
            
            # Remove padding
            pad_token = self.tokenizer.pad_token
            if pad_token:
                greedy_texts = [text.replace(pad_token, '') for text in greedy_texts]
            
            # Score Greedy
            greedy_responses = [[text] for text in greedy_texts]
            self._evaluate_batch(greedy_responses, data_sources, ground_truths, test_metrics, prefix='test/adaptive', mode='greedy')


            # ----------------------------------------------------------------
            # 2. Adaptive Evaluation (Temperature=0.7, N=8)
            # ----------------------------------------------------------------
            test_gen_batch = test_batch.select(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': True,
                'validate': True,
                'temperature': 0.7,
                'n': 8,
            }
            
            # Generate
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output = unpad_dataproto(test_output_padded, pad_size=pad_size)
            
            # Decode
            response_length = self.config.data.get('max_response_length', 4096)
            generated_texts = self.tokenizer.batch_decode(test_output.batch['input_ids'][:, -response_length:], skip_special_tokens=False)
            
            # Remove padding
            pad_token = self.tokenizer.pad_token
            if pad_token:
                generated_texts = [text.replace(pad_token, '') for text in generated_texts]
            
            # Reshape: [batch_size * n_samples] -> [batch_size, n_samples]
            n_samples = 8
            adaptive_responses = [generated_texts[i*n_samples : (i+1)*n_samples] for i in range(batch_size)]
            
            # Score
            self._evaluate_batch(adaptive_responses, data_sources, ground_truths, test_metrics, prefix='test/adaptive')


            # ----------------------------------------------------------------
            # 2. Router-Guided Evaluation (Temperature=0.7, N=8)
            # ----------------------------------------------------------------
            if self.use_router and self.config.algorithm.adv_estimator == 'router_grpo':
                test_batch2 = DataProto.from_single_dict(test_data)
                
                # Pad test batch for DP if needed
                padded_batch, padding_size = self._pad_for_dp(test_batch2)

                # Get router probabilities
                router_output = self.router_wg.forward(padded_batch)
                router_probs = router_output.batch['router_probs'] # [padded_batch, 4]
                
                # Unpad
                if padding_size > 0:
                    router_probs = router_probs[:len(router_probs) - padding_size]

                # ----------------------------------------------------------------
                # 2.1 Router-Guided GREEDY Evaluation (Pass@1, N=1, Deterministic)
                # ----------------------------------------------------------------
                self._evaluate_router_greedy(
                    router_probs=router_probs,
                    test_batch=test_batch2,
                    data_sources=data_sources,
                    ground_truths=ground_truths,
                    test_metrics=test_metrics
                )

                # ----------------------------------------------------------------
                # 2.2 Router-Guided SAMPLING Evaluation (Pass@8, SC, N=8, Stochastic)
                # ----------------------------------------------------------------
                self._evaluate_router_sampling(
                    router_probs=router_probs,
                    test_batch=test_batch2,
                    n_samples=n_samples,
                    data_sources=data_sources,
                    ground_truths=ground_truths,
                    test_metrics=test_metrics
                )

        # Aggregate and return final metrics
        final_metrics = {}
        for key, values in test_metrics.items():
            if 'format_count' in key:
                final_metrics[key] = np.sum(values)
            else:
                final_metrics[key] = np.mean(values)
        
        return final_metrics

    def _generate_with_format(self, test_batch, format_alloc, do_sample, temperature):
        """Generate responses with specified format allocation."""
        from verl.utils.router_utils import prepare_router_rollout_batch
        gen_batch = prepare_router_rollout_batch(
            batch=test_batch, format_alloc=format_alloc,
            tokenizer=self.tokenizer, max_prompt_length=self.config.data.max_prompt_length
        )
        # Only update sampling params - worker auto-injects eos_token_id/pad_token_id
        gen_batch.meta_info.update({'do_sample': do_sample, 'temperature': temperature})
        
        gen_batch_padded, pad_size = self._pad_for_dp(gen_batch)
        output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        out_ids = output_padded.batch['input_ids'][:-pad_size] if pad_size > 0 else output_padded.batch['input_ids']
        outs = self.tokenizer.batch_decode(out_ids[:, -self.config.data.max_response_length:], skip_special_tokens=False)
        pad_token = self.tokenizer.pad_token
        return [text.replace(pad_token, '') for text in outs] if pad_token else outs

    def _evaluate_router_greedy(self, router_probs, test_batch, data_sources, ground_truths, test_metrics):
        """Router-Guided Greedy Evaluation (Pass@1)"""
        batch_size = len(test_batch)
        greedy_formats = torch.argmax(router_probs, dim=-1)
        format_alloc = torch.zeros((batch_size, 4), dtype=torch.long, device=router_probs.device)
        format_alloc[torch.arange(batch_size), greedy_formats] = 1
        greedy_outs = self._generate_with_format(test_batch, format_alloc, do_sample=False, temperature=0.0)
        self._evaluate_batch([[txt] for txt in greedy_outs], data_sources, ground_truths, test_metrics, prefix='test/router_guided', mode='greedy')

    def _evaluate_router_sampling(self, router_probs, test_batch, n_samples, data_sources, ground_truths, test_metrics):
        """Router-Guided Sampling Evaluation (Pass@8, SC)"""
        from verl.utils.router_utils import FORMAT_NAMES
        batch_size = len(test_batch)
        dist = torch.distributions.Categorical(probs=router_probs)
        sampled_formats = dist.sample((n_samples,)).transpose(0, 1)  # [batch, n_samples]
        all_texts = [[] for _ in range(batch_size)]

        for i in range(n_samples):
            current_formats = sampled_formats[:, i]
            format_alloc = torch.zeros((batch_size, 4), dtype=torch.long, device=router_probs.device)
            format_alloc[torch.arange(batch_size), current_formats] = 1
            outs = self._generate_with_format(test_batch, format_alloc, do_sample=True, temperature=0.7)
            for b_i, txt in enumerate(outs):
                all_texts[b_i].append(txt)
            for f_idx, fmt_name in enumerate(FORMAT_NAMES):
                test_metrics[f'test/router_format_count/{fmt_name}'].append((current_formats == f_idx).sum().item())

        self._evaluate_batch(all_texts, data_sources, ground_truths, test_metrics, prefix='test/router_guided')


    def _pad_for_dp(self, data_proto):
        """Pads DataProto to make batch size divisible by world size."""
        bsz = len(data_proto.batch)
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        remainder = bsz % world_size
        if remainder != 0:
            padding_size = world_size - remainder
            # Simple padding by repeating the last element
            indices = torch.arange(bsz)
            pad_indices = torch.tensor([bsz-1] * padding_size)
            all_indices = torch.cat([indices, pad_indices])
            
            # Use indices to create padded batch
            padded_batch_item = data_proto[all_indices]
            # DataProto[indices] returns DataProtoItem, but we need DataProto for the decorator check
            padded_batch = DataProto(
                batch=padded_batch_item.batch,
                non_tensor_batch=padded_batch_item.non_tensor_batch,
                meta_info=padded_batch_item.meta_info
            )
            return padded_batch, padding_size
        else:
            return data_proto, 0

    def _evaluate_batch(self, batch_responses, data_sources, ground_truths, metrics_dict, prefix, mode='sampling'):
        """Helper to score a batch of generated responses.
        Args:
            mode: 'greedy' logs only pass@1; 'sampling' logs avg@8, pass@8, sc
        """
        for idx, responses in enumerate(batch_responses):
            ds = data_sources[idx]
            gt_data = ground_truths[idx]
            ground_truth = gt_data['ground_truth'] if isinstance(gt_data, dict) else gt_data
            reward_fn = select_reward_fn(ds)
            
            sample_scores = []
            valid_solutions = []
            
            for sample in responses:
                sol = extract_solution(sample)
                if sol is None and 'MATH' in ds:
                     sol = sample.strip().split('\n')[-1]
                if sol is not None:
                    s = reward_fn(sample, ground_truth)
                    sample_scores.append(s)
                    valid_solutions.append(sol)
                else:
                    sample_scores.append(0)
            
            if mode == 'greedy':
                # Greedy: only pass@1
                pass1 = sample_scores[0] if sample_scores else 0.0
                metrics_dict[f'{prefix}/{ds}/pass@1'].append(pass1)
                metrics_dict[f'{prefix}/all/pass@1'].append(pass1)
            else:
                # Sampling: avg@8, pass@8, sc
                if not sample_scores:
                    avg8, pass8, sc_score = 0.0, 0.0, 0.0
                else:
                    avg8 = np.mean(sample_scores)
                    pass8 = 1.0 if np.max(sample_scores) > 0 else 0.0
                    if not valid_solutions:
                        sc_score = 0.0
                    else:
                        sc_answer = find_most_common(valid_solutions, ds)
                        sc_score = reward_fn(sc_answer, ground_truth, method='sc')
                
                metrics_dict[f'{prefix}/{ds}/avg@8'].append(avg8)
                metrics_dict[f'{prefix}/{ds}/pass@8'].append(pass8)
                metrics_dict[f'{prefix}/{ds}/sc'].append(sc_score)
                metrics_dict[f'{prefix}/all/avg@8'].append(avg8)
                metrics_dict[f'{prefix}/all/pass@8'].append(pass8)
                metrics_dict[f'{prefix}/all/sc'].append(sc_score)

