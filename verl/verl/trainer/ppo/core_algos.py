# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import math
from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == "fixed":
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == "adaptive":
        assert config.kl_ctrl.horizon > 0, (
            f"horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}"
        )
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=config.critic.kl_ctrl.kl_coef,
            target_kl=config.critic.kl_ctrl.target_kl,
            horizon=config.critic.kl_ctrl.horizon,
        )
    else:
        raise ValueError("Unknown kl_ctrl type")

    return kl_ctrl


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


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


def compute_ada_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    sequences_strs: torch.Tensor,
    current_training_step: int,
    total_training_steps: int,
    num_repeat: int,
    epsilon: float = 1e-6,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2format = defaultdict(dict)
    id2mean = {}
    id2std = {}

    def cosine_decay_alpha(rollout_n, cur_format_num, current_training_step, total_training_steps):
        a = rollout_n / cur_format_num
        b = 1
        return b + 0.5 * (a - b) * (
            1 + math.cos(math.pi * current_training_step / total_training_steps)
        )

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            uid = index[i]
            solution_str = sequences_strs[i]
            _format = get_reasoning_format(solution_str)
            id2format[uid][_format] = id2format[uid].get(_format, 0) + 1

        for key, value in id2format.items():
            rollout_n = sum(value.values())
            assert rollout_n == num_repeat, f"rollout_n: {rollout_n}, num_repeat: {num_repeat}"

        for i in range(bsz):
            uid = index[i]
            _format = get_reasoning_format(sequences_strs[i])
            if _format == "UNKNOWN":
                assert scores[i] == 0.0, f"score: {scores[i]}"
            cur_format_num = id2format[uid][_format]
            assert cur_format_num > 0, f"cur_format_num: {cur_format_num}"
            alpha = cosine_decay_alpha(
                rollout_n, cur_format_num, current_training_step, total_training_steps
            )

            scores[i] = scores[i] * alpha
            id2score[uid].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_router_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                          eos_mask: torch.Tensor,
                                          index: torch.Tensor,
                                          format_indices: torch.Tensor,
                                          rho: float = 0.2,
                                          epsilon: float = 1e-6,
                                          format_reward_agg: str = 'mean',
                                          soft_best_of_tau: float = 1.0):
    """
    Compute advantage for Router GRPO using a rank-based reward scheme that
    dynamically promotes efficiency relative to the current batch.
    
    The reward scheme:
    - Let Y+(q) = {y ∈ Y(q) | Acc(y) = 1} be the set of correct responses for prompt q
    - Sort responses by length such that Len(y_(0)) ≤ Len(y_(1)) ≤ ...
    - For a generated response y:
        r_LLM(q, y) = ρ + (1 - ρ) · (1/2^k)  if y = y_(k) ∈ Y+(q)
                    = 0                       if Acc(y) = 0
    
    where k is the rank (0-indexed) of the response among correct candidates,
    and ρ ∈ (0, 1) is a base reward ensuring any correct response gets higher reward than incorrect.
    
    Args:
        token_level_rewards: (bs, response_length) - 1 for correct, 0 for incorrect
        eos_mask: (bs, response_length) - valid token mask
        index: (bs,) - prompt indices for grouping
        format_indices: (bs,) - format index for each sample (0=DIRECT, 1=COT, 2=CODE, 3=LONG_COT)
        rho: base reward for correct responses (default 0.2)
        epsilon: numerical stability (default 1e-6)
        format_reward_agg: aggregation method for format rewards:
            - 'mean': r_fmt(q,f) = (1/n_f) * sum(r_LLM(q, y_i, f))
            - 'best_of': r_fmt(q,f) = max(r_LLM(q, y_i, f))
            - 'soft_best_of': r_fmt(q,f) = (1/tau) * log(sum(exp(tau * r_LLM(q, y_i, f))))
        soft_best_of_tau: temperature for soft best-of (default 1.0)
    
    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
        format_rewards: dict[prompt_id -> tensor(4,)] - aggregated rewards for [DIRECT, COT, CODE, LONG_COT]
    """
    NUM_FORMATS = 4  # DIRECT, COT, CODE, LONG_COT
    
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    response_lengths = eos_mask.sum(dim=-1)  # (bs,)
    
    with torch.no_grad():
        bsz = scores.shape[0]
        device = scores.device
        
        # Group responses by prompt
        id2responses = defaultdict(list)
        for i in range(bsz):
            prompt_id = index[i].item() if hasattr(index[i], 'item') else index[i]
            fmt_idx = format_indices[i].item() if hasattr(format_indices[i], 'item') else int(format_indices[i])
            id2responses[prompt_id].append({
                'idx': i,
                'score': scores[i].item() if hasattr(scores[i], 'item') else float(scores[i]),
                'length': response_lengths[i].item() if hasattr(response_lengths[i], 'item') else int(response_lengths[i]),
                'format_idx': fmt_idx
            })
        
        # Compute rank-based rewards
        new_scores = torch.zeros_like(scores)
        
        for prompt_id, responses in id2responses.items():
            correct_responses = [r for r in responses if r['score'] == 1]
            
            if len(correct_responses) > 0:
                # Sort correct responses by length (ascending)
                correct_responses.sort(key=lambda x: x['length'])
                
                # Assign rank-based rewards
                for rank, response in enumerate(correct_responses):
                    # r_LLM(q, y) = ρ + (1 - ρ) · (1/2^k)
                    reward = rho + (1.0 - rho) * (1.0 / (2 ** rank))
                    new_scores[response['idx']] = reward
        
        # Aggregate rewards by format
        format_rewards = {}
        for prompt_id, responses in id2responses.items():
            format_scores = [[] for _ in range(NUM_FORMATS)]
            
            for response in responses:
                fmt_idx = response['format_idx']
                if 0 <= fmt_idx < NUM_FORMATS:
                    format_scores[fmt_idx].append(new_scores[response['idx']].item())
            
            # Compute aggregated reward per format based on method
            format_agg = torch.zeros(NUM_FORMATS, device=device)
            for f_idx in range(NUM_FORMATS):
                if len(format_scores[f_idx]) > 0:
                    scores_tensor = torch.tensor(format_scores[f_idx], device=device)
                    
                    if format_reward_agg == 'mean':
                        # Mean: r_fmt(q,f) = (1/n_f) * sum(r_LLM)
                        format_agg[f_idx] = scores_tensor.mean()
                    elif format_reward_agg == 'best_of':
                        # Best-of: r_fmt(q,f) = max(r_LLM)
                        format_agg[f_idx] = scores_tensor.max()
                    elif format_reward_agg == 'soft_best_of':
                        # Soft best-of: r_fmt(q,f) = (1/tau) * log(sum(exp(tau * r_LLM)))
                        format_agg[f_idx] = (1.0 / soft_best_of_tau) * torch.logsumexp(
                            soft_best_of_tau * scores_tensor, dim=0
                        )
                    else:
                        raise ValueError(f"Unknown format_reward_agg: {format_reward_agg}")
            
            format_rewards[prompt_id] = format_agg
        
        # Normalize within each prompt group
        id2mean = {}
        id2std = {}
        id2score_list = defaultdict(list)
        
        for i in range(bsz):
            prompt_id = index[i].item() if hasattr(index[i], 'item') else index[i]
            id2score_list[prompt_id].append(new_scores[i])
        
        for prompt_id in id2score_list:
            score_list = id2score_list[prompt_id]
            if len(score_list) == 1:
                id2mean[prompt_id] = torch.tensor(0.0, device=device)
                id2std[prompt_id] = torch.tensor(1.0, device=device)
            elif len(score_list) > 1:
                stacked = torch.stack(score_list)
                id2mean[prompt_id] = torch.mean(stacked)
                id2std[prompt_id] = torch.std(stacked)
            else:
                raise ValueError(f"No score in prompt index: {prompt_id}")
        
        for i in range(bsz):
            prompt_id = index[i].item() if hasattr(index[i], 'item') else index[i]
            new_scores[i] = (new_scores[i] - id2mean[prompt_id]) / (id2std[prompt_id] + epsilon)
        
        # Expand to token level
        new_scores = new_scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    
    return new_scores, new_scores, format_rewards




def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, eos_mask: torch.Tensor
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3", "k3+"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_router_target(ref_probs: torch.Tensor, 
                          format_rewards: torch.Tensor, 
                          current_training_step: int,
                          total_training_steps: int,
                          beta_max: float,
                          beta_min: float):
    """
    Compute closed-form target distribution π*(f|q) for the router.
    Formula: π*(f|q) = π_ref(f|q) * exp(r_fmt(q,f) / β) / Z
    
    Cosine decay: beta starts at beta_max (uniform) and decays to beta_min (concentrated).
    - High beta → uniform distribution (exploration)
    - Low beta → concentrated on best format (exploitation)
    
    Args:
        ref_probs: (bs, num_formats) - reference policy probabilities
        format_rewards: (bs, num_formats) - aggregated format rewards
        current_training_step: current step for cosine decay
        total_training_steps: total steps for cosine decay
        beta_max: starting beta (high = uniform)
        beta_min: ending beta (low = concentrated)
        
    Returns:
        target_log_probs: (bs, num_formats) - log of normalized target distribution
        effective_beta: the actual beta used (for logging)
    """
    # Cosine decay from beta_max to beta_min
    # cos(0) = 1 → beta = beta_max
    # cos(π) = -1 → beta = beta_min
    progress = current_training_step / max(total_training_steps, 1)
    effective_beta = beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(math.pi * progress))
    
    # π*(f|q) = π_ref(f|q) * exp(r_fmt(q,f) / β)
    # log π*(f|q) = log π_ref(f|q) + r_fmt(q,f) / β - log Z
    log_ref_probs = torch.log(ref_probs + 1e-8)
    log_unnorm = log_ref_probs + format_rewards / effective_beta
    
    # Log-softmax for numerical stability
    target_log_probs = log_unnorm - torch.logsumexp(log_unnorm, dim=-1, keepdim=True)
    
    return target_log_probs, effective_beta
