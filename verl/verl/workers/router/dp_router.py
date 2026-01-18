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
Single Process Router
"""

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.workers.router import BaseRouter

__all__ = ["DataParallelRouter"]


class DataParallelRouter(BaseRouter):
    def __init__(
        self,
        config,
        router_module: nn.Module,
        router_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Router"""
        super().__init__(config)
        self.router_module = router_module
        self.router_optimizer = router_optimizer

    def _forward_micro_batch(self, micro_batch):
        outputs = self.router_module(
            input_ids=micro_batch["tokenized_input_ids"],
            attention_mask=micro_batch["tokenized_attention_mask"],
        )
        scores = torch.log_softmax(outputs.logits, dim=-1)
        return scores

    def compute_router_score(self, data: DataProto, micro_batch_size: int):
        self.router_module.eval()
        select_keys = ["tokenized_input_ids", "tokenized_attention_mask"]
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)

        logprobs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                scores = self._forward_micro_batch(micro_batch)
            logprobs_lst.append(scores)

        output_logprobs = torch.cat(logprobs_lst, dim=0)
        return output_logprobs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.router_module, FSDP):
            grad_norm = self.router_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.router_module.parameters(), max_norm=self.config.grad_clip
            )
        self.router_optimizer.step()
        return grad_norm

    def update_router(self, data: DataProto):
        # make sure we are in training mode
        self.router_module.train()
        metrics = {}

        mini_batch_size = self.config.router_mini_batch_size
        micro_batch_size = self.config.router_micro_batch_size_per_gpu
        select_keys = ["tokenized_input_ids", "tokenized_attention_mask", "router_target_logprobs"]
        batch = data.select(batch_keys=select_keys).batch
        #update mini_batch_size
        dataloader = batch.split(mini_batch_size)
        print(f"[RouterDebug] Update Router Input Batch Size: {len(batch)}. Split into {len(dataloader)} Mini-Batches of size {mini_batch_size}")

        for batch_idx, data in enumerate(dataloader):
            mini_batch = data
            micro_batches = mini_batch.split(micro_batch_size)
            self.gradient_accumulation = mini_batch_size // micro_batch_size
            
            print(f"[RouterDebug] MiniBatch {batch_idx}: Size {len(mini_batch)}. Split into {len(micro_batches)} Micro-Batches. GradAccum={self.gradient_accumulation}")

            self.router_optimizer.zero_grad()

            micro_step_count = 0
            for data in micro_batches:
                micro_step_count += 1
                data.cuda()
                target_logprob = data["router_target_logprobs"]

                # Forward pass
                pred_logprobs = self._forward_micro_batch(data)

                # KL loss
                kld = core_algos.kl_penalty(
                    logprob=pred_logprobs,
                    ref_logprob=target_logprob,
                    kl_penalty=self.config.router_kl_loss_type,
                )

                loss = kld.sum() / len(mini_batch)
                loss.backward()

                data = {"router/kl_loss": kld.mean().detach().item()}
                append_to_dict(metrics, data)

            print(f"[RouterDebug] MiniBatch {batch_idx}: Completed {micro_step_count} micro-steps. Calling Optimizer Step.")
            grad_norm = self._optimizer_step()
            data = {"router/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)

        self.router_optimizer.zero_grad()

        return metrics
