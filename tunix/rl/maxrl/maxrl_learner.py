# Copyright 2025 Google LLC
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

"""MaxRL (Maximum Likelihood Reinforcement Learning) learner.

Based on the paper: "Maximum Likelihood Reinforcement Learning"
by Tajwar et al. (2026) - https://arxiv.org/abs/2602.02710
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Sequence, TypeVar

import flax
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  pass


@dataclasses.dataclass(slots=True, kw_only=True)
class MaxRLConfig(algo_config_lib.AlgorithmConfig):
  """Configuration for MaxRL algorithm.

  MaxRL (Maximum Likelihood Reinforcement Learning) is a policy gradient
  algorithm that uses a different advantage normalization compared to GRPO.
  Instead of normalizing by standard deviation, MaxRL normalizes by the mean,
  which provides better theoretical properties for maximum likelihood estimation.

  Attributes:
    algo_variant: The algorithm variant to use. Default: `maxrl`.
    advantage_estimator: The advantage estimator to use. Default: `maxrl`.
    policy_loss_fn: The policy loss function to use. Default: `grpo`.
    loss_agg_mode: The aggregation mode for the loss function. Default:
      `sequence-mean-token-mean`.
    reward_manager: The reward manager to use. Default: `sequence-level`.
    num_generations: The number of times the policy generates multiple responses
      for a given prompt within a single training step. This corresponds to the
      number of samples used to compute relative advantages.
    num_iterations: The number of iterations per batch.
    beta: The coefficient for the KL divergence penalty in the loss function.
      This term prevents policy updates from deviating too far from the
      reference model. A value of 0.0 means no KL penalty is applied.
    epsilon: Epsilon value for clipping in the loss function. Similar to
      PPO, it ensures stable updates.

  References:
    - MaxRL: https://arxiv.org/abs/2602.02710
  """

  algo_variant: str = "maxrl"
  advantage_estimator: str = "maxrl"
  policy_loss_fn: str = "grpo"
  loss_agg_mode: str = "sequence-mean-token-mean"
  reward_manager: str = "sequence-level"
  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.04
  epsilon: float = 0.2

  def __post_init__(self):
    if self.num_generations <= 1:
      raise ValueError(
          "num_generations must be greater than 1. Received: "
          f"{self.num_generations}"
      )


TMaxRLConfig = TypeVar("TMaxRLConfig", bound=MaxRLConfig)


class MaxRLLearner(rl_learner.RLLearner[TMaxRLConfig]):
  """MaxRL (Maximum Likelihood Reinforcement Learning) learner.

  MaxRL is a reinforcement learning algorithm designed to enhance reasoning
  abilities of large language models. It is a variant of policy gradient
  methods that uses a unique advantage normalization strategy: instead of
  normalizing by standard deviation (as in GRPO), MaxRL normalizes by the mean
  of the rewards. This approach is motivated by maximum likelihood principles
  and provides better theoretical properties for certain tasks.

  The key difference from GRPO is in the advantage computation:
  - GRPO: advantage = (reward - mean) / std
  - MaxRL: advantage = (reward - mean) / mean

  This makes MaxRL particularly effective for tasks where the reward magnitude
  is meaningful and should influence the policy update strength.

  References:
    - Paper: https://arxiv.org/abs/2602.02710
    - Code: https://github.com/tajwarfahim/maxrl
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TMaxRLConfig,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `MaxRLLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: An instance of `MaxRLConfig` containing all
        training-specific configuration options.
      reward_fns: A single callable or a list of callables that compute a
        scalar reward for given prompts and completions. Each function should
        accept `prompts`, `completions` and optional keyword arguments, and
        return a list of float rewards.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept ``prompts``, ``completions``,
        ``rewards``, ``advantages`` and optional keyword arguments, and return
        a dictionary of metric names to tuples of
        ``(metric_value, aggregation_fn)``:

           >>> def metric_fn(
           ...     prompts, completions, rewards, advantages, **kargs
           ... ):
           ...     return {
           ...       # ...
           ...       "prompt_min_len": (min(len(p) for p in prompts), np.min),
           ...       # ... }
      data_shuffle_seed: The seed used to shuffle the training data.
    """  # fmt: skip
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )

    policy_loss_fn = function_registry.get_policy_loss_fn(
        self.algo_config.policy_loss_fn
    )

    # Configure the loss function with maxrl-specific parameters
    loss_fn = lambda model, train_example, algo_config: policy_loss_fn(
        model,
        train_example,
        algo_config=self.algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
    )

    self.rl_cluster.actor_trainer.with_loss_fn(
        loss_fn,
        has_aux=True,
    )
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "algo_config": self.algo_config,
        }
    )
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({"kl": np.mean})
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        lambda: "kl" if self.algo_config.beta != 0.0 else None,
    ])

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Generate text completions and compute the advantages for MaxRL training.

    Args:
      training_input: A dictionary containing the training input data,
        containing the key 'prompts'.
      mode: The mode to use for logging metrics.

    Returns:
      A `TrainExample` instance containing the processed input data, including
      prompt IDs, completion IDs, masks, advantages, and per-token log
      probabilities from the reference and policy models.
    """
    training_input["prompts"] = list(training_input["prompts"])
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()
    rollout_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
        mode=mode,
        micro_batch_size=(
            self._rollout_micro_batch_size * self.algo_config.num_generations
        ),
    )
    completion_ids = rollout_output.tokens
    prompt_ids = jnp.array(rollout_output.left_padded_prompt_tokens)
    completion_text = rollout_output.text

    # Assemble masks
    prompt_mask = prompt_ids != pad_value
    completion_padding_mask = np.not_equal(completion_ids, pad_value)
    completion_mask = common.np_make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    # Apply the padding mask to the completion mask.
    completion_mask = completion_mask * completion_padding_mask

    # Convert completion_ids and completion_mask to jax arrays
    jax_completion_ids = jnp.array(completion_ids)
    jax_completion_mask = jnp.array(completion_mask)

    if self.algo_config.beta != 0.0:
      devices = self.rl_cluster.r2m[rl_cluster_lib.Role.REFERENCE].devices
      with self.rl_cluster.perf.span("refer_inference", devices) as interval:
        ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=jax_completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=(
                self._compute_logps_micro_batch_size
                * self.algo_config.num_generations
            ),
        )
        interval.device_end([ref_per_token_logps])
    else:
      ref_per_token_logps = None
    if self.algo_config.num_iterations > 1:
      devices = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR].devices
      with self.rl_cluster.perf.span(
          "old_actor_inference", devices
      ) as interval:
        old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=jax_completion_ids,
            micro_batch_size=(
                self._compute_logps_micro_batch_size
                * self.algo_config.num_generations
            ),
        )
        interval.device_end([old_per_token_logps])
    else:
      old_per_token_logps = None

    with self.rl_cluster.perf.span("advantage_computation"):
      # Compute rewards and advantages
      rewards = self._compute_rewards(
          prompts=training_input["prompts"],
          completions=completion_text,
          mode=mode,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      advantage_estimator = function_registry.get_advantage_estimator(
          self.algo_config.advantage_estimator
      )
      advantages = advantage_estimator(
          rewards=rewards, num_generations=self.algo_config.num_generations
      )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )
    for m_fn in self.metric_fns:
      user_defined_metric = m_fn(
          prompts=training_input["prompts"],
          completions=completion_text,
          advances=advantages,
          rewards=rewards,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      self.rl_cluster.buffer_metrics(user_defined_metric, mode=mode)

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=jax_completion_ids,
        completion_mask=jax_completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=jax.device_put(advantages),
        old_per_token_logps=old_per_token_logps,
    )

  def _compute_trajectory_ids(
      self, example: TrainingInputT, steps: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch.

    Trajectory id is a string of format {row_offset}_{group_offset} where
    row_offset is the row index of the example data source and
    group_offset is the group index of the example in the generation group.

    Args:
      example: The training input data.
      steps: The number of steps taken so far.

    Returns:
      A list of trajectory IDs, one for each prompt in the batch.
    """
    batch_size = len(example["prompts"]) // self.algo_config.num_generations
    row_offset = steps * batch_size
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.algo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.algo_config.num_generations),
        batch_size,
    )
    return [
        f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)
    ]

  def _num_iterations(self) -> int:
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    return self.algo_config.num_generations

  def train(  # pylint: disable=useless-parent-delegation
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """MaxRL training loop.

    MaxRL follows a similar training loop to GRPO but with a different
    advantage estimation strategy based on maximum likelihood principles.

    Args:
      train_ds: An iterable of training input data, where each element is a
        dictionary containing the key 'prompts'.
      eval_ds: An iterable of evaluation input data, where each element is a
        dictionary containing the key 'prompts'.
      skip_jit: Whether to skip JIT compilation of the training loop.
    """
    super().train(train_ds, eval_ds, skip_jit)


@function_registry.register_advantage_estimator("maxrl")
def compute_maxrl_advantages(
    rewards: np.ndarray, num_generations: int
) -> np.ndarray:
  """Compute MaxRL advantages.

  MaxRL uses a unique advantage normalization strategy where advantages are
  normalized by the mean instead of the standard deviation. This is motivated
  by maximum likelihood principles and provides better theoretical properties.

  The formula is:
    advantage = (reward - mean_reward) / (mean_reward + epsilon)

  This differs from GRPO which uses:
    advantage = (reward - mean_reward) / (std_reward + epsilon)

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    MaxRL advantages normalized by mean.
  """
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=-1)
  
  mean_grouped_rewards = mean_grouped_rewards.repeat(num_generations)
  
  # MaxRL normalization: divide by mean instead of std
  # This provides better scaling based on the absolute reward magnitude
  return (rewards - mean_grouped_rewards) / (mean_grouped_rewards + 1e-4)


# Aliases for backward compatibility
MaxrlConfig = MaxRLConfig
MaxrlLearner = MaxRLLearner
