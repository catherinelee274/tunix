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

"""Tests for MaxRL learner."""

import numpy as np
from absl.testing import absltest
from tunix.rl.maxrl import maxrl_learner


class MaxRLAdvantageTest(absltest.TestCase):
  """Test MaxRL advantage computation."""

  def test_compute_maxrl_advantages_basic(self):
    """Test basic MaxRL advantage computation."""
    # Create rewards for 2 prompts with 2 generations each
    rewards = np.array([1.0, 3.0, 2.0, 4.0])
    num_generations = 2
    
    advantages = maxrl_learner.compute_maxrl_advantages(
        rewards=rewards, num_generations=num_generations
    )
    
    # For first group: rewards [1.0, 3.0], mean = 2.0
    # advantages = [(1.0 - 2.0) / (2.0 + 1e-4), (3.0 - 2.0) / (2.0 + 1e-4)]
    #            = [-0.5, 0.5] approximately
    
    # For second group: rewards [2.0, 4.0], mean = 3.0
    # advantages = [(2.0 - 3.0) / (3.0 + 1e-4), (4.0 - 3.0) / (3.0 + 1e-4)]
    #            = [-0.333..., 0.333...] approximately
    
    self.assertEqual(advantages.shape, rewards.shape)
    
    # Check first group
    self.assertAlmostEqual(advantages[0], -0.5, places=3)
    self.assertAlmostEqual(advantages[1], 0.5, places=3)
    
    # Check second group
    self.assertAlmostEqual(advantages[2], -0.333, places=2)
    self.assertAlmostEqual(advantages[3], 0.333, places=2)

  def test_compute_maxrl_advantages_zero_mean(self):
    """Test MaxRL advantage computation when mean is near zero."""
    # Create rewards with zero mean in a group
    rewards = np.array([-1.0, 1.0])
    num_generations = 2
    
    advantages = maxrl_learner.compute_maxrl_advantages(
        rewards=rewards, num_generations=num_generations
    )
    
    # Mean = 0.0, so we use epsilon for stability
    # advantages = [(-1.0 - 0.0) / (0.0 + 1e-4), (1.0 - 0.0) / (0.0 + 1e-4)]
    
    self.assertEqual(advantages.shape, rewards.shape)
    # Both should have very large magnitude due to division by epsilon
    self.assertLess(advantages[0], 0)
    self.assertGreater(advantages[1], 0)

  def test_compute_maxrl_advantages_uniform_rewards(self):
    """Test MaxRL advantage computation with uniform rewards."""
    # All rewards are the same
    rewards = np.array([2.0, 2.0, 2.0, 2.0])
    num_generations = 2
    
    advantages = maxrl_learner.compute_maxrl_advantages(
        rewards=rewards, num_generations=num_generations
    )
    
    # All rewards equal mean, so advantages should all be 0
    self.assertEqual(advantages.shape, rewards.shape)
    np.testing.assert_array_almost_equal(advantages, np.zeros_like(rewards))

  def test_maxrl_vs_grpo_difference(self):
    """Test that MaxRL differs from GRPO in expected way."""
    from tunix.rl.grpo.grpo_learner import compute_advantages as grpo_compute_advantages
    
    rewards = np.array([1.0, 5.0, 2.0, 6.0])
    num_generations = 2
    
    maxrl_advantages = maxrl_learner.compute_maxrl_advantages(
        rewards=rewards, num_generations=num_generations
    )
    
    grpo_advantages = grpo_compute_advantages(
        rewards=rewards, num_generations=num_generations
    )
    
    # MaxRL and GRPO should give different results
    # (they normalize by mean vs std respectively)
    self.assertFalse(np.allclose(maxrl_advantages, grpo_advantages))


class MaxRLConfigTest(absltest.TestCase):
  """Test MaxRL configuration."""

  def test_maxrl_config_defaults(self):
    """Test MaxRL config default values."""
    config = maxrl_learner.MaxRLConfig()
    
    self.assertEqual(config.algo_variant, "maxrl")
    self.assertEqual(config.advantage_estimator, "maxrl")
    self.assertEqual(config.policy_loss_fn, "grpo")
    self.assertEqual(config.num_generations, 2)
    self.assertEqual(config.num_iterations, 1)
    self.assertEqual(config.beta, 0.04)
    self.assertEqual(config.epsilon, 0.2)

  def test_maxrl_config_validation(self):
    """Test MaxRL config validation."""
    # Should raise error if num_generations <= 1
    with self.assertRaises(ValueError):
      maxrl_learner.MaxRLConfig(num_generations=1)
    
    with self.assertRaises(ValueError):
      maxrl_learner.MaxRLConfig(num_generations=0)


if __name__ == "__main__":
  absltest.main()
