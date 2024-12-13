import unittest
from typing import NamedTuple

import chex
import jax

from rejax.algos.mixins import NormalizeRewardsMixin, RewardRMSState
from rejax.compat import create


class EpisodeState(NamedTuple):
    rng: chex.PRNGKey
    rms_state: RewardRMSState
    return_: chex.Array
    step: chex.Array


class TestRewardNormalization(unittest.TestCase):
    def test_reward_normalization(self):
        num_envs = 128
        gamma = 0.9

        mixin = NormalizeRewardsMixin()
        mixin.__dict__["num_envs"] = num_envs  # hacky, since frozen
        mixin.__dict__["reward_normalization_discount"] = gamma

        def complicated_reward_fn(i, rng):
            return 10 * (i - 5) + i**2 * jax.numpy.sin(i) * jax.random.normal(
                rng, (num_envs,)
            )

        def update(ep_state):
            rng, rng_reward = jax.random.split(ep_state.rng)
            step = jax.numpy.mod(ep_state.step + 1, 20)
            done = jax.numpy.where(step == 0, 1, 0)

            reward = complicated_reward_fn(step, rng_reward)
            rms_state = mixin.update_rew_rms(ep_state.rms_state, reward, done)
            reward = mixin.normalize_rew(rms_state, reward)

            return_ = reward + (1 - done) * gamma * ep_state.return_
            return EpisodeState(rng, rms_state, return_, step), return_

        rng = jax.random.PRNGKey(0)
        rms_state = mixin.initialize_reward_rms_state(rng)["rew_rms_state"]
        ep_state = EpisodeState(
            rng, rms_state, jax.numpy.zeros(num_envs), jax.numpy.zeros(num_envs)
        )

        new_ep_state = jax.lax.fori_loop(0, 300, lambda _, s: update(s)[0], ep_state)
        ep_state = ep_state._replace(rms_state=new_ep_state.rms_state)

        rolling_returns = jax.lax.scan(lambda s, _: update(s), ep_state, None, 99)[1]

        self.assertAlmostEqual(rolling_returns.var(), 1.0, delta=0.1)
