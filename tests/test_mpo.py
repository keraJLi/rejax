import typing
import unittest

import distrax
import jax
from jax import numpy as jnp

from rejax import MPO
from rejax.algos.mpo import (
    compute_cross_entropy_loss,
    compute_parametric_kl_penalty_and_dual_loss,
    compute_weights_and_temperature_loss,
)

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


class TestMPOLossParity(unittest.TestCase):
    def test_temperature_weights_match_acme_formula(self):
        q_values = jnp.array([[1.0, 2.0], [3.0, 0.0], [2.0, 1.0]])
        epsilon = 0.1
        temperature = jnp.array(2.0)

        weights, loss = compute_weights_and_temperature_loss(
            q_values, epsilon, temperature
        )

        tempered_q = q_values / temperature
        expected_weights = jax.nn.softmax(tempered_q, axis=0)
        expected_loss = temperature * (
            epsilon
            + jnp.mean(jax.scipy.special.logsumexp(tempered_q, axis=0))
            - jnp.log(q_values.shape[0])
        )

        self.assertTrue(jnp.allclose(weights, expected_weights))
        self.assertTrue(jnp.allclose(loss, expected_loss))

    def test_cross_entropy_loss_matches_weighted_target_policy(self):
        sampled_actions = jnp.array([[[0.0]], [[1.0]]])
        normalized_weights = jnp.array([[0.25], [0.75]])
        dist = distrax.MultivariateNormalDiag(
            loc=jnp.array([[0.0]]), scale_diag=jnp.array([[1.0]])
        )

        loss = compute_cross_entropy_loss(sampled_actions, normalized_weights, dist)
        expected = -jnp.sum(dist.log_prob(sampled_actions) * normalized_weights, axis=0)

        self.assertTrue(jnp.allclose(loss, jnp.mean(expected)))

    def test_parametric_kl_dual_loss_matches_acme_formula(self):
        kl = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        alpha = jnp.array([2.0, 3.0])
        epsilon = 0.25

        loss_kl, loss_alpha = compute_parametric_kl_penalty_and_dual_loss(
            kl, alpha, epsilon
        )

        mean_kl = jnp.mean(kl, axis=0)
        self.assertTrue(jnp.allclose(loss_kl, jnp.sum(alpha * mean_kl)))
        self.assertTrue(jnp.allclose(loss_alpha, jnp.sum(alpha * (epsilon - mean_kl))))


class TestEnvironmentsMPO(unittest.TestCase):
    args: typing.ClassVar[dict] = {
        "num_envs": 1,
        "learning_rate": 3e-4,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "skip_initial_evaluation": True,
        "num_action_samples": 10,
        "policy_eval_num_val_samples": 10,
        "batch_size": 64,
        "fill_buffer": 256,
    }

    def train_fn(self, mpo):
        return MPO.train(mpo, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        mpo = MPO.create(env=TestEnv1Continuous(), **self.args)
        ts, _ = self.train_fn(mpo)
        act = mpo.make_act(ts)

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 10)
        obs = jax.numpy.zeros((10, 1))
        actions = jax.vmap(act)(obs, rngs)

        actions = jax.numpy.expand_dims(actions, 1)
        value = mpo.critic.apply(ts.critic_ts.params, obs, actions)

        for v in value:
            self.assertAlmostEqual(v, 1.0, delta=0.1)

    def test_env2(self):
        mpo = MPO.create(env=TestEnv2Continuous(), **self.args)
        ts, _ = self.train_fn(mpo)
        act = mpo.make_act(ts)

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 10)
        obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
        actions = jax.vmap(act)(obs, rngs)

        actions = jax.numpy.expand_dims(actions, 1)
        value = mpo.critic.apply(ts.critic_ts.params, obs, actions)

        for v, r in zip(value, obs):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env3(self):
        mpo = MPO.create(env=TestEnv3Continuous(), **self.args)
        self.train_fn(mpo)

    def test_env4(self):
        mpo = MPO.create(env=TestEnv4Continuous(), **self.args)
        self.train_fn(mpo)

    def test_env5(self):
        mpo = MPO.create(env=TestEnv5Continuous(), **self.args)
        ts, _ = self.train_fn(mpo)

        rng = jax.random.PRNGKey(0)
        obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
        value = mpo.critic.apply(ts.critic_ts.params, obs, obs)
        for v in value:
            self.assertAlmostEqual(v, 0.0, delta=0.1)
