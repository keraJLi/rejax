import unittest
from functools import partial

import jax

from rejax import TD3

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


class TestEnvironmentsTD3(unittest.TestCase):
    args = {
        "num_envs": 1,
        "learning_rate": 0.0003,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, td3):
        return TD3.train(td3, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        env = TestEnv1Continuous()
        td3 = TD3.create(env=env, **self.args)
        ts, _ = self.train_fn(td3)
        act = td3.make_act(ts)

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 10)
        obs = jax.numpy.zeros((10, 1))
        actions = jax.vmap(act)(obs, rngs)

        actions = jax.numpy.expand_dims(actions, 1)
        q_fn = jax.vmap(td3.critic.apply, in_axes=(0, None, None))

        qs = q_fn(ts.critic_ts.params, obs, actions)
        value = qs.min(axis=0)

        for v in value:
            self.assertAlmostEqual(v, 1.0, delta=0.1)

    def test_env2(self):
        env = TestEnv2Continuous()
        td3 = TD3.create(env=env, **self.args)
        ts, _ = self.train_fn(td3)
        act = td3.make_act(ts)

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 10)
        obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
        actions = jax.vmap(act)(obs, rngs)
        actions = jax.numpy.expand_dims(actions, 1)
        q_fn = jax.vmap(td3.critic.apply, in_axes=(0, None, None))

        qs = q_fn(ts.critic_ts.params, obs, actions)
        value = qs.min(axis=0)

        for v, r in zip(value, obs):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env3(self):
        env = TestEnv3Continuous()
        td3 = TD3.create(env=env, **self.args)
        ts, _ = self.train_fn(td3)
        act = td3.make_act(ts)
        q_fn = jax.vmap(td3.critic.apply, in_axes=(0, None, None))

        @partial(jax.vmap, in_axes=(None, 0))
        def test_i(obs, rng):
            action = act(obs, rng)
            action = jax.numpy.expand_dims(action, 0)
            obs = jax.numpy.expand_dims(obs, 0)
            action = jax.numpy.expand_dims(action, 1)

            qs = q_fn(ts.critic_ts.params, obs, action)
            value = qs.min(axis=0)
            return value

        rngs = jax.random.split(jax.random.PRNGKey(0), 10)
        for obs in jax.numpy.array([[-1], [1]]):
            r = 1 * td3.gamma if obs == -1 else 1
            for v in test_i(obs, rngs):
                self.assertAlmostEqual(v, r, delta=0.1)

    def test_env4(self):
        env = TestEnv4Continuous()
        td3 = TD3.create(env=env, **self.args)
        ts, _ = self.train_fn(td3)
        act = td3.make_act(ts)
        q_fn = jax.vmap(td3.critic.apply, in_axes=(0, None, None))

        @partial(jax.vmap, in_axes=(None, 0))
        def test_i(obs, rng):
            action = act(obs, rng)
            action = jax.numpy.expand_dims(action, 0)
            obs = jax.numpy.expand_dims(obs, 0)
            action = jax.numpy.expand_dims(action, 1)

            qs = q_fn(ts.critic_ts.params, obs, action)
            value = qs.min(axis=0)
            return value, action

        rngs = jax.random.split(jax.random.PRNGKey(0), 10)
        obs = jax.numpy.array([0])
        vv, aa = test_i(obs, rngs)
        for v, a in zip(vv, aa):
            self.assertGreaterEqual(v, 1.0)  # very conservative, because minimum
            self.assertAlmostEqual(a, 2.0, delta=0.1)

    def test_env5(self):
        env = TestEnv5Continuous()
        td3 = TD3.create(env=env, **self.args)
        ts, _ = self.train_fn(td3)
        act = td3.make_act(ts)
        q_fn = jax.vmap(td3.critic.apply, in_axes=(0, None, None))

        @partial(jax.vmap, in_axes=(None, 0))
        def test_i(obs, rng):
            action = act(obs, rng)
            action = jax.numpy.expand_dims(action, 0)
            obs = jax.numpy.expand_dims(obs, 0)
            action = jax.numpy.expand_dims(action, 1)

            qs = q_fn(ts.critic_ts.params, obs, action)
            value = qs.min(axis=0)
            return value, action.squeeze(1)

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 10)
        obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
        for o in obs:
            vv, aa = test_i(o, rngs)
            for v, a in zip(vv, aa):
                self.assertAlmostEqual(v, 0.0, delta=0.1)
                self.assertAlmostEqual(a, o, delta=0.1)
