import unittest

import jax

from rejax import IQN

from .environments import (
    TestEnv1Discrete,
    TestEnv2Discrete,
    TestEnv3Discrete,
    TestEnv4Discrete,
    TestEnv5Discrete,
)


class TestEnvironmentsIQN(unittest.TestCase):
    args = {
        "learning_rate": 0.0003,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "skip_initial_evaluation": True,
    }

    rng = jax.random.PRNGKey(0)

    def train_fn(self, iqn):
        return IQN.train(iqn, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        env = TestEnv1Discrete()
        iqn = IQN.create(env=env, **self.args)
        ts, _ = self.train_fn(iqn)
        value = iqn.agent.apply(
            ts.q_ts.params, jax.numpy.array([0]), self.rng, method="q"
        )
        self.assertAlmostEqual(value, 1.0, delta=0.1)

    def test_env2(self):
        env = TestEnv2Discrete()
        iqn = IQN.create(env=env, **self.args)
        ts, _ = self.train_fn(iqn)

        obs = jax.numpy.array([[-1], [1]])
        rew = obs
        value = iqn.agent.apply(ts.q_ts.params, obs, self.rng, method="q")

        for v, r in zip(value, rew):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env3(self):
        env = TestEnv3Discrete()
        iqn = IQN.create(env=env, **self.args)
        ts, _ = self.train_fn(iqn)

        obs = jax.numpy.array([[-1], [1]])
        rew = [1 * iqn.gamma, 1]
        value = iqn.agent.apply(ts.q_ts.params, obs, self.rng, method="q")

        for v, r in zip(value, rew):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env4(self):
        env = TestEnv4Discrete()
        iqn = IQN.create(env=env, **self.args)
        ts, _ = self.train_fn(iqn)

        best_action = 1
        value = iqn.agent.apply(
            ts.q_ts.params, jax.numpy.array([0]), self.rng, method="q"
        )
        self.assertEqual(value.argmax(), best_action)

        act = iqn.make_act(ts)
        rngs = jax.random.split(jax.random.PRNGKey(0), 10)
        actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

        for a in actions:
            self.assertAlmostEqual(a, best_action, delta=0.1)

    def test_env5(self):
        env = TestEnv5Discrete()
        iqn = IQN.create(env=env, **self.args)
        ts, _ = self.train_fn(iqn)

        rng = jax.random.PRNGKey(0)
        obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

        act = iqn.make_act(ts)
        rngs = jax.random.split(rng, 10)
        actions = jax.vmap(act)(obs, rngs)

        for o, a in zip(obs, actions):
            self.assertEqual(a > 0.5, o > 0)
