import unittest

import jax

from rejax import PQN

from .environments import (
    TestEnv1Discrete,
    TestEnv2Discrete,
    TestEnv3Discrete,
    TestEnv4Discrete,
    TestEnv5Discrete,
)


class TestEnvironmentsPQN(unittest.TestCase):
    args = {
        "num_envs": 64,
        "num_steps": 16,
        "num_epochs": 10,
        "learning_rate": 0.0003,
        "total_timesteps": 131072,
        "eval_freq": 131072,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, pqn):
        return PQN.train(pqn, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        env = TestEnv1Discrete()
        pqn = PQN.create(env=env, **self.args)
        ts, _ = self.train_fn(pqn)
        value = pqn.agent.apply(ts.q_ts.params, jax.numpy.array([0]))
        self.assertAlmostEqual(value, 1.0, delta=0.1)

    def test_env2(self):
        env = TestEnv2Discrete()
        pqn = PQN.create(env=env, **self.args)
        ts, _ = self.train_fn(pqn)

        obs = jax.numpy.array([[-1], [1]])
        rew = obs
        value = pqn.agent.apply(ts.q_ts.params, obs)

        for v, r in zip(value, rew):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env3(self):
        env = TestEnv3Discrete()
        pqn = PQN.create(env=env, **self.args)
        ts, _ = self.train_fn(pqn)

        obs = jax.numpy.array([[-1], [1]])
        rew = [1 * pqn.gamma, 1]
        value = pqn.agent.apply(ts.q_ts.params, obs)

        for v, r in zip(value, rew):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env4(self):
        env = TestEnv4Discrete()
        pqn = PQN.create(env=env, **self.args)
        ts, _ = self.train_fn(pqn)

        best_action = 1
        value = pqn.agent.apply(ts.q_ts.params, jax.numpy.array([0]))
        self.assertEqual(value.argmax(), best_action)

        act = pqn.make_act(ts)
        rngs = jax.random.split(jax.random.PRNGKey(0), 10)
        actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

        for a in actions:
            self.assertAlmostEqual(a, best_action, delta=0.1)

    def test_env5(self):
        env = TestEnv5Discrete()
        pqn = PQN.create(env=env, **self.args)
        ts, _ = self.train_fn(pqn)

        rng = jax.random.PRNGKey(0)
        obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

        act = pqn.make_act(ts)
        rngs = jax.random.split(rng, 10)
        actions = jax.vmap(act)(obs, rngs)

        for o, a in zip(obs, actions):
            self.assertEqual(a > 0.5, o > 0)
