import unittest

import jax

from rejax import PPO

from .environments import (
    TestEnv1Continuous,
    TestEnv1Discrete,
    TestEnv2Continuous,
    TestEnv2Discrete,
    TestEnv3Continuous,
    TestEnv3Discrete,
    TestEnv4Continuous,
    TestEnv4Discrete,
    TestEnv5Continuous,
    TestEnv5Discrete,
)


class TestEnvironmentsPPO(unittest.TestCase):
    args = {
        "num_envs": 64,
        "num_steps": 16,
        "num_epochs": 10,
        "learning_rate": 0.0003,
        "total_timesteps": 131072,
        "eval_freq": 131072,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, ppo):
        return PPO.train(ppo, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                ppo = PPO.create(env=env, **self.args)
                ts, _ = self.train_fn(ppo)
                value = ppo.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
                self.assertAlmostEqual(value, 1.0, delta=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                ppo = PPO.create(env=env, **self.args)
                ts, _ = self.train_fn(ppo)

                obs = jax.numpy.array([[-1], [1]])
                rew = obs
                value = ppo.critic.apply(ts.critic_ts.params, obs)

                for v, r in zip(value, rew):
                    self.assertAlmostEqual(v, r, delta=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                ppo = PPO.create(env=env, **self.args)
                ts, _ = self.train_fn(ppo)

                obs = jax.numpy.array([[-1], [1]])
                rew = [1 * ppo.gamma, 1]
                value = ppo.critic.apply(ts.critic_ts.params, obs)

                for v, r in zip(value, rew):
                    self.assertAlmostEqual(v, r, delta=0.1)

    def test_env4(self):
        for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                ppo = PPO.create(env=env, **self.args)
                ts, _ = self.train_fn(ppo)

                best_action = jax.numpy.array(1.0 if discrete else 2.0)
                value = ppo.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
                self.assertAlmostEqual(value, best_action, delta=0.1)

                act = ppo.make_act(ts)
                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

                for a in actions:
                    self.assertAlmostEqual(a, best_action, delta=0.1)

    def test_env5(self):
        for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                ppo = PPO.create(env=env, **self.args)
                ts, _ = self.train_fn(ppo)

                rng = jax.random.PRNGKey(0)
                if not discrete:
                    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                else:
                    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

                if not discrete:
                    value = ppo.critic.apply(ts.critic_ts.params, obs)
                    for v in value:
                        self.assertAlmostEqual(v, 0.0, delta=0.1)

                act = ppo.make_act(ts)
                rngs = jax.random.split(rng, 10)
                actions = jax.vmap(act)(obs, rngs)

                for o, a in zip(obs, actions):
                    if discrete:
                        self.assertEqual(a > 0.5, o > 0)
                    else:
                        self.assertAlmostEqual(a, o, delta=0.2)
