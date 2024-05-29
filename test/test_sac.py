import unittest
from functools import partial

import jax

from rejax import SAC, SACConfig

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


class TestEnvironmentsSAC(unittest.TestCase):
    args = {
        "learning_rate": 0.01,
        "total_timesteps": 10_000,
        "eval_freq": 10_000,
        "buffer_size": 1000,
        "fill_buffer": 100,
        "batch_size": 100,
        "target_entropy_ratio": 0.0,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, config):
        return SAC.train(config, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        for discrete, env in enumerate([TestEnv1Continuous(), TestEnv1Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                config = SACConfig.create(env=env, **self.args)
                ts, _ = self.train_fn(config)
                act = SAC.make_act(config, ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.numpy.zeros((10, 1))
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: config.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

                q1, q2 = q_fn(ts.critic_ts.params, obs, actions)
                value = jax.numpy.minimum(q1, q2)

                for v in value:
                    self.assertAlmostEqual(v, 1.0, delta=0.1)

    def test_env2(self):
        for discrete, env in enumerate([TestEnv2Continuous(), TestEnv2Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                config = SACConfig.create(env=env, **self.args)
                ts, _ = self.train_fn(config)
                act = SAC.make_act(config, ts)

                rng = jax.random.PRNGKey(0)
                rngs = jax.random.split(rng, 10)
                obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
                actions = jax.vmap(act)(obs, rngs)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: config.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    actions = jax.numpy.expand_dims(actions, 1)
                    q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

                q1, q2 = q_fn(ts.critic_ts.params, obs, actions)
                value = jax.numpy.minimum(q1, q2)

                for v, r in zip(value, obs):
                    self.assertAlmostEqual(v, r, delta=0.1)

    def test_env3(self):
        for discrete, env in enumerate([TestEnv3Continuous(), TestEnv3Discrete()]):
            with self.subTest(discrete=bool(discrete)):
                config = SACConfig.create(env=env, **self.args)
                ts, _ = self.train_fn(config)
                act = SAC.make_act(config, ts)

                if discrete:
                    q_fn = jax.vmap(
                        lambda *args: config.critic.apply(*args, method="take"),
                        in_axes=(0, None, None),
                    )
                else:
                    q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

                @partial(jax.vmap, in_axes=(None, 0))
                def test_i(obs, rng):
                    action = act(obs, rng)
                    action = jax.numpy.expand_dims(action, 0)
                    obs = jax.numpy.expand_dims(obs, 0)
                    if not discrete:
                        action = jax.numpy.expand_dims(action, 1)

                    q1, q2 = q_fn(ts.critic_ts.params, obs, action)
                    value = jax.numpy.minimum(q1, q2)
                    return value

                rngs = jax.random.split(jax.random.PRNGKey(0), 10)
                for obs in jax.numpy.array([[-1], [1]]):
                    r = 1 * config.gamma if obs == -1 else 1
                    for v in test_i(obs, rngs):
                        self.assertAlmostEqual(v, r, delta=0.1)

    # TODO: The following tests does not work well, presumably because of the entropy
    # term in the optimization objective

    # def test_env4(self):
    #     for discrete, env in enumerate([TestEnv4Continuous(), TestEnv4Discrete()]):
    #         with self.subTest(discrete=bool(discrete)):
    #             config = SACConfig.create(env=env, **self.args)
    #             ts, _ = self.train_fn(config)
    #             act = SAC.make_act(config, ts)

    #             if discrete:
    #                 q_fn = jax.vmap(
    #                     lambda *args: config.critic.apply(*args, method="take"),
    #                     in_axes=(0, None, None),
    #                 )
    #             else:
    #                 q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

    #             @partial(jax.vmap, in_axes=(None, 0))
    #             def test_i(obs, rng):
    #                 action = act(obs, rng)
    #                 action = jax.numpy.expand_dims(action, 0)
    #                 obs = jax.numpy.expand_dims(obs, 0)
    #                 if not discrete:
    #                     action = jax.numpy.expand_dims(action, 1)

    #                 q1, q2 = q_fn(ts.critic_ts.params, obs, action)
    #                 value = jax.numpy.minimum(q1, q2)
    #                 return value, action

    #             rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    #             for obs in jax.numpy.array([[-1], [1]]):
    #                 threshold = 0.5 if discrete else 0.0
    #                 vv, aa = test_i(obs, rngs)
    #                 for v, a in zip(vv, aa):
    #                     # self.assertGreaterEqual(v, threshold)
    #                     self.assertAlmostEqual(v, a, delta=0.1)
    #                     self.assertGreaterEqual(a, threshold)

    # best_action = jax.numpy.array(1.0 if discrete else 2.0)
    # value = config.critic.apply(ts.critic_ts.params, jax.numpy.array([0]))
    # self.assertAlmostEqual(value, best_action, delta=0.1)

    # act = SAC.make_act(config, ts)
    # rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    # actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

    # for a in actions:
    #     self.assertAlmostEqual(a, best_action, delta=0.1)

    # def test_env5(self):
    #     for discrete, env in enumerate([TestEnv5Continuous(), TestEnv5Discrete()]):
    #         with self.subTest(discrete=bool(discrete)):
    #             config = SACConfig.create(env=env, **self.args)
    #             ts, _ = self.train_fn(config)

    #             rng = jax.random.PRNGKey(0)
    #             obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

    #             if not discrete:
    #                 value = config.critic.apply(ts.critic_ts.params, obs)
    #                 for v in value:
    #                     self.assertAlmostEqual(v, 0.0, delta=0.1)

    #             act = SAC.make_act(config, ts)
    #             rngs = jax.random.split(rng, 10)
    #             actions = jax.vmap(act)(obs, rngs)

    #             for o, a in zip(obs, actions):
    #                 if discrete:
    #                     self.assertEqual(a > 0.5, o > 0)
    #                 else:
    #                     self.assertAlmostEqual(a, o, delta=0.2)
    #                     self.assertAlmostEqual(a, o, delta=0.2)
