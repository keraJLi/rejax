import unittest
from functools import partial

import jax

from rejax import TD3, TD3Config

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


class TestEnvironmentsTD3(unittest.TestCase):
    args = {
        "learning_rate": 0.001,
        "total_timesteps": 50_000,
        "eval_freq": 50_000,
        "buffer_size": 1000,
        "fill_buffer": 100,
        "batch_size": 100,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, config):
        return TD3.train(config, rng=jax.random.PRNGKey(0))

    # def test_env1(self):
    #     env = TestEnv1Continuous()
    #     config = TD3Config.create(env=env, **self.args)
    #     ts, _ = self.train_fn(config)
    #     act = TD3.make_act(config, ts)

    #     rng = jax.random.PRNGKey(0)
    #     rngs = jax.random.split(rng, 10)
    #     obs = jax.numpy.zeros((10, 1))
    #     actions = jax.vmap(act)(obs, rngs)
        
    #     actions = jax.numpy.expand_dims(actions, 1)
    #     q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

    #     q1, q2 = q_fn(ts.critic_ts.params, obs, actions)
    #     value = jax.numpy.minimum(q1, q2)

    #     for v in value:
    #         self.assertAlmostEqual(v, 1.0, delta=0.1)

    # def test_env2(self):
    #     env = TestEnv2Continuous()
    #     config = TD3Config.create(env=env, **self.args)
    #     ts, _ = self.train_fn(config)
    #     act = TD3.make_act(config, ts)

    #     rng = jax.random.PRNGKey(0)
    #     rngs = jax.random.split(rng, 10)
    #     obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    #     actions = jax.vmap(act)(obs, rngs)
    #     actions = jax.numpy.expand_dims(actions, 1)
    #     q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

    #     q1, q2 = q_fn(ts.critic_ts.params, obs, actions)
    #     value = jax.numpy.minimum(q1, q2)

    #     for v, r in zip(value, obs):
    #         self.assertAlmostEqual(v, r, delta=0.1)

    # def test_env3(self):
    #     env = TestEnv3Continuous()
    #     config = TD3Config.create(env=env, **self.args)
    #     ts, _ = self.train_fn(config)
    #     act = TD3.make_act(config, ts)
    #     q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

    #     @partial(jax.vmap, in_axes=(None, 0))
    #     def test_i(obs, rng):
    #         action = act(obs, rng)
    #         action = jax.numpy.expand_dims(action, 0)
    #         obs = jax.numpy.expand_dims(obs, 0)
    #         action = jax.numpy.expand_dims(action, 1)

    #         q1, q2 = q_fn(ts.critic_ts.params, obs, action)
    #         value = jax.numpy.minimum(q1, q2)
    #         return value

    #     rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    #     for obs in jax.numpy.array([[-1], [1]]):
    #         r = 1 * config.gamma if obs == -1 else 1
    #         for v in test_i(obs, rngs):
    #             self.assertAlmostEqual(v, r, delta=0.1)

    # def test_env4(self):
        # env = TestEnv4Continuous()
        # config = TD3Config.create(env=env, **self.args)
        # ts, _ = self.train_fn(config)
        # act = TD3.make_act(config, ts)
        # q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

        # @partial(jax.vmap, in_axes=(None, 0))
        # def test_i(obs, rng):
        #     action = act(obs, rng)
        #     action = jax.numpy.expand_dims(action, 0)
        #     obs = jax.numpy.expand_dims(obs, 0)
        #     action = jax.numpy.expand_dims(action, 1)

        #     q1, q2 = q_fn(ts.critic_ts.params, obs, action)
        #     value = jax.numpy.minimum(q1, q2)
        #     return value, action

        # rngs = jax.random.split(jax.random.PRNGKey(0), 10)
        # for obs in jax.numpy.array([[-1], [1]]):
        #     vv, aa = test_i(obs, rngs)
        #     for v, a in zip(vv, aa):
        #         self.assertGreaterEqual(v, 1.0)  # very conservative, because minimum
        #         self.assertAlmostEqual(a, 2.0, delta=0.1)

    def test_env5(self):
        env = TestEnv5Continuous()
        config = TD3Config.create(env=env, **self.args)
        ts, _ = self.train_fn(config)
        act = TD3.make_act(config, ts)
        q_fn = jax.vmap(config.critic.apply, in_axes=(0, None, None))

        @partial(jax.vmap, in_axes=(None, 0))
        def test_i(obs, rng):
            action = act(obs, rng)
            action = jax.numpy.expand_dims(action, 0)
            obs = jax.numpy.expand_dims(obs, 0)
            action = jax.numpy.expand_dims(action, 1)

            q1, q2 = q_fn(ts.critic_ts.params, obs, action)
            value = jax.numpy.minimum(q1, q2)
            return value, action.squeeze(1)

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 10)
        obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
        for o in obs:
            vv, aa = test_i(o, rngs)
            for v, a in zip(vv, aa):
                self.assertAlmostEqual(v, 0.0, delta=0.1)
                self.assertAlmostEqual(a, o, delta=0.1)
