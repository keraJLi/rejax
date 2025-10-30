import unittest

import jax

from rejax.compat.gymnasium2gymnax import create_gymnasium


class TestGymnasiumCompat(unittest.TestCase):
    def test_create_gymnasium_environments(self):
        """Test creating and basic functionality of Gymnasium environments."""
        rng = jax.random.PRNGKey(0)

        # Test common Gymnasium environments
        # fmt: off
        gymnasium_envs = [
            "CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1",
        ]
        # fmt: on

        for env_name in gymnasium_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create_gymnasium(env_name)
                except Exception as e:
                    # Skip environments that might not be available
                    self.skipTest(f"Environment {env_name} not available: {e}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test reset
                try:
                    obs, state = jitted_reset(rng, params)
                except Exception as e:
                    self.fail(f"Failed to reset {env_name}: {type(e).__name__}: {e}")

                # Test observation space
                try:
                    obs_space = env.observation_space(params)
                    self.assertEqual(obs.dtype, obs_space.dtype)
                    self.assertEqual(obs.shape, obs_space.shape)
                except Exception as e:
                    self.fail(
                        f"Failed to get obs space for {env_name}: "
                        f"{type(e).__name__}: {e}"
                    )

                # Test action space and sampling
                try:
                    action_space = env.action_space(params)
                    action = action_space.sample(rng)
                    self.assertEqual(action.dtype, action_space.dtype)
                    self.assertEqual(action.shape, action_space.shape)
                except Exception as e:
                    self.fail(
                        f"Failed to sample action for {env_name}: "
                        f"{type(e).__name__}: {e}"
                    )

                # Test stepping
                for step in range(10):
                    try:
                        obs, state, reward, done, info = jitted_step(
                            rng, state, action, params
                        )

                        # Check types and shapes
                        self.assertEqual(obs.dtype, obs_space.dtype)
                        self.assertEqual(obs.shape, obs_space.shape)
                        self.assertTrue(hasattr(reward, "dtype"))
                        self.assertTrue(hasattr(done, "dtype"))

                        # If episode is done, break
                        if done:
                            break

                    except Exception as e:
                        self.fail(
                            f"Failed to step {env_name} at step {step}: "
                            f"{type(e).__name__}: {e}"
                        )

                    # Sample new action for next step
                    action = action_space.sample(rng)

    def test_gymnasium_discrete_environments(self):
        """Test discrete action environments specifically."""
        rng = jax.random.PRNGKey(0)

        # fmt: off
        discrete_envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
        # fmt: on

        for env_name in discrete_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create_gymnasium(env_name)
                except Exception:
                    self.skipTest(f"Environment {env_name} not available")

                # Test that action space is discrete
                action_space = env.action_space(params)
                from gymnax.environments.spaces import Discrete

                self.assertIsInstance(action_space, Discrete)

                # Test action sampling and stepping
                obs, state = env.reset(rng, params)
                action = action_space.sample(rng)
                obs, state, reward, done, info = env.step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_gymnasium_continuous_environments(self):
        """Test continuous action environments specifically."""
        rng = jax.random.PRNGKey(0)

        continuous_envs = ["Pendulum-v1"]

        for env_name in continuous_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create_gymnasium(env_name)
                except Exception:
                    self.skipTest(f"Environment {env_name} not available")

                # Test that action space is continuous (Box)
                action_space = env.action_space(params)
                from gymnax.environments.spaces import Box

                self.assertIsInstance(action_space, Box)

                # Test action sampling and stepping
                obs, state = env.reset(rng, params)
                action = action_space.sample(rng)
                obs, state, reward, done, info = env.step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_gymnasium_env_params(self):
        """Test that environment parameters work correctly."""
        rng = jax.random.PRNGKey(0)

        try:
            env, params = create_gymnasium("CartPole-v1")
        except Exception:
            self.skipTest("CartPole environment not available")

        # Test reset with params
        obs, state = env.reset(rng, params)
        self.assertIsNotNone(obs)
        self.assertIsNotNone(state)

        # Test that we can modify max_steps_in_episode
        if hasattr(params, "max_steps_in_episode"):
            self.assertIsInstance(params.max_steps_in_episode, int)
            self.assertGreater(params.max_steps_in_episode, 0)


if __name__ == "__main__":
    unittest.main()
