import unittest

import jax

from rejax.compat.brax2gymnax import create_brax


class TestBraxCompat(unittest.TestCase):
    def test_create_brax_environments(self):
        """Test creating and basic functionality of Brax environments."""
        rng = jax.random.PRNGKey(0)

        # Test common Brax environments
        # fmt: off
        brax_envs = [
            "ant", "fast", "halfcheetah", "hopper", "humanoid", "humanoidstandup",
            "inverted_pendulum", "inverted_double_pendulum", "pusher", "reacher",
            "swimmer", "walker2d"
        ]
        # fmt: on
        for env_name in brax_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create_brax(env_name)
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
                        f"Failed to get obs space for {env_name}: {type(e).__name__}: {e}"
                    )

                # Test action space and sampling
                try:
                    action_space = env.action_space(params)
                    action = action_space.sample(rng)
                    self.assertEqual(action.dtype, action_space.dtype)
                    self.assertEqual(action.shape, action_space.shape)
                except Exception as e:
                    self.fail(
                        f"Failed to sample action for {env_name}: {type(e).__name__}: {e}"
                    )

                # Test stepping
                for step in range(3):
                    try:
                        obs, state, reward, done, info = jitted_step(
                            rng, state, action, params
                        )

                        # Check types and shapes
                        self.assertEqual(obs.dtype, obs_space.dtype)
                        self.assertEqual(obs.shape, obs_space.shape)
                        self.assertTrue(hasattr(reward, "dtype"))  # Should be JAX array
                        self.assertTrue(hasattr(done, "dtype"))  # Should be JAX array

                    except Exception as e:
                        self.fail(
                            f"Failed to step {env_name} at step {step}: {type(e).__name__}: {e}"
                        )

                    # Sample new action for next step
                    action = action_space.sample(rng)

    def test_brax_env_params(self):
        """Test that environment parameters work correctly."""
        rng = jax.random.PRNGKey(0)

        try:
            env, params = create_brax("ant")
        except Exception:
            self.skipTest("Ant environment not available")

        # Test that params have expected structure
        self.assertTrue(hasattr(params, "max_steps_in_episode"))
        self.assertIsInstance(params.max_steps_in_episode, int)
        self.assertGreater(params.max_steps_in_episode, 0)

        # Test reset with params
        obs, state = env.reset(rng, params)
        self.assertIsNotNone(obs)
        self.assertIsNotNone(state)


if __name__ == "__main__":
    unittest.main()
