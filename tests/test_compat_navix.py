import unittest

import jax
from jax import numpy as jnp
from navix.environments.registry import registry

from rejax.compat.navix2gymnax import create_navix

# Standard set of Navix environments for testing: smallest size of every one
# fmt: off
NAVIX_TEST_ENVS = [
    "Navix-Empty-5x5-v0", "Navix-Empty-Random-5x5-v0", "Navix-DoorKey-5x5-v0",
    "Navix-DoorKey-Random-5x5-v0", "Navix-Dynamic-Obstacles-5x5-v0",
    "Navix-Dynamic-Obstacles-5x5-Random-v0", "Navix-LavaGapS5-v0",
    "Navix-SimpleCrossingS9N1-v0", "Navix-GoToDoor-5x5-v0", "Navix-KeyCorridorS3R1-v0",
    "Navix-DistShift1-v0", "Navix-DistShift2-v0", "Navix-FourRooms-v0",
]
# fmt: on


class TestNavixCompat(unittest.TestCase):
    def test_create_navix_environments(self):
        """Test creating and basic functionality of Navix environments."""
        rng = jax.random.PRNGKey(0)

        for env_name in NAVIX_TEST_ENVS:
            with self.subTest(env=env_name):
                env, params = create_navix(env_name)

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
                for step in range(5):
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

    def test_navix_float_obs_wrapper(self):
        """Test that FloatObsWrapper correctly converts observations to float."""
        rng = jax.random.PRNGKey(0)

        for env_name in NAVIX_TEST_ENVS:
            with self.subTest(env=env_name):
                env, params = create_navix(env_name)
                obs, state = env.reset(rng, params)

                # Check that observations are float type
                self.assertTrue(jnp.issubdtype(obs.dtype, jnp.floating))

                # Test that stepping also returns float observations
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = env.step(rng, state, action, params)
                self.assertTrue(jnp.issubdtype(obs.dtype, jnp.floating))

    def test_navix_env_params(self):
        """Test that environment parameters work correctly."""
        rng = jax.random.PRNGKey(0)

        for env_name in NAVIX_TEST_ENVS:
            with self.subTest(env=env_name):
                env, params = create_navix(env_name)

                # Test reset with params
                obs, state = env.reset(rng, params)
                self.assertIsNotNone(obs)
                self.assertIsNotNone(state)


if __name__ == "__main__":
    unittest.main()
