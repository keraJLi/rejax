import unittest

import jax

from rejax.compat.craftax2gymnax import create_craftax


class TestCraftaxCompat(unittest.TestCase):
    def test_create_craftax_environments(self):
        """Test creating and basic functionality of Craftax environments."""
        rng = jax.random.PRNGKey(0)

        # Test all 4 Craftax environment variants
        # fmt: off
        craftax_envs = [
            "Craftax-Symbolic-v1", "Craftax-Pixels-v1", "Craftax-Classic-Symbolic-v1",
            "Craftax-Classic-Pixels-v1",
        ]
        # fmt: on

        for env_name in craftax_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create_craftax(env_name)
                except Exception as e:
                    self.fail(f"Failed to create {env_name}: {type(e).__name__}: {e}")

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
                    # For pixel environments, obs might have different shape ordering
                    # Just check that they have the same dimensions
                    if hasattr(obs, "shape") and hasattr(obs_space, "shape"):
                        self.assertEqual(len(obs.shape), len(obs_space.shape))
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
                for step in range(3):
                    try:
                        obs, state, reward, done, _info = jitted_step(
                            rng, state, action, params
                        )

                        # Check types
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

    def test_craftax_env_params(self):
        """Test that environment parameters work correctly."""
        rng = jax.random.PRNGKey(0)

        for env_name in ["Craftax-Symbolic-v1", "Craftax-Classic-Symbolic-v1"]:
            with self.subTest(env=env_name):
                env, params = create_craftax(env_name)

                # Test that params have max_steps_in_episode attribute
                self.assertTrue(hasattr(params, "max_steps_in_episode"))
                self.assertIsInstance(params.max_steps_in_episode, int)

                # Test that max_steps_in_episode equals max_timesteps
                self.assertEqual(params.max_steps_in_episode, params.max_timesteps)

                # Test that other attributes are accessible
                self.assertTrue(hasattr(params, "day_length"))

                # Test that replacing max_steps_in_episode works and auto-syncs
                new_max_steps = params.max_steps_in_episode + 1000
                new_params = params.replace(max_steps_in_episode=new_max_steps)
                self.assertEqual(new_params.max_steps_in_episode, new_max_steps)
                # Auto-sync: max_timesteps should also be updated
                self.assertEqual(new_params.max_timesteps, new_max_steps)

                # Test that replacing max_timesteps works and auto-syncs
                new_max_timesteps = params.max_timesteps + 500
                new_params = params.replace(max_timesteps=new_max_timesteps)
                self.assertEqual(new_params.max_timesteps, new_max_timesteps)
                # Auto-sync: max_steps_in_episode should also be updated
                self.assertEqual(new_params.max_steps_in_episode, new_max_timesteps)

                # Test reset with params
                obs, state = env.reset(rng, params)
                self.assertIsNotNone(obs)
                self.assertIsNotNone(state)


if __name__ == "__main__":
    unittest.main()
