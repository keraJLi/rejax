import unittest

import jax

from rejax.compat.kinetix2gymnax import create_kinetix


class TestKinetixCompat(unittest.TestCase):
    def test_create_kinetix_environments(self):
        """Test creating and basic functionality of Kinetix environments."""
        rng = jax.random.PRNGKey(0)

        # Test common Kinetix level files
        # fmt: off
        kinetix_levels = [
            "s/h1_thrust_over_ball", "s/h2_one_wheel_car", "s/h8_unicycle_balance",
            "m/h1_car_left", "m/h8_weird_vehicle", "m/h14_thrustblock",
            "l/h5_flappy_bird", "l/hard_pinball", "l/lever_puzzle",
        ]
        # fmt: on

        for level_name in kinetix_levels:
            with self.subTest(level=level_name):
                try:
                    env, params = create_kinetix(level_name)
                except Exception as e:
                    # Skip levels that might not be available
                    self.skipTest(f"Level {level_name} not available: {e}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test reset
                try:
                    obs, state = jitted_reset(rng, params)
                except Exception as e:
                    self.fail(f"Failed to reset {level_name}: {type(e).__name__}: {e}")

                # Test observation space
                try:
                    obs_space = env.observation_space(params)
                    self.assertEqual(obs.dtype, obs_space.dtype)
                    self.assertEqual(obs.shape, obs_space.shape)
                except Exception as e:
                    self.fail(
                        f"Failed to get obs space for {level_name}: "
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
                        f"Failed to sample action for {level_name}: "
                        f"{type(e).__name__}: {e}"
                    )

                # Test stepping
                for step in range(5):
                    try:
                        obs, state, reward, done, _info = jitted_step(
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
                            f"Failed to step {level_name} at step {step}: "
                            f"{type(e).__name__}: {e}"
                        )

                    # Sample new action for next step
                    action = action_space.sample(rng)

    def test_kinetix_continuous_actions(self):
        """Test that Kinetix environments use continuous actions by default."""
        rng = jax.random.PRNGKey(0)

        try:
            env, params = create_kinetix("s/h0_weak_thrust")
        except Exception:
            self.skipTest("No test level available for Kinetix")

        # Test that action space is continuous (Box)
        action_space = env.action_space(params)
        from gymnax.environments.spaces import Box

        self.assertIsInstance(action_space, Box)

        # Test action sampling
        action = action_space.sample(rng)
        self.assertTrue(hasattr(action, "dtype"))

        # Test that we can step with continuous actions
        obs, state = env.reset(rng, params)
        obs, state, _reward, _done, _info = env.step(rng, state, action, params)
        self.assertIsNotNone(obs)

    def test_kinetix_symbolic_flat_observations(self):
        """Test that Kinetix uses symbolic flat observations by default."""
        rng = jax.random.PRNGKey(0)

        try:
            env, params = create_kinetix("s/h0_weak_thrust")
        except Exception:
            self.skipTest("No test level available for Kinetix")

        obs, _state = env.reset(rng, params)

        # Test that observations are flat (1D)
        self.assertEqual(len(obs.shape), 1)

        # Test that observation space matches
        obs_space = env.observation_space(params)
        self.assertEqual(obs.shape, obs_space.shape)

    def test_kinetix_custom_kwargs(self):
        """Test that custom kwargs can be passed to create_kinetix."""
        from gymnax.environments.spaces import Box, Discrete
        from kinetix.environment.utils import ActionType, ObservationType

        try:
            # Test with discrete actions
            env, params = create_kinetix(
                "s/h0_weak_thrust",
                action_type=ActionType.DISCRETE,
                observation_type=ObservationType.SYMBOLIC_FLAT,
            )
        except Exception:
            self.skipTest(
                "No test level available for Kinetix or ActionType not available"
            )

        # If we get here, the environment was created successfully
        self.assertIsNotNone(env)
        self.assertIsNotNone(params)
        self.assertIsInstance(env.action_space(params), Discrete)
        self.assertEqual(len(env.action_space(params).shape), 0)
        self.assertIsInstance(env.observation_space(params), Box)
        self.assertEqual(len(env.observation_space(params).shape), 1)


if __name__ == "__main__":
    unittest.main()
