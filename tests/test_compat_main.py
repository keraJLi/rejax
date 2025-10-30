import unittest

import jax

from rejax.compat import create


class TestCompatMain(unittest.TestCase):
    def test_create_gymnax_environments(self):
        """Test that create function works with native Gymnax environments."""
        rng = jax.random.PRNGKey(0)

        # Test native Gymnax environments (no prefix)
        # fmt: off
        gymnax_envs = ["CartPole-v1", "MountainCar-v0", "Pendulum-v1"]
        # fmt: on

        for env_name in gymnax_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create(env_name)
                except Exception as e:
                    self.skipTest(f"Environment {env_name} not available: {e!s}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test basic functionality
                obs, state = jitted_reset(rng, params)
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = jitted_step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_create_brax_environments(self):
        """Test that create function works with Brax environments."""
        rng = jax.random.PRNGKey(0)

        # fmt: off
        brax_envs = ["brax/ant", "brax/halfcheetah", "brax/hopper"]
        # fmt: on

        for env_name in brax_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create(env_name)
                except Exception as e:
                    self.skipTest(f"Environment {env_name} not available: {e}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test basic functionality
                obs, state = jitted_reset(rng, params)
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = jitted_step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_create_navix_environments(self):
        """Test that create function works with Navix environments."""
        rng = jax.random.PRNGKey(0)

        # fmt: off
        navix_envs = ["navix/Navix-Empty-6x6-v0", "navix/Navix-FourRooms-v0"]
        # fmt: on

        for env_name in navix_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create(env_name)
                except Exception as e:
                    self.skipTest(f"Environment {env_name} not available: {e}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test basic functionality
                obs, state = jitted_reset(rng, params)
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = jitted_step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_create_jumanji_environments(self):
        """Test that create function works with Jumanji environments."""
        rng = jax.random.PRNGKey(0)

        # fmt: off
        jumanji_envs = ["jumanji/Snake-v1", "jumanji/Tetris-v0"]
        # fmt: on

        for env_name in jumanji_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create(env_name)
                except Exception as e:
                    self.skipTest(f"Environment {env_name} not available: {e}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test basic functionality
                obs, state = jitted_reset(rng, params)
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = jitted_step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_create_gymnasium_environments(self):
        """Test that create function works with Gymnasium environments."""
        rng = jax.random.PRNGKey(0)

        # fmt: off
        gymnasium_envs = ["gymnasium/CartPole-v1", "gymnasium/Pendulum-v1"]
        # fmt: on

        for env_name in gymnasium_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create(env_name)
                except Exception as e:
                    self.skipTest(f"Environment {env_name} not available: {e}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test basic functionality
                obs, state = jitted_reset(rng, params)
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = jitted_step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_create_kinetix_environments(self):
        """Test that create function works with Kinetix environments."""
        rng = jax.random.PRNGKey(0)

        # fmt: off
        kinetix_envs = [
            "kinetix/s/h0_weak_thrust", "kinetix/m/h1_car_left",
            "kinetix/l/h5_flappy_bird",
        ]
        # fmt: on

        for env_name in kinetix_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create(env_name)
                except Exception as e:
                    self.skipTest(f"Environment {env_name} not available: {e}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test basic functionality
                obs, state = jitted_reset(rng, params)
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = jitted_step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_create_craftax_environments(self):
        """Test that create function works with Craftax environments."""
        rng = jax.random.PRNGKey(0)

        # fmt: off
        craftax_envs = [
            "craftax/Craftax-Symbolic-v1",
            "craftax/Craftax-Classic-Symbolic-v1"
        ]
        # fmt: on

        for env_name in craftax_envs:
            with self.subTest(env=env_name):
                try:
                    env, params = create(env_name)
                except Exception as e:
                    self.skipTest(f"Environment {env_name} not available: {e}")

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                # Test basic functionality
                obs, state = jitted_reset(rng, params)
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = jitted_step(rng, state, action, params)

                self.assertIsNotNone(obs)
                self.assertIsNotNone(reward)
                self.assertIsNotNone(done)

    def test_create_invalid_prefix(self):
        """Test that create raises appropriate error for invalid prefix."""
        with self.assertRaises(KeyError):
            create("invalid_prefix/some_env")

    def test_create_invalid_environment(self):
        """Test that create handles invalid environment names gracefully."""
        # This should raise an exception from the underlying library
        with self.assertRaises(ValueError):
            create("NonExistentEnvironment-v999")


if __name__ == "__main__":
    unittest.main()
