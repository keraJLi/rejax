import unittest

import jax
import jumanji

from rejax.compat.jumanji2gymnax import create_jumanji


class TestJumanjiCompat(unittest.TestCase):
    def test_create(self):
        rng = jax.random.PRNGKey(0)

        for env in jumanji.registered_environments():
            # This environment downloads some stuff from the internet.
            # I'll just assume that it works if the others do...
            if env.startswith("Sokoban"):
                continue

            with self.subTest(env=env):
                if len(jumanji.make(env).action_spec.shape) > 1:
                    with self.assertRaises(NotImplementedError):
                        create_jumanji(env)
                    continue

                env, params = create_jumanji(env)

                # JIT the reset and step functions
                jitted_reset = jax.jit(env.reset)
                jitted_step = jax.jit(env.step)

                obs, state = jitted_reset(rng, params)
                try:
                    env.observation_space(params)
                except Exception as e:
                    self.fail(f"Failed to get obs space: {type(e).__name__}: {e}")

                try:
                    a = env.action_space(params).sample(rng)
                except Exception as e:
                    self.fail(f"Failed to sample action: {type(e).__name__}: {e}")

                for _ in range(3):
                    try:
                        obs, state, reward, done, info = jitted_step(
                            rng, state, a, params
                        )
                    except Exception as e:
                        self.fail(f"Failed to step: {type(e).__name__}: {e}")

                self.assertEqual(obs.dtype, env.observation_space(params).dtype)
                self.assertEqual(obs.shape, env.observation_space(params).shape)
                self.assertEqual(a.dtype, env.action_space(params).dtype)
                self.assertEqual(a.shape, env.action_space(params).shape)
