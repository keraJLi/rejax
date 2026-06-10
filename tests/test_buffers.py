import unittest

import jax
import numpy as np
from gymnax.environments import spaces
from jax import numpy as jnp

from rejax.buffers import CircularBuffer, Minibatch, ReplayBuffer


def make_minibatch(ids):
    """Minibatch of transitions where every field encodes the transition id."""
    ids = jnp.asarray(ids)
    return Minibatch(
        obs=jnp.stack([ids, ids], axis=1).astype(jnp.float32),
        action=ids.astype(jnp.int32),
        reward=ids.astype(jnp.float32),
        done=ids % 2 == 0,
        next_obs=jnp.stack([ids + 1, ids + 1], axis=1).astype(jnp.float32),
    )


class TestCircularBuffer(unittest.TestCase):
    def empty_buffer(self, size):
        return CircularBuffer.empty(size, data=jnp.zeros(size))

    def test_extend_raises_if_batch_exceeds_size(self):
        buffer = self.empty_buffer(4)
        with self.assertRaises(ValueError):
            buffer.extend(jnp.arange(5, dtype=jnp.float32))

    def test_num_entries_grows_and_caps_at_size(self):
        buffer = self.empty_buffer(5)
        self.assertEqual(buffer.num_entries, 0)

        buffer = buffer.append(jnp.float32(1))
        self.assertEqual(buffer.num_entries, 1)

        buffer = buffer.extend(jnp.ones(3))
        self.assertEqual(buffer.num_entries, 4)

        buffer = buffer.extend(jnp.ones(3))
        self.assertEqual(buffer.num_entries, 5)

    def test_append_wraps_and_overwrites_oldest(self):
        size = 4
        buffer = self.empty_buffer(size)
        for i in range(size):
            buffer = buffer.append(jnp.float32(i))

        self.assertTrue(buffer.full)
        self.assertEqual(buffer.index, 0)
        np.testing.assert_array_equal(buffer.data, jnp.arange(size))

        buffer = buffer.append(jnp.float32(size))
        self.assertEqual(buffer.index, 1)
        self.assertEqual(buffer.num_entries, size)
        np.testing.assert_array_equal(buffer.data, jnp.array([4, 1, 2, 3]))

    def test_extend_with_dividing_batch_size(self):
        buffer = self.empty_buffer(8)
        buffer = buffer.extend(jnp.arange(4, dtype=jnp.float32))
        self.assertFalse(buffer.full)
        self.assertEqual(buffer.index, 4)

        buffer = buffer.extend(jnp.arange(4, 8, dtype=jnp.float32))
        self.assertTrue(buffer.full)
        self.assertEqual(buffer.index, 0)
        np.testing.assert_array_equal(buffer.data, jnp.arange(8))

    def test_extend_with_non_dividing_batch_size(self):
        # size=10, batches of 4: the third extend wraps (writes 8, 9, 0, 1) and
        # must set full even though the index never lands on 0
        buffer = self.empty_buffer(10)
        buffer = buffer.extend(jnp.arange(4, dtype=jnp.float32))
        buffer = buffer.extend(jnp.arange(4, 8, dtype=jnp.float32))
        self.assertFalse(buffer.full)
        self.assertEqual(buffer.index, 8)

        buffer = buffer.extend(jnp.arange(8, 12, dtype=jnp.float32))
        self.assertTrue(buffer.full)
        self.assertEqual(buffer.index, 2)
        self.assertEqual(buffer.num_entries, 10)
        np.testing.assert_array_equal(
            buffer.data, jnp.array([10, 11, 2, 3, 4, 5, 6, 7, 8, 9])
        )

        # full stays set on subsequent extends
        buffer = buffer.extend(jnp.arange(12, 16, dtype=jnp.float32))
        self.assertTrue(buffer.full)
        self.assertEqual(buffer.index, 6)

    def test_extend_with_full_size_batch(self):
        buffer = self.empty_buffer(6)
        buffer = buffer.extend(jnp.arange(6, dtype=jnp.float32))
        self.assertTrue(buffer.full)
        self.assertEqual(buffer.index, 0)
        np.testing.assert_array_equal(buffer.data, jnp.arange(6))


class TestReplayBuffer(unittest.TestCase):
    obs_space = spaces.Box(-1, 1, (2,), jnp.float32)
    action_space = spaces.Discrete(3)

    def empty_buffer(self, size):
        return ReplayBuffer.empty(size, self.obs_space, self.action_space)

    def test_empty_shapes_and_dtypes(self):
        buffer = self.empty_buffer(7)
        self.assertEqual(buffer.obs.shape, (7, 2))
        self.assertEqual(buffer.obs.dtype, jnp.float32)
        self.assertEqual(buffer.action.shape, (7,))
        self.assertEqual(buffer.reward.shape, (7,))
        self.assertEqual(buffer.done.dtype, jnp.bool_)
        self.assertEqual(buffer.next_obs.shape, (7, 2))

    def test_sample_shapes(self):
        buffer = self.empty_buffer(10)
        buffer = buffer.extend(make_minibatch(jnp.arange(10)))
        minibatch = buffer.sample(4, jax.random.PRNGKey(0))
        self.assertEqual(minibatch.obs.shape, (4, 2))
        self.assertEqual(minibatch.action.shape, (4,))
        self.assertEqual(minibatch.reward.shape, (4,))
        self.assertEqual(minibatch.done.shape, (4,))
        self.assertEqual(minibatch.next_obs.shape, (4, 2))

    def test_sample_without_replacement(self):
        size = 16
        buffer = self.empty_buffer(size)
        buffer = buffer.extend(make_minibatch(jnp.arange(size)))

        for seed in range(5):
            with self.subTest(seed=seed):
                minibatch = buffer.sample(size, jax.random.PRNGKey(seed))
                self.assertEqual(len(jnp.unique(minibatch.action)), size)

    def test_sample_only_returns_written_entries(self):
        num_written = 6
        buffer = self.empty_buffer(10)
        buffer = buffer.extend(make_minibatch(jnp.arange(num_written)))

        for seed in range(5):
            with self.subTest(seed=seed):
                minibatch = buffer.sample(num_written, jax.random.PRNGKey(seed))
                self.assertTrue(jnp.all(minibatch.action >= 0))
                self.assertTrue(jnp.all(minibatch.action < num_written))
                self.assertEqual(len(jnp.unique(minibatch.action)), num_written)

    def test_sampled_fields_stay_aligned(self):
        size = 12
        buffer = self.empty_buffer(size)
        buffer = buffer.extend(make_minibatch(jnp.arange(size)))

        minibatch = buffer.sample(8, jax.random.PRNGKey(0))
        ids = minibatch.action
        np.testing.assert_array_equal(minibatch.obs[:, 0], ids)
        np.testing.assert_array_equal(minibatch.reward, ids)
        np.testing.assert_array_equal(minibatch.done, ids % 2 == 0)
        np.testing.assert_array_equal(minibatch.next_obs[:, 0], ids + 1)


if __name__ == "__main__":
    unittest.main()
