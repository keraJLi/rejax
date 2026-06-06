import unittest

import jax.numpy as jnp

from rejax.buffers import CircularBuffer


class TestCircularBuffer(unittest.TestCase):
    def test_extend_marks_full_when_batch_wraps_without_landing_on_zero(self):
        buffer = CircularBuffer.empty(10, jnp.zeros(10, dtype=jnp.int32))

        buffer = buffer.extend(jnp.arange(8, dtype=jnp.int32))

        self.assertFalse(bool(buffer.full))

        buffer = buffer.extend(jnp.arange(8, 11, dtype=jnp.int32))

        self.assertTrue(bool(buffer.full))
        self.assertEqual(int(buffer.index), 1)
        self.assertEqual(int(buffer.num_entries), 10)

    def test_extend_with_batch_larger_than_buffer_keeps_latest_entries(self):
        buffer = CircularBuffer.empty(5, jnp.zeros(5, dtype=jnp.int32))

        buffer = buffer.extend(jnp.arange(-2, 0, dtype=jnp.int32))

        self.assertFalse(bool(buffer.full))

        buffer = buffer.extend(jnp.arange(7, dtype=jnp.int32))

        self.assertTrue(bool(buffer.full))
        self.assertEqual(int(buffer.index), 4)
        self.assertEqual(buffer.data.tolist(), [3, 4, 5, 6, 2])
