from functools import partial
from typing import NamedTuple, Union

import chex
import jax
from flax import struct
from gymnax.environments import spaces
from jax import numpy as jnp


class CircularBuffer(struct.PyTreeNode):
    size: int = struct.field(pytree_node=False)
    data: chex.ArrayTree
    index: int
    full: bool

    @classmethod
    def empty(cls, size: int, data: chex.ArrayTree) -> "CircularBuffer":
        return cls(size=size, data=data, index=0, full=False)

    @property
    def num_entries(self):
        return jnp.where(self.full, self.size, self.index)

    @jax.jit
    def append(self, a: chex.ArrayTree) -> "CircularBuffer":
        data = jax.tree_map(lambda arr, a_: arr.at[self.index].set(a_), self.data, a)
        next_index = (self.index + 1) % self.size
        full = jnp.logical_or(self.full, next_index == 0)
        return self.replace(data=data, index=next_index, full=full)

    @jax.jit
    def extend(self, batch: chex.ArrayTree) -> "CircularBuffer":
        batch_flat, _ = jax.tree_util.tree_flatten(batch)
        batch_size = batch_flat[0].shape[0]

        idx = self.index + jnp.arange(batch_size)
        idx = idx % self.size
        data = jax.tree_map(lambda arr, b: arr.at[idx].set(b), self.data, batch)

        next_index = (self.index + batch_size) % self.size
        full = jnp.logical_or(self.full, next_index == 0)
        return self.replace(data=data, index=next_index, full=full)


class Minibatch(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array


class ReplayBuffer(CircularBuffer):
    """
    Circular buffer for storing transitions. Implements appending and sampling
    while being `jit`-able.
    """

    data: Minibatch

    @classmethod
    def empty(
        cls,
        size: int,
        obs_space: Union[spaces.Discrete, spaces.Box],
        action_space: Union[spaces.Discrete, spaces.Box],
    ) -> "ReplayBuffer":
        """Returns an empty replay buffer with the given size and shapes.

        Args:
            size (int): Maximum number of transitions to store.
            obs_shape (chex.Shape): Shape of the observations.
            action_shape (chex.Shape): Shape of the actions.

        Returns:
            ReplayBuffer: The initialized replay buffer.
        """
        # Skip checking sizes as we know they are correct here
        data = Minibatch(
            obs=jnp.empty((size, *obs_space.shape)).astype(obs_space.dtype),
            action=jnp.empty((size, *action_space.shape)).astype(action_space.dtype),
            reward=jnp.empty(size),
            done=jnp.empty(size).astype(bool),
            next_obs=jnp.empty((size, *obs_space.shape)).astype(obs_space.dtype),
        )
        return cls(size=size, data=data, index=0, full=False)

    def __getattr__(self, name):
        if name in self.data._fields:
            return getattr(self.data, name)

    @partial(jax.jit, static_argnames=("num"))
    def sample(self, num: int, rng: chex.PRNGKey) -> Minibatch:
        """Samples a minibatch of transitions from the buffer, without replacement.
        Note that this function does not check if enough transitions are stored, and
        might return the same transition multiple times.

        Args:
            num (int): The size of the minibatch to sample.
            rng (chex.PRNGKey): The random number generator key.

        Returns:
            Minibatch: A minibatch of randomly sampled transitions.
        """
        minibatch_index = jax.random.randint(rng, (num,), 0, self.num_entries)
        return jax.tree_map(lambda arr: arr[minibatch_index], self.data)
