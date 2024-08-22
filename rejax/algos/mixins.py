import chex
from flax import struct
from optax import linear_schedule


class EpsilonGreedyMixin(struct.PyTreeNode):
    eps_start: chex.Scalar = struct.field(pytree_node=True, default=1.0)
    eps_end: chex.Scalar = struct.field(pytree_node=True, default=0.05)
    exploration_fraction: chex.Scalar = struct.field(pytree_node=False, default=0.1)

    @property
    def epsilon_schedule(self):
        return linear_schedule(
            self.eps_start,
            self.eps_end,
            int(self.exploration_fraction * self.total_timesteps),
        )


class ReplayBufferMixin(struct.PyTreeNode):
    buffer_size: int = struct.field(pytree_node=False, default=100_000)
    fill_buffer: int = struct.field(pytree_node=False, default=1_000)
    batch_size: int = struct.field(pytree_node=False, default=100)
