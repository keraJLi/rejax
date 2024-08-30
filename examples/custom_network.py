"""
This example demonstrates how to log to use a custom network.
"""

import jax
from flax import linen as nn
from jax import numpy as jnp

from rejax import DQN
from rejax.evaluate import evaluate
from rejax.networks import EpsilonGreedyPolicy


class ConvDuelingQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        value = nn.Dense(1)(x)
        advantage = nn.Dense(self.action_dim)(x)
        advantage = advantage - jnp.mean(advantage, axis=-1, keepdims=True)

        return value + advantage

    def take(self, obs, action):
        """Returns the Q-value of action `action`."""
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(1)


algo = DQN.create(
    env="Freeway-MinAtar",
    total_timesteps=1_000_000,
    eval_freq=50_000,
    num_epochs=1,
    learning_rate=0.00025,
    max_grad_norm=10,
    batch_size=32,
    num_envs=50,
    buffer_size=100_000,
    fill_buffer=5_000,
    target_update_freq=1_000,
    eps_start=1.0,
    eps_end=0.1,
    exploration_fraction=0.2,
    gamma=0.99,
    ddqn=True,
)


# Overwrite default evaluation callback to log episode lengths and returns
def log_callback(algo, train_state, rng):
    act = algo.make_act(train_state)
    lengths, returns = evaluate(act, rng, algo.env, algo.env_params, 200)
    jax.debug.print(
        "global step: {}, mean episode length: {} ± {}std, mean return: {} ± {}std",
        train_state.global_step,
        lengths.mean(),
        lengths.std(),
        returns.mean(),
        returns.std(),
    )
    return ()  # We don't need the results after printing them :-)


# We make a policy out of the Q-network to use it as an agent (that has an "act" method)
agent = EpsilonGreedyPolicy(ConvDuelingQNetwork)(algo.action_dim)

# Add custom network to config and train
rng = jax.random.PRNGKey(0)
algo = algo.replace(eval_callback=log_callback, agent=agent)
jax.jit(algo.train)(rng)
