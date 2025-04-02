"""
Adapted from https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
by Matteo Gallici et. al.
Thanks!
"""

import chex
import jax
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    EpsilonGreedyMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
)
from rejax.networks import DiscreteQNetwork, EpsilonGreedyPolicy


class Trajectory(struct.PyTreeNode):
    obs: chex.Array
    action: chex.Array
    next_q: chex.Array
    reward: chex.Array
    done: chex.Array


class TargetMinibatch(struct.PyTreeNode):
    trajectories: Trajectory
    targets: chex.Array


class PQN(
    OnPolicyMixin,
    EpsilonGreedyMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    Algorithm,
):
    agent: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    td_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.9)

    def make_act(self, ts):
        def act(obs, rng):
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.agent.apply(
                ts.q_ts.params, obs, rng, epsilon=0.005, method="act"
            )
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_kwargs = config.pop("agent_kwargs", {})
        agent_kwargs["activation"] = lambda x: nn.relu(nn.LayerNorm()(x))

        action_dim = env.action_space(env_params).n
        agent = EpsilonGreedyPolicy(DiscreteQNetwork)(
            hidden_layer_sizes=(64, 64), action_dim=action_dim, **agent_kwargs
        )
        return {"agent": agent}

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])
        q_params = self.agent.init(rng, obs_ph)
        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        q_ts = TrainState.create(apply_fn=(), params=q_params, tx=tx)
        return {"q_ts": q_ts}

    def train_iteration(self, ts):
        epsilon = self.epsilon_schedule(ts.global_step)
        ts, trajectories = self.collect_trajectories(ts, epsilon)

        max_last_q = self.agent.apply(ts.q_ts.params, ts.last_obs).max(axis=1)
        max_last_q = jnp.where(ts.last_done, 0, max_last_q)
        targets = self.calculate_targets(trajectories, max_last_q)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)

            batch = TargetMinibatch(trajectories, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def collect_trajectories(self, ts, epsilon):
        def env_step(ts, unused):
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_action, rng_step = jax.random.split(new_rng)
            action = self.agent.apply(
                ts.q_ts.params, ts.last_obs, rng_action, epsilon=epsilon, method="act"
            )

            rng_step = jax.random.split(rng_step, self.num_envs)
            transition = self.vmap_step(rng_step, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = transition
            next_q = self.agent.apply(ts.q_ts.params, next_obs)

            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(
                    ts.obs_rms_state, next_obs
                )
                ts = ts.replace(obs_rms_state=obs_rms_state)
            if self.normalize_rewards:
                rew_rms_state, reward = self.update_and_normalize_rew(
                    ts.rew_rms_state, reward, done
                )
                ts = ts.replace(rew_rms_state=rew_rms_state)

            # Return updated state and transition
            transition = Trajectory(ts.last_obs, action, next_q, reward, done)
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_targets(self, trajectories, max_last_q):
        def get_target(lambda_return_and_next_q, t):
            lambda_return, next_q = lambda_return_and_next_q
            return_bootstrap = next_q + self.td_lambda * (lambda_return - next_q)
            lambda_return = t.reward + (1 - t.done) * self.gamma * (return_bootstrap)
            max_next_q = t.next_q.max(axis=1)
            return (lambda_return, max_next_q), lambda_return

        max_last_q = jnp.where(trajectories.done[-1], 0, max_last_q)
        lambda_returns = trajectories.reward[-1] + self.gamma * max_last_q
        _, targets = jax.lax.scan(
            get_target,
            (lambda_returns, max_last_q),
            jax.tree.map(lambda x: x[:-1], trajectories),
            reverse=True,
        )
        targets = jnp.concatenate((targets, lambda_returns[None]))
        return targets

    def update(self, ts, minibatch):
        tr, ta = minibatch.trajectories, minibatch.targets

        def loss_fn(params):
            q_values = self.agent.apply(params, tr.obs, tr.action, method="take")
            return optax.l2_loss(q_values, ta).mean()

        grads = jax.grad(loss_fn)(ts.q_ts.params)
        ts = ts.replace(q_ts=ts.q_ts.apply_gradients(grads=grads))
        return ts
