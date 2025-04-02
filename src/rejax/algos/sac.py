import chex
import gymnax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks import (
    DiscretePolicy,
    DiscreteQNetwork,
    QNetwork,
    SquashedGaussianPolicy,
)


class SAC(
    ReplayBufferMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    TargetNetworkMixin,
    Algorithm,
):
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_critics: int = struct.field(pytree_node=False, default=2)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    target_entropy_ratio: chex.Scalar = struct.field(pytree_node=True, default=0.98)

    def make_act(self, ts):
        def act(obs, rng):
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.actor.apply(ts.actor_ts.params, obs, rng, method="act")
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "relu")
        agent_kwargs["activation"] = getattr(nn, activation)
        layers = config.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(layers)

        action_space = env.action_space(env_params)
        if isinstance(action_space, gymnax.environments.spaces.Discrete):
            actor = DiscretePolicy(action_space.n, **agent_kwargs)
            critic = DiscreteQNetwork(action_dim=action_space.n, **agent_kwargs)
        else:
            actor = SquashedGaussianPolicy(
                np.prod(action_space.shape),
                (action_space.low, action_space.high),
                log_std_range=(-10, 2),
                **agent_kwargs,
            )
            critic = QNetwork(**agent_kwargs)
        return {"actor": actor, "critic": critic}

    @property
    def target_entropy(self):
        if self.discrete:
            return -self.target_entropy_ratio * np.log(1 / self.action_dim)
        return -self.action_dim

    @property
    def vmap_critic(self, method="__call__"):
        def apply(*args):
            return self.critic.apply(*args, method=method)

        if self.discrete:
            return jax.vmap(apply, in_axes=(0, None))
        return jax.vmap(apply, in_axes=(0, None, None))

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty((1, *self.obs_space.shape))

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)

        rng_critic = jax.random.split(rng_critic, self.num_critics)
        if self.discrete:
            critic_params = jax.vmap(self.critic.init, in_axes=(0, None))(
                rng_critic, obs_ph
            )
        else:
            act_ph = jnp.empty((1, *self.env.action_space(self.env_params).shape))
            critic_params = jax.vmap(self.critic.init, in_axes=(0, None, None))(
                rng_critic, obs_ph, act_ph
            )

        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)
        critic_target_params = critic_params

        if self.target_entropy is None:
            self.target_entropy = -self.env.action_space(self.env_params).shape[0]

        alpha_params = FrozenDict({"log_alpha": jnp.array(0.0)})
        alpha_ts = TrainState.create(apply_fn=(), params=alpha_params, tx=tx)

        return {
            "actor_ts": actor_ts,
            "critic_ts": critic_ts,
            "critic_target_params": critic_target_params,
            "alpha_ts": alpha_ts,
        }

    def train_iteration(self, ts):
        # Collect transitions
        old_global_step = ts.global_step

        ts, batch = self.collect_transitions(ts)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        def update_iteration(ts):
            # Sample minibatch
            rng, rng_sample = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            minibatch = ts.replay_buffer.sample(self.batch_size, rng_sample)
            if self.normalize_observations:
                minibatch = minibatch._replace(
                    obs=self.normalize_obs(ts.obs_rms_state, minibatch.obs),
                    next_obs=self.normalize_obs(ts.obs_rms_state, minibatch.next_obs),
                )
            if self.normalize_rewards:
                minibatch = minibatch._replace(
                    reward=self.normalize_rew(ts.rew_rms_state, minibatch.reward)
                )

            # Update networks
            ts = self.update(ts, minibatch)
            return ts

        def do_updates(ts):
            return jax.lax.fori_loop(
                0, self.num_epochs, lambda _, ts: update_iteration(ts), ts
            )

        start_training = ts.global_step > self.fill_buffer
        ts = jax.lax.cond(start_training, lambda: do_updates(ts), lambda: ts)

        # Update target network
        if self.target_update_freq == 1:
            target_params = self.polyak_update(
                ts.critic_ts.params, ts.critic_target_params
            )
        else:
            update_target_params = (
                ts.global_step % self.target_update_freq
                <= old_global_step % self.target_update_freq
            )
            target_params = jax.tree.map(
                lambda q, qt: jax.lax.select(update_target_params, q, qt),
                self.polyak_update(ts.critic_ts.params, ts.critic_target_params),
                ts.critic_target_params,
            )
        ts = ts.replace(critic_target_params=target_params)

        return ts

    def collect_transitions(self, ts):
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_policy(rng):
            if self.normalize_observations:
                last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            actions = self.actor.apply(
                ts.actor_ts.params, last_obs, rng_action, method="act"
            )
            return actions

        actions = sample_policy(rng_action)

        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, actions, self.env_params
        )
        if self.normalize_observations:
            ts = ts.replace(
                obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs)
            )
        if self.normalize_rewards:
            ts = ts.replace(
                rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones)
            )

        minibatch = Minibatch(
            obs=ts.last_obs,
            action=actions,
            reward=rewards,
            next_obs=next_obs,
            done=dones,
        )
        ts = ts.replace(
            last_obs=next_obs,
            env_state=env_state,
            global_step=ts.global_step + self.num_envs,
        )
        return ts, minibatch

    def update_actor(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def actor_loss_fn(params):
            if self.discrete:
                logprob = jnp.log(
                    self.actor.apply(params, mb.obs, method="_action_dist").probs
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs)
                loss_pi = alpha * logprob - qs.min(axis=0)
                loss_pi = jnp.sum(jnp.exp(logprob) * loss_pi, axis=1)
            else:
                action, logprob = self.actor.apply(
                    params, mb.obs, action_rng, method="action_log_prob"
                )
                qs = self.vmap_critic(ts.critic_ts.params, mb.obs, action)
                loss_pi = alpha * logprob - qs.min(axis=0)
            return loss_pi.mean(), logprob

        grads, logprob = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, logprob

    def update_critic(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def critic_loss_fn(params):
            # Calculate target without gradient wrt `params`
            if self.discrete:
                action_dist = self.actor.apply(
                    ts.actor_ts.params, mb.next_obs, method="_action_dist"
                )
                logprob = jnp.log(action_dist.probs)
                qs = self.vmap_critic(ts.critic_target_params, mb.next_obs)
                q_target = jnp.min(qs, axis=0) - alpha * logprob
                q_target = jnp.sum(jnp.exp(logprob) * q_target, axis=1)
                qs = jax.vmap(
                    lambda *args: self.critic.apply(*args, method="take"),
                    in_axes=(0, None, None),
                )(params, mb.obs, mb.action)
            else:
                action, logprob = self.actor.apply(
                    ts.actor_ts.params,
                    mb.next_obs,
                    action_rng,
                    method="action_log_prob",
                )
                qs = self.vmap_critic(ts.critic_target_params, mb.next_obs, action)
                q_target = jnp.min(qs, axis=0) - alpha * logprob
                qs = self.vmap_critic(params, mb.obs, mb.action)

            target = mb.reward + self.gamma * (1 - mb.done) * q_target
            losses = jax.vmap(lambda q: optax.l2_loss(q, target))(qs)
            return losses.sum(axis=0).mean()

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts

    def update_alpha(self, ts, logprob):
        def alpha_loss_fn(params, logprob):
            alpha = jnp.exp(params["log_alpha"])
            loss_alpha = -alpha * (logprob + self.target_entropy)
            if self.discrete:
                loss_alpha = jnp.sum(jnp.exp(logprob) * loss_alpha, axis=1)
            return loss_alpha.mean()

        grads = jax.grad(alpha_loss_fn)(ts.alpha_ts.params, logprob)
        ts = ts.replace(alpha_ts=ts.alpha_ts.apply_gradients(grads=grads))
        return ts

    def update(self, ts, mb):
        ts, logprob = self.update_actor(ts, mb)
        ts = self.update_critic(ts, mb)
        ts = self.update_alpha(ts, logprob)
        return ts
