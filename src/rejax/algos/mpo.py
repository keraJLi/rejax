"""
Adapted from https://github.com/google-deepmind/acme/blob/master/acme/jax/losses/mpo.py
for rejax.
Main differences from original implementation:
- We use a scalar critic rather than a distributional one.
- We do a 1-step TD update rather than the default 5.
- We don't calculate any diagnostics.
- We reuse rejax's GaussianPolicy, whose standard deviation is a learned but
  observation-independent parameter, rather than Acme's state-dependent scale
  head (which outputs a per-observation scale).
- Actions are clipped to the action space within MPO (before feeding the critic
  and when acting in the env) rather than inside dedicated network modules.
- Target networks are updated on a global-step schedule via rejax's
  TargetNetworkMixin (supporting both hard updates via target_update_freq and
  polyak averaging), rather than Acme's fixed learner-step update period.
"""

from collections import namedtuple

import chex
import distrax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
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
from rejax.networks import GaussianPolicy, QNetwork


_MPO_EPS = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0

# Lagrange dual variables. log_penalty_temperature is only used when
# action_penalization is enabled, but is always present so the params remain a
# fixed pytree structure.
DualParams = namedtuple(
    "DualParams",
    [
        "log_temperature",
        "log_alpha_mean",
        "log_alpha_stddev",
        "log_penalty_temperature",
    ],
)


def compute_weights_and_temperature_loss(q_values, epsilon, temperature):
    """E-step: importance weights and temperature dual loss.

    Args:
        q_values: [N, B] Q-values for N actions sampled per B states.
        epsilon: KL constraint on the non-parametric policy.
        temperature: Lagrange dual variable (softplus-transformed before calling).

    Returns:
        normalized_weights: [N, B] stop-gradient importance weights.
        loss_temperature: scalar temperature dual loss.
    """
    tempered_q = jax.lax.stop_gradient(q_values) / temperature
    normalized_weights = jax.nn.softmax(tempered_q, axis=0)
    normalized_weights = jax.lax.stop_gradient(normalized_weights)

    q_logsumexp = jax.scipy.special.logsumexp(tempered_q, axis=0)
    log_num_actions = jnp.log(q_values.shape[0])
    loss_temperature = temperature * (epsilon + jnp.mean(q_logsumexp) - log_num_actions)

    return normalized_weights, loss_temperature


def compute_cross_entropy_loss(sampled_actions, normalized_weights, dist):
    """M-step: weighted negative log-likelihood under a fixed distribution.

    Args:
        sampled_actions: [N, B, D] raw Gaussian actions sampled from target policy.
        normalized_weights: [N, B] stop-gradient importance weights.
        dist: distrax.MultivariateNormalDiag with batch_shape [B].

    Returns:
        loss: scalar.
    """
    log_prob = dist.log_prob(sampled_actions)  # [N, B]
    return jnp.mean(-jnp.sum(log_prob * normalized_weights, axis=0))


def compute_parametric_kl_penalty_and_dual_loss(kl, alpha, epsilon):
    """Alpha-weighted KL penalty (M-step) and dual loss (alpha update).

    Args:
        kl: [B, D] (per-dim) or [B] (joint) KL between target and online policy.
        alpha: Lagrange dual variable (softplus-transformed before calling).
        epsilon: KL constraint threshold.

    Returns:
        loss_kl: scalar KL penalty added to actor loss (alpha stopped).
        loss_alpha: scalar dual loss for alpha update (KL stopped).
    """
    mean_kl = jnp.mean(kl, axis=0)  # [D] or scalar
    loss_kl = jnp.sum(jax.lax.stop_gradient(alpha) * mean_kl)
    loss_alpha = jnp.sum(alpha * (epsilon - jax.lax.stop_gradient(mean_kl)))
    return loss_kl, loss_alpha


class MPO(
    ReplayBufferMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    TargetNetworkMixin,
    Algorithm,
):
    """Maximum a Posteriori Policy Optimization (continuous actions only).

    Implements decoupled KL constraints as in Abdolmaleki et al. 2018.
    The Acme MPO loss is adapted here with TFP replaced by distrax.
    """

    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    # N actions sampled per state in the E-step
    num_action_samples: int = struct.field(pytree_node=False, default=20)
    # N actions sampled per next state for critic value estimation
    policy_eval_num_val_samples: int = struct.field(pytree_node=False, default=128)
    # Enforce KL constraint per action dimension rather than jointly
    per_dim_constraining: bool = struct.field(pytree_node=False, default=True)
    # MO-MPO action penalization from the Acme loss.
    action_penalization: bool = struct.field(pytree_node=False, default=True)

    # KL constraint thresholds
    epsilon: chex.Scalar = struct.field(pytree_node=True, default=0.1)
    epsilon_mean: chex.Scalar = struct.field(pytree_node=True, default=0.0025)
    epsilon_stddev: chex.Scalar = struct.field(pytree_node=True, default=1e-6)
    epsilon_penalty: chex.Scalar = struct.field(pytree_node=True, default=0.001)
    dual_learning_rate: chex.Scalar = struct.field(pytree_node=True, default=1e-2)

    # Initial log-space values for dual variables (only used during init)
    init_log_temperature: float = struct.field(pytree_node=False, default=10.0)
    init_log_alpha_mean: float = struct.field(pytree_node=False, default=10.0)
    init_log_alpha_stddev: float = struct.field(pytree_node=False, default=1000.0)

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
        layers = agent_kwargs.pop(
            "hidden_layer_sizes", config.pop("hidden_layer_sizes", (64, 64))
        )
        agent_kwargs["hidden_layer_sizes"] = tuple(layers)

        action_space = env.action_space(env_params)
        actor = GaussianPolicy(
            np.prod(action_space.shape),
            (action_space.low, action_space.high),
            **agent_kwargs,
        )
        critic = QNetwork(**agent_kwargs)
        return {"actor": actor, "critic": critic}

    @classmethod
    def clip_dual_params(cls, params):
        return DualParams(
            log_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE, params.log_temperature),
            log_alpha_mean=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_mean),
            log_alpha_stddev=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_stddev),
            log_penalty_temperature=jnp.maximum(
                _MIN_LOG_TEMPERATURE, params.log_penalty_temperature
            ),
        )

    def clip_action(self, action):
        action_space = self.env.action_space(self.env_params)
        return jnp.clip(action, action_space.low, action_space.high)

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty((1, *self.obs_space.shape))
        act_ph = jnp.empty((1, *self.env.action_space(self.env_params).shape))

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)
        critic_params = self.critic.init(rng_critic, obs_ph, act_ph)

        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        dual_tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.dual_learning_rate),
        )
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)

        action_dim = int(np.prod(self.env.action_space(self.env_params).shape))
        dual_shape = (action_dim,) if self.per_dim_constraining else ()
        dual_params = DualParams(
            log_temperature=jnp.full((), self.init_log_temperature),
            log_alpha_mean=jnp.full(dual_shape, self.init_log_alpha_mean),
            log_alpha_stddev=jnp.full(dual_shape, self.init_log_alpha_stddev),
            log_penalty_temperature=jnp.full((), self.init_log_temperature),
        )
        dual_ts = TrainState.create(apply_fn=(), params=dual_params, tx=dual_tx)

        return {
            "actor_ts": actor_ts,
            "actor_target_params": actor_params,
            "critic_ts": critic_ts,
            "critic_target_params": critic_params,
            "dual_ts": dual_ts,
        }

    def collect_transitions(self, ts):
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        if self.normalize_observations:
            last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
        else:
            last_obs = ts.last_obs

        actions = self.actor.apply(
            ts.actor_ts.params, last_obs, rng_action, method="act"
        )

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

    def update_critic(self, ts, mb):
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def critic_loss_fn(params):
            # Acme evaluates the bootstrap value under the target policy.
            target_dist = self.actor.apply(
                ts.actor_target_params, mb.next_obs, method="_action_dist"
            )
            next_actions = target_dist.sample(
                seed=action_rng, sample_shape=(self.policy_eval_num_val_samples,)
            )
            next_actions = self.clip_action(next_actions)

            qs_next = jax.vmap(
                lambda a: self.critic.apply(ts.critic_target_params, mb.next_obs, a)
            )(next_actions)
            q_target = jnp.mean(qs_next, axis=0)
            target = mb.reward + self.gamma * (1 - mb.done) * q_target

            qs = self.critic.apply(params, mb.obs, mb.action)
            return optax.l2_loss(qs, target).mean()

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))

    def update_actor_and_duals(self, ts, mb):
        rng, sample_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        # E-step: sample N actions from target actor and evaluate their Q-values.
        target_dist = self.actor.apply(
            ts.actor_target_params, mb.obs, method="_action_dist"
        )
        sampled_actions = target_dist.sample(
            seed=sample_rng, sample_shape=(self.num_action_samples,)
        )  # [N, B, D]

        # Raw samples are kept for the M-step log-probs and action penalty; the
        # critic only sees clipped actions.
        q_values = jax.vmap(
            lambda a: self.critic.apply(
                ts.critic_target_params, mb.obs, self.clip_action(a)
            )
        )(sampled_actions)  # [N, B]

        target_mean = target_dist.loc
        target_scale = target_dist.scale_diag

        def actor_and_dual_loss(actor_params, dual_params):
            sp = jax.nn.softplus
            online_dist = self.actor.apply(actor_params, mb.obs, method="_action_dist")
            online_mean = online_dist.loc  # [B, D]
            online_scale = online_dist.scale_diag  # [B, D]

            temperature = sp(dual_params.log_temperature) + _MPO_EPS
            alpha_mean = sp(dual_params.log_alpha_mean) + _MPO_EPS
            alpha_stddev = sp(dual_params.log_alpha_stddev) + _MPO_EPS

            # E-step: normalized importance weights
            normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
                q_values, self.epsilon, temperature
            )

            if self.action_penalization:
                penalty_temp = sp(dual_params.log_penalty_temperature) + _MPO_EPS
                diff_oob = sampled_actions - jnp.clip(sampled_actions, -1.0, 1.0)
                cost_oob = -jnp.linalg.norm(diff_oob, axis=-1)  # [N, B]
                penalty_weights, loss_penalty_temp = (
                    compute_weights_and_temperature_loss(
                        cost_oob, self.epsilon_penalty, penalty_temp
                    )
                )
                normalized_weights += penalty_weights
                loss_temperature += loss_penalty_temp

            # M-step: decomposed losses (fixed-mean and fixed-stddev distributions)
            fixed_stddev_dist = distrax.MultivariateNormalDiag(
                loc=online_mean, scale_diag=target_scale
            )
            fixed_mean_dist = distrax.MultivariateNormalDiag(
                loc=target_mean, scale_diag=online_scale
            )
            loss_pi_mean = compute_cross_entropy_loss(
                sampled_actions, normalized_weights, fixed_stddev_dist
            )
            loss_pi_stddev = compute_cross_entropy_loss(
                sampled_actions, normalized_weights, fixed_mean_dist
            )

            # KL between target and online policies
            if self.per_dim_constraining:
                target_normal = distrax.Normal(loc=target_mean, scale=target_scale)
                kl_mean = target_normal.kl_divergence(
                    distrax.Normal(loc=online_mean, scale=target_scale)
                )  # [B, D]
                kl_stddev = target_normal.kl_divergence(
                    distrax.Normal(loc=target_mean, scale=online_scale)
                )  # [B, D]
            else:
                target_mvn = distrax.MultivariateNormalDiag(
                    loc=target_mean, scale_diag=target_scale
                )
                kl_mean = target_mvn.kl_divergence(fixed_stddev_dist)  # [B]
                kl_stddev = target_mvn.kl_divergence(fixed_mean_dist)  # [B]

            loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
                kl_mean, alpha_mean, self.epsilon_mean
            )
            loss_kl_stddev, loss_alpha_stddev = (
                compute_parametric_kl_penalty_and_dual_loss(
                    kl_stddev, alpha_stddev, self.epsilon_stddev
                )
            )

            return (
                loss_pi_mean
                + loss_pi_stddev
                + loss_kl_mean
                + loss_kl_stddev
                + loss_alpha_mean
                + loss_alpha_stddev
                + loss_temperature
            )

        actor_grad, dual_grad = jax.grad(actor_and_dual_loss, argnums=(0, 1))(
            ts.actor_ts.params, ts.dual_ts.params
        )
        ts = ts.replace(
            actor_ts=ts.actor_ts.apply_gradients(grads=actor_grad),
            dual_ts=ts.dual_ts.apply_gradients(grads=dual_grad),
        )
        return ts.replace(
            dual_ts=ts.dual_ts.replace(params=self.clip_dual_params(ts.dual_ts.params))
        )

    def update(self, ts, mb):
        ts = self.update_critic(ts, mb)
        ts = self.update_actor_and_duals(ts, mb)
        return ts

    def train_iteration(self, ts):
        old_global_step = ts.global_step
        ts, batch = self.collect_transitions(ts)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        def update_iteration(ts):
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
            return self.update(ts, minibatch)

        def do_updates(ts):
            return jax.lax.fori_loop(
                0, self.num_epochs, lambda _, ts: update_iteration(ts), ts
            )

        ts = jax.lax.cond(
            ts.global_step > self.fill_buffer, lambda: do_updates(ts), lambda: ts
        )

        ts = ts.replace(
            actor_target_params=self.maybe_update_target_params(
                ts.actor_ts.params,
                ts.actor_target_params,
                ts.global_step,
                old_global_step,
            ),
            critic_target_params=self.maybe_update_target_params(
                ts.critic_ts.params,
                ts.critic_target_params,
                ts.global_step,
                old_global_step,
            ),
        )

        return ts
