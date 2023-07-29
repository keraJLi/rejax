import jax
import chex
import optax
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple, Any
from flax.training.train_state import TrainState

# TODO: fix step ratios and so on by implementing an eval tracker


class Trajectory(NamedTuple):
    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array


class AdvantageMinibatch(NamedTuple):
    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class PPOTrainState(TrainState):
    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    rng: chex.PRNGKey

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new


def initialize_train_state(config, rng):
    rng, rng_agent = jax.random.split(rng)
    agent_params = config.agent.init(
        rng_agent,
        jnp.zeros((1, *config.env.observation_space(config.env_params).shape)),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )

    # Initialize environment
    rngs = jax.random.split(rng, config.num_envs + 1)
    rng, env_rng = rngs[0], rngs[1:]
    vmap_reset = jax.vmap(config.env.reset, in_axes=(0, None))
    obs, env_state = vmap_reset(env_rng, config.env_params)

    train_state = PPOTrainState.create(
        apply_fn=config.agent.apply,
        params=agent_params,
        tx=tx,
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(config.num_envs, dtype=bool),
        rng=rng,
    )
    return train_state


def evaluate(config, ts):
    def act(obs, rng):
        obs = jnp.expand_dims(obs, 0)
        action = ts.apply_fn(ts.params, obs, rng, method="act")
        return jnp.squeeze(action, 0)

    return config.evaluate(act, ts.rng)


@jax.jit
def train(config, rng):
    ts = initialize_train_state(config, rng)

    if not config.skip_initial_evaluation:
        initial_evaluation = evaluate(config, ts)

    def eval_iteration(ts, unused):
        # Run a few training iterations
        iteration_steps = config.num_envs * config.num_steps
        ts = jax.lax.fori_loop(
            0,
            np.ceil(config.eval_freq / iteration_steps).astype(int),
            lambda _, ts: train_iteration(config, ts),
            ts,
        )

        # Run evaluation
        return ts, evaluate(config, ts)

    ts, evaluation = jax.lax.scan(
        eval_iteration,
        ts,
        None,
        np.ceil(config.total_timesteps / config.eval_freq).astype(int),
    )

    if not config.skip_initial_evaluation:
        evaluation = jax.tree_map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_evaluation,
            evaluation,
        )

    return ts, evaluation


def train_iteration(config, ts):
    ts, trajectories = collect_trajectories(config, ts)

    last_val = ts.apply_fn(
        ts.params,
        ts.last_obs,
        method=config.agent.value,
    )
    last_val = jnp.where(ts.last_done, jnp.zeros_like(last_val), last_val)
    advantages, targets = calculate_gae(config, trajectories, last_val)

    def update_epoch(ts: PPOTrainState, unused):
        ts, minibatch_rng = ts.get_rng()
        minibatches = make_minibatches(
            config, trajectories, advantages, targets, minibatch_rng
        )
        ts, _ = jax.lax.scan(
            lambda ts, mbs: (update(config, ts, mbs), None),
            ts,
            minibatches,
        )
        return ts, None

    ts, _ = jax.lax.scan(update_epoch, ts, None, config.num_epochs)
    return ts


def collect_trajectories(config, ts):
    def env_step(ts, unused):
        # Get keys for sampling action and stepping environment
        ts, rng = ts.get_rng()
        rng_steps, rng_action = jax.random.split(rng, 2)
        rng_steps = jax.random.split(rng_steps, config.num_envs)

        # Sample action
        action, log_prob = ts.apply_fn(
            ts.params, ts.last_obs, rng_action, method="action_log_prob"
        )
        value = ts.apply_fn(ts.params, ts.last_obs, method="value")

        # Step environment
        vmap_step = jax.vmap(config.env.step, in_axes=(0, 0, 0, None))
        transition = vmap_step(rng_steps, ts.env_state, action, config.env_params)
        next_obs, env_state, reward, done, _ = transition

        # Return updated runner state and transition
        transition = Trajectory(ts.last_obs, action, log_prob, reward, value, done)
        ts = ts.replace(env_state=env_state, last_obs=next_obs, last_done=done)
        return ts, transition

    ts, trajectories = jax.lax.scan(env_step, ts, None, config.num_steps)
    return ts, trajectories


def calculate_gae(config, trajectories, last_val):
    def get_advantages(advantage_and_next_value, transition):
        advantage, next_value = advantage_and_next_value
        delta = (
            transition.reward.squeeze()  # For gymnax envs that return shape (1, )
            + config.gamma * next_value * (1 - transition.done)
            - transition.value
        )
        advantage = (
            delta + config.gamma * config.gae_lambda * (1 - transition.done) * advantage
        )
        return (advantage, transition.value), advantage

    _, advantages = jax.lax.scan(
        get_advantages,
        (jnp.zeros_like(last_val), last_val),
        trajectories,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + trajectories.value


def make_minibatches(config, trajectories, advantages, targets, rng):
    iteration_size = config.minibatch_size * config.num_minibatches
    permutation = jax.random.permutation(rng, iteration_size)

    def shuffle_and_split(x):
        x = x.reshape((iteration_size, *x.shape[2:]))
        x = jnp.take(x, permutation, axis=0)
        return x.reshape(config.num_minibatches, -1, *x.shape[1:])

    batch = AdvantageMinibatch(trajectories, advantages, targets)
    minibatches = jax.tree_util.tree_map(shuffle_and_split, batch)
    return minibatches


def update(config, ts, batch):
    def loss_fn(params):
        value = config.agent.apply(params, batch.trajectories.obs, method="value")
        log_prob, entropy = config.agent.apply(
            params,
            batch.trajectories.obs,
            batch.trajectories.action,
            method="log_prob_entropy",
        )
        entropy = entropy.mean()

        # Calculate actor loss
        ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
        advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
        )
        clipped_ratio = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
        actor_loss1 = ratio * advantages
        actor_loss2 = clipped_ratio * advantages
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

        # Calculate value loss
        value_pred_clipped = batch.trajectories.value + (
            value - batch.trajectories.value
        ).clip(-config.clip_eps, config.clip_eps)
        value_losses = jnp.square(value - batch.targets)
        value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # Return total loss
        total_loss = (
            actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy
        )
        return total_loss

    grads = jax.grad(loss_fn)(ts.params)
    ts = ts.apply_gradients(grads=grads)
    return ts
