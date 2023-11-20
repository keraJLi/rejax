import jax
from jax import numpy as jnp

"""
This example demonstrates how to modify an existing algorithm. To this end, we change
the loss function of PPO to be the one from "Discovered Policy Optimisation" by Chris Lu
et al. (https://arxiv.org/abs/2210.05639). 
"""

from purerl.algos import PPO, PPOConfig


class DPO(PPO):
    @classmethod
    def update(cls, config, ts, batch):
        def v_loss_fn(params):
            # Standard value loss
            value = config.agent.apply(params, batch.trajectories.obs, method="value")
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-config.clip_eps, config.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return value_loss

        def pi_loss_fn(params):
            log_prob, entropy = config.agent.apply(
                params,
                batch.trajectories.obs,
                batch.trajectories.action,
                method="log_prob_entropy",
            )
            entropy = entropy.mean()

            # Calculate actor loss
            alpha, beta = 2, 0.6
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            pi_loss1 = jnp.maximum(
                0,
                (ratio - 1) * advantages
                - alpha * jnp.tanh((ratio - 1) * advantages / alpha),
            )
            pi_loss2 = jnp.maximum(
                0,
                jnp.log(ratio) * advantages
                - beta * jnp.tanh(jnp.log(ratio) * advantages / beta),
            )
            pi_loss = jnp.where(advantages >= 0, pi_loss1, pi_loss2).mean()

            return entropy, pi_loss

        def loss_fn(params):
            v_loss = v_loss_fn(params)
            entropy, pi_loss = pi_loss_fn(params)
            return pi_loss + config.vf_coef * v_loss - config.ent_coef * entropy

        grads = jax.grad(loss_fn)(ts.params)
        ts = ts.apply_gradients(grads=grads)
        return ts


config = PPOConfig.from_dict({
    "env": "CartPole-v1",
    "total_timesteps": 100_000,
    "eval_freq": 5000,
    "num_envs": 10,
    "num_steps": 100,
    "num_epochs": 10,
    "num_minibatches": 10,
    "max_grad_norm": 0.5,
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
})

eval_callback = config.eval_callback

def eval_with_print(c, ts, rng):
    lengths, returns = eval_callback(c, ts, rng)
    jax.debug.print("{}", returns.mean())
    return lengths, returns

config = config.replace(eval_callback=eval_with_print)

DPO.train(config, jax.random.PRNGKey(0))
