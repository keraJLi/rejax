"""
This example demonstrates how to log to wandb during training.
"""

import jax
import wandb

from rejax import PPO


CONFIG = {
    "env": "brax/ant",
    "env_params": {"backend": "positional"},
    "agent_kwargs": {"activation": "relu"},
    "total_timesteps": 10_000_000,
    "eval_freq": 100_000,
    "num_envs": 2_000,
    "num_steps": 5,
    "num_epochs": 4,
    "num_minibatches": 4,
    "learning_rate": 0.0003,
    "max_grad_norm": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
}

wandb.init(project="my-awesome-project", config=CONFIG)

ppo = PPO.create(**CONFIG)
eval_callback = ppo.eval_callback


def wandb_callback(ppo, train_state, rng):
    lengths, returns = eval_callback(ppo, train_state, rng)

    def log(step, data):
        # io_callback returns np.array, which wandb does not like.
        # In jax 0.4.27, this becomes a jax array, should check when upgrading...
        step = step.item()
        wandb.log(data, step=step)

    jax.experimental.io_callback(
        log,
        (),  # result_shape_dtypes (wandb.log returns None)
        train_state.global_step,
        {"episode_length": lengths.mean(), "return": returns.mean()},
    )

    # Since we log to wandb, we don't want to return anything that is collected
    # throughout training
    return ()


ppo = ppo.replace(eval_callback=wandb_callback)

rng = jax.random.PRNGKey(0)
print("Compiling...")
compiled_train = jax.jit(ppo.train).lower(rng).compile()
print("Training...")
compiled_train(rng)
