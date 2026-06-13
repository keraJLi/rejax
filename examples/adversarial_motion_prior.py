"""
This example demonstrates how to train a PPO policy on the Humanoid environment using Brax.
"""
import jax
import jax.numpy as jnp
import chex
from rejax import PPO, PPOAMP
import minari

CONFIG = {
    "algo": "PPOAMP",
    "env": "brax/ant",
    "agent_kwargs": {
        "activation": "relu",
        "hidden_layer_sizes": (256, 256),
    },
    "total_timesteps": 10_000_000,
    "eval_freq": 100_000,
    "num_envs": 2048,
    "num_steps": 8,
    "num_epochs": 4,
    "num_minibatches": 8,
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "normalize_rewards": True,
    "normalize_observations": True,
}

AMP_EXTRA_CONFIG = {
    "amp_data": "mujoco/ant/expert-v0",
    "discriminator_kwargs": {
        "activation": "relu",
        "hidden_layer_sizes": (256, 256),
    },
    "amp_learning_rate": 3e-4,
    "gp_lambda": 7.0,
}


def load_minari_data(dataset_id: str, num_episodes: int = 10) -> chex.Array:
    dataset = minari.load_dataset(dataset_id, download=True)
    observations = []
    for episode in dataset.sample_episodes(num_episodes):
        observations.append(episode.observations[..., :-78])  # remove contact forces
    return jnp.concatenate(observations)

def main():
    algo_class = PPO
    if CONFIG["algo"] == "PPOAMP":
      algo_class = PPOAMP
      CONFIG.update(AMP_EXTRA_CONFIG) 
      CONFIG["amp_data"] = load_minari_data(CONFIG["amp_data"])
    del CONFIG["algo"]
    algo = algo_class.create(**CONFIG)
    rng = jax.random.PRNGKey(42)

    eval_callback = algo.eval_callback
    def eval_with_print(c, ts, rng):
      _, returns = eval_callback(c, ts, rng)
      jax.debug.print("Step: {}, Mean return: {}", ts.global_step, returns.mean())
      return ()
    algo = algo.replace(eval_callback=eval_with_print)

    print("Compiling...")
    compiled_train = jax.jit(algo.train).lower(rng).compile()

    print("Training...")
    compiled_train(rng)


if __name__ == "__main__":
    main()