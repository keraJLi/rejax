"""Train MPO on Brax InvertedPendulum."""

import jax

from rejax import MPO


CONFIG = {
    "env": "brax/inverted_pendulum",
    "agent_kwargs": {"activation": "tanh"},
    "total_timesteps": 300_000,
    "eval_freq": 30_000,
    "num_envs": 32,
    "buffer_size": 100_000,
    "fill_buffer": 2_000,
    "batch_size": 256,
    "num_epochs": 16,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "num_action_samples": 20,
    "policy_eval_num_val_samples": 128,
    "normalize_observations": True,
    "skip_initial_evaluation": True,
}


def main():
    algo = MPO.create(**CONFIG)
    eval_callback = algo.eval_callback

    def log_callback(algo, train_state, rng):
        lengths, returns = eval_callback(algo, train_state, rng)
        jax.debug.print(
            "step: {}, episode length: {:.1f}, return: {:.1f}",
            train_state.global_step,
            lengths.mean(),
            returns.mean(),
        )
        return lengths, returns

    algo = algo.replace(eval_callback=log_callback)
    rng = jax.random.PRNGKey(0)

    _, (_, returns) = jax.jit(algo.train)(rng)
    print(f"Final mean return: {returns.mean(axis=-1)[-1]:.1f}")


if __name__ == "__main__":
    main()
