import timeit

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from rejax import get_algo


def main(algo_str, config, seed_id, num_seeds, time_fit):
    train_fn, config_cls = get_algo(algo_str)
    old_train_config = config_cls.from_dict(config)

    def eval_callback(config, ts, rng):
        lengths, returns = old_train_config.eval_callback(config, ts, rng)
        jax.debug.print(
            "Step {}, Mean episode length: {}, Mean return: {}",
            ts.global_step,
            lengths.mean(),
            returns.mean(),
        )
        return lengths, returns

    train_config = old_train_config.replace(eval_callback=eval_callback)

    # Train it
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)

    vmap_train = jax.jit(jax.vmap(train_fn, in_axes=(None, 0)))
    _, (_, returns) = vmap_train(train_config, keys)
    print(f"Achieved mean return of {returns.mean(axis=-1)[:, -1]}")

    t = jnp.arange(returns.shape[1]) * train_config.eval_freq
    colors = plt.cm.cool(jnp.linspace(0, 1, num_seeds))
    for i in range(num_seeds):
        plt.plot(t, returns.mean(axis=-1)[i], c=colors[i])
    plt.show()

    if time_fit:
        print("Fitting 3 times, getting a mean time of... ", end="", flush=True)

        def time_fn():
            return vmap_train(train_config, keys)

        time = timeit.timeit(time_fn, number=3) / 3
        print(
            f"{time:.1f} seconds total, equalling to "
            f"{time / num_seeds:.1f} seconds per seed"
        )


if __name__ == "__main__":
    import argparse

    from mle_logging import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cartpole.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--time-fit",
        action="store_true",
        help="Time how long it takes to fit the agent by fitting 3 times.",
    )
    parser.add_argument(
        "--seed_id",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to roll out.",
    )

    args, _ = parser.parse_known_args()
    config = load_config(args.config, True)
    main(
        args.algorithm,
        config[args.algorithm],
        args.seed_id,
        args.num_seeds,
        args.time_fit,
    )
