import jax
import timeit
import jax.numpy as jnp
from matplotlib import pyplot as plt

from purerl.algos import get_agent


def main(algo_str, config, seed_id, time_fit):
    train_fn, config_cls = get_agent(algo_str)
    train_config = config_cls.from_dict(config)

    # Train it
    num_seeds = 5
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)

    param = "gamma"
    param_values = jnp.array([0.98])
    train_config = train_config.replace(**{param: param_values})
    axes = jax.tree_map(lambda _: None, train_config).replace(**{param: 0})

    vmap_train = jax.vmap(train_fn, in_axes=(axes, None))
    vmap_train = jax.vmap(vmap_train, in_axes=(None, 0))
    _, (_, returns) = vmap_train(train_config, keys)
    returns = returns.mean(axis=-1)

    for i in range(len(param_values)):
        plt.plot(returns[:, i].mean(axis=0), c=f"C{i}", label=param_values[i])
        for j in range(num_seeds):
            plt.plot(returns[j, i], c=f"C{i}", alpha=0.1)

    plt.legend(title=param)
    plt.show()

    # colors = plt.cm.cool(jnp.linspace(0, 1, num_seeds))
    # for i in range(num_seeds):
    #     plt.plot(returns.mean(axis=-1)[i], c=colors[i])
    # plt.show()

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

    args, _ = parser.parse_known_args()
    config = load_config(args.config, True)
    main(args.algorithm, config[args.algorithm], args.seed_id, args.time_fit)
