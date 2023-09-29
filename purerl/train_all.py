import jax
import timeit
import jax.numpy as jnp
from flax import serialization
from matplotlib import pyplot as plt

from purerl.algos import get_agent


def main(algo_str, config, seed_id):
    train_fn, config_cls = get_agent(algo_str)
    train_config = config_cls.from_dict(config)

    # Train it
    key = jax.random.PRNGKey(seed_id)
    train_state, (lengths, returns) = train_fn(train_config, key)

    plt.plot(returns.mean(axis=-1))
    plt.show()

    with open(f"{algo_str}_{train_config.env.name}.pkl", "wb+") as f:
        f.write(serialization.to_bytes((train_state, train_config, lengths, returns)))


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
        "--seed_id",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    args, _ = parser.parse_known_args()
    config = load_config(args.config, True)
    main(
        args.algorithm,
        config[args.algorithm],
        args.seed_id,
    )
