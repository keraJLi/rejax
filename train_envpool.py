import jax
import yaml

from rejax import get_algo
from rejax.compat.envpool2gymnax import use_envpool


def main(algo_str, config, seed_id):
    algo_cls = get_algo(algo_str)
    algo_cls = use_envpool(algo_cls)
    algo = algo_cls.create(**config)

    old_eval_callback = algo.eval_callback

    def eval_callback(algo, ts, rng):
        lengths, returns = old_eval_callback(algo, ts, rng)
        jax.debug.print(
            "Step {}, Mean episode length: {}, Mean return: {}",
            ts.global_step,
            lengths.mean(),
            returns.mean(),
        )
        return lengths, returns

    algo = algo.replace(eval_callback=eval_callback)

    print(algo.config)
    rng = jax.random.PRNGKey(seed_id)
    ts, (_, returns) = jax.jit(algo.train)(rng)
    print(f"Achieved mean return of {returns.mean(axis=-1)[:, -1]}")


if __name__ == "__main__":
    import argparse

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
        "--seed-id",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())[args.algorithm]

    main(args.algorithm, config, args.seed_id)
