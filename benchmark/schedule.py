import os
import sys
import argparse
import subprocess
from yaml import load, Loader
from itertools import product


SLURM_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --output={results_dir}/{run_name}_%j.out
#SBATCH --partition=gpu,scioi_gpu,ex_scioi_gpu,ex_scioi_a100nv
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00

eval "$(conda shell.bash hook)"
conda activate jax

export http_proxy=http://frontend01:3128/
export https_proxy=http://frontend01:3128/
export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY={wandb_key}

python train.py \
    --config {config} \
    --algorithm {algorithm} \
    --num-seeds {num_seeds} \
    --log-dir {results_dir} \
    --use-wandb \
    --wandb-project purerl-test \
    --wandb-entity kerajli
"""


def run(results_dir, config, algorithm, num_seeds, wandb_key):
    config_fname = os.path.split(config)[-1].replace(".yaml", "")
    run_name = f"purerl_bench_{config_fname}_{algorithm}_{num_seeds}"
    run_file = f"{run_name}.sbatch.tmp"
    with open(run_file, "w+") as f:
        f.write(
            SLURM_SCRIPT.format(
                run_name=run_name,
                results_dir=results_dir,
                config=config,
                algorithm=algorithm,
                num_seeds=num_seeds,
                wandb_key=wandb_key,
            )
        )
    subprocess.run(["sbatch", run_file], stdout=sys.stdout, stderr=sys.stderr)
    os.remove(run_file)


def main(args):
    with open(args.config, "r") as f:
        config = load(f, Loader=Loader)

    algorithms = config.keys()
    num_seeds = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    for algorithm, num_seeds in list(product(algorithms, num_seeds)):
        run(args.results_dir, args.config, algorithm, num_seeds, args.wandb_key)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="Path to hyperparameter config.",
    required=True,
)
parser.add_argument(
    "--wandb-key",
    type=str,
    help="Wandb API key.",
    required=True,
)
parser.add_argument(
    "--results-dir",
    type=str,
    default="results",
    help="Directory to store results.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)