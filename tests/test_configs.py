import os
import unittest

from yaml import safe_load

from rejax import get_algo


class TestConfigs(unittest.TestCase):
    def setUp(self) -> None:
        configs = {}
        for root, _, files in os.walk("configs"):
            for file in files:
                if file.endswith(".yaml"):
                    with open(os.path.join(root, file)) as f:
                        configs[os.path.join(root, file)] = safe_load(f)
        self.configs = configs

    def test_configs(self) -> None:
        for config_path, configs_env in self.configs.items():
            for algo, config in configs_env.items():
                if config.get("env", "").startswith("navix"):
                    continue

                with self.subTest(config_opath=config_path, algo=algo):
                    try:
                        algo_cls = get_algo(algo)
                        algo_cls.create(**config)
                    except Exception as e:
                        self.fail(
                            f"Failed to create {algo} with config '{config_path}': "
                            f"{type(e).__name__}: {str(e)}"
                        )
