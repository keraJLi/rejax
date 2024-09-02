from functools import wraps

from flax import struct

from .evaluator import EnvpoolEvaluator
from .wrapper import create_envpool


def use_envpool(algo_cls):
    @wraps(algo_cls, updated=())
    class EnvpoolAlgorithm(EnvpoolMixin, algo_cls):
        pass

    return EnvpoolAlgorithm


class EnvpoolMixin(struct.PyTreeNode):
    eval_num_envs: int = struct.field(pytree_node=False, default=16)

    @classmethod
    def create(cls, **config):
        env_name = config["env"]
        env_params_dict = config.get("env_params", {})
        env, env_params = cls.create_env(config)
        agent = cls.create_agent(config, env, env_params)

        # For evaluation, overwrite number of environments
        eval_num_envs = config.get(
            "eval_num_envs", cls.__dataclass_fields__["eval_num_envs"].default
        )
        env_params_dict["num_envs"] = eval_num_envs
        evaluator = EnvpoolEvaluator(env_name=env_name, env_params=env_params_dict)

        def eval_callback(algo, ts, rng):
            act = algo.make_act(ts)
            return evaluator.evaluate(act, rng)

        return cls(
            env=env,
            env_params=env_params,
            eval_callback=eval_callback,
            **agent,
            **config,
        )

    @classmethod
    def create_env(cls, config):
        assert isinstance(config["env"], str)
        env_params = config.pop("env_params", {})
        env_params["num_envs"] = config.get(
            "num_envs", cls.__dataclass_fields__["num_envs"].default
        )
        env, env_params = create_envpool(config.pop("env"), **env_params)
        return env, env_params

    @property
    def vmap_reset(self):
        return self.env.reset

    @property
    def vmap_step(self):
        return self.env.step

    def replace(self, **kwargs):
        if "num_envs" in kwargs:
            raise ValueError("num_envs cannot be changed.")
        return super().replace(**kwargs)
