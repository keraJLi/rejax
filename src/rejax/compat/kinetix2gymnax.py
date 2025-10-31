from kinetix.environment import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.util import load_from_json_file


def create_kinetix(env_name, **kwargs):
    """Create a Kinetix environment. The `env_name` argument is the name of the
    level file, e.g. `"l/grasp_easy"`.
    """
    level, static_env_params, env_params = load_from_json_file(f"{env_name}.json")

    default_kwargs = {
        "action_type": ActionType.CONTINUOUS,
        "observation_type": ObservationType.SYMBOLIC_FLAT,
        "reset_fn": lambda rng: level,
        "env_params": env_params,
        "static_env_params": static_env_params,
    }
    default_kwargs.update(kwargs)

    env = make_kinetix_env(**default_kwargs)
    return env, env_params
