from craftax.craftax_env import make_craftax_env_from_name


def create_craftax(env_name, auto_reset=True):
    env = make_craftax_env_from_name(env_name, auto_reset=auto_reset)
    return env, env.default_params
