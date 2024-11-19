import importlib

from gymnax import make

_create_fns = {
    "brax": ("rejax.compat.brax2gymnax", "create_brax"),
    "navix": ("rejax.compat.navix2gymnax", "create_navix"),
    "jumanji": ("rejax.compat.jumanji2gymnax", "create_jumanji"),
    "gymnasium": ("rejax.compat.gymnasium2gymnax", "create_gymnasium"),
}


def create(env, **kwargs):
    if len(split := env.split("/", 1)) == 1:
        return make(env, **kwargs)

    prefix, env_name = split
    module, create_fn = _create_fns[prefix]
    module = importlib.import_module(module)
    create_fn = getattr(module, create_fn)
    return create_fn(env_name, **kwargs)


__all__ = ["create"]
