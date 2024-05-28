from setuptools import find_packages, setup

setup(
    name="rejax",
    version="0.0.1",
    url="https://github.com/keraJLi/rejax",
    author="Jarek Liesen",
    description="Lightweight library of RL algorithms in Jax",
    packages=find_packages(),
    install_requires=[
        "flax",
        "gymnax",
        "distrax",
        "optax",
        "numpy",
        "brax",
        "evosax",
    ],
)
