from setuptools import find_packages, setup

setup(
    name="fastrl",
    version="0.0.1",
    url="https://github.com/",
    author="...",
    description="Minimal implementations of RL algorithms in jax",
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
