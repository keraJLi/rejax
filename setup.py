from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="rejax",
    version="0.1.0",
    url="https://github.com/keraJLi/rejax",
    author="Jarek Liesen",
    description="Lightweight library of RL algorithms in Jax",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "flax",
        "gymnax",
        "distrax",
        "optax",
        "numpy",
        "brax",
        "evosax",
        "pyyaml",
    ],
)
