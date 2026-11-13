from setuptools import find_packages, setup

setup(
    name="franka_pick_place",
    version="0.1.0",
    description="Franka pick-and-place RL task (Isaac Lab extension) - Workflow 1: reward design",
    packages=find_packages(),
    python_requires=">=3.10",
)
