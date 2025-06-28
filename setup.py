from setuptools import setup, find_packages

setup(
    name = "river-evaluate",
    version="1.0",
    description="Evaluate river models",
    author="Caio Gimenes",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)