# setup.py
from setuptools import setup, find_packages

setup(
    name="eddata",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
