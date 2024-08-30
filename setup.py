from setuptools import setup, find_packages

setup(
    name='af-sdd',
    packages=[
        package for package in find_packages() if 'scripts' not in package
    ],
    version='0.1.0',
)
