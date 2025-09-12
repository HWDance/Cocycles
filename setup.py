try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='causal_cocycle',
    version="0.0.1dev0",
    description='cocycles',
    author='Hugh Dance',
    packages=['causal_cocycle'],
    python_requires='>=3.8',
)