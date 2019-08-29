#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

from setuptools import setup


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name="fit_nbinom",
    version="1.0.0",
    author="Gökçen Eraslan, Yui Tomo",
    packages=["fit_nbinom"],
    url="https://github.com/t-yui/fit_nbinom",
    license="GNU General Public License",
    description="Negative binomial maximum likelihood estimate implementation in Python using L-BFGS-B",
    install_requires=read_requirements()
)
