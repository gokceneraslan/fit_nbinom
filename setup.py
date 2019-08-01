#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup

setup(
    name="fit_nbinom",
    version="1.0.0",
    author="Gökçen Eraslan, Yui Tomo",
    packages=["fit_nbinom"],
    url="https://github.com/t-yui/fit_nbinom",
    license="GNU General Public License",
    description="Calculate parameters of Negative-Binomial distribution via maximum likelihood method.",
    install_requires=["numpy", "scipy"]
)
