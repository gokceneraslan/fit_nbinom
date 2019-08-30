#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import polygamma
from logzero import logger

from fit_nbinom.fit_nbinom import fit_nbinom
from fit_nbinom.fit_model import Fit_model


def fisher_information(params, *args):
    """
    Calculate fisher information of each parameters
    """
    r, p = params
    X = args[0]
    N = X.size
    infinitesimal = np.finfo(np.float).eps

    # calculate expectation using monte carlo method
    # https://en.wikipedia.org/wiki/Fisher_information#Definition
    # https://stats.stackexchange.com/questions/97715/fisher-information-matrix-of-negative-binomial-distribution
    n_rand = 100000
    rands = np.random.negative_binomial(n=r, p=p, size=(n_rand, 1))
    r_info = (-np.mean(polygamma(1, rands + r) - polygamma(1, r)) - (1 - p) / r) * N
    mu_info = ((p ** 2) / (r * (1 - (p if p < 1 else 1 - infinitesimal)))) * N

    information = {"size": r_info, "mu": mu_info}
    return information