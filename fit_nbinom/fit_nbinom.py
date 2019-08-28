#!/usr/bin/env python
# -*- coding: utf-8 -*-

# fit_nbinom
# Copyright (C) 2014 Gokcen Eraslan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import numpy as np
from scipy.special import gammaln
from scipy.special import psi
from scipy.special import factorial
from scipy.optimize import fmin_l_bfgs_b as optim
from scipy.special import polygamma

from fit_model import fit_model


def log_likelihood(params, *args):
    r, p = params
    X = args[0]
    N = X.size
    infinitesimal = np.finfo(np.float).eps

    # MLE estimate based on the formula on Wikipedia:
    # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
    result = (
        np.sum(gammaln(X + r))
        - np.sum(np.log(factorial(X)))
        - N * (gammaln(r))
        + N * r * np.log(p)
        + np.sum(X * np.log(1 - (p if p < 1 else 1 - infinitesimal)))
    )

    return -result


def log_likelihood_deriv(params, *args):
    r, p = params
    X = args[0]
    N = X.size
    infinitesimal = np.finfo(np.float).eps

    pderiv = (N * r) / p - np.sum(X) / (1 - (p if p < 1 else 1 - infinitesimal))
    rderiv = np.sum(psi(X + r)) - N * psi(r) + N * np.log(p)

    return np.array([-rderiv, -pderiv])


def fisher_information(params, *args):
    """
    Calculate fisher information about each params
    This function is written by Yui Tomo
    """
    r, p = params
    X = args[0]
    N = X.size
    infinitesimal = np.finfo(np.float).eps

    n_rand = 100000
    rands = np.random.negative_binomial(n=r, p=p, size=(n_rand, 1))

    # calculate expectation using monte carlo method
    # https://en.wikipedia.org/wiki/Fisher_information#Definition
    # https://stats.stackexchange.com/questions/97715/fisher-information-matrix-of-negative-binomial-distribution
    r_info = (-np.mean(polygamma(1, rands + r) - polygamma(1, r)) - (1 - p) / r) * N
    mu_info = ((p ** 2) / (r * (1 - (p if p < 1 else 1 - infinitesimal)))) * N

    information = {"size": r_info, "mu": mu_info}
    return information


class fit_nbinom:
    def __init__(self, X):
        self.X = X

    def fit(self):
        # X is a numpy array representing the data
        # initial params is a numpy array representing the initial values of
        # size and prob parameters
        infinitesimal = np.finfo(np.float).eps
        X = self.X

        # reasonable initial values (from fitdistr function in R)
        m = np.mean(X)
        v = np.var(X)
        size = (m ** 2) / (v - m) if v > m else 10

        # convert mu/size parameterization to prob/size
        p0 = size / ((size + m) if size + m != 0 else 1)
        r0 = size
        initial_params = np.array([r0, p0])

        bounds = [(infinitesimal, None), (infinitesimal, 1)]
        optimres = optim(
            log_likelihood,
            x0=initial_params,
            fprime=log_likelihood_deriv,
            args=(X,),
            approx_grad=1,
            bounds=bounds,
        )

        params = optimres[0]
        r = params[0]
        p = params[1]
        mu = (1 - p) * r / p

        information = fisher_information(params, X)
        r_info = information["size"]
        mu_info = information["mu"]
        N = X.size

        results = {
            "params": {"size": r, "mu": mu},
            "stderr": {
                "size": np.sqrt(1 / (r_info * N)),
                "mu": np.sqrt(1 / (mu_info * N)),
            },
        }
        nbinomfit = fit_model(results)
        return nbinomfit


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: %s size_param prob_param" % sys.argv[0])
        exit()

    size = float(sys.argv[1])
    mu = float(sys.argv[2])
    p = size / (size + mu)

    testset = np.random.negative_binomial(n=size, p=p, size=1000)

    nbinomfit = fit_nbinom(testset).fit()
    nbinomfit.summary()
