#!/usr/bin/env python

## fit_nbinom
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

from __future__ import print_function

import numpy as np
from scipy.special import gammaln
from scipy.special import psi
from scipy.misc import factorial
from scipy.optimize import fmin_l_bfgs_b as optim

import sys

# X is a numpy array representing the data
# initial params is a numpy array representing the initial values of
# size and prob parameters
def fit_nbinom(X, initial_params = None):
    def log_likelihood(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        #MLE estimate based on the formula on Wikipedia:
        # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
        result = np.sum(gammaln(X + r)) \
                 - np.sum(np.log(factorial(X))) \
                 - N*(gammaln(r)) \
                 + N*r*np.log(p) \
                 + np.sum(X*np.log(1-(p if p < 1 else 1-1e-10)))

        return -result

    def log_likelihood_deriv(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        pderiv = N*r / (N*r + np.sum(X))
        rderiv = np.sum(psi(X + r)) \
                 - N*psi(r) \
                 + N*np.log(r/(r+np.mean(X)))

        return np.array([-rderiv, -pderiv])

    if initial_params is None:
        #reasonable initial values from fitdistr function in R
        m = np.mean(X)
        v = np.var(X)
        size = (m**2)/(v-m) if v > m else 10

        #convert mu/size parameterization to prob/size
        p0 = size / (size+m)
        r0 = size
        initial_params = np.array([r0, p0])

    bounds = [(1e-10, None), (1e-10, 1)]
    optimres = optim(log_likelihood,
                     x0=initial_params,
                     fprime=log_likelihood_deriv,
                     args = (X,),
                     bounds=bounds)

    params = optimres[0]
    return {'size': params[0], 'prob': params[1]}


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: %s size_param prob_param" % sys.argv[0])
        exit()

    testset = np.random.negative_binomial(float(sys.argv[1]),
                                          float(sys.argv[2]),
                                          1000)
    print(fit_nbinom(testset))
