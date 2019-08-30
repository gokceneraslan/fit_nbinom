#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from logzero import logger

from fit_nbinom.fit_nbinom import fit_nbinom
from fit_nbinom.fisher_infomation import fisher_information
from fit_nbinom.fit_model import Fit_model


def nbinom_mle(X, seed=1):
    """
    Calculate maximum likelihood estimators and standard errors

    Parameters
    ----------
    X : list or ndarray
        a list or a numpy array representing the data
    seed : int32
        seed number of random when monte carlo calculation of fisher information

    Returns
    -------
    nbinomfit : instance variable
        instance to show and deliver results
    """

    if isinstance(X, list):
        X = np.array(X)
    elif type(X).__module__ == np.__name__:
        pass
    else:
        logger.error("Arg X must be list or numpy 1d array")
        exit(1)

    params_dict = fit_nbinom(X)
    r = params_dict['size']
    p = params_dict['prob']
    mu = (1 - p) * r / p

    params = (r, p)
    np.random.seed(seed=seed)
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
    nbinomfit = Fit_model(results)
    return nbinomfit