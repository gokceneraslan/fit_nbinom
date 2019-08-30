#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np

from fit_nbinom import nbinom_mle


class TestFitNbinom(unittest.TestCase):
    """
    test class of fit_nbinom.py
    """

    def test_fit_nbinom_array(self):
        """
        test method for fit_nbinom function
        """
        size = 5
        mu = 8

        p = size / (size + mu)
        X = np.random.negative_binomial(n=size, p=p, size=10000)

        res = nbinom_mle(X)
        size_hat = res.params()["size"]
        mu_hat = res.params()["mu"]

        self.assertEqual(size, round(size_hat))
        self.assertEqual(mu, round(mu_hat))

    def test_fit_nbinom_list(self):
        """
        test method for fit_nbinom function
        """
        size = 5
        mu = 8

        p = size / (size + mu)
        X = np.random.negative_binomial(n=size, p=p, size=10000)
        X = list(X)

        res = nbinom_mle(X)
        size_hat = res.params()["size"]
        mu_hat = res.params()["mu"]

        self.assertEqual(size, round(size_hat))
        self.assertEqual(mu, round(mu_hat))

    def test_fit_nbinom_stderr(self):
        """
        test method for fit_nbinom function
        """
        size = 5
        mu = 8

        p = size / (size + mu)
        X = np.random.negative_binomial(n=size, p=p, size=10000)
        X = list(X)

        res = nbinom_mle(X)
        size_se = res.stderr()["size"]
        mu_se = res.stderr()["mu"]

        # thresholds is calculated by fitdistr function of R
        # see `sim_nbinom.R` script of test directory
        size_se_lower = 0.0010
        size_se_upper = 0.0014
        mu_se_lower = 0.0004
        mu_se_upper = 0.0005

        self.assertGreater(size_se, size_se_lower)
        self.assertLess(size_se, size_se_upper)
        self.assertGreater(mu_se, mu_se_lower)
        self.assertLess(mu_se, mu_se_upper)


if __name__ == "__main__":
    unittest.main()
