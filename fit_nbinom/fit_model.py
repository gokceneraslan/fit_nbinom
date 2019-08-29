#!/usr/bin/env python
# -*- coding: utf-8 -*-

# fit_model
# Instance generated from fit_model class works
# when users get information about estimated parameters.

from tabulate import tabulate
from scipy.stats import norm


class fit_model:
    def __init__(self, results):
        self.results = results

    def summary(self, tablefmt="simple"):
        alpha=0.05
        params = self.results["params"]
        stderr = self.results["stderr"]
        norm_lower = norm.ppf(alpha / 2)
        norm_upper = norm.ppf(1 - alpha / 2)

        summary_list = []
        headers = [
            "parameter",
            "estimate",
            "std err",
            str(int(100 * (1 - alpha))) + "%CI lower",
            "95%CI upper",
        ]

        for par in ["size", "mu"]:
            ci_lower = params[par] + stderr[par] * norm_lower
            ci_upper = params[par] + stderr[par] * norm_upper
            table_row = [par, params[par], stderr[par], ci_lower, ci_upper]
            summary_list.append(table_row)

        summary_table = tabulate(summary_list, headers=headers, tablefmt=tablefmt)
        print(summary_table)

    def params(self):
        return self.results["params"]

    def stderr(self):
        return self.results["stderr"]
