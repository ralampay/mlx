"""Seed-paired, two-sided superiority summaries without equivalence claims."""

from __future__ import annotations

import itertools
import math
import numpy as np

from mlx.core.exceptions import MLXUserError


class AnalyzePairedSuperiority:
    def __init__(self, candidate, baseline):
        self.candidate, self.baseline = candidate, baseline

    def execute(self):
        from scipy import stats

        a, b = np.asarray(self.candidate, dtype=float), np.asarray(
            self.baseline, dtype=float
        )
        if (
            a.ndim != 1
            or a.shape != b.shape
            or a.size < 2
            or not np.isfinite([a, b]).all()
        ):
            raise MLXUserError(
                "Paired analysis requires at least two finite, aligned observations"
            )
        d = a - b
        n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
        se = sd / math.sqrt(n)
        width = float(stats.t.ppf(0.975, n - 1)) * se
        t = mean / se if se else None
        p = (
            float(2 * stats.t.sf(abs(t), n - 1))
            if t is not None
            else (1.0 if mean == 0 else 0.0)
        )
        exact = None
        if n <= 20:
            extreme = sum(
                abs(float(np.mean(d * np.asarray(signs)))) >= abs(mean) - 1e-14
                for signs in itertools.product((-1, 1), repeat=n)
            )
            exact = extreme / 2**n
        return {
            "n": n,
            "df": n - 1,
            "mean_difference": mean,
            "difference_sd": sd,
            "ci95_low": mean - width,
            "ci95_high": mean + width,
            "t": t,
            "p": p,
            "paired_dz": mean / sd if sd else None,
            "exact_sign_flip_p": exact,
            "degenerate_variance": sd == 0,
        }


def holm_adjust(pvalues):
    values = list(pvalues)
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, previous = [0.0] * len(values), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, values[index] * (len(values) - rank)))
        result[index] = previous
    return result
