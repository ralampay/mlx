"""Paired superiority and equivalence statistics for repeated experiments."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev
from typing import Sequence

from mlx.core.exceptions import MLXUserError


@dataclass(frozen=True)
class PairedComparisonResult:
    n: int
    mean_difference: float
    median_difference: float
    standard_deviation: float
    exact_sign_flip_p_two_sided: float
    bootstrap_95_ci: tuple[float, float]
    equivalence_margin: float
    tost_lower_p: float
    tost_upper_p: float
    tost_90_ci: tuple[float, float]
    equivalent: bool
    decision: str

    def to_dict(self) -> dict:
        return asdict(self)


class AnalyzePairedDifferences:
    """Analyze candidate-minus-control differences declared before evaluation."""

    def __init__(
        self,
        differences: Sequence[float],
        *,
        alpha: float = 0.05,
        equivalence_margin: float = 0.02,
        bootstrap_draws: int = 100_000,
        bootstrap_seed: int = 20261003,
    ) -> None:
        self.differences = tuple(float(value) for value in differences)
        self.alpha = float(alpha)
        self.equivalence_margin = float(equivalence_margin)
        self.bootstrap_draws = int(bootstrap_draws)
        self.bootstrap_seed = int(bootstrap_seed)

    def execute(self) -> PairedComparisonResult:
        self._validate()
        try:
            from scipy import stats
        except ImportError as exc:
            raise MLXUserError(
                "Paired equivalence analysis requires scipy. Install project dependencies."
            ) from exc

        values = self.differences
        observed = mean(values)
        permutations = (
            mean(tuple(value * sign for value, sign in zip(values, signs)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        )
        sign_flip_p = sum(
            abs(permuted) >= abs(observed) - 1e-15 for permuted in permutations
        ) / (2 ** len(values))

        rng = random.Random(self.bootstrap_seed)
        bootstrapped = sorted(
            mean(tuple(rng.choice(values) for _ in values))
            for _ in range(self.bootstrap_draws)
        )
        bootstrap_ci = (
            self._quantile(bootstrapped, 0.025),
            self._quantile(bootstrapped, 0.975),
        )

        margin = self.equivalence_margin
        lower = stats.ttest_1samp(values, -margin, alternative="greater")
        upper = stats.ttest_1samp(values, margin, alternative="less")
        sem = stats.sem(values)
        critical = stats.t.ppf(0.95, len(values) - 1)
        tost_ci = (observed - critical * sem, observed + critical * sem)
        equivalent = bool(lower.pvalue < self.alpha and upper.pvalue < self.alpha)
        if sign_flip_p < self.alpha:
            decision = "favor_candidate" if observed > 0 else "favor_control"
        elif equivalent:
            decision = "practically_equivalent"
        else:
            decision = "inconclusive"
        return PairedComparisonResult(
            n=len(values),
            mean_difference=observed,
            median_difference=median(values),
            standard_deviation=stdev(values),
            exact_sign_flip_p_two_sided=sign_flip_p,
            bootstrap_95_ci=bootstrap_ci,
            equivalence_margin=margin,
            tost_lower_p=float(lower.pvalue),
            tost_upper_p=float(upper.pvalue),
            tost_90_ci=(float(tost_ci[0]), float(tost_ci[1])),
            equivalent=equivalent,
            decision=decision,
        )

    def _validate(self) -> None:
        if len(self.differences) < 2:
            raise MLXUserError("Paired analysis requires at least two complete pairs.")
        if len(self.differences) > 20:
            raise MLXUserError(
                "Exact sign-flip analysis is limited to 20 pairs to bound runtime."
            )
        if not all(math.isfinite(value) for value in self.differences):
            raise MLXUserError("Paired differences must all be finite.")
        if not 0.0 < self.alpha < 1.0:
            raise MLXUserError("alpha must be between zero and one.")
        if self.equivalence_margin <= 0 or not math.isfinite(
            self.equivalence_margin
        ):
            raise MLXUserError("equivalence_margin must be finite and positive.")
        if self.bootstrap_draws < 1:
            raise MLXUserError("bootstrap_draws must be positive.")

    @staticmethod
    def _quantile(values: Sequence[float], probability: float) -> float:
        position = (len(values) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(values[lower])
        fraction = position - lower
        return float(values[lower] * (1 - fraction) + values[upper] * fraction)
