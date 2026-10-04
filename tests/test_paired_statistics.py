from __future__ import annotations

import pytest

from mlx.core.exceptions import MLXUserError
from mlx.core.paired_statistics import AnalyzePairedDifferences


def test_paired_statistics_favors_candidate_for_consistent_positive_effect():
    result = AnalyzePairedDifferences(
        [0.03, 0.04, 0.05, 0.035, 0.045, 0.04],
        bootstrap_draws=500,
        bootstrap_seed=7,
    ).execute()
    assert result.decision == "favor_candidate"
    assert result.exact_sign_flip_p_two_sided == pytest.approx(0.03125)
    assert result.bootstrap_95_ci[0] > 0


def test_paired_statistics_can_establish_equivalence_not_just_nonsignificance():
    result = AnalyzePairedDifferences(
        [-0.004, 0.003, -0.002, 0.004, 0.0, -0.001, 0.002, -0.003],
        equivalence_margin=0.02,
        bootstrap_draws=500,
    ).execute()
    assert result.decision == "practically_equivalent"
    assert result.equivalent is True


def test_paired_statistics_rejects_an_underidentified_analysis():
    with pytest.raises(MLXUserError, match="at least two"):
        AnalyzePairedDifferences([0.1]).execute()
