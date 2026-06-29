"""Tests for the Deflated Sharpe Ratio chain, anchored on the Bailey & Lopez de
Prado (2014) worked example.

Example inputs: annualized SR_hat = 2.5 over T=1250 daily obs (250/yr), skew=-3,
non-excess kurt=10. Converting to per-observation: SR = 2.5/sqrt(250). With the
implied cross-trial Sharpe dispersion sqrt(Var[SR]) ~ 0.044736 (per-obs), the
paper reports DSR ~ 0.90 at N=100 (REJECT) and ~0.9505 at N=46 (ACCEPT). This
single anchor validates PSR, the expected-max benchmark, and their composition.
"""
import numpy as np
import pytest

from edge.stats import dsr

OBS_PER_YEAR = 250
SR_ANN = 2.5
SR_OBS = SR_ANN / np.sqrt(OBS_PER_YEAR)
N_OBS = 1250
SKEW = -3.0
KURT = 10.0
SQRT_VAR_SR = 0.044736            # per-obs dispersion implied by the example
VAR_SR = SQRT_VAR_SR ** 2


def test_bailey_ldp_example_reject_at_100_trials():
    out = dsr.deflated_sharpe_ratio(SR_OBS, N_OBS, SKEW, KURT, VAR_SR, n_trials=100)
    assert out["dsr"] == pytest.approx(0.90, abs=0.01)
    assert out["dsr"] < 0.95          # rejected at the 95% bar


def test_bailey_ldp_example_accept_at_46_trials():
    out = dsr.deflated_sharpe_ratio(SR_OBS, N_OBS, SKEW, KURT, VAR_SR, n_trials=46)
    assert out["dsr"] == pytest.approx(0.9505, abs=0.01)
    assert out["dsr"] > 0.95          # accepted


def test_more_trials_lower_dsr():
    # The spec's core point: trial count alone can decide significance.
    dsrs = [dsr.deflated_sharpe_ratio(SR_OBS, N_OBS, SKEW, KURT, VAR_SR, n)["dsr"]
            for n in (10, 50, 100, 500)]
    assert all(a >= b for a, b in zip(dsrs, dsrs[1:]))  # monotonically non-increasing


def test_psr_monotonic_in_sr():
    a = dsr.probabilistic_sharpe_ratio(0.10, 0.0, 1000, 0.0, 3.0)
    b = dsr.probabilistic_sharpe_ratio(0.20, 0.0, 1000, 0.0, 3.0)
    assert b > a


def test_negative_skew_high_kurt_lowers_psr():
    normal = dsr.probabilistic_sharpe_ratio(0.12, 0.0, 1000, 0.0, 3.0)
    fat = dsr.probabilistic_sharpe_ratio(0.12, 0.0, 1000, -3.0, 10.0)
    assert fat < normal


def test_expected_max_sharpe_increases_with_trials():
    assert dsr.expected_max_sharpe(VAR_SR, 1000) > dsr.expected_max_sharpe(VAR_SR, 10)


def test_min_trl_finite_when_sr_above_benchmark():
    trl = dsr.min_track_record_length(SR_OBS, 0.0, SKEW, KURT, alpha=0.05)
    assert np.isfinite(trl) and trl > 0
    assert dsr.min_track_record_length(0.0, 0.1, 0.0, 3.0) == float("inf")
