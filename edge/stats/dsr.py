"""Probabilistic & Deflated Sharpe Ratio and Minimum Track Record Length.

Formulae (Bailey & López de Prado, 'The Deflated Sharpe Ratio', JPM 2014).
ALL inputs are in NON-ANNUALIZED (per-observation) units — mixing annualized SR
with non-annualized moments is the classic implementation bug, so this module
never annualizes internally.

    PSR(SR*) = Z[ (SR_hat - SR*) * sqrt(n - 1)
                  / sqrt(1 - g3*SR_hat + ((g4 - 1)/4)*SR_hat^2) ]

      Z   = standard-normal CDF
      n   = number of return observations
      g3  = skewness of returns
      g4  = kurtosis of returns (NON-excess; normal = 3)
      Negative skew / high kurtosis inflate the denominator => lower PSR.

    Expected maximum Sharpe over N independent trials (deflation benchmark):
    SR*_0 = sqrt(Var[SR_trials]) * [ (1-gamma)*Z^-1(1 - 1/N)
                                     + gamma*Z^-1(1 - 1/(N*e)) ]
      gamma = 0.5772156649  (Euler-Mascheroni),  e = Euler's number.

    DSR = PSR(SR*_0).   Decision rule: DSR > 0.95.

    MinTRL = (1 - g3*SR_hat + ((g4 - 1)/4)*SR_hat^2) * (Z_{1-alpha} / (SR_hat - SR*))^2
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329
_E = np.e


def sharpe_ratio(returns: np.ndarray, ddof: int = 1) -> float:
    """Non-annualized Sharpe = mean / std of the per-observation returns."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=ddof)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd)


def _moments(returns: np.ndarray) -> tuple[float, float]:
    """Return (skewness, NON-excess kurtosis)."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    m = r.mean()
    s = r.std(ddof=0)
    if s == 0:
        return 0.0, 3.0
    g3 = float(np.mean(((r - m) / s) ** 3))
    g4 = float(np.mean(((r - m) / s) ** 4))  # non-excess
    return g3, g4


def probabilistic_sharpe_ratio(sr_hat: float, sr_benchmark: float, n: int,
                               skew: float, kurt: float) -> float:
    """PSR(SR*) — probability the true SR exceeds `sr_benchmark`. Non-annualized."""
    denom = np.sqrt(1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2)
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(var_sr_trials: float, n_trials: int) -> float:
    """SR*_0: expected maximum Sharpe of `n_trials` independent worthless trials."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    term = ((1 - g) * norm.ppf(1 - 1.0 / n_trials)
            + g * norm.ppf(1 - 1.0 / (n_trials * _E)))
    return float(np.sqrt(var_sr_trials) * term)


def deflated_sharpe_ratio(sr_hat: float, n: int, skew: float, kurt: float,
                          var_sr_trials: float, n_trials: int) -> dict:
    """Compute DSR = PSR(SR*_0) and return all intermediate quantities.

    `var_sr_trials` is Var of the per-observation Sharpe ratios across the
    logged trials (the honest trial count N drives the deflation).
    """
    sr0 = expected_max_sharpe(var_sr_trials, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_hat, sr0, n, skew, kurt)
    return {"dsr": dsr, "sr_star0": sr0, "sr_hat": sr_hat, "n": n,
            "skew": skew, "kurt": kurt, "n_trials": n_trials,
            "var_sr_trials": var_sr_trials}


def deflated_sharpe_from_returns(returns: np.ndarray, var_sr_trials: float,
                                 n_trials: int) -> dict:
    """Convenience: compute DSR directly from a per-observation return series."""
    r = np.asarray(returns, dtype=float)
    sr = sharpe_ratio(r)
    g3, g4 = _moments(r)
    return deflated_sharpe_ratio(sr, r.size, g3, g4, var_sr_trials, n_trials)


def min_track_record_length(sr_hat: float, sr_benchmark: float, skew: float,
                            kurt: float, alpha: float = 0.05) -> float:
    """Minimum number of observations to reject SR <= sr_benchmark at level alpha."""
    if sr_hat <= sr_benchmark:
        return float("inf")
    z = norm.ppf(1 - alpha)
    denom_term = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    return float(denom_term * (z / (sr_hat - sr_benchmark)) ** 2 + 1.0)
