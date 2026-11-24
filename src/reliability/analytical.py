"""Exact adequacy indices from a capacity outage probability table (COPT).

These serve as the reference answer the Monte Carlo estimates are checked against.
"""

import numpy as np

from .rts import GenerationSystem


def available_capacity_pmf(system: GenerationSystem) -> np.ndarray:
    """pmf[a] = P(available capacity == a MW), for a = 0 .. installed MW.

    Built by convolving the units one at a time (recursive COPT construction).
    """
    outage = np.zeros(system.installed_mw + 1)
    outage[0] = 1.0
    for c, q in zip(system.capacity, system.forced_outage_rate):
        shifted = np.zeros_like(outage)
        shifted[c:] = outage[:-c]
        outage = (1 - q) * outage + q * shifted
    return outage[::-1]  # outage of k MW  <=>  available = installed - k


def adequacy_indices(system: GenerationSystem, load: np.ndarray) -> dict:
    """Expected shortfall indices over a load curve.

    For an hourly load curve this returns LOLE in hours/year and EENS in MWh/year.
    For a daily-peak load curve LOLE is in days/year (EENS is then not meaningful).
    """
    pmf = available_capacity_pmf(system)
    a = np.arange(len(pmf))
    cdf = np.cumsum(pmf)  # P(A <= a)
    partial_mean = np.cumsum(a * pmf)  # E[A ; A <= a]

    # Shortfall means A < L, i.e. A <= ceil(L) - 1.
    i = np.clip(np.ceil(load).astype(np.int64) - 1, -1, len(pmf) - 1)
    lolp = np.where(i >= 0, cdf[np.maximum(i, 0)], 0.0)
    unserved = np.where(i >= 0, load * lolp - partial_mean[np.maximum(i, 0)], 0.0)
    return {"lole": float(lolp.sum()), "eens": float(unserved.sum())}
