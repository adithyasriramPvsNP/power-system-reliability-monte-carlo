"""Monte Carlo estimation of generation adequacy indices.

Two methods:

* ``non_sequential`` (state sampling): draw independent snapshots of which units
  are up, and score each snapshot against the whole load curve. Fast; gives the
  expected indices (LOLE, EENS).
* ``sequential`` (chronological): simulate each unit's failure/repair history hour
  by hour for many years. Slower, but also yields frequency and duration of
  shortages and the year-to-year spread of the indices.
"""

from dataclasses import dataclass

import numpy as np

from .rts import GenerationSystem


class LoadCurve:
    """Sorted load curve with prefix sums, to score many capacity states at once."""

    def __init__(self, load: np.ndarray):
        self.sorted = np.sort(load)
        self.prefix = np.concatenate([[0.0], np.cumsum(self.sorted)])
        self.n = len(load)

    def shortfall(self, available: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each available capacity: (#periods with load > A, energy short)."""
        idx = np.searchsorted(self.sorted, available, side="right")
        periods = self.n - idx
        energy = (self.prefix[-1] - self.prefix[idx]) - available * periods
        return periods.astype(float), energy


@dataclass
class NonSequentialResult:
    lole: float
    lole_se: float
    eens: float
    eens_se: float
    n_samples: int
    trace_n: np.ndarray  # sample counts at checkpoints
    trace_lole: np.ndarray  # running LOLE estimate at checkpoints
    trace_se: np.ndarray  # running standard error at checkpoints


def non_sequential(system: GenerationSystem, load: np.ndarray, n_samples: int,
                   seed: int = 0, batch: int = 50_000) -> NonSequentialResult:
    """State-sampling Monte Carlo.

    Each sample draws every unit up/down with its forced outage rate and counts the
    load periods the resulting capacity cannot cover. That count is an unbiased
    estimate of LOLE; averaging over samples gives the estimate and its error bar.
    """
    rng = np.random.default_rng(seed)
    curve = LoadCurve(load)
    cap = system.capacity.astype(float)
    s_h = s_h2 = s_e = s_e2 = 0.0
    done = 0
    trace = []
    while done < n_samples:
        b = min(batch, n_samples - done)
        up = rng.random((b, system.n_units)) >= system.forced_outage_rate
        h, e = curve.shortfall(up @ cap)
        s_h += h.sum(); s_h2 += (h * h).sum()
        s_e += e.sum(); s_e2 += (e * e).sum()
        done += b
        mean = s_h / done
        se = np.sqrt(max(s_h2 / done - mean**2, 0.0) / done)
        trace.append((done, mean, se))
    lole, eens = s_h / done, s_e / done
    eens_se = np.sqrt(max(s_e2 / done - eens**2, 0.0) / done)
    tn, tl, ts = (np.array(col) for col in zip(*trace))
    return NonSequentialResult(lole, ts[-1], eens, eens_se, done, tn, tl, ts)


def _unit_down_mask(rng, mttf, mttr, q, n_hours) -> np.ndarray:
    """Boolean array: is the unit down at the middle of each hour?

    Up and down durations are exponential. The block starts in the steady state
    (down with probability q); by memorylessness the first residual duration is
    exponential with the same mean, so blocks can be simulated independently.
    """
    start_down = rng.random() < q
    means = np.array([mttr, mttf] if start_down else [mttf, mttr])
    n_pairs = int(n_hours / (mttf + mttr) * 1.5) + 10
    bounds = np.cumsum(rng.exponential(1.0, 2 * n_pairs) * np.tile(means, n_pairs))
    while bounds[-1] < n_hours:  # rare: extend the history
        extra = rng.exponential(1.0, 2 * n_pairs) * np.tile(means, n_pairs)
        bounds = np.concatenate([bounds, bounds[-1] + np.cumsum(extra)])
    interval = np.searchsorted(bounds, np.arange(n_hours) + 0.5, side="right")
    in_first_state = interval % 2 == 0
    return in_first_state if start_down else ~in_first_state


@dataclass
class SequentialResult:
    lole_per_year: np.ndarray  # hours of shortage in each simulated year
    eens_per_year: np.ndarray  # MWh not served in each simulated year
    events_per_year: np.ndarray  # number of separate shortage events in each year

    @property
    def n_years(self) -> int:
        return len(self.lole_per_year)

    def summary(self) -> dict:
        def mean_se(x):
            return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))

        lole, lole_se = mean_se(self.lole_per_year)
        eens, eens_se = mean_se(self.eens_per_year)
        lolf, lolf_se = mean_se(self.events_per_year)
        return {
            "years": self.n_years,
            "lole": lole, "lole_se": lole_se,
            "eens": eens, "eens_se": eens_se,
            "lolf": lolf, "lolf_se": lolf_se,
            "mean_event_duration_h": lole / lolf if lolf else float("nan"),
            "share_years_with_shortage": float((self.lole_per_year > 0).mean()),
        }


def sequential(system: GenerationSystem, hourly_load: np.ndarray, n_years: int,
               seed: int = 0, block_years: int = 100) -> SequentialResult:
    """Chronological Monte Carlo over ``n_years`` simulated years."""
    rng = np.random.default_rng(seed)
    hours_per_year = len(hourly_load)
    lole, eens, events = [], [], []
    for start in range(0, n_years, block_years):
        years = min(block_years, n_years - start)
        n_hours = years * hours_per_year
        available = np.full(n_hours, float(system.installed_mw))
        for c, f, r, q in zip(system.capacity, system.mttf, system.mttr,
                              system.forced_outage_rate):
            available -= c * _unit_down_mask(rng, f, r, q, n_hours)
        deficit = (np.tile(hourly_load, years) - available).reshape(years, -1)
        short = deficit > 0
        lole.append(short.sum(axis=1))
        eens.append(np.where(short, deficit, 0.0).sum(axis=1))
        starts = short[:, 1:] & ~short[:, :-1]
        events.append(starts.sum(axis=1) + short[:, 0])
    return SequentialResult(np.concatenate(lole).astype(float),
                            np.concatenate(eens), np.concatenate(events).astype(float))
