"""IEEE Reliability Test System (RTS-79): generating units and chronological load.

Source: IEEE RTS Task Force of the Application of Probability Methods
Subcommittee, "IEEE Reliability Test System", IEEE Transactions on Power
Apparatus and Systems, vol. PAS-98, no. 6, pp. 2047-2054, 1979.
"""

from dataclasses import dataclass

import numpy as np

# (unit size MW, number of units, forced outage rate, MTTF h, MTTR h)
UNIT_TYPES = [
    (12, 5, 0.02, 2940, 60),
    (20, 4, 0.10, 450, 50),
    (50, 6, 0.01, 1980, 20),
    (76, 4, 0.02, 1960, 40),
    (100, 3, 0.04, 1200, 50),
    (155, 4, 0.04, 960, 40),
    (197, 3, 0.05, 950, 50),
    (350, 1, 0.08, 1150, 100),
    (400, 2, 0.12, 1100, 150),
]

PEAK_LOAD_MW = 2850.0

# Weekly peak load, % of annual peak (Table 1). Week 1 starts on a Monday.
WEEKLY_PEAK_PCT = np.array([
    86.2, 90.0, 87.8, 83.4, 88.0, 84.1, 83.2, 80.6, 74.0, 73.7, 71.5, 72.7, 70.4,
    75.0, 72.1, 80.0, 75.4, 83.7, 87.0, 88.0, 85.6, 81.1, 90.0, 88.7, 89.6, 86.1,
    75.5, 81.6, 80.1, 88.0, 72.2, 77.6, 80.0, 72.9, 72.6, 70.5, 78.0, 69.5, 72.4,
    72.4, 74.3, 74.4, 80.0, 88.1, 88.5, 90.9, 94.0, 89.0, 94.2, 97.0, 100.0, 95.2,
])

# Daily peak load, % of weekly peak (Table 2), Monday .. Sunday.
DAILY_PEAK_PCT = np.array([93, 100, 98, 96, 94, 77, 75])

# Hourly load, % of daily peak (Table 3), hour 0 = midnight-1 am.
HOURLY_PCT = {
    "winter": (
        [67, 63, 60, 59, 59, 60, 74, 86, 95, 96, 96, 95,
         95, 95, 93, 94, 99, 100, 100, 96, 91, 83, 73, 63],
        [78, 72, 68, 66, 64, 65, 66, 70, 80, 88, 90, 91,
         90, 88, 87, 87, 91, 100, 99, 97, 94, 92, 87, 81],
    ),
    "summer": (
        [64, 60, 58, 56, 56, 58, 64, 76, 87, 95, 99, 100,
         99, 100, 100, 97, 96, 96, 93, 92, 92, 93, 87, 72],
        [74, 70, 66, 65, 64, 62, 62, 66, 81, 86, 91, 93,
         93, 92, 91, 91, 92, 94, 95, 95, 100, 93, 88, 80],
    ),
    "spring_fall": (
        [63, 62, 60, 58, 59, 65, 72, 85, 95, 99, 100, 99,
         93, 92, 90, 88, 90, 92, 96, 98, 96, 90, 80, 70],
        [75, 73, 69, 66, 65, 65, 68, 74, 83, 89, 92, 94,
         91, 90, 90, 86, 85, 88, 92, 100, 97, 95, 90, 85],
    ),
}

HOURS_PER_YEAR = 52 * 7 * 24  # 8736, the RTS study year


@dataclass(frozen=True)
class GenerationSystem:
    """Independent two-state (up/down) generating units."""

    capacity: np.ndarray  # MW, integer
    forced_outage_rate: np.ndarray  # long-run probability of being down
    mttf: np.ndarray  # mean time to failure, hours
    mttr: np.ndarray  # mean time to repair, hours

    @property
    def installed_mw(self) -> int:
        return int(self.capacity.sum())

    @property
    def n_units(self) -> int:
        return len(self.capacity)


def rts_generation() -> GenerationSystem:
    """The 32 RTS-79 generating units (3405 MW)."""
    rows = [(c, q, f, r) for c, n, q, f, r in UNIT_TYPES for _ in range(n)]
    cap, q, mttf, mttr = (np.array(col) for col in zip(*rows))
    return GenerationSystem(cap.astype(np.int64), q.astype(float),
                            mttf.astype(float), mttr.astype(float))


def _season(week: int) -> str:
    """Season for a 1-based week number, as defined by the RTS."""
    if week <= 8 or week >= 44:
        return "winter"
    if 18 <= week <= 30:
        return "summer"
    return "spring_fall"


def rts_daily_peak_load(peak_mw: float = PEAK_LOAD_MW) -> np.ndarray:
    """364 daily peak loads in MW."""
    weekly = WEEKLY_PEAK_PCT / 100.0
    daily = DAILY_PEAK_PCT / 100.0
    return peak_mw * np.outer(weekly, daily).ravel()


def rts_hourly_load(peak_mw: float = PEAK_LOAD_MW) -> np.ndarray:
    """8736 chronological hourly loads in MW."""
    daily_peaks = rts_daily_peak_load(peak_mw)
    hours = np.empty(HOURS_PER_YEAR)
    for day, day_peak in enumerate(daily_peaks):
        week = day // 7 + 1
        weekday, weekend = HOURLY_PCT[_season(week)]
        profile = weekend if day % 7 >= 5 else weekday
        hours[day * 24:(day + 1) * 24] = day_peak * np.array(profile) / 100.0
    return hours
