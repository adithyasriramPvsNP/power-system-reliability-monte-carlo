import numpy as np
import pytest

from reliability import (GenerationSystem, adequacy_indices, available_capacity_pmf,
                         non_sequential, rts_daily_peak_load, rts_generation,
                         rts_hourly_load, sequential)


@pytest.fixture(scope="module")
def rts():
    return rts_generation()


@pytest.fixture(scope="module")
def hourly():
    return rts_hourly_load()


def two_unit_system():
    return GenerationSystem(np.array([10, 20]), np.array([0.1, 0.2]),
                            np.array([90.0, 80.0]), np.array([10.0, 20.0]))


def test_rts_data(rts, hourly):
    assert rts.n_units == 32 and rts.installed_mw == 3405
    assert len(hourly) == 8736 and hourly.max() == pytest.approx(2850)
    # Forced outage rate must equal MTTR / (MTTF + MTTR) for every unit.
    assert np.allclose(rts.forced_outage_rate, rts.mttr / (rts.mttf + rts.mttr))


def test_pmf_by_hand():
    pmf = available_capacity_pmf(two_unit_system())
    expected = {0: 0.1 * 0.2, 10: 0.9 * 0.2, 20: 0.1 * 0.8, 30: 0.9 * 0.8}
    assert pmf.sum() == pytest.approx(1.0)
    for a, p in expected.items():
        assert pmf[a] == pytest.approx(p)


def test_indices_by_hand():
    # Load 15 MW: short when available is 0 (15 MW short) or 10 (5 MW short).
    r = adequacy_indices(two_unit_system(), np.array([15.0]))
    assert r["lole"] == pytest.approx(0.02 + 0.18)
    assert r["eens"] == pytest.approx(0.02 * 15 + 0.18 * 5)


def test_rts_published_values(rts, hourly):
    # Published RTS-79 indices: LOLE 9.3942 h/yr, EENS 1176 MWh/yr, 1.3689 d/yr.
    r = adequacy_indices(rts, hourly)
    assert r["lole"] == pytest.approx(9.3942, abs=1e-3)
    assert r["eens"] == pytest.approx(1176, abs=1)
    assert adequacy_indices(rts, rts_daily_peak_load())["lole"] == pytest.approx(1.3689, abs=1e-3)


def test_non_sequential_matches_exact(rts, hourly):
    exact = adequacy_indices(rts, hourly)
    r = non_sequential(rts, hourly, 400_000, seed=7)
    assert abs(r.lole - exact["lole"]) < 4 * r.lole_se
    assert abs(r.eens - exact["eens"]) < 4 * r.eens_se


def test_sequential_matches_exact(rts, hourly):
    exact = adequacy_indices(rts, hourly)
    s = sequential(rts, hourly, 300, seed=7).summary()
    assert abs(s["lole"] - exact["lole"]) < 4 * s["lole_se"]
    assert s["lolf"] > 0 and s["mean_event_duration_h"] >= 1
