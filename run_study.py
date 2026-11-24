"""Run the full RTS-79 adequacy study: results table, figures and results.json.

Usage:  python run_study.py           (full run, ~2 minutes)
        python run_study.py --quick   (fewer samples, ~15 seconds)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reliability import (PEAK_LOAD_MW, adequacy_indices, non_sequential,
                         rts_daily_peak_load, rts_generation, rts_hourly_load,
                         sequential)

TARGET_DAYS_PER_YEAR = 0.1  # "one day in ten years" planning criterion
FIGURES = Path("figures")


def daily_lole(system, peak_mw):
    return adequacy_indices(system, rts_daily_peak_load(peak_mw))["lole"]


def peak_load_carrying_capability(system, target, lo=1500.0, hi=PEAK_LOAD_MW):
    """Largest annual peak load (MW) whose daily LOLE meets the target (bisection)."""
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if daily_lole(system, mid) <= target else (lo, mid)
    return lo


def plot_convergence(ns, exact, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(ns.trace_n, ns.trace_lole - 1.96 * ns.trace_se,
                    ns.trace_lole + 1.96 * ns.trace_se, alpha=0.25, label="95% confidence band")
    ax.plot(ns.trace_n, ns.trace_lole, lw=1.5, label="Monte Carlo estimate")
    ax.axhline(exact, color="k", ls="--", lw=1, label=f"Exact (COPT) = {exact:.3f} h/yr")
    ax.set_xscale("log")
    ax.set_xlabel("Number of sampled system states")
    ax.set_ylabel("LOLE (hours / year)")
    ax.set_title("Non-sequential Monte Carlo converges to the exact LOLE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annual_distribution(seq, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = seq.lole_per_year
    top = np.percentile(x, 99)
    ax.hist(x, bins=np.arange(0, top + 2), edgecolor="white")
    ax.set_xlim(0, top + 1)
    ax.axvline(x.mean(), color="k", ls="--", lw=1, label=f"Mean = {x.mean():.2f} h/yr")
    ax.set_xlabel("Hours of capacity shortage in a simulated year")
    ax.set_ylabel("Number of years")
    ax.set_title(f"Year-to-year variability ({seq.n_years:,} simulated years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_load_sweep(system, plcc, mc_points, path):
    peaks = np.linspace(2200, 3000, 81)
    exact = [daily_lole(system, p) for p in peaks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(peaks, exact, lw=1.5, label="Exact (COPT)")
    p, m, se = (np.array(c) for c in zip(*mc_points))
    ax.errorbar(p, m, yerr=1.96 * se, fmt="o", ms=4, capsize=3, label="Monte Carlo ±95%")
    ax.axhline(TARGET_DAYS_PER_YEAR, color="tab:red", ls=":", label="Criterion: 0.1 day/yr")
    ax.axvline(plcc, color="tab:red", ls="--", lw=1)
    ax.axvline(PEAK_LOAD_MW, color="gray", ls="--", lw=1)
    ax.annotate(f"Max peak load\n{plcc:.0f} MW", (plcc, 0.02), ha="right",
                xytext=(-6, 0), textcoords="offset points", color="tab:red")
    ax.annotate(f"RTS peak\n{PEAK_LOAD_MW:.0f} MW", (PEAK_LOAD_MW, 0.02), ha="right",
                xytext=(-6, 0), textcoords="offset points", color="gray")
    ax.set_xlabel("Annual peak load (MW)")
    ax.set_ylabel("Daily-peak LOLE (days / year)")
    ax.set_title("How much peak load can the RTS fleet serve reliably?")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="fewer samples")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    n_states = 500_000 if args.quick else 5_000_000
    n_years = 1_000 if args.quick else 10_000

    FIGURES.mkdir(exist_ok=True)
    system = rts_generation()
    hourly = rts_hourly_load()

    exact = adequacy_indices(system, hourly)
    exact["lole_daily"] = daily_lole(system, PEAK_LOAD_MW)
    ns = non_sequential(system, hourly, n_states, seed=args.seed)
    seq_run = sequential(system, hourly, n_years, seed=args.seed)
    seq = seq_run.summary()

    plcc = peak_load_carrying_capability(system, TARGET_DAYS_PER_YEAR)
    mc_points = []
    for peak in [2300, 2400, 2500, plcc, 2700, 2850, 3000]:
        r = non_sequential(system, rts_daily_peak_load(peak), n_states, seed=args.seed)
        mc_points.append((peak, r.lole, r.lole_se))
    plcc_mc = next(m for p, m, _ in mc_points if p == plcc)
    plcc_mc_se = next(s for p, _, s in mc_points if p == plcc)

    print(f"IEEE RTS-79: {system.n_units} units, {system.installed_mw} MW installed, "
          f"{PEAK_LOAD_MW:.0f} MW peak\n")
    print(f"{'Method':<28}{'LOLE (h/yr)':>18}{'EENS (MWh/yr)':>20}")
    print(f"{'Exact (COPT)':<28}{exact['lole']:>18.3f}{exact['eens']:>20.1f}")
    print(f"{'Non-sequential MC':<28}{f'{ns.lole:.3f} ± {1.96 * ns.lole_se:.3f}':>18}"
          f"{f'{ns.eens:.1f} ± {1.96 * ns.eens_se:.1f}':>20}")
    print(f"{'Sequential MC':<28}{f'{seq['lole']:.3f} ± {1.96 * seq['lole_se']:.3f}':>18}"
          f"{f'{seq['eens']:.1f} ± {1.96 * seq['eens_se']:.1f}':>20}")
    print(f"\nSequential MC only: {seq['lolf']:.2f} shortage events/yr, "
          f"mean duration {seq['mean_event_duration_h']:.2f} h, "
          f"{100 * seq['share_years_with_shortage']:.0f}% of years have a shortage")
    print(f"\nDaily LOLE at 2850 MW peak: {exact['lole_daily']:.3f} days/yr "
          f"(criterion {TARGET_DAYS_PER_YEAR})")
    print(f"Peak load carrying capability at {TARGET_DAYS_PER_YEAR} day/yr: {plcc:.1f} MW "
          f"({100 * (1 - plcc / PEAK_LOAD_MW):.1f}% below the RTS peak); "
          f"MC check there: {plcc_mc:.4f} ± {1.96 * plcc_mc_se:.4f} days/yr")

    plot_convergence(ns, exact["lole"], FIGURES / "convergence.png")
    plot_annual_distribution(seq_run, FIGURES / "annual_lole_distribution.png")
    plot_load_sweep(system, plcc, mc_points, FIGURES / "peak_load_sweep.png")

    results = {
        "exact": exact,
        "non_sequential": {"samples": ns.n_samples, "lole": ns.lole, "lole_se": ns.lole_se,
                           "eens": ns.eens, "eens_se": ns.eens_se},
        "sequential": seq,
        "peak_load_carrying_capability_mw": plcc,
        "plcc_monte_carlo_check": {"lole_days": plcc_mc, "se": plcc_mc_se},
    }
    Path("results.json").write_text(json.dumps(results, indent=2))
    print("\nWrote results.json and figures/*.png")


if __name__ == "__main__":
    main()
