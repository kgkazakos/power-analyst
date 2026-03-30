"""Statistical power analysis engine — t-test, proportion, ANOVA, correlation."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from power_analyst.models import PowerResult

# Cohen's effect size benchmarks
COHEN_T = {"small": 0.2, "medium": 0.5, "large": 0.8}
COHEN_F = {"small": 0.1, "medium": 0.25, "large": 0.4}
COHEN_R = {"small": 0.1, "medium": 0.3, "large": 0.5}
COHEN_H = {"small": 0.2, "medium": 0.5, "large": 0.8}  # proportions


def _label_effect(d: float, benchmarks: dict) -> str:
    if d >= benchmarks["large"]:
        return "large"
    if d >= benchmarks["medium"]:
        return "medium"
    if d >= benchmarks["small"]:
        return "small"
    return "very small"


# ---------------------------------------------------------------------------
# Two-sample t-test power (Cohen's d)
# ---------------------------------------------------------------------------

def _ttest_power(n: int, d: float, alpha: float, tails: int = 2) -> float:
    """Compute power for a two-sample independent t-test."""
    df = 2 * n - 2
    ncp = d * math.sqrt(n / 2.0)
    t_crit = stats.t.ppf(1.0 - alpha / tails, df)
    nc_dist = stats.nct(df, ncp)
    if tails == 2:
        power = nc_dist.sf(t_crit) + nc_dist.cdf(-t_crit)
    else:
        power = nc_dist.sf(t_crit)
    return float(np.clip(power, 0.0, 1.0))


def ttest_sample_size(
    d: float,
    alpha: float = 0.05,
    power: float = 0.80,
    tails: int = 2,
    max_n: int = 100_000,
) -> PowerResult:
    """Solve for n per group given d, alpha, desired power."""
    for n in range(2, max_n + 1):
        achieved = _ttest_power(n, d, alpha, tails)
        if achieved >= power:
            curve = _build_curve(lambda nn: _ttest_power(nn, d, alpha, tails), n)
            return PowerResult(
                test_type="two_sample_ttest",
                effect_size=d,
                effect_size_label=_label_effect(abs(d), COHEN_T),
                alpha=alpha,
                power=achieved,
                n_per_group=n,
                n_total=2 * n,
                solved_for="n",
                power_curve=curve,
                notes=[f"Two-tailed={tails == 2}", f"Cohen's d={d}"],
            )
    return _insufficient_power_result("two_sample_ttest", d, alpha, power, max_n)


def ttest_power(n: int, d: float, alpha: float = 0.05, tails: int = 2) -> PowerResult:
    """Compute achieved power given n per group and effect size."""
    achieved = _ttest_power(n, d, alpha, tails)
    curve = _build_curve(lambda nn: _ttest_power(nn, d, alpha, tails), n)
    return PowerResult(
        test_type="two_sample_ttest",
        effect_size=d,
        effect_size_label=_label_effect(abs(d), COHEN_T),
        alpha=alpha,
        power=achieved,
        n_per_group=n,
        n_total=2 * n,
        solved_for="power",
        power_curve=curve,
    )


# ---------------------------------------------------------------------------
# Proportion test power (Cohen's h)
# ---------------------------------------------------------------------------

def _cohens_h(p1: float, p2: float) -> float:
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def _proportion_power(n: int, h: float, alpha: float, tails: int = 2) -> float:
    z_alpha = stats.norm.ppf(1.0 - alpha / tails)
    ncp = abs(h) * math.sqrt(n / 2.0)
    if tails == 2:
        power = stats.norm.sf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    else:
        power = stats.norm.sf(z_alpha - ncp)
    return float(np.clip(power, 0.0, 1.0))


def proportion_sample_size(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    tails: int = 2,
    max_n: int = 100_000,
) -> PowerResult:
    """Solve for n per group for a two-proportion z-test."""
    h = abs(_cohens_h(p1, p2))
    for n in range(2, max_n + 1):
        achieved = _proportion_power(n, h, alpha, tails)
        if achieved >= power:
            curve = _build_curve(lambda nn: _proportion_power(nn, h, alpha, tails), n)
            return PowerResult(
                test_type="proportion_test",
                effect_size=h,
                effect_size_label=_label_effect(h, COHEN_H),
                alpha=alpha,
                power=achieved,
                n_per_group=n,
                n_total=2 * n,
                solved_for="n",
                power_curve=curve,
                notes=[f"p1={p1}, p2={p2}", f"Cohen's h={h:.4f}"],
            )
    return _insufficient_power_result("proportion_test", h, alpha, power, max_n)


# ---------------------------------------------------------------------------
# One-way ANOVA power (Cohen's f)
# ---------------------------------------------------------------------------

def _anova_power(n_per_group: int, k: int, f: float, alpha: float) -> float:
    """Power for one-way ANOVA with k groups and n per group."""
    df1 = k - 1
    df2 = k * (n_per_group - 1)
    ncp = f ** 2 * k * n_per_group
    f_crit = stats.f.ppf(1.0 - alpha, df1, df2)
    power = stats.ncf.sf(f_crit, df1, df2, ncp)
    return float(np.clip(power, 0.0, 1.0))


def anova_sample_size(
    f: float,
    k: int = 3,
    alpha: float = 0.05,
    power: float = 0.80,
    max_n: int = 100_000,
) -> PowerResult:
    """Solve for n per group for one-way ANOVA."""
    for n in range(2, max_n + 1):
        achieved = _anova_power(n, k, f, alpha)
        if achieved >= power:
            curve = _build_curve(lambda nn: _anova_power(nn, k, f, alpha), n)
            return PowerResult(
                test_type="one_way_anova",
                effect_size=f,
                effect_size_label=_label_effect(f, COHEN_F),
                alpha=alpha,
                power=achieved,
                n_per_group=n,
                n_total=k * n,
                solved_for="n",
                power_curve=curve,
                notes=[f"k={k} groups", f"Cohen's f={f}"],
            )
    return _insufficient_power_result("one_way_anova", f, alpha, power, max_n)


# ---------------------------------------------------------------------------
# Correlation power (Pearson r)
# ---------------------------------------------------------------------------

def _correlation_power(n: int, r: float, alpha: float, tails: int = 2) -> float:
    """Power for testing H0: rho=0 using Fisher z transformation."""
    if abs(r) >= 1.0:
        return 1.0
    z_r = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3) if n > 3 else 1.0
    z_alpha = stats.norm.ppf(1.0 - alpha / tails)
    ncp = abs(z_r) / se
    if tails == 2:
        power = stats.norm.sf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    else:
        power = stats.norm.sf(z_alpha - ncp)
    return float(np.clip(power, 0.0, 1.0))


def correlation_sample_size(
    r: float,
    alpha: float = 0.05,
    power: float = 0.80,
    tails: int = 2,
    max_n: int = 100_000,
) -> PowerResult:
    """Solve for n to detect a Pearson correlation of size r."""
    for n in range(4, max_n + 1):
        achieved = _correlation_power(n, r, alpha, tails)
        if achieved >= power:
            curve = _build_curve(lambda nn: _correlation_power(nn, r, alpha, tails), n)
            return PowerResult(
                test_type="correlation",
                effect_size=abs(r),
                effect_size_label=_label_effect(abs(r), COHEN_R),
                alpha=alpha,
                power=achieved,
                n_per_group=n,
                n_total=n,
                solved_for="n",
                power_curve=curve,
                notes=[f"r={r}", f"Two-tailed={tails == 2}"],
            )
    return _insufficient_power_result("correlation", abs(r), alpha, power, max_n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_curve(power_fn, n_target: int, n_points: int = 30) -> list[dict[str, float]]:
    """Build a power curve from n=2 to 2*n_target."""
    max_n = max(n_target * 2, 20)
    step = max(1, max_n // n_points)
    ns = list(range(2, max_n + 1, step))
    if n_target not in ns:
        ns.append(n_target)
    ns = sorted(set(ns))
    return [{"n": float(n), "power": power_fn(n)} for n in ns]


def _insufficient_power_result(
    test_type: str, effect_size: float, alpha: float, power: float, max_n: int
) -> PowerResult:
    return PowerResult(
        test_type=test_type,
        effect_size=effect_size,
        effect_size_label="very small",
        alpha=alpha,
        power=_ttest_power(max_n, effect_size, alpha) if test_type == "two_sample_ttest" else 0.0,
        n_per_group=max_n,
        n_total=max_n * 2,
        solved_for="n",
        notes=[f"Could not reach power={power} within max_n={max_n}. Effect size may be too small."],
    )
