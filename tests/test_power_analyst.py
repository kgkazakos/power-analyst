"""Tests for power-analyst."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from power_analyst.engine import (
    _anova_power,
    _cohens_h,
    _correlation_power,
    _label_effect,
    _ttest_power,
    anova_sample_size,
    correlation_sample_size,
    proportion_sample_size,
    ttest_power,
    ttest_sample_size,
    COHEN_T,
    COHEN_F,
    COHEN_R,
)
from power_analyst.narrator import PowerNarrator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_label_effect_t():
    assert _label_effect(0.8, COHEN_T) == "large"
    assert _label_effect(0.5, COHEN_T) == "medium"
    assert _label_effect(0.2, COHEN_T) == "small"
    assert _label_effect(0.05, COHEN_T) == "very small"


def test_cohens_h_nonzero():
    h = _cohens_h(0.05, 0.10)
    assert h < 0  # p2 > p1 → negative h
    assert abs(h) > 0


def test_cohens_h_identical_proportions():
    assert _cohens_h(0.1, 0.1) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# t-test
# ---------------------------------------------------------------------------

def test_ttest_power_increases_with_n():
    p1 = _ttest_power(50, 0.5, 0.05)
    p2 = _ttest_power(200, 0.5, 0.05)
    assert p2 > p1


def test_ttest_power_increases_with_d():
    p_small = _ttest_power(100, 0.2, 0.05)
    p_large = _ttest_power(100, 0.8, 0.05)
    assert p_large > p_small


def test_ttest_power_bounded():
    p = _ttest_power(100, 0.5, 0.05)
    assert 0.0 <= p <= 1.0


def test_ttest_sample_size_medium_effect():
    result = ttest_sample_size(d=0.5, alpha=0.05, power=0.80)
    assert result.n_per_group > 0
    assert result.power >= 0.80
    assert result.effect_size_label == "medium"
    # Known approximate: ~64 per group for d=0.5, alpha=0.05, power=0.80
    assert 50 <= result.n_per_group <= 80


def test_ttest_sample_size_large_effect_smaller_n():
    small_d = ttest_sample_size(d=0.2)
    large_d = ttest_sample_size(d=0.8)
    assert large_d.n_per_group < small_d.n_per_group


def test_ttest_sample_size_total_is_double():
    result = ttest_sample_size(d=0.5)
    assert result.n_total == 2 * result.n_per_group


def test_ttest_achieved_power():
    result = ttest_power(n=64, d=0.5, alpha=0.05)
    assert result.solved_for == "power"
    assert 0.75 <= result.power <= 0.90


def test_ttest_serializable():
    import json
    result = ttest_sample_size(d=0.5)
    serialized = json.dumps(result.to_dict())
    parsed = json.loads(serialized)
    assert parsed["test_type"] == "two_sample_ttest"


# ---------------------------------------------------------------------------
# proportion test
# ---------------------------------------------------------------------------

def test_proportion_sample_size_basic():
    result = proportion_sample_size(p1=0.10, p2=0.15)
    assert result.n_per_group > 0
    assert result.power >= 0.80
    assert result.test_type == "proportion_test"


def test_proportion_larger_difference_needs_fewer():
    small_diff = proportion_sample_size(p1=0.10, p2=0.11)
    large_diff = proportion_sample_size(p1=0.10, p2=0.20)
    assert large_diff.n_per_group < small_diff.n_per_group


def test_proportion_notes_contain_rates():
    result = proportion_sample_size(p1=0.05, p2=0.10)
    assert any("p1" in note for note in result.notes)


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------

def test_anova_power_increases_with_n():
    p1 = _anova_power(20, 3, 0.25, 0.05)
    p2 = _anova_power(100, 3, 0.25, 0.05)
    assert p2 > p1


def test_anova_sample_size_medium_effect():
    result = anova_sample_size(f=0.25, k=3, alpha=0.05, power=0.80)
    assert result.n_per_group > 0
    assert result.power >= 0.80
    assert result.n_total == 3 * result.n_per_group


def test_anova_more_groups_needs_more_total():
    r3 = anova_sample_size(f=0.25, k=3)
    r5 = anova_sample_size(f=0.25, k=5)
    assert r5.n_total > r3.n_total


def test_anova_effect_label():
    result = anova_sample_size(f=0.4)
    assert result.effect_size_label == "large"


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

def test_correlation_power_increases_with_n():
    p1 = _correlation_power(30, 0.3, 0.05)
    p2 = _correlation_power(100, 0.3, 0.05)
    assert p2 > p1


def test_correlation_sample_size_medium():
    result = correlation_sample_size(r=0.3, alpha=0.05, power=0.80)
    assert result.n_per_group > 0
    assert result.power >= 0.80
    # Known approximate: ~84 for r=0.3, alpha=0.05, power=0.80
    assert 70 <= result.n_per_group <= 100


def test_correlation_perfect_r_high_power():
    p = _correlation_power(10, 0.99, 0.05)
    assert p > 0.99


def test_correlation_serializable():
    import json
    result = correlation_sample_size(r=0.3)
    parsed = json.loads(json.dumps(result.to_dict()))
    assert "power_curve" in parsed


# ---------------------------------------------------------------------------
# power curve
# ---------------------------------------------------------------------------

def test_power_curve_present_and_ordered():
    result = ttest_sample_size(d=0.5)
    curve = result.power_curve
    assert len(curve) > 5
    ns = [p["n"] for p in curve]
    assert ns == sorted(ns)


def test_power_curve_powers_bounded():
    result = ttest_sample_size(d=0.5)
    for point in result.power_curve:
        assert 0.0 <= point["power"] <= 1.0


# ---------------------------------------------------------------------------
# narrator
# ---------------------------------------------------------------------------

def test_narrator_rejects_unknown_backend():
    with pytest.raises(ValueError):
        PowerNarrator(backend="cohere")


def test_narrator_default_models():
    assert "gpt" in PowerNarrator(backend="openai").model
    assert "claude" in PowerNarrator(backend="anthropic").model
    assert "gemini" in PowerNarrator(backend="gemini").model


def test_narrator_mocked():
    narrator = PowerNarrator(backend="anthropic")
    result = ttest_sample_size(d=0.5)
    with patch.object(narrator, "_call_anthropic", return_value="You need 64 participants per group."):
        narrative = narrator.narrate(result.to_dict())
    assert "64" in narrative


def test_narrator_returns_empty_on_error():
    narrator = PowerNarrator(backend="openai")
    result = ttest_sample_size(d=0.5)
    with patch.object(narrator, "_call_openai", side_effect=ConnectionError("down")):
        assert narrator.narrate(result.to_dict()) == ""
