# power-analyst

Statistical power analysis for product and UX researchers — sample size calculations, achieved power, and full power curves for t-tests, proportions, ANOVA, and correlations.

```
$ power-analyst proportion --p1 0.05 --p2 0.08 --power 0.80

  n per group: 294  |  Total n: 588  |  Achieved power: 80.1%  |  Effect size: 0.118 (small)

  Parameter          Value
  Test               proportion test
  Effect size        0.1177 (small)
  Alpha (α)          0.05
  Achieved power     0.8012
  n per group        294
  Total N            588
```

## The problem

"How many participants do I need?" is one of the most common questions in research design — and one of the most commonly answered with a shrug. Underpowered studies miss real effects; overpowered studies waste budget. `power-analyst` solves for sample size or achieved power analytically across the four test types that cover most product research scenarios, and generates a power curve so you can see exactly where diminishing returns kick in.

## What it covers

**Two-sample independent t-test** — for comparing means between two groups (e.g. task completion time, NPS, session length). Uses Cohen's d as the effect size parameter.

**Two-proportion z-test** — for comparing conversion rates, click-through rates, or any binary outcome between two groups. Takes raw proportions (p1, p2) and computes Cohen's h internally.

**One-way ANOVA** — for comparing means across three or more groups (e.g. three design variants, four user segments). Uses Cohen's f.

**Pearson correlation** — for detecting a linear relationship between two continuous variables. Uses Fisher z transformation for accurate power estimation.

All four support both directions: solve for required sample size, or compute achieved power for a given n.

## Effect size benchmarks (Cohen's conventions)

| Test | Small | Medium | Large |
|------|-------|--------|-------|
| t-test (d) | 0.2 | 0.5 | 0.8 |
| ANOVA (f) | 0.1 | 0.25 | 0.4 |
| Correlation (r) | 0.1 | 0.3 | 0.5 |
| Proportion (h) | 0.2 | 0.5 | 0.8 |

## Install

```bash
pip install power-analyst
```

With optional LLM narrative layer:

```bash
pip install "power-analyst[openai]"
pip install "power-analyst[anthropic]"
pip install "power-analyst[gemini]"
pip install "power-analyst[all]"
```

## Usage

### CLI

```bash
# Sample size for a t-test (medium effect)
power-analyst ttest --d 0.5 --alpha 0.05 --power 0.80

# Sample size for an A/B test on conversion rate (3% → 5%)
power-analyst proportion --p1 0.03 --p2 0.05 --power 0.80

# Sample size for one-way ANOVA across 4 segments
power-analyst anova --f 0.25 --k 4 --power 0.80

# Sample size to detect a correlation of r=0.3
power-analyst correlation --r 0.3 --power 0.80

# Achieved power for an existing study (n=50 per group, d=0.4)
power-analyst ttest-achieved --n 50 --d 0.4 --alpha 0.05

# With LLM narrative and JSON output
power-analyst ttest --d 0.3 --power 0.90 \
  --llm-backend anthropic \
  --output power_result.json
```

### Python API

```python
from power_analyst import (
    ttest_sample_size,
    ttest_power,
    proportion_sample_size,
    anova_sample_size,
    correlation_sample_size,
)

# Solve for n
result = ttest_sample_size(d=0.5, alpha=0.05, power=0.80)
print(f"n per group: {result.n_per_group}")   # 64
print(f"Total N: {result.n_total}")            # 128
print(f"Achieved power: {result.power:.1%}")   # 80.2%
print(f"Effect size: {result.effect_size_label}")  # medium

# Power curve — see where diminishing returns begin
for point in result.power_curve:
    print(f"  n={int(point['n'])}: power={point['power']:.2%}")

# Achieved power
result = ttest_power(n=40, d=0.5, alpha=0.05)
print(f"Achieved power with n=40: {result.power:.1%}")  # ~60%

# Proportion test
result = proportion_sample_size(p1=0.10, p2=0.15, power=0.80)
print(result.n_total)

import json
print(json.dumps(result.to_dict(), indent=2))
```

## Output (JSON)

```json
{
  "test_type": "two_sample_ttest",
  "effect_size": 0.5,
  "effect_size_label": "medium",
  "alpha": 0.05,
  "power": 0.8022,
  "n_per_group": 64,
  "n_total": 128,
  "solved_for": "n",
  "power_curve": [
    {"n": 10, "power": 0.2934},
    {"n": 20, "power": 0.4637},
    {"n": 40, "power": 0.6008},
    {"n": 64, "power": 0.8022},
    {"n": 100, "power": 0.9405}
  ],
  "notes": ["Two-tailed=True", "Cohen's d=0.5"]
}
```

## Part of the CPR Ecosystem

`power-analyst` is Tool 4.9 in the [CPR Orbital](https://github.com/kgkazakos) series — 40 open-source tools for AI-native product research infrastructure, shipping every Friday from March to December 2026.

**Related tools in this series:**

| Tool | What it does |
|------|-------------|
| [`bayes-ab`](https://github.com/kgkazakos/bayes-ab) | Bayesian A/B testing with posteriors and ROPE decision rules |
| [`regression-diagnostics`](https://github.com/kgkazakos/regression-diagnostics) | OLS assumption checks, VIF, Cook's distance |
| [`scale-validator`](https://github.com/kgkazakos/scale-validator) | Cronbach's alpha, item analysis, factor retention |
| [`survey-cleaner`](https://github.com/kgkazakos/survey-cleaner) | Remove speeders, straight-liners and gibberish open-ends |

Part of the broader **Computational Product Research (CPR)** discipline. Core projects — CausalTrack, Synthetic User Council, Co-Reasoning Canvas — are developed separately and target CHI 2027/2028.

## License

MIT
