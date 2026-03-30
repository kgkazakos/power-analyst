"""Click CLI for power-analyst."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

if TYPE_CHECKING:
    from power_analyst.models import PowerResult


@click.group()
@click.version_option(package_name="power-analyst")
def cli() -> None:
    """Statistical power analysis — t-test, proportion, ANOVA, correlation."""


# ---------------------------------------------------------------------------
# t-test
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--d", "effect_size", required=True, type=float, help="Cohen's d effect size.")
@click.option("--alpha", default=0.05, show_default=True, help="Significance level.")
@click.option("--power", default=0.80, show_default=True, help="Desired power (0–1).")
@click.option("--tails", default=2, show_default=True, type=click.Choice(["1", "2"]))
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None)
@click.option("--quiet", "-q", is_flag=True)
def ttest(
    effect_size: float, alpha: float, power: float, tails: str,
    llm_backend: Optional[str], output: Optional[str], quiet: bool,
) -> None:
    """Sample size for a two-sample independent t-test."""
    from power_analyst.engine import ttest_sample_size

    result = ttest_sample_size(d=effect_size, alpha=alpha, power=power, tails=int(tails))
    _maybe_narrate(result, llm_backend)
    if not quiet:
        _print_result(result)
    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))


@cli.command()
@click.option("--n", "n_per_group", required=True, type=int, help="Sample size per group.")
@click.option("--d", "effect_size", required=True, type=float, help="Cohen's d.")
@click.option("--alpha", default=0.05, show_default=True)
@click.option("--tails", default=2, show_default=True, type=click.Choice(["1", "2"]))
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None)
@click.option("--quiet", "-q", is_flag=True)
def ttest_achieved(
    n_per_group: int, effect_size: float, alpha: float, tails: str,
    llm_backend: Optional[str], output: Optional[str], quiet: bool,
) -> None:
    """Compute achieved power for a two-sample t-test given n and effect size."""
    from power_analyst.engine import ttest_power

    result = ttest_power(n=n_per_group, d=effect_size, alpha=alpha, tails=int(tails))
    _maybe_narrate(result, llm_backend)
    if not quiet:
        _print_result(result)
    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# proportion
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--p1", required=True, type=float, help="Baseline conversion rate (0–1).")
@click.option("--p2", required=True, type=float, help="Expected treatment rate (0–1).")
@click.option("--alpha", default=0.05, show_default=True)
@click.option("--power", default=0.80, show_default=True)
@click.option("--tails", default=2, show_default=True, type=click.Choice(["1", "2"]))
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None)
@click.option("--quiet", "-q", is_flag=True)
def proportion(
    p1: float, p2: float, alpha: float, power: float, tails: str,
    llm_backend: Optional[str], output: Optional[str], quiet: bool,
) -> None:
    """Sample size for a two-proportion z-test (e.g. A/B conversion rates)."""
    from power_analyst.engine import proportion_sample_size

    result = proportion_sample_size(p1=p1, p2=p2, alpha=alpha, power=power, tails=int(tails))
    _maybe_narrate(result, llm_backend)
    if not quiet:
        _print_result(result)
    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--f", "effect_size", required=True, type=float, help="Cohen's f effect size.")
@click.option("--k", default=3, show_default=True, type=int, help="Number of groups.")
@click.option("--alpha", default=0.05, show_default=True)
@click.option("--power", default=0.80, show_default=True)
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None)
@click.option("--quiet", "-q", is_flag=True)
def anova(
    effect_size: float, k: int, alpha: float, power: float,
    llm_backend: Optional[str], output: Optional[str], quiet: bool,
) -> None:
    """Sample size for a one-way ANOVA."""
    from power_analyst.engine import anova_sample_size

    result = anova_sample_size(f=effect_size, k=k, alpha=alpha, power=power)
    _maybe_narrate(result, llm_backend)
    if not quiet:
        _print_result(result)
    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--r", "effect_size", required=True, type=float, help="Expected Pearson r.")
@click.option("--alpha", default=0.05, show_default=True)
@click.option("--power", default=0.80, show_default=True)
@click.option("--tails", default=2, show_default=True, type=click.Choice(["1", "2"]))
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None)
@click.option("--quiet", "-q", is_flag=True)
def correlation(
    effect_size: float, alpha: float, power: float, tails: str,
    llm_backend: Optional[str], output: Optional[str], quiet: bool,
) -> None:
    """Sample size to detect a Pearson correlation."""
    from power_analyst.engine import correlation_sample_size

    result = correlation_sample_size(r=effect_size, alpha=alpha, power=power, tails=int(tails))
    _maybe_narrate(result, llm_backend)
    if not quiet:
        _print_result(result)
    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _maybe_narrate(result: "PowerResult", llm_backend: Optional[str]) -> None:
    if llm_backend:
        from power_analyst.narrator import PowerNarrator
        narrator = PowerNarrator(backend=llm_backend)
        result.narrative = narrator.narrate(result.to_dict())


def _print_result(result: "PowerResult") -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        console.print()
        console.print(Panel(
            f"[bold]n per group:[/] {result.n_per_group}  "
            f"| Total n: {result.n_total}  "
            f"| Achieved power: [bold]{result.power:.1%}[/]  "
            f"| Effect size: {result.effect_size:.3f} ([italic]{result.effect_size_label}[/])",
            title=f"[bold]power-analyst[/] · {result.test_type.replace('_', ' ')}",
            expand=False,
        ))

        t = Table(box=box.SIMPLE_HEAVY, header_style="bold", show_header=True)
        t.add_column("Parameter")
        t.add_column("Value")
        t.add_row("Test", result.test_type.replace("_", " "))
        t.add_row("Effect size", f"{result.effect_size:.4f} ({result.effect_size_label})")
        t.add_row("Alpha (α)", str(result.alpha))
        t.add_row("Target power", "—")
        t.add_row("Achieved power", f"{result.power:.4f}")
        t.add_row("n per group", str(result.n_per_group))
        t.add_row("Total N", str(result.n_total))
        console.print(t)

        for note in result.notes:
            console.print(f"  [dim]{note}[/]")

        if result.narrative:
            console.print(Panel(result.narrative, title="Narrative", expand=False))
        console.print()

    except ImportError:
        click.echo(f"\npower-analyst · {result.test_type}")
        click.echo(f"  n per group: {result.n_per_group}  |  Total N: {result.n_total}")
        click.echo(f"  Achieved power: {result.power:.1%}  |  Effect size: {result.effect_size:.3f}")
