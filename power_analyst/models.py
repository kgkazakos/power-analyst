"""Core data models for power-analyst."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PowerResult:
    test_type: str
    effect_size: float
    alpha: float
    power: float
    n_per_group: int
    n_total: int

    # What was solved for
    solved_for: str  # "n" | "power" | "effect_size"

    # Curve data for visualisation
    power_curve: list[dict[str, float]] = field(default_factory=list)  # [{n, power}, ...]

    # Effect size interpretation
    effect_size_label: str = ""   # "small" | "medium" | "large"

    # Contextual notes
    notes: list[str] = field(default_factory=list)

    narrative: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_type": self.test_type,
            "effect_size": round(float(self.effect_size), 4),
            "effect_size_label": self.effect_size_label,
            "alpha": round(float(self.alpha), 4),
            "power": round(float(self.power), 4),
            "n_per_group": int(self.n_per_group),
            "n_total": int(self.n_total),
            "solved_for": self.solved_for,
            "power_curve": [
                {"n": int(p["n"]), "power": round(float(p["power"]), 4)}
                for p in self.power_curve
            ],
            "notes": self.notes,
            "narrative": self.narrative,
        }
