"""Victor Core — Orch-OR inspired quantum decision engine.

Integrates:
  - Quantum superposition modelling for decision candidates
  - Orch-OR decoherence collapse to select actions
  - Flower-of-Life neural topology for candidate weighting
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantumCandidate:
    """A decision candidate held in quantum superposition."""

    label: str
    amplitude: float        # Complex amplitude magnitude (0–1)
    phase: float            # Phase angle in radians
    coherence_time: float = 1.0  # Seconds before natural decoherence

    @property
    def probability(self) -> float:
        """Born-rule probability: |amplitude|²."""
        return self.amplitude ** 2


@dataclass
class OrchORResult:
    """Outcome of an Orchestrated Objective Reduction collapse event."""

    selected: str
    probability: float
    coherence_at_collapse: float
    collapse_time: float
    all_candidates: list[dict[str, Any]]


class FlowerOfLifeTopology:
    """
    Flower-of-Life inspired hexagonal neural weighting topology.

    Generates a geometric weighting mask for N candidates based on
    hexagonal packing geometry.
    """

    def weights(self, n: int) -> list[float]:
        """Return n L2-normalised weights derived from hexagonal geometry.

        L2 normalisation ensures sum(w²) == 1, satisfying the Born rule so
        that probability amplitudes collapse correctly.
        """
        if n <= 0:
            return []
        raw = []
        for i in range(n):
            # Angle evenly distributed around a circle; centre node (i=0) maximal
            angle = 2 * math.pi * i / max(n, 1)
            raw.append(1.0 / (1.0 + 0.4 * abs(math.sin(angle))))
        l2 = math.sqrt(sum(w * w for w in raw)) or 1.0
        return [round(w / l2, 6) for w in raw]


class VictorCore:
    """
    Victor Orch-OR quantum decision engine.

    Maintains a superposition of decision candidates; applies Flower-of-Life
    weighting; and collapses via Orch-OR decoherence to select an action.
    """

    def __init__(self, decoherence_threshold: float = 0.3) -> None:
        self.decoherence_threshold = decoherence_threshold
        self._topology = FlowerOfLifeTopology()
        self._candidates: list[QuantumCandidate] = []
        self._collapse_history: list[OrchORResult] = []

    # ------------------------------------------------------------------ superposition

    def superpose(
        self,
        candidates: list[str],
        amplitudes: list[float] | None = None,
    ) -> None:
        """
        Place candidate decisions into quantum superposition.

        If no amplitudes are given, equal superposition is assumed and
        Flower-of-Life geometry is applied to differentiate weights.
        """
        n = len(candidates)
        if n == 0:
            return

        if amplitudes is None:
            amplitudes = self._topology.weights(n)
        else:
            if len(amplitudes) != n:
                raise ValueError(
                    "Amplitudes list must match candidates list length."
                )
            # L2-normalise so that sum(amplitude²) == 1 (Born rule)
            l2 = math.sqrt(sum(a * a for a in amplitudes)) or 1.0
            amplitudes = [a / l2 for a in amplitudes]

        self._candidates = [
            QuantumCandidate(
                label=label,
                amplitude=amp,
                phase=2 * math.pi * i / max(n, 1),
                coherence_time=1.0,
            )
            for i, (label, amp) in enumerate(zip(candidates, amplitudes))
        ]

    def decohere(self, elapsed: float = 0.0) -> None:
        """Apply time-based amplitude decay to simulate decoherence."""
        for c in self._candidates:
            decay = math.exp(-elapsed / max(c.coherence_time, 1e-9))
            c.amplitude = max(0.0, c.amplitude * decay)

    # ------------------------------------------------------------------ collapse

    def collapse(self) -> OrchORResult | None:
        """
        Perform an Orch-OR collapse: select the candidate with the highest
        probability.  Returns None if the superposition is empty.
        """
        if not self._candidates:
            return None

        # Renormalise so probabilities sum to 1
        total_prob = sum(c.probability for c in self._candidates) or 1e-12
        coherence = min(1.0, total_prob)
        best = max(self._candidates, key=lambda c: c.probability)

        result = OrchORResult(
            selected=best.label,
            probability=round(best.probability / total_prob, 4),
            coherence_at_collapse=round(coherence, 4),
            collapse_time=time.time(),
            all_candidates=[
                {
                    "label": c.label,
                    "amplitude": round(c.amplitude, 4),
                    "probability": round(c.probability / total_prob, 4),
                }
                for c in self._candidates
            ],
        )
        self._collapse_history.append(result)
        self._candidates.clear()
        return result

    # ------------------------------------------------------------------ convenience

    def decide(
        self,
        candidates: list[str],
        amplitudes: list[float] | None = None,
    ) -> OrchORResult | None:
        """Convenience: superpose then immediately collapse."""
        self.superpose(candidates, amplitudes)
        return self.collapse()

    def get_state(self) -> dict[str, Any]:
        return {
            "superposition_size": len(self._candidates),
            "decoherence_threshold": self.decoherence_threshold,
            "collapse_count": len(self._collapse_history),
            "last_collapse": (
                self._collapse_history[-1].selected
                if self._collapse_history
                else None
            ),
        }
