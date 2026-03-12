"""Victor Analysis — foundational simulation analysis layer for the Victor AGI system.

Provides state-vector representation, simulation frame stepping, and multi-metric
scoring (coherence, entropy, drift) that downstream modules can consume without
heavy external dependencies.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulationFrame:
    """A single timestep snapshot captured during analysis."""

    frame_id: int
    timestamp: float
    state_vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def magnitude(self) -> float:
        """L2 norm of the state vector."""
        return math.sqrt(sum(v * v for v in self.state_vector))

    def dot(self, other: SimulationFrame) -> float:
        """Dot product with another frame's state vector."""
        if len(self.state_vector) != len(other.state_vector):
            raise ValueError("State vectors must have the same dimensionality.")
        return sum(a * b for a, b in zip(self.state_vector, other.state_vector))


class VictorAnalysis:
    """
    Foundational Victor-based simulation analysis system.

    Maintains a rolling window of simulation frames and exposes analytic
    primitives (coherence, entropy, drift) that higher-level modules rely on.
    """

    MAX_HISTORY = 256

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions
        self._history: list[SimulationFrame] = []
        self._frame_counter = 0

    # ------------------------------------------------------------------ frames

    def record_frame(
        self,
        state_vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> SimulationFrame:
        """Append a new frame to the rolling history window."""
        if len(state_vector) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions}-dimensional state vector, "
                f"got {len(state_vector)}."
            )
        self._frame_counter += 1
        frame = SimulationFrame(
            frame_id=self._frame_counter,
            timestamp=time.time(),
            state_vector=list(state_vector),
            metadata=metadata or {},
        )
        self._history.append(frame)
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)
        return frame

    # ------------------------------------------------------------------ metrics

    def coherence(self) -> float:
        """
        Mean pairwise cosine similarity of the last four frames.
        Returns 1.0 when there are fewer than two frames (trivially coherent).
        """
        window = self._history[-4:]
        if len(window) < 2:
            return 1.0
        scores: list[float] = []
        for i in range(len(window) - 1):
            a, b = window[i], window[i + 1]
            denom = (a.magnitude * b.magnitude) or 1e-12
            scores.append(a.dot(b) / denom)
        return sum(scores) / len(scores)

    def entropy(self) -> float:
        """
        Approximate Shannon entropy of the most recent frame's state vector,
        treating absolute values as an unnormalised probability distribution.
        """
        if not self._history:
            return 0.0
        vec = self._history[-1].state_vector
        total = sum(abs(v) for v in vec) or 1e-12
        probs = [abs(v) / total for v in vec if v != 0.0]
        return -sum(p * math.log(p + 1e-12) for p in probs)

    def drift(self) -> float:
        """
        Mean absolute change between consecutive frames over the last 8 steps.
        """
        window = self._history[-8:]
        if len(window) < 2:
            return 0.0
        deltas: list[float] = []
        for i in range(len(window) - 1):
            a, b = window[i].state_vector, window[i + 1].state_vector
            deltas.append(sum(abs(x - y) for x, y in zip(a, b)) / self.dimensions)
        return sum(deltas) / len(deltas)

    def get_summary(self) -> dict[str, Any]:
        """Return a snapshot of all key analytic metrics."""
        return {
            "dimensions": self.dimensions,
            "frame_count": self._frame_counter,
            "history_window": len(self._history),
            "coherence": round(self.coherence(), 4),
            "entropy": round(self.entropy(), 4),
            "drift": round(self.drift(), 4),
        }
