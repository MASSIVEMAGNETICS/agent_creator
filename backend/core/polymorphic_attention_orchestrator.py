"""Polymorphic Attention Orchestrator for Victor AGI.

Manages dynamic phase switching between cognitive attention modes:
  - solid       : precise, analytical — low entropy, high precision
  - fluid       : balanced creative/analytical work
  - gas         : high-entropy, divergent brainstorming
  - singularity : maximum focus — collapses all attention to one concern
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttentionPhase(str, Enum):
    SOLID = "solid"
    FLUID = "fluid"
    GAS = "gas"
    SINGULARITY = "singularity"


# Phase characteristics
_PHASE_PARAMS: dict[AttentionPhase, dict[str, float]] = {
    AttentionPhase.SOLID: {
        "entropy_bias": 0.1,
        "focus_weight": 0.9,
        "creativity_weight": 0.2,
        "precision_weight": 0.95,
    },
    AttentionPhase.FLUID: {
        "entropy_bias": 0.4,
        "focus_weight": 0.6,
        "creativity_weight": 0.6,
        "precision_weight": 0.6,
    },
    AttentionPhase.GAS: {
        "entropy_bias": 0.85,
        "focus_weight": 0.2,
        "creativity_weight": 0.95,
        "precision_weight": 0.3,
    },
    AttentionPhase.SINGULARITY: {
        "entropy_bias": 0.0,
        "focus_weight": 1.0,
        "creativity_weight": 0.05,
        "precision_weight": 1.0,
    },
}


@dataclass
class AttentionSnapshot:
    phase: AttentionPhase
    timestamp: float
    params: dict[str, float]
    trigger: str = ""


class PolymorphicAttentionOrchestrator:
    """
    Orchestrates cognitive-phase switching for Victor AGI.

    Tracks phase transition history and exposes helpers for downstream
    components to query the current attentional context.
    """

    def __init__(self, initial_phase: AttentionPhase = AttentionPhase.FLUID) -> None:
        self._phase = initial_phase
        self._history: list[AttentionSnapshot] = [
            AttentionSnapshot(
                phase=initial_phase,
                timestamp=time.time(),
                params=dict(_PHASE_PARAMS[initial_phase]),
                trigger="init",
            )
        ]

    # ------------------------------------------------------------------ phase

    @property
    def phase(self) -> AttentionPhase:
        return self._phase

    def transition(self, target: AttentionPhase, trigger: str = "") -> AttentionSnapshot:
        """Switch to a new attention phase and record the transition."""
        self._phase = target
        snapshot = AttentionSnapshot(
            phase=target,
            timestamp=time.time(),
            params=dict(_PHASE_PARAMS[target]),
            trigger=trigger,
        )
        self._history.append(snapshot)
        if len(self._history) > 128:
            self._history.pop(0)
        return snapshot

    def auto_transition(self, coherence: float, entropy: float) -> AttentionPhase:
        """
        Automatically select the best phase based on analysis metrics.

        Rules (in priority order):
        1. coherence > 0.85 and entropy < 0.2  → SINGULARITY
        2. entropy   > 0.7                      → GAS
        3. coherence > 0.5                      → SOLID
        4. otherwise                            → FLUID
        """
        if coherence > 0.85 and entropy < 0.2:
            target = AttentionPhase.SINGULARITY
        elif entropy > 0.7:
            target = AttentionPhase.GAS
        elif coherence > 0.5:
            target = AttentionPhase.SOLID
        else:
            target = AttentionPhase.FLUID

        if target != self._phase:
            self.transition(target, trigger="auto")
        return self._phase

    # ------------------------------------------------------------------ query

    def get_params(self) -> dict[str, float]:
        """Return the weight parameters for the current phase."""
        return dict(_PHASE_PARAMS[self._phase])

    def get_history(self) -> list[dict[str, Any]]:
        return [
            {
                "phase": s.phase.value,
                "timestamp": s.timestamp,
                "trigger": s.trigger,
                "params": s.params,
            }
            for s in self._history
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "current_phase": self._phase.value,
            "params": self.get_params(),
            "transition_count": len(self._history) - 1,
        }
