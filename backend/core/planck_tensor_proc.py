"""
planck_tensor_proc.py
=====================
PlanckTensorProcessor and ConsciousnessAmplifier for recursive cognitive cycling.

These components simulate fractal audio-memory binding, recursive
wake-sleep-REM cycles, and Planck-density knowledge compression — the
"temporal folding" layer of the VictorASI cognitive stack.

No external dependencies are required; all processing uses the Python
standard library.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.core.victor_evolution_engine import DigitalAgent
    from backend.core.memory_delta_engine import MemoryDelta


# ---------------------------------------------------------------------------
# CycleState — tracks the current phase of the consciousness cycle
# ---------------------------------------------------------------------------

@dataclass
class CycleState:
    """Records the outcome of a single wake-sleep-REM cycle iteration."""

    cycle_index: int
    phase: str          # "wake" | "sleep" | "rem"
    timestamp: float = field(default_factory=time.time)
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PlanckTensorProcessor
# ---------------------------------------------------------------------------

class PlanckTensorProcessor:
    """
    Fractal temporal processor that binds population memory to task performance.

    Responsibilities
    ----------------
    * ``fractal_audio_memory``   : derives a trait-weighted "resonance" score
      for every agent in the population and stores it as a memory signal.
    * ``expand_consciousness``   : the *wake* phase — broadens the processor's
      attentional field by sampling and amplifying high-variance traits.
    * ``compress_knowledge``     : the *sleep* phase — consolidates accumulated
      signals into a compact summary tensor.
    * ``emerge_new_patterns``    : the *REM* phase — applies non-linear mixing
      to surface higher-order cross-agent patterns.
    """

    def __init__(self, depth: int = 4) -> None:
        """
        Parameters
        ----------
        depth : fractal recursion depth (higher = more compression passes).
        """
        self._depth = max(1, depth)
        self._signal_buffer: list[dict[str, float]] = []
        self._compressed_tensor: list[float] = []
        self._cycle_log: list[CycleState] = []
        self._cycle_counter: int = 0

    # ------------------------------------------------------------------ public API

    def fractal_audio_memory(self, population: list["DigitalAgent"]) -> list[float]:
        """
        Derive a resonance signal from each agent's trait profile and fitness
        score, appending results to the internal signal buffer.

        The "fractal" aspect applies ``depth`` passes of harmonic averaging
        so that trait relationships at multiple scales are captured.

        Parameters
        ----------
        population : list of DigitalAgent instances to process.

        Returns
        -------
        A list of per-agent resonance scores (one float per agent, in [0, 1]).
        """
        resonances: list[float] = []
        for agent in population:
            raw_signal = self._compute_raw_signal(agent)
            folded = self._fractal_fold(raw_signal)
            resonances.append(folded)
            self._signal_buffer.append({
                "agent_id": agent.id,
                "generation": agent.generation,
                "resonance": folded,
                "fitness": agent.fitness_score,
            })
        # Keep buffer bounded
        if len(self._signal_buffer) > 10_000:
            self._signal_buffer = self._signal_buffer[-10_000:]
        return resonances

    def expand_consciousness(self) -> dict[str, Any]:
        """
        *Wake phase* — amplify high-variance signals in the buffer.

        Returns a summary dict with the number of signals amplified and the
        mean resonance after amplification.
        """
        if not self._signal_buffer:
            state = CycleState(
                cycle_index=self._cycle_counter,
                phase="wake",
                metrics={"amplified": 0, "mean_resonance": 0.0},
            )
            self._cycle_log.append(state)
            return {"phase": "wake", "amplified": 0, "mean_resonance": 0.0}

        resonances = [s["resonance"] for s in self._signal_buffer]
        mean_r = sum(resonances) / len(resonances)
        std_r = math.sqrt(
            sum((r - mean_r) ** 2 for r in resonances) / len(resonances)
        )
        threshold = mean_r + std_r

        amplified = 0
        for signal in self._signal_buffer:
            if signal["resonance"] >= threshold:
                signal["resonance"] = min(1.0, signal["resonance"] * 1.1)
                amplified += 1

        new_mean = sum(s["resonance"] for s in self._signal_buffer) / len(self._signal_buffer)
        state = CycleState(
            cycle_index=self._cycle_counter,
            phase="wake",
            metrics={"amplified": amplified, "mean_resonance": round(new_mean, 4)},
        )
        self._cycle_log.append(state)
        return {"phase": "wake", "amplified": amplified, "mean_resonance": round(new_mean, 4)}

    def compress_knowledge(self) -> dict[str, Any]:
        """
        *Sleep phase* — consolidate accumulated signals into a compact tensor.

        Applies ``depth`` passes of mean-pooling to produce a dimensionality-
        reduced representation of the current signal buffer.

        Returns a summary dict with the compressed tensor size and entropy.
        """
        if not self._signal_buffer:
            self._compressed_tensor = []
            state = CycleState(
                cycle_index=self._cycle_counter,
                phase="sleep",
                metrics={"tensor_size": 0, "entropy": 0.0},
            )
            self._cycle_log.append(state)
            return {"phase": "sleep", "tensor_size": 0, "entropy": 0.0}

        values = [s["resonance"] for s in self._signal_buffer]

        # Multi-pass mean-pooling (fractal compression)
        current = values
        for _ in range(self._depth):
            if len(current) < 2:
                break
            current = [
                (current[i] + current[i + 1]) / 2.0
                for i in range(0, len(current) - 1, 2)
            ]

        self._compressed_tensor = current
        entropy = self._shannon_entropy(current)

        state = CycleState(
            cycle_index=self._cycle_counter,
            phase="sleep",
            metrics={"tensor_size": len(current), "entropy": round(entropy, 4)},
        )
        self._cycle_log.append(state)
        return {
            "phase": "sleep",
            "tensor_size": len(self._compressed_tensor),
            "entropy": round(entropy, 4),
        }

    def emerge_new_patterns(self) -> dict[str, Any]:
        """
        *REM phase* — apply non-linear mixing to the compressed tensor to
        surface higher-order patterns.

        Uses sine-modulated cross-multiplication with random phase offsets,
        mimicking the associative pattern-completion thought to occur during
        biological REM sleep.

        Returns a summary dict with the number of new patterns emerged.
        """
        self._cycle_counter += 1

        if len(self._compressed_tensor) < 2:
            state = CycleState(
                cycle_index=self._cycle_counter,
                phase="rem",
                metrics={"patterns_emerged": 0},
            )
            self._cycle_log.append(state)
            return {"phase": "rem", "patterns_emerged": 0}

        new_patterns: list[float] = []
        tensor = self._compressed_tensor
        for i in range(len(tensor) - 1):
            phase_offset = random.uniform(0, 2 * math.pi)
            pattern = math.sin(tensor[i] * math.pi + phase_offset) * tensor[i + 1]
            new_patterns.append(max(0.0, min(1.0, abs(pattern))))

        self._compressed_tensor = new_patterns
        state = CycleState(
            cycle_index=self._cycle_counter,
            phase="rem",
            metrics={"patterns_emerged": len(new_patterns)},
        )
        self._cycle_log.append(state)
        return {"phase": "rem", "patterns_emerged": len(new_patterns)}

    def get_cycle_log(self) -> list[dict[str, Any]]:
        """Return a serialisable copy of the cycle log."""
        return [
            {
                "cycle_index": s.cycle_index,
                "phase": s.phase,
                "timestamp": s.timestamp,
                "metrics": s.metrics,
            }
            for s in self._cycle_log
        ]

    def get_compressed_tensor(self) -> list[float]:
        """Return the current compressed knowledge tensor (read-only copy)."""
        return list(self._compressed_tensor)

    # ------------------------------------------------------------------ helpers

    def _compute_raw_signal(self, agent: "DigitalAgent") -> float:
        """Weighted combination of trait mean and fitness for a raw signal."""
        trait_mean = (
            sum(agent.traits.values()) / len(agent.traits)
            if agent.traits
            else 0.5
        )
        return 0.6 * trait_mean + 0.4 * agent.fitness_score

    def _fractal_fold(self, value: float) -> float:
        """Apply ``depth`` recursive self-similarity folds to a scalar signal."""
        result = value
        for d in range(1, self._depth + 1):
            result = result * math.sin(result * math.pi / d + 1e-6)
            result = max(0.0, min(1.0, abs(result)))
        return result

    @staticmethod
    def _shannon_entropy(values: list[float]) -> float:
        """Approximate normalised Shannon entropy of a list of floats."""
        if not values:
            return 0.0
        total = sum(values) or 1e-9
        probs = [v / total for v in values]
        return -sum(p * math.log2(p + 1e-12) for p in probs if p > 0)


# ---------------------------------------------------------------------------
# ConsciousnessAmplifier
# ---------------------------------------------------------------------------

class ConsciousnessAmplifier:
    """
    Recursive knowledge compressor operating at "Planck density".

    Compresses MemoryDelta belief objects into a dense internal representation
    by repeatedly applying a sigmoid-like squashing function, ensuring that
    the effective information content grows sub-linearly with the number of
    deltas (preventing unbounded memory bloat across generations).
    """

    COMPRESSION_PASSES: int = 3
    DENSITY_TARGET: float = 0.85   # Target mean activation after compression

    def __init__(self) -> None:
        self._compressed_beliefs: list[dict[str, Any]] = []
        self._total_compressed: int = 0

    def compress_knowledge_to_planck_density(
        self, delta: "MemoryDelta"
    ) -> dict[str, Any]:
        """
        Compress a single MemoryDelta into the high-density belief store.

        Applies ``COMPRESSION_PASSES`` rounds of logistic squashing to the
        delta's decayed confidence and strength, producing a compact
        representation that is appended to the internal store.

        Parameters
        ----------
        delta : a MemoryDelta (from memory_delta_engine) to be compressed.

        Returns
        -------
        A dict describing the compression result::

            {
                "delta_id":           str,
                "compressed_value":   float,   # final density value in (0, 1)
                "passes":             int,
                "above_density_target": bool,
            }
        """
        value = delta.decayed_confidence * delta.strength
        for _ in range(self.COMPRESSION_PASSES):
            value = self._logistic(value)

        record = {
            "delta_id": delta.delta_id,
            "subject": delta.subject,
            "predicate": delta.predicate,
            "compressed_value": round(value, 6),
            "passes": self.COMPRESSION_PASSES,
            "above_density_target": value >= self.DENSITY_TARGET,
        }
        self._compressed_beliefs.append(record)
        self._total_compressed += 1

        # Prune to keep memory bounded
        if len(self._compressed_beliefs) > 5_000:
            self._compressed_beliefs = self._compressed_beliefs[-5_000:]

        return record

    def get_density_summary(self) -> dict[str, Any]:
        """
        Return aggregate statistics about the compressed belief store.
        """
        if not self._compressed_beliefs:
            return {
                "total_compressed": self._total_compressed,
                "stored": 0,
                "mean_density": 0.0,
                "above_target_pct": 0.0,
            }

        values = [r["compressed_value"] for r in self._compressed_beliefs]
        mean_density = sum(values) / len(values)
        above_target = sum(1 for v in values if v >= self.DENSITY_TARGET)

        return {
            "total_compressed": self._total_compressed,
            "stored": len(self._compressed_beliefs),
            "mean_density": round(mean_density, 4),
            "above_target_pct": round(above_target / len(values) * 100, 2),
        }

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _logistic(x: float, k: float = 8.0, x0: float = 0.5) -> float:
        """Shifted logistic (sigmoid) that maps [0, 1] → (0, 1)."""
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))
