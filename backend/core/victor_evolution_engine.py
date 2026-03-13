"""
victor_evolution_engine.py
==========================
EvolutionEngine and DigitalAgent for population-based agent evolution.

Implements crossover and mutation mechanics so that VictorASIOmniBrainGodcore
can breed successive generations of increasingly fit agents without any
external dependencies beyond the Python standard library.
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import ClassVar, Optional


# ---------------------------------------------------------------------------
# DigitalAgent
# ---------------------------------------------------------------------------

@dataclass
class DigitalAgent:
    """
    A single agent in an evolutionary population.

    Attributes
    ----------
    genome_id   : stable identifier (int or str) supplied at creation.
    generation  : the generation in which this agent was born.
    traits      : mutable dict of named float values (0.0–1.0) representing
                  the agent's heritable characteristics.
    fitness_score: assigned externally after `perform_cognitive_task()`.
    id          : unique UUID string, auto-generated at construction.
    """

    genome_id: int | str
    generation: int = 0
    traits: dict[str, float] = field(default_factory=dict)
    fitness_score: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    DEFAULT_TRAITS: ClassVar[dict[str, float]] = {
        "curiosity":    0.5,
        "caution":      0.5,
        "creativity":   0.5,
        "precision":    0.5,
        "autonomy":     0.5,
        "empathy":      0.5,
        "adaptability": 0.5,
        "resilience":   0.5,
    }

    def __post_init__(self) -> None:
        for trait, default_val in self.DEFAULT_TRAITS.items():
            self.traits.setdefault(trait, default_val)

    def perform_cognitive_task(self) -> float:
        """
        Simulate a cognitive benchmark and return a fitness score in [0, 1].

        The score is computed from the agent's trait profile with a small
        stochastic component so that identically-configured agents can still
        produce slightly different results across runs.
        """
        base_score = sum(self.traits.values()) / len(self.traits)
        noise = random.gauss(0.0, 0.05)
        return float(max(0.0, min(1.0, base_score + noise)))


# ---------------------------------------------------------------------------
# EvolutionEngine
# ---------------------------------------------------------------------------

class EvolutionEngine:
    """
    Genetic operators for DigitalAgent populations.

    Provides uniform crossover and Gaussian mutation so that
    VictorASIOmniBrainGodcore can advance agent generations each cycle.
    """

    MUTATION_RATE: float = 0.1    # Probability that a given trait is mutated
    MUTATION_SCALE: float = 0.05  # Standard deviation of Gaussian noise

    def crossover(
        self,
        parent_a: DigitalAgent,
        parent_b: DigitalAgent,
        generation: Optional[int] = None,
    ) -> DigitalAgent:
        """
        Produce a child agent by combining traits from two parents.

        Uses uniform crossover: for each trait the child randomly inherits
        from either parent_a or parent_b with equal probability.

        Parameters
        ----------
        parent_a, parent_b : source agents.
        generation         : optional explicit generation number; if omitted
                             the child's generation is max(parents) + 1.

        Returns
        -------
        A new DigitalAgent whose traits are a blend of both parents.
        """
        child_generation = (
            generation
            if generation is not None
            else max(parent_a.generation, parent_b.generation) + 1
        )
        child_traits: dict[str, float] = {}
        all_trait_keys = set(parent_a.traits) | set(parent_b.traits)
        for key in all_trait_keys:
            a_val = parent_a.traits.get(key, 0.5)
            b_val = parent_b.traits.get(key, 0.5)
            child_traits[key] = a_val if random.random() < 0.5 else b_val

        return DigitalAgent(
            genome_id=f"{parent_a.genome_id}x{parent_b.genome_id}",
            generation=child_generation,
            traits=child_traits,
        )

    def mutate(self, agent: DigitalAgent) -> None:
        """
        Apply in-place Gaussian mutations to an agent's traits.

        Each trait is mutated independently with probability `MUTATION_RATE`.
        Values are clamped to [0.0, 1.0] after mutation.
        """
        for trait in agent.traits:
            if random.random() < self.MUTATION_RATE:
                delta = random.gauss(0.0, self.MUTATION_SCALE)
                agent.traits[trait] = max(0.0, min(1.0, agent.traits[trait] + delta))
