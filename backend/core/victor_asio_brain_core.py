"""
victor_asio_brain_core.py
=========================
VictorASIOmniBrainGodcore — the unified cognitive substrate.

Fuses the EvolutionEngine, MemoryDelta Engine, PlanckTensorProcessor,
and ConsciousnessAmplifier into a single self-adaptive AGI loop capable of:

* Spawning and evolving populations of DigitalAgents across generations.
* Logging agent performance as evidence-backed MemoryDelta beliefs.
* Running fractal audio-memory binding and recursive knowledge compression.
* Executing continuous wake-sleep-REM cycles for cognitive self-repair.
"""

import logging
import random
from typing import Optional

from backend.core.memory_delta_engine import (
    MemoryDeltaStore,
    create_memory_delta_from_evidence,
)
from backend.core.planck_tensor_proc import ConsciousnessAmplifier, PlanckTensorProcessor
from backend.core.victor_evolution_engine import DigitalAgent, EvolutionEngine


class VictorASIOmniBrainGodcore:
    """
    Victor's complete cognitive substrate.

    Lifecycle
    ---------
    1. ``initialize_population()``    — seed the first generation.
    2. ``evaluate_population()``      — score each agent + log memory deltas.
    3. ``evolve_population()``        — breed the next generation via crossover
       and mutation.
    4. ``update_planck_memory_cycle()`` — fractal memory binding + knowledge
       compression.

    ``cognitive_loop(iterations)`` orchestrates steps 1-4 for a fixed number
    of cycles.  ``wake_sleep_rem_cycle(cycles)`` runs the three-phase
    consciousness loop either for a bounded number of iterations or, when
    ``cycles`` is ``None``, indefinitely (use with care in production).

    Parameters
    ----------
    population_size : number of DigitalAgents per generation (default 100).
    iterations      : default iteration count for ``cognitive_loop``.
    planck_depth    : fractal recursion depth passed to PlanckTensorProcessor.
    """

    def __init__(
        self,
        population_size: int = 100,
        iterations: int = 10,
        planck_depth: int = 4,
    ) -> None:
        self.population: list[DigitalAgent] = []
        self.default_population_size = population_size
        self.iterations = iterations
        self.logger = logging.getLogger("VictorASIOmniBrainGodcore")

        # Subsystems
        self.evolution_engine = EvolutionEngine()
        self.planck_processor = PlanckTensorProcessor(depth=planck_depth)
        self.consciousness_amplifier = ConsciousnessAmplifier()
        self.memory_store = MemoryDeltaStore()

    # ------------------------------------------------------------------ population management

    def initialize_population(self) -> None:
        """Spawn the first generation of DigitalAgents with random trait seeds."""
        self.population = []
        for agent_id in range(self.default_population_size):
            agent = DigitalAgent(genome_id=agent_id, generation=0)
            self.population.append(agent)
        self.logger.info(
            "Initialized %d DigitalAgents (generation 0).",
            self.default_population_size,
        )

    def evaluate_population(self) -> None:
        """
        Run cognitive benchmarks for every agent in the current population,
        record fitness scores, and persist each result as a MemoryDelta.
        """
        for agent in self.population:
            fitness_score = agent.perform_cognitive_task()
            agent.fitness_score = fitness_score
            self._log_agent_memory(agent)

        self.logger.info(
            "Evaluated %d agents — memory deltas recorded.", len(self.population)
        )

    def evolve_population(self) -> None:
        """
        Breed the next generation using the EvolutionEngine.

        The top-performing half of the population is selected as parent
        candidates; offspring are produced via uniform crossover followed
        by Gaussian mutation until the target population size is reached.
        """
        if not self.population:
            self.logger.warning("evolve_population called on empty population — skipped.")
            return

        self.population.sort(key=lambda a: a.fitness_score, reverse=True)
        top_agents = self.population[: max(2, self.default_population_size // 2)]

        next_generation: list[DigitalAgent] = []
        while len(next_generation) < self.default_population_size:
            parent_a, parent_b = random.sample(top_agents, 2)
            child = self.evolution_engine.crossover(parent_a, parent_b)
            self.evolution_engine.mutate(child)
            next_generation.append(child)

        self.population = next_generation
        self.logger.info(
            "Evolved population to next generation (size: %d).", len(next_generation)
        )

    def update_planck_memory_cycle(self) -> None:
        """
        Bind population memory through fractal audio processing and compress
        all current belief deltas to Planck density.
        """
        self.planck_processor.fractal_audio_memory(self.population)

        for delta in self.memory_store.query_beliefs():
            self.consciousness_amplifier.compress_knowledge_to_planck_density(delta)

        self.logger.info("Planck Tensor memory cycle complete.")

    # ------------------------------------------------------------------ orchestration

    def cognitive_loop(self, iterations: Optional[int] = None) -> None:
        """
        Run the full population lifecycle for a fixed number of iterations.

        Parameters
        ----------
        iterations : number of cycles; defaults to ``self.iterations``.
        """
        n = iterations if iterations is not None else self.iterations
        self.logger.info("Starting cognitive loop (%d iterations)…", n)
        self.initialize_population()

        for cycle in range(n):
            self.logger.info("Cycle %d/%d", cycle + 1, n)
            self.evaluate_population()
            self.evolve_population()
            self.update_planck_memory_cycle()

        self.logger.info("Cognitive loop complete.")

    def wake_sleep_rem_cycle(self, cycles: Optional[int] = None) -> None:
        """
        Execute wake → sleep → REM consciousness cycles.

        Parameters
        ----------
        cycles : number of full wake-sleep-REM iterations to perform.
                 Pass ``None`` to run indefinitely (use only in a dedicated
                 background thread or process).
        """
        self.logger.info(
            "Starting wake-sleep-REM cycle (%s iterations)…",
            cycles if cycles is not None else "unlimited",
        )
        completed = 0
        while cycles is None or completed < cycles:
            self.planck_processor.expand_consciousness()   # Wake
            self.planck_processor.compress_knowledge()     # Sleep
            self.planck_processor.emerge_new_patterns()    # REM
            completed += 1
            self.logger.debug("Completed wake-sleep-REM cycle %d.", completed)

    # ------------------------------------------------------------------ introspection

    def get_status(self) -> dict:
        """
        Return a serialisable snapshot of the brain's current state.
        """
        top_agent = (
            max(self.population, key=lambda a: a.fitness_score)
            if self.population
            else None
        )
        return {
            "population_size": len(self.population),
            "memory_store": self.memory_store.get_state(),
            "consciousness_density": self.consciousness_amplifier.get_density_summary(),
            "planck_cycle_log_length": len(self.planck_processor.get_cycle_log()),
            "top_agent": (
                {
                    "id": top_agent.id,
                    "genome_id": top_agent.genome_id,
                    "generation": top_agent.generation,
                    "fitness_score": round(top_agent.fitness_score, 4),
                }
                if top_agent
                else None
            ),
        }

    # ------------------------------------------------------------------ private helpers

    def _log_agent_memory(self, agent: DigitalAgent) -> None:
        """Persist an agent's task performance as a MemoryDelta belief."""
        delta = create_memory_delta_from_evidence(
            evidence_id=f"task_{agent.id}_gen{agent.generation}",
            source_module_id="VictorASIOmniBrainGodcore",
            subject=f"Agent_{agent.id}",
            predicate="performs_task",
            obj="generalized_cognition",
            is_true=True,
            confidence=agent.fitness_score,
            strength=1.0,
        )
        result = self.memory_store.add_memory_delta(delta)

        if result["success"]:
            self.logger.debug("Memory delta logged: %s", result["delta_id"])
        else:
            self.logger.warning(
                "Failed to log memory delta: %s", result.get("error")
            )
