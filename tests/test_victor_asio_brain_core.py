"""
Tests for the Victor ASI framework modules:
  - backend.core.victor_evolution_engine  (DigitalAgent, EvolutionEngine)
  - backend.core.memory_delta_engine      (MemoryDeltaStore, create_memory_delta_from_evidence)
  - backend.core.planck_tensor_proc       (PlanckTensorProcessor, ConsciousnessAmplifier)
  - backend.core.victor_asio_brain_core   (VictorASIOmniBrainGodcore)
"""

import pytest

from backend.core.victor_evolution_engine import DigitalAgent, EvolutionEngine
from backend.core.memory_delta_engine import (
    MemoryDelta,
    MemoryDeltaStore,
    create_memory_delta_from_evidence,
)
from backend.core.planck_tensor_proc import ConsciousnessAmplifier, PlanckTensorProcessor
from backend.core.victor_asio_brain_core import VictorASIOmniBrainGodcore


# ============================================================ DigitalAgent
class TestDigitalAgent:
    def test_default_traits_populated(self):
        agent = DigitalAgent(genome_id=1)
        assert len(agent.traits) > 0
        for trait, value in agent.traits.items():
            assert 0.0 <= value <= 1.0, f"{trait} out of range: {value}"

    def test_unique_ids(self):
        a = DigitalAgent(genome_id=1)
        b = DigitalAgent(genome_id=2)
        assert a.id != b.id

    def test_generation_default_zero(self):
        assert DigitalAgent(genome_id=0).generation == 0

    def test_perform_cognitive_task_returns_float_in_range(self):
        agent = DigitalAgent(genome_id=1)
        for _ in range(20):
            score = agent.perform_cognitive_task()
            assert 0.0 <= score <= 1.0

    def test_fitness_score_assignable(self):
        agent = DigitalAgent(genome_id=1)
        agent.fitness_score = 0.75
        assert agent.fitness_score == pytest.approx(0.75)

    def test_custom_traits_respected(self):
        agent = DigitalAgent(genome_id=1, traits={"curiosity": 0.9})
        assert agent.traits["curiosity"] == pytest.approx(0.9)


# ============================================================ EvolutionEngine
class TestEvolutionEngine:
    @pytest.fixture
    def engine(self) -> EvolutionEngine:
        return EvolutionEngine()

    @pytest.fixture
    def parent_pair(self):
        a = DigitalAgent(genome_id=0, generation=1, traits={"curiosity": 0.8, "caution": 0.2})
        b = DigitalAgent(genome_id=1, generation=2, traits={"curiosity": 0.4, "caution": 0.6})
        return a, b

    def test_crossover_returns_digital_agent(self, engine, parent_pair):
        child = engine.crossover(*parent_pair)
        assert isinstance(child, DigitalAgent)

    def test_crossover_child_generation_is_max_plus_one(self, engine, parent_pair):
        a, b = parent_pair
        child = engine.crossover(a, b)
        assert child.generation == max(a.generation, b.generation) + 1

    def test_crossover_child_traits_from_parents(self, engine, parent_pair):
        a, b = parent_pair
        child = engine.crossover(a, b)
        for key in ("curiosity", "caution"):
            assert child.traits[key] in (a.traits[key], b.traits[key])

    def test_crossover_child_unique_id(self, engine, parent_pair):
        a, b = parent_pair
        child = engine.crossover(a, b)
        assert child.id != a.id and child.id != b.id

    def test_mutate_keeps_traits_in_range(self, engine):
        engine.MUTATION_RATE = 1.0  # Force all traits to mutate
        agent = DigitalAgent(genome_id=1)
        engine.mutate(agent)
        for value in agent.traits.values():
            assert 0.0 <= value <= 1.0

    def test_mutate_changes_agent_in_place(self, engine):
        engine.MUTATION_RATE = 1.0
        engine.MUTATION_SCALE = 0.5   # Large noise ensures at least one trait changes
        agent = DigitalAgent(genome_id=1)
        before = dict(agent.traits)
        engine.mutate(agent)
        assert agent.traits != before

    def test_crossover_explicit_generation(self, engine, parent_pair):
        child = engine.crossover(*parent_pair, generation=99)
        assert child.generation == 99


# ============================================================ MemoryDeltaStore
class TestCreateMemoryDeltaFromEvidence:
    def test_returns_memory_delta(self):
        delta = create_memory_delta_from_evidence(
            evidence_id="ev-001",
            source_module_id="test",
            subject="AgentX",
            predicate="performs_task",
            obj="cognition",
            is_true=True,
            confidence=0.9,
            strength=1.0,
        )
        assert isinstance(delta, MemoryDelta)

    def test_confidence_clamped(self):
        delta = create_memory_delta_from_evidence(
            evidence_id="ev-002",
            source_module_id="test",
            subject="A",
            predicate="p",
            obj="o",
            is_true=True,
            confidence=2.5,   # out of range
            strength=1.0,
        )
        assert delta.confidence <= 1.0

    def test_strength_clamped(self):
        delta = create_memory_delta_from_evidence(
            evidence_id="ev-003",
            source_module_id="test",
            subject="A",
            predicate="p",
            obj="o",
            is_true=True,
            confidence=0.5,
            strength=-0.5,    # out of range
        )
        assert delta.strength >= 0.0

    def test_unique_delta_ids(self):
        d1 = create_memory_delta_from_evidence(
            evidence_id="ev-a", source_module_id="t",
            subject="A", predicate="p", obj="o",
            is_true=True, confidence=0.5, strength=1.0,
        )
        d2 = create_memory_delta_from_evidence(
            evidence_id="ev-b", source_module_id="t",
            subject="A", predicate="p", obj="o",
            is_true=True, confidence=0.5, strength=1.0,
        )
        assert d1.delta_id != d2.delta_id


class TestMemoryDeltaStore:
    @pytest.fixture
    def store(self) -> MemoryDeltaStore:
        return MemoryDeltaStore()

    @pytest.fixture
    def sample_delta(self):
        return create_memory_delta_from_evidence(
            evidence_id="ev-001",
            source_module_id="VictorASIOmniBrainGodcore",
            subject="Agent_A",
            predicate="performs_task",
            obj="generalized_cognition",
            is_true=True,
            confidence=0.85,
            strength=1.0,
        )

    def test_add_returns_success(self, store, sample_delta):
        result = store.add_memory_delta(sample_delta)
        assert result["success"] is True
        assert result["delta_id"] == sample_delta.delta_id

    def test_add_stores_delta(self, store, sample_delta):
        store.add_memory_delta(sample_delta)
        beliefs = store.query_beliefs()
        assert len(beliefs) == 1

    def test_query_beliefs_returns_list(self, store, sample_delta):
        store.add_memory_delta(sample_delta)
        beliefs = store.query_beliefs()
        assert isinstance(beliefs, list)
        assert beliefs[0].subject == "Agent_A"

    def test_evidence_deduplication(self, store, sample_delta):
        store.add_memory_delta(sample_delta)
        # Same evidence_id, updated confidence
        sample_delta.confidence = 0.5
        result = store.add_memory_delta(sample_delta)
        assert result["refreshed"] is True
        assert len(store.query_beliefs()) == 1

    def test_contradiction_detected(self, store):
        true_delta = create_memory_delta_from_evidence(
            evidence_id="ev-true",
            source_module_id="test",
            subject="AgentX",
            predicate="is_active",
            obj="yes",
            is_true=True,
            confidence=0.9,
            strength=1.0,
        )
        false_delta = create_memory_delta_from_evidence(
            evidence_id="ev-false",
            source_module_id="test",
            subject="AgentX",
            predicate="is_active",
            obj="yes",
            is_true=False,    # contradicts true_delta
            confidence=0.8,
            strength=1.0,
        )
        store.add_memory_delta(true_delta)
        store.add_memory_delta(false_delta)
        assert len(store.get_contradictions()) == 1

    def test_query_by_subject(self, store):
        d1 = create_memory_delta_from_evidence(
            evidence_id="ev-1", source_module_id="t",
            subject="AgentA", predicate="p", obj="o",
            is_true=True, confidence=0.9, strength=1.0,
        )
        d2 = create_memory_delta_from_evidence(
            evidence_id="ev-2", source_module_id="t",
            subject="AgentB", predicate="p", obj="o",
            is_true=True, confidence=0.8, strength=1.0,
        )
        store.add_memory_delta(d1)
        store.add_memory_delta(d2)
        results = store.query_beliefs(subject="AgentA")
        assert all(d.subject == "AgentA" for d in results)

    def test_get_state_shape(self, store, sample_delta):
        store.add_memory_delta(sample_delta)
        state = store.get_state()
        assert "total_deltas" in state
        assert "total_contradictions" in state
        assert "deltas" in state

    def test_decayed_confidence_non_negative(self, store, sample_delta):
        store.add_memory_delta(sample_delta)
        for delta in store.query_beliefs():
            assert delta.decayed_confidence >= 0.0


# ============================================================ PlanckTensorProcessor
class TestPlanckTensorProcessor:
    @pytest.fixture
    def processor(self) -> PlanckTensorProcessor:
        return PlanckTensorProcessor(depth=2)

    @pytest.fixture
    def small_population(self):
        agents = []
        for i in range(5):
            a = DigitalAgent(genome_id=i)
            a.fitness_score = 0.5 + i * 0.1
            agents.append(a)
        return agents

    def test_fractal_audio_memory_returns_scores(self, processor, small_population):
        scores = processor.fractal_audio_memory(small_population)
        assert len(scores) == len(small_population)
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_expand_consciousness_returns_dict(self, processor, small_population):
        processor.fractal_audio_memory(small_population)
        result = processor.expand_consciousness()
        assert result["phase"] == "wake"
        assert "mean_resonance" in result

    def test_compress_knowledge_returns_dict(self, processor, small_population):
        processor.fractal_audio_memory(small_population)
        result = processor.compress_knowledge()
        assert result["phase"] == "sleep"
        assert "tensor_size" in result

    def test_emerge_new_patterns_returns_dict(self, processor, small_population):
        processor.fractal_audio_memory(small_population)
        processor.compress_knowledge()
        result = processor.emerge_new_patterns()
        assert result["phase"] == "rem"
        assert "patterns_emerged" in result

    def test_cycle_log_grows_with_phases(self, processor, small_population):
        processor.fractal_audio_memory(small_population)
        processor.expand_consciousness()
        processor.compress_knowledge()
        processor.emerge_new_patterns()
        log = processor.get_cycle_log()
        assert len(log) == 3

    def test_empty_population_no_crash(self, processor):
        scores = processor.fractal_audio_memory([])
        assert scores == []

    def test_expand_consciousness_empty_buffer(self, processor):
        result = processor.expand_consciousness()
        assert result["amplified"] == 0

    def test_compress_knowledge_empty_buffer(self, processor):
        result = processor.compress_knowledge()
        assert result["tensor_size"] == 0

    def test_compressed_tensor_accessible(self, processor, small_population):
        processor.fractal_audio_memory(small_population)
        processor.compress_knowledge()
        tensor = processor.get_compressed_tensor()
        assert isinstance(tensor, list)


# ============================================================ ConsciousnessAmplifier
class TestConsciousnessAmplifier:
    @pytest.fixture
    def amplifier(self) -> ConsciousnessAmplifier:
        return ConsciousnessAmplifier()

    @pytest.fixture
    def delta(self):
        return create_memory_delta_from_evidence(
            evidence_id="ev-amp-001",
            source_module_id="test",
            subject="Agent_Z",
            predicate="knows",
            obj="deep_truth",
            is_true=True,
            confidence=0.9,
            strength=1.0,
        )

    def test_compress_returns_dict(self, amplifier, delta):
        result = amplifier.compress_knowledge_to_planck_density(delta)
        assert "compressed_value" in result
        assert "delta_id" in result

    def test_compressed_value_in_range(self, amplifier, delta):
        result = amplifier.compress_knowledge_to_planck_density(delta)
        assert 0.0 < result["compressed_value"] < 1.0

    def test_passes_count_correct(self, amplifier, delta):
        result = amplifier.compress_knowledge_to_planck_density(delta)
        assert result["passes"] == ConsciousnessAmplifier.COMPRESSION_PASSES

    def test_density_summary_structure(self, amplifier, delta):
        amplifier.compress_knowledge_to_planck_density(delta)
        summary = amplifier.get_density_summary()
        assert "total_compressed" in summary
        assert "mean_density" in summary
        assert "above_target_pct" in summary

    def test_total_compressed_increments(self, amplifier, delta):
        amplifier.compress_knowledge_to_planck_density(delta)
        amplifier.compress_knowledge_to_planck_density(delta)
        assert amplifier.get_density_summary()["total_compressed"] == 2

    def test_above_density_target_flag(self, amplifier, delta):
        result = amplifier.compress_knowledge_to_planck_density(delta)
        assert isinstance(result["above_density_target"], bool)

    def test_empty_store_summary(self, amplifier):
        summary = amplifier.get_density_summary()
        assert summary["mean_density"] == 0.0
        assert summary["stored"] == 0


# ============================================================ VictorASIOmniBrainGodcore
class TestVictorASIOmniBrainGodcore:
    @pytest.fixture
    def brain(self) -> VictorASIOmniBrainGodcore:
        return VictorASIOmniBrainGodcore(population_size=10, iterations=2, planck_depth=2)

    def test_initialization(self, brain):
        assert brain.default_population_size == 10
        assert brain.iterations == 2
        assert len(brain.population) == 0

    def test_initialize_population(self, brain):
        brain.initialize_population()
        assert len(brain.population) == 10
        assert all(isinstance(a, DigitalAgent) for a in brain.population)

    def test_evaluate_population_assigns_fitness(self, brain):
        brain.initialize_population()
        brain.evaluate_population()
        for agent in brain.population:
            assert 0.0 <= agent.fitness_score <= 1.0

    def test_evaluate_population_records_memory_deltas(self, brain):
        brain.initialize_population()
        brain.evaluate_population()
        beliefs = brain.memory_store.query_beliefs()
        assert len(beliefs) == brain.default_population_size

    def test_evolve_population_replaces_population(self, brain):
        brain.initialize_population()
        brain.evaluate_population()
        original_ids = {a.id for a in brain.population}
        brain.evolve_population()
        new_ids = {a.id for a in brain.population}
        assert len(new_ids) == brain.default_population_size
        assert new_ids != original_ids

    def test_evolve_population_maintains_size(self, brain):
        brain.initialize_population()
        brain.evaluate_population()
        brain.evolve_population()
        assert len(brain.population) == brain.default_population_size

    def test_evolve_population_increments_generation(self, brain):
        brain.initialize_population()
        brain.evaluate_population()
        brain.evolve_population()
        assert all(a.generation >= 1 for a in brain.population)

    def test_update_planck_memory_cycle_no_crash(self, brain):
        brain.initialize_population()
        brain.evaluate_population()
        brain.update_planck_memory_cycle()  # Should not raise

    def test_cognitive_loop_runs_to_completion(self, brain):
        brain.cognitive_loop(iterations=2)
        assert len(brain.population) == brain.default_population_size

    def test_cognitive_loop_uses_default_iterations(self, brain):
        brain.cognitive_loop()   # uses brain.iterations = 2
        assert len(brain.population) == brain.default_population_size

    def test_wake_sleep_rem_cycle_bounded(self, brain):
        brain.initialize_population()
        brain.wake_sleep_rem_cycle(cycles=3)
        log = brain.planck_processor.get_cycle_log()
        assert len(log) == 3 * 3   # 3 phases × 3 cycles

    def test_get_status_shape(self, brain):
        brain.initialize_population()
        brain.evaluate_population()
        status = brain.get_status()
        assert "population_size" in status
        assert "memory_store" in status
        assert "consciousness_density" in status
        assert "top_agent" in status

    def test_get_status_top_agent_populated(self, brain):
        brain.initialize_population()
        brain.evaluate_population()
        top = brain.get_status()["top_agent"]
        assert top is not None
        assert 0.0 <= top["fitness_score"] <= 1.0

    def test_evolve_on_empty_population_safe(self, brain):
        brain.evolve_population()   # Should log a warning, not crash

    def test_subsystems_instantiated(self, brain):
        assert isinstance(brain.evolution_engine, EvolutionEngine)
        assert isinstance(brain.planck_processor, PlanckTensorProcessor)
        assert isinstance(brain.consciousness_amplifier, ConsciousnessAmplifier)
        assert isinstance(brain.memory_store, MemoryDeltaStore)

    def test_small_population_size_two(self):
        """Ensure the system works with a minimal population of 2 agents."""
        brain = VictorASIOmniBrainGodcore(population_size=2, iterations=1)
        brain.cognitive_loop()
        assert len(brain.population) == 2
