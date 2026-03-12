import pytest

from backend.core.victor_thought_engine_v2 import (
    BeliefSynthesis,
    Thought,
    VictorThoughtEngineV2,
)


@pytest.fixture
def engine() -> VictorThoughtEngineV2:
    return VictorThoughtEngineV2()


# ------------------------------------------------------------------ think

class TestThink:
    def test_think_returns_thought(self, engine):
        t = engine.think("First thought")
        assert isinstance(t, Thought)

    def test_thought_id_monotonic(self, engine):
        t1 = engine.think("Alpha")
        t2 = engine.think("Beta")
        assert t2.thought_id == t1.thought_id + 1

    def test_root_thought_depth_zero(self, engine):
        t = engine.think("Root")
        assert t.depth == 0

    def test_child_thought_depth_increments(self, engine):
        root = engine.think("Root")
        child = engine.think("Child", parent_id=root.thought_id)
        assert child.depth == 1

    def test_child_registered_in_parent(self, engine):
        root = engine.think("Root")
        child = engine.think("Child", parent_id=root.thought_id)
        assert child.thought_id in engine._thoughts[root.thought_id].children

    def test_belief_weight_clamped(self, engine):
        t = engine.think("Over", belief_weight=2.0)
        assert t.belief_weight <= 1.0

    def test_depth_capped_at_max(self, engine):
        current_id = engine.think("Root").thought_id
        for _ in range(20):
            child = engine.think("Deep", parent_id=current_id)
            current_id = child.thought_id
        assert child.depth <= VictorThoughtEngineV2.MAX_DEPTH

    def test_memory_links_populated(self, engine):
        engine.think("neural network architecture")
        assert "neural" in engine._memory_links or "network" in engine._memory_links


# ------------------------------------------------------------------ reflect

class TestReflect:
    def test_reflect_creates_child_thought(self, engine):
        t = engine.think("Original belief")
        reflection = engine.reflect(t.thought_id)
        assert reflection is not None
        assert reflection.parent_id == t.thought_id

    def test_reflect_marks_source(self, engine):
        t = engine.think("Original belief")
        engine.reflect(t.thought_id)
        assert engine._thoughts[t.thought_id].reflected is True

    def test_reflect_nonexistent_returns_none(self, engine):
        assert engine.reflect(9999) is None

    def test_reflect_at_max_depth_returns_none(self, engine):
        current_id = engine.think("Root").thought_id
        for _ in range(VictorThoughtEngineV2.MAX_DEPTH):
            child = engine.think("Deep", parent_id=current_id)
            current_id = child.thought_id
        assert engine.reflect(current_id) is None


# ------------------------------------------------------------------ crawl_memory

class TestCrawlMemory:
    def test_crawl_finds_matching_thought(self, engine):
        engine.think("quantum computing breakthrough")
        results = engine.crawl_memory("quantum")
        assert len(results) > 0

    def test_crawl_returns_empty_for_unknown(self, engine):
        results = engine.crawl_memory("xyzzy_never_used")
        assert results == []

    def test_crawl_returns_thought_objects(self, engine):
        engine.think("cognitive architecture design")
        results = engine.crawl_memory("cognitive")
        assert all(isinstance(r, Thought) for r in results)


# ------------------------------------------------------------------ synthesize_beliefs

class TestSynthesizeBeliefs:
    def test_empty_engine_returns_synthesis(self, engine):
        synthesis = engine.synthesize_beliefs()
        assert isinstance(synthesis, BeliefSynthesis)
        assert "No thoughts" in synthesis.core_belief

    def test_synthesis_picks_highest_weight_root(self, engine):
        engine.think("Low weight thought", belief_weight=0.2)
        engine.think("High weight thought", belief_weight=0.9)
        synthesis = engine.synthesize_beliefs()
        assert "High weight thought" in synthesis.core_belief

    def test_confidence_in_range(self, engine):
        engine.think("Some belief")
        synthesis = engine.synthesize_beliefs()
        assert 0.0 <= synthesis.confidence <= 1.0

    def test_depth_reached_reported(self, engine):
        root = engine.think("Root")
        engine.think("Child", parent_id=root.thought_id)
        synthesis = engine.synthesize_beliefs()
        assert synthesis.depth_reached >= 1


# ------------------------------------------------------------------ stats

class TestStats:
    def test_stats_keys(self, engine):
        engine.think("Test")
        stats = engine.get_stats()
        for key in ("total_thoughts", "reflected_count", "memory_keywords", "max_depth_reached"):
            assert key in stats

    def test_stats_total_increments(self, engine):
        engine.think("A")
        engine.think("B")
        assert engine.get_stats()["total_thoughts"] == 2
