import pytest

from backend.core.adaptive_agent import AdaptiveAgent
from backend.models.agent import AgentArchetype, AgentConfig, BehaviorProfile


@pytest.fixture
def base_config() -> AgentConfig:
    return AgentConfig(
        name="TestAgent",
        description="A test agent",
        archetype=AgentArchetype.EXPLORER,
        behavior_profile=BehaviorProfile(
            curiosity=0.8,
            caution=0.3,
            creativity=0.7,
            precision=0.5,
            autonomy=0.6,
            empathy=0.4,
        ),
        capabilities=["web_search"],
        system_prompt="Test prompt",
        memory_enabled=True,
        can_spawn_children=True,
        max_children=3,
        ecosystem_tags=["test"],
    )


@pytest.fixture
def agent(base_config: AgentConfig) -> AdaptiveAgent:
    return AdaptiveAgent(agent_id="test-001", config=base_config)


# ------------------------------------------------------------------ creation

class TestAdaptiveAgentCreation:
    def test_fields_populated(self, agent):
        assert agent.agent_id == "test-001"
        assert agent.config.name == "TestAgent"
        assert agent.behavior_profile["curiosity"] == pytest.approx(0.8)

    def test_memory_enabled(self, agent):
        assert agent.memory is not None

    def test_memory_disabled(self, base_config):
        cfg = base_config.model_copy(update={"memory_enabled": False})
        a = AdaptiveAgent(agent_id="no-mem", config=cfg)
        assert a.memory is None

    def test_initial_generation_zero(self, agent):
        assert agent._generation == 0
        assert agent.get_evolution_history() == []

    def test_behavior_profile_reflects_config(self, agent):
        for trait in ("curiosity", "caution", "creativity", "precision", "autonomy", "empathy"):
            assert trait in agent.behavior_profile


# ------------------------------------------------------------------ run

class TestAdaptiveAgentRun:
    def test_returns_agent_response(self, agent):
        resp = agent.run(input="Hello!", context={})
        assert resp.agent_id == "test-001"
        assert len(resp.output) > 0
        assert resp.processing_time_ms >= 0

    def test_intent_populated(self, agent):
        resp = agent.run(input="Create a new report", context={})
        assert resp.intent is not None
        assert "goal" in resp.intent
        assert "confidence" in resp.intent

    def test_capabilities_list(self, agent):
        resp = agent.run(input="Search for AI news", context={})
        assert isinstance(resp.capabilities_used, list)

    def test_memory_updated_when_enabled(self, agent):
        agent.run(input="Remember project alpha", context={})
        assert len(agent.memory.short_term) > 0

    def test_memory_not_updated_when_disabled(self, base_config):
        cfg = base_config.model_copy(update={"memory_enabled": False})
        a = AdaptiveAgent(agent_id="no-mem-run", config=cfg)
        resp = a.run(input="test", context={})
        assert resp.memory_updated is False

    def test_all_archetypes_produce_output(self, base_config):
        for archetype in AgentArchetype:
            cfg = base_config.model_copy(update={"archetype": archetype})
            a = AdaptiveAgent(agent_id=f"arch-{archetype}", config=cfg)
            assert len(a.run(input="What is your purpose?", context={}).output) > 0

    def test_context_stored_in_memory(self, agent):
        agent.run(input="analyse", context={"project": "apollo", "phase": "2"})
        keys = {item.key for item in agent.memory.short_term}
        assert "project" in keys
        assert "phase" in keys


# ------------------------------------------------------------------ adapt

class TestAdaptiveAgentAdapt:
    def test_direct_trait_delta(self, agent):
        old = agent.behavior_profile["curiosity"]
        agent.adapt({"curiosity": 0.1})
        assert agent.behavior_profile["curiosity"] == pytest.approx(old + 0.1)

    def test_clamps_upper_bound(self, agent):
        agent.adapt({"curiosity": 10.0})
        assert agent.behavior_profile["curiosity"] <= 1.0

    def test_clamps_lower_bound(self, agent):
        agent.adapt({"curiosity": -10.0})
        assert agent.behavior_profile["curiosity"] >= 0.0

    def test_increments_generation(self, agent):
        agent.adapt({"curiosity": 0.05})
        assert agent._generation == 1
        agent.adapt({"caution": -0.05})
        assert agent._generation == 2

    def test_records_evolution_history(self, agent):
        agent.adapt({"curiosity": 0.1})
        history = agent.get_evolution_history()
        assert len(history) == 1
        assert history[0]["event_type"] == "adapt"

    def test_positive_feedback_amplifies_tendencies(self, agent):
        # curiosity=0.8 is above 0.5 → should increase
        old_curiosity = agent.behavior_profile["curiosity"]
        agent.adapt({"positive": 1.0})
        assert agent.behavior_profile["curiosity"] >= old_curiosity

    def test_negative_feedback_nudges_toward_neutral(self, agent):
        agent.behavior_profile["curiosity"] = 0.9
        agent.behavior_profile["caution"] = 0.1
        agent.adapt({"negative": 1.0})
        assert agent.behavior_profile["curiosity"] <= 0.9
        assert agent.behavior_profile["caution"] >= 0.1

    def test_empty_feedback_produces_no_changes(self, agent):
        original = dict(agent.behavior_profile)
        changes = agent.adapt({})
        assert changes == {}
        assert agent.behavior_profile == original

    def test_config_profile_synced_after_adapt(self, agent):
        agent.adapt({"creativity": 0.15})
        assert agent.config.behavior_profile.creativity == pytest.approx(
            agent.behavior_profile["creativity"]
        )

    def test_behavior_summary(self, agent):
        summary = agent.get_behavior_summary()
        assert "profile" in summary
        assert "dominant_traits" in summary
        assert "personality_summary" in summary


# ------------------------------------------------------------------ spawn

class TestAdaptiveAgentSpawn:
    def test_child_has_unique_id(self, agent):
        child = agent.spawn_child("analyse data")
        assert child.agent_id != agent.agent_id

    def test_child_name_derived_from_parent(self, agent):
        child = agent.spawn_child("analyse data")
        assert child.config.name.startswith(agent.config.name)

    def test_spawn_records_history(self, agent):
        agent.spawn_child("test task")
        assert any(e["event_type"] == "spawn" for e in agent.get_evolution_history())

    def test_spawn_tracks_children_list(self, agent):
        agent.spawn_child("task A")
        agent.spawn_child("task B")
        assert len(agent._children) == 2

    def test_spawn_raises_when_disabled(self, base_config):
        cfg = base_config.model_copy(update={"can_spawn_children": False})
        a = AdaptiveAgent(agent_id="no-spawn", config=cfg)
        with pytest.raises(ValueError, match="not configured to spawn"):
            a.spawn_child("some task")

    def test_spawn_raises_at_max_children(self, base_config):
        cfg = base_config.model_copy(update={"can_spawn_children": True, "max_children": 1})
        a = AdaptiveAgent(agent_id="max-spawn", config=cfg)
        a.spawn_child("task 1")
        with pytest.raises(ValueError, match="max children"):
            a.spawn_child("task 2")

    def test_spawned_child_is_functional(self, agent):
        child = agent.spawn_child("process this data")
        resp = child.run(input="process this data", context={})
        assert len(resp.output) > 0

    def test_child_cannot_spawn_by_default(self, agent):
        child = agent.spawn_child("sub-task")
        assert child.config.can_spawn_children is False
