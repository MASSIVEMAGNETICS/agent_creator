import pytest

from backend.core.polymorphic_attention_orchestrator import (
    AttentionPhase,
    PolymorphicAttentionOrchestrator,
)


@pytest.fixture
def orchestrator() -> PolymorphicAttentionOrchestrator:
    return PolymorphicAttentionOrchestrator()


# ------------------------------------------------------------------ initialisation

class TestOrchestatorInit:
    def test_default_phase_is_fluid(self, orchestrator):
        assert orchestrator.phase == AttentionPhase.FLUID

    def test_custom_initial_phase(self):
        o = PolymorphicAttentionOrchestrator(initial_phase=AttentionPhase.SOLID)
        assert o.phase == AttentionPhase.SOLID

    def test_history_starts_with_init_entry(self, orchestrator):
        history = orchestrator.get_history()
        assert len(history) == 1
        assert history[0]["trigger"] == "init"


# ------------------------------------------------------------------ manual transitions

class TestManualTransition:
    def test_transition_changes_phase(self, orchestrator):
        orchestrator.transition(AttentionPhase.SOLID)
        assert orchestrator.phase == AttentionPhase.SOLID

    def test_transition_appends_to_history(self, orchestrator):
        orchestrator.transition(AttentionPhase.GAS, trigger="test")
        history = orchestrator.get_history()
        assert len(history) == 2
        assert history[-1]["phase"] == AttentionPhase.GAS.value
        assert history[-1]["trigger"] == "test"

    def test_transition_returns_snapshot(self, orchestrator):
        snap = orchestrator.transition(AttentionPhase.SINGULARITY)
        assert snap.phase == AttentionPhase.SINGULARITY

    def test_each_phase_has_params(self, orchestrator):
        for phase in AttentionPhase:
            orchestrator.transition(phase)
            params = orchestrator.get_params()
            assert "focus_weight" in params
            assert "creativity_weight" in params


# ------------------------------------------------------------------ auto transition

class TestAutoTransition:
    def test_high_coherence_low_entropy_gives_singularity(self, orchestrator):
        result = orchestrator.auto_transition(coherence=0.9, entropy=0.1)
        assert result == AttentionPhase.SINGULARITY

    def test_high_entropy_gives_gas(self, orchestrator):
        result = orchestrator.auto_transition(coherence=0.3, entropy=0.8)
        assert result == AttentionPhase.GAS

    def test_medium_coherence_gives_solid(self, orchestrator):
        result = orchestrator.auto_transition(coherence=0.6, entropy=0.3)
        assert result == AttentionPhase.SOLID

    def test_low_coherence_medium_entropy_gives_fluid(self, orchestrator):
        result = orchestrator.auto_transition(coherence=0.3, entropy=0.3)
        assert result == AttentionPhase.FLUID

    def test_no_transition_recorded_when_phase_unchanged(self, orchestrator):
        # Already FLUID; auto-transition that also resolves to FLUID should not add entry
        initial_len = len(orchestrator.get_history())
        orchestrator.auto_transition(coherence=0.3, entropy=0.3)
        assert len(orchestrator.get_history()) == initial_len


# ------------------------------------------------------------------ state

class TestOrchestatorState:
    def test_get_state_keys(self, orchestrator):
        state = orchestrator.get_state()
        assert "current_phase" in state
        assert "params" in state
        assert "transition_count" in state

    def test_transition_count_increments(self, orchestrator):
        orchestrator.transition(AttentionPhase.GAS)
        assert orchestrator.get_state()["transition_count"] == 1
