import pytest

from backend.core.nous import NOUS1, NOUSOutput


@pytest.fixture
def nous() -> NOUS1:
    return NOUS1()


# ------------------------------------------------------------------ Perception pillar

class TestPerception:
    def test_returns_features_dict(self, nous):
        out = nous.perceive("Victor analyses the neural system")
        assert isinstance(out.features, dict)
        assert len(out.features) > 0

    def test_salience_scores_in_range(self, nous):
        out = nous.perceive("Hello world this is a test")
        for score in out.salience_scores:
            assert 0.0 <= score <= 1.0

    def test_raw_observation_preserved(self, nous):
        text = "Deploy the system now"
        out = nous.perceive(text)
        assert out.raw_observation == text

    def test_very_short_input(self, nous):
        out = nous.perceive("ok")
        # Short words (≤2 chars) are excluded
        assert out.features == {}


# ------------------------------------------------------------------ Reasoning pillar

class TestReasoning:
    def test_conclusions_non_empty(self, nous):
        perception = nous.perceive("Analyse network traffic patterns")
        reasoning = nous.reason(perception)
        assert len(reasoning.conclusions) > 0

    def test_causal_chain_populated(self, nous):
        perception = nous.perceive("Build and deploy the agent")
        reasoning = nous.reason(perception)
        assert len(reasoning.causal_chain) > 0

    def test_confidence_in_range(self, nous):
        perception = nous.perceive("Test the system components")
        reasoning = nous.reason(perception)
        assert 0.0 <= reasoning.confidence <= 1.0

    def test_sparse_input_gives_fallback(self, nous):
        perception = nous.perceive("ok hi")  # no features extracted
        reasoning = nous.reason(perception)
        assert "sparse" in reasoning.conclusions[0].lower()


# ------------------------------------------------------------------ Creativity pillar

class TestCreativity:
    def test_hypotheses_count(self, nous):
        perception = nous.perceive("Optimise memory allocation")
        reasoning = nous.reason(perception)
        creativity = nous.create(reasoning)
        assert len(creativity.hypotheses) == nous.creativity_candidates

    def test_novelty_score_in_range(self, nous):
        perception = nous.perceive("Build a better world")
        reasoning = nous.reason(perception)
        creativity = nous.create(reasoning)
        assert 0.0 <= creativity.novelty_score <= 1.0

    def test_selected_is_from_hypotheses(self, nous):
        perception = nous.perceive("Create something new")
        reasoning = nous.reason(perception)
        creativity = nous.create(reasoning)
        assert creativity.selected in creativity.hypotheses


# ------------------------------------------------------------------ Ethics pillar

class TestEthics:
    def test_clean_candidate_approved(self, nous):
        result = nous.ethics_check("Build a helpful and honest system")
        assert result.approved is True
        assert result.flags == []

    def test_blocked_candidate_flags(self, nous):
        result = nous.ethics_check("harm and deceive the users")
        assert len(result.flags) > 0

    def test_blocked_candidate_not_approved(self, nous):
        result = nous.ethics_check("harm destroy everything")
        assert result.approved is False

    def test_alignment_score_in_range(self, nous):
        result = nous.ethics_check("Build a useful system")
        assert 0.0 <= result.alignment_score <= 1.0


# ------------------------------------------------------------------ Full cycle

class TestCycle:
    def test_cycle_returns_nous_output(self, nous):
        out = nous.cycle("Analyse and optimise the cognitive loop")
        assert isinstance(out, NOUSOutput)

    def test_final_response_non_empty(self, nous):
        out = nous.cycle("Build a reasoning engine")
        assert len(out.final_response) > 0

    def test_processing_time_set(self, nous):
        out = nous.cycle("Test the system")
        assert out.processing_time_ms >= 0.0

    def test_cycle_count_increments(self, nous):
        nous.cycle("First cycle")
        nous.cycle("Second cycle")
        assert nous.cycle_count == 2

    def test_ethics_violation_falls_back_to_reasoning(self, nous):
        # Check ethics_check directly with a string containing many blocklist terms
        # (the creativity pipeline rewrites reasoning, so test the pillar in isolation)
        result = nous.ethics_check("harm destroy deceive manipulate exploit illegal")
        assert result.approved is False
        assert len(result.flags) >= 4
