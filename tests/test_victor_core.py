import math

import pytest

from backend.core.victor_core import (
    FlowerOfLifeTopology,
    OrchORResult,
    QuantumCandidate,
    VictorCore,
)


# ------------------------------------------------------------------ FlowerOfLifeTopology

class TestFlowerOfLifeTopology:
    def test_empty_returns_empty(self):
        topo = FlowerOfLifeTopology()
        assert topo.weights(0) == []

    def test_single_candidate(self):
        topo = FlowerOfLifeTopology()
        weights = topo.weights(1)
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        topo = FlowerOfLifeTopology()
        for n in (2, 3, 5, 8):
            weights = topo.weights(n)
            # L2-normalised: sum of squares should equal 1
            assert sum(w * w for w in weights) == pytest.approx(1.0, abs=1e-5)

    def test_weights_all_positive(self):
        topo = FlowerOfLifeTopology()
        for w in topo.weights(6):
            assert w > 0.0


# ------------------------------------------------------------------ QuantumCandidate

class TestQuantumCandidate:
    def test_probability_is_amplitude_squared(self):
        c = QuantumCandidate(label="X", amplitude=0.5, phase=0.0)
        assert c.probability == pytest.approx(0.25)

    def test_zero_amplitude_probability(self):
        c = QuantumCandidate(label="X", amplitude=0.0, phase=0.0)
        assert c.probability == pytest.approx(0.0)


# ------------------------------------------------------------------ VictorCore superposition

class TestSuperposition:
    def test_superpose_creates_candidates(self):
        core = VictorCore()
        core.superpose(["A", "B", "C"])
        assert len(core._candidates) == 3

    def test_superpose_equal_amplitudes_sum_to_one(self):
        core = VictorCore()
        core.superpose(["A", "B", "C"])
        total = sum(c.probability for c in core._candidates)
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_superpose_custom_amplitudes_normalised(self):
        core = VictorCore()
        core.superpose(["A", "B"], amplitudes=[3.0, 1.0])
        # L2-normalised: sum of squares (probabilities) should equal 1
        total_prob = sum(c.probability for c in core._candidates)
        assert total_prob == pytest.approx(1.0, abs=1e-5)

    def test_superpose_empty_is_no_op(self):
        core = VictorCore()
        core.superpose([])
        assert core._candidates == []

    def test_superpose_mismatched_amplitudes_raises(self):
        core = VictorCore()
        with pytest.raises(ValueError):
            core.superpose(["A", "B"], amplitudes=[1.0])


# ------------------------------------------------------------------ VictorCore decoherence

class TestDecoherence:
    def test_decohere_reduces_amplitudes(self):
        core = VictorCore()
        core.superpose(["A", "B"])
        initial = [c.amplitude for c in core._candidates]
        core.decohere(elapsed=1.0)
        for i, c in enumerate(core._candidates):
            assert c.amplitude <= initial[i]

    def test_decohere_zero_elapsed_no_change(self):
        core = VictorCore()
        core.superpose(["A"])
        initial = core._candidates[0].amplitude
        core.decohere(elapsed=0.0)
        assert core._candidates[0].amplitude == pytest.approx(initial)


# ------------------------------------------------------------------ VictorCore collapse

class TestCollapse:
    def test_collapse_returns_orch_or_result(self):
        core = VictorCore()
        core.superpose(["X", "Y"])
        result = core.collapse()
        assert isinstance(result, OrchORResult)

    def test_collapse_clears_superposition(self):
        core = VictorCore()
        core.superpose(["X", "Y"])
        core.collapse()
        assert core._candidates == []

    def test_collapse_empty_returns_none(self):
        core = VictorCore()
        assert core.collapse() is None

    def test_selected_is_one_of_candidates(self):
        core = VictorCore()
        candidates = ["alpha", "beta", "gamma"]
        core.superpose(candidates)
        result = core.collapse()
        assert result.selected in candidates

    def test_probability_in_range(self):
        core = VictorCore()
        core.superpose(["A", "B", "C"])
        result = core.collapse()
        assert 0.0 <= result.probability <= 1.0

    def test_collapse_appended_to_history(self):
        core = VictorCore()
        core.decide(["A", "B"])
        assert len(core._collapse_history) == 1


# ------------------------------------------------------------------ VictorCore decide

class TestDecide:
    def test_decide_returns_result(self):
        core = VictorCore()
        result = core.decide(["move", "wait", "attack"])
        assert result is not None

    def test_decide_with_biased_amplitudes(self):
        core = VictorCore()
        # "move" gets huge amplitude — should almost certainly be selected
        result = core.decide(["move", "wait"], amplitudes=[100.0, 1.0])
        assert result.selected == "move"


# ------------------------------------------------------------------ state

class TestVictorCoreState:
    def test_state_keys(self):
        core = VictorCore()
        state = core.get_state()
        for key in ("superposition_size", "decoherence_threshold", "collapse_count", "last_collapse"):
            assert key in state

    def test_collapse_count_tracked(self):
        core = VictorCore()
        core.decide(["A", "B"])
        core.decide(["C", "D"])
        assert core.get_state()["collapse_count"] == 2
