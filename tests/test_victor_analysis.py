import pytest

from backend.core.victor_analysis import SimulationFrame, VictorAnalysis


@pytest.fixture
def analysis() -> VictorAnalysis:
    return VictorAnalysis(dimensions=4)


def _vec(analysis: VictorAnalysis, values: list[float]) -> SimulationFrame:
    return analysis.record_frame(values)


# ------------------------------------------------------------------ SimulationFrame

class TestSimulationFrame:
    def test_magnitude_unit_vector(self):
        frame = SimulationFrame(frame_id=1, timestamp=0.0, state_vector=[1.0, 0.0, 0.0, 0.0])
        assert frame.magnitude == pytest.approx(1.0)

    def test_magnitude_general(self):
        frame = SimulationFrame(frame_id=1, timestamp=0.0, state_vector=[3.0, 4.0])
        assert frame.magnitude == pytest.approx(5.0)

    def test_dot_product(self):
        a = SimulationFrame(frame_id=1, timestamp=0.0, state_vector=[1.0, 2.0])
        b = SimulationFrame(frame_id=2, timestamp=0.0, state_vector=[3.0, 4.0])
        assert a.dot(b) == pytest.approx(11.0)

    def test_dot_dimension_mismatch_raises(self):
        a = SimulationFrame(frame_id=1, timestamp=0.0, state_vector=[1.0, 2.0])
        b = SimulationFrame(frame_id=2, timestamp=0.0, state_vector=[1.0])
        with pytest.raises(ValueError):
            a.dot(b)


# ------------------------------------------------------------------ VictorAnalysis

class TestVictorAnalysisRecording:
    def test_record_increments_counter(self, analysis):
        analysis.record_frame([1.0, 0.0, 0.0, 0.0])
        assert analysis._frame_counter == 1

    def test_record_stores_frame(self, analysis):
        frame = analysis.record_frame([1.0, 0.0, 0.0, 0.0])
        assert len(analysis._history) == 1
        assert frame.state_vector == [1.0, 0.0, 0.0, 0.0]

    def test_record_wrong_dimension_raises(self, analysis):
        with pytest.raises(ValueError):
            analysis.record_frame([1.0, 2.0])  # 2 instead of 4

    def test_rolling_window_capped(self):
        a = VictorAnalysis(dimensions=2)
        a.MAX_HISTORY = 3
        for i in range(5):
            a.record_frame([float(i), 0.0])
        assert len(a._history) == 3

    def test_frame_id_monotonic(self, analysis):
        f1 = analysis.record_frame([1.0, 0.0, 0.0, 0.0])
        f2 = analysis.record_frame([0.0, 1.0, 0.0, 0.0])
        assert f2.frame_id == f1.frame_id + 1


class TestVictorAnalysisMetrics:
    def test_coherence_single_frame(self, analysis):
        analysis.record_frame([1.0, 0.0, 0.0, 0.0])
        assert analysis.coherence() == pytest.approx(1.0)

    def test_coherence_identical_frames(self, analysis):
        for _ in range(3):
            analysis.record_frame([1.0, 0.0, 0.0, 0.0])
        assert analysis.coherence() == pytest.approx(1.0)

    def test_coherence_orthogonal_drops(self, analysis):
        analysis.record_frame([1.0, 0.0, 0.0, 0.0])
        analysis.record_frame([0.0, 1.0, 0.0, 0.0])
        assert analysis.coherence() == pytest.approx(0.0, abs=1e-6)

    def test_entropy_empty(self, analysis):
        assert analysis.entropy() == pytest.approx(0.0)

    def test_entropy_uniform_positive(self, analysis):
        analysis.record_frame([1.0, 1.0, 1.0, 1.0])
        assert analysis.entropy() > 0.0

    def test_drift_no_frames(self, analysis):
        assert analysis.drift() == pytest.approx(0.0)

    def test_drift_no_change(self, analysis):
        for _ in range(4):
            analysis.record_frame([1.0, 1.0, 1.0, 1.0])
        assert analysis.drift() == pytest.approx(0.0)

    def test_drift_detects_change(self, analysis):
        analysis.record_frame([0.0, 0.0, 0.0, 0.0])
        analysis.record_frame([1.0, 1.0, 1.0, 1.0])
        assert analysis.drift() > 0.0


class TestVictorAnalysisSummary:
    def test_summary_keys(self, analysis):
        analysis.record_frame([1.0, 0.0, 0.0, 0.0])
        s = analysis.get_summary()
        for key in ("dimensions", "frame_count", "history_window", "coherence", "entropy", "drift"):
            assert key in s

    def test_summary_dimensions_matches(self, analysis):
        assert analysis.get_summary()["dimensions"] == 4
