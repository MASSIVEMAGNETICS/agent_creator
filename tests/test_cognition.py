import pytest

from backend.core.cognition import CognitionPipeline, FocusResult, SynthesisResult


@pytest.fixture
def pipeline() -> CognitionPipeline:
    return CognitionPipeline()


# ------------------------------------------------------------------ Focus stage

class TestFocusStage:
    def test_returns_focus_result(self, pipeline):
        result = pipeline.focus("Build a new agent system")
        assert isinstance(result, FocusResult)

    def test_salient_tokens_non_empty(self, pipeline):
        result = pipeline.focus("Deploy the database cluster")
        assert len(result.salient_tokens) > 0

    def test_stop_words_excluded(self, pipeline):
        result = pipeline.focus("the and or a an")
        # All tokens are stop words — salient list should be empty
        assert result.salient_tokens == []

    def test_attention_score_in_range(self, pipeline):
        result = pipeline.focus("Create optimized neural networks")
        assert 0.0 <= result.attention_score <= 1.0

    def test_raw_input_preserved(self, pipeline):
        text = "Analyse performance metrics"
        result = pipeline.focus(text)
        assert result.raw_input == text


# ------------------------------------------------------------------ Comprehend stage

class TestComprehendStage:
    def test_returns_comprehension_result(self, pipeline):
        focus = pipeline.focus("Create a neural model")
        comp = pipeline.comprehend(focus)
        assert comp.topic != ""

    def test_create_intent_detected(self, pipeline):
        focus = pipeline.focus("create the agent")
        comp = pipeline.comprehend(focus)
        assert comp.intent_hint == "create"

    def test_query_intent_detected(self, pipeline):
        focus = pipeline.focus("find the nearest node")
        comp = pipeline.comprehend(focus)
        assert comp.intent_hint == "query"

    def test_deploy_intent_detected(self, pipeline):
        focus = pipeline.focus("deploy microservice container")
        comp = pipeline.comprehend(focus)
        assert comp.intent_hint == "deploy"

    def test_general_intent_fallback(self, pipeline):
        focus = pipeline.focus("something unknown blah")
        comp = pipeline.comprehend(focus)
        assert comp.intent_hint == "general"

    def test_confidence_in_range(self, pipeline):
        focus = pipeline.focus("analyse and build something")
        comp = pipeline.comprehend(focus)
        assert 0.0 <= comp.confidence <= 1.0

    def test_entities_list_type(self, pipeline):
        focus = pipeline.focus("Victor processes Alpha tasks")
        comp = pipeline.comprehend(focus)
        assert isinstance(comp.entities, list)


# ------------------------------------------------------------------ Process (full pipeline)

class TestFullPipeline:
    def test_process_returns_synthesis(self, pipeline):
        result = pipeline.process("Build a scalable system")
        assert isinstance(result, SynthesisResult)

    def test_summary_non_empty(self, pipeline):
        result = pipeline.process("Deploy the model to production")
        assert len(result.summary) > 0

    def test_action_items_non_empty(self, pipeline):
        result = pipeline.process("Create the neural network")
        assert len(result.action_items) > 0

    def test_processing_time_set(self, pipeline):
        result = pipeline.process("Analyse the dataset")
        assert result.processing_time_ms >= 0.0

    def test_confidence_in_range(self, pipeline):
        result = pipeline.process("Search for relevant patterns")
        assert 0.0 <= result.confidence <= 1.0

    def test_stages_in_result(self, pipeline):
        result = pipeline.process("Build a classifier")
        assert "focus" in result.stages
        assert "comprehension" in result.stages

    def test_run_count_increments(self, pipeline):
        pipeline.process("first call")
        pipeline.process("second call")
        assert pipeline.run_count == 2
