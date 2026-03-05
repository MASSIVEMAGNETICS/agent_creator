import pytest

from backend.core.intent_graph import Intent, IntentGraph


@pytest.fixture
def graph() -> IntentGraph:
    return IntentGraph()


# ------------------------------------------------------------------ extract_intent

class TestExtractIntent:
    def test_create_intent(self, graph):
        intent = graph.extract_intent("Create a new agent for data analysis")
        assert intent.archetype == "CREATE"
        assert intent.confidence > 0.3

    def test_query_intent(self, graph):
        assert graph.extract_intent("What is the current status?").archetype == "QUERY"

    def test_deploy_intent(self, graph):
        assert graph.extract_intent("Deploy the agent to production").archetype == "DEPLOY"

    def test_monitor_intent(self, graph):
        assert graph.extract_intent("Monitor the health of all services").archetype == "MONITOR"

    def test_transform_intent(self, graph):
        assert graph.extract_intent("Transform and convert the data format").archetype == "TRANSFORM"

    def test_collaborate_intent(self, graph):
        assert graph.extract_intent("Collaborate with team members").archetype == "COLLABORATE"

    def test_goal_is_non_empty(self, graph):
        intent = graph.extract_intent("Build a dashboard for analytics")
        assert len(intent.goal) > 0

    def test_entities_list(self, graph):
        intent = graph.extract_intent("Create an agent with API access")
        assert isinstance(intent.entities, list)

    def test_context_requirements_non_empty(self, graph):
        intent = graph.extract_intent("Deploy to production environment")
        assert len(intent.context_requirements) > 0

    def test_confidence_in_bounds(self, graph):
        for text in ("Hello", "Deploy agent", "search find retrieve"):
            intent = graph.extract_intent(text)
            assert 0.0 <= intent.confidence <= 1.0

    def test_unknown_defaults_to_query(self, graph):
        intent = graph.extract_intent("xyz zzz nnn")
        assert intent.archetype == "QUERY"
        assert intent.confidence < 0.5

    def test_caching_returns_same_object(self, graph):
        text = "Create a new database"
        assert graph.extract_intent(text) is graph.extract_intent(text)

    def test_raw_input_preserved(self, graph):
        text = "Create a report"
        intent = graph.extract_intent(text)
        assert intent.raw_input == text

    def test_keywords_populated(self, graph):
        intent = graph.extract_intent("Create a comprehensive analytics platform")
        assert len(intent.keywords) > 0

    def test_security_context_requirement(self, graph):
        intent = graph.extract_intent("Deploy with secure encrypted configuration")
        assert "security_config" in intent.context_requirements

    def test_priority_context_requirement(self, graph):
        intent = graph.extract_intent("Quickly search for urgent data")
        assert "priority_level" in intent.context_requirements


# ------------------------------------------------------------------ expand_intent

class TestExpandIntent:
    def test_expand_compound_splits_on_and(self, graph):
        intent = graph.extract_intent("Create a report and then deploy it")
        sub_intents = graph.expand_intent(intent)
        assert len(sub_intents) >= 2

    def test_expand_single_deploy_gives_steps(self, graph):
        intent = graph.extract_intent("Deploy the system to production")
        sub_intents = graph.expand_intent(intent)
        assert len(sub_intents) > 0

    def test_expand_single_create_gives_steps(self, graph):
        intent = graph.extract_intent("Create a comprehensive analytics platform")
        sub_intents = graph.expand_intent(intent)
        assert len(sub_intents) > 0

    def test_sub_intents_are_intent_objects(self, graph):
        intent = graph.extract_intent("Build a system and monitor it")
        for sub in graph.expand_intent(intent):
            assert hasattr(sub, "goal")
            assert hasattr(sub, "archetype")
            assert hasattr(sub, "confidence")

    def test_expand_attaches_sub_intents_to_parent(self, graph):
        intent = graph.extract_intent("Create something and deploy it")
        subs = graph.expand_intent(intent)
        assert intent.sub_intents is subs

    def test_sub_intent_confidence_in_bounds(self, graph):
        # Sub-intents from a compound "X and Y" may score higher than the parent
        # (each part is more focused).  Just verify they are valid confidence values.
        intent = graph.extract_intent("Create a report and deploy it")
        for sub in graph.expand_intent(intent):
            assert 0.0 <= sub.confidence <= 1.0


# ------------------------------------------------------------------ graph stats

class TestIntentGraphStats:
    def test_empty_graph_stats(self, graph):
        stats = graph.get_graph_stats()
        assert stats["total_intents"] == 0
        assert stats["cached_intents"] == 0

    def test_stats_after_extraction(self, graph):
        graph.extract_intent("Create something")
        graph.extract_intent("Query the database")
        stats = graph.get_graph_stats()
        assert stats["total_intents"] == 2

    def test_cache_hit_does_not_double_count(self, graph):
        graph.extract_intent("Create something")
        graph.extract_intent("Create something")
        assert graph.get_graph_stats()["total_intents"] == 1

    def test_archetype_distribution_tracked(self, graph):
        graph.extract_intent("Create something")
        graph.extract_intent("Deploy something")
        dist = graph.get_graph_stats()["archetype_distribution"]
        assert "CREATE" in dist
        assert "DEPLOY" in dist
