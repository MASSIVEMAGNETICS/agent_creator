import pytest

from backend.core.capability_registry import Capability, CapabilityRegistry


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry()


# ------------------------------------------------------------------ registration

class TestCapabilityRegistration:
    def test_builtins_present(self, registry):
        names = {c.name for c in registry.list_all()}
        assert {"web_search", "code_execute", "memory_recall", "send_message", "spawn_agent"} <= names

    def test_register_custom(self, registry):
        cap = registry.register(
            name="custom",
            description="A custom capability",
            handler=lambda x: x,
            tags=["custom", "test"],
        )
        assert cap.name == "custom"
        assert cap.usage_count == 0

    def test_register_overwrites(self, registry):
        registry.register("cap", "v1", lambda: "v1", ["a"])
        registry.register("cap", "v2", lambda: "v2", ["b"])
        assert registry.get("cap").description == "v2"

    def test_get_existing(self, registry):
        cap = registry.get("web_search")
        assert cap is not None
        assert cap.name == "web_search"

    def test_get_missing_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_list_all_returns_all(self, registry):
        initial = len(registry.list_all())
        registry.register("extra", "extra cap", lambda: None)
        assert len(registry.list_all()) == initial + 1

    def test_register_without_tags(self, registry):
        cap = registry.register("notags", "no tags cap", lambda: None)
        assert cap.tags == []


# ------------------------------------------------------------------ discovery

class TestCapabilityDiscovery:
    def test_discover_web_search(self, registry):
        names = [c.name for c in registry.discover("search for recent news")]
        assert "web_search" in names

    def test_discover_code_execute(self, registry):
        names = [c.name for c in registry.discover("execute python code")]
        assert "code_execute" in names

    def test_discover_memory_recall(self, registry):
        names = [c.name for c in registry.discover("recall past memories")]
        assert "memory_recall" in names

    def test_discover_send_message_near_top(self, registry):
        results = registry.discover("send a message to notify user")
        assert len(results) > 0
        assert "send_message" in [c.name for c in results[:3]]

    def test_discover_spawn_agent(self, registry):
        names = [c.name for c in registry.discover("spawn a child agent to delegate tasks")]
        assert "spawn_agent" in names

    def test_discover_empty_intent(self, registry):
        # Should not raise
        results = registry.discover("")
        assert isinstance(results, list)

    def test_discover_irrelevant_intent_returns_few(self, registry):
        results = registry.discover("zzz xyz nnn")
        assert len(results) <= 2

    def test_discover_sorted_by_score(self, registry):
        # Add a highly relevant capability
        registry.register(
            "super_search",
            "Search and find and lookup information on the web",
            lambda q: q,
            tags=["search", "find", "lookup", "web"],
        )
        results = registry.discover("search find lookup")
        # super_search should rank highly
        top_names = [c.name for c in results[:3]]
        assert "super_search" in top_names or "web_search" in top_names


# ------------------------------------------------------------------ execution

class TestCapabilityExecution:
    def test_execute_web_search(self, registry):
        result = registry.get("web_search").execute(query="test query")
        assert result["status"] == "simulated"
        assert "results" in result

    def test_execute_code(self, registry):
        result = registry.get("code_execute").execute(code="print('hi')")
        assert result["exit_code"] == 0

    def test_execute_memory_recall(self, registry):
        result = registry.get("memory_recall").execute(query="past interactions")
        assert result["status"] == "simulated"

    def test_execute_send_message(self, registry):
        result = registry.get("send_message").execute(
            recipient="user@example.com", content="Hello!"
        )
        assert result["delivered"] is True

    def test_execute_spawn_agent(self, registry):
        result = registry.get("spawn_agent").execute(task="analyse data", archetype="oracle")
        assert "agent_id" in result

    def test_execute_increments_usage_count(self, registry):
        cap = registry.get("code_execute")
        before = cap.usage_count
        cap.execute(code="x = 1")
        assert cap.usage_count == before + 1

    def test_execute_via_capability_execute_method(self, registry):
        def double(x: int) -> int:
            return x * 2

        cap = registry.register("doubler", "Doubles a number", double, tags=["math"])
        assert cap.execute(x=5) == 10
        assert cap.usage_count == 1
