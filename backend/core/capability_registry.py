from dataclasses import dataclass, field
from typing import Callable, Any, Optional


@dataclass
class Capability:
    name: str
    description: str
    tags: list[str]
    handler: Callable
    usage_count: int = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.usage_count += 1
        return self.handler(*args, **kwargs)


class CapabilityRegistry:
    """
    Dynamic capability discovery and management.

    Agents register named capabilities (functions / tools) and later discover
    relevant ones through keyword-based semantic matching against intent text.
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._register_builtins()

    def register(
        self,
        name: str,
        description: str,
        handler: Callable,
        tags: Optional[list[str]] = None,
    ) -> Capability:
        """Register (or overwrite) a named capability."""
        capability = Capability(
            name=name,
            description=description,
            tags=tags or [],
            handler=handler,
        )
        self._capabilities[name] = capability
        return capability

    def discover(self, intent: str) -> list[Capability]:
        """
        Return capabilities sorted by relevance to the given intent string.
        Matching is keyword-based (name, description, tags).
        """
        intent_words = set(intent.lower().split())
        scored: list[tuple[float, Capability]] = []

        for cap in self._capabilities.values():
            score = self._score_capability(cap, intent_words, intent)
            if score > 0:
                scored.append((score, cap))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cap for _, cap in scored]

    def get(self, name: str) -> Optional[Capability]:
        """Retrieve a capability by exact name."""
        return self._capabilities.get(name)

    def list_all(self) -> list[Capability]:
        """Return all registered capabilities."""
        return list(self._capabilities.values())

    # ------------------------------------------------------------------ helpers

    def _score_capability(
        self, cap: Capability, intent_words: set[str], raw_intent: str
    ) -> float:
        score = 0.0

        cap_name_words = set(cap.name.lower().replace("_", " ").split())
        score += len(intent_words & cap_name_words) * 3.0  # Name overlap

        desc_words = set(cap.description.lower().split())
        score += len(intent_words & desc_words) * 1.0  # Description overlap

        for tag in cap.tags:
            tag_words = set(tag.lower().replace("-", " ").split())
            score += len(intent_words & tag_words) * 2.0  # Tag overlap
            if tag.lower() in raw_intent.lower():
                score += 3.0  # Exact tag substring bonus

        for word in intent_words:
            if len(word) > 3:
                if word in cap.description.lower() or word in cap.name.lower():
                    score += 0.5

        return score

    def _register_builtins(self) -> None:
        """Seed the registry with the five built-in capabilities."""

        def web_search(query: str) -> dict:
            return {
                "status": "simulated",
                "query": query,
                "results": [
                    {
                        "title": f"Result for: {query}",
                        "snippet": "Simulated search result",
                        "url": "https://example.com",
                    }
                ],
            }

        def code_execute(code: str, language: str = "python") -> dict:
            return {
                "status": "simulated",
                "language": language,
                "output": f"[Simulated execution of {len(code)} chars of {language} code]",
                "exit_code": 0,
            }

        def memory_recall(query: str) -> dict:
            return {
                "status": "simulated",
                "query": query,
                "memories": [],
                "note": "Memory system active",
            }

        def send_message(recipient: str, content: str) -> dict:
            return {
                "status": "simulated",
                "recipient": recipient,
                "content_preview": content[:100],
                "delivered": True,
            }

        def spawn_agent(task: str, archetype: str = "executor") -> dict:
            return {
                "status": "simulated",
                "task": task,
                "archetype": archetype,
                "agent_id": "child-agent-placeholder",
            }

        self.register(
            name="web_search",
            description="Search the web for information, news, and data",
            handler=web_search,
            tags=["search", "web", "information", "query", "lookup", "find"],
        )
        self.register(
            name="code_execute",
            description="Execute code snippets in various programming languages",
            handler=code_execute,
            tags=["code", "execute", "run", "programming", "script", "compute"],
        )
        self.register(
            name="memory_recall",
            description="Recall information from agent memory and past interactions",
            handler=memory_recall,
            tags=["memory", "recall", "remember", "history", "past", "context"],
        )
        self.register(
            name="send_message",
            description="Send a message to a user, system, or another agent",
            handler=send_message,
            tags=["message", "send", "notify", "communicate", "alert", "contact"],
        )
        self.register(
            name="spawn_agent",
            description="Spawn a child agent to handle a specialized sub-task",
            handler=spawn_agent,
            tags=["spawn", "agent", "child", "delegate", "orchestrate", "multi-agent"],
        )


# Module-level singleton
_global_registry: Optional[CapabilityRegistry] = None


def get_global_registry() -> CapabilityRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityRegistry()
    return _global_registry
