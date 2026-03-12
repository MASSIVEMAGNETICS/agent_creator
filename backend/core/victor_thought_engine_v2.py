"""Victor Thought Engine v2 — recursive thought-processing engine.

Supports embedded reflection, recursive memory crawling, and cohesive
belief synthesis through a directed tree of Thought nodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Thought:
    thought_id: int
    content: str
    depth: int
    parent_id: int | None
    timestamp: float = field(default_factory=time.time)
    belief_weight: float = 1.0
    reflected: bool = False
    children: list[int] = field(default_factory=list)


@dataclass
class BeliefSynthesis:
    core_belief: str
    supporting_thoughts: list[str]
    contradiction_count: int
    confidence: float
    depth_reached: int


class VictorThoughtEngineV2:
    """
    Recursive thought engine with reflection and belief synthesis.

    Thoughts are stored as a directed tree.  Reflection deepens the tree by
    spawning meta-thoughts that interrogate existing nodes.  Synthesis
    collapses the tree into a single coherent belief statement.
    """

    MAX_DEPTH = 8

    def __init__(self, max_thoughts: int = 512) -> None:
        self.max_thoughts = max_thoughts
        self._thoughts: dict[int, Thought] = {}
        self._id_counter = 0
        self._memory_links: dict[str, list[int]] = {}  # keyword → thought ids

    # ------------------------------------------------------------------ creation

    def think(
        self,
        content: str,
        parent_id: int | None = None,
        belief_weight: float = 1.0,
    ) -> Thought:
        """Add a new thought to the tree."""
        depth = 0
        if parent_id is not None:
            parent = self._thoughts.get(parent_id)
            if parent:
                depth = min(parent.depth + 1, self.MAX_DEPTH)

        self._id_counter += 1
        thought = Thought(
            thought_id=self._id_counter,
            content=content,
            depth=depth,
            parent_id=parent_id,
            belief_weight=min(1.0, max(0.0, belief_weight)),
        )
        self._thoughts[self._id_counter] = thought

        if parent_id is not None and parent_id in self._thoughts:
            self._thoughts[parent_id].children.append(self._id_counter)

        # Update memory links
        for word in content.lower().split():
            if len(word) > 3:
                self._memory_links.setdefault(word, []).append(self._id_counter)

        # Prune oldest thought if over capacity
        if len(self._thoughts) > self.max_thoughts:
            oldest_id = min(self._thoughts)
            self._thoughts.pop(oldest_id, None)

        return thought

    # ------------------------------------------------------------------ reflection

    def reflect(self, thought_id: int) -> Thought | None:
        """
        Generate a meta-thought that interrogates the given thought.
        Returns the new reflective thought, or None if source is not found.
        """
        source = self._thoughts.get(thought_id)
        if not source:
            return None
        if source.depth >= self.MAX_DEPTH:
            return None

        reflection_content = (
            f"[Reflection] On '{source.content[:80]}': Is this belief well-founded?"
        )
        source.reflected = True
        return self.think(
            content=reflection_content,
            parent_id=thought_id,
            belief_weight=source.belief_weight * 0.9,
        )

    def crawl_memory(self, keyword: str) -> list[Thought]:
        """Return thoughts whose content contains the keyword."""
        kw = keyword.lower()
        ids = self._memory_links.get(kw, [])
        return [self._thoughts[i] for i in ids if i in self._thoughts]

    # ------------------------------------------------------------------ synthesis

    def synthesize_beliefs(self) -> BeliefSynthesis:
        """
        Collapse the thought tree into a coherent belief statement.

        The core belief is derived from the highest-weighted root thought.
        """
        if not self._thoughts:
            return BeliefSynthesis(
                core_belief="No thoughts recorded.",
                supporting_thoughts=[],
                contradiction_count=0,
                confidence=0.0,
                depth_reached=0,
            )

        roots = [t for t in self._thoughts.values() if t.parent_id is None]
        if not roots:
            roots = list(self._thoughts.values())

        # Highest-weight root is the core belief anchor
        anchor = max(roots, key=lambda t: t.belief_weight)
        supporting = [
            t.content
            for t in self._thoughts.values()
            if t.thought_id != anchor.thought_id
        ][:5]

        # Simple contradiction count: reflected thoughts without children
        contradictions = sum(
            1 for t in self._thoughts.values() if t.reflected and not t.children
        )

        max_depth = max(t.depth for t in self._thoughts.values())
        confidence = min(1.0, anchor.belief_weight * (1.0 - 0.05 * contradictions))

        return BeliefSynthesis(
            core_belief=anchor.content,
            supporting_thoughts=supporting,
            contradiction_count=contradictions,
            confidence=round(confidence, 4),
            depth_reached=max_depth,
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_thoughts": len(self._thoughts),
            "reflected_count": sum(1 for t in self._thoughts.values() if t.reflected),
            "memory_keywords": len(self._memory_links),
            "max_depth_reached": max(
                (t.depth for t in self._thoughts.values()), default=0
            ),
        }
