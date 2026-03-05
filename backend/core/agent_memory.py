import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryItem:
    key: str
    value: str
    memory_type: str  # "short_term", "long_term", "episodic"
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5


@dataclass
class Episode:
    episode_id: str
    summary: str
    items: list[MemoryItem]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)


class AgentMemory:
    """
    Living memory system for adaptive agents.

    Maintains three memory tiers:
    - Short-term: recent interaction context (capped at 50 items)
    - Long-term: persistent key facts learned over time
    - Episodic: structured episode records of complete interactions
    """

    SHORT_TERM_CAPACITY = 50
    CONSOLIDATION_THRESHOLD = 3  # Access count required to promote to long-term

    def __init__(self) -> None:
        self.short_term: list[MemoryItem] = []
        self.long_term: dict[str, MemoryItem] = {}
        self.episodic: list[Episode] = []
        self._episode_counter = 0

    def remember(self, key: str, value: str, memory_type: str = "short_term") -> None:
        """Store a memory item in the specified tier."""
        item = MemoryItem(key=key, value=str(value), memory_type=memory_type)

        if memory_type == "short_term":
            existing = next((i for i in self.short_term if i.key == key), None)
            if existing:
                existing.value = str(value)
                existing.timestamp = time.time()
                existing.access_count += 1
            else:
                self.short_term.append(item)
                if len(self.short_term) > self.SHORT_TERM_CAPACITY:
                    self.short_term.pop(0)

        elif memory_type == "long_term":
            existing_lt = self.long_term.get(key)
            if existing_lt:
                existing_lt.value = str(value)
                existing_lt.timestamp = time.time()
                existing_lt.access_count += 1
            else:
                self.long_term[key] = item

        elif memory_type == "episodic":
            self._episode_counter += 1
            episode = Episode(
                episode_id=f"ep_{self._episode_counter:04d}",
                summary=f"{key}: {str(value)[:100]}",
                items=[item],
                tags=self._extract_tags(str(value)),
            )
            self.episodic.append(episode)
            if len(self.episodic) > 100:
                self.episodic.pop(0)

    def recall(self, query: str) -> list[dict]:
        """
        Retrieve relevant memories matching the query.
        Uses keyword matching across all memory tiers.
        """
        query_words = set(query.lower().split())
        results: list[tuple[float, MemoryItem]] = []

        for item in self.short_term:
            score = self._score_memory(item, query_words)
            if score > 0:
                item.access_count += 1
                results.append((score, item))

        for item in self.long_term.values():
            score = self._score_memory(item, query_words)
            if score > 0:
                item.access_count += 1
                results.append((score + 0.2, item))  # Long-term reliability bonus

        for episode in self.episodic:
            score = self._score_episode(episode, query_words)
            if score > 0:
                ep_item = MemoryItem(
                    key=episode.episode_id,
                    value=episode.summary,
                    memory_type="episodic",
                )
                results.append((score, ep_item))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "key": item.key,
                "value": item.value,
                "memory_type": item.memory_type,
                "relevance_score": round(score, 3),
                "access_count": item.access_count,
            }
            for score, item in results[:10]
        ]

    def consolidate(self) -> dict:
        """
        Move frequently-accessed short-term items to long-term storage.
        Prunes stale, low-access short-term items.
        Returns stats about the consolidation run.
        """
        promoted = 0
        pruned = 0
        items_to_keep: list[MemoryItem] = []

        for item in self.short_term:
            should_promote = (
                item.access_count >= self.CONSOLIDATION_THRESHOLD
                or item.importance > 0.7
            )
            if should_promote:
                self.long_term[item.key] = MemoryItem(
                    key=item.key,
                    value=item.value,
                    memory_type="long_term",
                    timestamp=item.timestamp,
                    access_count=item.access_count,
                    importance=min(1.0, item.importance + 0.1),
                )
                promoted += 1
            else:
                age = time.time() - item.timestamp
                if age < 3600 or item.access_count > 0:
                    items_to_keep.append(item)
                else:
                    pruned += 1

        self.short_term = items_to_keep

        return {
            "promoted_to_long_term": promoted,
            "pruned": pruned,
            "short_term_remaining": len(self.short_term),
            "long_term_total": len(self.long_term),
        }

    def get_context_window(self, max_tokens: int = 2000) -> str:
        """
        Format memory as a context string for agent use.
        Approximates token count as 4 chars = 1 token.
        """
        lines: list[str] = []
        char_limit = max_tokens * 4
        current_chars = 0

        if self.long_term:
            lines.append("=== Long-term Knowledge ===")
            for key, item in list(self.long_term.items())[-20:]:
                line = f"  {key}: {item.value}"
                if current_chars + len(line) > char_limit:
                    break
                lines.append(line)
                current_chars += len(line)

        if self.short_term:
            lines.append("=== Recent Context ===")
            for item in self.short_term[-15:]:
                line = f"  {item.key}: {item.value}"
                if current_chars + len(line) > char_limit:
                    break
                lines.append(line)
                current_chars += len(line)

        if self.episodic:
            lines.append("=== Recent Episodes ===")
            for episode in self.episodic[-5:]:
                line = f"  [{episode.episode_id}] {episode.summary}"
                if current_chars + len(line) > char_limit:
                    break
                lines.append(line)
                current_chars += len(line)

        return "\n".join(lines)

    def get_state(self) -> dict:
        """Return the full memory state as a serialisable dict."""
        return {
            "short_term": [
                {
                    "key": item.key,
                    "value": item.value,
                    "timestamp": item.timestamp,
                    "access_count": item.access_count,
                }
                for item in self.short_term
            ],
            "long_term": {
                key: {
                    "value": item.value,
                    "timestamp": item.timestamp,
                    "access_count": item.access_count,
                    "importance": item.importance,
                }
                for key, item in self.long_term.items()
            },
            "episodic": [
                {
                    "episode_id": ep.episode_id,
                    "summary": ep.summary,
                    "created_at": ep.created_at.isoformat(),
                    "tags": ep.tags,
                }
                for ep in self.episodic
            ],
            "stats": {
                "short_term_count": len(self.short_term),
                "long_term_count": len(self.long_term),
                "episode_count": len(self.episodic),
            },
        }

    # ------------------------------------------------------------------ helpers

    def _score_memory(self, item: MemoryItem, query_words: set[str]) -> float:
        key_words = set(item.key.lower().replace("_", " ").split())
        value_words = set(item.value.lower().split())
        score = len(query_words & key_words) * 2.0
        score += len(query_words & value_words) * 1.0
        age = time.time() - item.timestamp
        score += max(0.0, 1.0 - age / 86400) * 0.3  # Recency bonus (decays over 24 h)
        return score

    def _score_episode(self, episode: Episode, query_words: set[str]) -> float:
        summary_words = set(episode.summary.lower().split())
        tag_words = set(" ".join(episode.tags).lower().split())
        return len(query_words & summary_words) * 1.0 + len(query_words & tag_words) * 1.5

    def _extract_tags(self, text: str) -> list[str]:
        stop_words = {
            "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "this", "that", "with",
        }
        words = [w.lower() for w in text.split() if len(w) > 3 and w.lower() not in stop_words]
        return [word for word, _ in Counter(words).most_common(5)]
