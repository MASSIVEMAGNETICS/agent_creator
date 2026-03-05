import re
from dataclasses import dataclass, field
from enum import Enum


class IntentArchetype(str, Enum):
    CREATE = "CREATE"
    QUERY = "QUERY"
    TRANSFORM = "TRANSFORM"
    DEPLOY = "DEPLOY"
    MONITOR = "MONITOR"
    COLLABORATE = "COLLABORATE"


@dataclass
class Intent:
    goal: str
    entities: list[str]
    context_requirements: list[str]
    confidence: float
    sub_intents: list["Intent"]
    archetype: str
    raw_input: str = ""
    keywords: list[str] = field(default_factory=list)


class IntentGraph:
    """
    Semantic intent extraction and decomposition engine.

    Parses free-form user input into structured Intent objects using
    keyword/pattern matching — no external ML dependencies required.
    Builds an in-memory graph of intent relationships for complex tasks.
    """

    ARCHETYPE_PATTERNS: dict[str, list[str]] = {
        "CREATE": [
            "create", "build", "make", "generate", "design", "construct",
            "develop", "write", "produce", "initialize", "new", "start",
            "launch", "establish", "form", "compose", "draft",
        ],
        "QUERY": [
            "what", "how", "why", "when", "where", "who", "find", "get",
            "retrieve", "show", "list", "search", "lookup", "fetch", "tell",
            "explain", "describe", "analyze", "check", "examine",
        ],
        "TRANSFORM": [
            "transform", "convert", "change", "modify", "update", "edit",
            "refactor", "optimize", "improve", "enhance", "process",
            "translate", "format", "parse", "filter", "sort", "map",
        ],
        "DEPLOY": [
            "deploy", "release", "publish", "ship", "push", "install",
            "provision", "configure", "setup", "activate", "enable",
            "rollout", "distribute",
        ],
        "MONITOR": [
            "monitor", "watch", "track", "observe", "log", "audit",
            "alert", "notify", "measure", "metric", "status", "health",
            "report", "dashboard", "telemetry",
        ],
        "COLLABORATE": [
            "collaborate", "share", "sync", "coordinate", "team",
            "together", "help", "assist", "support", "delegate",
            "assign", "communicate", "discuss", "review",
        ],
    }

    def __init__(self) -> None:
        self._intent_cache: dict[str, Intent] = {}
        self._node_count = 0

    def extract_intent(self, raw_input: str) -> Intent:
        """
        Parse user input into a structured Intent.
        Results are cached — identical inputs return the same object.
        """
        cache_key = raw_input.lower().strip()
        if cache_key in self._intent_cache:
            return self._intent_cache[cache_key]

        normalized = raw_input.lower().strip()
        words = re.findall(r"\b\w+\b", normalized)

        # Score every archetype
        archetype_scores: dict[str, float] = {}
        for archetype, keywords in self.ARCHETYPE_PATTERNS.items():
            score = sum(1.0 for w in words if w in keywords)
            early_bonus = sum(0.5 for w in words[:3] if w in keywords)
            archetype_scores[archetype] = score + early_bonus

        best_archetype = max(archetype_scores, key=lambda k: archetype_scores[k])
        best_score = archetype_scores[best_archetype]

        if best_score == 0:
            best_archetype = "QUERY"
            confidence = 0.3
        else:
            max_possible = max(len(words) * 1.5, 1)
            confidence = min(0.95, best_score / max_possible + 0.4)

        entities = self._extract_entities(raw_input)
        context_requirements = self._extract_context_requirements(words, best_archetype)

        stop_words = {
            "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "this", "that", "with", "from",
        }
        keywords = [w for w in words if w not in stop_words and len(w) > 2][:10]

        goal = self._summarize_goal(raw_input, best_archetype)

        intent = Intent(
            goal=goal,
            entities=entities,
            context_requirements=context_requirements,
            confidence=round(confidence, 3),
            sub_intents=[],
            archetype=best_archetype,
            raw_input=raw_input,
            keywords=keywords,
        )

        self._intent_cache[cache_key] = intent
        self._node_count += 1
        return intent

    def expand_intent(self, intent: Intent) -> list[Intent]:
        """
        Decompose a complex intent into ordered sub-intents.
        Splits on conjunctions first; falls back to archetype-specific decomposition.
        """
        parts = re.split(
            r"\band\b|\bthen\b|\bafter\b|\bnext\b",
            intent.raw_input,
            flags=re.IGNORECASE,
        )

        sub_intents: list[Intent] = []
        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if len(part) > 5:
                    sub_intents.append(self.extract_intent(part))
        else:
            sub_intents = self._decompose_single_intent(intent)

        intent.sub_intents = sub_intents
        return sub_intents

    def get_graph_stats(self) -> dict:
        archetype_dist: dict[str, int] = {}
        for intent in self._intent_cache.values():
            archetype_dist[intent.archetype] = archetype_dist.get(intent.archetype, 0) + 1
        return {
            "total_intents": self._node_count,
            "cached_intents": len(self._intent_cache),
            "archetype_distribution": archetype_dist,
        }

    # ------------------------------------------------------------------ helpers

    def _extract_entities(self, text: str) -> list[str]:
        entities: list[str] = []
        seen: set[str] = set()

        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', text):
            entity = match.group(1) or match.group(2)
            if entity.lower() not in seen:
                entities.append(entity)
                seen.add(entity.lower())

        tech_pattern = (
            r"\b(agent|model|system|service|api|database|file|user|data|"
            r"config|endpoint|token)\b"
        )
        for match in re.finditer(tech_pattern, text, re.IGNORECASE):
            entity = match.group(1).lower()
            if entity not in seen:
                entities.append(entity)
                seen.add(entity)

        noun_after_verb = (
            r"\b(?:create|build|deploy|monitor|update|delete|search|find)"
            r"\s+(\w+(?:\s+\w+)?)\b"
        )
        for match in re.finditer(noun_after_verb, text, re.IGNORECASE):
            entity = match.group(1).strip()
            if entity.lower() not in seen and len(entity) > 2:
                entities.append(entity)
                seen.add(entity.lower())

        return entities[:8]

    def _extract_context_requirements(
        self, words: list[str], archetype: str
    ) -> list[str]:
        archetype_requirements: dict[str, list[str]] = {
            "CREATE": ["target_name", "specifications", "constraints"],
            "QUERY": ["search_scope", "filters", "output_format"],
            "TRANSFORM": ["source_data", "transformation_rules", "target_format"],
            "DEPLOY": ["environment", "configuration", "dependencies"],
            "MONITOR": ["target_system", "metrics", "alert_thresholds"],
            "COLLABORATE": ["participants", "shared_context", "communication_channel"],
        }
        requirements = list(archetype_requirements.get(archetype, ["context"]))

        if any(w in words for w in ["all", "everything", "complete"]):
            requirements.append("full_scope")
        if any(w in words for w in ["fast", "quick", "urgent", "asap"]):
            requirements.append("priority_level")
        if any(w in words for w in ["secure", "safe", "private", "encrypted"]):
            requirements.append("security_config")

        return list(dict.fromkeys(requirements))  # Deduplicate, preserving order

    def _summarize_goal(self, raw_input: str, archetype: str) -> str:
        first_sentence = re.split(r"[.!?]", raw_input)[0].strip()
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."
        return f"{archetype.capitalize()}: {first_sentence}"

    def _decompose_single_intent(self, intent: Intent) -> list[Intent]:
        archetype_steps: dict[str, list[tuple[str, str]]] = {
            "CREATE": [
                ("plan the structure and requirements", "QUERY"),
                ("generate the core content", "CREATE"),
                ("validate and refine the output", "TRANSFORM"),
            ],
            "DEPLOY": [
                ("check environment readiness", "MONITOR"),
                ("configure deployment parameters", "TRANSFORM"),
                ("execute deployment", "DEPLOY"),
                ("verify deployment health", "MONITOR"),
            ],
            "TRANSFORM": [
                ("analyze input data", "QUERY"),
                ("apply transformation", "TRANSFORM"),
            ],
            "MONITOR": [
                ("establish monitoring baseline", "QUERY"),
                ("set up alert rules", "MONITOR"),
            ],
        }
        steps = archetype_steps.get(intent.archetype)
        if not steps:
            return []

        return [
            Intent(
                goal=f"{sub_archetype.capitalize()}: {sub_text}",
                entities=list(intent.entities),
                context_requirements=[],
                confidence=round(intent.confidence * 0.9, 3),
                sub_intents=[],
                archetype=sub_archetype,
                raw_input=sub_text,
                keywords=[],
            )
            for sub_text, sub_archetype in steps
        ]
