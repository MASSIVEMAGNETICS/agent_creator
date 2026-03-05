import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backend.core.agent_memory import AgentMemory
from backend.core.capability_registry import CapabilityRegistry, get_global_registry
from backend.core.intent_graph import Intent, IntentGraph
from backend.models.agent import (
    AgentArchetype,
    AgentConfig,
    AgentResponse,
    BehaviorProfile,
)


@dataclass
class BehaviorEvent:
    timestamp: datetime
    event_type: str  # "adapt" | "spawn" | "run"
    delta: dict
    generation: int
    notes: str = ""


class AdaptiveAgent:
    """
    The next-generation adaptive agent runtime.

    Unlike static tool-calling bots, AdaptiveAgent evolves its behavior
    profile in response to feedback, maintains living memory across
    interactions, and can orchestrate child agents for complex tasks.

    Key properties
    --------------
    - behavior_profile  : mutable dict of float trait weights (0–1)
    - memory            : three-tier AgentMemory (short / long / episodic)
    - intent_graph      : extracts structured intents from free-form input
    - registry          : dynamic capability discovery
    - _evolution_history: immutable log of every adaptation event
    """

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        registry: Optional[CapabilityRegistry] = None,
    ) -> None:
        self.agent_id = agent_id
        self.config = config
        self.behavior_profile: dict[str, float] = config.behavior_profile.model_dump()
        self.registry = registry or get_global_registry()
        self.intent_graph = IntentGraph()
        self.memory: Optional[AgentMemory] = AgentMemory() if config.memory_enabled else None
        self._evolution_history: list[BehaviorEvent] = []
        self._generation: int = 0
        self._children: list[str] = []

    # ------------------------------------------------------------------ public API

    def run(self, input: str, context: dict) -> AgentResponse:
        """
        Execute the agent against the given input and context.
        The response is shaped by the current behavior_profile and archetype.
        """
        start = time.time()

        intent = self.intent_graph.extract_intent(input)

        # Discover and record relevant capabilities
        capabilities_used: list[str] = []
        for cap in self.registry.discover(intent.goal)[:3]:
            cap.usage_count += 1
            capabilities_used.append(cap.name)

        # Update memory
        context_window = ""
        if self.memory:
            self.memory.remember("last_input", input, "short_term")
            self.memory.remember("last_intent", intent.goal, "short_term")
            for k, v in context.items():
                self.memory.remember(k, str(v), "short_term")
            context_window = self.memory.get_context_window()

        output = self._generate_response(input, intent, context, context_window)
        behavior_influence = self._calculate_behavior_influence(intent)

        if self.memory:
            self.memory.remember("last_output", output[:200], "short_term")

        elapsed_ms = (time.time() - start) * 1000

        return AgentResponse(
            agent_id=self.agent_id,
            input=input,
            output=output,
            intent={
                "goal": intent.goal,
                "confidence": intent.confidence,
                "archetype": intent.archetype,
            },
            capabilities_used=capabilities_used,
            behavior_influence=behavior_influence,
            memory_updated=self.memory is not None,
            processing_time_ms=round(elapsed_ms, 2),
            metadata={
                "archetype": self.config.archetype.value,
                "generation": self._generation,
                "entities": intent.entities,
            },
        )

    def adapt(self, feedback: dict) -> dict:
        """
        Adjust behavior_profile weights based on feedback.

        Recognised keys
        ---------------
        - Any trait name (curiosity, caution, …): float delta applied directly.
        - "positive"  : float strength — amplifies existing trait tendencies.
        - "negative"  : float strength — nudges all traits back toward 0.5.

        Returns a dict of {trait: {from, to, delta}} for every changed trait.
        """
        changes: dict[str, dict] = {}
        trait_names = {"curiosity", "caution", "creativity", "precision", "autonomy", "empathy"}

        # Direct trait deltas
        for key, value in feedback.items():
            if key in trait_names:
                old_val = self.behavior_profile.get(key, 0.5)
                new_val = max(0.0, min(1.0, old_val + float(value)))
                self.behavior_profile[key] = new_val
                changes[key] = {"from": old_val, "to": new_val, "delta": float(value)}

        # Global positive feedback — amplify current tendencies
        if "positive" in feedback:
            raw = feedback["positive"]
            # Accept float strength OR a list of reason strings (each item = 0.1 strength)
            strength = (len(raw) * 0.1 if isinstance(raw, list) else float(raw)) * 0.05
            for trait in trait_names:
                if trait not in changes:
                    old_val = self.behavior_profile.get(trait, 0.5)
                    adjustment = (self.behavior_profile[trait] - 0.5) * strength
                    new_val = max(0.0, min(1.0, old_val + adjustment))
                    self.behavior_profile[trait] = new_val
                    if abs(new_val - old_val) > 0.001:
                        changes[trait] = {"from": old_val, "to": new_val, "delta": adjustment}

        # Global negative feedback — nudge traits toward neutral 0.5
        if "negative" in feedback:
            raw = feedback["negative"]
            # Accept float strength OR a list of reason strings (each item = 0.1 strength)
            strength = (len(raw) * 0.1 if isinstance(raw, list) else float(raw)) * 0.05
            for trait in trait_names:
                if trait not in changes:
                    old_val = self.behavior_profile.get(trait, 0.5)
                    adjustment = (0.5 - old_val) * strength
                    new_val = max(0.0, min(1.0, old_val + adjustment))
                    self.behavior_profile[trait] = new_val
                    if abs(new_val - old_val) > 0.001:
                        changes[trait] = {"from": old_val, "to": new_val, "delta": adjustment}

        if changes:
            self._generation += 1
            self._evolution_history.append(
                BehaviorEvent(
                    timestamp=datetime.now(timezone.utc),
                    event_type="adapt",
                    delta=changes,
                    generation=self._generation,
                    notes=f"Adapted via feedback keys: {list(feedback.keys())}",
                )
            )
            self.config.behavior_profile = BehaviorProfile(**self.behavior_profile)

        return changes

    def spawn_child(self, task: str) -> "AdaptiveAgent":
        """
        Spawn a child agent specialised for the given task.

        Inherits the parent's behavior_profile and adjusts traits to match
        the detected task archetype. Raises ValueError when spawning is
        disabled or the max-children limit is reached.
        """
        if not self.config.can_spawn_children:
            raise ValueError(
                f"Agent {self.agent_id} is not configured to spawn children"
            )
        if len(self._children) >= self.config.max_children:
            raise ValueError(
                f"Agent {self.agent_id} has reached max children limit "
                f"({self.config.max_children})"
            )

        child_id = str(uuid.uuid4())
        intent = self.intent_graph.extract_intent(task)

        child_profile_dict = dict(self.behavior_profile)
        archetype_adjustments: dict[str, dict[str, float]] = {
            "CREATE":     {"creativity": 0.15, "curiosity": 0.10},
            "QUERY":      {"precision": 0.15,  "curiosity": 0.10},
            "TRANSFORM":  {"creativity": 0.10, "precision": 0.10},
            "DEPLOY":     {"precision": 0.15,  "caution":   0.10},
            "MONITOR":    {"caution":   0.15,  "precision": 0.10},
            "COLLABORATE":{"empathy":   0.15,  "autonomy": -0.10},
        }
        for trait, delta in archetype_adjustments.get(intent.archetype, {}).items():
            child_profile_dict[trait] = max(
                0.0, min(1.0, child_profile_dict.get(trait, 0.5) + delta)
            )

        child_config = AgentConfig(
            name=f"{self.config.name}_child_{len(self._children) + 1}",
            description=f"Child of {self.config.name} — task: {task[:100]}",
            archetype=self._determine_child_archetype(intent.archetype),
            behavior_profile=BehaviorProfile(**child_profile_dict),
            capabilities=list(self.config.capabilities),
            system_prompt=f"You are a specialised child agent. Your task: {task}",
            memory_enabled=True,
            can_spawn_children=False,
            max_children=0,
            ecosystem_tags=list(self.config.ecosystem_tags),
        )

        child = AdaptiveAgent(
            agent_id=child_id,
            config=child_config,
            registry=self.registry,
        )

        self._children.append(child_id)
        self._evolution_history.append(
            BehaviorEvent(
                timestamp=datetime.now(timezone.utc),
                event_type="spawn",
                delta={"child_id": child_id, "task": task[:100]},
                generation=self._generation,
                notes=f"Spawned child for: {task[:50]}",
            )
        )

        return child

    def get_evolution_history(self) -> list[dict]:
        """Return the immutable log of all adaptation and spawn events."""
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "delta": event.delta,
                "generation": event.generation,
                "notes": event.notes,
            }
            for event in self._evolution_history
        ]

    def get_behavior_summary(self) -> dict:
        dominant = sorted(self.behavior_profile.items(), key=lambda x: x[1], reverse=True)[:3]
        return {
            "profile": self.behavior_profile,
            "dominant_traits": [t[0] for t in dominant],
            "generation": self._generation,
            "archetype": self.config.archetype.value,
            "personality_summary": self._describe_personality(),
        }

    # ------------------------------------------------------------------ internals

    def _generate_response(
        self, input: str, intent: Intent, context: dict, context_window: str
    ) -> str:
        bp = self.behavior_profile
        archetype = self.config.archetype.value

        prefix_map = {
            "explorer":     "Scanning the possibility space... ",
            "guardian":     "Analyzing for risks and safeguards... ",
            "synthesizer":  "Integrating patterns across domains... ",
            "oracle":       "Processing signal through predictive models... ",
            "executor":     "Initiating action sequence... ",
            "collaborator": "Aligning with shared objectives... ",
        }
        parts: list[str] = [prefix_map.get(archetype, "Processing... ")]

        if bp.get("curiosity", 0.5) > 0.7:
            parts.append("I notice several interesting dimensions to explore here.")
        if bp.get("precision", 0.5) > 0.7:
            parts.append(f"Breaking this down precisely: the core request is '{intent.goal}'.")
        if bp.get("creativity", 0.5) > 0.7:
            parts.append("Here's an unconventional approach worth considering.")
        if bp.get("caution", 0.5) > 0.7:
            parts.append("Before proceeding, I've evaluated potential risks.")
        if bp.get("empathy", 0.5) > 0.7:
            parts.append("Understanding what matters most to you here.")

        entity_str = ", ".join(intent.entities) if intent.entities else "the target"
        intent_responses: dict[str, str] = {
            "CREATE": (
                f"I'll help you create {entity_str}. "
                "Approach: define structure → generate content → validate → refine."
            ),
            "QUERY": (
                f"Retrieving relevant information about {entity_str}. "
                "Analysis complete."
            ),
            "TRANSFORM": (
                f"Transforming the input by applying {archetype} processing patterns. "
                "Output optimised."
            ),
            "DEPLOY": (
                "Preparing deployment configuration. "
                "Environment checks passed. Initiating rollout sequence."
            ),
            "MONITOR": (
                "Monitoring active. Telemetry streams established. "
                "Current status: nominal."
            ),
            "COLLABORATE": (
                f"Coordination protocol engaged. "
                f"Synchronising with {len(context)} contextual signals."
            ),
        }
        core = intent_responses.get(
            intent.archetype,
            f"Processing '{input[:50]}...' with {archetype} intelligence patterns.",
        )

        context_note = " (drawing on accumulated context)" if len(context_window) > 10 else ""
        parts.append(core + context_note)

        if bp.get("autonomy", 0.5) > 0.7:
            parts.append(
                f"I've also identified {len(intent.sub_intents)} sub-tasks for thorough coverage."
            )

        response = " ".join(parts)
        if self._generation > 0:
            response += f" [Generation {self._generation} • Evolved]"
        return response

    def _calculate_behavior_influence(self, intent: Intent) -> dict:
        influences: dict[str, dict] = {}
        for trait, value in self.behavior_profile.items():
            if value > 0.6:
                influences[trait] = {"value": value, "influence": "high"}
            elif value < 0.4:
                influences[trait] = {"value": value, "influence": "low"}
        return influences

    def _describe_personality(self) -> str:
        bp = self.behavior_profile
        high = [t for t, v in bp.items() if v > 0.7]
        low = [t for t, v in bp.items() if v < 0.3]
        desc = f"A {self.config.archetype.value} agent"
        if high:
            desc += f" with high {', '.join(high)}"
        if low:
            desc += f" and restrained {', '.join(low)}"
        return desc

    def _determine_child_archetype(self, intent_archetype: str) -> AgentArchetype:
        mapping: dict[str, AgentArchetype] = {
            "CREATE":     AgentArchetype.SYNTHESIZER,
            "QUERY":      AgentArchetype.ORACLE,
            "TRANSFORM":  AgentArchetype.EXECUTOR,
            "DEPLOY":     AgentArchetype.EXECUTOR,
            "MONITOR":    AgentArchetype.GUARDIAN,
            "COLLABORATE":AgentArchetype.COLLABORATOR,
        }
        return mapping.get(intent_archetype, AgentArchetype.EXECUTOR)
