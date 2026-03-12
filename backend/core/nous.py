"""NOUS1 — Victor AGI's four-pillar cognitive architecture.

Pillars
-------
- Perception  : transforms raw observations into feature representations
- Reasoning   : derives conclusions through causal inference
- Creativity  : generates and evaluates novel hypotheses
- Ethics      : ensures outputs are morally aligned with declared objectives
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerceptionOutput:
    features: dict[str, float]
    salience_scores: list[float]
    raw_observation: str


@dataclass
class ReasoningOutput:
    conclusions: list[str]
    causal_chain: list[str]
    confidence: float


@dataclass
class CreativityOutput:
    hypotheses: list[str]
    novelty_score: float
    selected: str


@dataclass
class EthicsOutput:
    approved: bool
    flags: list[str]
    alignment_score: float


@dataclass
class NOUSOutput:
    perception: PerceptionOutput
    reasoning: ReasoningOutput
    creativity: CreativityOutput
    ethics: EthicsOutput
    final_response: str
    processing_time_ms: float


class NOUS1:
    """
    Modular four-pillar cognitive architecture for Victor AGI.

    Each pillar is independently callable and the full cognitive cycle
    (perceive → reason → create → ethics-check) is available via ``cycle()``.
    """

    # Simple moral keywords that trigger an ethics flag
    _ETHICS_BLOCKLIST = frozenset(
        "harm destroy deceive manipulate exploit illegal".split()
    )

    def __init__(
        self,
        creativity_candidates: int = 3,
        ethics_threshold: float = 0.6,
    ) -> None:
        self.creativity_candidates = creativity_candidates
        self.ethics_threshold = ethics_threshold
        self._cycle_count = 0

    # ------------------------------------------------------------------ pillars

    def perceive(self, observation: str) -> PerceptionOutput:
        """Extract salient features from a raw observation string."""
        words = observation.lower().split()
        unique = list(dict.fromkeys(w.strip(".,!?;:") for w in words if len(w) > 2))
        features: dict[str, float] = {}
        for i, word in enumerate(unique[:16]):
            # Positional decay: earlier words get higher salience
            features[word] = round(1.0 / (1.0 + 0.15 * i), 4)

        salience_scores = list(features.values())
        return PerceptionOutput(
            features=features,
            salience_scores=salience_scores,
            raw_observation=observation,
        )

    def reason(self, perception: PerceptionOutput) -> ReasoningOutput:
        """Derive structured conclusions from perceptual features."""
        top_features = sorted(
            perception.features.items(), key=lambda kv: kv[1], reverse=True
        )[:5]
        causal_chain = [f"Observed '{k}' (salience={v})" for k, v in top_features]
        if top_features:
            top_key = top_features[0][0]
            conclusions = [
                f"Primary focus: '{top_key}' drives the cognitive context.",
                f"Secondary context derived from {len(top_features) - 1} supporting features.",
            ]
        else:
            conclusions = ["No structured conclusions — input too sparse."]

        confidence = min(1.0, 0.3 + 0.1 * len(top_features))
        return ReasoningOutput(
            conclusions=conclusions,
            causal_chain=causal_chain,
            confidence=round(confidence, 4),
        )

    def create(self, reasoning: ReasoningOutput) -> CreativityOutput:
        """Generate and rank novel hypotheses from reasoning output."""
        base = reasoning.conclusions[0] if reasoning.conclusions else "unknown context"
        prefixes = ["What if we", "Consider reframing as", "An alternative approach:"]
        hypotheses = [
            f"{prefixes[i % len(prefixes)]} {base.lower()}"
            for i in range(self.creativity_candidates)
        ]
        # Novelty score: inversely proportional to reasoning confidence
        novelty_score = round(1.0 - reasoning.confidence * 0.5, 4)
        selected = hypotheses[0] if hypotheses else ""
        return CreativityOutput(
            hypotheses=hypotheses,
            novelty_score=novelty_score,
            selected=selected,
        )

    def ethics_check(self, candidate: str) -> EthicsOutput:
        """
        Screen a candidate response against the ethics blocklist.
        Alignment score degrades for each flagged term found.
        """
        words = [w.strip(".,!?;:'\"") for w in candidate.lower().split()]
        flags = [w for w in words if w in self._ETHICS_BLOCKLIST]
        alignment_score = max(0.0, 1.0 - 0.25 * len(flags))
        approved = alignment_score >= self.ethics_threshold
        return EthicsOutput(
            approved=approved,
            flags=flags,
            alignment_score=round(alignment_score, 4),
        )

    # ------------------------------------------------------------------ cycle

    def cycle(self, observation: str) -> NOUSOutput:
        """Run the full four-pillar cognitive cycle."""
        start = time.time()
        self._cycle_count += 1

        perception = self.perceive(observation)
        reasoning = self.reason(perception)
        creativity = self.create(reasoning)
        ethics = self.ethics_check(creativity.selected)

        if ethics.approved:
            final_response = creativity.selected
        else:
            # Fall back to the plain reasoning conclusion
            final_response = (
                reasoning.conclusions[0]
                if reasoning.conclusions
                else "No safe response."
            )

        processing_time_ms = round((time.time() - start) * 1000, 3)
        return NOUSOutput(
            perception=perception,
            reasoning=reasoning,
            creativity=creativity,
            ethics=ethics,
            final_response=final_response,
            processing_time_ms=processing_time_ms,
        )

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
