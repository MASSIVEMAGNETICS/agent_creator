"""CognitionPipeline — orchestrates inputs through three processing stages:
  1. Focus    : extract salient signals from raw input
  2. Comprehend: build structured understanding
  3. Synthesize: produce an actionable cognitive output
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FocusResult:
    """Output of the Focus stage."""

    salient_tokens: list[str]
    attention_score: float
    raw_input: str


@dataclass
class ComprehensionResult:
    """Output of the Comprehend stage."""

    topic: str
    entities: list[str]
    intent_hint: str
    confidence: float


@dataclass
class SynthesisResult:
    """Final output of the CognitionPipeline."""

    summary: str
    action_items: list[str]
    confidence: float
    processing_time_ms: float
    stages: dict[str, Any] = field(default_factory=dict)


class CognitionPipeline:
    """
    Multi-stage cognitive processing pipeline.

    Each stage refines the representation of an input so that the final
    synthesis step produces a structured, actionable result.
    """

    # Common stop-words excluded from salient token extraction
    _STOP_WORDS = frozenset(
        "a an the is it in on at to for of and or but this that with "
        "i you we they be do have had was were".split()
    )

    def __init__(self, attention_threshold: float = 0.3) -> None:
        self.attention_threshold = attention_threshold
        self._run_count = 0

    # ------------------------------------------------------------------ stages

    def focus(self, raw_input: str) -> FocusResult:
        """Extract high-salience tokens from raw text input."""
        words = raw_input.lower().split()
        salient = [
            w.strip(".,!?;:")
            for w in words
            if w not in self._STOP_WORDS and len(w) > 2
        ]
        # Simple attention score: ratio of salient tokens to total tokens
        total = len(words) or 1
        score = min(1.0, len(salient) / total + 0.1)
        return FocusResult(
            salient_tokens=salient,
            attention_score=round(score, 4),
            raw_input=raw_input,
        )

    def comprehend(self, focus_result: FocusResult) -> ComprehensionResult:
        """Build structured understanding from focused tokens."""
        tokens = focus_result.salient_tokens

        # Heuristic topic extraction: longest token is often the key concept
        topic = max(tokens, key=len) if tokens else "unknown"

        # Entities: capitalised words from the raw input
        entities = [
            w.strip(".,!?;:")
            for w in focus_result.raw_input.split()
            if w and w[0].isupper()
        ]

        # Intent hint from known action verbs
        action_verbs = {
            "create": "create",
            "build": "create",
            "make": "create",
            "find": "query",
            "search": "query",
            "get": "query",
            "deploy": "deploy",
            "launch": "deploy",
            "run": "deploy",
            "analyse": "analyse",
            "analyze": "analyse",
            "check": "analyse",
        }
        intent_hint = "general"
        for token in tokens:
            if token in action_verbs:
                intent_hint = action_verbs[token]
                break

        confidence = min(1.0, 0.4 + 0.06 * len(tokens))
        return ComprehensionResult(
            topic=topic,
            entities=entities,
            intent_hint=intent_hint,
            confidence=round(confidence, 4),
        )

    def synthesize(
        self,
        focus_result: FocusResult,
        comprehension_result: ComprehensionResult,
    ) -> SynthesisResult:
        """Produce the final actionable cognitive output."""
        summary = (
            f"Topic '{comprehension_result.topic}' detected with intent "
            f"'{comprehension_result.intent_hint}'. Key signals: "
            + ", ".join(focus_result.salient_tokens[:5])
            + "."
        )

        action_items: list[str] = []
        if comprehension_result.intent_hint == "create":
            action_items.append(
                f"Initialize creation workflow for '{comprehension_result.topic}'"
            )
        elif comprehension_result.intent_hint == "query":
            action_items.append(
                f"Run discovery query for '{comprehension_result.topic}'"
            )
        elif comprehension_result.intent_hint == "deploy":
            action_items.append(
                f"Prepare deployment pipeline for '{comprehension_result.topic}'"
            )
        else:
            action_items.append(
                f"Process '{comprehension_result.topic}' through general workflow"
            )

        if comprehension_result.entities:
            action_items.append(
                "Resolve entities: " + ", ".join(comprehension_result.entities[:3])
            )

        confidence = (focus_result.attention_score + comprehension_result.confidence) / 2
        return SynthesisResult(
            summary=summary,
            action_items=action_items,
            confidence=round(confidence, 4),
            processing_time_ms=0.0,  # filled by process()
            stages={
                "focus": {
                    "salient_tokens": focus_result.salient_tokens,
                    "attention_score": focus_result.attention_score,
                },
                "comprehension": {
                    "topic": comprehension_result.topic,
                    "intent_hint": comprehension_result.intent_hint,
                    "confidence": comprehension_result.confidence,
                },
            },
        )

    # ------------------------------------------------------------------ main

    def process(self, raw_input: str) -> SynthesisResult:
        """Run the full three-stage pipeline and return a SynthesisResult."""
        start = time.time()
        self._run_count += 1

        focus_result = self.focus(raw_input)
        comprehension_result = self.comprehend(focus_result)
        synthesis = self.synthesize(focus_result, comprehension_result)

        synthesis.processing_time_ms = round((time.time() - start) * 1000, 3)
        return synthesis

    @property
    def run_count(self) -> int:
        return self._run_count
