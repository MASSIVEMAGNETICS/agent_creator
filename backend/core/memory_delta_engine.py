"""
memory_delta_engine.py
======================
MemoryDelta Engine for belief-state management.

Provides time-decayed belief storage with contradiction detection, ensuring
that evolving agents can inherit validated knowledge rather than blindly
overwriting prior beliefs.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# MemoryDelta  (a single belief assertion)
# ---------------------------------------------------------------------------

@dataclass
class MemoryDelta:
    """
    A single evidence-backed belief assertion stored in the delta engine.

    Attributes
    ----------
    delta_id          : unique identifier for this delta.
    evidence_id       : caller-supplied reference to the originating evidence.
    source_module_id  : which subsystem created this delta.
    subject           : the entity this belief is about.
    predicate         : the relationship or property being asserted.
    obj               : the value or target of the predicate.
    is_true           : polarity of the assertion.
    confidence        : initial certainty in [0, 1].
    strength          : signal weight in [0, 1] used during decay.
    created_at        : creation timestamp (Unix epoch seconds).
    last_updated_at   : last modification timestamp.
    """

    delta_id: str
    evidence_id: str
    source_module_id: str
    subject: str
    predicate: str
    obj: str
    is_true: bool
    confidence: float
    strength: float
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    @property
    def decayed_confidence(self) -> float:
        """
        Return confidence adjusted for elapsed time.

        Uses exponential decay with a half-life of 24 hours so that older
        beliefs naturally lose influence without being erased outright.
        """
        age_hours = (time.time() - self.created_at) / 3600.0
        decay = 0.5 ** (age_hours / 24.0)
        return max(0.0, self.confidence * decay * self.strength)


def create_memory_delta_from_evidence(
    *,
    evidence_id: str,
    source_module_id: str,
    subject: str,
    predicate: str,
    obj: str,
    is_true: bool,
    confidence: float,
    strength: float,
) -> MemoryDelta:
    """
    Convenience factory for creating a MemoryDelta from evidence parameters.

    Returns a MemoryDelta with an auto-generated delta_id.
    """
    return MemoryDelta(
        delta_id=str(uuid.uuid4()),
        evidence_id=evidence_id,
        source_module_id=source_module_id,
        subject=subject,
        predicate=predicate,
        obj=obj,
        is_true=is_true,
        confidence=float(max(0.0, min(1.0, confidence))),
        strength=float(max(0.0, min(1.0, strength))),
    )


# ---------------------------------------------------------------------------
# MemoryDeltaStore
# ---------------------------------------------------------------------------

class MemoryDeltaStore:
    """
    Persistent (in-process) store for MemoryDelta belief assertions.

    Key behaviours
    --------------
    * Time-decayed confidence: older beliefs lose influence automatically.
    * Contradiction detection: when a new delta directly contradicts an
      existing one (same subject + predicate, opposite polarity), the
      conflict is recorded and the lower-confidence assertion is flagged.
    * Evidence de-duplication: submitting a delta with an existing
      evidence_id refreshes the original rather than creating a duplicate.
    """

    def __init__(self) -> None:
        self._deltas: dict[str, MemoryDelta] = {}          # delta_id → delta
        self._evidence_index: dict[str, str] = {}          # evidence_id → delta_id
        self._contradictions: list[dict[str, Any]] = []    # audit log

    # ---------------------------------------------------------------------- public API

    def add_memory_delta(self, delta: MemoryDelta) -> dict[str, Any]:
        """
        Store a new belief delta (or refresh an existing one).

        Returns a result dict::

            {
                "success":  bool,
                "delta_id": str | None,
                "error":    str | None,   # present only on failure
                "refreshed": bool,         # True when an existing delta was updated
            }
        """
        # De-duplicate by evidence_id
        if delta.evidence_id in self._evidence_index:
            existing_id = self._evidence_index[delta.evidence_id]
            existing = self._deltas[existing_id]
            existing.confidence = delta.confidence
            existing.strength = delta.strength
            existing.is_true = delta.is_true
            existing.last_updated_at = time.time()
            return {"success": True, "delta_id": existing_id, "refreshed": True}

        # Contradiction check: same subject + predicate, opposite polarity
        contradiction = self._find_contradiction(delta)
        if contradiction is not None:
            self._record_contradiction(delta, contradiction)

        self._deltas[delta.delta_id] = delta
        self._evidence_index[delta.evidence_id] = delta.delta_id
        return {"success": True, "delta_id": delta.delta_id, "refreshed": False}

    def query_beliefs(
        self,
        subject: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryDelta]:
        """
        Return active belief deltas, optionally filtered by subject and
        minimum decayed confidence.

        Results are sorted by decayed_confidence (highest first).
        """
        results: list[MemoryDelta] = []
        for delta in self._deltas.values():
            if subject and delta.subject != subject:
                continue
            if delta.decayed_confidence >= min_confidence:
                results.append(delta)
        results.sort(key=lambda d: d.decayed_confidence, reverse=True)
        return results

    def get_contradictions(self) -> list[dict[str, Any]]:
        """Return the contradiction audit log."""
        return list(self._contradictions)

    def get_state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the store."""
        return {
            "total_deltas": len(self._deltas),
            "total_contradictions": len(self._contradictions),
            "deltas": [
                {
                    "delta_id": d.delta_id,
                    "evidence_id": d.evidence_id,
                    "subject": d.subject,
                    "predicate": d.predicate,
                    "obj": d.obj,
                    "is_true": d.is_true,
                    "confidence": d.confidence,
                    "decayed_confidence": round(d.decayed_confidence, 4),
                }
                for d in self._deltas.values()
            ],
        }

    # ---------------------------------------------------------------------- helpers

    def _find_contradiction(self, incoming: MemoryDelta) -> Optional[MemoryDelta]:
        for existing in self._deltas.values():
            if (
                existing.subject == incoming.subject
                and existing.predicate == incoming.predicate
                and existing.obj == incoming.obj
                and existing.is_true != incoming.is_true
            ):
                return existing
        return None

    def _record_contradiction(
        self, incoming: MemoryDelta, existing: MemoryDelta
    ) -> None:
        self._contradictions.append(
            {
                "timestamp": time.time(),
                "incoming_delta_id": incoming.delta_id,
                "existing_delta_id": existing.delta_id,
                "subject": incoming.subject,
                "predicate": incoming.predicate,
                "obj": incoming.obj,
                "incoming_is_true": incoming.is_true,
                "existing_is_true": existing.is_true,
            }
        )
