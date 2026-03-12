"""BandoTech Enterprise — modular factory-driven task orchestration layer.

Simulates a distributed, multi-role agent factory for decomposing and
executing complex tasks across specialised sub-agents.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class AgentRole(str, Enum):
    ANALYST = "analyst"
    BUILDER = "builder"
    REVIEWER = "reviewer"
    OPTIMIZER = "optimizer"
    ORCHESTRATOR = "orchestrator"


@dataclass
class Ticket:
    ticket_id: str
    title: str
    description: str
    priority: TicketPriority
    assigned_role: AgentRole
    status: TicketStatus = TicketStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: Any = None
    error: str | None = None


@dataclass
class SubAgentResult:
    role: AgentRole
    output: Any
    duration_ms: float
    success: bool


class BandoTechFactory:
    """
    Factory-driven orchestration layer for distributed sub-agent tasks.

    Maintains a ticket queue and a registry of role handlers.  When a ticket
    is dispatched the factory routes it to the appropriate handler, records
    the result, and updates ticket status.
    """

    def __init__(self) -> None:
        self._tickets: dict[str, Ticket] = {}
        self._handlers: dict[AgentRole, Callable[..., Any]] = {}
        self._completed: list[Ticket] = []
        self._register_default_handlers()

    # ------------------------------------------------------------------ setup

    def _register_default_handlers(self) -> None:
        self._handlers[AgentRole.ANALYST] = self._default_analyst
        self._handlers[AgentRole.BUILDER] = self._default_builder
        self._handlers[AgentRole.REVIEWER] = self._default_reviewer
        self._handlers[AgentRole.OPTIMIZER] = self._default_optimizer
        self._handlers[AgentRole.ORCHESTRATOR] = self._default_orchestrator

    def register_handler(self, role: AgentRole, handler: Callable[..., Any]) -> None:
        """Replace the default handler for a role with a custom implementation."""
        self._handlers[role] = handler

    # ------------------------------------------------------------------ tickets

    def create_ticket(
        self,
        title: str,
        description: str,
        priority: TicketPriority = TicketPriority.MEDIUM,
        role: AgentRole = AgentRole.ANALYST,
    ) -> Ticket:
        ticket_id = str(uuid.uuid4())[:8]
        ticket = Ticket(
            ticket_id=ticket_id,
            title=title,
            description=description,
            priority=priority,
            assigned_role=role,
        )
        self._tickets[ticket_id] = ticket
        return ticket

    def dispatch(self, ticket_id: str) -> SubAgentResult:
        """Execute the ticket using its assigned role handler."""
        ticket = self._tickets.get(ticket_id)
        if not ticket:
            raise KeyError(f"No ticket with id '{ticket_id}'.")
        if ticket.status not in (TicketStatus.PENDING, TicketStatus.FAILED):
            raise RuntimeError(
                f"Ticket '{ticket_id}' is not dispatchable (status={ticket.status})."
            )

        ticket.status = TicketStatus.IN_PROGRESS
        handler = self._handlers.get(ticket.assigned_role, self._default_analyst)
        start = time.time()
        try:
            output = handler(ticket)
            duration_ms = (time.time() - start) * 1000
            ticket.status = TicketStatus.COMPLETE
            ticket.completed_at = time.time()
            ticket.result = output
            result = SubAgentResult(
                role=ticket.assigned_role,
                output=output,
                duration_ms=round(duration_ms, 3),
                success=True,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.time() - start) * 1000
            ticket.status = TicketStatus.FAILED
            ticket.error = str(exc)
            result = SubAgentResult(
                role=ticket.assigned_role,
                output=None,
                duration_ms=round(duration_ms, 3),
                success=False,
            )
        self._completed.append(ticket)
        return result

    def dispatch_all_pending(self) -> list[SubAgentResult]:
        """Dispatch every PENDING ticket in priority order."""
        priority_order = [
            TicketPriority.CRITICAL,
            TicketPriority.HIGH,
            TicketPriority.MEDIUM,
            TicketPriority.LOW,
        ]
        pending = [t for t in self._tickets.values() if t.status == TicketStatus.PENDING]
        pending.sort(key=lambda t: priority_order.index(t.priority))
        return [self.dispatch(t.ticket_id) for t in pending]

    def get_stats(self) -> dict[str, Any]:
        statuses = [t.status.value for t in self._tickets.values()]
        return {
            "total": len(self._tickets),
            "pending": statuses.count("pending"),
            "in_progress": statuses.count("in_progress"),
            "complete": statuses.count("complete"),
            "failed": statuses.count("failed"),
        }

    # ------------------------------------------------------------------ default handlers

    @staticmethod
    def _default_analyst(ticket: Ticket) -> str:
        return (
            f"[ANALYST] Analysis complete for '{ticket.title}': "
            f"{ticket.description[:80]}"
        )

    @staticmethod
    def _default_builder(ticket: Ticket) -> str:
        return f"[BUILDER] Build artifact created for '{ticket.title}'."

    @staticmethod
    def _default_reviewer(ticket: Ticket) -> str:
        return f"[REVIEWER] Review passed for '{ticket.title}'."

    @staticmethod
    def _default_optimizer(ticket: Ticket) -> str:
        return f"[OPTIMIZER] Optimisation applied to '{ticket.title}'."

    @staticmethod
    def _default_orchestrator(ticket: Ticket) -> str:
        return f"[ORCHESTRATOR] Orchestration complete for '{ticket.title}'."
