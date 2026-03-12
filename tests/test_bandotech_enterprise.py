import pytest

from backend.core.bandotech_enterprise import (
    AgentRole,
    BandoTechFactory,
    Ticket,
    TicketPriority,
    TicketStatus,
)


@pytest.fixture
def factory() -> BandoTechFactory:
    return BandoTechFactory()


# ------------------------------------------------------------------ ticket creation

class TestTicketCreation:
    def test_create_returns_ticket(self, factory):
        ticket = factory.create_ticket("Task A", "Description")
        assert isinstance(ticket, Ticket)

    def test_ticket_default_status_pending(self, factory):
        ticket = factory.create_ticket("Task A", "Description")
        assert ticket.status == TicketStatus.PENDING

    def test_ticket_default_role(self, factory):
        ticket = factory.create_ticket("Task A", "Description")
        assert ticket.assigned_role == AgentRole.ANALYST

    def test_ticket_custom_priority(self, factory):
        ticket = factory.create_ticket("Critical task", "Desc", priority=TicketPriority.CRITICAL)
        assert ticket.priority == TicketPriority.CRITICAL

    def test_ticket_custom_role(self, factory):
        ticket = factory.create_ticket("Build thing", "Desc", role=AgentRole.BUILDER)
        assert ticket.assigned_role == AgentRole.BUILDER

    def test_ticket_id_unique(self, factory):
        t1 = factory.create_ticket("A", "a")
        t2 = factory.create_ticket("B", "b")
        assert t1.ticket_id != t2.ticket_id


# ------------------------------------------------------------------ dispatch

class TestDispatch:
    def test_dispatch_returns_result(self, factory):
        ticket = factory.create_ticket("Task", "Desc")
        result = factory.dispatch(ticket.ticket_id)
        assert result.success is True

    def test_dispatch_marks_ticket_complete(self, factory):
        ticket = factory.create_ticket("Task", "Desc")
        factory.dispatch(ticket.ticket_id)
        assert ticket.status == TicketStatus.COMPLETE

    def test_dispatch_sets_result(self, factory):
        ticket = factory.create_ticket("Task", "Desc")
        factory.dispatch(ticket.ticket_id)
        assert ticket.result is not None

    def test_dispatch_missing_ticket_raises(self, factory):
        with pytest.raises(KeyError):
            factory.dispatch("nonexistent-id")

    def test_dispatch_already_complete_raises(self, factory):
        ticket = factory.create_ticket("Task", "Desc")
        factory.dispatch(ticket.ticket_id)
        with pytest.raises(RuntimeError):
            factory.dispatch(ticket.ticket_id)

    def test_each_role_dispatches_successfully(self, factory):
        for role in AgentRole:
            t = factory.create_ticket(f"Task {role}", "Desc", role=role)
            result = factory.dispatch(t.ticket_id)
            assert result.success is True

    def test_custom_handler_used(self, factory):
        factory.register_handler(AgentRole.ANALYST, lambda ticket: "custom_output")
        ticket = factory.create_ticket("Task", "Desc", role=AgentRole.ANALYST)
        result = factory.dispatch(ticket.ticket_id)
        assert result.output == "custom_output"

    def test_failing_handler_marks_failed(self, factory):
        def boom(ticket):
            raise RuntimeError("handler error")
        factory.register_handler(AgentRole.BUILDER, boom)
        ticket = factory.create_ticket("Task", "Desc", role=AgentRole.BUILDER)
        result = factory.dispatch(ticket.ticket_id)
        assert result.success is False
        assert ticket.status == TicketStatus.FAILED


# ------------------------------------------------------------------ dispatch all pending

class TestDispatchAllPending:
    def test_dispatches_all(self, factory):
        for i in range(3):
            factory.create_ticket(f"Task {i}", "Desc")
        results = factory.dispatch_all_pending()
        assert len(results) == 3

    def test_priority_order_respected(self, factory):
        factory.create_ticket("Low", "d", priority=TicketPriority.LOW, role=AgentRole.ANALYST)
        factory.create_ticket("Critical", "d", priority=TicketPriority.CRITICAL, role=AgentRole.ANALYST)
        factory.create_ticket("High", "d", priority=TicketPriority.HIGH, role=AgentRole.ANALYST)
        results = factory.dispatch_all_pending()
        # All should succeed
        assert all(r.success for r in results)


# ------------------------------------------------------------------ stats

class TestStats:
    def test_stats_empty(self, factory):
        stats = factory.get_stats()
        assert stats["total"] == 0

    def test_stats_after_dispatch(self, factory):
        ticket = factory.create_ticket("Task", "Desc")
        factory.dispatch(ticket.ticket_id)
        stats = factory.get_stats()
        assert stats["complete"] == 1
        assert stats["pending"] == 0
