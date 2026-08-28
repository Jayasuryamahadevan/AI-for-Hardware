"""The human-approval broker: request/grant/deny/expire, and digest binding."""

from __future__ import annotations

import asyncio

import pytest

from labbench.bridge.approval import ApprovalBroker, ApprovalDenied, ApprovalNotFound, ApprovalState


@pytest.fixture
def broker():
    return ApprovalBroker(default_timeout_s=60)


class TestRequestAndGrant:
    async def test_grant_resolves_a_waiter(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={"x": 1})
        waiter = asyncio.ensure_future(broker.wait(req.id))
        await asyncio.sleep(0.01)
        granted = await broker.grant(req.id, approver="human:alice", reason="ok")
        result = await waiter
        assert result.state is ApprovalState.GRANTED
        assert granted.decided_by == "human:alice"

    async def test_deny(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={})
        denied = await broker.deny(req.id, approver="human:bob", reason="not now")
        assert denied.state is ApprovalState.DENIED

    async def test_agent_cannot_approve_its_own_request(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={}, actor="agent:main")
        with pytest.raises(ApprovalDenied):
            await broker.grant(req.id, approver="agent:main")

    async def test_anonymous_approver_rejected(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={})
        with pytest.raises(ApprovalDenied):
            await broker.grant(req.id, approver="")
        with pytest.raises(ApprovalDenied):
            await broker.grant(req.id, approver="agent")

    async def test_double_resolution_raises_on_the_second_attempt(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={})
        await broker.grant(req.id, approver="human:alice")
        with pytest.raises(ApprovalDenied):
            await broker.deny(req.id, approver="human:bob")


class TestExpiry:
    async def test_unanswered_request_expires_and_denies(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={}, timeout_s=0.05)
        result = await broker.wait(req.id)
        assert result.state is ApprovalState.EXPIRED

    async def test_reap_moves_expired_requests_out_of_pending(self, broker):
        await broker.request(device="d", feature="F", command="c", args={}, timeout_s=0.01)
        await asyncio.sleep(0.05)
        assert broker.pending() == []


class TestDigestBinding:
    async def test_verify_accepts_the_exact_call_that_was_granted(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={"x": 1})
        await broker.grant(req.id, approver="human:alice")
        verified = broker.verify(req.id, device="d", feature="F", command="c", args={"x": 1})
        assert verified.state is ApprovalState.GRANTED

    async def test_verify_rejects_a_different_argument(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={"x": 1})
        await broker.grant(req.id, approver="human:alice")
        with pytest.raises(ApprovalDenied):
            broker.verify(req.id, device="d", feature="F", command="c", args={"x": 999})

    async def test_verify_rejects_when_not_granted(self, broker):
        req = await broker.request(device="d", feature="F", command="c", args={})
        with pytest.raises(ApprovalDenied):
            broker.verify(req.id, device="d", feature="F", command="c", args={})

    def test_verify_unknown_id_raises(self, broker):
        with pytest.raises(ApprovalNotFound):
            broker.verify("apr_nope", device="d", feature="F", command="c", args={})


class TestBroadcast:
    async def test_request_and_resolution_are_broadcast(self, broker):
        events = []

        async def sink(topic, payload):
            events.append((topic, payload.get("state")))

        broker._broadcast = sink
        req = await broker.request(device="d", feature="F", command="c", args={})
        await broker.grant(req.id, approver="human:alice")
        topics = [e[0] for e in events]
        assert topics == ["approval.requested", "approval.resolved"]
