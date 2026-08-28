"""The human signature channel.

The safety kernel can decide that an action needs a person to approve it. Until
something carries that decision to an actual person and carries an answer back,
the decision is just an exception -- which is where most "human in the loop"
designs quietly stop.

This is that channel. A pending approval is broadcast to every connected
operator console, held open, and resolved by a named human via
`approval.grant` or `approval.deny`. The identity of the approver, their stated
reason, and the exact arguments they saw are written to the provenance ledger,
because "who authorised this run" is a question that gets asked a year later.

Three properties are non-negotiable:

**Timeout denies.** An approval nobody answers must not become an approval.
An agent that asks to open a laser shutter at 3am and gets no reply gets a
refusal, not a hang and not a grant.

**Approval is bound to the exact request.** The grant carries the hash of the
arguments that were shown. If the agent alters so much as a coordinate between
asking and acting, the approval no longer matches and the call is refused. This
closes the obvious attack and the more likely accident.

**The approver is never the agent.** An actor may not sign its own request.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import ApprovalDenied, LabBenchError
from ..protocol.jsonrpc import serialise


def _format_seconds(seconds: float) -> str:
    """Human-readable duration for the audit trail.

    A sub-second window rounded with `.0f` reads as "0s", which in a ledger
    looks like a bug rather than a short timeout.
    """
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}min"
    return f"{seconds / 3600:.1f}h"


class ApprovalState(str, Enum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"


class ApprovalNotFound(LabBenchError):
    code = "approval_not_found"


def request_digest(device: str, feature: str, command: str, args: dict[str, Any]) -> str:
    """Stable digest of exactly what the human was shown.

    Canonical JSON, so the same request digests identically across processes
    and Python versions. A grant is valid only for this digest.
    """
    canonical = serialise(
        {"device": device, "feature": feature, "command": command, "args": args}
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


class ApprovalRequest(BaseModel):
    """One pending question to a human."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"apr_{uuid.uuid4().hex[:12]}")
    device: str
    feature: str
    command: str
    args: dict[str, Any] = Field(default_factory=dict)
    #: The kernel's rendered prompt. Shown to the operator verbatim.
    prompt: str = ""
    hazard: str = "benign"
    reasons: list[str] = Field(default_factory=list)
    #: Who is asking.
    actor: str = "agent"
    #: The agent's stated purpose. Required for hazardous actions.
    intent: str = ""
    session_id: str = ""
    run_id: str | None = None
    digest: str = ""
    created: float = Field(default_factory=time.time)
    expires_at: float = 0.0
    state: ApprovalState = ApprovalState.PENDING
    #: Set once resolved.
    decided_by: str = ""
    decided_at: float | None = None
    decision_reason: str = ""

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "approval_id": self.id,
            "state": self.state.value,
            "device": self.device,
            "action": f"{self.feature}.{self.command}",
            "args": self.args,
            "hazard": self.hazard,
            "actor": self.actor,
            "intent": self.intent,
            "reasons": self.reasons,
            "prompt": self.prompt,
        }
        if self.state is ApprovalState.PENDING:
            out["expires_in_s"] = round(self.seconds_remaining, 1)
        else:
            # Who signed and why is the whole point of the record once the
            # question has been answered; a summary that omitted it would send
            # an auditor back to the ledger for the one field they came for.
            out["decided_by"] = self.decided_by
            out["decided_at"] = self.decided_at
            out["decision_reason"] = self.decision_reason
        return out


Broadcaster = Callable[[str, dict[str, Any]], Awaitable[Any]]


class ApprovalBroker:
    """Holds open questions and matches answers to them."""

    def __init__(
        self,
        *,
        default_timeout_s: float = 300.0,
        broadcast: Broadcaster | None = None,
        on_decision: Callable[[ApprovalRequest], Awaitable[None] | None] | None = None,
    ) -> None:
        self.default_timeout_s = default_timeout_s
        self._broadcast = broadcast
        self._on_decision = on_decision
        self._pending: dict[str, ApprovalRequest] = {}
        self._waiters: dict[str, asyncio.Future[ApprovalRequest]] = {}
        #: Resolved requests, kept so a reconnecting agent can still read the
        #: outcome and so `authorise` can verify a grant it did not wait for.
        self._history: dict[str, ApprovalRequest] = {}

    # -- asking -----------------------------------------------------------

    async def request(
        self,
        *,
        device: str,
        feature: str,
        command: str,
        args: dict[str, Any],
        prompt: str = "",
        hazard: str = "benign",
        reasons: list[str] | None = None,
        actor: str = "agent",
        intent: str = "",
        session_id: str = "",
        run_id: str | None = None,
        timeout_s: float | None = None,
    ) -> ApprovalRequest:
        """Register a question and announce it. Does not wait."""
        timeout = self.default_timeout_s if timeout_s is None else timeout_s
        req = ApprovalRequest(
            device=device, feature=feature, command=command, args=args,
            prompt=prompt, hazard=hazard, reasons=reasons or [], actor=actor,
            intent=intent, session_id=session_id, run_id=run_id,
            digest=request_digest(device, feature, command, args),
            expires_at=time.time() + timeout,
        )
        self._pending[req.id] = req
        self._waiters[req.id] = asyncio.get_running_loop().create_future()
        if self._broadcast is not None:
            await self._broadcast("approval.requested", req.summary())
        return req

    async def wait(self, approval_id: str, timeout_s: float | None = None) -> ApprovalRequest:
        """Block until a human answers, or until the request expires.

        Expiry denies. An unanswered question is not consent.
        """
        req = self._pending.get(approval_id)
        if req is None:
            resolved = self._history.get(approval_id)
            if resolved is not None:
                return resolved
            raise ApprovalNotFound(f"no approval {approval_id!r}", approval_id=approval_id)
        waiter = self._waiters[approval_id]
        remaining = timeout_s if timeout_s is not None else req.seconds_remaining
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), remaining)
        except TimeoutError:
            # Report the window the operator actually had, not the fraction of
            # a second left when this coroutine got around to timing out. The
            # ledger reader a year from now needs the former.
            window = req.expires_at - req.created
            return await self._resolve(
                approval_id, ApprovalState.EXPIRED, "system",
                f"no answer within {_format_seconds(window)}; denied by default",
            )

    async def request_and_wait(self, **kwargs: Any) -> ApprovalRequest:
        req = await self.request(**kwargs)
        return await self.wait(req.id)

    # -- answering --------------------------------------------------------

    async def grant(
        self, approval_id: str, *, approver: str, reason: str = ""
    ) -> ApprovalRequest:
        """Sign off on a pending request.

        `approver` must name a person. An agent signing its own request would
        make the whole gate decorative, so that case is refused outright.
        """
        req = self._require_pending(approval_id)
        if not approver or approver.strip().lower() in ("", "agent", "unknown"):
            raise ApprovalDenied(
                "an approval must be signed by an identified human; "
                "pass approver='<name or id>'",
                approval_id=approval_id,
            )
        if approver == req.actor:
            raise ApprovalDenied(
                f"{approver!r} raised this request and may not also approve it",
                approval_id=approval_id, actor=req.actor,
            )
        return await self._resolve(approval_id, ApprovalState.GRANTED, approver, reason)

    async def deny(
        self, approval_id: str, *, approver: str = "operator", reason: str = ""
    ) -> ApprovalRequest:
        self._require_pending(approval_id)
        return await self._resolve(approval_id, ApprovalState.DENIED, approver, reason)

    async def withdraw(self, approval_id: str, *, reason: str = "") -> ApprovalRequest:
        """The agent gives up on a question it asked."""
        self._require_pending(approval_id)
        return await self._resolve(approval_id, ApprovalState.WITHDRAWN, "agent", reason)

    def _require_pending(self, approval_id: str) -> ApprovalRequest:
        req = self._pending.get(approval_id)
        if req is None:
            done = self._history.get(approval_id)
            if done is not None:
                raise ApprovalDenied(
                    f"approval {approval_id!r} was already {done.state.value} "
                    f"by {done.decided_by or 'the system'}",
                    approval_id=approval_id, state=done.state.value,
                )
            raise ApprovalNotFound(f"no approval {approval_id!r}", approval_id=approval_id)
        return req

    async def _resolve(
        self, approval_id: str, state: ApprovalState, by: str, reason: str
    ) -> ApprovalRequest:
        req = self._pending.pop(approval_id, None)
        if req is None:  # already resolved by a racing caller
            return self._history[approval_id]
        req.state = state
        req.decided_by = by
        req.decided_at = time.time()
        req.decision_reason = reason
        self._history[approval_id] = req
        waiter = self._waiters.pop(approval_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(req)
        if self._on_decision is not None:
            result = self._on_decision(req)
            if asyncio.iscoroutine(result):
                await result
        if self._broadcast is not None:
            await self._broadcast("approval.resolved", req.summary() | {"by": by})
        return req

    # -- verification -----------------------------------------------------

    def verify(
        self,
        approval_id: str,
        *,
        device: str,
        feature: str,
        command: str,
        args: dict[str, Any],
    ) -> ApprovalRequest:
        """Check that a grant covers exactly this call.

        The digest binding is what makes an approval mean something. Without
        it, an agent could ask permission to move to a safe coordinate, get a
        signature, and then use that signature to move somewhere else.
        """
        req = self._history.get(approval_id) or self._pending.get(approval_id)
        if req is None:
            raise ApprovalNotFound(f"no approval {approval_id!r}", approval_id=approval_id)
        if req.state is not ApprovalState.GRANTED:
            raise ApprovalDenied(
                f"approval {approval_id!r} is {req.state.value}, not granted",
                approval_id=approval_id, state=req.state.value,
                decided_by=req.decided_by, reason=req.decision_reason,
            )
        digest = request_digest(device, feature, command, args)
        if digest != req.digest:
            raise ApprovalDenied(
                f"approval {approval_id!r} does not cover this call: it was granted for "
                f"{req.feature}.{req.command} with different arguments. "
                "Request approval again for the call you actually intend to make.",
                approval_id=approval_id,
                approved={"action": f"{req.feature}.{req.command}", "args": req.args},
                attempted={"action": f"{feature}.{command}", "args": args},
            )
        return req

    # -- inspection -------------------------------------------------------

    def pending(self) -> list[ApprovalRequest]:
        self.reap()
        return sorted(self._pending.values(), key=lambda r: r.created)

    def get(self, approval_id: str) -> ApprovalRequest:
        req = self._pending.get(approval_id) or self._history.get(approval_id)
        if req is None:
            raise ApprovalNotFound(f"no approval {approval_id!r}", approval_id=approval_id)
        return req

    def reap(self) -> list[str]:
        """Expire anything past its deadline.

        Called on inspection rather than by a timer: a pending approval that
        nobody is waiting on and nobody is looking at has no way to do harm,
        and one fewer background task is one fewer thing to shut down cleanly.
        """
        now = time.time()
        expired = [rid for rid, req in self._pending.items() if req.expires_at <= now]
        for rid in expired:
            req = self._pending.pop(rid)
            req.state = ApprovalState.EXPIRED
            req.decided_by = "system"
            req.decided_at = now
            req.decision_reason = "expired without an answer"
            self._history[rid] = req
            waiter = self._waiters.pop(rid, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(req)
        return expired
