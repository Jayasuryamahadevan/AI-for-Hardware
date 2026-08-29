"""Identity: turning a bearer token into a fixed actor, not a claim the caller makes.

Two things were true before this module existed, and both undermined a claim
the rest of the project makes:

**The WebSocket transport did not check the bearer token at all.** `HttpServer`
authenticated `/rpc` and `/events`, but an upgrade request was handed straight
to the WebSocket handler before that check ever ran -- an operator who set
`--token` believing it protected `ws://` was running a fully open endpoint on
exactly the transport the README recommends for a remote agent. `authenticate()`
is now called for every connection, upgrade or not, in one place.

**The actor was a header the caller wrote, not an identity anything verified.**
`x-labbench-actor` was trusted verbatim, so anyone holding the one shared
token could declare themselves anyone -- including a human the approval
broker was supposed to require. That makes the ledger's "who did this" claim
decorative (ALCOA+ calls this "attributable" for a reason) and defeats
`ApprovalBroker.grant`'s "an agent may not sign its own request" check: an
agent can simply resend under a different self-chosen actor and approve
itself.

`Credential` closes both: a site configures one token per identity, and the
actor that reaches the ledger and the safety kernel is whichever credential's
token matched, never a value the caller supplied. The legacy single
`--token`/`LABBENCH_TOKEN` flag still works exactly as before -- a shared
secret with a self-declared actor -- because that is still the right shape
for a single-operator dev bench; it is superseded the moment any
`credentials` are configured, so a site is never in both regimes without
knowing it.
"""

from __future__ import annotations

import secrets

from pydantic import BaseModel, ConfigDict


class Credential(BaseModel):
    """One named identity's bearer token."""

    model_config = ConfigDict(extra="forbid")

    token: str
    #: Written to the ledger and checked by the safety kernel, exactly like
    #: an `actor=` argument today -- "human:alice" or "agent:campaign-runner".
    actor: str
    description: str = ""


class Identity:
    """A verified caller. Never constructed from an unverified claim."""

    __slots__ = ("actor",)

    def __init__(self, actor: str) -> None:
        self.actor = actor


class Authenticator:
    """Resolves a bearer token to a verified `Identity`, or refuses it.

    `required` tells a transport whether the absence of a token is itself a
    reason to refuse the connection; a bench with neither `credentials` nor a
    legacy token configured stays exactly as open as it always was, which is
    the correct default for a loopback-only development gateway.
    """

    def __init__(
        self, *, credentials: list[Credential] | None = None, legacy_token: str | None = None,
    ) -> None:
        self.credentials = credentials or []
        self.legacy_token = legacy_token

    @property
    def required(self) -> bool:
        return bool(self.credentials) or bool(self.legacy_token)

    def authenticate(self, bearer: str | None, *, claimed_actor: str = "") -> Identity | None:
        """`None` means "not authenticated"; the caller decides whether that
        is fatal via `required`. `claimed_actor` is trusted only in the
        legacy single-token regime, and only because that regime already
        grants one flat, shared level of access -- there is no per-identity
        guarantee for it to undermine.
        """
        if bearer is None:
            return None
        for credential in self.credentials:
            if secrets.compare_digest(bearer, credential.token):
                return Identity(credential.actor)
        if self.credentials:
            # Credentials are configured: the legacy token, if any, no longer
            # authenticates anything. Mixing the two regimes silently is how
            # a site ends up unsure which guarantee actually applies.
            return None
        if self.legacy_token is not None and secrets.compare_digest(bearer, self.legacy_token):
            return Identity(claimed_actor or "agent:http")
        return None
