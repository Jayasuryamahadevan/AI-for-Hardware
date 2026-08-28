"""The agent-facing tool surface.

Deliberately small and *fixed*. The obvious design -- one tool per device
command -- produces two hundred-odd tools for a six-instrument lab, which
buries the model's attention in an enumeration it must re-read every turn and
which changes shape whenever someone plugs in a new box.

Instead the capability model is served as *data*: `device.describe` returns
features, commands and a JSON Schema per command, and `device.invoke` takes a
feature, a command and arguments. Validation still happens against the real
schema, because `Command.validate_args` does it. The tool list stays the same
size whether the lab has one instrument or fifty, and an agent that has learned
this surface has learned every LabBench installation.

The tools fall into five groups: find out what is here, look at it, act on it,
supervise long work, and audit what happened.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.capability import Command, Feature
from ..core.device import DeviceState
from ..core.errors import ValidationError
from ..protocol.router import Router, RpcContext
from .schema import ToolSpec, emit


def register_tools(router: Router, gateway: Any) -> None:
    """Attach every gateway method to a router."""

    # -- discovery --------------------------------------------------------

    @router.method("lab.describe")
    async def lab_describe() -> dict[str, Any]:
        """What this laboratory contains, and what autonomy you have been granted."""
        return gateway.describe()

    @router.method("lab.find")
    async def lab_find(
        feature: str | None = None,
        kind: str | None = None,
        state: str | None = None,
        simulated: bool | None = None,
    ) -> dict[str, Any]:
        """Find devices by capability rather than by name.

        Asking for a feature instead of a model number is what keeps an agent
        vendor-agnostic: anything implementing MotionControl can be driven the
        same way.
        """
        wanted_state = DeviceState(state) if state else None
        matches = gateway.devices.find(kind=kind, feature=feature, state=wanted_state)
        if simulated is not None:
            matches = [d for d in matches if d.descriptor.simulated is simulated]
        return {
            "query": {"feature": feature, "kind": kind, "state": state, "simulated": simulated},
            "matches": [
                {
                    "id": d.id, "kind": d.descriptor.kind, "state": d.state.value,
                    "simulated": d.descriptor.simulated,
                    "display_name": d.descriptor.display_name,
                    "features": sorted(d.features()),
                }
                for d in matches
            ],
        }

    @router.method("device.describe")
    async def device_describe(
        device: str, feature: str | None = None, include_schemas: bool = True
    ) -> dict[str, Any]:
        """Full capability model for one device: properties, commands, schemas.

        This is the enumeration step. Everything an agent needs to construct a
        valid `device.invoke` call is here, including units, constraints,
        hazard class, reversibility and preconditions.
        """
        dev = gateway.device(device)
        features = dev.features()
        if feature is not None:
            if feature not in features:
                raise ValidationError(
                    f"device {device!r} has no feature {feature!r}",
                    available=sorted(features),
                )
            features = {feature: features[feature]}
        return {
            "device": device,
            "descriptor": dev.descriptor.model_dump(mode="json"),
            "state": dev.state.value,
            "fault": dev.fault,
            "features": [
                _describe_feature(f, include_schemas=include_schemas)
                for f in features.values()
            ],
        }

    # -- observation ------------------------------------------------------

    @router.method("device.read")
    async def device_read(
        device: str, feature: str | None = None, property: str | None = None
    ) -> dict[str, Any]:
        """Read one property, or snapshot every readable property.

        Reads are never gated: observing an instrument cannot change it.
        """
        dev = gateway.device(device)
        if property is not None:
            if feature is None:
                raise ValidationError("reading a single property needs its feature too")
            sample = await dev.read(feature, property)
            return sample.model_dump(mode="json")
        return {"device": device, "properties": await dev.read_all(feature)}

    @router.method("device.write")
    async def device_write(
        device: str, feature: str, property: str, value: Any, ctx: RpcContext
    ) -> dict[str, Any]:
        """Set a writable property."""
        dev = gateway.device(device)
        await dev.write(feature, property, value)
        gateway.ledger.log(
            "property_write", actor=ctx.actor, device_id=device,
            feature=feature, command=property, payload={"value": value},
        )
        sample = await dev.read(feature, property)
        return sample.model_dump(mode="json")

    @router.method("device.simulate")
    async def device_simulate(
        device: str, feature: str, command: str, args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Predict what a command would do, without doing it.

        Free to call and always safe. `fidelity` says how much to trust the
        answer: "none" means the driver has no model and the prediction is
        worthless, which is itself worth knowing before acting.
        """
        dev = gateway.device(device)
        result = await dev.simulate(feature, command, args or {})
        return {"device": device, "action": f"{feature}.{command}",
                **result.model_dump(mode="json")}

    # -- action -----------------------------------------------------------

    @router.method("device.invoke")
    async def device_invoke(
        device: str,
        feature: str,
        command: str,
        args: dict[str, Any] | None = None,
        reason: str = "",
        approval_id: str | None = None,
        run_id: str | None = None,
        wait_for_approval_s: float | None = None,
        *,
        ctx: RpcContext,
    ) -> dict[str, Any]:
        """Run a command on an instrument. Every physical action goes through here.

        `reason` is not decoration. It is written to the provenance ledger and
        shown to the human who may be asked to approve the action, and for a
        hazardous command it is the difference between a signature and a
        refusal.

        A long-running command returns a job handle immediately; poll
        `job.status` or subscribe to job updates. A command needing a human
        signature raises with an `approval_id` to be granted and then replayed.
        """
        return await gateway.invoke(
            device, feature, command, args or {},
            actor=ctx.actor, reason=reason, approval_id=approval_id,
            run_id=run_id, wait_for_approval=wait_for_approval_s,
            progress=lambda fraction, message: ctx.progress(
                fraction, message, device=device, action=f"{feature}.{command}"
            ),
        )

    @router.method("device.connect")
    async def device_connect(device: str) -> dict[str, Any]:
        """Bring a device online."""
        dev = gateway.device(device)
        await dev.connect()
        return {"device": device, "state": dev.state.value}

    @router.method("device.initialize")
    async def device_initialize(device: str, ctx: RpcContext) -> dict[str, Any]:
        """Home or calibrate a device. Many commands are gated behind this."""
        dev = gateway.device(device)
        await dev.initialize()
        gateway.ledger.log("run_step", actor=ctx.actor, device_id=device,
                           command="initialize", payload={"state": dev.state.value})
        return {"device": device, "state": dev.state.value,
                "properties": await dev.read_all()}

    @router.method("device.clear_fault")
    async def device_clear_fault(device: str, reason: str, ctx: RpcContext) -> dict[str, Any]:
        """Clear a fault after the cause has been dealt with.

        `reason` is required. Clearing a fault without recording why is how a
        recurring hardware problem becomes invisible.
        """
        if not reason.strip():
            raise ValidationError(
                "clearing a fault requires a reason describing what was fixed"
            )
        dev = gateway.device(device)
        previous = dev.fault
        await dev.clear_fault()
        gateway.ledger.log(
            "note", actor=ctx.actor, device_id=device, reason=reason,
            payload={"cleared_fault": previous},
        )
        return {"device": device, "state": dev.state.value, "cleared": previous}

    @router.method("estop")
    async def estop(reason: str = "operator e-stop", ctx: RpcContext = None) -> dict[str, Any]:
        """Stop every device immediately. Never gated and never refused."""
        return await gateway.estop(reason, actor=ctx.actor if ctx else "agent")

    # -- long-running work ------------------------------------------------

    @router.method("job.status")
    async def job_status(job_id: str, include_history: bool = False) -> dict[str, Any]:
        """Progress, result or error for one job."""
        job = gateway.jobs.get(job_id)
        out = job.summary()
        if include_history:
            out["history"] = [h.model_dump(mode="json") for h in job.history]
            out["artifacts"] = [a.model_dump(mode="json") for a in job.artifacts]
        return out

    @router.method("job.list")
    async def job_list(
        status: str | None = None, device: str | None = None,
        run_id: str | None = None, limit: int = 50,
    ) -> dict[str, Any]:
        """Recent jobs, newest first."""
        from ..core.jobs import JobStatus

        jobs = gateway.jobs.list(
            status=JobStatus(status) if status else None,
            device_id=device, run_id=run_id, limit=limit,
        )
        return {"jobs": [j.summary() for j in jobs], "count": len(jobs)}

    @router.method("job.cancel")
    async def job_cancel(job_id: str, reason: str = "agent requested") -> dict[str, Any]:
        """Ask a job to stop.

        Cooperative: the driver is told to stop at its next safe point, so the
        instrument is parked rather than abandoned mid-move. Use `job.kill`
        only for a driver that ignores this.
        """
        job = await gateway.jobs.cancel(job_id, reason=reason)
        return job.summary()

    @router.method("job.kill")
    async def job_kill(job_id: str) -> dict[str, Any]:
        """Force a job's task to stop. Leaves hardware in an unknown state."""
        job = await gateway.jobs.kill(job_id)
        return job.summary()

    @router.method("job.wait")
    async def job_wait(job_id: str, timeout_s: float = 60.0) -> dict[str, Any]:
        """Block until a job finishes or the timeout expires.

        Returns the job either way; check `status` rather than assuming the
        wait succeeded.
        """
        job = await gateway.jobs.wait(job_id, timeout_s)
        return job.summary()

    @router.method("job.artifacts")
    async def job_artifacts(job_id: str) -> dict[str, Any]:
        """Files a job produced: images, tables, traces.

        References, never contents. A tile scan can produce hundreds of
        megabytes, and putting that in a tool result would destroy the context
        window it was meant to inform.
        """
        job = gateway.jobs.get(job_id)
        return {
            "job_id": job_id,
            "artifacts": [a.model_dump(mode="json") for a in job.artifacts],
            "count": len(job.artifacts),
        }

    # -- human approval ---------------------------------------------------

    @router.method("approval.list")
    async def approval_list() -> dict[str, Any]:
        """Actions waiting on a human signature."""
        pending = gateway.approvals.pending()
        return {"pending": [r.summary() for r in pending], "count": len(pending)}

    @router.method("approval.get")
    async def approval_get(approval_id: str) -> dict[str, Any]:
        """One approval, pending or resolved."""
        return gateway.approvals.get(approval_id).summary()

    @router.method("approval.grant")
    async def approval_grant(
        approval_id: str, approver: str, reason: str = ""
    ) -> dict[str, Any]:
        """Sign off on a pending action. For humans, not agents.

        `approver` must identify a person. The grant is bound to the exact
        arguments that were shown, so it cannot be spent on a different call.
        """
        request = await gateway.approvals.grant(
            approval_id, approver=approver, reason=reason
        )
        return request.summary() | {
            "next_step": f"retry the original call passing approval_id={approval_id!r}"
        }

    @router.method("approval.deny")
    async def approval_deny(
        approval_id: str, approver: str = "operator", reason: str = ""
    ) -> dict[str, Any]:
        """Refuse a pending action."""
        request = await gateway.approvals.deny(
            approval_id, approver=approver, reason=reason
        )
        return request.summary()

    # -- provenance -------------------------------------------------------

    @router.method("ledger.query")
    async def ledger_query(
        run_id: str | None = None, device: str | None = None,
        kind: str | None = None, since_s: float | None = None, limit: int = 100,
    ) -> dict[str, Any]:
        """Read the audit trail."""
        since = time.time() - since_s if since_s else None
        records = gateway.ledger.query(
            run_id=run_id, device_id=device, kind=kind, since=since, limit=limit
        )
        return {
            "records": [r.model_dump(mode="json") for r in records],
            "count": len(records),
        }

    @router.method("ledger.verify")
    async def ledger_verify() -> dict[str, Any]:
        """Re-walk the hash chain and report the first break, if any.

        This is what an auditor runs, and what answers "did anything touch this
        dataset after the fact".
        """
        return gateway.ledger.verify()

    @router.method("ledger.note")
    async def ledger_note(
        note: str, run_id: str | None = None, device: str | None = None,
        ctx: RpcContext = None,
    ) -> dict[str, Any]:
        """Write an observation into the permanent record.

        For the reasoning that would otherwise be lost: why a parameter was
        chosen, what looked wrong, what to check next time.
        """
        record = gateway.ledger.log(
            "note", actor=ctx.actor if ctx else "agent",
            device_id=device, run_id=run_id, reason=note, payload={},
        )
        return {"seq": record.seq, "id": record.id, "timestamp": record.timestamp}

    # -- memory -------------------------------------------------------------

    @router.method("memory.write")
    async def memory_write(
        content: str, title: str = "", kind: str = "note", tags: list[str] | None = None,
        run_id: str | None = None, device: str | None = None, store: str | None = None,
        ctx: RpcContext = None,
    ) -> dict[str, Any]:
        """Write a durable, searchable note or document.

        Unlike `ledger.note` -- a timestamped, immutable entry in the audit
        trail -- this is meant to be found again next week by `memory.search`.
        Write what is worth keeping: an SOP, a calibration offset, what a
        field of the plate looked like and why it was excluded.
        """
        from ..memory.store import MemoryRecord

        record = MemoryRecord(
            content=content, title=title, kind=kind, tags=tags or [], run_id=run_id,
            device_id=device, actor=ctx.actor if ctx else "agent",
        )
        saved = await gateway.memory.store(store).write(record)
        return saved.model_dump(mode="json")

    @router.method("memory.search")
    async def memory_search(
        query: str = "", kind: str | None = None, tags: list[str] | None = None,
        run_id: str | None = None, device: str | None = None, store: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Find notes and documents relevant to `query`. Omit it to browse by filter alone."""
        records = await gateway.memory.store(store).search(
            query, kind=kind, tags=tags, run_id=run_id, device_id=device, limit=limit,
        )
        return {"records": [r.summary() for r in records], "count": len(records)}

    @router.method("memory.get")
    async def memory_get(id: str, store: str | None = None) -> dict[str, Any]:
        """Read one memory record in full, including its whole content."""
        record = await gateway.memory.store(store).get(id)
        return record.model_dump(mode="json")

    @router.method("memory.list")
    async def memory_list(
        kind: str | None = None, run_id: str | None = None, store: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Recent memory records, newest first."""
        records = await gateway.memory.store(store).list(kind=kind, run_id=run_id, limit=limit)
        return {"records": [r.summary() for r in records], "count": len(records)}

    @router.method("memory.delete")
    async def memory_delete(
        id: str, reason: str, store: str | None = None, ctx: RpcContext = None,
    ) -> dict[str, Any]:
        """Remove a memory record. Requires a reason, which is written to the ledger."""
        if not reason.strip():
            raise ValidationError("deleting a memory record requires a reason")
        await gateway.memory.store(store).delete(id)
        gateway.ledger.log(
            "note", actor=ctx.actor if ctx else "agent", reason=reason,
            payload={"deleted_memory_id": id},
        )
        return {"deleted": True, "id": id}

    # -- experiments ----------------------------------------------------------

    @router.method("experiment.validate")
    async def experiment_validate(protocol: dict[str, Any]) -> dict[str, Any]:
        """Check a protocol's steps against the current capability model, without running it.

        Catches an unknown device, a wrong argument name, and any literal
        value outside its declared envelope, before the first step executes.
        A `${...}` reference to a later step's result cannot be checked here;
        see `experiment.dry_run` for what that needs.
        """
        from ..experiment import Protocol

        parsed = Protocol.model_validate(protocol)
        problems = parsed.validate_against(gateway)
        return {"valid": not problems, "problems": problems, "steps": len(parsed.steps)}

    @router.method("experiment.dry_run")
    async def experiment_dry_run(protocol: dict[str, Any]) -> dict[str, Any]:
        """Simulate every step in order, without touching hardware.

        The protocol-level sibling of `device.simulate`: resolves `${...}`
        references against *simulated* results, so it also catches a
        downstream step whose predicted state would fail an envelope check.
        """
        from ..experiment import Protocol, dry_run_protocol

        return await dry_run_protocol(Protocol.model_validate(protocol), gateway)

    @router.method("experiment.start")
    async def experiment_start(
        protocol: dict[str, Any], variables: dict[str, Any] | None = None,
        ctx: RpcContext = None,
    ) -> dict[str, Any]:
        """Begin executing a protocol. Long-running: returns a run handle immediately.

        Every step still crosses the safety kernel exactly as a direct
        `device.invoke` would. A step whose hazard needs a human signature
        parks the whole run on status `awaiting_approval` rather than failing
        it; a human calls `approval.grant`, then `experiment.resume`.
        """
        from ..core.errors import ValidationError as _ValidationError
        from ..experiment import Protocol

        parsed = Protocol.model_validate(protocol)
        problems = parsed.validate_against(gateway)
        if problems:
            raise _ValidationError(
                f"protocol has {len(problems)} problem(s); call experiment.validate first",
                problems=problems,
            )
        protocol_id = gateway.experiments.define(parsed)
        run = gateway.experiments.start(
            parsed, protocol_id=protocol_id, variables=variables or {},
            actor=ctx.actor if ctx else "agent",
        )
        return {"run": run.summary(), "protocol_id": protocol_id}

    @router.method("experiment.status")
    async def experiment_status(run_id: str, include_steps: bool = True) -> dict[str, Any]:
        """Progress, per-step results, and any pending approval for one run."""
        return gateway.experiments.get(run_id).summary(include_steps=include_steps)

    @router.method("experiment.list")
    async def experiment_list(status: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Recent runs, newest first."""
        from ..experiment import RunStatus

        runs = gateway.experiments.list(
            status=RunStatus(status) if status else None, limit=limit,
        )
        return {"runs": [r.summary(include_steps=False) for r in runs], "count": len(runs)}

    @router.method("experiment.cancel")
    async def experiment_cancel(run_id: str, reason: str = "agent requested") -> dict[str, Any]:
        """Stop a run. The in-flight step is asked to finish or park; nothing new starts."""
        run = await gateway.experiments.cancel(run_id, reason=reason)
        return run.summary()

    @router.method("experiment.resume")
    async def experiment_resume(run_id: str) -> dict[str, Any]:
        """Continue a run parked on `awaiting_approval`, after a human has answered.

        Retries the step that asked for a signature with the same approval id,
        so it succeeds only if that exact call -- the one the human actually
        saw -- was the one granted.
        """
        run = gateway.experiments.get(run_id)
        protocol = gateway.experiments.get_protocol(run.protocol_id) if run.protocol_id else None
        if protocol is None:
            raise ValidationError(f"run {run_id!r} has no stored protocol to resume with")
        resumed = gateway.experiments.resume(run_id, protocol)
        return resumed.summary()

    @router.method("experiment.replay")
    async def experiment_replay(run_id: str) -> dict[str, Any]:
        """Reconstruct exactly what a finished run did, from the ledger alone.

        Never re-executes anything: see the module docstring in
        `experiment/replay.py` for why that is the point, not an omission.
        """
        from ..experiment import replay_run

        return replay_run(gateway.ledger, run_id).model_dump(mode="json")

    # -- self-description -------------------------------------------------

    @router.method("tools.schema")
    async def tools_schema(dialect: str = "jsonschema", strict: bool = False) -> Any:
        """This gateway's own tool definitions, in your model's dialect.

        Supported: anthropic, openai, openai-responses, gemini, jsonschema,
        openapi. This is how a client wires LabBench into an agent loop without
        anyone hand-writing schemas: fetch them, hand them to the model.
        """
        specs = tool_specs(gateway)
        return emit(
            specs, dialect, strict=strict,
            title=f"LabBench - {gateway.config.name}",
        )

    @router.method("tools.list")
    async def tools_list() -> dict[str, Any]:
        """Every method this gateway exposes, with a one-line summary."""
        return {"methods": router.methods, "count": len(router.methods)}


def _describe_feature(feature: Feature, *, include_schemas: bool = True) -> dict[str, Any]:
    """Project one feature into the form an agent reads."""
    return {
        "identifier": feature.identifier,
        "display_name": feature.display_name,
        "description": feature.description,
        "version": feature.version,
        "fqid": feature.fqid,
        "properties": [
            {
                "name": p.name,
                "description": p.description,
                "unit": p.schema_.unit,
                "type": p.schema_.type,
                "writable": p.writable,
                "observable": p.observable,
                **({"schema": p.schema_.to_json_schema()} if include_schemas else {}),
            }
            for p in feature.properties
        ],
        "commands": [_describe_command(c, include_schemas=include_schemas)
                     for c in feature.commands],
        "events": [
            {"name": e.name, "description": e.description, "severity": e.severity}
            for e in feature.events
        ],
    }


def _describe_command(command: Command, *, include_schemas: bool = True) -> dict[str, Any]:
    """Everything an agent needs to decide whether and how to call a command."""
    out: dict[str, Any] = {
        "name": command.name,
        "description": command.description,
        "hazard": command.hazard.value,
        "reversibility": command.reversibility.value,
        "observable": command.observable,
        "duration_estimate_s": command.duration_estimate_s,
        "exclusive": command.exclusive,
        "simulatable": command.simulatable,
    }
    if command.inverse:
        out["inverse"] = command.inverse
    if command.tags:
        out["tags"] = sorted(command.tags)
    if command.preconditions:
        out["preconditions"] = [
            {
                "property": p.property, "operator": p.operator, "value": p.value,
                "message": p.message,
            }
            for p in command.preconditions
        ]
    if include_schemas:
        out["input_schema"] = command.input_schema()
        if command.returns:
            out["returns"] = {
                r.name: r.to_json_schema() for r in command.returns
            }
    return out


def tool_specs(gateway: Any) -> list[ToolSpec]:
    """Neutral tool descriptions, ready to be projected into any AI dialect.

    Hand-written rather than reflected off the router, because a schema an
    agent will plan against deserves prose written for it -- describing what
    the tool is *for*, not merely what arguments it takes.
    """
    devices = sorted(gateway.devices.all())
    device_hint = f" Known devices: {', '.join(devices)}." if devices else ""

    def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        }

    string = {"type": "string"}
    device_arg = {"type": "string", "description": f"Device id.{device_hint}"}

    return [
        ToolSpec(
            name="lab.describe",
            description="List every instrument in this laboratory with its state, and report "
                        "the autonomy level you have been granted. Call this first.",
            parameters=obj({}),
            read_only=True, idempotent=True, hazard="none", tags=["discovery"],
        ),
        ToolSpec(
            name="lab.find",
            description="Find instruments by capability rather than by name, e.g. every device "
                        "implementing MotionControl. Asking for a feature keeps you independent "
                        "of vendor and model.",
            parameters=obj({
                "feature": {"type": "string", "description": "Feature identifier, e.g. 'Camera'."},
                "kind": {"type": "string", "description": "e.g. 'microscope', 'incubator'."},
                "state": {"type": "string", "enum": [s.value for s in DeviceState]},
                "simulated": {"type": "boolean", "description": "Filter simulated vs real."},
            }),
            read_only=True, idempotent=True, hazard="none", tags=["discovery"],
        ),
        ToolSpec(
            name="device.describe",
            description="Full capability model for one instrument: every feature, property and "
                        "command, with JSON Schema, units, constraints, hazard class, "
                        "reversibility and preconditions. This is how you learn what arguments "
                        "device.invoke will accept.",
            parameters=obj({
                "device": device_arg,
                "feature": {"type": "string", "description": "Restrict to one feature."},
                "include_schemas": {"type": "boolean", "default": True},
            }, ["device"]),
            read_only=True, idempotent=True, hazard="none", tags=["discovery"],
        ),
        ToolSpec(
            name="device.read",
            description="Read instrument state. Omit 'property' to snapshot everything readable. "
                        "Reads are never gated and never change anything.",
            parameters=obj({
                "device": device_arg,
                "feature": string,
                "property": string,
            }, ["device"]),
            read_only=True, idempotent=True, hazard="none", tags=["observation"],
        ),
        ToolSpec(
            name="device.simulate",
            description="Predict what a command would do WITHOUT doing it. Free, always safe, "
                        "and the right thing to call before anything irreversible. Check "
                        "'fidelity': 'none' means the driver has no model and the prediction "
                        "is worthless.",
            parameters=obj({
                "device": device_arg,
                "feature": string,
                "command": string,
                "args": {"type": "object", "description": "Arguments you intend to pass."},
            }, ["device", "feature", "command"]),
            read_only=True, hazard="none", tags=["safety", "planning"],
        ),
        ToolSpec(
            name="device.invoke",
            description="Run a command on an instrument. Every physical action goes through "
                        "this tool. Give a real 'reason': it is written to the permanent record "
                        "and shown to whichever human may be asked to approve the action. "
                        "Long-running commands return a job handle rather than blocking; "
                        "commands above your autonomy level raise with an approval_id that a "
                        "human must grant before you retry.",
            parameters=obj({
                "device": device_arg,
                "feature": string,
                "command": string,
                "args": {"type": "object", "description": "Arguments, per device.describe."},
                "reason": {"type": "string",
                           "description": "Why you are doing this, in one sentence."},
                "approval_id": {"type": "string",
                                "description": "A granted approval from a previous attempt."},
                "run_id": {"type": "string", "description": "Groups actions into one experiment."},
            }, ["device", "feature", "command"]),
            hazard="varies", destructive=True, tags=["action"],
        ),
        ToolSpec(
            name="device.initialize",
            description="Home or calibrate an instrument. Many commands refuse to run until "
                        "this has happened, because before it the controller's zero is wherever "
                        "it powered up.",
            parameters=obj({"device": device_arg}, ["device"]),
            hazard="motion", tags=["action"],
        ),
        ToolSpec(
            name="job.status",
            description="Progress, result or error for a long-running command.",
            parameters=obj({
                "job_id": string,
                "include_history": {"type": "boolean", "default": False},
            }, ["job_id"]),
            read_only=True, idempotent=True, hazard="none", tags=["jobs"],
        ),
        ToolSpec(
            name="job.list",
            description="Recent jobs, newest first.",
            parameters=obj({
                "status": {"type": "string",
                           "enum": ["pending", "running", "succeeded", "failed", "cancelled"]},
                "device": string, "run_id": string,
                "limit": {"type": "integer", "default": 50},
            }),
            read_only=True, idempotent=True, hazard="none", tags=["jobs"],
        ),
        ToolSpec(
            name="job.cancel",
            description="Ask a running job to stop at its next safe point, so the instrument "
                        "is parked rather than abandoned mid-move.",
            parameters=obj({"job_id": string, "reason": string}, ["job_id"]),
            hazard="benign", tags=["jobs"],
        ),
        ToolSpec(
            name="job.wait",
            description="Block until a job finishes or the timeout expires. Returns the job "
                        "either way, so check 'status' rather than assuming it finished.",
            parameters=obj({
                "job_id": string,
                "timeout_s": {"type": "number", "default": 60},
            }, ["job_id"]),
            read_only=True, hazard="none", tags=["jobs"],
        ),
        ToolSpec(
            name="job.artifacts",
            description="Files a job produced - images, tables, traces - as references with "
                        "metadata, never as inline content.",
            parameters=obj({"job_id": string}, ["job_id"]),
            read_only=True, idempotent=True, hazard="none", tags=["jobs", "data"],
        ),
        ToolSpec(
            name="approval.list",
            description="Actions currently waiting on a human signature, including anything "
                        "you requested.",
            parameters=obj({}),
            read_only=True, idempotent=True, hazard="none", tags=["safety"],
        ),
        ToolSpec(
            name="ledger.query",
            description="Read the append-only audit trail: what was requested, what the safety "
                        "kernel decided, what happened, and who approved it.",
            parameters=obj({
                "run_id": string, "device": string, "kind": string,
                "since_s": {"type": "number", "description": "Look back this many seconds."},
                "limit": {"type": "integer", "default": 100},
            }),
            read_only=True, idempotent=True, hazard="none", tags=["provenance"],
        ),
        ToolSpec(
            name="ledger.note",
            description="Write an observation into the permanent record - why you chose a "
                        "parameter, what looked wrong, what to check next time. This is the "
                        "reasoning that would otherwise be lost when your context ends.",
            parameters=obj({
                "note": string, "run_id": string, "device": string,
            }, ["note"]),
            hazard="none", tags=["provenance"],
        ),
        ToolSpec(
            name="memory.write",
            description="Save a durable, searchable note or document - unlike ledger.note, "
                        "this is meant to be found again later by memory.search. Use it for "
                        "things worth keeping: an SOP, a calibration offset, a conclusion.",
            parameters=obj({
                "content": string, "title": string,
                "kind": {"type": "string", "description": "Free-form, e.g. 'note', 'sop'."},
                "tags": {"type": "array", "items": string},
                "run_id": string, "device": string,
            }, ["content"]),
            hazard="none", tags=["memory"],
        ),
        ToolSpec(
            name="memory.search",
            description="Find notes and documents relevant to a query. Call this before asking "
                        "an operator something the lab may already have written down.",
            parameters=obj({
                "query": string,
                "kind": string, "tags": {"type": "array", "items": string},
                "run_id": string, "device": string,
                "limit": {"type": "integer", "default": 20},
            }),
            read_only=True, idempotent=True, hazard="none", tags=["memory"],
        ),
        ToolSpec(
            name="experiment.validate",
            description="Check a multi-step protocol against the current capability model "
                        "before running it: unknown devices, wrong argument names, "
                        "out-of-envelope literal values. Call this before experiment.start.",
            parameters=obj({
                "protocol": {"type": "object",
                             "description": "name, variables, and a list of steps "
                                            "(device, feature, command, args, reason)."},
            }, ["protocol"]),
            read_only=True, idempotent=True, hazard="none", tags=["experiment", "planning"],
        ),
        ToolSpec(
            name="experiment.start",
            description="Begin executing a validated multi-step protocol. Long-running: "
                        "returns a run handle immediately. A step whose hazard needs a human "
                        "signature parks the whole run rather than failing it; check "
                        "experiment.status and resume it once approved.",
            parameters=obj({
                "protocol": {"type": "object"},
                "variables": {"type": "object",
                              "description": "Overrides for the protocol's declared variables."},
            }, ["protocol"]),
            destructive=True, hazard="varies", tags=["experiment", "action"],
        ),
        ToolSpec(
            name="experiment.status",
            description="Progress, per-step results, and any pending approval for one run.",
            parameters=obj({
                "run_id": string,
                "include_steps": {"type": "boolean", "default": True},
            }, ["run_id"]),
            read_only=True, idempotent=True, hazard="none", tags=["experiment"],
        ),
        ToolSpec(
            name="experiment.replay",
            description="Reconstruct exactly what a finished run did, in order, from the "
                        "tamper-evident ledger - what was decided, what ran, who approved what.",
            parameters=obj({"run_id": string}, ["run_id"]),
            read_only=True, idempotent=True, hazard="none", tags=["experiment", "provenance"],
        ),
        ToolSpec(
            name="estop",
            description="EMERGENCY STOP. Halts every instrument immediately and cancels every "
                        "running job. Never gated, never queued, never refused. Use it the "
                        "moment something is wrong; recovering from an unnecessary stop is far "
                        "cheaper than not stopping.",
            parameters=obj({"reason": string}),
            hazard="none", tags=["safety"],
        ),
    ]
