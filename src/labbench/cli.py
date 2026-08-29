"""Operator entry point.

Argparse rather than a CLI framework, for the same reason the HTTP server is
hand-written: this is the program someone runs at two in the morning when an
experiment is misbehaving, and it should depend on as little as possible.

    labbench serve   --config lab.yaml [--transport stdio|http|ws|all]
    labbench doctor  [--config lab.yaml]
    labbench devices --config lab.yaml
    labbench tools   [--dialect anthropic|openai|gemini|jsonschema|openapi]
    labbench ledger     verify|query
    labbench call       --config lab.yaml <method> [key=value ...]
    labbench experiment run --config lab.yaml protocol.yaml [var=value ...]
    labbench campaign   run --config lab.yaml campaign.yaml
    labbench eval       list
    labbench eval       run --dialect anthropic [--task ID ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

#: Which optional extra provides each driver. The driver name and the extra
#: are not always the same word, and telling someone to install a
#: non-existent extra is worse than saying nothing.
_EXTRA_FOR_DRIVER = {
    "micromanager": "micromanager",
    "scpi": "scpi",
    "sila2": "sila2",
    "opcua_lads": "opcua",
    "wot": "http",
    "opentrons": "http",
}


def _setup_logging(verbosity: int, *, to_stderr: bool = True) -> None:
    level = logging.WARNING if verbosity == 0 else (
        logging.INFO if verbosity == 1 else logging.DEBUG
    )
    # Always stderr. On the stdio transport stdout carries protocol, and a log
    # line written there corrupts the session.
    logging.basicConfig(
        level=level, format=LOG_FORMAT,
        stream=sys.stderr if to_stderr else sys.stdout,
    )


def _load_config(path: str | None) -> Any:
    from .core.registry import LabConfig

    if path is None:
        return LabConfig()
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise SystemExit(f"labbench: no such config file: {config_path}")
    try:
        return LabConfig.load(config_path)
    except Exception as exc:  # noqa: BLE001 - any failure here becomes a clean operator-facing message
        raise SystemExit(f"labbench: {config_path} is not a valid lab config: {exc}") from None


def _emit(payload: Any, *, compact: bool = False) -> None:
    print(json.dumps(payload, indent=None if compact else 2, default=str))


# -- serve -----------------------------------------------------------------


async def _serve(args: argparse.Namespace) -> int:
    from .gateway import Gateway
    from .protocol.http import HttpServer
    from .protocol.stdio import StdioServer
    from .protocol.websocket import WebSocketEndpoint

    config = _load_config(args.config)
    gateway = Gateway(config, data_dir=args.data_dir)
    boot = await gateway.start()

    for device_id, state in boot["connected"].items():
        (log.warning if state.startswith("error") else log.info)(
            "device %s: %s", device_id, state
        )
    for device_id, problem in boot["unavailable"].items():
        log.warning("device %s unavailable: %s", device_id, problem)

    transports = (
        ["stdio", "http", "ws"] if args.transport == "all" else [args.transport]
    )
    http_server: HttpServer | None = None
    tasks: list[asyncio.Task[Any]] = []

    if "http" in transports or "ws" in transports:
        token = args.token or os.environ.get("LABBENCH_TOKEN")
        http_server = HttpServer(
            gateway.router, host=args.host, port=args.port, token=token,
            server_name=f"labbench/{__version__}",
        )
        if "ws" in transports:
            endpoint = WebSocketEndpoint(gateway.router)
            http_server.set_upgrade_handler(endpoint.handle_upgrade)
            gateway.add_event_sink(endpoint.broadcast)
        gateway.add_event_sink(http_server.broadcast)
        await http_server.start()
        print(
            f"labbench: serving {config.name} on {http_server.url}\n"
            f"  POST {http_server.url}/rpc      JSON-RPC 2.0\n"
            f"  GET  {http_server.url}/events   live event stream (SSE)\n"
            f"  GET  {http_server.url}/healthz  liveness\n"
            f"  auth: {'bearer token' if token else 'none (loopback only)'}",
            file=sys.stderr,
        )
        tasks.append(asyncio.create_task(http_server.serve_forever()))

    if "stdio" in transports:
        from .protocol.stdio import stdin_is_usable

        if stdin_is_usable():
            print(f"labbench: serving {config.name} over stdio", file=sys.stderr)
            tasks.append(asyncio.create_task(StdioServer(gateway.router).serve()))
        elif args.transport == "stdio":
            # Explicitly asked for, and impossible: that is an error.
            raise SystemExit(
                "labbench: stdin is not a pipe, so the stdio transport cannot run. "
                "Use --transport http or --transport ws for a daemon."
            )
        else:
            # Part of --transport all: skip it and say so once, rather than
            # letting asyncio log a traceback nobody can act on.
            log.info("stdin is not a pipe; skipping the stdio transport")

    try:
        if not tasks:
            raise SystemExit("labbench: no transport selected")
        # Whichever finishes first ends the process: a closed stdin means the
        # parent went away, and there is no one left to serve.
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nlabbench: shutting down", file=sys.stderr)
    finally:
        if http_server is not None:
            await http_server.close()
        await gateway.close()
    return 0


# -- doctor ----------------------------------------------------------------


async def _doctor(args: argparse.Namespace) -> int:
    """Report what is installed, what is not, and what that costs you."""
    from .core.registry import DriverRegistry

    ok = True
    print(f"labbench {__version__}")
    print(f"  python      {sys.version.split()[0]} at {sys.executable}")

    print("\nruntime dependencies")
    for module, purpose in (
        ("pydantic", "capability model and validation"),
        ("yaml", "lab configuration"),
        ("numpy", "simulated instrument physics"),
    ):
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            print(f"  [ok]   {module:12s} {version:10s} {purpose}")
        except ImportError:
            ok = False
            print(f"  [MISS] {module:12s} {'':10s} {purpose}")

    print("\ndrivers")
    catalog = DriverRegistry().catalog()
    for name in catalog["available"]:
        print(f"  [ok]   {name}")
    for name, problem in sorted(catalog["unavailable"].items()):
        extra = _EXTRA_FOR_DRIVER.get(name, name.split("_")[0])
        print(f"  [--]   {name:18s} {problem[:60]}")
        print(f"         install with: pip install 'labbench[{extra}]'")
    if not catalog["available"]:
        ok = False
        print("  no drivers available at all - the package may be installed incorrectly")

    print("\noptional instrument libraries")
    for module, extra in (
        ("pyvisa", "scpi"), ("pymmcore_plus", "micromanager"), ("sila2", "sila2"),
        ("asyncua", "opcua"), ("zeroconf", "discovery"), ("httpx", "http"),
        ("tifffile", "imaging"),
    ):
        try:
            __import__(module)
            print(f"  [ok]   {module}")
        except ImportError:
            print(f"  [--]   {module:16s} pip install 'labbench[{extra}]'")

    if args.config:
        print(f"\nconfiguration: {args.config}")
        config = _load_config(args.config)
        print(f"  lab         {config.name}")
        print(f"  devices     {len(config.devices)}")
        from .core.safety import SafetyPolicy

        policy = SafetyPolicy.model_validate(config.safety or {})
        print(f"  autonomy    {int(policy.autonomy)} ({policy.autonomy.name})")
        print(f"  ceiling     {policy.ceiling().value}")
        print(f"  rules       {len(policy.rules)}")
        if not policy.approve_irreversible:
            print("  [warn] approve_irreversible is false: irreversible actions will run "
                  "without a signature.")
            print("         This is only defensible when every device is simulated.")
        registry = DriverRegistry()
        for device in config.devices:
            try:
                registry.get(device.driver)
                print(f"  [ok]   {device.id:12s} -> {device.driver}")
            except Exception as exc:  # noqa: BLE001 - one bad driver must not stop the whole report
                ok = False
                print(f"  [FAIL] {device.id:12s} -> {device.driver}: {exc}")

    print("\n" + ("all good" if ok else "problems found - see [MISS]/[FAIL] above"))
    return 0 if ok else 1


# -- devices ---------------------------------------------------------------


async def _devices(args: argparse.Namespace) -> int:
    from .gateway import Gateway

    gateway = Gateway(_load_config(args.config), data_dir=args.data_dir)
    await gateway.start()
    try:
        description = gateway.describe()
        if args.json:
            _emit(description)
            return 0
        print(f"{description['lab']}  autonomy {description['autonomy']['level']} "
              f"({description['autonomy']['name']}), "
              f"ceiling {description['autonomy']['hazard_ceiling']}")
        for device in description["devices"]:
            flag = " [SIM]" if device["simulated"] else ""
            print(f"\n  {device['id']}{flag}  {device['display_name']}")
            print(f"    kind    {device['kind']}   state {device['state']}"
                  + (f"   FAULT: {device['fault']}" if device["fault"] else ""))
            for feature in device["features"]:
                detail = gateway.device(device["id"]).features()[feature]
                commands = ", ".join(c.name for c in detail.commands)
                print(f"    {feature:18s} {commands}")
        return 0
    finally:
        await gateway.close()


# -- tools -----------------------------------------------------------------


async def _tools(args: argparse.Namespace) -> int:
    from .bridge.schema import emit
    from .bridge.toolset import tool_specs
    from .gateway import Gateway

    gateway = Gateway(_load_config(args.config), data_dir=args.data_dir)
    if args.config:
        await gateway.start()
    try:
        _emit(emit(tool_specs(gateway), args.dialect, strict=args.strict))
        return 0
    finally:
        await gateway.close()


# -- ledger ----------------------------------------------------------------


async def _ledger(args: argparse.Namespace) -> int:
    from .core.provenance import Ledger

    path = Path(args.path or Path(args.data_dir or "./labbench-data") / "provenance.sqlite")
    if not path.exists():
        raise SystemExit(f"labbench: no ledger at {path}")
    ledger = Ledger(path)
    try:
        if args.ledger_command == "verify":
            result = ledger.verify()
            _emit(result)
            if not result["valid"]:
                print(
                    f"\nTHE CHAIN IS BROKEN at record {result['broken_at']}: "
                    f"{result['reason']}\nRecords at or after this point cannot be trusted.",
                    file=sys.stderr,
                )
                return 2
            print(f"\nchain intact across {result['records']} records", file=sys.stderr)
            return 0
        records = ledger.query(
            run_id=args.run, device_id=args.device, kind=args.kind, limit=args.limit
        )
        if args.json:
            _emit([r.model_dump(mode="json") for r in records])
            return 0
        for record in records:
            import time as _time

            stamp = _time.strftime("%H:%M:%S", _time.localtime(record.timestamp))
            target = ".".join(x for x in (record.device_id, record.feature, record.command) if x)
            print(f"  {record.seq:>5} {stamp} {record.kind:<16} {record.actor:<18} {target}"
                  + (f"  // {record.reason}" if record.reason else ""))
        return 0
    finally:
        ledger.close()


# -- call ------------------------------------------------------------------


async def _call(args: argparse.Namespace) -> int:
    """Invoke one gateway method from the shell. The debugging tool."""
    from .gateway import Gateway
    from .protocol.jsonrpc import Request
    from .protocol.router import RpcContext

    params: dict[str, Any] = {}
    for pair in args.params:
        if "=" not in pair:
            raise SystemExit(f"labbench: expected key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw  # a bare string is the common case

    gateway = Gateway(_load_config(args.config), data_dir=args.data_dir)
    await gateway.start()
    try:
        context = RpcContext(actor=args.actor, transport="cli")
        response = await gateway.router.dispatch(Request(args.method, params, id=1), context)
        assert response is not None
        payload = response.to_dict()
        if context.buffered:
            payload["notifications"] = context.buffered
        _emit(payload)
        return 1 if "error" in payload else 0
    finally:
        await gateway.close()


# -- experiment --------------------------------------------------------------


async def _experiment_run(args: argparse.Namespace) -> int:
    """Run a protocol to completion, prompting the operator for any approval.

    A one-shot command holds its own Gateway for the duration of the run --
    the same shape `labbench call` uses -- so a step that needs a human
    signature is answered right here, interactively, rather than requiring a
    second terminal talking to a long-lived server.
    """
    from .experiment import Protocol, RunStatus
    from .gateway import Gateway

    variables: dict[str, Any] = {}
    for pair in args.var:
        if "=" not in pair:
            raise SystemExit(f"labbench: expected key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            variables[key] = json.loads(raw)
        except json.JSONDecodeError:
            variables[key] = raw

    try:
        protocol = Protocol.load(args.protocol)
    except Exception as exc:  # noqa: BLE001 - any failure here becomes a clean operator-facing message
        raise SystemExit(f"labbench: {args.protocol} is not a valid protocol: {exc}") from None

    gateway = Gateway(_load_config(args.config), data_dir=args.data_dir)
    await gateway.start()
    try:
        problems = protocol.validate_against(gateway)
        if problems:
            print(f"labbench: {protocol.name} has {len(problems)} problem(s):", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        run = gateway.experiments.start(protocol, variables=variables, actor="human:cli")
        print(f"labbench: running {protocol.name!r} as {run.id}", file=sys.stderr)

        while True:
            current = gateway.experiments.get(run.id)
            if current.status.terminal:
                break
            if current.status is not RunStatus.AWAITING_APPROVAL:
                # Every branch of this loop must yield at least once per
                # iteration: `experiments.resume()` below only *schedules* the
                # background task that eventually moves the run out of
                # AWAITING_APPROVAL, and a loop that never awaits would starve
                # that task forever on this single-threaded event loop rather
                # than genuinely wait for it.
                await asyncio.sleep(0.1)
                continue

            pending_for_run = [p for p in gateway.approvals.pending() if p.run_id == run.id]
            if not pending_for_run:  # pragma: no cover - resolved by a racing caller
                await asyncio.sleep(0.1)
                continue
            pending = pending_for_run[0]
            print(f"\n{pending.prompt}\n", file=sys.stderr)
            # input()'s own prompt argument always writes to stdout, which
            # would land inside the JSON this command prints there on exit;
            # the prompt is written to stderr explicitly instead, exactly the
            # stdout-carries-only-the-result discipline `protocol/stdio.py`
            # enforces for the same reason.
            print("Grant this action? [y/N] ", end="", file=sys.stderr)
            answer = input().strip().lower()
            if answer == "y":
                print("Your name or id: ", end="", file=sys.stderr)
                approver = input().strip() or "human:cli"
                await gateway.approvals.grant(pending.id, approver=approver)
            else:
                await gateway.approvals.deny(pending.id, approver="human:cli", reason="declined at the CLI")
            gateway.experiments.resume(run.id, protocol)
            await asyncio.sleep(0.1)  # let the resumed run actually start before re-checking it

        finished = gateway.experiments.get(run.id)
        _emit(finished.summary())
        return 0 if finished.status is RunStatus.SUCCEEDED else 1
    finally:
        await gateway.close()


# -- campaign ----------------------------------------------------------------


async def _campaign_run(args: argparse.Namespace) -> int:
    """Run a closed-loop campaign to completion, prompting the operator for any approval.

    Same one-shot shape as `_experiment_run`: one Gateway held for the whole
    campaign, so a trial that needs a human signature is answered right here
    rather than requiring a second terminal talking to a long-lived server.
    """
    from .campaign import CampaignSpec, CampaignStatus
    from .gateway import Gateway

    try:
        spec = CampaignSpec.load(args.campaign)
    except Exception as exc:  # noqa: BLE001 - any failure here becomes a clean operator-facing message
        raise SystemExit(f"labbench: {args.campaign} is not a valid campaign: {exc}") from None

    gateway = Gateway(_load_config(args.config), data_dir=args.data_dir)
    await gateway.start()
    try:
        problems = spec.validate_against(gateway)
        if problems:
            print(f"labbench: {spec.name} has {len(problems)} problem(s):", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        campaign_id = gateway.campaigns.define(spec)
        state = gateway.campaigns.start(campaign_id, actor="human:cli")
        print(f"labbench: running {spec.name!r} as {state.id} "
              f"(budget {spec.budget}, {len(spec.objectives)} objective(s))", file=sys.stderr)

        last_trial = -1
        while True:
            current = gateway.campaigns.get(state.id)
            if current.status.terminal:
                break
            if current.status is not CampaignStatus.AWAITING_APPROVAL:
                # See the matching comment in _experiment_run: every branch must
                # yield at least once, or campaign.resume()'s background task
                # never actually gets a turn on this single-threaded event loop.
                if current.trial != last_trial:
                    print(f"  trial {current.trial}/{spec.budget}...", file=sys.stderr)
                    last_trial = current.trial
                await asyncio.sleep(0.1)
                continue

            pending_for_run = [
                p for p in gateway.approvals.pending() if p.run_id == current.current_run_id
            ]
            if not pending_for_run:  # pragma: no cover - resolved by a racing caller
                await asyncio.sleep(0.1)
                continue
            pending = pending_for_run[0]
            print(f"\n{pending.prompt}\n", file=sys.stderr)
            print("Grant this action? [y/N] ", end="", file=sys.stderr)
            answer = input().strip().lower()
            if answer == "y":
                print("Your name or id: ", end="", file=sys.stderr)
                approver = input().strip() or "human:cli"
                await gateway.approvals.grant(pending.id, approver=approver)
            else:
                await gateway.approvals.deny(pending.id, approver="human:cli", reason="declined at the CLI")
            gateway.campaigns.resume(state.id)
            await asyncio.sleep(0.1)  # let the resumed campaign actually start before re-checking it

        finished = gateway.campaigns.get(state.id)
        best = gateway.campaigns.best(state.id)
        _emit({"campaign": finished.summary(), "best": best})
        return 0 if finished.status is CampaignStatus.SUCCEEDED else 1
    finally:
        await gateway.close()


# -- eval ----------------------------------------------------------------


async def _eval_list(args: argparse.Namespace) -> int:
    from .evals.tasks import all_tasks

    for task in all_tasks():
        print(f"{task.id:<24} {task.category:<12} {task.description}")
    return 0


async def _eval_run(args: argparse.Namespace) -> int:
    """Run one or more eval tasks against a real model and grade the result.

    Deliberately not part of `uv run pytest`: this spends a real API key and
    is not deterministic in the way a unit test must be. `tests/test_evals.py`
    covers the harness and every grader with `ScriptedPolicy` instead, which
    needs neither.
    """
    from .evals.harness import EvalRunner
    from .evals.policy import AnthropicPolicy, GeminiPolicy, OpenAIPolicy
    from .evals.report import render_table, summarize
    from .evals.tasks import all_tasks, get

    tasks = [get(task_id) for task_id in args.task] if args.task else all_tasks()

    def make_policy():
        if args.dialect == "anthropic":
            return AnthropicPolicy(args.model or "claude-sonnet-5")
        if args.dialect == "openai":
            return OpenAIPolicy(args.model or "gpt-4o", base_url=args.base_url)
        return GeminiPolicy(args.model or "gemini-2.0-flash")

    runner = EvalRunner(
        data_dir=Path(args.data_dir or "./labbench-data") / "evals", max_turns=args.max_turns,
    )
    results = []
    for task in tasks:
        print(f"labbench: running {task.id} ({args.dialect})...", file=sys.stderr)
        results.append(await runner.run(task, make_policy()))

    if args.json:
        _emit(summarize(results))
    else:
        print(render_table(results))
    return 0 if all(r.passed for r in results) else 1


# -- argument parsing ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labbench",
        description="Connect any AI agent to laboratory hardware.",
    )
    parser.add_argument("--version", action="version", version=f"labbench {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for info, -vv for debug. Always to stderr.")
    parser.add_argument("--data-dir", default=None,
                        help="Where the ledger and artifacts live.")

    # The same flags on every subcommand as well as the top level, so both
    # `labbench --data-dir X serve` and `labbench serve --data-dir X` work.
    # Requiring one particular order is the kind of papercut that makes a tool
    # feel hostile at two in the morning.
    #
    # The default here MUST be `argparse.SUPPRESS`, not a concrete value: a
    # subparser applies its own default for every argument it does not see on
    # the command line, and with a concrete default that unconditionally
    # overwrites whatever the top-level parser already set from
    # `labbench --data-dir X <command>` -- silently discarding it, with no
    # error, the moment the flag is not repeated after the subcommand too.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="count", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    common.add_argument("--data-dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", parents=[common], help="Run the gateway.")
    serve.add_argument("-c", "--config", required=True, help="Lab configuration YAML.")
    serve.add_argument("-t", "--transport", default="http",
                       choices=["stdio", "http", "ws", "all"],
                       help="stdio for a local agent, ws for duplex, http for anything else.")
    serve.add_argument("--host", default="127.0.0.1",
                       help="Binding off-loopback requires --token.")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--token", default=None,
                       help="Bearer token. Also read from LABBENCH_TOKEN.")
    serve.set_defaults(func=_serve)

    doctor = sub.add_parser("doctor", parents=[common], help="Check the installation and a config.")
    doctor.add_argument("-c", "--config", default=None)
    doctor.set_defaults(func=_doctor)

    devices = sub.add_parser("devices", parents=[common], help="List instruments and their capabilities.")
    devices.add_argument("-c", "--config", required=True)
    devices.add_argument("--json", action="store_true")
    devices.set_defaults(func=_devices)

    tools = sub.add_parser("tools", parents=[common], help="Print tool schemas in an AI dialect.")
    tools.add_argument("-c", "--config", default=None)
    tools.add_argument("-d", "--dialect", default="jsonschema",
                       choices=["anthropic", "openai", "openai-responses", "gemini",
                                "jsonschema", "openapi"])
    tools.add_argument("--strict", action="store_true",
                       help="OpenAI strict mode: guaranteed schema adherence.")
    tools.set_defaults(func=_tools)

    ledger = sub.add_parser("ledger", parents=[common], help="Read or verify the provenance ledger.")
    ledger.add_argument("ledger_command", choices=["verify", "query"])
    ledger.add_argument("--path", default=None, help="Path to provenance.sqlite.")
    ledger.add_argument("--run", default=None)
    ledger.add_argument("--device", default=None)
    ledger.add_argument("--kind", default=None)
    ledger.add_argument("--limit", type=int, default=100)
    ledger.add_argument("--json", action="store_true")
    ledger.set_defaults(func=_ledger)

    call = sub.add_parser("call", parents=[common], help="Invoke one gateway method and print the reply.")
    call.add_argument("-c", "--config", required=True)
    call.add_argument("method", help="e.g. lab.describe, device.read")
    call.add_argument("params", nargs="*", help="key=value; values are parsed as JSON.")
    call.add_argument("--actor", default="human:cli")
    call.set_defaults(func=_call)

    experiment = sub.add_parser("experiment", parents=[common], help="Run a protocol.")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    run = experiment_sub.add_parser("run", parents=[common], help="Run a protocol to completion.")
    run.add_argument("-c", "--config", required=True)
    run.add_argument("protocol", help="Path to a protocol YAML file.")
    run.add_argument("var", nargs="*", help="key=value overrides for the protocol's variables.")
    run.set_defaults(func=_experiment_run)

    campaign = sub.add_parser("campaign", parents=[common], help="Run a closed-loop campaign.")
    campaign_sub = campaign.add_subparsers(dest="campaign_command", required=True)
    campaign_run = campaign_sub.add_parser(
        "run", parents=[common], help="Run a campaign to completion."
    )
    campaign_run.add_argument("-c", "--config", required=True)
    campaign_run.add_argument("campaign", help="Path to a campaign YAML file.")
    campaign_run.set_defaults(func=_campaign_run)

    eval_parser = sub.add_parser(
        "eval", parents=[common], help="Run scored agent evals against the simulated lab."
    )
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_list = eval_sub.add_parser("list", parents=[common], help="List available eval tasks.")
    eval_list.set_defaults(func=_eval_list)

    eval_run = eval_sub.add_parser(
        "run", parents=[common], help="Run one or more eval tasks against a real model."
    )
    eval_run.add_argument("-d", "--dialect", required=True, choices=["anthropic", "openai", "gemini"])
    eval_run.add_argument("-m", "--model", default=None, help="Defaults to a sane per-dialect model.")
    eval_run.add_argument("--base-url", default=None,
                          help="For --dialect openai against a local OpenAI-compatible server.")
    eval_run.add_argument("-t", "--task", action="append", default=None,
                          help="Repeatable; omit to run every task (see 'labbench eval list').")
    eval_run.add_argument("--max-turns", type=int, default=8)
    eval_run.add_argument("--json", action="store_true")
    eval_run.set_defaults(func=_eval_run)

    return parser


log = logging.getLogger("labbench.cli")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        if args.verbose:
            raise
        # A traceback is noise for an operator; -v gets you the real thing.
        print(f"labbench: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
