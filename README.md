# LabBench

**A USB cable for laboratory hardware.** Plug any AI model into any supported instrument.

```
   ANY AI MODEL                  LabBench                    ANY INSTRUMENT
 ┌───────────────┐         ┌──────────────────┐         ┌───────────────────┐
 │ Claude        │         │                  │         │ Microscope  MMCore│
 │ GPT           │◄───────►│  capability model│◄───────►│ Oscilloscope SCPI │
 │ Gemini        │  tool   │  safety kernel   │ driver  │ Plate reader SiLA2│
 │ Llama / local │ schemas │  provenance ledge│         │ Incubator    LADS │
 │ your own loop │         │                  │         │ Liquid handler    │
 └───────────────┘         └──────────────────┘         └───────────────────┘
      host end                  the cable                    device end
```

A USB cable does not care which computer is on one end or which device is on
the other. That only works because both ends agree on a small, boring contract:
how you enumerate a device, how you describe what it can do, how power is
negotiated, and what a *class* of device looks like so one driver serves every
vendor's mouse.

LabBench is that contract for lab instruments and AI agents.

---

## Status

**Under active construction.** This table is the honest state of the tree and is
updated as each layer lands.

| Layer | What it is | State |
|---|---|---|
| `core/` | Capability model, safety kernel, jobs, provenance ledger | ✅ complete |
| `protocol/` | JSON-RPC 2.0, dispatch router | ✅ complete |
| `protocol/` | HTTP/1.1 server, SSE, WebSocket (RFC 6455), stdio | ✅ complete |
| `protocol/client.py` | Client for all three transports, one call surface | ✅ complete |
| `bridge/` | Tool schemas (6 AI dialects), human-approval broker | ✅ complete |
| `bridge/toolset.py` | The gateway tool surface (16 tools, 26 methods) | ✅ complete |
| `gateway.py` | Assembly: request → ledger → safety → approval → act | ✅ complete |
| `drivers/simulated/` | Microscope, plate reader, liquid handler, incubator | ✅ complete |
| `drivers/` | SCPI, WoT, MicroManager, SiLA2, OPC UA LADS, Opentrons | ⬜ |
| `configs/` | A complete four-instrument simulated lab | ✅ complete |
| `memory/` | Durable notes and documents an agent can search | 🔨 building |
| `experiment/` | Protocols, runs, replay | ⬜ |
| `cli.py` | `serve` · `doctor` · `tools` · `ledger` | ⬜ |
| `tests/` | Test suite | ⬜ |

---

## Why a cable and not an integration

The usual way to give an AI agent control of an instrument is to write a script
that wires one model to one device. That works exactly once. Change the model
and the tool schemas are wrong; change the instrument and the whole script is
wrong; and nothing in it knows whether the command it just sent was safe.

USB solved the equivalent problem by refusing to be an integration. It defined
a host end, a device end, and a set of *device classes* in between — so a new
mouse needs no new driver, and a new laptop needs no new mouse. LabBench copies
that shape:

**The device end — one driver per protocol, not per instrument.**
Every lab-automation standard converged on the same three nouns:

| LabBench | SiLA 2 | W3C WoT | OPC UA LADS |
|---|---|---|---|
| Property | Property | Property | Variable |
| Command | Command | Action | Method |
| Event | Observable property | Event | Event / Notifier |

LabBench adopts that triple as its internal model and treats each real protocol
as a *projection* of it. Adding SCPI support means writing one projection, not
touching anything an agent sees.

**Device classes — `Feature`, the thing that makes instruments substitutable.**
Capabilities are grouped by *function*, not by model number. An agent that can
drive `MotionControl/move_absolute` drives every stage that implements it, from
any vendor. This is USB HID for lab hardware: the class is the contract.

**The host end — one schema emitter per AI dialect.**
The same capability model is projected outward into whatever the model on the
other end speaks: Anthropic tool definitions, OpenAI function-calling schemas,
Gemini declarations, or plain JSON Schema over HTTP for a local model and a
hand-rolled loop. No agent SDK is a dependency.

**Four ways to plug in.** The call surface is identical across all of them:

| Transport | For | Notifications |
|---|---|---|
| `stdio://` | A local agent that spawns the gateway | Streamed live |
| `ws://` | A remote agent wanting duplex over one socket | Streamed live |
| `http://` | Anything with an HTTP client and a JSON parser | Ride back with the reply |
| SSE `/events` | Dashboards, watchdogs, `curl -N` | Streamed live, filterable by topic |

**Power negotiation — graduated autonomy.**
USB devices do not simply draw whatever current they like; they ask, and the
host grants. Neither does an agent here. A session is granted an autonomy level
0–5, and that level caps which *hazard classes* may execute without a human
signature.

---

## The part that is not a cable

A cable is passive. This one is not, and it is the reason the project exists.

An LLM emits commands that are perfectly well-formed and physically
catastrophic — the syntax-to-safety gap. The answer is not a better prompt. It
is an interposed layer that knows the declared operating envelope, simulates
before it actuates, and escalates to a human when the action's hazard exceeds
the granted autonomy.

Every command crosses three gates, cheapest first:

1. **Policy** — is this actor allowed to run this command at all? *(glob rules, rate limits, site-tightened parameter limits, permitted hours)*
2. **Envelope** — do the arguments sit inside the declared operating domain?
3. **Simulation** — does a model of the device say the trajectory stays safe?

Only then does anything physical happen. The ordering matters: the expensive
check runs last, and the irreversible one runs never if an earlier gate says no.

**Hazard classes** are ordered, and the ordering is load-bearing:

```
none → benign → motion → sample → thermal → chemical → biological → radiological
```

Biological and radiological always require a human signature, at any autonomy
level. Containment and ionising-radiation decisions are not delegated to a model.

**Everything is recorded.** Each action is written to an append-only,
hash-chained ledger before and after execution — SQLite for queries, mirrored
JSONL for the auditor who has none of this software. Any retroactive edit is
detectable by re-walking the chain. That is what GxP audit-trail rules and
ALCOA+ ask for, and what a reproducibility claim needs regardless of regulation.

---

## Design decisions worth knowing

**The tool surface is fixed, not generated per command.** A six-instrument lab
would generate two hundred–odd tools and bury the agent. Instead there is a
small constant set — `device.describe` returns the capability model *as data*,
including JSON Schema per command, and `device.invoke` takes a feature, a
command and arguments. The tool list does not grow as the lab does.

**Units are mandatory on physical quantities.** Unit mismatch is the single most
common class of automation error, and a model reading `"unit": "um"` is far less
likely to send millimetres than one reading a bare `float`.

**Long operations return a handle, not a blocked call.** A tool call is
request/response; a tile scan is forty minutes. Anything that can outlive a
timeout runs as a job with progress, cancellation and artifacts.

**A driver that cannot predict must say so.** `simulate()` returns a fidelity,
and `"none"` is a legal answer. A driver that silently returned success would
turn the safety gate into theatre.

**Simulated instruments run real physics.** Defocus actually blurs, illumination
actually bleaches the sample, and the camera has real shot noise — so an agent
doing closed-loop autofocus has to genuinely search, and a badly planned
time-lapse genuinely destroys the specimen. A stub driver would teach an agent
nothing and would validate none of the safety machinery.

---

## Dependencies

Three, at runtime: `pydantic`, `pyyaml`, `numpy`.

The wire protocol, the HTTP server, the WebSocket framing and the tool-schema
emitters are all written from scratch in this package. There is no web
framework and no agent-vendor SDK, so the agent-facing contract of a laboratory
gateway is not hostage to two third-party release cadences — and an instrument
part-way through a five-year experiment cannot be broken by someone else's
major version. An auditor reading the ledger needs none of it either.

Instrument libraries are optional extras, one per protocol. A driver whose
vendor library is missing is reported as unavailable, with the extra needed to
install it; a lab with no hardware attached still starts.

```bash
pip install labbench              # gateway + simulated instruments
pip install 'labbench[scpi]'      # + oscilloscopes, power supplies, DMMs
pip install 'labbench[all]'       # + every supported protocol
```

---

## Licence

Apache-2.0
