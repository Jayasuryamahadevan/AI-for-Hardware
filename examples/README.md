# Agent loop examples

Four reference implementations of the same loop, one per corner of the
README's "ANY AI MODEL" box: connect to a gateway, fetch tool declarations
in that model's dialect, run whatever it calls, feed the result back.

| Script | Model | Extra dependency |
|---|---|---|
| `agent_anthropic.py` | Claude | `pip install anthropic` |
| `agent_openai.py` | GPT, or any OpenAI-compatible local server | `pip install openai` |
| `agent_gemini.py` | Gemini | `pip install google-generativeai` |
| `agent_generic.py` | anything else -- a local model, a bespoke API, your own loop | none |

None of these are LabBench dependencies; they are installed only if you run
the corresponding example, matching the project's own "three dependencies at
runtime" rule.

## Running one

```bash
# terminal 1
labbench serve -c ../configs/simulated-lab.yaml --transport ws

# terminal 2
pip install anthropic  # or openai / google-generativeai, matching the example
export ANTHROPIC_API_KEY=...
python agent_anthropic.py "Home the microscope and take a snapshot."
```

`agent_generic.py` needs no API key and no SDK -- it runs a scripted
three-step sequence against the gateway so you can see the exact
request/response shape a model needs to produce, and is the file to copy
when wiring up a model with no maintained Python client at all.

## What's shared

`_shared.py` is the ~60 lines every example has in common: connecting with
`labbench.protocol.client` (LabBench's own dependency-free client, used here
exactly as an external process would use it), fetching `tools.schema` in the
right dialect, and mapping a tool name a model emits (`device_invoke`, no
dots -- every AI dialect requires that) back to the real, dotted RPC method
(`device.invoke`) before calling the gateway. Skipping that mapping is the
most common mistake wiring a new dialect up by hand; `LabBenchTools.dispatch`
is the one place it happens so no example has to remember it.

A `SafetyViolation` or `ApprovalRequired` from the gateway is handed back to
the model as the tool's *result*, not raised as a client-side exception --
the model needs to see that its action was refused, and often the
`approval_id` to retry with once a human grants it, to recover the way an
operator would rather than the loop just crashing.

## Proof, not just prose

`tests/test_examples.py` runs `agent_generic.py` against a real
`labbench serve` process and asserts on its output -- the one example with no
paid API key requirement is verified by actually executing it, the same
"real protocol, not a mock" standard the rest of the test suite holds
drivers to. The three SDK-backed examples are checked for the failure mode
that matters without credentials: they must fail with the documented
`pip install` message, not an unrelated crash.
