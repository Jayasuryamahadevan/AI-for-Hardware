"""The example agent loops: real subprocesses, a real gateway.

`agent_generic.py` has no optional SDK dependency, so it is run for real
against a real `labbench serve` process -- the same proof-by-execution this
suite applies everywhere else, rather than trusting the example by
inspection. The three SDK-backed examples (`agent_anthropic.py`,
`agent_openai.py`, `agent_gemini.py`) cannot be run for real without a paid
API key, so they are held to a weaker but still meaningful bar: they must
import, parse their arguments and fail with the *documented* `pip install`
message when the SDK is absent, not with an unrelated crash. That is enough
to catch a typo or a bad import without needing credentials in CI.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
CONFIG = REPO_ROOT / "configs" / "simulated-lab.yaml"

SDK_BACKED_EXAMPLES = ["agent_anthropic.py", "agent_openai.py", "agent_gemini.py"]


@pytest.fixture
def running_gateway(tmp_path):
    port = 18901
    data_dir = tmp_path / "labbench-data"
    proc = subprocess.Popen(
        [sys.executable, "-m", "labbench.cli", "serve", "-c", str(CONFIG),
         "--transport", "ws", "--port", str(port), "--data-dir", str(data_dir)],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            line = proc.stdout.readline()
            if "serving" in line:
                break
            if proc.poll() is not None:
                raise RuntimeError("gateway exited before it started serving")
        yield f"ws://127.0.0.1:{port}/ws"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


class TestGenericExample:
    def test_runs_the_scripted_sequence_against_a_real_gateway(self, running_gateway):
        result = subprocess.run(
            [sys.executable, "agent_generic.py", "--gateway", running_gateway, "smoke test"],
            cwd=EXAMPLES, capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "lab.find" in result.stderr
        assert "device.invoke" in result.stderr
        assert '"accepted": true' in result.stderr  # the home command was actually accepted
        assert "artifact_uri" in result.stderr  # the snap actually produced an image


class TestSdkBackedExamplesFailCleanlyWithoutTheSdk:
    @pytest.mark.parametrize("script", SDK_BACKED_EXAMPLES)
    def test_missing_sdk_gives_a_pip_install_message(self, script):
        # These SDKs are not installed in this environment; the point of the
        # test is exactly that fact, not a skip condition.
        result = subprocess.run(
            [sys.executable, script, "--gateway", "ws://127.0.0.1:1/ws"],
            cwd=EXAMPLES, capture_output=True, text=True, timeout=15, check=False,
        )
        assert result.returncode != 0
        assert "pip install" in result.stderr

    @pytest.mark.parametrize("script", SDK_BACKED_EXAMPLES)
    def test_help_text_works_without_the_sdk_or_a_gateway(self, script):
        # Argument parsing must not itself require the optional SDK to be
        # importable -- a user should be able to run --help before installing
        # anything.
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=EXAMPLES, capture_output=True, text=True, timeout=15, check=False,
        )
        assert result.returncode == 0
        assert "--gateway" in result.stdout
