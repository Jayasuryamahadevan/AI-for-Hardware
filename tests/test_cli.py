"""CLI smoke tests: real subprocesses, the real console script.

Deliberately black-box (subprocess, not importing `cli.main` in-process):
what matters here is the thing an operator actually runs at the terminal,
argv and all.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "configs" / "simulated-lab.yaml"


def run_cli(*args: str, data_dir: Path, input_text: str | None = None, timeout: float = 30) -> subprocess.CompletedProcess:
    # --data-dir before the subcommand: see TestGlobalFlagOrder for why this
    # order specifically must work, not just --data-dir after it.
    return subprocess.run(
        [sys.executable, "-m", "labbench.cli", "--data-dir", str(data_dir), *args],
        capture_output=True, text=True, input=input_text, timeout=timeout,
        cwd=REPO_ROOT, check=False,
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "labbench-data"


class TestDoctor:
    def test_reports_all_good_for_the_shipped_config(self, data_dir):
        result = run_cli("doctor", "-c", str(CONFIG), data_dir=data_dir)
        assert result.returncode == 0
        assert "all good" in result.stdout

    def test_runs_with_no_config_at_all(self, data_dir):
        result = run_cli("doctor", data_dir=data_dir)
        assert result.returncode == 0


class TestDevices:
    def test_lists_every_configured_device(self, data_dir):
        result = run_cli("devices", "-c", str(CONFIG), data_dir=data_dir)
        assert result.returncode == 0
        for device_id in ("scope1", "reader1", "handler1", "incubator1"):
            assert device_id in result.stdout

    def test_json_output_is_valid_json(self, data_dir):
        result = run_cli("devices", "-c", str(CONFIG), "--json", data_dir=data_dir)
        payload = json.loads(result.stdout)
        assert len(payload["devices"]) == 4


class TestTools:
    @pytest.mark.parametrize("dialect", ["anthropic", "openai", "gemini", "jsonschema", "openapi"])
    def test_every_dialect_emits_valid_json(self, dialect, data_dir):
        result = run_cli("tools", "--dialect", dialect, data_dir=data_dir)
        assert result.returncode == 0
        json.loads(result.stdout)  # must not raise


class TestCall:
    def test_lab_describe(self, data_dir):
        result = run_cli("call", "-c", str(CONFIG), "lab.describe", data_dir=data_dir)
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["result"]["lab"] == "simulated-lab"

    def test_unknown_method_is_a_clean_error_not_a_traceback(self, data_dir):
        result = run_cli("call", "-c", str(CONFIG), "not.a.real.method", data_dir=data_dir)
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert "error" in payload

    def test_key_value_params_are_parsed_as_json(self, data_dir):
        result = run_cli(
            "call", "-c", str(CONFIG), "device.read", "device=scope1", "feature=MotionControl",
            "property=x_um", data_dir=data_dir,
        )
        payload = json.loads(result.stdout)
        assert payload["result"]["property"] == "x_um"


class TestGlobalFlagOrder:
    """`--data-dir` (and `-v`) are declared on both the top-level parser and
    every subcommand so either order works. A subparser's own default for a
    shared destination applies unconditionally when that subparser does not
    see the flag itself, which silently discards whatever the top-level
    parser already parsed unless that default is `argparse.SUPPRESS` --
    exactly the bug this locks in against regressing.
    """

    def test_data_dir_before_the_subcommand_is_honoured(self, data_dir, tmp_path):
        protocol_marker = data_dir / "provenance.sqlite"
        assert not protocol_marker.exists()
        result = run_cli("call", "-c", str(CONFIG), "lab.describe", data_dir=data_dir)
        assert result.returncode == 0
        assert protocol_marker.exists(), "the ledger was written somewhere other than --data-dir"

    def test_data_dir_after_the_subcommand_also_works(self, data_dir):
        result = subprocess.run(
            [sys.executable, "-m", "labbench.cli", "call", "-c", str(CONFIG),
             "lab.describe", "--data-dir", str(data_dir)],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=30, check=False,
        )
        assert result.returncode == 0
        assert (data_dir / "provenance.sqlite").exists()


class TestLedger:
    def test_verify_on_a_fresh_ledger(self, data_dir):
        # Create the ledger first via any command that starts a gateway.
        run_cli("call", "-c", str(CONFIG), "lab.describe", data_dir=data_dir)
        result = run_cli("ledger", "verify", data_dir=data_dir)
        assert result.returncode == 0
        assert "chain intact" in result.stderr

    def test_verify_with_no_ledger_is_a_clean_error(self, data_dir):
        result = run_cli("ledger", "verify", data_dir=data_dir)
        assert result.returncode != 0
        assert "no ledger" in result.stderr


class TestExperimentRun:
    PROTOCOL = """
name: quick-image
variables:
  exposure: 40.0
steps:
  - label: home
    device: scope1
    feature: MotionControl
    command: home
  - label: snap
    device: scope1
    feature: Camera
    command: snap
    args: {exposure_ms: "${exposure}"}
"""

    def test_successful_run(self, data_dir, tmp_path):
        protocol_path = tmp_path / "protocol.yaml"
        protocol_path.write_text(self.PROTOCOL)
        result = run_cli(
            "experiment", "run", "-c", str(CONFIG), str(protocol_path), data_dir=data_dir,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["status"] == "succeeded"
        assert len(payload["steps"]) == 2

    def test_invalid_protocol_is_reported_without_running_anything(self, data_dir, tmp_path):
        protocol_path = tmp_path / "bad.yaml"
        protocol_path.write_text(
            "name: bad\nsteps:\n  - device: no_such_device\n    feature: F\n    command: c\n"
        )
        result = run_cli(
            "experiment", "run", "-c", str(CONFIG), str(protocol_path), data_dir=data_dir,
        )
        assert result.returncode == 1
        assert "no_such_device" in result.stderr

    def test_variable_override_from_the_command_line(self, data_dir, tmp_path):
        protocol_path = tmp_path / "protocol.yaml"
        protocol_path.write_text(self.PROTOCOL)
        result = run_cli(
            "experiment", "run", "-c", str(CONFIG), str(protocol_path), "exposure=15.0",
            data_dir=data_dir,
        )
        payload = json.loads(result.stdout)
        snap_step = next(s for s in payload["steps"] if s["label"] == "snap")
        assert snap_step["result"]["exposure_ms"] == 15.0

    def test_approval_required_step_prompts_and_honours_a_grant(self, data_dir, tmp_path):
        protocol_path = tmp_path / "gas.yaml"
        protocol_path.write_text(
            "name: gas-change\nsteps:\n"
            "  - device: incubator1\n    feature: GasControl\n    command: set_co2\n"
            "    args: {co2_pct: 5.0}\n"
        )
        result = run_cli(
            "experiment", "run", "-c", str(CONFIG), str(protocol_path),
            data_dir=data_dir, input_text="y\nhuman:test\n",
        )
        assert result.returncode == 0, result.stderr
        assert "APPROVAL REQUIRED" in result.stderr
        payload = json.loads(result.stdout)
        assert payload["status"] == "succeeded"

    def test_approval_required_step_honours_a_denial(self, data_dir, tmp_path):
        protocol_path = tmp_path / "gas.yaml"
        protocol_path.write_text(
            "name: gas-change\nsteps:\n"
            "  - device: incubator1\n    feature: GasControl\n    command: set_co2\n"
            "    args: {co2_pct: 5.0}\n"
        )
        result = run_cli(
            "experiment", "run", "-c", str(CONFIG), str(protocol_path),
            data_dir=data_dir, input_text="n\n",
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["status"] == "failed"


class TestCampaignRun:
    CAMPAIGN = """
name: focus-search
protocol:
  name: focus-trial
  steps:
    - label: focus
      device: scope1
      feature: FocusControl
      command: move_z
      args: {z_um: "${z_um}"}
    - label: snap
      device: scope1
      feature: Camera
      command: snap
      args: {exposure_ms: "${exposure_ms}"}
space:
  dimensions:
    - {name: z_um, low: 0.0, high: 190.0, unit: um}
    - {name: exposure_ms, low: 5.0, high: 200.0, unit: ms, log: true}
objectives:
  - {name: focus, path: steps.snap.result.focus_score, direction: maximize}
budget: 4
initial_design_size: 2
seed: 3
"""

    def test_successful_run(self, data_dir, tmp_path):
        campaign_path = tmp_path / "campaign.yaml"
        campaign_path.write_text(self.CAMPAIGN)
        result = run_cli(
            "campaign", "run", "-c", str(CONFIG), str(campaign_path), data_dir=data_dir,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["campaign"]["status"] == "succeeded"
        assert payload["campaign"]["trial"] == 4
        assert len(payload["campaign"]["observations"]) == 4
        assert payload["best"]["best_trial"] is not None

    def test_invalid_campaign_is_reported_without_running_anything(self, data_dir, tmp_path):
        campaign_path = tmp_path / "bad.yaml"
        campaign_path.write_text(
            "name: bad\n"
            "protocol:\n  name: x\n  steps:\n"
            "    - {device: no_such_device, feature: F, command: c}\n"
            "space:\n  dimensions:\n    - {name: z_um, low: 0.0, high: 1.0}\n"
            "objectives:\n  - {name: focus, path: steps.a.result.x, direction: maximize}\n"
        )
        result = run_cli(
            "campaign", "run", "-c", str(CONFIG), str(campaign_path), data_dir=data_dir,
        )
        assert result.returncode == 1
        assert "no_such_device" in result.stderr

    def test_approval_required_trial_prompts_and_honours_a_grant(self, data_dir, tmp_path):
        campaign_path = tmp_path / "gas.yaml"
        campaign_path.write_text(
            "name: gas-sweep\n"
            "protocol:\n  name: gas-trial\n  steps:\n"
            "    - {label: set_gas, device: incubator1, feature: GasControl, command: set_co2, "
            "args: {co2_pct: \"${co2_pct}\"}}\n"
            "space:\n  dimensions:\n    - {name: co2_pct, low: 3.0, high: 8.0}\n"
            "objectives:\n  - {name: co2, path: steps.set_gas.result.co2_pct, direction: maximize}\n"
            "budget: 1\ninitial_design_size: 1\n"
        )
        result = run_cli(
            "campaign", "run", "-c", str(CONFIG), str(campaign_path),
            data_dir=data_dir, input_text="y\nhuman:test\n",
        )
        assert result.returncode == 0, result.stderr
        assert "APPROVAL REQUIRED" in result.stderr
        payload = json.loads(result.stdout)
        assert payload["campaign"]["status"] == "succeeded"
        assert payload["campaign"]["trial"] == 1
