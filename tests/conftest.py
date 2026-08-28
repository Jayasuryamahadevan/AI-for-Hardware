"""Shared fixtures.

`asyncio_mode = "auto"` (pyproject.toml) means an `async def test_...` needs no
`@pytest.mark.asyncio` marker; pytest-asyncio runs it for us.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from labbench.core.registry import LabConfig
from labbench.gateway import Gateway


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "labbench-data"


@pytest.fixture
def simulated_lab_config() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "simulated-lab.yaml"


@pytest.fixture
async def gateway(simulated_lab_config: Path, data_dir: Path):
    """A fully started Gateway over the shipped four-instrument simulated lab."""
    config = LabConfig.load(simulated_lab_config)
    gw = Gateway(config, data_dir=data_dir)
    await gw.start()
    try:
        yield gw
    finally:
        await gw.close()


@pytest.fixture
async def bare_gateway(data_dir: Path):
    """An empty gateway (no devices) for tests that only need core plumbing."""
    gw = Gateway(data_dir=data_dir)
    await gw.start()
    try:
        yield gw
    finally:
        await gw.close()


async def wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.02) -> None:
    """Poll `predicate()` until it is true or `timeout` elapses.

    Used instead of a fixed `asyncio.sleep` wherever a test waits on a
    background job or run: a fixed sleep is either too slow (annoying) or too
    fast (flaky) depending on the machine, and this is neither.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(interval)
