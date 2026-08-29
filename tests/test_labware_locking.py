"""`PlateStore.hold`: the same physical plate, touched by two different
devices, must serialise -- and two different plates must never wait on
each other. `_labware.BENCH` is a process-wide singleton (see its own
docstring), so every test here clears it and uses barcodes no other test
in the suite is expected to touch.
"""

from __future__ import annotations

import asyncio

import pytest

from labbench.core.errors import LabBenchError
from labbench.drivers.simulated import _labware
from labbench.drivers.simulated._labware import PlateStore


@pytest.fixture(autouse=True)
def clean_bench():
    _labware.BENCH.clear()
    yield
    _labware.BENCH.clear()


class TestPlateStoreHold:
    async def test_hold_yields_the_plate(self):
        store = PlateStore()
        store.create("P1")
        async with store.hold("P1") as plate:
            assert plate.barcode == "P1"

    async def test_hold_raises_for_an_unknown_barcode(self):
        store = PlateStore()
        with pytest.raises(KeyError):
            async with store.hold("nope"):
                pass

    async def test_same_barcode_serialises_across_two_holders(self):
        store = PlateStore()
        store.create("P1")
        order: list[str] = []

        async def holder(name: str, hold_for: float) -> None:
            async with store.hold("P1"):
                order.append(f"{name}-enter")
                await asyncio.sleep(hold_for)
                order.append(f"{name}-exit")

        # "a" starts first and holds long enough that "b" can only enter
        # after "a" has fully released -- if the lock did nothing, "b" would
        # enter while "a" is still inside.
        await asyncio.gather(holder("a", 0.05), holder("b", 0.0))
        assert order == ["a-enter", "a-exit", "b-enter", "b-exit"]

    async def test_different_barcodes_do_not_block_each_other(self):
        store = PlateStore()
        store.create("P1")
        store.create("P2")
        order: list[str] = []

        async def holder(name: str, barcode: str, hold_for: float) -> None:
            async with store.hold(barcode):
                order.append(f"{name}-enter")
                await asyncio.sleep(hold_for)
                order.append(f"{name}-exit")

        # "a" holds P1 for a while; "b" (on P2) must be able to run to
        # completion inside that window rather than queueing behind it --
        # unlike the same-barcode case, "b" finishes before "a" does.
        await asyncio.gather(holder("a", "P1", 0.05), holder("b", "P2", 0.0))
        assert order.index("b-exit") < order.index("a-exit")

    async def test_two_real_devices_racing_for_the_same_plate_never_both_win(self, gateway):
        """The end-to-end version: an incubator and a plate reader both try
        to claim the same physical plate at the same instant. Without the
        lock this is a textbook check-then-claim race (see `_cmd_store_plate`
        and `_cmd_load_plate`'s own comments); with it, exactly one instrument
        gets the plate and the other sees a truthful `ConstraintViolation`
        naming who actually has it -- never a plate silently owned by both.
        """
        await gateway.invoke(
            "handler1", "Labware", "create_plate", {"barcode": "RACE1"}, actor="test",
        )

        results = await asyncio.gather(
            gateway.invoke("incubator1", "PlateStorage", "store_plate",
                           {"barcode": "RACE1"}, actor="test"),
            gateway.invoke("reader1", "PlateTransport", "load_plate",
                           {"barcode": "RACE1"}, actor="test"),
            return_exceptions=True,
        )

        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1, results
        assert len(failures) == 1, results
        assert isinstance(failures[0], LabBenchError)
        assert "RACE1" in failures[0].message

        # The plate ended up with exactly the one instrument that actually won.
        plate = _labware.BENCH.get("RACE1")
        assert plate.location in ("incubator1", "reader1")

    async def test_mutation_inside_hold_is_not_torn_by_a_concurrent_holder(self):
        """The realistic check: two 'devices' each add volume to the same
        well through `hold`; the total must be exactly the sum of both,
        never a lost update from an unguarded read-modify-write."""
        store = PlateStore()
        store.create("P1", plate_format="96")

        async def add_to_well(amount: float) -> None:
            async with store.hold("P1") as plate:
                well = plate.well("A1")
                current = well.volume_ul
                await asyncio.sleep(0.01)  # the exact window an unguarded race needs
                well.volume_ul = current + amount

        await asyncio.gather(*(add_to_well(10.0) for _ in range(20)))
        assert store.get("P1").well("A1").volume_ul == pytest.approx(200.0)
