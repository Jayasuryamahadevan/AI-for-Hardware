"""The append-only, hash-chained ledger."""

from __future__ import annotations

import json

import pytest

from labbench.core.provenance import GENESIS, Ledger


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(tmp_path / "provenance.sqlite", session_id="test-session")
    try:
        yield led
    finally:
        led.close()


class TestChain:
    def test_first_record_chains_to_genesis(self, ledger):
        record = ledger.log("note", actor="test", reason="first")
        assert record.seq == 1
        assert record.prev_hash == GENESIS
        assert record.hash

    def test_records_chain_to_predecessor(self, ledger):
        first = ledger.log("note", actor="test")
        second = ledger.log("note", actor="test")
        assert second.prev_hash == first.hash
        assert second.seq == first.seq + 1

    def test_verify_passes_on_untouched_chain(self, ledger):
        for _ in range(5):
            ledger.log("note", actor="test")
        result = ledger.verify()
        assert result["valid"] is True
        assert result["records"] == 5

    def test_verify_detects_tampered_payload(self, ledger):
        ledger.log("note", actor="test", payload={"value": 1})
        ledger.log("note", actor="test")
        # Simulate an operator editing history directly in SQLite.
        ledger._conn.execute(
            "UPDATE ledger SET payload = ? WHERE seq = 1", (json.dumps({"value": 999}),)
        )
        ledger._conn.commit()
        result = ledger.verify()
        assert result["valid"] is False
        assert result["broken_at"] == 1

    def test_verify_detects_broken_link(self, ledger):
        ledger.log("note", actor="test")
        ledger.log("note", actor="test")
        ledger._conn.execute("UPDATE ledger SET prev_hash = ? WHERE seq = 2", ("f" * 64,))
        ledger._conn.commit()
        result = ledger.verify()
        assert result["valid"] is False
        assert result["broken_at"] == 2


class TestJsonlMirror:
    def test_jsonl_file_mirrors_sqlite(self, ledger, tmp_path):
        ledger.log("note", actor="test", reason="hello")
        jsonl_path = tmp_path / "provenance.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["reason"] == "hello"


class TestQuery:
    def test_query_filters_by_kind_and_device(self, ledger):
        ledger.log("command_request", device_id="scope1")
        ledger.log("command_result", device_id="scope1")
        ledger.log("command_request", device_id="reader1")
        results = ledger.query(kind="command_request", device_id="scope1")
        assert len(results) == 1

    def test_query_orders_oldest_first_within_the_limit(self, ledger):
        for i in range(3):
            ledger.log("note", reason=str(i))
        results = ledger.query(limit=100)
        assert [r.reason for r in results] == ["0", "1", "2"]
