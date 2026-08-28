"""JSON-RPC framing/parsing and the method router."""

from __future__ import annotations

import pytest

from labbench.core.errors import DeviceNotFound, ValidationError
from labbench.protocol.jsonrpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcError,
    Request,
    parse_message,
)
from labbench.protocol.router import Router, RpcContext


class TestParseMessage:
    def test_single_request(self):
        requests, is_batch = parse_message('{"jsonrpc":"2.0","method":"m","id":1}')
        assert not is_batch
        assert requests[0].method == "m"
        assert requests[0].id == 1

    def test_notification_has_no_id(self):
        requests, _ = parse_message('{"jsonrpc":"2.0","method":"m"}')
        assert requests[0].is_notification

    def test_batch_preserves_order(self):
        requests, is_batch = parse_message(
            '[{"jsonrpc":"2.0","method":"a","id":1},{"jsonrpc":"2.0","method":"b","id":2}]'
        )
        assert is_batch
        assert [r.method for r in requests] == ["a", "b"]

    def test_empty_batch_is_invalid(self):
        with pytest.raises(JsonRpcError) as exc:
            parse_message("[]")
        assert exc.value.code == INVALID_REQUEST

    def test_bad_json_is_parse_error(self):
        with pytest.raises(JsonRpcError) as exc:
            parse_message("{not json")
        assert exc.value.code == PARSE_ERROR

    def test_wrong_version_rejected(self):
        with pytest.raises(JsonRpcError):
            parse_message('{"jsonrpc":"1.0","method":"m","id":1}')

    def test_positional_params_are_legal_at_the_wire_level(self):
        # Rejected later by the router (see TestRouter), not by parsing.
        requests, _ = parse_message('{"jsonrpc":"2.0","method":"m","params":[1,2],"id":1}')
        assert requests[0].kwargs == {}


class TestRouter:
    @pytest.fixture
    def router(self):
        r = Router()

        @r.method("echo")
        async def echo(value: int) -> dict:
            return {"value": value}

        @r.method("boom")
        async def boom() -> None:
            raise ValidationError("bad", parameter="x")

        @r.method("device_missing")
        async def device_missing() -> None:
            raise DeviceNotFound("nope", device="x")

        @r.method("crashes")
        async def crashes() -> None:
            raise RuntimeError("driver bug")

        @r.method("wants_context")
        async def wants_context(ctx: RpcContext) -> dict:
            return {"actor": ctx.actor}

        return r

    async def test_dispatch_success(self, router):
        ctx = RpcContext()
        response = await router.dispatch(Request("echo", {"value": 5}, id=1), ctx)
        assert response.result == {"value": 5}

    async def test_unknown_method(self, router):
        ctx = RpcContext()
        response = await router.dispatch(Request("nope", {}, id=1), ctx)
        assert response.error.code == METHOD_NOT_FOUND

    async def test_positional_params_rejected_with_a_clear_reason(self, router):
        ctx = RpcContext()
        response = await router.dispatch(Request("echo", [1], id=1), ctx)
        assert response.error.code == INVALID_PARAMS
        assert "named parameters" in response.error.message

    async def test_labbench_error_carries_structured_data(self, router):
        ctx = RpcContext()
        response = await router.dispatch(Request("boom", {}, id=1), ctx)
        assert response.error.data["error"] == "validation_error"
        assert response.error.data["detail"]["parameter"] == "x"

    async def test_specific_error_gets_a_distinct_code(self, router):
        ctx = RpcContext()
        response = await router.dispatch(Request("device_missing", {}, id=1), ctx)
        assert response.error.code == -32001

    async def test_unexpected_exception_is_reported_as_state_uncertain(self, router):
        ctx = RpcContext()
        response = await router.dispatch(Request("crashes", {}, id=1), ctx)
        assert response.error.data["state_uncertain"] is True

    async def test_notification_swallows_errors(self, router):
        ctx = RpcContext()
        response = await router.dispatch(Request("boom", {}, is_notification=True), ctx)
        assert response is None

    async def test_context_injection(self, router):
        ctx = RpcContext(actor="human:alice")
        response = await router.dispatch(Request("wants_context", {}, id=1), ctx)
        assert response.result == {"actor": "human:alice"}

    async def test_dispatch_many_runs_concurrently_and_preserves_order(self, router):
        ctx = RpcContext()
        requests = [Request("echo", {"value": i}, id=i) for i in range(5)]
        responses = await router.dispatch_many(requests, ctx)
        assert [r.result["value"] for r in responses] == [0, 1, 2, 3, 4]

    def test_duplicate_registration_raises(self, router):
        with pytest.raises(ValueError):
            router.add("echo", lambda: None)
