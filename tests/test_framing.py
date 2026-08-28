"""Message framing for stream transports: LINE and HEADER, and their failure modes."""

from __future__ import annotations

import asyncio

import pytest

from labbench.protocol.framing import MAX_FRAME_BYTES, FrameError, Framing, encode, read_frame


def make_reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


class TestLineFraming:
    async def test_round_trip(self):
        payload = encode('{"a":1}', Framing.LINE)
        assert payload == b'{"a":1}\n'
        frame = await read_frame(make_reader(payload), Framing.LINE)
        assert frame == '{"a":1}'

    async def test_blank_lines_are_skipped_as_keepalive(self):
        reader = make_reader(b"\n\n{\"a\":1}\n")
        frame = await read_frame(reader, Framing.LINE)
        assert frame == '{"a":1}'

    async def test_clean_eof_returns_none(self):
        frame = await read_frame(make_reader(b""), Framing.LINE)
        assert frame is None

    async def test_mid_frame_eof_raises(self):
        reader = make_reader(b'{"a":1}')  # no trailing newline
        with pytest.raises(FrameError):
            await read_frame(reader, Framing.LINE)

    def test_oversized_frame_is_rejected_at_encode_time(self):
        with pytest.raises(FrameError):
            encode("x" * (MAX_FRAME_BYTES + 1), Framing.LINE)


class TestHeaderFraming:
    async def test_round_trip(self):
        payload = encode('{"a":1}', Framing.HEADER)
        assert payload.startswith(b"Content-Length: 7\r\n\r\n")
        frame = await read_frame(make_reader(payload), Framing.HEADER)
        assert frame == '{"a":1}'

    async def test_missing_content_length_raises(self):
        reader = make_reader(b"X-Other: 1\r\n\r\n{}")
        with pytest.raises(FrameError):
            await read_frame(reader, Framing.HEADER)

    async def test_malformed_content_length_raises(self):
        reader = make_reader(b"Content-Length: notanumber\r\n\r\n{}")
        with pytest.raises(FrameError):
            await read_frame(reader, Framing.HEADER)

    async def test_truncated_body_raises(self):
        reader = make_reader(b"Content-Length: 100\r\n\r\n{}")
        with pytest.raises(FrameError):
            await read_frame(reader, Framing.HEADER)

    async def test_clean_eof_before_headers_returns_none(self):
        frame = await read_frame(make_reader(b""), Framing.HEADER)
        assert frame is None
