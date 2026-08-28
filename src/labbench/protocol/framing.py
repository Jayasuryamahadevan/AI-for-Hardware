"""Message framing for stream transports.

A TCP stream or a pipe has no message boundaries, so one has to be imposed.
Two schemes are supported because both are in the field and neither is
universally better:

`LINE`  -- one JSON object per line. Trivially greppable, works with `tee`,
           and a human can read the session in a terminal. Its cost is that no
           message may contain a raw newline, which JSON compact encoding
           guarantees anyway.

`HEADER` -- an HTTP-style `Content-Length` prefix. Safe for any payload, and
           what most editor/LSP-style tooling already speaks.

Both readers are written to be robust against the thing that actually happens
in the field: a partially-written frame from a peer that died mid-send. Neither
blocks forever on it, and neither desynchronises the stream after it.
"""

from __future__ import annotations

import asyncio
from enum import Enum

#: Refuse frames larger than this. A microscope image must travel as an
#: artifact reference, never inline, and a 64 MiB "JSON message" is a bug or an
#: attack rather than a legitimate instrument reply.
MAX_FRAME_BYTES = 16 * 1024 * 1024


class Framing(str, Enum):
    LINE = "line"
    HEADER = "header"


class FrameError(Exception):
    """The stream is malformed in a way that cannot be resynchronised."""


def encode(payload: str, framing: Framing = Framing.LINE) -> bytes:
    body = payload.encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise FrameError(f"frame of {len(body)} bytes exceeds the {MAX_FRAME_BYTES} byte limit")
    if framing is Framing.LINE:
        return body + b"\n"
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


async def read_frame(reader: asyncio.StreamReader, framing: Framing = Framing.LINE) -> str | None:
    """Read one frame. Returns None at clean end-of-stream."""
    if framing is Framing.LINE:
        return await _read_line_frame(reader)
    return await _read_header_frame(reader)


async def _read_line_frame(reader: asyncio.StreamReader) -> str | None:
    try:
        line = await reader.readline()
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None
    if not line:
        return None
    if not line.endswith(b"\n"):
        # EOF mid-line: the peer died part-way through a frame. There is no
        # resynchronising from that, and a truncated JSON object must never be
        # handed on as if it were complete.
        raise FrameError("stream ended mid-frame")
    if len(line) > MAX_FRAME_BYTES:
        raise FrameError(f"frame exceeds the {MAX_FRAME_BYTES} byte limit")
    text = line.decode("utf-8", errors="replace").strip()
    # Blank lines are keep-alive padding, not messages.
    return text if text else await _read_line_frame(reader)


async def _read_header_frame(reader: asyncio.StreamReader) -> str | None:
    length: int | None = None
    while True:
        try:
            raw = await reader.readline()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None
        if not raw:
            return None if length is None else _truncated()
        line = raw.decode("ascii", errors="replace").strip()
        if not line:  # blank line terminates the header block
            break
        name, _, value = line.partition(":")
        if name.strip().lower() == "content-length":
            try:
                length = int(value.strip())
            except ValueError:
                raise FrameError(f"malformed Content-Length: {value.strip()!r}") from None
    if length is None:
        raise FrameError("header frame has no Content-Length")
    if length < 0 or length > MAX_FRAME_BYTES:
        raise FrameError(f"Content-Length {length} is out of range")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return _truncated()
    return body.decode("utf-8", errors="replace")


def _truncated() -> str:
    raise FrameError("stream ended mid-frame")
