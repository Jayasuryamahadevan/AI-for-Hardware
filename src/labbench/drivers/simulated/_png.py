"""A minimal PNG encoder for 8- and 16-bit greyscale.

An artifact an operator cannot open is barely an artifact. The imaging stack
(`tifffile`, `imageio`) is an optional extra, so on a base install there would
otherwise be nothing to write camera frames into but `.npy` -- which needs
Python and numpy to look at, exactly when someone is trying to see what the
instrument did.

PNG is the right floor: the format is small enough to implement correctly in
sixty lines, it carries 16-bit greyscale natively (so a scientific camera's
full depth survives), it is lossless, and every operating system opens it
without asking. When `tifffile` *is* installed the microscope prefers it,
because OME-TIFF carries the metadata a downstream analysis pipeline needs.
This is the floor, not the ceiling.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """One PNG chunk: length, type, payload, CRC over type+payload."""
    return (
        struct.pack("!I", len(payload))
        + tag
        + payload
        + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def encode(image: np.ndarray, *, bit_depth: int | None = None) -> bytes:
    """Encode a 2-D greyscale array as PNG bytes.

    `bit_depth` defaults to 16 for integer arrays wider than a byte, which is
    what a scientific camera produces. Downcasting to 8 bits by default would
    silently discard most of the dynamic range the simulation went to the
    trouble of producing.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D greyscale array, got shape {image.shape}")
    if bit_depth is None:
        bit_depth = 8 if image.dtype == np.uint8 else 16
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    height, width = image.shape
    if bit_depth == 8:
        data = np.clip(image, 0, 255).astype(np.uint8)
        raw = data.tobytes()
        stride = width
    else:
        # PNG is big-endian regardless of the host's byte order.
        data = np.clip(image, 0, 65535).astype(">u2")
        raw = data.tobytes()
        stride = width * 2

    # Each scanline is prefixed with a filter byte. Filter 0 (None) keeps the
    # encoder trivial; zlib still compresses the result well because
    # microscope backgrounds are highly repetitive.
    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)
        scanlines += raw[row * stride : (row + 1) * stride]

    ihdr = struct.pack(
        "!IIBBBBB",
        width, height,
        bit_depth,
        0,  # colour type 0: greyscale
        0,  # deflate
        0,  # adaptive filtering
        0,  # no interlace
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6))
        + _chunk(b"IEND", b"")
    )


def write(path: str, image: np.ndarray, *, bit_depth: int | None = None) -> int:
    """Write a PNG and return the number of bytes written."""
    payload = encode(image, bit_depth=bit_depth)
    with open(path, "wb") as fh:
        fh.write(payload)
    return len(payload)
