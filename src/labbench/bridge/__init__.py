"""The host end of the connector: one capability model, every AI dialect.

`protocol/` decides how bytes move. This package decides what the model on the
other end is *told*, and it exists because the industry has not converged. The
same tool must be described as `input_schema` to one vendor, nested under
`function.parameters` for another, and stripped down to an OpenAPI subset for a
third -- and a local model with a hand-written loop wants none of that, just
JSON Schema.

Writing those projections once, here, is what stops "connect any AI" from
meaning "connect the one AI we happened to test against".
"""

from .schema import (
    DIALECTS,
    Dialect,
    ToolSpec,
    emit,
    sanitise_name,
)

__all__ = ["Dialect", "DIALECTS", "ToolSpec", "emit", "sanitise_name"]
