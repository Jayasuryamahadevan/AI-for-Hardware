"""Southbound drivers — one module per protocol.

Every module here is optional. A driver whose vendor library is not installed
raises on import, `DriverRegistry.discover` catches that, and the driver is
reported as unavailable with the extra needed to install it. A lab with no
hardware attached still starts.
"""
