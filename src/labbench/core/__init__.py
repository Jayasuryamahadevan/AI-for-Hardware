"""Protocol-independent heart of LabBench.

Nothing in this package imports MCP or any instrument library. That is a hard
rule, not a stylistic one: the safety kernel and the provenance ledger have to
be reviewable, testable and runnable in isolation from the transport that
happens to be carrying commands this year.
"""
