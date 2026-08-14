"""vdb - local retrieval over the captain's own Claude Code history.

Phase 1: ingest, cleaning, message-boundary chunking, a BM25 keyword index and
an explicit query command. No model, no vectors, no network, no dependencies
beyond the Python standard library.

Built to report `vdbqual` §13. Phase 2 adds the dense index and reciprocal-rank
fusion; it is not stubbed here.
"""

__version__ = "0.1.0"
