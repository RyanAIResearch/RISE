"""RISE: Readout Influence Sketching Estimator.

Scalable, forward-only data attribution and valuation for LLMs. Part of the
Hammer Engine research line.

Layers:
    rise.runtime  -- trunks (hidden-state providers), length-sorted batching, prefetch
    rise.head     -- LM-head influence sketching (the estimator itself)
    rise.index    -- sharded, resumable, memory-mapped signature index
    rise.search   -- streaming exhaustive top-k and full scoring over an index
    rise.pipeline -- build / query pipelines tying the layers together
"""

__version__ = "0.1.0.dev0"

from .config import RiseConfig  # noqa: E402,F401
