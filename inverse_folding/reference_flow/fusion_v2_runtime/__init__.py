"""V2 runtime: the torch-aware half of the V2 implementation.

Two V2 packages exist on purpose, and the split is load-bearing:

``fusion_v2``
    The typed contract layer (V2F1) and the pure projection kernel (V2F2).  **torch-free and
    model-free**, asserted by test.  That property is what lets ``--print-config`` and ``--dry-run``
    validate an entire campaign -- config, schedule, seeds, coordinates, identities -- with no model
    loaded at all (PLAN §5, V2F7).

``fusion_v2_runtime`` (this package)
    Everything that needs tensors: scoring, the projected-segment executor, capture providers.

**This package never modifies the V1 sampler.**  ``inverse_folding/reference_flow/sampler.py`` is
byte-identical to HEAD; the V1 reference-flow results are published and PLAN V2F3 requires "V1
continuation tests remain byte-identical".  PLAN §3.1 offers "a V2-specific sampler input **or
segment API**" -- this package is the segment-API option, and it is also why PLAN's other rule,
"do not implement V2 by passing ``continuation_resume`` and ``continuation`` simultaneously", is
satisfied trivially: V2 never calls that entry point.

**The step math is imported, never copied.**  A duplicated denoising loop would be the single most
dangerous thing in this codebase: if the V2 loop drifted from the V1 loop in the schedule, the
unmask probability, the remask lifecycle order, or fixed-token enforcement, then (a) A2 would stop
being a matched control of the same process, and (b) the ``B(r)`` band table -- calibrated on the
V1 loop -- would no longer describe the substrate V2 actually runs on.  So this package owns only
CONTROL FLOW (where to start, where to stop, when to assimilate, when protection expires) and
imports every numerical primitive from ``sampler``, ``schedule``, and ``amplification``.
"""

from __future__ import annotations

__all__: list[str] = []
