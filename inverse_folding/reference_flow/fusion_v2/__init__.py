"""RF-Refine Fusion V2: trajectory-coupled pre-terminal exact-endpoint feedback.

Scientific authority: ``doc/FUSION_V2.md``. Implementation contract:
``PLAN_RF_REFINE_FUSION_V2.md``. Frozen module map: ``doc/FUSION_V2_Interface_Map.md``.

Hard architectural boundaries for every module in this package:

1. **No torch, no model.** This package is pure state algebra over typed records. The sampler
   segment and the Head/structure oracles are injected by the caller. Anything that needs torch
   lives in the sampler layer.
2. **One-way dependency.** ``fusion_v2`` may import v0 ``fusion`` primitives it genuinely reuses;
   ``fusion`` must never import ``fusion_v2``. Pinned by
   ``tests/inverse_folding/test_fusion_v2_reuse_boundary.py``.
3. **No scientific defaults.** A numeric field whose value is an open scientific decision is a
   required input with no default; missing means fail closed, never a guess.

Submodules are imported explicitly by consumers; this file intentionally re-exports nothing, so an
import of one leaf module cannot drag in the whole package.
"""

from __future__ import annotations
