"""The one exception base for ``fusion_v2``.

Separate module so ``seeds.py`` can raise a catchable V2 error without importing ``identity`` or
``state`` -- the structural form of the PLAN §2.6 exclusion law, which keeps treatment content out
of the seed layer.

``V2Error`` derives from ``ValueError`` (so ordinary callers keep working) but deliberately **not**
from ``fusion.state.FusionStateError``: a v0 ``except FusionStateError`` must not silently swallow a
V2 failure.
"""

from __future__ import annotations


class V2Error(ValueError):
    """Base of every ``fusion_v2`` failure."""
