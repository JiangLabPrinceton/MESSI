"""V2F5A: the runtime adapter behind the policy's leave-one-out counterfactual (PLAN §2.5, A.3).

:func:`fusion_v2.evidence.score_leave_one_out` is pure -- it builds the counterfactual sequences,
matches results by digest and computes ``a_i`` -- but it must not know what a Head request looks
like, must not import torch, and must not reach the cost journal.  This module is the one seam that
does: it turns ``(protein_id, sequences)`` into the same typed :class:`OracleRequest` batch every
other V2 Head call travels through, so the policy's counterfactuals are scored by the SAME frozen
Head, on the SAME grid, as the endpoints they are compared against.

That sameness is the point.  ``a_i`` is a DIFFERENCE of two complete-sequence risks; if the
counterfactual took a second scoring path, the difference would carry whatever those two paths
disagree about, and the sign the policy ranks on would be an artifact of the plumbing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..fusion.state import sequence_md5
from ..fusion_v2.errors import V2Error
from .lookahead import OracleRequest

__all__ = ["V2ContributionError", "CounterfactualHeadScorer"]


class V2ContributionError(V2Error):
    """The counterfactual scorer was wired to something that is not the run's frozen Head."""


@dataclass(frozen=True)
class CounterfactualHeadScorer:
    """Adapt ``head_oracle.score(list[OracleRequest])`` to the pure scorer contract.

    Deliberately thin: it validates that it was handed complete canonical sequences of the donor's
    own length (``OracleRequest`` does the AA20 check itself, and a masked or truncated
    counterfactual would be scored fail-open by a real Head), and otherwise adds nothing.  It
    recomputes no score and caches nothing -- a cache keyed on anything but the exact sequence is
    how a counterfactual acquires another design's risk.
    """

    head_oracle: Any

    def __post_init__(self) -> None:
        if self.head_oracle is None or not hasattr(self.head_oracle, "score"):
            raise V2ContributionError(
                "head_oracle must expose score(list[OracleRequest]); without the frozen Head there "
                "is no applied-dose evidence and the Head-directed policy has nothing to rank on"
            )

    def __call__(self, protein_id: str, sequences: Sequence[str]) -> list[Any]:
        requests = [
            OracleRequest(
                protein_id=str(protein_id), sequence=str(sequence),
                sequence_md5=sequence_md5(str(sequence)), sequence_length=len(str(sequence)),
            )
            for sequence in sequences
        ]
        if not requests:
            return []
        return list(self.head_oracle.score(requests))
