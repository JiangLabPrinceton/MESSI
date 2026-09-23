"""RF-Refine Fusion: Head-guided, structure-constrained edit-and-repair reference flow.

Scientific source: ``doc/RF-Refine-Fusion.md``. Execution contract:
``PLAN_RF_REFINE_FUSION.md``.

Import discipline (PLAN F1): this package ``__init__`` must stay lightweight — it must
NOT eagerly import torch, ESMFold, Head checkpoints, or external repos. Only the pure
state/config/objective/selection layers are re-exported here; the torch-backed RF repair
kernel (``moves`` RF adapter), the real oracle wiring (``oracles``), and the runner are
imported lazily by their consumers / the driver.
"""

from __future__ import annotations

# Pure, torch-free layers are safe to re-export from the package root.
from .config import (
    FusionConfig,
    FusionConfigError,
    load_fusion_config,
)
from .state import (
    CandidateEvaluation,
    EliteState,
    FusionStateError,
    ParticleState,
    PopulationState,
    Proposal,
    make_particle_id,
    make_proposal_id,
    sequence_md5,
    uniform_population,
)

__all__ = [
    "CandidateEvaluation",
    "EliteState",
    "FusionConfig",
    "FusionConfigError",
    "FusionStateError",
    "ParticleState",
    "PopulationState",
    "Proposal",
    "load_fusion_config",
    "make_particle_id",
    "make_proposal_id",
    "sequence_md5",
    "uniform_population",
]
