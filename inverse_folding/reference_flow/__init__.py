"""Phase C reference-flow sampling utilities."""

from .amplification import amplification_factor, normalize_h_values, shuffle_h_values
from .config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    ReferenceFlowConfigError,
    RemaskConfig,
    SamplerConfig,
    ScheduleConfig,
    load_reference_flow_config,
    reference_flow_config_to_dict,
    with_reference_flow_overrides,
)
# The torch-backed sampler is imported LAZILY (PEP 562) so importing lightweight submodules
# (e.g. reference_flow.fusion.config / .state) does not eagerly pull torch. Accessing any of the
# names below still works exactly as before — the import fires on first attribute access.
_LAZY_SAMPLER = {"PositionDependentDFMSampler", "SamplerBatchLane", "SamplerOutput"}


def __getattr__(name):  # noqa: D401 - module-level lazy attribute hook
    if name in _LAZY_SAMPLER:
        import importlib

        mod = importlib.import_module(".sampler", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AmplificationConfig",
    "HShuffleConfig",
    "PositionDependentDFMSampler",
    "ReferenceFlowConfig",
    "ReferenceFlowConfigError",
    "RemaskConfig",
    "SamplerConfig",
    "SamplerBatchLane",
    "SamplerOutput",
    "ScheduleConfig",
    "amplification_factor",
    "load_reference_flow_config",
    "normalize_h_values",
    "reference_flow_config_to_dict",
    "shuffle_h_values",
    "with_reference_flow_overrides",
]
