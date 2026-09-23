"""Inference runtime for Epitope Head (Module F)."""

from .predictor import InferencePredictor, build_epitope_scorer_from_config

__all__ = ["InferencePredictor", "build_epitope_scorer_from_config"]
