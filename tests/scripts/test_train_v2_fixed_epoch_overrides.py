"""Fixed-epoch training overrides for the full-data (production) refit.

A CV run picks its epoch on a held-out val split. A full-data refit trains on
train+val+test, so there is no honest val to stop on: the epoch budget has to be
transferred from CV and held fixed, with early stopping switched off so the run
actually reaches it.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "train_head", REPO / "scripts" / "train_head.py"
)
train_v2 = importlib.util.module_from_spec(spec)
sys.modules["train_v2_ablation"] = train_v2
spec.loader.exec_module(train_v2)

apply_fixed_epoch_overrides = train_v2.apply_fixed_epoch_overrides


def _cfg():
    return {"max_epochs": 50, "early_stopping_patience": 15,
            "checkpoint_every_n_epochs": 5}


def test_noop_when_neither_flag_is_set():
    cfg = _cfg()
    assert apply_fixed_epoch_overrides(cfg, None, False) == _cfg()


def test_max_epochs_overrides_config():
    cfg = apply_fixed_epoch_overrides(_cfg(), 30, False)
    assert cfg["max_epochs"] == 30
    # untouched unless explicitly disabled
    assert cfg["early_stopping_patience"] == 15


def test_no_early_stopping_makes_patience_unreachable():
    cfg = apply_fixed_epoch_overrides(_cfg(), 30, True)
    assert cfg["max_epochs"] == 30
    assert cfg["early_stopping_patience"] > cfg["max_epochs"]


def test_no_early_stopping_without_max_epochs_uses_config_budget():
    cfg = apply_fixed_epoch_overrides(_cfg(), None, True)
    assert cfg["max_epochs"] == 50
    assert cfg["early_stopping_patience"] > 50


def test_checkpoint_grid_is_not_silently_changed():
    # the deliverable is the ckpt at the fixed epoch, so the save grid must stay
    # whatever the config says — the caller is responsible for aligning them.
    cfg = apply_fixed_epoch_overrides(_cfg(), 30, True)
    assert cfg["checkpoint_every_n_epochs"] == 5


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_max_epochs(bad):
    with pytest.raises(ValueError, match="max-epochs"):
        apply_fixed_epoch_overrides(_cfg(), bad, False)


def test_cli_exposes_both_flags():
    argv = ["--seed", "42", "--variant-id", "LC1", "--max-epochs", "25",
            "--no-early-stopping"]
    old = sys.argv
    try:
        sys.argv = ["train_v2_ablation.py"] + argv
        args = train_v2.parse_args()
    finally:
        sys.argv = old
    assert args.max_epochs == 25
    assert args.no_early_stopping is True


def test_flags_default_to_inactive():
    old = sys.argv
    try:
        sys.argv = ["train_v2_ablation.py", "--seed", "42", "--variant-id", "LC1"]
        args = train_v2.parse_args()
    finally:
        sys.argv = old
    assert args.max_epochs is None
    assert args.no_early_stopping is False
