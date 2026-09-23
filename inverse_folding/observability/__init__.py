"""Optional observability hooks for IF-stage scripts (wandb)."""

from .wandb_runtime import (
    add_wandb_cli_args,
    finish_wandb,
    init_wandb_from_args,
    is_active,
    log_metrics,
    set_summary,
)

__all__ = [
    "add_wandb_cli_args",
    "finish_wandb",
    "init_wandb_from_args",
    "is_active",
    "log_metrics",
    "set_summary",
]
