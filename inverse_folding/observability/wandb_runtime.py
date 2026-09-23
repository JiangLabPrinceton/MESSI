"""Optional wandb integration for Phase C scripts.

Designed so the rest of the code never has to know whether wandb is installed,
enabled, or disabled — every hook is a safe no-op when ``run is None``.

CLI conventions:
  --wandb                       opt-in switch (off by default)
  --wandb-project NAME          project for the stage (each stage gets its own)
  --wandb-mode online|offline|disabled
  --wandb-group GROUP           optional run grouping
  --wandb-tags a,b,c            optional run tags (comma-separated)
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - import guard exercised on cluster only
    import wandb as _wandb

    _AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means wandb unavailable
    _wandb = None
    _AVAILABLE = False


def add_wandb_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_project: str,
    default_mode: str = "online",
) -> None:
    """Register the standard wandb CLI surface on a parser."""
    group = parser.add_argument_group("wandb")
    group.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb monitoring (off by default).",
    )
    group.add_argument(
        "--wandb-project",
        default=default_project,
        help=f"wandb project name (default: {default_project}).",
    )
    group.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=default_mode,
        help="wandb mode; 'disabled' suppresses init even when --wandb is passed.",
    )
    group.add_argument(
        "--wandb-group",
        default=None,
        help="Optional wandb run group (e.g. allele tag for cross-arm comparison).",
    )
    group.add_argument(
        "--wandb-tags",
        default=None,
        help="Comma-separated wandb tags.",
    )


def init_wandb_from_args(
    args: argparse.Namespace,
    *,
    run_name: str,
    config: dict[str, Any],
    extra_tags: Sequence[str] | None = None,
) -> Any:
    """Init a wandb run from parsed args.

    Returns the wandb run object, or ``None`` when wandb is off / unavailable.
    Callers should treat the return value opaquely and pass it back to the
    other helpers in this module.
    """
    enabled = bool(getattr(args, "wandb", False))
    mode = str(getattr(args, "wandb_mode", "online"))
    if not enabled or mode == "disabled":
        return None
    if not _AVAILABLE:
        print(
            "[wandb] package not installed; --wandb flag is a no-op",
            flush=True,
        )
        return None

    tags: list[str] = []
    raw_tags = getattr(args, "wandb_tags", None)
    if raw_tags:
        tags.extend([t.strip() for t in str(raw_tags).split(",") if t.strip()])
    if extra_tags:
        tags.extend(extra_tags)

    project = str(getattr(args, "wandb_project"))
    group = getattr(args, "wandb_group", None)

    run = _wandb.init(
        project=project,
        name=run_name,
        config=config,
        mode=mode,
        group=group,
        tags=_dedup_preserving_order(tags),
        reinit=True,
    )
    print(
        f"[wandb] active project={project} run={run_name} mode={mode} "
        f"group={group} tags={tags}",
        flush=True,
    )
    return run


def is_active(run: Any) -> bool:
    return run is not None


def log_metrics(run: Any, metrics: dict[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    payload = {k: v for k, v in metrics.items() if v is not None}
    if not payload:
        return
    run.log(payload, step=step)


def set_summary(run: Any, summary: dict[str, Any]) -> None:
    if run is None:
        return
    for key, value in summary.items():
        if value is None:
            continue
        try:
            run.summary[key] = value
        except Exception:  # noqa: BLE001 - never break the host script on telemetry
            pass


def finish_wandb(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:  # noqa: BLE001
        pass


def _dedup_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def wandb_dir_default() -> str | None:
    """Resolve the default WANDB_DIR (env-driven; helper for diagnostics only)."""
    return os.environ.get("WANDB_DIR")
