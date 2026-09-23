"""Torch-free backbone/structure path resolution shared by Phase C runtimes and preflight gates.

Extracted from ``reference_flow.runtime`` (which imports torch and inserts the byprot source tree
onto ``sys.path``) so a model-free launch gate can resolve exactly the SAME structure file the run
will later load. Duplicating the candidate order in a validator would be worse than useless: the
dry run would certify a path the run does not use.

``runtime`` re-exports these names, so existing imports and monkeypatch targets are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def safe_file_id(protein_id: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in protein_id)


def infer_chain_id(protein_id: str) -> str | None:
    parts = protein_id.split("_", maxsplit=1)
    if len(parts) == 2 and len(parts[0]) == 4 and parts[0][0].isdigit():
        return parts[1]
    return None


def structure_path_candidates(entry: Mapping[str, Any], pdb_root: str | Path) -> list[Path]:
    """The ordered candidate paths for a test-set row's structure, most specific first."""
    row = dict(entry)
    root = Path(pdb_root)
    protein_id = str(row["protein_id"])
    pdb_path = str(row.get("pdb_path") or "")
    safe_id = safe_file_id(protein_id)

    candidates: list[Path] = []
    raw_path = Path(pdb_path) if pdb_path else None
    if raw_path:
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(root / raw_path)
            candidates.append(root / raw_path.name)
            stem = raw_path.stem
            candidates.append(root / f"{stem}.pdb")
            candidates.append(root / f"{stem}.cif")

    candidates.extend(
        [
            root / f"{protein_id}.pdb",
            root / f"{protein_id}.cif",
            root / f"{safe_id}.pdb",
            root / f"{safe_id}.cif",
        ]
    )

    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def resolve_structure_path(entry: Mapping[str, Any], pdb_root: str | Path) -> Path:
    for candidate in structure_path_candidates(entry, pdb_root):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not resolve structure for protein_id={str(dict(entry)['protein_id'])} "
        f"under pdb_root={Path(pdb_root)}"
    )
