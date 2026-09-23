"""Reuse boundary between frozen v0/V1 modules and the new ``fusion_v2`` package.

Step 0 of the V2F1 build order (``doc/FUSION_V2_Interface_Map.md`` §6). ``fusion_v2`` must reuse
the audited v0/V1 primitives listed in the map's §5 reuse ledger rather than re-deriving them: a
second canonical-JSON encoder, AA20 guard, or bool coercer would fork a digest or re-open a fixed
bug. Those primitives are currently private, so this module pins an additive **public alias** for
each one.

The aliases are zero-behavior: ``public is _private`` object identity, no wrapper, no signature
change, no call-site change. That keeps PLAN §5.1's frozen-system-boundary rule intact while making
the reuse visible instead of reaching across packages for underscore names.

Deliberate deviation from the interface map: the map proposed "alias plus an ``__all__`` entry".
None of the five v0/V1 modules currently defines ``__all__``, so adding one would *narrow*
``from module import *`` from "every public name" to "the listed names" — a real behavior change,
not a zero-behavior alias. Aliases only.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from inverse_folding.reference_flow.fusion import config as v0_config
from inverse_folding.reference_flow.fusion import objective as v0_objective
from inverse_folding.reference_flow.fusion import state as v0_state
from inverse_folding.reference_flow.fusion import v1_records, v1_seeds

# (module, public alias, private original) -- the §5 reuse ledger, one row per aliased symbol.
ALIASES = [
    (v1_records, "canonical_json_bytes", "_canonical_json"),
    (v1_records, "sha256_hex", "_sha256_hex"),
    (v1_records, "require_int_token", "_require_int_token"),
    (v1_records, "normalize_score", "_norm_score"),
    (v0_state, "require_canonical_aa20", "_require_canonical"),
    (v0_state, "require_finite", "_require_finite"),
    (v0_objective, "aligned_window_z", "_aligned_z"),
    (v0_objective, "finite_window_z", "_finite_z"),
    (v0_objective, "window_coord", "_coord"),
    (v1_seeds, "exact_index", "_exact_index"),
    (v0_config, "require_key", "_require"),
    (v0_config, "require_int", "_req_int"),
    (v0_config, "require_float", "_req_float"),
    (v0_config, "require_str", "_req_str"),
    (v0_config, "require_explicit_bool", "_as_bool"),
    (v0_config, "require_submapping", "_sub"),
    (v0_config, "require_finite_nonneg", "_finite_nonneg"),
    (v0_config, "require_finite_pos", "_finite_pos"),
]


@pytest.mark.parametrize(
    ("module", "public", "private"),
    ALIASES,
    ids=[f"{m.__name__.rsplit('.', 1)[-1]}.{pub}" for m, pub, _ in ALIASES],
)
def test_public_alias_is_the_private_symbol(module, public, private):
    """The alias must be the *same object*, not a wrapper with its own semantics."""
    assert hasattr(module, public), (
        f"{module.__name__}.{public} is missing; fusion_v2 would have to import "
        f"the private {private} or re-derive it"
    )
    assert getattr(module, public) is getattr(module, private)


@pytest.mark.parametrize(
    "module",
    [v1_records, v0_state, v0_objective, v1_seeds, v0_config],
    ids=lambda m: m.__name__.rsplit(".", 1)[-1],
)
def test_alias_commit_does_not_introduce_star_export_narrowing(module):
    """Adding ``__all__`` to a module that lacked one silently narrows ``import *``.

    Pins the deviation documented in this module's docstring so a later "tidy-up" cannot turn a
    zero-behavior alias commit into a behavior change.
    """
    assert not hasattr(module, "__all__")


def test_frozen_v0_package_never_imports_fusion_v2():
    """PLAN dependency direction: ``fusion_v2`` may use ``fusion``; ``fusion`` may never use it.

    A fence, not a behavior test: it passes before ``fusion_v2`` exists and must keep passing
    afterwards. Without it the import cycle is only caught the first time someone runs the v0
    suite in isolation.

    Matched on the import graph rather than on the raw text, so a docstring or a reuse comment
    naming ``fusion_v2`` is not an offence -- only an actual dependency edge is.
    """
    pkg = pathlib.Path(v1_records.__file__).parent
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level>0 is a relative import; ``from .fusion_v2 import x`` has module="fusion_v2"
                targets = [node.module or ""] + [a.name for a in node.names]
            else:
                continue
            if any(_names_fusion_v2(t) for t in targets):
                offenders.append(f"{path.relative_to(pkg.parent)}:{node.lineno}")
    assert offenders == []


def _names_fusion_v2(dotted: str) -> bool:
    """True iff ``dotted`` refers to the ``fusion_v2`` package at any position."""
    return "fusion_v2" in dotted.split(".")


@pytest.mark.parametrize(
    "statement",
    [
        "import inverse_folding.reference_flow.fusion_v2",
        "import inverse_folding.reference_flow.fusion_v2.identity",
        "from inverse_folding.reference_flow.fusion_v2 import identity",
        "from inverse_folding.reference_flow.fusion_v2.identity import LineageRef",
        "from ..fusion_v2 import identity",
        "from ..fusion_v2.identity import LineageRef",
        "from . import fusion_v2",
        "import inverse_folding.reference_flow.fusion_v2 as fv2",
    ],
)
def test_import_fence_predicate_catches_every_spelling(statement):
    """The fence is only as good as its matcher.

    An earlier version tested ``t.split('.')[-1] == 'fusion_v2'``, which misses
    ``from ..fusion_v2.identity import X`` -- a dotted relative import, the exact idiom the frozen
    package already uses. Pinning every spelling stops the gap silently reopening.
    """
    tree = ast.parse(statement)
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
    if isinstance(node, ast.Import):
        targets = [a.name for a in node.names]
    else:
        targets = [node.module or ""] + [a.name for a in node.names]
    assert any(_names_fusion_v2(t) for t in targets), statement
