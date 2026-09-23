"""V2F1 steps 1-4: ``fusion_v2.errors`` and ``fusion_v2.identity``.

Build order and first-behaviors come from ``doc/FUSION_V2_Interface_Map.md`` §6; the adversarial
cases referenced in each test name come from ``PLAN_RF_REFINE_FUSION_V2.md`` §7.2.

``identity.py`` mints every V2 identifier and content digest from typed pre-declared fields, so no
ID is ever derived from ``hash()``, row order, sequence similarity, or a score coincidence.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from inverse_folding.reference_flow.fusion import state as v0_state
from inverse_folding.reference_flow.fusion import v1_records
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2.errors import V2Error

D = "a" * 64  # a stand-in 64-hex content digest


def _conditioning(**over):
    kw = dict(
        dplm_checkpoint=D, tokenizer=D, backbone_row=D, coordinate_mask=D,
        entry_config=D, fixed_token_policy=D,
    )
    kw.update(over)
    return ident.make_v2_conditioning(**kw)


def _identity_bundle(**over):
    kw = {role: D for role in ident.OWN_CONDITIONING_FIELDS}
    kw.update(over)
    return ident.V2ConditioningIdentity(base=_conditioning(), **kw)


def _evaluator(**over):
    kw = dict(
        allele="DRB1_0701", score_scale="nats", window_k_min=13, window_k_max=25,
        head_config_hash=D, head_checkpoint_digest=D,
    )
    kw.update(over)
    return ident.HeadEvaluatorIdentity(**kw)


def _binding(**over):
    kw = dict(
        protein_id="5ZHV_B", sequence_md5="0" * 32, sequence_length=244,
        window_grid_digest=D, evaluator=_evaluator(),
    )
    kw.update(over)
    return ident.HeadScoreBinding(**kw)


def _safety_ref(**over):
    kw = dict(
        reference_id="ref:5ZHV_B:wt", reference_label="wt_native", sequence_md5="0" * 32,
        sequence_length=244, reference_content_digest=D, head_binding=_binding(),
        head_score_digest=D, bound_at_depth=0, source_kind="predeclared_external",
    )
    kw.update(over)
    return ident.SafetyReferenceBinding(**kw)


# --------------------------------------------------------------------------------------------
# Step 1 - errors.py
# --------------------------------------------------------------------------------------------

def test_v2_error_is_a_value_error_but_not_a_fusion_state_error():
    """A v0 ``except FusionStateError`` must not swallow a V2 failure (map §6 step 1, §7.2 #30)."""
    assert issubclass(V2Error, ValueError)
    assert not issubclass(V2Error, v0_state.FusionStateError)
    assert not issubclass(v0_state.FusionStateError, V2Error)


def test_identity_error_is_a_v2_error():
    assert issubclass(ident.V2IdentityError, V2Error)


def test_importing_fusion_v2_never_pulls_torch():
    """``fusion_v2`` is pure state algebra; anything needing torch lives in the sampler layer.

    Asserted on ``torch`` only. ``numpy`` is unavoidably present because
    ``reference_flow/__init__.py`` eagerly imports ``.amplification`` (map §1 correction).
    """
    code = (
        "import sys;"
        "import inverse_folding.reference_flow.fusion_v2.identity as m;"
        "import inverse_folding.reference_flow.fusion_v2.errors as e;"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


# --------------------------------------------------------------------------------------------
# Step 2 - digests
# --------------------------------------------------------------------------------------------

def test_canonical_digest_is_byte_identical_to_the_v1_encoder():
    """A second encoder would digest the same payload differently and fork V1/V2 identities."""
    payload = {"b": 2, "a": [1, {"z": None, "y": "x"}]}
    assert ident.canonical_digest(payload) == v1_records.sha256_hex(
        v1_records.canonical_json_bytes(payload)
    )
    assert ident.canonical_json_bytes is v1_records.canonical_json_bytes
    assert ident.sha256_hex is v1_records.sha256_hex


def test_canonical_digest_is_insensitive_to_key_insertion_order():
    assert ident.canonical_digest({"a": 1, "b": 2}) == ident.canonical_digest({"b": 2, "a": 1})


@pytest.mark.parametrize("bad", ["", "unset:backbone", "unset:", 0, None, b"x" * 64, 1.0, True])
def test_require_digest_fails_closed_on_missing_or_placeholder(bad):
    """PLAN §5.2 'Missing content identity fails closed' (§7.2 #28)."""
    with pytest.raises(ident.V2IdentityError):
        ident.require_digest(bad, "backbone")


def test_require_digest_returns_the_value_when_valid():
    assert ident.require_digest(D, "backbone") == D


# --------------------------------------------------------------------------------------------
# Step 3 - IDs and lineage
# --------------------------------------------------------------------------------------------

def test_ids_bind_their_content_digest_and_detect_a_payload_edit():
    """§7.2 #7: an ID that survived an edit to its payload must not validate."""
    sid = ident.make_live_state_id("5ZHV_B", "fam0", depth=0, step=50, content_digest=D)
    ident.assert_id_binds_digest(sid, D)
    with pytest.raises(ident.V2IdentityError):
        ident.assert_id_binds_digest(sid, "b" * 64)


def test_projected_id_is_rejected_where_a_live_id_is_expected():
    """Five typed layers (PLAN §2.2); a namespace token makes the confusion unrepresentable."""
    live = ident.make_live_state_id("5ZHV_B", "fam0", depth=1, step=95, content_digest=D)
    proj = ident.make_projected_state_id("5ZHV_B", "fam0", depth=1, r_step=90, content_digest=D)
    assert ident.id_namespace(live) == "live"
    assert ident.id_namespace(proj) == "proj"
    assert live != proj
    with pytest.raises(ident.V2IdentityError):
        ident.require_id_namespace(proj, "live")


def test_archive_id_wraps_the_endpoint_identity_without_a_second_hash():
    """PLAN §2.3: the archive entry *is* the endpoint identity."""
    src = ident.make_live_state_id("5ZHV_B", "fam0", depth=0, step=50, content_digest=D)
    ep = ident.make_endpoint_id(src, fork_index=2, sequence_md5_hex="0" * 32)
    arch = ident.make_archive_entry_id(ep)
    assert ident.id_namespace(ep) == "endpoint"
    assert ident.id_namespace(arch) == "archive"
    assert arch.endswith(ep)


E = "b" * 64

#: (minter, baseline kwargs) -- every keyword is a declared component of the ID grammar.
_MINTERS = {
    "live": (ident.make_live_state_id,
             dict(protein_id="5ZHV_B", family_id="fam0", depth=1, step=95, content_digest=D)),
    "proj": (ident.make_projected_state_id,
             dict(protein_id="5ZHV_B", family_id="fam0", depth=1, r_step=90, content_digest=D)),
    "txn": (ident.make_transition_id,
            dict(protein_id="5ZHV_B", family_id="fam0", depth=1, r_step=90, c_next_step=97,
                 content_digest=D)),
}
_ALT = {"protein_id": "9L2Q_A", "family_id": "fam1", "depth": 2, "step": 96, "r_step": 91,
        "c_next_step": 98, "content_digest": E}


def _mint(minter, kw):
    kw = dict(kw)
    return minter(kw.pop("protein_id"), kw.pop("family_id"), **kw)


@pytest.mark.parametrize("kind", sorted(_MINTERS))
def test_every_declared_id_component_is_load_bearing(kind):
    """Each component must actually reach the identifier.

    ``depth`` and the sampler coordinate are the load-bearing pair: under
    ``stationary_checkpoint`` recurrence two feedback depths legitimately share a sampler step and
    must not alias (§7.2 #10). A test that only varies ``depth`` would let every coordinate be
    dropped from the grammar unnoticed.
    """
    minter, base = _MINTERS[kind]
    baseline = _mint(minter, base)
    ids = {"baseline": baseline}
    for component in base:
        variant = _mint(minter, {**base, component: _ALT[component]})
        assert variant != baseline, f"{kind} id ignores {component!r}"
        ids[component] = variant
    ident.assert_no_id_collisions(ids)


def test_endpoint_fork_index_and_sequence_are_load_bearing():
    src = ident.make_live_state_id("5ZHV_B", "fam0", depth=0, step=50, content_digest=D)
    base = ident.make_endpoint_id(src, fork_index=0, sequence_md5_hex="0" * 32)
    assert base != ident.make_endpoint_id(src, fork_index=1, sequence_md5_hex="0" * 32)
    assert base != ident.make_endpoint_id(src, fork_index=0, sequence_md5_hex="1" * 32)


@pytest.mark.parametrize("bad", [":", "a:b", "live", "proj", "endpoint", "txn", "archive"])
@pytest.mark.parametrize("leaf", ["protein_id", "family_id"])
def test_leaf_id_components_may_not_carry_the_delimiter_or_a_namespace_token(leaf, bad):
    """Otherwise ``(protein, family, depth, step, digest) -> id`` is not injective.

    ``5ZHV:B``/``fam0`` and ``5ZHV``/``B:fam0`` would mint one identifier, merging two lineages in
    every downstream join; and a component equal to ``archive`` could inject the archive prefix.
    """
    minter, base = _MINTERS["live"]
    with pytest.raises(ident.V2IdentityError):
        _mint(minter, {**base, leaf: bad})


@pytest.mark.parametrize("bad_digest", ["live:1234567890ab", "endpoint:abcdef", "abc:defghijkl",
                                        "short", "a" * 11])
def test_a_digest_that_breaks_the_id_grammar_is_refused_at_mint_time(bad_digest):
    """``require_digest`` deliberately admits non-SHA-256 identities (a git revision is legal), so
    the ID grammar -- not the digest domain -- is where a ``:`` must be refused.

    Without this, a projected-state ID minted with a ``live:``-prefixed digest reports namespace
    ``live`` and is accepted where a live-state ID is required: exactly the PLAN §2.2 five-layer
    confusion the namespace token exists to prevent.
    """
    minter, base = _MINTERS["proj"]
    with pytest.raises(ident.V2IdentityError):
        _mint(minter, {**base, "content_digest": bad_digest})


def test_endpoint_must_be_minted_over_a_live_state():
    """An endpoint forks from a committed live state, never from a projected state or a
    transition; otherwise a Layer-4 object is archived as a Layer-2 one (PLAN §2.2, §2.3)."""
    live = ident.make_live_state_id("5ZHV_B", "fam0", depth=0, step=50, content_digest=D)
    ident.make_endpoint_id(live, fork_index=0, sequence_md5_hex="0" * 32)
    for wrong in (
        ident.make_projected_state_id("5ZHV_B", "fam0", depth=0, r_step=40, content_digest=D),
        ident.make_transition_id("5ZHV_B", "fam0", depth=0, r_step=40, c_next_step=50,
                                 content_digest=D),
    ):
        with pytest.raises(ident.V2IdentityError):
            ident.make_endpoint_id(wrong, fork_index=0, sequence_md5_hex="0" * 32)


def test_transition_id_carries_both_reentry_and_next_checkpoint():
    t = ident.make_transition_id(
        "5ZHV_B", "fam0", depth=0, r_step=90, c_next_step=97, content_digest=D
    )
    assert ident.id_namespace(t) == "txn"
    ident.assert_id_binds_digest(t, D)
    other = ident.make_transition_id(
        "5ZHV_B", "fam0", depth=0, r_step=90, c_next_step=98, content_digest=D
    )
    assert t != other


def test_assert_no_id_collisions_reports_the_duplicated_id():
    ident.assert_no_id_collisions({"a": "x:1", "b": "x:2"})
    with pytest.raises(ident.V2IdentityError, match="x:1"):
        ident.assert_no_id_collisions({"a": "x:1", "b": "x:1"})


def test_depth_zero_lineage_must_have_no_parent_edges():
    """PLAN §2.2/§5.3: every edge replayable, never guessed from sequence."""
    ok = ident.LineageRef(
        protein_id="5ZHV_B", root_id="5ZHV_B:v2:d0:r0", family_id="fam0", depth=0,
        parent_state_id=None, parent_transition_id=None, origin_endpoint_id=None,
    )
    assert ok.depth == 0
    with pytest.raises(ident.V2IdentityError):
        ident.LineageRef(
            protein_id="5ZHV_B", root_id="r", family_id="fam0", depth=0,
            parent_state_id="live:x", parent_transition_id=None, origin_endpoint_id=None,
        )


@pytest.mark.parametrize(
    "missing", ["parent_state_id", "parent_transition_id", "origin_endpoint_id"]
)
def test_depth_positive_lineage_requires_every_parent_edge(missing):
    kw = dict(
        protein_id="5ZHV_B", root_id="r", family_id="fam0", depth=1,
        parent_state_id="s", parent_transition_id="t", origin_endpoint_id="e",
    )
    kw[missing] = None
    with pytest.raises(ident.V2IdentityError):
        ident.LineageRef(**kw)


def test_lineage_family_id_is_required_and_never_derived():
    """Map OQ7: ``family_id`` is a declared input; fusion_v2 may not derive it."""
    with pytest.raises(ident.V2IdentityError):
        ident.LineageRef(
            protein_id="5ZHV_B", root_id="r", family_id="", depth=0,
            parent_state_id=None, parent_transition_id=None, origin_endpoint_id=None,
        )


# --------------------------------------------------------------------------------------------
# Step 4 - Head evaluator / score binding / safety reference
# --------------------------------------------------------------------------------------------

def test_head_evaluator_digest_changes_when_only_the_window_grid_changes():
    """§7.2 #18: 'Head identities differ' must be detectable independently of a sequence change."""
    assert _evaluator().digest() != _evaluator(window_k_max=24).digest()
    assert _evaluator().digest() != _evaluator(window_k_min=12).digest()
    assert _evaluator().digest() != _evaluator(allele="DRB1_0401").digest()
    assert _evaluator().digest() == _evaluator().digest()


def test_head_score_binding_separates_score_identity_from_evaluator_identity():
    """Map Conflict 6: collapsing the two makes the adversarial case untestable.

    The property is that two bindings over the *same sequence* differ when only the *evaluator*
    differs -- which is what makes "the Head identities differ" detectable independently of a
    sequence change (§7.2 #18).
    """
    same_seq_other_evaluator = _binding(evaluator=_evaluator(window_k_max=24))
    assert same_seq_other_evaluator != _binding()
    assert same_seq_other_evaluator.sequence_md5 == _binding().sequence_md5


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("sequence_md5", "NOT-AN-MD5"),
        ("sequence_md5", "A" * 32),      # uppercase hex is not the canonical join key
        ("sequence_md5", "0" * 31),
        ("sequence_md5", ""),
        ("sequence_length", 0),
        ("sequence_length", -1),
        ("sequence_length", True),
        ("window_grid_digest", ""),
        ("window_grid_digest", "unset:grid"),
        ("protein_id", ""),
        ("evaluator", {"allele": "x"}),
        ("evaluator", None),
    ],
)
def test_head_score_binding_fails_closed(field, bad):
    with pytest.raises(ident.V2IdentityError):
        _binding(**{field: bad})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("reference_id", ""),
        ("reference_label", ""),
        ("source_kind", ""),
        ("reference_content_digest", ""),
        ("reference_content_digest", "unset:ref"),
        ("head_score_digest", ""),
        ("head_score_digest", "unset:hs"),
        ("head_binding", None),
        ("head_binding", "not-a-binding"),
        ("bound_at_depth", 1),
        ("bound_at_depth", -1),
        ("bound_at_depth", 0.0),        # a float zero forks the record digest
        ("bound_at_depth", False),      # bool is not an int declaration here
        ("bound_at_depth", "0"),
    ],
)
def test_safety_reference_binding_fails_closed(field, bad):
    with pytest.raises(ident.V2IdentityError):
        _safety_ref(**{field: bad})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("policy_id", ""),
        ("policy_version", ""),
        ("policy_config_digest", ""),
        ("policy_config_digest", "unset:p"),
        ("policy_spec_digest", ""),
        ("is_diagnostic_only", "false"),
        ("is_diagnostic_only", 0),
        ("is_diagnostic_only", 1),
        ("is_diagnostic_only", None),
    ],
)
def test_projection_policy_identity_fails_closed(field, bad):
    kw = dict(policy_id="p", policy_version="1", policy_config_digest=D,
              policy_spec_digest=D, is_diagnostic_only=False)
    kw[field] = bad
    with pytest.raises(ident.V2IdentityError):
        ident.ProjectionPolicyIdentity(**kw)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("allele", ""),
        ("score_scale", ""),
        ("window_k_min", 0),
        ("window_k_max", 0),
        ("window_k_min", 26),          # min > max
        ("window_k_min", True),
        ("head_config_hash", ""),
        ("head_checkpoint_digest", "unset:x"),
    ],
)
def test_head_evaluator_identity_fails_closed(field, bad):
    with pytest.raises(ident.V2IdentityError):
        _evaluator(**{field: bad})


@pytest.mark.parametrize("edge", ["parent_state_id", "parent_transition_id", "origin_endpoint_id"])
def test_depth_positive_lineage_rejects_an_empty_or_mistyped_edge(edge):
    """A blank edge is not an edge, and each edge is layer-typed: the transition edge must name a
    transition and the origin edge must name an endpoint (map C3). Without the type check any layer
    could occupy any edge, and all three could be one string."""
    live = ident.make_live_state_id("5ZHV_B", "fam0", depth=0, step=50, content_digest=D)
    txn = ident.make_transition_id("5ZHV_B", "fam0", depth=0, r_step=40, c_next_step=50,
                                   content_digest=D)
    endpoint = ident.make_endpoint_id(live, fork_index=0, sequence_md5_hex="0" * 32)
    good = dict(protein_id="5ZHV_B", root_id="r", family_id="fam0", depth=1,
                parent_state_id=live, parent_transition_id=txn, origin_endpoint_id=endpoint)
    ident.LineageRef(**good)  # the well-typed lineage constructs
    for bad in ("", live if edge != "parent_state_id" else txn):
        with pytest.raises(ident.V2IdentityError):
            ident.LineageRef(**{**good, edge: bad})


def test_safety_reference_must_be_bound_at_depth_zero():
    """PLAN §2.7: content-bound before any depth-0 endpoint is scored; immutable thereafter."""
    assert _safety_ref().bound_at_depth == 0
    with pytest.raises(ident.V2IdentityError):
        _safety_ref(bound_at_depth=1)


def test_safety_reference_length_must_match_its_head_binding():
    """A differing-length reference is structurally inadmissible: ``objective._aligned_z``
    (``objective.py:96-98``) raises on differing window-coordinate sets, and the grid for a fixed
    k-range is determined by sequence length (map OQ5)."""
    with pytest.raises(ident.V2IdentityError):
        _safety_ref(sequence_length=243)
    with pytest.raises(ident.V2IdentityError):
        _safety_ref(sequence_md5="1" * 32)


def test_projection_policy_identity_requires_an_explicit_diagnostic_flag():
    """PLAN §2.5: ``explicit_probe`` may never become a silent production default."""
    ok = ident.ProjectionPolicyIdentity(
        policy_id="source_writeback_v1", policy_version="1",
        policy_config_digest=D, policy_spec_digest=D, is_diagnostic_only=False,
    )
    assert ok.is_diagnostic_only is False
    with pytest.raises(ident.V2IdentityError):
        ident.ProjectionPolicyIdentity(
            policy_id="p", policy_version="1", policy_config_digest=D,
            policy_spec_digest=D, is_diagnostic_only="false",  # a coercible str is not a declaration
        )


# --------------------------------------------------------------------------------------------
# Conditioning identity: h-maps are obsolete -- asserted, never declared
# --------------------------------------------------------------------------------------------

def test_v2_conditioning_never_asks_the_caller_to_declare_h_maps():
    """h-maps are retired. The substrate guarantee stays; the declaration burden does not.

    ``make_v2_conditioning`` fixes ``controller_enabled``/``h_maps_present`` internally, so no V2
    config surface, artifact, or call site carries an h-map field.
    """
    cd = _conditioning()
    assert cd.controller_enabled is False
    assert cd.h_maps_present is False
    for banned in ("h_maps_present", "controller_enabled", "h_source", "h_maps"):
        with pytest.raises(TypeError):
            ident.make_v2_conditioning(
                dplm_checkpoint=D, tokenizer=D, backbone_row=D, coordinate_mask=D,
                entry_config=D, fixed_token_policy=D, **{banned: False},
            )


def test_v2_conditioning_identity_rejects_a_placeholder_in_any_role():
    _identity_bundle()  # all real digests -> constructs
    for role in ident.OWN_CONDITIONING_FIELDS:
        with pytest.raises(ident.V2IdentityError):
            _identity_bundle(**{role: "unset:x"})


#: The PLAN §5.2 role set, spelled out so an omission fails. Asserted as a SUBSET of the
#: implementation's keys because PLAN §5.2 says "at least" -- additions are legal, omissions are not.
_REQUIRED_CONTENT_ROLES = frozenset({
    "cohort_table", "reference_sequences", "backbone", "coordinate_mask", "constraint_manifest",
    "rf_sampler_config", "tokenizer", "dplm_checkpoint", "fixed_token_policy", "head_config",
    "head_checkpoint", "structure_backend", "structure_config", "v0_structure_gate_config",
    "projection_policy_spec", "schedule_band_calibration", "complete_reference_sequence",
    "code_revision",
})


def test_v2_conditioning_identity_carries_every_content_role():
    """PLAN §5.2 says 'at least'; additions are legal, omissions are not.

    The earlier version of this test asserted ``set(values) <= covered``, which gets *easier* as
    roles vanish: three roles and the whole frozen set could be deleted with the suite green. The
    required-role set is therefore pinned literally.
    """
    assert _REQUIRED_CONTENT_ROLES <= set(ident.CONTENT_ROLE_TO_FIELD)
    covered = set(ident.OWN_CONDITIONING_FIELDS) | set(ident.BASE_CONDITIONING_FIELDS)
    assert set(ident.CONTENT_ROLE_TO_FIELD.values()) <= covered
    assert ident.FROZEN_DIGEST_ROLES == {
        "projection_policy_spec", "schedule_band_calibration", "complete_reference_sequence",
    }
    assert "h_maps" not in ident.CONTENT_ROLE_TO_FIELD


# --------------------------------------------------------------------------------------------
# Digest completeness: every declared field must reach the digest it claims to identify
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field", ident.OWN_CONDITIONING_FIELDS)
def test_every_own_conditioning_field_reaches_the_run_signature(field):
    """Otherwise two runs with different scientific inputs share one signature.

    Without this sweep the whole payload projection can be gutted: ``canonical_payload`` collapses
    to ``{}``, every run digests to the SHA-256 of the empty object, and a Head-checkpoint swap is
    indistinguishable from a DPLM swap.
    """
    assert _identity_bundle(**{field: E}).digest() != _identity_bundle().digest()


@pytest.mark.parametrize("dotted", ident.BASE_CONDITIONING_FIELDS)
def test_every_base_conditioning_field_reaches_the_run_signature(dotted):
    name = dotted.split(".", 1)[1]
    bundle = ident.V2ConditioningIdentity(
        base=_conditioning(**{name: E}), **{r: D for r in ident.OWN_CONDITIONING_FIELDS}
    )
    assert bundle.digest() != _identity_bundle().digest()


def test_conditioning_digest_is_stable_and_schema_tagged():
    assert _identity_bundle().digest() == _identity_bundle().digest()
    assert _identity_bundle().canonical_payload()["schema"] == ident.V2_STATE_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("field", "other"),
    [("allele", "DRB1_0401"), ("score_scale", "logit"), ("window_k_min", 12),
     ("window_k_max", 24), ("head_config_hash", E), ("head_checkpoint_digest", E)],
)
def test_every_head_evaluator_field_reaches_its_digest(field, other):
    assert _evaluator(**{field: other}).digest() != _evaluator().digest()


@pytest.mark.parametrize(
    ("field", "other"),
    [("protein_id", "9L2Q_A"), ("sequence_md5", "1" * 32), ("sequence_length", 245),
     ("window_grid_digest", E), ("evaluator", _evaluator(window_k_max=24))],
)
def test_every_head_score_binding_field_reaches_its_digest(field, other):
    assert _binding(**{field: other}).digest() != _binding().digest()


@pytest.mark.parametrize(
    ("field", "other"),
    [("reference_id", "ref:other"), ("reference_label", "external"),
     ("reference_content_digest", E), ("head_score_digest", E),
     ("source_kind", "frozen_calibration")],
)
def test_every_safety_reference_field_reaches_its_digest(field, other):
    assert _safety_ref(**{field: other}).digest() != _safety_ref().digest()


@pytest.mark.parametrize(
    ("field", "other"),
    [("policy_id", "other"), ("policy_version", "2"),
     ("policy_config_digest", E), ("policy_spec_digest", E),
     ("is_diagnostic_only", True)],
)
def test_every_projection_policy_field_reaches_its_digest(field, other):
    base = dict(
        policy_id="p", policy_version="1", policy_config_digest=D,
        policy_spec_digest=D, is_diagnostic_only=False,
    )
    assert ident.ProjectionPolicyIdentity(**{**base, field: other}).digest() != (
        ident.ProjectionPolicyIdentity(**base).digest()
    )


class _W:
    """Minimal structural stand-in for ``head_scoring.WindowRiskRecord`` (no torch)."""

    def __init__(self, start_0b, end_0b, k, z=0.0):
        self.start_0b, self.end_0b, self.k, self.z = start_0b, end_0b, k, z


def test_window_grid_digest_is_order_invariant_but_coordinate_sensitive():
    grid = [_W(0, 13, 13), _W(1, 14, 13), _W(0, 25, 25)]
    assert ident.window_grid_digest(grid) == ident.window_grid_digest(list(reversed(grid)))
    assert ident.window_grid_digest(grid) != ident.window_grid_digest(grid[:2])
    # same span, different k -- a real distinction in the Head's window enumeration
    assert ident.window_grid_digest([_W(0, 13, 13)]) != ident.window_grid_digest([_W(0, 13, 12)])


def test_window_grid_digest_refuses_a_duplicated_coordinate():
    """The digest pre-checks ``objective.aligned_window_z``, which keys a *dict* by coordinate and
    silently keeps the last row. A multiset digest therefore disagrees with the comparator in both
    directions: two grids whose duplicated rows carry swapped ``z`` digest identically and then
    align to a fabricated per-window delta, while ``[A, A, B]`` and ``[A, B]`` -- the same
    coordinate set, which the comparator aligns fine -- digest differently and are wrongly refused.

    ``head_scoring`` emits one record per parquet row with no dedup, so a duplicate is reachable
    input, not a hypothetical. Fail closed rather than dedupe: silently collapsing would let a
    score carrying two conflicting ``z`` at one coordinate reach the cumulative hotspot gate
    unflagged.
    """
    with pytest.raises(ident.V2IdentityError, match="duplicate"):
        ident.window_grid_digest([_W(0, 13, 13), _W(0, 13, 13), _W(1, 14, 13)])


def test_window_grid_digest_refuses_an_empty_grid():
    with pytest.raises(ident.V2IdentityError):
        ident.window_grid_digest([])


@pytest.mark.parametrize("blank", ["   ", "\t", "\n", " \n "])
def test_require_digest_rejects_whitespace_only(blank):
    with pytest.raises(ident.V2IdentityError):
        ident.require_digest(blank, "backbone")


@pytest.mark.parametrize("placeholder", ["UNSET:backbone", "Unset:x", "  unset:x  "])
def test_require_digest_rejects_case_varied_and_padded_placeholders(placeholder):
    """``unset:`` is the project's "absent" spelling; a capital U is the same absence."""
    with pytest.raises(ident.V2IdentityError):
        ident.require_digest(placeholder, "backbone")
