# Configuration Scope

## Official MESSI

`examples/messi_official.json` records the realized single-allele official hardset
recipe, independently of the constrained uricase family experiment. It is a
parameter/provenance reference, not a resolved runtime config.

Use `examples/messi_official.template.yaml` with the existing materializer and
`--exploratory-profile highrisk_d4_k12_r40 --run-max-head-calls 6000`.
The template intentionally has unresolved content bindings: supply the cohort,
references, model files, measured schedule bands, allele-bound policy/hotspot
artifacts and structure configuration through the materializer. Never substitute
dummy digests. The official template retains the actual zero-retry setting; the
older canary template's two retries do not reproduce this campaign.

The common search is D4/K12/r40, 100 steps, temperature 1, checkpoints
50/60/70/80/90, four seeds 20260811-20260814 and scTM >= 0.70. Official selection
returns up to eight designs per protein from the complete feasible archive.

Actual policy scalars differ across the archived allele runs:

| Allele | Donor epsilon | Local tolerance | Write fraction |
|---|---:|---:|---:|
| DRB0701 | 0.0000019073486328125 | 0.0000019073486328125 | 0.05 |
| DRB0401 | 0.005 | 0.017012596130371094 | 0.05 |
| DRB1501 | 0.005 | 0.017012596130371094 | 0.05 |

These values were checked across all 3,000 original single-allele cell configs.
They preserve the realized runs, rather than silently applying the earlier
protocol's uniform thresholds. The returned 1501 panel also contains a separately
repaired cell; this recipe describes the original campaign, not its later repair.
External input identities must be rematerialized for the published DPLM package.

Required shared files are the sampler, policy-v2 specification, structure-gate
config, Head inference configs and the materializer template. Canary/probe and
dual fixtures still used by retained tests are not alternative official recipes.

## Selected Target Constraints

`examples/constraints/manifest.json` lists the only selected target manifests:
LuxSit-i (22 fixed positions), PD1_b2 (57 fixed positions), and ten DRB1501
uricases. Each uricase file preserves its original bytes, full reference sequence,
sequence MD5, free positions, hard anchors and scientific provenance. LuxSit-i
and PD1 entries are unchanged; only unavailable path/history annotations were
removed from their publication copies.

Do not project these manifests onto another sequence or allele. Uricase masks
were prepared using NMP-core-informed P1/P4/P6/P9 opening; that is a separate
input-preparation step, not NMP use inside the Head-guided search.

`examples/uricase_family_historical.json` separately records the family's
D3/K12/r40, two-root, local-tolerance-zero experiment. Its production profile
exists at the recorded source revision, not in this release's materializer.
Do not substitute a pilot or the official D4 profile to claim reproduction.
Only five of the ten selected uricases occur in that 2,539-parent production
launch; `family_production_cells=0` means absent from that launch, not that a
constraint is missing. No new experiments were run to fill the difference.

## Full-Data Head Training

`epitope_head/configs/full_data/drb{0701,0401,1501}.yaml` wraps the exact checkpoint
companion `resolved_config.yaml` under `train`, for use with `--override-config`.
All resolved training values are preserved, not reconstructed from old ablations.
The unchanged model/ablation/inference files remain required for LC1 architecture
and content identity. Supply the allele's own full-data manifest directory.

```bash
python scripts/train_head.py --variant-id LC1 --seed 42 --profile strict \
  --data-dir DATA_DIR --splits-subdir full --device cuda \
  --override-config epitope_head/configs/full_data/drb1501.yaml \
  --max-epochs 25 --no-early-stopping --output-root OUTPUT_DIR
```

Use 30 epochs for 0701 and 35 for 0401. The checkpoint epochs are zero-based:
29/34/24. Full-data training includes the validation IDs in training; validation
is only a monitoring curve. Publish the fixed-epoch checkpoint, never `best.pt`,
and do not report its monitoring score as held-out performance.
