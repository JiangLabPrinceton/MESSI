#!/usr/bin/env python
"""Convert per-design ColabFold a3m -> Protenix homo-tetramer JSON (count=4 + 4 ligands).

For a homo-oligomer, Protenix pairing MSA = query-only (identical chains self-pair);
non_pairing MSA = the full ColabFold a3m. Validated on design_0000 (pairing=1, non_pairing=3076).
"""
import argparse, json, os, sys
import pandas as pd

LIGAND_SMILES = "O=C1NC(=O)C2=C(N1)NC(=O)N2"  # uric acid (CCD URC)


def _protein_chain(seq, count, paired=None, unpaired=None):
    chain = {"sequence": seq, "count": count}
    if paired is not None:
        chain["pairedMsaPath"] = paired
    if unpaired is not None:
        chain["unpairedMsaPath"] = unpaired
    return {"proteinChain": chain}


def build_sequences_block(seq, n_copies, ligands=None, apo=False, ligand_copies=None,
                          paired=None, unpaired=None, partners=None):
    """Protenix `sequences` list: proteinChain(s) + one ligand entity per SMILES.

    ``ligands`` is a LIST so a holo job can carry several distinct entities (e.g. a metal
    cofactor plus a substrate: ``['[Zn+2]', '<adenosine>']``). ``apo=True`` omits all ligands.
    ``ligand_copies`` defaults to ``n_copies``.

    ``partners`` adds further DISTINCT protein chains for a hetero-complex (e.g. a de novo
    binder plus its target): a list of ``{sequence, count?, paired?, unpaired?}``. Each partner
    carries its own MSA paths — Protenix loads MSAs per entity, and a designed binder's a3m is
    query-depth while its target's is deep, so the two must never be shared. Partner chains are
    emitted after the primary chain and before any ligand entity.
    """
    out = [_protein_chain(seq, n_copies, paired, unpaired)]
    for p in partners or []:
        out.append(_protein_chain(p["sequence"], p.get("count", 1),
                                  p.get("paired"), p.get("unpaired")))
    if apo:
        return out
    smis = list(ligands) if ligands else [LIGAND_SMILES]
    k = n_copies if ligand_copies is None else ligand_copies
    for smi in smis:
        out.append({"ligand": {"ligand": smi, "count": k}})
    return out


def a3m_query(a3m_path):
    """Return (header_seq) of the first record = the query."""
    hdr = seq = None
    with open(a3m_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if hdr is not None:
                    break
                hdr = line.strip()
                seq = ""
            elif hdr is not None:
                seq += line.strip()
    return seq


def build_a3m_index(msa_dirs):
    """Index a disjoint union of ColabFold output directories by filename stem."""
    index = {}
    for raw_dir in msa_dirs:
        msa_dir = os.path.realpath(os.path.expanduser(raw_dir))
        if not os.path.isdir(msa_dir):
            raise ValueError(f"A3M directory does not exist: {msa_dir}")
        for name in sorted(os.listdir(msa_dir)):
            if not name.endswith(".a3m"):
                continue
            pred_id = name[:-4]
            path = os.path.realpath(os.path.join(msa_dir, name))
            if pred_id in index:
                raise ValueError(
                    f"duplicate A3M for {pred_id}: {index[pred_id]}, {path}"
                )
            index[pred_id] = path
    return {pred_id: index[pred_id] for pred_id in sorted(index)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--id-column", default="pred_id",
                    help="manifest ID column (renamed internally to pred_id)")
    ap.add_argument("--sequence-column", default="sequence_pred",
                    help="manifest sequence column (renamed internally to sequence_pred)")
    ap.add_argument("--msa-cf", action="append", required=True,
                    help="dir with <pred_id>.a3m from colabfold_search; repeat for disjoint shards")
    ap.add_argument("--msa-px", required=True, help="output dir for <pred_id>/{pairing,non_pairing}.a3m")
    ap.add_argument("--inputs", required=True, help="output dir for <pred_id>-update-msa.json")
    ap.add_argument("--ready-list", required=True)
    ap.add_argument("--ligand-smiles", action="append", default=None,
                    help="ligand SMILES; REPEAT for several distinct entities "
                         "(e.g. --ligand-smiles '[Zn+2]' --ligand-smiles '<substrate>'). Default: uric acid.")
    ap.add_argument("--ligand-copies", type=int, default=None,
                    help="count per ligand entity (default: --n-copies)")
    ap.add_argument("--n-copies", type=int, default=4)
    ap.add_argument("--apo", action="store_true", help="omit the ligand entity (monomer/apo prediction)")
    ap.add_argument("--name-prefix", default="tetra_", help="prediction name prefix (e.g. 'mono_')")
    ap.add_argument("--partner-id", action="append", default=None,
                    help="pred_id of an additional DISTINCT protein chain (hetero-complex, e.g. a "
                         "binder's target). Its sequence and MSA are read from --msa-cf/<id>.a3m. "
                         "REPEAT for several partners.")
    ap.add_argument("--partner-copies", action="append", type=int, default=None,
                    help="copies per --partner-id, in the same order (default 1 for each)")
    ap.add_argument("--require-complete", action="store_true",
                    help="fail before writes unless every manifest/partner ID has an A3M")
    ap.add_argument("--require-query-match", action="store_true",
                    help="fail before writes instead of warning on an A3M/manifest sequence mismatch")
    ap.add_argument("--refuse-existing-output", action="store_true",
                    help="require absent msa-px, inputs, and ready-list outputs")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    required = {args.id_column, args.sequence_column}
    missing_columns = sorted(required - set(man.columns))
    if missing_columns:
        sys.exit(f"FATAL: manifest missing columns: {missing_columns}")
    if args.id_column != "pred_id" and "pred_id" in man.columns:
        sys.exit("FATAL: manifest already contains pred_id while --id-column names another column")
    if args.sequence_column != "sequence_pred" and "sequence_pred" in man.columns:
        sys.exit(
            "FATAL: manifest already contains sequence_pred while --sequence-column names another column"
        )
    man = man.rename(
        columns={args.id_column: "pred_id", args.sequence_column: "sequence_pred"}
    )
    if man["pred_id"].isna().any() or man["pred_id"].astype(str).duplicated().any():
        sys.exit("FATAL: manifest pred_id must be non-null and unique")
    man = man.copy()
    man["pred_id"] = man["pred_id"].astype(str)
    a3m_index = build_a3m_index(args.msa_cf)
    partner_ids = args.partner_id or []
    required_ids = set(man.pred_id) | set(partner_ids)
    missing_preflight = sorted(required_ids - set(a3m_index))
    if args.require_complete and missing_preflight:
        sys.exit(f"FATAL: missing A3M for {len(missing_preflight)} IDs: {missing_preflight[:10]}")
    if args.require_query_match:
        sequence_by_id = dict(zip(man.pred_id, man.sequence_pred.astype(str), strict=True))
        for pred_id in sorted(set(man.pred_id) & set(a3m_index)):
            observed = a3m_query(a3m_index[pred_id]).replace("-", "").upper()
            expected = sequence_by_id[pred_id].upper()
            if observed != expected:
                sys.exit(
                    f"FATAL: {pred_id} A3M query differs from manifest sequence "
                    f"(observed_len={len(observed)}, expected_len={len(expected)})"
                )
    if args.refuse_existing_output:
        existing = [
            path for path in (args.msa_px, args.inputs, args.ready_list)
            if os.path.exists(path)
        ]
        if existing:
            sys.exit(f"FATAL: output paths already exist: {existing}")
    os.makedirs(args.msa_px, exist_ok=True)
    os.makedirs(args.inputs, exist_ok=True)

    # Partner chains are shared across every design, so split their a3m once up front.
    copies = args.partner_copies or []
    if copies and len(copies) != len(partner_ids):
        sys.exit(f"FATAL: {len(copies)} --partner-copies for {len(partner_ids)} --partner-id")
    partners = []
    for i, pid in enumerate(partner_ids):
        a3m = a3m_index.get(pid)
        if a3m is None:
            sys.exit(f"FATAL: partner a3m absent: {a3m}")
        pseq = a3m_query(a3m).replace("-", "").upper()
        d = os.path.join(args.msa_px, pid)
        os.makedirs(d, exist_ok=True)
        with open(a3m) as src, open(os.path.join(d, "non_pairing.a3m"), "w") as dst:
            dst.write(src.read())
        with open(os.path.join(d, "pairing.a3m"), "w") as f:
            f.write(f">query\n{pseq}\n")
        partners.append({"sequence": pseq, "count": copies[i] if copies else 1,
                         "paired": os.path.join(d, "pairing.a3m"),
                         "unpaired": os.path.join(d, "non_pairing.a3m")})
        print(f"partner chain {pid}: len={len(pseq)} copies={partners[-1]['count']}")

    ready, missing = [], []
    for r in man.itertuples():
        pid = r.pred_id
        a3m = a3m_index.get(pid)
        if a3m is None:
            missing.append(pid)
            continue
        qseq = a3m_query(a3m)
        # cross-check the a3m query matches the manifest sequence (upper, gaps removed)
        exp = r.sequence_pred.upper()
        if qseq.replace("-", "").upper() != exp:
            message = f"{pid}: a3m query != manifest seq (len {len(qseq)} vs {len(exp)})"
            if args.require_query_match:
                sys.exit(f"FATAL: {message}")
            print(f"WARN {message}", file=sys.stderr)
        d = os.path.join(args.msa_px, pid)
        os.makedirs(d, exist_ok=True)
        # non_pairing = full a3m verbatim
        with open(a3m) as src, open(os.path.join(d, "non_pairing.a3m"), "w") as dst:
            dst.write(src.read())
        # pairing = query only
        with open(os.path.join(d, "pairing.a3m"), "w") as f:
            f.write(f">query\n{exp}\n")
        seqs_list = build_sequences_block(
            exp, args.n_copies, ligands=args.ligand_smiles, apo=args.apo,
            ligand_copies=args.ligand_copies,
            paired=os.path.join(d, "pairing.a3m"),
            unpaired=os.path.join(d, "non_pairing.a3m"),
            partners=partners,
        )
        obj = [{"name": f"{args.name_prefix}{pid}", "sequences": seqs_list}]
        jp = os.path.join(args.inputs, f"{args.name_prefix}{pid}-update-msa.json")
        with open(jp, "w") as f:
            json.dump(obj, f, indent=2)
        ready.append(jp)
    with open(args.ready_list, "w") as f:
        f.write("\n".join(ready) + "\n")
    print(f"built {len(ready)} JSONs; missing a3m for {len(missing)}: {missing[:10]}")


if __name__ == "__main__":
    main()
