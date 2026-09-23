"""Build Tier 1 candidate entries from manifests + IEDB span data + SIFTS.

Reverse-engineered from the existing `outputs/if/test_set/tier1_candidates.json`
(0701 set, 15 entries). Span extraction is exact: every span in the existing
JSON is reproduced bit-identically from `data/mhc_if_v2.tsv` for the matching
(uniprot, allele, chain_range) tuple — a sweep over all 15 entries / 261 spans
returned `exact_match=True` on every entry. The PDB-chain selection uses the
PDBe SIFTS REST API (`/mappings/best_structures/{uniprot}`) ranked by SIFTS
quality and filtered for X-ray structures within the requested resolution and
chain length bounds.

Coordinate contract (matches `inverse_folding/analysis/structural_features.py`):

  * IEDB `peptide_position_info[*].start` / `end` are **1-based inclusive**
    UniProt residue positions (verified empirically: `end - start + 1 ==
    len(peptide_seq)`).
  * Tier 1 JSON `start_0b` / `end_0b` are **0-based half-open** absolute
    UniProt positions, so `start_0b = iedb_start - 1`, `end_0b = iedb_end`.
  * `chain_range` is **1-based inclusive** UniProt range covered by the chain;
    chain residue 1 (`sequence[0]`) corresponds to UniProt residue
    `chain_range_start`.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


PDBE_BEST_STRUCTURES = (
    "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}"
)


# ─────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────


def load_test_uniprots(manifest_dir: Path) -> List[str]:
    """Read `splits/test_ids.txt` from a per-allele manifest directory."""
    p = Path(manifest_dir) / "splits" / "test_ids.txt"
    if not p.is_file():
        raise FileNotFoundError(f"Test split file missing: {p}")
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def load_uniprot_seqs(fasta_path: Path) -> Dict[str, str]:
    """Parse a FASTA into `{uniprot_id: sequence}`.

    Accepts both bare-accession headers (`>P12345`) and SwissProt-style
    headers (`>sp|P12345|NAME`). NCBI-style `>NP_xxx` headers are kept as-is.
    Versioned accessions (`P12345.2`) are stripped to bare ID.
    """
    seqs: Dict[str, str] = {}
    cur_id: Optional[str] = None
    cur_lines: List[str] = []
    with Path(fasta_path).open() as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_lines)
                header = line[1:].strip()
                if header.startswith("sp|") or header.startswith("tr|"):
                    parts = header.split("|")
                    cur_id = parts[1] if len(parts) >= 2 else parts[0]
                else:
                    cur_id = header.split()[0]
                cur_id = cur_id.split(".")[0]
                cur_lines = []
            else:
                cur_lines.append(line.strip())
    if cur_id is not None:
        seqs[cur_id] = "".join(cur_lines)
    return seqs


def _safe_json_loads(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


def load_iedb_spans(
    mhc_if_v2_path: Path,
    allele: str,
) -> Dict[str, List[dict]]:
    """Return `{uniprot: [{start_0b, end_0b, peptide, source}, ...]}` for one allele.

    Filters rows whose `alleles` JSON list contains `allele` exactly. Spans are
    deduplicated by `(start_0b, end_0b, peptide)` per UniProt — preserves all
    length variants at the same start, matching the convention used by the
    shipped 0701 JSON.
    """
    df = pd.read_csv(mhc_if_v2_path, sep="\t", low_memory=False)

    def has_allele(s):
        alist = _safe_json_loads(s)
        return bool(alist) and allele in alist

    sub = df[df["alleles"].apply(has_allele)]
    out: Dict[str, List[dict]] = {}
    for _, row in sub.iterrows():
        accs = _safe_json_loads(row.get("protein_accessions")) or []
        ppi = _safe_json_loads(row.get("peptide_position_info")) or []
        peptide = row.get("peptide_seq")
        source = row.get("source") or "iedb"
        for it in ppi:
            try:
                pid = str(it.get("protein_id", ""))
                start = int(it["start"])
                end = int(it["end"])
            except (KeyError, TypeError, ValueError):
                continue
            uni = pid.split(".")[0]
            if uni not in accs:
                continue
            entry = {
                "start_0b": start - 1,   # 1-based inclusive → 0-based half-open
                "end_0b": end,
                "peptide": peptide,
                "source": str(source),
            }
            out.setdefault(uni, []).append(entry)

    # Dedup per uniprot, keeping order of first appearance
    for uni in out:
        seen = set()
        unique = []
        for sp in out[uni]:
            k = (sp["start_0b"], sp["end_0b"], sp["peptide"])
            if k in seen:
                continue
            seen.add(k)
            unique.append(sp)
        out[uni] = unique
    return out


# ─────────────────────────────────────────────────────────────────────────
# Span extraction within chain
# ─────────────────────────────────────────────────────────────────────────


def extract_spans_for_chain(
    spans_full: Sequence[dict],
    chain_seq: str,
    chain_range_start: int,  # 1-based UniProt position of sequence[0]
    chain_range_end: int,    # 1-based UniProt position of sequence[-1]
    iedb_ref_label: str = "IEDB",
) -> List[dict]:
    """Return spans whose [start_0b, end_0b) lies fully inside the chain range.

    For each kept span the chain-local slice `chain_seq[start-(s-1):end-(s-1)]`
    is verified to equal `peptide`; mismatches are silently dropped (they
    indicate IEDB position info that disagrees with the chosen UniProt
    sequence — e.g. an isoform). The returned list uses the JSON schema
    expected by `tier1_candidates.json::experimental_epitopes`.
    """
    chain_start_0b = chain_range_start - 1
    chain_end_0b = chain_range_end          # half-open upper bound
    out: List[dict] = []
    for sp in spans_full:
        s0 = sp["start_0b"]
        e0 = sp["end_0b"]
        if s0 < chain_start_0b or e0 > chain_end_0b:
            continue
        local_s = s0 - chain_start_0b
        local_e = e0 - chain_start_0b
        slice_aa = chain_seq[local_s:local_e]
        peptide = sp.get("peptide") or ""
        if peptide and slice_aa != peptide:
            continue
        out.append({
            "start_0b": s0,
            "end_0b": e0,
            "peptide": peptide,
            "assay_type": "EL",
            "iedb_ref": iedb_ref_label,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────
# SIFTS PDBe API
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SiftsChain:
    pdb_id: str
    chain_id: str
    unp_start: int
    unp_end: int
    pdb_start: int
    pdb_end: int
    resolution: Optional[float]
    method: str
    coverage: Optional[float]


def _parse_sifts_payload(payload: dict, uniprot: str) -> List[SiftsChain]:
    items = payload.get(uniprot) or []
    out: List[SiftsChain] = []
    for it in items:
        try:
            out.append(SiftsChain(
                pdb_id=str(it["pdb_id"]).upper(),
                chain_id=str(it["chain_id"]),
                unp_start=int(it["unp_start"]),
                unp_end=int(it["unp_end"]),
                pdb_start=int(it["start"]),
                pdb_end=int(it["end"]),
                resolution=float(it["resolution"]) if it.get("resolution") is not None else None,
                method=str(it.get("experimental_method") or ""),
                coverage=float(it["coverage"]) if it.get("coverage") is not None else None,
            ))
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("Skipping malformed SIFTS entry for %s: %s (%s)", uniprot, it, e)
    return out


def query_sifts_best_structures(
    uniprot: str,
    cache_dir: Optional[Path] = None,
    api_delay_s: float = 0.10,
    timeout_s: float = 20.0,
    user_agent: str = "build_tier1_candidates/1.0",
) -> List[SiftsChain]:
    """Hit PDBe `/mappings/best_structures/{uniprot}`.

    Returns the SIFTS-ranked list (best first). Network failures and 404 yield
    an empty list and a logged warning — the caller treats "no PDB mapping" as
    a soft skip. Successful responses are JSON-cached under `cache_dir`.
    """
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{uniprot}.json"
        if cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text())
                return _parse_sifts_payload(payload, uniprot)
            except Exception as e:
                logger.warning("Cache read failed for %s (%s); refetching", uniprot, e)

    url = PDBE_BEST_STRUCTURES.format(uniprot=uniprot)
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": user_agent},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            payload = {}
        else:
            logger.warning("SIFTS HTTP %s for %s", e.code, uniprot)
            return []
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        logger.warning("SIFTS network error for %s: %s", uniprot, e)
        return []

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
    if api_delay_s > 0:
        time.sleep(api_delay_s)
    return _parse_sifts_payload(payload, uniprot)


# ─────────────────────────────────────────────────────────────────────────
# Candidate construction
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FilterParams:
    min_len: int = 100
    max_len: int = 500
    max_resolution: float = 2.5
    min_spans: int = 2
    min_coverage: float = 0.10
    max_coverage: float = 0.50


def _is_xray(method: str) -> bool:
    m = method.lower()
    return ("x-ray" in m) or ("crystal" in m)


def build_candidate_entry(
    uniprot: str,
    sifts_chain: SiftsChain,
    spans_full: Sequence[dict],
    full_uniprot_seq: str,
    allele: str,
    epitope_head_split: str,
    filters: FilterParams,
) -> Optional[dict]:
    """Try to construct one Tier 1 entry. Returns None if the chain is rejected.

    Rejection reasons (silent — caller may keep stats):
      - method not X-ray
      - resolution missing or > max_resolution
      - chain length outside [min_len, max_len]
      - UniProt sequence too short to cover the chain
      - chain has fewer than `min_spans` valid spans within range
    """
    if not _is_xray(sifts_chain.method):
        return None
    if sifts_chain.resolution is None or sifts_chain.resolution > filters.max_resolution:
        return None
    chain_len = sifts_chain.unp_end - sifts_chain.unp_start + 1
    if chain_len < filters.min_len or chain_len > filters.max_len:
        return None
    if not full_uniprot_seq or sifts_chain.unp_end > len(full_uniprot_seq):
        return None

    chain_seq = full_uniprot_seq[sifts_chain.unp_start - 1:sifts_chain.unp_end]
    if len(chain_seq) != chain_len:
        return None

    spans_chain = extract_spans_for_chain(
        spans_full,
        chain_seq=chain_seq,
        chain_range_start=sifts_chain.unp_start,
        chain_range_end=sifts_chain.unp_end,
    )
    if len(spans_chain) < filters.min_spans:
        return None

    covered = set()
    for sp in spans_chain:
        covered.update(range(sp["start_0b"], sp["end_0b"]))
    n_covered = len(covered)
    coverage = n_covered / chain_len

    if coverage < filters.min_coverage or coverage > filters.max_coverage:
        return None

    return {
        "protein_id": f"{sifts_chain.pdb_id}_{sifts_chain.chain_id}",
        "uniprot_id": uniprot,
        "pdb_id": sifts_chain.pdb_id,
        "chain": sifts_chain.chain_id,
        "sequence": chain_seq,
        "sequence_length": chain_len,
        "resolution": float(sifts_chain.resolution),
        "chain_range": f"{sifts_chain.unp_start}-{sifts_chain.unp_end}",
        "n_epitope_spans": len(spans_chain),
        "n_covered_residues": n_covered,
        "n_cold_residues": chain_len - n_covered,
        "epitope_coverage": round(coverage, 4),
        "experimental_epitopes": spans_chain,
        "epitope_head_split": epitope_head_split,
        "selection_reason": (
            f"Test split; {len(spans_chain)} {allele} spans; "
            f"{coverage:.1%} coverage; X-ray {sifts_chain.resolution}Å"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# Top-level pipeline (callable from CLI or tests)
# ─────────────────────────────────────────────────────────────────────────


def build_tier1_candidates(
    test_uniprots: Sequence[str],
    spans_by_uniprot: Dict[str, List[dict]],
    uniprot_seqs: Dict[str, str],
    allele: str,
    filters: FilterParams,
    n_select: int,
    rank_by: str = "epitope_coverage",
    rank_descending: bool = True,
    sifts_cache_dir: Optional[Path] = None,
    api_delay_s: float = 0.10,
    epitope_head_split: str = "test",
    progress_log_every: int = 25,
) -> Tuple[List[dict], dict]:
    """End-to-end candidate construction. Returns (selected, stats)."""
    if rank_by not in ("epitope_coverage", "n_epitope_spans", "n_covered_residues"):
        raise ValueError(f"Unsupported rank_by={rank_by!r}")

    candidates: List[dict] = []
    stats = {
        "n_test_uniprots": len(test_uniprots),
        "n_with_spans": 0,
        "n_with_seq": 0,
        "n_with_sifts_chains": 0,
        "n_candidates_built": 0,
        "uniprots_no_qualifying_chain": [],
    }

    for i, uni in enumerate(test_uniprots):
        spans = spans_by_uniprot.get(uni, [])
        if not spans:
            continue
        stats["n_with_spans"] += 1
        full = uniprot_seqs.get(uni)
        if not full:
            stats["uniprots_no_qualifying_chain"].append((uni, "no_uniprot_seq"))
            continue
        stats["n_with_seq"] += 1

        sifts_chains = query_sifts_best_structures(
            uni, cache_dir=sifts_cache_dir, api_delay_s=api_delay_s,
        )
        if not sifts_chains:
            stats["uniprots_no_qualifying_chain"].append((uni, "no_sifts"))
            continue
        stats["n_with_sifts_chains"] += 1

        chosen: Optional[dict] = None
        for ch in sifts_chains:
            entry = build_candidate_entry(
                uniprot=uni,
                sifts_chain=ch,
                spans_full=spans,
                full_uniprot_seq=full,
                allele=allele,
                epitope_head_split=epitope_head_split,
                filters=filters,
            )
            if entry is not None:
                chosen = entry
                break
        if chosen is None:
            stats["uniprots_no_qualifying_chain"].append((uni, "no_chain_passed_filters"))
            continue

        candidates.append(chosen)
        stats["n_candidates_built"] += 1

        if progress_log_every and ((i + 1) % progress_log_every == 0):
            logger.info(
                "  processed %d/%d (with-spans=%d, candidates=%d)",
                i + 1, len(test_uniprots),
                stats["n_with_spans"], stats["n_candidates_built"],
            )

    candidates.sort(key=lambda c: c[rank_by], reverse=rank_descending)
    selected = candidates[:n_select]
    selected.sort(key=lambda c: c["epitope_coverage"])  # final ascending order matches existing JSON
    stats["n_selected"] = len(selected)
    return selected, stats
