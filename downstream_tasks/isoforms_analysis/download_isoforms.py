#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isoform-centric metadata + AlphaFold downloads from a local UniProt HUMAN .json.gz dump.

Input:
  --from-uniprot-json  path/to/human_uniprot.json.gz   # UniProt dump (search results, NDJSON, or concatenated JSON)
Options:
  --reviewed-only      keep only reviewed (handles both reviewed: true and entryType: "Reviewed")
  --meta-out           output JSON (isoform-centric), default: isoform_meta.json
  --pdb-dir            folder for AFDB files, default: afdb_pdb
  --prefer {pdb,cif}   preferred file type to download from AFDB API, default: pdb
  --quiet              suppress info logs and progress bars

Output:
  - isoform_meta.json: {
        "P04637-2": [{
            "full_name": str,
            "gene_name": str,
            "ec_numbers": [str, ...],
            "active_sites": [int, ...],
            "base_id": "P04637"
        }],
        ...
    }
  - afdb_pdb/: downloaded AFDB structures (PDB by default; if missing, entry goes to afdb_missing.tsv)
  - afdb_missing.tsv: isoforms without downloadable file (or network issues)
"""

import argparse
import gzip
import json
import logging
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

# --------------------------- Logging ---------------------------

log = logging.getLogger("isoforms")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_handler)

# --------------------------- Constants ---------------------------

HUMAN_TAXID = 9606
BATCH = 200
SLEEP = 0.05

AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"

# --------------------------- Robust readers ---------------------------

def iter_records_json_gz(path: str) -> Iterable[dict]:
    """Yield UniProt records from a .json.gz dump: big JSON, NDJSON, or concatenated JSON objects."""
    # 1) Try a single big JSON
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            data = json.load(f)
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    yield rec
            return
        if isinstance(data, dict):
            items = data.get("results", data)
            if isinstance(items, list):
                for rec in items:
                    if isinstance(rec, dict):
                        yield rec
                return
            if isinstance(items, dict):
                yield items
                return
    except Exception:
        pass

    # 2) NDJSON (one JSON per line)
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            any_yielded = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    any_yielded = True
                    yield obj
            if any_yielded:
                return
    except Exception:
        pass

    # 3) Concatenated JSON objects (raw_decode loop)
    dec = json.JSONDecoder()
    buf = ""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB
            if not chunk and not buf:
                break
            buf += chunk
            i = 0
            n = len(buf)
            while True:
                while i < n and buf[i].isspace():
                    i += 1
                if i >= n:
                    break
                try:
                    obj, j = dec.raw_decode(buf, i)
                except JSONDecodeError:
                    break
                if isinstance(obj, dict):
                    yield obj
                elif isinstance(obj, list):
                    for x in obj:
                        if isinstance(x, dict):
                            yield x
                i = j
            buf = buf[i:]
            if not chunk:
                break

# --------------------------- Extractors ---------------------------

def is_human(rec: dict, organism_id: int = HUMAN_TAXID) -> bool:
    org = rec.get("organism") or {}
    tid = org.get("taxonId")
    if tid is None:
        # dump is already human-only quite often; be permissive
        return True
    try:
        return int(tid) == organism_id
    except Exception:
        return False

def is_reviewed_entry(rec: dict) -> bool:
    v = rec.get("reviewed")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower().startswith("review")
    et = rec.get("entryType")
    if isinstance(et, str):
        return et.lower().startswith("review")
    # If we cannot tell, don't exclude
    return True

def acc(rec: dict) -> Optional[str]:
    a = rec.get("primaryAccession") or rec.get("accession") or rec.get("uniProtkbId")
    return str(a).upper() if a else None

def extract_full_name(rec: dict) -> Optional[str]:
    pd = rec.get("proteinDescription") or {}
    rn = pd.get("recommendedName") or {}
    fn = ((rn.get("fullName") or {}).get("value")) if isinstance(rn, dict) else None
    if isinstance(fn, str) and fn.strip():
        return fn.strip()
    for key in ("submissionNames", "alternativeNames"):
        blocks = pd.get(key) or []
        if not isinstance(blocks, list):
            blocks = [blocks]
        for block in blocks:
            if not isinstance(block, dict):
                continue
            v = ((block.get("fullName") or {}).get("value"))
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None

def extract_gene_name(rec: dict) -> Optional[str]:
    gs = rec.get("genes") or []
    if not isinstance(gs, list) or not gs:
        return None
    g0 = gs[0]
    if isinstance(g0, dict):
        gn = (g0.get("geneName") or {}).get("value")
        if isinstance(gn, str) and gn.strip():
            return gn.strip()
        syns = g0.get("synonyms") or []
        if isinstance(syns, list) and syns:
            v = (syns[0] or {}).get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
    for g in gs:
        if not isinstance(g, dict):
            continue
        gn = (g.get("geneName") or {}).get("value")
        if isinstance(gn, str) and gn.strip():
            return gn.strip()
        for s in (g.get("synonyms") or []):
            v = (s or {}).get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None

def seq_string(rec: dict) -> Optional[str]:
    s = (rec.get("sequence") or {}).get("value")
    return s if isinstance(s, str) and s else None

def extract_active_site_positions(rec: dict, seq: Optional[str]) -> List[int]:
    """
    Return list of integer positions for active sites.
    Supports feature.type 'ACT_SITE' (new) or 'Active site' (older dumps).
    Keeps exact single residues only.
    """
    feats = rec.get("features") or []
    out: Set[int] = set()
    for ft in feats:
        if not isinstance(ft, dict):
            continue
        tp = ft.get("type")
        if tp not in {"ACT_SITE", "Active site"}:
            continue
        loc = ft.get("location") or {}
        pos = None
        if isinstance(loc.get("position"), dict) and "value" in loc["position"]:
            try:
                pos = int(loc["position"]["value"])
            except Exception:
                pos = None
        else:
            st = (loc.get("start") or {}).get("value")
            en = (loc.get("end") or {}).get("value")
            mod_st = (loc.get("start") or {}).get("modifier")
            mod_en = (loc.get("end") or {}).get("modifier")
            if st is not None and en is not None and st == en and (mod_st in (None, "EXACT")) and (mod_en in (None, "EXACT")):
                try:
                    pos = int(st)
                except Exception:
                    pos = None
        if pos is not None:
            out.add(pos)
    return sorted(out)

def extract_ec_numbers(rec: dict) -> List[str]:
    """
    Collect EC numbers from multiple possible locations in UniProt JSON schema.
    Permissive and deduplicated.
    """
    ecs: Set[str] = set()

    def _add(x):
        if isinstance(x, str) and x.strip():
            ecs.add(x.strip())

    def _from_block(b: dict):
        if not isinstance(b, dict):
            return
        v = b.get("ecNumber")
        if isinstance(v, str):
            _add(v)
        elif isinstance(v, list):
            for x in v:
                _add(x)
        v2 = b.get("ecNumbers")
        if isinstance(v2, list):
            for x in v2:
                if isinstance(x, dict):
                    _add(x.get("value"))
                else:
                    _add(x)

    # comments → catalytic activity
    for c in (rec.get("comments") or []):
        if isinstance(c, dict) and c.get("commentType") in {"CATALYTIC_ACTIVITY", "catalytic activity"}:
            _add(c.get("ecNumber"))
            rxn = c.get("reaction") or {}
            _add(rxn.get("ecNumber"))

    # cross-references → EC
    for xref in (rec.get("uniProtKBCrossReferences") or []):
        if isinstance(xref, dict) and xref.get("database") == "EC":
            _add(xref.get("id"))

    # proteinDescription variants
    pd = rec.get("proteinDescription") or {}
    for key in ("recommendedName", "submissionNames", "alternativeNames"):
        block = pd.get(key)
        blocks = block if isinstance(block, list) else [block] if block else []
        for b in blocks:
            _from_block(b)

    for key in ("includes", "contains", "fragments"):
        items = pd.get(key) or []
        if not isinstance(items, list):
            items = [items]
        for it in items:
            if not isinstance(it, dict):
                continue
            _from_block(it)
            _from_block(it.get("recommendedName") or {})
            alts = it.get("alternativeNames")
            alts = alts if isinstance(alts, list) else [alts] if alts else []
            for a in alts:
                _from_block(a)

    return sorted(ecs)

# --------------------------- Scan dump once ---------------------------

def scan_dump(json_gz_path: str, reviewed_only: bool = False) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, int]]:
    """
    Single-pass scan:
      - meta_by_parent[parent] = {canonical_full_name, gene_name, ec_numbers, active_sites}
      - iso_by_parent[parent] = [isoform IDs]
      - counters for diagnostics
    """
    meta_by_parent: Dict[str, Dict[str, Any]] = {}
    iso_by_parent: Dict[str, List[str]] = {}

    n_seen = n_human = n_after_review = n_parents = 0

    for rec in iter_records_json_gz(json_gz_path):
        n_seen += 1
        if not is_human(rec):
            continue
        n_human += 1
        if reviewed_only and not is_reviewed_entry(rec):
            continue
        n_after_review += 1

        a = acc(rec)
        if not a:
            continue
        n_parents += 1

        # metadata from dump
        meta_by_parent[a] = {
            "canonical_full_name": extract_full_name(rec) or "",
            "gene_name": (extract_gene_name(rec) or ""),
            "ec_numbers": extract_ec_numbers(rec),
            "active_sites": extract_active_site_positions(rec, seq_string(rec)),
        }

        # isoforms from ALTERNATIVE PRODUCTS
        iso_ids: List[str] = []
        for c in (rec.get("comments") or []):
            if not isinstance(c, dict):
                continue
            if c.get("commentType") == "ALTERNATIVE PRODUCTS":
                for iso in c.get("isoforms", []) or []:
                    for iid in iso.get("isoformIds", []) or []:
                        if isinstance(iid, str) and iid.strip():
                            iso_ids.append(iid.strip())

        if iso_ids:
            iso_by_parent[a] = sorted(set(iso_ids))

    counters = dict(seen=n_seen, human=n_human, after_review=n_after_review, parents=n_parents)
    return meta_by_parent, iso_by_parent, counters

# --------------------------- Build iso-centric JSON ---------------------------

def build_iso_centric_out(multi_iso: Dict[str, List[str]],
                          meta_by_parent: Dict[str, Dict[str, Any]],
                          quiet: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    it = multi_iso.items()
    it = tqdm(it, desc="Metadata", disable=quiet)
    for parent, iso_list in it:
        meta = meta_by_parent.get(parent, {})
        for iso in iso_list:
            rec = {
                "full_name": meta.get("canonical_full_name", "") or "",
                "gene_name": meta.get("gene_name", "") or "",
                "ec_numbers": meta.get("ec_numbers", []) or [],
                "active_sites": meta.get("active_sites", []) or [],
                "base_id": parent,
            }
            out.setdefault(iso, []).append(rec)
    return out

# --------------------------- AFDB API download ---------------------------

def resolve_af_urls(uid: str) -> Dict[str, Optional[str]]:
    try:
        r = requests.get(AF_API.format(uid=uid), timeout=60, headers={"User-Agent": "afdb-scraper/1.0 (email@example.com)"})
        if r.status_code != 200:
            return {}
        data = r.json()
        d0 = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
        if not isinstance(d0, dict):
            return {}
        # разные возможные ключи, старые и новые
        candidates_pdb = ["pdbUrl", "modelUrlPDB", "pdb_url"]
        candidates_cif = ["cifUrl", "bcifUrl", "mmCifUrl", "cif_url", "bcif_url", "mmcif_url"]
        def pick(d, keys):
            for k in keys:
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            return None
        return {
            "pdb": pick(d0, candidates_pdb),
            "cif": pick(d0, candidates_cif),
            # entryId → modelEntityId (на части ответов уже так)
            "entryId": d0.get("entryId") or d0.get("modelEntityId"),
        }
    except Exception:
        return {}

def guess_direct_url(uniprot: str, prefer: str = "pdb") -> Optional[str]:
    base = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4"
    if prefer == "pdb":
        return base + ".pdb"
    return base + ".cif"


def download_afdb_models(all_isoforms: Set[str],
                         out_dir: str = "afdb_pdb",
                         prefer: str = "pdb",
                         quiet: bool = False) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    missing = []

    for iso in tqdm(sorted(all_isoforms), desc="AFDB download", disable=quiet):
        info = resolve_af_urls(iso)
        # prefer requested type, then fallback to the other type
        url = None
        if prefer == "pdb":
            url = info.get("pdb") or info.get("cif")
        else:
            url = info.get("cif") or info.get("pdb")

        # If isoform missing, try base accession as a last resort
        if not url:
            base = iso.split("-")[0]
            info_b = resolve_af_urls(base)
            url = (info_b.get(prefer) or info_b.get("pdb") or info_b.get("cif"))
            if not url:
                url = guess_direct_url(base, prefer)
                
        if not url:
            missing.append((iso, "no_api_link"))
            continue

        ext = ".pdb" if url.endswith(".pdb") else (".cif" if ".cif" in url or ".bcif" in url else "")
        dest = outp / f"{iso}{ext or ''}"

        try:
            rr = requests.get(url, timeout=180)
            if rr.status_code == 200 and rr.content and len(rr.content) > 1000:
                dest.write_bytes(rr.content)
            else:
                missing.append((iso, f"http_{rr.status_code}"))
        except Exception as e:
            missing.append((iso, f"exc:{e.__class__.__name__}"))

        time.sleep(SLEEP)

    if missing:
        with open("afdb_missing.tsv", "w") as m:
            m.write("isoform_id\tstatus\n")
            for iso, st in missing:
                m.write(f"{iso}\t{st}\n")

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Isoform metadata + AFDB downloads from a local UniProt HUMAN .json.gz")
    ap.add_argument("--from-uniprot-json", required=True, help="Path to UniProt HUMAN dump (.json.gz)")
    ap.add_argument("--reviewed-only", action="store_true", help="Keep only reviewed entries (robust).")
    ap.add_argument("--meta-out", default="isoform_meta.json", help="Output iso-centric JSON.")
    ap.add_argument("--pdb-dir", default="afdb_pdb", help="Folder for AlphaFold files.")
    ap.add_argument("--prefer", choices=["pdb", "cif"], default="pdb", help="Preferred AFDB file type to download.")
    ap.add_argument("--quiet", action="store_true", help="Suppress info logs and progress bars.")
    args = ap.parse_args()

    if args.quiet:
        log.setLevel(logging.WARNING)

    # Scan dump once (metadata + isoforms, offline)
    meta_by_parent, iso_by_parent, cnt = scan_dump(args.from_uniprot_json, reviewed_only=args.reviewed_only)
    log.info(f"[dump] seen={cnt['seen']} | human={cnt['human']} | after_review={cnt['after_review']} | parents={cnt['parents']}")

    parents = list(iso_by_parent.keys())
    log.info(f"Parents with any isoforms (raw): {len(parents)}")

    # Keep only parents with ≥2 isoforms
    multi_iso = {p: isos for p, isos in iso_by_parent.items() if len(isos) >= 1}
    log.info(f"Parents with ≥1 isoforms: {len(multi_iso)}")

    # Build iso-centric JSON
    out = build_iso_centric_out(multi_iso, meta_by_parent, quiet=args.quiet)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info(f"Saved meta: {args.meta_out} (isoform keys: {len(out)})")

    # Download AFDB files
    all_iso = {iso for isos in multi_iso.values() for iso in isos}
    log.info(f"Total isoforms to try downloading: {len(all_iso)}")
    
    download_afdb_models(all_iso, out_dir=args.pdb_dir, prefer=args.prefer, quiet=args.quiet)
    log.info("Done.")

if __name__ == "__main__":
    main()
