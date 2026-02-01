#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a single UniProt HUMAN metadata cache from one .json.gz dump.

Output per ACC:
{
  "full_name": str|null,
  "gene_name": str|null,
  "ec_numbers": [str, ...],
  "active_sites": [ {"pos": int, "aa": "X"|null, "description": str|null}, ... ]
}
"""

import os, gzip, json, argparse
from typing import Dict, Any, Optional, Iterable, List, Tuple, Set

# --------------------- robust readers ---------------------

def iter_records_json_gz(path: str) -> Iterable[dict]:
    """
    Порядок попыток:
    1) load-весь-файл как единый JSON:
       - если dict с ключом 'results' -> итерируем results
       - если list -> итерируем список
       - если одиночный dict -> одна запись
    2) NDJSON (по строкам)
    3) Конкатенированные JSON-объекты без разделителей (raw_decode)
    """
    # 1) единый JSON
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            data = json.load(f)
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    yield rec
            return
        if isinstance(data, dict):
            if isinstance(data.get("results"), list):
                for rec in data["results"]:
                    if isinstance(rec, dict):
                        yield rec
                return
            # одиночный словарь как запись
            yield data
            return
    except Exception:
        pass

    # 2) NDJSON
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

    # 3) конкатенированные объекты
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
                except json.JSONDecodeError:
                    # нужно больше данных
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

# --------------------- helpers & extractors ---------------------

def is_human(rec: dict, organism_id: int = 9606) -> bool:
    """
    Пермишсивно: если taxonId отсутствует (часто уже отфильтровано upstream),
    считаем human; иначе сравниваем.
    """
    org = rec.get("organism") or {}
    tid = org.get("taxonId")
    if tid is None:
        return True
    try:
        return int(tid) == organism_id
    except Exception:
        return False

def acc(rec: dict) -> Optional[str]:
    a = rec.get("primaryAccession") or rec.get("uniProtkbId")
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

def extract_active_sites(rec: dict, seq: Optional[str]) -> List[Dict[str, Optional[str]]]:
    feats = rec.get("features") or []
    out = []
    seen: Set[Tuple[int, Optional[str], Optional[str]]] = set()
    for ft in feats:
        if not isinstance(ft, dict):
            continue
        if ft.get("type") != "Active site":
            continue
        loc = ft.get("location") or {}
        st = (loc.get("start") or {}).get("value")
        en = (loc.get("end") or {}).get("value")
        mod_st = (loc.get("start") or {}).get("modifier")
        mod_en = (loc.get("end") or {}).get("modifier")
        try:
            if st is None or en is None:
                continue
            st_i = int(st); en_i = int(en)
            if st_i != en_i:
                continue
            if (mod_st and mod_st != "EXACT") or (mod_en and mod_en != "EXACT"):
                continue
            aa = None
            if seq and 1 <= st_i <= len(seq):
                aa = seq[st_i - 1]
            desc = ft.get("description")
            if isinstance(desc, str):
                desc = desc.strip() or None
            key = (st_i, aa, desc)
            if key not in seen:
                seen.add(key)
                out.append({"pos": st_i, "aa": aa, "description": desc})
        except Exception:
            continue
    out.sort(key=lambda x: (x["pos"], x.get("aa") or "", x.get("description") or ""))
    return out

def extract_ec_numbers(rec: dict) -> list[str]:
    ecs: set[str] = set()

    def _add(x):
        if isinstance(x, str) and x.strip():
            ecs.add(x.strip())

    def _from_block(b: dict):
        if not isinstance(b, dict):
            return
        # старые/редкие варианты: ecNumber: "1.2.3.4" или список строк
        v = b.get("ecNumber")
        if isinstance(v, str):
            _add(v)
        elif isinstance(v, list):
            for x in v:
                _add(x)
        # новый вариант: ecNumbers: [{"value": "1.2.3.4"}, ...] или список строк
        v2 = b.get("ecNumbers")
        if isinstance(v2, list):
            for x in v2:
                if isinstance(x, dict):
                    _add(x.get("value"))
                else:
                    _add(x)

    # 1) comments → catalytic activity
    for c in (rec.get("comments") or []):
        if isinstance(c, dict) and c.get("commentType") in {"CATALYTIC_ACTIVITY", "catalytic activity"}:
            _add(c.get("ecNumber"))  # иногда встречается
            rxn = c.get("reaction") or {}
            _add(rxn.get("ecNumber"))

    # 2) cross-references → EC
    for xref in (rec.get("uniProtKBCrossReferences") or []):
        if isinstance(xref, dict) and xref.get("database") == "EC":
            _add(xref.get("id"))

    # 3) proteinDescription во всех ветках
    pd = rec.get("proteinDescription") or {}

    # recommendedName + submissionNames/alternativeNames (могут быть list или dict)
    for key in ("recommendedName", "submissionNames", "alternativeNames"):
        block = pd.get(key)
        blocks = block if isinstance(block, list) else [block] if block else []
        for b in blocks:
            _from_block(b)

    # includes / contains: внутри снова могут быть recommendedName/alternativeNames
    for key in ("includes", "contains", "fragments"):
        items = pd.get(key) or []
        if not isinstance(items, list):
            items = [items]
        for it in items:
            if not isinstance(it, dict):
                continue
            _from_block(it)
            _from_block(it.get("recommendedName") or {})
            # иногда EC дублируется и в alternativeNames внутри include/contain
            alts = it.get("alternativeNames")
            alts = alts if isinstance(alts, list) else [alts] if alts else []
            for a in alts:
                _from_block(a)

    # финал
    return sorted(ecs)


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Build HUMAN UniProt metadata JSON from a single dump (.json.gz)")
    ap.add_argument("--in_json_gz", required=True, help="UniProt human dump (.json.gz)")
    ap.add_argument("--out_json", required=True, help="Output JSON path")
    ap.add_argument("--organism_id", type=int, default=9606)
    ap.add_argument("--reviewed_only", action="store_true")
    args = ap.parse_args()

    out: Dict[str, Dict[str, Any]] = {}
    n_seen = 0
    n_human = 0

    for rec in iter_records_json_gz(args.in_json_gz):
        n_seen += 1
        if not is_human(rec, args.organism_id):
            continue
        n_human += 1

        if args.reviewed_only and not rec.get("reviewed", False):
            continue

        a = acc(rec)
        if not a:
            continue

        full_name = extract_full_name(rec)
        gene_name = extract_gene_name(rec)
        ec_numbers = extract_ec_numbers(rec)
        seq = seq_string(rec)
        active_sites = extract_active_sites(rec, seq)

        out[a] = {
            "full_name": full_name,
            "gene_name": gene_name,
            "ec_numbers": ec_numbers,
            "active_sites": active_sites,
        }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"[OK] scanned: {n_seen} | human: {n_human} | wrote entries: {len(out)} -> {args.out_json}")

if __name__ == "__main__":
    main()
