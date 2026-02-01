#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import sys

# ---------------- ScanNet_Ub in sys.path ----------------
PROJECT_ROOT = "/home/iscb/wolfson/annab4/main_scannet/ScanNet_Ub/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utilities.chimera import annotate_pdb_file # noqa: E402


def make_scaled_csv(
    csv_in: Path,
    csv_out: Path,
    field: str,
    prob_scale: float,
) -> Path:
    """
    Create a temporary CSV where df[field] is multiplied by prob_scale.
    Does NOT modify the original file.
    """
    if prob_scale == 1.0:
        return csv_in

    df = pd.read_csv(csv_in, sep=",")
    if field not in df.columns:
        raise KeyError(f"Field '{field}' not found in {csv_in}. Columns: {list(df.columns)}")

    df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0.0) * float(prob_scale)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    return csv_out



HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{base_id} isoforms</title>

  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar/build/viewer/molstar.css" />
  <script src="https://cdn.jsdelivr.net/npm/molstar/build/viewer/molstar.js"></script>

  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }}
    #viewer {{ width: 100%; height: 70vh; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
    th {{ text-align: left; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
  </style>
</head>

<body>
  <h1 class="mono">{base_id}</h1>
  <div id="viewer"></div>

  <table>
    <thead>
      <tr>
        <th>Show</th>
        <th>Isoform</th>
        <th>File</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>

  <script>
    const MS = window.molstar;

    function guessFormat(path) {{
      const p = path.toLowerCase();
      if (p.endsWith('.cif') || p.endsWith('.mmcif')) return 'mmcif';
      if (p.endsWith('.pdb')) return 'pdb';
      return 'mmcif';
    }}

    async function main() {{
      const spec = await fetch('isoforms.json').then(r => r.json());

      const viewer = await MS.Viewer.create('viewer', {{
        layoutIsExpanded: true,
        layoutShowControls: true,
        layoutShowSequence: true
      }});

      const refByIso = new Map();

      async function setVisible(ref, visible) {{
        await viewer.plugin.build().to(ref).updateState({{ isHidden: !visible }}).commit();
      }}

      const tbody = document.getElementById('rows');

      for (const iso of spec.isoforms) {{
        const url = iso.file;
        const format = iso.format || guessFormat(url);

        await viewer.loadStructureFromUrl(url, format);

        const hs = viewer.plugin.managers.structure.hierarchy.current.structures;
        const last = hs[hs.length - 1];
        const ref = last.cell.transform.ref;
        refByIso.set(iso.isoform_id, ref);

        const tr = document.createElement('tr');

        const tdCheck = document.createElement('td');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true;
        cb.addEventListener('change', async () => {{
          await setVisible(refByIso.get(iso.isoform_id), cb.checked);
        }});
        tdCheck.appendChild(cb);

        const tdId = document.createElement('td');
        tdId.textContent = iso.isoform_id;
        tdId.className = 'mono';

        const tdFile = document.createElement('td');
        tdFile.textContent = iso.file;
        tdFile.className = 'mono';

        tr.appendChild(tdCheck);
        tr.appendChild(tdId);
        tr.appendChild(tdFile);
        tbody.appendChild(tr);
      }}
    }}

    main().catch(err => {{
      console.error(err);
      document.body.insertAdjacentHTML('afterbegin', '<p style="color:red">Failed to load. See console.</p>');
    }});
  </script>
</body>
</html>
"""


ROOT_INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>Isoform browser</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; }
    input { font-size: 16px; padding: 8px 10px; width: 260px; }
    button { font-size: 16px; padding: 8px 12px; margin-left: 8px; }
    .hint { color: #666; margin-top: 10px; }
  </style>
</head>
<body>
  <h1>Isoform viewer</h1>
  <div>
    <input id="base" placeholder="e.g. O14492" />
    <button id="go">Open</button>
  </div>
  <div class="hint">Откроет: site/proteins/&lt;base_id&gt;/index.html</div>

  <script>
    document.getElementById('go').onclick = () => {
      const id = document.getElementById('base').value.trim();
      if (!id) return;
      window.location.href = `proteins/${encodeURIComponent(id)}/index.html`;
    };
    document.getElementById('base').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') document.getElementById('go').click();
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Mol* isoform site under site/")

    p.add_argument("--base-ids", required=True, help="Path to file with base_id per line.")
    p.add_argument("--groups-dir", required=True, help="Directory with <base_id>_isoforms.txt")
    p.add_argument("--pred-dir", required=True, help="Directory with per-isoform CSV predictions")
    p.add_argument("--site-dir", default="site", help="Output site directory (default: site/)")

    p.add_argument("--field", default="Binding site probability", help="CSV field to write into B-factor")
    p.add_argument("--mini", type=float, default=0.0, help="Min value for probabilities")
    p.add_argument("--maxi", type=float, default=0.8, help="Max value for probabilities")
    p.add_argument("--prob-scale", type=float, default=1.0, help="Multiply probabilities before writing (e.g. 100)")

    p.add_argument("--chimerax", default="chimerax", help="ChimeraX executable")
    p.add_argument("--no-align", action="store_true", help="Skip ChimeraX alignment step")
    p.add_argument("--chain", default=None, help="Optional chain ID to align only that chain, e.g. A")

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing per-protein folders")
    return p.parse_args()


def read_base_ids(path: Path) -> List[str]:
    ids = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def read_isoforms_txt(path: Path) -> List[Path]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(Path(s))
    return out


def isoform_id_from_structure_path(p: Path) -> str:
    # O14492-1.pdb -> O14492-1
    return p.name.split(".")[0]


def find_csv(pred_dir: Path, isoform_id: str) -> Optional[Path]:
    # Prefer exact match
    exact = pred_dir / f"{isoform_id}.csv"
    if exact.exists():
        return exact

    # Fallback: glob
    matches = sorted(pred_dir.glob(f"{isoform_id}*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return None


def write_alignment_cxc(
    annotated_files: List[Path],
    out_cxc: Path,
    aligned_dir: Path,
    chain: Optional[str] = None,
) -> None:
    """
    Open #1 as reference, then for each next file:
      open -> mm to #1 -> save aligned mmCIF -> close
    """
    aligned_dir.mkdir(parents=True, exist_ok=True)

    def mm_sel(model_num: int) -> str:
        if chain is None:
            return f"#{model_num}"
        return f"#{model_num}/{chain}"

    with out_cxc.open("w") as f:
        f.write("# Auto-generated alignment script\n")

        # Reference
        f.write(f'open "{annotated_files[0].as_posix()}"\n')

        # Align each next, save as mmCIF
        for i, src in enumerate(annotated_files[1:], start=2):
            f.write(f'open "{src.as_posix()}"\n')
            f.write(f"mm {mm_sel(i)} to {mm_sel(1)}\n")

            iso_id = src.stem.replace(".annot", "").replace(".aln", "")
            out_path = aligned_dir / f"{iso_id}.aln.annot.cif"
            f.write(f'save "{out_path.as_posix()}" models #{i}\n')
            f.write(f"close #{i}\n")

        # Optionally save reference as well
        ref_id = annotated_files[0].stem.replace(".annot", "").replace(".aln", "")
        ref_out = aligned_dir / f"{ref_id}.aln.annot.cif"
        f.write(f'save "{ref_out.as_posix()}" models #1\n')

        f.write("exit\n")


def run_chimerax(chimerax: str, cxc_path: Path) -> None:
    cmd = [chimerax, "--nogui", "--cmd", f'open "{cxc_path.as_posix()}"', "--exit"]
    subprocess.run(cmd, check=True)


def write_protein_page(protein_dir: Path, base_id: str) -> None:
    (protein_dir / "index.html").write_text(HTML_TEMPLATE.format(base_id=base_id))


def write_root_page(site_dir: Path) -> None:
    (site_dir / "index.html").write_text(ROOT_INDEX_HTML)


def build_one_base_id(
    base_id: str,
    groups_dir: Path,
    pred_dir: Path,
    site_dir: Path,
    field: str,
    mini: float,
    maxi: float,
    prob_scale: float,
    chimerax: str,
    do_align: bool,
    chain: Optional[str],
    overwrite: bool,
) -> Tuple[int, int]:
    """
    Returns (n_ok, n_skipped)
    """
    isoforms_txt = groups_dir / f"{base_id}_isoforms.txt"
    if not isoforms_txt.exists():
        print(f"[{base_id}] SKIP: missing {isoforms_txt}")
        return 0, 1

    struct_paths = read_isoforms_txt(isoforms_txt)
    if not struct_paths:
        print(f"[{base_id}] SKIP: empty isoforms list")
        return 0, 1

    protein_dir = site_dir / "proteins" / base_id
    structures_dir = protein_dir / "structures"
    work_dir = site_dir / "_work" / base_id

    if protein_dir.exists() and overwrite:
        shutil.rmtree(protein_dir)
    protein_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = work_dir / "annotated"
    aligned_dir = work_dir / "aligned"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    annotated_files: List[Path] = []
    n_ok = 0
    n_skipped = 0

    # 1) Annotate each isoform structure with predictions into B-factor
    for sp in struct_paths:
        iso_id = isoform_id_from_structure_path(sp)
        csv_path = find_csv(pred_dir, iso_id)
        if csv_path is None:
            print(f"[{base_id}] WARN: no CSV for {iso_id} in {pred_dir}, skip isoform")
            n_skipped += 1
            continue

        # Keep same extension as input for annotation step (PDB->PDB, CIF->CIF)
        out_ext = sp.suffix.lower()
        if out_ext not in [".pdb", ".cif", ".mmcif"]:
            out_ext = ".pdb"

        out_annot = annotated_dir / f"{iso_id}.annot{out_ext}"

        # If you want scaling (e.g. *100), easiest is: temporarily write a scaled copy CSV or
        # scale in annotate_pdb_file. Here we do a simple approach: pass field and then post-scale
        # is not possible without rewriting. So: recommended prob_scale=1.0 OR implement scaling inside annotate.
        tmp_scaled_csv = work_dir / "scaled_csv" / f"{iso_id}.scaled.csv"
        csv_for_annot = make_scaled_csv(
            csv_in=csv_path,
            csv_out=tmp_scaled_csv,
            field=field,
            prob_scale=prob_scale,
        )

        annotate_pdb_file(
            pdb_file=str(sp),
            csv_file=str(csv_for_annot),
            output_file=str(out_annot),
            output_script=False,
            mini=mini,
            maxi=maxi,
            version="default",
            field=field,
        )

        annotated_files.append(out_annot)
        n_ok += 1

    if len(annotated_files) == 0:
        print(f"[{base_id}] SKIP: no annotated files")
        return 0, n_skipped + 1

    # 2) Align in ChimeraX headless (optional)
    final_structs: List[Path]
    if do_align and len(annotated_files) > 1:
        cxc_path = work_dir / f"{base_id}.align.cxc"
        write_alignment_cxc(annotated_files, cxc_path, aligned_dir, chain=chain)
        run_chimerax(chimerax, cxc_path)

        final_structs = sorted(aligned_dir.glob("*.aln.annot.cif"))
        if not final_structs:
            print(f"[{base_id}] WARN: alignment produced no files, fallback to annotated")
            final_structs = annotated_files
    else:
        final_structs = annotated_files

    # 3) Copy structures into site/proteins/<base_id>/structures/
    isoforms_spec = []
    for fp in final_structs:
        dst = structures_dir / fp.name
        shutil.copy2(fp, dst)

        iso_id = fp.name.split(".")[0]  # up to first dot
        isoforms_spec.append(
            {
                "isoform_id": iso_id,
                "file": f"structures/{dst.name}",
                "format": "mmcif" if dst.suffix.lower() in [".cif", ".mmcif"] else "pdb",
            }
        )

    # 4) Write isoforms.json + page
    (protein_dir / "isoforms.json").write_text(
        json.dumps({"base_id": base_id, "isoforms": isoforms_spec}, indent=2)
    )
    write_protein_page(protein_dir, base_id)

    print(f"[{base_id}] OK: {len(isoforms_spec)} isoforms -> {protein_dir}")
    return n_ok, n_skipped


def main() -> None:
    args = parse_args()

    base_ids = read_base_ids(Path(args.base_ids))
    groups_dir = Path(args.groups_dir)
    pred_dir = Path(args.pred_dir)
    site_dir = Path(args.site_dir)

    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "proteins").mkdir(parents=True, exist_ok=True)
    (site_dir / "_work").mkdir(parents=True, exist_ok=True)

    write_root_page(site_dir)

    manifest = []
    total_ok = 0
    total_skip = 0

    for base_id in base_ids:
        n_ok, n_skipped = build_one_base_id(
            base_id=base_id,
            groups_dir=groups_dir,
            pred_dir=pred_dir,
            site_dir=site_dir,
            field=args.field,
            mini=args.mini,
            maxi=args.maxi,
            prob_scale=args.prob_scale,
            chimerax=args.chimerax,
            do_align=(not args.no_align),
            chain=args.chain,
            overwrite=args.overwrite,
        )
        if n_ok > 0:
            manifest.append(base_id)
        total_ok += n_ok
        total_skip += n_skipped

    (site_dir / "manifest.json").write_text(json.dumps({"base_ids": manifest}, indent=2))
    print(f"DONE. annotated_ok={total_ok} skipped={total_skip}")
    print(f"Open: {site_dir / 'index.html'}")


if __name__ == "__main__":
    main()
