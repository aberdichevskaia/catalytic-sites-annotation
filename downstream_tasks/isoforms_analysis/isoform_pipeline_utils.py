#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for isoform prediction and ranking pipelines.

Keep this module free of machine-specific paths; pass all paths via CLI.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Dict, Optional

import numpy as np


def ensure_trailing_slash(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    return path if path.endswith(os.sep) else path + os.sep


def stem_from_path(path: str) -> str:
    """Return file stem (case-sensitive, important for MSA/ESM2 matching)."""
    return os.path.splitext(os.path.basename(path))[0]


def collect_structure_paths(structures_dir: str) -> Dict[str, str]:
    """
    Collect structure paths and map them by stem.

    Preference:
    - If both .cif and .pdb exist for the same stem, keep .pdb.
    """
    cif_paths = glob(os.path.join(structures_dir, "*.cif"))
    pdb_paths = glob(os.path.join(structures_dir, "*.pdb"))
    by_stem: Dict[str, str] = {}

    for path in cif_paths + pdb_paths:
        stem = stem_from_path(path)
        prev = by_stem.get(stem)
        if prev is None or (prev.lower().endswith(".cif") and path.lower().endswith(".pdb")):
            by_stem[stem] = path

    return by_stem


def esm_cache_path(esm2_root: str, origin: str) -> str:
    sub = origin[:2] if len(origin) >= 2 else "__"
    return os.path.join(esm2_root, sub, f"{origin}.npy")


def esm_exists_for_origin(esm2_root: Optional[str], origin: str) -> bool:
    """
    Check cached ESM2 embedding:
      <esm2_root>/<origin[:2]>/<origin>.npy
    Try both exact and lowercased origin (filesystem variability).
    """
    if not esm2_root:
        return False
    p1 = esm_cache_path(esm2_root, origin)
    if os.path.exists(p1):
        return True
    p2 = esm_cache_path(esm2_root, origin.lower())
    return os.path.exists(p2)


def catalytic_channel(arr: np.ndarray) -> np.ndarray:
    """Extract catalytic probability channel from model output."""
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return (arr[:, 0] if arr.shape[1] == 1 else arr[:, 1]).astype(np.float32)
    return np.asarray(arr).reshape(-1).astype(np.float32)
