#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Фильтрация таблицы предсказаний с форматами вроде:
  ['0', 'A', '716'],['0','A','836']
— берём ТОЛЬКО третий элемент из каждой тройки (индекс остатка).

Ожидаемые колонки по умолчанию:
- "protein id (uniprot / PDB)"
- "predicted with 35% threshold"
- "predicted with 65% threshold"
- "predicted with 85% threshold"
- "known catalytic sites"
- "EC number (if exists)"
- "protein name"
- "gene name"

Выход:
- CSV с подмножеством строк, где есть хотя бы одна новая позиция,
  плюс колонки novel_sites_35/65/85, n_novel_35/65/85, any_novel.
"""

import argparse
import re
import pandas as pd

# Находим ТОЛЬКО третий элемент внутри каждой тройки ['..','..','123']
TRIPLET_IDX_RE = re.compile(r"\[\s*'[^']*'\s*,\s*'[^']*'\s*,\s*'(\d+)'\s*\]")
# Универсальный «ловец» чисел — на случай простых списков "12,34"
ANY_NUM_RE = re.compile(r"\d+")

def parse_pred_triplets(cell) -> set[int]:
    """
    Для предсказательных колонок: извлекаем ТРЕТИЙ элемент из каждой тройки.
    Примеры:
      "['0','A','125']"                       -> {125}
      "['0','A','216'],['0','A','236']"       -> {216,236}
      "" / NaN                                 -> {}
    Если вдруг тройки не распознаны, fallback: вытащить любые числа.
    """
    if cell is None:
        return set()
    if isinstance(cell, float) and pd.isna(cell):
        return set()
    s = str(cell)
    hits = TRIPLET_IDX_RE.findall(s)
    if hits:
        return {int(x) for x in hits}
    # fallback: любой числовой мусор (на случай другого формата)
    return {int(x) for x in ANY_NUM_RE.findall(s)}

def parse_known_any(cell) -> set[int]:
    """
    Для колонки 'known catalytic sites': просто вытянуть все числа.
    Примеры:
      "125, 236"   -> {125,236}
      "" / NaN     -> {}
    """
    if cell is None:
        return set()
    if isinstance(cell, float) and pd.isna(cell):
        return set()
    return {int(x) for x in ANY_NUM_RE.findall(str(cell))}

def main():
    ap = argparse.ArgumentParser(description="Filter rows with novel (non-UniProt) catalytic sites (triplet format).")
    ap.add_argument("--in_csv", required=True, help="Входной CSV от predict.py")
    ap.add_argument("--out_csv", required=True, help="Выходной CSV (только новизна)")
    # Можешь переопределить имена колонок при необходимости:
    ap.add_argument("--pred_cols", nargs="+",
                    default=["predicted with 35% threshold",
                             "predicted with 65% threshold",
                             "predicted with 85% threshold"],
                    help="Список предсказательных колонок (в формате из predict.py)")
    ap.add_argument("--col_known", default="known catalytic sites",
                    help="Колонка с UniProt-известными позициями")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Парсим known
    known_sets = df[args.col_known].apply(parse_known_any) if args.col_known in df.columns else pd.Series([set()]*len(df))

    # Для каждой предсказательной колонки считаем новизну
    novel_cols = []   # имена колонок с novel_sites_XX
    count_cols = []   # имена колонок с n_novel_XX
    any_novel_mask = pd.Series(False, index=df.index)

    for col in args.pred_cols:
        if col not in df.columns:
            # пропускаем отсутствующие колонки тихо
            continue
        preds = df[col].apply(parse_pred_triplets)
        novel = [sorted(p - k) for p, k in zip(preds, known_sets)]
        tag = re.search(r"(\d+)%", col)
        suffix = tag.group(1) if tag else col.replace("predicted with ", "").replace(" threshold", "").strip()
        col_sites = f"novel_sites_{suffix}"
        col_count = f"n_novel_{suffix}"
        df[col_sites] = [",".join(map(str, x)) if x else "" for x in novel]
        df[col_count] = [len(x) for x in novel]
        novel_cols.append(col_sites)
        count_cols.append(col_count)
        any_novel_mask = any_novel_mask | (df[col_count] > 0)

    # any_novel
    df["any_novel"] = any_novel_mask

    # Фильтруем только строки с новыми позициями
    out = df[df["any_novel"]].copy()

    # Сортировка: сначала по самому строгому порогу, который есть, потом по остальным
    sort_by = [c for c in ["n_novel_85", "n_novel_65", "n_novel_35"] if c in out.columns]
    if sort_by:
        out = out.sort_values(sort_by, ascending=False)

    out.to_csv(args.out_csv, index=False)
    print(f"[OK] {len(out)} rows with novel catalytic sites -> {args.out_csv}")

if __name__ == "__main__":
    main()
