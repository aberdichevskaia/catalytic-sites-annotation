#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast

import pandas as pd


def count_sites(cell) -> int:
    """
    Подсчитать количество предсказанных аминокислот в ячейке.

    Поддерживает форматы:
    - "" или NaN -> 0
    - "243,286" -> 2
    - "243" -> 1
    - "['0', 'A', '243']" -> 1
    - "['0', 'A', '243', '0', 'A', '286']" -> 2
    """
    if cell is None:
        return 0

    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none"}:
        return 0

    # 1) Пытаемся распарсить как литерал Python (список/кортеж)
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        v = None

    if isinstance(v, (list, tuple)):
        items = list(v)

        # кейс вида ['0','A','243', ...] -> предполагаем тройки (model, chain, resid)
        if len(items) % 3 == 0:
            return len(items) // 3

        # иначе пробуем посчитать только числовые элементы
        cnt_num = 0
        for it in items:
            try:
                int(it)
                cnt_num += 1
            except (TypeError, ValueError):
                pass
        if cnt_num > 0:
            return cnt_num
        return len(items)

    # 2) Фоллбек: считаем, что это строка с разделителем запятая
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    if not tokens:
        return 0

    cnt_num = 0
    for t in tokens:
        try:
            int(t)
            cnt_num += 1
        except ValueError:
            pass

    return cnt_num if cnt_num > 0 else len(tokens)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Отфильтровать строки по base uniprot id, оставив только те группы, "
            "где различается число предсказанных сайтов при заданном пороге."
        )
    )
    parser.add_argument("--in_csv", required=True, help="Входной CSV с предсказаниями")
    parser.add_argument("--out_csv", required=True, help="Выходной CSV")
    parser.add_argument(
        "--thr",
        type=int,
        choices=[35, 65, 85],
        required=True,
        help="Порог (в процентах): 35, 65 или 85",
    )
    parser.add_argument(
        "--base_col",
        default="base uniprot id",
        help="Имя колонки с base id (по умолчанию 'base uniprot id')",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    thr_col = f"predicted with {args.thr}% threshold"
    if thr_col not in df.columns:
        raise SystemExit(f"Колонка '{thr_col}' не найдена в CSV.")

    if args.base_col not in df.columns:
        raise SystemExit(f"Колонка с base id '{args.base_col}' не найдена в CSV.")

    # 1) считаем число предсказанных сайтов
    df["_n_sites"] = df[thr_col].apply(count_sites)

    # 2) для каждого base id считаем, есть ли различия в числе сайтов
    grp = df.groupby(args.base_col)["_n_sites"].transform(lambda x: x.nunique() > 1)

    # 3) оставляем только те строки, где в группе есть различия
    df_out = df[grp].copy()

    # вспомогательную колонку можно убрать, если не нужна
    df_out.drop(columns=["_n_sites"], inplace=True)

    df_out.to_csv(args.out_csv, index=False)
    print(
        f"Сохранено {len(df_out)} строк в {args.out_csv} "
        f"(из {len(df)} исходных), порог {args.thr}%."
    )


if __name__ == "__main__":
    main()
