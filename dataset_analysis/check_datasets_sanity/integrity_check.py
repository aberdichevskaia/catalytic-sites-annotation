#!/usr/bin/env python3
import pandas as pd
import sys

def check_column(df, col):
    """Возвращает словарь {значение: set типов сплитов}, где len(set)>1."""
    bad = {}
    for val, group in df.groupby(col):
        types = set(group['Set_Type'])
        if len(types) > 1:
            bad[val] = types
    return bad

def main(path):
    df = pd.read_csv(path)
    # Проверяем наличие нужных колонок
    required = {'Cluster_1','Cluster_2','Component_ID','Set_Type'}
    if not required.issubset(df.columns):
        print("В файле не найдены все необходимые колонки:", required - set(df.columns))
        sys.exit(1)

    overall_ok = True
    for col in ['Cluster_1','Cluster_2','Component_ID']:
        bad = check_column(df, col)
        if bad:
            overall_ok = False
            print(f"\n❌ Найдены нарушения по «{col}»:")
            for val, types in bad.items():
                print(f"  {col} = {val!r} → встречается в сплитах {sorted(types)}")
        else:
            print(f"✅ Все значения «{col}» целиком в одном сплите.")

    if overall_ok:
        print("\n✔ Валидация пройдена: с целостностью компонент всё в порядке.")
    else:
        print("\n❗ Обнаружены нарушения. Исправьте распределение указанных значений.")

table_path = "/home/iscb/wolfson/annab4/DB/all_proteins/cross_validation_chem/weight_based_v5/dataset.csv"
main(table_path)
