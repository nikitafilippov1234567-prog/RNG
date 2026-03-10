#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МОДУЛЬ 2B (ИСПРАВЛЕННЫЙ): RATIO + RAW (которые не использовались в RATIO)
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

OUTPUT_FOLDER = r"G:\downloads\price_factors_results_FULL"
THRESHOLD = 0.8

print("="*100)
print("МОДУЛЬ 2B: RATIO + НЕЗАВИСИМЫЕ RAW")
print("="*100)

# ============================================================================
# 1. ЗАГРУЗКА
# ============================================================================

print("\n[1] Загрузка...")

df = pd.read_csv(f"{OUTPUT_FOLDER}/panel_FULL_normalized.csv", sep=";", parse_dates=['date'])

# RATIO переменные
ratio_vars = ['real_rate', 'overdue_ratio', 'credit_availability', 
              'supply_intensity', 'primary_share', 'price_ratio']
ratio_vars = [v for v in ratio_vars if v in df.columns]

# RAW переменные, которые НЕ использовались для создания RATIO
# ИСКЛЮЧАЕМ: rate, inflation (→ real_rate), mortgage_debt, mortgage_overdue (→ overdue_ratio),
#            housing_loans, price (→ credit_availability), offers, housing_completed (→ supply_intensity)

independent_raw = ['sentiment', 'expectations', 'current_state']  # Только индексы!

print(f"RATIO переменных: {len(ratio_vars)}")
print(f"Независимых RAW: {len(independent_raw)}")

# Все переменные для RATIO подхода
all_ratio_vars = ratio_vars + independent_raw

print(f"\nВсе переменные для RATIO подхода:")
for i, v in enumerate(all_ratio_vars, 1):
    print(f"   {i}. {v}")

# Агрегация
df_agg = df.groupby('date')[all_ratio_vars].mean().reset_index()

print(f"\n Агрегировано: {len(df_agg)} наблюдений")

# ============================================================================
# 2. КОРРЕЛЯЦИИ И PCA
# ============================================================================

print("\n[2] Корреляции...")

corr_matrix = df_agg[all_ratio_vars].corr()

graph = defaultdict(set)
n_pairs = 0

for i in range(len(all_ratio_vars)):
    for j in range(i+1, len(all_ratio_vars)):
        c = abs(corr_matrix.iloc[i,j])
        if c > THRESHOLD:
            graph[all_ratio_vars[i]].add(all_ratio_vars[j])
            graph[all_ratio_vars[j]].add(all_ratio_vars[i])
            n_pairs += 1

print(f" Пар >0.8: {n_pairs}")

# Группировка
visited = set()
groups = []

def dfs(node, group):
    visited.add(node)
    group.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, group)

for var in all_ratio_vars:
    if var not in visited:
        if var in graph:
            group = set()
            dfs(var, group)
            if len(group) > 1:
                groups.append(list(group))
        visited.add(var)

independent = [v for v in all_ratio_vars if v not in graph]

print(f"Групп: {len(groups)}")
print(f"Независимых: {len(independent)}")

# PCA
print("\n[3] PCA...")

pca_features = []
pca_info = []

for i, group in enumerate(groups, 1):
    X = df_agg[group].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    n_comp = pca.n_components_
    
    print(f"   Группа {i}: {len(group)} → {n_comp} компонент")
    
    for j in range(n_comp):
        col_name = f'PC_RATIO_{i}_{j+1}'
        df_agg[col_name] = X_pca[:, j]
        pca_features.append(col_name)
    
    loadings = pd.DataFrame(
        pca.components_[:n_comp],
        columns=group,
        index=[f'PC_RATIO_{i}_{j+1}' for j in range(n_comp)]
    )
    
    pca_info.append({
        'group': i,
        'variables': group,
        'loadings': loadings
    })

final_features = independent + pca_features

print(f"\n{len(all_ratio_vars)} → {len(final_features)} признаков")

# ============================================================================
# 4. СОХРАНЕНИЕ
# ============================================================================

print("\n[4] Сохранение...")

df_agg[['date'] + final_features].to_csv(f"{OUTPUT_FOLDER}/data_RATIO_with_PCA.csv", sep=";", index=False)

with open(f"{OUTPUT_FOLDER}/PCA_RATIO_info.txt", 'w', encoding='utf-8') as f:
    f.write("PCA ДЛЯ RATIO + НЕЗАВИСИМЫЕ RAW\n")
    f.write("="*60 + "\n\n")
    for info in pca_info:
        f.write(f"Группа {info['group']}:\n")
        f.write(f"Переменные: {', '.join(info['variables'])}\n\n")
        f.write("Loadings:\n")
        f.write(info['loadings'].to_string())
        f.write("\n\n" + "-"*60 + "\n\n")
    if independent:
        f.write(f"Независимые: {', '.join(independent)}\n")

print(f"data_RATIO_with_PCA.csv")
print(f"PCA_RATIO_info.txt")