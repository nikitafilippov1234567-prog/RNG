#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МОДУЛЬ 1: ПАНЕЛЬНЫЕ ТЕСТЫ UNIT ROOT
ПРАВИЛЬНЫЙ подход для панельных данных
Не удаляем нестационарные, просто диагностика для выбора метода
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller

OUTPUT_FOLDER = r"G:\downloads\price_factors_results_FULL"

print("="*100)
print("МОДУЛЬ 1.5: ПАНЕЛЬНЫЕ ТЕСТЫ UNIT ROOT (ДИАГНОСТИКА)")
print("="*100)

# ============================================================================
# 1. ЗАГРУЗКА
# ============================================================================

print("\n[1] Загрузка нормализованных данных...")
df = pd.read_csv(f"{OUTPUT_FOLDER}/panel_FULL_normalized.csv", sep=";", parse_dates=['date'])

all_pred = [c for c in df.columns if c not in ['date','region','price'] and '_lag' not in c]

print(f"✓ {len(df)} наблюдений, {df['region'].nunique()} регионов")
print(f"✓ Переменных: {len(all_pred)}")

# ============================================================================
# 2. ПАНЕЛЬНЫЕ UNIT ROOT ТЕСТЫ
# ============================================================================

print("\n[2] Панельные unit root тесты (по каждому региону)...")

def panel_adf_test(df, variable, max_regions=10):
    """ADF тест для панели (подвыборка регионов)"""
    regions = df['region'].unique()[:max_regions]
    
    results = []
    
    for region in regions:
        series = df[df['region'] == region][variable].dropna()
        
        if len(series) < 20:
            continue
        
        try:
            adf_stat, adf_pval, _, _, _, _ = adfuller(series, maxlag=6, regression='ct')
            
            results.append({
                'region': region,
                'adf_pval': adf_pval,
                'stationary': adf_pval < 0.05
            })
        except:
            continue
    
    return results

test_results = {}

print(f"\n{'Переменная':<30s} {'Стац. регионов':<20s} {'%':<10s}")
print("="*70)

for var in all_pred:
    results = panel_adf_test(df, var)
    
    if results:
        n_stationary = sum(r['stationary'] for r in results)
        pct_stationary = (n_stationary / len(results)) * 100
        
        test_results[var] = {
            'n_tested': len(results),
            'n_stationary': n_stationary,
            'pct_stationary': pct_stationary
        }
        
        ratio_str = f"{n_stationary}/{len(results)}"
        print(f"{var:<30s} {ratio_str:<20s} {pct_stationary:>6.1f}%")

# ============================================================================
# 3. РЕКОМЕНДАЦИИ
# ============================================================================

print("\n[3] Анализ и рекомендации...")

highly_non_stat = {k: v for k, v in test_results.items() if v['pct_stationary'] < 30}
mostly_stat = {k: v for k, v in test_results.items() if v['pct_stationary'] >= 70}
mixed = {k: v for k, v in test_results.items() if 30 <= v['pct_stationary'] < 70}

print(f"\nКлассификация:")
print(f"   • Стационарные (≥70%): {len(mostly_stat)}")
print(f"   • Смешанные (30-70%): {len(mixed)}")
print(f"   • Нестационарные (<30%): {len(highly_non_stat)}")

if highly_non_stat:
    print(f"\n⚠ Явно нестационарные:")
    for var in highly_non_stat:
        print(f"   • {var} ({highly_non_stat[var]['pct_stationary']:.0f}% стац.)")


# ============================================================================
# 4. СОЗДАНИЕ ДАТАСЕТА С ПЕРВЫМИ РАЗНОСТЯМИ
# ============================================================================

print("\n[4] Создание датасета с первыми разностями...")

df_diff = df.copy()

# Сортируем
df_diff = df_diff.sort_values(['region', 'date'])

# Первые разности для каждого региона
for var in all_pred:
    df_diff[f'{var}_diff'] = df_diff.groupby('region')[var].diff()

# Удаляем NaN
df_diff = df_diff.dropna()

print(f"Создан датасет с первыми разностями")
print(f"Наблюдений: {len(df_diff)}")

# ============================================================================
# 5. СОХРАНЕНИЕ
# ============================================================================

print("\n[5] Сохранение...")

# Результаты тестов
results_df = pd.DataFrame.from_dict(test_results, orient='index')
results_df['variable'] = results_df.index
results_df = results_df[['variable', 'n_tested', 'n_stationary', 'pct_stationary']]
results_df = results_df.sort_values('pct_stationary')

results_df.to_csv(f"{OUTPUT_FOLDER}/panel_unit_root_tests.csv", sep=";", index=False)

# Датасет с разностями (для проверки робастности)
df_diff.to_csv(f"{OUTPUT_FOLDER}/panel_FULL_first_diff.csv", sep=";", index=False)

# Рекомендации
with open(f"{OUTPUT_FOLDER}/stationarity_recommendations.txt", 'w', encoding='utf-8') as f:
    f.write("РЕКОМЕНДАЦИИ ПО СТАЦИОНАРНОСТИ\n")
    f.write("="*60 + "\n\n")
    f.write("ВАЖНО: НЕ удаляем нестационарные переменные!\n\n")
    f.write("Панельная регрессия с FE устойчива к нестационарности\n")
    f.write("благодаря within-трансформации.\n\n")
    f.write("Стационарные (≥70%):\n")
    for var in mostly_stat:
        f.write(f"   {var} ({mostly_stat[var]['pct_stationary']:.0f}%)\n")
    f.write("\nНестационарные (<30%):\n")
    for var in highly_non_stat:
        f.write(f"   {var} ({highly_non_stat[var]['pct_stationary']:.0f}%)\n")
    f.write("\n→ Используем ВСЕ переменные в FE-модели\n")
    f.write("→ Первые разности доступны для проверки робастности\n")

print(f"✓ panel_unit_root_tests.csv")
print(f"✓ panel_FULL_first_diff.csv (для робастности)")
print(f"✓ stationarity_recommendations.txt")

print(f"\nМОДУЛЬ 1 ЗАВЕРШЕН")
print(f"\nМодуль 2 будет работать со ВСЕМИ переменными (не фильтруем!)")
print("="*100 + "\n")