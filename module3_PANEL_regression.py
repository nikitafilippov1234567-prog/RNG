#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МОДУЛЬ 3: ПАНЕЛЬНАЯ РЕГРЕССИЯ (ПРАВИЛЬНО!)
Используем полную панельную структуру: регионы × время
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os

try:
    from linearmodels.panel import PanelOLS
    HAS_PANEL = True
except:
    HAS_PANEL = False
    print("Установите: pip install linearmodels")
    exit(1)

OUTPUT_FOLDER = r"G:\downloads\price_factors_results_FULL"


# ============================================================================
# 1. ЗАГРУЗКА ПАНЕЛЬНЫХ ДАННЫХ
# ============================================================================

print("\n[1] Загрузка панельных данных...")

df = pd.read_csv(f"{OUTPUT_FOLDER}/panel_FULL_normalized.csv", sep=";", parse_dates=['date'])

print(f"Наблюдений: {len(df)}")
print(f"Регионов: {df['region'].nunique()}")
print(f"Периодов: {df['date'].nunique()}")

# RAW предикторы (без лагов, без ratio)
raw_vars = []
ratio_keywords = ['ratio', 'real_rate', 'credit_availability', 
                  'supply_intensity', 'primary_share', 'price_ratio']

for col in df.columns:
    if col not in ['date','region','price','price_secondary'] and '_lag' not in col:
        if not any(kw in col.lower() for kw in ratio_keywords):
            raw_vars.append(col)

print(f"\nRAW предикторов: {len(raw_vars)}")
for v in raw_vars:
    print(f"   • {v}")

# ============================================================================
# 2. ФУНКЦИЯ ПАНЕЛЬНОЙ РЕГРЕССИИ
# ============================================================================

def run_panel_regression(df, y_var, X_vars, name="Model"):
    """Панельная регрессия с FE"""
    
    # Подготовка
    panel_data = df[['date','region',y_var] + X_vars].copy()
    panel_data = panel_data.dropna()
    
    if len(panel_data) < 100:
        print(f"   {name}: мало данных")
        return None
    
    panel_data = panel_data.set_index(['region','date'])
    
    # Нормализация
    for col in [y_var] + X_vars:
        panel_data[f'{col}_n'] = (panel_data[col] - panel_data[col].mean()) / panel_data[col].std()
    
    # Формула с FE
    X_norm = [f'{v}_n' for v in X_vars]
    formula = f"{y_var}_n ~ {' + '.join(X_norm)} + EntityEffects"
    
    print(f"\n   {name}:")
    print(f"      Наблюдений: {len(panel_data)}")
    print(f"      Регионов: {panel_data.index.get_level_values(0).nunique()}")
    print(f"      Признаков: {len(X_vars)}")
    
    try:
        model = PanelOLS.from_formula(formula, data=panel_data)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        print(f"      R² within: {results.rsquared_within:.4f}")
        print(f"      R² between: {results.rsquared_between:.4f}")
        print(f"      R² overall: {results.rsquared_overall:.4f}")
        
        # Коэффициенты
        coefs = pd.DataFrame({
            'feature': results.params.index,
            'coefficient': results.params.values,
            'std_error': results.std_errors.values,
            'pvalue': results.pvalues.values,
            'significant': ['***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '' 
                           for p in results.pvalues.values]
        })
        
        # Значимые
        sig = coefs[coefs['pvalue'] < 0.05].sort_values('pvalue')
        
        if len(sig) > 0:
            print(f"\n      Значимые (p<0.05): {len(sig)}")
            for idx, row in sig.head(5).iterrows():
                print(f"         {row['feature']:<30s} β={row['coefficient']:>7.4f} {row['significant']}")
        
        return {
            'name': name,
            'r2_within': results.rsquared_within,
            'r2_between': results.rsquared_between,
            'r2_overall': results.rsquared_overall,
            'n_obs': len(panel_data),
            'n_regions': panel_data.index.get_level_values(0).nunique(),
            'n_features': len(X_vars),
            'coefs': coefs,
            'results': results
        }
    
    except Exception as e:
        print(f"      ✗ Ошибка: {e}")
        return None

# ============================================================================
# 3. МОДЕЛЬ 1: PRICE (PRIMARY) ~ RAW
# ============================================================================

print("\n" + "="*100)
print("МОДЕЛЬ 1: PRICE (PRIMARY) ~ RAW (ПАНЕЛЬНАЯ)")
print("="*100)

result1 = run_panel_regression(df, 'price', raw_vars, "Model1_Price_RAW_Panel")

if result1:
    result1['coefs'].to_csv(f"{OUTPUT_FOLDER}/Model1_Price_RAW_panel_coefs.csv", sep=";", index=False)
    
    with open(f"{OUTPUT_FOLDER}/Model1_Price_RAW_panel_summary.txt", 'w') as f:
        f.write(str(result1['results'].summary))

# ============================================================================
# 4. МОДЕЛЬ 2: PRICE (SECONDARY) ~ RAW
# ============================================================================

print("\n" + "="*100)
print("МОДЕЛЬ 2: PRICE (SECONDARY) ~ RAW (ПАНЕЛЬНАЯ)")
print("="*100)

result2 = run_panel_regression(df, 'price_secondary', raw_vars, "Model2_PriceSecondary_RAW_Panel")

if result2:
    result2['coefs'].to_csv(f"{OUTPUT_FOLDER}/Model2_PriceSecondary_RAW_panel_coefs.csv", sep=";", index=False)
    
    with open(f"{OUTPUT_FOLDER}/Model2_PriceSecondary_RAW_panel_summary.txt", 'w') as f:
        f.write(str(result2['results'].summary))

# ============================================================================
# 5. МОДЕЛЬ 3: FIRST DIFF (проверка робастности)
# ============================================================================

print("\n" + "="*100)
print("МОДЕЛЬ 3: FIRST DIFFERENCES (ПРОВЕРКА РОБАСТНОСТИ)")
print("="*100)

# Создаем первые разности
df_diff = df.copy()
df_diff = df_diff.sort_values(['region','date'])

for var in ['price'] + raw_vars:
    df_diff[f'{var}_diff'] = df_diff.groupby('region')[var].diff()

df_diff = df_diff.dropna()

raw_diff_vars = [f'{v}_diff' for v in raw_vars]

result3 = run_panel_regression(df_diff, 'price_diff', raw_diff_vars, "Model3_Price_FirstDiff_Panel")

if result3:
    result3['coefs'].to_csv(f"{OUTPUT_FOLDER}/Model3_Price_FirstDiff_panel_coefs.csv", sep=";", index=False)

# ============================================================================
# 6. СРАВНЕНИЕ
# ============================================================================

print("\n" + "="*100)
print("СРАВНЕНИЕ ПАНЕЛЬНЫХ МОДЕЛЕЙ")
print("="*100)

results = []
for res in [result1, result2, result3]:
    if res:
        results.append({
            'Model': res['name'],
            'R2_within': res['r2_within'],
            'R2_between': res['r2_between'],
            'R2_overall': res['r2_overall'],
            'N_obs': res['n_obs'],
            'N_regions': res['n_regions'],
            'N_features': res['n_features']
        })

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('R2_within', ascending=False)

print(f"\n{comparison_df.to_string(index=False)}")

comparison_df.to_csv(f"{OUTPUT_FOLDER}/panel_models_comparison.csv", sep=";", index=False)

best = comparison_df.iloc[0]
print(f"\nЛучшая: {best['Model']}")
print(f"   R² within: {best['R2_within']:.4f}")
print(f"   N obs: {int(best['N_obs'])} (вместо 39!)")

print(f"\nПанельная регрессия:")
print(f"   • Использует ВСЮ информацию ({int(best['N_obs'])} наблюдений)")
print(f"   • FE контролирует региональные эффекты")
print(f"   • Кластеризованные SE по регионам")
print(f"   • Within R² - качество объяснения временной вариации")

print(f"\nМОДУЛЬ 3 ЗАВЕРШЕН")
print("="*100 + "\n")
