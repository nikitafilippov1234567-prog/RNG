#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

try:
    from linearmodels.panel import PanelOLS
    HAS_PANEL = True
except:
    print("⚠ Установите: pip install linearmodels")
    exit(1)

OUTPUT_FOLDER = r"G:\downloads\price_factors_results_FULL"
MAX_LAG = 6

print("="*100)
print("МОДУЛЬ 4 (ПОЛНЫЙ): ЛАГИ - НЕЛИНЕЙНОСТИ - PCA - ВАЖНОСТЬ")
print("="*100)

# ============================================================================
# 1. ЗАГРУЗКА
# ============================================================================

print("\n[1] Загрузка данных с лагами...")

df = pd.read_csv(f"{OUTPUT_FOLDER}/panel_FULL_with_lags.csv", sep=";", parse_dates=['date'])  # Используем файл с лагами для независимого отбора

print(f"✓ Наблюдений: {len(df)}")
print(f"✓ Регионов: {df['region'].nunique()}")

# Базовые переменные (без лагов, без ratio)
base_vars = []
ratio_keywords = ['ratio', 'real_rate', 'credit_availability', 
                  'supply_intensity', 'primary_share', 'price_ratio']

for col in df.columns:
    if col not in ['date','region','price','price_secondary'] and '_lag' not in col:
        if not any(kw in col.lower() for kw in ratio_keywords):
            base_vars.append(col)

print(f"\n✓ Базовых переменных: {len(base_vars)}")
for v in base_vars:
    print(f"   • {v}")

# ============================================================================
# 2. САМОСТОЯТЕЛЬНЫЙ ОТБОР ОПТИМАЛЬНЫХ ЛАГОВ
# ============================================================================

print("\n" + "="*100)
print("[2] САМОСТОЯТЕЛЬНЫЙ ОТБОР ОПТИМАЛЬНЫХ ЛАГОВ")
print("="*100)

def estimate_panel_model(df, y_var, X_vars, name="Model"):
    """Оценивает панельную модель и возвращает статистики"""
    
    panel_data = df[['date','region',y_var] + X_vars].copy()
    panel_data = panel_data.dropna()
    
    if len(panel_data) < 100:
        return None
    
    panel_data = panel_data.set_index(['region','date'])
    
    # Нормализация
    for col in [y_var] + X_vars:
        panel_data[f'{col}_n'] = (panel_data[col] - panel_data[col].mean()) / panel_data[col].std()
    
    X_norm = [f'{v}_n' for v in X_vars]
    formula = f"{y_var}_n ~ {' + '.join(X_norm)} + EntityEffects"
    
    try:
        model = PanelOLS.from_formula(formula, data=panel_data)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        return {
            'name': name,
            'n_obs': results.nobs,
            'r2_within': results.rsquared_within,
            'r2_overall': results.rsquared_overall,
            'aic': results.aic if hasattr(results, 'aic') else None,
            'bic': results.bic if hasattr(results, 'bic') else None,
            'loglik': results.loglik if hasattr(results, 'loglik') else None,
            'results': results
        }
    except:
        return None

print("\nОценка базовой модели...")

baseline = estimate_panel_model(df, 'price', base_vars, "Baseline")

if baseline:
    print(f"\n   Baseline:")
    print(f"   R² within:  {baseline['r2_within']:.4f}")
    print(f"   R² overall: {baseline['r2_overall']:.4f}")
    print(f"   N obs:      {baseline['n_obs']}")
    
    # Сохраняем
    with open(f"{OUTPUT_FOLDER}/Lag_Baseline_summary.txt", 'w', encoding='utf-8') as f:
        f.write(str(baseline['results'].summary))
else:
    print("Не удалось оценить базовую модель")
    exit(1)

# Тестирование лагов
print(f"\nТЕСТИРОВАНИЕ ЛАГОВ (1-{MAX_LAG}) ДЛЯ КАЖДОЙ ПЕРЕМЕННОЙ")

lag_results = []

for base_var in base_vars:
    print(f"\n{'='*100}")
    print(f"Переменная: {base_var}")
    print(f"{'='*100}")
    
    # Проверяем доступность лагов
    available_lags = []
    for lag in range(1, MAX_LAG + 1):
        if f'{base_var}_lag{lag}' in df.columns:
            available_lags.append(lag)
    
    if not available_lags:
        print(f"Лаги недоступны")
        continue
    
    print(f"   Доступные лаги: {available_lags}")
    
    # Тестируем каждый лаг ОТДЕЛЬНО
    for lag in available_lags:
        lag_var = f'{base_var}_lag{lag}'
        
        # Модель: все базовые + ОДИН лаг
        test_vars = base_vars.copy()
        test_vars.append(lag_var)
        
        result = estimate_panel_model(df, 'price', test_vars, f"{base_var}_lag{lag}")
        
        if result:
            # Сравнение с baseline
            delta_r2 = result['r2_within'] - baseline['r2_within']
            
            # Значимость лага
            lag_coef_name = f'{lag_var}_n'
            if lag_coef_name in result['results'].params.index:
                coef = result['results'].params[lag_coef_name]
                pval = result['results'].pvalues[lag_coef_name]
                se = result['results'].std_errors[lag_coef_name]
                
                lag_results.append({
                    'variable': base_var,
                    'lag': lag,
                    'lag_var': lag_var,
                    'r2_within': result['r2_within'],
                    'delta_r2': delta_r2,
                    'coefficient': coef,
                    'std_error': se,
                    'pvalue': pval,
                    'significant': pval < 0.05,
                    'n_obs': result['n_obs']
                })
                
                sig = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else ''
                print(f"   lag{lag}: R²={result['r2_within']:.4f} (ΔR²={delta_r2:+.4f}), "
                      f"β={coef:>7.4f} (p={pval:.4f}) {sig}")

# Анализ результатов
print("\n" + "="*100)
print("[2.1] АНАЛИЗ РЕЗУЛЬТАТОВ ОТБОРА ЛАГОВ")
print("="*100)

if not lag_results:
    print("Нет результатов")
    exit(1)

lag_df = pd.DataFrame(lag_results)

# Сохраняем все результаты
lag_df.to_csv(f"{OUTPUT_FOLDER}/Lag_All_Tests.csv", sep=";", index=False)

print(f"\nПротестировано: {len(lag_df)} комбинаций (переменная × лаг)")

# Для каждой переменной: лучший лаг
print(f"\n{'='*100}")
print("ОПТИМАЛЬНЫЕ ЛАГИ ДЛЯ КАЖДОЙ ПЕРЕМЕННОЙ")
print(f"{'='*100}")

print(f"\n{'Переменная':<25s} {'Лучший лаг':<12s} {'ΔR²':<10s} {'β':<10s} {'p-value':<10s} {'Использ.'}")
print("="*100)

optimal_lags = []
optimal_vars = []

for var in lag_df['variable'].unique():
    var_results = lag_df[lag_df['variable'] == var]
    
    # Критерии отбора:
    # 1. Значимость (p<0.05)
    # 2. Максимальный прирост R²
    
    significant = var_results[var_results['significant'] == True]
    
    if len(significant) > 0:
        # Выбираем лаг с максимальным приростом R²
        best = significant.loc[significant['delta_r2'].idxmax()]
        use_lag = True
    else:
        # Нет значимых лагов
        best = var_results.loc[var_results['delta_r2'].idxmax()]
        use_lag = False
    
    optimal_lags.append({
        'variable': var,
        'optimal_lag': int(best['lag']),
        'lag_var': best['lag_var'],
        'delta_r2': best['delta_r2'],
        'coefficient': best['coefficient'],
        'pvalue': best['pvalue'],
        'use_lag': use_lag
    })
    
    marker = "ДА" if use_lag else "НЕТ"
    print(f"{var:<25s} lag{int(best['lag']):<11d} {best['delta_r2']:+.4f}     "
          f"{best['coefficient']:>7.4f}   {best['pvalue']:.4f}     {marker}")
    
    # Собираем optimal_vars
    if use_lag:
        optimal_vars.append(best['lag_var'])
    else:
        optimal_vars.append(var)

optimal_df = pd.DataFrame(optimal_lags)
optimal_df.to_csv(f"{OUTPUT_FOLDER}/Lag_Optimal_Selection.csv", sep=";", index=False)

print(f"\n✓ Переменных с оптимальными лагами: {len(optimal_vars)}")

# ============================================================================
# 3. ТЕСТИРОВАНИЕ КВАДРАТИЧНЫХ ТЕРМОВ
# ============================================================================

print("\n" + "="*100)
print("[3] ТЕСТИРОВАНИЕ НЕЛИНЕЙНОСТЕЙ (КВАДРАТИЧНЫЕ ТЕРМЫ)")
print("="*100)

# Агрегация для тестирования
regional_vars = [v for v in optimal_vars if '_lag' not in v or 'rate' not in v and 'inflation' not in v and 'sentiment' not in v and 'expectations' not in v and 'current_state' not in v]
common_vars = [v for v in optimal_vars if '_lag' in v and ('rate' in v or 'inflation' in v or 'sentiment' in v or 'expectations' in v or 'current_state' in v)]

df_agg = df.groupby('date')[regional_vars].mean().reset_index()
for v in common_vars:
    df_agg[v] = df.groupby('date')[v].first().values

# Создаем квадратичные термы для ВСЕХ переменных
print(f"\nСоздание квадратичных термов для {len(optimal_vars)} переменных...")

for var in optimal_vars:
    df_agg[f'{var}_sq'] = df_agg[var] ** 2

# Функция оценки модели
def estimate_with_quadratic(vars_list, df_data):
    """Оценивает панельную модель через агрегированные данные"""
    
    # Расширяем до панельной структуры
    df_panel = df[['date','region','price']].copy()
    
    for feat in vars_list:
        df_panel = df_panel.merge(
            df_data[['date', feat]],
            on='date',
            how='left'
        )
    
    df_panel = df_panel.dropna()
    
    if len(df_panel) < 100:
        return None
    
    panel_data = df_panel.set_index(['region','date'])
    
    # Нормализация
    for col in ['price'] + vars_list:
        panel_data[f'{col}_n'] = (panel_data[col] - panel_data[col].mean()) / panel_data[col].std()
    
    X_norm = [f'{v}_n' for v in vars_list]
    formula = f"price_n ~ {' + '.join(X_norm)} + EntityEffects"
    
    try:
        model = PanelOLS.from_formula(formula, data=panel_data)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        return results
    except:
        return None

# Базовая модель без квадратов
print(f"\n   Базовая модель (без квадратов)...")
baseline = estimate_with_quadratic(optimal_vars, df_agg)

if baseline:
    baseline_r2 = baseline.rsquared_within
    print(f"      R² within = {baseline_r2:.4f}")
else:
    print(f"      Не удалось оценить")
    exit(1)

# Тестируем квадратичные термы для каждой переменной
quadratic_results = []

print(f"\n   Тестирование квадратичных термов...")

for var in optimal_vars:
    test_vars = optimal_vars + [f'{var}_sq']
    
    result = estimate_with_quadratic(test_vars, df_agg)
    
    if result:
        r2 = result.rsquared_within
        delta_r2 = r2 - baseline_r2
        
        # Коэффициент при квадратичном терме
        sq_var_n = f'{var}_sq_n'
        
        if sq_var_n in result.params.index:
            coef_sq = result.params[sq_var_n]
            pval_sq = result.pvalues[sq_var_n]
            
            # Также смотрим на линейный коэффициент
            var_n = f'{var}_n'
            coef_lin = result.params[var_n] if var_n in result.params.index else 0
            
            quadratic_results.append({
                'variable': var,
                'r2_with_quad': r2,
                'delta_r2': delta_r2,
                'coef_linear': coef_lin,
                'coef_quadratic': coef_sq,
                'pvalue_quad': pval_sq,
                'significant': pval_sq < 0.05,
                'form': 'U-форма' if coef_sq > 0 else '∩-форма'
            })
            
            sig = '***' if pval_sq<0.001 else '**' if pval_sq<0.01 else '*' if pval_sq<0.05 else ''
            
            print(f"      {var:<30s} ΔR²={delta_r2:+.4f}, β²={coef_sq:>7.4f} (p={pval_sq:.4f}) {sig}")

quadratic_df = pd.DataFrame(quadratic_results)

# Отбираем значимые квадратичные термы
significant_quads = quadratic_df[quadratic_df['significant'] == True].sort_values('delta_r2', ascending=False)

print(f"\n Значимых квадратичных термов (p<0.05): {len(significant_quads)}")

if len(significant_quads) > 0:
    print(f"\n   ТОП значимых нелинейностей:")
    for idx, row in significant_quads.head(10).iterrows():
        print(f"      {row['variable']:<30s} {row['form']:<10s} ΔR²={row['delta_r2']:+.4f} β²={row['coef_quadratic']:>7.4f}")

# Сохраняем результаты
quadratic_df.to_csv(f"{OUTPUT_FOLDER}/Quadratic_Terms_Tests.csv", sep=";", index=False)

# Собираем финальный список переменных (линейные + значимые квадраты)
final_vars_for_pca = optimal_vars.copy()

for idx, row in significant_quads.iterrows():
    final_vars_for_pca.append(f"{row['variable']}_sq")
    df_agg[f"{row['variable']}_sq"] = df_agg[row['variable']] ** 2

print(f"\n Финальных переменных для PCA: {len(final_vars_for_pca)}")
print(f"   • Линейных: {len(optimal_vars)}")
print(f"   • Квадратичных: {len(significant_quads)}")

# ============================================================================
# 4. PCA ДЛЯ ФИНАЛЬНОЙ МОДЕЛИ
# ============================================================================

print("\n" + "="*100)
print("[4] PCA ДЛЯ УСТРАНЕНИЯ МУЛЬТИКОЛЛИНЕАРНОСТИ")
print("="*100)

THRESHOLD = 0.8
corr_matrix = df_agg[final_vars_for_pca].corr()

# Граф связей
graph = defaultdict(set)
n_pairs = 0

for i in range(len(final_vars_for_pca)):
    for j in range(i+1, len(final_vars_for_pca)):
        c = abs(corr_matrix.iloc[i,j])
        if c > THRESHOLD:
            graph[final_vars_for_pca[i]].add(final_vars_for_pca[j])
            graph[final_vars_for_pca[j]].add(final_vars_for_pca[i])
            n_pairs += 1

print(f" Пар с корреляцией >{THRESHOLD}: {n_pairs}")

# Группировка
visited = set()
groups = []

def dfs(node, group):
    visited.add(node)
    group.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, group)

for var in final_vars_for_pca:
    if var not in visited:
        if var in graph:
            group = set()
            dfs(var, group)
            if len(group) > 1:
                groups.append(list(group))
        visited.add(var)

independent = [v for v in final_vars_for_pca if v not in graph]

print(f" Групп: {len(groups)}")
for i, group in enumerate(groups, 1):
    print(f"   Группа {i} ({len(group)})")

print(f" Независимых: {len(independent)}")

# PCA для каждой группы
print(f"\nPCA...")

pca_features = []
pca_info = []
feature_importance = []

for i, group in enumerate(groups, 1):
    X = df_agg[group].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    n_comp = pca.n_components_
    explained_var = pca.explained_variance_ratio_
    
    print(f"   Группа {i}: {len(group)} → {n_comp} компонент ({explained_var.sum()*100:.1f}%)")
    
    for j in range(n_comp):
        col_name = f'PC_FINAL_{i}_{j+1}'
        df_agg[col_name] = X_pca[:, j]
        pca_features.append(col_name)
    
    loadings = pd.DataFrame(
        pca.components_[:n_comp],
        columns=group,
        index=[f'PC_FINAL_{i}_{j+1}' for j in range(n_comp)]
    )
    
    pca_info.append({
        'group': i,
        'variables': group,
        'n_components': n_comp,
        'explained_variance': explained_var,
        'loadings': loadings
    })
    
    # Важность
    for var in group:
        importance = sum((loadings.loc[f'PC_FINAL_{i}_{j+1}', var]**2) * explained_var[j] 
                        for j in range(n_comp))
        feature_importance.append({
            'variable': var,
            'group': i,
            'importance': importance,
            'is_quadratic': '_sq' in var
        })

final_features = independent + pca_features

print(f"\n✓ {len(final_vars_for_pca)} переменных → {len(final_features)} признаков")

# ============================================================================
# 5. ФИНАЛЬНАЯ МОДЕЛЬ
# ============================================================================

print("\n" + "="*100)
print("[5] ФИНАЛЬНАЯ ПАНЕЛЬНАЯ РЕГРЕССИЯ")
print("="*100)

df_panel = df[['date','region','price']].copy()

for feat in final_features:
    df_panel = df_panel.merge(
        df_agg[['date', feat]],
        on='date',
        how='left'
    )

df_panel = df_panel.dropna()
panel_data = df_panel.set_index(['region','date'])

# Нормализация
for col in ['price'] + final_features:
    panel_data[f'{col}_n'] = (panel_data[col] - panel_data[col].mean()) / panel_data[col].std()

X_norm = [f'{v}_n' for v in final_features]
formula = f"price_n ~ {' + '.join(X_norm)} + EntityEffects"

try:
    model = PanelOLS.from_formula(formula, data=panel_data)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    print(f"\n   РЕЗУЛЬТАТЫ:")
    print(f"   R² within:  {results.rsquared_within:.4f}")
    print(f"   R² overall: {results.rsquared_overall:.4f}")
    print(f"   N obs:      {results.nobs}")
    
    # Улучшение относительно baseline
    improvement = results.rsquared_within - baseline_r2
    
    print(f"\n   УЛУЧШЕНИЕ:")
    print(f"   Baseline (лаги без квадратов): R²={baseline_r2:.4f}")
    print(f"   Final (лаги + квадраты + PCA): R²={results.rsquared_within:.4f}")
    print(f"   Прирост: ΔR²={improvement:+.4f} ({improvement*100:+.2f}%)")
    
    pc_coefs = pd.DataFrame({
        'feature': results.params.index,
        'coefficient': results.params.values,
        'std_error': results.std_errors.values,
        'pvalue': results.pvalues.values
    })
    
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    results = None

# ============================================================================
# 6. ТАБЛИЦА ВАЖНОСТИ
# ============================================================================

print("\n" + "="*100)
print("[6] ТАБЛИЦА ВАЖНОСТИ ПРИЗНАКОВ")
print("="*100)

if results is not None and len(feature_importance) > 0:
    
    importance_df = pd.DataFrame(feature_importance)
    
    # Вычисляем общий эффект
    for idx, row in importance_df.iterrows():
        group = row['group']
        total_effect = 0
        
        for j in range(pca_info[group-1]['n_components']):
            pc_name = f'PC_FINAL_{group}_{j+1}_n'
            
            if pc_name in pc_coefs['feature'].values:
                pc_coef = pc_coefs[pc_coefs['feature']==pc_name]['coefficient'].values[0]
                loading = pca_info[group-1]['loadings'].loc[f'PC_FINAL_{group}_{j+1}', row['variable']]
                total_effect += pc_coef * loading
        
        importance_df.loc[idx, 'total_effect'] = total_effect
    
    # Независимые
    for var in independent:
        var_norm = f'{var}_n'
        if var_norm in pc_coefs['feature'].values:
            coef = pc_coefs[pc_coefs['feature']==var_norm]['coefficient'].values[0]
            
            importance_df = pd.concat([importance_df, pd.DataFrame([{
                'variable': var,
                'group': 0,
                'importance': 1.0,
                'total_effect': coef,
                'is_quadratic': '_sq' in var
            }])], ignore_index=True)
    
    importance_df['abs_effect'] = importance_df['total_effect'].abs()
    importance_df = importance_df.sort_values('abs_effect', ascending=False)
    
    print(f"\n{'Переменная':<40s} {'Тип':<12s} {'Важность':<12s} {'Эффект':<12s}")
    print("-"*100)
    
    for idx, row in importance_df.head(20).iterrows():
        var_type = "Квадратичная" if row['is_quadratic'] else "Линейная"
        print(f"{row['variable']:<40s} {var_type:<12s} {row['importance']:>10.4f}   {row['total_effect']:>10.4f}")
    
    importance_df.to_csv(f"{OUTPUT_FOLDER}/Feature_Importance_FINAL.csv", sep=";", index=False)
    
    print(f"\n✓ Feature_Importance_FINAL.csv")

# ============================================================================
# 7. СОХРАНЕНИЕ
# ============================================================================

print("\n[7] Сохранение...")

if results is not None:
    with open(f"{OUTPUT_FOLDER}/Final_Model_with_Nonlinear_PCA_summary.txt", 'w', encoding='utf-8') as f:
        f.write("ФИНАЛЬНАЯ МОДЕЛЬ: ЛАГИ + НЕЛИНЕЙНОСТИ + PCA\n")
        f.write("="*80 + "\n\n")
        f.write(f"Baseline R² (лаги): {baseline_r2:.4f}\n")
        f.write(f"Final R² (лаги + квадраты + PCA): {results.rsquared_within:.4f}\n")
        f.write(f"Improvement: {improvement:+.4f}\n\n")
        f.write(f"Линейных переменных: {len(optimal_vars)}\n")
        f.write(f"Квадратичных термов: {len(significant_quads)}\n")
        f.write(f"Главных компонент: {len(pca_features)}\n")
        f.write(f"Независимых: {len(independent)}\n\n")
        f.write("="*80 + "\n\n")
        f.write(str(results.summary))
    
    pc_coefs.to_csv(f"{OUTPUT_FOLDER}/Final_Model_PCA_coefs.csv", sep=";", index=False)

for info in pca_info:
    info['loadings'].to_csv(
        f"{OUTPUT_FOLDER}/PCA_FINAL_group{info['group']}_loadings.csv",
        sep=";"
    )

print(f"Quadratic_Terms_Tests.csv")
print(f"Feature_Importance_FINAL.csv")
print(f"Final_Model_with_Nonlinear_PCA_summary.txt")
print(f"Final_Model_PCA_coefs.csv")
print(f"PCA_FINAL_group*_loadings.csv")

print("\n" + "="*100)
print("МОДУЛЬ 4 (ПОЛНЫЙ) ЗАВЕРШЕН")
print("="*100)