#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МОДУЛЬ 0 (ПОЛНЫЙ): ПОДГОТОВКА ВСЕХ ПЕРЕМЕННЫХ + RATIO
Включает ВСЕ факторы для анализа:
- Региональные: цены, кредиты, задолженности, предложения, жилье
- Общие: ставка, инфляция, индексы
- RATIO: доля просрочки, доступность и т.д.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os

INPUT_FILE = r"G:\downloads\housingdata_combined.csv"
OUTPUT_FOLDER = r"G:\downloads\price_factors_results_FULL"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MAX_LAG = 6

print("="*100)
print("МОДУЛЬ 0 (ПОЛНЫЙ): ПОДГОТОВКА ВСЕХ ПЕРЕМЕННЫХ")
print("="*100)

# ============================================================================
# 1. ЗАГРУЗКА И ОЧИСТКА
# ============================================================================

print("\n[1] Загрузка...")
df = pd.read_csv(INPUT_FILE, sep=";", encoding='utf-8-sig', parse_dates=['date'])
print(f"✓ Загружено: {len(df)} × {len(df.columns)}")

print("\n[2] Очистка чисел...")

def clean_num(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        cleaned = val.replace(' ', '').replace('\xa0', '').replace(',', '').strip()
        if cleaned in ['', '-']:
            return np.nan
        try:
            return float(cleaned)
        except:
            return val
    return val

for col in [c for c in df.columns if c != 'date']:
    df[col] = df[col].apply(clean_num)
    df[col] = pd.to_numeric(df[col], errors='ignore')

print(f"✓ Очищено")

# ============================================================================
# 2. ИДЕНТИФИКАЦИЯ РЕГИОНОВ
# ============================================================================

print("\n[3] Поиск регионов...")

price_cols = [c for c in df.columns if c.startswith('real_estate_deals_primary_market-')]
regions_all = [c.replace('real_estate_deals_primary_market-', '') for c in price_cols]

print(f"Найдено регионов: {len(regions_all)}")

# Выбираем регионы с ≤10% пропусков (смягчаем требование)
regions_ok = []

for region in regions_all:
    price_col = f'real_estate_deals_primary_market-{region}'
    
    # Проверяем только цену
    n_missing = df[price_col].isna().sum()
    pct = (n_missing / len(df)) * 100
    
    if pct <= 10.0:
        regions_ok.append(region)

print(f"✓ Регионов с данными: {len(regions_ok)}")

# ============================================================================
# 3. СОЗДАНИЕ ПАНЕЛИ СО ВСЕМИ ПЕРЕМЕННЫМИ
# ============================================================================

print("\n[4] Создание панели со ВСЕМИ переменными...")

panel_data = []

for region in regions_ok:
    for idx, row in df.iterrows():
        record = {
            'date': row['date'],
            'region': region,
            'price': row[f'real_estate_deals_primary_market-{region}']  # Целевая
        }
        
        # РЕГИОНАЛЬНЫЕ ПЕРЕМЕННЫЕ
        regional_vars = {
            # Цены
            'price_secondary': f'real_estate_deals_secondary_market-{region}',
            # Предложения
            'offers_primary': f'predlozheniya-novostroek-{region}',
            'offers_secondary': f'predlozheniya-vtorichnoi-nedvizhimosti-{region}',
            # Кредиты и задолженности
            'housing_loans': f'housing_loans_{region}',
            'mortgage_debt': f'mortgage_debt_{region}',
            'mortgage_overdue': f'mortgage_overdue_{region}',
            # Жилье
            'housing_completed': f'housing_completed_{region}'
        }
        
        for var_name, col_name in regional_vars.items():
            if col_name in df.columns:
                record[var_name] = row[col_name]
            else:
                record[var_name] = np.nan
        
        # ОБЩИЕ ПЕРЕМЕННЫЕ
        common_vars = {
            'rate': 'Ключевая ставка, %',
            'inflation': 'Базовая инфляция по трем месяцам, %',
            'sentiment': 'индекс потребительских настроений (в пунктах)',
            'expectations': 'индекс ожиданий  (в пунктах)',
            'current_state': 'индекс текущего состояния  (в пунктах)'
        }
        
        for var_name, col_name in common_vars.items():
            if col_name in df.columns:
                record[var_name] = row[col_name]
            else:
                record[var_name] = np.nan
        
        panel_data.append(record)

df_panel = pd.DataFrame(panel_data)

print(f"Создано: {len(df_panel)} наблюдений")
print(f"Базовых переменных: {len(df_panel.columns) - 3}")  # -3 для date, region, price

# ============================================================================
# 4. СОЗДАНИЕ RATIO ПЕРЕМЕННЫХ
# ============================================================================

print("\n[5] Создание RATIO переменных...")

# КРИТИЧНО: Принудительно конвертируем все в numeric
print(f"   Проверка типов данных...")

numeric_cols = [c for c in df_panel.columns if c not in ['date', 'region']]

for col in numeric_cols:
    if df_panel[col].dtype == 'object':
        print(f"{col} - строковый тип, конвертируем...")
        df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce')

print(f"Все колонки числовые")

# Реальная ставка
df_panel['real_rate'] = df_panel['rate'] - df_panel['inflation']
print(f"real_rate = rate - inflation")

# Доля просрочки
df_panel['overdue_ratio'] = (df_panel['mortgage_overdue'] / df_panel['mortgage_debt']) * 100
print(f"overdue_ratio = (просрочка / задолженность) × 100")

# Кредитная доступность
df_panel['credit_availability'] = df_panel['housing_loans'] / df_panel['price']
print(f"credit_availability = кредиты / цена")

# Интенсивность предложения
df_panel['supply_intensity'] = df_panel['offers_primary'] / df_panel['housing_completed']
print(f" supply_intensity = предложения / введено")

# Доля первичного рынка
total_offers = df_panel['offers_primary'] + df_panel['offers_secondary']
df_panel['primary_share'] = (df_panel['offers_primary'] / total_offers) * 100
print(f"primary_share = предложения_первички / всего_предложений")

# Относительная цена (первичка vs вторичка)
df_panel['price_ratio'] = df_panel['price'] / df_panel['price_secondary']
print(f"price_ratio = цена_первички / цена_вторички")

print(f"\nСоздано {6} RATIO переменных")
print(f"Итого переменных: {len(df_panel.columns) - 3}")

# ============================================================================
# 5. ЗАПОЛНЕНИЕ ПРОПУСКОВ
# ============================================================================

print("\n[6] Заполнение пропусков...")

df_panel = df_panel.sort_values(['region', 'date'])

for col in df_panel.columns:
    if col not in ['date', 'region']:
        df_panel[col] = df_panel.groupby('region')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

n_before = len(df_panel)
df_panel = df_panel.dropna()
n_after = len(df_panel)

print(f"Удалено строк с NaN: {n_before - n_after}")
print(f"Итого: {n_after} наблюдений")

# ============================================================================
# 6. СОЗДАНИЕ ЛАГОВ
# ============================================================================

print(f"\n[7] Создание лагов (до {MAX_LAG})...")

df_panel = df_panel.sort_values(['region', 'date']).reset_index(drop=True)

# Все предикторы (кроме date, region, price)
predictors = [c for c in df_panel.columns if c not in ['date', 'region', 'price']]

print(f"   Предикторов (без лагов): {len(predictors)}")

for var in predictors:
    for lag in range(1, MAX_LAG + 1):
        df_panel[f'{var}_lag{lag}'] = df_panel.groupby('region')[var].shift(lag)

n_before = len(df_panel)
df_panel = df_panel.dropna()
n_after = len(df_panel)

print(f"Создано лагов: {MAX_LAG} × {len(predictors)} = {MAX_LAG * len(predictors)}")
print(f"Удалено NaN от лагов: {n_before - n_after}")
print(f"Итого: {n_after} наблюдений")
print(f"Всего столбцов: {len(df_panel.columns)}")

# ============================================================================
# 7. НОРМАЛИЗАЦИЯ
# ============================================================================

print("\n[8] Нормализация данных по регионам...")

from sklearn.preprocessing import RobustScaler

# Нормализуем внутри каждого региона (устойчива к выбросам)
numeric_cols = [c for c in df_panel.columns if c not in ['date', 'region']]

df_normalized_list = []

for region in df_panel['region'].unique():
    region_data = df_panel[df_panel['region'] == region].copy()
    
    scaler = RobustScaler()
    region_data[numeric_cols] = scaler.fit_transform(region_data[numeric_cols])
    
    df_normalized_list.append(region_data)

df_panel_norm = pd.concat(df_normalized_list, ignore_index=True)

print(f"✓ Нормализовано: {len(df_panel_norm)} наблюдений")
print(f"   Метод: RobustScaler (устойчив к выбросам)")

# ============================================================================
# 8. СОХРАНЕНИЕ
# ============================================================================

print("\n[9] Сохранение...")

# Оригинальные данные
df_panel.to_csv(f"{OUTPUT_FOLDER}/panel_FULL_with_lags.csv", sep=";", index=False)
print(f"panel_FULL_with_lags.csv (оригинал)")

# Нормализованные данные
df_panel_norm.to_csv(f"{OUTPUT_FOLDER}/panel_FULL_normalized.csv", sep=";", index=False)
print(f"panel_FULL_normalized.csv (нормализованные)")

# Список всех переменных
all_vars = [c for c in df_panel.columns if c not in ['date', 'region', 'price']]
base_vars = [c for c in all_vars if '_lag' not in c]
lag_vars = [c for c in all_vars if '_lag' in c]

print(f"\nИТОГОВАЯ СТРУКТУРА:")
print(f"   • Регионов: {df_panel['region'].nunique()}")
print(f"   • Наблюдений: {len(df_panel)}")
print(f"   • Период: {df_panel['date'].min().strftime('%Y-%m')} — {df_panel['date'].max().strftime('%Y-%m')}")
print(f"   • Базовых переменных: {len(base_vars)}")
print(f"   • С лагами: {len(lag_vars)}")
print(f"   • Всего предикторов: {len(all_vars)}")

print(f"\nБазовые переменные:")
for var in base_vars:
    print(f"   • {var}")

print(f"\nМОДУЛЬ 1 (ПОЛНЫЙ) ЗАВЕРШЕН")
print(f"Результат: {OUTPUT_FOLDER}/panel_FULL_with_lags.csv")
print("="*100 + "\n")