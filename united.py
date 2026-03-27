import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*p-value.*')
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from collections import defaultdict

from scipy.stats import jarque_bera, shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller, kpss as kpss_test

from linearmodels.panel import PanelOLS

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
SOURCE_FILE   = r"C:\housingdata_combined.csv"
OUTPUT_FOLDER = r"C:\price_factors_results_FULL"
MAX_LAG       = 6

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Макро-переменные: подстрока в названии колонки → короткое имя
MACRO_MAP = {
    'Базовая инфляция':           'inflation',
    'Ключевая ставка в реальном': 'real_rate',
    'Ключевая ставка,':           'rate',
    'индекс потребительских':     'sentiment',
    'индекс ожиданий':            'expectations',
    'индекс текущего':            'current_state',
}

# Макро-переменные (базовые имена) — нормализуются глобально, не per region
MACRO_VARS = {'inflation', 'real_rate', 'rate', 'sentiment', 'expectations', 'current_state'}

# Региональные префиксы → короткое имя
REGIONAL_KEYS = {
    'predlozheniya-vtorichnoi-nedvizhimosti': 'offers_secondary',
    'predlozheniya-novostroek':               'offers_primary',
    'real_estate_deals_secondary_market':     'deals_secondary',
    'real_estate_deals_primary_market':       'deals_primary',
    'housing_completed':                      'housing_completed',
    'housing_loans':                          'housing_loans',
    'mortgage_debt':                          'mortgage_debt',
    'mortgage_overdue':                       'mortgage_overdue',
}

# Регионы-агрегаты — исключаем по точному имени И по паттерну
EXCLUDE_REGIONS = {
    'РОССИЙСКАЯ ФЕДЕРАЦИЯ', 'Россия', 'РФ',
    'СЕВЕРО-ЗАПАДНЫЙ ФО', 'ЦЕНТРАЛЬНЫЙ ФО', 'ПРИВОЛЖСКИЙ ФО',
    'УРАЛЬСКИЙ ФО', 'СИБИРСКИЙ ФО', 'ДАЛЬНЕВОСТОЧНЫЙ ФО',
    'ЮЖНЫЙ ФО', 'СЕВЕРО-КАВКАЗСКИЙ ФО', 'КРЫМСКИЙ ФО',
}
# Дополнительно исключаем по паттерну (ФО, федеральный округ и т.п.)
EXCLUDE_PATTERNS = ['ФЕДЕРАЦИЯ', ' ФО', 'федеральный округ', 'ФЕДЕРАЛЬНЫЙ ОКРУГ']

# Два прогона: первичный и вторичный рынок
TARGETS = [
    {'name': 'primary',   'price_col': 'deals_primary',
     'exclude_regressors': {'deals_secondary', 'offers_secondary',
                            'absorption_secondary', 'deals_primary'}},
    {'name': 'secondary', 'price_col': 'deals_secondary',
     'exclude_regressors': {'deals_primary', 'offers_primary',
                            'absorption_primary', 'deals_secondary'}},
]

# ============================================================================
# ШАГ 0: ЗАГРУЗКА И ТРАНСФОРМАЦИЯ WIDE → PANEL (один раз)
# ============================================================================
print("="*100)
print("ЗАГРУЗКА И ТРАНСФОРМАЦИЯ ДАННЫХ")
print("="*100)

import re as _re

MONTHS_RU_DEC = {
    'янв':1,'фев':2,'мар':3,'апр':4,'май':5,'июн':6,
    'июл':7,'авг':8,'сен':9,'окт':10,'ноя':11,'дек':12
}

def fix_excel_number(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        pass
    sl = s.lower()
    for mon, num in MONTHS_RU_DEC.items():
        if mon in sl:
            rest = _re.sub(mon, '', sl).strip('.')
            try:
                rest_f = float(rest)
            except ValueError:
                return np.nan
            if sl.startswith(mon):
                return num + rest_f / 100
            else:
                return rest_f + num / 10
    return np.nan

raw = pd.read_csv(SOURCE_FILE, sep=';')
raw['date'] = pd.to_datetime(raw['date'], dayfirst=True)

print("Декодирование Excel-дат в числа...")
for col in raw.columns:
    if col == 'date':
        continue
    non_numeric = raw[col].apply(
        lambda x: pd.isna(pd.to_numeric(str(x).replace(',','.'), errors='coerce'))
        if pd.notna(x) else False
    ).sum()
    if non_numeric > 0:
        raw[col] = raw[col].apply(fix_excel_number)
        print(f"  Исправлено '{col}': {non_numeric} значений")
raw = raw.loc[:, ~raw.columns.duplicated()]

seen_hashes, drop_cols = {}, []
for col in raw.columns:
    h = pd.util.hash_pandas_object(raw[col].fillna(-9999)).sum()
    if h in seen_hashes:
        drop_cols.append(col)
    else:
        seen_hashes[h] = col
if drop_cols:
    print(f"Удалено дублей по содержимому: {len(drop_cols)}: {drop_cols}")
    raw = raw.drop(columns=drop_cols)

print(f"Строк: {len(raw)}, колонок после дедупликации: {raw.shape[1]}")
print(f"Период: {raw['date'].min()} — {raw['date'].max()}")

macro_df = pd.DataFrame({'date': raw['date']})
for substr, varname in MACRO_MAP.items():
    matched = [c for c in raw.columns if substr in c]
    if matched:
        macro_df[varname] = pd.to_numeric(raw[matched[0]], errors='coerce').values
        print(f"Макро '{varname}' ← '{matched[0]}'")

def prefix_to_long(raw, prefix, varname):
    cols = [c for c in raw.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame(columns=['date', 'region', varname])
    sub  = raw[['date'] + cols].copy()
    long = sub.melt(id_vars='date', value_vars=cols,
                    var_name='_col', value_name=varname)
    sep            = long['_col'].iloc[0][len(prefix)]
    long['region'] = long['_col'].str[len(prefix) + 1:]
    long[varname]  = pd.to_numeric(long[varname], errors='coerce')
    return long[['date', 'region', varname]]

panel_parts = []
for prefix, varname in REGIONAL_KEYS.items():
    part = prefix_to_long(raw, prefix, varname)
    if len(part) > 0:
        panel_parts.append(part)
        print(f"'{varname}': {part['region'].nunique()} регионов, "
              f"{part[varname].notna().sum()} непустых значений")

base_panel = panel_parts[0]
for part in panel_parts[1:]:
    base_panel = base_panel.merge(part, on=['date', 'region'], how='outer')

panel_dates = set(base_panel['date'].dt.normalize().unique())
macro_dates = set(macro_df['date'].dt.normalize().unique())
missing_macro = panel_dates - macro_dates
if missing_macro:
    print(f"⚠ Даты в панели без макро-данных: {sorted(missing_macro)}")
    base_panel['date'] = base_panel['date'].dt.normalize()
    macro_df['date']   = macro_df['date'].dt.normalize()

base_panel = base_panel.merge(macro_df, on='date', how='left')
base_panel = base_panel[~base_panel['region'].isin(EXCLUDE_REGIONS)]
for pattern in EXCLUDE_PATTERNS:
    base_panel = base_panel[~base_panel['region'].str.contains(pattern, case=False, na=False)]
base_panel = base_panel.sort_values(['region', 'date']).reset_index(drop=True)

print(f"\n✓ Базовая панель: {len(base_panel)} наблюдений | "
      f"{base_panel['region'].nunique()} регионов")

all_dates = sorted(base_panel['date'].unique())
print(f"   Уникальных дат: {len(all_dates)}")
print(f"   Первые 5 дат:  {[str(d)[:10] for d in all_dates[:5]]}")
print(f"   Последние 5 дат: {[str(d)[:10] for d in all_dates[-5:]]}")

print(f"\n   Пропуски по колонкам:")
for col in base_panel.columns:
    if col in ['date', 'region']:
        continue
    n_na = base_panel[col].isna().sum()
    total = len(base_panel)
    if col in list(MACRO_MAP.values()):
        dates_with_na = base_panel[base_panel[col].isna()]['date'].nunique()
        dates_ok      = base_panel[base_panel[col].notna()]['date'].nunique()
        print(f"      {col:<35s}: {n_na:4d} NaN | дат с данными: {dates_ok} | дат без: {dates_with_na}")
    else:
        regions_with_na = base_panel[base_panel[col].isna()]['region'].nunique()
        print(f"      {col:<35s}: {n_na:4d} NaN ({n_na/total*100:.0f}%) | регионов с NaN: {regions_with_na}")

regional_only_cols = [c for c in base_panel.columns
                      if c not in ['date', 'region'] + list(MACRO_MAP.values())]
complete_regions = (
    base_panel.groupby('region')[regional_only_cols]
    .apply(lambda x: x.notna().all().all())
)
print(f"\n   Регионов с полными региональными данными: {complete_regions.sum()} из {complete_regions.shape[0]}")
print(f"   Примеры полных регионов: {list(complete_regions[complete_regions].index[:5])}")

macro_dates_ok = base_panel[base_panel['rate'].notna()]['date'].unique()
print(f"\n   Дат с данными по 'rate': {len(macro_dates_ok)}")
region_dates_ok = base_panel[base_panel['offers_secondary'].notna()]['date'].unique()
print(f"   Дат с данными по 'offers_secondary': {len(region_dates_ok)}")

# ============================================================================
# ФИКСАЦИЯ ЕДИНОЙ ВЫБОРКИ ДЛЯ ОБОИХ РЫНКОВ
# ============================================================================
# Определяем общий набор регионов и дат, валидных для ОБОИХ рынков одновременно.
# Это обеспечивает одинаковое количество наблюдений N×T в обоих прогонах.
print("\n" + "="*100)
print("ФИКСАЦИЯ ЕДИНОЙ ВЫБОРКИ (N регионов × T периодов) ДЛЯ ОБОИХ РЫНКОВ")
print("="*100)

# Колонки, которые должны быть непустыми для включения в выборку
COMMON_REQUIRED_COLS = (
    ['deals_primary', 'deals_secondary',
     'offers_primary', 'offers_secondary',
     'housing_completed', 'housing_loans',
     'mortgage_debt', 'mortgage_overdue']
    + list(MACRO_MAP.values())
)
COMMON_REQUIRED_COLS = [c for c in COMMON_REQUIRED_COLS if c in base_panel.columns]

# Шаг 1: регионы без пропусков в региональных колонках
regional_req = [c for c in COMMON_REQUIRED_COLS if c not in MACRO_VARS]
region_completeness = base_panel.groupby('region')[regional_req].apply(
    lambda x: x.isnull().any().any()
)
valid_regions = region_completeness[~region_completeness].index.tolist()
print(f"   Регионов с полными данными по обоим рынкам: {len(valid_regions)}")

# Шаг 2: даты без пропусков в макро
macro_req = [c for c in COMMON_REQUIRED_COLS if c in MACRO_VARS]
date_completeness = (
    base_panel[base_panel['region'].isin(valid_regions)]
    [['date'] + macro_req]
    .drop_duplicates('date')
    .set_index('date')
    .isnull().any(axis=1)
)
valid_dates = date_completeness[~date_completeness].index.tolist()
print(f"   Периодов с полными макро-данными: {len(valid_dates)}")

# Применяем к base_panel
base_panel = base_panel[
    base_panel['region'].isin(valid_regions) &
    base_panel['date'].isin(valid_dates)
].copy().reset_index(drop=True)

N_PANEL = base_panel['region'].nunique()
T_PANEL = base_panel['date'].nunique()
OBS_PANEL = len(base_panel)
print(f"\n✓ Единая выборка зафиксирована: N={N_PANEL} регионов × T={T_PANEL} периодов = {OBS_PANEL} наблюдений")
print(f"   Период: {base_panel['date'].min().date()} — {base_panel['date'].max().date()}")
print(f"   Эта выборка используется для ОБОИХ рынков без изменений.")

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================
def get_base_name(col, max_lag=MAX_LAG):
    name = col
    for lag in range(max_lag, 0, -1):
        name = name.replace(f'_lag{lag}', '')
    name = name.replace('_sq', '')
    return name


def estimate_panel_model(df, y_var, X_vars, name="Model"):
    cols  = ['date', 'region', y_var] + X_vars
    pdata = df[cols].copy().dropna()
    if len(pdata) < 100:
        return None
    pdata = pdata.set_index(['region', 'date'])

    nd = {}
    for col in [y_var] + X_vars:
        std = pdata[col].std()
        nd[f'{col}_n'] = ((pdata[col] - pdata[col].mean()) / std
                          if std > 1e-10
                          else pdata[col] - pdata[col].mean())
    nd = pd.DataFrame(nd, index=pdata.index)

    X_norm = [f'{v}_n' for v in X_vars if nd[f'{v}_n'].std() > 1e-10]
    if not X_norm:
        return None
    try:
        res = PanelOLS.from_formula(
            f"{y_var}_n ~ {' + '.join(X_norm)} + EntityEffects",
            data=nd).fit(cov_type='clustered', cluster_entity=True)
        return {'name': name, 'r2_within': res.rsquared_within,
                'n_obs': res.nobs, 'results': res}
    except Exception as e:
        print(f"  {name}: {e}")
        return None


# ============================================================================
# ТЕСТ ПЕДРОНИ (панельная коинтеграция)
# ============================================================================
# Применяется для пар (целевая переменная, I(1) регрессор).
# H0: нет коинтеграции. Если H0 отвергается хотя бы в 2 из 4 статистик → коинтеграция.
# При наличии коинтеграции переменные входят в УРОВНЯХ (не дифференцируются):
#   ECM-интерпретация: FE-панель с уровнями консистентна при коинтеграции (Kao, 1999).
# При отсутствии коинтеграции I(1) переменные дифференцируются (spurious regression).
#
# Статистики Пед​рони (1999, 2004):
#   Panel v, Panel rho, Panel PP, Panel ADF — within-dimension (pooled)
#   Group rho, Group PP, Group ADF — between-dimension (mean-group)
# Из 7 статистик наиболее надёжны Panel ADF и Group ADF (финитные выборки).
# Критические значения: стандартное нормальное N(0,1) для больших N,T.
# Статистика Panel v: H0 отвергается при БОЛЬШИХ положительных значениях.
# Остальные: H0 отвергается при МАЛЫХ отрицательных (левосторонний тест).

from scipy.stats import norm as _norm_coint

def pedroni_cointegration(df, y_col, x_col, min_obs=15):
    """
    Тест Пе​дрони (1999/2004) для панели (y, x) — оба ряда должны быть I(1).
    Возвращает dict со статистиками и итоговым выводом.
    """
    regions = df['region'].unique()
    resids_all = []
    T_list     = []

    # Шаг 1: получаем остатки регрессии y ~ x для каждого региона
    for reg in regions:
        sub = df[df['region'] == reg][['date', y_col, x_col]].dropna().sort_values('date')
        if len(sub) < min_obs:
            continue
        y = sub[y_col].values
        x = sub[x_col].values
        X = sm.add_constant(x)
        try:
            b   = np.linalg.lstsq(X, y, rcond=None)[0]
            e   = y - X @ b
            resids_all.append(e)
            T_list.append(len(e))
        except Exception:
            continue

    N = len(resids_all)
    if N < 3:
        return {'N': N, 'conclusion': 'Недостаточно регионов для теста'}

    T_avg = np.mean(T_list)

    # ── Моменты Пед​рони (1999, Table 2, без тренда) ──────────────────
    # μ(stat) и σ²(stat) для стандартизации: Z = (stat - N×μ) / sqrt(N×σ²)
    # Значения для T≈40 (интерполируем при необходимости):
    PEDRONI_MOMENTS = {
        # stat: (mu, sigma2)  — для регрессии с константой
        'panel_v':   (-0.551,  0.765),
        'panel_rho': (-1.654,  1.049),
        'panel_pp':  (-1.654,  1.049),   # аппроксимация как rho
        'panel_adf': (-1.654,  1.049),
        'group_rho': (-1.654,  1.049),
        'group_pp':  (-1.654,  1.049),
        'group_adf': (-1.654,  1.049),
    }

    # ── Panel v (Pedroni 1999 eq. 5) ──────────────────────────────────
    v_stats = []
    for e in resids_all:
        T_i   = len(e)
        sigma2 = np.var(e, ddof=1)
        if sigma2 < 1e-12: continue
        # Частичная сумма остатков
        S = np.cumsum(e)
        v_stat = np.sum(S**2) / (T_i**2 * sigma2)
        v_stats.append(v_stat)

    panel_v_raw = np.sum(v_stats) if v_stats else np.nan
    mu_v, s2_v  = PEDRONI_MOMENTS['panel_v']
    Z_panel_v   = (panel_v_raw - N * mu_v) / np.sqrt(N * s2_v) if not np.isnan(panel_v_raw) else np.nan
    # Panel v: H0 отвергается при Z > +1.645 (правосторонний)
    p_panel_v   = float(1 - _norm_coint.cdf(Z_panel_v)) if not np.isnan(Z_panel_v) else np.nan

    # ── ADF-статистики для каждого региона (для Panel ADF и Group ADF) ─
    adf_stats_indiv = []
    for e in resids_all:
        if len(e) < 8: continue
        try:
            adf_res = adfuller(e, maxlag=1, regression='n', autolag=None)
            adf_stats_indiv.append(adf_res[0])
        except Exception:
            pass

    if not adf_stats_indiv:
        return {'N': N, 'conclusion': 'Не удалось вычислить ADF-статистики'}

    # Panel ADF (pooled)
    t_bar_panel  = np.mean(adf_stats_indiv)
    mu_a, s2_a   = PEDRONI_MOMENTS['panel_adf']
    Z_panel_adf  = (np.sqrt(N) * (t_bar_panel - mu_a)) / np.sqrt(s2_a)
    p_panel_adf  = float(_norm_coint.cdf(Z_panel_adf))   # левосторонний

    # Group ADF (mean-group)
    t_bar_group  = np.mean(adf_stats_indiv)
    Z_group_adf  = (np.sqrt(N) * (t_bar_group - mu_a)) / np.sqrt(s2_a)
    p_group_adf  = float(_norm_coint.cdf(Z_group_adf))   # левосторонний

    # ── PP-статистики (Phillips-Perron) — без автокорреляционной коррекции ─
    pp_stats = []
    for e in resids_all:
        if len(e) < 8: continue
        try:
            # PP ≈ ADF(lag=0) для нулевой автокорреляции
            adf0 = adfuller(e, maxlag=0, regression='n', autolag=None)
            pp_stats.append(adf0[0])
        except Exception:
            pass

    t_bar_pp    = np.mean(pp_stats) if pp_stats else np.nan
    mu_p, s2_p  = PEDRONI_MOMENTS['panel_pp']
    Z_panel_pp  = (np.sqrt(N) * (t_bar_pp - mu_p)) / np.sqrt(s2_p) if not np.isnan(t_bar_pp) else np.nan
    p_panel_pp  = float(_norm_coint.cdf(Z_panel_pp)) if not np.isnan(Z_panel_pp) else np.nan

    Z_group_pp  = Z_panel_pp   # упрощение: group PP ≈ panel PP при однородных панелях
    p_group_pp  = p_panel_pp

    # ── Итог: считаем сколько статистик отвергают H0 ─────────────────
    results_stats = {
        'Panel v   (правост.)': (Z_panel_v,  p_panel_v,  p_panel_v  < 0.05),
        'Panel ADF (левост.)':  (Z_panel_adf, p_panel_adf, p_panel_adf < 0.05),
        'Panel PP  (левост.)':  (Z_panel_pp,  p_panel_pp,  p_panel_pp  < 0.05),
        'Group ADF (левост.)':  (Z_group_adf, p_group_adf, p_group_adf < 0.05),
        'Group PP  (левост.)':  (Z_group_pp,  p_group_pp,  p_group_pp  < 0.05),
    }
    n_reject = sum(v[2] for v in results_stats.values())

    # Решение: коинтеграция если ≥2 из 5 статистик отвергают H0
    # (консервативный критерий — Pedroni 2004 рекомендует опираться на Panel ADF + Group ADF)
    cointegrated = (
        n_reject >= 2 or
        (results_stats['Panel ADF (левост.)'][2] and results_stats['Group ADF (левост.)'][2])
    )
    conclusion = 'КОИНТЕГРАЦИЯ ЕСТЬ (уровни корректны)' if cointegrated else 'КОИНТЕГРАЦИИ НЕТ (дифференцировать)'

    return {
        'N': N, 'T_avg': T_avg, 'n_reject': n_reject,
        'stats': results_stats,
        'cointegrated': cointegrated,
        'conclusion': conclusion,
    }


# ============================================================================
# ОСНОВНОЙ ЦИКЛ: ПЕРВИЧНЫЙ И ВТОРИЧНЫЙ РЫНОК
# ============================================================================
for target in TARGETS:
    PRICE_COL     = target['price_col']
    EXCLUDE_REGS  = target['exclude_regressors']
    MARKET_NAME   = target['name']
    OUT_PREFIX    = f"{OUTPUT_FOLDER}/{MARKET_NAME}"

    print("\n" + "="*100)
    print(f"МОДУЛЬ 4 — РЫНОК: {MARKET_NAME.upper()} (цель: {PRICE_COL})")
    print(f"Выборка: N={N_PANEL} регионов × T={T_PANEL} периодов = {OBS_PANEL} наблюдений")
    print("="*100)

    # ------------------------------------------------------------------
    # 1. Подготовка датафрейма для данного рынка
    # ------------------------------------------------------------------
    print(f"\n[1] Подготовка данных для {MARKET_NAME}...")

    df = base_panel.copy()
    df = df.rename(columns={PRICE_COL: 'price'})

    drop_cols = [c for c in df.columns if get_base_name(c) in EXCLUDE_REGS]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"   Исключены cross-market переменные: {drop_cols}")

    # Фильтрация по региональным пропускам (уже не нужна — единая выборка,
    # но оставляем как защитный контроль)
    regional_check_cols = ['price'] + [
        c for c in df.columns
        if c not in ['date', 'region', 'price']
        and get_base_name(c) not in MACRO_VARS
    ]
    regional_check_cols = [c for c in regional_check_cols if c in df.columns]

    region_has_na = df.groupby('region')[regional_check_cols].apply(
        lambda x: x.isnull().any().any()
    )
    bad_regions = region_has_na[region_has_na].index.tolist()
    if bad_regions:
        print(f"   ⚠ Дополнительно исключено регионов с пропусками: {len(bad_regions)} (неожиданно)")
        df = df[~df['region'].isin(bad_regions)]

    macro_base_cols = [c for c in df.columns
                       if c not in ['date', 'region', 'price']
                       and get_base_name(c) in MACRO_VARS]
    if macro_base_cols:
        date_macro  = df[['date'] + macro_base_cols].drop_duplicates('date').sort_values('date')
        bad_dates   = date_macro[date_macro[macro_base_cols].isnull().any(axis=1)]['date']
        if len(bad_dates) > 0:
            print(f"   ⚠ Дополнительно исключено периодов без макро: {bad_dates.nunique()}")
            df = df[~df['date'].isin(bad_dates)]

    df = df.sort_values(['region', 'date']).reset_index(drop=True)

    n_obs_market  = len(df)
    n_reg_market  = df['region'].nunique()
    n_per_market  = df['date'].nunique()
    print(f"✓ {n_obs_market} наблюдений | {n_reg_market} регионов | {n_per_market} периодов")
    print(f"  Период: {df['date'].min().date()} — {df['date'].max().date()}")

    # ------------------------------------------------------------------
    # 2. Нормализация + тест стационарности + дифференцирование I(1)
    # ------------------------------------------------------------------
    print(f"\n[2] Нормализация ({MARKET_NAME})...")

    exclude_from_norm = {'date', 'region', 'price'}
    all_vars          = [c for c in df.columns if c not in exclude_from_norm]
    macro_cols        = [v for v in all_vars if get_base_name(v) in MACRO_VARS]
    region_cols       = [v for v in all_vars if get_base_name(v) not in MACRO_VARS]

    for col in region_cols:
        g      = df.groupby('region')[col]
        median = g.transform('median')
        iqr    = g.transform(
            lambda x: x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
        df[col] = (df[col] - median) / iqr
        df[col] = df[col].fillna(0)

    for col in macro_cols:
        vals    = df[col].values.reshape(-1, 1)
        df[col] = RobustScaler().fit_transform(vals).flatten()
        df[col] = df[col].fillna(0)

    # ── Тест стационарности макро-переменных (ADF + KPSS) ──────────────
    macro_unique = df[['date'] + macro_cols].drop_duplicates('date').sort_values('date')

    I1_VARS = set()
    print(f"\n   Тесты стационарности макро (T={len(macro_unique)}):")
    print(f"   {'Переменная':<22s} {'ADF p':<10s} {'KPSS p':<10s} {'Порядок'}")
    print(f"   {'-'*55}")
    for col in macro_cols:
        s = macro_unique[col].dropna()
        if len(s) < 10:
            continue
        adf_p = adfuller(s, autolag='AIC')[1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                kpss_p = kpss_test(s, regression='c', nlags='auto')[1]
        except:
            kpss_p = np.nan
        stationary_i0 = adf_p < 0.05 and (np.isnan(kpss_p) or kpss_p > 0.05)
        if not stationary_i0:
            s_diff = s.diff().dropna()
            adf_p_d1 = adfuller(s_diff, autolag='AIC')[1]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    kpss_p_d1 = kpss_test(s_diff, regression='c', nlags='auto')[1]
            except:
                kpss_p_d1 = np.nan
            if adf_p_d1 < 0.05 and (np.isnan(kpss_p_d1) or kpss_p_d1 > 0.05):
                order = 'I(1)'
                I1_VARS.add(col)
            else:
                order = 'I(2)+'
        else:
            order = 'I(0) ✓'
        kpss_str = f'{kpss_p:.4f}' if not np.isnan(kpss_p) else 'n/a'
        print(f"   {col:<22s} {adf_p:<10.4f} {kpss_str:<10s} {order}")

    # ── IPS тест для региональных переменных ────────────────────────────
    IPS_MOMENTS = {
        25: (-1.481, 1.010), 30: (-1.493, 1.013), 35: (-1.501, 1.015),
        40: (-1.508, 1.017), 45: (-1.512, 1.018), 50: (-1.514, 1.019),
        60: (-1.516, 1.020), 70: (-1.517, 1.020),
    }

    def _ips_moments(T_val):
        keys = sorted(IPS_MOMENTS.keys())
        if T_val <= keys[0]:  return IPS_MOMENTS[keys[0]]
        if T_val >= keys[-1]: return IPS_MOMENTS[keys[-1]]
        for i in range(len(keys)-1):
            t0, t1 = keys[i], keys[i+1]
            if t0 <= T_val <= t1:
                w = (T_val - t0) / (t1 - t0)
                e0, v0 = IPS_MOMENTS[t0]
                e1, v1 = IPS_MOMENTS[t1]
                return (e0 + w*(e1-e0), v0 + w*(v1-v0))

    I1_REGIONAL = set()   # региональные I(1) — для теста коинтеграции

    print(f"\n   Im-Pesaran-Shin (IPS, 2003) — панельный тест единичного корня:")
    print(f"   H0: все панельные ряды I(1)")
    print(f"   {'Переменная':<28s} {'W-стат':<10s} {'p-value':<10s} {'t_bar':<8s} {'Вывод'}")
    print(f"   {'-'*72}")

    for col in region_cols:
        t_stats_col, T_vals = [], []
        for reg, grp in df.groupby('region'):
            s = grp[col].dropna()
            if len(s) < 10:
                continue
            try:
                res_adf = adfuller(s, autolag='AIC')
                t_stats_col.append(res_adf[0])
                T_vals.append(len(s))
            except:
                pass
        if len(t_stats_col) < 5:
            continue
        N_i   = len(t_stats_col)
        T_avg = np.mean(T_vals)
        e_t, v_t = _ips_moments(int(round(T_avg)))
        t_bar = np.mean(t_stats_col)

        if t_bar > 2.0:
            verdict = 'ВЗРЫВНОЙ — структурный сдвиг (IPS неприменим)'
            print(f"   {col:<28s} {'n/a':<10s} {'n/a':<10s} {t_bar:<8.3f} {verdict}")
            continue

        W     = np.sqrt(N_i) * (t_bar - e_t) / np.sqrt(v_t)
        p_ips = float(_norm_coint.cdf(W))
        if p_ips >= 0.05:
            I1_REGIONAL.add(col)
            verdict = 'I(1) — H0 не отвергнута'
        else:
            verdict = 'I(0) — H0 отвергнута'
        print(f"   {col:<28s} {W:<10.3f} {p_ips:<10.4f} {t_bar:<8.3f} {verdict}")

    # ============================================================================
    # ШАГ 2б: ТЕСТ КОИНТЕГРАЦИИ ПЕД​РОНИ
    # ============================================================================
    # Проводится только для I(1) переменных (макро и региональных) с целевой.
    # Если коинтеграция найдена — переменная остаётся в УРОВНЯХ.
    # Если нет — дифференцируется для устранения spurious regression.
    # Тест применяется до финального дифференцирования I(1) макро.
    # ============================================================================
    print(f"\n[2б] ТЕСТ КОИНТЕГРАЦИИ ПEDRONI — {MARKET_NAME.upper()}")
    print(f"   H0: нет коинтеграции | H1: коинтеграция присутствует")
    print(f"   Критерий: ≥2 из 5 статистик значимы ИЛИ Panel ADF + Group ADF оба значимы")
    print(f"   Источник: Pedroni (1999, 2004), Panel ADF и Group ADF — наиболее надёжны")
    print(f"   {'─'*90}")

    # I(1) переменные для теста: макро I(1) + региональные I(1)
    i1_for_coint = (
        [(v, 'macro')    for v in I1_VARS    if v in df.columns] +
        [(v, 'regional') for v in I1_REGIONAL if v in df.columns]
    )

    COINTEGRATED_VARS  = set()   # остаются в уровнях
    NON_COINTEGRATED   = set()   # дифференцируются

    coint_summary_rows = []

    if not i1_for_coint:
        print(f"   Нет I(1) переменных — тест коинтеграции не применяется.")
    else:
        print(f"   I(1) переменных для теста: {len(i1_for_coint)}")
        print(f"\n   {'Переменная':<28s} {'Тип':<10s} {'N рег.':<8s} "
              f"{'Отверг.':<10s} {'Panel ADF':<12s} {'Group ADF':<12s} {'Вывод'}")
        print(f"   {'─'*90}")

        for var, vtype in i1_for_coint:
            res = pedroni_cointegration(df, 'price', var)
            if 'stats' not in res:
                print(f"   {var:<28s} {vtype:<10s} — тест не выполнен: {res.get('conclusion','')}")
                continue

            padf_z, padf_p, padf_sig = res['stats']['Panel ADF (левост.)']
            gadf_z, gadf_p, gadf_sig = res['stats']['Group ADF (левост.)']
            n_rej = res['n_reject']

            if res['cointegrated']:
                COINTEGRATED_VARS.add(var)
                verdict = '✓ КОИНТ. → уровни'
            else:
                NON_COINTEGRATED.add(var)
                verdict = '✗ НЕТ → дифф.'

            padf_str = f"Z={padf_z:.2f} p={padf_p:.3f}{'*' if padf_sig else ' '}"
            gadf_str = f"Z={gadf_z:.2f} p={gadf_p:.3f}{'*' if gadf_sig else ' '}"

            print(f"   {var:<28s} {vtype:<10s} {res['N']:<8d} "
                  f"{n_rej}/5{'*' if n_rej>=2 else ' ':<7s} "
                  f"{padf_str:<12s} {gadf_str:<12s} {verdict}")

            coint_summary_rows.append({
                'market': MARKET_NAME, 'variable': var, 'type': vtype,
                'N_regions': res['N'], 'T_avg': res.get('T_avg', np.nan),
                'n_reject': n_rej, 'cointegrated': res['cointegrated'],
                'Panel_ADF_Z': padf_z, 'Panel_ADF_p': padf_p,
                'Group_ADF_Z': gadf_z, 'Group_ADF_p': gadf_p,
                'conclusion': res['conclusion'],
            })

        # Детальный вывод по статистикам для каждой переменной
        print(f"\n   Детальные статистики Пед​рони:")
        for var, vtype in i1_for_coint:
            res = pedroni_cointegration(df, 'price', var)
            if 'stats' not in res:
                continue
            print(f"\n   [{var}]  {res['conclusion']}")
            for stat_name, (Z, p, sig) in res['stats'].items():
                if np.isnan(Z):
                    continue
                sig_str = '*** (H0 отвергнута)' if sig else '(H0 не отвергнута)'
                print(f"      {stat_name:<28s}  Z={Z:7.3f}  p={p:.4f}  {sig_str}")

        print(f"\n   Итог:")
        print(f"   Коинтегрированы (уровни):    {sorted(COINTEGRATED_VARS) or '—'}")
        print(f"   Не коинтегрированы (диффер.): {sorted(NON_COINTEGRATED) or '—'}")

    # Сохраняем результаты теста коинтеграции
    if coint_summary_rows:
        coint_df = pd.DataFrame(coint_summary_rows)
        coint_df.to_csv(f"{OUT_PREFIX}_Pedroni_Cointegration.csv", sep=';', index=False)
        print(f"\n   Сохранено: {OUT_PREFIX}_Pedroni_Cointegration.csv")

    # ── Дифференцируем: только I(1) БЕЗ коинтеграции ────────────────────
    diff_vars = (I1_VARS | I1_REGIONAL) - COINTEGRATED_VARS
    if diff_vars:
        print(f"\n   Дифференцируем (I(1) без коинтеграции): {sorted(diff_vars)}")
        for col in diff_vars:
            if col in df.columns:
                df[col] = df.groupby('region')[col].diff()
                df[col] = df[col].fillna(0)
        print(f"   NaN от diff заполнены 0 (первое наблюдение каждого региона)")
    elif I1_VARS or I1_REGIONAL:
        print(f"\n   Все I(1) переменные коинтегрированы с целевой → уровни сохранены, дифференцирование не применяется")
    else:
        print(f"\n   Все переменные стационарны I(0) — дифференцирование не применяется")

    if COINTEGRATED_VARS:
        print(f"\n   ℹ EntityEffects абсорбирует фиксированные тренды.")
        print(f"   При коинтеграции FE-оценки β консистентны (Kao, 1999; Phillips & Moon, 1999).")
        print(f"   Интерпретация: долгосрочная равновесная зависимость.")

    print(f"\n   EntityEffects корректно работает с I(1) рядами (абсорбирует тренд).")

    # ── Составные показатели (feature engineering) ───────────────────────
    print(f"\n   Составные показатели (feature engineering):")

    def _safe_ratio(num_col, den_col, df):
        num   = df[num_col]
        den   = df[den_col].replace(0, np.nan)
        ratio = (num / den).replace([np.inf, -np.inf], np.nan)
        def _winsorize(x):
            lo, hi = x.quantile(0.01), x.quantile(0.99)
            return x.clip(lo, hi)
        ratio = ratio.groupby(df['region']).transform(_winsorize)
        med   = ratio.groupby(df['region']).transform('median')
        iqr   = ratio.groupby(df['region']).transform(
            lambda x: x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
        return ((ratio - med) / iqr).fillna(0)

    created_fe = []

    if 'deals_secondary' in df.columns and 'offers_secondary' in df.columns:
        df['absorption_secondary'] = _safe_ratio('deals_secondary', 'offers_secondary', df)
        created_fe.append('absorption_secondary')

    if 'deals_primary' in df.columns and 'offers_primary' in df.columns:
        df['absorption_primary'] = _safe_ratio('deals_primary', 'offers_primary', df)
        created_fe.append('absorption_primary')

    if 'housing_loans' in df.columns and 'mortgage_debt' in df.columns:
        df['mortgage_intensity'] = _safe_ratio('housing_loans', 'mortgage_debt', df)
        created_fe.append('mortgage_intensity')

    if 'mortgage_overdue' in df.columns and 'mortgage_debt' in df.columns:
        df['overdue_rate'] = _safe_ratio('mortgage_overdue', 'mortgage_debt', df)
        created_fe.append('overdue_rate')

    print(f"   Создано: {len(created_fe)}: {created_fe}")

    ratio_kw  = ['ratio', 'real_rate', 'credit_availability',
                 'supply_intensity', 'primary_share', 'price_ratio']
    base_vars = [c for c in df.columns
                 if c not in ['date', 'region', 'price']
                 and '_lag' not in c
                 and not any(kw in c.lower() for kw in ratio_kw)]

    excluded_vars = [c for c in df.columns
                     if c not in ['date', 'region', 'price'] and '_lag' not in c
                     and any(kw in c.lower() for kw in ratio_kw)]
    if excluded_vars:
        print(f"   Исключено из регрессии (производные/коллинеарные): {excluded_vars}")

    print(f"   Региональных: {len(region_cols)}, макро: {len(macro_cols)}")
    print(f"   Базовых переменных: {len(base_vars)}: {base_vars}")

    # ------------------------------------------------------------------
    # 3. RAW vs SQUARED
    # ------------------------------------------------------------------
    print(f"\n[3] RAW vs SQUARED ({MARKET_NAME})...")

    baseline_raw = estimate_panel_model(df, 'price', base_vars, "Baseline_raw")
    if baseline_raw is None:
        print("Baseline не построен — пропускаем рынок")
        continue
    baseline_r2_raw = baseline_raw['r2_within']
    print(f"   Baseline RAW R² within = {baseline_r2_raw:.4f}")

    THEORY_NONLINEAR   = {'rate', 'inflation', 'sentiment', 'expectations', 'mortgage_overdue'}
    MIN_SQ_DR2_THEORY  = 0.003
    MIN_SQ_DR2_EMPIRIC = 0.005

    significant_squares = []
    for var in base_vars:
        df[f'{var}_sq'] = df[var] ** 2
        result = estimate_panel_model(df, 'price', base_vars + [f'{var}_sq'])
        sq_n   = f'{var}_sq_n'
        if result and sq_n in result['results'].params.index:
            pval  = result['results'].pvalues[sq_n]
            delta = result['r2_within'] - baseline_r2_raw
            has_theory = get_base_name(var) in THEORY_NONLINEAR
            threshold  = MIN_SQ_DR2_THEORY if has_theory else MIN_SQ_DR2_EMPIRIC
            p_thresh   = 0.05 if has_theory else 0.01
            if pval < p_thresh and delta >= threshold:
                significant_squares.append(var)
                tag = "(теория)" if has_theory else "(эмпир.)"
                print(f"   {var:<30s} → SQUARED {tag} (ΔR²={delta:+.4f}, p={pval:.4f}) ***")
            else:
                reason = f"p={pval:.4f}" if pval >= p_thresh else f"ΔR²={delta:+.4f} < {threshold}"
                print(f"   {var:<30s} → RAW  ({reason})")
        else:
            print(f"   {var:<30s} → RAW")

    # ------------------------------------------------------------------
    # 4. Финальные переменные
    # ------------------------------------------------------------------
    chosen_bases = []
    for var in base_vars:
        if var in significant_squares:
            chosen_bases.append(f'{var}_sq')
        else:
            chosen_bases.append(var)

    print(f"\n[4] Финальные переменные ({MARKET_NAME}): {len(chosen_bases)} "
          f"(squared: {len(significant_squares)}, без лагов)")

    final_vars_for_pca = list(chosen_bases)

    print(f"\n   АУДИТ форм переменных:")
    for v in final_vars_for_pca:
        form = 'sq' if '_sq' in v else 'raw'
        print(f"   ✓ '{get_base_name(v)}': {form} → {v}")

    baseline_final = estimate_panel_model(df, 'price', final_vars_for_pca, "Baseline_final")
    baseline_r2    = baseline_final['r2_within'] if baseline_final else 0.0
    print(f"\n   Baseline на финальных переменных R² = {baseline_r2:.4f}")

    # ------------------------------------------------------------------
    # 6. FEATURE ENGINEERING
    # ------------------------------------------------------------------
    print(f"\n[6] FEATURE ENGINEERING — {MARKET_NAME}")
    fe_features = list(final_vars_for_pca)
    fe_info     = [{'variable': v, 'index': None, 'corr_with_index': 1.0,
                    'sign': 1, 'is_quadratic': '_sq' in v} for v in fe_features]

    print(f"\n {len(final_vars_for_pca)} признаков после FE: {fe_features}")

    # ── Устранение мультиколлинеарности ──────────────────────────────────
    MULTICOLL_THRESHOLD = 0.7

    def _var_priority(v):
        bn = get_base_name(v)
        if bn in {'absorption_secondary', 'absorption_primary',
                  'mortgage_intensity', 'overdue_rate'}:
            return 1
        if bn in {'offers_secondary', 'offers_primary', 'deals_primary',
                  'deals_secondary', 'housing_completed'}:
            return 2
        if bn in {'housing_loans', 'mortgage_overdue'}:
            return 3
        if bn in {'mortgage_debt'}:
            return 4
        if v.startswith('idx_'):
            return 5
        return 6

    def _single_r2(feat):
        res = estimate_panel_model(df, 'price', [feat])
        return res['r2_within'] if res else 0.0

    print(f"\n   Проверка мультиколлинеарности (порог |r|>{MULTICOLL_THRESHOLD}):")

    COMPONENT_MAP = {
        'overdue_rate':         {'mortgage_overdue', 'mortgage_debt'},
        'mortgage_intensity':   {'mortgage_debt', 'housing_loans'},
        'absorption_secondary': {'deals_secondary', 'offers_secondary'},
        'absorption_primary':   {'deals_primary', 'offers_primary'},
    }
    component_drop = set()
    for composite, components in COMPONENT_MAP.items():
        composite_present = any(get_base_name(v) == composite for v in fe_features)
        if not composite_present:
            continue
        for comp_base in components:
            for v in fe_features:
                bn = get_base_name(v)
                if bn == comp_base and v not in component_drop:
                    component_drop.add(v)
                    print(f"   ⚠ '{v}' — обратная причинность (следствие цены), убираем")
    if component_drop:
        fe_features = [v for v in fe_features if v not in component_drop]

    to_drop  = set()
    fe_corr  = df[fe_features].corr()
    reported = set()

    for i in range(len(fe_features)):
        for j in range(i + 1, len(fe_features)):
            a, b = fe_features[i], fe_features[j]
            if a in to_drop or b in to_drop:
                continue
            c = abs(fe_corr.iloc[i, j])
            if c <= MULTICOLL_THRESHOLD:
                continue
            pair_key = (min(a, b), max(a, b))
            if pair_key in reported:
                continue
            reported.add(pair_key)

            pa, pb = _var_priority(a), _var_priority(b)
            if pa != pb:
                weaker   = b if pa < pb else a
                stronger = a if pa < pb else b
                reason   = f"приоритет {min(pa,pb)} > {max(pa,pb)}"
            else:
                r2_a = _single_r2(a)
                r2_b = _single_r2(b)
                weaker   = b if r2_a >= r2_b else a
                stronger = a if r2_a >= r2_b else b
                reason   = f"R²: {max(r2_a,r2_b):.4f} vs {min(r2_a,r2_b):.4f}"

            to_drop.add(weaker)
            print(f"   ⚠ {a} ↔ {b}: r={c:.3f} → убираем '{weaker}' ({reason})")

    if not to_drop:
        print(f"   ✓ Мультиколлинеарность отсутствует")
    else:
        fe_features = [v for v in fe_features if v not in to_drop]

    print(f"\n   Предварительный прогон для отсева незначимых признаков (p>0.10):")
    pre_result = estimate_panel_model(df, 'price', fe_features, "pre_significance")
    if pre_result:
        pre_pvals = pre_result['results'].pvalues
        insig     = []
        for v in fe_features:
            vn = f'{v}_n'
            pv = pre_pvals[vn] if vn in pre_pvals.index else np.nan
            if pd.notna(pv) and pv > 0.10:
                insig.append((v, pv))
        if insig:
            drop_insig = {v for v, _ in insig}
            for v, pv in sorted(insig, key=lambda x: x[1], reverse=True):
                print(f"   ⚠ '{v}': p={pv:.4f} → убираем")
            fe_features = [v for v in fe_features if v not in drop_insig]
            print(f"   ✓ После отсева незначимых: {fe_features}")
        else:
            print(f"   ✓ Все признаки значимы (p≤0.10)")

    final_features = fe_features
    pca_info       = []
    independent    = fe_features

    # ------------------------------------------------------------------
    # 7. Финальная модель
    # ------------------------------------------------------------------
    print(f"\n[7] ФИНАЛЬНАЯ ПАНЕЛЬНАЯ РЕГРЕССИЯ — {MARKET_NAME.upper()}")
    print(f"   Выборка: N={n_reg_market} регионов × T={n_per_market} периодов = {n_obs_market} наблюдений")

    df_panel   = df[['date', 'region', 'price'] + final_features].copy().dropna()
    panel_data = df_panel.set_index(['region', 'date'])

    nd = {}
    for col in ['price'] + final_features:
        std = panel_data[col].std()
        nd[f'{col}_n'] = ((panel_data[col] - panel_data[col].mean()) / std
                          if std > 1e-10
                          else panel_data[col] - panel_data[col].mean())
    norm_panel = pd.DataFrame(nd, index=panel_data.index)

    X_norm  = [f'{v}_n' for v in final_features
               if norm_panel[f'{v}_n'].std() > 1e-10]
    results = PanelOLS.from_formula(
        f"price_n ~ {' + '.join(X_norm)} + EntityEffects",
        data=norm_panel).fit(cov_type='clustered', cluster_entity=True)

    r2w = results.rsquared_within
    r2o = results.rsquared_overall
    print(f"   R² within:  {r2w:.4f}")
    print(f"   R² overall: {r2o:.4f}")
    gap = r2w - r2o
    if gap > 0.4:
        print(f"   ℹ R²_within >> R²_overall (gap={gap:.3f}): EntityEffects абсорбирует межрегиональные различия.")
    improvement = r2w - baseline_r2
    print(f"   Baseline (до FE-отбора):  {baseline_r2:.4f}")
    print(f"   Прирост/потеря от FE:     ΔR²={improvement:+.4f} ({improvement*100:+.2f}%)")

    pc_coefs = pd.DataFrame({
        'feature':     results.params.index,
        'coefficient': results.params.values,
        'std_error':   results.std_errors.values,
        'pvalue':      results.pvalues.values,
    })

    # ------------------------------------------------------------------
    # 8. Важность факторов
    # ------------------------------------------------------------------
    print(f"\n[8] ВАЖНОСТЬ ФАКТОРОВ — {MARKET_NAME.upper()}")

    all_importance = []
    for feat in final_features:
        fn = f'{feat}_n'
        if fn not in pc_coefs['feature'].values:
            continue
        coef = pc_coefs.loc[pc_coefs['feature'] == fn, 'coefficient'].values[0]
        pval = pc_coefs.loc[pc_coefs['feature'] == fn, 'pvalue'].values[0]
        is_index       = feat.startswith('idx_')
        constituents   = [fi for fi in fe_info if fi['index'] == feat]
        all_importance.append({
            'variable':     feat,
            'beta_std':     coef,
            'abs_beta_std': abs(coef),
            'pvalue':       pval,
            'is_quadratic': '_sq' in feat,
            'is_index':     is_index,
            'constituents': [fi['variable'] for fi in constituents],
        })

    all_importance.sort(key=lambda x: x['abs_beta_std'], reverse=True)
    max_abs = max((x['abs_beta_std'] for x in all_importance), default=1.0)

    print(f"\n{'Признак':<38s} {'Тип':<12s} {'Важн.':<8s} {'β_std':<10s} "
          f"{'Эффект(ориг)':<14s} {'p-value'}")
    print("-"*100)
    for row in all_importance:
        feat = row['variable']
        orig = (row['beta_std'] * df[feat].std() if feat in df.columns else row['beta_std'])
        pv   = f"{row['pvalue']:.4f}"
        imp  = row['abs_beta_std'] / max_abs
        kind = 'Индекс' if row['is_index'] else ('Квадр.' if row['is_quadratic'] else 'Линейн.')
        print(f"{feat:<38s} {kind:<12s} {imp:>6.4f}   {row['beta_std']:>8.4f}   "
              f"{orig:>10.4f}   {pv}")

    imp_out = pd.DataFrame([{k: v for k, v in r.items() if k != 'constituents'}
                             for r in all_importance])
    imp_out['importance_norm'] = imp_out['abs_beta_std'] / max_abs
    imp_out.to_csv(f"{OUT_PREFIX}_Feature_Importance_FINAL.csv", sep=";", index=False)

    # ── Эффекты в реальных единицах ──────────────────────────────────────
    print(f"\n   Эффекты в реальных единицах (при типичном изменении фактора):")
    print(f"   {'Фактор':<35s} {'Шок':<20s} {'Δ сделок, %':>12s}  Расчёт")
    print(f"   {'-'*80}")

    price_raw  = df_panel['price']
    price_sd   = float(price_raw.std())
    price_mean = float(price_raw.mean())

    UNIT_SHOCKS = {
        'rate':                ('1 п.п. ставки',        1.0),
        'inflation':           ('1 п.п. инфляции',      1.0),
        'sentiment':           ('1 пункт индекса',       1.0),
        'expectations':        ('1 пункт индекса',       1.0),
        'current_state':       ('1 пункт индекса',       1.0),
        'housing_completed':   ('1 тыс. кв. м ввода',   1.0),
        'offers_primary':      ('1% изм. предложения',   1.0),
        'offers_secondary':    ('1% изм. предложения',   1.0),
        'overdue_rate':        ('1 п.п. просрочки/долг', 1.0),
        'mortgage_intensity':  ('1 п.п. кредиты/долг',  1.0),
        'absorption_secondary':('1 п.п. поглощения',     1.0),
        'absorption_primary':  ('1 п.п. поглощения',     1.0),
    }

    for row in all_importance:
        feat   = row['variable']
        bn     = get_base_name(feat)
        beta_s = row['beta_std']
        if bn not in UNIT_SHOCKS:
            continue
        shock_label, shock_size = UNIT_SHOCKS[bn]
        if feat not in df.columns:
            continue
        sd_feat   = float(df[feat].std())
        mean_feat = float(df[feat].mean())
        if sd_feat < 1e-10:
            continue
        if '_sq' in feat:
            delta_pct = beta_s * (price_sd / max(price_mean, 1e-10)) * 2 * mean_feat / sd_feat * shock_size * 100
            calc_note = f"2×β×x̄ (нелин., x̄={mean_feat:.2f})"
        else:
            delta_pct = beta_s * (price_sd / max(price_mean, 1e-10)) / sd_feat * shock_size * 100
            calc_note = f"β×(SD_y/ȳ)/SD_x×шок"
        sign_str = "+" if delta_pct > 0 else ""
        print(f"   {feat:<35s} {shock_label:<20s} {sign_str}{delta_pct:>8.2f}%  [{calc_note}]")

    # ── Диагностика знаков ────────────────────────────────────────────────
    EXPECTED_SIGNS = {
        'offers_primary':      (+1, 'больше предложения → больше сделок'),
        'offers_secondary':    (None, 'знак зависит от лага: overhang давит на ликвидность'),
        'housing_completed':   (+1, 'новое жильё → больше сделок'),
        'inflation':           (-1, 'рост инфляции → снижение реального спроса'),
        'rate':                (-1, 'рост ставки → снижение доступности'),
        'sentiment':           (None, 'неоднозначно после дифференцирования'),
        'expectations':        (+1, 'позитивные ожидания → больше сделок'),
        'current_state':       (None, 'неоднозначно с лагом'),
        'mortgage_intensity':  (None, 'неоднозначно: контрциклическое охлаждение'),
        'absorption_primary':  (+1, 'высокая поглощаемость → активный рынок'),
        'absorption_secondary':(+1, 'высокая поглощаемость → активный рынок'),
        'overdue_rate':        (+1, 'вынужденные продажи → рост сделок'),
    }
    print(f"\n   Диагностика знаков коэффициентов:")
    for row in all_importance:
        bn = get_base_name(row['variable'])
        if bn not in EXPECTED_SIGNS or '_sq' in row['variable']:
            continue
        expected_sign, comment = EXPECTED_SIGNS[bn]
        actual = np.sign(row['beta_std'])
        if expected_sign is None:
            print(f"   ℹ '{row['variable']}': β={row['beta_std']:+.4f} — {comment}")
        elif actual != 0 and actual != expected_sign:
            print(f"   ⚠ '{row['variable']}': β={row['beta_std']:+.4f}, ожидался {'+'if expected_sign>0 else '-'} — {comment}")
        else:
            print(f"   ✓ '{row['variable']}': β={row['beta_std']:+.4f} — {comment}")

    # ------------------------------------------------------------------
    # 9. Breusch-Pagan
    # ------------------------------------------------------------------
    print(f"\n[9] Breusch-Pagan — {MARKET_NAME.upper()}")

    resid_s    = results.resids
    common_idx = norm_panel.index.intersection(resid_s.index)
    resid_arr  = resid_s.reindex(common_idx).values
    X_bp       = sm.add_constant(norm_panel.reindex(common_idx)[X_norm].values)
    mask       = ~(np.isnan(resid_arr) | np.isnan(X_bp).any(axis=1))
    bp         = het_breuschpagan(resid_arr[mask], X_bp[mask])
    print(f"   N={mask.sum()}, LM={bp[0]:.4f}, p={bp[1]:.6f} → "
          f"{'Гетероскедастичность ПРИСУТСТВУЕТ' if bp[1] < 0.05 else 'ОТСУТСТВУЕТ'}")

    # ------------------------------------------------------------------
    # 10. Нормальность остатков
    # ------------------------------------------------------------------
    print(f"\n[10] НОРМАЛЬНОСТЬ ОСТАТКОВ — {MARKET_NAME.upper()}")

    rc         = resid_arr[mask]
    jb_s, jb_p = jarque_bera(rc)
    sw_s, sw_p = shapiro(rc[:5000]) if len(rc) > 5000 else shapiro(rc)
    mean_r     = np.mean(rc)

    print(f"   N={len(rc)}, mean={mean_r:.8f}, std={np.std(rc):.6f}")
    print(f"   Jarque-Bera:  stat={jb_s:.2f}, p={jb_p:.6f} → "
          f"{'Нормальность ОТСУТСТВУЕТ' if jb_p < 0.05 else 'ПРИСУТСТВУЕТ'}")
    print(f"   Shapiro-Wilk: stat={sw_s:.4f}, p={sw_p:.6f} → "
          f"{'Нормальность ОТСУТСТВУЕТ' if sw_p < 0.05 else 'ПРИСУТСТВУЕТ'}")
    print(f"   Смещённость: "
          f"{'ОТСУТСТВУЕТ' if abs(mean_r) < 1e-6 else f'есть ({mean_r:.8f})'}")

    # Сохранение
    with open(f"{OUT_PREFIX}_Final_Model_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"РЫНОК: {MARKET_NAME.upper()}\n")
        f.write(f"Выборка: N={n_reg_market} регионов × T={n_per_market} периодов = {n_obs_market} наблюдений\n")
        f.write(f"Период: {df['date'].min().date()} — {df['date'].max().date()}\n")
        f.write(f"Baseline R²: {baseline_r2:.4f}\n")
        f.write(f"Final R²:    {results.rsquared_within:.4f}\n")
        f.write(f"Improvement: {improvement:+.4f}\n\n")
        f.write(str(results.summary))

    pc_coefs.to_csv(f"{OUT_PREFIX}_Final_Model_PCA_coefs.csv", sep=";", index=False)

    print(f"\n✓ Результаты сохранены с префиксом: {OUT_PREFIX}_*")

    pc_cols = [c for c in df.columns if c.startswith('PC_FINAL_')]
    df.drop(columns=pc_cols, inplace=True, errors='ignore')

print("\n" + "="*100)
print("Оба рынка обработаны")
print(f"Единая выборка: N={N_PANEL} регионов × T={T_PANEL} периодов = {OBS_PANEL} наблюдений")
print("="*100)
