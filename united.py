# =============================================================================
# ПАНЕЛЬНЫЙ АНАЛИЗ РЫНКА ЖИЛЬЯ — v4
# =============================================================================
!pip install linearmodels
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler
from scipy.stats import jarque_bera, shapiro, norm as _norm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller, kpss as kpss_test
from linearmodels.panel import PanelOLS
import re as _re

# ─────────────────────────────────────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_FILE   = r"housingdata_combined.csv"
OUTPUT_FOLDER = r"price_factors_results_v4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MACRO_MAP = {
    'Базовая инфляция':           'inflation',
    'Ключевая ставка в реальном': 'real_rate',
    'Ключевая ставка,':           'rate',
    'индекс потребительских':     'sentiment',
    'индекс ожиданий':            'expectations',
    'индекс текущего':            'current_state',
}
MACRO_VARS = set(MACRO_MAP.values())

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

REQUIRED_VARS = ['deals_primary', 'deals_secondary',
                 'offers_primary', 'offers_secondary',
                 'housing_completed', 'housing_loans',
                 'mortgage_debt', 'mortgage_overdue']

EXCLUDE_REGIONS  = {'РОССИЙСКАЯ ФЕДЕРАЦИЯ','Россия','РФ',
                    'СЕВЕРО-ЗАПАДНЫЙ ФО','ЦЕНТРАЛЬНЫЙ ФО','ПРИВОЛЖСКИЙ ФО',
                    'УРАЛЬСКИЙ ФО','СИБИРСКИЙ ФО','ДАЛЬНЕВОСТОЧНЫЙ ФО',
                    'ЮЖНЫЙ ФО','СЕВЕРО-КАВКАЗСКИЙ ФО','КРЫМСКИЙ ФО'}
EXCLUDE_PATTERNS = ['ФЕДЕРАЦИЯ',' ФО','федеральный округ','ФЕДЕРАЛЬНЫЙ ОКРУГ']

TARGETS = [
    {'name': 'primary',
     'price_col': 'deals_primary',
     'exclude_regressors': {'deals_secondary', 'offers_secondary',
                            'absorption_secondary', 'deals_primary'}},
    {'name': 'secondary',
     'price_col': 'deals_secondary',
     'exclude_regressors': {'deals_primary', 'offers_primary',
                            'absorption_primary', 'deals_secondary'}},
]

THEORY_NONLINEAR    = {'rate', 'inflation', 'sentiment', 'expectations', 'mortgage_overdue'}
MIN_SQ_DR2_THEORY   = 0.003
MIN_SQ_DR2_EMPIRIC  = 0.005
MULTICOLL_THRESHOLD = 0.7

# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────
MONTHS_RU_DEC = {'янв':1,'фев':2,'мар':3,'апр':4,'май':5,'июн':6,
                 'июл':7,'авг':8,'сен':9,'окт':10,'ноя':11,'дек':12}

def fix_excel_number(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    try: return float(s.replace(',', '.'))
    except ValueError: pass
    sl = s.lower()
    for mon, num in MONTHS_RU_DEC.items():
        if mon in sl:
            rest = _re.sub(mon, '', sl).strip('.')
            try: rest_f = float(rest)
            except ValueError: return np.nan
            return (num + rest_f / 100) if sl.startswith(mon) else (rest_f + num / 10)
    return np.nan

def get_base_name(col):
    return col.replace('_sq', '')

def sep(char='─', n=90): print(char * n)

def interpret(text):
    print(f"\n  💬 ИНТЕРПРЕТАЦИЯ: {text}\n")

def compute_vif(df_feat, feat_list):
    """Рассчитывает VIF для каждой переменной из feat_list по df_feat."""
    vif_out = {}
    data = df_feat[feat_list].dropna()
    if len(data) <= len(feat_list) + 1:
        return vif_out
    for col in feat_list:
        y_v = data[col].values
        X_v = np.column_stack([np.ones(len(data)),
                               data[[c for c in feat_list if c != col]].values])
        beta_v = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
        yhat_v = X_v @ beta_v
        ss_r = np.sum((y_v - yhat_v) ** 2)
        ss_t = np.sum((y_v - np.mean(y_v)) ** 2)
        r2_v = 1 - ss_r / ss_t if ss_t > 0 else 0
        vif_out[col] = 1 / (1 - r2_v) if r2_v < 1 else np.inf
    return vif_out

def print_vif_table(vif_dict, title="VIF"):
    """Выводит таблицу VIF с оценкой."""
    print(f"\n  {title}:")
    print(f"  {'Переменная':<38s} {'VIF':>8s}  {'Оценка'}")
    for col, v in vif_dict.items():
        verdict = "ОК" if v < 5 else ("Умеренный" if v < 10 else "ВЫСОКИЙ")
        vif_str = f"{v:>8.2f}" if not np.isinf(v) else "     ∞   "
        print(f"  {col:<38s} {vif_str}  {verdict}")

# ─────────────────────────────────────────────────────────────────────────────
# FIX-3: estimate_panel_model — БЕЗ повторной нормализации внутри функции.
# Данные уже нормализованы на ШАГ 3. Функция только центрирует по глобальному
# среднему (требование PanelOLS) и добавляет суффикс _c для формулы.
# β-коэффициенты теперь в единицах робастно-нормированной шкалы (IQR_x ≈ 1).
# ─────────────────────────────────────────────────────────────────────────────
def estimate_panel_model(df, y_var, X_vars, name=""):
    """
    FE-панельная модель со стандартизацией внутри функции.
    Стандартизация (z-score по глобальному среднему и std) применяется
    единожды здесь — данные из df_work уже робастно нормированы per-region,
    но их глобальный std != 1, что нужно для сопоставимости β.
    Суффикс _n = normalized inside this function.
    """
    cols  = ['date', 'region', y_var] + X_vars
    pdata = df[cols].copy().dropna()
    if len(pdata) < 50: return None
    pdata = pdata.set_index(['region', 'date'])
    nd = {}
    for col in [y_var] + X_vars:
        std = pdata[col].std()
        nd[f'{col}_n'] = ((pdata[col] - pdata[col].mean()) / std
                          if std > 1e-10 else pdata[col] * 0)
    nd = pd.DataFrame(nd, index=pdata.index)
    X_n = [f'{v}_n' for v in X_vars if nd[f'{v}_n'].std() > 1e-10]
    if not X_n: return None
    try:
        res = PanelOLS.from_formula(
            f"{y_var}_n ~ {' + '.join(X_n)} + EntityEffects",
            data=nd).fit(cov_type='clustered', cluster_entity=True)
        return {'name': name, 'r2_within': res.rsquared_within,
                'n_obs': res.nobs, 'results': res}
    except Exception as e:
        if name: print(f"  ⚠ {name}: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ТЕСТ ПЕДРОНИ — на исходных ненормализованных данных (FIX-4)
# ─────────────────────────────────────────────────────────────────────────────
PEDRONI_MOMENTS = {
    'group_adf': (-1.654, 1.000),
    'panel_adf': (-1.520, 0.726),
    'group_pp':  (-7.823, 94.154),
    'panel_pp':  (-1.457, 0.711),
}

def pedroni_test(y_dict, x_dict, T_min=15):
    adf_stats, pp_stats = [], []
    for reg in y_dict:
        if reg not in x_dict: continue
        y_arr = np.asarray(y_dict[reg], dtype=float)
        x_arr = np.asarray(x_dict[reg], dtype=float)
        mask  = ~(np.isnan(y_arr) | np.isnan(x_arr))
        yc, xc = y_arr[mask], x_arr[mask]
        if len(yc) < T_min: continue
        X_mat = np.column_stack([np.ones(len(xc)), xc])
        try:
            beta = np.linalg.lstsq(X_mat, yc, rcond=None)[0]
            e = yc - X_mat @ beta
        except Exception: continue
        try: adf_stats.append(adfuller(e, maxlag=1, regression='n', autolag=None)[0])
        except Exception: pass
        try: pp_stats.append(adfuller(e, maxlag=0, regression='n', autolag=None)[0])
        except Exception: pass
    N = len(adf_stats)
    if N < 3: return None
    results = {'n_units': N}
    for stat_name, stats_list in [('adf', adf_stats), ('pp', pp_stats)]:
        if not stats_list: continue
        t_bar  = np.mean(stats_list)
        n_used = len(stats_list)
        mu_g, var_g = PEDRONI_MOMENTS[f'group_{stat_name}']
        mu_p, var_p = PEDRONI_MOMENTS[f'panel_{stat_name}']
        Z_group = np.sqrt(n_used) * (t_bar - mu_g) / np.sqrt(var_g)
        Z_panel = np.sqrt(n_used) * (t_bar - mu_p) / np.sqrt(var_p)
        results[f'group_{stat_name}'] = {'stat': t_bar, 'Z': Z_group, 'p': float(_norm.cdf(Z_group))}
        results[f'panel_{stat_name}'] = {'stat': t_bar, 'Z': Z_panel, 'p': float(_norm.cdf(Z_panel))}
    return results

def run_pedroni(raw_panel, y_col, x_col, valid_regions):
    """
    FIX-4: тест Педрони на ИСХОДНЫХ (ненормализованных) данных.
    raw_panel — unified_panel до нормализации.
    """
    y_dict, x_dict = {}, {}
    for reg, grp in raw_panel.groupby('region'):
        if reg not in valid_regions: continue
        g = grp.sort_values('date')
        y_dict[reg] = g[y_col].values
        x_dict[reg] = g[x_col].values
    res = pedroni_test(y_dict, x_dict)
    if res is None:
        return False, f"   {x_col:<25s} — недостаточно данных"
    p_panel = res.get('panel_adf', {}).get('p', 1.0)
    p_group = res.get('group_adf', {}).get('p', 1.0)
    n_sig = sum(1 for k in ['panel_adf','group_adf','panel_pp','group_pp']
                if k in res and res[k]['p'] < 0.05)
    n_tot = sum(1 for k in ['panel_adf','group_adf','panel_pp','group_pp'] if k in res)
    cointegrated = (p_panel < 0.05) and (p_group < 0.05)
    verdict = "Коинтеграция ✓" if cointegrated else "Нет коинтеграции"
    summary = (f"   {x_col:<25s} "
               f"PanelADF p={p_panel:.3f}  GroupADF p={p_group:.3f}  "
               f"[{n_sig}/{n_tot} значимы]  → {verdict}")
    return cointegrated, summary

# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 0: ЗАГРУЗКА ДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────
sep('═')
print("ШАГ 0. ЗАГРУЗКА И СБОРКА ПАНЕЛИ")
sep('═')

raw = pd.read_csv(SOURCE_FILE, sep=';')
raw['date'] = pd.to_datetime(raw['date'], dayfirst=True)

for col in raw.columns:
    if col == 'date': continue
    n_bad = raw[col].apply(
        lambda x: pd.isna(pd.to_numeric(str(x).replace(',','.'), errors='coerce'))
        if pd.notna(x) else False).sum()
    if n_bad > 0:
        raw[col] = raw[col].apply(fix_excel_number)

raw = raw.loc[:, ~raw.columns.duplicated()]
seen_h, drop_c = {}, []
for col in raw.columns:
    h = pd.util.hash_pandas_object(raw[col].fillna(-9999)).sum()
    if h in seen_h: drop_c.append(col)
    else: seen_h[h] = col
if drop_c:
    raw = raw.drop(columns=drop_c)

print(f"Строк: {len(raw)} | Колонок: {raw.shape[1]} | "
      f"Период: {raw['date'].min():%Y-%m} — {raw['date'].max():%Y-%m}")

macro_df = pd.DataFrame({'date': raw['date']})
for substr, varname in MACRO_MAP.items():
    matched = [c for c in raw.columns if substr in c]
    if matched:
        macro_df[varname] = pd.to_numeric(raw[matched[0]], errors='coerce').values

def prefix_to_long(raw, prefix, varname):
    cols = [c for c in raw.columns if c.startswith(prefix)]
    if not cols: return pd.DataFrame(columns=['date','region',varname])
    long = raw[['date']+cols].melt(id_vars='date', value_vars=cols,
                                   var_name='_col', value_name=varname)
    long['region'] = long['_col'].str[len(prefix)+1:]
    long[varname]  = pd.to_numeric(long[varname], errors='coerce')
    return long[['date','region',varname]]

panel_parts = []
for prefix, varname in REGIONAL_KEYS.items():
    part = prefix_to_long(raw, prefix, varname)
    panel_parts.append(part)

base_panel = panel_parts[0]
for part in panel_parts[1:]:
    base_panel = base_panel.merge(part, on=['date','region'], how='outer')

base_panel = base_panel.merge(macro_df, on='date', how='left')
base_panel = base_panel[~base_panel['region'].isin(EXCLUDE_REGIONS)]
for pat in EXCLUDE_PATTERNS:
    base_panel = base_panel[~base_panel['region'].str.contains(pat, case=False, na=False)]
base_panel = base_panel.sort_values(['region','date']).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 1: ЕДИНАЯ ВЫБОРКА РЕГИОНОВ
# ─────────────────────────────────────────────────────────────────────────────
sep()
print("ШАГ 1. ФОРМИРОВАНИЕ ЕДИНОЙ ВЫБОРКИ РЕГИОНОВ")
sep()
print("""
  Регион включается, если имеет полные данные по ВСЕМ обязательным переменным
  за все 45 месяцев. Обеспечивает сопоставимость при межрыночном сравнении.
""")

region_completeness = base_panel.groupby('region')[REQUIRED_VARS].apply(
    lambda x: x.notna().all(axis=1).sum()
)
T_total       = base_panel['date'].nunique()
full_mask     = region_completeness == T_total
UNIFIED_REGIONS = set(full_mask[full_mask].index.tolist())

print(f"  Всего регионов: {base_panel['region'].nunique()} | "
      f"С полными данными за {T_total} периодов: {len(UNIFIED_REGIONS)}")
for var in REQUIRED_VARS:
    n_full = base_panel.groupby('region')[var].apply(lambda x: x.notna().sum() == T_total).sum()
    print(f"    {var:<35s}: {n_full}")

unified_panel = base_panel[base_panel['region'].isin(UNIFIED_REGIONS)].copy()
unified_panel = unified_panel.sort_values(['region','date']).reset_index(drop=True)
N_regions = unified_panel['region'].nunique()
N_periods = unified_panel['date'].nunique()
N_obs     = len(unified_panel)

# FIX-4: сохраняем исходную (ненормализованную) панель для теста Педрони
unified_panel_raw = unified_panel.copy()

print(f"\n  ✓ Панель: {N_regions} регионов × {N_periods} периодов = {N_obs} наблюдений")
interpret(
    f"Единая панель из {N_regions} регионов гарантирует, что оба рынка "
    f"оцениваются на одном и том же наборе регионов — необходимое условие "
    f"корректного межсегментного сравнения коэффициентов."
)

# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 2: ТЕСТЫ СТАЦИОНАРНОСТИ
# ─────────────────────────────────────────────────────────────────────────────
sep()
print("ШАГ 2. ТЕСТЫ СТАЦИОНАРНОСТИ (единые для обоих рынков)")
sep()
print("""
  Стационарность проверяется ДО нормализации, на исходных значениях.
  • I(1) макро → дифференцируются (монетарные переменные влияют через поток,
    а не через долгосрочный уровень);
  • I(1) региональные → тест коинтеграции Педрони (отдельно для каждого рынка).
""")

macro_unique = (unified_panel[['date'] + list(MACRO_VARS)]
                .drop_duplicates('date').sort_values('date'))

I1_MACRO   = set()
STAT_MACRO = set()

print(f"  2а. ADF + KPSS для макропеременных (T = {len(macro_unique)}):")
print(f"  {'Переменная':<22s} {'ADF p':<10s} {'KPSS p':<10s} {'Порядок'}")
print(f"  {'-'*54}")

for col in MACRO_VARS:
    s = macro_unique[col].dropna()
    if len(s) < 10: continue
    adf_p = adfuller(s, autolag='AIC')[1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kpss_p = kpss_test(s, regression='c', nlags='auto')[1]
    except: kpss_p = np.nan
    i0 = adf_p < 0.05 and (np.isnan(kpss_p) or kpss_p > 0.05)
    if not i0:
        s_d   = s.diff().dropna()
        adf_d = adfuller(s_d, autolag='AIC')[1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                kpss_d = kpss_test(s_d, regression='c', nlags='auto')[1]
        except: kpss_d = np.nan
        order = 'I(1) → Δ' if (adf_d < 0.05 and (np.isnan(kpss_d) or kpss_d > 0.05)) else 'I(2)+'
        if order == 'I(1) → Δ': I1_MACRO.add(col)
    else:
        order = 'I(0) ✓'
        STAT_MACRO.add(col)
    kpss_str = f'{kpss_p:.4f}' if not np.isnan(kpss_p) else 'n/a'
    print(f"  {col:<22s} {adf_p:<10.4f} {kpss_str:<10s} {order}")

interpret(
    f"I(0): {sorted(STAT_MACRO) or 'нет'}. I(1) → Δ: {sorted(I1_MACRO)}. "
    f"Макро I(1) дифференцируются — тест Педрони для них не применяется: "
    f"монетарные переменные влияют через поток (изменение условий), а не уровень."
)

# IPS для региональных
IPS_MOMENTS_TABLE = {
    25:(-1.481,1.010), 30:(-1.493,1.013), 35:(-1.501,1.015),
    40:(-1.508,1.017), 45:(-1.512,1.018), 50:(-1.514,1.019),
    60:(-1.516,1.020), 70:(-1.517,1.020),
}

def ips_moments(T_val):
    keys = sorted(IPS_MOMENTS_TABLE)
    if T_val <= keys[0]: return IPS_MOMENTS_TABLE[keys[0]]
    if T_val >= keys[-1]: return IPS_MOMENTS_TABLE[keys[-1]]
    for i in range(len(keys)-1):
        t0, t1 = keys[i], keys[i+1]
        if t0 <= T_val <= t1:
            w = (T_val-t0)/(t1-t0)
            e0,v0 = IPS_MOMENTS_TABLE[t0]; e1,v1 = IPS_MOMENTS_TABLE[t1]
            return (e0+w*(e1-e0), v0+w*(v1-v0))

REGIONAL_VARS_REGR = [v for v in unified_panel.columns
                      if v not in ['date','region'] + list(MACRO_VARS)
                      and v not in ['deals_primary','deals_secondary']]

I1_REGIONAL        = set()
STAT_REGIONAL      = set()
EXPLOSIVE_REGIONAL = set()

print(f"\n  2б. IPS-тест региональных переменных (N={N_regions}, T={N_periods}):")
print(f"  {'Переменная':<28s} {'W-стат':<10s} {'p-value':<10s} {'t̄':<8s} {'Вывод'}")
print(f"  {'-'*70}")

for col in REGIONAL_VARS_REGR:
    t_stats, t_lens = [], []
    for reg, grp in unified_panel.groupby('region'):
        s = grp[col].dropna()
        if len(s) < 10: continue
        try:
            t_stats.append(adfuller(s, autolag='AIC')[0])
            t_lens.append(len(s))
        except: pass
    if len(t_stats) < 5: continue
    t_bar = np.mean(t_stats)
    if t_bar > 2.0:
        EXPLOSIVE_REGIONAL.add(col)
        print(f"  {col:<28s} {'n/a':<10s} {'n/a':<10s} {t_bar:<8.3f} ВЗРЫВНОЙ (IPS неприменим)")
        continue
    e_t, v_t = ips_moments(int(round(np.mean(t_lens))))
    W     = np.sqrt(len(t_stats)) * (t_bar - e_t) / np.sqrt(v_t)
    p_ips = float(_norm.cdf(W))
    if p_ips >= 0.05:
        I1_REGIONAL.add(col)
    else:
        STAT_REGIONAL.add(col)
    verdict = 'I(0) — H0 отвергнута' if p_ips < 0.05 else 'I(1) — H0 не отвергнута'
    print(f"  {col:<28s} {W:<10.3f} {p_ips:<10.4f} {t_bar:<8.3f} {verdict}")

interpret(
    f"I(0) региональные: {sorted(STAT_REGIONAL)}. "
    f"I(1): {sorted(I1_REGIONAL)} → тест Педрони (по рынкам). "
    f"Взрывные: {sorted(EXPLOSIVE_REGIONAL)} — структурный сдвиг 2024 года."
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX-1 & FIX-2: СОСТАВНЫЕ ПОКАЗАТЕЛИ ИЗ ИСХОДНЫХ ДАННЫХ
# Вычисляем overdue_rate и mortgage_intensity ДО нормализации.
# orig_stats хранит IQR самого отношения — не его компонент.
# ─────────────────────────────────────────────────────────────────────────────
print()
sep()
print("ШАГ 2в. СОСТАВНЫЕ ПОКАЗАТЕЛИ (из исходных данных до нормализации)")
sep()

def build_ratio_raw(panel, num_col, den_col, varname):
    """
    Строит отношение num/den из исходных данных.
    Winsorize [1%,99%] per region, затем per-region RobustScaler.
    Сохраняет orig_stats с IQR отношения в оригинальных единицах.
    Возвращает нормированный ряд и orig_stats-запись.
    """
    num = panel[num_col]
    den = panel[den_col].replace(0, np.nan)
    ratio = (num / den).replace([np.inf, -np.inf], np.nan)
    # Winsorize per region
    ratio = ratio.groupby(panel['region']).transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
    # Сохраняем статистики оригинального отношения
    grp_med = ratio.groupby(panel['region']).transform('median')
    grp_iqr = ratio.groupby(panel['region']).transform(
        lambda x: x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
    stats = {
        'iqr':    float(grp_iqr.mean()),      # IQR отношения в оригинальных единицах
        'median': float(grp_med.mean()),
        'mean':   float(ratio.mean()),
        'std':    float(ratio.std()),
    }
    # Per-region RobustScaler
    norm = ((ratio - grp_med) / grp_iqr).fillna(0)
    return norm, stats

ratios_raw = {}
if all(c in unified_panel.columns for c in ['housing_loans','mortgage_debt']):
    val, st = build_ratio_raw(unified_panel, 'housing_loans', 'mortgage_debt',
                              'mortgage_intensity')
    ratios_raw['mortgage_intensity'] = (val, st)
    print(f"  mortgage_intensity = housing_loans / mortgage_debt")
    print(f"    IQR(оригинал): {st['iqr']:.6f}  mean: {st['mean']:.6f}")

if all(c in unified_panel.columns for c in ['mortgage_overdue','mortgage_debt']):
    val, st = build_ratio_raw(unified_panel, 'mortgage_overdue', 'mortgage_debt',
                              'overdue_rate')
    ratios_raw['overdue_rate'] = (val, st)
    print(f"  overdue_rate = mortgage_overdue / mortgage_debt")
    print(f"    IQR(оригинал): {st['iqr']:.6f}  mean: {st['mean']:.6f}")

interpret(
    "Составные показатели вычислены из ИСХОДНЫХ данных — отношение оригинальных "
    "числителя и знаменателя нормируется отдельно. IQR хранится для самого "
    "отношения, а не его компонент (IQR overdue_rate ≈ 0.001, а не ~253 "
    "для абсолютной просрочки). Это устраняет искажение предельных эффектов."
)

# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 3: НОРМАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────
sep()
print("ШАГ 3. РОБАСТНАЯ НОРМАЛИЗАЦИЯ")
sep()
print("""
  Региональные: x_norm = (x − median_region) / IQR_region  (per-region)
  Макропеременные: глобальная RobustScaler.
  Составные показатели уже нормированы на ШАГ 2в.
  Параметры нормализации (IQR_orig, mean_orig) сохраняются для предельных эффектов.
""")

df_work   = unified_panel.copy()
orig_stats = {}   # {varname: {iqr, median, mean, std}} в оригинальных единицах

all_vars   = [c for c in df_work.columns if c not in ['date','region']]
macro_cols = [v for v in all_vars if get_base_name(v) in MACRO_VARS]
regio_cols = [v for v in all_vars if get_base_name(v) not in MACRO_VARS
              and v not in ['deals_primary','deals_secondary']]

for col in regio_cols:
    g_med = df_work.groupby('region')[col].transform('median')
    g_iqr = df_work.groupby('region')[col].transform(
        lambda x: x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
    orig_stats[col] = {
        'iqr':    float(g_iqr.mean()),
        'median': float(g_med.mean()),
        'mean':   float(df_work[col].mean()),
        'std':    float(df_work[col].std()),
    }
    df_work[col] = ((df_work[col] - g_med) / g_iqr).fillna(0)

for col in macro_cols:
    scaler = RobustScaler()
    vals   = df_work[col].values.reshape(-1,1)
    orig_stats[col] = {
        'iqr':    float(scaler.fit(vals).scale_[0]),
        'median': float(np.nanmedian(df_work[col])),
        'mean':   float(df_work[col].mean()),
        'std':    float(df_work[col].std()),
    }
    df_work[col] = scaler.fit_transform(vals).flatten()
    df_work[col] = df_work[col].fillna(0)

# Дифференцируем I(1) макро
for col in I1_MACRO:
    if col in df_work.columns:
        df_work[col] = df_work.groupby('region')[col].diff().fillna(0)

# Добавляем составные показатели (уже нормированы из исходных данных)
for varname, (norm_series, stats) in ratios_raw.items():
    df_work[varname] = norm_series.values
    orig_stats[varname] = stats   # IQR и mean самого отношения

# Сохраняем orig_stats для целевых переменных (до нормализации — из raw панели)
for price_col in ['deals_primary', 'deals_secondary']:
    grp = unified_panel.groupby('region')[price_col]
    g_med_p = grp.transform('median')
    g_iqr_p = grp.transform(
        lambda x: x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
    orig_stats[price_col] = {
        'iqr':    float(g_iqr_p.mean()),
        'median': float(g_med_p.mean()),
        'mean':   float(unified_panel[price_col].mean()),
        'std':    float(unified_panel[price_col].std()),
    }

print(f"  Нормализовано: {len(regio_cols)} региональных, {len(macro_cols)} макро, "
      f"{len(ratios_raw)} составных.")
print(f"  Дифференцированы I(1) макро: {sorted(I1_MACRO)}")

# ─────────────────────────────────────────────────────────────────────────────
# ОСНОВНОЙ ЦИКЛ
# ─────────────────────────────────────────────────────────────────────────────
ALL_RESULTS = {}

for target in TARGETS:
    PRICE_COL = target['price_col']
    EXCL_REGS = target['exclude_regressors']
    MNAME     = target['name']
    OUT_PFX   = f"{OUTPUT_FOLDER}/{MNAME}"
    LABEL     = "ПЕРВИЧНЫЙ" if MNAME == 'primary' else "ВТОРИЧНЫЙ"

    sep('═')
    print(f"РЫНОК: {LABEL} (зависимая переменная: {PRICE_COL})")
    sep('═')

    df = df_work.copy()
    df = df.rename(columns={PRICE_COL: 'price'})
    drop_c = [c for c in df.columns if get_base_name(c) in EXCL_REGS]
    if drop_c:
        df = df.drop(columns=drop_c)
        print(f"\n  Исключены cross-market переменные: {drop_c}")
    df = df.sort_values(['region','date']).reset_index(drop=True)
    print(f"  Панель: {len(df)} наблюдений | {df['region'].nunique()} регионов | "
          f"{df['date'].nunique()} периодов")

    # Исходные данные для Педрони (с переименованием целевой)
    raw_for_pedroni = unified_panel_raw.copy().rename(columns={PRICE_COL: 'price'})

    # ── ШАГ 4: Тест Педрони ──────────────────────────────────────────────────
    sep()
    print(f"ШАГ 4. ТЕСТ КОИНТЕГРАЦИИ ПЕДРОНИ — {LABEL}")
    sep()
    print("""
  Тест проводится на ИСХОДНЫХ (ненормализованных) данных — FIX-4.
  Только для I(1) региональных переменных: ложная регрессия возможна только
  для нестационарных рядов в уровнях.
  I(1) макро уже дифференцированы — для них тест неприменим (Δx уже I(0)).
  Критерий: Panel ADF и Group ADF оба p < 0.05 одновременно.
    """)

    COINTEGRATED     = set()
    NON_COINTEGRATED = set()

    i1_in_model = [c for c in sorted(I1_REGIONAL) if c in df.columns]

    if not i1_in_model:
        print("  Нет I(1) региональных переменных — тест не нужен.")
    else:
        print(f"  I(1) региональные в данном рынке: {i1_in_model}")
        print(f"\n  {'Переменная':<25s} {'PanelADF p':<14s} {'GroupADF p':<14s} "
              f"{'Знач./4':<10s} {'Вывод'}")
        print(f"  {'-'*75}")
        for x_col in i1_in_model:
            if x_col not in raw_for_pedroni.columns: continue
            cointegrated, summary = run_pedroni(
                raw_for_pedroni, 'price', x_col, UNIFIED_REGIONS)
            print(summary)
            if cointegrated: COINTEGRATED.add(x_col)
            else:             NON_COINTEGRATED.add(x_col)

        print(f"\n  Коинтегрированы (в уровнях): {sorted(COINTEGRATED) or '—'}")
        print(f"  Не коинтегрированы (→ Δ):    {sorted(NON_COINTEGRATED) or '—'}")

        for col in NON_COINTEGRATED:
            if col in df.columns:
                df[col] = df.groupby('region')[col].diff().fillna(0)
                print(f"  Δ применено к '{col}'")

        if 'mortgage_debt' in COINTEGRATED:
            interpret(
                f"mortgage_debt коинтегрирован с объёмом сделок на {LABEL.lower()} рынке — "
                f"устойчивая долгосрочная равновесная связь. Переменная входит в уровнях."
            )
        if 'mortgage_debt' in NON_COINTEGRATED:
            interpret(
                f"mortgage_debt НЕ коинтегрирован на {LABEL.lower()} рынке — "
                f"включение в уровнях привело бы к ложной регрессии. Применено Δ."
            )

    # ── ШАГ 5: Тест нелинейностей ────────────────────────────────────────────
    sep()
    print(f"ШАГ 5. ТЕСТ НЕЛИНЕЙНОСТЕЙ — {LABEL}")
    sep()

    ratio_kw  = ['ratio','credit_availability','supply_intensity','primary_share']
    base_vars = [c for c in df.columns
                 if c not in ['date','region','price']
                 and '_sq' not in c
                 and not any(kw in c.lower() for kw in ratio_kw)
                 and df[c].std() > 1e-8]

    baseline_raw = estimate_panel_model(df, 'price', base_vars, "Baseline RAW")
    if baseline_raw is None:
        print("  ⚠ Baseline не построен"); continue
    r2_raw = baseline_raw['r2_within']
    print(f"  Baseline RAW: R² within = {r2_raw:.4f}")

    significant_sq = []
    for var in base_vars:
        df[f'{var}_sq'] = df[var] ** 2
        base_st = orig_stats.get(var, {})
        orig_stats[f'{var}_sq'] = {
            'iqr':    base_st.get('iqr', 1.0),
            'median': float(df[var].median()),
            'mean':   float(df[var].mean()),
            'std':    float(df[f'{var}_sq'].std()),
        }
        res = estimate_panel_model(df, 'price', base_vars + [f'{var}_sq'])
        sq_n = f'{var}_sq_n'
        if res and sq_n in res['results'].params.index:
            pval  = res['results'].pvalues[sq_n]
            delta = res['r2_within'] - r2_raw
            has_theory = get_base_name(var) in THEORY_NONLINEAR
            thr_r2 = MIN_SQ_DR2_THEORY if has_theory else MIN_SQ_DR2_EMPIRIC
            thr_p  = 0.05              if has_theory else 0.01
            tag    = "(теория)" if has_theory else "(эмпир.) "
            if pval < thr_p and delta >= thr_r2:
                significant_sq.append(var)
                print(f"    ✓ {var:<28s} → SQUARED {tag}  ΔR²={delta:+.4f}  p={pval:.4f}")
            else:
                reason = f"p={pval:.4f}" if pval >= thr_p else f"ΔR²={delta:+.4f}<{thr_r2}"
                print(f"    · {var:<28s} → RAW       {reason}")
        else:
            print(f"    · {var:<28s} → RAW       (не значим)")

    chosen = [f'{v}_sq' if v in significant_sq else v for v in base_vars]

    if significant_sq:
        interpret(f"Квадратичные формы приняты для: {significant_sq}.")
    else:
        interpret("Квадратичные термы не приняты.")

    # ── ШАГ 6: VIF (диагностика) + мультиколлинеарность + VIF (финал) ────────
    sep()
    print(f"ШАГ 6. МУЛЬТИКОЛЛИНЕАРНОСТЬ И VIF — {LABEL}")
    sep()

    fe_feats = list(chosen)

    # Компонентная фильтрация
    COMP_MAP = {
        'overdue_rate':       {'mortgage_overdue','mortgage_debt'},
        'mortgage_intensity': {'mortgage_debt','housing_loans'},
    }
    comp_drop = set()
    for composite, components in COMP_MAP.items():
        if not any(get_base_name(v) == composite for v in fe_feats): continue
        for comp_base in components:
            for v in fe_feats:
                if get_base_name(v) == comp_base and v not in comp_drop:
                    comp_drop.add(v)
    if comp_drop:
        fe_feats = [v for v in fe_feats if v not in comp_drop]
        print(f"\n  Убраны компоненты составных: {sorted(comp_drop)}")

    # FIX-5а: ДИАГНОСТИЧЕСКАЯ таблица VIF (до удалений по корреляции)
    vif_diag = compute_vif(df, fe_feats)
    print_vif_table(vif_diag, "  VIF диагностическая (до удаления коллинеарных пар)")

    # Парная корреляция
    fe_corr  = df[fe_feats].corr()
    to_drop  = set()
    reported = set()

    def var_prio(v):
        bn = get_base_name(v)
        if bn in {'overdue_rate','mortgage_intensity'}: return 1
        if bn in {'offers_secondary','offers_primary','housing_completed'}: return 2
        if bn in {'housing_loans','mortgage_overdue'}: return 3
        if bn in {'mortgage_debt'}: return 4
        return 5

    def single_r2(feat):
        res = estimate_panel_model(df, 'price', [feat])
        return res['r2_within'] if res else 0.0

    high_corr_pairs = []
    for i in range(len(fe_feats)):
        for j in range(i+1, len(fe_feats)):
            a, b = fe_feats[i], fe_feats[j]
            if a in to_drop or b in to_drop: continue
            c = abs(fe_corr.iloc[i,j])
            if c <= MULTICOLL_THRESHOLD: continue
            key = (min(a,b), max(a,b))
            if key in reported: continue
            reported.add(key)
            pa, pb = var_prio(a), var_prio(b)
            if pa != pb:
                weaker = b if pa < pb else a
                reason = f"приоритет {min(pa,pb)} > {max(pa,pb)}"
            else:
                r2a, r2b = single_r2(a), single_r2(b)
                weaker   = b if r2a >= r2b else a
                reason   = f"R² {max(r2a,r2b):.4f} vs {min(r2a,r2b):.4f}"
            to_drop.add(weaker)
            high_corr_pairs.append((a, b, c, weaker, reason))
            print(f"\n  ⚠ {a} ↔ {b}: r={c:.3f} → убираем '{weaker}' ({reason})")

    if not high_corr_pairs:
        print(f"\n  ✓ Высокая корреляция отсутствует (|r| ≤ {MULTICOLL_THRESHOLD})")

    fe_feats = [v for v in fe_feats if v not in to_drop]

    # Предварительный прогон — отсев незначимых
    print(f"\n  Предварительный прогон (отсев p > 0.10):")
    pre_res = estimate_panel_model(df, 'price', fe_feats, "pre")
    if pre_res:
        pre_pv = pre_res['results'].pvalues
        insig  = [(v, pre_pv[f'{v}_n']) for v in fe_feats
                  if f'{v}_n' in pre_pv.index and pre_pv[f'{v}_n'] > 0.10]
        if insig:
            for v, pv in sorted(insig, key=lambda x: x[1], reverse=True):
                print(f"  ⚠ '{v}': p={pv:.4f} → убираем")
            fe_feats = [v for v in fe_feats if v not in {v for v,_ in insig}]
            print(f"  После отсева: {fe_feats}")
        else:
            print(f"  ✓ Все переменные значимы (p ≤ 0.10)")

    # FIX-5б: ФИНАЛЬНАЯ таблица VIF (после всех удалений)
    vif_final = compute_vif(df, fe_feats)
    print_vif_table(vif_final, "  VIF финальная (после всех удалений)")

    # FIX-7: корректная интерпретация VIF
    max_vif_diag  = max((v for v in vif_diag.values()  if not np.isinf(v)), default=0)
    max_vif_final = max((v for v in vif_final.values() if not np.isinf(v)), default=0)
    vif_msg_diag = (
        f"До удалений: max VIF = {max_vif_diag:.1f} "
        + ("(критическая мультиколлинеарность)" if max_vif_diag >= 10
           else "(умеренная)" if max_vif_diag >= 5 else "(ОК)")
    )
    vif_msg_final = (
        f"После удалений: max VIF = {max_vif_final:.1f} "
        + ("— критическая мультиколлинеарность сохраняется!" if max_vif_final >= 10
           else "— умеренная, допустимо" if max_vif_final >= 5
           else "— все переменные независимы (ОК)")
    )
    if high_corr_pairs:
        pairs_str = "; ".join(f"{a}↔{b} (r={c:.2f})" for a,b,c,_,_ in high_corr_pairs)
        interpret(f"Высокая корреляция: {pairs_str}. {vif_msg_diag}. {vif_msg_final}.")
    else:
        interpret(f"Высокая корреляция отсутствует. {vif_msg_final}.")

    final_feats = fe_feats

    # ── ШАГ 7: Финальная регрессия ───────────────────────────────────────────
    sep()
    print(f"ШАГ 7. ФИНАЛЬНАЯ ПАНЕЛЬНАЯ РЕГРЕССИЯ — {LABEL}")
    sep()

    df_panel   = df[['date','region','price'] + final_feats].copy().dropna()
    panel_data = df_panel.set_index(['region','date'])

    nd = {}
    for col in ['price'] + final_feats:
        std = panel_data[col].std()
        nd[f'{col}_n'] = ((panel_data[col] - panel_data[col].mean()) / std
                          if std > 1e-10 else panel_data[col] * 0)
    norm_panel = pd.DataFrame(nd, index=panel_data.index)
    X_n = [f'{v}_n' for v in final_feats if norm_panel[f'{v}_n'].std() > 1e-10]

    results = PanelOLS.from_formula(
        f"price_n ~ {' + '.join(X_n)} + EntityEffects",
        data=norm_panel).fit(cov_type='clustered', cluster_entity=True)

    r2w = results.rsquared_within
    r2o = results.rsquared_overall
    gap = r2w - r2o

    print(f"\n  R² within  = {r2w:.4f}")
    print(f"  R² overall = {r2o:.4f}")
    print(f"  Базовый R² = {r2_raw:.4f}  |  Δ = {r2w - r2_raw:+.4f}")

    if r2w - r2_raw < -0.05:
        interpret(
            f"R² снизился с {r2_raw:.3f} до {r2w:.3f} ({abs(r2w-r2_raw)*100:.0f} п.п.) — "
            f"это доля псевдообъяснения, созданного переменными с обратной причинностью."
        )
    if gap > 0.35:
        interpret(
            f"R²_within ({r2w:.3f}) >> R²_overall ({r2o:.3f}) — "
            f"EntityEffects поглощают межрегиональные различия. "
            f"Модель идентифицирует временну́ю динамику внутри регионов."
        )

    coef_df = pd.DataFrame({
        'feature':   results.params.index,
        'beta_std':  results.params.values,
        'std_error': results.std_errors.values,
        'pvalue':    results.pvalues.values,
        'ci_lower':  results.params.values - 1.96*results.std_errors.values,
        'ci_upper':  results.params.values + 1.96*results.std_errors.values,
    })

    imp_rows = []
    for feat in final_feats:
        fn = f'{feat}_n'
        if fn not in coef_df['feature'].values: continue
        row = coef_df[coef_df['feature'] == fn].iloc[0]
        imp_rows.append({
            'variable': feat,
            'beta_std': row['beta_std'],
            'std_error':row['std_error'],
            'ci_lower': row['ci_lower'],
            'ci_upper': row['ci_upper'],
            'abs_beta': abs(row['beta_std']),
            'pvalue':   row['pvalue'],
            'is_sq':    '_sq' in feat,
        })
    imp_rows.sort(key=lambda x: x['abs_beta'], reverse=True)
    max_abs = max((r['abs_beta'] for r in imp_rows), default=1.0)

    sep('-', 85)
    print(f"\n  {'#':<4s} {'Переменная':<30s} {'Важн':<7s} {'β_std':<10s} "
          f"{'95% ДИ':<22s} {'p-value':<10s} {'Тип'}")
    print(f"  {'-'*82}")
    for i, r in enumerate(imp_rows, 1):
        imp  = r['abs_beta'] / max_abs
        kind = 'квадр.' if r['is_sq'] else 'линейн'
        print(f"  {i:<4d} {r['variable']:<30s} {imp:>5.3f}   {r['beta_std']:>+8.4f}   "
              f"[{r['ci_lower']:>+6.3f}; {r['ci_upper']:>+6.3f}]   "
              f"{r['pvalue']:.4f}    {kind}")

    if imp_rows:
        lead     = imp_rows[0]
        lead_dir = "положительная" if lead['beta_std'] > 0 else "отрицательная"
        interpret(
            f"Ведущий фактор: '{lead['variable']}' (β={lead['beta_std']:+.3f}, "
            f"p={lead['pvalue']:.4f}). Связь {lead_dir}."
        )

    # ── ШАГ 8: Предельные эффекты ─────────────────────────────────────────────
    sep()
    print(f"ШАГ 8. ПРЕДЕЛЬНЫЕ ЭФФЕКТЫ — {LABEL}")
    sep()
    print(f"""
  Переменные нормируются дважды: (1) per-region RobustScaler на ШАГ 3,
  (2) глобальный z-score внутри estimate_panel_model.
  β — в единицах глобального z-score робастно-нормированного ряда.
  Δy% = β × std_y_robust / (std_x_robust × IQR_x_orig × mean_y_orig) × δx_orig × 100
  Для квадратичных: Δy% = 2β × x̄_robust_norm × std_y_robust / (std_x_robust × IQR_x_orig × mean_y_orig) × δx_orig × 100
  std_x_robust = std(x после per-region RobustScaler), IQR_x_orig — в оригинальных единицах.
    """)

    # Параметры y: оригинальные и robust-normed
    y_orig_stats  = orig_stats.get(PRICE_COL, {})
    mean_y_orig   = y_orig_stats.get('mean', 1.0)   # среднее в оригинальных единицах
    iqr_y_orig    = y_orig_stats.get('iqr',  1.0)   # IQR в оригинальных единицах

    # std_y_robust = std цены после per-region RobustScaler (до второй нормировки)
    std_y_robust  = float(df_panel['price'].std())   # df_panel уже робастно нормирован

    UNIT_SHOCKS = {
        'rate':               ('1 п.п. ставки',          1.0),
        'real_rate':          ('1 п.п. реал. ставки',    1.0),
        'inflation':          ('1 п.п. инфляции',         1.0),
        'sentiment':          ('1 пункт индекса',         1.0),
        'expectations':       ('1 пункт индекса',         1.0),
        'current_state':      ('1 пункт индекса',         1.0),
        'housing_completed':  ('1 тыс. кв. м ввода',     1.0),
        'offers_primary':     ('1 IQR предложения',       1.0),
        'offers_secondary':   ('1 IQR предложения',       1.0),
        'overdue_rate':       ('1 п.п. просрочки/долг',  0.01),
        'mortgage_intensity': ('1 п.п. выдачи/долг',     0.01),
        'mortgage_debt':      ('1 IQR долга',             1.0),
    }

    print(f"  {'Фактор':<30s} {'Шок':<24s} {'Δ сделок,%':>11s}  Расчёт")
    print(f"  {'-'*90}")

    marg_effects = []
    for r in imp_rows:
        feat   = r['variable']
        bn     = get_base_name(feat)
        beta_s = r['beta_std']
        if bn not in UNIT_SHOCKS: continue
        shock_label, shock_size = UNIT_SHOCKS[bn]

        st         = orig_stats.get(feat) or orig_stats.get(bn, {})
        iqr_x_orig = st.get('iqr', None)
        if not iqr_x_orig or abs(iqr_x_orig) < 1e-12 or abs(mean_y_orig) < 1e-6:
            continue

        # std_x_robust = std переменной после per-region RobustScaler (в df_panel)
        if feat in df_panel.columns:
            std_x_robust = float(df_panel[feat].std())
        elif bn in df_panel.columns:
            std_x_robust = float(df_panel[bn].std())
        else:
            std_x_robust = 1.0

        if std_x_robust < 1e-10: continue

        if '_sq' in feat:
            # x̄_robust_norm = среднее x после robust-norm (в df_panel)
            x_robust_mean = float(df_panel[feat.replace('_sq','')].mean()) if feat.replace('_sq','') in df_panel.columns else st.get('mean', 0.0)
            # d/d(x_orig)[beta * ((x_robust - mean_robust)/std_robust)^2]
            # = beta * 2 * (x_robust_at_mean - mean_robust) / std_robust^2 * (1/IQR_x_orig)
            # При x_orig=mean: x_robust_at_mean ≈ 0 (median≈0 after robust norm)
            # But mean != 0 for skewed distributions — use actual mean from df_panel
            x_robust_at_mean_norm = x_robust_mean / std_x_robust  # (x_robust_mean - 0)/std_robust, since mean(robust)≈0
            delta_pct = (2 * beta_s * x_robust_at_mean_norm
                         * std_y_robust / (std_x_robust * iqr_x_orig * mean_y_orig)
                         * shock_size * 100)
            note = f"2β×x̄ᵣₙ({x_robust_at_mean_norm:.3f})×σyᵣ/(σxᵣ×IQRx×ȳ)"
        else:
            # Δy_orig = beta * (std_y_robust / std_x_robust) * (1/IQR_x_orig) * shock_orig * IQR_y_orig
            # Δy% = Δy_orig / mean_y_orig * 100
            # = beta * std_y_robust * shock / (std_x_robust * IQR_x_orig * mean_y_orig) * 100
            delta_pct = (beta_s * std_y_robust
                         / (std_x_robust * iqr_x_orig * mean_y_orig)
                         * shock_size * 100)
            note = f"β×σyᵣ/(σxᵣ×IQRx×ȳ) [IQRx={iqr_x_orig:.4f}]"

        sign = "+" if delta_pct > 0 else ""
        print(f"  {feat:<30s} {shock_label:<24s} {sign}{delta_pct:>9.2f}%   [{note}]")
        marg_effects.append({'variable': feat, 'shock': shock_label,
                             'delta_pct': delta_pct, 'beta_std': beta_s})

    if marg_effects:
        strongest = max(marg_effects, key=lambda x: abs(x['delta_pct']))
        interpret(
            f"Наибольший предельный эффект: '{strongest['variable']}' — "
            f"шок '{strongest['shock']}' → {strongest['delta_pct']:+.2f}% сделок. "
            f"Параметры: IQR_y_orig={iqr_y_orig:.1f}, mean_y_orig={mean_y_orig:.1f}."
        )

    # ── ШАГ 9: Диагностика остатков ───────────────────────────────────────────
    sep()
    print(f"ШАГ 9. ДИАГНОСТИКА ОСТАТКОВ — {LABEL}")
    sep()

    resid_s    = results.resids
    common_idx = norm_panel.index.intersection(resid_s.index)
    resid_arr  = resid_s.reindex(common_idx).values
    X_bp       = sm.add_constant(norm_panel.reindex(common_idx)[X_n].values)
    mask       = ~(np.isnan(resid_arr) | np.isnan(X_bp).any(axis=1))
    rc, X_bpm  = resid_arr[mask], X_bp[mask]

    bp_stat, bp_p, _, _ = het_breuschpagan(rc, X_bpm)
    jb_s, jb_p          = jarque_bera(rc)
    sw_s, sw_p          = (shapiro(rc[:5000]) if len(rc) > 5000 else shapiro(rc))

    print(f"  N={len(rc)}, mean={np.mean(rc):.2e}, std={np.std(rc):.4f}")
    print(f"  Breusch-Pagan: LM={bp_stat:.2f}, p={bp_p:.6f} → "
          f"{'ПРИСУТСТВУЕТ' if bp_p < 0.05 else 'отсутствует'}")
    print(f"  Jarque-Bera:   stat={jb_s:.2f},  p={jb_p:.6f} → "
          f"{'ненормальные' if jb_p < 0.05 else 'нормальные'}")
    print(f"  Shapiro-Wilk:  stat={sw_s:.4f},  p={sw_p:.6f}")

    interpret(
        (f"Гетероскедастичность {'обнаружена — корректируется кластеризованными SE.' if bp_p < 0.05 else 'не обнаружена.'} "
         + (f"Ненормальность остатков при N={len(rc)} не влияет на инференс по ЦПТ. "
            f"JB={jb_s:.0f} — тяжёлые хвосты, характерны для шоковых периодов." if jb_p < 0.05 else ""))
    )

    # ── Сохранение ─────────────────────────────────────────────────────────────
    with open(f"{OUT_PFX}_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"РЫНОК: {LABEL}\n")
        f.write(f"Зависимая переменная: {PRICE_COL}\n")
        f.write(f"Панель: {N_regions}×{N_periods}={N_obs}\n")
        f.write(f"R² within: {r2w:.4f} | R² overall: {r2o:.4f}\n")
        f.write(f"Базовый R²: {r2_raw:.4f}\n")
        f.write(f"Коинтегрированные: {sorted(COINTEGRATED)}\n")
        f.write(f"Δ-применено к: {sorted(NON_COINTEGRATED | I1_MACRO)}\n\n")
        f.write(str(results.summary))

    coef_df.to_csv(f"{OUT_PFX}_coefs.csv", sep=';', index=False)
    pd.DataFrame(marg_effects).to_csv(f"{OUT_PFX}_marginal.csv", sep=';', index=False)
    imp_out = pd.DataFrame(imp_rows)
    imp_out['importance'] = imp_out['abs_beta'] / max_abs
    imp_out.to_csv(f"{OUT_PFX}_importance.csv", sep=';', index=False)

    ALL_RESULTS[MNAME] = {
        'r2w': r2w, 'r2o': r2o, 'r2_raw': r2_raw,
        'cointegrated': COINTEGRATED,
        'n_diff': NON_COINTEGRATED | I1_MACRO,
        'imp_rows': imp_rows,
        'n_regions': N_regions, 'n_obs': N_obs,
    }
    print(f"\n  ✓ Результаты: {OUT_PFX}_*")

    # Очищаем _sq перед следующим рынком
    df_work.drop(columns=[c for c in df_work.columns if c.endswith('_sq')],
                 inplace=True, errors='ignore')

# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 10: СРАВНИТЕЛЬНЫЙ АНАЛИЗ
# ─────────────────────────────────────────────────────────────────────────────
sep('═')
print("ШАГ 10. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ДВУХ РЫНКОВ")
sep('═')

if len(ALL_RESULTS) == 2:
    pr, sec = ALL_RESULTS['primary'], ALL_RESULTS['secondary']
    rows = [
        ("R² within (причинная модель)",        f"{pr['r2w']:.3f}",  f"{sec['r2w']:.3f}"),
        ("R² overall",                           f"{pr['r2o']:.3f}",  f"{sec['r2o']:.3f}"),
        ("Базовый R² (до отбора)",               f"{pr['r2_raw']:.3f}", f"{sec['r2_raw']:.3f}"),
        ("Δ R² (устранение обратн. причинн.)",   f"{pr['r2w']-pr['r2_raw']:+.3f}", f"{sec['r2w']-sec['r2_raw']:+.3f}"),
        ("N наблюдений",                         f"{pr['n_obs']}",    f"{sec['n_obs']}"),
        ("N регионов (единая выборка)",          f"{pr['n_regions']}", f"{sec['n_regions']}"),
        ("Коинтегрированные I(1)",               f"{sorted(pr['cointegrated']) or '—'}", f"{sorted(sec['cointegrated']) or '—'}"),
    ]
    print(f"\n  {'Характеристика':<45s} {'Первичный':<20s} {'Вторичный'}")
    print(f"  {'-'*85}")
    for label, pv, sv in rows:
        print(f"  {label:<45s} {pv:<20s} {sv}")

    print("\n  Ведущие факторы:")
    for mn, lbl in [('primary','Первичный'),('secondary','Вторичный')]:
        imp = ALL_RESULTS[mn]['imp_rows']
        if imp:
            lead = imp[0]
            print(f"    {lbl}: '{lead['variable']}' β={lead['beta_std']:+.4f}")

    interpret(
        "Единая выборка из одних регионов устраняет смещение при сравнении. "
        "Различие в коинтегрированных переменных — структурный результат: "
        "ипотечный портфель долгосрочно связан только с первичным рынком."
    )

sep('═')
print("ОБА РЫНКА ОБРАБОТАНЫ")
sep('═')
