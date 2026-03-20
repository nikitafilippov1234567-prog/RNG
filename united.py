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
     # Исключаем сделки и предложение вторичного рынка: они не влияют на первичный рынок,
     # а определяются теми же факторами (ставка, доходы, настроения) — common cause bias.
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
# ФИКС ЕСЛИ ОТКРЫТЬ РУ ЭКСЕЛЬ (!!!)
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
                # МЕС.ДД -> номер_месяца + дробная/100
                return num + rest_f / 100
            else:
                # ДД.МЕС -> целая + номер_месяца/10
                return rest_f + num / 10
    return np.nan

raw = pd.read_csv(SOURCE_FILE, sep=';')
raw['date'] = pd.to_datetime(raw['date'], dayfirst=True)  # DD.MM.YYYY

# Применяем декодирование ко всем нечисловым колонкам (кроме date)
print("Декодирование Excel-дат в числа...")
for col in raw.columns:
    if col == 'date':
        continue
    # Проверяем есть ли нечисловые значения
    non_numeric = raw[col].apply(
        lambda x: pd.isna(pd.to_numeric(str(x).replace(',','.'), errors='coerce'))
        if pd.notna(x) else False
    ).sum()
    if non_numeric > 0:
        raw[col] = raw[col].apply(fix_excel_number)
        print(f"  Исправлено '{col}': {non_numeric} значений")
raw = raw.loc[:, ~raw.columns.duplicated()]

# Удаляем колонки с идентичным содержимым
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

# Макро-датафрейм
macro_df = pd.DataFrame({'date': raw['date']})
for substr, varname in MACRO_MAP.items():
    matched = [c for c in raw.columns if substr in c]
    if matched:
        macro_df[varname] = pd.to_numeric(raw[matched[0]], errors='coerce').values
        print(f"Макро '{varname}' ← '{matched[0]}'")

def prefix_to_long(raw, prefix, varname):
    """Wide → long для одного префикса."""
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

# Собираем панель через outer join
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

# Макро: merge по дате — даты должны совпадать точно
# Диагностика совпадения дат
panel_dates = set(base_panel['date'].dt.normalize().unique())
macro_dates = set(macro_df['date'].dt.normalize().unique())
missing_macro = panel_dates - macro_dates
if missing_macro:
    print(f"⚠ Даты в панели без макро-данных: {sorted(missing_macro)}")
    print(f"  Нормализуем даты до месяца для корректного merge...")
    base_panel['date'] = base_panel['date'].dt.normalize()
    macro_df['date']   = macro_df['date'].dt.normalize()

base_panel = base_panel.merge(macro_df, on='date', how='left')
base_panel = base_panel[~base_panel['region'].isin(EXCLUDE_REGIONS)]
# Фильтруем агрегаты по паттерну
for pattern in EXCLUDE_PATTERNS:
    base_panel = base_panel[~base_panel['region'].str.contains(pattern, case=False, na=False)]
base_panel = base_panel.sort_values(['region', 'date']).reset_index(drop=True)

# Диагностика пропусков после сборки
print(f"\n✓ Базовая панель: {len(base_panel)} наблюдений | "
      f"{base_panel['region'].nunique()} регионов")

# Диагностика: уникальные даты
all_dates = sorted(base_panel['date'].unique())
print(f"   Уникальных дат: {len(all_dates)}")
print(f"   Первые 5 дат:  {[str(d)[:10] for d in all_dates[:5]]}")
print(f"   Последние 5 дат: {[str(d)[:10] for d in all_dates[-5:]]}")

# Диагностика: пропуски по колонкам
print(f"\n   Пропуски по колонкам:")
for col in base_panel.columns:
    if col in ['date', 'region']:
        continue
    n_na = base_panel[col].isna().sum()
    total = len(base_panel)
    # Для макро: считаем уникальные даты с NaN
    if col in list(MACRO_MAP.values()):
        dates_with_na = base_panel[base_panel[col].isna()]['date'].nunique()
        dates_ok      = base_panel[base_panel[col].notna()]['date'].nunique()
        print(f"      {col:<35s}: {n_na:4d} NaN | дат с данными: {dates_ok} | дат без: {dates_with_na}")
    else:
        regions_with_na = base_panel[base_panel[col].isna()]['region'].nunique()
        print(f"      {col:<35s}: {n_na:4d} NaN ({n_na/total*100:.0f}%) | регионов с NaN: {regions_with_na}")

# Диагностика: сколько регионов имеют полные данные
regional_only_cols = [c for c in base_panel.columns
                      if c not in ['date', 'region'] + list(MACRO_MAP.values())]
complete_regions = (
    base_panel.groupby('region')[regional_only_cols]
    .apply(lambda x: x.notna().all().all())
)
print(f"\n   Регионов с полными региональными данными: {complete_regions.sum()} из {complete_regions.shape[0]}")
print(f"   Примеры полных регионов: {list(complete_regions[complete_regions].index[:5])}")

# Диагностика: даты макро-данных vs региональных
macro_dates_ok = base_panel[base_panel['rate'].notna()]['date'].unique()
print(f"\n   Дат с данными по 'rate': {len(macro_dates_ok)}")
print(f"   Пример дат rate: {[str(d)[:10] for d in sorted(macro_dates_ok)[:5]]}")
region_dates_ok = base_panel[base_panel['offers_secondary'].notna()]['date'].unique()
print(f"   Дат с данными по 'offers_secondary': {len(region_dates_ok)}")
print(f"   Пример дат offers: {[str(d)[:10] for d in sorted(region_dates_ok)[:5]]}")


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
# ОСНОВНОЙ ЦИКЛ: ПЕРВИЧНЫЙ И ВТОРИЧНЫЙ РЫНОК
# ============================================================================
for target in TARGETS:
    PRICE_COL     = target['price_col']
    EXCLUDE_REGS  = target['exclude_regressors']
    MARKET_NAME   = target['name']
    OUT_PREFIX    = f"{OUTPUT_FOLDER}/{MARKET_NAME}"

    print("\n" + "="*100)
    print(f"МОДУЛЬ 4 — РЫНОК: {MARKET_NAME.upper()} (цель: {PRICE_COL})")
    print("="*100)

    # ------------------------------------------------------------------
    # 1. Подготовка датафрейма для данного рынка
    # ------------------------------------------------------------------
    print(f"\n[1] Подготовка данных для {MARKET_NAME}...")

    df = base_panel.copy()
    df = df.rename(columns={PRICE_COL: 'price'})

    # Убираем сделки/предложение другого рынка: common cause bias —
    # оба рынка определяются одними факторами, а не влияют друг на друга
    drop_cols = [c for c in df.columns if get_base_name(c) in EXCLUDE_REGS]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"   Исключены cross-market переменные: {drop_cols}")

    # ← Фильтруем регионы с пропусками ТОЛЬКО по региональным переменным и price
    # Макро-переменные одинаковы для всех регионов — их пропуски не зависят от региона
    regional_check_cols = ['price'] + [
        c for c in df.columns
        if c not in ['date', 'region', 'price']
        and get_base_name(c) not in MACRO_VARS
    ]
    # Оставляем только колонки которые реально есть в df
    regional_check_cols = [c for c in regional_check_cols if c in df.columns]

    region_has_na = df.groupby('region')[regional_check_cols].apply(
        lambda x: x.isnull().any().any()
    )
    bad_regions = region_has_na[region_has_na].index.tolist()
    print(f"   Всего регионов до фильтрации: {df['region'].nunique()}")
    print(f"   Исключено регионов с пропусками: {len(bad_regions)}")
    # Показываем какие колонки дают пропуски
    for col in regional_check_cols:
        n_bad = df.groupby('region')[col].apply(lambda x: x.isnull().any()).sum()
        if n_bad > 0:
            print(f"      '{col}': пропуски в {n_bad} регионах")
    df = df[~df['region'].isin(bad_regions)]
    print(f"   Регионов после фильтрации: {df['region'].nunique()}")

    # Макро-переменные: пропуск в одну дату = пропуск для ВСЕХ регионов в эту дату
    # Исключаем целые временные периоды (не строки!), где макро отсутствует
    macro_base_cols = [c for c in df.columns
                       if c not in ['date', 'region', 'price']
                       and get_base_name(c) in MACRO_VARS]
    if macro_base_cols:
        # Смотрим заполненность каждой макро-переменной по датам
        date_macro = df[['date'] + macro_base_cols].drop_duplicates('date').sort_values('date')
        print(f"   Макро колонки ({len(macro_base_cols)}): {macro_base_cols}")
        for col in macro_base_cols:
            n_ok  = date_macro[col].notna().sum()
            n_bad = date_macro[col].isna().sum()
            print(f"      '{col}': {n_ok} дат с данными, {n_bad} без данных")
        bad_dates = date_macro[date_macro[macro_base_cols].isnull().any(axis=1)]['date']
        if len(bad_dates) > 0:
            print(f"   Исключено периодов без макро: {bad_dates.nunique()} уникальных дат")
            print(f"   Даты: {[str(d)[:10] for d in sorted(bad_dates.unique())]}")
            df = df[~df['date'].isin(bad_dates)]
        else:
            print(f"   Все макро-данные присутствуют для {len(date_macro)} дат")

    df = df.sort_values(['region', 'date']).reset_index(drop=True)
    print(f"✓ {len(df)} наблюдений | {df['region'].nunique()} регионов | "
          f"{df['date'].nunique()} периодов")

    # ------------------------------------------------------------------
    # 2. Нормализация + тест стационарности + дифференцирование I(1)
    # ------------------------------------------------------------------
    print(f"\n[2] Нормализация ({MARKET_NAME})...")

    exclude_from_norm = {'date', 'region', 'price'}
    all_vars          = [c for c in df.columns if c not in exclude_from_norm]
    macro_cols        = [v for v in all_vars if get_base_name(v) in MACRO_VARS]
    region_cols       = [v for v in all_vars if get_base_name(v) not in MACRO_VARS]

    # Региональные — per region RobustScaler
    for col in region_cols:
        g      = df.groupby('region')[col]
        median = g.transform('median')
        iqr    = g.transform(
            lambda x: x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
        df[col] = (df[col] - median) / iqr
        df[col] = df[col].fillna(0)

    # Макро — глобально RobustScaler
    for col in macro_cols:
        vals    = df[col].values.reshape(-1, 1)
        df[col] = RobustScaler().fit_transform(vals).flatten()
        df[col] = df[col].fillna(0)

    # ── Тест стационарности макро-переменных (ADF + KPSS) ────────────
    # Макро одинаковы для всех регионов → тестируем на уникальных датах
    # adfuller и kpss_test импортированы глобально

    macro_unique = df[['date'] + macro_cols].drop_duplicates('date').sort_values('date')

    I1_VARS = set()   # нестационарные I(1) — нужно дифференцировать
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
                order = 'I(1) → Δ'
                I1_VARS.add(col)
            else:
                order = 'I(2)+'
        else:
            order = 'I(0) ✓'
        kpss_str = f'{kpss_p:.4f}' if not np.isnan(kpss_p) else 'n/a'
        print(f"   {col:<22s} {adf_p:<10.4f} {kpss_str:<10s} {order}")

    # Дифференцируем I(1) макро
    if I1_VARS:
        print(f"\n   Дифференцируем I(1) макро: {sorted(I1_VARS)}")
        for col in I1_VARS:
            df[col] = df.groupby('region')[col].diff()
            df[col] = df[col].fillna(0)
        print(f"   NaN от diff заполнены 0 (первое наблюдение каждого региона)")
    else:
        print(f"\n   Все макро-переменные стационарны I(0)")

    # ── IPS тест (Im, Pesaran, Shin 2003) для региональных переменных ──────
    # IPS H0: все ряды I(1). W = sqrt(N)*(t_bar - E[t]) / sqrt(Var[t]) ~ N(0,1)
    # Таблица моментов IPS (2003, Table 3, regression без тренда)
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

    from scipy.stats import norm as _norm

    print(f"\n   Im-Pesaran-Shin (IPS, 2003) — панельный тест единичного корня:")
    print(f"   H0: все панельные ряды I(1)  |  IPS применим только при t_bar < E[t]")
    print(f"   Взрывные ряды (t_bar >> 0) диагностируются отдельно.")
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

        # Взрывной ряд: t_bar >> 0 означает AR(1) с ρ>1 (структурный сдвиг или рост)
        # IPS не применим — выводим отдельный вердикт
        if t_bar > 2.0:
            verdict = 'ВЗРЫВНОЙ — структурный сдвиг (IPS неприменим)'
            W, p_ips = np.nan, np.nan
            print(f"   {col:<28s} {'n/a':<10s} {'n/a':<10s} {t_bar:<8.3f} {verdict}")
            continue

        W     = np.sqrt(N_i) * (t_bar - e_t) / np.sqrt(v_t)
        p_ips = float(_norm.cdf(W))
        verdict = 'I(0) — H0 отвергнута' if p_ips < 0.05 else 'I(1) — H0 не отвергнута'
        print(f"   {col:<28s} {W:<10.3f} {p_ips:<10.4f} {t_bar:<8.3f} {verdict}")

    print(f"\n   EntityEffects корректно работает с I(1) рядами (абсорбирует тренд).")
    print(f"   Взрывные ряды: структурный сдвиг (напр. резкий рост просрочки в 2024).")
    print(f"   Дифференцирование региональных НЕ применяется.")

    # ── Составные показатели (feature engineering) ───────────────────────
    # Строим ИЗ УРОВНЕЙ (до нормализации) чтобы отношение было корректным.
    # Нормализуем per region через RobustScaler после вычисления.
    print(f"\n   Составные показатели (feature engineering):")

    def _safe_ratio(num_col, den_col, df):
        """
        Отношение num/den, нормализованное per region через RobustScaler.
        Защита от нуля через replace(0, NaN).
        Важно: нормализуем само отношение (не компоненты) чтобы _sq был корректен.
        Результат ≈ z-score относительно медианы региона.
        """
        num = df[num_col]
        den = df[den_col].replace(0, np.nan)
        ratio = (num / den).replace([np.inf, -np.inf], np.nan)
        # Winsorize на уровне 1%–99% per region чтобы выбросы не искажали нормализацию
        def _winsorize(x):
            lo, hi = x.quantile(0.01), x.quantile(0.99)
            return x.clip(lo, hi)
        ratio = ratio.groupby(df['region']).transform(_winsorize)
        # Per-region RobustScaler
        med = ratio.groupby(df['region']).transform('median')
        iqr = ratio.groupby(df['region']).transform(
            lambda x: x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
        return ((ratio - med) / iqr).fillna(0)

    created_fe = []

    # absorption = deals / offers (скорость реализации предложения)
    # Строим из RAW данных base_panel чтобы не зависеть от уже нормализованных колонок
    raw_cols = {}
    for col in ['deals_secondary', 'deals_primary', 'offers_secondary',
                'offers_primary', 'housing_loans', 'mortgage_debt', 'mortgage_overdue']:
        # Ищем в base_panel (до нормализации) — но у нас уже нормализованный df.
        # Вместо этого используем знак: нормализованные ряды сохраняют порядок,
        # отношение нормализованных значений интерпретируемо как z-score ratio.
        pass

    # Используем уже нормализованные значения (RobustScaler сохраняет монотонность)
    if 'deals_secondary' in df.columns and 'offers_secondary' in df.columns:
        df['absorption_secondary'] = _safe_ratio('deals_secondary', 'offers_secondary', df)
        created_fe.append('absorption_secondary')
        print(f"   absorption_secondary = deals_secondary / offers_secondary")

    if 'deals_primary' in df.columns and 'offers_primary' in df.columns:
        df['absorption_primary'] = _safe_ratio('deals_primary', 'offers_primary', df)
        created_fe.append('absorption_primary')
        print(f"   absorption_primary   = deals_primary / offers_primary")

    # mortgage_intensity и overdue_rate: знаменатель = mortgage_debt (остаток, стационарен в уровнях)
    # housing_loans и mortgage_overdue стационарны (I(0) по ADF) → деление корректно
    if 'housing_loans' in df.columns and 'mortgage_debt' in df.columns:
        df['mortgage_intensity'] = _safe_ratio('housing_loans', 'mortgage_debt', df)
        created_fe.append('mortgage_intensity')
        print(f"   mortgage_intensity   = housing_loans / mortgage_debt")

    if 'mortgage_overdue' in df.columns and 'mortgage_debt' in df.columns:
        df['overdue_rate'] = _safe_ratio('mortgage_overdue', 'mortgage_debt', df)
        created_fe.append('overdue_rate')
        print(f"   overdue_rate         = mortgage_overdue / mortgage_debt")

    print(f"   Создано: {len(created_fe)}: {created_fe}")

    # Базовые переменные: исходные переменные ОСТАЮТСЯ вместе с составными показателями.
    # Составные (absorption, mortgage_intensity, overdue_rate) дополняют модель —
    # они несут информацию об ОТНОШЕНИЯХ которую уровни не содержат.
    # Коллинеарность между уровнями и составными устраняется на шаге мультиколлинеарности
    # через приоритеты (составные = приоритет 1, уровни = 2-4).
    ratio_kw = ['ratio', 'real_rate', 'credit_availability',
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

    # ── Теоретически обоснованные нелинейности ──────────────────────────────
    # Квадратичные термы добавляются только для переменных с явным экономическим
    # обоснованием нелинейного эффекта. Статистика подтверждает теорию, но не заменяет её.
    #
    # ОБОСНОВАНИЕ КВАДРАТИЧНЫХ ФОРМ:
    #   rate_sq: Кривая доступности ипотеки нелинейна — при ставке выше ~15%
    #            ежемесячный платёж превышает типичный доход, рынок обваливается нелинейно.
    #            Теоретически: Poterba (1984), user cost of housing model.
    #   inflation_sq: U-образный эффект Пигу — при умеренной инфляции жильё защитный актив,
    #                 при высокой (>8%) покупательская способность коллапсирует быстрее роста цен.
    #                 Теоретически: Brunnermeier & Julliard (2008), money illusion in housing.
    #   sentiment_sq / expectations_sq: Нелинейность психологического порога — настроения
    #                 влияют слабо вблизи нейтрального уровня, сильно — при экстремальных значениях.
    #                 Теоретически: Shiller (2015), narrative economics, threshold effects.
    #   mortgage_overdue_sq: Ускорение вынужденных продаж — небольшая просрочка не создаёт
    #                 давления, высокая вызывает лавину дистрессовых сделок.
    #
    # Переменные БЕЗ теоретического обоснования нелинейности тестируются,
    # но принимаются только при p < 0.01 И ΔR² >= 0.005 (более жёсткий критерий). ПОМЕНЯТЬ (!!!) если данные странные
    THEORY_NONLINEAR = {'rate', 'inflation', 'sentiment', 'expectations', 'mortgage_overdue'}
    MIN_SQ_DR2_THEORY = 0.003   # порог для теоретически обоснованных
    MIN_SQ_DR2_EMPIRIC = 0.005  # более жёсткий порог для остальных

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
    # 4. Финальные переменные (без лагов)
    # ------------------------------------------------------------------
    # Лаги исключены: для нормализованных per-region рядов с EntityEffects
    # оператор within уже устраняет автокорреляцию. Дифференцированные
    # макро-переменные (Δrate, Δsentiment) действуют contemporaneously.
    # Использование raw/sq форм без лагов обеспечивает максимальную
    # прозрачность и избегает data-snooping при подборе лагов.

    chosen_bases = []
    for var in base_vars:
        if var in significant_squares:
            chosen_bases.append(f'{var}_sq')
        else:
            chosen_bases.append(var)

    print(f"\n[4] Финальные переменные ({MARKET_NAME}): {len(chosen_bases)} "
          f"(squared: {len(significant_squares)}, без лагов)")

    final_vars_for_pca = list(chosen_bases)

    # Аудит форм
    print(f"\n   АУДИТ форм переменных:")
    for v in final_vars_for_pca:
        form = 'sq' if '_sq' in v else 'raw'
        print(f"   ✓ '{get_base_name(v)}': {form} → {v}")

    baseline_final = estimate_panel_model(df, 'price', final_vars_for_pca, "Baseline_final")
    baseline_r2 = baseline_final['r2_within'] if baseline_final else 0.0
    print(f"\n   Baseline на финальных переменных R² = {baseline_r2:.4f}")



    # ------------------------------------------------------------------
    # 6. FEATURE ENGINEERING (замена PCA)
    # ------------------------------------------------------------------
    # Логика: вместо слепого PCA создаём содержательные индексы по экономическим группам.
    # Только составные показатели (feature engineering) — без индексов.
    # Индексы (idx_*) некорректны экономически: объединяют разнородные переменные.
    # Составные показатели (отношения) — экономически содержательны и интерпретируемы.
    # ------------------------------------------------------------------
    print(f"\n[6] FEATURE ENGINEERING — {MARKET_NAME}")
    print(f"   Входных переменных: {final_vars_for_pca}")
    print(f"   Составные показатели уже созданы на шаге нормализации.")
    print(f"   Индексы (idx_*) не используются: экономически некорректное объединение.")
    print(f"   Все переменные входят как есть — прозрачная интерпретация каждого коэффициента.")

    fe_features = list(final_vars_for_pca)
    fe_info     = [{'variable': v, 'index': None, 'corr_with_index': 1.0,
                    'sign': 1, 'is_quadratic': '_sq' in v} for v in fe_features]

    print(f"\n {len(final_vars_for_pca)} признаков после FE: {fe_features}")

    # ── Устранение мультиколлинеарности ──────────────────────────────────
    # Критерий выбора при коллинеарной паре — приоритет по типу переменной:
    #   1 (высший) — рыночные региональные (offers, deals, housing_*)
    #   2           — ипотечные потоки (housing_loans, mortgage_overdue)
    #   3           — ипотечные остатки (mortgage_debt) — накопленный, менее динамичен
    #   4           — сводные индексы (idx_*)
    #   5 (низший)  — макро (inflation, rate, sentiment и их _sq/_lag)
    # Внутри одного приоритета — оставляем с большим |β_std| в полной модели
    # (считаем через однофакторный R² как proxy до финальной регрессии)
    # ПОМЕНЯТЬ (!!!) ПРИ ИЗМЕНЕНИИ CSV
    MULTICOLL_THRESHOLD = 0.7

    def _var_priority(v):
        bn = get_base_name(v)
        # Составные показатели (отношения) — наиболее интерпретируемы, без коллинеарности
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
        return 6   # макро: inflation, rate, sentiment, expectations, current_state

    def _single_r2(feat):
        res = estimate_panel_model(df, 'price', [feat])
        return res['r2_within'] if res else 0.0

    print(f"\n   Проверка мультиколлинеарности (порог |r|>{MULTICOLL_THRESHOLD}):")

    # ── Шаг 0: компонентная коллинеарность + spurious correlates ────────
    # COMPONENT_MAP: переменные которые убираются в пользу составного показателя.
    #
    # Логика включения/исключения:
    #   mortgage_overdue, mortgage_debt - следствие цены и ставки, убираем
    #   deals_secondary, deals_primary  - объём сделок = следствие цены, убираем
    #   offers_secondary, offers_primary - предложение реагирует на цену, убираем
    #   housing_loans - убираем: r=0.96 с mortgage_intensity (числитель примерно отношение при стаб. знаменателе).
    #                  mortgage_intensity предпочтительнее: нормирует на размер портфеля.
    #   overdue_rate - spurious: просрочка и восстановительный спрос совпали во времени
    #                  (2024: ставка 21% - и просрочка растёт, и спрос восстанавливается).
    #                  Сигнал ставки уже есть в rate/rate_sq, overdue_rate дублирует его
    #                  со случайным знаком. Убираем.
    COMPONENT_MAP = {
        'overdue_rate':        {'mortgage_overdue', 'mortgage_debt'},
        'mortgage_intensity':  {'mortgage_debt', 'housing_loans'},  # housing_loans ↔ mortgage_intensity r=0.96 (числитель=знаменатель)
        'absorption_secondary':{'deals_secondary', 'offers_secondary'},
        'absorption_primary':  {'deals_primary', 'offers_primary'},
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
        print(f" После удаления: {fe_features}")
    # ─────────────────────────────────────────────────────────────────────

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
                # Разные приоритеты: убираем с низшим (более высоким числом)
                weaker   = b if pa < pb else a
                stronger = a if pa < pb else b
                reason   = f"приоритет {min(pa,pb)} > {max(pa,pb)}"
            else:
                # Одинаковый приоритет: убираем с меньшим R²
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
        print(f"   ✓ После устранения: {fe_features}")

    # ── Убираем незначимые признаки (предварительный прогон) ─────────────
    print(f"\n   Предварительный прогон для отсева незначимых признаков (p>0.10):")
    pre_result = estimate_panel_model(df, 'price', fe_features, "pre_significance")
    if pre_result:
        pre_pvals = pre_result['results'].pvalues  # индекс вида 'varname_n'
        insig = []
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

        # ── Проверка доминирования перекрёстного рыночного регрессора ────
        # Если deals_* другого рынка имеет |β_std| > 2× следующего по важности,
        # выводим предупреждение: он поглощает объяснение у структурных факторов.
        # Только deals_* могут быть перекрёстными (уже исключены, но на всякий случай).
        # offers_* — предложение своего рынка, не перекрёстный регрессор.
        cross_bases = {'deals_primary', 'deals_secondary'}
        pre_params = pre_result['results'].params if pre_result else {}
        cross_betas = {}
        other_betas = {}
        for v in fe_features:
            vn = f'{v}_n'
            if vn in pre_params.index:
                b = abs(pre_params[vn])
                if get_base_name(v) in cross_bases:
                    cross_betas[v] = b
                else:
                    other_betas[v] = b
        if cross_betas and other_betas:
            max_cross = max(cross_betas.values())
            max_other = max(other_betas.values())
            if max_cross > 2 * max_other:
                dominant = max(cross_betas, key=cross_betas.get)
                print(f"\n   ℹ ДОМИНИРОВАНИЕ: '{dominant}' (|β|={max_cross:.4f}) >> прочих (max |β|={max_other:.4f})")
                print(f"   Перекрёстный регрессор поглощает ~{max_cross/(max_cross+max_other)*100:.0f}% объяснения.")
                print(f"   Интерпретация: сильная взаимосвязь первичного и вторичного рынков.")

    final_features = fe_features
    pca_info       = []
    independent    = fe_features

    # ------------------------------------------------------------------
    # 7. Финальная модель
    # ------------------------------------------------------------------
    print(f"\n[7] ФИНАЛЬНАЯ ПАНЕЛЬНАЯ РЕГРЕССИЯ — {MARKET_NAME.upper()}")

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
    print(f"   R² within:  {r2w:.4f}  (объяснённая within-вариация: динамика внутри регионов)")
    print(f"   R² overall: {r2o:.4f}  (межрегиональное обобщение)")
    gap = r2w - r2o
    if gap > 0.4:
        print(f"   ℹ R²_within >> R²_overall (gap={gap:.3f}): EntityEffects абсорбирует")
        print(f"     межрегиональные различия. Модель идентифицирует ВРЕМЕННУ́Ю динамику,")
        print(f"     не уровневые различия между регионами. Это ожидаемо для FE-панели.")
    improvement = r2w - baseline_r2
    print(f"   Baseline (до FE-отбора):  {baseline_r2:.4f}")
    print(f"   Прирост/потеря от FE:     ΔR²={improvement:+.4f} ({improvement*100:+.2f}%)")
    if improvement < -0.10:
        print(f"   ℹ Значительная потеря R² после FE — ожидаема если удалены переменные")
        print(f"     с обратной причинностью (deals_*, mortgage_debt, mortgage_overdue).")
        print(f"     Оставшийся R²={r2w:.4f} отражает ПРИЧИННЫЕ факторы влияния на цену.")
    elif improvement < -0.01:
        print(f" Потеря >1% R² — проверьте состав финальных признаков")

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

    # После FE все признаки независимы — β_std напрямую из нормализованной регрессии
    # Для индексов дополнительно показываем вклад составляющих через corr с индексом
    all_importance = []
    for feat in final_features:
        fn = f'{feat}_n'
        if fn not in pc_coefs['feature'].values:
            continue
        coef = pc_coefs.loc[pc_coefs['feature'] == fn, 'coefficient'].values[0]
        pval = pc_coefs.loc[pc_coefs['feature'] == fn, 'pvalue'].values[0]

        # Определяем тип признака
        is_index = feat.startswith('idx_')
        # Для индекса: раскладываем на составляющие
        constituents = [fi for fi in fe_info if fi['index'] == feat]

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
        orig = (row['beta_std'] * df[feat].std()
                if feat in df.columns else row['beta_std'])
        pv   = f"{row['pvalue']:.4f}"
        imp  = row['abs_beta_std'] / max_abs
        kind = 'Индекс' if row['is_index'] else ('Квадр.' if row['is_quadratic'] else 'Линейн.')
        print(f"{feat:<38s} {kind:<12s} {imp:>6.4f}   {row['beta_std']:>8.4f}   "
              f"{orig:>10.4f}   {pv}")
        # Если индекс — показываем составляющие с их корреляцией
        if row['is_index'] and row['constituents']:
            for fi in fe_info:
                if fi['index'] == feat:
                    cr = fi.get('corr_with_index', np.nan)
                    print(f"   {'└─ ' + fi['variable']:<35s} {'':12s} "
                          f"{'':8s} corr={cr:+.3f}")

    # Сохраняем
    imp_out = pd.DataFrame([{k: v for k, v in r.items() if k != 'constituents'}
                             for r in all_importance])
    imp_out['importance_norm'] = imp_out['abs_beta_std'] / max_abs
    imp_out.to_csv(f"{OUT_PREFIX}_Feature_Importance_FINAL.csv", sep=";", index=False)

    # ── Эффекты в реальных единицах ──────────────────────────────────────
    # Пересчёт β_std → изменение целевой переменной (сделок) в %
    # при изменении фактора на 1 единицу в оригинальных единицах.
    #
    # Формула: Δdeals% = β_std × (SD_price / mean_price) × (1 / SD_factor) × factor_unit × 100
    # где factor_unit = типичный шок фактора (1 п.п. для ставки, 1% для инфляции и т.д.)
    #
    # Для нелинейных (_sq): эффект оценивается в точке среднего значения фактора (x̄):
    # d(f(x²))/dx|_{x=x̄} = 2 × β_sq × x̄  (производная квадратичной функции)
    print(f"\n   Эффекты в реальных единицах (при типичном изменении фактора):")
    print(f"   {'Фактор':<35s} {'Шок':<20s} {'Δ сделок, %':>12s}  Расчёт")
    print(f"   {'-'*80}")

    # Получаем SD и среднее price из финального датасета
    price_raw = df_panel['price'] if 'df_panel' in dir() else df['price'].dropna()
    price_sd   = float(price_raw.std())
    price_mean = float(price_raw.mean())

    # Типичные шоки для интерпретации
    UNIT_SHOCKS = {
        'rate':              ('1 п.п. ставки',      1.0),
        'inflation':         ('1 п.п. инфляции',    1.0),
        'sentiment':         ('1 пункт индекса',    1.0),
        'expectations':      ('1 пункт индекса',    1.0),
        'current_state':     ('1 пункт индекса',    1.0),
        # housing_loans исключён (r=0.96 с mortgage_intensity)
        # 'housing_loans':   ('1 млрд руб. кредитов', 1.0),
        'housing_completed': ('1 тыс. кв. м ввода', 1.0),
        'offers_primary':    ('1% изм. предложения', 1.0),
        'offers_secondary':  ('1% изм. предложения', 1.0),
        'overdue_rate':      ('1 п.п. просрочки/долг', 1.0),
        'mortgage_intensity':('1 п.п. кредиты/долг', 1.0),
        'absorption_secondary':('1 п.п. поглощения', 1.0),
        'absorption_primary':  ('1 п.п. поглощения', 1.0),
    }

    for row in all_importance:
        feat   = row['variable']
        bn     = get_base_name(feat)
        beta_s = row['beta_std']   # β в стандартизованных единицах
        if bn not in UNIT_SHOCKS:
            continue

        shock_label, shock_size = UNIT_SHOCKS[bn]

        # SD фактора в оригинальных единицах
        if feat in df.columns:
            sd_feat = float(df[feat].std())
            mean_feat = float(df[feat].mean())
        else:
            continue

        if '_sq' in feat:
            # Квадратичный терм: дифференцируем в точке среднего
            # β_std × (price_sd / price_mean) × 2 × mean_feat / sd_feat × shock × 100
            if sd_feat < 1e-10: continue
            delta_pct = beta_s * (price_sd / max(price_mean, 1e-10)) * 2 * mean_feat / sd_feat * shock_size * 100
            calc_note = f"2×β×x̄ (нелин., x̄={mean_feat:.2f})"
        else:
            # Линейный терм: β_std / sd_feat × shock × price_sd/price_mean × 100
            if sd_feat < 1e-10: continue
            delta_pct = beta_s * (price_sd / max(price_mean, 1e-10)) / sd_feat * shock_size * 100
            calc_note = f"β×(SD_y/ȳ)/SD_x×шок"

        sign_str = "+" if delta_pct > 0 else ""
        print(f"   {feat:<35s} {shock_label:<20s} {sign_str}{delta_pct:>8.2f}%  [{calc_note}]")

    # ── Диагностика знаков коэффициентов ─────────────────────────────────
    # Ожидаемые знаки для целевой переменной = объём сделок (deals_*)
    EXPECTED_SIGNS = {
        # base name: (ожидаемый знак, комментарий)
        # Знак None = неоднозначный/контекстно-зависимый, проверяем отдельно
        'offers_primary':    (+1, 'больше предложения → больше сделок'),
        'offers_secondary':  (None, 'знак зависит от лага: без лага — избыток предложения в '
                              'текущем месяце давит на ликвидность (β<0 корректен для overhang); '
                              'с лагом 6+ мес. тот же эффект'),
        'housing_completed': (+1, 'новое жильё → больше сделок'),
        'housing_loans':     (None, 'исключается: r=0.96 с mortgage_intensity — числитель≈отношение'),
        'inflation':         (-1, 'рост инфляции → снижение реального спроса'),
        'rate':              (-1, 'рост ставки → снижение доступности'),
        'sentiment':         (None, 'неоднозначно после дифференцирования I(1): '
                              'Δsentiment отрицателен в периоды ажиотажных покупок 2022–23'),
        'expectations':      (+1, 'позитивные ожидания → больше сделок'),
        'current_state':     (None, 'неоднозначно с лагом: высокое состояние предшествует '
                              'охлаждению рынка (индикатор перегрева при лаге 2–6 мес.)'),
        'mortgage_intensity':(None, 'неоднозначно с лагом: высокая интенсивность lag6 → '
                              'контрциклическое охлаждение рынка'),
        'absorption_primary': (+1, 'высокая поглощаемость → активный рынок'),
        'absorption_secondary':(+1, 'высокая поглощаемость → активный рынок'),
        'overdue_rate':      (+1, 'финансовый стресс → вынужденные продажи → рост сделок'),
        # Положительный знак overdue_rate — экономически объяснимый механизм:
        # просрочка (frac. overdue/debt) растёт при доходном шоке домохозяйств,
        # что вынуждает их продавать жильё. Кредиты фиксированные → ставка ЦБ
        # не влияет напрямую на существующие обязательства.
    }
    print(f"\n   Диагностика знаков коэффициентов:")
    sign_issues = []
    for row in all_importance:
        bn = get_base_name(row['variable'])
        if bn not in EXPECTED_SIGNS:
            continue
        expected = EXPECTED_SIGNS[bn]
        actual   = np.sign(row['beta_std'])
        # Для _sq переменных знак инвертируется (нелинейный эффект — сложнее)
        if '_sq' in row['variable']:
            continue
        expected_sign, comment = expected
        if expected_sign is None:
            print(f"   ℹ '{row['variable']}': β={row['beta_std']:+.4f} — {comment}")
            continue
        if actual != 0 and actual != expected_sign:
            sign_issues.append((row['variable'], row['beta_std'], expected_sign))
            print(f" '{row['variable']}': β={row['beta_std']:+.4f}, ожидался {'+'if expected_sign>0 else '-'} — {comment}")
        else:
            print(f"  '{row['variable']}': β={row['beta_std']:+.4f} — {comment}")
    if not sign_issues:
        print(f" Все знаки соответствуют экономической логике")

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

    rc          = resid_arr[mask]
    jb_s, jb_p  = jarque_bera(rc)
    sw_s, sw_p  = shapiro(rc[:5000]) if len(rc) > 5000 else shapiro(rc)
    mean_r      = np.mean(rc)

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
        f.write(f"Baseline R²: {baseline_r2:.4f}\n")
        f.write(f"Final R²:    {results.rsquared_within:.4f}\n")
        f.write(f"Improvement: {improvement:+.4f}\n\n")
        f.write(str(results.summary))

    pc_coefs.to_csv(f"{OUT_PREFIX}_Final_Model_PCA_coefs.csv", sep=";", index=False)
    for info in pca_info:
        info['loadings'].to_csv(
            f"{OUT_PREFIX}_PCA_group{info['group']}_loadings.csv", sep=";")

    print(f"\n Результаты сохранены с префиксом: {OUT_PREFIX}_*")

    # Чистим PC-колонки перед следующим прогоном
    pc_cols = [c for c in df.columns if c.startswith('PC_FINAL_')]
    df.drop(columns=pc_cols, inplace=True, errors='ignore')

print("\n" + "="*100)
print("Оба рынка обработаны")
print("="*100)
