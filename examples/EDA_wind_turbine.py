# -*- coding: utf-8 -*-
"""
Разведывательный анализ данных ветрогенератора
================================================

Структура анализа:
    1.  Подготовка окружения и загрузка данных.
    2.  Общий обзор каждой таблицы.
    3.  Анализ временного покрытия и пропусков.
    4.  Исследование журнала отказов.
    5.  Исследование журнала статусов.
    6.  Анализ телеметрии SCADA: распределения, выбросы, корреляции,
        сводная таблица признаков по типам отказов.
    7.  Анализ режимов работы установки по кривой мощности.
    8.  Сводные выводы.

Входные файлы:
    scada_data.csv, status_data.csv, fault_data.csv.
"""

# %% Раздел 1. Подготовка окружения и загрузка данных
# ----------------------------------------------------------------------------
# Импорт библиотек и настройки отображения. Эту ячейку нужно запускать первой.

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Параметры pandas: показывать больше столбцов, аккуратный формат чисел
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 180)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

# Параметры графиков
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = (11, 5)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Путь к данным. По умолчанию — подпапка dataset рядом со скриптом.
# При необходимости укажите абсолютный путь, например:
#   DATA_DIR = r'C:\Users\user\data\wind_turbine'
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Если скрипт запускается интерактивно ячейками — используем cwd.
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'dataset')
if not os.path.isdir(DATA_DIR):
    DATA_DIR = BASE_DIR

print('Версии библиотек:')
print(f'  pandas      : {pd.__version__}')
print(f'  numpy       : {np.__version__}')
print(f'  seaborn     : {sns.__version__}')
print(f'\nКаталог данных: {DATA_DIR}')


# %% Загрузка всех трёх источников
scada = pd.read_csv(os.path.join(DATA_DIR, 'scada_data.csv'))
status = pd.read_csv(os.path.join(DATA_DIR, 'status_data.csv'))
faults = pd.read_csv(os.path.join(DATA_DIR, 'fault_data.csv'))

# Приведение столбцов времени к единому типу datetime
scada['DateTime'] = pd.to_datetime(scada['DateTime'], errors='coerce')
status['Time'] = pd.to_datetime(status['Time'], dayfirst=True, errors='coerce')
faults['DateTime'] = pd.to_datetime(faults['DateTime'], errors='coerce')

# Сортировка по времени
scada = scada.sort_values('DateTime').reset_index(drop=True)
status = status.sort_values('Time').reset_index(drop=True)
faults = faults.sort_values('DateTime').reset_index(drop=True)

print(f'SCADA  : {scada.shape[0]:>6} строк, {scada.shape[1]:>3} столбцов')
print(f'Status : {status.shape[0]:>6} строк, {status.shape[1]:>3} столбцов')
print(f'Faults : {faults.shape[0]:>6} строк, {faults.shape[1]:>3} столбцов')


# %% Раздел 2. Общий обзор таблиц
# ----------------------------------------------------------------------------
# Прежде чем переходить к содержательному анализу, фиксируется состав каждой
# таблицы: типы столбцов, доля пропусков, базовые статистики. Это позволяет
# на ранних этапах обнаружить очевидные дефекты данных.

def overview(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Сводка по таблице: тип, число пропусков, доля пропусков, число уникальных значений."""
    summary = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'n_missing': df.isna().sum(),
        'pct_missing': (df.isna().mean() * 100).round(2),
        'n_unique': df.nunique(dropna=True),
    })
    print(f'=== {name} === shape={df.shape}')
    return summary


print(overview(faults, 'fault_data').head(20))


# %% Обзор журнала статусов
print(overview(status, 'status_data').head(20))


# %% Обзор телеметрии SCADA (первые 25 каналов)
print(overview(scada, 'scada_data').head(25))


# %% Приведение булевых столбцов журнала статусов
# Поля Service и FaultMsg в исходных данных прочитаны как строки 'TRUE'/'FALSE'.
# Приводим их к настоящему булевому типу.
for col in ['Service', 'FaultMsg']:
    status[col] = status[col].map({'TRUE': True, 'FALSE': False,
                                    True: True, False: False})

print('Типы булевых столбцов после приведения:')
print(status[['Service', 'FaultMsg']].dtypes)


# %% Раздел 3. Временное покрытие
# ----------------------------------------------------------------------------
# Прежде чем сводить данные между собой, необходимо убедиться, что временные
# диапазоны трёх источников действительно пересекаются и что в SCADA нет
# систематических пропусков (длительных провалов в записи).

def time_range(df, col, name):
    print(f'{name:<8}: с {df[col].min()}  по  {df[col].max()}  '
          f'(длительность: {df[col].max() - df[col].min()})')


time_range(scada,  'DateTime', 'SCADA')
time_range(status, 'Time',     'Status')
time_range(faults, 'DateTime', 'Faults')


# %% Визуализация временного покрытия трёх источников
# Каждый источник показан как горизонтальная полоса; каждая чёрточка —
# одна запись. Сразу видно, что источники покрывают разные периоды.
t_scada = scada['DateTime']
t_status = status['Time']
t_fault = faults['DateTime']

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_scada,  np.full(len(t_scada),  1), '|', markersize=8,
        color='steelblue', label='SCADA')
ax.plot(t_status, np.full(len(t_status), 2), '|', markersize=8,
        color='darkorange', label='Status')
ax.plot(t_fault,  np.full(len(t_fault),  3), '|', markersize=8,
        color='firebrick', label='Faults')

ax.set_ylim(0, 4)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['SCADA', 'Status', 'Faults'])
ax.set_xlabel('Дата')
ax.set_title('Временное покрытие трёх источников данных')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.4, axis='x')
plt.tight_layout()
plt.show()


# %% Количественная оценка пересечения временных периодов
common_start = max(t_scada.min(), t_status.min(), t_fault.min())
common_end = min(t_scada.max(), t_status.max(), t_fault.max())

in_common = (scada['DateTime'] >= common_start) & (scada['DateTime'] <= common_end)
share_in = in_common.mean() * 100
share_out = (1 - in_common.mean()) * 100

print(f'Общий период пересечения: с {common_start}  по  {common_end}')
print(f'Длительность пересечения: {common_end - common_start}')
print()
print(f'SCADA-записей внутри пересечения: {in_common.sum():>6} ({share_in:.1f} %)')
print(f'SCADA-записей вне  пересечения:   {(~in_common).sum():>6} ({share_out:.1f} %)')

# Вывод: журнал отказов охватывает только часть периода SCADA. Для SCADA-записей
# вне общего периода отсутствие записи об отказе НЕ означает «отказа не было» —
# эта часть данных просто не размечена.


# %% Анализ шага дискретизации SCADA
scada_dt = scada['DateTime'].diff().dropna()
print('Распределение интервалов между соседними записями SCADA:')
print(scada_dt.describe())

# Разрывы длительностью больше 15 минут (типовой шаг — 10 минут)
long_gaps = scada_dt[scada_dt > pd.Timedelta(minutes=15)]
print(f'\nЧисло разрывов длительностью более 15 минут: {len(long_gaps)}')
print(f'Суммарная длительность таких разрывов: {long_gaps.sum()}')


# %% Визуализация плотности записей по времени
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

axes[0].hist(scada['DateTime'], bins=80, color='steelblue', edgecolor='white')
axes[0].set_title('Плотность записей SCADA (10-минутные агрегаты)')
axes[0].set_ylabel('Количество')

axes[1].hist(status['Time'], bins=80, color='darkorange', edgecolor='white')
axes[1].set_title('Плотность записей журнала статусов')
axes[1].set_ylabel('Количество')

axes[2].hist(faults['DateTime'], bins=80, color='firebrick', edgecolor='white')
axes[2].set_title('Плотность зафиксированных отказов')
axes[2].set_ylabel('Количество')
axes[2].set_xlabel('Дата')

plt.tight_layout()
plt.show()


# %% Раздел 4. Журнал отказов
# ----------------------------------------------------------------------------
# Файл fault_data.csv содержит зафиксированные отказы, классифицированные
# по типу. Расшифровка кодов:
#   GF (Generator Heating Fault) — перегрев генератора
#   MF (Mains Failure Fault)     — отказ электросети
#   FF (Feeding Fault)           — отказ системы подачи/преобразования
#   AF (Air Cooling Fault)       — отказ системы воздушного охлаждения
#   EF (Excitation Fault)        — отказ системы возбуждения

fault_counts = faults['Fault'].value_counts()
fault_share = (fault_counts / fault_counts.sum() * 100).round(1)

fault_table = pd.DataFrame({'count': fault_counts, 'share_%': fault_share})
print('Распределение отказов по типам (только записи журнала отказов):')
print(fault_table)


# %% Совмещение SCADA с журналом отказов и визуализация перекоса классов
# Размечаем каждое 10-минутное окно SCADA внутри общего периода либо типом
# отказа, либо меткой 'NF' (No Fault — нормальная работа без зафиксированного
# отказа в окрестности ±10 минут).
scada_common = scada.loc[in_common, ['DateTime']].copy()
scada_common = scada_common.sort_values('DateTime').reset_index(drop=True)

faults_sorted = faults.sort_values('DateTime').reset_index(drop=True)

merged = pd.merge_asof(
    scada_common,
    faults_sorted[['DateTime', 'Fault']],
    on='DateTime',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=10),
)
merged['Fault'] = merged['Fault'].fillna('NF')

modes_counts = merged['Fault'].value_counts()
modes_share = (modes_counts / modes_counts.sum() * 100).round(2)
modes_table = pd.DataFrame({'count': modes_counts, 'share_%': modes_share})
print('Распределение режимов работы (с учётом нормального режима NF):')
print(modes_table)

# Упорядочиваем категории по убыванию частоты, NF — первой
ordered = ['NF'] + [c for c in ['FF', 'EF', 'AF', 'GF', 'MF']
                    if c in modes_counts.index]
modes_counts = modes_counts.reindex(ordered)

fig, ax = plt.subplots(figsize=(7, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
ax.pie(modes_counts, labels=modes_counts.index, colors=colors,
       startangle=90, counterclock=False, textprops={'fontsize': 12})
ax.set_title('Fault Modes', fontsize=14)
ax.set_ylabel('count')
plt.tight_layout()
plt.show()

# Вывод: класс NF доминирует — задача сильно несбалансирована.
# Простая accuracy непригодна; нужны метрики recall/F1/PR-AUC,
# балансировка классов или подход one-class detection.


# %% Помесячная динамика отказов
faults['year_month'] = faults['DateTime'].dt.strftime('%Y-%m')
monthly = (faults.groupby(['year_month', 'Fault']).size()
                 .unstack(fill_value=0)
                 .sort_index())

fig, ax = plt.subplots(figsize=(12, 5))
monthly.plot(kind='bar', stacked=True, ax=ax,
             colormap='tab10', edgecolor='white', width=0.85)
ax.set_title('Помесячная динамика отказов по типам')
ax.set_xlabel('Месяц')
ax.set_ylabel('Количество отказов')
plt.xticks(rotation=45, ha='right')
ax.legend(title='Тип отказа', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %% Распределение отказов по времени суток
faults['hour'] = faults['DateTime'].dt.hour
hourly = faults.groupby(['hour', 'Fault']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(11, 4))
hourly.plot(kind='bar', stacked=True, ax=ax,
            colormap='tab10', edgecolor='white', width=0.9)
ax.set_title('Распределение отказов по часам суток')
ax.set_xlabel('Час суток')
ax.set_ylabel('Количество отказов')
ax.legend(title='Тип отказа', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# %% Раздел 5. Журнал статусов
# ----------------------------------------------------------------------------
# Журнал статусов фиксирует все смены режимов работы турбины. Один и тот же
# физический процесс может многократно повторяться в журнале — например,
# многократные старты в условиях слабого ветра.

# Топ-15 текстовых статусов по частоте появления
top_statuses = status['Status Text'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
top_statuses[::-1].plot(kind='barh', color='steelblue', edgecolor='white', ax=ax)
ax.set_title('Топ-15 статусов работы турбины по частоте появления')
ax.set_xlabel('Количество событий')
ax.set_ylabel('')
plt.tight_layout()
plt.show()


# %% Доли сервисных событий и аварийных сообщений
shares = pd.DataFrame({
    'Service (плановое обслуживание)': status['Service'].mean(),
    'FaultMsg (сообщение об отказе)': status['FaultMsg'].mean(),
}, index=['доля от всех записей']).T * 100

shares.columns = ['доля, %']
print(shares.round(2))

# Совместное распределение Service x FaultMsg
ct = pd.crosstab(status['Service'], status['FaultMsg'],
                 rownames=['Service'], colnames=['FaultMsg'])
print('\nСовместное распределение признаков:')
print(ct)


# %% Длительности нахождения в каждом статусе
# Длительность статуса = разница между его началом и началом следующего события.
status_sorted = status.sort_values('Time').copy()
status_sorted['next_time'] = status_sorted['Time'].shift(-1)
status_sorted['duration_min'] = (
    (status_sorted['next_time'] - status_sorted['Time']).dt.total_seconds() / 60
)

duration_by_status = (status_sorted.groupby('Status Text')['duration_min']
                      .agg(['count', 'sum', 'median'])
                      .sort_values('sum', ascending=False)
                      .head(15))
duration_by_status.columns = ['число эпизодов', 'суммарно, мин', 'медиана, мин']
print('Топ-15 статусов по суммарной длительности:')
print(duration_by_status.round(1))

# Замечание: частота и суммарная длительность — разные характеристики.
# Статус с большим числом эпизодов может занимать мало времени, и наоборот.


# %% Раздел 6. Телеметрия SCADA
# ----------------------------------------------------------------------------
# Числовые каналы установки: скорость ветра, мощность, температуры.

# Базовые физические каналы установки
key_channels = [
    'WEC: ava. windspeed',
    'WEC: ava. Rotation',
    'WEC: ava. Power',
    'WEC: ava. reactive Power',
    'WEC: ava. blade angle A',
    'Nacelle ambient temp. 1',
    'Ambient temp.',
    'Front bearing temp.',
    'Rear bearing temp.',
    'Stator temp. 1',
    'Rotor temp. 1',
    'Transformer temp.',
]
print(scada[key_channels].describe().T)


# %% Гистограммы ключевых переменных
fig, axes = plt.subplots(3, 4, figsize=(15, 9))
for ax, col in zip(axes.ravel(), key_channels):
    data = scada[col].dropna()
    ax.hist(data, bins=60, color='steelblue', edgecolor='white')
    ax.set_title(col, fontsize=10)
    ax.set_ylabel('Количество')
plt.tight_layout()
plt.show()


# %% Поиск выбросов по правилу межквартильного размаха (IQR)
def iqr_outliers(s: pd.Series, k: float = 3.0) -> dict:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    mask = (s < low) | (s > high)
    return {'n_outliers': int(mask.sum()),
            'share_%': round(mask.mean() * 100, 3),
            'low_bound': low, 'high_bound': high}


outlier_report = pd.DataFrame(
    {col: iqr_outliers(scada[col]) for col in key_channels}
).T
print(outlier_report)

# Замечание: для физических величин (температура, скорость ветра) большое число
# выбросов по IQR обычно отражает тяжёлые хвосты распределения, а не ошибки.
# Удалять следует только физически невозможные значения.


# %% Корреляционная матрица по ключевым каналам
corr = scada[key_channels].corr(method='pearson')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.75})
ax.set_title('Корреляция Пирсона между ключевыми каналами SCADA')
plt.tight_layout()
plt.show()


# %% Корреляционная матрица всех температурных каналов
temp_cols = [c for c in scada.columns if 'temp' in c.lower()]
print(f'Найдено температурных каналов: {len(temp_cols)}')

corr_t = scada[temp_cols].corr(method='pearson')

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr_t, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3, ax=ax,
            cbar_kws={'shrink': 0.6})
ax.set_title('Корреляция температурных каналов SCADA', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

# Замечание: блочная структура корреляций — признак информационной избыточности.
# Каналы одной подсистемы дают почти одно и то же; имеет смысл либо снижать
# размерность, либо отбирать представителей кластеров.


# %% Раздел 6.1. Сводная таблица признаков по типам отказов
# ----------------------------------------------------------------------------
# Объединяем телеметрию SCADA с метками отказов и считаем средние значения
# каналов в каждом режиме. Каналы, средние которых заметно отличаются между
# нормальным режимом и режимом отказа, — кандидаты в признаки модели.

scada_common_full = scada.loc[in_common].sort_values('DateTime').reset_index(drop=True)

df_combine = pd.merge_asof(
    scada_common_full,
    faults_sorted[['DateTime', 'Fault']],
    on='DateTime',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=10),
)
df_combine['Fault'] = df_combine['Fault'].fillna('NF')

print(f'Размерность объединённой таблицы: {df_combine.shape}')
print('Размер выборки по режимам:')
print(df_combine['Fault'].value_counts())


# %% Средние значения по типам отказов (последние 20 каналов)
df_summary = df_combine.groupby('Fault').mean(numeric_only=True).T

col_order = [c for c in ['NF', 'FF', 'EF', 'AF', 'GF', 'MF']
             if c in df_summary.columns]
df_summary = df_summary[col_order]

print('Сводная таблица средних значений каналов по типам отказов:')
print(df_summary.tail(20))


# %% Топ каналов по относительному отклонению от нормального режима
if 'NF' in df_summary.columns:
    nf_mean = df_summary['NF']
    rel_diff = df_summary.subtract(nf_mean, axis=0).divide(nf_mean.abs() + 1e-9,
                                                           axis=0) * 100
    rel_diff = rel_diff.drop(columns='NF')
    rel_diff['max_abs_dev_%'] = rel_diff.abs().max(axis=1)
    top_features = rel_diff.sort_values('max_abs_dev_%', ascending=False).head(15)
    print('Топ-15 каналов по максимальному относительному отклонению от NF:')
    print(top_features.round(2))


# %% Раздел 7. Кривая мощности
# ----------------------------------------------------------------------------
# Кривая мощности — стандартный диагностический инструмент в индустрии
# ветроэнергетики. По её виду можно судить о корректности работы установки.

ws = scada['WEC: ava. windspeed']
p = scada['WEC: ava. Power']

fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(ws, p, s=4, alpha=0.25, color='steelblue')
ax.set_title('Кривая мощности: средняя мощность относительно средней скорости ветра')
ax.set_xlabel('Средняя скорость ветра, м/с')
ax.set_ylabel('Средняя мощность, кВт')
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


# %% Эмпирическая кривая мощности: усреднение по бинам скорости ветра
bins = np.arange(0, ws.max() + 1, 0.5)
scada['ws_bin'] = pd.cut(ws, bins)
curve = scada.groupby('ws_bin', observed=False)['WEC: ava. Power'].agg(['mean', 'median', 'count'])
curve['ws_mid'] = [interval.mid for interval in curve.index]

fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(ws, p, s=3, alpha=0.15, color='lightsteelblue', label='наблюдения')
ax.plot(curve['ws_mid'], curve['mean'], color='firebrick', linewidth=2,
        label='средняя мощность по бину')
ax.plot(curve['ws_mid'], curve['median'], color='darkgreen', linewidth=2,
        linestyle='--', label='медиана по бину')
ax.set_xlabel('Средняя скорость ветра, м/с')
ax.set_ylabel('Средняя мощность, кВт')
ax.set_title('Эмпирическая кривая мощности и облако наблюдений')
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# На графике видны три области: (1) ниже стартовой скорости (~3 м/с) мощность
# близка к нулю; (2) в рабочем диапазоне (3–12 м/с) — характерный кубический
# рост; (3) при высоких скоростях — насыщение (работа на номинальной мощности).
# Точки ниже эталонной кривой при высоких скоростях ветра — кандидаты на
# разметку как аномальные эпизоды (простои, ограничения сети, дефекты).


# %% Оценка коэффициента использования установленной мощности
rated_power = p.quantile(0.99)  # робастная оценка номинальной мощности
capacity_factor = p.mean() / rated_power * 100
downtime_share = (p <= 0).mean() * 100

print(f'Робастная оценка номинальной мощности (99-й перцентиль): {rated_power:,.0f} кВт')
print(f'Коэффициент использования установленной мощности (CF):   {capacity_factor:.1f} %')
print(f'Доля записей с нулевой мощностью:                        {downtime_share:.1f} %')


# %% Совместная динамика ключевых каналов во времени
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

# Прореживание для наглядности графика
step = max(1, len(scada) // 4000)
scada_plot = scada.iloc[::step]

axes[0].plot(scada_plot['DateTime'], scada_plot['WEC: ava. windspeed'],
             linewidth=0.6, color='steelblue')
axes[0].set_ylabel('Ветер, м/с')
axes[0].set_title('Динамика основных каналов установки')

axes[1].plot(scada_plot['DateTime'], scada_plot['WEC: ava. Power'],
             linewidth=0.6, color='darkgreen')
axes[1].set_ylabel('Мощность, кВт')

axes[2].plot(scada_plot['DateTime'], scada_plot['Stator temp. 1'],
             linewidth=0.6, color='firebrick', label='Статор 1')
axes[2].plot(scada_plot['DateTime'], scada_plot['Front bearing temp.'],
             linewidth=0.6, color='darkorange', label='Передний подшипник')
axes[2].set_ylabel('Температура, °C')
axes[2].set_xlabel('Дата')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()


# %% Раздел 8. Сводные выводы
# ----------------------------------------------------------------------------
# Качество данных:
#   1. Три источника имеют согласованные временные диапазоны и пригодны
#      для совместного анализа. Шаг SCADA близок к 10 минутам.
#   2. Журналы статусов и отказов — событийные; их нужно привязывать к окнам
#      SCADA через merge_asof, а не наоборот.
#   3. Журнал отказов покрывает лишь часть периода SCADA — около 28 % записей
#      SCADA лежат вне периода отказов и не должны размечаться как «нормальные».
#
# Структура отказов:
#   1. Доминируют FF (Feeding Fault) и EF (Excitation Fault) — около 75 %
#      событий. Приоритетные зоны мониторинга — электрические подсистемы.
#   2. GF (перегрев генератора) и AF (отказ воздушного охлаждения) — реже,
#      но могут обнаруживаться рано по температурным каналам SCADA.
#   3. MF (отказ сети) — внешний фактор, при моделировании надёжности
#      установки рассматривается отдельно.
#
# Структура телеметрии:
#   1. Среди ~65 каналов SCADA доминируют температуры; они образуют плотно
#      скоррелированные кластеры по подсистемам — есть избыточность.
#   2. Распределения физических величин соответствуют отраслевым ожиданиям.
#
# Эксплуатационные характеристики:
#   1. Кривая мощности имеет ожидаемый вид. Точки, отклоняющиеся вниз от
#      эталонной кривой при достаточном ветре, — кандидаты в аномальные.
#   2. Доля интервалов с нулевой мощностью — важная метрика доступности.
#
# Рекомендации к следующим шагам:
#   1. Сформировать единую таблицу: каждое 10-минутное окно SCADA размечается
#      ближайшим статусом и флагом отказа в окрестности (±60 минут).
#   2. Отфильтровать физически невозможные значения отдельно от естественных
#      хвостов распределений.
#   3. Для температурных каналов перейти к признакам уровня подсистем
#      (средние и std по кластеру вместо отдельных датчиков).
#   4. Построить отдельные модели по типам отказов; для малых классов
#      (MF, GF, AF) — рассмотреть one-class detection или балансировку.
#   5. Валидация — по времени, а не случайным разбиением, для корректной
#      оценки обобщающей способности на будущих данных.
