# Script to compute TOU pricing for each time period in a dataset and return a modified dataset.
# Input: CSV file of daily consumption with time/date data as one column
# Output: CSV file of daily consumption with TOU pricing data added

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Tariff rate data for TOU-DR1
SUMMER_MONTHS = [6, 7, 8, 9, 10]  # June 1 through Oct 31
WINTER_MONTHS = [1, 2, 3, 4, 5, 11, 12]  # Nov 1 through May 31th
ON_PEAK = [16, 17, 18, 19, 20]  # 4pm - 9pm, same for all days
SUMMER_OFF_PEAK = [
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    21,
    22,
    23,
]  # 6am - 4pm, 9pm - midnight
SUPER_OFF_PEAK = [
    0,
    1,
    2,
    3,
    4,
    5,
]  # midnight - 6am, same for all days except in March and April
WINTER_OFF_PEAK = [
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    21,
    22,
    23,
]  # 6am - 4pm, 9pm - midnight
WINTER_OFF_PEAK_MAR_APR = [
    6,
    7,
    8,
    9,
    14,
    15,
    21,
    22,
    23,
]  # 6am - 4pm, 9pm - midnight, excluding 10:00 a.m. – 2:00 p.m
WINTER_SUPER_OFF_PEAK_MAR_APR = [
    0,
    1,
    2,
    3,
    4,
    5,
    10,
    11,
    12,
    13,
]  # midnight - 6am; 10am = 2pm
OFF_PEAK_WEEKEND = [14, 15, 21, 22, 23]  # 2pm - 4pm; 9pm - midnight
SUPER_OFF_PEAK_WEEKEND = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
]  # midnight - 2pm

# since we have 15-minute periods, therefore $/kw-hour must be divided by (15/60) = 4
SUM_ON_PEAK_TOU = 0.50199 / 4
SUM_OFF_PEAK_TOU = 0.30462 / 4
SUM_SUP_OFF_PEAK_TOU = 0.25900 / 4

WIN_ON_PEAK_TOU = 0.35630 / 4
WIN_OFF_PEAK_TOU = 0.34747 / 4
WIN_SUP_OFF_PEAK_TOU = 0.3376 / 4

# df = pd.read_csv('9836.csv')
#
# # drop NaNs (0 in original CSV -- not metered load quantities)
# df.dropna(axis=1, how='all', inplace=True)
#
# # Remove unnecessary voltage data and dataid columns
# df.drop(['dataid', 'leg1v', 'leg2v'], axis=1, inplace=True)
#
# # Create column subtracting out PV output:
# df['gridnopv'] = df['grid'] - df['solar']

# Keep only grid and solar data:
df = pd.read_csv("9836.csv", usecols=["local_15min", "grid", "solar"])

# Create column subtracting out PV output:
df["gridnopv"] = df["grid"] + df["solar"]

# Convert first column to datetime:

df["dt"] = pd.to_datetime(df["local_15min"], format="%m/%d/%Y %H:%M")

# Plot!
fig = plt.figure(figsize=(8, 6), dpi=150)
ax = plt.gca()
# df.plot(kind='line', x='dt', y='grid', ax=ax, xlabel='Date', ylabel='Power, kW')
# df.plot(kind='line', x='dt', y='solar', color='red', ax=ax, xlabel='Date', ylabel='Power, kW')
df.plot(
    kind="line",
    x="dt",
    y="gridnopv",
    color="green",
    ax=ax,
    xlabel="Date",
    ylabel="Power, kW",
)
fig.savefig("load_data.png")

df = df.assign(tariff="")

# Summer TOU pricing, weekdays:
df.loc[
    df["dt"].dt.month.isin(SUMMER_MONTHS)
    & df["dt"].dt.hour.isin(ON_PEAK)
    & df["dt"].dt.weekday.isin([1, 2, 3, 4, 5]),
    "tariff",
] = SUM_ON_PEAK_TOU
df.loc[
    df["dt"].dt.month.isin(SUMMER_MONTHS)
    & df["dt"].dt.hour.isin(SUMMER_OFF_PEAK)
    & df["dt"].dt.weekday.isin([1, 2, 3, 4, 5]),
    "tariff",
] = SUM_OFF_PEAK_TOU
df.loc[
    df["dt"].dt.month.isin(SUMMER_MONTHS)
    & df["dt"].dt.hour.isin(SUPER_OFF_PEAK)
    & df["dt"].dt.weekday.isin([1, 2, 3, 4, 5]),
    "tariff",
] = SUM_SUP_OFF_PEAK_TOU

# Winter TOU pricing, weekdays:
df.loc[
    df["dt"].dt.month.isin(WINTER_MONTHS)
    & df["dt"].dt.hour.isin(ON_PEAK)
    & df["dt"].dt.weekday.isin([1, 2, 3, 4, 5]),
    "tariff",
] = WIN_ON_PEAK_TOU
df.loc[
    df["dt"].dt.month.isin(WINTER_MONTHS)
    & df["dt"].dt.hour.isin(WINTER_OFF_PEAK)
    & df["dt"].dt.weekday.isin([1, 2, 3, 4, 5]),
    "tariff",
] = WIN_OFF_PEAK_TOU
df.loc[
    df["dt"].dt.month.isin(WINTER_MONTHS)
    & df["dt"].dt.hour.isin(SUPER_OFF_PEAK)
    & df["dt"].dt.weekday.isin([1, 2, 3, 4, 5]),
    "tariff",
] = WIN_SUP_OFF_PEAK_TOU
# Adjust March and April TOU periods:
df.loc[
    df["dt"].dt.month.isin([3, 4])
    & df["dt"].dt.hour.isin(WINTER_SUPER_OFF_PEAK_MAR_APR)
    & df["dt"].dt.weekday.isin([1, 2, 3, 4, 5]),
    "tariff",
] = WIN_SUP_OFF_PEAK_TOU

# Summer TOU pricing, weekends:
df.loc[
    df["dt"].dt.month.isin(SUMMER_MONTHS)
    & df["dt"].dt.hour.isin(ON_PEAK)
    & df["dt"].dt.weekday.isin([0, 6]),
    "tariff",
] = SUM_ON_PEAK_TOU
df.loc[
    df["dt"].dt.month.isin(SUMMER_MONTHS)
    & df["dt"].dt.hour.isin(OFF_PEAK_WEEKEND)
    & df["dt"].dt.weekday.isin([0, 6]),
    "tariff",
] = SUM_OFF_PEAK_TOU
df.loc[
    df["dt"].dt.month.isin(SUMMER_MONTHS)
    & df["dt"].dt.hour.isin(SUPER_OFF_PEAK_WEEKEND)
    & df["dt"].dt.weekday.isin([0, 6]),
    "tariff",
] = SUM_SUP_OFF_PEAK_TOU

# Winter TOU pricing, weekends:
df.loc[
    df["dt"].dt.month.isin(WINTER_MONTHS)
    & df["dt"].dt.hour.isin(ON_PEAK)
    & df["dt"].dt.weekday.isin([0, 6]),
    "tariff",
] = WIN_ON_PEAK_TOU
df.loc[
    df["dt"].dt.month.isin(WINTER_MONTHS)
    & df["dt"].dt.hour.isin(OFF_PEAK_WEEKEND)
    & df["dt"].dt.weekday.isin([0, 6]),
    "tariff",
] = WIN_OFF_PEAK_TOU
df.loc[
    df["dt"].dt.month.isin(WINTER_MONTHS)
    & df["dt"].dt.hour.isin(SUPER_OFF_PEAK_WEEKEND)
    & df["dt"].dt.weekday.isin([0, 6]),
    "tariff",
] = WIN_SUP_OFF_PEAK_TOU

# # Plot tariff rate!
# fig = plt.figure(figsize=(8, 6), dpi=150)
# ax = plt.gca()
# df.plot(kind='line', x='dt', y='tariff', color='black', ax=ax, xlabel='Date', ylabel='$/kWh')
# fig.savefig('tariff_data.png')

df.to_csv("load_tariff.csv")
