# File to compute optimal TOU dispatch from load data and tariff rate pricing
# Kevin Moy, 11/3/2020

import cvxpy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import load and tariff rate data; convert to numpy array and get length
df = pd.read_csv('load_tariff.csv')
# load = df.gridnopv[0:288].to_numpy()
# tariff = df.tariff[0:288].to_numpy()
# times = pd.to_datetime(df.local_15min[0:288])
load = df.gridnopv.to_numpy()
tariff = df.tariff.to_numpy()
times = pd.to_datetime(df.local_15min)


# Set environment variables:
LOAD_LEN = load.size  # length of optimization
BAT_KW = 7  # Rated power of battery, in kW
BAT_KWH = 14  # Rated energy of battery, in kWh.
# Note Tesla Powerwall rates their energy at 13.5kWh, but at 100% DoD,
# but I have also seen that it's actually 14kwh, 13.5kWh usable
BAT_KWH_MIN = 0.1*BAT_KWH  # Minimum SOE of battery, 10% of rated
BAT_KWH_MAX = 0.9*BAT_KWH  # Maximum SOE of battery, 90% of rated
BAT_KWH_INIT = 0.5*BAT_KWH  # Starting SOE of battery, 50% of rated
HR_FRAC = 15/60  # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh

# Create optimization variables.
grd_pow = cp.Variable(LOAD_LEN)  # Total power consumed from grid
lod_pow = cp.Variable(LOAD_LEN)  # Power consumed by load from grid
chg_pow = cp.Variable(LOAD_LEN)  # Power charged to the battery
dch_pow = cp.Variable(LOAD_LEN)  # Power discharged from the battery
bat_eng = cp.Variable(LOAD_LEN)  # Energy stored in the battery

# Create constraints.
constraints = [bat_eng[0] == BAT_KWH_INIT]

for i in range(LOAD_LEN):
    constraints += [grd_pow[i] == chg_pow[i] + lod_pow[i],  # Power flow constraints
                    load[i] == dch_pow[i] + lod_pow[i],
                    chg_pow[i] <= BAT_KW,
                    dch_pow[i] <= BAT_KW,
                    bat_eng[i] <= BAT_KWH_MAX,  # Prevent overcharging
                    bat_eng[i] >= BAT_KWH_MIN,  # Prevent undercharging
                    bat_eng[i] >= HR_FRAC * dch_pow[i],  # Prevent undercharging from overdischarging
                    # Convexity requirements:
                    grd_pow[i] >= 0,
                    chg_pow[i] >= 0,
                    dch_pow[i] >= 0,
                    bat_eng[i] >= 0,
                    lod_pow[i] >= 0]

for i in range(1, LOAD_LEN):
    constraints += [bat_eng[i] == HR_FRAC * chg_pow[i-1] + (bat_eng[i-1] - HR_FRAC * dch_pow[i-1])]  # Energy flow constraints

print('constraints complete')

# Form objective.
obj = cp.Minimize(grd_pow.T @ tariff)
# obj = cp.Minimize(lod_pow.T @ np.ones(LOAD_LEN))


# Form and solve problem.
prob = cp.Problem(obj, constraints)
print('solving...')
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)

# Calculate relevant quantities.
bat_pow = dch_pow.value - chg_pow.value
cumulative_cost = np.cumsum(grd_pow.value * tariff)


# Save output to CSV.
print('saving to CSV')
outputdf = pd.DataFrame(np.transpose([load, grd_pow.value, bat_pow, bat_eng.value, tariff, cumulative_cost]))
outputdf.columns = ['load_power', 'grid_power', 'battery_power', 'battery_energy', 'tariff_rate', 'cumulative_cost']
outputdf.set_index(times, inplace=True)
outputdf.to_csv('opt_tou.csv')


# # PLOTTING !
#
# fig, ax1 = plt.subplots(1, 1, figsize=(10,6))
# fig.autofmt_xdate()
# plt.gcf().autofmt_xdate()
# xfmt = mdates.DateFormatter('%m-%d-%y %H:%M')
# ax1.xaxis.set_major_formatter(xfmt)
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Power, kW')
# p1 = ax1.plot(times, bat_pow)
# p2 = ax1.plot(times, load)
# p3 = ax1.plot(times, grd_pow.value)
#
# color = 'tab:red'
# ax2 = ax1.twinx()
# ax2.set_ylabel('Energy Price, $/kWh', color=color)
# p4 = ax2.plot(times, tariff, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylim([0,1.1*max(tariff)])
# ax2.xaxis.set_major_formatter(xfmt)
#
# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Battery Power', 'Load Power', 'Grid Power', 'Tariff Rate'), loc='best')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
#
# plt.savefig('opt_ex_tariff.png')
#
# fig, ax1 = plt.subplots(1, 1, figsize=(10,6))
# fig.autofmt_xdate()
# plt.gcf().autofmt_xdate()
# xfmt = mdates.DateFormatter('%m-%d-%y %H:%M')
# ax1.xaxis.set_major_formatter(xfmt)
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Power, kW')
# p1 = ax1.plot(times, bat_pow)
# p2 = ax1.plot(times, load)
# p3 = ax1.plot(times, grd_pow.value)
#
# color = 'tab:purple'
# ax2 = ax1.twinx()
# ax2.set_ylabel('Energy, kWh', color=color)
# p4 = ax2.plot(times, bat_eng.value, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylim([0,BAT_KWH])
# ax2.xaxis.set_major_formatter(xfmt)
#
# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Battery Power', 'Load Power', 'Grid Power', 'Battery Energy'), loc='best')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
#
# plt.savefig('opt_ex_energy.png')