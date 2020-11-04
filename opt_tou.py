# File to compute optimal TOU dispatch from load data and tariff rate pricing
# Kevin Moy, 11/3/2020

import cvxpy as cp
import pandas as pd
import numpy as np

# Import load and tariff rate data; convert to numpy array and get length
df = pd.read_csv('load_tariff.csv')
# load = df.gridnopv[0:100].to_numpy()
# tariff = df.tariff[0:100].to_numpy()
load = df.gridnopv.to_numpy()
tariff = df.tariff.to_numpy()


# Set environment variables:
LOAD_LEN = load.size  # length of optimization
BAT_KW = 7  # Rated power of battery, in kW
BAT_KWH = 10  # Rated energy of battery, in kWh
BAT_KWH_MIN = 0.1*BAT_KWH  # Minimum SOE of battery, 10% of rated
BAT_KWH_MAX = 0.9*BAT_KWH  # Maximum SOE of battery, 90% of rated
BAT_KWH_INIT = 0.5*BAT_KWH  # Starting SOE of battery, 50% of rated
HR_FRAC = 15/60  # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh

# Create optimization variables.
grd_pow = cp.Variable(LOAD_LEN)  # Power consumed from grid
chg_pow = cp.Variable(LOAD_LEN)  # Power charged to the battery
dch_pow = cp.Variable(LOAD_LEN)  # Power discharged from the battery
bat_eng = cp.Variable(LOAD_LEN)  # Energy stored in the battery

# Create constraints.
constraints = [bat_eng[0] == BAT_KWH_INIT]

for i in range(LOAD_LEN):
    constraints += [grd_pow[i] == chg_pow[i] + dch_pow[i] + load[i],  # Power flow constraints
                    chg_pow[i] <= BAT_KW,
                    dch_pow[i] <= BAT_KW,
                    bat_eng[i] <= BAT_KWH_MAX,  # Prevent overcharging
                    bat_eng[i] >= BAT_KWH_MIN,  # Prevent undercharging
                    bat_eng[i] >= HR_FRAC * dch_pow[i],  # Prevent undercharging from overdischarging
                    # Convexity requirements:
                    grd_pow[i] >= 0,
                    chg_pow[i] >= 0,
                    dch_pow[i] >= 0,
                    bat_eng[i] >= 0]

for i in range(1, LOAD_LEN):
    constraints += [bat_eng[i] == HR_FRAC * chg_pow[i-1] - HR_FRAC * dch_pow[i-1]]  # Energy flow constraints

print('constraints complete')

# Form objective.
obj = cp.Minimize(grd_pow.T @ tariff)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
print('solving...')
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
