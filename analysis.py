# Analysis of our results!!
# Kevin Moy
#11/8/2020

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np


currentDirectory = os.getcwd().replace('\\', '/')
edDir = currentDirectory + '/Ed'

df_tou = pd.read_csv('opt_tou_5kW_14kWh.csv')
df_base = pd.read_csv(edDir + '/base.csv')
df_naive = pd.read_csv(edDir + '/naive.csv')
df_rl = pd.read_csv('output2.csv')

# Preprocess LMP data
df_lmp = pd.read_csv('df_LMP.csv')
lmpcv = df_lmp['Cumulative Additive Revenue'].to_numpy()
lmpcv_rep = np.repeat(lmpcv, 4)
lmpcv_load = df_base.R_base - lmpcv_rep[0:-4]

# Obtain cumulative rewards
df_rewards = pd.concat([df_base.local_15min, df_base.R_base, df_naive.R_naive, df_tou.cumulative_cost,
                        pd.Series(lmpcv_load), -df_rl.cumulative_revenue], axis=1)
df_rewards.set_index('local_15min', inplace=True)
df_rewards.index = pd.to_datetime(df_rewards.index)
df_rewards.columns = ['base', 'naive', 'optimal TOU', 'optimal LMP', 'Q-learned policy']

plot = df_rewards.plot()
fig = plot.get_figure()
fig.autofmt_xdate()
plot.set_xlabel('Date')
plot.set_ylabel('Cumulative cost, $')
fig.savefig("cumulative_cost_comparison.png")

actions = pd.concat([df_base.local_15min, df_rl.actions], axis=1)
# actions.actions = pd.Categorical(actions.actions)
plot2 = actions['actions'].value_counts().plot(kind='pie', legend=None)
fig = plot2.get_figure()
# plot2.set_xlabel('Action')
plot2.set_ylabel('')
fig.tight_layout()
fig.savefig("action_pie.png")

# action_types = ['LMP_buy', 'LMP_sell', 'wait', 'TOU_buy', 'TOU_discharge']
# actions['act_codes'] = pd.Categorical(actions.actions, categories=action_types).codes
# actions.set_index('local_15min', inplace=True)
# actions.index = pd.to_datetime(actions.index)
#
# actions.act_codes.plot()
