import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import datetime


"""
Base Case:

Generate Cumulative Rewards with no battery use


"""


def make_base_csv(df):
    load = df.gridnopv.to_numpy()
    tariff = df.tariff.to_numpy()
    times = pd.to_datetime(df.local_15min)

    df["R_base"] = np.cumsum(load * tariff)
    rewards = df["R_base"].to_numpy()

    df.to_csv(r"base.csv")
    # #plotting
    # fig, ax1 = plt.subplots(1, 1, figsize=(10,6))
    # fig.autofmt_xdate()
    # plt.gcf().autofmt_xdate()
    # xfmt = mdates.DateFormatter('%m-%d-%y %H:%M')
    # ax1.xaxis.set_major_formatter(xfmt)
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel('Power, kW')

    # #load
    # p1 = ax1.plot(times, tariff)
    # p2 = ax1.plot(times, load)
    # p3 = ax1.plot(times, rewards)
    # # p3 = ax1.plot(times, grd_pow.value)

    # color = 'tab:red'
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Energy Price, $/kWh', color=color)
    # # p4 = ax2.plot(times, tariff, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim([0,1.1*max(tariff)])
    # ax2.xaxis.set_major_formatter(xfmt)

    # # plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Battery Power', 'Load Power', 'Grid Power', 'Tariff Rate'), loc='best')
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # # plt.savefig('opt_ex_tariff.png')
    # ax2.set_xlim([datetime.date(2014, 7, 8), datetime.date(2014, 7, 11)])
    # plt.show()


"""
Naive TOU Case:

Generate Cumulative Rewards optimized on peak/non-peak TOU

nonpeak: lowest tariff price
peak: all other prices

battery charges/discharges at power rating 


"""


def make_naiveTOU_csv(df):
    load = df.gridnopv.to_numpy()
    tariff = df.tariff.to_numpy()
    times = pd.to_datetime(df.local_15min)

    min_tariff = min(tariff)
    max_tariff = max(tariff)

    # constants
    BAT_KW = 5  # Rated power of battery, in kW
    BAT_KWH = 14  # Rated energy of battery, in kWh.
    # Note Tesla Powerwall rates their energy at 13.5kWh, but at 100% DoD,
    # but I have also seen that it's actually 14kwh, 13.5kWh usable
    BAT_KWH_MIN = 0.1 * BAT_KWH  # Minimum SOE of battery, 10% of rated
    BAT_KWH_MAX = 0.9 * BAT_KWH  # Maximum SOE of battery, 90% of rated
    BAT_KWH_INIT = 0.5 * BAT_KWH  # Starting SOE of battery, 50% of rated
    HR_FRAC = (
        15 / 60
    )  # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh

    # df['bat_eng'], df['R_naive'], df['bat_charge']

    # set all battery charges to 0 initially
    df["bat_charge"] = 0
    df["bat_discharge"] = 0
    df["bat_eng"] = 0
    df["R_naive"] = 0

    for index, row in df.iterrows():

        # first row initialize
        if index == 0:
            df.loc[index, "bat_eng"], df.loc[index, "R_naive"] = BAT_KWH_INIT, 0
        else:
            df.loc[index, "bat_eng"], df.loc[index, "R_naive"] = bat_eng_old, R_old

        # when tariff is lowest, charge the battery all you can
        if row["tariff"] == min_tariff:

            if df.loc[index, "bat_eng"] < BAT_KWH_MAX:
                df.loc[index, "bat_charge"] = min(
                    BAT_KWH_MAX - df.loc[index, "bat_eng"], HR_FRAC * BAT_KW
                )
                df.loc[index, "bat_eng"] += df.loc[index, "bat_charge"]

        # try to discharge as much as possible at max tariff
        elif row["tariff"] == max_tariff:

            if df.loc[index, "bat_eng"] > BAT_KWH_MIN:
                df.loc[index, "bat_discharge"] = min(
                    df.loc[index, "bat_eng"] - BAT_KWH_MIN, HR_FRAC * BAT_KW
                )
                df.loc[index, "bat_eng"] -= df.loc[index, "bat_discharge"]

        # account for battery charge/discharge in the reward
        df.loc[index, "R_naive"] += (
            row["gridnopv"]
            - df.loc[index, "bat_discharge"]
            + df.loc[index, "bat_charge"]
        ) * row["tariff"]

        # update old values:
        bat_eng_old, R_old = df.loc[index, "bat_eng"], df.loc[index, "R_naive"]

    # to csv
    df.to_csv("naive.csv")

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig.autofmt_xdate()
    plt.gcf().autofmt_xdate()
    xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Power, kW")

    # p1 = ax1.plot(times, bat_pow)

    # load
    p2 = ax1.plot(times, load)
    # p3 = ax1.plot(times, grd_pow.value)

    p1 = ax1.plot(times, df["bat_charge"].to_numpy())
    p3 = ax1.plot(times, df["bat_discharge"].to_numpy())
    # p4 = ax1.plot(times, df['R_naive'].to_numpy())

    color = "tab:red"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Energy Price, $/kWh", color=color)
    # p4 = ax2.plot(times, tariff, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim([0, 1.1 * max(tariff)])
    ax2.xaxis.set_major_formatter(xfmt)

    # plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Battery Power', 'Load Power', 'Grid Power', 'Tariff Rate'), loc='best')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # plt.savefig('opt_ex_tariff.png')
    ax2.set_xlim([datetime.date(2014, 7, 8), datetime.date(2014, 7, 11)])
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(r"load_tariff.csv")
    # make_base_csv(df)
    make_naiveTOU_csv(df)

    # #Plot compare
    # df1= pd.read_csv(r'naive.csv')
    # df2 = pd.read_csv(r'base.csv')
    # times = pd.to_datetime(df1.local_15min)
    # fig, ax1 = plt.subplots(1, 1, figsize=(10,6))

    # #Plot

    # fig.autofmt_xdate()
    # plt.gcf().autofmt_xdate()
    # xfmt = mdates.DateFormatter('%m-%d-%y %H:%M')
    # ax1.xaxis.set_major_formatter(xfmt)
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel('Power, kW')
    # # p1 = ax1.plot(times, bat_pow)
    # p2 = ax1.plot(times, df1['R_naive'])
    # p3 = ax1.plot(times, df2['R_base'])
    ############
    # color = 'tab:red'
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Energy Price, $/kWh', color=color)
    # # p4 = ax2.plot(times, tariff, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim([0,1.1*max(tariff)])
    # ax2.xaxis.set_major_formatter(xfmt)

    # # plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Battery Power', 'Load Power', 'Grid Power', 'Tariff Rate'), loc='best')
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # # plt.savefig('opt_ex_tariff.png')
    plt.show()

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

    # color = 'tab:purple'
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Energy, kWh', color=color)
    # p4 = ax2.plot(times, bat_eng.value, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim([0,BAT_KWH])
    # ax2.xaxis.set_major_formatter(xfmt)
