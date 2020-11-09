import cvxpy as cp
import numpy as np
import pandas as pd
import datetime
import random
import pickle
import matplotlib.pyplot as plt

def script(filename):
    """Read Dataset and parse it into datetime, and respective LMP prices at each hour."""
    data_2 = pd.read_csv(filename)
    data_2.drop(["NODE_ID_XML","NODE_ID","NODE","MARKET_RUN_ID","PNODE_RESMRID","GRP_TYPE","POS","OPR_INTERVAL"],axis=1,inplace=True)
    data_2[data_2["LMP_TYPE"]=="LMP"]
    data_2["DATETIME"]=pd.to_datetime(data_2["INTERVALSTARTTIME_GMT"])
    data_2 = data_2[data_2["LMP_TYPE"]=="LMP"].sort_values("DATETIME")
    data_2.drop(["INTERVALSTARTTIME_GMT","INTERVALENDTIME_GMT","OPR_DT","OPR_HR","LMP_TYPE","XML_DATA_ITEM","GROUP"],axis=1,inplace=True)
    return data_2

def optimization_problem(data_frame):
    
    """Defines the optimization problem, and solves it for the maximum revenue along with saving the relevant result 
    as a dataframe and a plot."""
    #LMP Prices
    prices = data_frame["LMP_kWh"]

    #Initialize Variables for optimization Problem
    rate = cp.Variable((len(data_frame),1))
    E = cp.Variable((len(data_frame),1))
    
    #Create max, min for the 3 optimization variables 
    discharge_max = 5
    charge_max = -5
    SOC_max = 0.9*14
    SOC_min = 0.1*14
    
    #Initialize constraints and revenue
    constraints = []
    revenue = 0
    
    print("Starting Constraint Creation")
    #Create constraints for the each time step along with revenue.
    for i in range(len(data_frame)):
        if i%1000 == 0: print(i)
        constraints += [rate[i] <= discharge_max, #Rate should be lower than or equal to max rate,
                        rate[i] >= charge_max,
                        E[i]<= SOC_max, #Overall kW should be within the range of [SOC_min,SOC_max]
                        E[i] >= SOC_min]
        revenue += prices[i] *(rate[i]) #Revenue = sum of (prices ($/kWh) * (energy sold (kW) * 1hr - energy bought (kW) * 1hr) at timestep t)

    for i in range(1,len(data_frame)):
        if i%1000 == 0: print(i)
        constraints += [E[i] == E[i-1] + rate[i-1]] #Current SOC constraint

    constraints += [E[0] == random.uniform(SOC_min,SOC_max), rate[0] == 0] #create first time step constraints
    
    print("Solving problem")
    #Create Problem and solve to find Optimal Revenue and Times to sell.
    prob = cp.Problem(cp.Maximize(revenue),constraints)
    prob.solve(solver=cp.ECOS,verbose=True)
    print("Optimal Maximum Revenue is {0}".format(prob.value))
    
    #Convert values for the variables into arrays
    E_val = [E.value[i][0] for i in range(len(data_frame))]
    charge_val = [rate.value[i][0] for i in range(len(data_frame)) ]
    
    #Join values to the data frame
    data_frame["E"] = E_val
    data_frame["Charge"] = charge_val
    data_frame["DATETIME"] = data_frame.index
    revenue = [0]
    for i in range(1,len(data_frame)):
        revenue.append(revenue[-1] + prices[i]*charge_val[i])
        
    data_frame["Cumulative Additive Revenue"] = revenue
    print("Saving DF")
    #Save dataframe
    data_frame.to_csv("df_LMP.csv")
    print("Plotting 2 day timeline")
    #Plot dataframe for 2 days
    f = plt.figure(figsize=(20,20))
    data_frame.iloc[:48].plot(x="DATETIME",y=["E","Charge"])
    plt.xlabel("DateTime")
    plt.ylabel("Power- kW")
    plt.show()
    plt.savefig('2_day_battery_energy_arbitrage.png')
    
    return data_frame

def state_space_creation(data_frame,load_bins_number = 10, lmp_bins_number = 10):
    """Generates State Space for the problem containing TOU, LMP, Load, and SOC in both binned and unbinned formats."""
    #Use a 15 min resampled dataset for joining datasets and creating state space
    data_frame_2 = data_frame.resample('15T').pad()
    SOC_max = 0.9*14
    SOC_min = 0.1*14
    
    data_frame_2["MW"] = data_frame_2["MW"].apply(lambda x:x/1000)
    data_frame_2.rename(columns={"MW":"LMP_kWh"},inplace=True)
    
    ##Drop last value added for ease of parsing
    data_frame_2.drop(data_frame_2.iloc[len(data_frame_2)-1].name,axis=0,inplace=True)
    
    # For 15 min sampled data, create a new column to match indices to the load, tariff dataset from 2014.
    data_frame_2["date_month_2014"] = data_frame_2.index

    new_dateimt = []
    for i in data_frame_2["date_month_2014"]:
        if i.year == 2018:
            i = datetime.datetime(2014,i.month,i.day,i.hour,i.minute,i.second)
        else:
            i = datetime.datetime(2015,i.month,i.day,i.hour,i.minute,i.second)
        new_dateimt.append(i)

    data_frame_2["date_month_2014"]=  new_dateimt
    
    #Load in the load_tariff dataset
    data_sep = pd.read_csv("load_tariff.csv")
    
    #Convert data into relevant types
    data_sep["dt"] = pd.to_datetime(data_sep["dt"])
    data_sep["tariff"] = data_sep["tariff"].to_numpy()
    data_sep["solar"] = data_sep["solar"].to_numpy()
    data_sep["grid"] = data_sep["grid"].to_numpy()
    data_sep["gridnopv"] = data_sep["gridnopv"].to_numpy()
    
    #Parse data
    data_sep.drop("local_15min",axis=1,inplace=True)
    data_sep.drop("Unnamed: 0",axis=1,inplace=True)
        
    #Set index to be DateTime Index
    data_sep.set_index("dt",inplace=True)
    
    #Merge the 2 datasets on the 2014 datetime column
    df = pd.merge(data_frame_2, data_sep,left_on="date_month_2014",right_index=True)
    
    #Drop Column
    df.drop("date_month_2014",axis=1,inplace=True)
    df.drop("grid",axis=1,inplace=True)
    df.drop("solar",axis=1,inplace=True)
    
    df.rename(columns={"tariff":"TOU"},inplace=True)
    df.rename(columns={"gridnopv":"Load"},inplace=True)
    
    #Create Binned LMP, Load, TOU columns
    df["binned_LMP"],bins_LMP = pd.cut(df['LMP_kWh'], lmp_bins_number,labels = range(lmp_bins_number),retbins=True)
    df["binned_Load"],bins_Load = pd.cut(df["Load"],load_bins_number,labels=range(load_bins_number),retbins=True)
    
    #Create bins for SOC
    bins_SOC = []
    for i in np.arange(SOC_min,SOC_max,5*.25):
        bins_SOC.append(round(i,2))
    bins_SOC.append(SOC_max)
    
    #Create bins for TOU
    unique_TOU = {j:i for i,j in enumerate(df["TOU"].unique())}
    rows_TOU = []
    rows_Load = []
    for i, j in enumerate(df.iterrows()):
        rows_TOU.append(unique_TOU[j[1]["TOU"]])
    
    
    bins_TOU = df["TOU"].unique()

    #Create binned TOU to be mapping to indices
    df["binned_TOU"] = rows_TOU
    
    #Create mapping bins Dict
    bins_dict = {"LMP":bins_LMP,"Load":bins_Load,"TOU":bins_TOU,"SOC":bins_SOC}
    
    #Create SOC binned/unbinned column
    df["SOC"] = [5.15] + [0]*(len(df) -1)
    df["binned_SOC"] = [3] + [0]*(len(df)-1)
    
    #save csv and dictionary
    df.set_index(data_sep.index,drop=True,inplace=True)
    df.to_csv("Discretized_State_Space.csv")
    
    def save_obj(obj, name ):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    save_obj(bins_dict,"bins_Dict")
        


#Append the different datasets from each month into one dataset to range from July 2018 to June 2019
t = 0
filename_list = ["20190208_20190309_PRC_LMP_DAM_20201017_00_48_54_v1.csv",
                 "20190608_20190701_PRC_LMP_DAM_20201017_00_55_15_v1.csv",
                 "20190408_20190509_PRC_LMP_DAM_20201017_00_52_30_v1.csv",
                 "20190308_20190409_PRC_LMP_DAM_20201017_00_51_06_v1.csv",
                 "20181108_20181209_PRC_LMP_DAM_20201017_00_23_08_v1.csv",
                 "20180908_20181009_PRC_LMP_DAM_20201017_00_19_49_v1.csv",
                 "20190508_20190608_PRC_LMP_DAM_20201105_01_18_27_v1.csv",
                 "20180608_20180708_PRC_LMP_DAM_20201105_00_51_29_v1.csv",
                 "20180708_20180808_PRC_LMP_DAM_20201105_00_33_06_v1.csv",
                 "20180808_20180908_PRC_LMP_DAM_20201105_00_11_49_v1.csv",
                 "20181008_20181108_PRC_LMP_DAM_20201105_00_09_20_v1.csv",
                 "20181208_20190108_PRC_LMP_DAM_20201105_00_06_18_v1.csv",
                 "20190108_20190208_PRC_LMP_DAM_20201105_00_03_31_v1.csv"]
for i in filename_list:
    data_temp = script(i)
    if t==0:
        data_frame = data_temp
    else:
        data_frame = pd.concat([data_frame, data_temp], ignore_index=True)
    t += 1

#Drop Duplicates and reset index.
data_frame.drop_duplicates(inplace=True)
data_frame.reset_index(inplace=True,drop = True)

#Further Parse the datetime, and limit the dataset into the dates in the original load dataset.
datestime= []
for i in data_frame["DATETIME"]:
    if (i.date() < datetime.date(2019,7,1)) & (i.date() > datetime.date(2018,7,7)):
        datestime.append(str(i).replace("+00:00",""))
    else:
        data_frame.drop(data_frame.loc[data_frame["DATETIME"]==i].index.values[0],inplace=True)
#Convert index to DateTimeIndex.
data_frame["DATETIME"] = pd.to_datetime(datestime)

data_frame = data_frame.sort_values("DATETIME").reset_index(drop=True)

data_frame = data_frame.append({"DATETIME" :datetime.datetime(2019,7,1,0,0,0),"MW":42069},ignore_index=True)

data_frame.set_index("DATETIME",inplace=True)

state_space_creation(data_frame)

#Rename Column for both datsets
data_frame["MW"] = data_frame["MW"].apply(lambda x:x/1000)
data_frame.rename(columns={"MW":"LMP_kWh"},inplace=True)

##Drop last value added for ease of parsing
data_frame.drop(data_frame.iloc[len(data_frame)-1].name,axis=0,inplace=True)

print("Finished DF creation, starting optimization")
optimization_problem(data_frame)

