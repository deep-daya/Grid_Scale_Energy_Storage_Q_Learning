import pandas as pd
import numpy as np

from collections import defaultdict

class Residential():
    def __init__(self, df, gamma, state_dict):

        self.df = df
        
        # self.states = np.zeros(len(df), 4)  


        self.gamma = gamma

        # self.Q = np.zeros((len(self.states), len(self.actions)))
        # self.Q = defaultdict(lambda: np.zeros(len(self.actions))) 
        
        # print(self.Q)
        # self.policy = np.zeros(len(self.states))    

        self.LMP_bins = state_dict['LMP']
        self.Load_bins = state_dict['Load']
        self.TOU_bins = state_dict['TOU']
        # self.TOU_bins = [0.06475, 0.076155, 0.1254975, 0.0844, 0.0868675, 0.089075]
        # self.SOC_bins = []


        self.BAT_KWH_MIN = 0.1*14    # Minimum SOE of battery, 10% of rated
        self.BAT_KWH_MAX = 0.9*14    # Maximum SOE of battery, 90% of rated
        self.BAT_KW = 5
        self.HR_FRAC = 15/60  # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh
        

        # D means discharge the battery to help the utility, H means hold current battery energy
        self.actions = [self.LMP_buy_D, self.LMP_sell_D, self.wait_D, self.TOU_buy_D, self.TOU_sell_D, self.LMP_buy_H, self.LMP_sell_H, self.wait_H, self.TOU_buy_H, self.TOU_sell_H]
        #state parameterized by: LMP, TOU, load
        self.S = np.zeros([len(self.LMP_bins), len(self.TOU_bins), len(self.Load_bins)])
        self.Q = np.zeros([len(self.LMP_bins), len(self.TOU_bins), len(self.Load_bins), len(self.actions)])


    def get_allowed_actions(self, state):
        actions = []
        EnergyDiff = self.BAT_KW * self.HR_FRAC
        loss = -self.BAT_KW * self.HR_FRAC

        #if can discharge and doesn't put you under limit
        if state['SOC'] - EnergyDiff > self.BAT_KWH_MIN
        #can buy if it doesn't put you over the limit
        if state['SOC'] + EnergyDiff < self.BAT_KWH_MAX:
            actions.append(self.LMP_buy)
            actions.append(self.TOU_buy)

        #vice versa for sell
        if state['SOC'] - EnergyDiff > self.BAT_KWH_MIN:
            actions.append(self.LMP_sell_H)
            actions.append(self.TOU_sell_H)
            actions.append(self.LMP_buy_)
            actions.append(self.TOU_sell_H)
        

        
        actions.append(self.wait)

        return actions


    def LMP_buy_D(self, state):
        #buy 15 mins of power from LMP
        #kWh * $/kWh
        LMP_cost = -state['LMP'] 
        

        return LMP_cost * (self.BAT_KW * self.HR_FRAC), SOC


    def LMP_sell_D(self, state):
        #sell 15 mins of power to LMP
        #kWh * $/kWh
        LMP_comp = state['LMP'] 

        return LMP_comp * (self.BAT_KW * self.HR_FRAC)

    def TOU_buy_D(self, state):
        #buy 15 mins of power from TOU
        #kWh * $/kWh
        TOU_cost = -state['TOU'] 

        return TOU_cost * (self.BAT_KW * self.HR_FRAC)


    def TOU_sell_D(self, state):
        #sell 15 mins of power to TOU
        #kWh * $/kWh
        TOU_comp = state['TOU'] 

        return TOU_comp * (self.BAT_KW * self.HR_FRAC)

    def wait_D(self, state):
        #do nothing
        return 0

    def LMP_buy_H(self, state):
        #buy 15 mins of power from LMP
        #kWh * $/kWh
        LMP_cost = -state['LMP'] 

        return LMP_cost * (self.BAT_KW * self.HR_FRAC)


    def LMP_sell_H(self, state):
        #sell 15 mins of power to LMP
        #kWh * $/kWh
        LMP_comp = state['LMP'] 

        return LMP_comp * (self.BAT_KW * self.HR_FRAC)

    def TOU_buy_H(self, state):
        #buy 15 mins of power from TOU
        #kWh * $/kWh
        TOU_cost = -state['TOU'] 

        return TOU_cost * (self.BAT_KW * self.HR_FRAC)


    def TOU_sell_H(self, state):
        #sell 15 mins of power to TOU
        #kWh * $/kWh
        TOU_comp = state['TOU'] 

        return TOU_comp * (self.BAT_KW * self.HR_FRAC)

    def wait_H(self, state):
        #do nothing
        return 0

    def createEpsilonGreedyPolicy(self, Q, epsilon, num_actions): 
        """ 
        Creates an epsilon-greedy policy based 
        on a given Q-function and epsilon. 
        
        Returns a function that takes the state 
        as an input and returns the probabilities 
        for each action in the form of a numpy array  
        of length of the action space(set of possible actions). 
        """
        def policyFunction(LMP_ind, TOU_ind, Load_ind): 
    
            Action_probabilities = np.ones(num_actions, 
                    dtype = float) * epsilon / num_actions 
                    
            best_action = np.argmax(Q[LMP_ind][TOU_ind][Load_ind][:]) 
            Action_probabilities[best_action] += (1.0 - epsilon) 
            return Action_probabilities 
    
        return policyFunction 



    def Q_learning(self, epsilon):

        # Create an epsilon greedy policy function 
        # appropriately for environment action space 
        policy = self.createEpsilonGreedyPolicy(self.Q, epsilon, len(self.actions) * 2) 
        
        #starting SOC to be kept track of
        SOC = .5
        for ind, row in self.df.iterrows():
            
            LMP_ind = row["binned_LMP"]
            TOU_ind = row['binned_TOU']
            Load_ind = row['binned_Load']

            #  np.zeros([len(self.LMP_bins), len(self.TOU_bins), len(self.Load_bins), len(self.actions)])
            self.Q[LMP_ind][TOU_ind][Load_ind]

            # self.Q[s_ind, a_ind] = self.Q[s_ind, a_ind] + 0.2 * (
            #     row["r"] + self.gamma * max(self.Q[sp_ind, :]) - self.Q[s_ind, a_ind]
            # )
            # self.Q

            # get probabilities of all actions from current state 
            action_probabilities = policy(state) 
   
            # choose action according to  
            # the probability distribution 
            action = np.random.choice(np.arange( 
                      len(action_probabilities)), 
                       p = action_probabilities) 
   
            # take action and get reward, transit to next state 
            next_state, reward, done, _ = env.step(action) 
               
            # TD Update 
            best_next_action = np.argmax(Q[next_state])     
            td_target = reward + discount_factor * Q[next_state][best_next_action] 
            td_delta = td_target - Q[state][action] 
            Q[state][action] += alpha * td_delta 
   
            # done is True if episode terminated    
            if done: 
                break
                   
            state = next_state 


        print(self.Q)
        self.extract_policy()

    def extract_policy(self):
        for s in range(self.numStates):
            if not np.any(self.Q[s]):
                self.policy[s] = np.random.randint(1, self.numActions + 1)
            else:
                self.policy[s] = np.argmax(self.Q[s]) + 1
                


def main():
    df = pd.read_csv(r"Discretized_State_Space.csv")
    state_dict = pd.read_pickle(r'bins_Dict.pkl')

    print(df.head())
    # print(df2)
    a = Residential(df, .95, state_dict)
    # a = Residential()
    # a.Q_learning()


if __name__ == "__main__":
    main()
