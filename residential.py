import pandas as pd
import numpy as np

from collections import defaultdict


class Residential:
    def __init__(self, df, state_dict):

        self.df = df

        self.gamma = .95
        self.alpha = .2
        self.epsilon = 0.65
        self.LMP_Mavg = 0
        

        self.LMP_bins = state_dict["LMP"]
        self.Load_bins = state_dict["Load"]
        self.TOU_bins = state_dict["TOU"]
        self.SOC_bins = state_dict["SOC"]

        self.BAT_KWH_MIN = 0.1 * 14  # Minimum SOE of battery, 10% of rated
        self.BAT_KWH_MAX = 0.9 * 14  # Maximum SOE of battery, 90% of rated
        self.BAT_KW = 5
        # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh
        self.HR_FRAC = (
            15 / 60
        )  

        # D means discharge the battery to help the utility, H means hold current battery energy
        # THIS ORDER MATTERS
        self.action_map = {
            0: self.LMP_buy,
            1: self.LMP_sell,
            2: self.wait,
            3: self.TOU_buy,
            4: self.TOU_discharge,
        }

        # state parameterized by: LMP, TOU, load
        self.S = np.zeros(
            [
                len(self.LMP_bins),
                len(self.TOU_bins),
                len(self.Load_bins),
                len(self.SOC_bins),
            ]
        )
        self.Q = np.zeros(
            [
                len(self.LMP_bins),
                len(self.TOU_bins),
                len(self.Load_bins),
                len(self.SOC_bins),
                len(self.actions),
            ]
        )

    def get_allowed_actions(self, state):
        # [self.LMP_buy, self.LMP_sell, self.wait, self.TOU_buy, self.TOU_discharge]

        actions = []
        EnergyDiff = self.BAT_KW * self.HR_FRAC
        loss = -self.BAT_KW * self.HR_FRAC

        # can buy if it doesn't put you over the limit
        if state["SOC"] + EnergyDiff < self.BAT_KWH_MAX:
            # actions.append(self.LMP_buy)
            # actions.append(self.TOU_buy)
            actions.append(0)
            actions.append(3)

        # vice versa for sell
        if state["SOC"] - EnergyDiff > self.BAT_KWH_MIN:
            # actions.append(self.LMP_sell)
            # actions.append(self.TOU_discharge)
            actions.append(1)
            actions.append(4)

        # actions.append(self.wait)
        actions.append(2)

        return actions

    def LMP_buy(self, state):
        # buy 15 mins of power from LMP
        # kWh * $/kWh
        LMP_cost = state["LMP"]
        energy_change = self.BAT_KW * self.HR_FRAC

        return (self.LMP_Mavg - LMP_cost) * energy_change

    def LMP_sell(self, state):
        # sell 15 mins of power to LMP
        # kWh * $/kWh
        LMP_comp = state["LMP"]
        energy_change = self.BAT_KW * self.HR_FRAC

        return (LMP_comp - self.LMP_Mavg) * energy_change

    def TOU_buy(self, state):
        # buy 15 mins of power from TOU
        # kWh * $/kWh
        TOU_cost = -state["TOU"]
        energy_change = self.BAT_KW * self.HR_FRAC

        return TOU_cost * energy_change

    def TOU_discharge(self, state):
        # discharge 15 mins of power to TOU, offsetting load
        # kWh * $/kWh
        TOU_comp = state["TOU"]
        energy_change = self.BAT_KW * self.HR_FRAC

        return TOU_comp * energy_change

    def wait(self, state):
        # do nothing
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

        def policyFunction(state):
            LMP_ind, TOU_ind, Load_ind, SOC_ind = state

            allowed = self.get_allowed_actions(state)
            num_allowed = len(allowed)
            Action_probabilities = [
                float(epsilon / num_allowed) for i in range(num_actions) if i in allowed else 0.0
            ]
            # print(Action_probabilities)
            # Action_probabilities = np.ones(num_actions,
            #         dtype = float) * epsilon / num_actions

            best_action = np.argmax(Q[LMP_ind][TOU_ind][Load_ind][SOC_ind][:])
            # print(best_action)
            Action_probabilities[best_action] += 1.0 - epsilon
            return Action_probabilities

        return policyFunction

    def Q_learning(self, epsilon):

        # Create an epsilon greedy policy function
        # appropriately for environment action space
        policy = self.createEpsilonGreedyPolicy(self.Q, self.epsilon, len(self.action_map))

        #initialize
        SOC_ind = 3

        for ind, row in self.df.iterrows():
            #update
            self.LMP_Mavg 

            LMP_ind = row["binned_LMP"]
            TOU_ind = row["TOU"]
            Load_ind = row["binned_Load"]
            

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(
                np.arange(len(action_probabilities)), p=action_probabilities
            )

            # take action and get reward, transit to next state
            # next_state, reward, done, _ = env.step(action)?

            next_state, reward, done = self.get_next(state, action)

            # reward = action() + residual * state['TOU']
            # if

            # TD Update
            LMP_ind_new, TOU_ind_new, Load_ind_new, SOC_ind_new = next_state
            best_next_action = np.argmax(self.Q[LMP_ind_new][TOU_ind_new][Load_ind_new][SOC_ind_new][:])
            td_target = reward + self.gamma * Q[LMP_ind_new][TOU_ind_new][Load_ind_new][SOC_ind_new][best_next_action]
            td_delta = td_target - Q[LMP_ind][TOU_ind][Load_ind][SOC_ind][action]
            Q[LMP_ind][TOU_ind][Load_ind][SOC_ind][action] += self.alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            # state = next_state
            SOC_ind = SOC_ind_new

        # print(self.Q)
        # self.extract_policy()

    def get_next(self, state, action):
        """
        Given starting state and action

        return: next state, reward, and boolean (done or not)
        """
        # self.action_map = {0: self.LMP_buy, 1: self.LMP_sell, 2: self.wait, 3: self.TOU_buy, 4: self.TOU_discharge}
        # LMP_ind, TOU_ind, Load_ind, SOC_ind = state
        # no terminating states in this problem
        done = False

        reward = 0

        newstate = state.copy()
        # action can only alter the SOC of the state variables

        # if charged: increase SOC, keep load the same
        if action in [0, 3]:
            newstate["SOC_ind"] += 1

        # if discharged TOU: decrease SOC, decrease load
        if action == 4:
            newstate["SOC_ind"] -= 1
            newstate["load"] -= self.BAT_KW

        # if discharged LMP: decrease SOC
        if action == 1:
            newstate["SOC_ind"] -= 1

        # if wait do nothing
        if action == 2:
            pass

        # reward is based on action + residual of load
        # load to kWh * TOU rate + action component
        action_component = self.action_map[action](state)
        reward = -newstate['load'] * self.HR_FRAC * state['TOU'] + action_component


        return newstate, reward, done

    # def extract_policy(self):
    #     for s in range(self.numStates):
    #         if not np.any(self.Q[s]):
    #             self.policy[s] = np.random.randint(1, self.numActions + 1)
    #         else:
    #             self.policy[s] = np.argmax(self.Q[s]) + 1


def main():
    df = pd.read_csv(r"Discretized_State_Space.csv")
    state_dict = pd.read_pickle(r"bins_Dict.pkl")

    print(df.head())
    # print(df2)
    print(state_dict)
    a = Residential(df, 0.95, state_dict)
    # a = Residential()
    # a.Q_learning()


if __name__ == "__main__":
    main()
