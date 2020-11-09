import pandas as pd
import numpy as np

from collections import defaultdict


class Residential:
    def __init__(self, df, state_dict):

        self.df = df

        self.gamma = 0.95
        self.alpha = 0.2
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
        self.HR_FRAC = 15 / 60

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
                len(self.action_map),
            ]
        )
        self.Policy = np.zeros(
            [
                len(self.LMP_bins),
                len(self.TOU_bins),
                len(self.Load_bins),
                len(self.SOC_bins),
            ]
        )

    def get_allowed_actions(self, state):
        # [self.LMP_buy, self.LMP_sell, self.wait, self.TOU_buy, self.TOU_discharge]

        actions = []

        # can buy if it doesn't put you over the limit
        if state["SOC"] + 1 < len(self.TOU_bins):
            # actions.append(self.LMP_buy)
            # actions.append(self.TOU_buy)
            actions.append(0)
            actions.append(3)

        # vice versa for sell
        if state["SOC"] - 1 > 0:
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
        LMP_cost = self.LMP_bins[state["LMP"]]
        
        energy_change = self.BAT_KW * self.HR_FRAC

        return (self.LMP_Mavg - LMP_cost) * energy_change

    def LMP_sell(self, state):
        # sell 15 mins of power to LMP
        # kWh * $/kWh
        LMP_comp = self.LMP_bins[state["LMP"]]
        energy_change = self.BAT_KW * self.HR_FRAC

        return (LMP_comp - self.LMP_Mavg) * energy_change

    def TOU_buy(self, state):
        # buy 15 mins of power from TOU
        # kWh * $/kWh
        TOU_cost = -self.TOU_bins[state["TOU"]]
        energy_change = self.BAT_KW * self.HR_FRAC

        return TOU_cost * energy_change

    def TOU_discharge(self, state):
        # discharge 15 mins of power to TOU, offsetting load
        # kWh * $/kWh
        TOU_comp = self.TOU_bins[state["TOU"]]
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
            LMP_ind, TOU_ind, Load_ind, SOC_ind = (
                state["LMP"],
                state["TOU"],
                state["Load"],
                state["SOC"],
            )

            allowed = self.get_allowed_actions(state)

            num_allowed = len(allowed)
            Action_probabilities = [
                float(epsilon / num_allowed) if i in allowed else 0.0
                for i in range(num_actions)
            ]
        
            if all(Q[LMP_ind][TOU_ind][Load_ind][SOC_ind][:]):
                best_action = np.argmax(Q[LMP_ind][TOU_ind][Load_ind][SOC_ind][:])
            else:
                best_action = np.random.choice(allowed)

            Action_probabilities[best_action] += 1.0 - epsilon
            return Action_probabilities

        return policyFunction

    def Q_learning(self):

        # Create an epsilon greedy policy function
        # appropriately for environment action space
        policy = self.createEpsilonGreedyPolicy(
            self.Q, self.epsilon, len(self.action_map)
        )

        # initialize
        SOC_ind = 3
        n = 0
        for ind, row in self.df.iterrows():

            LMP_ind = row["binned_LMP"]
            TOU_ind = row["binned_TOU"]
            Load_ind = row["binned_Load"]

            state = {
                "LMP": LMP_ind,
                "TOU": TOU_ind,
                "Load": Load_ind,
                "SOC": SOC_ind,
            }

            # update
            n += 1
            self.LMP_Mavg = (self.LMP_Mavg + self.LMP_bins[LMP_ind]) / n

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(
                np.arange(len(action_probabilities)), p=action_probabilities
            )

            # take action and get reward, transit to next state
            next_state, reward, done = self.get_next(state, action)
            LMP_ind_new, TOU_ind_new, Load_ind_new, SOC_ind_new = (
                next_state["LMP"],
                next_state["TOU"],
                next_state["Load"],
                next_state["SOC"],
            )

            # TD Update

            # print(LMP_ind_new, TOU_ind_new, Load_ind_new, SOC_ind_new )
            allowed = self.get_allowed_actions(next_state)
            if all(self.Q[LMP_ind_new][TOU_ind_new][Load_ind_new][SOC_ind_new][:]):
                best_next_action = np.argmax(
                    self.Q[LMP_ind_new][TOU_ind_new][Load_ind_new][SOC_ind_new][:]
                )
            else:
                best_next_action = np.random.choice(allowed)
            td_target = (
                reward
                + self.gamma
                * self.Q[LMP_ind_new][TOU_ind_new][Load_ind_new][SOC_ind_new][
                    best_next_action
                ]
            )
            td_delta = td_target - self.Q[LMP_ind][TOU_ind][Load_ind][SOC_ind][action]
            self.Q[LMP_ind][TOU_ind][Load_ind][SOC_ind][action] += self.alpha * td_delta

            # update the policy with the action
            self.Policy[LMP_ind][TOU_ind][Load_ind][SOC_ind] = action

            # done is True if episode terminated
            if done:
                break

            state = next_state
            SOC_ind = SOC_ind_new

    def get_next(self, state, action):
        """
        Given starting state and action

        return: next state, reward, and boolean (done or not)
        """
        # self.action_map = {0: self.LMP_buy, 1: self.LMP_sell, 2: self.wait, 3: self.TOU_buy, 4: self.TOU_discharge}
        # no terminating states in this problem
        done = False

        reward = 0

        newstate = state.copy()
        # action can only alter the SOC of the state variables

        # if charged: increase SOC, keep load the same
        if action in [0, 3]:
            newstate["SOC"] += 1

        # if discharged TOU: decrease SOC, decrease load
        if action == 4:
            newstate["SOC"] -= 1
            load  = max(0, self.Load_bins[state["Load"]]-self.BAT_KW)
        else:
            load = self.Load_bins[state["Load"]]

        # if discharged LMP: decrease SOC
        if action == 1:
            newstate["SOC"] -= 1

        # if wait do nothing
        if action == 2:
            pass

        # reward is based on action + residual of load
        # load to kWh * TOU rate + action component
        action_component = self.action_map[action](state)
        reward = -load * self.HR_FRAC * self.TOU_bins[state["TOU"]] + action_component

        return newstate, reward, done

    def calc_revenue(self):
        revenue = 0
        SOC_ind = 3
        actions = []
        rewards = []
        for ind, row in self.df.iterrows():

            LMP_ind = row["binned_LMP"]
            TOU_ind = row["binned_TOU"]
            Load_ind = row["binned_Load"]

            state = {
                "LMP": LMP_ind,
                "TOU": TOU_ind,
                "Load": Load_ind,
                "SOC": SOC_ind,
            }
            # print(state)
            allowed = self.get_allowed_actions(state)
            # print(allowed)
            if int(self.Policy[LMP_ind][TOU_ind][Load_ind][SOC_ind]) in allowed:
                action = int(self.Policy[LMP_ind][TOU_ind][Load_ind][SOC_ind])
            else:
                action = np.random.choice(allowed)

            #make new state
            newstate = state.copy()

            # CALCULATE REWARD COMPONENTS
            LMP_cost = self.LMP_bins[state["LMP"]]
            TOU_cost = self.TOU_bins[state["TOU"]]
            energy_change = self.BAT_KW * self.HR_FRAC
            # self.action_map = {0: self.LMP_buy, 1: self.LMP_sell, 2: self.wait, 3: self.TOU_buy, 4: self.TOU_discharge}

            if action == 0:
                action_component = -LMP_cost * energy_change
            elif action == 1:
                action_component = LMP_cost * energy_change
            elif action == 2:
                action_component = 0
            elif action == 3:
                action_component = -TOU_cost * energy_change
            elif action == 4:
                action_component = TOU_cost * energy_change

            # CHANGE STATE
            # if charged: increase SOC, keep load the same
            if action in [0, 3]:
                newstate["SOC"] += 1

            # if discharged TOU: decrease SOC, decrease load
            if action == 4:
                newstate["SOC"] -= 1
                load  = max(0, self.Load_bins[state["Load"]]-self.BAT_KW)
            else:
                load = self.Load_bins[state["Load"]]

            # if discharged LMP: decrease SOC
            if action == 1:
                newstate["SOC"] -= 1

            # if wait do nothing
            if action == 2:
                pass

            # reward is based on action + residual of load
            # load to kWh * TOU rate + action component
            reward = -load * self.HR_FRAC * self.TOU_bins[state["TOU"]] + action_component

            # calc revenue and transition SOC
            revenue += reward
            SOC_ind = newstate["SOC"]

            #record
            actions.append(action)
            rewards.append(reward)

        print(revenue)
        self.write(actions, rewards)

    def write(self, actions, rewards):
        # for action in self.Policy:
        #     with open('out.policy', 'w') as f:
        #         f.write(action)


        action2string = {
            0: 'LMP_buy',
            1: 'LMP_sell',
            2: 'wait',
            3: 'TOU_buy',
            4: 'TOU_discharge',
        }

        df = pd.DataFrame(list(zip(actions, rewards)), 
               columns =['action_inds', 'rewards']) 
        
        df['actions'] = df['action_inds'].map(action2string)
        df['cumulative_revenue'] = np.cumsum(df['rewards'])

        df.to_csv('output.csv')

def main():
    df = pd.read_csv(r"Discretized_State_Space.csv")
    state_dict = pd.read_pickle(r"bins_Dict.pkl")

    a = Residential(df, state_dict)
    a.Q_learning()
    a.calc_revenue()




if __name__ == "__main__":
    main()
