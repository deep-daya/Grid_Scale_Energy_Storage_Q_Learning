import pandas as pd
import numpy as np



class Residential():
    def __init__(self, df, gamma):

        self.df = df
        
        self.actions = [LMP_buy, LMP_sell, wait, TOU_buy, TOU_sell]
        self.states = np.zeros(len(df), 4)  


        self.gamma = gamma

        self.Q = np.zeros((len(self.states), len(self.actions)))
        self.policy = np.zeros(len(self.states))    

    def Q_learning(self):

        for ind, row in self.df.iterrows():

            s_ind = row["s"] - 1
            a_ind = row["a"] - 1
            sp_ind = row["sp"] - 1

            self.Q[s_ind, a_ind] = self.Q[s_ind, a_ind] + 0.2 * (
                row["r"] + self.gamma * max(self.Q[sp_ind, :]) - self.Q[s_ind, a_ind]
            )

        print(self.Q)
        self.extract_policy()

    def extract_policy(self):
        for s in range(self.numStates):
            if not np.any(self.Q[s]):
                self.policy[s] = np.random.randint(1, self.numActions + 1)
            else:
                self.policy[s] = np.argmax(self.Q[s]) + 1
                


def main():
    df = pd.read_csv(r"")
    a = Residential()
    a.Q_learning()


if __name__ == "__main__":
    main()
