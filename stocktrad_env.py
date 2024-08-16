import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnv(gym.Env):
    """A stock trading environment for Gymnasium"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy, Sell, Hold
        self.action_space = spaces.Discrete(3)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, 5), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.df.index[self.current_step: self.current_step + 5], 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.df.index[self.current_step: self.current_step + 5], 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.df.index[self.current_step: self.current_step + 5], 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.df.index[self.current_step: self.current_step + 5], 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.df.index[self.current_step: self.current_step + 5], 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Transpose frame to have shape (5, 5)
        frame = frame.T

        return frame

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.df.index[self.current_step], "Open"], self.df.loc[self.df.index[self.current_step], "Close"])

        action_type = action

        if action_type == 0:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * random.uniform(0, 1))
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type == 1:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * random.uniform(0, 1))
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            kwargs.pop('seed')  # Remove the 'seed' argument if present but not used

        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')

def main():
    # Load your historical stock data into a pandas dataframe
    df = pd.read_csv("AAPL.csv", index_col='Date', parse_dates=True)

    # Create the stock trading environment
    env = StockTradingEnv(df)

    # Instantiate the agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=1)

    # Number of episodes to run
    episodes = 1

    # Loop through episodes
    for episode in range(episodes):
        # Reset the environment for a new episode
        obs = env.reset()

        # Loop through steps within the episode
        while True:
            # Render the environment (optional)
            env.render()

            # Predict action
            action, states = model.predict(obs)

            # Perform a step in the environment
            obs, rewards, done, info = env.step(action)

            # Check if the episode is finished
            if done:
                print("Episode finished!")
                break

    # Close the environment (optional)
    env.close()

if __name__ == "__main__":
    main()
