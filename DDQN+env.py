import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import random
from env1 import StockTradingEnv

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        self.value_stream = nn.Linear(32, 1)
        self.advantage_stream = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + (advantage - advantage.mean())
        return q_values


class Agent:
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        self.state_dim = state_dim
        self.action_dim = 2  # 0: Buy, 1: Sell 
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0  
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.995
        self.is_eval = is_eval
        
        self.model = DuelingDQN(state_dim, self.action_dim)
        self.target_model = DuelingDQN(state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        if self.is_eval:
            self.model.load_state_dict(torch.load(f'saved_models/{model_name}'))
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            action_type = random.randrange(self.action_dim) 
            amount = random.random()  
            return [action_type, amount]

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        action_type = torch.argmax(q_values[0]).item()
        amount = random.random()  
        return [action_type, amount]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = self.model(state)[0]

            if done:
                target[action[0]] = reward
            else:
                t = self.target_model(next_state)[0]
                next_action = torch.argmax(self.model(next_state)[0]).item()
                target[action[0]] = reward + self.gamma * t[next_action].item()
                
            target = target.unsqueeze(0)
            output = self.model(state)
            loss = self.loss_fn(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 2000

INITIAL_ACCOUNT_BALANCE = 10000

df = pd.read_csv('AAPL.csv')
df = df.sort_values('Date')
df.dropna(inplace=True)
df = df.reset_index(drop=True)

env = StockTradingEnv(df, render_mode='human')
state_dim = env.observation_space.shape[1] * env.observation_space.shape[0]
agent = Agent(state_dim=state_dim, balance=INITIAL_ACCOUNT_BALANCE)

episodes = 50
net_worths = []

for e in range(episodes):
    state, _ = env.reset()
    state = state.flatten()  
    done = False
    for time in range(MAX_STEPS):
        if time % 100 == 0:
            print(f"Time: {time} episode: {e+1}/{episodes} score: {env.net_worth}")
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = next_state.flatten()  
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            print(f'Episode {e+1}/{episodes} - Net Worth: {env.net_worth}')
            break
        agent.replay()
    net_worths.append(env.net_worth)
    if (e + 1) % 10 == 0:
        agent.save(f"model_ddqn_{e+1}.pth")
    print(f'Episode {e+1}/{episodes} - Net Worth: {env.net_worth}')

print(net_worths)

plt.plot(range(episodes), net_worths)
plt.xlabel('Episodes')
plt.ylabel('Net Worth')
plt.title('Net Worth over Episodes')
plt.show()

def evaluate(env, agent, episodes=10):
    total_rewards = 0
    for _ in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            total_rewards += reward
            state = next_state
    avg_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {avg_reward}")


agent.load("model_ddqn_50.pth")


evaluate(env, agent)
