import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import matplotlib.pyplot as plt
import datetime
from Preprocess import *
from build_Q import *

env = gym.make('Breakout-v4',render_mode='human')

# Initialize parameters
input_shape = (80, 80, 1)
num_actions = env.action_space.n
memory_size = 100000
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.00025
target_update_freq = 1000
max_steps_per_episode = 1000 

q_network = build_q_network(input_shape, num_actions)
target_network = build_q_network(input_shape, num_actions)
target_network.set_weights(q_network.get_weights())

# Compile the model
q_network.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')

# Replay memory
memory = deque(maxlen=memory_size)

def add_experience(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def sample_experience(memory, batch_size):
    return random.sample(memory, batch_size)

env = gym.make('Breakout-v4',render_mode='human')

total_rewards = []
def train_dqn(env, num_episodes):
    global epsilon
    for episode in range(num_episodes):
        state = preprocess_frame(env.reset()[0])
        done = False
        total_reward = 0
        step = 0
        while not done and step < max_steps_per_episode:
            if episode % 10 == 0:
                env.render()
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                q_values = q_network.predict(state, verbose=0)
                action = np.argmax(q_values)

            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            total_reward += reward
            step += 1

            add_experience(memory, state, action, reward, next_state, done)
            state = next_state

            if len(memory) > batch_size:
                batch = sample_experience(memory, batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
                states = np.concatenate(states)
                next_states = np.concatenate(next_states)

                q_values = q_network.predict(states, verbose=0)
                target_q_values = target_network.predict(next_states, verbose=0)

                for i in range(batch_size):
                    if dones[i]:
                        q_values[i, actions[i]] = rewards[i]
                    else:
                        q_values[i, actions[i]] = rewards[i] + gamma * np.max(target_q_values[i])

                q_network.fit(states, q_values, epochs=1, verbose=0)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                epsilon = max(epsilon, epsilon_min)
            print(step)
            if step % target_update_freq == 0:
                target_network.set_weights(q_network.get_weights())
         

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")
