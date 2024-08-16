import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import matplotlib.pyplot as plt
import datetime
from Train_DQN import *
from Preprocess import *
from build_Q import build_q_network

env = gym.make('Breakout-v4',render_mode='human')

train_dqn(env, num_episodes=50)