__author__ = "Unathi Matu (440882)"
__maintainer__ = "Timothy Reeder (455840)"
__version__ = '0.1.0'
__status__ = 'Development'

import os
import minihack
from nle import nethack
from stable_baselines3 import PPO
from gym.core import ObservationWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = gym.make("MiniHack-MultiRoom-N6-OpenDoor-v0")
vec_env = DummyVecEnv([lambda: env])
#vec_env.seed(0)

# Hyperparameters to tune
learning_rates = [1e-4, 1e-3, 1e-2]
gammas = [0.9, 0.99, 0.999]
ent_coefs = [0.01, 0.001]

TIMESTEPS = 300_000

# Lists to store metrics
all_rewards = []
all_lengths = []

for lr in learning_rates:
    for gamma in gammas:
        for ent_coef in ent_coefs:
            # Create and train the model with the current hyperparameters
            model = PPO(
                "MultiInputPolicy", vec_env, 
                verbose=1,  
                tensorboard_log=logdir, 
                device=device,
                learning_rate=lr,
                gamma=gamma,
                ent_coef=ent_coef
            )
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

            # Evaluate the model and track metrics
            obs = vec_env.reset()
            episode_rewards = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _ = vec_env.step(action)
                episode_rewards += reward[0]
                episode_length += 1

            all_rewards.append(episode_rewards)
            all_lengths.append(episode_length)

# Plotting the metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(all_rewards)
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(all_lengths)
plt.title('Episode Length per Episode')
plt.xlabel('Episode')
plt.ylabel('Episode Length')

plt.tight_layout()
plt.show()
