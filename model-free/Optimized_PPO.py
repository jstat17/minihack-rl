__author__ = "Unathi Matu (440882)"
__maintainer__ = "Timothy Reeder (455840)"
__version__ = '0.1.0'
__status__ = 'Development'

import os
import minihack
from nle import nethack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = gym.make("MiniHack-MultiRoom-N10-v0")
vec_env = DummyVecEnv([lambda: env])
vec_env.seed(0)

# Hyperparameters to tune
learning_rates = [ 1e-3]
gammas = [ 0.99]
ent_coefs = [0.01]

TIMESTEPS = 300_000

# Lists to store metrics
all_rewards = []
all_lengths = []
all_losses = []
all_entropies = []
all_value_estimates = []
all_actions = []


            
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
episode_actions = []
episode_values = []
episode_entropies = []

while not done:
  action, _ = model.predict(obs)
  #values = model.critic(obs).cpu().detach().numpy()
  #entropy = model.policy.entropy().mean().cpu().detach().numpy()
  
  obs, reward, done, _ = vec_env.step(action)
  episode_rewards += reward[0]
  episode_length += 1
  episode_actions.append(action[0])
  #episode_values.append(values[0])
  #episode_entropies.append(entropy)

all_rewards.append(episode_rewards)
all_lengths.append(episode_length)
all_actions.append(episode_actions)
# all_value_estimates.extend(episode_values)
#all_entropies.extend(episode_entropies)

# Extract training losses (this might vary based on your setup)
# all_losses.extend(model.get_vec_normalize_env().get_original_env().get_attr('get_logging_data')[0]['losses'])

# Plotting the metrics
plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.plot(all_rewards)
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')






# Uncomment when you have loss data
# plt.subplot(2, 3, 5)
# plt.plot(all_losses)
# plt.title('Loss Over Time')
# plt.xlabel('Update')
# plt.ylabel('Loss')

plt.tight_layout()
plt.show()
