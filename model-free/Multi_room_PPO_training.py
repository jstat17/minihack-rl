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
from gym.core import ObservationWrapper



class StandardizeObservation(ObservationWrapper):
    def __init__(self, env):
        super(StandardizeObservation, self).__init__(env)

    def observation(self, observation):
        # Remove 'inv_letters' key or add any other preprocessing you need
        if 'inv_letters' in observation:
            del observation['inv_letters']
        return observation

device = "cuda" if torch.cuda.is_available() else "cpu"
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)
models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
# Define training environments
training_envs = [
    "MiniHack-Room-5x5-v0", "MiniHack-Room-15x15-v0",
    "MiniHack-MazeWalk-9x9-v0", "MiniHack-MazeWalk-15x15-v0",
    "MiniHack-MazeWalk-45x19-v0"
]

# Hyperparameters
lr = 1e-3
gamma = 0.99
ent_coef = 0.01
TIMESTEPS = 250000

training_rewards = []

# Train the model on each training environment
for train_env_name in training_envs:
    env = StandardizeObservation(gym.make(train_env_name))

    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(0)
    
    model = PPO(
        "MultiInputPolicy", vec_env, 
        verbose=1,  
        tensorboard_log=logdir, 
        device=device,
        learning_rate=lr,
        gamma=gamma,
        ent_coef=ent_coef
    )
    
    print(f"Training on {train_env_name}")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

    # Evaluate on the training environment
    obs = vec_env.reset()
    episode_rewards = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = vec_env.step(action)
        episode_rewards += reward[0]
    training_rewards.append(episode_rewards)
model.save(f"{models_dir}/{TIMESTEPS}")
# Test the model on MiniHack-Quest-Easy-v0
test_env_name = "MiniHack-Quest-Easy-v0"
env = StandardizeObservation(gym.make(test_env_name))
vec_env = DummyVecEnv([lambda: env])
vec_env.seed(0)

obs = vec_env.reset()
test_rewards = 0
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = vec_env.step(action)
    test_rewards += reward[0]

# Plotting the average rewards
plt.figure(figsize=(10, 5))
plt.bar(training_envs, training_rewards, label="Training Environments", alpha=0.6)
plt.axhline(test_rewards, color='r', linestyle='dashed', linewidth=1, label="Test Environment")
plt.ylabel("Average Reward")
plt.title("Average Reward per Episode")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()