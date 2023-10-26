import minihack
from agent import Agent, device
import torch as th
import gym
from nle.nethack import actions
import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from typing import Callable
from functools import partial



def linear_schedule(x_start: int, x_end: int, y_start: float, y_end: float) -> Callable[[int], float]:
    def linear(x: int, a: float, b: float) -> float:
        return a*x + b
    
    def schedule(x: int, func: Callable[[int], float], x_term: int, y_term: float) -> float:
        if x >= x_term:
            return y_term
        else:
            return func(x)
    
    a = (y_end - y_start) / (x_end - x_start)
    b = y_start - a*x_start
    decay_func = partial(linear, a=a, b=b)
    sched_func = partial(schedule, func=decay_func, x_term=x_end, y_term=y_end)
    
    return sched_func
    

def main(hyper_params: dict):
    # Create save folders and logs for trained models
    # parent_folder = './runs/'

    # if not os.path.exists(parent_folder):
    #     os.makedirs(parent_folder)

    # existing_folders = natsorted(os.listdir(parent_folder))

    # if not existing_folders:
    #     new_folder_name = '01'
    # else:
    #     last_folder_name = existing_folders[-1]
    #     last_folder_number = int(last_folder_name)
        
    #     new_folder_number = last_folder_number + 1
    #     new_folder_name = f'{new_folder_number:02d}'

    # new_folder_path = os.path.join(parent_folder, new_folder_name)
    # os.makedirs(new_folder_path)
    # print(f'Created folder: {new_folder_path}')
    
    # Q_weights_path = os.path.join(new_folder_path, "Q-weights-")
    # M_weights_path = os.path.join(new_folder_path, "M-weights-")
    # logs_path = os.path.join(new_folder_path, "logs.pkl")

    np.random.seed(hyper_params['seed'])
    
    agent = Agent(
        obs_shape = (4, 9, 9),
        obs_keys = ["pixel", "colors_crop", "chars_crop", "message"],
        obs_dtype = np.uint8,
        act_shape = len(hyper_params['env_actions']),
        batch_size = hyper_params['batch_size'],
        max_replay_buffer_len = hyper_params['max_replay_buffer_len'],
        priority_default = hyper_params['priority_default'],
        alpha = hyper_params['alpha'],
        beta = hyper_params['beta'],
        phi = hyper_params['phi'],
        c = hyper_params['c'],
        gamma = hyper_params['gamma'],
        lr = hyper_params['lr'],
        lamb = hyper_params['lamb']
    )
    
    episode_rewards = [0.0]
    episode_average_action_value = [0.0]
    n_action_values = 0
    episode_average_Q_loss = [0.0]
    episode_average_M_loss = [0.0]
    n_Q_updates = 0
    n_M_updates = 0
    
    switch_env = True
    env_counter = 0
    n_envs = len(hyper_params['env_names'])
    
    for t in range(1, hyper_params['total_steps']):
        if switch_env:
            epsilon_schedule = linear_schedule(
                x_start = t,
                x_end = t + hyper_params['steps_epsilon_end'],
                y_start = hyper_params['epsilon_start'],
                y_end = hyper_params['epsilon_end']
            )
            
            env = gym.make(
                hyper_params['env_names'][env_counter],
                observation_keys = tuple(agent.obs_keys),
                actions = hyper_params['env_actions'],
                reward_lose = -1.0
            )
            
            env_num_actions = hyper_params['env_action_spaces'][env_counter]
            
            state_dict = env.reset()
            state = np.zeros(agent.obs_shape, agent.obs_dtype)
            state[0] = state_dict['colors_crop']
            state[1] = state_dict['chars_crop']
            state[2] = state_dict['colors_crop']
            state[3] = state_dict['chars_crop']
            
            switch_env = False
            
        state_tensor = th.tensor(agent.normalize_state(state), dtype=th.float32)[None, :]
        state_tensor = state_tensor.to(device)
            
        # epsilon-greedy
        epsilon = epsilon_schedule(t)
        if np.random.random() < epsilon:
            action = np.random.randint(0, env_num_actions)
        else:
            Q_values = agent.act(state_tensor)[:, :env_num_actions]
            action = th.argmax(Q_values).item()
            action_value = th.max(Q_values).item()
            
            n_action_values += 1
            episode_average_action_value[-1] += action_value
            
        # agent takes real step in environment
        state_next_dict, reward, done, _info = env.step(action)
        
        state_next = np.zeros(agent.obs_shape, agent.obs_dtype)
        state_next[0] = state_next_dict['colors_crop']
        state_next[1] = state_next_dict['chars_crop']
        state_next[2] = state[0]
        state_next[3] = state[1]

        agent.replay_buffer.add_to_buffer(
            state = state,
            action = action,
            reward = reward,
            state_next = state_next,
            done = float(done)
        )
        
        state = state_next
        episode_rewards[-1] += reward
        n_episodes = len(episode_rewards)
        
        # agent fails or wins level
        if done:
            # switch envs periodically
            if n_envs > 1 and n_episodes % hyper_params['change_env_episode_freq'] == 0:
                switch_env = True
                env_counter  = (env_counter + 1) % n_envs
            
            # otherwise reset current env
            else:
                state_dict = env.reset()
                state = np.zeros(agent.obs_shape, agent.obs_dtype)
                state[0] = state_dict['colors_crop']
                state[1] = state_dict['chars_crop']
                state[2] = state_dict['colors_crop']
                state[3] = state_dict['chars_crop']
            
            # logging
            episode_rewards.append(0.0)
            
            if n_action_values > 0:
                episode_average_action_value[-1] /= n_action_values
            episode_average_action_value.append(0.0)
            n_action_values = 0
            
            if n_Q_updates != 0:
                episode_average_Q_loss[-1] /= n_Q_updates
            episode_average_Q_loss.append(0.0)
            n_Q_updates = 0
            
            if n_M_updates != 0:
                episode_average_M_loss[-1] /= n_M_updates
            episode_average_M_loss.append(0.0)
            n_M_updates = 0

        # Learning
        if t > hyper_params['learning_starts'] and t % hyper_params['learning_steps_freq']:
            # get samples from buffer
            idxs, (states, actions, rewards, state_nexts, dones) = agent.replay_buffer.sample(hyper_params['batch_size'])
            states = th.tensor(
                agent.normalize_state(states),
                dtype=th.float32
            ).to(device)
            actions = th.tensor(actions, dtype=th.int32).to(device)
            rewards = th.tensor(rewards, dtype=th.float32).to(device)
            state_nexts = th.tensor(
                agent.normalize_state(state_nexts),
                dtype=th.float32
            ).to(device)
            frame_nexts = state_nexts[:, 0:2, :, :].to(device)
            dones = th.tensor(dones, dtype=th.float32).to(device)
            
            # optimize
            Q_loss = agent.optimise_Q_loss(
                state = states,
                action = actions,
                reward = rewards,
                state_next = state_nexts,
                done = dones
            )
            M_losses, M_loss = agent.optimise_M_loss(
                state = states,
                action = actions,
                reward = rewards,
                frame_next = frame_nexts
            )
            
            # update buffer priorities
            agent.replay_buffer.update_priority(
                idxs = idxs,
                model_losses = M_losses
            )
            
            # logging
            n_Q_updates += 1
            episode_average_Q_loss[-1] += Q_loss.to("cpu").item()
            
            n_M_updates += 1
            episode_average_M_loss[-1] += M_loss.to("cpu").item()
            
        # Updating Q_target periodically
        if t > hyper_params['learning_starts'] and t % hyper_params['update_Q_target_steps_freq'] == 0:
            agent.update_target_network()
        
        # printing logs
        if done and n_episodes % hyper_params['print_episode_freq'] == 0:
            print("---------------------------------------------------------------------")
            print(f"steps: {t}\tepisodes: {n_episodes}\tepsilon: {epsilon:.4f}")
            print(f"Mean {hyper_params['print_episode_freq']} episode reward: {np.mean(episode_rewards[-10:]):.2f}")
            print("---------------------------------------------------------------------")
            
        # # periodically save DQN/M model weights and list of episode rewards
        # if (t % hyper_params["save_episode_freq"] == 0):
        #     this_model_weights_path = model_weights_path + str(int(t/hyper_params["save_freq"])) + ".pth"
        #     th.save(agent.DQN_learning.state_dict(), this_model_weights_path)
            
        #     logs = {
        #         'episode_rewards': episode_rewards,
        #         'episode_average_action_value': episode_average_action_value,
        #         'episode_average_loss': episode_average_loss,
        #         'hyper_params': hyper_params
        #     }
        #     with open(logs_path, 'wb') as f:
        #         pk.dump(logs, f)
                
        #     print("-->Saved model weights + logs<--")
        



if __name__ == "__main__":
    hyper_params = {
        'env_names': ["MiniHack-MazeWalk-9x9-v0"],
        'env_action_spaces': [4],
        'env_actions': tuple(actions.CompassCardinalDirection),
        'change_env_episode_freq': 32,
        'seed': 0,
        'total_steps': int(1e6),
        'batch_size': 256,
        'max_replay_buffer_len': int(5e3),
        'priority_default': 1e5,
        'alpha': 0.7,
        'beta': 0.7,
        'phi': 0.01,
        'c': 1e4,
        'gamma': 0.99,
        'lr': 1e-4,
        'lamb': 1.,
        'learning_starts': 1024,
        'update_Q_target_steps_freq': 1000,
        'learning_steps_freq': 32,
        'planning_starts': 20_000,
        'planning_batch_size': 512,
        'planning_steps_freq': 10,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'steps_epsilon_end': int(1e5),
        'print_episode_freq': 10,
        'save_episode_freq': 20_000
    }
    
    main(hyper_params)