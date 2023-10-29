__author__ = "John Statheros (1828326)"
__maintainer__ = "Timothy Reeder (455840)"
__version__ = '0.1.0'
__status__ = 'Development'

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
import pickle as pk


def linear_schedule(x_start: int,
                    x_end: int,
                    y_start: float,
                    y_end: float) -> Callable[[int], float]:
    """
    
    Args:
        x_start: 
        x_end: 
        y_start: 
        y_end: 

    Returns:

    """

    def linear(x: int, a: float, b: float) -> float:
        return a * x + b

    def schedule(x: int,
                 func: Callable[[int], float],
                 x_term: int, y_term: float) -> float:
        if x >= x_term:
            return y_term
        else:
            return func(x)

    a = (y_end - y_start) / (x_end - x_start)
    b = y_start - a * x_start
    decay_func = partial(linear, a=a, b=b)
    sched_func = partial(schedule, func=decay_func, x_term=x_end, y_term=y_end)

    return sched_func


def main(h_params: dict):
    """
    The main training function that will use DQN to train an agent
    
    Args:
        h_params: All the hyperparameters required to train the agent 

    Returns:

    """
    # Create save folders and logs for trained models
    parent_folder = './runs/'

    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    existing_folders = natsorted(os.listdir(parent_folder))

    if not existing_folders:
        new_folder_name = '01'
    else:
        last_folder_name = existing_folders[-1]
        last_folder_number = int(last_folder_name)

        new_folder_number = last_folder_number + 1
        new_folder_name = f'{new_folder_number:02d}'

    new_folder_path = os.path.join(parent_folder, new_folder_name)
    os.makedirs(new_folder_path)
    print(f'Created folder: {new_folder_path}')

    Q_weights_path = os.path.join(new_folder_path, "Q-weights-")
    M_weights_path = os.path.join(new_folder_path, "M-weights-")
    logs_path = os.path.join(new_folder_path, "logs.pkl")

    np.random.seed(h_params['seed'])

    obs_keys = ["pixel", "pixel_crop", "colors_crop", "chars_crop",
                "message", "tty_cursor"]

    agent = Agent(
        obs_shape=(3, 9, 9),
        obs_keys=obs_keys,
        obs_dtype=np.uint8,
        act_shape=len(h_params['env_available_actions']),
        batch_size=h_params['batch_size'],
        max_replay_buffer_len=h_params['max_replay_buffer_len'],
        priority_default=h_params['priority_default'],
        alpha=h_params['alpha'],
        beta=h_params['beta'],
        phi=h_params['phi'],
        c=h_params['c'],
        gamma=h_params['gamma'],
        lr_Q=h_params['lr_Q'],
        lr_M=h_params['lr_M'],
        lamb=h_params['lamb']
    )

    episode_rewards = [0.0]
    episode_average_action_value = [0.0]
    n_action_values = 0
    episode_average_Q_loss = [0.0]
    episode_average_M_loss = [0.0]
    n_Q_updates = 0
    n_M_updates = 0
    episode_steps = [0]
    episode_win = [0]

    switch_env = True
    env_counter = 0
    n_envs = len(h_params['env_names'])

    for t in range(1, h_params['total_steps']):
        if switch_env:
            # agent.replay_buffer.reset_buffer()
            if t > 10:
                h_params['epsilon_start'] *= 0.9

            epsilon_schedule = linear_schedule(
                x_start=t,
                x_end=t + h_params['steps_epsilon_end'],
                y_start=h_params['epsilon_start'],
                y_end=h_params['epsilon_end']
            )

            env = gym.make(h_params['env_names'][env_counter],
                           observation_keys=tuple(agent.obs_keys),
                           actions=h_params['env_actions'],
                           reward_win=50.0,
                           reward_lose=-2.5,
                           penalty_step=-0.5,
                           max_episode_steps=500,
                           penalty_time=-0.5)
            max_reward = 50

            env_num_actions = h_params['env_action_spaces'][env_counter]

            state_dict = env.reset()
            state = np.zeros(agent.obs_shape, agent.obs_dtype)
            state[0] = state_dict['colors_crop']
            state[1] = state_dict['chars_crop']
            state[2] = agent.inv

            # object available to be immediately picked up
            obj = agent.chars_to_obj(state_dict["message"])
            if obj:
                agent.update_inv(obj)
                state[2] = agent.inv
                for i, procedure_action in enumerate(
                        agent.get_pickup_meta_action(obj)):
                    if procedure_action == -1:
                        _action = agent.tool_hotkeys[obj]
                    else:
                        _action = procedure_action

                    _action = h_params["env_actions"].index(_action)
                    _state_dict, _reward, _done, _info = env.step(_action)
                    if _done:
                        break

                    if i == 0:
                        msg = agent.get_text(_state_dict["message"])
                        hotkey = ord(msg[0])
                        agent.add_tool_hotkey(obj, hotkey)

            switch_env = False
            n_wins = 0

        state_tensor = th.tensor(agent.normalize_state(state),
                                 dtype=th.float32)[None, :]
        state_tensor = state_tensor.to(device)

        # epsilon-greedy
        epsilon = epsilon_schedule(t)
        if np.random.random() < epsilon:
            action_idx = np.random.randint(0, env_num_actions)
        else:
            Q_values = agent.act(state_tensor)[:, :env_num_actions]
            # if t % 10 == 0:
            #     print(Q_values)
            action_idx = th.argmax(Q_values).item()
            action_value = th.max(Q_values).item()

            n_action_values += 1
            episode_average_action_value[-1] += action_value

        # agent takes real step in environment
        # handle meta actions
        action = h_params["env_available_actions"][action_idx]
        if type(action) == str:
            # print("meta")
            if action == "ZAP_META":
                obj = "wand"
            elif action == "APPLY_META":
                obj = "horn"

            action = agent.get_tool_meta_action(obj)
            # print(168, f"actions list: {action}")
            # print(169, agent.inv_objs)
        else:
            action = (action,)

        # convert available to index of all actions:
        a_temp = []
        for a in action:
            if a == -1:
                # print(agent.inv_objs)
                # print(agent.tool_hotkeys)
                _action = agent.tool_hotkeys[obj]
            else:
                _action = a

            # print(f"{_action = }")
            # if type(_action) == int:
            #     print(chr(_action))
            try:
                a_temp.append(
                    h_params['env_actions'].index(_action)
                )
            except ValueError:
                continue
        actions_to_do = tuple(a_temp)

        for action in actions_to_do:
            state_next_dict, reward, done, _info = env.step(action)
            if done:
                break
        if done and reward > 0.5:
            n_wins += 1
            episode_win[-1] = 1
            # print("win")
        episode_steps[-1] += 1

        state_next = np.zeros(agent.obs_shape, agent.obs_dtype)
        state_next[0] = state_next_dict['colors_crop']
        state_next[1] = state_next_dict['chars_crop']
        state_next[2] = agent.inv
        # if agent.inv > 0:
        #     print(state_next[2])

        # object available to be immediately picked up
        obj = agent.chars_to_obj(state_dict["message"])
        if obj:
            agent.update_inv(obj)
            state[2] = agent.inv
            for i, procedure_action in enumerate(
                    agent.get_pickup_meta_action(obj)):
                if procedure_action == -1:
                    _action = agent.tool_hotkeys[obj]
                else:
                    _action = procedure_action

                # print(_action)
                # if type(_action) == int:
                #     print(chr(_action))
                _action = h_params["env_actions"].index(_action)
                _state_dict, _reward, _done, _info = env.step(_action)
                if _done:
                    break

                if i == 0:
                    msg = agent.get_text(_state_dict["message"])
                    # print("message: ", msg)
                    hotkey = ord(msg[0])
                    agent.add_tool_hotkey(obj, hotkey)

        agent.replay_buffer.add_to_buffer(state=state,
                                          action=action_idx,
                                          reward=reward / max_reward,
                                          state_next=state_next,
                                          done=float(done))

        old_state = state
        state = state_next
        episode_rewards[-1] += reward
        n_episodes = len(episode_rewards)

        # if n_episodes > 50:
        #     print("display")
        #     print(action)
        #     plt.imshow(state_next_dict['pixel_crop'])
        #     plt.show()

        # agent fails or wins level
        if done:
            # switch envs periodically
            if n_envs > 1 and \
                    n_episodes % h_params['change_env_episode_freq'] == 0:
                switch_env = True
                env_counter = (env_counter + 1) % n_envs

                h_params['lr_Q'] *= 0.1
                h_params['lr_M'] *= 0.5
                agent.lr_Q = h_params['lr_Q']

            # otherwise reset current env
            else:
                state_dict = env.reset()
                agent.reset_inv()
                state = np.zeros(agent.obs_shape, agent.obs_dtype)
                state[0] = state_dict['colors_crop']
                state[1] = state_dict['chars_crop']
                state[2] = agent.inv

                agent.lr_Q = max(agent.lr_Q * 0.985, h_params['min_lr_Q'])
                for g in agent.Q_optimizer.param_groups:
                    g['lr'] = agent.lr_Q

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

            episode_steps.append(0)
            episode_win.append(0)

        if t > 1_000 and t % 2_000 == 0:
            pred_frame, pred_reward = agent.M(state_tensor,
                                              th.tensor([action_idx],
                                                        dtype=th.int32).to(
                                                  device))
            # print(f"{pred_reward = }")
            # print(f"dones = {(th.floor(th.sum(pred_frame, dim=(1, 2, 3)))).type(th.float32)}")
            # print(f"correct done = {done}")
            # print(f"{pred_frame = }")
            fig = plt.figure(figsize=(12, 6))
            ax = plt.subplot(1, 3, 1)
            ax.matshow(old_state[1])
            ax.set_title(f"Current state, action: {action_idx}")
            ax = plt.subplot(1, 3, 2)
            ax.matshow(state_next[1])
            ax.set_title("Next state")
            ax = plt.subplot(1, 3, 3)
            ax.matshow(pred_frame[0][1].to("cpu").detach().numpy())
            plt.tight_layout()
            plt.savefig(os.path.join(new_folder_path, f"img{t}.png"), dpi=100)
            plt.close('all')
            # plt.show()

        # Learning
        if t > h_params['learning_starts'] and \
                t % h_params['learning_steps_freq']:
            # get samples from buffer
            idxs, (states, actions, rewards, state_nexts,
                   dones) = agent.replay_buffer.sample(h_params['batch_size'])
            states = th.tensor(agent.normalize_state(states),
                               dtype=th.float32).to(device)
            actions = th.tensor(actions, dtype=th.int32).to(device)
            rewards = th.tensor(rewards, dtype=th.float32).to(device)
            state_nexts = th.tensor(agent.normalize_state(state_nexts),
                                    dtype=th.float32).to(device)
            dones = th.tensor(dones, dtype=th.float32).to(device)

            # optimize
            Q_loss = agent.optimise_Q_loss(state=states,
                                           action=actions,
                                           reward=rewards,
                                           state_next=state_nexts,
                                           done=dones)
            M_losses, M_loss = agent.optimise_M_loss(state=states,
                                                     action=actions,
                                                     reward=rewards,
                                                     state_next=state_nexts)

            # update buffer priorities
            agent.replay_buffer.update_priority(idxs=idxs,
                                                model_losses=M_losses)

            # logging
            n_Q_updates += 1
            episode_average_Q_loss[-1] += Q_loss.to("cpu").item()

            n_M_updates += 1
            episode_average_M_loss[-1] += M_loss.to("cpu").item()

        # Planning
        if t > h_params['planning_starts'] and \
                t % h_params['planning_steps_freq']:
            # get samples from buffer
            _idxs, (states, actions, rewards, _state_nexts,
                    _dones) = agent.replay_buffer.sample(h_params['batch_size'])

            states = th.tensor(agent.normalize_state(states),
                               dtype=th.float32).to(device)

            # random actions
            actions = th.tensor(np.random.randint(0, env_num_actions, size=(
            h_params['batch_size'], 1)), dtype=th.int32).to(device)

            # use model
            state_nexts, rewards = agent.M(states, actions)
            dones = (th.round(th.sum(state_nexts, dim=(1, 2, 3))) < 2).type(
                th.float32).to(device)

            Q_loss = agent.optimise_Q_loss(state=states,
                                           action=actions,
                                           reward=rewards,
                                           state_next=state_nexts,
                                           done=dones)
            n_Q_updates += 1
            episode_average_Q_loss[-1] += Q_loss.to("cpu").item()

        # Updating Q_target periodically
        if t > h_params['learning_starts'] and \
                t % h_params['update_Q_target_steps_freq'] == 0:
            agent.update_target_network()

        # printing logs
        if done and n_episodes % h_params['print_episode_freq'] == 0:
            freq = h_params['print_episode_freq']
            print(
                "-"*96)
            print(
                f"env: {h_params['env_names'][env_counter]} "
                f"steps: {t}\t"
                f"episodes: {n_episodes}\t"
                f"epsilon: {epsilon:.4f}\t"
                f"lr_Q: {agent.lr_Q:.4e}")
            print(
                f"Mean {freq} "
                f"episode reward: {np.mean(episode_rewards[-freq:]):.2f}")
            print(f"\t\t Q loss: {np.mean(episode_average_Q_loss[-freq:]):.6f}")
            print(f"\t\t M loss: {np.mean(episode_average_M_loss[-freq:]):.6f}")
            print(
                f"\t\t action value: {np.mean(episode_average_action_value[-freq:]):.6f}")
            print(
                f"\t\t steps per episode: {np.mean(episode_steps[-freq:]):.2f}")
            print(f"wins: {n_wins} / {freq}")
            print(
                "-"*96)
            n_wins = 0

        # periodically save DQN/M model weights and list of episode rewards
        if (t % h_params["save_episode_freq"] == 0):
            this_Q_model_weights_path = Q_weights_path + str(
                int(t / h_params["save_episode_freq"])) + ".pth"
            this_M_model_weights_path = M_weights_path + str(
                int(t / h_params["save_episode_freq"])) + ".pth"
            th.save(agent.Q_learning.state_dict(), this_Q_model_weights_path)
            th.save(agent.M.state_dict(), this_M_model_weights_path)

            logs = {
                'episode_rewards': episode_rewards,
                'episode_average_action_value': episode_average_action_value,
                'episode_average_Q_loss': episode_average_Q_loss,
                'episode_average_M_loss': episode_average_M_loss,
                'episode_steps': episode_steps,
                'episode_win': episode_win,
                'h_params': h_params
            }
            with open(logs_path, 'wb') as f:
                pk.dump(logs, f)

            print("-->Saved model weights + logs<--")


if __name__ == "__main__":
    env_names = ["MiniHack-Room-5x5-v0",
                 "MiniHack-LavaCross-Full-v0",
                 "MiniHack-MazeWalk-9x9-v0",
                 "MiniHack-Quest-Hard-v0"]

    env_actions = tuple(actions.CompassCardinalDirection) + \
        (actions.Command.PICKUP,
         actions.Command.QUAFF,
         actions.Command.PUTON,
         actions.Command.APPLY,
         actions.Command.ZAP,
         ord("f"), ord("g"), ord("r"), ord("y"))

    env_available_actions = tuple(actions.CompassCardinalDirection) \
        + ("ZAP_META", "APPLY_META")

    h_params = {
        'env_names': env_names,
        'env_action_spaces': [6, 6, 6, 6],
        'env_actions': env_actions,
        'env_available_actions': env_available_actions,
        'change_env_episode_freq': 100,
        'seed': np.random.randint(0, 2 ** 32),
        'total_steps': int(1e6),
        'batch_size': 32,
        'max_replay_buffer_len': 10_000,
        'priority_default': 1e5,
        'alpha': 0.7,
        'beta': 0.7,
        'phi': 0.01,
        'c': 1e4,
        'gamma': 1,
        'lr_Q': 5e-5,
        'min_lr_Q': 1e-8,
        'lr_M': 8e-4,
        'lamb': 0.2,
        'learning_starts': 1024,
        'update_Q_target_steps_freq': 2048,
        'learning_steps_freq': 3,
        'planning_starts': 150_000,
        'planning_batch_size': 512,
        'planning_steps_freq': 10,
        'epsilon_start': 1.0,
        'epsilon_end': 0.0,
        'steps_epsilon_end': 15_000,
        'print_episode_freq': 5,
        'save_episode_freq': 10_000
    }

    main(h_params)
