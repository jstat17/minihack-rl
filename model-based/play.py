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
import cv2
import pygame


def scale_observation(observation, new_size):
    """
    Scale an observation (image) to a new size using Pygame.
    Args:
        observation (pygame.Surface): The input Pygame observation.
        new_size (tuple): The new size (width, height) for scaling.
    Returns:
        pygame.Surface: The scaled observation.
    """
    return pygame.transform.scale(observation, new_size)

# Function to render the game observation
def render(obs, screen, font, text_color):
    """
    Render the game observation on the Pygame screen.
    Args:
        obs (dict): Observation dictionary containing "pixel" and "message" keys.
        screen (pygame.Surface): The Pygame screen to render on.
        font (pygame.Font): The Pygame font for rendering text.
        text_color (tuple): The color for rendering text.
    """
    img = obs["pixel"]
    msg = obs["message"]
    msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")
    rotated_array = np.rot90(img, k=-1)

    window_size = screen.get_size()
    image_surface = pygame.surfarray.make_surface(rotated_array)
    image_surface = scale_observation(image_surface, window_size)

    screen.fill((0, 0, 0))
    screen.blit(image_surface, (0, 0))

    text_surface = font.render(msg, True, text_color)
    text_position = (window_size[0] // 2 - text_surface.get_width() // 2, window_size[1] - text_surface.get_height() - 20)
    screen.blit(text_surface, text_position)
    pygame.display.flip()

    

def main(hyper_params: dict):
    # Create save folders and logs for trained models
    parent_folder = './videos/'

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
    
    video_save_path = os.path.join(new_folder_path, f"{hyper_params['env_names']}.mp4")
    
    # Q_weights_path = './runs/44/Q-weights-40.pth'
    Q_weights_path = './runs/44/Q-weights-70.pth'
    M_weights_path = './runs/44/M-weights-40.pth'
    
    agent = Agent(
        obs_shape = (3, 9, 9),
        obs_keys = ["pixel", "pixel_crop", "colors_crop", "chars_crop", "message", "tty_cursor"],
        obs_dtype = np.uint8,
        act_shape = len(hyper_params['env_available_actions']),
        batch_size = hyper_params['batch_size'],
        max_replay_buffer_len = hyper_params['max_replay_buffer_len'],
        priority_default = hyper_params['priority_default'],
        alpha = hyper_params['alpha'],
        beta = hyper_params['beta'],
        phi = hyper_params['phi'],
        c = hyper_params['c'],
        gamma = hyper_params['gamma'],
        lr_Q = hyper_params['lr_Q'],
        lr_M = hyper_params['lr_M'],
        lamb = hyper_params['lamb']
    )
    
    agent.Q_target.load_state_dict(th.load(Q_weights_path))
    agent.Q_learning.load_state_dict(th.load(Q_weights_path))
    agent.M.load_state_dict(th.load(M_weights_path))
    
    episode_rewards = [0.0]
    episode_average_action_value = [0.0]
    episode_steps = [0]
    
    env = gym.make(
        hyper_params['env_names'],
        observation_keys = tuple(agent.obs_keys),
        actions = hyper_params['env_actions'],
        reward_win = 50.0,
        reward_lose = -2.5,
        penalty_step = -0.5,
        max_episode_steps = 500,
        penalty_time = -0.05
    )
    max_reward = 50
    
    env_num_actions = hyper_params['env_action_spaces']
    
    # setup video
    frame_width = env.observation_space["pixel"].shape[1]
    frame_height = env.observation_space["pixel"].shape[0]
    pygame_frame_rate=1
    video_frame_rate=5
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_save_path), fourcc=fourcc, fps=video_frame_rate, frameSize=(frame_width, frame_height), apiPreference=cv2.CAP_FFMPEG)

    pygame.init()
    screen = pygame.display.set_mode((frame_width, frame_height))
    font = pygame.font.Font(None, 36)
    text_color = (255, 255, 255)
    
    # start agent
            
    state_dict = env.reset()
    clock = pygame.time.Clock()
    state = np.zeros(agent.obs_shape, agent.obs_dtype)
    state[0] = state_dict['colors_crop']
    state[1] = state_dict['chars_crop']
    state[2] = agent.inv
    
    # object available to be immediately picked up
    obj = agent.chars_to_obj(state_dict["message"])
    if obj:
        agent.update_inv(obj)
        state[2] = agent.inv
        for i, procedure_action in enumerate(agent.get_pickup_meta_action(obj)):
            if procedure_action == -1:
                _action = agent.tool_hotkeys[obj]
            else:
                _action = procedure_action
                
            _action = hyper_params["env_actions"].index(_action)
            _state_dict, _reward, _done, _info = env.step(_action)
            render(_state_dict, screen, font, text_color)

            # Capture the current frame and save it to the video
            pygame.image.save(screen, "temp_frame.png")
            frame = cv2.imread("temp_frame.png")
            out.write(frame)
            
            clock.tick(pygame_frame_rate)
            
            if _done:
                break
            
            if i == 0:
                msg = agent.get_text(_state_dict["message"])
                hotkey = ord(msg[0])
                agent.add_tool_hotkey(obj, hotkey)
    
    for t in range(1, hyper_params['total_steps']):
        state_tensor = th.tensor(agent.normalize_state(state), dtype=th.float32)[None, :]
        state_tensor = state_tensor.to(device)
            
        Q_values = agent.act(state_tensor)[:, :env_num_actions]
        action_idx = th.argmax(Q_values).item()
        action_value = th.max(Q_values).item()
        
        episode_average_action_value[-1] += action_value
            
            
        # agent takes real step in environment
        # handle meta actions
        action = hyper_params["env_available_actions"][action_idx]
        if type(action) == str:
            # print("meta")
            if action == "ZAP_META":
                obj = "wand"
            elif action == "APPLY_META":
                obj = "horn"
            
            action = agent.get_tool_meta_action(obj)
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
                    hyper_params['env_actions'].index(_action)
                )
            except ValueError:
                continue
        actions_to_do = tuple(a_temp)
        
        for action in actions_to_do:
            state_next_dict, reward, done, _info = env.step(action)
            render(state_next_dict, screen, font, text_color)

            # Capture the current frame and save it to the video
            pygame.image.save(screen, "temp_frame.png")
            frame = cv2.imread("temp_frame.png")
            out.write(frame)
            
            clock.tick(pygame_frame_rate)
            if done:
                break
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
            for i, procedure_action in enumerate(agent.get_pickup_meta_action(obj)):
                if procedure_action == -1:
                    _action = agent.tool_hotkeys[obj]
                else:
                    _action = procedure_action
                    
                # print(_action)
                # if type(_action) == int:
                #     print(chr(_action))
                _action = hyper_params["env_actions"].index(_action)
                _state_dict, _reward, _done, _info = env.step(_action)
                render(_state_dict, screen, font, text_color)

                # Capture the current frame and save it to the video
                pygame.image.save(screen, "temp_frame.png")
                frame = cv2.imread("temp_frame.png")
                out.write(frame)
                
                clock.tick(pygame_frame_rate)
                
                if _done:
                    break
                
                if i == 0:
                    msg = agent.get_text(_state_dict["message"])
                    # print("message: ", msg)
                    hotkey = ord(msg[0])
                    agent.add_tool_hotkey(obj, hotkey)
        
        old_state = state
        state = state_next
        episode_rewards[-1] += reward
        n_episodes = len(episode_rewards)
        
        
        # agent fails or wins level
        if done:
            out.release()  # Release the video writer
            cv2.destroyAllWindows()  # Close any OpenCV windows
            os.remove("temp_frame.png")  # Remove the temporary frame file
            break
            


if __name__ == "__main__":
    hyper_params = {
        'env_names': "MiniHack-MazeWalk-9x9-v0",
        # 'env_names': "MiniHack-Room-5x5-v0",
        # 'env_names': "MiniHack-LavaCross-Full-v0",
        # 'env_action_spaces': 6,
        'env_action_spaces': 4,
        'env_actions': tuple(actions.CompassCardinalDirection) + (actions.Command.PICKUP, actions.Command.QUAFF, actions.Command.PUTON, actions.Command.APPLY, actions.Command.ZAP, ord("f"), ord("g"), ord("r"), ord("y")),
        'env_available_actions': tuple(actions.CompassCardinalDirection) + ("ZAP_META", "APPLY_META"),
        'change_env_episode_freq': 100,
        'seed': np.random.randint(0, 2**32),
        'total_steps': int(1e6),
        'batch_size': 32,
        'max_replay_buffer_len': 10_000,
        'priority_default': 1e5,
        'alpha': 0.7,
        'beta': 0.7,
        'phi': 0.01,
        'c': 1e4,
        'gamma': 1,
        'lr_Q': 5e-5,#1e-4,
        # 'lr_Q': 5e-6,
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
        # 'steps_epsilon_end': 25_000,
        'steps_epsilon_end': 15_000,
        'print_episode_freq': 5,
        'save_episode_freq': 10_000
    }
    
    main(hyper_params)