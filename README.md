# MiniHack RL Project
![minihack_pic](https://raw.githubusercontent.com/facebookresearch/minihack/main/docs/imgs/minihack.png)

<div align="center">

![Version](https://img.shields.io/badge/version-0.0.1-green)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-darkblue)
[![Discord](https://img.shields.io/badge/discord-blue)](https://discord.gg/f2MyUrHY)

</div>

In this repository a group of four students explore various methods in an 
attempt to get an agent to successfully complete tasks in the 
[minihack](https://github.com/facebookresearch/minihack) environment . The 
tasks chosen escalated in  difficulty and therefore processing/training time 
requirements. Starting  with getting the agent to move around the map 
without dying, and then  attempting to complete levels. This code is very 
much still in development and so, with taking a instruction from the name 
of the environment, getting  this code to work requires a mini hack. 

Two methods were explored: one model-based (DQN) and one model-free (PPO). 
More success was achieved from the model free method. The two methods are 
split into two folders and to train and then run needs to be done manually.

## Get Started
Please see the `requirements.txt` file to see what Python libraries are 
required to run this project. Or alternatively create a conda environment 
using `environment.yml`.

> Note: minihack should not be run on a Windows machine. If you only have 
> access to Windows, please set up a Docker container running a linux OS 
> since it will be much easier to install.

To run the model-based (DQN):
 - to train the agent: `python train.py`
 - the to watch the agent in action: `python play.py`

To run the model-free (PPO):
 - to train the agent: `python Multi_room_PPO_training.py`
 - the to watch the agent in action: `python video.py`
 
 ## Videos
 