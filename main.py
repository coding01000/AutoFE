# -*- coding: utf-8 -*-

import numpy as np
import scipy
import gc
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from typing import Dict, List, Tuple
from itertools import combinations, permutations
from sklearn import linear_model
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate

from typing import Deque, Dict, List, Tuple
from collections import deque
import math

from catboost.datasets import amazon
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch
import torch.nn.functional as F
from PPO import Memory, PPO
from environment.HousePriceEnv import HousePriceEnv

device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    ############## Hyperparameters ##############
    env_name = "HousePrice"
    # creating environment
    env = HousePriceEnv(20)
    state_dim = env.state_dim
    action_dim = env.action.action_dim
    render = False
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 500000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    # ppo.policy.load_state_dict(torch.load('./model/PPO_HousePriceEnv.pth'))
    # ppo.policy_old.load_state_dict(torch.load('./model/PPO_HousePriceEnv.pth'))

    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # #........
    # cnt = 0
    # #........

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                print(i_episode, '---------------------', t, '---------', reward)
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval * solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
        #     break

        # logging
        if i_episode % log_interval == 0:
            avg_length = (avg_length / log_interval)
            running_reward = ((running_reward / log_interval))
            torch.save(ppo.policy.state_dict(), './model/PPO_{}.pth'.format(env_name))
            torch.save(i_episode, './model/PPO_{}.pth'.format('nn'))
            os.system('clear')
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


main()
