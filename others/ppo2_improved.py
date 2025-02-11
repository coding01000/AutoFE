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

device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, transform_dim, feature_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.transform_dim = transform_dim
        self.feature_dim = feature_dim

        # actor
        self.transform_action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, transform_dim),
            nn.Softmax(dim=-1)
        )

        self.feature_action_layer = nn.Sequential(
            # nn.Linear(state_dim + transform_dim, n_latent_var),
            nn.Linear(state_dim + 1, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, feature_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def action_layer(self, state):
        # state = torch.from_numpy(state).float().to(device)
        transform_probs = self.transform_action_layer(state)
        t_dist = Categorical(transform_probs)
        transform = t_dist.sample()

        tmp = torch.zeros(1)
        tmp[0] = transform
        # state = state.resize_(len(state)+1)
        # state[-1] = transform
        state = torch.cat([state, tmp])

        if transform < self.transform_dim - 1:
            feature_probs = self.feature_action_layer(state)
            f_dist = Categorical(feature_probs)
            feature = f_dist.sample()
            action = transform * self.feature_dim + feature
            log_probs = t_dist.log_prob(transform) + f_dist.log_prob(feature)
        else:
            feature_probs = self.feature2_action_layer(state)
            f_dist = Categorical(feature_probs)
            features = f_dist.sample((2,))
            action = transform * self.feature_dim + features[0] * self.feature_dim + features[1]
            log_probs = t_dist.log_prob(transform) + f_dist.log_prob(features[0]) + f_dist.log_prob(features[1])

        return action, log_probs, t_dist, f_dist

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        memory.states.append(state)
        action, log_probs, t_dist, f_dist = self.action_layer(state)

        memory.actions.append(action)
        memory.logprobs.append(log_probs)

        return action.item()

    def evaluate(self, states, actions):
        action_logprobs = []
        dist_entropys = []
        for i in range(states.shape[0]):
            action, log_probs, t_dist, f_dist = self.action_layer(states[i])
            action = actions[i]
            if action < (self.transform_dim - 1) * self.feature_dim:
                transform = action / self.feature_dim
                feature = action % self.feature_dim
                action_logprobs.append(t_dist.log_prob(transform) + f_dist.log_prob(feature))
            else:
                transform = self.transform_dim - 1
                feature0 = (action - transform * self.feature_dim) / self.feature_dim
                feature1 = (action - transform * self.feature_dim) % self.feature_dim
                action_logprobs.append(t_dist.log_prob(torch.tensor(transform)) + f_dist.log_prob(feature0) + f_dist.log_prob(feature1))
            dist_entropys.append(t_dist.entropy() + f_dist.entropy())
        action_logprobs = torch.stack(action_logprobs)
        dist_entropys = torch.stack(dist_entropys)

        state_value = self.value_layer(states)

        return action_logprobs, torch.squeeze(state_value).double(), dist_entropys


class PPO:
    def __init__(self, state_dim, action_dim, feature_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, feature_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, feature_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.DoubleTensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class Action(object):
    def __init__(self, data, bounds):
        self.data = data
        self.feature_dim = data.shape[1]
        self.action_num = 2 + 1  # 处理方式数目 + feature cross
        # self.combinations = list(combinations([i for i in range(self.feature_nums)], 2))  # 特征组合
        # self.action_dim = self.feature_nums * self.action_num + len(self.combinations)
        self.action_dim = (self.action_num - 1) * self.feature_dim + self.feature_dim * self.feature_dim
        self.actions = [i for i in range(self.action_dim)]  # 动态搜索空间
        # self.selected = [] #所有已经处理过的action
        self.processed = {}  # 处理过后数据的存储
        self.episode_done = []  # 单次episode做过的action
        self.one_fe = []
        self.bounds = bounds
        self.ptr = 0

    def reset(self):
        self.ptr = 0
        self.actions = [i for i in range(self.action_dim)]
        self.episode_done = []
        self.one_fe = []

    def step(self, _action):
        action = np.zeros(2, dtype='int')

        is_combinations = _action >= (self.action_num - 1) * self.feature_dim
        if not is_combinations:
            action[0] = _action % self.feature_dim
            action[1] = _action / self.feature_dim
        else:
            action[0] = (_action - (self.action_num - 1) * self.feature_dim) / self.feature_dim
            action[1] = (_action - (self.action_num - 1) * self.feature_dim) % self.feature_dim
            action.sort()

        if _action not in self.actions:
            return True
        # if is_combinations and action[0] == action[1]:
        #     return True
        if (not is_combinations) and action[0] in self.one_fe:
            return True

        print(is_combinations, '.................', _action, '---------', action)

        self.episode_done.append(_action)

        if _action not in self.processed:
            if not is_combinations:
                column_name = self.data.columns[action[0]]
                if action[1] == 0:
                    tmp = (self.data[[column_name]] + 1).apply(np.log)
                    tmp = scipy.sparse.csr_matrix(tmp)
                elif action[1] == 1:
                    ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
                    tmp = ohe.fit_transform(self.data[[column_name]])
            else:
                c1 = self.data.columns[action[0]]
                c2 = self.data.columns[action[1]]
                tmp = self.data[c1].apply(str) + self.data[c2].apply(str)
                ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
                tmp = ohe.fit_transform(tmp.values.reshape(-1, 1))
            self.processed[_action] = tmp
        self.actions.remove(_action)
        # if is_combinations:
        #     self.actions.remove(_action)
        # else:
        if not is_combinations:
            self.one_fe.append(action[0])
        #
        #     for i in range(len(self.actions)):
        #         self.actions.remove(i)
        return self.isDone()

    def isDone(self):
        self.ptr += 1
        if self.ptr >= self.bounds:
            return True
        else:
            return False

    def data_sets(self):
        x = scipy.sparse.hstack([self.processed[i] for i in self.episode_done])
        return x


class AmazonEmployeeEvn(object):
    def __init__(self, bounds):
        data, _ = amazon()
        # data = pd.read_csv('train.csv')
        self.label = data['ACTION'].values
        data.drop('ACTION', axis=1, inplace=True)
        self.action = Action(data, bounds)
        self.base_score = self.auc_score(data.values)
        self.state_dim = self.action.action_dim
        self.state = np.zeros(self.state_dim)
        self.one_episode = []  # 存储action执行序列
        self.episodes = {}

    def reset(self):
        self.state = np.zeros(self.state_dim)
        self.one_episode = []
        self.action.reset()
        return self.state

    def step(self, _action):
        self.one_episode.append(_action)
        done = self.action.step(_action)
        if done:
            reward = self._reward()
        else:
            reward = 0
        self.state[_action] += 1
        return self.state, reward, done

    def _reward(self):
        if self.action.ptr < self.action.bounds:
            return -10
        one_episode = "".join(map(str, self.one_episode))
        if one_episode in self.episodes:
            return self.episodes[one_episode]
        self.episodes[one_episode] = self.auc_score(self.action.data_sets()) - self.base_score
        return self.episodes[one_episode]

    def auc_score(self, x):
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            fit_intercept=True,
            random_state=432,
            solver='liblinear',
            max_iter=1000,
            # n_jobs=-1,
        )
        y = self.label
        stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                               cv=5, return_train_score=True)
        return stats['test_score'].mean() * 100


def main():
    ############## Hyperparameters ##############
    env_name = "amazon_test"
    # creating environment
    env = AmazonEmployeeEvn(15)
    state_dim = env.state_dim
    action_dim = env.action.action_num
    feature_dim = env.action.feature_dim
    render = False
    solved_reward = 100  # stop training if avg_reward > solved_reward
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
    ppo = PPO(state_dim, action_dim, feature_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    # ppo.policy.load_state_dict(torch.load('./PPO_amazon_test_improve.pth'))
    # ppo.policy_old.load_state_dict(torch.load('./PPO_amazon_test_improve.pth'))

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
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            torch.save(i_episode, './PPO_{}.pth'.format('nn'))
            os.system('clear')
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


main()
