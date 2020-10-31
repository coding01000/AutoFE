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
from sklearn import tree

from typing import Deque, Dict, List, Tuple
from collections import deque
import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch
import dataprocessing.house_price_data as hpd
import torch.nn.functional as F
import sys

from transform_function.transform_function import transform_list
from utils import init_seed

device = torch.device("cpu")     # "cuda:0" if torch.cuda.is_available() else "cpu"


class Action(object):
    def __init__(self, data, bounds):
        self.data = data
        self.feature_nums = data.shape[1]
        self.combinations = list(combinations([i for i in range(self.feature_nums)], 2))  # 特征组合
        self.transform_list = transform_list
        self.action_num = len(transform_list)  # 处理方式数目
        self.action_dim = self.feature_nums * self.action_num + len(self.combinations)
        self.actions = [i for i in range(self.action_dim)]  # 动态搜索空间
        # self.selected = [] #所有已经处理过的action
        self.processed = {}  # 处理过后数据的存储
        self.episode_done = []  # 单次episode做过的action
        self.bounds = bounds
        self.ptr = 0

    def reset(self):
        self.ptr = 0
        self.actions = [i for i in range(self.action_dim)]
        self.episode_done = []

    def step(self, _action):
        action = np.zeros(2, dtype='int')

        is_combinations = _action / self.action_num >= self.feature_nums
        if not is_combinations:
            action[0] = _action / self.action_num
            action[1] = _action % self.action_num
        else:
            action[0] = self.combinations[_action - self.feature_nums * self.action_num - 1][0]
            action[1] = self.combinations[_action - self.feature_nums * self.action_num - 1][1]

        print(is_combinations, '..............', _action, f'---------[{action[0]}, {action[1]}], ')

        # if _action not in self.actions:
        #     return True

        self.episode_done.append(_action)

        if _action not in self.processed:
            if not is_combinations:
                column_name = self.data.columns[action[0]]
                tmp = self.transform_list[action[1]](self.data[[column_name]])
            else:
                c1 = self.data.columns[action[0]]
                c2 = self.data.columns[action[1]]
                tmp = self.data[c1].apply(str) + self.data[c2].apply(str)
                ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
                tmp = ohe.fit_transform(tmp.values.reshape(-1, 1))
            self.processed[_action] = tmp
        # if is_combinations:
        # self.actions.remove(_action)
        # else:
        #     start = action[0] * self.action_num
        #     end = start + self.action_num
        #     for i in range(start, end):
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


class HousePriceEnv(object):
    def __init__(self, bounds):
        # data, _ = amazon()
        data, self.label = hpd.get_data(False)
        self.action = Action(data, bounds)
        self.base_score = self.rmse_score(data.values)
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
            return -sys.maxsize - 1
        one_episode = "".join(map(str, self.one_episode))
        if one_episode in self.episodes:
            return self.episodes[one_episode]
        self.episodes[one_episode] = (self.rmse_score(self.action.data_sets()) - self.base_score)
        return self.episodes[one_episode]

    def rmse_score(self, x):
        model = tree.DecisionTreeRegressor(random_state=init_seed.get_seed())
        # model = linear_model.LinearRegression()
        y = self.label
        stats = cross_validate(model, x, y, groups=None, scoring='r2',
                               cv=5, return_train_score=True)
        return stats['test_score'].mean() * 100

