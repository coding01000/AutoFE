import random

import numpy as np
import scipy
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from dataprocessing import dataset
import pandas as pd
from transform_function.transform_function import transform_list
from utils import init_seed


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

        if _action not in self.actions:
            return True

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
        self.actions.remove(_action)
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

    def replay(self, _action):
        action = np.zeros(2, dtype='int')
        is_combinations = _action / self.action_num >= self.feature_nums
        if not is_combinations:
            action[0] = _action / self.action_num
            action[1] = _action % self.action_num
        else:
            action[0] = self.combinations[_action - self.feature_nums * self.action_num - 1][0]
            action[1] = self.combinations[_action - self.feature_nums * self.action_num - 1][1]
        print(is_combinations, '..............', _action, f'---------[{action[0]}, {action[1]}], ')
        if not is_combinations:
            column_name = self.data.columns[action[0]]
            if action[1] is 1:
                is_combinations = True
                tmp = self.data[column_name]
            else:
                tmp = self.transform_list[action[1]](self.data[[column_name]])
                tmp = pd.Series(tmp.toarray().reshape(-1, ))
        else:
            c1 = self.data.columns[action[0]]
            c2 = self.data.columns[action[1]]
            tmp = self.data[c1].apply(str) + self.data[c2].apply(str)
        return tmp, is_combinations


class AmazonEnv(object):
    def __init__(self, bounds):
        data, self.label = dataset.get_amazon(False)
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
            # random_state=432,
            solver='liblinear',
            max_iter=1000,
            random_state=init_seed.get_seed(),
            # n_jobs=-1,
        )
        y = self.label
        stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                               cv=5, return_train_score=True)
        return stats['test_score'].mean() * 100

    def sample(self):
        return random.randint(0, self.action.action_dim - 1)
