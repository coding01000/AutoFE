import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_validate
from itertools import combinations
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr
from dataprocessing import dataset
# from sklearn import tree
import random
from sklearn import preprocessing

from utils import init_seed


class Action(object):
    def __init__(self, data: pd.DataFrame, bounds, y):
        self.row_data = data
        self.data = self.row_data.copy()
        self.X = sparse.csr_matrix(data.values)
        self.y = y
        self.base_score = self.get_score(self.X)
        self.bounds = bounds
        self.ptr = 0
        self.processing = [
            lambda x: x.apply(np.log1p),
            lambda x: x.apply(np.square),
            lambda x: x.apply(np.sqrt),
            lambda x: x.apply(preprocessing.scale),
            lambda x: x.apply(preprocessing.minmax_scale),
            lambda x: x.apply(np.sin),
            lambda x: x.apply(np.cos),
            lambda x: x.apply(np.tanh),
            lambda x1, x2: x1 - x2,
            lambda x1, x2: x1 + x2,
            lambda x1, x2: x1 * x2,
            lambda x1, x2: x1 / x2,
        ]
        self.action_dim = len(self.processing)
        self.action_seq = []
        self.Xs = {}
        self.datas = {}

    def reset(self):
        self.ptr = 0
        self.data = self.row_data.copy()
        self.X = sparse.csr_matrix(self.row_data.values)
        self.base_score = self.get_score(self.X)
        self.action_seq = []

    def step(self, action):

        self.action_seq.append(action)
        actions = str(self.action_seq)
        if actions in self.Xs.keys():
            self.X = self.Xs[actions]
            self.base_score = self.get_score(self.X)
            self.data = self.datas[actions]
            return self.is_done()

        df = self.data
        columns = df.columns
        if action < 8:
            for i in columns:
                tmp1: pd.DataFrame = self.processing[action](df[[i]])
                tmp1 = tmp1.replace([np.inf, -np.inf], np.nan)
                if tmp1.isna().any().any():
                    return True
                new_col_name = f'{i}___{action}'
                tmp2 = scipy.sparse.hstack([self.X, sparse.csr_matrix(tmp1.values)])
                if new_col_name not in columns:
                    if self.feature_selection(tmp2):
                        df[new_col_name] = tmp1
                        self.X = tmp2
        else:
            combs = combinations(columns, 2)
            for i in combs:
                tmp1: pd.DataFrame = self.processing[action](df[[i[0]]], df[[i[1]]])
                tmp1 = tmp1.replace([np.inf, -np.inf], np.nan)
                if tmp1.isna().any().any():
                    return True
                new_col_name = f'{i}___{action}'
                tmp2 = scipy.sparse.hstack([self.X, sparse.csr_matrix(tmp1.values)])
                if new_col_name not in columns:
                    if self.feature_selection(tmp2):
                        df[new_col_name] = tmp1
                        self.X = tmp2
        self.base_score = self.get_score(self.X)
        self.Xs[actions] = self.X
        self.datas[actions] = self.data.copy()
        return self.is_done()

    def is_done(self):
        self.ptr += 1
        if self.ptr >= self.bounds:
            return True
        else:
            return False

    def data_sets(self):
        return self.X

    def feature_selection(self, features):
        score = self.get_score(features)
        if score - self.base_score > 1:
            return True
        return False

    def get_score(self, x):
        model = LinearRegression()
        y = self.y
        stats = cross_validate(model, x, y, groups=None, scoring='r2',
                               cv=5, return_train_score=True)
        return stats['test_score'].mean() * 100


class MarketEnv(object):
    def __init__(self, bounds):
        data, label = dataset.get_market_sales()
        self.label = label.values
        self.action = Action(data, bounds, self.label)
        self.base_score = self.get_score(self.action.X)
        self.state_dim = self.action.action_dim
        self.state = np.zeros(self.state_dim)

    def reset(self):
        self.state = np.zeros(self.state_dim)
        self.action.reset()
        self.base_score = self.get_score(self.action.X)
        return self.state

    def step(self, _action):

        done = self.action.step(_action)
        reward = self._reward()
        self.state[_action] += 1
        return self.state, reward, done

    def _reward(self):
        score = self.get_score(self.action.data_sets())
        reward = score - self.base_score
        self.base_score = max(self.base_score, score)
        return reward

    def get_score(self, x):
        model = LinearRegression()
        y = self.label
        stats = cross_validate(model, x, y, groups=None, scoring='r2',
                               cv=5, return_train_score=True)
        return stats['test_score'].mean() * 100

    def sample(self):
        return random.randint(0, self.action.action_dim - 1)
