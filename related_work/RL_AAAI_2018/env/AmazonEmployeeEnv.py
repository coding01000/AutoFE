from catboost.datasets import amazon
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


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
            lambda x: x.apply(np.tanh),
        ]
        self.action_dim = len(self.processing)

    def reset(self):
        self.ptr = 0
        self.data = self.row_data.copy()
        self.X = sparse.csr_matrix(self.row_data.values)

    def step(self, action):

        df = self.data.copy()
        columns = df.columns
        for i in columns:
            tmp1 = self.processing[action](df[[i]])
            tmp2 = scipy.sparse.hstack([self.X, sparse.csr_matrix(tmp1.values)])
            if self.feature_selection(tmp2):
                self.data[f'{i}_{1}'] = tmp1
                self.X = tmp2
        self.base_score = self.get_score(self.X)

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
        if score - self.base_score > 0.1:
            return True
        return False

    def get_score(self, x):
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            fit_intercept=True,
            random_state=432,
            solver='liblinear',
            max_iter=1000,
        )
        y = self.y
        stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                               cv=5, return_train_score=True)
        return stats['test_score'].mean() * 100


class AmazonEmployeeEvn(object):
    def __init__(self, bounds):
        data, _ = amazon()
        self.label = data['ACTION'].values
        data.drop('ACTION', axis=1, inplace=True)
        self.action = Action(data, bounds, self.label)
        self.base_score = self.auc_score(self.action.X)
        self.state_dim = self.action.action_dim
        self.state = np.zeros(self.state_dim)

    def reset(self):
        self.state = np.zeros(self.state_dim)
        self.action.reset()
        return self.state

    def step(self, _action):
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
        reward = self.auc_score(self.action.data_sets()) - self.base_score
        return reward

    def auc_score(self, x):
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            fit_intercept=True,
            random_state=432,
            solver='liblinear',
            max_iter=1000,
        )
        y = self.label
        stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                               cv=5, return_train_score=True)
        return stats['test_score'].mean() * 100

# if action == 0:
#     for i in columns:
#         tmp1 = df[[i]].apply(np.log1p)
#         tmp2 = scipy.sparse.hstack([self.X, sparse.csr_matrix(tmp1.values)])
#         if self.feature_selection(tmp2):
#             print(f'action0---------{i}')
#             self.data[i+'_1'] = tmp1
#             self.X = tmp2
# elif action == 1:
#     for i in columns:
#         tmp1 = df[[i]].apply(np.square)
#         tmp2 = scipy.sparse.hstack([self.X, sparse.csr_matrix(tmp1.values)])
#         if self.feature_selection(tmp2):
#             print(f'action0---------{i}')
#             self.data[i + '_1'] = tmp1
#             self.X = tmp2
#         # ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
#         # tmp = ohe.fit_transform(df[[i]])
#         # tmp = scipy.sparse.hstack([self.X, tmp])
#         # if self.feature_selection(tmp):
#         #     print(f'action1---------{i}')
#         #     self.X = tmp
# elif action == 2:
#     for i in columns:
#         tmp1 = df[[i]].apply(np.sqrt)
#         tmp2 = scipy.sparse.hstack([self.X, sparse.csr_matrix(tmp1.values)])
#         if self.feature_selection(tmp2):
#             print(f'action0---------{i}')
#             self.data[i + '_1'] = tmp1
#             self.X = tmp2
# elif action == 3:
#     for i in columns:
#         tmp1 = df[[i]].apply(preprocessing.scale)
#         tmp2 = scipy.sparse.hstack([self.X, sparse.csr_matrix(tmp1.values)])
#         if self.feature_selection(tmp2):
#             print(f'action0---------{i}')
#             self.data[i + '_1'] = tmp1
#             self.X = tmp2
# elif action == 4:
#     for i in columns:
#         tmp1 = df[[i]].apply(preprocessing.scale)
#         tmp2 = scipy.sparse.hstack([self.X, sparse.csr_matrix(tmp1.values)])
#         if self.feature_selection(tmp2):
#             print(f'action0---------{i}')
#             self.data[i + '_1'] = tmp1
#             self.X = tmp2