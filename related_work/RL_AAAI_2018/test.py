from catboost.datasets import amazon
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from related_work.RL_AAAI_2018.env import AmazonEmployeeEvn
from catboost.datasets import amazon
from scipy.stats import pearsonr
from sklearn import preprocessing
from scipy import sparse
# 30872 0.233535

# RESOURCE

# p = [
#             lambda x: x.apply(np.log1p),
#             lambda x: x.apply(np.square),
#             lambda x: x.apply(np.sqrt),
#             lambda x: x.apply(preprocessing.scale),
#             lambda x: x.apply(preprocessing.minmax_scale),
#             lambda x: x.apply(np.sin),
#             lambda x: x.apply(np.cos),
#             lambda x: x.apply(np.tanh),
#             lambda x: x.apply(np.tanh),
#         ]
# df, _ = amazon()
# a = pearsonr(df[['RESOURCE']].values.reshape([-1, ]), df['ACTION'].values)
# print(a)
# action = 3
# columns = df.columns
# tmp2 = 0
# for i in columns:
#     tmp1 = p[action](df[[i]])
#     sparse.csr_matrix(tmp1.values)
#     tmp2 = scipy.sparse.hstack([tmp2, sparse.csr_matrix(tmp1.values)])
#     df[f'{i}_{action}'] = tmp1
#     print(i)
env = AmazonEmployeeEvn(9)

for i in range(9):
    env.step(i)
    print(f'{i}----{env.auc_score(env.action.X)}')

#
print(env.action.data.columns)
print(env.auc_score(env.action.X))

# print(env.base_score)
# print(env.action.X.shape)
# print(env.step(1))
# print(env.action.X.shape)
# print(env.step(0))
# print(env.action.X.shape)
# print(env.state)
# print(env.action.X.shape)
# print(env.auc_score(env.action.X))

# train, _ = amazon()
# print(train.shape)
# y = train['ACTION'].values
# X = train.drop('ACTION', axis=1).values
# clf = SelectFromModel(LinearSVC())
#
# X_new = clf.fit_transform(X, y)
# print(X_new.shape)