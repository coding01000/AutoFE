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
from sklearn import preprocessing
# 30872 0.233535

# RESOURCE

env = AmazonEmployeeEvn(9)

for i in range(9):
    print(f'{i}----{env.auc_score(env.action.X)}')
    env.step(i)

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