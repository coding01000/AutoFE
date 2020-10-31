import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from typing import Dict, List, Tuple
from IPython.display import clear_output
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_validate
import itertools
import scipy
import gc
from dataprocessing.house_price_data import get_data


# train = pd.read_csv('dataset/Bikeshare/train_unt.csv')
# test = pd.read_csv('dataset/Bikeshare/test_unt.csv')
# train_sets = None
# test_sets = None
#
#
# target = ['count', 'registered', 'casual']
# col4train = [x for x in train.columns if x not in target]
# y = train["count"].values
# train = train[col4train]
# submit = pd.DataFrame()
# submit["datetime"] = test["datetime"]
# test = test[col4train]
# train['datetime'] = pd.to_datetime(train['datetime']).dt.hour
# test['datetime'] = pd.to_datetime(test['datetime']).dt.hour

# train = pd.read_csv('./dataset/Titanic/train.csv')
# test = pd.read_csv('./dataset/Titanic/test.csv')
# train_sets = None
# test_sets = None
#
# target = ['PassengerId', 'Survived', 'Ticket', 'Name', 'Cabin']
# col4train = [x for x in train.columns if x not in target]
# y = train["Survived"].values
# train = train[col4train]
# submit = pd.DataFrame()
# submit["PassengerId"] = test["PassengerId"]
# test = test[col4train]
#
# em = LabelEncoder()
# sex = LabelEncoder()
# train['Embarked'] = em.fit_transform(train['Embarked'].astype(str))
# train['Sex'] = sex.fit_transform(train['Sex'])
# test['Embarked'] = em.transform(test['Embarked'].astype(str))
# test['Sex'] = sex.transform(test['Sex'])
#
# train = train.fillna(0)
# test = test.fillna(0)
train, test = get_data(True)
train_sets = None
test_sets = None
y = train["SalePrice"].apply(np.log1p).values
submit = pd.DataFrame()
submit["Id"] = test["Id"]
print(submit.shape)
train.drop('SalePrice', axis=1, inplace=True)
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
col4train = train.columns


def cos(column_name):
    global train_sets
    global test_sets
    train_sets = scipy.sparse.hstack((train_sets, scipy.sparse.csr_matrix(train[[column_name]].apply(np.cos).values)))
    test_sets = scipy.sparse.hstack((test_sets, scipy.sparse.csr_matrix(test[[column_name]].apply(np.cos).values)))


def log(column_name):
    global train_sets
    global test_sets
    train[column_name] = train[column_name] + 1
    test[column_name] = test[column_name] + 1
    train_sets = scipy.sparse.hstack((train_sets, scipy.sparse.csr_matrix(train[[column_name]].apply(np.log).values)))
    test_sets = scipy.sparse.hstack((test_sets, scipy.sparse.csr_matrix(test[[column_name]].apply(np.log).values)))
    # train_sets = scipy.sparse.hstack((train_sets, train[column_name].apply(np.log)))
    # test_sets = scipy.sparse.hstack((test_sets, test[column_name].apply(np.log)))


def onehot(column_name):
    global train_sets
    global test_sets
    ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
    train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(train[[column_name]])))
    test_sets = scipy.sparse.hstack((test_sets, ohe.transform(test[[column_name]])))


def comb(n1, n2):
    global train_sets
    global test_sets
    ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
    c1 = train.columns[n1]
    c2 = train.columns[n2]
    print('{}:{},{}:{}'.format(n1, c1, n2, c2))

    tmp1 = train[c1].apply(str) + train[c2].apply(str)
    tmp2 = test[c1].apply(str) + test[c2].apply(str)
    train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(tmp1.values.reshape(-1, 1))))
    test_sets = scipy.sparse.hstack((test_sets, ohe.transform(tmp2.values.reshape(-1, 1))))


# combs = [[2, 7], [0, 2], [1, 8], [0, 3], [0, 4], [6, 8], [7, 8], [0, 7], [1, 6], [2, 6], [3, 8], [1, 3], [6, 7], [5, 6]]
# employee amazon
# combs = [[2, 3], [3, 6], [3, 8], [0, 3], [5, 6], [1, 8], [1, 3], [1, 7], [4, 5], [1, 5], [0, 6], [0, 4], [1, 4], [4, 6]]
# bike share
# combs = [[0, 4], [1, 2], [1, 4], [0, 2], [3, 5], [6, 8], [5, 8], [3, 6], [7, 0], [0, 3], [1, 5], [2, 5]]
# Titanic
# combs = [[3, 5], [4, 5], [0, 5], [2, 4], [1, 2], [1, 4], [2, 6], [0, 1], [2, 3], [1, 5], [0, 3], [3, 4], [1, 6]]
# house price
combs = [[5, 24], [43, 46], [48, 50], [0, 39], [39, 65], [54, 56], [9, 27], [28, 62], [55, 60], [15, 23], [26, 52],
         [24, 47], [7, 10], [22, 77], [15, 30], [7, 35], [1, 5], [12, 29], [24, 25]]
#
for i in combs:
    comb(i[0], i[1])

l0 = []
l1 = [55]
for i in range(9):
    if i in l0:
        cos(col4train[i])
        print(1)
    elif i in l1:
        onehot(col4train[i])
        print(2)
# onehot(col4train[1])
# train.to_csv("train.csv", index=False)
# test.to_csv("test.csv", index=False)


from sklearn import tree
model = tree.DecisionTreeRegressor()
# from sklearn import linear_model
# model = linear_model.SGDRegressor()
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# model = GradientBoostingRegressor()


print('train...')
model.fit(train.values, y)
# predictions = model.predict_proba(test_sets)[:, 1]
predictions = model.predict(test.values)
predictions = np.expm1(predictions)
# print(predictions)
submit["SalePrice"] = predictions

submit.to_csv("submission.csv", index=False)
