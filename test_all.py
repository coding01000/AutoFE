from catboost.datasets import amazon
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
import itertools
import scipy
import gc

train, test = amazon()
train_sets = None
test_sets = None


target = "ACTION"
col4train = [x for x in train.columns if x != target]
y = train["ACTION"].values
train = train[col4train]
submit = pd.DataFrame()
submit["Id"] = test["id"]
test = test[col4train]


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


combs = [[2, 7], [0, 2], [1, 8], [0, 3], [0, 4], [6, 8], [7, 8], [0, 7], [1, 6], [2, 6], [3, 8], [1, 3], [6, 7], [5, 6]]
for i in combs:
    comb(i[0], i[1])

for i in range(9):
    if i == 2 or i == 8:
        cos(col4train[i])
        print(1)
    else:
        onehot(col4train[i])
        print(2)
# onehot(col4train[1])
# train.to_csv("train.csv", index=False)
# test.to_csv("test.csv", index=False)

model = LogisticRegression(
    penalty='l2',
    C=1.0,
    fit_intercept=True,
    random_state=432,
    solver='liblinear',
    max_iter=1000,
    n_jobs=-1,
)

del train
del test
gc.collect()
print('train...')
model.fit(train_sets, y)
predictions = model.predict_proba(test_sets)[:, 1]

submit["ACTION"] = predictions

submit.to_csv("submission.csv", index=False)
