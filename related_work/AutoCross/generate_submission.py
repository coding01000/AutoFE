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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_validate
import itertools
import scipy
import gc
from dataprocessing.house_price_data import get_data
from related_work.AutoCross.discretization.discretization import mul_granularity_discretization


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


def parsing_feature(feature):
    return feature.split('___')


model = LogisticRegression(
    penalty='l2',
    C=1.0,
    fit_intercept=True,
    random_state=432,
    solver='liblinear',
    max_iter=1000,
    # n_jobs=-1,
)

train, test = amazon()


target = "ACTION"
col4train = [x for x in train.columns if x != target]
y = train["ACTION"].values
train = train[col4train]
submit = pd.DataFrame()
submit["Id"] = test["id"]
test = test[col4train]
train = mul_granularity_discretization(train)
test = mul_granularity_discretization(test)
p = ['MGR_ID__3___ROLE_FAMILY_DESC__3', 'RESOURCE__3___MGR_ID__3___ROLE_FAMILY_DESC__3', 'ROLE_DEPTNAME__3___MGR_ID__3___RESOURCE__3___ROLE_FAMILY_DESC__3', 'ROLE_DEPTNAME__3___RESOURCE__3___ROLE_FAMILY__3___MGR_ID__3___ROLE_FAMILY_DESC__3', 'ROLE_DEPTNAME__3___ROLE_CODE__3___RESOURCE__3___ROLE_FAMILY__3___MGR_ID__3___ROLE_FAMILY_DESC__3', 'ROLE_DEPTNAME__3___ROLE_CODE__3___RESOURCE__3___ROLE_FAMILY__3___MGR_ID__3___ROLE_ROLLUP_2__3___ROLE_FAMILY_DESC__3', 'ROLE_DEPTNAME__3___ROLE_CODE__3___RESOURCE__3___ROLE_FAMILY__3___MGR_ID__3___ROLE_ROLLUP_2__3___ROLE_ROLLUP_1__3___ROLE_FAMILY_DESC__3', 'ROLE_DEPTNAME__3___ROLE_CODE__3___RESOURCE__3___ROLE_FAMILY__3___MGR_ID__3___ROLE_ROLLUP_2__3___ROLE_ROLLUP_1__3___ROLE_FAMILY_DESC__3___ROLE_TITLE__3', 'ROLE_DEPTNAME__3___ROLE_CODE__3___RESOURCE__3___ROLE_FAMILY__3___MGR_ID__3___ROLE_ROLLUP_2__3___RESOURCE__1___ROLE_ROLLUP_1__3___ROLE_FAMILY_DESC__3___ROLE_TITLE__3', 'ROLE_DEPTNAME__3___ROLE_CODE__3___RESOURCE__3___ROLE_FAMILY__3___MGR_ID__3___ROLE_ROLLUP_2__3___ROLE_ROLLUP_1__3___ROLE_FAMILY_DESC__3___RESOURCE__2___ROLE_TITLE__3']
if __name__ == '__main__':
    train_sets = None
    test_sets = None
    for f in p:
        features = parsing_feature(f)
        df1 = train[features[0]].apply(str)
        df2 = test[features[0]].apply(str)
        for i in range(1, len(features)):
            df1 += train[features[i]].apply(str)
            df2 += test[features[i]].apply(str)
        ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
        if train_sets is None:
            train_sets = ohe.fit_transform(df1.values.reshape(-1, 1))
            test_sets = ohe.transform(df2.values.reshape(-1, 1))
        else:
            train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(df1.values.reshape(-1, 1))))
            test_sets = scipy.sparse.hstack((test_sets, ohe.transform(df2.values.reshape(-1, 1))))

    model.fit(train_sets, y)
    predictions = model.predict_proba(test_sets)[:, 1]
    # predictions = model.predict(test_sets)
    # predictions = np.expm1(predictions)
    # print(predictions)
    submit["ACTION"] = predictions

    submit.to_csv("submission.csv", index=False)