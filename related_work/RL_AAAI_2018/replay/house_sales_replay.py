import sys

import scipy
from scipy import sparse
import os
sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from related_work.RL_AAAI_2018.env.HouseSalesEnv import HouseSalesEnv
from related_work.Explorekit.evaluation import evalution
from utils import init_seed
from test_my_work import run_agent
from multiprocessing import Pool
from dataprocessing import dataset
import environment as env
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
init_seed.init_seed()
np.set_printoptions(threshold=np.inf)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# env = HouseSalesEnv(5)
# print('---start---')
# print(env.base_score)
# print(env.step(6))
# print(env.step(1))
# print(env.step(4))
# print(env.step(5))
# print(env.step(3))
# print(env.action.data.columns.values)
# print(env.get_score(env.action.X))
# df, label = dataset.get_house_sales()

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront'
, 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built'
, 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year'
, 'month', 'day', 'bathrooms___1', 'floors___1', 'waterfront___1', 'view___1'
, 'grade___1', 'lat___1', 'long___1', 'sqft_living15___1', 'year___1'
, 'month___1', 'day___1', 'waterfront___3', 'view___3', 'condition___3'
, 'grade___3', 'sqft_above___3', 'sqft_basement___3', 'yr_built___3'
, 'yr_renovated___3', 'zipcode___3', 'lat___3', 'long___3', 'sqft_living15___3'
, 'sqft_lot15___3', 'year___3', 'month___3', 'day___3', 'bathrooms___1___3'
, 'floors___1___3', 'waterfront___1___3', 'view___1___3', 'grade___1___3'
, 'lat___1___3', 'long___1___3', 'sqft_living15___1___3', 'year___1___3'
, 'month___1___3', 'day___1___3']
df, label = dataset.get_house_sales()
columns = df.columns
train = None
action = HouseSalesEnv(5).action
print('---start---')
for feature in features:
    print(feature)
    f = feature.split('___')
    tmp = df[[f[0]]]
    for i in range(1, len(f)):
        tmp = action.processing[int(f[i])](tmp)
    train = sparse.hstack([train, sparse.csr_matrix(tmp)])
print('---train---')
print(train.shape)
print(evalution.r2_score(train, label))
