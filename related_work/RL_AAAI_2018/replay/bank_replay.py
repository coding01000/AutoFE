import sys

import scipy
from scipy import sparse
import os
sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from related_work.RL_AAAI_2018.env.BankEnv import BankEnv
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

# env = BankEnv(5)
# print('---start---')
# print(env.base_score)
# print(env.step(5))
# print(env.step(4))
# print(env.step(1))
# print(env.step(9))
# print(env.auc_score(env.action.X))
# # df, label = dataset.get_customer_satisfaction()
# print(env.action.data.columns.values)

features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact'
, 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
, 'empvarrate', 'conspriceidx', 'consconfidx', 'euribor3m', 'nremployed'
, 'marital___5', 'loan___5', 'duration___5', 'campaign___5', 'previous___5'
, 'conspriceidx___5', 'consconfidx___5', 'euribor3m___5', 'age___4'
, 'nremployed___4', 'consconfidx___5___4', 'euribor3m___5___4', 'housing___1'
, 'duration___1', 'campaign___1', 'pdays___1', 'previous___1', 'poutcome___1'
, 'empvarrate___1', 'consconfidx___1', 'euribor3m___1', 'nremployed___1'
, 'loan___5___1', 'duration___5___1', 'previous___5___1'
, 'consconfidx___5___1', 'age___4___1', 'nremployed___4___1'
, 'consconfidx___5___4___1', 'euribor3m___5___4___1']
df, label = dataset.get_bank()
columns = df.columns
train = None
action = BankEnv(5).action
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
print(evalution.auc_score(train, label))
