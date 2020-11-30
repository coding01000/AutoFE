import sys

import scipy
from scipy import sparse
import os
sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from related_work.RL_AAAI_2018.env.CreditEnv import CreditEnv
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

# env = CreditEnv(5)
# print('---start---')
# print(env.base_score)
# print(env.step(6))
# print(env.step(1))
# print(env.step(3))
# print(env.step(10))
# print(env.auc_score(env.action.X))
# # df, label = dataset.get_customer_satisfaction()
# print(env.action.data.columns.values)
# print(1)
# print(2)
# ['LIMIT_BAL' 'SEX' 'EDUCATION' 'MARRIAGE' 'AGE' 'PAY_0' 'PAY_2' 'PAY_3'
#  'PAY_4' 'PAY_5' 'PAY_6' 'BILL_AMT1' 'BILL_AMT2' 'BILL_AMT3' 'BILL_AMT4'
#  'BILL_AMT5' 'BILL_AMT6' 'PAY_AMT1' 'PAY_AMT2' 'PAY_AMT3' 'PAY_AMT4'
#  'PAY_AMT5' 'PAY_AMT6' 'EDUCATION___6' 'MARRIAGE___6' 'PAY_2___6'
#  'PAY_3___6' 'PAY_4___6' 'PAY_5___6' 'PAY_6___6' 'BILL_AMT1___6'
#  'BILL_AMT2___6' 'BILL_AMT3___6' 'BILL_AMT4___6' 'BILL_AMT5___6'
#  'BILL_AMT6___6' 'PAY_AMT1___6' 'PAY_AMT2___6' 'PAY_AMT5___6'
#  'PAY_AMT6___6' 'LIMIT_BAL___3' 'EDUCATION___3' 'AGE___3' 'PAY_0___3'
#  'PAY_2___3' 'PAY_3___3' 'PAY_4___3' 'PAY_5___3' 'PAY_6___3'
#  'BILL_AMT2___3' 'BILL_AMT3___3' 'BILL_AMT5___3' 'BILL_AMT6___3'
#  'PAY_AMT1___3' 'PAY_AMT2___3' 'PAY_AMT3___3' 'PAY_AMT4___3'
#  'PAY_AMT6___3' 'EDUCATION___6___3' 'MARRIAGE___6___3' 'PAY_2___6___3'
#  'PAY_3___6___3' 'PAY_4___6___3' 'PAY_5___6___3' 'PAY_6___6___3'
#  'BILL_AMT1___6___3' 'BILL_AMT2___6___3' 'BILL_AMT3___6___3'
#  'BILL_AMT4___6___3' 'BILL_AMT5___6___3' 'BILL_AMT6___6___3'
#  'PAY_AMT1___6___3' 'PAY_AMT2___6___3' 'PAY_AMT5___6___3'
#  'PAY_AMT6___6___3' 'SEX___3' 'MARRIAGE___3' 'BILL_AMT1___3'
#  'BILL_AMT4___3' 'PAY_AMT5___3' 'PAY_0___3___3' 'PAY_2___3___3'
#  'PAY_4___3___3' 'PAY_5___3___3' 'BILL_AMT2___3___3' 'BILL_AMT5___3___3'
#  'PAY_AMT1___3___3' 'PAY_AMT2___3___3' 'PAY_AMT3___3___3'
#  'PAY_AMT4___3___3' 'PAY_4___6___3___3' 'PAY_5___6___3___3'
#  'PAY_6___6___3___3' 'BILL_AMT1___6___3___3' 'BILL_AMT2___6___3___3'
#  'BILL_AMT3___6___3___3' 'BILL_AMT4___6___3___3' 'BILL_AMT5___6___3___3'
#  'BILL_AMT6___6___3___3' 'PAY_AMT2___6___3___3' 'PAY_AMT5___6___3___3'
#  'LIMIT_BAL___7' 'SEX___7' 'EDUCATION___7' 'MARRIAGE___7' 'PAY_2___7'
#  'PAY_3___7' 'PAY_6___7' 'BILL_AMT1___7' 'BILL_AMT2___7' 'BILL_AMT4___7'
#  'BILL_AMT5___7' 'PAY_AMT1___7' 'PAY_AMT2___7' 'PAY_AMT3___7'
#  'PAY_AMT4___7' 'PAY_AMT5___7' 'PAY_AMT6___7' 'PAY_2___6___7'
#  'PAY_3___6___7' 'PAY_4___6___7' 'PAY_6___6___7' 'BILL_AMT1___6___7'
#  'BILL_AMT2___6___7' 'BILL_AMT3___6___7' 'BILL_AMT4___6___7'
#  'BILL_AMT6___6___7' 'PAY_AMT2___6___7' 'PAY_AMT6___6___7'
#  'LIMIT_BAL___3___7' 'EDUCATION___3___7' 'AGE___3___7' 'PAY_0___3___7'
#  'PAY_2___3___7' 'PAY_3___3___7' 'PAY_4___3___7' 'PAY_5___3___7'
#  'BILL_AMT2___3___7' 'BILL_AMT3___3___7' 'BILL_AMT5___3___7'
#  'BILL_AMT6___3___7' 'EDUCATION___6___3___7' 'MARRIAGE___6___3___7'
#  'PAY_2___6___3___7' 'PAY_3___6___3___7' 'PAY_4___6___3___7'
#  'PAY_5___6___3___7' 'BILL_AMT1___6___3___7' 'BILL_AMT2___6___3___7'
#  'BILL_AMT3___6___3___7' 'BILL_AMT6___6___3___7' 'PAY_AMT1___6___3___7'
#  'PAY_AMT2___6___3___7' 'SEX___3___7' 'BILL_AMT1___3___7'
#  'BILL_AMT4___3___7' 'PAY_AMT5___3___7' 'PAY_0___3___3___7'
#  'PAY_2___3___3___7' 'PAY_4___3___3___7' 'PAY_5___3___3___7'
#  'BILL_AMT2___3___3___7' 'BILL_AMT5___3___3___7' 'PAY_AMT1___3___3___7'
#  'PAY_AMT2___3___3___7' 'PAY_AMT4___3___3___7' 'PAY_4___6___3___3___7'
#  'PAY_5___6___3___3___7' 'PAY_6___6___3___3___7'
#  'BILL_AMT1___6___3___3___7' 'BILL_AMT4___6___3___3___7'
#  'BILL_AMT5___6___3___3___7' 'BILL_AMT6___6___3___3___7'
#  'PAY_AMT2___6___3___3___7' 'PAY_AMT5___6___3___3___7']
# print(evalution.auc_score(env.action.data_sets(), label))
# df, label = dataset.get_customer_satisfaction()
features = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3'
,'PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4'
,'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4'
,'PAY_AMT5','PAY_AMT6','EDUCATION___6','MARRIAGE___6','PAY_2___6'
,'PAY_3___6','PAY_4___6','PAY_5___6','PAY_6___6','BILL_AMT1___6'
,'BILL_AMT2___6','BILL_AMT3___6','BILL_AMT4___6','BILL_AMT5___6'
,'BILL_AMT6___6','PAY_AMT1___6','PAY_AMT2___6','PAY_AMT5___6'
,'PAY_AMT6___6','SEX___1','EDUCATION___1','MARRIAGE___1','AGE___1'
,'PAY_0___1','PAY_2___1','PAY_3___1','PAY_4___1','PAY_5___1','PAY_6___1'
,'EDUCATION___6___1','MARRIAGE___6___1','PAY_2___6___1','PAY_3___6___1'
,'PAY_4___6___1','PAY_5___6___1','PAY_6___6___1','BILL_AMT1___6___1'
,'BILL_AMT2___6___1','BILL_AMT3___6___1','BILL_AMT4___6___1'
,'BILL_AMT6___6___1','PAY_AMT1___6___1','PAY_AMT2___6___1'
,'PAY_AMT5___6___1','PAY_AMT6___6___1','LIMIT_BAL___3','EDUCATION___3'
,'AGE___3','PAY_0___3','PAY_2___3','PAY_3___3','PAY_4___3','PAY_5___3'
,'PAY_6___3','BILL_AMT1___3','BILL_AMT2___3','BILL_AMT3___3'
,'BILL_AMT4___3','BILL_AMT5___3','BILL_AMT6___3','PAY_AMT2___3'
,'PAY_AMT3___3','PAY_AMT4___3','PAY_AMT5___3','PAY_AMT6___3'
,'PAY_2___6___3','PAY_3___6___3','PAY_4___6___3','BILL_AMT1___6___3'
,'BILL_AMT3___6___3','BILL_AMT5___6___3','BILL_AMT6___6___3'
,'PAY_AMT1___6___3','PAY_AMT2___6___3','PAY_AMT5___6___3'
,'PAY_AMT6___6___3','SEX___1___3','EDUCATION___1___3','MARRIAGE___1___3'
,'AGE___1___3','PAY_0___1___3','PAY_2___1___3','PAY_3___1___3'
,'PAY_4___1___3','PAY_5___1___3','PAY_6___1___3','EDUCATION___6___1___3'
,'MARRIAGE___6___1___3','PAY_2___6___1___3','PAY_4___6___1___3'
,'PAY_5___6___1___3','PAY_6___6___1___3','BILL_AMT1___6___1___3'
,'BILL_AMT2___6___1___3','BILL_AMT3___6___1___3','BILL_AMT4___6___1___3'
,'BILL_AMT6___6___1___3','PAY_AMT1___6___1___3','PAY_AMT2___6___1___3'
,'PAY_AMT5___6___1___3','PAY_AMT6___6___1___3']
df, label = dataset.get_credit_card()
columns = df.columns
train = None
action = CreditEnv(5).action
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
