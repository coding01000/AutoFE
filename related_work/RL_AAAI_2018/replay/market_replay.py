import sys

import scipy
from scipy import sparse
import os
sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from related_work.RL_AAAI_2018.env.MarketEnv import MarketEnv
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

# env = MarketEnv(5)
# print('---start---')
# print(env.base_score)
# print(env.step(5))
# print(env.step(11))
# print(env.step(4))
# print(env.step(5))
# print(env.step(3))
# print(env.action.data.columns.values)
# print(env.get_score(env.action.X))
# df, label = dataset.get_house_sales()

features = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility'
, 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year'
, 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
, 'Outlet_Establishment_Year___5', 'Outlet_Size___5'
, 'Outlet_Location_Type___5', 'Outlet_Type___5']
df, label = dataset.get_market_sales()
columns = df.columns
train = None
action = MarketEnv(5).action
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
