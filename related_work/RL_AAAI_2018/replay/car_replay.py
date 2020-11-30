import sys

import scipy
from scipy import sparse
import os
sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from related_work.RL_AAAI_2018.env.CarEnv import CarEnv
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

# env = CarEnv(5)
# print('---start---')
# print(env.base_score)
# print(env.step(6))
# print(env.step(5))
# print(env.step(3))
# print(env.step(0))
# print(env.step(7))
# print(env.action.data.columns.values)
# print(env.get_score(env.action.X))
# df, label = dataset.get_customer_satisfaction()

features = ['Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
 'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Market Category',
 'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg', 'Popularity',
 'Engine Fuel Type___6', 'Engine HP___6', 'Engine Cylinders___6',
 'Transmission Type___6', 'Driven_Wheels___6', 'Number of Doors___6',
 'Market Category___6', 'Vehicle Size___6', 'Vehicle Style___6',
 'highway MPG___6', 'city mpg___6', 'Popularity___6', 'Engine Fuel Type___5',
 'Engine Cylinders___5', 'Transmission Type___5', 'Driven_Wheels___5',
 'Number of Doors___5', 'Market Category___5', 'Vehicle Size___5',
 'Vehicle Style___5', 'highway MPG___5', 'city mpg___5', 'Popularity___5',
 'Engine Fuel Type___6___5', 'Engine HP___6___5', 'Engine Cylinders___6___5',
 'Transmission Type___6___5', 'Driven_Wheels___6___5',
 'Number of Doors___6___5', 'Market Category___6___5',
 'Vehicle Size___6___5', 'Vehicle Style___6___5', 'highway MPG___6___5',
 'city mpg___6___5', 'Popularity___6___5', 'Engine HP___0',
 'Engine Cylinders___0', 'Transmission Type___0', 'Driven_Wheels___0',
 'Number of Doors___0', 'Market Category___0', 'Vehicle Size___0',
 'Vehicle Style___0', 'highway MPG___0', 'city mpg___0', 'Popularity___0',
 'Engine Fuel Type___6___0', 'Engine HP___6___0',
 'Transmission Type___6___0', 'Driven_Wheels___6___0',
 'Number of Doors___6___0', 'Market Category___6___0',
 'Vehicle Size___6___0', 'Vehicle Style___6___0', 'highway MPG___6___0',
 'city mpg___6___0', 'Popularity___6___0', 'Engine Fuel Type___5___0',
 'Engine Cylinders___5___0', 'Transmission Type___5___0',
 'Driven_Wheels___5___0', 'Number of Doors___5___0',
 'Market Category___5___0', 'Vehicle Size___5___0', 'Vehicle Style___5___0',
 'highway MPG___5___0', 'city mpg___5___0', 'Popularity___5___0',
 'Engine Fuel Type___6___5___0', 'Engine HP___6___5___0',
 'Engine Cylinders___6___5___0', 'Transmission Type___6___5___0',
 'Driven_Wheels___6___5___0', 'Number of Doors___6___5___0',
 'Market Category___6___5___0', 'Vehicle Size___6___5___0',
 'Vehicle Style___6___5___0', 'highway MPG___6___5___0',
 'city mpg___6___5___0', 'Popularity___6___5___0']
df, label = dataset.get_car()
columns = df.columns
train = None
action = CarEnv(5).action
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
