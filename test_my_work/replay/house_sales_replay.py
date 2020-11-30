import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from utils import init_seed
from sklearn import metrics
from test_my_work import run_agent
from scipy import sparse
from dataprocessing import dataset
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import environment as env


if __name__ == '__main__':
    # init_seed.init_seed()
    house_sales = env.HouseSalesEnv(25)
    # run_agent.replay(amazon, 'PPO_Amazon_15')
    actions = [37, 251, 178, 17, 263, 193, 318, 219, 310, 278, 320, 262, 263, 151, 318, 263, 310, 193, 224, 321, 224, 53, 203, 320, 136]
    actions = set(actions)
    _, label = dataset.get_house_sales()
    train = None
    for action in actions:
        house_sales.step(action)
    train = house_sales.action.data_sets()

    model = LinearRegression()
    y = label
    stats = cross_validate(model, train, y, groups=None, scoring='r2',
                           cv=5, return_train_score=True)
    print(stats['test_score'].mean())
