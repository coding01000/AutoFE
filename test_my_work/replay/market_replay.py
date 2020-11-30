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
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import environment as env


if __name__ == '__main__':
    # init_seed.init_seed()
    market = env.MarketEnv(20)
    # run_agent.replay(amazon, 'PPO_Amazon_15')
    actions = [17, 49, 35, 49, 113, 120, 40, 60, 3, 3, 8, 117, 120, 120, 3, 66, 62, 35, 52, 120]
    actions = set(actions)
    _, label = dataset.get_market_sales()
    train = None
    for action in actions:
        market.step(action)
    train = market.action.data_sets()

    model = LinearRegression()

    y = label
    stats = cross_validate(model, train, y, groups=None, scoring='r2',
                           cv=5, return_train_score=True)
    print(stats['test_score'].mean())
