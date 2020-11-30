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
    car = env.CarEnv(25)
    # run_agent.replay(amazon, 'PPO_Amazon_15')
    actions = [190, 29, 13, 33, 33, 172, 109, 145, 154, 147, 35, 13, 33, 162, 109, 35, 30, 55, 109, 24]
    actions = set(actions)
    _, label = dataset.get_car()
    train = None
    for action in actions:
        car.step(action)
    train = car.action.data_sets()

    model = LinearRegression()
    y = label
    stats = cross_validate(model, train, y, groups=None, scoring='r2',
                           cv=5, return_train_score=True)
    print(stats['test_score'].mean())
