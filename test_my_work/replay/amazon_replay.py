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
import environment as env


if __name__ == '__main__':
    # init_seed.init_seed()
    amazon = env.AmazonEnv(15)
    # run_agent.replay(amazon, 'PPO_Amazon_15')
    actions = [61, 79, 86, 82, 72, 68, 21, 62, 73, 63, 60, 57, 55, 81, 66]
    actions = set(actions)
    _, label = dataset.get_amazon(False)
    train = None
    for action in actions:
        amazon.step(action)
    train = amazon.action.data_sets()

    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=True,
        solver='liblinear',
        max_iter=1000,
        random_state=init_seed.get_seed(),
    )
    # kf = KFold(n_splits=5, shuffle=True, random_state=init_seed.get_seed())
    # for train_index, test_index in kf.split(train, label):
    #     train_X = train[train_index]
    #     test_X = train[test_index]
    #     train_y = label[train_index]
    #     test_y = label[test_index]
    #     model = LogisticRegression(
    #         penalty='l2',
    #         C=1.0,
    #         fit_intercept=True,
    #         solver='liblinear',
    #         max_iter=1000,
    #         random_state=init_seed.get_seed(),
    #     )
    #     model.fit(train_X, train_y)
    #     y = model.predict(test_X)
    #     print(metrics.roc_auc_score(test_y, y))
    train_X, test_X, train_y, test_y = train_test_split(train, label, test_size=0.2)
    y = label
    stats = cross_validate(model, train, y, groups=None, scoring='roc_auc',
                           cv=5, return_train_score=True)
    print(stats['test_score'])
