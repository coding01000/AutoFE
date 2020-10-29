import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from utils import init_seed


def get_regress_predict(x, y):
    model = DecisionTreeRegressor(random_state=init_seed.get_seed())
    model.fit(x, y)
    return model.predict(x)


def get_class_predict(x, y):
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=True,
        random_state=init_seed.get_seed(),
        solver='liblinear',
        max_iter=1000,
    )
    model.fit(x, y)
    return model.predict_proba(x)[:, 1]
    # return model.predict(x)


def evaluation_with_auc(x, y):
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=True,
        random_state=init_seed.get_seed(),
        solver='liblinear',
        max_iter=1000,
    )
    stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                           cv=5, return_train_score=True)
    return stats['test_score'].mean() * 100


def evaluation_with_r2(x, y):
    model = DecisionTreeRegressor(random_state=init_seed.get_seed())
    stats = cross_validate(model, x, y, groups=None, scoring='r2',
                           cv=5, return_train_score=True)
    return stats['test_score'].mean() * 100




