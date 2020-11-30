import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_validate
# from sklearn import tree

from utils import init_seed


def auc_score(x, y):
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=True,
        random_state=init_seed.get_seed(),
        solver='liblinear',
        max_iter=1000,
        # n_jobs=-1
    )
    stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                           cv=5, return_train_score=True)
    return stats['test_score'].mean() * 100


def r2_score(x, y):
    model = LinearRegression()
    stats = cross_validate(model, x, y, groups=None, scoring='r2',
                           cv=5, return_train_score=True)
    return stats['test_score'].mean() * 100


def evaluate_a_class_candidate(original_dataset: pd.DataFrame, candidate, label):
    df = original_dataset.copy()
    df[candidate.name] = candidate
    score = auc_score(df.values, label)
    return score


def evaluate_a_class_dataset(df: pd.DataFrame, label):
    score = auc_score(df.values, label)
    return score


def evaluate_a_regress_candidate(original_dataset: pd.DataFrame, candidate, label):
    df = original_dataset.copy()
    df[candidate.name] = candidate
    score = r2_score(df.values, label)
    return score


def evaluate_a_regress_dataset(df: pd.DataFrame, label):
    score = r2_score(df.values, label)
    return score



