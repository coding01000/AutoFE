import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


def auc_score(x, y):
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=True,
        random_state=432,
        solver='liblinear',
        max_iter=1000,
    )
    stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                           cv=5, return_train_score=True)
    return stats['test_score'].mean() * 100


def evaluate_a_candidate(original_dataset: pd.DataFrame, candidate, label):
    df = original_dataset.copy()
    df[candidate.name] = candidate
    score = auc_score(df.values, label)
    return score


def evaluate_a_dataset(df: pd.DataFrame, label):
    score = auc_score(df.values, label)
    return score



