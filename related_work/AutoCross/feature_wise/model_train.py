import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


def get_class_predict(x, y):
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=True,
        random_state=432,
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
        random_state=432,
        solver='liblinear',
        max_iter=1000,
    )
    stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                           cv=5, return_train_score=True)
    return stats['test_score'].mean() * 100




