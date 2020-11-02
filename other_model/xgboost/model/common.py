from xgboost import XGBClassifier, XGBRegressor
from utils.init_seed import get_seed
from sklearn.model_selection import cross_validate


def class_model(x, y):
    model = XGBClassifier(random_state=get_seed())
    stats = cross_validate(model, x, y, groups=None, scoring='roc_auc',
                           cv=5, return_train_score=True)
    return stats['test_score'].mean()


def regress_model(x, y):
    model = XGBRegressor(random_state=get_seed())
    stats = cross_validate(model, x, y, groups=None, scoring='r2',
                           cv=5, return_train_score=True)
    return stats['test_score'].mean()

