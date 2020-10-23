from sklearn import preprocessing
import pandas as pd


def uniform_discretizer(x: pd.DataFrame):
    est = preprocessing.KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform')
    return list(est.fit_transform(x).reshape([-1, ]))


def scale(x: pd.DataFrame):
    return preprocessing.scale(x)


def addition(x: pd.DataFrame, y: pd.DataFrame):
    return x + y


def subtraction(x: pd.DataFrame, y: pd.DataFrame):
    return x - y


def multiplication(x: pd.DataFrame, y: pd.DataFrame):
    return x * y


def division(x: pd.DataFrame, y: pd.DataFrame):
    return x / y


def group_by_max(df: pd.DataFrame, group_name: str, cal_name):
    return df.groupby(group_name)[cal_name].transform('max')


def group_by_min(df: pd.DataFrame, group_name: str, cal_name):
    return df.groupby(group_name)[cal_name].transform('min')


def group_by_mean(df: pd.DataFrame, group_name: str, cal_name):
    return df.groupby(group_name)[cal_name].transform('mean')


unary_operator_list = {
    'uniform_discretizer': uniform_discretizer,
    'scale': scale,
}

binary_operator_list = {
    'addition': addition,
    'multiplication': multiplication,
    'subtraction': subtraction,
    'division': division,
}

consider_order = {
    'subtraction',
    'division',
}


group_by_operator_list = {
    'group_by_max': group_by_max,
    'group_by_min': group_by_min,
    'group_by_mean': group_by_mean,
}