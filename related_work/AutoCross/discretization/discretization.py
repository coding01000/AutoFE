from sklearn import preprocessing
import pandas as pd


def discretization(x: pd.DataFrame, n_bins):
    est = preprocessing.KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    return list(est.fit_transform(x).reshape([-1, ]))


def mul_granularity_discretization(x: pd.DataFrame):
    df = pd.DataFrame()
    columns = x.columns
    for col in columns:
        if len(x[col].unique()) <= 200:
            df[col] = x[col]
            continue
        df[f'{col}__1'] = discretization(x[[col]], 10)
        df[f'{col}__2'] = discretization(x[[col]], 100)
        df[f'{col}__3'] = discretization(x[[col]], 200)

    return df
