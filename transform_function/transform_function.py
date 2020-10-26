import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, scale


def log1p(df: pd.DataFrame):
    tmp = df.apply(np.log1p)
    return sparse.csr_matrix(tmp)


def one_hot(df: pd.DataFrame):
    ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
    return ohe.fit_transform(df)


def tanh(df: pd.DataFrame):
    tmp = df.apply(np.tanh)
    return sparse.csr_matrix(tmp)


def sin(df: pd.DataFrame):
    tmp = df.apply(np.sin)
    return sparse.csr_matrix(tmp)


def cos(df: pd.DataFrame):
    tmp = df.apply(np.cos)
    return sparse.csr_matrix(tmp)


def uniform_discretizer(x: pd.DataFrame):
    est = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform')
    return list(est.fit_transform(x).reshape([-1, ]))


def scale_(df: pd.DataFrame):
    tmp = scale(df)
    return sparse.csr_matrix(tmp)


transform_list = [
    log1p,
    one_hot,
    tanh,
    sin,
    cos,
    scale,
]