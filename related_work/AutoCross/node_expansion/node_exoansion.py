import pandas as pd
from itertools import combinations
from sklearn.preprocessing import OneHotEncoder
from related_work.AutoCross.discretization.discretization import mul_granularity_discretization
import numpy as np


def parsing_feature(feature):
    return feature.split('___')


def pair_is_eq(pair):
    # p0 = [i.split('__')[0] for i in pair[0].split('___')]
    # p1 = [i.split('__')[0] for i in pair[1].split('___')]
    # return sorted(p0) == sorted(p1)
    return pair[0].split('__')[0] == pair[1].split('__')[0]


expand_set = {}
columns = []
df = pd.DataFrame()


def expanded_root_node(original_data: pd.DataFrame):
    global expand_set
    global columns
    global df

    df = mul_granularity_discretization(original_data)
    columns = list(df.columns)
    combs = combinations(columns, 2)
    for pair in combs:
        if pair_is_eq(pair) is True:
            continue
        name = f'{pair[0]}___{pair[1]}'
        tmp = df[pair[0]].apply(str) + df[pair[1]].apply(str)
        ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
        tmp = ohe.fit_transform(tmp.values.reshape(-1, 1))
        expand_set[name] = tmp
    for col in columns:
        ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
        tmp = ohe.fit_transform(df[[col]])
        expand_set[name] = tmp
    return expand_set


def expansion(new_feature):
    global expand_set
    global columns
    global df

    # delete new feature in expand_set
    del expand_set[new_feature]

    new_feature_parsing = parsing_feature(new_feature)
    for col in columns:
        if col in new_feature:
            continue

        col_parsing = parsing_feature(col)
        cols = list(set(new_feature_parsing+col_parsing))

        name = cols[0]
        tmp = df[cols[0]].apply(str)
        for i in range(1, len(cols)):
            tmp += df[cols[i]].apply(str)
            name += f'___{cols[i]}'
        ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
        tmp = ohe.fit_transform(tmp.values.reshape(-1, 1))

        expand_set[name] = tmp
        columns.append(new_feature)

    return expand_set
