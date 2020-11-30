import os
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import scipy
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from dataprocessing import dataset
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
import pandas as pd
import environment as env



# if __name__ == '__main__':
#     combs = [[0, 7], [5, 7], [3, 7], [2, 5], [4, 6], [1, 7], [0, 8], [2, 6], [0, 1], [0, 3], [0, 6],[1, 5], [4, 5], [2, 8], [5, 8]]
#     ohe_list = []
#     train_sets = None
#     train, y = dataset.get_amazon(False)
#     # for i in train.columns:
#     #     ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
#     #     ohe.fit(train[[i]])
#     #     # print(ohe.categories_)
#     #     ohe_list.append(ohe)
#     for i in train.columns:
#         train[i] = train[i].apply(str)
#     for i in combs:
#         ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
#         ohe_list.append(ohe)
#         c1 = train.columns[i[0]]
#         c2 = train.columns[i[1]]
#
#         tmp1 = train[c1] + train[c2]
#         train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(tmp1.values.reshape(-1, 1))))
#
#     model = LogisticRegression()
#     model.fit(train_sets, y)
#     train_sets = None
#
#     start = time.time()
#     for i in range(len(ohe_list)):
#         ohe = ohe_list[i]
#         c1 = train.columns[combs[i][0]]
#         c2 = train.columns[combs[i][1]]
#         tmp1 = train[c1] + train[c2]
#         # tmp1 = tmp1.values.reshape(-1, 1)
#         # ohe.transform(tmp1)
#         train_sets = scipy.sparse.hstack((train_sets, ohe.transform(tmp1.values.reshape(-1, 1))))
#     model.predict(train_sets)
#     end = time.time()
#     print(end-start)


# if __name__ == '__main__':
#     combs = [[0, 7], [5, 7], [3, 7], [2, 5], [4, 6], [1, 7], [0, 8], [2, 6], [0, 1], [0, 3], [0, 6],[1, 5], [4, 5], [2, 8], [5, 8]]
#     ohe_list = []
#     train_sets = None
#     train, y = dataset.get_amazon(False)
#     print(train.shape)
#     train = train.values.astype('str')
#     print(train.dtype)
#     # print(train[:, 0])
#     for i in combs:
#         ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
#         ohe_list.append(ohe)
#         tmp1 = np.char.add(train[:, i[0]], train[:, i[1]])
#         train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(tmp1.reshape(-1, 1))))
#
#     model = LogisticRegression()
#     model.fit(train_sets, y)
#     train_sets = None
#     start = time.time()
#     for i in range(len(ohe_list)):
#         ohe = ohe_list[i]
#         # print('---')
#         tmp1 = np.char.add(train[:, combs[i][0]], train[:, combs[i][1]])
#         train_sets = scipy.sparse.hstack((train_sets, ohe.transform(tmp1.reshape(-1, 1))))
#     model.predict(train_sets)
#     end = time.time()
#     print(end-start)

# if __name__ == '__main__':
#     combs = [[0, 7], [5, 7], [3, 7], [2, 5], [4, 6], [1, 7], [0, 8], [2, 6], [0, 1], [0, 3], [0, 6],[1, 5], [4, 5], [2, 8], [5, 8]]
#     ohe_list = []
#     train_sets = None
#     train, y = dataset.get_amazon(False)
#
#     for i in train.columns:
#         train[i] = train[i].apply(str)
#     for i in combs:
#         ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
#         ohe_list.append(ohe)
#         c1 = train.columns[i[0]]
#         c2 = train.columns[i[1]]
#
#         tmp1 = train[c1] + train[c2]
#         train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(tmp1.values.reshape(-1, 1))))
#
#     model = LogisticRegression()
#     model.fit(train_sets, y)
#     train_sets = None
#
#     start = time.time()
#     for i in range(len(ohe_list)):
#         ohe = ohe_list[i]
#         c1 = train.columns[combs[i][0]]
#         c2 = train.columns[combs[i][1]]
#         tmp1 = train[c1] + train[c2]
#         # tmp1 = tmp1.values.reshape(-1, 1)
#         # ohe.transform(tmp1)
#         train_sets = scipy.sparse.hstack((train_sets, ohe.transform(tmp1.values.reshape(-1, 1))))
#     model.predict(train_sets)
#     end = time.time()
#     print(end-start)
# if __name__ == '__main__':
#     print(env.CreditEnv(20).base_score)
#     combs = [[0, 7], [5, 7], [3, 7], [2, 5], [4, 6], [1, 7], [0, 8], [2, 6], [0, 1], [0, 3], [0, 6],[1, 5], [4, 5], [2, 8], [5, 8]]
#     ohe_list = []
#     train_sets = None
#     train, y = dataset.get_amazon(False)
#     for i in train.columns:
#         train[i] = train[i].apply(str)
#     t = {}
#     for i in combs:
#         ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
#         ohe_list.append(ohe)
#         c1 = train.columns[i[0]]
#         c2 = train.columns[i[1]]
#         t[c1+c2] = (train[c1] + train[c2]).values.reshape(-1, 1)
#         # tmp1 = train[c1] + train[c2]
#         train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(t[c1+c2])))
#
#     model = LogisticRegression(
#         penalty='l2',
#         C=1.0,
#         fit_intercept=True,
#         # random_state=432,
#         solver='liblinear',
#         max_iter=1000,
#         n_jobs=-1,
#     )
#     model.fit(train_sets, y)
#     train_sets = None
#     from multiprocessing import Pool
#
#     def f(i):
#         # ohe = ohe_list[i]
#         c1 = train.columns[combs[i][0]]
#         c2 = train.columns[combs[i][1]]
#         # tmp1 = t[c1+c2]
#         # tmp1 = tmp1.values.reshape(-1, 1)
#         return ohe_list[i].transform(t[c1+c2])
#     # def ff(x):
#     #     return scipy.sparse.hstack(p.map(f, a))
#     # x = Pool(2)
#     # print()
#     a = [i for i in range(len(ohe_list))]
#     p = Pool(32)
#     start = time.time()
#     x = p.map(f, a)
#     train_sets = scipy.sparse.hstack(x)
#     end = time.time()
#     print(end-start)
from utils import init_seed

if __name__ == '__main__':
    a = list(combinations([i for i in range(380)], 2))
    b = list(combinations([i for i in range(380)], 3))
    c = list(combinations([i for i in range(20)], 4))
    print(len(a), len(b), len(c))
    # a = env.BankEnv(15)
    # print(a.base_score)
    # print(a.action.data.shape)
    # train = pd.read_csv(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE/dataset/Amazon/train.csv')
    # train, label = dataset.get_amazon(False)
    # kf = KFold(n_splits=5, shuffle=True, random_state=init_seed.get_seed())
    # for train_index, test_index in kf.split(train, label):
    #     train_X = train.iloc[train_index]
    #     y = label.iloc[train_index]
    #     print(y)
    #
    #     test_X = train.loc[test_index]
    # b = []
    # for i in a.columns:
    #     b.append(i.replace('.', ''))
    # a.columns = b
    # print(a.columns)
    # a = a.dropna(axis=0)
    # print(a.shape)
    # print(np.isnan(a.values).any())
#     import math
#     n = 384 * 3
#     m = 2
#     print(math.factorial(n) // (math.factorial(m) * math.factorial(n - m)) - 383 * 9)
