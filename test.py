import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import scipy
from sklearn.preprocessing import OneHotEncoder

from dataprocessing import dataset
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
import pandas as pd
import environment as env


def onehot(column_name):

    ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
    train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(train[[column_name]])))


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

if __name__ == '__main__':
    combs = [[0, 7], [5, 7], [3, 7], [2, 5], [4, 6], [1, 7], [0, 8], [2, 6], [0, 1], [0, 3], [0, 6],[1, 5], [4, 5], [2, 8], [5, 8]]
    ohe_list = []
    train_sets = None
    train, y = dataset.get_amazon(False)

    for i in train.columns:
        train[i] = train[i].apply(str)
    for i in combs:
        ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
        ohe_list.append(ohe)
        c1 = train.columns[i[0]]
        c2 = train.columns[i[1]]

        tmp1 = train[c1] + train[c2]
        train_sets = scipy.sparse.hstack((train_sets, ohe.fit_transform(tmp1.values.reshape(-1, 1))))

    model = LogisticRegression()
    model.fit(train_sets, y)
    train_sets = None

    start = time.time()
    for i in range(len(ohe_list)):
        ohe = ohe_list[i]
        c1 = train.columns[combs[i][0]]
        c2 = train.columns[combs[i][1]]
        tmp1 = train[c1] + train[c2]
        # tmp1 = tmp1.values.reshape(-1, 1)
        # ohe.transform(tmp1)
        train_sets = scipy.sparse.hstack((train_sets, ohe.transform(tmp1.values.reshape(-1, 1))))
    model.predict(train_sets)
    end = time.time()
    print(end-start)