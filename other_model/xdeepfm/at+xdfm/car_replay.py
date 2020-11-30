import sys
import os
sys.path.append(os.path.abspath('.'))
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from other_model.xdeepfm.deepctr_torch.inputs import SparseFeat, get_feature_names, DenseFeat
from other_model.xdeepfm.deepctr_torch.models import *
import environment as env
from dataprocessing import dataset
import numpy as np
from utils import init_seed

if __name__ == "__main__":
    init_seed.init_seed()
    data, label = dataset.get_car()
    path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(path, 'model')
    path = os.path.join(path, 'at+car.pth')
    print(path)
    house = env.HousePriceEnv(80)
    actions = [1261, 103, 1876, 1613, 103, 2033, 2033, 2657, 1248, 3076, 2657, 3431, 241, 422, 192, 785, 3076, 487,
               1953, 1907, 1472, 822, 1907, 487, 264, 487, 487, 1393, 1953, 2528, 422, 822, 154, 1220, 232, 137, 1708,
               556, 922, 706, 481, 2528, 706, 2528, 232, 1089, 706, 232, 269, 1220, 232, 2528, 580, 96, 2098, 1145,
               2930, 3272, 1722, 1722, 2734, 595, 1373, 481, 1544, 1907, 98, 51, 216, 2098, 269, 96, 221, 2921, 1373,
               1126, 221, 1373, 2098, 2098]
    actions = set(actions)
    # data = pd.DataFrame()
    sparse_features = []
    dense_features = []

    for i in data.columns:
        if len(data[i].unique()) < 50:
            sparse_features.append(i)
        else:
            dense_features.append(i)
    for action in actions:
        data[f'{action}'], is_combinations = house.action.replay(action)
        if is_combinations:
            sparse_features.append(f'{action}')
        else:
            dense_features.append(f'{action}')
    mms = MinMaxScaler(feature_range=(0, 1))
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        data[feat] = mms.fit_transform(data[[feat]])
    if len(dense_features) > 0:
        data[dense_features] = mms.fit_transform(data[dense_features])

    if len(dense_features) > 0:
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
    else:
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                  for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    # print((feature_names))
    # 3.generate input data for model
    train, test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    model.compile("adam", "mse", metrics=['mse'], )
    print(train.shape)
    history = model.fit(train_model_input, y_train.values, batch_size=256, epochs=1000, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(r2_score(y_test.values, pred_ans), 4))

