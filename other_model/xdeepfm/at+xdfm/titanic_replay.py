import sys
import os
sys.path.append(os.path.abspath('.'))
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, log_loss, roc_auc_score
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
    data, label = dataset.get_titanic(False)
    print(data.shape)
    path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(path, 'model')
    path = os.path.join(path, 'at+xdfm_titanic.pth')
    print(path)
    bike = env.TitanicEnv(15)
    actions = [57, 60, 48, 64, 66, 74, 51, 58, 55, 59, 37, 65, 23, 56, 50]
    actions = set(actions)
    sparse_features = []
    dense_features = data.columns.values.tolist()
    for action in actions:
        data[f'{action}'], is_combinations = bike.action.replay(action)
        if is_combinations:
            sparse_features.append(f'{action}')
        else:
            dense_features.append(f'{action}')
    # data.to_csv('aa.scv')
    mms = MinMaxScaler(feature_range=(0, 1))
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        data[feat] = mms.fit_transform(data[[feat]])

    data[dense_features] = mms.fit_transform(data[dense_features])
    # data = data.replace([np.nan], 0)
    # print(np.isnan(data).any())
    # print(data.shape)
    # print(data['0'])
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
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

    model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    task='binary',
                    l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    model.fit(train_model_input, y_train.values, batch_size=256, epochs=15, verbose=2, validation_split=0.2)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(y_test.values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(y_test.values, pred_ans), 4))

