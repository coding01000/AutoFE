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
    path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(path, 'model')
    path = os.path.join(path, 'at+xdfm_bank.pth')
    print(path)
    data, label = dataset.get_bank()
    print(data.shape)
    credit = env.BankEnv(15)
    actions = [46, 72, 63, 23, 44, 75, 65, 45, 75, 119, 65, 69, 69, 42, 119, 63, 14, 69, 20, 40]
    actions = set(actions)
    sparse_features = []
    dense_features = []
    for i in data.columns:
        if i == label.name:
            continue
        if len(data[i].unique()) < 800:
            sparse_features.append(i)
        else:
            dense_features.append(i)
    for action in actions:
        data[f'{action}'], is_combinations = credit.action.replay(action)
        if is_combinations:
            sparse_features.append(f'{action}')
        else:
            dense_features.append(f'{action}')
    # data.to_csv('aa.scv')
    print(dense_features)
    print(sparse_features)
    mms = MinMaxScaler(feature_range=(0, 1))
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    data[dense_features] = mms.fit_transform(data[dense_features])

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
    torch.save(model, path)
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(y_test.values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(y_test.values, pred_ans), 4))

