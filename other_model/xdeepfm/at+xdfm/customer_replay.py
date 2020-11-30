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
    path = os.path.join(path, 'at+xdfm_customer.pth')
    print(path)
    data, label = dataset.get_customer_satisfaction()
    print(data.shape)
    customer = env.CustomerSatisfactionEnv(80)
    print('-------start-------')
    actions = [18517, 24900, 5136, 23322, 10559, 34291, 29670, 26738, 640, 40553, 42737, 40782, 6135, 40806, 41117, 38158, 2389, 21745, 37555, 43106, 22224, 25349, 17514, 7977, 29960, 41118, 12965, 33515, 29049, 35365, 37218, 37217, 3858, 988, 39788, 12885, 2054, 33465, 37555, 33065, 44050, 9168, 46535, 21993, 40832, 1224, 28583, 32403, 29020, 3164, 42186, 1036, 30115, 14536, 45581, 44020, 17944, 46815, 2427, 39845, 39114, 22511, 6627, 12784, 45179, 17107, 41422, 18989, 30589, 3893, 40128, 19612, 8446, 12562, 2276, 38293, 29541, 34452, 32304, 8754]
    actions = set(actions)
    sparse_features = []
    dense_features = data.columns.values.tolist()
    # dense_features = []
    for action in actions:
        data[f'{action}'], is_combinations = customer.action.replay(action)
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

    model.fit(train_model_input, y_train.values, batch_size=1024, epochs=15, verbose=2, validation_split=0.2)
    torch.save(model, path)
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(y_test.values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(y_test.values, pred_ans), 4))

