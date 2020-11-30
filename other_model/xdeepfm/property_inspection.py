# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
sys.path.append(os.path.abspath('.'))
from other_model.xdeepfm.deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from other_model.xdeepfm.deepctr_torch.models import *
from dataprocessing import dataset
from utils import init_seed
import time


if __name__ == "__main__":
    init_seed.init_seed()

    data, label = dataset.get_property_inspection()
    sparse_features = []
    dense_features = data.columns
    for i in data.columns:
        if i == label.name:
            continue
        if len(data[i].unique()) < 800:
            sparse_features.append(i)
        else:
            dense_features.append(i)
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    if len(dense_features) > 0:
        data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    if len(dense_features) > 0:
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
    else:
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                  for feat in sparse_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    # 3.generate input data for model
    train, test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=init_seed.get_seed())
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    # 4.Define Model,train,predict and evaluate
    # print(len(train[target].values), train[target].values.sum())
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    model.compile("adam", "mse", metrics=['mse'], )
    print(len(train_model_input))
    history = model.fit(train_model_input, y_train.values, batch_size=256, epochs=15, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(r2_score(y_test.values, pred_ans), 4))
