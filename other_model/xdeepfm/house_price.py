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

from dataprocessing import dataset
from utils import init_seed

if __name__ == "__main__":
    init_seed.init_seed()
    data, label = dataset.get_house_price(False)
    sparse_features = []
    dense_features = []

    for i in data.columns:
        if len(data[i].unique()) < 50:
            sparse_features.append(i)
        else:
            dense_features.append(i)
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
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

    history = model.fit(train_model_input, y_train.values, batch_size=256, epochs=30, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(r2_score(y_test.values, pred_ans), 4))
