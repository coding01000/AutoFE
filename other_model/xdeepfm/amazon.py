# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
from other_model.xdeepfm.deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from other_model.xdeepfm.deepctr_torch.models import *
from dataprocessing import dataset
from utils import init_seed


if __name__ == "__main__":
    init_seed.init_seed()
    # data = pd.read_csv('./criteo_sample.txt')
    data, label = dataset.get_amazon(False)
    print(len(label))
    sparse_features = []
    dense_features = []
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
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)


    # print(feature_names)
    # 4.Define Model,train,predict and evaluate
    # print(len(train[target].values), train[target].values.sum())
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    kf = KFold(n_splits=5, shuffle=True, random_state=init_seed.get_seed())
    score = 0
    for train_index, test_index in kf.split(data, label):
        train = data.iloc[train_index]
        y_train = label.iloc[train_index]
        test = data.iloc[test_index]
        y_test = label.iloc[test_index]
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                        task='binary',
                        l2_reg_embedding=1e-5, device=device)

        model.compile("adagrad", "binary_crossentropy",
                      metrics=["binary_crossentropy", "auc"], )

        model.fit(train_model_input, y_train.values, batch_size=256, epochs=15, verbose=2, validation_split=0.2)
        pred_ans = model.predict(test_model_input, 256)
        auc = round(roc_auc_score(y_test.values, pred_ans), 4)
        score += auc
        print("test AUC", auc)
    print(score)
    print(score/5)
    # 3.generate input data for model
    # train, test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=init_seed.get_seed())
    # train_model_input = {name: train[name] for name in feature_names}
    # test_model_input = {name: test[name] for name in feature_names}
    #
    # model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary',
    #                l2_reg_embedding=1e-5, device=device)
    #
    # model.compile("adagrad", "binary_crossentropy",
    #               metrics=["binary_crossentropy", "auc"], )
    #
    # model.fit(train_model_input, y_train.values, batch_size=256, epochs=15, verbose=2, validation_split=0.2)
    # # torch.save(model, './aaa.pth')
    # # model = torch.load('./aaa.pth')
    # import time
    # start = time.time()
    # pred_ans = model.predict(train_model_input, 256)
    # pred_ans = model.predict(test_model_input, 256)
    # end = time.time()
    # print(end - start)
    # print(len(pred_ans))
    # print("")
    # print("test LogLoss", round(log_loss(y_test.values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(y_test.values, pred_ans), 4))
