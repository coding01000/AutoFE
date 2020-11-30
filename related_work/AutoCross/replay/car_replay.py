import sys

from sklearn.preprocessing import OneHotEncoder

sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
from related_work.AutoCross.discretization.discretization import discretization
from utils.init_seed import init_seed
from related_work.AutoCross.beam_search.beam_search import search
from related_work.AutoCross.feature_wise.model_train import evaluation_with_auc, evaluation_with_r2
import scipy
# from dataprocessing.house_price_data import get_data
import pandas as pd
import numpy as np
from dataprocessing import dataset

features = [['Transmission Type', 'Market Category'], ['Engine HP__1', 'Engine Cylinders'], ['Driven_Wheels', 'Vehicle Style'], ['Year', 'Vehicle Size'], ['Engine Cylinders', 'Engine HP__1', 'Engine Fuel Type'], ['Vehicle Style', 'Driven_Wheels', 'Number of Doors'], ['Vehicle Size', 'Vehicle Style'], ['Vehicle Style', 'Driven_Wheels', 'Engine Fuel Type'], ['Engine Cylinders', 'Engine HP__1', 'Engine Fuel Type', 'Number of Doors']]
if __name__ == '__main__':
    init_seed()
    df, label = dataset.get_car()
    train = None
    granularity = {'1': 10, '2': 100, '3': 200}
    columns = df.columns
    print('---start---')
    for feature in features:
        print(feature)
        if feature[0] in columns:
            tmp = df[feature[0]].apply(str)
        else:
            t = feature[0].split('__')[0]
            g = feature[0].split('__')[1]
            tmp = pd.Series(discretization(df[[t]], granularity[g])).apply(str)
        for i in range(1, len(feature)):
            if feature[i] in columns:
                tmp += df[feature[i]].apply(str)
            else:
                t = feature[i].split('__')[0]
                g = feature[i].split('__')[1]
                tmp += pd.Series(discretization(df[[t]], granularity[g])).apply(str)
        ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
        tmp = ohe.fit_transform(tmp.values.reshape(-1, 1))
        train = scipy.sparse.hstack([train, tmp])
    print(train.shape)
    print(evaluation_with_r2(train, label))
