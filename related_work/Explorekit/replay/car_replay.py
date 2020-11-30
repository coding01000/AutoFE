import os
import sys

sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from utils.init_seed import init_seed
# from related_work.Explorekit.dataset import dataset
from related_work.Explorekit.evaluation.evalution import evaluate_a_class_dataset, evaluate_a_regress_dataset
from dataprocessing import dataset
import warnings
from related_work.Explorekit.generater.operator_list import group_by_operator_list, unary_operator_list
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    init_seed()
    original_dataset, label = dataset.get_car()
    print('-------start-----------')
    original_dataset['tmp'] = unary_operator_list['scale'](original_dataset[['Engine HP']])
    original_dataset['1'] = original_dataset['tmp'] / original_dataset['city mpg']
    original_dataset['2'] = group_by_operator_list['group_by_mean'](original_dataset, 'Make', '1')
    original_dataset['3'] = group_by_operator_list['group_by_mean'](original_dataset, 'Engine Cylinders', '2')
    original_dataset['4'] = group_by_operator_list['group_by_mean'](original_dataset, 'Make', '3')
    original_dataset['tmp'] = unary_operator_list['uniform_discretizer'](original_dataset[['Engine Cylinders']])
    original_dataset['5'] = group_by_operator_list['group_by_mean'](original_dataset, 'Make', 'tmp')
    original_dataset['tmp1'] = unary_operator_list['uniform_discretizer'](original_dataset[['Engine Cylinders']])
    original_dataset['tmp2'] = unary_operator_list['scale'](original_dataset[['Driven_Wheels']])
    original_dataset['6'] = group_by_operator_list['group_by_min'](original_dataset, 'tmp1', 'tmp2')

    del original_dataset['tmp']
    del original_dataset['tmp1']
    del original_dataset['tmp2']
    print('------------------')
    print(f'score:{evaluate_a_regress_dataset(original_dataset, label)}')
    print(original_dataset.columns)
