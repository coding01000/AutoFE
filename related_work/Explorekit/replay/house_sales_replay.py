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
    original_dataset, label = dataset.get_house_sales()
    print('-------start-----------')
    original_dataset['tmp1'] = unary_operator_list['uniform_discretizer'](original_dataset[['lat']])
    original_dataset['tmp2'] = unary_operator_list['scale'](original_dataset[['yr_renovated']])
    original_dataset['1'] = group_by_operator_list['group_by_mean'](original_dataset, 'tmp1', 'tmp2')
    original_dataset['tmp3'] = unary_operator_list['scale'](original_dataset[['sqft_living']])
    original_dataset['2'] = group_by_operator_list['group_by_mean'](original_dataset, 'grade', 'tmp3')
    original_dataset['3'] = group_by_operator_list['group_by_mean'](original_dataset, 'tmp1', 'floors')
    original_dataset['4'] = original_dataset['sqft_living15'] / original_dataset['yr_built']
    original_dataset['tmp4'] = unary_operator_list['uniform_discretizer'](original_dataset[['sqft_living']])
    original_dataset['tmp5'] = unary_operator_list['uniform_discretizer'](original_dataset[['long']])
    original_dataset['5'] = group_by_operator_list['group_by_mean'](original_dataset, 'tmp4', 'tmp5')
    original_dataset['6'] = original_dataset['tmp3'] / original_dataset['long']
    original_dataset['tmp6'] = unary_operator_list['uniform_discretizer'](original_dataset[['zipcode']])
    original_dataset['7'] = group_by_operator_list['group_by_max'](original_dataset, 'tmp4', 'tmp6')
    original_dataset['8'] = group_by_operator_list['group_by_mean'](original_dataset, 'tmp6', '7')

    for i in range(1, 7):
        del original_dataset[f'tmp{i}']
    print('------------------')
    print(f'score:{evaluate_a_regress_dataset(original_dataset, label)}')
    print(original_dataset.columns)
