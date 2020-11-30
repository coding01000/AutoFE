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
    original_dataset, label = dataset.get_market_sales()
    print('-------start-----------')
    original_dataset['tmp1'] = unary_operator_list['uniform_discretizer'](original_dataset[['Item_Type']])
    original_dataset['1'] = group_by_operator_list['group_by_mean'](original_dataset, 'Outlet_Type', 'tmp1')
    original_dataset['tmp2'] = unary_operator_list['scale'](original_dataset[['Item_Visibility']])
    original_dataset['2'] = original_dataset['tmp2'] * original_dataset['Item_MRP']
    original_dataset['tmp3'] = unary_operator_list['scale'](original_dataset[['Item_Weight']])
    original_dataset['3'] = original_dataset['tmp3'] / original_dataset['Item_MRP']

    for i in range(1, 4):
        del original_dataset[f'tmp{i}']
    print('------------------')
    print(f'score:{evaluate_a_regress_dataset(original_dataset, label)}')
    print(original_dataset.columns)
