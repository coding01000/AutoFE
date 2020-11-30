import os
import sys

sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from utils.init_seed import init_seed
# from related_work.Explorekit.dataset import dataset
from related_work.Explorekit.evaluation.evalution import evaluate_a_class_dataset
from dataprocessing import dataset
import warnings
from related_work.Explorekit.generater.operator_list import group_by_operator_list, unary_operator_list
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    init_seed()
    original_dataset, label = dataset.get_bank()

    original_dataset['1'] = original_dataset['euribor3m'] * original_dataset['duration']
    original_dataset['tmp'] = unary_operator_list['uniform_discretizer'](original_dataset[['euribor3m']])
    original_dataset['2'] = group_by_operator_list['group_by_max'](original_dataset, 'tmp', '1')
    original_dataset['3'] = group_by_operator_list['group_by_min'](original_dataset, 'nremployed', 'euribor3m')

    del original_dataset['tmp']

    print(f'score:{evaluate_a_class_dataset(original_dataset, label)}')
    print(original_dataset.columns)
