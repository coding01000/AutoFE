import os
import sys

sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')
from utils.init_seed import init_seed
# from related_work.Explorekit.dataset import dataset
from related_work.Explorekit.evaluation.evalution import evaluate_a_class_dataset
from dataprocessing import dataset
import warnings
from related_work.Explorekit.generater.operator_list import group_by_operator_list
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    init_seed()
    original_dataset, label = dataset.get_credit_card()
    original_dataset['1'] = group_by_operator_list['group_by_min'](original_dataset, 'PAY_0', 'BILL_AMT3')
    original_dataset['2'] = group_by_operator_list['group_by_mean'](original_dataset, 'PAY_0', 'BILL_AMT4')
    original_dataset['3'] = group_by_operator_list['group_by_max'](original_dataset, 'PAY_0', 'BILL_AMT3')
    original_dataset['4'] = group_by_operator_list['group_by_max'](original_dataset, 'PAY_5', '1')
    original_dataset['5'] = group_by_operator_list['group_by_max'](original_dataset, 'PAY_3', 'PAY_AMT6')


    print(f'score:{evaluate_a_class_dataset(original_dataset, label)}')
    print(original_dataset.columns)
