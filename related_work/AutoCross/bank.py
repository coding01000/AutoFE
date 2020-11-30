import sys
import os
import os
sys.path.append(os.path.abspath(__file__).split('AutoFE')[0]+'AutoFE')

from utils.init_seed import init_seed
from related_work.AutoCross.beam_search.beam_search import search
from related_work.AutoCross.feature_wise.model_train import evaluation_with_auc, evaluation_with_r2
import scipy
# from dataprocessing.house_price_data import get_data
from dataprocessing import dataset
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    init_seed()
    train, label = dataset.get_bank()
    chosen_list = search(train, label, 20, 'class')
    print(chosen_list.keys())

    # train = scipy.sparse.csr_matrix(train.values)
    train = None
    for i in chosen_list:
        if train is None:
            train = chosen_list[i]
        else:
            train = scipy.sparse.hstack([train, chosen_list[i]])
    print(train.shape)
    print(evaluation_with_auc(train, label))
