from catboost.datasets import amazon
from related_work.AutoCross.feature_wise.model_train import get_class_predict, evaluation_with_auc
from related_work.AutoCross.beam_search.beam_search import search
from related_work.AutoCross.dataset.dataset import get_amazon
from related_work.AutoCross.feature_wise.model_train import evaluation_with_auc
import scipy
import pandas as pd

if __name__ == '__main__':
    train, label = get_amazon()
    chosen_list = search(train, label, 15)
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
