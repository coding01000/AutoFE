import pandas as pd
from tqdm import tqdm
from related_work.AutoCross.node_expansion.node_exoansion import expansion, expanded_root_node
from related_work.AutoCross.feature_wise.model_train import get_class_predict, get_regress_predict
import scipy
from related_work.AutoCross.feature_wise.model_train import evaluation_with_auc, evaluation_with_r2


def search(df: pd.DataFrame, label, expand_depth, kind):
    # row_label = label
    chosen_ = None
    # init the label to train
    # label = label - get_regress_predict(df.values, label)
    # the list of chosen crossing feature
    chosen_list = {}
    # init root node
    print('----------start--------------')
    expand_set = expanded_root_node(df)
    print(len(expand_set))
    bar = tqdm(total=expand_depth, desc='beam search')
    for i in range(expand_depth):
        # store the score
        prediction = {}
        for set_name in expand_set:
            if kind == 'class':
                score = evaluation_with_auc(scipy.sparse.hstack([chosen_, expand_set[set_name]]), label)
            else:
                score = evaluation_with_r2(scipy.sparse.hstack([chosen_, expand_set[set_name]]), label)
            prediction[set_name] = score

        # find the best and unselected features
        chosen = sorted(prediction.items(), key=lambda item: item[1], reverse=True)[0][0]
        # for can in candidate:
        #     if can[0] not in chosen_list.keys():
        #         chosen = can[0]
        #         break

        chosen_list[chosen] = expand_set[chosen]
        # update label for the next step to train
        # label = label - get_regress_predict(expand_set[chosen], label)
        # get the expand set: {feature_name: feature}
        chosen_ = scipy.sparse.hstack([chosen_, expand_set[chosen]])
        if kind == 'class':
            print(evaluation_with_auc(chosen_, label))
        else:
            print(evaluation_with_r2(chosen_, label))
        print(chosen_list.keys())

        expand_set = expansion(chosen)

        bar.update()

    bar.close()
    return chosen_list
