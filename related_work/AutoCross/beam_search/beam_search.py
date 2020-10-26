import pandas as pd
from tqdm import tqdm
from related_work.AutoCross.node_expansion.node_exoansion import expansion, expanded_root_node
from related_work.AutoCross.feature_wise.model_train import get_class_predict, get_regress_predict


def search(df: pd.DataFrame, label, expand_depth):
    # init the label to train
    # label = label - get_regress_predict(df.values, label)
    # the list of chosen crossing feature
    chosen_list = {}
    # init root node
    expand_set = expanded_root_node(df)

    bar = tqdm(total=expand_depth, desc='beam search')
    for i in range(expand_depth):
        # store the score
        prediction = {}
        for set_name in expand_set:
            prediction[set_name] = abs(get_regress_predict(expand_set[set_name], label) - label).sum()

        # find the best and unselected features
        chosen = sorted(prediction.items(), key=lambda item: item[1])[0][0]
        # for can in candidate:
        #     if can[0] not in chosen_list.keys():
        #         chosen = can[0]
        #         break

        chosen_list[chosen] = expand_set[chosen]
        # update label for the next step to train
        label = label - get_regress_predict(expand_set[chosen], label)
        # get the expand set: {feature_name: feature}
        expand_set = expansion(chosen)
        print(i, len(expand_set))
        bar.update()

    bar.close()
    return chosen_list
